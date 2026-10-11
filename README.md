# hierarchical-spacecraft-control

Tests whether a three-layer AI controller can make fast spacecraft decisions while staying aligned
with a larger mission. A slow model sets strategy, a mid-speed model updates short-term goals, and a
fast layer picks actions.

| Layer | Default model | Runs | Output |
|---|---|---|---|
| Mission strategy | Claude Opus (`claude-opus-5-5`) | once before the episode | mission brief: objective, approach, keep-out radius, speed limits, fuel reserve, re-plan period |
| Tactical planning | Claude Haiku (`claude-haiku-5-5`) | asynchronously, every `tactical_period_s` | short-term goal: waypoint (Lady-relative), limits, mode hint |
| Immediate control | TypeSafe Jev (`jev-latest`) | every 0.25 s whenever free | one of six maneuver primitives |
| Executor | code, no model | every 0.25 s | lbg1 thrust command |

Jev is a classifier. It answers a `choice` question over a fixed option list and doesn't write text,
so the immediate-control layer is split in two. Jev picks a maneuver primitive (`APPROACH_LADY`,
`GO_TO_WAYPOINT`, `BRAKE`, `EVADE_GUARD`, `HOLD_STANDOFF`, `COAST`). A deterministic executor then
turns that primitive into thrust, using Clohessy-Wiltshire relative-motion guidance in the Lady's
LVLH frame. Every configuration shares the same executor, so they differ only in *who decides* and
*how long deciding takes*.

## Simulator: `lbg1_lite`

The real environment is KSPDG lbg1, which needs Kerbal Space Program and kRPC. The competition
docker image is private. `hsc/sim/lbg1_lite.py` is a pure-Python stand-in that copies lbg1's
interface:

- the 21-float observation layout;
- the info keys and score, `dm_lb^2 + 1e6/(dm_bg + 0.1)`, where lower is better;
- 5 m capture distances and the 240 s timeout;
- RCS thrust and Isp taken from the KSPDG source.

Vessel masses and exact orbits are approximations, because the KSP save files aren't public.

**Scenario of record: `LBG1_LG7_I2_V1`.**
- **Starting positions (`i2`):** all three craft share one circular orbit. The Guard is 600 m
  behind the Lady, and the Bandit is 2000 m behind the Guard.
- **Guard (`lg7`):** like LG3 and above, LG7 is code-obfuscated upstream. `lg7` here is a proxy:
  an aggressive pursuer that re-aims every step, closes at up to about 12 m/s, has a 1 s attitude
  lag and never coasts.
- **Other Guards:** `lg0` (passive) and `lg1` (the zero, pursue, coast heuristic) follow the public
  source, and `i1` is also available.

**Actions match the controller API for this project.** Each command is a forward/backward throttle
and a right/left throttle, each from -1 to 1, plus a duration of 0 to 10 s. There is no up/down axis.
The Bandit is assumed to be held prograde and rolled so those two axes lie in the orbital plane:
forward is along-track and right is roughly radial-in. Since every craft starts in that plane, the
game is planar. Pass `two_axis=False` to get KSPDG's full three-axis `burn_vec` action instead.

**Latency is charged in sim time.** KSPDG runs in wall-clock time, so the spacecraft keeps flying on
its last command while the agent thinks. `hsc/runner.py` reproduces this. A decision launched at time
`t` takes effect at `t + latency`, and a layer can't start a new call until its previous one returns.
Model layers use their measured API latency. No-model layers use a synthetic latency, which makes it
possible to sweep latency for free.

## Running

```bash
pip install -r requirements.txt
python -m pytest -q tests

# no API keys needed
python -m hsc.experiment --configs fast_only,rule_hier,latency_sweep --seeds 5

# needs ANTHROPIC_API_KEY
python -m hsc.experiment --configs hier_rule_fast,flat_haiku,flat_opus --seeds 3

# also needs TYPESAFE_API_KEY and network access to api.typesafe.ai
python -m hsc.experiment --configs hier_jev,hier_no_opus,flat_jev --seeds 3
```

Add `--haiku-images` to send Haiku a rendered plot of recent motion alongside the numeric history.
Results go to `results/<timestamp>/`: `episodes.jsonl` (per episode, with trajectories),
`summary.md`, `summary.json`, and `latency_sweep.png` when the sweep ran.

| Config | Strategy | Tactical | Immediate |
|---|---|---|---|
| `fast_only` | fixed brief | none | rules, 0 s |
| `rule_hier` | fixed brief | rules (radial bypass) | rules, 0 s |
| `latency_sweep` | fixed brief | none | rules with 0 to 16 s latency |
| `flat_opus` / `flat_haiku` | fixed brief | none | the model picks every maneuver |
| `hier_rule_fast` | Opus | Haiku | rules |
| `hier_jev` | Opus | Haiku | Jev |
| `hier_no_opus` | fixed brief | Haiku | Jev |
| `flat_jev` | fixed brief | none | Jev |

## First result (no models)

`results/baseline-no-models/`: 5 seeds per scenario, two-axis actions.

![latency sweep](results/baseline-no-models/latency_sweep.png)

In the sweep the controller and strategy are identical in every run, and only decision latency
changes:

- **`i2` + `lg1`** (Guard pursues, sitting between the Bandit and the Lady): latency up to 4 s
  makes no difference. At 6 s the closest Guard approach drops from about 680 m to about 150 m. At
  8 s the Guard catches the Bandit in 80% of runs, and at 16 s in 100%.
- **`i2` + `lg0`** (passive Guard in the way): the closest Guard approach shrinks steadily with
  latency, from 101 m at 0 s to 39 m at 4 s.
- **`i2` + `lg7` (scenario of record):** the advanced-Guard proxy catches the rule controller in
  every run, at every latency. That includes the rule tactical layer, which still gets about 65 m
  closer to the Lady before being caught. So the no-model baselines can't win LG7, and that is the
  gap the Opus and Haiku layers have to close.
- **Rule tactical layer on `i2` + `lg0`:** it doubles the closest Guard distance (218 m vs 101 m) and
  uses half the fuel, at no latency cost to the fast loop.

This sets the bar for the model runs. A flat controller whose decisions take about 6 s or more loses
to the pursuing Guard. A hierarchy keeps the fast loop under 1 s, so it only helps if its slow layers
plan better than the rules do.

## Layout

```
hsc/sim/        lbg1_lite environment and orbital helpers
hsc/control/    situation summary, plans (brief/goal/primitives), executor
hsc/layers/     strategist (Opus), tactician (Haiku), deciders (rules, Jev, flat LLM), API wrapper
hsc/runner.py   latency-aware episode loop
hsc/experiment.py  named configs, CLI, summary table, plots
tests/          sim, runner and API-shape tests (no network)
```

Opus calls opt into the server-side refusal fallback (`fallbacks: "default"`). Jev request and
response fields follow TypeSafe's public docs. Check them against your account's API reference if a
call fails.
