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
- the `{"burn_vec", "vec_type", "ref_frame"}` action;
- the info keys and score, `dm_lb^2 + 1e6/(dm_bg + 0.1)`, where lower is better;
- 5 m capture distances and the 240 s timeout;
- RCS thrust and Isp taken from the KSPDG source;
- initial conditions `i1` and `i2`, and Guard policies `lg0` (passive) and `lg1` (the zero, pursue,
  coast heuristic from `lg1_envs.py`).

Vessel masses and exact orbits are approximations, because the KSP save files aren't public.

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
| `rule_hier` | fixed brief | rules (out-of-plane bypass) | rules, 0 s |
| `latency_sweep` | fixed brief | none | rules with 0 to 16 s latency |
| `flat_opus` / `flat_haiku` | fixed brief | none | the model picks every maneuver |
| `hier_rule_fast` | Opus | Haiku | rules |
| `hier_jev` | Opus | Haiku | Jev |
| `hier_no_opus` | fixed brief | Haiku | Jev |
| `flat_jev` | fixed brief | none | Jev |

## First result (no models)

`results/baseline-no-models/`: 5 seeds per scenario.

![latency sweep](results/baseline-no-models/latency_sweep.png)

The controller and strategy are identical in every run, and only decision latency changes:

- **`i2` + `lg1`** (Guard pursues, sitting between the Bandit and the Lady): latency up to 4 s
  makes no difference. At 6 s the closest Guard approach drops from about 680 m to about 150 m. At
  8 s the Guard catches the Bandit in 60% of runs, and at 16 s in 100%.
- **`i2` + `lg0`** (passive Guard in the way): the closest Guard approach shrinks steadily with
  latency, from 103 m at 0 s to 44 m at 4 s.
- **`i1` + `lg1`**: little effect, because the Guard is never a threat on that geometry.
- **Rule tactical layer**: on `i2` + `lg0` it doubles the closest Guard distance (218 m vs 103 m) and
  uses less than half the fuel, at zero latency cost to the fast loop.

This sets the bar for the model runs. A flat controller whose decisions take about 6 s or more
(typical for a large model reasoning over state) loses to the pursuing Guard. A hierarchy keeps the
fast loop under 1 s, so it only wins if its slow layers add better plans than the rules do.

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
