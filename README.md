# hierarchical-spacecraft-control

Tests whether a three-layer AI controller can make fast spacecraft decisions while staying aligned
with a larger mission. A slow model sets strategy, a mid-speed model updates short-term goals, and a
fast layer picks actions.

| Layer | Default model | Runs | Output |
|---|---|---|---|
| Mission strategy | Claude Opus (`claude-opus-5-5`) | once before the episode | mission brief: objective, approach, keep-out radius, speed limits, fuel reserve, re-plan period |
| Tactical planning | Claude Haiku (`claude-haiku-5-5`) | asynchronously, every `tactical_period_s` | short-term goal: waypoint (Lady-relative), limits, mode hint |
| Immediate control | Kev-0.8B (open weights, local server); hosted Jev also supported | every 0.25 s whenever free | one raw action (coast/forward/backward/left/right/up/down) or one of six guided maneuvers |
| Executor | code, no model | every 0.25 s | lbg1 thrust command |

Kev is an open-source, Jev-compatible classifier ([jaredpalmer/kev](https://github.com/jaredpalmer/kev),
weights on Hugging Face). It runs locally and serves TypeSafe's System One API, so the same client
talks to Kev or to hosted Jev. It answers a `choice` question over a fixed option list and doesn't
write text. Two modes are wired up:

- **Raw actions (the first-version plan):** Kev returns one action ID: coast, forward, backward, left,
  right, up or down. An adapter maps it to a fixed-strength throttle command. `up` and `down` execute
  as a coast, because the controller API has no up/down axis.
- **Guided maneuvers:** the immediate-control layer is split in two. Jev picks a maneuver primitive (`APPROACH_LADY`,
`GO_TO_WAYPOINT`, `BRAKE`, `EVADE_GUARD`, `HOLD_STANDOFF`, `COAST`). A deterministic executor then
turns that primitive into thrust, using Clohessy-Wiltshire relative-motion guidance in the Lady's
LVLH frame. Guided configurations share the same executor, so they differ only in *who decides* and *how
long deciding takes*.

## Real loop: KSPDG observation, Kev choice, burn, next observation

`hsc/kspdg_loop.py` is the execution path for the real game. Its target is the KSPDG scenario
`LBG1_LG7_I2_V1`. Each step reads one observation, asks the local Kev server for one action ID,
validates the burn, sends it, and reads the next observation. Burns never overlap: the next request
is only sent after `env.step()` returns.

**Action catalog** (inertial axes, right-handed CBCI, `ref_frame=1`, `vec_type=0` throttle, 1.0 s):

| ID | burn_vec |
|---|---|
| `COAST` | `[0, 0, 0, 1.0]` |
| `POS_X` / `NEG_X` | `[±0.2, 0, 0, 1.0]` |
| `POS_Y` / `NEG_Y` | `[0, ±0.2, 0, 1.0]` |
| `POS_Z` / `NEG_Z` | `[0, 0, ±0.2, 1.0]` |

Kev gets `POST /v1/systemone` with `model: "kev-latest"`, a JSON `state` (mission text plus derived
features: relative position/velocity, distance and closing speed to the Lady and the Guard, time and
fuel, recent history), and one `choice` question named `action` over the seven IDs. The loop reads
`answers.action.choice`.

**Stops, with no fallback burn:** invalid observation (not 21 finite values), unknown or malformed
answer, non-finite confidence or probabilities, request timeout, connection error, an observation
older than `--max-observation-age-s` by the time the answer arrives, an invalid command, or a
simulator exception. The reason goes into the trace and the run summary.

**Trace:** `results/traces/<run_id>.jsonl`. One `run_header` (effective config, Python, platform,
repo commit, `kspdg` version, Kev `/v1/models`), one `decision` record per step (step, wall time,
mission time, raw observation, features, model, context version and age, choice, confidence,
probabilities, burn command, next observation, model and decision latency, observation age,
terminal or error reason), and one `run_end`.

### Exact commands

1. Kev server (Python 3.11+, [uv](https://docs.astral.sh/uv/)). Weights come from Hugging Face:
   `jaredpalmer/kev-0.8b@v1.0`, a LoRA adapter on `Qwen/Qwen3.5-0.8B-Base`. Don't use `pip install kev`.

   ```bash
   git clone https://github.com/jaredpalmer/kev.git && cd kev
   uv sync --extra serve
   uv run --extra serve python -m kev.serve --run jaredpalmer/kev-0.8b@v1.0 --host 127.0.0.1 --port 8009
   curl -s http://127.0.0.1:8009/v1/models    # check it is up; the run header records this response
   ```

2. KSPDG: Kerbal Space Program with the KSPDG mod and kRPC running, and the `kspdg` Python package
   (with `kspdg_envs` for LG3 and above) installed in this repo's environment. Follow the
   competition's `docs/docker_execution.md` for the Docker setup. If the loop runs inside Docker
   and Kev runs on the host, pass `--kev-base-url http://host.docker.internal:8009`.

3. This repo:

   ```bash
   pip install -r requirements.txt
   python -m hsc.kspdg_loop --env kspdg --steps 1     # one-action connection test
   python -m hsc.kspdg_loop --env kspdg --episode     # full episode, until KSPDG ends it
   ```

   Every setting is a flag, with these defaults: `--scenario LBG1_LG7_I2_V1 --episode-timeout-s 240
   --thrust-magnitude 0.2 --burn-duration-s 1.0 --kev-base-url http://127.0.0.1:8009
   --kev-model kev-latest --request-timeout-s 10 --max-observation-age-s 15 --history-len 5
   --trace-dir results/traces`.

`--env fixture` runs the same loop on the `lbg1_lite` stand-in below. It is a **test fixture only**,
not the real game, and its LG7 Guard is a proxy. `tests/test_kspdg_loop.py` covers parsing, feature
math (including zero distance), all seven mappings and limits, bad and non-finite responses,
timeout, connection loss, stale observations, both Gym step formats and cleanup, using a local stub
Kev server.

### Running Kev

```bash
git clone https://github.com/jaredpalmer/kev.git && cd kev && uv sync --extra serve
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-0.8b@v1.0 --host 127.0.0.1 --port 8009
```

The server needs Hugging Face to download weights. It runs on any Apple Silicon Mac or a 4 GB GPU,
and on CPU more slowly. Point this repo at it with `KEV_BASE_URL`, which defaults to
`http://127.0.0.1:8009`.

**First milestone** (one observation, then a Kev decision, then the executed command, then the next
observation):

```bash
python -m hsc.milestone                            # Kev, raw actions; logs to results/milestone.jsonl
python -m hsc.milestone --decider rule_discrete    # same loop with no model
```

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

# local Kev server
python -m hsc.experiment --configs kev_discrete,kev --seeds 3
# Kev plus ANTHROPIC_API_KEY for the upper layers
python -m hsc.experiment --configs hier_kev,hier_kev_discrete --seeds 3

# hosted Jev instead: TYPESAFE_API_KEY and network access to api.typesafe.ai
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
| `rule_discrete` | fixed brief | none | rules limited to the 7 raw actions |
| `kev_discrete` | fixed brief | none | Kev, raw actions (first-version plan) |
| `kev` | fixed brief | none | Kev, guided maneuvers |
| `hier_kev` / `hier_kev_discrete` | Opus | Haiku | Kev, guided / raw actions |

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
- **Raw actions** (`results/baseline-rule-discrete/`): the rule controller limited to the 7 raw
  actions at half throttle still catches the Lady against `lg0` and `lg1` on `i2`. Its closest Guard
  approach is about half that of the guided version, and the LG7 proxy catches it every time.
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
hsc/kspdg_loop.py  real KSPDG + local Kev loop (7 inertial actions, validation, JSONL trace)
hsc/runner.py   latency-aware episode loop
hsc/experiment.py  named configs, CLI, summary table, plots
tests/          sim, runner and API-shape tests (no network)
```

Opus calls opt into the server-side refusal fallback (`fallbacks: "default"`). Jev request and
response fields follow TypeSafe's public docs. Check them against your account's API reference if a
call fails.
