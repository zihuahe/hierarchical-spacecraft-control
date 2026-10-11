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

- **Raw actions (`lbg1_lite` experiments):** Kev returns one action ID: coast, forward, backward, left,
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

### Status (2026-10-11)

| Stage | Status | Evidence |
|---|---|---|
| A. Environment | **Blocked here.** `kspdg` 0.15.0 installs and `LBG1_LG7_I2_V1` builds; `reset()` needs KSP with kRPC, which this container doesn't have. | The loop exits with `env_unavailable: kRPC server not reachable at 127.0.0.1:50000` |
| B. Local model | **Done.** A real Kev-0.8B server on CPU answered 22 of the loop's own Choice requests, all valid. | [`results/kev-local/kev_check.json`](results/kev-local/kev_check.json) |
| C. First control loop | **Not run against KSP.** The loop ran with real Kev on the `lbg1_lite` test fixture only. | [`results/kev-fixture-check/`](results/kev-fixture-check/) (fixture, not the game) |
| D. Episode | Not run against KSP. | — |
| E. Hierarchy | Not started on the real loop. | — |

The KSP game archive (`KSP_linux.zip`) and the Docker guide (`docs/docker_execution.md`) are not in
`mit-ll/spacegym-kspdg` v0.15.0 or `ARCLab-MIT/kspdg` main. They came with the speaker materials.
On a host with the game, start with `--steps 0` (stage A), then `--steps 1` (stage C).

### What the installed KSPDG actually does

Checked against `kspdg` 0.15.0 (`mit-ll/spacegym-kspdg` at `6251ca3`, whose code matches tag `v0.15.0`, installed with
`[adv_bots,viz_tools]` on Python 3.12 and Linux x86_64):

- **LG7 rules differ from the base defaults.** `LBG1_LG7_I2_V1` resolves to the obfuscated
  `LBG1_LG7_I2_Env` (parent `LBG1_LG6_ParentEnv`). An instance reports `episode_timeout=240.0`,
  `lady_capture_dist=5.0` and **`bandit_capture_dist=10.0`**, so the Guard intercepts at 10 m, not
  5 m. `loadfile` is `lbg1_i2_init`. The loop reads these values from the env, records them in the
  run header as `scenario_settings`, and builds Kev's mission text from them.
- **Constructor:** `__init__(self, **kwargs)`. The loop always passes `episode_timeout`, then checks
  the value the env reports.
- **Without `juliacall`**, `kspdg.LBG1_LG7_I2_V1` is a placeholder that returns an error *string*.
  The loop reports this as `env_unavailable`. LG7's Guard runs Julia (`iLQGames`, `FromFile`):
  install Julia and run `kspdg-install-julia-deps`. `lbg1_base.py` imports `dearpygui`, so the
  `viz_tools` extra is needed too. Obfuscated environments ship for Python 3.9, 3.10, 3.12 and 3.13,
  not 3.11.
- **API:** `reset()` returns `(obs, info)` and connects through `krpc.connect(name='kspdg_env')`, which
  uses kRPC's defaults of `127.0.0.1:50000` (RPC) and `50001` (stream). `step(action)` returns
  `(obs, reward, done, info)`. The burn runs as a **wall-clock** busy-wait of `burn_vec[3]` seconds and
  stops early when the episode ends.
- **Termination:** a background thread sets `env.is_lady_captured` and `env.is_bandit_captured`, or
  times out when the Bandit's MET passes `episode_timeout`. These flags live on the env object, not in
  `info`. The loop reads them and reports `capture`, `interception`, `capture_and_interception`,
  `timeout` or `simulator_done`.
- **Fuel:** `obs[2]` is `resources.amount('MonoPropellant')`, which is in KSP resource units even though
  the docstring says kg. The loop keeps it as `bandit_fuel_raw` and `fuel_used_raw`, and also logs
  `mass_used_kg` from `obs[1]`. KSPDG's `info["bandit_fuel_usage"]` is a mass difference in kg.
- **Action:** a `Dict` of `burn_vec` (Box, −1..1 ×3 and 0..10 s), `ref_frame` (Discrete(3); 1 = right-handed
  CBCI) and `vec_type` (Discrete(3); 0 = throttle). With `ref_frame=1` the env rotates the vector into the
  body frame. The forward/reverse axes have 8 thrusters and the others 4, so the same inertial throttle
  can give different accelerations depending on attitude.

### Action catalog and request

Inertial axes (right-handed CBCI), `ref_frame=1`, `vec_type=0` throttle, 1.0 s:

| ID | burn_vec |
|---|---|
| `COAST` | `[0, 0, 0, 1.0]` |
| `POS_X` / `NEG_X` | `[±0.2, 0, 0, 1.0]` |
| `POS_Y` / `NEG_Y` | `[0, ±0.2, 0, 1.0]` |
| `POS_Z` / `NEG_Z` | `[0, 0, ±0.2, 1.0]` |

Kev gets `POST /v1/systemone` with `model: "kev-latest"`, a JSON `state`, and one `choice` question
named `action` over the seven IDs. The state holds the mission text from the effective rules and the
derived features: relative position and velocity, distance and closing speed for both the Lady and the
Guard, time remaining, fuel and mass used since the start, and recent history. The loop reads
`answers.action.choice`. It logs `confidence` and `probabilities`, but these are not a calibrated
chance of mission success.

**Stops, with no fallback burn:** an invalid observation (not 21 finite values); an unknown or malformed
answer; non-finite confidence or probabilities; a request timeout (`model_timeout`, enforced on the
whole request); an HTTP error such as Kev's 422 for oversized states (`model_http_error`); a connection
error; an observation older than `--max-observation-age-s` when the answer arrives (`stale_decision`);
an invalid command; or a simulator exception. The reason goes into the trace and the run summary.

**Trace:** `results/traces/<run_id>.jsonl`, with these records in order:

- `run_header`: effective config, `scenario_settings` read from the env, mission text, Python, platform,
  repo commit, `kspdg`, `krpc`, `gymnasium` and `juliacall` versions, and Kev's `/v1/models`.
- `warmup`: one per untimed warm-up request. The state is a labelled placeholder and the answer is
  discarded.
- `observation`: the reset observation and info, plus the reset wall time.
- `decision`, one per step: step, wall time, mission time, raw observation, features, model, context
  version and age, choice, confidence, probabilities, burn command, next observation, `mission_dt_s`
  and `step_wall_s` (the two clocks), model and decision latency, observation age, info, and the
  terminal or error reason.
- `run_end`.

### Exact commands

1. **Kev server.** Use Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/). Don't use
   `pip install kev`, which is an unrelated package. This was verified in this repo's cloud container
   (Linux x86_64, 4 vCPU, no GPU). Kev source was at `2ecafc49`. `@v1.0` resolved to Hub snapshot
   `bf75a6a8848ea6960ff2ed108d9ed44c2941174f`, a LoRA adapter on
   `Qwen/Qwen3.5-0.8B-Base@dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`.

   ```bash
   git clone https://github.com/jaredpalmer/kev.git && cd kev
   uv sync --extra serve
   HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 uv run --extra serve python -m kev.serve \
       --run jaredpalmer/kev-0.8b@v1.0 --host 127.0.0.1 --port 8009
   curl --fail http://127.0.0.1:8009/v1/models          # metadata: the server is up
   ```

   On CPU the server runs in float32 and logs `serving jaredpalmer/kev-0.8b@v1.0 (...) on cpu via torch
   (float32)`. The first start downloads about 7 GB of Python packages plus the weights.

2. **Kev inference check** (stage B, no game needed). This sends the loop's exact request on synthetic
   states and validates each answer with the loop's parser:

   ```bash
   pip install -r requirements.txt
   python -m hsc.kev_check --requests 20 --out results/kev-local/kev_check.json
   ```

3. **KSPDG** on the machine that runs the loop. Use Python 3.12 or 3.13, since 3.11 has no LG7 build:

   ```bash
   pip install "kspdg[full] @ git+https://github.com/mit-ll/spacegym-kspdg@v0.15.0"
   kspdg-install-ksp-files                      # copies mission saves into your KSP install
   curl -fsSL https://install.julialang.org | sh && kspdg-install-julia-deps   # LG7 Guard (Julia)
   ```

   Then start KSP with the KSPDG files, kRPC and PhysicsRangeExtender. Load a game, open the kRPC window,
   enable **Auto-accept new clients**, and start the server. The env loads its own save, `lbg1_i2_init`,
   on `reset()`. If the game runs in a container, publish ports 50000 and 50001 or pass
   `--krpc-address`. If the loop runs in Docker and Kev on the host, pass
   `--kev-base-url http://host.docker.internal:8009`. The first LG7 reset may compile Julia packages.

4. **Loop**, from this repo:

   ```bash
   python -m hsc.kspdg_loop --env kspdg --steps 0      # stage A: one observation + effective rules, no Kev, no burn
   python -m hsc.kspdg_loop --env kspdg --steps 1      # stage C: one real action, then the next observation
   python -m hsc.kspdg_loop --env kspdg --episode      # stage D: until KSPDG ends the episode
   ```

   Every setting is a flag. Defaults: `--scenario LBG1_LG7_I2_V1 --episode-timeout-s 240
   --thrust-magnitude 0.2 --burn-duration-s 1.0 --kev-base-url http://127.0.0.1:8009
   --kev-model kev-latest --request-timeout-s 10 --max-observation-age-s 15 --history-len 5
   --warmup-requests 1 --krpc-address 127.0.0.1 --krpc-rpc-port 50000 --krpc-stream-port 50001
   --trace-dir results/traces`. The timeout and age limits are not tuned. Measured Kev latency on a
   4-vCPU CPU is about 2 s, so 10 s and 15 s leave room. Set them from your host's numbers.

### Measured so far (Kev only, no game)

`results/kev-local/kev_check.json`: 20 timed requests after 2 warm-up requests, CPU float32.
Decision latency was **median 1.97 s, p95 2.30 s** (min 1.72, max 2.44), covering request building,
HTTP, inference and validation. Every answer was a valid action ID, and every one was **`COAST`**, at
about 0.27 probability against 0.14 for uniform. That held even with the Lady 2.5 km along +Y, where
`POS_Y` got about 0.10. Pretrained Kev-0.8B wires up correctly, but this check shows no sign that it
picks a direction from these features. A Kev-only run would probably coast, so compare it with the
deterministic baseline before reading anything into the outcomes.

`results/kev-fixture-check/` holds the same loop with real Kev on the `lbg1_lite` **test fixture**,
not KSP, to exercise the trace and stop logic end to end:

- `20261011-202335-4f08c1.jsonl`: a full fixture episode. All 171 decisions were valid `COAST` answers.
  Decision latency was median 2.72 s and p95 3.08 s; the 5-step history makes the state longer than in
  `kev_check`. The fixture's proxy Guard intercepted at mission time 170.2 s. The Bandit used no fuel
  and its closest Lady approach was 2,600 m, its starting distance. That is a wiring check on a
  stand-in, not a flight result.
- `20261011-202159-bb3c44.jsonl`: an earlier run that stopped at step 20 with `model_timeout`. One
  request took more than 10 s while other CPU-heavy work ran on the same 4 vCPUs. The loop sent no burn
  and closed the env.

`--env fixture` runs the loop on the `lbg1_lite` stand-in below. It is a **test fixture only**, not
the real game, and its LG7 Guard is a proxy that captures at 5 m, not LG7's 10 m.
`tests/test_kspdg_loop.py` covers parsing, the feature maths (including zero distance), all seven
mappings and limits, bad and non-finite responses, HTTP errors, timeouts, connection loss, stale
observations, effective-rule mission text, terminal reasons from env flags, the juliacall placeholder,
kRPC routing and reachability, both step formats, observation-only mode and cleanup. The tests use a
local stub Kev server and fake environments.

The older `python -m hsc.milestone` runs the same idea on `lbg1_lite` with the body-frame action set
used by the experiments below. It is a stand-in, not the stage C milestone.

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
| `kev_discrete` | fixed brief | none | Kev, raw body-frame actions |
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
hsc/kev_check.py   stage B check: the loop's Choice request against a local Kev server
hsc/runner.py   latency-aware episode loop
hsc/experiment.py  named configs, CLI, summary table, plots
tests/          sim, runner and API-shape tests (no network)
```

Opus calls opt into the server-side refusal fallback (`fallbacks: "default"`). Jev request and
response fields follow TypeSafe's public docs. Check them against your account's API reference if a
call fails.
