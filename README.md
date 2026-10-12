# Hierarchical AI for Autonomous Spacecraft

**Sundai Hack 144 · Hack144: Autonomous Spacecraft**

Can a slow, capable planner and a fast, small decision model fly a spacecraft together? This project
tests a three-layer controller on Lady-Bandit-Guard, a scenario from KSPDG (the Kerbal Space Program
Differential Games suite). We fly the Bandit. The goal is to reach the Lady without being
intercepted by the Guard.

- **Play the game:** <https://zihuahe.github.io/hierarchical-spacecraft-control/>. You fly the Bandit
  yourself in a browser, on a computer or a phone.
- **Project page:** [Sundai Club project](https://www.sundai.club/projects/2f46263e-d3db-45c1-8091-df5b2683bf22)
- **Code:** the [`claude/project-thread-w6vikd`](https://github.com/zihuahe/hierarchical-spacecraft-control/tree/claude/project-thread-w6vikd) branch has the controller, tests, results and the game source. This `main` branch holds only this README.
- **Replay video:** [Kev-0.8B vs a deterministic baseline](https://github.com/zihuahe/hierarchical-spacecraft-control/blob/claude/project-thread-w6vikd/results/replay/kev_vs_rule_fixture.mp4)
  (on the stand-in simulator)

## Where it stands (2026-10-12)

The real-game control loop is written and tested, and the local Kev-0.8B model answers its requests
correctly. The loop has not yet run against Kerbal Space Program itself, because the cloud machine we
built on has no KSP install with kRPC. Every result marked **fixture** ran on `lbg1_lite`, a
pure-Python stand-in for the game, not on KSP.

| Stage | Done when | Status |
|---|---|---|
| A. Environment | One real observation and the effective scenario settings are recorded. | **Blocked.** Needs a host running KSP with kRPC. KSPDG 0.15.0 installs, the scenario builds, and its effective rules were read from the environment: 240 s timeout, 5 m capture, 10 m interception. |
| B. Local model | A valid structured response returns from the local Kev server. | **Done.** A real Kev-0.8B server on CPU answered 22 of 22 Choice requests with valid action IDs. Median latency was 1.97 s, p95 2.30 s. |
| C. First control loop | One validated action executes and the next observation is recorded. | **Fixture only.** It works end to end on `lbg1_lite` with real Kev, but has not run in KSP. |
| D. Episode | An actual episode has a local trace and a reported outcome. | **Fixture only.** Kev coasted all 171 steps and the proxy Guard intercepted the Bandit at 170.2 s. |
| E. Hierarchy experiment | Comparable trials with latency and outcome measurements are available. | **Not started** on the real loop. No-model latency experiments ran on the fixture. |

Also built: a deterministic baseline over the same seven actions (`--decider rule`), a replay renderer
for loop traces, and the browser game. The game's physics is checked against the Python simulator by
the test suite.

**What we learned so far.** Pretrained Kev-0.8B is wired up correctly but shows no sign of steering:
every answer was `COAST`, at about 0.27 probability against 0.14 for a uniform guess. The
deterministic baseline closed to 1,588 m of the Lady, but at 0.2 throttle it can't outrun the proxy
Guard either. Neither run tests the hierarchy hypothesis yet.

## 1. Goal and original pitch

- **Mission:** control the Bandit to reach the Lady while avoiding interception by the Guard.
- **Scenario:** `LBG1_LG7_I2_V1`, with a 240-second mission.
- **Immediate model:** local Kev-0.8B, pretrained, with no fine-tuning.
- **First milestone:** real observation → Kev decision → validated thrust command → next observation.
- **Presentation goal:** show a working controller and, if integration permitted, compare it with
  hierarchical guidance at the 20:00 final presentations.

### Heiko's original pitch: hierarchical spacecraft control

> For the presented space control game, use a hierarchical controller: a global policy (e.g., Opus)
> encodes a context (mission objectives etc.), a mid-level controller (e.g., Haiku) takes several
> images and creates a short-term set of objectives on a rolling basis, and a low-level controller
> (e.g., JEV) selects actions based on the higher-level contexts quickly.

### Current adaptation

We keep the three-layer idea, but start with numerical spacecraft observations and Kev-0.8B as the
open-source immediate decision model. Screenshots and the higher-level planners come after the basic
connection works. The original pitch remains the project direction; the first deliverable is
deliberately smaller.

### Hypothesis

Slower planners can set strategy and update short-term objectives while a fast decision model handles
immediate actions. We need to measure whether this separation improves control while keeping decision
latency manageable. A Kev-only run proves integration, not the hierarchy hypothesis.

## 2. Spacecraft task

Lady-Bandit-Guard is a KSPDG scenario that runs inside Kerbal Space Program. We control the Bandit.
The Lady is the target and the Guard is the opponent. Capture means crossing a distance threshold,
not docking or colliding.

| Outcome | Rule | Value in `LBG1_LG7_I2_V1` (read from KSPDG 0.15.0) |
|---|---|---|
| Capture | The Bandit gets within `lady_capture_dist` of the Lady. | 5 m |
| Interception | The Guard gets within `bandit_capture_dist` of the Bandit. | **10 m.** The base default is 5 m, but LG7 overrides it. |
| Timeout | Mission time reaches `episode_timeout`. | 240 s |

Use the environment's own termination result and metrics. Don't infer success from a screenshot, and
don't substitute a different scenario without saying so.

## 3. Controller inputs and outputs

### Inputs

Each observation has nine input groups with 21 numbers in total. Positions are in meters and
velocities in meters per second, all in the same right-handed celestial-body-centered inertial frame
(CBCI).

| Index | Input | Shape and units | Meaning |
|---|---|---|---|
| 0 | Mission time | 1 number; s | Elapsed mission time. |
| 1 | Bandit mass | 1 number; kg | Current spacecraft mass. |
| 2 | Bandit fuel | 1 number; KSP resource units | Remaining MonoPropellant. The upstream docstring says kg, but the value is in KSP resource units. |
| 3:6 | Bandit position | 3 numbers; m | Our position: x, y, z. |
| 6:9 | Bandit velocity | 3 numbers; m/s | Our velocity: vx, vy, vz. |
| 9:12 | Lady position | 3 numbers; m | Target position. |
| 12:15 | Lady velocity | 3 numbers; m/s | Target velocity. |
| 15:18 | Guard position | 3 numbers; m | Opponent position. |
| 18:21 | Guard velocity | 3 numbers; m/s | Opponent velocity. |

**Features computed in code,** for both the Lady and the Guard:

- Time remaining: `max(0, episode_timeout − mission time)`.
- Relative position: target position − Bandit position.
- Relative velocity: target velocity − Bandit velocity.
- Distance: the length of the relative position.
- Closing speed: `−dot(relative position, relative velocity) / distance`. Positive means approaching.
  The zero-distance case is handled.
- Recent state and action history, and fuel used, from locally stored previous readings.

The state sent to Kev stays compact and includes units. Kev is never asked to do the vector maths.
The 21-number observation has no attitude and no screenshots.

### Outputs

A simulator command holds a three-axis burn vector with a duration, a reference frame and a vector
type:

```json
{"burn_vec": [x, y, z, duration], "ref_frame": 1, "vec_type": 0}
```

Throttle runs from −1 to +1 on each axis, and duration from 0 to 10 s. `ref_frame = 1` is right-handed
CBCI, the same axes as the observation. `vec_type = 0` means throttle. The environment rotates the
vector into spacecraft controls.

`ref_frame = 0` is the spacecraft's body frame, which the first action catalog doesn't use. Its axes
are not interchangeable with inertial x, y and z:

| Body axis | Range | Sign |
|---|---|---|
| Forward / backward | −1 to +1 | Positive is forward. |
| Right / left | −1 to +1 | Positive is right. |
| Down / up | −1 to +1 | Positive is down. |
| Duration | 0 to 10 s | |

Things to keep in mind:

- Zero thrust doesn't stop the spacecraft. It coasts.
- The forward and backward axes have 8 thrusters and the others have 4. The same inertial throttle can
  therefore give a different acceleration depending on attitude.
- Mission time and wall time are different clocks, and the burn runs for `duration` seconds of
  **wall-clock** time. The loop logs both clocks.

### Action catalog

Seven inertial-axis actions, so no attitude calculation is needed. The magnitude (0.2) and duration
(1.0 s) are configurable connection-test settings, not a demonstrated flight policy.

| Action ID | Burn vector: x, y, z, duration |
|---|---|
| `COAST` | `[0, 0, 0, 1.0]` |
| `POS_X` | `[0.2, 0, 0, 1.0]` |
| `NEG_X` | `[−0.2, 0, 0, 1.0]` |
| `POS_Y` | `[0, 0.2, 0, 1.0]` |
| `NEG_Y` | `[0, −0.2, 0, 1.0]` |
| `POS_Z` | `[0, 0, 0.2, 1.0]` |
| `NEG_Z` | `[0, 0, −0.2, 1.0]` |

Kev picks only an action ID. The adapter looks up the vector, adds `ref_frame = 1` and `vec_type = 0`,
and checks the ID, finite values, limits and duration before anything executes.

## 4. Controller architecture

| Layer | Responsibility | Proposed model | As built |
|---|---|---|---|
| Mission strategy | Set the mission, priorities and constraints. | Claude Opus | The real loop uses fixed mission text built from the scenario's effective rules. An Opus strategist is implemented for the fixture experiments. |
| Tactical planning | Review recent states and update a short-term objective. | Claude Haiku | Implemented for the fixture experiments, but not yet on the real loop. |
| Immediate decisions | Pick one option from the fixed action catalog. | Kev-0.8B | **Done.** A local server answers one Choice question per observation. |
| Decision engine / adapter | Prepare features, validate the choice, map it to a command and log it. | None, plain Python | [`hsc/kspdg_loop.py`](https://github.com/zihuahe/hierarchical-spacecraft-control/blob/claude/project-thread-w6vikd/hsc/kspdg_loop.py) |

Jev is TypeSafe's original typed-decision model. Kev is an independent open-source alternative that
serves the same API. A compatible interface doesn't mean the same accuracy, confidence or performance.

### Adding the hierarchy after the first loop

The comparison keeps the same observation adapter and action catalog, so it isolates the value of
guidance. Using local Kev for immediate decisions doesn't make hosted Opus or Haiku inference offline.

| Layer | Proposed output | Update policy |
|---|---|---|
| Mission strategy | Mission objective, priorities and constraints, plus a strategy version. | Once per episode at first. |
| Tactical guidance | Short-term objective, preferred and avoided actions, context version and expiry. | Rolling updates from recent numerical history, on an interval chosen after measuring latency. |
| Immediate selection | One action ID from the fixed catalog. | One validated decision per control step. |

These schemas and schedules are proposed interfaces. Slow planners run separately from the immediate
loop. The loop keeps the latest valid context, discards expired or stale results, and logs the context
age and version with every action. Screenshots remain optional later work.

## 5. Build plan

**Priorities.** P0: a real one-action trace, then a reproducible episode. P1: a deterministic and
Kev-only comparison, then the hierarchy if P0 works. P2: screenshots and a frontend. There is no
fine-tuning in this hack.

### Handoff checklist

| Item | Status |
|---|---|
| Environment host, launch command and connection verified | **Blocked.** No KSP with kRPC on the build machine. |
| Selected scenario and effective rules recorded | **Done** from the KSPDG environment object (240 s, 5 m, 10 m), not yet from a live game. |
| Local Kev Choice request verified | **Done.** [`results/kev-local/kev_check.json`](https://github.com/zihuahe/hierarchical-spacecraft-control/blob/claude/project-thread-w6vikd/results/kev-local/kev_check.json) |
| Real one-action control loop and trace captured | **Fixture only.** [`results/kev-fixture-check/`](https://github.com/zihuahe/hierarchical-spacecraft-control/tree/claude/project-thread-w6vikd/results/kev-fixture-check/) |
| Episode run and metrics captured | **Fixture only.** Kev and baseline traces, plus the replay video. |
| Hierarchy comparison completed | **Not started** on the real loop. |

### Environment facts

| Fact | Recorded |
|---|---|
| Simulator host OS, architecture and where the game runs | Not yet. It needs a host with KSP. |
| Repository and extracted KSP paths on that host | Not yet. |
| Launch command, Python version and KSPDG revision | Python 3.12; `kspdg` 0.15.0 (`mit-ll/spacegym-kspdg` at `6251ca3`); `python -m hsc.kspdg_loop --env kspdg --steps 0` |
| kRPC address and ports | KSPDG connects with kRPC's defaults, `127.0.0.1:50000` (RPC) and `50001` (stream). Container networking is not set up yet. |
| Loaded scenario, effective timeout and capture distances | `LBG1_LG7_I2_V1`, 240 s, 5 m capture, 10 m interception. These were read from the environment object, not yet from a live game. |

### Setup

Heiko's original setup approach was to clone both speaker repositories, unzip the supplied KSP game
next to them, and install Python, Docker and uv. Neighboring folders alone don't configure mounts or
networking, so follow the checked-out runtime guide. The verified commands for KSPDG and the loop are
in [Exact commands](https://github.com/zihuahe/hierarchical-spacecraft-control/blob/claude/project-thread-w6vikd/README.md#exact-commands).

**Kev, local.** The selected checkpoint is `jaredpalmer/kev-0.8b@v1.0`. It has downloadable weights
and Apache-2.0 licensing, and returns typed decisions with probabilities instead of generated prose.
Its general benchmarks say nothing about spacecraft competence, so we measure this task directly. Use
Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/). Don't use `pip install kev`, which is an
unrelated package.

```bash
git clone https://github.com/jaredpalmer/kev.git && cd kev
uv sync --extra serve
HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 uv run --extra serve python -m kev.serve \
    --run jaredpalmer/kev-0.8b@v1.0 --host 127.0.0.1 --port 8009
curl --fail http://127.0.0.1:8009/v1/models   # metadata means the server is up
```

A metadata response confirms the server is reachable, and a real Choice request confirms inference.
If the controller runs inside Docker, its `localhost` is the container, so point `--kev-base-url` at
the host.

<details>
<summary>Choice request and response</summary>

`POST /v1/systemone` with compact numerical features in `state` and one `action` Choice question:

```json
{
  "model": "kev-latest",
  "state": "Mission objective and numerical spacecraft features go here.",
  "questions": {
    "action": {
      "type": "choice",
      "instructions": "Choose the next short burn to approach Lady while avoiding Guard. Use the stated inertial coordinate frame.",
      "criteria": {
        "COAST": "No thrust for this interval.",
        "POS_X": "Positive inertial X throttle.",
        "NEG_X": "Negative inertial X throttle.",
        "POS_Y": "Positive inertial Y throttle.",
        "NEG_Y": "Negative inertial Y throttle.",
        "POS_Z": "Positive inertial Z throttle.",
        "NEG_Z": "Negative inertial Z throttle."
      }
    }
  }
}
```

Read `answers.action.choice`. The loop logs the returned confidence and probabilities, but they are
not a calibrated chance of mission success, and the answer is not itself a simulator command.

</details>

### One sequential control loop

1. Read and validate a fresh observation.
2. Calculate features and prepare a compact model state.
3. Request one action from Kev, with a finite timeout.
4. Check the response, the action ID and the observation's age.
5. Map the ID to a validated simulator command and execute it.
6. Record the next observation and episode metrics, and repeat only while the simulator is active.

Burns never overlap. On an invalid input, a timeout, a stale decision or a lost connection, the loop
stops with a clear reason and cleans up the environment. It never executes a guessed fallback, because
coasting isn't always safe.

### Configuration and trace

The scenario, episode timeout, thrust magnitude and duration, model endpoint, request timeout, maximum
observation age and history length are all flags. Timing thresholds should be set from measurements on
the host, not treated as verified defaults. Each run writes a local JSONL trace with a run header
(configuration, effective rules, dependency and model revisions) and one record per decision: step,
both clocks, raw observation, features, context version and age, choice and confidence, burn command,
next observation, latency, and the terminal or error reason. Details are in the
[technical reference](https://github.com/zihuahe/hierarchical-spacecraft-control/blob/claude/project-thread-w6vikd/README.md#action-catalog-and-request).

<details>
<summary>Coding-agent handoff (from the build brief)</summary>

Implement the smallest real KSPDG + local Kev-0.8B control loop in the existing team repository.

- Inspect the repository instructions and existing code first. Use sections 2–5 as the interface
  specification, and check the installed upstream version before assuming signatures or settings.
- Keep the real KSPDG scenario as the execution target. Don't replace it with a generated simulator, a
  fake model response or an unannounced alternate scenario.
- Reuse the working environment setup. Discover host paths, launch commands and connection details
  where possible, and ask only for runtime facts that can't be discovered.
- First prove local Kev inference on its own. Implement feature preparation, one Choice request, a
  fixed action mapping, validation and local JSONL logging.
- Expose a one-action connection test and an episode mode. Use the environment's actual
  reset/step/termination API rather than assuming the standard Gym return format.
- Handle finite request timeouts, invalid observations or responses, stale results and connection
  errors explicitly. Keep inference and logging local, and turn off cloud experiment logging.
- Write focused checks for observation parsing, action mapping and invalid-response handling, and label
  fixtures and mocks clearly.
- Record the exact working startup and run commands, dependency revisions and model revision in the
  README.
- Produce a trace showing a real observation, a real Kev selection, the executed command and the next
  observation. Then run and report an episode if the simulator is available.
- If the simulator is blocked, finish the model and adapter checks, report the exact blocker, and keep
  the real-run task open.

Acceptance: a real observation → local Kev decision → valid command → next observation, with a
reproducible command and trace. Integration success doesn't yet mean successful navigation or a
measured hierarchy benefit.

</details>

## 6. Demo and evaluation

| Comparison | Status |
|---|---|
| A simple deterministic controller using the same seven actions | Built (`--decider rule`). On the fixture it closed to 1,588 m of the Lady and was intercepted at 170.2 s. |
| Kev with the fixed mission objective | On the fixture, Kev coasted every step and was intercepted at 170.2 s. |
| Kev with mission strategy and rolling tactical guidance | Not run on the real loop yet. |

Comparisons keep the scenario, initial conditions, opponent, action catalog and command limits the
same, and repeat trials where feasible. A coast-only run checks wiring but is a weak performance
baseline, and a single lucky run is not proof of the hypothesis.

**What to record:** capture, interception or timeout, with the simulator's terminal metrics; closest
distances to the Lady and the Guard; fuel used; median and p95 end-to-end decision latency, measured
after warm-up on changing observations; invalid selections, timeouts, stale results and failed runs;
and model calls with planner and context versions. Simulation time stays separate from wall time.

**Presentation:** explain the Lady, Bandit and Guard and the chosen task; show the control loop and
which parts are Kev and which are ordinary code; play a replay or show a trajectory plot; present
measured outcomes and latency, then explain what remains unfinished. The replay video and the browser
game were built for this.

## 7. Materials

**Speaker materials**

- KSPDG Lady-Bandit-Guard scenario, in [mit-ll/spacegym-kspdg](https://github.com/mit-ll/spacegym-kspdg):
  the scenario description and the observation and action implementation.
- ARCLab KSPDG Docker setup, in [ARCLab-MIT/kspdg](https://github.com/ARCLab-MIT/kspdg): the setup entry
  point for the supplied Linux game environment. Some instructions may need adjusting.
- The supplied `KSP_linux.zip` archive with the Linux game files. A downloaded archive alone doesn't
  prove the scenario runs.
- A `docs/docker_execution.md` guide was suggested during setup. It isn't in `mit-ll/spacegym-kspdg`
  v0.15.0 or in `ARCLab-MIT/kspdg` main, so it presumably came with the speaker materials.

**Implementation references**

- KSPDG observation source, and action and frame-conversion source, in `mit-ll/spacegym-kspdg`.
- Kev: [source and setup](https://github.com/jaredpalmer/kev) and the
  [Kev-0.8B model card](https://huggingface.co/jaredpalmer/kev-0.8b) with its limitations.
- TypeSafe Jev, the original System One concept.

**Event.** Sundai Hack 144. The morning covered AI news, autonomous-spacecraft research, project
pitches and team formation, with a 14:00 lunch check-in, an optional 16:00 run or bike ride, and final
presentations at 20:00.

## Team and credits

- Built at Sundai Hack 144, hosted by [Sundai Club](https://www.sundai.club/). The team is listed on the
  [project page](https://www.sundai.club/projects/2f46263e-d3db-45c1-8091-df5b2683bf22).
- Original pitch by Heiko.
- Much of the code, the replay renderer and the browser game were written with
  [Claude Code](https://claude.com/claude-code).
