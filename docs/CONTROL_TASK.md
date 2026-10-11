# Controlling the Bandit

Adapted from Heiko's supplied `CONTROL_TASK.MD` and checked against the public KSPDG interface. Paths below are relative to `spacegym-kspdg/src/kspdg/` unless stated. Heiko's `run_agent.py` belongs to the prepared cluster setup and is not present on the reviewed team `main` branch.

## Mission and score

Steer Bandit toward Lady while avoiding Guard. Public LBG1 defaults are capture at a distance strictly below 5 m and timeout when mission elapsed time exceeds 240 seconds. These base defaults do not establish LG7's rules. PR #1 reports that an installed KSPDG 0.15.0 LG7 instance has a **5 m Lady capture distance and 10 m Bandit interception distance**. That inspection did not include a real game reset. Read the loaded environment's actual thresholds and termination flags, and record them before a run.

The terminal weighted score is:

```text
score = min_lady_bandit_distance² + 1,000,000 / (min_bandit_guard_distance + 0.1)
reward = -score
```

Lower score is better. The distance terms are minima over the episode, not just final distances. Default reward is terminal; do not assume each burn receives a useful learning signal. Capture/interception and this score are separate measures.

Our originally selected `LBG1_LG7_I2_V1` is not confirmed as the running instance. `MISSION=lbg1_i2` selects initial conditions. LG0/LG1/LG2 have public implementations; LG7 needs the matching obfuscated environment build and Julia dependencies. PR #1 reports loading it with KSPDG 0.15.0 on Python 3.12/Linux; game connectivity remains unverified here. Confirm the Guard policy before comparing results.

## Observations: nine groups, 21 numbers

```text
KSP → kRPC → environment.get_observation()
    → runner pipe → agent.get_action(observation=observation)
```

| Python index | Value | Units / meaning |
| --- | --- | --- |
| `0` | Mission elapsed time | Seconds |
| `1` | Bandit mass | Kilograms |
| `2` | Bandit monopropellant | Resource reading; see note below |
| `3:6` | Bandit position | Metres |
| `6:9` | Bandit velocity | Metres/second |
| `9:12` | Lady position | Metres |
| `12:15` | Lady velocity | Metres/second |
| `15:18` | Guard position | Metres |
| `18:21` | Guard velocity | Metres/second |

Positions and velocities use **right-handed celestial-body-centred inertial coordinates**, abbreviated rhCBCI. Positions are measured from the planet's centre, not from Bandit. Attitude and angular rate are not included.

The upstream docstring labels propellant as kg, but the implementation directly reads `resources.amount('MonoPropellant')`. Keep it as a raw resource reading until units/conversion are confirmed for the running version. Do not silently equate it to spacecraft mass.

Calculate relative features in code, not in the language model:

```text
r_lady = lady_position - bandit_position
v_lady = lady_velocity - bandit_velocity
distance = norm(r_lady)
closing_speed = -dot(r_lady, v_lady) / distance
time_remaining = max(0, configured_mission_timeout - mission_time)
```

Positive closing speed means approaching. Define closing speed as zero when distance is near zero. Compute the same features for Guard, and track recent observations/actions and fuel change. Reject malformed or non-finite observations.

## Existing programmed controller

Heiko describes `PDPursuit` as:

```python
r = observation[9:12] - observation[3:6]
v = observation[12:15] - observation[6:9]
F = observation[1] * (kp * r + kd * v)
n = max(abs(F))
if n > 3500:
    F *= 3500 / n
```

It returns `burn_vec=[Fx, Fy, Fz, dt]`, `vec_type=1`, `ref_frame=1`. Reported first guesses are `kp=2e-4`, `kd=2e-2`, and `dt=0.5` seconds. It pursues Lady and damps relative motion; it **ignores Guard**. Keep it as a baseline, not proof of collision avoidance.

The vector rescaling preserves direction. The 3500 N cap is on inertial components; actual force feasibility depends on transformation into the spacecraft body frame and its per-axis limits.

## Actions: three axes plus duration

```json
{"burn_vec": [1000, 0, 0, 0.5], "ref_frame": 1, "vec_type": 1}
```

This is an illustrative 1000 N command along inertial X for 0.5 seconds. It is not an observed model output.

| Field | Meaning |
| --- | --- |
| `burn_vec[0:3]` | Three command components |
| `burn_vec[3]` | Burn duration, wall-clock seconds |
| `ref_frame=0` | Spacecraft body axes: forward, right, down |
| `ref_frame=1` | rhCBCI inertial axes, matching observations |
| `ref_frame=2` | NTW orbital frame; Y follows velocity, Z is orbit normal |
| `vec_type=0` | Dimensionless throttle; body-axis range −1 to +1 |
| `vec_type=1` | Force in newtons; converted to body-axis throttle internally |

Always set `ref_frame` and `vec_type`. Force and throttle are different units. The fourth number is duration, not a fourth engine. Zero thrust means coasting.

Public LBG1 nominal body-axis limits are 8 kN forward/reverse and 4 kN laterally/vertically. Out-of-range commands need not raise an error, but may not be achieved. Validate finite values, positive bounded duration and feasible transformed commands. When rescaling a force, use one factor to preserve direction. The declared action duration range is 0–10 seconds; the adapter should enforce its configured bound rather than rely on simulator rejection.

### Kev's first action catalog

Kev selects an ID; Python supplies the numerical command. The proposed direct bridge uses `ref_frame=1`, `vec_type=0` and configurable strength/duration:

| ID | Throttle components |
| --- | --- |
| `COAST` | `[0, 0, 0]` |
| `POS_X` / `NEG_X` | `[±h, 0, 0]` |
| `POS_Y` / `NEG_Y` | `[0, ±h, 0]` |
| `POS_Z` / `NEG_Z` | `[0, 0, ±h]` |

Do not call these forward/right/up: they are inertial directions. Some stand-in experiments intentionally use only two body axes; that limitation does not describe the real three-axis KSPDG API.

Send one `choice` question named `action` to `POST /v1/systemone`, with mission guidance, numeric features and optional planner context in `state`. Read `answers.action.choice`; reject unknown IDs and malformed/non-finite response data. Confidence is not a probability of mission success. [The example request](examples/kev-choice.json) is a transport smoke test, not a flight observation.

## Agent boundary and timing

In the standard runner, subclass `KSPDGBaseAgent` and implement `get_action(self, observation)`. The keyword parameter must be named `observation`. The environment process owns kRPC; the agent receives observations through a pipe. Do not open another kRPC connection inside that agent.

The environment sets thrust, waits for the wall-clock burn duration or termination, zeroes thrust, and reads state. The interface loop can re-run the last received action while the policy computes. Returning `None` means **keep the previous action**, not coast or stop. A new observation is served between steps; record when it was sampled and how old it is when a decision is applied. This timing does not mean the spacecraft freezes while the model computes.

The proposed PR #1 bridge instead calls `env.reset()` and `env.step()` directly and waits for each step to return before the next model request. These paths have different during-inference behavior. The latest bridge reads effective scenario settings and environment capture flags; it also adds an observation-only mode, discarded warm-up requests, kRPC routing and request timeouts. Pick one harness and use it consistently in matched experiments. A fixture that advances only on `step()` does not reproduce real wall-clock motion automatically.

Use finite request timeouts. On invalid input, stale decisions or model failure, record the reason and use the chosen runner/environment's supported termination and cleanup. Test that old actions stop. Returning `None` or merely abandoning the agent loop does not prove safe shutdown.

## Planner interface, after the first loop

Mission guidance contains objective, priorities and constraints. Tactical guidance contains a short-term objective, mode, limits, version, creation time and expiry. The fast controller reads the latest valid context without waiting for a planner call. Log the context version used for each action. Keep the action catalog and executor unchanged across the Kev-only and hierarchical comparison.

## Sources

- [LBG1 observations, scoring and limits](https://github.com/mit-ll/spacegym-kspdg/blob/main/src/kspdg/lbg1/lbg1_base.py)
- [Action conversion and burn timing](https://github.com/mit-ll/spacegym-kspdg/blob/main/src/kspdg/base_envs.py)
- [Agent runner](https://github.com/mit-ll/spacegym-kspdg/blob/main/src/kspdg/agent_api/runner.py) and [environment interface](https://github.com/mit-ll/spacegym-kspdg/blob/main/src/kspdg/agent_api/ksp_interface.py)
- [Scenario exports and optional dependencies](https://github.com/mit-ll/spacegym-kspdg/blob/main/src/kspdg/__init__.py)

- [Reviewed real-loop implementation and installed-version findings](https://github.com/zihuahe/hierarchical-spacecraft-control/blob/70412e311d9ba7bb5045b1b061cf10282b480c1d/README.md)
