# Experiment plan

**Question:** does slower strategic/tactical guidance improve mission outcomes while keeping immediate decisions responsive?

## Milestones and evidence

| Milestone | Evidence required |
| --- | --- |
| Reproduce environment | Host, revisions, launch command, actual scenario and effective rules |
| Verify local model | Saved `/v1/models` metadata and a real Choice request/response |
| Connect real control | Real observation → Kev response → validated action → next observation, in one trace |
| Complete episode | Terminal outcome, trajectory, fuel and timing records |
| Test hierarchy | Repeated matched trials against the baselines below |

A fixture test proves software plumbing. A real-game loop proves integration. Neither alone demonstrates better navigation or a benefit from hierarchy.

## Compare like with like

Compare a deterministic seven-action controller, Kev with fixed mission guidance, and Kev with mission/tactical guidance. Keep the same simulator, initial conditions, Guard policy, action catalog, executor, thrust/burn limits, shutdown rules and execution harness.

Heiko's continuous-force PD controller is useful for environment bring-up. If it is compared against discrete Kev actions, disclose the different action spaces. Likewise, a guided maneuver executor and raw thrust selection are different experiments. Do not attribute their difference solely to model quality or hierarchy.

Use the same predefined initial-condition/seed set for every controller where supported. Record the number of episodes and all failures. Choose trial count before running; this prototype has no justified universal trial count. Keep the scenario name distinct from any proxy Guard used in the stand-in simulation.

## Measure

- Capture, interception, simulator timeout, runner timeout and error counts.
- Closest Lady–Bandit and Bandit–Guard distances, fuel change and terminal weighted score.
- End-to-end decision latency, median and p95, including preparation, request, parsing and validation.
- Model-reported latency separately from client latency; cold start separately from warmed requests.
- Observation age at application, action duration, planner latency/context age, and rejected or stale decisions.

Keep mission time and monotonic wall time separate. Record what the spacecraft does while inference is pending: coast, repeat prior burn, or another explicitly implemented behavior. Model calls that take seconds cannot be described as a measured millisecond control loop.

Fixture latency experiments must state whether wall-clock inference delays advance simulated time. An older one-step Kev fixture trace advances mission time by about 1 second while the decision takes about 6.44 seconds; that trace is not a real-time spacecraft latency benchmark.

The newer stored synthetic check reports 20 timed requests, median 1.97 s and p95 2.30 s, all choosing `COAST`. These are one host's artifact results, not independently reproduced measurements or evidence of useful control. Do not average them with the older fixture latency.

## Trace contract

Save JSONL locally with these record types:

1. `run_header`: run ID, environment kind (`kspdg`, `fixture`, or `mock`), scenario/Guard implementation, initial condition/seed, revisions, model metadata, effective configuration and both timeout definitions.
2. `warmup` and `observation`, when present: identify discarded warm-up calls and the initial reset observation separately.
3. `decision`: step index, observation and sampling/application times, derived features, action ID, confidence/probabilities, numerical command and frame/units, context version/age, request/client latency, next observation, terminal/error information.
4. `run_end`: steps completed, terminal reason and artifact locations.

Identify synthetic latency and mocked responses explicitly. Publish sanitized logs with no keys, private host paths or personal recordings.

## Result summary

Use one row per controller/environment combination:

| Controller | Real game or fixture | Scenario / Guard | Trials | Captures / interceptions / timeouts / errors | Closest distances | Fuel change | Decision median / p95 | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pending | Pending | Pending | Not run here | Pending | Pending | Pending | Pending | Pending |

For the final demo, show a trajectory or replay, the chosen actions and planner context, one outcome summary, and the limits of the comparison. Record unsuccessful runs as well as successes.
