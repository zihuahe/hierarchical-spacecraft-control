# Status

Snapshot: **October 11, 2026**. This distinguishes meeting reports, inspected artifacts, proposed code and independently reproduced execution. No simulator or model was run during this documentation preparation.

## What we have

| Item | Evidence | Status |
| --- | --- | --- |
| Linux simulation and programmed `PDPursuit` | Team's latest meeting report and Heiko's supplied control explainer | Reported working; launch command, actual scenario and runtime trace not independently checked |
| Public `main` branch | Commit `57325775631bce1515aa02c7d49773b08f76c374` | Inspected: README title only |
| Proposed real-game bridge | [PR #1](https://github.com/zihuahe/hierarchical-spacecraft-control/pull/1), reviewed head `70412e311d9ba7bb5045b1b061cf10282b480c1d` | `hsc/kspdg_loop.py` and updated README inspected; effective rules, observation-only mode and kRPC preflight added; real execution not reproduced |
| Proposed controller/fixture code and results | [PR #2](https://github.com/zihuahe/hierarchical-spacecraft-control/pull/2), reviewed head `a5488389cc12185fe6087ff6d6d75ac2650f6973` | Source and stored trace inspected; not merged in the reviewed snapshot |
| Local Kev response on fixture state | [Stored one-step trace](https://github.com/zihuahe/hierarchical-spacecraft-control/blob/a5488389cc12185fe6087ff6d6d75ac2650f6973/results/kev-cpu/kspdg_loop_real_kev_choice_fixture_obs.jsonl) | Artifact labels `env: fixture`, CPU Kev-0.8B, `COAST`, about 6.44 s decision time, one step; ends at `max_steps_reached` |
| New synthetic Kev check | [Stored report](https://github.com/zihuahe/hierarchical-spacecraft-control/blob/70412e311d9ba7bb5045b1b061cf10282b480c1d/results/kev-local/kev_check.json) | 20 timed requests after two warm-ups; median 1.97 s / p95 2.30 s; all `COAST`; inspected, not reproduced |
| New fixture loop | [Stored trace](https://github.com/zihuahe/hierarchical-spacecraft-control/blob/70412e311d9ba7bb5045b1b061cf10282b480c1d/results/kev-fixture-check/20261011-202159-bb3c44.jsonl) | 20 completed steps; ends with `model_timeout`; fixture only, inspected, not reproduced |
| Real KSPDG controlled by Kev | No real-game trace inspected | Pending |
| Real-game hierarchy comparison | No matched result inspected | Pending |

The older one-step fixture trace records PR #1's earlier head `0235575bd56d65aac404954c24954534450f90ae`, while that artifact is stored in PR #2. Do not infer that either PR alone contains every integration change. Reconcile the branches before publishing a unified run command on `main`.

## Open decisions and blockers

- **Scenario:** the original target is `LBG1_LG7_I2_V1`. Heiko's mission save is I2; the running LG variant is not established. PR #1 now reports constructing LG7 with KSPDG 0.15.0 and its Julia/obfuscated dependencies, with Lady capture at 5 m and Bandit interception at 10 m. This is installed-environment evidence, not a real-game trace. Confirm host availability and effective rules or explicitly choose a supported variant.
- **Harness:** Heiko's standard agent runner repeats the prior action while computing; PR #1's direct bridge follows a different step/request sequence. Choose and document one real execution path.
- **Timing:** the supplied cluster command uses a 150-second wall-clock runner limit, distinct from the 240-second mission limit. Burn length and observation-age limits need measurement on the chosen host.
- **Visualization and reproduction:** game rendering/browser access and Mac setup remain unresolved in the meeting notes. Use trajectory logs if needed.
- **Integration ownership:** agree who owns runtime, Kev adapter, planners and evaluation. No named coding assignments are inferred from the transcript.

All-`COAST` synthetic responses demonstrate valid API outputs, not a useful pretrained policy. Treat model action quality as an open experiment before adding planner complexity.

## Next checkpoint

1. Record the running Linux scenario, exact commands and a real programmed-control trace.
2. Confirm Kev reachability from that runtime and replace action selection through the chosen harness.
3. Save one real Kev-controlled action and next observation, then a complete episode.
4. Add planners and perform matched comparisons only after integration is established.

Update this snapshot after a merge or new execution evidence. The presence of source code, a passing stub test, a fixture trace or a meeting report is not interchangeable with a reproduced real-game result.
