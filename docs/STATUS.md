# Status

Updated **October 11, 2026**. Real-game Kev control and a hierarchy comparison remain pending.

## Progress

| Component | Current evidence | Next step |
| --- | --- | --- |
| Linux KSP with programmed pursuit | Reported working by the team | Save the running scenario, launch command and state/action trace |
| KSPDG bridge | [PR #1](https://github.com/zihuahe/hierarchical-spacecraft-control/pull/1) adds observation-only mode, effective-rule logging, action validation and kRPC preflight | Connect to the game host |
| Local Kev API | [Synthetic-state check](https://github.com/zihuahe/hierarchical-spacecraft-control/blob/70412e311d9ba7bb5045b1b061cf10282b480c1d/results/kev-local/kev_check.json): 20 timed requests after two warm-ups, all `COAST`; median 1.97 s, p95 2.30 s | Test action selection on real observations |
| Fixture integration | [Fixture trace](https://github.com/zihuahe/hierarchical-spacecraft-control/blob/70412e311d9ba7bb5045b1b061cf10282b480c1d/results/kev-fixture-check/20261011-202159-bb3c44.jsonl): 20 completed steps, ending in `model_timeout` | Complete a real-game episode |
| Hierarchical planning | Proposed controller work in [PR #2](https://github.com/zihuahe/hierarchical-spacecraft-control/pull/2) | Compare against programmed control and Kev alone |

The stored Kev results use synthetic states or a fixture, not KSP gameplay. They establish API integration, not navigation performance. All-`COAST` responses leave pretrained action quality unresolved.

## Integration decisions

- **Scenario:** target `LBG1_LG7_I2_V1`. The cluster's I2 save does not identify its Guard policy. PR #1 reports constructing LG7 under KSPDG 0.15.0 with a 5 m Lady capture distance and 10 m Bandit interception distance. Confirm the effective rules on the game host.
- **Harness:** the standard agent runner can repeat prior actions during inference; the direct bridge uses a different step/request sequence. Use the same harness across comparisons.
- **Timing:** the cluster command's 150-second wall-clock runner limit differs from the 240-second mission limit. Measure burn length, inference delay and observation age on the selected host.
- **Rendering:** game display and Mac reproduction remain open. Trajectory logs can support debugging.

## Next milestone

One real observation → Kev decision → validated command → next observation, saved in a trace. Then complete an episode and run matched controller comparisons.

The setup guide pins PR #1 at `70412e311d9ba7bb5045b1b061cf10282b480c1d`; PR #2 was reviewed at `a5488389cc12185fe6087ff6d6d75ac2650f6973`. Both contain proposed work. Reconcile overlapping changes when merging and update this status as real-game evidence becomes available.
