# Setup

**Source-checked instructions, not an independently reproduced installation.** Start with [status](STATUS.md) to choose a checkout and runtime. The reviewed `main` branch contains only a README title.

## 1. Choose the execution path

| Path | What it provides | What to verify |
| --- | --- | --- |
| Heiko's Linux/Slurm job | Existing KSP installation and `PDPursuit` agent | Job, mission, environment class, kRPC connection, and launch command |
| Speaker's Docker setup | Container setup for the supplied Linux game | Host architecture, mounts, mods, networking, and installed dependencies |
| Proposed PR #1 bridge | Direct `env.step()` loop with Kev | Correct checkout and real KSPDG dependencies |
| Stand-in fixture | Development without the game | Label results as fixture-only |

The original setup idea was to clone both speaker repositories, place the extracted game nearby, and install Python, Docker, and `uv`. Follow the checked-out runtime guide: neighboring folders alone do not configure mounts or networking. See the [speaker's Docker directory](https://github.com/ARCLab-MIT/kspdg-2026/tree/main/docker).

### Existing cluster job

Adapted from Heiko's supplied `CONTROL_TASK.MD`. Replace both placeholders. The directory must contain the prepared `ksp.sif`, `run_agent.py`, and job setup; this command does not create them or submit a new job.

```bash
cd <YOUR_CLUSTER_DIR>
mkdir -p cache tmp
SINGULARITY_CACHEDIR="$PWD/cache" SINGULARITY_TMPDIR="$PWD/tmp" \
srun --jobid=<YOUR_JOB_ID> --overlap -n1 \
  singularity exec --bind "$PWD:/w" ksp.sif \
  python3 /w/run_agent.py 150
```

`150` is this script's runner timeout in wall-clock seconds, not the mission's 240-second limit. Expected evidence: agent log lines containing mission time, target distance, and relative speed. Save those logs and one observation/action pair. A successful launch alone does not prove a completed mission.

Heiko reports `MISSION=lbg1_i2`. That identifies the initial-condition save, not the Guard policy. Confirm the actual environment class. Public LG0/LG1/LG2 classes have I2 variants; LG7 requires `juliacall`, Julia packages and a matching obfuscated environment build. PR #1 reports constructing an LG7 instance with KSPDG 0.15.0 on Python 3.12/Linux, without a real game connection. Its reported rules are Lady capture at 5 m and Bandit interception at 10 m; confirm these on your running host. Record an explicit scenario change if LG7 is unavailable.

## 2. Start local Kev

Upstream requires Python 3.12 or 3.13 and `uv`. Reuse an existing Kev checkout if present. First startup downloads pretrained weights. These commands adapt the [upstream instructions](https://github.com/jaredpalmer/kev#run-it-locally) to our selected 0.8B checkpoint.

```bash
cd <YOUR_WORKSPACE_DIR>
git clone https://github.com/jaredpalmer/kev.git
cd <YOUR_WORKSPACE_DIR>/kev
uv sync --extra serve
HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 \
uv run --extra serve python -m kev.serve \
  --run jaredpalmer/kev-0.8b@v1.0 --host 127.0.0.1 --port 8009
```

In a second terminal:

```bash
curl -fsS --max-time 60 http://127.0.0.1:8009/v1/models
```

Expect JSON describing the loaded model, device, and backend. Record it. The PyPI package named `kev` is unrelated; install this model from source.

The controller must reach Kev from its own host/container/job. `127.0.0.1` refers to that process's machine or container. Confirm connectivity from the compute job; a working request on a laptop does not prove cluster connectivity. Use the actual endpoint in the adapter. The transcript's incomplete localhost address is not a configuration value.

## 3. Inspect and try the proposed bridge

This reproducible source snapshot is PR #1 at `70412e311d9ba7bb5045b1b061cf10282b480c1d`. It is not the current `main` implementation. Reuse an existing checkout rather than cloning over it.

```bash
cd <YOUR_WORKSPACE_DIR>
git clone https://github.com/zihuahe/hierarchical-spacecraft-control.git
cd <YOUR_WORKSPACE_DIR>/hierarchical-spacecraft-control
git checkout 70412e311d9ba7bb5045b1b061cf10282b480c1d
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q tests/test_kspdg_loop.py
```

Use Python 3.12 for this checkout. Expected result: tests pass; they use a stub model server and do not establish real game or real model integration. These commands have not been run as part of this documentation task.

With Kev running at an endpoint reachable from this checkout:

```bash
cd <YOUR_WORKSPACE_DIR>/hierarchical-spacecraft-control
.venv/bin/python -m hsc.kspdg_loop --env fixture --steps 1 \
  --kev-base-url <YOUR_REACHABLE_KEV_URL>
```

Expect a JSONL trace with `env: fixture`, a decision, a next observation, and a `run_end`. For a real game connection, first install the matching KSPDG dependencies and start KSP/kRPC using the chosen runtime guide:

```bash
cd <YOUR_WORKSPACE_DIR>/hierarchical-spacecraft-control
.venv/bin/python -m hsc.kspdg_loop --env kspdg --steps 1 \
  --scenario <YOUR_AVAILABLE_I2_SCENARIO> \
  --kev-base-url <YOUR_REACHABLE_KEV_URL>
```

Expect the run header to identify KSPDG and a real observation/action/next-observation trace. Confirm effective rules rather than relying on defaults. The bridge defaults to 0.2 throttle and a 1.0-second burn; Heiko recommends testing roughly 0.2–0.5-second burns in his runner. Choose and log settings for the actual harness.

### Test the real observation before model control

The latest bridge supports an observation-only connection check. This still resets/loads the selected mission, but issues no Kev request or thrust command:

```bash
cd <YOUR_WORKSPACE_DIR>/hierarchical-spacecraft-control
.venv/bin/python -m hsc.kspdg_loop --env kspdg --steps 0 \
  --scenario <YOUR_AVAILABLE_I2_SCENARIO> \
  --krpc-address <YOUR_KRPC_HOST> --krpc-rpc-port 50000 --krpc-stream-port 50001
```

Expect an `observation` record, effective `scenario_settings`, and `run_end` with `observation_only`. An `env_unavailable` result establishes a connection/dependency problem, not model failure. Use the same kRPC settings for the subsequent one-action check.

The reviewed bridge defaults to one discarded warm-up request before decisions. Record warm-ups separately from timed inference. Its synthetic model check can run without KSP:

```bash
cd <YOUR_WORKSPACE_DIR>/hierarchical-spacecraft-control
.venv/bin/python -m hsc.kev_check --requests 20 \
  --kev-base-url <YOUR_REACHABLE_KEV_URL> --out results/kev-local/kev_check.json
```

Expected evidence: valid choices and a latency summary for synthetic states. This does not establish navigation skill or a real-game episode. See [status](STATUS.md).

## 4. Capture the reproducible setup

Record host architecture, Python/KSPDG/Kev revisions, environment class, mission/save, capture distances, mission timeout, runner timeout, burn duration, model endpoint, and exact commands. Confirm shutdown behavior before a model timeout test.

Rendering is a separate workstream. Use logged trajectories if game display is unavailable. Do not treat a browser port, game process, and kRPC endpoint as interchangeable services.
