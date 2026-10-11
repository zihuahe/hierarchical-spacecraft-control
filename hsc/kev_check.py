"""Stage B check: is local Kev serving, and does it answer the loop's own Choice request?

    python -m hsc.kev_check                       # Kev at --kev-base-url (default http://127.0.0.1:8009)
    python -m hsc.kev_check --requests 20 --out results/kev-local/kev_check.json

Sends exactly the request ``hsc.kspdg_loop`` sends (same mission text, features layout, instructions and
seven action IDs) and validates every answer with the loop's parser. The states are SYNTHETIC check states
built from varying geometry, not simulator observations: this proves inference and measures latency, not
flight competence. The first ``--warmup`` answers are excluded from the latency figures.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import platform
import statistics
import time
import urllib.request

from hsc import kspdg_loop as K

# LBG1_LG7_I2_V1 rules as read from kspdg 0.15.0 (LBG1_LG7_I2_Env attributes)
LG7_SETTINGS = {"episode_timeout_s": 240.0, "lady_capture_dist_m": 5.0, "bandit_capture_dist_m": 10.0}


def synthetic_observation(i: int) -> list[float]:
    """A 21-value observation with the KSPDG layout: Lady ahead along +y, Guard between, both offsets varying."""
    r, v = 700_000.0, 2_200.0
    a = 0.3 * i
    lady = [r, 2600.0 - 40 * i, 30 * math.sin(a)]
    guard = [r + 15 * math.cos(a), 2000.0 - 25 * i, 10 * math.sin(2 * a)]
    return ([12.0 + i, 6800.0 - 0.5 * i, 120.0 - 0.1 * i, r, 0.0, 0.0, 0.0, v + 0.05 * i, 0.0]
            + lady + [0.0, v, 0.0] + guard + [0.3 * math.sin(a), v + 0.4, 0.0])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kev-base-url", default=K.LoopConfig.kev_base_url)
    ap.add_argument("--kev-model", default=K.LoopConfig.kev_model)
    ap.add_argument("--request-timeout-s", type=float, default=60.0)
    ap.add_argument("--requests", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", default=None, help="write the full report (requests, answers, latencies) as JSON")
    a = ap.parse_args(argv)
    cfg = K.LoopConfig(kev_base_url=a.kev_base_url, kev_model=a.kev_model, request_timeout_s=a.request_timeout_s)
    with urllib.request.urlopen(f"{cfg.kev_base_url.rstrip('/')}/v1/models", timeout=10) as r:
        models = json.loads(r.read())
    mission = K.mission_text(LG7_SETTINGS)
    calls, initial = [], synthetic_observation(0)
    for i in range(a.warmup + a.requests):
        obs = K.parse_observation(synthetic_observation(i))
        req = K.build_request(K.features(obs, [], LG7_SETTINGS["episode_timeout_s"], initial), cfg, None, mission)
        t0 = time.perf_counter()
        parsed, model_latency = K.kev_choose(req, cfg)
        cmd = K.command_for(parsed["choice"], cfg)
        calls.append({"i": i, "warmup": i < a.warmup, "choice": parsed["choice"], "confidence": parsed["confidence"],
                      "probabilities": parsed["probabilities"], "burn_command": cmd,
                      "model_latency_s": round(model_latency, 4),
                      "decision_latency_s": round(time.perf_counter() - t0, 4),
                      **({"request": req} if i in (0, a.warmup) else {})})
    timed = sorted(c["decision_latency_s"] for c in calls if not c["warmup"])
    p95 = timed[min(len(timed) - 1, math.ceil(0.95 * len(timed)) - 1)] if timed else None
    report = {"note": "SYNTHETIC check states (not simulator observations); validates Kev inference and latency only",
              "kev_base_url": cfg.kev_base_url, "models": models, "python": platform.python_version(),
              "platform": platform.platform(), "timed_requests": len(timed),
              "decision_latency_s": {"median": statistics.median(timed) if timed else None, "p95": p95,
                                     "min": timed[0] if timed else None, "max": timed[-1] if timed else None},
              "choices": {c: sum(x["choice"] == c for x in calls if not x["warmup"]) for c in K.ACTION_IDS},
              "calls": calls}
    if a.out:
        pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.out).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "calls"}, indent=1))


if __name__ == "__main__":
    main()
