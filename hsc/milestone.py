"""First milestone: one real observation -> fast-layer decision -> executed command -> next observation.

    python -m hsc.milestone                       # local Kev server, raw action IDs (the first-version plan)
    python -m hsc.milestone --decider rule_discrete   # no model, to check the loop itself
    python -m hsc.milestone --decider jev --options primitives

Prints one JSON record per step and appends it to results/milestone.jsonl.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from hsc.control.executor import Executor
from hsc.control.plans import MissionBrief, TacticalGoal
from hsc.control.situation import situation_from_obs
from hsc.sim.lbg1_lite import LBG1LiteEnv


def make_decider(kind: str, options: str, kev_url: str | None):
    from hsc.layers import deciders as D

    if kind == "kev":
        return D.KevDecider(base_url=kev_url, options=options)
    if kind == "jev":
        return D.JevDecider(options=options)
    if kind == "rule":
        return D.RuleDecider()
    if kind == "rule_discrete":
        return D.DiscreteRuleDecider()
    raise ValueError(kind)


def run(decider, steps: int = 1, init_cond: str = "i2", policy: str = "lg7", hold_s: float = 1.0) -> list[dict]:
    env = LBG1LiteEnv(init_cond, policy)
    obs, _ = env.reset()
    p = env.bandit_params
    executor = Executor(hold_s, p.thrust_fwd, p.thrust_rev, p.thrust_lat)
    brief, goal = MissionBrief(), TacticalGoal()
    records = []
    for _ in range(steps):
        s = situation_from_obs(obs, p.prop_mass, env.episode_timeout)
        if hasattr(decider, "obs"):
            decider.obs = obs
        t0 = time.perf_counter()
        choice, _, info = decider.decide(s, goal, brief)
        wall = time.perf_counter() - t0
        action = executor.act(s, obs, choice, goal, brief)
        next_obs, _, done, env_info = env.step(action)
        records.append({
            "decider": decider.name,
            "observation": [round(float(x), 3) for x in obs],
            "decision": choice,
            "decision_latency_s": round(info.latency_s if info is not None else wall, 4),
            "confidence": None if info is None else info.extra.get("confidence"),
            "command": {"forward": action["burn_vec"][0], "right": action["burn_vec"][1],
                        "duration_s": action["burn_vec"][3]},
            "next_observation": [round(float(x), 3) for x in next_obs],
            "lady_dist_m": round(float(np.linalg.norm(next_obs[3:6] - next_obs[9:12])), 2),
            "guard_dist_m": round(float(np.linalg.norm(next_obs[3:6] - next_obs[15:18])), 2),
            "fuel_used_kg": round(env_info["bandit_fuel_usage"], 3),
        })
        obs = next_obs
        if done:
            break
    return records


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decider", default="kev", choices=["kev", "jev", "rule", "rule_discrete"])
    ap.add_argument("--options", default="discrete", choices=["discrete", "primitives"])
    ap.add_argument("--kev-url", default=None, help="default $KEV_BASE_URL or http://127.0.0.1:8009")
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--hold", type=float, default=1.0, help="seconds each command is held (0-10)")
    ap.add_argument("--scenario", default="i2:lg7")
    ap.add_argument("--out", default="results/milestone.jsonl")
    args = ap.parse_args(argv)
    ic, lg = args.scenario.split(":")
    records = run(make_decider(args.decider, args.options, args.kev_url), args.steps, ic, lg, args.hold)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
