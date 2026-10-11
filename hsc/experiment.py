"""Named controller configurations and the experiment driver.

    python -m hsc.experiment --configs fast_only,rule_hier --seeds 3
    python -m hsc.experiment --configs latency_sweep --seeds 3
    python -m hsc.experiment --configs hier_rule_fast,flat_haiku,flat_opus --seeds 2   # needs ANTHROPIC_API_KEY
    python -m hsc.experiment --configs hier_jev,hier_no_opus --seeds 2                  # also needs TYPESAFE_API_KEY
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib

import numpy as np

from hsc.runner import ControllerConfig, run_episode

SWEEP_LATENCIES = [0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0]

DESCRIPTIONS = {
    "fast_only": "No models. Rule decider with the default brief, zero latency.",
    "rule_hier": "No models. Rule tactician (out-of-plane bypass) + rule decider.",
    "flat_opus": "Opus picks every maneuver directly (blocking).",
    "flat_haiku": "Haiku picks every maneuver directly (blocking).",
    "hier_rule_fast": "Opus brief + Haiku tactical goals + rule decider (no Jev).",
    "hier_jev": "Full proposal: Opus brief + Haiku tactical goals + Jev picks maneuvers.",
    "hier_no_opus": "Ablation: default brief + Haiku tactical goals + Jev.",
    "flat_jev": "Jev picks every maneuver with the default brief and no tactical layer.",
    "latency_sweep": "Rule decider with synthetic decision latency L seconds (fast_only@L).",
}


def make_config(name: str, haiku_images: bool = False) -> list[ControllerConfig]:
    from hsc.layers.deciders import FlatLLMDecider, JevDecider, RuleDecider
    from hsc.layers.llm import HAIKU_MODEL, OPUS_MODEL
    from hsc.layers.strategist import FixedStrategist, OpusStrategist
    from hsc.layers.tactician import HaikuTactician, RuleTactician

    if name == "fast_only":
        return [ControllerConfig(name, FixedStrategist(), None, RuleDecider())]
    if name == "rule_hier":
        return [ControllerConfig(name, FixedStrategist(), RuleTactician(), RuleDecider())]
    if name == "flat_opus":
        return [ControllerConfig(name, FixedStrategist(), None, FlatLLMDecider(OPUS_MODEL, effort="low"))]
    if name == "flat_haiku":
        return [ControllerConfig(name, FixedStrategist(), None, FlatLLMDecider(HAIKU_MODEL, effort="low"))]
    if name == "hier_rule_fast":
        return [ControllerConfig(name, OpusStrategist(), HaikuTactician(use_images=haiku_images), RuleDecider())]
    if name == "hier_jev":
        return [ControllerConfig(name, OpusStrategist(), HaikuTactician(use_images=haiku_images), JevDecider())]
    if name == "hier_no_opus":
        return [ControllerConfig(name, FixedStrategist(), HaikuTactician(use_images=haiku_images), JevDecider())]
    if name == "flat_jev":
        return [ControllerConfig(name, FixedStrategist(), None, JevDecider())]
    if name == "latency_sweep":
        return [ControllerConfig(f"fast_only@{L:g}s", FixedStrategist(), None, RuleDecider(), decider_latency_s=L)
                for L in SWEEP_LATENCIES]
    raise ValueError(f"unknown config {name}; choose from {sorted(DESCRIPTIONS)}")


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    order = {}
    for r in rows:
        order.setdefault(r["config"], len(order))
    keys = sorted({(r["config"], r["scenario"]) for r in rows}, key=lambda k: (k[1], order[k[0]]))
    for cfg, sc in keys:
        rs = [r for r in rows if r["config"] == cfg and r["scenario"] == sc]
        dec = [r["layers"]["decider"] for r in rs]
        lat = [d["latency_p50_s"] for d in dec if d["latency_p50_s"] is not None]
        tac = [r["layers"]["tactical"]["latency_p50_s"] for r in rs if r["layers"]["tactical"]["latency_p50_s"] is not None]
        out.append({
            "config": cfg,
            "scenario": sc,
            "n": len(rs),
            "score_median": float(np.median([r["score"] for r in rs])),
            "score_mean": float(np.mean([r["score"] for r in rs])),
            "lady_captured_pct": 100 * float(np.mean([r["lady_captured"] for r in rs])),
            "guard_captured_pct": 100 * float(np.mean([r["bandit_captured"] for r in rs])),
            "closest_lb_m": float(np.mean([r["closest_lb_m"] for r in rs])),
            "closest_bg_m": float(np.mean([r["closest_bg_m"] for r in rs])),
            "fuel_kg": float(np.mean([r["bandit_fuel_kg"] for r in rs])),
            "decider_latency_p50_s": float(np.median(lat)) if lat else None,
            "tactical_latency_p50_s": float(np.median(tac)) if tac else None,
            "action_age_mean_s": float(np.mean([r["action_age_mean_s"] for r in rs])),
            "decider_calls": int(np.sum([d["calls"] for d in dec])),
            "errors": int(np.sum([sum(l["errors"] for l in r["layers"].values()) for r in rs])),
        })
    return out


def _fmt(v, nd=1):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:,.{nd}f}"
    return str(v)


def summary_markdown(summary: list[dict]) -> str:
    cols = [("config", "config"), ("scenario", "scenario"), ("n", "n"), ("score_median", "median score"),
            ("lady_captured_pct", "Lady caught %"), ("guard_captured_pct", "Guard caught us %"),
            ("closest_lb_m", "closest Lady m"), ("closest_bg_m", "closest Guard m"), ("fuel_kg", "fuel kg"),
            ("decider_latency_p50_s", "decision latency p50 s"), ("tactical_latency_p50_s", "tactical latency p50 s"),
            ("action_age_mean_s", "mean action age s"), ("errors", "errors")]
    lines = ["| " + " | ".join(c[1] for c in cols) + " |", "|" + "---|" * len(cols)]
    for row in summary:
        lines.append("| " + " | ".join(_fmt(row[k], 0 if k in ("score_median",) else 2 if "latency" in k or "age" in k else 1)
                                        for k, _ in cols) + " |")
    return "\n".join(lines)


def plot_latency_sweep(summary: list[dict], path: pathlib.Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweep = [r for r in summary if r["config"].startswith("fast_only@")]
    if not sweep:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for sc in sorted({r["scenario"] for r in sweep}):
        rs = sorted([r for r in sweep if r["scenario"] == sc], key=lambda r: float(r["config"].split("@")[1][:-1]))
        L = [float(r["config"].split("@")[1][:-1]) for r in rs]
        axes[0].plot(L, [r["score_median"] for r in rs], "o-", label=sc)
        axes[1].plot(L, [r["closest_bg_m"] for r in rs], "o-", label=sc)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("median lbg1 score (lower is better)")
    axes[1].set_ylabel("closest Guard approach [m] (higher is better)")
    for ax in axes:
        ax.set_xlabel("decision latency [s]")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Same controller, slower decisions")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default="fast_only,rule_hier")
    ap.add_argument("--scenarios", default="i1:lg1,i2:lg1,i2:lg0", help="comma list of init_cond:lady_guard_policy")
    ap.add_argument("--seeds", type=int, default=3, help="seeds 0..N-1 (perturbed initial states)")
    ap.add_argument("--haiku-images", action="store_true", help="send a rendered plot of recent motion to Haiku")
    ap.add_argument("--out", default=None, help="output directory (default results/<timestamp>)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    out = pathlib.Path(args.out or f"results/{dt.datetime.now():%Y%m%d-%H%M%S}")
    out.mkdir(parents=True, exist_ok=True)
    scenarios = [tuple(s.split(":")) for s in args.scenarios.split(",")]
    rows = []
    with open(out / "episodes.jsonl", "w") as f:
        for cname in args.configs.split(","):
            for cfg in make_config(cname, args.haiku_images):
                for ic, lg in scenarios:
                    for seed in range(args.seeds):
                        r = run_episode(cfg, ic, lg, seed=seed, verbose=args.verbose)
                        rows.append(r)
                        f.write(json.dumps(r, default=float) + "\n")
                        f.flush()
                        print(f"{r['config']:>18} {r['scenario']} seed={seed} score={r['score']:,.0f} "
                              f"lady={r['closest_lb_m']:.1f}m guard={r['closest_bg_m']:.1f}m "
                              f"age={r['action_age_mean_s']:.2f}s wall={r['wall_s']:.1f}s", flush=True)
    summary = summarize(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    md = summary_markdown(summary)
    (out / "summary.md").write_text(md + "\n")
    if plot_latency_sweep(summary, out / "latency_sweep.png"):
        print(f"wrote {out / 'latency_sweep.png'}")
    print(md)
    print(f"\nresults in {out}")


if __name__ == "__main__":
    main()
