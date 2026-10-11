"""Replay video of one or more ``hsc.kspdg_loop`` traces: Lady, Bandit and Guard in the Lady's LVLH frame.

    python -m hsc.replay results/kev-fixture-check/20261011-202335-4f08c1.jsonl --out results/replay/kev.mp4
    python -m hsc.replay KEV.jsonl RULE.jsonl --labels "Kev-0.8B" "Rule baseline" --out results/replay/side.mp4

Several traces render side by side on one mission clock. Everything drawn comes from the trace: the
observations (positions), each decision's action and probabilities, the scenario settings in the run
header and the terminal reason. Traces from the ``lbg1_lite`` fixture are stamped as such. Needs ffmpeg.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from hsc.kspdg_loop import ACTION_IDS
from hsc.sim.orbits import lvlh_basis, relative_lvlh

BG = "#05070d"
FG = "#d8dee9"
DIM = "#6b7385"
LADY_C = "#ffd166"
BANDIT_C = "#4cc9f0"
GUARD_C = "#ef476f"
BANNERS = {"capture": ("LADY CAPTURED", "#06d6a0"), "interception": ("INTERCEPTED BY GUARD", GUARD_C),
           "capture_and_interception": ("CAPTURE + INTERCEPTION", "#f4a261"), "timeout": ("TIMEOUT", LADY_C)}


def load_trace(path) -> dict:
    """Mission-time series from a trace: LVLH positions relative to the Lady, decisions and the outcome."""
    recs = [json.loads(line) for line in open(path)]
    header = next(r for r in recs if r["record"] == "run_header")
    first = next(r for r in recs if r["record"] == "observation")
    decisions = [r for r in recs if r["record"] == "decision" and "next_observation" in r]
    end = next((r for r in recs if r["record"] == "run_end"), {})
    obs = [first["observation"]] + [r["next_observation"] for r in decisions]
    rel = {"bandit": [], "guard": []}
    for o in obs:
        lp, lv = o[9:12], o[12:15]
        for name, sl in (("bandit", slice(3, 9)), ("guard", slice(15, 21))):
            pos, _ = relative_lvlh(lp, lv, o[sl][:3], o[sl][3:])
            rel[name].append(pos)
    burns = []
    for r in decisions:  # commanded inertial throttle direction, expressed in the Lady's LVLH frame
        o = r["observation"]
        burns.append(lvlh_basis(np.asarray(o[9:12]), np.asarray(o[12:15])) @ np.asarray(r["burn_command"]["burn_vec"][:3]))
    obs = np.asarray(obs)
    settings = header.get("scenario_settings", {})
    return {
        "path": str(path), "env": header.get("versions", {}).get("env"), "settings": settings,
        "config": header.get("config", {}),
        "model": decisions[0]["model"] if decisions else header.get("config", {}).get("kev_model"),
        "t": obs[:, 0], "bandit": np.asarray(rel["bandit"]), "guard": np.asarray(rel["guard"]),
        "lady_dist": np.linalg.norm(obs[:, 9:12] - obs[:, 3:6], axis=1),
        "guard_dist": np.linalg.norm(obs[:, 15:18] - obs[:, 3:6], axis=1),
        "mass_used": obs[0, 1] - obs[:, 1],
        "choices": [r["choice"] for r in decisions], "probs": [r.get("probabilities") or {} for r in decisions],
        "burns": np.asarray(burns) if burns else np.zeros((0, 3)),
        "latency": [r.get("model_latency_s") for r in decisions],
        "terminal_reason": end.get("terminal_reason"),
    }


def _interp(tr: dict, T: float):
    """Interpolated state at mission time T (clamped to the trace) and the index of the decision in force."""
    t = tr["t"]
    T = min(max(T, t[0]), t[-1])
    k = int(np.clip(np.searchsorted(t, T, side="right") - 1, 0, len(t) - 1))
    if k >= len(t) - 1:
        a = 0.0
        k = len(t) - 1
    else:
        a = (T - t[k]) / max(t[k + 1] - t[k], 1e-9)
    def lerp(x):
        return x[k] if a == 0.0 else x[k] + a * (x[k + 1] - x[k])
    return T, k, lerp(tr["bandit"]), lerp(tr["guard"]), lerp(tr["lady_dist"]), lerp(tr["guard_dist"]), lerp(tr["mass_used"])


class _Panel:
    def __init__(self, fig, gs_col, tr: dict, label: str, rng):
        import matplotlib.pyplot as plt  # noqa: F401

        self.tr = tr
        self.label = label
        sub = gs_col.subgridspec(2, 2, height_ratios=[3.2, 1.0], hspace=0.28, wspace=0.32)
        self.ax = fig.add_subplot(sub[0, :])
        self.ax_p = fig.add_subplot(sub[1, 0])
        self.ax_d = fig.add_subplot(sub[1, 1])
        for a in (self.ax, self.ax_p, self.ax_d):
            a.set_facecolor(BG)
            for s in a.spines.values():
                s.set_color("#1c2333")
            a.tick_params(colors=DIM, labelsize=7)
        ax = self.ax
        self.px_ratio = None  # axes width / height in pixels, so metres are square on screen
        n = 220
        ax.scatter(rng.random(n), rng.random(n), s=rng.random(n) * 2.2, c="white", alpha=0.35,
                   transform=ax.transAxes, zorder=0, linewidths=0)
        ax.set_xlabel("along-track from Lady (m)", color=DIM, fontsize=7)
        ax.set_ylabel("radial from Lady (m)", color=DIM, fontsize=7)
        title = f"{label}"
        ax.set_title(title, color=FG, fontsize=11, loc="left", fontweight="bold")
        if tr["env"] == "fixture":
            ax.text(1.0, 1.02, "lbg1_lite TEST FIXTURE, not KSP", transform=ax.transAxes, ha="right",
                    va="bottom", color="#ff9f1c", fontsize=7.5, fontweight="bold")
        self.lady = ax.scatter([0], [0], marker="*", s=220, c=LADY_C, zorder=5, edgecolors="none")
        ax.text(0, 0, "  Lady", color=LADY_C, fontsize=7, va="bottom", zorder=5)
        (self.b_trail,) = ax.plot([], [], color=BANDIT_C, lw=1.2, alpha=0.6, zorder=3)
        (self.g_trail,) = ax.plot([], [], color=GUARD_C, lw=1.2, alpha=0.6, zorder=3)
        self.b_dot = ax.scatter([], [], s=55, c=BANDIT_C, zorder=6, edgecolors="white", linewidths=0.6)
        self.g_dot = ax.scatter([], [], s=55, c=GUARD_C, marker="^", zorder=6, edgecolors="white", linewidths=0.6)
        self.b_txt = ax.text(0, 0, "", color=BANDIT_C, fontsize=7, zorder=6, va="top")
        self.g_txt = ax.text(0, 0, "", color=GUARD_C, fontsize=7, zorder=6, va="bottom")
        self.burn = None
        self.hud = ax.text(0.015, 0.98, "", transform=ax.transAxes, va="top", ha="left", color=FG, fontsize=8,
                           family="DejaVu Sans Mono", zorder=8,
                           bbox=dict(boxstyle="round,pad=0.4", fc="#0b1020", ec="#1c2333", alpha=0.85))
        self.banner = ax.text(0.5, 0.80, "", transform=ax.transAxes, ha="center", va="center", fontsize=17,
                              fontweight="bold", zorder=10, color=FG)
        # close-up of the Bandit, where the Guard's intercept happens
        self.inset = ax.inset_axes([0.70, 0.04, 0.28, 0.36])
        self.inset.set_facecolor("#080c18")
        self.inset.set_xticks([])
        self.inset.set_yticks([])
        for s in self.inset.spines.values():
            s.set_color(BANDIT_C)
            s.set_alpha(0.5)
        self.inset_r = 60.0
        self.inset.set_xlim(-self.inset_r, self.inset_r)
        self.inset.set_ylim(-self.inset_r, self.inset_r)
        self.inset.set_aspect("equal")
        self.inset.text(0.04, 0.94, "Bandit close-up ±60 m", transform=self.inset.transAxes, color=DIM,
                        fontsize=6, va="top")
        import matplotlib.patches as mp

        bcd = tr["settings"].get("bandit_capture_dist_m") or 0.0
        self.inset.add_patch(mp.Circle((0, 0), bcd, fill=False, ec=GUARD_C, ls="--", lw=0.8, alpha=0.8))
        self.inset.scatter([0], [0], s=30, c=BANDIT_C, zorder=3)
        self.in_g = self.inset.scatter([], [], s=34, c=GUARD_C, marker="^", zorder=4)
        (self.in_gtrail,) = self.inset.plot([], [], color=GUARD_C, lw=1.0, alpha=0.6)
        self.in_lbl = self.inset.text(0.96, 0.06, f"intercept ring {bcd:g} m", transform=self.inset.transAxes,
                                      color=GUARD_C, fontsize=5.5, ha="right", alpha=0.8)
        # action probabilities
        ap = self.ax_p
        self.bars = ap.barh(range(len(ACTION_IDS)), [0] * len(ACTION_IDS), color="#2b3550", height=0.7)
        ap.set_yticks(range(len(ACTION_IDS)), ACTION_IDS, fontsize=6.5, color=FG)
        ap.invert_yaxis()
        ap.set_xlim(0, 1)
        has_probs = any(tr["probs"])
        ap.set_title("model probabilities" if has_probs else "chosen action (no model)", color=DIM, fontsize=7.5,
                     loc="left")
        # distances
        ad = self.ax_d
        ad.set_yscale("log")
        ad.set_xlim(0, max(tr["settings"].get("episode_timeout_s") or tr["t"][-1], tr["t"][-1]))
        ad.set_ylim(1, 5000)
        lcd = tr["settings"].get("lady_capture_dist_m")
        if lcd:
            ad.axhline(lcd, color=LADY_C, ls=":", lw=0.8, alpha=0.7)
        if bcd:
            ad.axhline(bcd, color=GUARD_C, ls=":", lw=0.8, alpha=0.7)
        (self.d_l,) = ad.plot([], [], color=LADY_C, lw=1.2, label="to Lady")
        (self.d_g,) = ad.plot([], [], color=GUARD_C, lw=1.2, label="to Guard")
        ad.legend(fontsize=6, loc="lower left", facecolor=BG, edgecolor="#1c2333", labelcolor=FG)
        ad.set_title("distance (m) vs mission time (s)", color=DIM, fontsize=7.5, loc="left")
        self.view = None

    def draw(self, T: float, final: bool):
        tr = self.tr
        T, k, b, g, dl, dg, mu = _interp(tr, T)
        ended = T >= tr["t"][-1] - 1e-9
        sel = tr["t"] <= T + 1e-9
        bt = np.vstack([tr["bandit"][sel], b])
        gt = np.vstack([tr["guard"][sel], g])
        self.b_trail.set_data(bt[:, 1], bt[:, 0])
        self.g_trail.set_data(gt[:, 1], gt[:, 0])
        self.b_dot.set_offsets([[b[1], b[0]]])
        self.g_dot.set_offsets([[g[1], g[0]]])
        self.b_txt.set_position((b[1], b[0]))
        self.b_txt.set_text("  Bandit")
        self.g_txt.set_position((g[1], g[0]))
        self.g_txt.set_text("  Guard")
        # camera: fit all three with margin, eased between frames
        pts = np.array([[0.0, 0.0], [b[1], b[0]], [g[1], g[0]]])
        lo, hi = pts.min(0), pts.max(0)
        c = (lo + hi) / 2
        span = max(float(np.max(hi - lo)) * 1.35, 150.0)
        target = np.array([c[0], c[1], span])
        self.view = target if self.view is None else self.view + 0.15 * (target - self.view)
        cx, cy, sp = self.view
        if self.px_ratio is None:
            bb = self.ax.get_window_extent()
            self.px_ratio = bb.width / bb.height
        self.ax.set_xlim(cx - sp / 2 * self.px_ratio, cx + sp / 2 * self.px_ratio)
        self.ax.set_ylim(cy - sp / 2, cy + sp / 2)
        # burn arrow for the decision in force
        if self.burn is not None:
            self.burn.remove()
            self.burn = None
        choice = tr["choices"][min(k, len(tr["choices"]) - 1)] if tr["choices"] else None
        if choice and choice != "COAST" and not ended:
            d = tr["burns"][min(k, len(tr["burns"]) - 1)]
            d2 = np.array([d[1], d[0]])
            if np.linalg.norm(d2) > 1e-9:
                d2 = d2 / np.linalg.norm(d2) * sp * 0.08
                self.burn = self.ax.annotate("", xy=(b[1] + d2[0], b[0] + d2[1]), xytext=(b[1], b[0]), zorder=7,
                                             arrowprops=dict(arrowstyle="-|>", color="#ffb703", lw=1.8))
        # close-up
        dgv = g - b
        gv = np.array([dgv[1], dgv[0]])
        if np.linalg.norm(gv) <= self.inset_r:
            self.in_g.set_offsets([gv])
            tail = gt[-40:] - np.array([b])
            self.in_gtrail.set_data(tail[:, 1], tail[:, 0])
        else:
            self.in_g.set_offsets(np.empty((0, 2)))
            self.in_gtrail.set_data([], [])
        # HUD
        timeout = tr["settings"].get("episode_timeout_s")
        lat = tr["latency"][min(k, len(tr["latency"]) - 1)] if tr["latency"] else None
        probs = tr["probs"][min(k, len(tr["probs"]) - 1)] if tr["probs"] else {}
        p = probs.get(choice) if choice else None
        lines = [f"T+ {T:6.1f} s" + (f" / {timeout:g} s" if timeout else ""),
                 f"Lady  {dl:7.1f} m", f"Guard {dg:7.1f} m", f"fuel used {mu:5.1f} kg",
                 f"action {choice or '-':<6}" + (f" p={p:.2f}" if p is not None else "")]
        if lat is not None and "rule" not in str(tr["model"]):
            lines.append(f"decision {lat:4.2f} s wall")
        self.hud.set_text("\n".join(lines))
        for i, a in enumerate(ACTION_IDS):
            if probs:
                self.bars[i].set_width(probs.get(a, 0.0))
            else:
                self.bars[i].set_width(1.0 if a == choice else 0.0)
            self.bars[i].set_color(BANDIT_C if a == choice else "#2b3550")
        self.d_l.set_data(np.append(tr["t"][sel], T), np.append(tr["lady_dist"][sel], dl))
        self.d_g.set_data(np.append(tr["t"][sel], T), np.append(tr["guard_dist"][sel], dg))
        if ended and final is not None:
            text, color = BANNERS.get(tr["terminal_reason"], (str(tr["terminal_reason"]).upper(), FG))
            self.banner.set_text(f"{text}\nT+ {tr['t'][-1]:.1f} s")
            self.banner.set_color(color)
            self.banner.set_bbox(dict(boxstyle="round,pad=0.5", fc=BG, ec=color, alpha=0.85))


def render(traces: list[dict], labels: list[str], out: str, fps: int = 24, speed: float = 8.0,
           hold_s: float = 2.5, title: str | None = None, poster: str | None = None) -> dict:
    """Render the traces side by side. ``speed`` is mission seconds per video second."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    n = len(traces)
    fig = plt.figure(figsize=(6.4 * n if n > 1 else 9.6, 7.2), dpi=100, facecolor=BG)
    gs = fig.add_gridspec(1, n, left=0.06 if n > 1 else 0.08, right=0.98, top=0.88, bottom=0.07, wspace=0.18)
    rng = np.random.default_rng(7)
    panels = [_Panel(fig, gs[0, i], tr, lab, rng) for i, (tr, lab) in enumerate(zip(traces, labels))]
    s0 = traces[0]["settings"]
    fig.suptitle(title or "Lady-Bandit-Guard replay", color=FG, fontsize=14, fontweight="bold", y=0.975)
    fig.text(0.5, 0.925, f"Bandit must get within {s0.get('lady_capture_dist_m', '?'):g} m of the Lady before the "
             f"Guard gets within {s0.get('bandit_capture_dist_m', '?'):g} m of the Bandit. "
             f"Seven inertial actions at {traces[0]['config'].get('thrust_magnitude', '?'):g} throttle for "
             f"{traces[0]['config'].get('burn_duration_s', '?'):g} s each.", ha="center", color=DIM, fontsize=8.5)
    t_end = max(tr["t"][-1] for tr in traces)
    n_frames = int(np.ceil(t_end / speed * fps)) + 1
    n_hold = int(hold_s * fps)
    out_p = pathlib.Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=2400,
                          extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    with writer.saving(fig, str(out_p), dpi=100):
        for i in range(n_frames + n_hold):
            T = min(i * speed / fps, t_end)
            for p in panels:
                p.draw(T, final=True)
            writer.grab_frame(facecolor=BG)
    if poster:
        fig.savefig(poster, facecolor=BG, dpi=100)
    plt.close(fig)
    return {"out": str(out_p), "frames": n_frames + n_hold, "fps": fps, "mission_s": float(t_end),
            "outcomes": [tr["terminal_reason"] for tr in traces]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+", help="hsc.kspdg_loop JSONL traces, rendered side by side")
    ap.add_argument("--labels", nargs="+", help="panel titles (default: the model recorded in each trace)")
    ap.add_argument("--out", default="results/replay/replay.mp4")
    ap.add_argument("--poster", help="also save the final frame as a PNG")
    ap.add_argument("--title")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--speed", type=float, default=8.0, help="mission seconds per video second")
    args = ap.parse_args(argv)
    traces = [load_trace(p) for p in args.traces]
    labels = args.labels or [str(tr["model"]) for tr in traces]
    if len(labels) != len(traces):
        ap.error("--labels needs one label per trace")
    print(json.dumps(render(traces, labels, args.out, fps=args.fps, speed=args.speed, title=args.title,
                            poster=args.poster), indent=2))


if __name__ == "__main__":
    main()
