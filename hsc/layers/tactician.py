"""Tactical planning layer (medium speed): reviews recent states and updates short-term goals."""

from __future__ import annotations

import base64
import io
import json

import numpy as np

from hsc.control.plans import PRIMITIVES, MissionBrief, TacticalGoal
from hsc.control.situation import Situation
from hsc.layers.llm import HAIKU_MODEL, StructuredLLM
from hsc.layers.strategist import SCENARIO_TEXT

GOAL_SCHEMA = {
    "type": "object",
    "properties": {
        "mode_hint": {"type": "string", "enum": ["AUTO", *PRIMITIVES.keys()]},
        "waypoint_lvlh_m": {"anyOf": [{"type": "array", "items": {"type": "number"}}, {"type": "null"}]},
        "max_closing_speed_mps": {"type": "number"},
        "guard_keepout_m": {"type": "number"},
        "standoff_m": {"type": "number"},
        "valid_for_s": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["mode_hint", "waypoint_lvlh_m", "max_closing_speed_mps", "guard_keepout_m",
                 "standoff_m", "valid_for_s", "rationale"],
    "additionalProperties": False,
}

TACTICAL_SYSTEM = SCENARIO_TEXT + """
You are the TACTICAL PLANNER. Each call you see the mission brief and the recent history. Return the
short-term goal the fast controller should pursue until your next update: an optional Lady-relative
waypoint (use one to route around the Guard, for example out of plane on z), speed and keep-out limits,
and a mode hint (AUTO lets the fast controller choose; otherwise it is strongly preferred).
Keep the rationale to one sentence.
"""


def _goal_from_dict(d: dict, t: float, brief: MissionBrief) -> TacticalGoal:
    wp = d.get("waypoint_lvlh_m")
    if wp is not None and len(wp) != 3:
        wp = None
    return TacticalGoal(
        mode_hint=d.get("mode_hint", "AUTO"),
        waypoint_lvlh_m=None if wp is None else [float(x) for x in wp],
        max_closing_speed_mps=float(np.clip(d.get("max_closing_speed_mps", brief.max_closing_speed_mps), 0.5, 40)),
        guard_keepout_m=float(np.clip(d.get("guard_keepout_m", brief.guard_keepout_m), 10, 1000)),
        standoff_m=float(np.clip(d.get("standoff_m", 300.0), 10, 3000)),
        rationale=str(d.get("rationale", ""))[:300],
        issued_at=t,
        valid_for_s=float(np.clip(d.get("valid_for_s", 30.0), 2, 120)),
    )


def render_history_png(history: list[Situation]) -> bytes:
    """Two-panel plot of recent Lady-relative motion: along-track vs radial, along-track vs normal."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b = np.array([h.bandit_rel_pos for h in history])
    g = np.array([h.guard_rel_pos for h in history])
    fig, axes = plt.subplots(1, 2, figsize=(6, 3), dpi=80)
    for ax, j, lab in ((axes[0], 0, "radial x [m]"), (axes[1], 2, "normal z [m]")):
        ax.plot(b[:, 1], b[:, j], "b-", label="bandit")
        ax.plot(g[:, 1], g[:, j], "r-", label="guard")
        ax.plot(b[-1, 1], b[-1, j], "bo")
        ax.plot(g[-1, 1], g[-1, j], "ro")
        ax.plot(0, 0, "k*", ms=10, label="lady")
        ax.set_xlabel("along-track y [m]")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


class HaikuTactician:
    name = "haiku"

    def __init__(self, model: str = HAIKU_MODEL, effort: str = "low", use_images: bool = False,
                 history_points: int = 8, llm: StructuredLLM | None = None):
        self.llm = llm or StructuredLLM(model, effort=effort, max_tokens=4000)
        self.use_images = use_images
        self.history_points = history_points

    def update(self, history: list[Situation], brief: MissionBrief, current: TacticalGoal):
        idx = np.linspace(0, len(history) - 1, min(self.history_points, len(history))).astype(int)
        hist = [history[i].compact() for i in idx]
        text = (f"Mission brief:\n{json.dumps(brief.to_dict())}\n\nCurrent goal:\n{json.dumps(current.to_dict())}\n\n"
                f"Recent states, oldest first:\n{json.dumps(hist)}\n\nReturn the updated tactical goal.")
        content = [{"type": "text", "text": text}]
        if self.use_images:
            png = render_history_png(history)
            content.insert(0, {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                           "data": base64.b64encode(png).decode()}})
        res = self.llm.call(TACTICAL_SYSTEM, content, GOAL_SCHEMA)
        return _goal_from_dict(res.data, history[-1].t, brief), res


class RuleTactician:
    """No-model tactical layer: routes out of plane around the Guard when it sits between Bandit and Lady."""

    name = "rule"

    def update(self, history: list[Situation], brief: MissionBrief, current: TacticalGoal):
        s = history[-1]
        goal = TacticalGoal.from_brief(brief)
        goal.issued_at = s.t
        if s.guard_between and s.lady_dist > 3 * brief.guard_keepout_m:
            # pass the guard's along-track station offset out of plane by twice the keep-out radius
            side = 1.0 if s.bandit_rel_pos[2] >= s.guard_rel_pos[2] else -1.0
            goal.waypoint_lvlh_m = [float(s.guard_rel_pos[0]), float(s.guard_rel_pos[1]),
                                    float(s.guard_rel_pos[2] + side * 2 * brief.guard_keepout_m)]
            goal.mode_hint = "GO_TO_WAYPOINT"
            goal.rationale = "guard blocks the direct line; bypass out of plane"
        else:
            goal.rationale = "direct approach"
        return goal, None
