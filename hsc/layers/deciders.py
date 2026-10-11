"""Immediate-control deciders: pick one maneuver primitive from the current state and context.

- RuleDecider: hand-written rules, no model (the no-LLM fast baseline and a Jev stand-in).
- JevDecider: TypeSafe Jev "System One" classifier, one choice question per tick.
- FlatLLMDecider: a Claude model picks the primitive directly every decision (flat baselines).
"""

from __future__ import annotations

import json
import os
import time
import urllib.request

import numpy as np

from hsc.control.plans import PRIMITIVES, MissionBrief, TacticalGoal
from hsc.control.situation import Situation
from hsc.layers.llm import CallResult, StructuredLLM
from hsc.layers.strategist import SCENARIO_TEXT


def context_flags(s: Situation, goal: TacticalGoal) -> dict:
    wp_dist = None
    if goal.waypoint_lvlh_m is not None:
        wp_dist = float(np.linalg.norm(np.asarray(goal.waypoint_lvlh_m) - s.bandit_rel_pos))
    return {
        "guard_inside_keepout": s.guard_dist < goal.guard_keepout_m,
        "guard_closing": s.guard_range_rate < -0.2,
        "waypoint_dist_m": None if wp_dist is None else round(wp_dist, 1),
    }


class RuleDecider:
    name = "rule"

    def decide(self, s: Situation, goal: TacticalGoal, brief: MissionBrief):
        f = context_flags(s, goal)
        if f["guard_inside_keepout"] and f["guard_closing"]:
            return "EVADE_GUARD", None, None
        if goal.mode_hint in PRIMITIVES:
            if goal.mode_hint == "GO_TO_WAYPOINT" and (f["waypoint_dist_m"] is None or f["waypoint_dist_m"] < 20):
                return "APPROACH_LADY", None, None
            return goal.mode_hint, None, None
        if f["waypoint_dist_m"] is not None and f["waypoint_dist_m"] > 20:
            return "GO_TO_WAYPOINT", None, None
        return "APPROACH_LADY", None, None


JEV_INSTRUCTIONS = (
    "You are the fast controller of a spacecraft (the Bandit) trying to reach within 5 m of a passive "
    "target (the Lady) while never letting a pursuing Guard within 5 m. Follow the mission brief and the "
    "current tactical goal. Given the current state, which maneuver should the Bandit execute for the "
    "next half second?"
)


class JevDecider:
    """TypeSafe Jev via POST {base}/v1/systemone.

    Request/response fields follow TypeSafe's public docs (LiteLLM pass-through page): a ``state``
    string, a ``questions`` map of typed questions (``choice`` with ``instructions`` and ``criteria``),
    and ``answers[name].choice`` / ``probabilities`` / ``confidence`` in the response.
    """

    name = "jev"

    def __init__(self, model: str = "jev-latest", api_key: str | None = None, base_url: str | None = None,
                 timeout: float = 5.0):
        self.model = model
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        self.base_url = (base_url or os.environ.get("TYPESAFE_API_BASE") or "https://api.typesafe.ai").rstrip("/")
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError("TYPESAFE_API_KEY is not set")

    def _state_text(self, s: Situation, goal: TacticalGoal, brief: MissionBrief) -> str:
        return json.dumps({
            "mission": {"objective": brief.objective, "approach": brief.approach,
                        "guard_keepout_m": brief.guard_keepout_m},
            "tactical_goal": {"mode_hint": goal.mode_hint, "waypoint_lvlh_m": goal.waypoint_lvlh_m,
                              "rationale": goal.rationale},
            "state": s.compact(),
            "flags": context_flags(s, goal),
        })

    def decide(self, s: Situation, goal: TacticalGoal, brief: MissionBrief):
        body = {
            "model": self.model,
            "state": self._state_text(s, goal, brief),
            "questions": {"maneuver": {"type": "choice", "instructions": JEV_INSTRUCTIONS,
                                       "criteria": dict(PRIMITIVES)}},
        }
        req = urllib.request.Request(
            f"{self.base_url}/v1/systemone", data=json.dumps(body).encode(), method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            resp = json.loads(r.read())
        latency = time.perf_counter() - t0
        ans = resp["answers"]["maneuver"]
        choice = ans["choice"]
        if choice not in PRIMITIVES:
            raise RuntimeError(f"Jev returned unknown choice {choice!r}")
        usage = resp.get("usage", {})
        info = CallResult(data=ans, latency_s=latency, model=resp.get("model", self.model),
                          input_tokens=usage.get("input_tokens", 0), output_tokens=usage.get("output_tokens", 0),
                          extra={"confidence": ans.get("confidence")})
        return choice, None, info


DECIDER_SCHEMA = {
    "type": "object",
    "properties": {
        "primitive": {"type": "string", "enum": list(PRIMITIVES)},
        "waypoint_lvlh_m": {"anyOf": [{"type": "array", "items": {"type": "number"}}, {"type": "null"}]},
    },
    "required": ["primitive", "waypoint_lvlh_m"],
    "additionalProperties": False,
}


class FlatLLMDecider:
    """One Claude model makes every control decision directly (no hierarchy)."""

    def __init__(self, model: str, effort: str = "low", llm: StructuredLLM | None = None):
        self.llm = llm or StructuredLLM(model, effort=effort, max_tokens=4000)
        self.name = f"flat:{model}"

    def decide(self, s: Situation, goal: TacticalGoal, brief: MissionBrief):
        content = (f"Current state:\n{json.dumps(s.compact())}\nFlags: {json.dumps(context_flags(s, goal))}\n"
                   "Choose the maneuver primitive to execute now (and a waypoint if you choose GO_TO_WAYPOINT).")
        res = self.llm.call(SCENARIO_TEXT + "\nYou are the controller. Decide the next maneuver.", content,
                            DECIDER_SCHEMA)
        override = None
        wp = res.data.get("waypoint_lvlh_m")
        if wp is not None and len(wp) == 3:
            override = {"waypoint_lvlh_m": [float(x) for x in wp]}
        return res.data["primitive"], override, res
