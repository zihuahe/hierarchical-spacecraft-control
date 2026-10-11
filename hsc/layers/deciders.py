"""Immediate-control deciders: pick one maneuver primitive from the current state and context.

- RuleDecider: hand-written rules, no model (the no-LLM fast baseline and a Jev stand-in).
- SystemOneDecider: one choice question per tick to a System One server: TypeSafe's hosted Jev
  (JevDecider) or a local open-weights Kev server such as Kev-0.8B (KevDecider). Either a guided
  maneuver or a raw action ID.
- DiscreteRuleDecider: no-model baseline in the raw action space.
- FlatLLMDecider: a Claude model picks the primitive directly every decision (flat baselines).
"""

from __future__ import annotations

import json
import os
import time
import urllib.request

import numpy as np

from hsc.control.plans import DISCRETE_ACTIONS, PRIMITIVES, MissionBrief, TacticalGoal
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

DISCRETE_INSTRUCTIONS = (
    "You are the fast controller of a spacecraft (the Bandit) trying to reach within 5 m of a passive "
    "target (the Lady) while never letting a pursuing Guard within 5 m. Follow the mission brief and the "
    "current tactical goal. Positions are Lady-relative: +y is ahead along the orbit, +x is away from the "
    "planet. Which single thrust action should the Bandit take right now?"
)


def _relative_hints(s: Situation) -> dict:
    """Plain-language cues a small classifier can use to pick a raw thrust direction."""
    rb, vb = s.bandit_rel_pos, s.bandit_rel_vel
    return {
        "lady_is": ("ahead" if rb[1] < 0 else "behind") + " and " + ("below" if rb[0] > 0 else "above"),
        "along_track_gap_m": round(float(-rb[1]), 1),
        "radial_gap_m": round(float(-rb[0]), 1),
        "along_track_closing_mps": round(float(-np.sign(rb[1]) * vb[1]) if abs(rb[1]) > 1e-6 else 0.0, 2),
        "radial_closing_mps": round(float(-np.sign(rb[0]) * vb[0]) if abs(rb[0]) > 1e-6 else 0.0, 2),
    }


class SystemOneDecider:
    """Any TypeSafe System One server: hosted Jev, or a local Kev server (same API).

    POST {base}/v1/systemone with a ``state`` string and one ``choice`` question; read
    ``answers[name].choice`` / ``probabilities`` / ``confidence`` from the response. Fields follow
    TypeSafe's public docs and Kev's README, whose server is a drop-in for Jev.

    ``options="primitives"`` asks for a guided maneuver (the executor flies it);
    ``options="discrete"`` asks for a raw action ID (coast/forward/backward/left/right/up/down) that
    the adapter maps to a fixed-strength throttle, as in the project's first-version plan.
    """

    def __init__(self, name: str, base_url: str, model: str, api_key: str, options: str = "primitives",
                 timeout: float = 10.0):
        assert options in ("primitives", "discrete"), options
        self.name = name if options == "primitives" else f"{name}:discrete"
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.options = options
        self.timeout = timeout

    def _state_text(self, s: Situation, goal: TacticalGoal, brief: MissionBrief) -> str:
        state = {
            "mission": {"objective": brief.objective, "approach": brief.approach,
                        "guard_keepout_m": brief.guard_keepout_m},
            "tactical_goal": {"mode_hint": goal.mode_hint, "waypoint_lvlh_m": goal.waypoint_lvlh_m,
                              "rationale": goal.rationale},
            "state": s.compact(),
            "flags": context_flags(s, goal),
        }
        if self.options == "discrete":
            state["hints"] = _relative_hints(s)
        return json.dumps(state)

    def request_body(self, s: Situation, goal: TacticalGoal, brief: MissionBrief) -> dict:
        if self.options == "discrete":
            question = {"type": "choice", "instructions": DISCRETE_INSTRUCTIONS, "criteria": dict(DISCRETE_ACTIONS)}
        else:
            question = {"type": "choice", "instructions": JEV_INSTRUCTIONS, "criteria": dict(PRIMITIVES)}
        return {"model": self.model, "state": self._state_text(s, goal, brief), "questions": {"maneuver": question}}

    def decide(self, s: Situation, goal: TacticalGoal, brief: MissionBrief):
        body = self.request_body(s, goal, brief)
        req = urllib.request.Request(
            f"{self.base_url}/v1/systemone", data=json.dumps(body).encode(), method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            resp = json.loads(r.read())
        latency = time.perf_counter() - t0
        ans = resp["answers"]["maneuver"]
        choice = ans["choice"]
        allowed = DISCRETE_ACTIONS if self.options == "discrete" else PRIMITIVES
        if choice not in allowed:
            raise RuntimeError(f"{self.name} returned unknown choice {choice!r}")
        usage = resp.get("usage", {})
        info = CallResult(data=ans, latency_s=latency, model=resp.get("model", self.model),
                          input_tokens=usage.get("input_tokens", 0), output_tokens=usage.get("output_tokens", 0),
                          extra={"confidence": ans.get("confidence"), "server_latency_ms": resp.get("latency_ms")})
        return choice, None, info


def JevDecider(model: str = "jev-latest", api_key: str | None = None, base_url: str | None = None,
               options: str = "primitives", timeout: float = 5.0) -> SystemOneDecider:
    """TypeSafe's hosted Jev (needs TYPESAFE_API_KEY and network access to api.typesafe.ai)."""
    api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
    if not api_key:
        raise RuntimeError("TYPESAFE_API_KEY is not set")
    base_url = base_url or os.environ.get("TYPESAFE_API_BASE") or "https://api.typesafe.ai"
    return SystemOneDecider("jev", base_url, model, api_key, options, timeout)


def KevDecider(model: str = "kev-latest", base_url: str | None = None, api_key: str | None = None,
               options: str = "primitives", timeout: float = 10.0) -> SystemOneDecider:
    """A local Kev server (open weights, e.g. Kev-0.8B), started with
    ``uv run --extra serve python -m kev.serve --run jaredpalmer/kev-0.8b --port 8009``."""
    base_url = base_url or os.environ.get("KEV_BASE_URL") or "http://127.0.0.1:8009"
    api_key = api_key or os.environ.get("KEV_API_KEY") or "local"
    return SystemOneDecider("kev", base_url, model, api_key, options, timeout)


class DiscreteRuleDecider:
    """No-model baseline in the raw action space: bang-bang version of the rule controller.

    Runs the rule decider's maneuver through the guided executor, then fires whichever single axis
    the guided command leans on most (or coasts below a deadband).
    """

    name = "rule:discrete"

    def __init__(self, deadband: float = 0.15):
        from hsc.control.executor import Executor

        self.rule = RuleDecider()
        self.executor = Executor(0.25)
        self.deadband = deadband
        self.obs = None  # the runner sets this before each decision

    def decide(self, s: Situation, goal: TacticalGoal, brief: MissionBrief):
        primitive, _, _ = self.rule.decide(s, goal, brief)
        fwd, right = self.executor.act(s, self.obs, primitive, goal, brief)["burn_vec"][:2]
        if max(abs(fwd), abs(right)) < self.deadband:
            return "coast", None, None
        if abs(fwd) >= abs(right):
            return ("forward" if fwd > 0 else "backward"), None, None
        return ("right" if right > 0 else "left"), None, None


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
