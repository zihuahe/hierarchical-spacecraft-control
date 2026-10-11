"""Mission strategy layer (slow): sets objectives, constraints and approach once per episode."""

from __future__ import annotations

import json

from hsc.control.plans import PRIMITIVES, MissionBrief
from hsc.control.situation import Situation
from hsc.layers.llm import OPUS_MODEL, StructuredLLM

SCENARIO_TEXT = """\
You command the Bandit spacecraft in the Lady-Bandit-Guard orbital game (KSPDG lbg1) around Kerbin
(about 75 km altitude, orbital period about 31 minutes). The Lady is a passive target. The Guard has
the same capability as the Bandit and pursues it (in the scenario of record, an advanced Guard that
re-aims continuously and closes at up to about 12 m/s; easier variants zero relative velocity, burn
straight at the Bandit, then coast). Episodes end after 240 s, when the Bandit gets within
5 m of the Lady (win), or when the Guard gets within 5 m of the Bandit (loss).

Score (lower is better): closest_lady_bandit_distance^2 + 1e6 / (closest_bandit_guard_distance + 0.1).
So 100 m of miss distance from the Lady costs about as much as letting the Guard get within 100 m.

Bandit capability: RCS only, two commandable axes in the orbital plane: forward/backward along-track
(about 2 m/s^2) and right/left roughly radial (about 1 m/s^2). No out-of-plane thrust. Roughly 250 m/s
of delta-v in total, so the Bandit cannot afford long chases. Scenario of record: LBG1_LG7_I2_V1
(advanced Guard; all three vessels start in the same circular orbit, Guard between Bandit and Lady).

Control hierarchy under you: a tactical planner re-plans short-term goals every few seconds from
recent states, and a fast controller picks one of these maneuver primitives every 0.5 s:
""" + "\n".join(f"- {k}: {v}" for k, v in PRIMITIVES.items()) + """

Relative coordinates are in the Lady's LVLH frame: x radial out, y along-track (Lady's direction of
motion), z orbit normal, metres.
"""

BRIEF_SCHEMA = {
    "type": "object",
    "properties": {
        "objective": {"type": "string"},
        "approach": {"type": "string", "description": "2-4 sentences of strategy for the lower layers"},
        "guard_keepout_m": {"type": "number"},
        "max_closing_speed_mps": {"type": "number"},
        "final_approach_speed_mps": {"type": "number"},
        "fuel_reserve_fraction": {"type": "number"},
        "tactical_period_s": {"type": "number"},
        "phases": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["objective", "approach", "guard_keepout_m", "max_closing_speed_mps",
                 "final_approach_speed_mps", "fuel_reserve_fraction", "tactical_period_s", "phases"],
    "additionalProperties": False,
}

_LIMITS = {
    "guard_keepout_m": (20.0, 1000.0),
    "max_closing_speed_mps": (1.0, 40.0),
    "final_approach_speed_mps": (0.2, 5.0),
    "fuel_reserve_fraction": (0.0, 0.5),
    "tactical_period_s": (2.0, 60.0),
}


def brief_from_dict(d: dict) -> MissionBrief:
    kw = {k: d[k] for k in MissionBrief.__dataclass_fields__ if k in d}
    for k, (lo, hi) in _LIMITS.items():
        if k in kw:
            kw[k] = float(min(hi, max(lo, kw[k])))
    return MissionBrief(**kw)


class OpusStrategist:
    name = "opus"

    def __init__(self, model: str = OPUS_MODEL, effort: str = "medium", llm: StructuredLLM | None = None):
        self.llm = llm or StructuredLLM(model, effort=effort, max_tokens=16000)

    def plan(self, s: Situation, scenario_label: str):
        content = (f"Scenario: {scenario_label}\nInitial situation:\n{json.dumps(s.compact())}\n\n"
                   "Write the mission brief for the layers below you.")
        res = self.llm.call(SCENARIO_TEXT, content, BRIEF_SCHEMA)
        return brief_from_dict(res.data), res


class FixedStrategist:
    """No-model strategy layer: returns the hand-written default brief."""

    name = "fixed"

    def plan(self, s: Situation, scenario_label: str):
        return MissionBrief(), None
