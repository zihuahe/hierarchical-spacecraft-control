"""Data passed down the hierarchy: mission brief (strategy) and tactical goal (short-term)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# Maneuver primitives the immediate-control layer chooses between. The executor turns
# each one into a thrust command, so every layer shares the same physical capability.
PRIMITIVES: dict[str, str] = {
    "APPROACH_LADY": "Close on the Lady along a braking glide slope, capped at the allowed closing speed.",
    "GO_TO_WAYPOINT": "Fly to the tactical waypoint (Lady-relative LVLH metres), then hold there.",
    "BRAKE": "Null velocity relative to the Lady.",
    "EVADE_GUARD": "Accelerate away from the Guard, sidestepping around it toward the Lady.",
    "HOLD_STANDOFF": "Hold position at the standoff distance from the Lady on the current bearing.",
    "COAST": "No thrust; save fuel.",
}


@dataclass
class MissionBrief:
    objective: str = "Get the Bandit within 5 m of the Lady while never letting the Guard within 5 m."
    approach: str = "Default brief: approach directly, evade the Guard when it closes inside the keep-out radius."
    guard_keepout_m: float = 150.0
    max_closing_speed_mps: float = 15.0
    final_approach_speed_mps: float = 2.0
    fuel_reserve_fraction: float = 0.1
    tactical_period_s: float = 10.0
    phases: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TacticalGoal:
    mode_hint: str = "AUTO"  # a PRIMITIVES key, or AUTO to let the fast layer decide
    waypoint_lvlh_m: list[float] | None = None
    max_closing_speed_mps: float = 15.0
    guard_keepout_m: float = 150.0
    standoff_m: float = 300.0
    rationale: str = ""
    issued_at: float = 0.0
    valid_for_s: float = 30.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_brief(cls, brief: MissionBrief) -> "TacticalGoal":
        return cls(max_closing_speed_mps=brief.max_closing_speed_mps, guard_keepout_m=brief.guard_keepout_m,
                   rationale="initial goal from mission brief")
