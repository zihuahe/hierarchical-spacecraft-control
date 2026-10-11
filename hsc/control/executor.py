"""Deterministic executor: turns a maneuver primitive into an lbg1 thrust action.

This is the "game code keeps state authority" half of the immediate-control layer. It
runs every control tick in microseconds and never calls a model. Guidance is done in the
Lady's LVLH frame with a Clohessy-Wiltshire feed-forward for relative orbital dynamics.
"""

from __future__ import annotations

import numpy as np

from hsc.control.plans import PRIMITIVES, MissionBrief, TacticalGoal
from hsc.control.situation import Situation
from hsc.sim import orbits as O

TAU = 2.0  # [s] velocity-tracking time constant
GLIDE_DECEL = 0.6  # [m/s^2] planned braking deceleration (below the ~1 m/s^2 lateral capability)
EVADE_SPEED = 6.0  # [m/s]


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros_like(v)


def _cw_free_accel(r, v, n):
    return np.array([3 * n * n * r[0] + 2 * n * v[1], -2 * n * v[0], -n * n * r[2]])


def _goto(r, target, vmax, v_final):
    """Desired velocity to reach target along a braking glide slope."""
    d = target - r
    dist = np.linalg.norm(d)
    speed = min(vmax, np.sqrt(v_final**2 + 2 * GLIDE_DECEL * dist), 0.2 * dist + v_final)
    return _unit(d) * speed


class Executor:
    def __init__(self, control_period: float = 0.5):
        self.control_period = control_period

    def desired_velocity(self, s: Situation, primitive: str, goal: TacticalGoal, brief: MissionBrief):
        r, v = s.bandit_rel_pos, s.bandit_rel_vel
        vmax = min(goal.max_closing_speed_mps, brief.max_closing_speed_mps)
        vf = brief.final_approach_speed_mps
        if primitive == "APPROACH_LADY":
            return _goto(r, np.zeros(3), vmax, vf)
        if primitive == "GO_TO_WAYPOINT":
            wp = np.zeros(3) if goal.waypoint_lvlh_m is None else np.asarray(goal.waypoint_lvlh_m, dtype=float)
            return _goto(r, wp, vmax, 0.0)
        if primitive == "BRAKE":
            return np.zeros(3)
        if primitive == "HOLD_STANDOFF":
            target = _unit(r) * goal.standoff_m if np.linalg.norm(r) > 1 else np.array([0.0, -goal.standoff_m, 0.0])
            return _goto(r, target, vmax, 0.0)
        if primitive == "EVADE_GUARD":
            away = _unit(r - s.guard_rel_pos)
            to_lady = _unit(-r)
            side = _unit(to_lady - np.dot(to_lady, away) * away)  # around the guard, toward the lady
            if np.linalg.norm(side) < 1e-6:
                side = np.array([0.0, 0.0, 1.0])
            return _unit(away + side) * EVADE_SPEED
        if primitive == "COAST":
            return None
        raise ValueError(f"unknown primitive {primitive}")

    def act(self, s: Situation, lady_pos, lady_vel, primitive: str, goal: TacticalGoal, brief: MissionBrief) -> dict:
        assert primitive in PRIMITIVES, primitive
        v_des = self.desired_velocity(s, primitive, goal, brief)
        if v_des is None:
            force_cbci = np.zeros(3)
        else:
            a_cmd = (v_des - s.bandit_rel_vel) / TAU
            a_thrust_lvlh = a_cmd - _cw_free_accel(s.bandit_rel_pos, s.bandit_rel_vel, s.mean_motion)
            basis = O.lvlh_basis(np.asarray(lady_pos), np.asarray(lady_vel))
            force_cbci = basis.T @ (a_thrust_lvlh * s.mass_kg)
            if s.fuel_fraction <= brief.fuel_reserve_fraction and primitive != "EVADE_GUARD":
                force_cbci = np.zeros(3)  # keep the reserve for evasion
        return {"burn_vec": [*force_cbci.tolist(), self.control_period], "vec_type": 1, "ref_frame": 1}
