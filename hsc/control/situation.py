"""Compress the raw 21-float lbg1 observation into a small semantic snapshot.

All relative quantities are in the Lady's LVLH frame: x = radial out, y = along-track
(Lady's direction of motion), z = orbit normal. Distances in metres, speeds in m/s.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from hsc.sim import orbits as O


@dataclass
class Situation:
    t: float
    time_left: float
    fuel_kg: float
    fuel_fraction: float
    mass_kg: float
    bandit_rel_pos: np.ndarray  # bandit relative to lady
    bandit_rel_vel: np.ndarray
    guard_rel_pos: np.ndarray  # guard relative to lady
    guard_rel_vel: np.ndarray
    lady_dist: float
    lady_range_rate: float  # negative = closing
    guard_dist: float  # bandit to guard
    guard_range_rate: float  # negative = guard closing on bandit
    guard_between: bool  # guard roughly on the line from bandit to lady
    mean_motion: float

    def compact(self) -> dict:
        """Rounded dict that is cheap to send to a model."""
        r = lambda a: [round(float(x), 1) for x in a]  # noqa: E731
        return {
            "t_s": round(self.t, 1),
            "time_left_s": round(self.time_left, 1),
            "fuel_fraction": round(self.fuel_fraction, 3),
            "lady_dist_m": round(self.lady_dist, 1),
            "lady_range_rate_mps": round(self.lady_range_rate, 2),
            "guard_dist_m": round(self.guard_dist, 1),
            "guard_range_rate_mps": round(self.guard_range_rate, 2),
            "guard_between_bandit_and_lady": self.guard_between,
            "bandit_pos_lvlh_m": r(self.bandit_rel_pos),
            "bandit_vel_lvlh_mps": r(self.bandit_rel_vel),
            "guard_pos_lvlh_m": r(self.guard_rel_pos),
            "guard_vel_lvlh_mps": r(self.guard_rel_vel),
        }


def situation_from_obs(obs, prop_capacity: float, episode_timeout: float) -> Situation:
    obs = np.asarray(obs, dtype=float)
    b_p, b_v = obs[3:6], obs[6:9]
    l_p, l_v = obs[9:12], obs[12:15]
    g_p, g_v = obs[15:18], obs[18:21]
    rb, vb = O.relative_lvlh(l_p, l_v, b_p, b_v)
    rg, vg = O.relative_lvlh(l_p, l_v, g_p, g_v)
    lady_dist = float(np.linalg.norm(rb))
    lady_rr = float(np.dot(rb, vb) / lady_dist) if lady_dist > 0 else 0.0
    d_bg = rg - rb
    guard_dist = float(np.linalg.norm(d_bg))
    guard_rr = float(np.dot(d_bg, vg - vb) / guard_dist) if guard_dist > 0 else 0.0
    # guard "between" if it is closer to the lady than we are and within 30 deg of the bandit->lady line
    to_lady = -rb
    cosang = np.dot(d_bg, to_lady) / (guard_dist * lady_dist + 1e-9)
    guard_between = bool(cosang > np.cos(np.deg2rad(30)) and guard_dist < lady_dist)
    return Situation(
        t=float(obs[0]),
        time_left=max(0.0, episode_timeout - float(obs[0])),
        fuel_kg=float(obs[2]),
        fuel_fraction=float(obs[2] / prop_capacity) if prop_capacity > 0 else 0.0,
        mass_kg=float(obs[1]),
        bandit_rel_pos=rb,
        bandit_rel_vel=vb,
        guard_rel_pos=rg,
        guard_rel_vel=vg,
        lady_dist=lady_dist,
        lady_range_rate=lady_rr,
        guard_dist=guard_dist,
        guard_range_rate=guard_rr,
        guard_between=guard_between,
        mean_motion=O.mean_motion(float(np.linalg.norm(l_p))),
    )


def situation_dict(s: Situation) -> dict:
    d = asdict(s)
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            d[k] = v.tolist()
    return d
