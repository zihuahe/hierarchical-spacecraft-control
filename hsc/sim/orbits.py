"""Two-body orbital helpers and reference frames used by the lbg1 stand-in.

Frames (all right-handed):
- CBCI: celestial-body-centred inertial (what lbg1 observations use).
- NTW: x = in-plane normal to velocity (roughly radial out), y = along velocity, z = orbit normal.
- LVLH (Hill) of a reference vessel: x = radial out, y = along-track, z = orbit normal.
- Body of the Bandit: forward = +T, right = -W, down = -N (vessel held prograde).
"""

from __future__ import annotations

import numpy as np

# Kerbin, as used by Kerbal Space Program
MU_KERBIN = 3.5316e12  # [m^3/s^2]
R_KERBIN = 600_000.0  # [m]
G0 = 9.80665  # [m/s^2]


def two_body_accel(pos: np.ndarray, mu: float = MU_KERBIN) -> np.ndarray:
    """Gravitational acceleration for one or more positions, shape (..., 3)."""
    r = np.linalg.norm(pos, axis=-1, keepdims=True)
    return -mu * pos / r**3


def propagate_coast(state: np.ndarray, dt: float, step: float = 1.0, mu: float = MU_KERBIN) -> np.ndarray:
    """RK4-propagate unforced states, shape (..., 6), by dt seconds (dt may be negative)."""
    state = np.array(state, dtype=float)
    n = max(1, int(np.ceil(abs(dt) / step)))
    h = dt / n

    def f(s):
        return np.concatenate([s[..., 3:], two_body_accel(s[..., :3], mu)], axis=-1)

    for _ in range(n):
        k1 = f(state)
        k2 = f(state + 0.5 * h * k1)
        k3 = f(state + 0.5 * h * k2)
        k4 = f(state + h * k3)
        state = state + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return state


def circular_speed(r: float, mu: float = MU_KERBIN) -> float:
    return float(np.sqrt(mu / r))


def ntw_basis(pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """Rows are the N, T, W unit vectors expressed in CBCI."""
    t = vel / np.linalg.norm(vel)
    w = np.cross(pos, vel)
    w /= np.linalg.norm(w)
    n = np.cross(t, w)
    return np.vstack([n, t, w])


def body_basis(pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """Rows are the forward, right, down unit vectors of the prograde-held Bandit, in CBCI."""
    n, t, w = ntw_basis(pos, vel)
    return np.vstack([t, -w, -n])


def lvlh_basis(pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """Rows are the radial, along-track, normal unit vectors of a reference vessel, in CBCI."""
    x = pos / np.linalg.norm(pos)
    z = np.cross(pos, vel)
    z /= np.linalg.norm(z)
    y = np.cross(z, x)
    return np.vstack([x, y, z])


def relative_lvlh(ref_pos, ref_vel, pos, vel):
    """Position and velocity of a vessel relative to a reference, in the reference's rotating LVLH frame."""
    ref_pos, ref_vel, pos, vel = (np.asarray(a, dtype=float) for a in (ref_pos, ref_vel, pos, vel))
    basis = lvlh_basis(ref_pos, ref_vel)
    omega = np.cross(ref_pos, ref_vel) / np.dot(ref_pos, ref_pos)
    dr = pos - ref_pos
    dv = vel - ref_vel - np.cross(omega, dr)
    return basis @ dr, basis @ dv


def mean_motion(r: float, mu: float = MU_KERBIN) -> float:
    return float(np.sqrt(mu / r**3))
