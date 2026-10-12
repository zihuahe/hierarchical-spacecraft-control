"""lbg1_lite: a pure-Python stand-in for the KSPDG lbg1 Lady-Bandit-Guard environment.

It copies lbg1's interface so an agent written against it can later run on real KSPDG:
- observation: 21 floats, same layout as ``LadyBanditGuardGroup1Env.get_observation``
- action: ``{"burn_vec": [x, y, z, dur_s], "vec_type": 0|1, "ref_frame": 0|1|2}``
  (vec_type 0 = throttle, 1 = thrust in N; ref_frame 0 = body, 1 = CBCI, 2 = NTW)
- info keys and the single-value score: ``dm_lb**2 + 1e6 / (dm_bg + 0.1)`` (lower is better)
- termination: Lady captured (< 5 m), Bandit captured by Guard (< 5 m), or 240 s timeout
- the controller API used for this project (scenario LBG1_LG7_I2_V1): forward/backward and
  right/left throttle in [-1, 1] plus a duration of 0-10 s. With ``two_axis=True`` (default) the
  up/down axis is ignored and durations are clipped to that range.

Guard policies: ``lg0`` (passive) and ``lg1`` (zero/pursue/coast heuristic) follow the public
KSPDG source. LG3 and above are code-obfuscated upstream, so ``lg7`` here is a proxy: an
aggressive continuous pursuer that re-points every step (with a 1 s attitude lag) and never coasts.

Physics is point-mass two-body motion around Kerbin with RK4 integration. RCS thrust and
Isp come from the KSPDG source (8 x 1 kN fore/aft, 4 x 1 kN per lateral axis, Isp 240 s).
Vessel masses and the exact initial orbits are approximations of the KSP save files, which
are not public. The real environment runs in wall-clock time inside KSP; here time only
advances when ``step`` is called, and the experiment runner charges agent latency explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hsc.sim import orbits as O

DEFAULT_EPISODE_TIMEOUT = 240.0  # [s], as in kspdg lbg1_base
DEFAULT_CAPTURE_DIST = 5.0  # [m], as in kspdg lbg1_base
SCORE_BG_SCALE = 1e6
SCORE_BG_OFFSET = 0.1

LADY_ORBIT_RADIUS = O.R_KERBIN + 75_000.0  # [m] approximate


@dataclass
class VesselParams:
    dry_mass: float = 3600.0  # [kg] approximate
    prop_mass: float = 400.0  # [kg] approximate
    isp: float = 240.0  # [s] kspdg PARAMS.BANDIT.RCS.VACUUM_SPECIFIC_IMPULSE
    thrust_fwd: float = 8000.0  # [N] 8 nozzles x 1000 N
    thrust_rev: float = 8000.0
    thrust_lat: float = 4000.0  # [N] 4 nozzles x 1000 N, each of right/left/up/down


@dataclass
class GuardPursuitParams:
    """Parameters of the lg1 heuristic Guard, defaults copied from kspdg lg1_envs."""

    pursuit_throttle: float = 1.0
    min_speed_thresh: float = 0.2  # [m/s]
    max_speed_thresh: float = 10.0  # [m/s]
    ang_vel_pos_thresh: float = np.deg2rad(15.0)
    loop_timeout: float = 30.0  # [s]
    reorient_time: float = 1.0  # [s]
    min_burn_time: float = 1.0  # [s]


@dataclass
class _GuardState:
    phase: str = "reorient_zero"
    phase_start: float = 0.0


@dataclass
class EpisodeMetrics:
    closest_lb: float = np.inf
    closest_lb_time: float = 0.0
    closest_bg: float = np.inf
    closest_lg: float = np.inf
    min_lb_distspeed: float = np.inf
    trajectory: list = field(default_factory=list)


class LBG1LiteEnv:
    INIT_CONDS = ("i1", "i2")
    LADY_GUARD_POLICIES = ("lg0", "lg1", "lg7")
    MAX_BURN_DURATION = 10.0  # [s] controller API limit

    def __init__(
        self,
        init_cond: str = "i2",
        lady_guard_policy: str = "lg1",
        episode_timeout: float = DEFAULT_EPISODE_TIMEOUT,
        capture_dist: float = DEFAULT_CAPTURE_DIST,
        sim_dt: float = 0.1,
        seed: int | None = None,
        record_every: float = 1.0,
        two_axis: bool = True,
    ):
        assert init_cond in self.INIT_CONDS, init_cond
        assert lady_guard_policy in self.LADY_GUARD_POLICIES, lady_guard_policy
        self.init_cond = init_cond
        self.lady_guard_policy = lady_guard_policy
        self.episode_timeout = episode_timeout
        self.capture_dist = capture_dist
        self.sim_dt = sim_dt
        self.seed = seed
        self.record_every = record_every
        self.two_axis = two_axis
        self.bandit_params = VesselParams()
        self.guard_params = VesselParams()  # Bandit and Guard have identical capabilities
        self.lady_mass = 4000.0
        self.guard_pursuit = GuardPursuitParams()
        self.reset()

    # ------------------------------------------------------------------ setup

    def _initial_states(self) -> np.ndarray:
        """States (lady, bandit, guard) x (pos, vel) in CBCI at t = 0."""
        r_l = LADY_ORBIT_RADIUS
        v_l = O.circular_speed(r_l)

        def on_circle(arc_offset):
            th = arc_offset / r_l
            return np.array([r_l * np.cos(th), r_l * np.sin(th), 0.0, -v_l * np.sin(th), v_l * np.cos(th), 0.0])

        lady = on_circle(0.0)
        if self.init_cond == "i2":
            # all in the same circular orbit; Guard ~600 m retrograde of Lady, Bandit ~2000 m retrograde of Guard
            guard = on_circle(-600.0)
            bandit = on_circle(-2600.0)
        else:
            # i1: Guard ~600 m prograde of Lady in the same orbit; Bandit on an elliptical orbit with an
            # upcoming tangential conjunction at its apoapsis with the Lady (here at t = 120 s, 500 m below)
            t_conj = 120.0
            lady_conj = O.propagate_coast(lady, t_conj)
            r_hat = lady_conj[:3] / np.linalg.norm(lady_conj[:3])
            t_hat = lady_conj[3:] / np.linalg.norm(lady_conj[3:])
            r_apo = r_l - 500.0
            r_peri = r_l - 8000.0
            a = 0.5 * (r_apo + r_peri)
            v_apo = np.sqrt(O.MU_KERBIN * (2.0 / r_apo - 1.0 / a))
            bandit_conj = np.concatenate([r_hat * r_apo, t_hat * v_apo])
            bandit = O.propagate_coast(bandit_conj, -t_conj)
            guard = on_circle(600.0)

        states = np.vstack([lady, bandit, guard])
        if self.seed is not None:
            rng = np.random.default_rng(self.seed)
            dp = rng.normal(0.0, 30.0, size=(2, 3))
            dv = rng.normal(0.0, 0.1, size=(2, 3))
            if self.two_axis:  # keep the game planar: the Bandit has no out-of-plane thrust
                dp[:, 2] = 0.0
                dv[:, 2] = 0.0
            states[1:, :3] += dp
            states[1:, 3:] += dv
        return states

    def reset(self):
        self.t = 0.0
        self.states = self._initial_states()
        self.bandit_prop = self.bandit_params.prop_mass
        self.guard_prop = self.guard_params.prop_mass
        self.guard_state = _GuardState(phase_start=0.0)
        self.guard_pointing = None
        self.is_lady_captured = False
        self.is_bandit_captured = False
        self.done = False
        self.metrics = EpisodeMetrics()
        self._next_record = 0.0
        self._update_metrics()
        return self.get_observation(), self.get_info()

    # -------------------------------------------------------------- interface

    def get_observation(self) -> np.ndarray:
        lady, bandit, guard = self.states
        obs = np.empty(21)
        obs[0] = self.t
        obs[1] = self.bandit_params.dry_mass + self.bandit_prop
        obs[2] = self.bandit_prop
        obs[3:9] = bandit
        obs[9:15] = lady
        obs[15:21] = guard
        return obs

    def step(self, action):
        """Hold the commanded burn for its duration (burn_vec[3] seconds), like kspdg vessel_step."""
        if self.done:
            raise RuntimeError("episode is done; call reset()")
        if isinstance(action, dict):
            burn_vec = np.asarray(action["burn_vec"], dtype=float)
            vec_type = int(action.get("vec_type", 0))
            ref_frame = int(action.get("ref_frame", 0))
        else:  # legacy 4-list of body-frame throttles plus duration
            burn_vec, vec_type, ref_frame = np.asarray(action, dtype=float), 0, 0
        duration = max(0.0, float(burn_vec[3]))
        if self.two_axis:
            duration = min(duration, self.MAX_BURN_DURATION)
        t_end = min(self.t + duration, self.episode_timeout)
        while self.t < t_end - 1e-9 and not self.done:
            h = min(self.sim_dt, t_end - self.t)
            thr_body = self._throttle_body(burn_vec[:3], vec_type, ref_frame)
            self._substep(h, thr_body)
        obs = self.get_observation()
        info = self.get_info()
        reward = -info["weighted_score"] if self.done else 0.0
        return obs, reward, self.done, info

    def get_info(self) -> dict:
        m = self.metrics
        return {
            "closest_lady_bandit_approach": m.closest_lb,
            "closest_lady_bandit_approach_time": m.closest_lb_time,
            "closest_bandit_guard_approach": m.closest_bg,
            "closest_lady_guard_approach": m.closest_lg,
            "minimum_lady_bandit_distance_speed_product": m.min_lb_distspeed,
            "bandit_fuel_usage": float(self.bandit_params.prop_mass - self.bandit_prop),
            "guard_fuel_usage": float(self.guard_params.prop_mass - self.guard_prop),
            "lady_fuel_usage": 0.0,
            "weighted_score": self.weighted_score(m.closest_lb, m.closest_bg),
            "is_lady_captured": self.is_lady_captured,
            "is_bandit_captured": self.is_bandit_captured,
            "mission_elapsed_time": self.t,
        }

    @staticmethod
    def weighted_score(lb_dist: float, bg_dist: float) -> float:
        return lb_dist**2 + SCORE_BG_SCALE / (bg_dist + SCORE_BG_OFFSET)

    # ---------------------------------------------------------------- physics

    def _throttle_body(self, vec, vec_type, ref_frame) -> np.ndarray:
        bandit = self.states[1]
        body = O.body_basis(bandit[:3], bandit[3:])
        if ref_frame == 0:
            v_body = vec
        elif ref_frame == 1:
            v_body = body @ vec
        elif ref_frame == 2:
            v_body = body @ (O.ntw_basis(bandit[:3], bandit[3:]).T @ vec)
        else:
            raise ValueError(f"unknown ref_frame {ref_frame}")
        if vec_type == 0:
            thr = v_body
        elif vec_type == 1:
            p = self.bandit_params
            f_fwd = p.thrust_fwd if v_body[0] >= 0 else p.thrust_rev
            thr = np.array([v_body[0] / f_fwd, v_body[1] / p.thrust_lat, v_body[2] / p.thrust_lat])
        else:
            raise ValueError(f"unknown vec_type {vec_type}")
        thr = np.clip(thr, -1.0, 1.0)
        if self.two_axis:
            thr[2] = 0.0  # only forward/backward and right/left are commandable
        return thr

    def _bandit_force(self, thr_body: np.ndarray, h: float):
        p = self.bandit_params
        if self.bandit_prop <= 0.0:
            return np.zeros(3), 0.0
        f_body = np.array([thr_body[0] * (p.thrust_fwd if thr_body[0] >= 0 else p.thrust_rev),
                           thr_body[1] * p.thrust_lat, thr_body[2] * p.thrust_lat])
        mdot = np.sum(np.abs(f_body)) / (p.isp * O.G0)
        used = min(self.bandit_prop, mdot * h)
        scale = used / (mdot * h) if mdot > 0 else 0.0
        bandit = self.states[1]
        body = O.body_basis(bandit[:3], bandit[3:])
        return body.T @ (f_body * scale), used

    def _guard_force(self, h: float):
        if self.lady_guard_policy == "lg0" or self.guard_prop <= 0.0:
            return np.zeros(3), 0.0
        if self.lady_guard_policy == "lg7":
            return self._guard_force_lg7(h)
        g = self.guard_pursuit
        gs = self.guard_state
        _, bandit, guard = self.states
        r_rel = bandit[:3] - guard[:3]
        v_rel = bandit[3:] - guard[3:]
        spd = np.linalg.norm(v_rel)
        ang = _angle(-v_rel, r_rel)
        dt_phase = self.t - gs.phase_start

        def switch(phase):
            gs.phase, gs.phase_start = phase, self.t

        direction = None
        if gs.phase == "reorient_zero":
            if dt_phase >= g.reorient_time:
                switch("zero")
        elif gs.phase == "zero":
            if spd < g.min_speed_thresh or dt_phase > g.loop_timeout:
                switch("reorient_direct")
            else:
                direction = v_rel / spd
        elif gs.phase == "reorient_direct":
            if dt_phase >= g.reorient_time:
                switch("direct")
        elif gs.phase == "direct":
            if spd > g.max_speed_thresh or dt_phase > g.loop_timeout or (
                ang > g.ang_vel_pos_thresh and dt_phase > g.min_burn_time):
                switch("coast")
            else:
                direction = r_rel / np.linalg.norm(r_rel)
        elif gs.phase == "coast":
            if ang > g.ang_vel_pos_thresh:
                switch("reorient_zero")
        if direction is None:
            return np.zeros(3), 0.0
        thrust = g.pursuit_throttle * self.guard_params.thrust_fwd
        mdot = thrust / (self.guard_params.isp * O.G0)
        used = min(self.guard_prop, mdot * h)
        return direction * thrust * (used / (mdot * h)), used

    def _guard_force_lg7(self, h: float):
        """Proxy for the obfuscated advanced Guard: continuous velocity-matching pursuit."""
        _, bandit, guard = self.states
        r_rel = bandit[:3] - guard[:3]
        v_rel = bandit[3:] - guard[3:]
        dist = np.linalg.norm(r_rel)
        thrust = self.guard_params.thrust_fwd
        a_max = thrust / (self.guard_params.dry_mass + self.guard_prop)
        # desired closing velocity along the line of sight, braking so it can stay on the bandit
        v_close = min(12.0, np.sqrt(2 * 0.7 * a_max * dist))
        v_des_rel = -r_rel / max(dist, 1e-6) * v_close  # bandit relative to guard should shrink
        a_cmd = (v_rel - v_des_rel) / 2.0  # guard accelerates to cancel the error
        if np.linalg.norm(a_cmd) < 1e-3:
            return np.zeros(3), 0.0
        target = a_cmd / np.linalg.norm(a_cmd)
        # first-order attitude lag of about 1 s
        if self.guard_pointing is None:
            self.guard_pointing = target
        alpha = min(1.0, h / self.guard_pursuit.reorient_time)
        p = self.guard_pointing + alpha * (target - self.guard_pointing)
        self.guard_pointing = p / np.linalg.norm(p)
        throttle = min(1.0, np.linalg.norm(a_cmd) / a_max)
        if np.dot(self.guard_pointing, target) < np.cos(np.deg2rad(30)):
            throttle = 0.0  # don't burn while badly mis-pointed, as the lg1 autopilot does
        f = thrust * throttle
        mdot = f / (self.guard_params.isp * O.G0)
        if mdot == 0:
            return np.zeros(3), 0.0
        used = min(self.guard_prop, mdot * h)
        return self.guard_pointing * f * (used / (mdot * h)), used

    def _substep(self, h: float, thr_body: np.ndarray):
        f_b, used_b = self._bandit_force(thr_body, h)
        f_g, used_g = self._guard_force(h)
        m_b = self.bandit_params.dry_mass + self.bandit_prop - 0.5 * used_b
        m_g = self.guard_params.dry_mass + self.guard_prop - 0.5 * used_g
        a_thrust = np.vstack([np.zeros(3), f_b / m_b, f_g / m_g])

        def f(s):
            return np.concatenate([s[:, 3:], O.two_body_accel(s[:, :3]) + a_thrust], axis=1)

        s = self.states
        k1 = f(s)
        k2 = f(s + 0.5 * h * k1)
        k3 = f(s + 0.5 * h * k2)
        k4 = f(s + h * k3)
        self.states = s + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        self.bandit_prop -= used_b
        self.guard_prop -= used_g
        self.t += h
        self._update_metrics()

    def _update_metrics(self):
        lady, bandit, guard = self.states
        m = self.metrics
        d_lb = float(np.linalg.norm(bandit[:3] - lady[:3]))
        d_bg = float(np.linalg.norm(bandit[:3] - guard[:3]))
        d_lg = float(np.linalg.norm(lady[:3] - guard[:3]))
        s_lb = float(np.linalg.norm(bandit[3:] - lady[3:]))
        if d_lb < m.closest_lb:
            m.closest_lb, m.closest_lb_time = d_lb, self.t
        m.closest_bg = min(m.closest_bg, d_bg)
        m.closest_lg = min(m.closest_lg, d_lg)
        m.min_lb_distspeed = min(m.min_lb_distspeed, d_lb * s_lb)
        if self.t >= self._next_record - 1e-9:
            m.trajectory.append((self.t, *bandit[:3], *lady[:3], *guard[:3]))
            self._next_record += self.record_every
        if d_lb < self.capture_dist:
            self.is_lady_captured = True
        if d_bg < self.capture_dist:
            self.is_bandit_captured = True
        if self.is_lady_captured or self.is_bandit_captured or self.t >= self.episode_timeout - 1e-9:
            self.done = True


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)))
