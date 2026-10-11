"""Latency-aware episode runner for the three-layer controller.

Real KSPDG runs in wall-clock time: while an agent thinks, the spacecraft keeps flying on its
previous command. The stand-in sim only advances when stepped, so this runner charges each
layer's latency explicitly, as a discrete-event simulation in sim time:

- the executor runs every control tick (default 0.25 s) with the latest primitive and goal;
- a decider (Jev, rules, or a flat LLM) launched at time t on the state at t has its choice take
  effect at t + latency, and it cannot start another decision until then;
- the tactical layer runs the same way, asynchronously, every ``tactical_period_s``;
- the strategy layer plans once before the episode starts (its latency is recorded, not charged).

Model layers use their measured wall-clock latency. No-model layers use a configurable synthetic
latency (zero by default), which also lets us sweep latency without any API calls.
"""

from __future__ import annotations

import dataclasses
import time
import traceback
from dataclasses import dataclass, field

import numpy as np

from hsc.control.executor import Executor
from hsc.control.plans import TacticalGoal
from hsc.control.situation import situation_from_obs
from hsc.sim.lbg1_lite import LBG1LiteEnv


@dataclass
class LayerLog:
    calls: int = 0
    errors: int = 0
    latencies: list = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    last_error: str = ""

    def record(self, latency, info):
        self.calls += 1
        self.latencies.append(float(latency))
        if info is not None:
            self.input_tokens += info.input_tokens
            self.output_tokens += info.output_tokens

    def summary(self) -> dict:
        lat = np.array(self.latencies)
        stat = (lambda f: float(f(lat))) if len(lat) else (lambda f: None)
        return {
            "calls": self.calls,
            "errors": self.errors,
            "latency_p50_s": stat(lambda a: np.percentile(a, 50)),
            "latency_p95_s": stat(lambda a: np.percentile(a, 95)),
            "latency_mean_s": stat(np.mean),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "last_error": self.last_error,
        }


@dataclass
class ControllerConfig:
    name: str
    strategist: object
    tactician: object | None
    decider: object
    decider_latency_s: float = 0.0  # synthetic latency for layers that report none
    tactician_latency_s: float = 0.0
    control_period_s: float = 0.25
    min_decision_interval_s: float = 0.0


def _latency(info, synthetic):
    return info.latency_s if info is not None else synthetic


def run_episode(cfg: ControllerConfig, init_cond: str, lady_guard_policy: str, seed: int | None = None,
                verbose: bool = False) -> dict:
    env = LBG1LiteEnv(init_cond, lady_guard_policy, seed=seed)
    obs, _ = env.reset()
    sit = lambda o: situation_from_obs(o, env.bandit_params.prop_mass, env.episode_timeout)  # noqa: E731
    executor = Executor(cfg.control_period_s)
    logs = {"strategy": LayerLog(), "tactical": LayerLog(), "decider": LayerLog()}
    wall0 = time.perf_counter()

    s = sit(obs)
    label = f"lbg1_{lady_guard_policy}_{init_cond}"
    try:
        brief, info = cfg.strategist.plan(s, label)
        logs["strategy"].record(_latency(info, 0.0), info)
    except Exception as e:  # keep the episode going on the default brief
        from hsc.control.plans import MissionBrief
        brief = MissionBrief()
        logs["strategy"].errors += 1
        logs["strategy"].last_error = repr(e)

    goal = TacticalGoal.from_brief(brief)
    history = [s]
    primitive, primitive_obs_t = "COAST", 0.0
    pending_goal = None  # (apply_t, goal)
    pending_dec = None  # (apply_t, primitive, override, obs_t)
    decider_free_at = tactician_free_at = 0.0
    next_tactical = 0.0
    last_decision_launch = -np.inf
    action_ages, timeline, goals = [], [], []
    done = False

    def call(layer_log, fn, *args):
        try:
            return fn(*args), None
        except Exception as e:
            layer_log.errors += 1
            layer_log.last_error = repr(e)
            if verbose:
                traceback.print_exc()
            return None, e

    while not done:
        t = float(obs[0])
        s = sit(obs)
        history.append(s)

        def deliver():
            nonlocal pending_goal, pending_dec, goal, primitive, primitive_obs_t
            if pending_goal is not None and pending_goal[0] <= t + 1e-9:
                goal = pending_goal[1]
                goals.append({"t_applied": t, **goal.to_dict()})
                pending_goal = None
            if pending_dec is not None and pending_dec[0] <= t + 1e-9:
                _, primitive, override, primitive_obs_t = pending_dec
                if override:
                    goal = dataclasses.replace(goal, **override)
                pending_dec = None

        deliver()

        if cfg.tactician is not None and t >= tactician_free_at - 1e-9 and t >= next_tactical - 1e-9:
            out, err = call(logs["tactical"], cfg.tactician.update, history, brief, goal)
            if out is not None:
                new_goal, info = out
                lat = _latency(info, cfg.tactician_latency_s)
                logs["tactical"].record(lat, info)
                pending_goal = (t + lat, new_goal)
                tactician_free_at = t + lat
            next_tactical = t + brief.tactical_period_s

        if t >= decider_free_at - 1e-9 and t - last_decision_launch >= cfg.min_decision_interval_s - 1e-9:
            out, err = call(logs["decider"], cfg.decider.decide, s, goal, brief)
            last_decision_launch = t
            if out is not None:
                p, override, info = out
                lat = _latency(info, cfg.decider_latency_s)
                logs["decider"].record(lat, info)
                pending_dec = (t + lat, p, override, t)
                decider_free_at = t + lat
            else:
                decider_free_at = t + cfg.control_period_s

        deliver()  # zero-latency layers take effect this tick

        action = executor.act(s, obs[9:12], obs[12:15], primitive, goal, brief)
        action_ages.append(t - primitive_obs_t)
        timeline.append((round(t, 2), primitive))
        obs, _, done, info = env.step(action)

    ages = np.array(action_ages)
    switches = [timeline[0]] + [timeline[i] for i in range(1, len(timeline)) if timeline[i][1] != timeline[i - 1][1]]
    return {
        "config": cfg.name,
        "scenario": label,
        "seed": seed,
        "score": info["weighted_score"],
        "closest_lb_m": info["closest_lady_bandit_approach"],
        "closest_lb_time_s": info["closest_lady_bandit_approach_time"],
        "closest_bg_m": info["closest_bandit_guard_approach"],
        "lady_captured": info["is_lady_captured"],
        "bandit_captured": info["is_bandit_captured"],
        "bandit_fuel_kg": info["bandit_fuel_usage"],
        "episode_s": info["mission_elapsed_time"],
        "action_age_mean_s": float(ages.mean()),
        "action_age_p95_s": float(np.percentile(ages, 95)),
        "wall_s": time.perf_counter() - wall0,
        "brief": brief.to_dict(),
        "layers": {k: v.summary() for k, v in logs.items()},
        "primitive_switches": switches,
        "goals": goals,
        "trajectory": env.metrics.trajectory,
    }
