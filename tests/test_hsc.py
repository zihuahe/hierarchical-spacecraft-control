import io
import json

import numpy as np
import pytest

from hsc.control.plans import MissionBrief, TacticalGoal
from hsc.control.situation import situation_from_obs
from hsc.layers import deciders as D
from hsc.layers.llm import CallResult
from hsc.layers.strategist import FixedStrategist, OpusStrategist
from hsc.layers.tactician import HaikuTactician, RuleTactician
from hsc.runner import ControllerConfig, run_episode
from hsc.sim.lbg1_lite import LBG1LiteEnv

COAST = {"burn_vec": [0, 0, 0, 10], "vec_type": 0, "ref_frame": 0}


def test_observation_layout_and_score():
    env = LBG1LiteEnv("i2", "lg0")
    obs, info = env.reset()
    assert obs.shape == (21,)
    assert obs[0] == 0 and obs[2] == env.bandit_params.prop_mass
    assert np.linalg.norm(obs[3:6] - obs[9:12]) == pytest.approx(2600, rel=1e-3)  # bandit 2600 m behind lady
    assert np.linalg.norm(obs[15:18] - obs[9:12]) == pytest.approx(600, rel=1e-3)
    assert LBG1LiteEnv.weighted_score(10.0, 100.0) == pytest.approx(100 + 1e6 / 100.1)


def test_coast_keeps_circular_formation_and_times_out():
    env = LBG1LiteEnv("i2", "lg0")
    done = False
    while not done:
        obs, _, done, info = env.step(COAST)
    assert info["mission_elapsed_time"] == pytest.approx(240.0)
    assert info["bandit_fuel_usage"] == 0.0
    assert np.linalg.norm(obs[3:6] - obs[9:12]) == pytest.approx(2600, rel=1e-3)


def test_thrust_uses_fuel_and_saturates():
    env = LBG1LiteEnv("i2", "lg0")
    env.step({"burn_vec": [1e6, 0, 0, 1.0], "vec_type": 1, "ref_frame": 0})  # saturates to full forward
    used = env.bandit_params.prop_mass - env.bandit_prop
    assert used == pytest.approx(8000 / (240 * 9.80665), rel=1e-6)


def test_i1_conjunction_without_control():
    env = LBG1LiteEnv("i1", "lg0")
    done = False
    while not done:
        _, _, done, info = env.step(COAST)
    assert info["closest_lady_bandit_approach"] == pytest.approx(500, abs=5)
    assert info["closest_lady_bandit_approach_time"] == pytest.approx(120, abs=1)


def test_rule_controller_catches_lady():
    r = run_episode(ControllerConfig("fast_only", FixedStrategist(), None, D.RuleDecider()), "i2", "lg1")
    assert r["lady_captured"] and not r["bandit_captured"]


class _FakeLLM:
    def __init__(self, data, latency):
        self.data, self.latency, self.calls = data, latency, []

    def call(self, system, content, schema):
        self.calls.append((system, content, schema))
        return CallResult(data=self.data, latency_s=self.latency, model="fake", input_tokens=10, output_tokens=5)


def test_latency_is_charged_in_sim_time():
    llm = _FakeLLM({"primitive": "BRAKE", "waypoint_lvlh_m": None}, latency=3.0)
    cfg = ControllerConfig("flat_fake", FixedStrategist(), None, D.FlatLLMDecider("fake", llm=llm))
    r = run_episode(cfg, "i2", "lg0")
    assert r["primitive_switches"][0] == (0.0, "COAST")
    assert r["primitive_switches"][1] == (3.0, "BRAKE")  # first decision lands 3 s after it was asked
    assert r["layers"]["decider"]["calls"] == pytest.approx(240 / 3, abs=1)
    assert r["action_age_mean_s"] > 2.0


def test_hierarchy_with_fake_models():
    brief = {"objective": "o", "approach": "a", "guard_keepout_m": 200, "max_closing_speed_mps": 10,
             "final_approach_speed_mps": 1, "fuel_reserve_fraction": 0.05, "tactical_period_s": 5, "phases": []}
    goal = {"mode_hint": "AUTO", "waypoint_lvlh_m": [0, -1000, 400], "max_closing_speed_mps": 10,
            "guard_keepout_m": 200, "standoff_m": 300, "valid_for_s": 20, "rationale": "bypass"}
    strat = OpusStrategist(llm=_FakeLLM(brief, 8.0))
    tac = HaikuTactician(llm=_FakeLLM(goal, 1.5), use_images=True)
    r = run_episode(ControllerConfig("hier_fake", strat, tac, D.RuleDecider()), "i2", "lg1")
    assert r["brief"]["guard_keepout_m"] == 200
    assert r["goals"][0]["t_applied"] == pytest.approx(1.5)
    assert r["goals"][0]["waypoint_lvlh_m"] == [0, -1000, 400]
    assert r["layers"]["tactical"]["calls"] >= 10
    assert any(p == "GO_TO_WAYPOINT" for _, p in r["primitive_switches"])


def test_rule_tactician_bypasses_blocking_guard():
    env = LBG1LiteEnv("i2", "lg0")
    obs, _ = env.reset()
    s = situation_from_obs(obs, env.bandit_params.prop_mass, env.episode_timeout)
    assert s.guard_between
    g, _ = RuleTactician().update([s], MissionBrief(), TacticalGoal())
    assert g.mode_hint == "GO_TO_WAYPOINT" and abs(g.waypoint_lvlh_m[2]) >= 300


def test_jev_request_shape(monkeypatch):
    sent = {}

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout):
        sent["url"] = req.full_url
        sent["auth"] = req.get_header("Authorization")
        sent["body"] = json.loads(req.data)
        payload = {"model": "jev-1.13.0", "answers": {"maneuver": {"type": "choice", "choice": "EVADE_GUARD",
                   "probabilities": {"EVADE_GUARD": 0.8, "BRAKE": 0.2}, "confidence": 0.9}},
                   "usage": {"input_tokens": 300, "output_tokens": 1}}
        return _Resp(json.dumps(payload).encode())

    monkeypatch.setattr(D.urllib.request, "urlopen", fake_urlopen)
    env = LBG1LiteEnv("i2", "lg0")
    obs, _ = env.reset()
    s = situation_from_obs(obs, env.bandit_params.prop_mass, env.episode_timeout)
    choice, override, info = D.JevDecider(api_key="k").decide(s, TacticalGoal(), MissionBrief())
    assert choice == "EVADE_GUARD" and override is None
    assert sent["url"] == "https://api.typesafe.ai/v1/systemone"
    assert sent["auth"] == "Bearer k"
    q = sent["body"]["questions"]["maneuver"]
    assert q["type"] == "choice" and "EVADE_GUARD" in q["criteria"]
    assert info.model == "jev-1.13.0" and info.extra["confidence"] == 0.9
