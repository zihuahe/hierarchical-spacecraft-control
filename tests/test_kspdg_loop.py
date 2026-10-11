"""Checks for the real-loop adapter (hsc/kspdg_loop.py). No network: Kev is a local stub server, and the
environment is either a scripted fake or the labelled lbg1_lite test fixture."""

import http.server
import json
import math
import threading
import time

import pytest

from hsc import kspdg_loop as K

OBS = [12.0, 5000.0, 400.0,            # met, bandit mass, bandit prop
       0, 0, 0, 1, 0, 0,               # bandit pos, vel
       100, 0, 0, 0, 0, 0,             # lady pos, vel
       0, 50, 0, 0, -2, 0]             # guard pos, vel


class _FakeEnv:
    """Scripted environment shaped like KSPDG 0.15's LBG1 envs: 4-tuple step, capture flags and rules as attributes.
    ``five_tuple`` switches to the Gymnasium step format."""

    episode_timeout, lady_capture_dist, bandit_capture_dist, loadfile = 240.0, 5.0, 10.0, "lbg1_i2_init"

    def __init__(self, five_tuple=False, done_after=None, info=None, fail_step=False, reset_tuple=True, on_done=None):
        self.five_tuple, self.done_after, self.info = five_tuple, done_after, info or {}
        self.fail_step, self.reset_tuple, self.on_done = fail_step, reset_tuple, on_done or {}
        self.is_lady_captured = self.is_bandit_captured = False
        self.cmds, self.closed = [], False

    def reset(self):
        return (list(OBS), {}) if self.reset_tuple else list(OBS)

    def step(self, cmd):
        if self.fail_step:
            raise ConnectionResetError("krpc gone")
        self.cmds.append(cmd)
        obs = list(OBS)
        obs[0] += cmd["burn_vec"][3] * len(self.cmds)
        done = self.done_after is not None and len(self.cmds) >= self.done_after
        if done:
            for k, v in self.on_done.items():
                setattr(self, k, v) if k.startswith("is_") else obs.__setitem__(int(k), v)
        if self.five_tuple:
            return obs, 0.0, done, False, self.info
        return obs, 0.0, done, self.info

    def close(self):
        self.closed = True


def _ok(choice="POS_X", **extra):
    return lambda req: ({"choice": choice, "confidence": 0.7, "probabilities": {choice: 0.7}, "model": "stub"}, 0.01)


def _records(path, kind=None):
    recs = [json.loads(l) for l in open(path)]
    return [r for r in recs if r["record"] == kind] if kind else recs


# ---------------------------------------------------------------- parsing and features

def test_parse_observation_accepts_21_and_rejects_bad():
    assert K.parse_observation(OBS) == [float(x) for x in OBS]
    for bad in (OBS[:20], OBS + [0], OBS[:5] + [math.nan] + OBS[6:], OBS[:3] + [math.inf] + OBS[4:],
                [OBS[0], 0.0] + OBS[2:], None, ["x"] * 21):
        with pytest.raises(K.LoopError) as e:
            K.parse_observation(bad)
        assert e.value.reason == "invalid_observation"


def test_relative_vectors_and_closing_speed():
    r = K.relative([0, 0, 0], [1, 0, 0], [100, 0, 0], [0, 0, 0])
    assert r["rel_pos_m"] == [100, 0, 0] and r["rel_vel_mps"] == [-1, 0, 0]
    assert r["distance_m"] == 100 and r["closing_speed_mps"] == 1.0          # approaching at 1 m/s
    assert K.relative([0, 0, 0], [0, 0, 0], [0, 30, 40], [0, 3, 4])["closing_speed_mps"] == -5.0  # receding
    z = K.relative([5, 5, 5], [1, 2, 3], [5, 5, 5], [0, 0, 0])               # zero distance: no division
    assert z["distance_m"] == 0 and z["closing_speed_mps"] == 0.0


def test_features_from_observation():
    f = K.features(K.parse_observation(OBS), [], 240.0)
    assert f["time_remaining_s"] == 228.0 and f["lady"]["distance_m"] == 100
    assert f["guard"]["distance_m"] == 50 and f["guard"]["closing_speed_mps"] == pytest.approx(2.0)
    assert f["fuel_used_raw"] == 0 and f["mass_used_kg"] == 0 and f["recent"] == []
    later = list(OBS)
    later[0], later[1], later[2] = 300.0, 4990.0, 397.5
    g = K.features(later, [{**f, "action": "POS_X"}], 240.0, initial=OBS)
    assert g["time_remaining_s"] == 0.0                              # clamped at zero
    assert g["fuel_used_raw"] == 2.5 and g["mass_used_kg"] == 10.0   # raw resource units vs kg, since episode start
    assert g["recent"] == [{"t_s": 12.0, "lady_distance_m": 100, "guard_distance_m": 50, "action": "POS_X"}]


def test_mission_text_uses_effective_rules():
    s = K.env_settings(_FakeEnv(), K.LoopConfig())
    assert s == {"env_class": "_FakeEnv", "loadfile": "lbg1_i2_init", "episode_timeout_s": 240.0,
                 "episode_timeout_source": "env", "lady_capture_dist_m": 5.0, "bandit_capture_dist_m": 10.0}
    text = K.mission_text(s)
    assert "within 5 m of the Lady" in text and "within 10 m of the Bandit" in text and "240 s" in text
    assert "capture distance" in K.mission_text({}) and "time limit" in K.mission_text({})


# ---------------------------------------------------------------- action catalog and validation

def test_all_seven_mappings_and_limits():
    cfg = K.LoopConfig()
    expected = {"COAST": [0, 0, 0], "POS_X": [.2, 0, 0], "NEG_X": [-.2, 0, 0], "POS_Y": [0, .2, 0],
                "NEG_Y": [0, -.2, 0], "POS_Z": [0, 0, .2], "NEG_Z": [0, 0, -.2]}
    assert set(expected) == set(K.ACTION_IDS)
    for a, vec in expected.items():
        cmd = K.command_for(a, cfg)
        assert cmd == {"burn_vec": vec + [1.0], "ref_frame": 1, "vec_type": 0}
    with pytest.raises(K.LoopError):
        K.command_for("forward", cfg)
    for bad in ({"burn_vec": [1.5, 0, 0, 1], "ref_frame": 1, "vec_type": 0},
                {"burn_vec": [0, 0, 0, 11], "ref_frame": 1, "vec_type": 0},
                {"burn_vec": [0, 0, 0, -1], "ref_frame": 1, "vec_type": 0},
                {"burn_vec": [math.nan, 0, 0, 1], "ref_frame": 1, "vec_type": 0},
                {"burn_vec": [0, 0, 1], "ref_frame": 1, "vec_type": 0},
                {"burn_vec": [0, 0, 0, 1], "ref_frame": 0, "vec_type": 0},
                {"burn_vec": [0, 0, 0, 1], "ref_frame": 1, "vec_type": 1}):
        with pytest.raises(K.LoopError) as e:
            K.validate_command(bad)
        assert e.value.reason == "invalid_command"


def test_config_validation():
    for kw in ({"thrust_magnitude": 0}, {"thrust_magnitude": 1.5}, {"burn_duration_s": 12},
               {"burn_duration_s": 0}, {"request_timeout_s": math.nan}, {"history_len": 0}):
        with pytest.raises(K.LoopError) as e:
            K.LoopConfig(**kw).validate()
        assert e.value.reason == "invalid_config"


def test_request_shape():
    req = K.build_request({"a": 1}, K.LoopConfig(), {"version": "brief-1"}, "Mission: test")
    assert req["model"] == "kev-latest"
    q = req["questions"]["action"]
    assert q["type"] == "choice" and q["instructions"] == K.INSTRUCTIONS
    assert list(q["criteria"]) == list(K.ACTION_IDS) and q["criteria"]["COAST"] == "No thrust for this interval."
    state = json.loads(req["state"])
    assert state["features"] == {"a": 1} and state["context"]["version"] == "brief-1"
    assert state["mission"] == "Mission: test"


@pytest.mark.parametrize("resp", [
    {}, {"answers": {}}, {"answers": {"action": {}}}, None, [],
    {"answers": {"action": {"choice": "forward"}}},                                  # unknown ID
    {"answers": {"action": {"choice": "POS_X", "confidence": float("nan")}}},
    {"answers": {"action": {"choice": "POS_X", "probabilities": {"POS_X": float("inf")}}}},
    {"answers": {"action": {"choice": "POS_X", "confidence": "high"}}},
])
def test_bad_responses_rejected(resp):
    with pytest.raises(K.LoopError) as e:
        K.parse_response(resp)
    assert e.value.reason == "invalid_response"


# ---------------------------------------------------------------- Kev over HTTP (stub server)

def _stub(handler_body, delay=0.0, status=200):
    seen = []

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            time.sleep(delay)
            data = handler_body if isinstance(handler_body, bytes) else json.dumps(handler_body).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except OSError:
                pass

        def do_GET(self):
            data = json.dumps({"models": [{"name": "kev-latest", "run": "stub"}]}).encode()  # kev.serve shape
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, seen


VALID = {"model": "kev-latest", "answers": {"action": {"type": "choice", "choice": "NEG_Y", "confidence": 0.6,
                                                       "probabilities": {"NEG_Y": 0.6, "COAST": 0.4}}}}


def test_kev_http_roundtrip_one_action(tmp_path):
    srv, seen = _stub(VALID)
    try:
        cfg = K.LoopConfig(kev_base_url=f"http://127.0.0.1:{srv.server_port}")
        env = _FakeEnv()
        out = K.run(env, cfg, 1, choose=None, trace_path=tmp_path / "t.jsonl")
    finally:
        srv.shutdown()
    assert out["terminal_reason"] == "max_steps_reached" and out["steps"] == 1 and env.closed
    assert len(seen) == 2                                   # one warm-up request, then the real decision
    path, body = seen[1]
    assert path == "/v1/systemone" and set(body["questions"]["action"]["criteria"]) == set(K.ACTION_IDS)
    assert "within 10 m of the Bandit" in json.loads(body["state"])["mission"]
    assert env.cmds == [{"burn_vec": [0.0, -0.2, 0.0, 1.0], "ref_frame": 1, "vec_type": 0}]
    header, warm, obs, dec, end = _records(tmp_path / "t.jsonl")
    assert header["record"] == "run_header" and header["config"]["scenario"] == "LBG1_LG7_I2_V1"
    assert header["scenario_settings"]["bandit_capture_dist_m"] == 10.0
    assert header["versions"]["kev_models"]["models"][0]["name"] == "kev-latest"
    assert warm["record"] == "warmup" and obs["record"] == "observation" and obs["observation"] == OBS
    assert dec["record"] == "decision"
    for k in ("run_id", "step", "wall_time", "mission_time_s", "observation", "features", "model", "choice",
              "confidence", "burn_command", "next_observation", "decision_latency_s", "model_latency_s"):
        assert k in dec, k
    assert dec["choice"] == "NEG_Y" and dec["next_mission_time_s"] == 13.0 and dec["mission_dt_s"] == 1.0
    assert end["record"] == "run_end" and end["terminal_reason"] == "max_steps_reached"


@pytest.mark.parametrize("body,reason", [
    (b"not json", "invalid_response"),
    ({"answers": {"action": {"choice": "UP"}}}, "invalid_response"),
    ({"answers": {"action": {"choice": "POS_X", "confidence": None, "probabilities": {"POS_X": None}}}},
     "invalid_response"),
])
def test_bad_http_response_stops_without_burn(tmp_path, body, reason):
    srv, _ = _stub(body)
    try:
        env = _FakeEnv()
        out = K.run(env, K.LoopConfig(kev_base_url=f"http://127.0.0.1:{srv.server_port}"), 3,
                    trace_path=tmp_path / "t.jsonl")
    finally:
        srv.shutdown()
    assert out["terminal_reason"] == reason and env.cmds == [] and env.closed
    assert _records(tmp_path / "t.jsonl", "run_end")[0]["terminal_reason"] == reason


def test_http_error_is_reported(tmp_path):
    srv, _ = _stub({"detail": "state too long"}, status=422)
    try:
        env = _FakeEnv()
        out = K.run(env, K.LoopConfig(kev_base_url=f"http://127.0.0.1:{srv.server_port}"), 1,
                    trace_path=tmp_path / "t.jsonl")
    finally:
        srv.shutdown()
    assert out["terminal_reason"] == "model_http_error" and "422" in out["error"] and env.cmds == []


def test_timeout_stops_without_burn(tmp_path):
    srv, _ = _stub(VALID, delay=1.5)
    try:
        env = _FakeEnv()
        cfg = K.LoopConfig(kev_base_url=f"http://127.0.0.1:{srv.server_port}", request_timeout_s=0.3,
                           warmup_requests=0)
        out = K.run(env, cfg, 1, trace_path=tmp_path / "t.jsonl")
    finally:
        srv.shutdown()
    assert out["terminal_reason"] == "model_timeout" and env.cmds == [] and env.closed


def test_connection_refused_stops(tmp_path):
    env = _FakeEnv()
    out = K.run(env, K.LoopConfig(kev_base_url="http://127.0.0.1:9", request_timeout_s=1), 1,
                trace_path=tmp_path / "t.jsonl")
    assert out["terminal_reason"] == "model_connection_error" and env.cmds == [] and env.closed


# ---------------------------------------------------------------- loop behaviour

def test_stale_observation_stops_without_burn(tmp_path):
    def slow(req):
        time.sleep(0.2)
        return _ok()(req)

    env = _FakeEnv()
    out = K.run(env, K.LoopConfig(max_observation_age_s=0.05, kev_base_url="http://127.0.0.1:9", warmup_requests=0),
                1, choose=slow, trace_path=tmp_path / "t.jsonl")
    assert out["terminal_reason"] == "stale_decision" and env.cmds == [] and env.closed


@pytest.mark.parametrize("five", [False, True])
def test_episode_mode_both_step_formats(tmp_path, five):
    env = _FakeEnv(five_tuple=five, done_after=3, on_done={"is_lady_captured": True}, reset_tuple=not five)
    out = K.run(env, K.LoopConfig(kev_base_url="http://127.0.0.1:9"), None, choose=_ok("COAST"),
                trace_path=tmp_path / "t.jsonl")
    assert out["steps"] == 3 and out["terminal_reason"] == "capture" and env.closed
    decs = _records(tmp_path / "t.jsonl", "decision")
    assert [r["step"] for r in decs] == [0, 1, 2]
    assert decs[0]["features"]["recent"] == []
    assert [h["action"] for h in decs[2]["features"]["recent"]] == ["COAST", "COAST"]
    assert len(env.cmds) == 3                            # warm-up answers are never executed


@pytest.mark.parametrize("on_done,info,reason", [
    ({"is_bandit_captured": True}, {}, "interception"),            # KSPDG: flag on the env, not in info
    ({"is_lady_captured": True, "is_bandit_captured": True}, {}, "capture_and_interception"),
    ({"0": 240.5}, {}, "timeout"),                                  # mission time past the effective timeout
    ({}, {"is_bandit_captured": True}, "interception"),             # fixture: flag in info
    ({}, {}, "simulator_done"),
])
def test_terminal_reason(tmp_path, on_done, info, reason):
    env = _FakeEnv(done_after=1, on_done=on_done, info=info)
    out = K.run(env, K.LoopConfig(warmup_requests=0), None, choose=_ok("COAST"), trace_path=tmp_path / "t.jsonl")
    assert out["terminal_reason"] == reason and out["steps"] == 1


def test_observation_only_mode(tmp_path):
    calls = []
    env = _FakeEnv()
    out = K.run(env, K.LoopConfig(), 0, choose=lambda r: calls.append(r), trace_path=tmp_path / "t.jsonl")
    assert out["terminal_reason"] == "observation_only" and calls == [] and env.cmds == [] and env.closed
    assert [r["record"] for r in _records(tmp_path / "t.jsonl")] == ["run_header", "observation", "run_end"]


def test_simulator_error_and_cleanup(tmp_path):
    env = _FakeEnv(fail_step=True)
    out = K.run(env, K.LoopConfig(kev_base_url="http://127.0.0.1:9"), 1, choose=_ok(),
                trace_path=tmp_path / "t.jsonl")
    assert out["terminal_reason"] == "simulator_error" and env.closed
    assert _records(tmp_path / "t.jsonl")[-1]["record"] == "run_end"


class _FakeKrpc:
    def connect(self, name=None, address="127.0.0.1", rpc_port=50000, stream_port=50001):
        return (name, address, rpc_port, stream_port)


def _fake_kspdg(monkeypatch, scenario):
    import types

    mod = types.ModuleType("kspdg")
    mod.__version__ = "0.15.0-fake"
    mod.LBG1_LG7_I2_V1 = scenario
    monkeypatch.setitem(__import__("sys").modules, "kspdg", mod)
    krpc = types.ModuleType("krpc")
    krpc.connect = _FakeKrpc().connect
    monkeypatch.setitem(__import__("sys").modules, "krpc", krpc)
    return krpc


def test_kspdg_juliacall_placeholder_is_reported(monkeypatch):
    # kspdg 0.15 binds LBG1_LG7_* to a lambda returning this message when juliacall is not installed
    _fake_kspdg(monkeypatch, lambda *a, **k: "Unmet dependency juliacall for using LBG1_LG7 environments.\n")
    with pytest.raises(K.LoopError) as e:
        K.make_env("kspdg", K.LoopConfig())
    assert e.value.reason == "env_unavailable" and "juliacall" in e.value.detail


def test_kspdg_env_gets_timeout_and_krpc_address(monkeypatch):
    class LG7Like(_FakeEnv):  # LG3+ envs take **kwargs only
        def __init__(self, **kwargs):
            super().__init__()
            self.episode_timeout = kwargs.get("episode_timeout", 240.0)

    krpc = _fake_kspdg(monkeypatch, LG7Like)
    cfg = K.LoopConfig(episode_timeout_s=120.0, krpc_address="172.17.0.2", krpc_rpc_port=51000)
    env = K.make_env("kspdg", cfg, check_krpc=False)
    assert env.episode_timeout == 120.0 and K.env_settings(env, cfg)["episode_timeout_s"] == 120.0
    assert krpc.connect(name="kspdg_env") == ("kspdg_env", "172.17.0.2", 51000, 50001)


def test_kspdg_timeout_mismatch_and_unreachable_krpc(monkeypatch):
    class Fixed(_FakeEnv):
        def __init__(self, **kwargs):
            super().__init__()                         # ignores episode_timeout: stays 240

    _fake_kspdg(monkeypatch, Fixed)
    with pytest.raises(K.LoopError) as e:
        K.make_env("kspdg", K.LoopConfig(episode_timeout_s=60.0), check_krpc=False)
    assert e.value.reason == "invalid_config"
    with pytest.raises(K.LoopError) as e:
        K.make_env("kspdg", K.LoopConfig(krpc_rpc_port=9))
    assert e.value.reason == "env_unavailable" and "kRPC server not reachable" in e.value.detail


def test_kspdg_missing_is_reported():
    try:
        import kspdg  # noqa: F401
        pytest.skip("kspdg installed")
    except ImportError:
        pass
    with pytest.raises(K.LoopError) as e:
        K.make_env("kspdg", K.LoopConfig())
    assert e.value.reason == "env_unavailable"


def test_fixture_episode_runs_to_terminal(tmp_path):
    """lbg1_lite TEST FIXTURE (not the real game): the full loop terminates on a simulator outcome."""
    cfg = K.LoopConfig(kev_base_url="http://127.0.0.1:9")
    out = K.run(K.make_env("fixture", cfg), cfg, None, env_kind="fixture", choose=_ok("POS_X"),
                trace_path=tmp_path / "t.jsonl")
    assert out["terminal_reason"] in K.SIMULATOR_OUTCOMES
    assert out["steps"] > 1


def test_kev_check_against_stub(tmp_path, capsys):
    from hsc import kev_check

    for i in range(5):
        assert len(K.parse_observation(kev_check.synthetic_observation(i))) == K.OBS_LEN
    srv, seen = _stub(VALID)
    try:
        kev_check.main(["--kev-base-url", f"http://127.0.0.1:{srv.server_port}", "--requests", "3", "--warmup", "1",
                        "--out", str(tmp_path / "k.json")])
    finally:
        srv.shutdown()
    rep = json.loads((tmp_path / "k.json").read_text())
    assert len(seen) == 4 and rep["timed_requests"] == 3 and rep["choices"]["NEG_Y"] == 3
    assert rep["calls"][1]["burn_command"] == {"burn_vec": [0.0, -0.2, 0.0, 1.0], "ref_frame": 1, "vec_type": 0}
    assert "within 10 m of the Bandit" in json.loads(seen[0][1]["state"])["mission"]


def _feat(lady_pos, lady_vel=(0, 0, 0), guard_pos=(5000, 0, 0)):
    return K.features([0, 4000, 400, 0, 0, 0, 0, 0, 0, *lady_pos, *lady_vel, *guard_pos, 0, 0, 0], [], 240.0)


def test_rule_action_steers_toward_lady_and_away_from_guard():
    assert K.rule_action(_feat((1000, 0, 0))) == "POS_X"
    assert K.rule_action(_feat((0, -1000, 0))) == "NEG_Y"
    # already closing at the desired speed (Lady's velocity relative to the Bandit is -8 m/s along X)
    assert K.rule_action(_feat((1000, 0, 0), lady_vel=(-8, 0, 0))) == "COAST"
    # Guard 50 m away along +Z: the push away from it dominates
    assert K.rule_action(_feat((0, 0, 0.0), guard_pos=(0, 0, 50))) == "NEG_Z"


def test_rule_decider_runs_without_kev(tmp_path):
    """lbg1_lite TEST FIXTURE: decider=rule needs no model server, records no warm-up and names itself."""
    cfg = K.LoopConfig(decider="rule", kev_base_url="http://127.0.0.1:9")
    out = K.run(K.make_env("fixture", cfg), cfg, 5, env_kind="fixture", trace_path=tmp_path / "t.jsonl")
    assert out["terminal_reason"] == "max_steps_reached" and out["steps"] == 5
    assert not _records(tmp_path / "t.jsonl", "warmup")
    decisions = _records(tmp_path / "t.jsonl", "decision")
    assert {d["model"] for d in decisions} == {K.RULE_MODEL}
    assert all(d["choice"] in K.ACTION_IDS for d in decisions)


def test_unknown_decider_is_rejected():
    with pytest.raises(K.LoopError) as e:
        K.LoopConfig(decider="jev").validate()
    assert e.value.reason == "invalid_config"
