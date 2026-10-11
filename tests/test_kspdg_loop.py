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
    """Scripted environment. ``five_tuple`` switches to the Gymnasium step format."""

    def __init__(self, five_tuple=False, done_after=None, info=None, fail_step=False, reset_tuple=True):
        self.five_tuple, self.done_after, self.info = five_tuple, done_after, info or {}
        self.fail_step, self.reset_tuple = fail_step, reset_tuple
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
        if self.five_tuple:
            return obs, 0.0, done, False, self.info
        return obs, 0.0, done, self.info

    def close(self):
        self.closed = True


def _ok(choice="POS_X", **extra):
    return lambda req: ({"choice": choice, "confidence": 0.7, "probabilities": {choice: 0.7}, "model": "stub"}, 0.01)


def _records(path):
    return [json.loads(l) for l in open(path)]


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
    req = K.build_request({"a": 1}, K.LoopConfig(), {"version": "brief-1"})
    assert req["model"] == "kev-latest"
    q = req["questions"]["action"]
    assert q["type"] == "choice" and q["instructions"] == K.INSTRUCTIONS
    assert list(q["criteria"]) == list(K.ACTION_IDS) and q["criteria"]["COAST"] == "No thrust for this interval."
    state = json.loads(req["state"])
    assert state["features"] == {"a": 1} and state["context"]["version"] == "brief-1"


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

def _stub(handler_body, delay=0.0):
    seen = []

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            time.sleep(delay)
            data = handler_body if isinstance(handler_body, bytes) else json.dumps(handler_body).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except OSError:
                pass

        def do_GET(self):
            data = json.dumps({"data": [{"id": "kev-latest", "revision": "stub"}]}).encode()
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
    path, body = seen[0]
    assert path == "/v1/systemone" and set(body["questions"]["action"]["criteria"]) == set(K.ACTION_IDS)
    assert env.cmds == [{"burn_vec": [0.0, -0.2, 0.0, 1.0], "ref_frame": 1, "vec_type": 0}]
    header, dec, end = _records(tmp_path / "t.jsonl")
    assert header["record"] == "run_header" and header["config"]["scenario"] == "LBG1_LG7_I2_V1"
    assert header["versions"]["kev_models"]["data"][0]["id"] == "kev-latest"
    for k in ("run_id", "step", "wall_time", "mission_time_s", "observation", "features", "model", "choice",
              "confidence", "burn_command", "next_observation", "decision_latency_s", "model_latency_s"):
        assert k in dec, k
    assert dec["choice"] == "NEG_Y" and dec["next_mission_time_s"] == 13.0
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
    assert _records(tmp_path / "t.jsonl")[1]["terminal_reason"] == reason


def test_timeout_stops_without_burn(tmp_path):
    srv, _ = _stub(VALID, delay=1.5)
    try:
        env = _FakeEnv()
        cfg = K.LoopConfig(kev_base_url=f"http://127.0.0.1:{srv.server_port}", request_timeout_s=0.3)
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
    out = K.run(env, K.LoopConfig(max_observation_age_s=0.05, kev_base_url="http://127.0.0.1:9"), 1,
                choose=slow, trace_path=tmp_path / "t.jsonl")
    assert out["terminal_reason"] == "stale_decision" and env.cmds == [] and env.closed


@pytest.mark.parametrize("five", [False, True])
def test_episode_mode_both_step_formats(tmp_path, five):
    env = _FakeEnv(five_tuple=five, done_after=3, info={"is_lady_captured": True}, reset_tuple=not five)
    out = K.run(env, K.LoopConfig(kev_base_url="http://127.0.0.1:9"), None, choose=_ok("COAST"),
                trace_path=tmp_path / "t.jsonl")
    assert out["steps"] == 3 and out["terminal_reason"] == "capture" and env.closed
    recs = _records(tmp_path / "t.jsonl")
    assert [r["step"] for r in recs if r["record"] == "decision"] == [0, 1, 2]
    assert recs[2]["features"]["recent"] == []          # history excludes the oldest entry
    assert recs[3]["features"]["recent"][-1]["action"] == "COAST"


def test_simulator_error_and_cleanup(tmp_path):
    env = _FakeEnv(fail_step=True)
    out = K.run(env, K.LoopConfig(kev_base_url="http://127.0.0.1:9"), 1, choose=_ok(),
                trace_path=tmp_path / "t.jsonl")
    assert out["terminal_reason"] == "simulator_error" and env.closed
    assert _records(tmp_path / "t.jsonl")[-1]["record"] == "run_end"


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
    assert out["terminal_reason"] in ("capture", "interception", "simulator_done")
    assert out["steps"] > 1
