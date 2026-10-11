"""Smallest real control loop: KSPDG observation -> local Kev-0.8B choice -> validated burn -> next observation.

Execution target is the real KSPDG scenario (default LBG1_LG7_I2_V1). The pure-Python ``lbg1_lite``
environment is available only as a clearly labelled test fixture (``--env fixture``).

    # one-action connection test against the real game (KSPDG + kRPC running, Kev server up)
    python -m hsc.kspdg_loop --env kspdg --steps 1
    # full episode until the simulator terminates it
    python -m hsc.kspdg_loop --env kspdg --episode

Action catalog: seven inertial-axis (right-handed CBCI) throttle actions, sent with ref_frame=1 and
vec_type=0. Kev selects only the action ID; this adapter supplies and validates the command.
Any invalid observation, invalid response, timeout, stale decision or connection error stops the run
with a diagnostic. No fallback action is executed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import platform
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass

OBS_LEN = 21
ACTION_IDS = ("COAST", "POS_X", "NEG_X", "POS_Y", "NEG_Y", "POS_Z", "NEG_Z")
ACTION_CRITERIA = {
    "COAST": "No thrust for this interval.",
    "POS_X": "Positive inertial X throttle.",
    "NEG_X": "Negative inertial X throttle.",
    "POS_Y": "Positive inertial Y throttle.",
    "NEG_Y": "Negative inertial Y throttle.",
    "POS_Z": "Positive inertial Z throttle.",
    "NEG_Z": "Negative inertial Z throttle.",
}
INSTRUCTIONS = ("Choose the next short burn to approach Lady while avoiding Guard. "
                "Use the stated inertial coordinate frame.")
REF_FRAME_CBCI = 1
VEC_TYPE_THROTTLE = 0
MAX_DURATION_S = 10.0
SIMULATOR_OUTCOMES = ("capture", "interception", "capture_and_interception", "timeout", "simulator_done")


class LoopError(RuntimeError):
    """Stops the run; ``reason`` goes into the trace as the terminal/error reason."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass
class LoopConfig:
    scenario: str = "LBG1_LG7_I2_V1"
    episode_timeout_s: float = 240.0
    thrust_magnitude: float = 0.2
    burn_duration_s: float = 1.0
    kev_base_url: str = "http://127.0.0.1:8009"
    kev_model: str = "kev-latest"
    request_timeout_s: float = 10.0
    max_observation_age_s: float = 15.0
    history_len: int = 5
    warmup_requests: int = 1
    krpc_address: str = "127.0.0.1"
    krpc_rpc_port: int = 50000
    krpc_stream_port: int = 50001
    trace_dir: str = "results/traces"

    def validate(self):
        if self.warmup_requests < 0:
            raise LoopError("invalid_config", "warmup_requests must be >= 0")
        for name in ("episode_timeout_s", "burn_duration_s", "request_timeout_s", "max_observation_age_s"):
            v = getattr(self, name)
            if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
                raise LoopError("invalid_config", f"{name}={v!r} must be positive and finite")
        if not (0.0 < self.thrust_magnitude <= 1.0):
            raise LoopError("invalid_config", f"thrust_magnitude={self.thrust_magnitude} must be in (0, 1]")
        if self.burn_duration_s > MAX_DURATION_S:
            raise LoopError("invalid_config", f"burn_duration_s={self.burn_duration_s} exceeds {MAX_DURATION_S}")
        if self.history_len < 1:
            raise LoopError("invalid_config", "history_len must be >= 1")


# ---------------------------------------------------------------- observation and features

def parse_observation(obs) -> list[float]:
    try:
        vals = [float(x) for x in obs]
    except (TypeError, ValueError) as e:
        raise LoopError("invalid_observation", repr(e))
    if len(vals) != OBS_LEN:
        raise LoopError("invalid_observation", f"expected {OBS_LEN} values, got {len(vals)}")
    if not all(math.isfinite(v) for v in vals):
        raise LoopError("invalid_observation", "non-finite value")
    if vals[1] <= 0:
        raise LoopError("invalid_observation", f"bandit mass {vals[1]} must be positive")
    return vals


def _sub(a, b):
    return [x - y for x, y in zip(a, b)]


def _norm(a):
    return math.sqrt(sum(x * x for x in a))


def relative(bandit_pos, bandit_vel, target_pos, target_vel) -> dict:
    """Target relative to the Bandit. Closing speed > 0 means approaching; 0 at zero distance."""
    rp = _sub(target_pos, bandit_pos)
    rv = _sub(target_vel, bandit_vel)
    d = _norm(rp)
    closing = -sum(p * v for p, v in zip(rp, rv)) / d if d > 1e-9 else 0.0
    return {"rel_pos_m": [round(x, 2) for x in rp], "rel_vel_mps": [round(x, 3) for x in rv],
            "distance_m": round(d, 2), "closing_speed_mps": round(closing, 3)}


def features(obs: list[float], history: list[dict], timeout_s: float, initial: list[float] | None = None) -> dict:
    """Compact model state. ``initial`` is the episode's first observation (fuel and mass used since then).

    obs[2] is KSP's MonoPropellant resource amount (``resources.amount``), in resource units, not kg, so it
    stays ``_raw``. Mass is kg, so ``mass_used_kg`` is the unit-safe propellant figure.
    """
    bp, bv = obs[3:6], obs[6:9]
    initial = initial or obs
    return {
        "mission_time_s": round(obs[0], 2),
        "time_remaining_s": round(max(0.0, timeout_s - obs[0]), 2),
        "bandit_mass_kg": round(obs[1], 2),
        "bandit_fuel_raw": round(obs[2], 3),
        "fuel_used_raw": round(initial[2] - obs[2], 3),
        "mass_used_kg": round(initial[1] - obs[1], 3),
        "lady": relative(bp, bv, obs[9:12], obs[12:15]),
        "guard": relative(bp, bv, obs[15:18], obs[18:21]),
        "recent": [{"t_s": h["mission_time_s"], "lady_distance_m": h["lady"]["distance_m"],
                    "guard_distance_m": h["guard"]["distance_m"], "action": h.get("action")}
                   for h in history],
    }


def mission_text(settings: dict) -> str:
    """Mission statement built from the loaded scenario's effective rules, not assumed defaults."""
    def m(key):
        v = settings.get(key)
        return f"{v:g} m" if isinstance(v, (int, float)) else "the scenario's capture distance"

    t = settings.get("episode_timeout_s")
    limit = f"before mission time {t:g} s" if isinstance(t, (int, float)) else "within the episode time limit"
    return (f"Mission: move the Bandit within {m('lady_capture_dist_m')} of the Lady before the Guard gets within "
            f"{m('bandit_capture_dist_m')} of the Bandit, {limit}. Coordinates: right-handed celestial-body-centred "
            "inertial (CBCI), metres and metres per second.")


# ---------------------------------------------------------------- Kev request / response

def build_request(feat: dict, cfg: LoopConfig, context: dict | None = None, mission: str | None = None) -> dict:
    state = {"mission": mission or mission_text({}), "features": feat}
    if context:
        state["context"] = context
    return {
        "model": cfg.kev_model,
        "state": json.dumps(state, separators=(",", ":")),
        "questions": {"action": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": dict(ACTION_CRITERIA)}},
    }


def parse_response(resp) -> dict:
    try:
        ans = resp["answers"]["action"]
        choice = ans["choice"]
    except (KeyError, TypeError) as e:
        raise LoopError("invalid_response", f"missing answers.action.choice ({e!r})")
    if choice not in ACTION_IDS:
        raise LoopError("invalid_response", f"unknown action {choice!r}")
    conf = ans.get("confidence")
    if conf is not None and not (isinstance(conf, (int, float)) and math.isfinite(conf)):
        raise LoopError("invalid_response", f"non-finite confidence {conf!r}")
    probs = ans.get("probabilities") or {}
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in probs.values()):
        raise LoopError("invalid_response", "non-finite probability")
    return {"choice": choice, "confidence": conf, "probabilities": probs, "model": resp.get("model")}


def kev_choose(request: dict, cfg: LoopConfig) -> tuple[dict, float]:
    req = urllib.request.Request(
        f"{cfg.kev_base_url.rstrip('/')}/v1/systemone", data=json.dumps(request).encode(), method="POST",
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=cfg.request_timeout_s) as r:
            raw = r.read()
    except TimeoutError as e:
        raise LoopError("model_timeout", repr(e))
    except urllib.error.HTTPError as e:  # e.g. 422: state over the server's token limit
        raise LoopError("model_http_error", f"HTTP {e.code}: {e.read()[:300]!r}")
    except urllib.error.URLError as e:
        if isinstance(e.reason, TimeoutError):
            raise LoopError("model_timeout", repr(e))
        raise LoopError("model_connection_error", repr(e))
    except (ConnectionError, OSError) as e:  # reset mid-response
        raise LoopError("model_connection_error", repr(e))
    latency = time.perf_counter() - t0
    if latency > cfg.request_timeout_s:  # urllib's timeout is per socket operation, not per request
        raise LoopError("model_timeout", f"request took {latency:.2f}s > {cfg.request_timeout_s}s")
    try:
        resp = json.loads(raw)
    except ValueError as e:
        raise LoopError("invalid_response", f"not JSON ({e!r})")
    return parse_response(resp), latency


# ---------------------------------------------------------------- action mapping and validation

def command_for(action_id: str, cfg: LoopConfig) -> dict:
    if action_id not in ACTION_IDS:
        raise LoopError("invalid_action", repr(action_id))
    m = cfg.thrust_magnitude
    vec = {"COAST": (0, 0, 0), "POS_X": (m, 0, 0), "NEG_X": (-m, 0, 0), "POS_Y": (0, m, 0),
           "NEG_Y": (0, -m, 0), "POS_Z": (0, 0, m), "NEG_Z": (0, 0, -m)}[action_id]
    cmd = {"burn_vec": [float(vec[0]), float(vec[1]), float(vec[2]), float(cfg.burn_duration_s)],
           "ref_frame": REF_FRAME_CBCI, "vec_type": VEC_TYPE_THROTTLE}
    validate_command(cmd)
    return cmd


def validate_command(cmd: dict):
    bv = cmd.get("burn_vec")
    if not (isinstance(bv, list) and len(bv) == 4):
        raise LoopError("invalid_command", f"burn_vec must have 4 values: {bv!r}")
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in bv):
        raise LoopError("invalid_command", "non-finite burn_vec")
    if any(abs(x) > 1.0 for x in bv[:3]):
        raise LoopError("invalid_command", "throttle outside [-1, 1]")
    if not (0.0 <= bv[3] <= MAX_DURATION_S):
        raise LoopError("invalid_command", f"duration {bv[3]} outside [0, {MAX_DURATION_S}]")
    if cmd.get("ref_frame") != REF_FRAME_CBCI or cmd.get("vec_type") != VEC_TYPE_THROTTLE:
        raise LoopError("invalid_command", "expected ref_frame=1 (CBCI) and vec_type=0 (throttle)")


# ---------------------------------------------------------------- environments

def _step(env, cmd):
    """KSPDG 0.15 returns (obs, reward, done, info); a Gymnasium-style 5-tuple is also accepted."""
    try:
        out = env.step(cmd)
    except Exception as e:  # noqa: BLE001  (kRPC connection loss, game crash)
        raise LoopError("simulator_error", repr(e))
    if len(out) == 5:
        obs, rew, term, trunc, info = out
        return obs, rew, bool(term or trunc), info
    if len(out) == 4:
        return out
    raise LoopError("env_api", f"unexpected step() result of length {len(out)}")


def _reset(env):
    try:
        out = env.reset()
    except Exception as e:  # noqa: BLE001
        raise LoopError("simulator_error", repr(e))
    return out if isinstance(out, tuple) and len(out) == 2 else (out, {})


def make_env(kind: str, cfg: LoopConfig, check_krpc: bool = True):
    if kind == "kspdg":
        try:
            import kspdg  # real game: needs KSP running with kRPC and the KSPDG mod
        except ImportError as e:
            raise LoopError("env_unavailable", f"kspdg is not installed ({e!r})")

        env_cls = getattr(kspdg, cfg.scenario, None)
        if env_cls is None:
            raise LoopError("env_unavailable", f"kspdg {getattr(kspdg, '__version__', '?')} has no scenario {cfg.scenario}")
        try:
            # LG3+ classes take **kwargs only, so pass the timeout explicitly; LadyBanditGuardGroup1Env accepts it
            env = env_cls(episode_timeout=cfg.episode_timeout_s)
        except Exception as e:  # noqa: BLE001
            raise LoopError("env_unavailable", repr(e))
        if isinstance(env, str):  # kspdg's placeholder for LG3-LG7 when juliacall is missing returns a message
            raise LoopError("env_unavailable", " ".join(env.split()))
        eff = getattr(env, "episode_timeout", None)
        if eff is not None and float(eff) != float(cfg.episode_timeout_s):
            raise LoopError("invalid_config", f"env episode_timeout={eff} != configured {cfg.episode_timeout_s}")
        _route_krpc(cfg)
        if check_krpc:
            _check_krpc(cfg)
        return env
    if kind == "fixture":
        # TEST FIXTURE ONLY: pure-Python stand-in, not the real game. lg7 is a proxy Guard.
        from hsc.sim.lbg1_lite import LBG1LiteEnv

        return LBG1LiteEnv("i2", "lg7", episode_timeout=cfg.episode_timeout_s, two_axis=False)
    raise LoopError("invalid_config", f"unknown env {kind!r}")


def _route_krpc(cfg: LoopConfig):
    """KSPDG calls ``krpc.connect(name=...)`` with the default 127.0.0.1:50000/50001. Point it at the configured
    server instead (for example the KSP container's address) by binding the address into ``krpc.connect``."""
    import functools

    import krpc

    base = getattr(krpc.connect, "func", krpc.connect)
    krpc.connect = functools.partial(base, address=cfg.krpc_address, rpc_port=cfg.krpc_rpc_port,
                                     stream_port=cfg.krpc_stream_port)


def _check_krpc(cfg: LoopConfig, timeout_s: float = 3.0):
    import socket

    for port in (cfg.krpc_rpc_port, cfg.krpc_stream_port):
        try:
            socket.create_connection((cfg.krpc_address, port), timeout=timeout_s).close()
        except OSError as e:
            raise LoopError("env_unavailable", f"kRPC server not reachable at {cfg.krpc_address}:{port} ({e!r}); "
                                               "start KSP, load the game and start the kRPC server")


def env_settings(env, cfg: LoopConfig) -> dict:
    """Effective scenario rules read back from the environment object (None when it does not expose them)."""
    def num(name):
        v = getattr(env, name, None)
        return float(v) if isinstance(v, (int, float)) else None

    shared = num("capture_dist")  # the lbg1_lite fixture uses one distance for both
    timeout = num("episode_timeout")
    return {"env_class": type(env).__name__, "loadfile": getattr(env, "loadfile", None),
            "episode_timeout_s": timeout if timeout is not None else cfg.episode_timeout_s,
            "episode_timeout_source": "env" if timeout is not None else "config",
            "lady_capture_dist_m": num("lady_capture_dist") or shared,
            "bandit_capture_dist_m": num("bandit_capture_dist") or shared}


# ---------------------------------------------------------------- loop

def _versions(env_kind: str, cfg: LoopConfig) -> dict:
    v = {"python": platform.python_version(), "platform": platform.platform(), "env": env_kind}
    try:
        v["repo_rev"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                       cwd=pathlib.Path(__file__).parent, timeout=5).stdout.strip()
    except Exception:  # noqa: BLE001
        v["repo_rev"] = None
    if env_kind == "kspdg":
        import importlib.metadata as md

        for pkg in ("kspdg", "krpc", "gymnasium", "juliacall"):
            try:
                v[pkg] = md.version(pkg)
            except Exception:  # noqa: BLE001
                v[pkg] = None
    try:
        with urllib.request.urlopen(f"{cfg.kev_base_url.rstrip('/')}/v1/models", timeout=5) as r:
            v["kev_models"] = json.loads(r.read())
    except Exception as e:  # noqa: BLE001
        v["kev_models"] = f"unavailable: {e!r}"
    return v


def run(env, cfg: LoopConfig, max_steps: int | None, env_kind: str = "custom", choose=None,
        trace_path: str | None = None, context_provider=None) -> dict:
    """Run the sequential loop. ``choose(request) -> (parsed, latency_s)`` defaults to the Kev server."""
    cfg.validate()
    choose = choose or (lambda req: kev_choose(req, cfg))
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    path = pathlib.Path(trace_path or pathlib.Path(cfg.trace_dir) / f"{run_id}.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    settings = env_settings(env, cfg)
    mission = mission_text(settings)
    outcome = {"run_id": run_id, "trace": str(path), "steps": 0, "terminal_reason": None}
    with open(path, "w") as f:
        f.write(json.dumps({"record": "run_header", "run_id": run_id, "config": dataclasses.asdict(cfg),
                            "scenario_settings": settings, "mission": mission,
                            "versions": _versions(env_kind, cfg)}) + "\n")
        f.flush()
        try:
            if max_steps != 0:
                _warmup(choose, cfg, mission, f)
            wall_before = time.time()
            obs_raw, info = _reset(env)
            obs = parse_observation(obs_raw)
            obs_wall = time.time()
            initial = obs
            f.write(json.dumps({"record": "observation", "run_id": run_id, "wall_time": obs_wall,
                                "reset_wall_s": round(obs_wall - wall_before, 3), "mission_time_s": obs[0],
                                "observation": obs, "features": features(obs, [], settings["episode_timeout_s"]),
                                "info": _plain(info)}) + "\n")
            f.flush()
            step = 0
            while max_steps is None or step < max_steps:
                feat = features(obs, history, settings["episode_timeout_s"], initial)
                ctx = context_provider(feat) if context_provider else None
                request = build_request(feat, cfg, ctx, mission)
                rec = {"record": "decision", "run_id": run_id, "step": step, "wall_time": time.time(),
                       "mission_time_s": obs[0], "observation": obs, "features": feat,
                       "model": cfg.kev_model, "context_version": None if not ctx else ctx.get("version"),
                       "context_age_s": None if not ctx else ctx.get("age_s")}
                try:
                    t0 = time.perf_counter()
                    parsed, model_latency = choose(request)
                    obs_age = time.time() - obs_wall
                    if obs_age > cfg.max_observation_age_s:
                        raise LoopError("stale_decision", f"observation age {obs_age:.2f}s > {cfg.max_observation_age_s}s")
                    cmd = command_for(parsed["choice"], cfg)
                    decision_latency = time.perf_counter() - t0
                    rec.update({"choice": parsed["choice"], "confidence": parsed["confidence"],
                                "probabilities": parsed["probabilities"], "burn_command": cmd,
                                "model_latency_s": round(model_latency, 4),
                                "decision_latency_s": round(decision_latency, 4), "observation_age_s": round(obs_age, 4)})
                    wall_before = time.time()
                    next_raw, _, done, info = _step(env, cmd)
                    rec["step_wall_s"] = round(time.time() - wall_before, 3)
                    next_obs = parse_observation(next_raw)
                    obs_wall = time.time()
                except LoopError as e:
                    rec.update({"terminal_reason": e.reason, "error": e.detail})
                    f.write(json.dumps(rec, default=float) + "\n")
                    raise
                rec.update({"next_observation": next_obs, "next_mission_time_s": next_obs[0],
                            "mission_dt_s": round(next_obs[0] - obs[0], 3),  # vs step_wall_s: the two clocks
                            "done": bool(done), "info": _plain(info)})
                if done:
                    rec["terminal_reason"] = _terminal_reason(env, info, next_obs, settings)
                f.write(json.dumps(rec, default=float) + "\n")
                f.flush()
                history = (history + [{**feat, "action": parsed["choice"]}])[-cfg.history_len:]
                obs = next_obs
                step += 1
                outcome["steps"] = step
                if done:
                    outcome["terminal_reason"] = rec["terminal_reason"]
                    outcome["info"] = rec["info"]
                    break
            else:
                outcome["terminal_reason"] = "observation_only" if max_steps == 0 else "max_steps_reached"
        except LoopError as e:
            outcome["terminal_reason"] = e.reason
            outcome["error"] = e.detail
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as e:  # noqa: BLE001
                    outcome["cleanup_error"] = repr(e)
            f.write(json.dumps({"record": "run_end", **outcome}, default=float) + "\n")
    return outcome


def _warmup(choose, cfg: LoopConfig, mission: str, f):
    """Untimed requests before the episode so the first real decision doesn't pay model load costs. The state is a
    labelled placeholder (no observation exists yet) and nothing is executed."""
    for i in range(cfg.warmup_requests):
        feat = {"note": "WARMUP ONLY: placeholder state, not an observation; the answer is discarded",
                "lady": relative([0, 0, 0], [0, 0, 0], [1000.0 + 100 * i, 0, 0], [0, 0, 0]),
                "guard": relative([0, 0, 0], [0, 0, 0], [500.0, 50.0 * i, 0], [0, 0, 0])}
        _, latency = choose(build_request(feat, cfg, None, mission))
        f.write(json.dumps({"record": "warmup", "i": i, "model_latency_s": round(latency, 4)}) + "\n")


def _plain(info) -> dict:
    """JSON-safe scalars from an info dict (KSPDG returns numpy floats)."""
    out = {}
    for k, v in (info or {}).items():
        if hasattr(v, "item") and not hasattr(v, "__len__"):
            v = v.item()
        if isinstance(v, (int, float, bool, str)):
            out[k] = v
    return out


def _terminal_reason(env, info, obs, settings) -> str:
    """KSPDG keeps the capture flags on the env object (set by its termination thread), not in ``info``."""
    info = info or {}
    lady = bool(getattr(env, "is_lady_captured", False) or info.get("is_lady_captured"))
    bandit = bool(getattr(env, "is_bandit_captured", False) or info.get("is_bandit_captured"))
    if lady and bandit:
        return "capture_and_interception"
    if lady:
        return "capture"
    if bandit:
        return "interception"
    if obs[0] >= settings["episode_timeout_s"]:
        return "timeout"
    return "simulator_done"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", choices=["kspdg", "fixture"], default="kspdg",
                    help="kspdg = real game (default); fixture = pure-Python test stand-in")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--steps", type=int, default=1,
                   help="number of decisions (default 1: connection test; 0: read one observation, no Kev, no burn)")
    g.add_argument("--episode", action="store_true", help="run until the simulator terminates the episode")
    for fld in dataclasses.fields(LoopConfig):
        ap.add_argument("--" + fld.name.replace("_", "-"), type=type(fld.default), default=fld.default)
    args = ap.parse_args(argv)
    cfg = LoopConfig(**{fld.name: getattr(args, fld.name) for fld in dataclasses.fields(LoopConfig)})
    try:
        cfg.validate()
        if args.steps < 0:
            raise LoopError("invalid_config", "--steps must be >= 0")
        env = make_env(args.env, cfg)
    except LoopError as e:
        print(json.dumps({"terminal_reason": e.reason, "error": e.detail}))
        raise SystemExit(2)
    out = run(env, cfg, None if args.episode else args.steps, env_kind=args.env)
    print(json.dumps(out, indent=2, default=float))
    raise SystemExit(0 if out["terminal_reason"] in SIMULATOR_OUTCOMES + ("max_steps_reached", "observation_only") else 1)


if __name__ == "__main__":
    main()
