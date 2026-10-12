"""The browser game (web/lady-bandit-guard.html) ports lbg1_lite to JavaScript. Run the same command sequence
through both and require the same trajectories, for every start and Guard policy. Needs node."""

import json
import pathlib
import re
import shutil
import subprocess

import numpy as np
import pytest

from hsc.sim.lbg1_lite import LBG1LiteEnv

GAME = pathlib.Path(__file__).resolve().parents[1] / "web" / "lady-bandit-guard.html"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs node")


def _commands(seed: int, n: int = 240):
    """Deterministic mix of body-frame (ref_frame 0) and inertial (ref_frame 1) burns, coasts and odd durations."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        ref = int(rng.integers(0, 2))
        vec = [float(x) for x in rng.choice([-1.0, -0.2, 0.0, 0.0, 0.2, 1.0], size=3)]
        if ref == 0:
            vec[2] = 0.0  # the game's Bandit is planar in body axes
        out.append({"burn_vec": vec + [float(rng.choice([0.5, 1.0, 1.0, 2.3]))], "ref_frame": ref, "vec_type": 0})
    return out


def _run_js(init, policy, cmds, tmp_path):
    src = GAME.read_text()
    physics = re.search(r"/\* LBG-PHYSICS-BEGIN.*?\*/(.*?)/\* LBG-PHYSICS-END \*/", src, re.S).group(1)
    runner = tmp_path / "run.js"
    runner.write_text(physics + f"""
const cmds = {json.dumps(cmds)};
const s = LBG.create({{ init: {json.dumps(init)}, policy: {json.dumps(policy)} }});
const out = [LBG.observation(s)];
for (const c of cmds) {{ if (s.done) break; out.push(LBG.step(s, c.burn_vec, c.ref_frame)); }}
console.log(JSON.stringify({{ obs: out, lady: s.ladyCaptured, bandit: s.banditCaptured, m: s.m }}));
""")
    res = subprocess.run(["node", str(runner)], capture_output=True, text=True, timeout=120, check=True)
    return json.loads(res.stdout)


@pytest.mark.parametrize("init", ["i1", "i2"])
@pytest.mark.parametrize("policy", ["lg0", "lg1", "lg7"])
def test_js_port_matches_python(init, policy, tmp_path):
    cmds = _commands(seed=["i1", "i2"].index(init) * 10 + ["lg0", "lg1", "lg7"].index(policy))
    env = LBG1LiteEnv(init, policy, two_axis=False)
    obs, _ = env.reset()
    py = [obs.tolist()]
    for c in cmds:
        if env.done:
            break
        obs, _, _, info = env.step(c)
        py.append(obs.tolist())
    js = _run_js(init, policy, cmds, tmp_path)
    assert len(js["obs"]) == len(py)
    a, b = np.asarray(js["obs"]), np.asarray(py)
    assert np.allclose(a[:, 0], b[:, 0], atol=1e-9)                       # mission time
    assert np.allclose(a[:, 1:3], b[:, 1:3], atol=1e-6)                   # mass and propellant
    assert np.max(np.abs(a[:, 3:] - b[:, 3:])) < 1e-3                     # positions [m] and velocities [m/s]
    assert (js["lady"], js["bandit"]) == (env.is_lady_captured, env.is_bandit_captured)
    assert js["m"]["closestLB"] == pytest.approx(info["closest_lady_bandit_approach"], abs=1e-3)
    assert js["m"]["closestBG"] == pytest.approx(info["closest_bandit_guard_approach"], abs=1e-3)
