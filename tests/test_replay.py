"""Replay renderer (hsc/replay.py) on short lbg1_lite TEST FIXTURE traces (not the real game)."""

import shutil

import numpy as np
import pytest

from hsc import kspdg_loop as K
from hsc import replay as R


def _trace(tmp_path, steps=6):
    cfg = K.LoopConfig(decider="rule", kev_base_url="http://127.0.0.1:9")
    path = tmp_path / "t.jsonl"
    K.run(K.make_env("fixture", cfg), cfg, steps, env_kind="fixture", trace_path=path)
    return path


def test_load_trace_in_lady_frame(tmp_path):
    tr = R.load_trace(_trace(tmp_path))
    assert tr["env"] == "fixture" and len(tr["t"]) == 7 and len(tr["choices"]) == 6
    # i2 initial conditions: Guard ~600 m and Bandit ~2600 m behind the Lady along-track
    assert tr["bandit"][0][1] == pytest.approx(-2600, abs=20) and tr["guard"][0][1] == pytest.approx(-600, abs=20)
    assert np.allclose(np.linalg.norm(tr["bandit"], axis=1), tr["lady_dist"], rtol=1e-3)
    assert tr["terminal_reason"] == "max_steps_reached"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_render_side_by_side(tmp_path):
    tr = R.load_trace(_trace(tmp_path))
    out = R.render([tr, tr], ["a", "b"], str(tmp_path / "r.mp4"), fps=4, speed=6.0, hold_s=0.5,
                   poster=str(tmp_path / "r.png"))
    assert (tmp_path / "r.mp4").stat().st_size > 0 and (tmp_path / "r.png").stat().st_size > 0
    assert out["frames"] == 5 + 2 and out["outcomes"] == ["max_steps_reached"] * 2
