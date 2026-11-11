"""The watcher counts owed reps by itself (no `hermes vape reps` needed)."""
import importlib.util
from pathlib import Path

import numpy as np

from vapecoach.cli import AutoReps
from vapecoach.coach import Coach
from vapecoach.config import load_settings

_spec = importlib.util.spec_from_file_location("_reps_helpers", Path(__file__).with_name("test_reps.py"))
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
pose = _helpers.pose


class FakeLms:
    """Stands in for MediaPipe: returns the next scripted pose on every Pose call."""

    def __init__(self, poses):
        self.poses = list(poses)

    def run(self, rgb, t, which):
        return {"pose": self.poses.pop(0) if self.poses else None}


def jacks(n, legs=True):
    out = []
    for _ in range(n):
        out += [pose(arms_up=False, legs_visible=legs)] * 2 + [pose(arms_up=True, feet_apart=True, legs_visible=legs)] * 2
    return out + [pose(arms_up=False, legs_visible=legs)] * 2


def run(reps, frames, t0=0.0):
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    msgs, t = [], t0
    for _ in range(frames):
        m = reps.feed(frame, t)
        if m:
            msgs.append(m)
        t += 0.25
    return msgs, t


def test_owed_challenge_is_counted_and_cleared_automatically(store):
    c = Coach(store, load_settings())
    store.add_challenge("jumping_jacks", 3)
    reps = AutoReps(c, FakeLms(jacks(4)))
    assert "counting automatically" in reps.sync()
    assert reps.active and store.get("live_reps")["active"] is True
    msgs, t = run(reps, 40)
    assert store.open_challenges() == []
    assert store.reps()[0]["verified"] == "camera"
    assert not reps.active and store.get("live_reps")["active"] is False
    assert any("Challenge cleared" in m for m in msgs)
    assert reps.recently_moving(t + 5)        # cooldown: walking back to the desk
    assert not reps.recently_moving(t + 30)


def test_each_rep_is_credited_as_it_happens(store):
    c = Coach(store, load_settings())
    store.add_challenge("jumping_jacks", 10)
    reps = AutoReps(c, FakeLms(jacks(2)))
    reps.sync()
    run(reps, 20)
    assert store.open_challenge()["done"] == 2
    assert store.get("live_reps")["count"] == 2


def test_honor_mode_never_starts_the_counter(store):
    c = Coach(store, load_settings(verify_reps_with_camera=False))
    store.add_challenge("jumping_jacks", 3)
    reps = AutoReps(c, FakeLms([]))
    assert reps.sync() is None and not reps.active


def test_counter_stops_when_challenge_closes_elsewhere(store):
    c = Coach(store, load_settings())
    store.add_challenge("jumping_jacks", 5)
    reps = AutoReps(c, FakeLms([]))
    reps.sync()
    store.waive_challenges()
    assert "nothing owed" in reps.sync() and not reps.active
