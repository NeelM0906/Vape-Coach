"""Rep counters on synthetic Pose landmark frames."""
import math

from vapecoach.reps import RepCounter


def pose(*, arms_up=False, feet_apart=False, legs_visible=True, knee_deg=175, elbow_deg=175):
    p = [(0.5, 0.5, 0.0, 1.0)] * 33
    p = list(p)
    sh_y, hip_y, ank_y = 0.35, 0.6, 0.95
    p[0] = (0.5, 0.25, 0, 1)
    p[11], p[12] = (0.56, sh_y, 0, 1), (0.44, sh_y, 0, 1)
    p[23], p[24] = (0.54, hip_y, 0, 1), (0.46, hip_y, 0, 1)
    if arms_up:
        p[15], p[16] = (0.62, 0.12, 0, 1), (0.38, 0.12, 0, 1)
    else:
        p[15], p[16] = (0.58, 0.62, 0, 1), (0.42, 0.62, 0, 1)
    lv = 1.0 if legs_visible else 0.05
    ax = 0.70 if feet_apart else 0.55
    # knee angle: place knee then ankle at angle knee_deg
    for side, hx, kx in ((0, 0.54, 0.54), (1, 0.46, 0.46)):
        hip = p[23 + side]
        knee = (kx, hip_y + 0.17, 0, lv)
        rad = math.radians(180 - knee_deg)
        sign = 1 if side == 0 else -1
        ank = (knee[0] + sign * 0.17 * math.sin(rad) + (ax - 0.55) * sign, knee[1] + 0.17 * math.cos(rad), 0, lv)
        p[25 + side], p[27 + side] = knee, ank
    # elbow angle for pushups
    for side in (0, 1):
        sh = p[11 + side]
        el = (sh[0], sh[1] + 0.12, 0, 1)
        rad = math.radians(180 - elbow_deg)
        if not arms_up:
            p[13 + side] = el
            p[15 + side] = (el[0] + 0.12 * math.sin(rad), el[1] + 0.12 * math.cos(rad), 0, 1)
    return p


def feed(counter, frames, dt=0.2):
    t = 0.0
    for f in frames:
        counter.update(f, t)
        t += dt
    return counter.count


def test_jumping_jacks_full_body():
    c = RepCounter("jumping_jacks")
    frames = []
    for _ in range(5):
        frames += [pose(), pose(arms_up=True, feet_apart=True), pose(arms_up=True, feet_apart=True), pose()]
    assert feed(c, frames) == 5 and c.st.mode == "full"


def test_jumping_jacks_arms_only_when_legs_out_of_frame():
    c = RepCounter("jumping_jacks")
    frames = []
    for _ in range(4):
        frames += [pose(legs_visible=False), pose(arms_up=True, legs_visible=False), pose(legs_visible=False)]
    assert feed(c, frames) == 4 and c.st.mode == "arms_only"


def test_jacks_need_feet_apart_when_legs_visible():
    c = RepCounter("jumping_jacks")
    frames = [pose(), pose(arms_up=True, feet_apart=False), pose()] * 3
    assert feed(c, frames) == 0


def test_jitter_does_not_double_count():
    c = RepCounter("jumping_jacks", min_phase_s=0.12)
    o, cl = pose(arms_up=True, feet_apart=True), pose()
    # one real rep with a 1-frame flicker back to "open" while closing (50 ms frames)
    frames = [cl] * 4 + [o] * 4 + [cl] * 2 + [o] + [cl] * 4
    assert feed(c, frames, dt=0.05) == 1


def test_squats():
    c = RepCounter("squats")
    frames = [pose(knee_deg=175), pose(knee_deg=80), pose(knee_deg=175)] * 3
    assert feed(c, frames) == 3


def test_pushups():
    c = RepCounter("pushups")
    frames = [pose(elbow_deg=175), pose(elbow_deg=70), pose(elbow_deg=175)] * 2
    assert feed(c, frames) == 2


def test_missing_pose_ignored():
    c = RepCounter("jumping_jacks")
    assert not c.update(None, 0.0) and c.count == 0
