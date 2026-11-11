"""Hand-to-mouth state machine on synthetic landmark-derived sequences."""
from vapecoach.detect_fsm import FSMConfig, HandToMouthFSM, Obs


def run(seq):
    fsm = HandToMouthFSM()
    evs = [e for o in seq if (e := fsm.update(o))]
    last = seq[-1].t if seq else 0
    tail = fsm.flush(last + 10)
    return (evs + ([tail] if tail else [])), fsm


def gesture(t0, *, dwell=1.5, grip=0.2, exhale=0.6, jaw=0.1, fps=12, exhale_delay=0.8, approach_s=0.6,
            idle_after=6.0, at=0.15, source="face_hand", pitch=None):
    """approach -> dwell at the mouth -> release -> exhale bump -> idle."""
    dt, out, t = 1.0 / fps, [], t0
    for i in range(int(approach_s * fps)):  # hand rising from the desk
        out.append(Obs(t, 3.0 - 2.4 * i / max(1, approach_s * fps - 1), grip, 0.05, jaw, pitch=-0.15)); t += dt
    for _ in range(int(dwell * fps)):
        out.append(Obs(t, at, grip, 0.05, jaw, source, pitch=pitch)); t += dt
    for _ in range(int(exhale_delay * fps)):
        out.append(Obs(t, 2.5, None, 0.05, 0.1)); t += dt
    for _ in range(int(1.0 * fps)):
        out.append(Obs(t, 3.0, None, exhale, 0.15)); t += dt
    for _ in range(int(idle_after * fps)):
        out.append(Obs(t, 3.5, None, 0.05, 0.05)); t += dt
    return out


def test_classic_puff_detected():
    evs, _ = run(gesture(0.0))
    assert len(evs) == 1
    e = evs[0]
    assert 0.45 <= e.t_start <= 0.8 and 1.2 <= e.dwell_s <= 1.7 and e.exhale > 0.9 and e.score >= 0.55


def test_stealth_hit_small_exhale_with_pinch_grip():
    evs, _ = run(gesture(0.0, exhale=0.3, grip=0.25, dwell=1.2))
    assert len(evs) == 1


def test_coffee_sip_rejected():
    evs, fsm = run(gesture(0.0, grip=1.6, exhale=0.08, dwell=2.0))
    assert evs == [] and fsm.rejected and "weak" in fsm.rejected[-1]["reason"]


def test_chin_rest_rejected():
    evs, fsm = run(gesture(0.0, dwell=20.0, exhale=0.6))
    assert evs == [] and "too long" in fsm.rejected[-1]["reason"]


def test_scratch_rejected():
    evs, fsm = run(gesture(0.0, dwell=0.17, exhale=0.0))
    assert evs == [] and "too short" in fsm.rejected[-1]["reason"]


def test_eating_wide_jaw_rejected():
    evs, fsm = run(gesture(0.0, jaw=0.8, exhale=0.2, grip=0.3))
    assert evs == [] and "jaw" in fsm.rejected[-1]["reason"]


def test_two_puffs_counted_separately():
    seq = gesture(0.0) + gesture(12.0)
    evs, _ = run(seq)
    assert len(evs) == 2 and evs[1].t_start > 12


def test_double_tap_is_one_puff():
    """Hand leaves briefly (<1 s) and comes back: still one dwell, one puff."""
    seq = [Obs(0.0, 3.0), Obs(0.1, 1.5), Obs(0.2, 0.15, 0.2)]
    t = 0.2
    for _ in range(8):
        t += 0.1; seq.append(Obs(t, 0.15, 0.2))
    t += 0.1; seq.append(Obs(t, 1.8))          # quick release
    t += 0.2; seq.append(Obs(t, 0.15, 0.2))     # back at the mouth
    for _ in range(6):
        t += 0.1; seq.append(Obs(t, 0.15, 0.2))
    for _ in range(30):
        t += 0.1; seq.append(Obs(t, 3.0, None, 0.7))
    evs, _ = run(seq)
    assert len(evs) == 1


def test_tracking_lost_mid_dwell_drops_candidate():
    seq = [Obs(0.0, 2.0), Obs(0.1, 0.5, 0.2), Obs(0.2, 0.15, 0.2)] + [Obs(0.3 + i * 0.1, None) for i in range(30)]
    evs, fsm = run(seq)
    assert evs == [] and fsm.state == "idle"


def test_low_frame_rate_still_works():
    evs, _ = run(gesture(0.0, fps=4, dwell=2.0))
    assert len(evs) == 1


def test_refractory_merges_bounce():
    cfg = FSMConfig(refractory_s=30)
    fsm = HandToMouthFSM(cfg)
    evs = [e for o in gesture(0.0) + gesture(9.0) if (e := fsm.update(o))]
    assert len(evs) == 1


def test_wants_high_rate_tracks_candidate():
    fsm = HandToMouthFSM()
    assert not fsm.wants_high_rate
    fsm.update(Obs(0.0, 1.8))
    assert fsm.wants_high_rate


# --- physical gates against common look-alikes ---

def test_bottle_sip_hand_never_at_lips_rejected():
    """Bottle sips: the hand holds the bottle body, 0.4-1 face widths from the mouth."""
    seq, t = [Obs(0.0, 3.0, pitch=-0.15)], 0.0
    for d in (1.2, 0.8, 0.5, 0.58, 0.9, 1.4):   # one frame or two near the lips, then away
        t += 0.1; seq.append(Obs(t, d, 0.4, 0.9))
    for _ in range(60):
        t += 0.1; seq.append(Obs(t, 3.0, None, 0.1))
    evs, fsm = run(seq)
    assert evs == [] and fsm.rejected and ("never at the lips" in fsm.rejected[-1]["reason"]
                                           or "too short" in fsm.rejected[-1]["reason"])


def test_face_hidden_by_bottle_rejected():
    evs, fsm = run(gesture(0.0, source="pose", exhale=0.9))
    assert evs == [] and "face hidden" in fsm.rejected[-1]["reason"]


def test_head_tilted_back_rejected():
    evs, fsm = run(gesture(0.0, pitch=0.15, exhale=0.9))   # baseline -0.15 -> +0.30 tilt
    assert evs == [] and "tilted back" in fsm.rejected[-1]["reason"]


def test_level_head_still_detected():
    evs, _ = run(gesture(0.0, pitch=-0.10))
    assert len(evs) == 1 and evs[0].pitch_up is not None and evs[0].pitch_up < 0.2


def test_fingers_at_lips_then_hit_scores_the_hit_segment():
    """fingers rest at the lips (~0.6), a soft release, then a 5 s deep hit.

    The whole episode is >7 s; the hit segment must still be detected, starting at the hit.
    """
    seq, t = [Obs(0.0, 3.0)], 0.0
    for _ in range(50):                      # 5 s hovering at the lips, shallow
        t += 0.1; seq.append(Obs(t, 0.45, 0.3, 0.1))
    for _ in range(5):                       # soft release, still under the release distance
        t += 0.1; seq.append(Obs(t, 0.7, 0.3, 0.1))
    t_hit = t + 0.1
    for _ in range(50):                      # 5 s device at the lips
        t += 0.1; seq.append(Obs(t, 0.1, 0.3, 0.1))
    for _ in range(60):
        t += 0.1; seq.append(Obs(t, 3.0, None, 0.7))
    evs, _ = run(seq)
    assert len(evs) == 1 and abs(evs[0].t_start - t_hit) < 0.15 and evs[0].dwell_s <= 7.0


def test_long_chin_rest_without_valid_segment_still_rejected():
    seq, t = [Obs(0.0, 3.0)], 0.0
    for i in range(150):                     # 15 s on the chin, shallow, with wobbles
        t += 0.1; seq.append(Obs(t, 0.45 if i % 30 else 0.7, 0.3, 0.6))
    for _ in range(60):
        t += 0.1; seq.append(Obs(t, 3.0, None, 0.1))
    evs, fsm = run(seq)
    assert evs == []


def test_exhale_into_shirt_right_after_hit_is_same_puff():
    """device at the lips, release, then the shirt pulled up to the mouth 0.5 s later."""
    seq, t = [Obs(0.0, 3.0)], 0.0
    for _ in range(20):
        t += 0.1; seq.append(Obs(t, 0.1, 0.3, 0.6))
    for _ in range(4):                       # release ...
        t += 0.1; seq.append(Obs(t, 2.0, None, 0.3))
    for _ in range(12):
        t += 0.1; seq.append(Obs(t, 0.4, None, 0.3))  # ... no, wait: shirt over the mouth (shallow)
    for _ in range(10):
        t += 0.1; seq.append(Obs(t, 0.1, 0.4, 0.6))
    for _ in range(60):
        t += 0.1; seq.append(Obs(t, 3.0, None, 0.1))
    evs, _ = run(seq)
    assert len(evs) == 1

