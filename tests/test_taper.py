from vapecoach.config import load_settings
from vapecoach.taper import compute_state, judge_new_session, target_for_day

H = 3600.0


def sess(starts_h, dur_s=120):
    return [{"session_id": i + 1, "start": h * H, "end": h * H + dur_s, "puffs": 3} for i, h in enumerate(starts_h)]


def test_no_data():
    assert compute_state([], now=0).phase == "no_data"


def test_baseline_is_observe_only():
    s = load_settings()
    ss = sess([0, 1, 2])
    st = compute_state(ss, now=5 * H, s=s)
    assert st.phase == "baseline" and st.target_gap_s is None
    v = judge_new_session(ss, 5 * H, s)
    assert v.on_time and v.severity == 0 and v.phase == "baseline"


def test_baseline_gap_is_median_and_target_grows_10pct_per_day():
    s = load_settings()
    ss = sess([0, 1, 2, 3, 5])  # gaps ~ 58,58,58,118 min -> median 58 min
    st = compute_state(ss, now=25 * H, s=s)
    assert st.phase == "coaching" and st.day_index == 1
    assert abs(st.baseline_gap_s - (H - 120)) < 1
    assert abs(st.target_gap_s - st.baseline_gap_s * 1.1) < 1
    st3 = compute_state(ss, now=24 * H + 2 * 86400 + 60, s=s)
    assert st3.day_index == 3 and abs(st3.target_gap_s - st.baseline_gap_s * 1.1 ** 3) < 1


def test_fallback_gap_when_baseline_sparse():
    s = load_settings(fallback_baseline_gap_min=45)
    st = compute_state(sess([0]), now=30 * H, s=s)
    assert abs(st.baseline_gap_s - 45 * 60) < 1


def test_target_capped():
    s = load_settings(max_target_gap_hours=2)
    assert target_for_day(H, 100, s) == 2 * H


def test_verdict_severity():
    s = load_settings()
    ss = sess([0, 1, 2, 3, 25])  # last session at 25h (coaching), target ~ 63.8 min
    last_end = ss[-1]["end"]
    target = compute_state(ss, now=last_end + 1, s=s).target_gap_s
    on = judge_new_session(ss, last_end + target + 5, s)
    mild = judge_new_session(ss, last_end + 0.8 * target, s)
    med = judge_new_session(ss, last_end + 0.5 * target, s)
    bad = judge_new_session(ss, last_end + 0.1 * target, s)
    assert on.on_time and on.severity == 0
    assert (mild.severity, med.severity, bad.severity) == (1, 2, 3)
    assert bad.early_by_s > med.early_by_s > mild.early_by_s > 0


def test_seconds_until_allowed():
    s = load_settings()
    ss = sess([0, 1, 2, 3, 25])
    st = compute_state(ss, now=ss[-1]["end"] + 600, s=s)
    assert abs(st.seconds_until_allowed - (st.target_gap_s - 600)) < 1
