import time

from vapecoach.coach import Coach
from vapecoach.config import load_settings
from vapecoach.nudge import turn_context

H = 3600.0


def test_storage_sessions_group_puffs(store):
    for t in (0, 30, 60, 1000, 1030):
        store.add_event(ts=t, session_gap_s=300)
    ss = store.sessions()
    assert [s["puffs"] for s in ss] == [3, 2]
    assert ss[0]["start"] == 0 and ss[0]["end"] == 60


def test_void_and_reset(store):
    store.add_event(ts=1)
    store.add_event(ts=2)
    assert store.void_last_event()["ts"] == 2
    assert len(store.events()) == 1
    store.put("x", {"a": 1})
    assert store.get("x") == {"a": 1}
    store.reset()
    assert store.events() == [] and store.get("x") is None


def test_no_frames_or_landmarks_in_schema(store):
    cols = {r[1] for t in ("events", "interventions", "challenges", "reps")
            for r in store._db.execute(f"PRAGMA table_info({t})").fetchall()}
    assert not cols & {"frame", "image", "landmarks", "jpeg", "png", "crop"}


def test_baseline_then_coaching_flow(store):
    s = load_settings(seed=1)
    c = Coach(store, s)
    base = 1_000_000.0
    # Baseline: sessions every hour, observe-only.
    for h in range(5):
        out = c.record_hit(ts=base + h * H)
        assert out["intervention"]["kind"] == "observe"
    # Puff inside the same session: counted but not judged.
    assert c.record_hit(ts=base + 4 * H + 30)["intervention"] is None
    # Coaching day 1: target ~ 1.1 h. Early sessions climb the ladder one rung each.
    t = base + 25 * H
    assert c.record_hit(ts=t)["intervention"]["kind"] == "praise"   # big gap: on time
    kinds = []
    for i in range(1, 6):
        # 50 min after the previous session (~75% of the target: severity 1, no extra rung)
        out = c.record_hit(ts=t + i * 50 * 60)
        assert out["verdict"]["severity"] in (1, 2)
        kinds.append(out["intervention"]["kind"])
    assert kinds[:4] == ["meme", "pr_threat", "protein_shake", "exercise"]
    assert kinds[4] == "exercise"
    targets = [ch["target"] for ch in store.open_challenges()]
    assert targets == [s.reps_base, s.reps_base + s.reps_step]
    st = c.status(now=t + 6 * 50 * 60)
    assert st["phase"] == "coaching" and len(st["open_challenges"]) == 2


def test_very_early_session_skips_a_rung(store):
    s = load_settings(seed=1)
    c = Coach(store, s)
    c.record_hit(ts=0.0)
    t = 25 * H
    c.record_hit(ts=t)
    out = c.record_hit(ts=t + 600)          # 10 min after: severity 3
    assert out["verdict"]["severity"] == 3 and out["intervention"]["kind"] == "pr_threat"


def test_disabled_rung_falls_back_to_nearest_enabled(store):
    s = load_settings(seed=1, interventions=["exercise", "meme"])
    c = Coach(store, s)
    c.record_hit(ts=0.0)
    t = 25 * H
    c.record_hit(ts=t)
    out = c.record_hit(ts=t + 600)          # rung 2 (pr_threat) disabled -> milder meme
    assert out["intervention"]["kind"] == "meme"


def test_undo_false_alarm_cancels_its_penalty_and_verdict(store):
    s = load_settings(seed=1, interventions=["exercise"])
    c = Coach(store, s)
    c.record_hit(ts=0.0)
    t = 25 * H
    c.record_hit(ts=t)
    streak_before = c.status(now=t + 1)["streak"]
    out = c.record_hit(ts=t + 600)
    assert out["challenge"] and store.open_challenges()
    assert c.undo_last()["id"] == out["event"]["id"]
    assert store.open_challenges() == []
    assert c.status(now=t + 700)["streak"] == streak_before


def test_undo_one_puff_of_many_keeps_the_penalty(store):
    s = load_settings(seed=1, interventions=["exercise"])
    c = Coach(store, s)
    c.record_hit(ts=0.0)
    t = 25 * H
    c.record_hit(ts=t)
    c.record_hit(ts=t + 600)
    c.record_hit(ts=t + 630)                # second puff, same session
    c.undo_last()
    assert len(store.open_challenges()) == 1


def test_honor_done_refused_while_camera_verifies(store):
    store.add_challenge("jumping_jacks", 10)
    assert Coach(store, load_settings()).honor_done()["ok"] is False
    assert store.open_challenge()["done"] == 0
    res = Coach(store, load_settings(verify_reps_with_camera=False)).honor_done()
    assert res["ok"] and store.open_challenges() == []


def test_credit_reps_with_overflow(store):
    s = load_settings()
    c = Coach(store, s)
    a = store.add_challenge("jumping_jacks", 10)
    b = store.add_challenge("jumping_jacks", 5)
    res = c.credit_reps("jumping_jacks", 12, "camera")
    assert res["challenge"]["id"] == a and res["challenge"]["status"] == "done"
    left = store.open_challenges()
    assert len(left) == 1 and left[0]["id"] == b and left[0]["done"] == 2
    c.credit_reps("jumping_jacks", 10, "camera")
    assert store.open_challenges() == []
    assert sum(r["count"] for r in store.reps()) == 22  # overflow kept as free reps


def test_wrong_exercise_does_not_credit(store):
    c = Coach(store, load_settings())
    store.add_challenge("jumping_jacks", 10)
    c.credit_reps("squats", 10, "camera")
    assert store.open_challenge()["done"] == 0


def test_nudge_is_context_only_and_delivered_once(store):
    s = load_settings()
    store.add_intervention("meme", "roast text")
    store.add_challenge("jumping_jacks", 15)
    ctx1 = turn_context(store, s, 1)
    assert ctx1.startswith("[vape-coach]") and "roast text" in ctx1 and "15 jumping jacks" in ctx1
    assert "Never refuse" in ctx1
    ctx2 = turn_context(store, s, 2)
    assert "roast text" not in ctx2 and "15 jumping jacks" in ctx2  # roast delivered once, debt persists
    store.waive_challenges()
    assert turn_context(store, s, 3) is None


def test_nudge_off(store):
    store.add_challenge("jumping_jacks", 15)
    assert turn_context(store, load_settings(gate_mode="off"), 1) is None


def test_block_mode_does_not_exist():
    s = load_settings(gate_mode="block")
    assert s.gate_mode == "nag"


def test_pr_threats_are_marked_as_jokes():
    from vapecoach.interventions import PR_THREATS
    tells = ("kidding", "/s", "joking", "empty threat", "can't", "sarcasm", "won't")
    assert all(any(t in p.lower() for t in tells) for p in PR_THREATS)


def test_stats(store):
    c = Coach(store, load_settings())
    now = time.time()
    for i in range(3):
        c.record_hit(ts=now - 3 * H + i * H)
    st = c.stats(now=now)
    assert st["total_puffs"] == 3 and st["total_sessions"] == 3 and st["daily"][-1]["puffs"] >= 1


def test_record_puffs_returns_session_nudge(store):
    c = Coach(store, load_settings())
    out = c.record_puffs(3, ts=1_000_000.0)
    assert out["puffs"] == 3 and out["intervention"]["kind"] == "observe"
    assert len(store.events()) == 3 and len(store.sessions()) == 1


def test_first_session_without_baseline_is_not_praised_as_na(tmp_path):
    from vapecoach.config import load_settings
    from vapecoach.coach import Coach
    from vapecoach.storage import Store
    c = Coach(Store(tmp_path / "d.db"), load_settings(lambda k=None, d=None: d, baseline_hours=0))
    r = c.record_hit(ts=1_000_000.0)
    assert "n/a" not in r["intervention"]["text"]
    assert r["intervention"]["kind"] == "observe"
