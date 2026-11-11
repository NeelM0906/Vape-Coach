"""Orchestration: a new hit -> session grouping -> taper verdict -> intervention -> storage.

Called by every producer of hits (camera watcher, `/vape hit`, the `vape_log_hit` tool), so the
behaviour is identical whatever surface the demo uses. Delivery is someone else's job: we only
queue an ``interventions`` row and the surface adapters (Desktop poll/broadcast, post_message,
pre_llm_call context) pick it up.
"""

from __future__ import annotations

import random
import time
from typing import Optional

from .config import Settings
from .interventions import EXERCISE_LABEL, fmt_dur, pick
from .storage import Store
from .taper import compute_state, daily_stats, judge_new_session, on_time_streak

MAX_VERDICTS = 500


class Coach:
    def __init__(self, store: Store, settings: Optional[Settings] = None,
                 rng: Optional[random.Random] = None):
        self.store = store
        self.s = settings or Settings()
        self.rng = rng or random.Random(self.s.seed)

    # -- producers ---------------------------------------------------------------
    def record_hit(self, ts: Optional[float] = None, source: str = "manual", confidence: float = 1.0,
                   dwell_s: Optional[float] = None, exhale: Optional[float] = None,
                   meta: Optional[dict] = None) -> dict:
        ts = time.time() if ts is None else ts
        started = self.store.get("started_ts")
        if started is None:
            self.store.put("started_ts", ts)
            started = ts
        before = self.store.sessions()
        ev = self.store.add_event(ts=ts, source=source, confidence=confidence, dwell_s=dwell_s,
                                  exhale=exhale, session_gap_s=self.s.session_gap_s, meta=meta)
        out = {"event": ev, "new_session": ev["new_session"], "intervention": None, "challenge": None}
        if not ev["new_session"]:
            return out  # another puff in the same session: counted, not judged twice
        verdict = judge_new_session(before, ts, self.s, started_ts=started)
        verdicts = self.store.get("verdicts", [])
        early_today = sum(1 for v in verdicts if not v.get("on_time") and v.get("ts", 0) >= _local_midnight(ts))
        st = compute_state(before, now=ts, s=self.s, started_ts=started)
        p = pick(verdict, self.s, early_today=early_today, streak=on_time_streak(verdicts),
                 baseline_sessions=st.baseline_sessions + 1, rng=self.rng)
        verdicts.append({**verdict.as_dict(), "ts": ts, "event_id": ev["id"], "kind": p.kind})
        self.store.put("verdicts", verdicts[-MAX_VERDICTS:])
        iid = self.store.add_intervention(p.kind, p.text, event_id=ev["id"], ts=ts)
        out["intervention"] = {"id": iid, "kind": p.kind, "text": p.text, "level": p.level}
        out["verdict"] = verdict.as_dict()
        if p.kind == "exercise":
            cid = self.store.add_challenge(p.exercise, p.reps, intervention_id=iid, ts=ts)
            out["challenge"] = {"id": cid, "exercise": p.exercise, "target": p.reps}
        return out

    def record_puffs(self, n: int = 1, ts: Optional[float] = None, source: str = "manual") -> dict:
        """Log ``n`` puffs one second apart; returns the session's verdict/nudge (from the first puff)."""
        ts = time.time() if ts is None else ts
        merged: dict = {"intervention": None, "challenge": None, "verdict": None, "puffs": 0}
        for i in range(max(1, n)):
            out = self.record_hit(ts=ts + i, source=source)
            merged["puffs"] += 1
            for k in ("intervention", "challenge", "verdict"):
                merged[k] = merged[k] or out.get(k)
        return merged

    def credit_reps(self, exercise: str, count: int, verified: str) -> dict:
        ch = self.store.credit_reps(exercise, count, verified)
        left = self.store.open_challenges()
        msg = None
        if ch and ch["status"] == "done":
            msg = (f"Challenge cleared: {ch['target']} {EXERCISE_LABEL.get(exercise, exercise)} "
                   f"({verified}). Respect. Now stay off it for {fmt_dur(self.state().target_gap_s)}.")
            self.store.add_intervention("praise", msg)
        return {"challenge": ch, "open": left, "message": msg}

    def honor_done(self) -> dict:
        """Honor-system completion. Refused while camera verification is on: the camera counts."""
        ch = self.store.open_challenge()
        if not ch:
            return {"ok": False, "message": "no owed challenge."}
        if self.s.verify_reps_with_camera:
            return {"ok": False, "message": (f"{ch['target'] - ch['done']} {EXERCISE_LABEL.get(ch['exercise'], ch['exercise'])} "
                                             "still owed, and these are camera-counted: start the camera from the "
                                             "Vape Coach pane (or `hermes vape watch`), step back and do them.")}
        out = self.credit_reps(ch["exercise"], ch["target"] - ch["done"], "honor")
        return {"ok": True, "message": out["message"] or "credited."}

    def undo_last(self) -> Optional[dict]:
        """Void the most recent hit (a false positive). If that empties its session, the session's
        verdict and any challenge it created go too, so a false alarm leaves no penalty behind."""
        ev = self.store.void_last_event()
        if not ev:
            return None
        sid = ev["session_id"]
        if sid is not None and not self.store.session_events(sid):
            ids = {e["id"] for e in self.store.session_events(sid, include_voided=True)}
            self.store.cancel_challenges_for_events(ids)
            verdicts = [v for v in self.store.get("verdicts", []) if v.get("event_id") not in ids]
            self.store.put("verdicts", verdicts)
        return ev

    # -- readers -----------------------------------------------------------------
    def state(self, now: Optional[float] = None):
        return compute_state(self.store.sessions(), now=now, s=self.s, started_ts=self.store.get("started_ts"))

    def status(self, now: Optional[float] = None) -> dict:
        now = time.time() if now is None else now
        st = self.state(now)
        sessions = self.store.sessions()
        events = self.store.events()
        verdicts = self.store.get("verdicts", [])
        today = daily_stats(sessions, events, now=now, days=1)[0]
        open_ch = self.store.open_challenges()
        return {
            "phase": st.phase,
            "day_index": st.day_index,
            "baseline_gap": fmt_dur(st.baseline_gap_s),
            "target_gap": fmt_dur(st.target_gap_s),
            "target_gap_s": st.target_gap_s,
            "next_allowed_ts": st.next_allowed_ts,
            "seconds_until_allowed": round(st.seconds_until_allowed),
            "until_allowed": fmt_dur(st.seconds_until_allowed),
            "baseline_ends_ts": st.baseline_end_ts,
            "last_session_end": st.last_session_end,
            "since_last": fmt_dur(now - st.last_session_end) if st.last_session_end else None,
            "today": today,
            "streak": on_time_streak(verdicts),
            "open_challenges": open_ch,
            "watcher": self.store.get("watcher", {}),
            "live_reps": self.store.get("live_reps", {}),
        }

    def stats(self, days: int = 7, now: Optional[float] = None) -> dict:
        now = time.time() if now is None else now
        sessions, events = self.store.sessions(), self.store.events()
        reps = self.store.reps()
        verdicts = self.store.get("verdicts", [])
        coached = [v for v in verdicts if v.get("phase") == "coaching"]
        return {
            "daily": daily_stats(sessions, events, now=now, days=days),
            "total_puffs": len(events),
            "total_sessions": len(sessions),
            "camera_puffs": sum(1 for e in events if e["source"] == "camera"),
            "manual_puffs": sum(1 for e in events if e["source"] == "manual"),
            "on_time_rate": round(sum(1 for v in coached if v["on_time"]) / len(coached), 2) if coached else None,
            "reps": {ex: sum(r["count"] for r in reps if r["exercise"] == ex) for ex in {r["exercise"] for r in reps}},
            "streak": on_time_streak(verdicts),
        }


def _local_midnight(ts: float) -> float:
    lt = time.localtime(ts)
    return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))
