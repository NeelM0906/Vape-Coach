"""Taper engine: baseline -> target gap that grows ~10%/day -> verdict per new vaping session.

Pure functions over the session list from ``Store.sessions()`` so they are trivially testable.
A *session* is a cluster of puffs (``session_gap_s``); the taper stretches the gap between
the end of one session and the start of the next.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

from .config import Settings

DAY = 86400.0


@dataclass
class TaperState:
    phase: str                     # no_data | baseline | coaching
    started_ts: Optional[float]
    baseline_end_ts: Optional[float]
    baseline_gap_s: Optional[float]
    day_index: int                 # 0 during baseline, 1 = first coaching day
    target_gap_s: Optional[float]
    last_session_end: Optional[float]
    next_allowed_ts: Optional[float]
    seconds_until_allowed: float
    baseline_sessions: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Verdict:
    phase: str
    gap_s: Optional[float]         # gap since previous session end (None for the very first)
    target_gap_s: Optional[float]
    ratio: Optional[float]         # gap / target (>=1 means on time)
    on_time: bool
    early_by_s: float = 0.0
    severity: int = 0              # 0 on time / baseline, 1..3 how early

    def as_dict(self) -> dict:
        return asdict(self)


def baseline_gap(sessions: list[dict], start: float, end: float, s: Settings) -> tuple[float, int]:
    """Median end->start gap of sessions that began inside the baseline window."""
    inside = [x for x in sessions if start <= x["start"] < end]
    gaps = [b["start"] - a["end"] for a, b in zip(inside, inside[1:]) if b["start"] > a["end"]]
    if len(gaps) >= 2:
        g = statistics.median(gaps)
    else:
        g = s.fallback_baseline_gap_min * 60
    return max(g, s.min_target_gap_min * 60), len(inside)


def target_for_day(base_gap_s: float, day_index: int, s: Settings) -> float:
    t = base_gap_s * (1.0 + s.gap_growth_per_day) ** max(0, day_index)
    return min(t, s.max_target_gap_hours * 3600)


def compute_state(sessions: list[dict], now: Optional[float] = None, s: Optional[Settings] = None,
                  started_ts: Optional[float] = None) -> TaperState:
    s = s or Settings()
    now = time.time() if now is None else now
    if not sessions and started_ts is None:
        return TaperState("no_data", None, None, None, 0, None, None, None, 0.0)
    start = started_ts if started_ts is not None else sessions[0]["start"]
    b_end = start + s.baseline_hours * 3600
    last_end = sessions[-1]["end"] if sessions else None
    if now < b_end:
        return TaperState("baseline", start, b_end, None, 0, None, last_end, None, 0.0,
                          baseline_sessions=sum(1 for x in sessions if x["start"] < b_end))
    gap, n = baseline_gap(sessions, start, b_end, s)
    day = int(math.floor((now - b_end) / DAY)) + 1
    target = target_for_day(gap, day, s)
    nxt = (last_end + target) if last_end is not None else None
    wait = max(0.0, nxt - now) if nxt is not None else 0.0
    return TaperState("coaching", start, b_end, gap, day, target, last_end, nxt, wait, baseline_sessions=n)


def judge_new_session(sessions_before: list[dict], session_start: float, s: Optional[Settings] = None,
                      started_ts: Optional[float] = None) -> Verdict:
    """Was a session starting at ``session_start`` early relative to the target gap?"""
    s = s or Settings()
    st = compute_state(sessions_before, now=session_start, s=s, started_ts=started_ts)
    if st.phase != "coaching":
        gap = (session_start - st.last_session_end) if st.last_session_end else None
        return Verdict(st.phase, gap, None, None, True)
    if st.last_session_end is None:
        return Verdict("coaching", None, st.target_gap_s, None, True)
    gap = max(0.0, session_start - st.last_session_end)
    ratio = gap / st.target_gap_s if st.target_gap_s else 1.0
    if ratio >= 1.0:
        return Verdict("coaching", gap, st.target_gap_s, ratio, True)
    severity = 1 if ratio >= 0.75 else 2 if ratio >= 0.4 else 3
    return Verdict("coaching", gap, st.target_gap_s, ratio, False, st.target_gap_s - gap, severity)


def daily_stats(sessions: list[dict], events: list[dict], now: Optional[float] = None,
                days: int = 7, tz_offset_s: Optional[float] = None) -> list[dict]:
    """Per local-day counts: puffs, sessions, mean/longest gap (minutes)."""
    now = time.time() if now is None else now
    if tz_offset_s is None:
        tz_offset_s = -time.altzone if time.localtime(now).tm_isdst > 0 else -time.timezone

    def day_key(ts: float) -> int:
        return int((ts + tz_offset_s) // DAY)

    today = day_key(now)
    out = []
    for d in range(today - days + 1, today + 1):
        ss = [x for x in sessions if day_key(x["start"]) == d]
        ev = [e for e in events if day_key(e["ts"]) == d]
        gaps = [(b["start"] - a["end"]) / 60 for a, b in zip(ss, ss[1:])]
        out.append({
            "day_offset": d - today,
            "puffs": len(ev),
            "sessions": len(ss),
            "mean_gap_min": round(statistics.mean(gaps), 1) if gaps else None,
            "longest_gap_min": round(max(gaps), 1) if gaps else None,
        })
    return out


def on_time_streak(verdicts: list[dict]) -> int:
    n = 0
    for v in reversed(verdicts):
        if v.get("phase") != "coaching":
            continue
        if not v.get("on_time"):
            break
        n += 1
    return n
