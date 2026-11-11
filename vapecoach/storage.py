"""Durable event log in ``<hermes home>/plugin-data/vape-coach/data.db`` (SQLite, WAL).

Shared by two processes: ``hermes vape watch`` (writer: camera events, reps) and the
in-session plugin (reader + writer: manual hits, interventions, challenges). Only numbers and
labels are stored -- never frames, crops, or landmarks of the user's face.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from . import PLUGIN_NAME

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    source TEXT NOT NULL,            -- camera | manual | replay
    confidence REAL NOT NULL DEFAULT 1.0,
    dwell_s REAL,
    exhale REAL,
    session_id INTEGER,
    voided INTEGER NOT NULL DEFAULT 0,
    meta TEXT
);
CREATE INDEX IF NOT EXISTS events_ts ON events(ts);
CREATE TABLE IF NOT EXISTS interventions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    event_id INTEGER,
    kind TEXT NOT NULL,              -- exercise | pr_threat | protein_shake | meme | praise
    text TEXT NOT NULL,
    delivered INTEGER NOT NULL DEFAULT 0,
    delivered_via TEXT
);
CREATE TABLE IF NOT EXISTS challenges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    intervention_id INTEGER,
    exercise TEXT NOT NULL,
    target INTEGER NOT NULL,
    done INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'owed',   -- owed | done | waived
    verified TEXT,                          -- camera | honor | waived
    closed_ts REAL,
    blocked_turns INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS reps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    exercise TEXT NOT NULL,
    count INTEGER NOT NULL,
    challenge_id INTEGER,
    verified TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT NOT NULL);
"""


def default_db_path() -> Path:
    """Profile-scoped path via Hermes' public ``plugins.plugin_storage``; standalone fallback.

    ``VAPE_COACH_DB`` (set by the Desktop backend when it starts the camera watcher) pins the
    child process to exactly the database the backend reads.
    """
    pinned = os.environ.get("VAPE_COACH_DB")
    if pinned:
        return Path(pinned)
    try:
        from plugins.plugin_storage import plugin_data_dir  # public plugin API
        return plugin_data_dir(PLUGIN_NAME) / "data.db"
    except Exception:
        home = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
        root = home / "plugin-data" / PLUGIN_NAME
        root.mkdir(parents=True, exist_ok=True)
        return root / "data.db"


class Store:
    def __init__(self, path: Optional[os.PathLike | str] = None):
        self.path = Path(path) if path else default_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False, timeout=10)
        self._db.row_factory = sqlite3.Row
        try:
            self._db.execute("PRAGMA journal_mode=WAL")
        except sqlite3.DatabaseError:
            pass
        self._db.executescript(SCHEMA)
        self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # -- low level ---------------------------------------------------------
    def _exec(self, sql: str, args: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._db.execute(sql, tuple(args))
            self._db.commit()
            return cur

    def _all(self, sql: str, args: Iterable[Any] = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, tuple(args)).fetchall()]

    def _one(self, sql: str, args: Iterable[Any] = ()) -> Optional[dict]:
        rows = self._all(sql, args)
        return rows[0] if rows else None

    # -- kv ----------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        row = self._one("SELECT v FROM kv WHERE k=?", (key,))
        return json.loads(row["v"]) if row else default

    def put(self, key: str, value: Any) -> None:
        self._exec("INSERT INTO kv(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
                   (key, json.dumps(value)))

    # -- events --------------------------------------------------------------
    def add_event(self, ts: Optional[float] = None, source: str = "manual", confidence: float = 1.0,
                  dwell_s: Optional[float] = None, exhale: Optional[float] = None,
                  session_gap_s: float = 300.0, meta: Optional[dict] = None) -> dict:
        """Insert a hit; assigns a session id (puffs within ``session_gap_s`` share one)."""
        ts = time.time() if ts is None else float(ts)
        with self._lock:
            prev = self._one("SELECT * FROM events WHERE voided=0 AND ts<=? ORDER BY ts DESC LIMIT 1", (ts,))
            if prev and ts - prev["ts"] <= session_gap_s and prev["session_id"] is not None:
                session_id, new_session = prev["session_id"], False
            else:
                row = self._one("SELECT COALESCE(MAX(session_id),0)+1 AS n FROM events")
                session_id, new_session = row["n"], True
            cur = self._exec(
                "INSERT INTO events(ts,source,confidence,dwell_s,exhale,session_id,meta) VALUES(?,?,?,?,?,?,?)",
                (ts, source, confidence, dwell_s, exhale, session_id, json.dumps(meta or {})))
            ev = self._one("SELECT * FROM events WHERE id=?", (cur.lastrowid,))
        ev["new_session"] = new_session
        return ev

    def void_last_event(self) -> Optional[dict]:
        ev = self._one("SELECT * FROM events WHERE voided=0 ORDER BY ts DESC LIMIT 1")
        if ev:
            self._exec("UPDATE events SET voided=1 WHERE id=?", (ev["id"],))
        return ev

    def session_events(self, session_id: int, include_voided: bool = False) -> list[dict]:
        extra = "" if include_voided else " AND voided=0"
        return self._all(f"SELECT * FROM events WHERE session_id=?{extra} ORDER BY ts", (session_id,))

    def cancel_challenges_for_events(self, event_ids: Iterable[int]) -> int:
        """Cancel still-owed challenges created by nudges for these (voided) events."""
        ids = [int(i) for i in event_ids]
        if not ids:
            return 0
        marks = ",".join("?" * len(ids))
        cur = self._exec(
            f"UPDATE challenges SET status='cancelled', verified='undo', closed_ts=? WHERE status='owed' AND "
            f"intervention_id IN (SELECT id FROM interventions WHERE event_id IN ({marks}))",
            (time.time(), *ids))
        return cur.rowcount

    def events(self, since: float = 0.0, until: Optional[float] = None) -> list[dict]:
        until = time.time() + 1e9 if until is None else until
        return self._all("SELECT * FROM events WHERE voided=0 AND ts>=? AND ts<? ORDER BY ts", (since, until))

    def events_after_id(self, last_id: int) -> list[dict]:
        return self._all("SELECT * FROM events WHERE voided=0 AND id>? ORDER BY id", (last_id,))

    def sessions(self, since: float = 0.0) -> list[dict]:
        """One row per vaping session: start, end, puff count."""
        return self._all(
            "SELECT session_id, MIN(ts) AS start, MAX(ts) AS end, COUNT(*) AS puffs, "
            "GROUP_CONCAT(DISTINCT source) AS sources FROM events WHERE voided=0 AND ts>=? "
            "GROUP BY session_id ORDER BY start", (since,))

    # -- interventions ---------------------------------------------------------
    def add_intervention(self, kind: str, text: str, event_id: Optional[int] = None,
                         ts: Optional[float] = None) -> int:
        cur = self._exec("INSERT INTO interventions(ts,event_id,kind,text) VALUES(?,?,?,?)",
                         (time.time() if ts is None else ts, event_id, kind, text))
        return int(cur.lastrowid)

    def pending_interventions(self) -> list[dict]:
        return self._all("SELECT * FROM interventions WHERE delivered=0 ORDER BY ts")

    def mark_delivered(self, ids: Iterable[int], via: str) -> None:
        for i in ids:
            self._exec("UPDATE interventions SET delivered=1, delivered_via=? WHERE id=?", (via, i))

    def interventions(self, since: float = 0.0) -> list[dict]:
        return self._all("SELECT * FROM interventions WHERE ts>=? ORDER BY ts", (since,))

    # -- challenges ------------------------------------------------------------
    def add_challenge(self, exercise: str, target: int, intervention_id: Optional[int] = None,
                      ts: Optional[float] = None) -> int:
        cur = self._exec("INSERT INTO challenges(ts,intervention_id,exercise,target) VALUES(?,?,?,?)",
                         (time.time() if ts is None else ts, intervention_id, exercise, int(target)))
        return int(cur.lastrowid)

    def open_challenge(self) -> Optional[dict]:
        return self._one("SELECT * FROM challenges WHERE status='owed' ORDER BY ts LIMIT 1")

    def open_challenges(self) -> list[dict]:
        return self._all("SELECT * FROM challenges WHERE status='owed' ORDER BY ts")

    def credit_reps(self, exercise: str, count: int, verified: str, ts: Optional[float] = None) -> Optional[dict]:
        """Log reps and apply them to the oldest owed challenge for that exercise.

        Overflow carries to the next owed challenge. Returns the (updated) first challenge touched.
        """
        ts = time.time() if ts is None else ts
        remaining, touched = int(count), None
        with self._lock:
            for ch in self.open_challenges():
                if ch["exercise"] != exercise or remaining <= 0:
                    continue
                need = ch["target"] - ch["done"]
                use = min(need, remaining)
                remaining -= use
                done = ch["done"] + use
                status = "done" if done >= ch["target"] else "owed"
                self._exec("UPDATE challenges SET done=?, status=?, verified=?, closed_ts=? WHERE id=?",
                           (done, status, verified, ts if status == "done" else None, ch["id"]))
                self._exec("INSERT INTO reps(ts,exercise,count,challenge_id,verified) VALUES(?,?,?,?,?)",
                           (ts, exercise, use, ch["id"], verified))
                touched = touched or self._one("SELECT * FROM challenges WHERE id=?", (ch["id"],))
            if remaining > 0:
                self._exec("INSERT INTO reps(ts,exercise,count,challenge_id,verified) VALUES(?,?,?,?,?)",
                           (ts, exercise, remaining, None, verified))
        return touched

    def waive_challenges(self) -> int:
        cur = self._exec("UPDATE challenges SET status='waived', verified='waived', closed_ts=? WHERE status='owed'",
                         (time.time(),))
        return cur.rowcount

    def bump_blocked(self, challenge_id: int) -> None:
        self._exec("UPDATE challenges SET blocked_turns=blocked_turns+1 WHERE id=?", (challenge_id,))

    def reps(self, since: float = 0.0) -> list[dict]:
        return self._all("SELECT * FROM reps WHERE ts>=? ORDER BY ts", (since,))

    def reset(self) -> None:
        with self._lock:
            for t in ("events", "interventions", "challenges", "reps", "kv"):
                self._db.execute(f"DELETE FROM {t}")
            self._db.commit()
