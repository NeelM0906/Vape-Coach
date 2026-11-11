"""Everything that touches the Hermes ``ctx``: tools, hooks, the /vape command, the CLI, the skill.

Nag-only: no hook here returns ``{"action": "block"}``, no ``pre_tool_call`` is registered, and no
tool writes files or runs git. Sarcastic PR threats are text, nothing more.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from . import PLUGIN_NAME
from .cli import build_parser, format_stats, format_status, make_handler
from .coach import Coach
from .config import load_settings
from .interventions import MEMES, PR_THREATS, SHAKES
from .nudge import turn_context
from .storage import Store

log = logging.getLogger(__name__)

TOOLSET = "vape_coach"

STATUS_SCHEMA = {
    "name": "vape_status",
    "description": ("Read the user's vape-tapering status: phase (baseline/coaching), today's target gap "
                    "between vaping sessions, time until the next allowed hit, puffs today, on-time streak "
                    "and any owed exercise challenge. Read-only."),
    "parameters": {"type": "object", "properties": {}, "required": []},
}
LOG_SCHEMA = {
    "name": "vape_log_hit",
    "description": ("Log a vape hit the user tells you about (e.g. 'I just hit my vape', 'took 3 puffs 10 "
                    "minutes ago'). Returns the coach's verdict and the nudge to relay. Only call when the "
                    "user explicitly says they vaped."),
    "parameters": {"type": "object", "properties": {
        "puffs": {"type": "integer", "minimum": 1, "maximum": 20, "description": "number of puffs (default 1)"},
        "minutes_ago": {"type": "number", "minimum": 0, "maximum": 720, "description": "when (default now)"},
    }, "required": []},
}
STATS_SCHEMA = {
    "name": "vape_stats",
    "description": "Daily vape stats for the last N days (puffs, sessions, gaps), reps done, on-time rate. Read-only.",
    "parameters": {"type": "object", "properties": {
        "days": {"type": "integer", "minimum": 1, "maximum": 30}}, "required": []},
}

HELP = """/vape - vape-coach (nag-only; never blocks, never touches your code)
  /vape status        taper phase, target gap, time until next allowed hit
  /vape hit [n]       log an off-camera hit (n puffs)
  /vape stats         last 7 days
  /vape done          honor system: mark the owed challenge done (only when camera counting is off)
  /vape roast         a free roast, on the house
Camera: Start it from the Vape Coach pane in Hermes Desktop (or `hermes vape watch` in a terminal).
Owed jumping jacks are counted automatically while the camera is on: just step back and do them."""


class Glue:
    def __init__(self, ctx):
        self.ctx = ctx
        self._store = None
        self._turns = 0

    # Settings are re-read on every use so Settings-page edits apply without a reload.
    @property
    def settings(self):
        return load_settings(getattr(self.ctx, "get_config", None))

    @property
    def store(self) -> Store:
        if self._store is None:
            self._store = Store()
        return self._store

    def coach(self) -> Coach:
        return Coach(self.store, self.settings)

    # -- delivery -------------------------------------------------------------------
    def broadcast(self, event: str, payload: dict) -> None:
        """Desktop half (``plugin.vape-coach.<event>``); a logged no-op where no Desktop is attached."""
        try:
            from hermes_cli.plugin_events import broadcast_plugin_event  # public plugin API
            broadcast_plugin_event(PLUGIN_NAME, event, payload)
        except Exception as e:  # never let UI delivery break a turn
            log.debug("vape-coach broadcast failed: %s", e)

    def deliver_pending(self) -> int:
        """Post queued nudges into the bound conversation (post_message) when delivery allows it."""
        s = self.settings
        pending = [p for p in self.store.pending_interventions() if p["kind"] != "observe"]
        if not pending or s.delivery not in ("auto", "post"):
            return 0
        text = "\n".join(f"💨 vape-coach: {p['text']}" for p in pending[-3:])
        try:
            res = self.ctx.post_message(text)
            ok = bool(getattr(res, "ok", res))
        except Exception as e:
            log.debug("vape-coach post_message failed: %s", e)
            ok = False
        if ok:
            self.store.mark_delivered([p["id"] for p in pending], "post")
        return len(pending) if ok else 0

    # -- tools --------------------------------------------------------------------
    def tool_status(self, args: dict, **_kw) -> str:
        return json.dumps(self.coach().status(), default=str)

    def tool_log_hit(self, args: dict, **_kw) -> str:
        c = self.coach()
        puffs = max(1, min(20, int(args.get("puffs") or 1)))
        base = time.time() - float(args.get("minutes_ago") or 0) * 60
        out = c.record_puffs(puffs, ts=base)
        iv = out.get("intervention")
        if iv:
            self.store.mark_delivered([iv["id"]], "tool")
            self.broadcast("intervention", iv)
        return json.dumps({"logged_puffs": puffs, "nudge": iv, "challenge": out.get("challenge"),
                           "status": c.status()}, default=str)

    def tool_stats(self, args: dict, **_kw) -> str:
        return json.dumps(self.coach().stats(days=int(args.get("days") or 7)), default=str)

    # -- hooks ----------------------------------------------------------------------
    def on_pre_llm_call(self, **kw: Any):
        self._turns += 1
        try:
            ctx_text = turn_context(self.store, self.settings, self._turns)
        except Exception as e:
            log.debug("vape-coach pre_llm_call failed: %s", e)
            return None
        return {"context": ctx_text} if ctx_text else None  # context only; never {"action": "block"}

    def on_post_llm_call(self, **kw: Any) -> None:
        try:
            self.deliver_pending()
            self.broadcast("status", self.coach().status())
        except Exception as e:
            log.debug("vape-coach post_llm_call failed: %s", e)

    def on_session_start(self, **kw: Any) -> None:
        try:
            self.broadcast("status", self.coach().status())
        except Exception as e:
            log.debug("vape-coach on_session_start failed: %s", e)

    # -- /vape ------------------------------------------------------------------------
    def slash(self, raw_args: str = "", **_kw) -> str:
        parts = (raw_args or "").split()
        sub = parts[0].lower() if parts else "status"
        c = self.coach()
        if sub in ("help", "-h", "--help"):
            return HELP
        if sub == "status":
            return format_status(c.status())
        if sub == "stats":
            return format_stats(c.stats())
        if sub == "hit":
            n = max(1, min(int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1, 20))
            iv = c.record_puffs(n).get("intervention")
            if iv:
                self.store.mark_delivered([iv["id"]], "command")
                self.broadcast("intervention", iv)
            self.broadcast("status", c.status())
            return f"logged {n} puff(s)." + (f"\n{iv['text']}" if iv else "")
        if sub == "undo":
            return ("Hits can't be undone from chat. If the camera logged a false alarm, run "
                    "`hermes vape undo` in a terminal.")
        if sub == "done":
            out = c.honor_done()
            if out["ok"]:
                self.broadcast("status", c.status())
            return out["message"]
        if sub in ("roast", "meme"):
            import random
            return random.choice(MEMES + PR_THREATS + SHAKES)
        return HELP


def register(ctx) -> None:
    g = Glue(ctx)
    ctx.register_tool(name="vape_status", toolset=TOOLSET, schema=STATUS_SCHEMA, handler=g.tool_status, emoji="💨")
    ctx.register_tool(name="vape_log_hit", toolset=TOOLSET, schema=LOG_SCHEMA, handler=g.tool_log_hit, emoji="💨")
    ctx.register_tool(name="vape_stats", toolset=TOOLSET, schema=STATS_SCHEMA, handler=g.tool_stats, emoji="📉")
    ctx.register_hook("pre_llm_call", g.on_pre_llm_call)
    ctx.register_hook("post_llm_call", g.on_post_llm_call)
    ctx.register_hook("on_session_start", g.on_session_start)
    ctx.register_command("vape", g.slash, description="vape-coach: status, log a hit, stats, challenges",
                         args_hint="[status|hit|stats|done|roast]")
    ctx.register_cli_command(name="vape", help="vape-coach: webcam vape tracking + taper (local only)",
                             setup_fn=build_parser, handler_fn=make_handler(getattr(ctx, "get_config", None)),
                             description="Local-first webcam vape tracker and taper coach. "
                                         "Start with: hermes vape watch (or the Vape Coach pane in Desktop)")
    skill = Path(__file__).resolve().parent.parent / "skills" / "vape-coach" / "SKILL.md"
    if skill.is_file():
        try:
            ctx.register_skill("vape-coach", skill, description="Persona + rules for the vape-coach plugin")
        except Exception as e:
            log.debug("vape-coach skill registration failed: %s", e)
