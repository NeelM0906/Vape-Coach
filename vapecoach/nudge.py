"""Nag-only nudges: what the agent hears (pre_llm_call context) and what surfaces display.

vape-coach never blocks a request or a tool call. While a challenge is owed, or an undelivered
roast is waiting, the next turn carries a short context note so the agent can tease the user in
character; the user's actual request is always answered.
"""

from __future__ import annotations

from typing import Optional

from .config import Settings
from .interventions import EXERCISE_LABEL
from .storage import Store

CONTEXT_HEADER = "[vape-coach]"


def turn_context(store: Store, s: Settings, turn_no: int) -> Optional[str]:
    """Context string for ``pre_llm_call`` (or None). Marks queued roasts as delivered via context."""
    if s.gate_mode == "off":
        return None
    lines: list[str] = []
    pending = [p for p in store.pending_interventions() if p["kind"] != "observe"]
    if pending and s.delivery in ("auto", "context"):
        newest = pending[-1]
        lines.append(f"The user just vaped. Queued nudge (tone level by kind={newest['kind']}): \"{newest['text']}\"")
        store.mark_delivered([p["id"] for p in pending], "context")
    owed = store.open_challenges()
    if owed and (turn_no % max(1, s.nag_every_turns) == 0):
        c = owed[0]
        lines.append(f"They still owe {c['target'] - c['done']} {EXERCISE_LABEL.get(c['exercise'], c['exercise'])} "
                     f"(challenge #{c['id']}).")
    if not lines:
        return None
    lines.append("Answer their request fully first, then add ONE short playful line about it. "
                 "Never refuse, delay, or degrade the work; never modify code as a 'punishment'.")
    return f"{CONTEXT_HEADER} " + " ".join(lines)
