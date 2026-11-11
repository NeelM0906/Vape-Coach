"""Intervention picker + text banks. Text only (no bundled images).

Escalation climbs a fixed ladder (roast -> sarcastic PR threat -> protein shake -> camera-counted
exercise), one rung per early session today, plus one rung when a session is very early. Everything is a nudge: nothing here can block a request or touch code.
PR "threats" are sarcasm and always carry an explicit tell; the plugin ships no tool that writes
to a repo, so the joke cannot become real.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from .config import Settings
from .taper import Verdict

EXERCISE_LABEL = {"jumping_jacks": "jumping jacks", "squats": "squats", "pushups": "push-ups"}

MEMES = [
    "Breaking: local dev outsources thinking to an AI, keeps the lung damage in-house.",
    "Vaping while the agent codes is like paying a plumber and then licking the pipes.",
    "That cloud you just blew has more mass than your last commit.",
    "Nobody: ... Absolutely nobody: ... You: *mango ice exhale at 2pm standup*",
    "Achievement unlocked: 'Fog Machine (Unpaid Intern)'.",
    "Your vape has a battery indicator. Your lungs don't. Think about it.",
    "POV: you're the only thing in this repo still running on legacy nicotine.",
    "Real 10x devs: ship features. You: ship flavor pods.",
    "The agent is compiling. You are... recompiling your dopamine. Badly.",
    "If puffs were story points you'd be a principal engineer by now.",
]

PR_THREATS = [
    "One more hit and I rename every variable in your PR to `vape_cloud_{n}`. (Kidding. I can't touch your code.)",
    "Hit that again and your PR description becomes a 4,000-word essay on mango ice. /s - your code is safe.",
    "Keep puffing and I'll convert your whole diff to tabs. Mixed tabs. Joking: I have zero write access.",
    "Another cloud and I tell the reviewer you hand-wrote that regex. (Empty threat. Your repo is untouched.)",
    "I could add `// written under the influence of blue razz` to every function. I won't. I literally can't. But I could.",
    "Your PR will be approved... the moment your lungs are. (Sarcasm - nothing in your repo changes.)",
]

SHAKES = [
    "Protein shake. Now. Replace the hand-to-mouth habit with 25g of something useful.",
    "Mandatory hydration + protein break: shake in hand before the agent finishes this task.",
    "Your hands want something to do. Give them a shaker bottle. Chug, don't puff.",
    "Swap the pod for a protein shake. Gains > clouds.",
]

PRAISE = [
    "On-time! {gap} since the last session (target {target}). The taper is working.",
    "Gap held: {gap} vs target {target}. Your lungs filed a thank-you ticket.",
    "{gap} clean. That beats the target ({target}). Streak: {streak}.",
]

OBSERVE = [
    "Logged. Baseline day: I'm just watching for now ({n} sessions so far).",
]


@dataclass
class Pick:
    kind: str                      # exercise | pr_threat | protein_shake | meme | praise | observe
    text: str
    level: int                     # 0 praise/observe, 1 mild, 2 medium, 3 spicy
    exercise: Optional[str] = None
    reps: int = 0


def fmt_dur(seconds: Optional[float]) -> str:
    if seconds is None:
        return "n/a"
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, _ = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m"


# The escalation ladder, mildest first. Each early session today climbs one rung; a session that
# started at < 40% of the target gap (severity 3) climbs one extra.
LADDER = ("meme", "pr_threat", "protein_shake", "exercise")


def step_for(verdict: Verdict, early_today: int) -> int:
    """0-based rung on ``LADDER`` (may run past the top: extra rungs add reps)."""
    return max(0, early_today) + (1 if verdict.severity >= 3 else 0)


def level_for(verdict: Verdict, early_today: int) -> int:
    """0 on time; 1..4 = meme, PR threat, protein shake, exercise."""
    if verdict.on_time:
        return 0
    return min(len(LADDER), 1 + step_for(verdict, early_today))


def reps_for(step: int, s: Settings) -> int:
    """Base reps on the first exercise rung, then ``reps_step`` more per rung above it."""
    return int(min(s.reps_max, s.reps_base + s.reps_step * max(0, step - (len(LADDER) - 1))))


def kind_for(level: int, enabled: list) -> str:
    """The ladder rung's kind, or the nearest enabled one (milder first, then harsher)."""
    i = level - 1
    order = list(LADDER[i::-1]) + list(LADDER[i + 1:])
    return next((k for k in order if k in enabled), "meme")


def exercise_how(s: Settings) -> str:
    if s.verify_reps_with_camera:
        return ("Step back so your head and arms are in frame: the camera counts them automatically "
                "(start it from the Vape Coach pane if it's off).")
    return "Honor system: hit Done in the Vape Coach pane (or /vape done) when finished."


def pick(verdict: Verdict, s: Settings, *, early_today: int = 0, streak: int = 0,
         baseline_sessions: int = 0, rng: Optional[random.Random] = None) -> Pick:
    rng = rng or random.Random(s.seed)
    if verdict.phase != "coaching":
        return Pick("observe", rng.choice(OBSERVE).format(n=baseline_sessions), 0)
    if verdict.on_time and verdict.gap_s is None:
        return Pick("observe", f"First session logged. Next target gap: {fmt_dur(verdict.target_gap_s)}.", 0)
    if verdict.on_time:
        return Pick("praise", rng.choice(PRAISE).format(gap=fmt_dur(verdict.gap_s),
                                                        target=fmt_dur(verdict.target_gap_s),
                                                        streak=streak + 1), 0)
    level = level_for(verdict, early_today)
    kind = kind_for(level, list(s.interventions))
    early = f"{fmt_dur(verdict.early_by_s)} early (target gap {fmt_dur(verdict.target_gap_s)})"
    if kind == "exercise":
        reps = reps_for(step_for(verdict, early_today), s)
        text = f"{early}. Penalty: {reps} {EXERCISE_LABEL[s.exercise]}. {exercise_how(s)}"
        return Pick("exercise", text, level, s.exercise, reps)
    bank = {"meme": MEMES, "pr_threat": PR_THREATS, "protein_shake": SHAKES}[kind]
    return Pick(kind, f"{rng.choice(bank)} [{early}]", level)
