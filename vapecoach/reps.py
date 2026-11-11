"""Camera-checked rep counters from Pose landmarks (joint angles + hysteresis).

* jumping_jacks (default): "open" = both arms raised (shoulder abduction angle hip-shoulder-wrist
  > open_deg, wrists above shoulders) and, when ankles are visible, feet apart (> 1.8x hip width);
  "closed" = arms down (< closed_deg) and feet together. One rep = closed -> open -> closed.
  Arms-only mode kicks in automatically when the legs are out of frame (desk webcam), so the user
  only has to step back far enough to show their arms and head.
* squats: knee angle hip-knee-ankle < 100 = down, > 160 = up.
* pushups: elbow angle shoulder-elbow-wrist < 90 = down, > 155 = up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .geometry import (L_ANK, L_EL, L_HIP, L_KNEE, L_SH, L_WR, R_ANK, R_EL, R_HIP, R_KNEE, R_SH,
                       R_WR, angle, dist, vis)

MIN_VIS = 0.5


@dataclass
class RepState:
    exercise: str
    count: int = 0
    phase: str = "unknown"            # open/closed (jacks) or up/down
    mode: str = ""                    # jacks: full | arms_only
    last_change_t: float = -1e9
    metric: float = 0.0               # last angle used (debug overlay)


class RepCounter:
    def __init__(self, exercise: str = "jumping_jacks", min_phase_s: float = 0.12):
        if exercise not in ("jumping_jacks", "squats", "pushups"):
            raise ValueError(f"unsupported exercise: {exercise}")
        self.st = RepState(exercise)
        self.min_phase_s = min_phase_s
        self._seen_open = False

    @property
    def count(self) -> int:
        return self.st.count

    def update(self, pose: Optional[Sequence[Sequence[float]]], t: float) -> bool:
        """Feed one pose frame; returns True when a rep completes on this frame."""
        if pose is None or len(pose) < 29:
            return False
        fn = {"jumping_jacks": self._jacks, "squats": self._squats, "pushups": self._pushups}[self.st.exercise]
        new_phase = fn(pose)
        if new_phase is None or new_phase == self.st.phase:
            return False
        if t - self.st.last_change_t < self.min_phase_s:
            return False
        prev, self.st.phase, self.st.last_change_t = self.st.phase, new_phase, t
        top = {"jumping_jacks": "open", "squats": "down", "pushups": "down"}[self.st.exercise]
        rest = {"jumping_jacks": "closed", "squats": "up", "pushups": "up"}[self.st.exercise]
        if new_phase == top:
            self._seen_open = True
        elif new_phase == rest and prev == top and self._seen_open:
            self._seen_open = False
            self.st.count += 1
            return True
        return False

    # -- per exercise ------------------------------------------------------------
    def _jacks(self, p) -> Optional[str]:
        if min(vis(p[L_SH]), vis(p[R_SH]), vis(p[L_WR]), vis(p[R_WR])) < MIN_VIS:
            return None
        abd = min(angle(p[L_HIP], p[L_SH], p[L_WR]), angle(p[R_HIP], p[R_SH], p[R_WR]))
        abd_max = max(angle(p[L_HIP], p[L_SH], p[L_WR]), angle(p[R_HIP], p[R_SH], p[R_WR]))
        wrists_up = p[L_WR][1] < p[L_SH][1] and p[R_WR][1] < p[R_SH][1]
        self.st.metric = abd
        legs = min(vis(p[L_ANK]), vis(p[R_ANK]), vis(p[L_HIP]), vis(p[R_HIP])) >= MIN_VIS
        self.st.mode = "full" if legs else "arms_only"
        if legs:
            hip_w = max(dist(p[L_HIP], p[R_HIP]), 1e-3)
            spread = dist(p[L_ANK], p[R_ANK]) / hip_w
            if abd > 120 and wrists_up and spread > 1.8:
                return "open"
            if abd_max < 50 and spread < 1.5:
                return "closed"
            return None
        if abd > 125 and wrists_up:
            return "open"
        if abd_max < 45:
            return "closed"
        return None

    def _squats(self, p) -> Optional[str]:
        side = [(L_HIP, L_KNEE, L_ANK), (R_HIP, R_KNEE, R_ANK)]
        side = [s for s in side if min(vis(p[i]) for i in s) >= MIN_VIS]
        if not side:
            return None
        a = min(angle(p[h], p[k], p[an]) for h, k, an in side)
        self.st.metric = a
        return "down" if a < 100 else "up" if a > 160 else None

    def _pushups(self, p) -> Optional[str]:
        side = [(L_SH, L_EL, L_WR), (R_SH, R_EL, R_WR)]
        side = [s for s in side if min(vis(p[i]) for i in s) >= MIN_VIS]
        if not side:
            return None
        a = min(angle(p[s], p[e], p[w]) for s, e, w in side)
        self.st.metric = a
        return "down" if a < 90 else "up" if a > 155 else None
