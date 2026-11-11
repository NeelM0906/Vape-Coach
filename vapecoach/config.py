"""Settings: defaults + a loader that reads ``plugins.entries.vape-coach.settings`` via ctx.

Every surface/OS/exercise/gating choice the user has not made yet lives here, so changing
the demo surface or the gate strength is a config edit, never a code change.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Callable, Optional

GATE_MODES = ("nag", "off")  # nag-only by design: vape-coach never blocks a request
DELIVERY_MODES = ("auto", "desktop", "post", "context", "none")
EXERCISES = ("jumping_jacks", "squats", "pushups")
CAMERA_VERIFIED = ("jumping_jacks", "squats", "pushups")
INTERVENTIONS = ("exercise", "pr_threat", "protein_shake", "meme")


@dataclass
class Settings:
    # taper
    baseline_hours: float = 24.0            # observe-only window after the first logged hit
    gap_growth_per_day: float = 0.10        # target gap grows ~10%/day after the baseline
    fallback_baseline_gap_min: float = 30.0  # used when the baseline has < 3 sessions
    min_target_gap_min: float = 10.0
    max_target_gap_hours: float = 12.0
    session_gap_s: float = 300.0            # puffs closer than this are one vaping session
    # interventions
    interventions: list = field(default_factory=lambda: list(INTERVENTIONS))
    exercise: str = "jumping_jacks"    # camera-counted; squats is the alt
    reps_base: int = 15
    reps_step: int = 10
    reps_max: int = 60
    verify_reps_with_camera: bool = True     # False = honor system (`/vape done`)
    # gating
    gate_mode: str = "nag"                  # nag = owed challenges ride along as context; never blocks
    nag_every_turns: int = 1                # remind the agent every N turns while a challenge is owed
    # delivery of unprompted nudges (surface not decided yet)
    delivery: str = "auto"
    # detector
    camera_index: int = 0
    camera_backend: str = "auto"            # auto | dshow | msmf | avfoundation | v4l2 | any
    idle_fps: float = 2.0
    active_fps: float = 12.0
    model_variant: str = "lite"             # pose model: lite | full
    seed: Optional[int] = None              # deterministic intervention picks (tests/demo)

    def validate(self) -> "Settings":
        if self.gate_mode not in GATE_MODES:
            self.gate_mode = "nag"
        if self.delivery not in DELIVERY_MODES:
            self.delivery = "auto"
        if self.exercise not in EXERCISES:
            self.exercise = "jumping_jacks"
        self.interventions = [i for i in (self.interventions or []) if i in INTERVENTIONS] or ["meme"]
        self.gap_growth_per_day = max(0.0, min(float(self.gap_growth_per_day), 1.0))
        self.idle_fps = max(0.5, min(float(self.idle_fps), 10.0))
        self.active_fps = max(self.idle_fps, min(float(self.active_fps), 30.0))
        return self

    def as_dict(self) -> dict:
        return asdict(self)


def load_settings(get_config: Optional[Callable[[str, Any], Any]] = None, **overrides) -> Settings:
    """Build Settings from ``ctx.get_config`` (or nothing, in tests/standalone) plus overrides."""
    s = Settings()
    for f in fields(Settings):
        val = None
        if get_config is not None:
            try:
                val = get_config(f.name, None)
            except Exception:
                val = None
        if val is not None:
            setattr(s, f.name, _coerce(val, getattr(s, f.name)))
    for k, v in overrides.items():
        if hasattr(s, k) and v is not None:
            setattr(s, k, v)
    return s.validate()


def _coerce(val: Any, default: Any) -> Any:
    try:
        if isinstance(default, bool):
            return val if isinstance(val, bool) else str(val).lower() in ("1", "true", "yes", "on")
        if isinstance(default, int):
            return int(val)
        if isinstance(default, float):
            return float(val)
        if isinstance(default, list):
            return list(val) if isinstance(val, (list, tuple)) else [x.strip() for x in str(val).split(",") if x.strip()]
    except (TypeError, ValueError):
        return default
    return val
