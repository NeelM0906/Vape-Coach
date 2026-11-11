"""Hand-to-mouth puff detector: per-frame observations -> APPROACH -> DWELL -> RELEASE -> EXHALE -> event.

Pure Python over ``Obs`` (no OpenCV/MediaPipe) so it is unit-tested with synthetic sequences and
reused unchanged for live camera, ``--replay`` and evaluation. Timestamps drive everything, so a
variable frame rate (2 fps idle, 12 fps active) is fine.

Distances are in *face widths* (cheek-to-cheek); pose-only frames approximate a face width as
half the shoulder width. Scoring rewards a short dwell (0.4-6 s), a pinch grip (thin device) and an
exhale cue after release (mouthPucker / mouthFunnel / cheekPuff blendshapes, i.e. the "O" mouth
of blowing out; pursed lips on the device during the dwell count at 0.8 weight). Look-alikes: a coffee sip has an open grip and no exhale; hand-on-chin dwells
too long; scratching is too short; eating shows a wide jaw instead of a pucker.

Three physical gates reject common look-alikes:

- *Time at the lips*: a puff keeps the hand within ~0.55 face widths of the lips for >= 0.3 s in
  total; a quick sip or a touch on the way past does not. (A tighter 0.25 fit pod-style devices
  but lost a stock box-mod clip where the hand sits ~0.55 away, so it was relaxed.)
- *Face visible*: a small device leaves the face trackable; a bottle or mug hides it.
- *Head level*: drinking tilts the head back (nose rises above the ear line).

A long hand-near-mouth episode (fingers resting at the lips, then the actual hit) is split into
segments at every soft release, and the best valid segment is scored instead of rejecting it all.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class Obs:
    t: float                          # seconds (video time or wall clock)
    hand_mouth: Optional[float]       # min hand->mouth distance in face widths; None = no hand/face
    grip: Optional[float] = None      # thumb-index tip distance / hand size (small = pinch)
    exhale: Optional[float] = None    # 0..1 blow-out cue from face blendshapes
    jaw_open: Optional[float] = None  # 0..1
    source: str = "face_hand"         # pose | face_hand
    hint: Optional[float] = None      # low-confidence Pose hand->mouth distance, only used to wake the models
    pitch: Optional[float] = None     # head tilt back: (ear line - nose) height in face widths; drinking >> 0


@dataclass
class FSMConfig:
    approach: float = 2.0             # face widths: hand is coming toward the face
    contact: float = 0.6              # face widths: hand (fingertips/device) at the lips
    release: float = 0.85             # face widths: hand has left the lips (hysteresis vs contact)
    deep: float = 0.55                # face widths: hand really at the lips (0.25 fit pods but lost a box-mod grip at ~0.55)
    min_deep_s: float = 0.3           # a puff needs this much time actually at the lips (a quick sip rarely gets it)
    min_face_frac: float = 0.7        # share of contact frames with a tracked face (bottle/mug hides it)
    max_pitch_up: float = 0.2         # head tilt back vs the pre-contact baseline, face widths (drinking)
    min_dwell_s: float = 0.35
    max_dwell_s: float = 7.0          # longer = chin rest / phone call, not a puff
    exhale_window_s: float = 5.0      # look for the exhale this long after release
    exhale_floor: float = 0.18        # blendshape level that counts as zero evidence
    exhale_span: float = 0.32         # floor + span = full evidence
    # Pursed lips while AT the lips count as exhale evidence at this weight (stealth hits show no cloud).
    # A mug rim purses them too: 0.0 + score_threshold 0.65 removed 2/2 synthetic mug-sip false alarms
    # but cost 1/2 synthetic stealth hits, with real and stock clips unchanged, so the default stays 0.8.
    dwell_pucker_w: float = 0.8
    eat_jaw: float = 0.55             # jawOpen above this during dwell = chewing/eating
    score_threshold: float = 0.6      # live desk test: real hits scored >= 0.889, some false alarms 0.57-0.58
    max_grip: float = 0.6             # thumb-index gap / hand size above this = open hand, not holding a
                                      # device. Live test: real hits <= 0.49, most low-score false alarms 0.57-0.91
    refractory_s: float = 8.0         # a contact this soon after the last puff's release is the same puff
                                      # (live test: one hit was logged twice 7 s apart at 2.5 s)
    escalate: float = 1.6             # face widths: a hand this close wakes the high-rate models
    approach_hold_s: float = 3.0      # how long a hovering hand may hold the high rate
    lost_s: float = 1.2               # hand/face missing this long in DWELL = abort
    w_dwell: float = 0.35
    w_exhale: float = 0.45
    w_grip: float = 0.20


@dataclass
class PuffEvent:
    t_start: float                    # contact start
    t_end: float                      # end of exhale evidence window actually used
    dwell_s: float
    exhale: float                     # peak exhale evidence 0..1
    grip: Optional[float]
    score: float
    reason: str = ""
    deep_s: float = 0.0
    face_frac: float = 1.0
    pitch_up: Optional[float] = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Seg:
    """Stats of one continuous stretch with the hand at the lips (d < contact)."""
    t0: float
    t1: float
    deep_s: float = 0.0
    n: int = 0
    n_face: int = 0
    grip_min: Optional[float] = None
    jaw_max: float = 0.0
    exhale: float = 0.0
    pitch_max: Optional[float] = None
    d_min: float = 9.0

    @property
    def length(self) -> float:
        return self.t1 - self.t0

    def merge(self, o: "_Seg") -> "_Seg":
        mn = lambda a, b: b if a is None else a if b is None else min(a, b)
        mx = lambda a, b: b if a is None else a if b is None else max(a, b)
        return _Seg(min(self.t0, o.t0), max(self.t1, o.t1), self.deep_s + o.deep_s, self.n + o.n,
                    self.n_face + o.n_face, mn(self.grip_min, o.grip_min), max(self.jaw_max, o.jaw_max),
                    max(self.exhale, o.exhale), mx(self.pitch_max, o.pitch_max), min(self.d_min, o.d_min))


class HandToMouthFSM:
    IDLE, APPROACH, DWELL, EXHALE = "idle", "approach", "dwell", "exhale"

    def __init__(self, cfg: Optional[FSMConfig] = None):
        self.cfg = cfg or FSMConfig()
        self.reset()

    def reset(self) -> None:
        self.state = self.IDLE
        self.t_contact = 0.0
        self.t_release = 0.0
        self.t_last_seen = 0.0
        self.dwell_s = 0.0
        self.segs: list[_Seg] = []
        self._in_contact = False
        self._t_prev: Optional[float] = None
        self.post_exhale = 0.0
        self.pitch_base: Optional[float] = None
        self._pitch_hist: deque = deque()
        self.last_emit_t = -1e9
        self.last_release_t = -1e9
        self.rejected: list[dict] = []   # look-alikes we saw and declined (for --debug / eval)
        self.t_approach = 0.0
        self._t_now = 0.0

    # Back-compat views used by the overlay / older callers.
    @property
    def exhale_peak(self) -> float:
        return max([s.exhale for s in self.segs] + [self.post_exhale])

    @property
    def wants_high_rate(self) -> bool:
        """True while a candidate is in flight: the pipeline escalates to Face+Hand at active fps.

        A hand merely hovering in APPROACH range (resting on the chin, typing close to the camera)
        only holds the high rate for ``approach_hold_s``; DWELL/EXHALE always hold it.
        """
        if self.state in (self.DWELL, self.EXHALE):
            return True
        return self.state == self.APPROACH and self._t_now - self.t_approach < self.cfg.approach_hold_s

    def _exhale_evidence(self, e: Optional[float]) -> float:
        if e is None:
            return 0.0
        c = self.cfg
        return max(0.0, min(1.0, (e - c.exhale_floor) / c.exhale_span))

    def _pick(self) -> tuple[_Seg, float]:
        """Whole dwell, or (if it ran too long) the best valid segment inside it."""
        whole = self.segs[0]
        for s in self.segs[1:]:
            whole = whole.merge(s)
        c = self.cfg
        if self.dwell_s <= c.max_dwell_s or len(self.segs) < 2:
            return whole, self.dwell_s
        ok = [s for s in self.segs if c.min_dwell_s <= s.length <= c.max_dwell_s and s.deep_s >= c.min_deep_s]
        if not ok:
            return whole, self.dwell_s
        best = max(ok, key=lambda s: (s.deep_s, s.t0))
        # Rescue only a segment that went clearly closer than the rest of the episode (device at the
        # lips after fingers resting near them); an even chin rest with wobbles stays rejected.
        others = min((s.d_min for s in self.segs if s is not best), default=9.0)
        if best.d_min > 0.3 or best.d_min > 0.6 * others:
            return whole, self.dwell_s
        return best, best.length

    def _score(self, seg: _Seg, dwell: float, exhale: float) -> tuple[float, str]:
        c = self.cfg
        if dwell < c.min_dwell_s:
            return 0.0, "dwell too short (scratch/touch)"
        if dwell > c.max_dwell_s:
            return 0.0, "dwell too long (chin rest/phone)"
        if seg.n >= 3 and seg.n_face / seg.n < c.min_face_frac:
            return 0.0, "face hidden at the lips (bottle/mug)"
        if self.pitch_base is not None and seg.pitch_max is not None and seg.pitch_max - self.pitch_base > c.max_pitch_up:
            return 0.0, "head tilted back (drinking)"
        if seg.deep_s < c.min_deep_s:
            return 0.0, "hand never at the lips (drink/gesture)"
        if seg.jaw_max >= c.eat_jaw and exhale < 0.5:
            return 0.0, "wide jaw (eating/talking)"
        grip_score = 0.0 if seg.grip_min is None else max(0.0, min(1.0, (0.9 - seg.grip_min) / 0.6))
        s = c.w_dwell + c.w_exhale * exhale + c.w_grip * grip_score
        if s >= c.score_threshold and seg.grip_min is not None and seg.grip_min > c.max_grip:
            return 0.0, "open hand at the lips (snack/scratch/phone, not a device)"
        return round(s, 3), "ok" if s >= c.score_threshold else "weak evidence (sip/touch)"

    def _finish(self, t: float) -> Optional[PuffEvent]:
        seg, dwell = self._pick()
        exhale = max(seg.exhale, self.post_exhale)
        score, reason = self._score(seg, dwell, exhale)
        ev = PuffEvent(round(seg.t0, 3), t, round(dwell, 2), round(exhale, 3),
                       None if seg.grip_min is None else round(seg.grip_min, 3), score, reason,
                       round(seg.deep_s, 2), round(seg.n_face / seg.n, 2) if seg.n else 1.0,
                       None if self.pitch_base is None or seg.pitch_max is None
                       else round(seg.pitch_max - self.pitch_base, 3))
        self.state = self.IDLE
        self.segs, self.post_exhale = [], 0.0
        if score >= self.cfg.score_threshold:
            # e.g. inhale, then pull the shirt up to the mouth to exhale into it
            if ev.t_start - self.last_release_t >= self.cfg.refractory_s and ev.t_start - self.last_emit_t >= self.cfg.refractory_s:
                self.last_emit_t = ev.t_start
                self.last_release_t = self.t_release if self.t_release >= ev.t_start else t
                return ev
            ev.reason, ev.score = "refractory (same puff)", 0.0
        self.rejected.append(ev.as_dict())
        del self.rejected[:-50]
        return None

    def _track_pitch(self, o: Obs) -> None:
        if o.pitch is None:
            return
        h = self._pitch_hist
        h.append((o.t, o.pitch))
        while h and o.t - h[0][0] > 6.0:
            h.popleft()

    def _baseline_pitch(self, t: float) -> Optional[float]:
        xs = sorted(p for (tt, p) in self._pitch_hist if t - 6.0 <= tt <= t - 0.3)
        return xs[len(xs) // 2] if len(xs) >= 2 else None

    def _accumulate(self, o: Obs, d: float) -> None:
        c = self.cfg
        dt = 0.0 if self._t_prev is None else max(0.0, min(0.5, o.t - self._t_prev))
        if d < c.contact:
            if not self._in_contact or not self.segs:
                self.segs.append(_Seg(o.t, o.t))
                self._in_contact = True
            s = self.segs[-1]
            s.t1 = o.t
            s.n += 1
            s.d_min = min(s.d_min, d)
            if o.source != "pose":
                s.n_face += 1
            if d < c.deep:
                s.deep_s += dt
            if o.grip is not None:
                s.grip_min = o.grip if s.grip_min is None else min(s.grip_min, o.grip)
            if o.jaw_open is not None:
                s.jaw_max = max(s.jaw_max, o.jaw_open)
            if o.pitch is not None:
                s.pitch_max = o.pitch if s.pitch_max is None else max(s.pitch_max, o.pitch)
            # Lips pursed on the device (mouthPucker/Funnel) count too: inhale and exhale share the shape.
            s.exhale = max(s.exhale, self.cfg.dwell_pucker_w * self._exhale_evidence(o.exhale))
        else:
            self._in_contact = False  # soft release: the next contact starts a new segment

    def update(self, o: Obs) -> Optional[PuffEvent]:
        c = self.cfg
        d = o.hand_mouth
        self._t_now = o.t
        if d is not None:
            self.t_last_seen = o.t
        if self.state not in (self.DWELL,):
            self._track_pitch(o)
        try:
            return self._step(o, d, c)
        finally:
            self._t_prev = o.t

    def _step(self, o: Obs, d: Optional[float], c: FSMConfig) -> Optional[PuffEvent]:
        if self.state == self.IDLE:
            if d is not None and d < c.approach:
                self.state, self.t_approach = self.APPROACH, o.t
            if d is not None and d < c.contact:
                self._enter_dwell(o)
            return None
        if self.state == self.APPROACH:
            if d is None or d >= c.approach * 1.1:
                self.state = self.IDLE
            elif d < c.contact:
                self._enter_dwell(o)
            return None
        if self.state == self.DWELL:
            if d is None:
                if o.t - self.t_last_seen > c.lost_s:
                    self.state, self.segs = self.IDLE, []  # lost tracking mid-gesture: drop, don't guess
                return None
            self.dwell_s = o.t - self.t_contact
            self._accumulate(o, d)
            if self.dwell_s > c.max_dwell_s * 3:
                return self._finish(o.t)
            if d > c.release:
                self.state, self.t_release = self.EXHALE, o.t
                self.post_exhale = max(self.post_exhale, self._exhale_evidence(o.exhale))
            return None
        if self.state == self.EXHALE:
            self.post_exhale = max(self.post_exhale, self._exhale_evidence(o.exhale))
            if d is not None and d < c.contact and o.t - self.t_release < 1.0:
                # Hand came straight back (double tap / adjusting): keep it one dwell.
                self.state = self.DWELL
                self._in_contact = False
                self._accumulate(o, d)
                return None
            if o.t - self.t_release >= c.exhale_window_s or self.exhale_peak >= 0.999:
                return self._finish(o.t)
        return None

    def _enter_dwell(self, o: Obs) -> None:
        self.state, self.t_contact, self.dwell_s = self.DWELL, o.t, 0.0
        self.pitch_base = self._baseline_pitch(o.t)
        self.segs, self.post_exhale, self._in_contact = [], 0.0, False
        self._t_prev = None
        self._accumulate(o, o.hand_mouth if o.hand_mouth is not None else 0.0)

    def flush(self, t: float) -> Optional[PuffEvent]:
        """End of stream: close a pending EXHALE window."""
        if self.state == self.EXHALE:
            return self._finish(t)
        return None
