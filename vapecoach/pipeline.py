"""OpenCV + MediaPipe pipeline: camera/replay frames -> Obs -> HandToMouthFSM -> puff events.

Two rates (the "lightweight" part):
  * idle:   Pose Landmarker lite only, ~2 fps. Cheap enough to leave running all day.
  * active: Pose + Face (blendshapes) + Hand landmarkers at ~12 fps, entered when a hand gets
            near the face and held while the FSM has a candidate in flight (+ ``hold_s``).
Frames live only in memory for the duration of one inference; nothing is written to disk unless
the user explicitly passes ``--debug-out`` on a replay of their own clip.
"""

from __future__ import annotations

import math
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .detect_fsm import FSMConfig, HandToMouthFSM, Obs, PuffEvent
from .geometry import (pose_pitch, H_INDEX_MCP, H_INDEX_TIP, H_MIDDLE_TIP, H_THUMB_TIP, H_WRIST, L_INDEX, L_THUMB,
                       L_WR, R_INDEX, R_THUMB, R_WR, dist, face_mouth, face_scale, pose_face_unit,
                       pose_mouth, vis)

MIN_FACE_PX = 28  # faces smaller than this (in source pixels) are too far for reliable cues
from .reps import RepCounter


def _lm_list(lms) -> list[tuple[float, float, float, float]]:
    return [(p.x, p.y, p.z, getattr(p, "visibility", None) if getattr(p, "visibility", None) is not None else 1.0)
            for p in lms]


class Landmarkers:
    """Lazily-built MediaPipe Tasks landmarkers in VIDEO mode (each keeps its own timestamps)."""

    def __init__(self, model_paths: dict, pose_key: str = "pose_lite"):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self._mp = mp
        self._vision = vision
        self._BaseOptions = BaseOptions
        self.paths = model_paths
        self.pose_key = pose_key
        self._pose = self._face = self._hand = None
        self._ts = {"pose": -1, "face": -1, "hand": -1}

    def _opts(self, key):
        return self._BaseOptions(model_asset_path=str(self.paths[key]))

    @property
    def pose(self):
        if self._pose is None:
            v = self._vision
            self._pose = v.PoseLandmarker.create_from_options(v.PoseLandmarkerOptions(
                base_options=self._opts(self.pose_key), running_mode=v.RunningMode.VIDEO, num_poses=1))
        return self._pose

    @property
    def face(self):
        if self._face is None:
            v = self._vision
            self._face = v.FaceLandmarker.create_from_options(v.FaceLandmarkerOptions(
                base_options=self._opts("face"), running_mode=v.RunningMode.VIDEO, num_faces=1,
                output_face_blendshapes=True))
        return self._face

    @property
    def hand(self):
        if self._hand is None:
            v = self._vision
            self._hand = v.HandLandmarker.create_from_options(v.HandLandmarkerOptions(
                base_options=self._opts("hand"), running_mode=v.RunningMode.VIDEO, num_hands=2))
        return self._hand

    def _stamp(self, key: str, t: float) -> int:
        ms = max(int(t * 1000), self._ts[key] + 1)  # VIDEO mode needs strictly increasing ms
        self._ts[key] = ms
        return ms

    def _img(self, rgb):
        import numpy as np
        return self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

    @staticmethod
    def head_roi(pose_px, W: int, H: int):
        """Square crop (x0, y0, side) around the head + room for a hand at the mouth, or None.

        May extend past the frame; ``crop_square`` pads with black so MediaPipe always gets a
        square image (its landmark projection assumes a square ROI without IMAGE_DIMENSIONS).
        """
        nose, ls, rs = pose_px[0], pose_px[11], pose_px[12]
        if nose[3] < 0.5:
            return None
        sw = math.hypot(ls[0] - rs[0], ls[1] - rs[1]) if min(ls[3], rs[3]) > 0.3 else 0.0
        ear = math.hypot(pose_px[7][0] - pose_px[8][0], pose_px[7][1] - pose_px[8][1])
        half = max(1.5 * sw, 3.2 * ear, 72.0)
        half = min(half, max(W, H) / 2)
        x0, y0 = int(nose[0] - half), int(nose[1] - 0.9 * half)
        return x0, y0, int(2 * half)

    @staticmethod
    def crop_square(rgb, x0: int, y0: int, side: int):
        import cv2
        H, W = rgb.shape[:2]
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x0 + side), min(H, y0 + side)
        if cx1 - cx0 < 16 or cy1 - cy0 < 16:
            return None
        crop = rgb[cy0:cy1, cx0:cx1]
        pad = (cy0 - y0, (y0 + side) - cy1, cx0 - x0, (x0 + side) - cx1)
        if any(pad):
            crop = cv2.copyMakeBorder(crop, *pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return crop

    def run(self, rgb, t: float, which: tuple[str, ...]):
        """Landmarks in *height units*: x in [0, W/H], y in [0, 1] (so distances are isotropic).

        Face and Hand run on a head-centred crop taken from the pose (upsampled when small): far
        and small faces are found far more reliably, and the crop costs less than the full frame.
        """
        import cv2
        H, W = rgb.shape[:2]
        aspect = W / H
        out = {"pose": None, "face": None, "blend": {}, "hands": [], "aspect": aspect, "roi": None, "h_px": H}
        pose_px = None
        if "pose" in which:
            r = self.pose.detect_for_video(self._img(rgb), self._stamp("pose", t))
            if r.pose_landmarks:
                lms = _lm_list(r.pose_landmarks[0])
                out["pose"] = [(x * aspect, y, z, v) for x, y, z, v in lms]
                pose_px = [(x * W, y * H, z, v) for x, y, z, v in lms]
        if not ({"face", "hand"} & set(which)):
            return out
        roi = self.head_roi(pose_px, W, H) if pose_px else None
        crop = self.crop_square(rgb, *roi) if roi else None
        if crop is not None:
            x0, y0, side = roi
            target = min(max(side, 256), 384)
            if target != side:
                crop = cv2.resize(crop, (target, target),
                                  interpolation=cv2.INTER_LINEAR if target > side else cv2.INTER_AREA)
            to_full = lambda p: ((p[0] * side + x0) / H, (p[1] * side + y0) / H, p[2], p[3])
            out["roi"] = (x0 / H, y0 / H, (x0 + side) / H, (y0 + side) / H)
        else:
            crop = rgb
            to_full = lambda p: (p[0] * aspect, p[1], p[2], p[3])
        img = self._img(crop)
        if "face" in which:
            r = self.face.detect_for_video(img, self._stamp("face", t))
            if r.face_landmarks:
                out["face"] = [to_full(p) for p in _lm_list(r.face_landmarks[0])]
                if r.face_blendshapes:
                    out["blend"] = {c.category_name: c.score for c in r.face_blendshapes[0]}
        if "hand" in which:
            r = self.hand.detect_for_video(img, self._stamp("hand", t))
            out["hands"] = [[to_full(p) for p in _lm_list(h)] for h in (r.hand_landmarks or [])]
        return out

    def close(self):
        for lm in (self._pose, self._face, self._hand):
            if lm is not None:
                lm.close()


def observe(t: float, lm: dict) -> Obs:
    """Landmarks -> one FSM observation (distances in face widths)."""
    pose, face, hands, blend = lm["pose"], lm["face"], lm["hands"], lm["blend"]
    mouth, unit = None, None
    if face:
        mouth, unit = face_mouth(face), face_scale(face)
    if (mouth is None or unit is None) and pose:
        pu = pose_face_unit(pose)
        if pu and min(vis(pose[9]), vis(pose[10])) > 0.5:
            mouth, unit = pose_mouth(pose), pu
    if mouth is None or not unit or unit * lm.get("h_px", 10_000) < MIN_FACE_PX:
        return Obs(t, None, source="none")
    best, grip = None, None
    if hands and pose:
        # Cross-check: Hand Landmarker on a face-heavy crop can hallucinate a "hand" on the face.
        # Keep a hand only if a visible Pose wrist agrees with its wrist (or no Pose wrist is visible).
        wrists = [pose[i] for i in (L_WR, R_WR) if vis(pose[i]) > 0.5]
        if wrists:
            hands = [h for h in hands if min(dist(h[H_WRIST], w) for w in wrists) < 1.5 * unit]
    if hands:
        for h in hands:
            d = min(dist(h[i], mouth) for i in (H_WRIST, H_THUMB_TIP, H_INDEX_TIP, H_MIDDLE_TIP)) / unit
            if best is None or d < best:
                size = max(dist(h[H_WRIST], h[H_INDEX_MCP]), 1e-3)
                best, grip = d, dist(h[H_THUMB_TIP], h[H_INDEX_TIP]) / size
        src = "face_hand"
    elif pose:
        pts = [pose[i] for i in (L_WR, R_WR, L_INDEX, R_INDEX, L_THUMB, R_THUMB) if vis(pose[i]) > 0.5]
        if pts:
            best = min(dist(p, mouth) for p in pts) / unit
        src = "pose"
    else:
        src = "face"
    exhale = None
    jaw = None
    if blend:
        exhale = max(blend.get("mouthPucker", 0.0), blend.get("mouthFunnel", 0.0), 0.8 * blend.get("cheekPuff", 0.0))
        jaw = blend.get("jawOpen")
    pitch, hint = None, None
    if pose:
        pu = pose_face_unit(pose)
        pitch = pose_pitch(pose, pu) if pu else None
        # Pose rates a hand held at the face at visibility ~0.3-0.5 (occluded/foreshortened) while its
        # position is still right; that's enough to wake the Face+Hand models.
        weak = [pose[i] for i in (L_WR, R_WR, L_INDEX, R_INDEX, L_THUMB, R_THUMB) if vis(pose[i]) > 0.35]
        if weak:
            hint = min(dist(p, mouth) for p in weak) / unit
    return Obs(t, best, grip, exhale, jaw, src, hint=hint, pitch=pitch)


def _pct(xs: list, q: float) -> Optional[float]:
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))], 2)


@dataclass
class Metrics:
    lat_idle: list = field(default_factory=list)
    lat_active: list = field(default_factory=list)
    frames_seen: int = 0
    frames_idle: int = 0
    frames_active: int = 0
    ms_idle: float = 0.0
    ms_active: float = 0.0
    cpu_s: float = 0.0
    wall_s: float = 0.0
    media_s: float = 0.0

    def summary(self) -> dict:
        return {
            "frames_seen": self.frames_seen,
            "frames_processed_idle": self.frames_idle,
            "frames_processed_active": self.frames_active,
            "ms_per_idle_frame": round(self.ms_idle / self.frames_idle, 2) if self.frames_idle else None,
            "ms_per_active_frame": round(self.ms_active / self.frames_active, 2) if self.frames_active else None,
            "idle_ms_p50_p95": [_pct(self.lat_idle, 0.5), _pct(self.lat_idle, 0.95)],
            "active_ms_p50_p95": [_pct(self.lat_active, 0.5), _pct(self.lat_active, 0.95)],
            "active_share": round(self.frames_active / max(1, self.frames_idle + self.frames_active), 3),
            "process_cpu_s": round(self.cpu_s, 2),
            "wall_s": round(self.wall_s, 2),
            "media_s": round(self.media_s, 2),
            # CPU seconds burned per second of watched video, in % of ONE core (realtime cost).
            "cpu_pct_of_one_core_realtime": round(100 * self.cpu_s / self.media_s, 1) if self.media_s else None,
        }


class CloudCue:
    """Classical vapor cue: a cloud in front of the mouth brightens and blurs that patch.

    Compares the patch's Laplacian variance (texture) and mean brightness with a slow baseline
    learned while no gesture is in flight. Returns 0..1 or None (no mouth). Only ever consulted
    after a hand-to-mouth contact, so lighting changes alone cannot create an event.
    """

    def __init__(self, alpha: float = 0.08):
        self.alpha = alpha
        self.base_lv: Optional[float] = None
        self.base_br: Optional[float] = None

    def update(self, frame_bgr, lm: dict, o: Obs, learn: bool) -> Optional[float]:
        import cv2
        face, pose = lm.get("face"), lm.get("pose")
        if face:
            mouth, unit = face_mouth(face), face_scale(face)
        elif pose and pose_face_unit(pose):
            mouth, unit = pose_mouth(pose), pose_face_unit(pose)
        else:
            return None
        if not unit or unit * frame_bgr.shape[0] < MIN_FACE_PX:
            return None
        h, w = frame_bgr.shape[:2]
        cx, cy, r = mouth[0] * h, mouth[1] * h, unit * h
        x0, x1 = int(max(0, cx - 1.6 * r)), int(min(w, cx + 1.6 * r))
        y0, y1 = int(max(0, cy - 0.2 * r)), int(min(h, cy + 1.8 * r))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        g = cv2.cvtColor(cv2.resize(frame_bgr[y0:y1, x0:x1], (64, 64)), cv2.COLOR_BGR2GRAY)
        lv, br = float(cv2.Laplacian(g, cv2.CV_32F).var()), float(g.mean())
        if self.base_lv is None:
            self.base_lv, self.base_br = lv, br
            return 0.0
        blur_drop = max(0.0, 1.0 - lv / (self.base_lv + 1e-6))
        bright_up = max(0.0, (br - self.base_br) / 40.0)
        if learn:
            self.base_lv += self.alpha * (lv - self.base_lv)
            self.base_br += self.alpha * (br - self.base_br)
        return round(min(1.0, 0.7 * blur_drop + 0.5 * bright_up), 3)


class Detector:
    """Rate-scheduled puff detector over a stream of (frame_bgr, t)."""

    def __init__(self, landmarkers: Landmarkers, idle_fps: float = 2.0, active_fps: float = 12.0,
                 hold_s: float = 4.0, fsm_cfg: Optional[FSMConfig] = None,
                 on_event: Optional[Callable[[PuffEvent], None]] = None, always_active: bool = False):
        self.lm = landmarkers
        self.idle_dt, self.active_dt = 1.0 / idle_fps, 1.0 / active_fps
        self.hold_s = hold_s
        self.fsm = HandToMouthFSM(fsm_cfg)
        self.on_event = on_event
        self.always_active = always_active
        self.events: list[PuffEvent] = []
        self.metrics = Metrics()
        self._last_proc = -1e9
        self._active_until = -1e9
        self._hint_prev = False
        self.last = {"obs": None, "lm": None, "mode": "idle"}
        self.last_t = 0.0
        self.trace: Optional[list] = None   # per-processed-frame numbers for tuning (--trace)
        self.cloud = CloudCue()

    @property
    def active(self) -> bool:
        return self.always_active or self.fsm.wants_high_rate or self.last_t < self._active_until

    def feed(self, frame_bgr, t: float) -> Optional[PuffEvent]:
        import cv2
        self.metrics.frames_seen += 1
        self.last_t = t
        active = self.active
        if t - self._last_proc < (self.active_dt if active else self.idle_dt) - 1e-6:
            return None
        self._last_proc = t
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        which = ("pose", "face", "hand") if active else ("pose",)
        lm = self.lm.run(rgb, t, which)
        o = observe(t, lm)
        if active:
            cloud = self.cloud.update(frame_bgr, lm, o, learn=self.fsm.state == "idle")
            if cloud is not None:
                o.exhale = max(o.exhale or 0.0, cloud)
        ms = (time.perf_counter() - t0) * 1000
        if active:
            self.metrics.frames_active += 1
            self.metrics.ms_active += ms
            self.metrics.lat_active.append(ms)
        else:
            self.metrics.frames_idle += 1
            self.metrics.ms_idle += ms
            self.metrics.lat_idle.append(ms)
        # Escalate when a hand comes close to the mouth (pose-only check in idle mode).
        # A low-visibility Pose hand only wakes the models when it is right at the mouth (<1 face width).
        # It must persist for two idle frames (~1 s): typing/fidgeting hands flicker through, a hit doesn't.
        hint_now = o.hint is not None and o.hint < 1.0
        hint = o.hint if hint_now and (active or self._hint_prev) else None
        if not active:
            self._hint_prev = hint_now
        near_d = min([x for x in (o.hand_mouth, hint) if x is not None], default=None)
        near = near_d is not None and near_d < self.fsm.cfg.escalate
        if near and not active:
            strong = o.hand_mouth is not None and o.hand_mouth < self.fsm.cfg.escalate
            # A weak (hint-only) wake is a short look: if Face+Hand don't confirm, drop back fast.
            self._active_until = t + (self.hold_s if strong else 1.2)
        if not active and near:
            # Don't feed coarse idle frames into a DWELL: the next active frames start the gesture.
            self.fsm.update(Obs(t, max(near_d, self.fsm.cfg.contact + 1e-3), source="pose", pitch=o.pitch))
            self.last = {"obs": o, "lm": lm, "mode": "idle"}
            if self.trace is not None:
                self.trace.append((round(t, 3), "idle", o.source, o.hand_mouth, o.grip, o.exhale, o.jaw_open,
                                   self.fsm.state, 0, round(ms, 1), o.pitch, o.hint))
            return None
        ev = self.fsm.update(o)
        self.last = {"obs": o, "lm": lm, "mode": "active" if active else "idle"}
        if self.trace is not None:
            self.trace.append((round(t, 3), "active" if active else "idle", o.source, o.hand_mouth, o.grip,
                               o.exhale, o.jaw_open, self.fsm.state, int(ev is not None), round(ms, 1), o.pitch, o.hint))
        if ev:
            self.events.append(ev)
            if self.on_event:
                self.on_event(ev)
        return ev

    def finish(self, t: float) -> Optional[PuffEvent]:
        ev = self.fsm.flush(t)
        if ev:
            self.events.append(ev)
            if self.on_event:
                self.on_event(ev)
        return ev


# -- camera ---------------------------------------------------------------------------

def open_camera(index: int = 0, backend: str = "auto", width: int = 640, height: int = 480):
    """Open a webcam with the right OS backend; Windows tries DirectShow then Media Foundation."""
    import cv2
    names = {"dshow": "CAP_DSHOW", "msmf": "CAP_MSMF", "avfoundation": "CAP_AVFOUNDATION",
             "v4l2": "CAP_V4L2", "any": "CAP_ANY"}
    if backend == "auto":
        sysname = platform.system()
        order = (["dshow", "msmf", "any"] if sysname == "Windows" else
                 ["avfoundation", "any"] if sysname == "Darwin" else ["v4l2", "any"])
    else:
        order = [backend, "any"]
    tried = []
    for name in order:
        api = getattr(cv2, names.get(name, "CAP_ANY"), cv2.CAP_ANY)
        cap = cv2.VideoCapture(index, api)
        if cap is not None and cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            ok, _ = cap.read()
            if ok:
                return cap, name
        tried.append(name)
        if cap is not None:
            cap.release()
    hint = ""
    if platform.system() == "Windows":
        hint = (" On Windows check Settings > Privacy & security > Camera: 'Camera access', 'Let apps "
                "access your camera' and 'Let desktop apps access your camera' must be On, and close "
                "other apps (Teams/Zoom) holding the camera.")
    elif platform.system() == "Darwin":
        hint = " On macOS grant your terminal Camera access in System Settings > Privacy & Security > Camera."
    raise RuntimeError(f"could not open camera {index} (tried {', '.join(tried)}).{hint}")


# -- debug overlay ------------------------------------------------------------------

def draw_overlay(frame, det: Optional[Detector] = None, reps: Optional[RepCounter] = None, extra: str = ""):
    import cv2
    h, w = frame.shape[:2]
    lm = (det.last.get("lm") if det else None) or {}
    px = lambda p: (int(p[0] * h), int(p[1] * h))  # landmarks are in height units
    for p in (lm.get("pose") or []):
        if p[3] > 0.5:
            cv2.circle(frame, px(p), 3, (0, 255, 0), -1)
    for hand in lm.get("hands") or []:
        for p in hand:
            cv2.circle(frame, px(p), 2, (255, 128, 0), -1)
    face = lm.get("face")
    if face:
        for i in (13, 14, 61, 291):
            cv2.circle(frame, px(face[i]), 3, (0, 0, 255), -1)
    if lm.get("roi"):
        x0, y0, x1, y1 = lm["roi"]
        cv2.rectangle(frame, px((x0, y0)), px((x1, y1)), (200, 200, 0), 1)
    lines = []
    if det:
        o = det.last.get("obs")
        lines.append(f"mode={det.last.get('mode')} fsm={det.fsm.state} puffs={len(det.events)}")
        if o is not None:
            fmt = lambda v: "-" if v is None else f"{v:.2f}"
            lines.append(f"hand-mouth={fmt(o.hand_mouth)} grip={fmt(o.grip)} exhale={fmt(o.exhale)} jaw={fmt(o.jaw_open)}")
    if reps:
        lines.append(f"{reps.st.exercise} reps={reps.count} phase={reps.st.phase} {reps.st.mode} angle={reps.st.metric:.0f}")
    if extra:
        lines.append(extra)
    for i, txt in enumerate(lines):
        cv2.putText(frame, txt, (8, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(frame, txt, (8, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return frame


class Display:
    """``cv2.imshow`` when a GUI build is present; silently off on headless OpenCV."""

    def __init__(self, enabled: bool, title: str = "vape-coach debug"):
        self.enabled, self.title = enabled, title

    def show(self, frame) -> bool:
        """Returns False when the user pressed q/Esc."""
        if not self.enabled:
            return True
        import cv2
        try:
            cv2.imshow(self.title, frame)
            k = cv2.waitKey(1) & 0xFF
            return k not in (ord("q"), 27)
        except cv2.error:
            print("debug window unavailable (headless OpenCV); use --debug-out on replays", file=sys.stderr)
            self.enabled = False
            return True

    def close(self):
        if self.enabled:
            import cv2
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


def iter_video(path: Path):
    """Yield (frame_bgr, t_seconds) from a video file, using its own clock."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame, i / fps
            i += 1
    finally:
        cap.release()
