"""``hermes vape ...`` terminal commands (registered via ``ctx.register_cli_command``).

The camera lives here, in a foreground process the user starts and stops, so the OS camera
permission belongs to their terminal and they always know when it is on.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

from .config import EXERCISES, Settings, load_settings

SUBCOMMANDS = (
    ("setup", "download the MediaPipe models (SHA256-pinned) and check the camera stack"),
    ("watch", "watch the webcam for vape hits and count owed reps automatically (Ctrl+C to stop)"),
    ("reps", "count exercise reps on camera right now (the watcher also does this by itself)"),
    ("hit", "log an off-camera hit"),
    ("undo", "void the most recent hit (false positive) and any penalty it created"),
    ("status", "taper phase, target gap, time until the next allowed hit, owed challenges"),
    ("stats", "last 7 days"),
    ("done", "honor-system completion (only when verify_reps_with_camera is off)"),
    ("waive", "waive owed challenges"),
    ("eval", "score replay predictions against a labels CSV"),
    ("reset", "delete all vape-coach data (requires --yes)"),
)


def build_parser(sub: argparse.ArgumentParser) -> None:
    subs = sub.add_subparsers(dest="vape_command")
    p = {n: subs.add_parser(n, help=h, description=h) for n, h in SUBCOMMANDS}
    for name in ("watch", "reps"):
        a = p[name]
        a.add_argument("--replay", type=Path, help="video file instead of the webcam")
        a.add_argument("--camera", type=int, default=None, help="camera index (default from config)")
        a.add_argument("--backend", default=None, help="auto|dshow|msmf|avfoundation|v4l2|any")
        a.add_argument("--debug", action="store_true", help="show a landmark/state overlay window")
        a.add_argument("--debug-out", type=Path, help="(replay only) write the overlay video here")
        a.add_argument("--duration", type=float, default=None, help="stop after N seconds")
    w = p["watch"]
    w.add_argument("--out", type=Path, help="(replay) write predictions JSON here")
    w.add_argument("--log", action="store_true", help="(replay) also log detected hits to the coach")
    w.add_argument("--idle-fps", type=float, default=None)
    w.add_argument("--active-fps", type=float, default=None)
    w.add_argument("--always-active", action="store_true", help="skip idle mode (max recall, more CPU)")
    w.add_argument("--realtime", action="store_true", help="(replay) pace frames at video speed")
    w.add_argument("--trace", type=Path, help="write per-frame numbers (distances, blendshapes, FSM state) as CSV")
    r = p["reps"]
    r.add_argument("--exercise", choices=EXERCISES, default=None)
    r.add_argument("--target", type=int, default=None, help="default: what the open challenge still needs")
    p["hit"].add_argument("--count", type=int, default=1, help="puffs to log")
    p["hit"].add_argument("--ago", type=float, default=0.0, help="minutes ago")
    p["stats"].add_argument("--days", type=int, default=7)
    p["stats"].add_argument("--json", action="store_true")
    p["status"].add_argument("--json", action="store_true")
    e = p["eval"]
    e.add_argument("labels", type=Path, help="labels CSV (see eval/labels_template.csv)")
    e.add_argument("predictions", type=Path, nargs="+", help="replay JSON(s) from `watch --replay --out`")
    e.add_argument("--tolerance", type=float, default=2.0, help="seconds of slack when matching")
    e.add_argument("--sweep", action="store_true", help="also report P/R/F1 across score thresholds")
    e.add_argument("--brief", action="store_true", help="omit the per-clip breakdown")
    p["reset"].add_argument("--yes", action="store_true")


def _quiet_native_logs() -> None:
    """MediaPipe/TFLite log through glog to stderr; keep the terminal readable."""
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def make_handler(get_config: Optional[Callable[[str, Any], Any]] = None) -> Callable[[argparse.Namespace], int]:
    def handler(args: argparse.Namespace) -> int:
        s = load_settings(get_config)
        _quiet_native_logs()
        cmd = getattr(args, "vape_command", None)
        if not cmd:
            print("usage: hermes vape {" + ",".join(n for n, _ in SUBCOMMANDS) + "}")
            return 2
        try:
            return int(_DISPATCH[cmd](args, s) or 0)
        except KeyboardInterrupt:
            return 130
    return handler


def _coach(s: Settings):
    from .coach import Coach
    from .storage import Store
    return Coach(Store(), s)


def _print_intervention(out: dict) -> None:
    iv = out.get("intervention")
    if iv:
        print(f"\n  >> [{iv['kind']}] {iv['text']}\n", flush=True)


# -- setup --------------------------------------------------------------------------

def cmd_setup(args, s: Settings) -> int:
    from . import models
    keys = ("pose_full" if s.model_variant == "full" else "pose_lite", "face", "hand")
    paths = models.ensure(keys)
    for k, v in paths.items():
        print(f"ok  {k}: {v}")
    try:
        import cv2
        import mediapipe
        print(f"ok  opencv {cv2.__version__}, mediapipe {mediapipe.__version__}")
    except ImportError as e:
        print(f"missing dependency: {e}. Install: pip install 'mediapipe>=0.10.14' opencv-python")
        return 1
    print("Camera check: run `hermes vape watch --debug` (Ctrl+C to stop). Frames never leave this machine.")
    return 0


def _landmarkers(s: Settings):
    from . import models
    from .pipeline import Landmarkers
    pose_key = "pose_full" if s.model_variant == "full" else "pose_lite"
    keys = (pose_key, "face", "hand")
    if not models.is_ready(keys):
        print("first run: downloading the MediaPipe models (SHA256-pinned)...", flush=True)
        models.ensure(keys)
    return Landmarkers({k: models.model_path(k) for k in keys}, pose_key=pose_key)


def _frames(args, s: Settings, duration: Optional[float]):
    """Yield (frame, t_media, wall_ts). Webcam: wall clock; replay: the file's clock."""
    from .pipeline import iter_video, open_camera
    if args.replay:
        start = time.time()
        for frame, t in iter_video(args.replay):
            if duration and t > duration:
                break
            if getattr(args, "realtime", False):
                lag = t - (time.time() - start)
                if lag > 0:
                    time.sleep(lag)
            yield frame, t, None
        return
    cap, backend = open_camera(s.camera_index if args.camera is None else args.camera,
                               args.backend or s.camera_backend)
    print(f"camera open ({backend}). Frames are processed in memory only. Ctrl+C to stop.", flush=True)
    t0 = time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            t = time.monotonic() - t0
            if duration and t > duration:
                break
            yield frame, t, time.time()
    finally:
        cap.release()


REP_FPS = 10.0          # Pose rate while counting owed reps
REP_PAUSE_PUFFS_S = 4.0  # no puff detection this soon after the counter saw exercise movement
REP_COOLDOWN_S = 15.0   # ...nor this soon after the last owed rep (walking back, wiping your face)
CHECK_EVERY_S = 2.0     # how often the watcher re-reads owed challenges and the stop flag


class AutoReps:
    """Counts owed reps inside the running watcher: nobody has to start a separate command.

    While a camera-verified challenge is owed, Pose runs at ``REP_FPS`` and every completed rep is
    credited at once (so the Desktop pane fills live and nothing is lost if the camera stops).
    Puff detection keeps running, but pauses while the counter sees exercise movement.
    """

    def __init__(self, coach, lms):
        from .reps import RepCounter
        self._RepCounter = RepCounter
        self.coach, self.lms = coach, lms
        self.counter = None
        self.exercise: Optional[str] = None
        self._last_proc = -1e9
        self._finished_t = -1e9
        self.last_lm = None

    @property
    def active(self) -> bool:
        return self.counter is not None

    def owed(self) -> list:
        return self.coach.store.open_challenges()

    def sync(self) -> Optional[str]:
        """Start/stop counting to match the owed challenges; returns a message to print, if any."""
        s = self.coach.s
        owed = [c for c in self.owed() if s.verify_reps_with_camera]
        if owed and self.counter is None:
            self.exercise = owed[0]["exercise"]
            self.counter = self._RepCounter(self.exercise)
            left = sum(c["target"] - c["done"] for c in owed if c["exercise"] == self.exercise)
            self._publish(active=True)
            return (f"[reps] {left} {self.exercise.replace('_', ' ')} owed. Step back so your head and arms "
                    f"are in frame: counting automatically.")
        if not owed and self.counter is not None:
            self.counter = None
            self._publish(active=False)
            return "[reps] nothing owed any more: back to watching for hits."
        return None

    def feed(self, frame_bgr, t: float) -> Optional[str]:
        """Run Pose on this frame if due; credit a completed rep. Returns a message to print, if any."""
        if self.counter is None or t - self._last_proc < 1.0 / REP_FPS:
            return None
        import cv2
        self._last_proc = t
        r = self.lms.run(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), t, ("pose",))
        self.last_lm = r
        if not self.counter.update(r["pose"], t):
            return None
        out = self.coach.credit_reps(self.exercise, 1, "camera")
        self._publish(active=True)
        left = sum(c["target"] - c["done"] for c in out["open"] if c["exercise"] == self.exercise)
        msg = f"  rep {self.counter.count} ({left} left)"
        if out["message"]:
            msg += f"\n  >> {out['message']}"
        if not out["open"]:
            self.counter = None
            self._finished_t = t
            self._publish(active=False)
            msg += "\n[reps] all done: back to watching for hits."
        return msg

    def recently_moving(self, t: float) -> bool:
        if t - self._finished_t < REP_COOLDOWN_S:
            return True
        return self.counter is not None and t - self.counter.st.last_change_t < REP_PAUSE_PUFFS_S

    def _publish(self, active: bool) -> None:
        owed = [c for c in self.owed() if c["exercise"] == self.exercise]
        self.coach.store.put("live_reps", {
            "exercise": self.exercise, "active": active, "ts": time.time(),
            "count": owed[0]["done"] if owed else 0, "target": owed[0]["target"] if owed else 0,
            "session_count": self.counter.count if self.counter else 0})


def _stop_requested(coach, started: float) -> bool:
    cmd = coach.store.get("watcher_cmd") or {}
    return bool(cmd.get("stop")) and float(cmd.get("ts", 0)) >= started


def cmd_watch(args, s: Settings) -> int:
    from .detect_fsm import HandToMouthFSM
    from .pipeline import Detector, Display, draw_overlay
    lms = _landmarkers(s)
    live = args.replay is None
    coach = _coach(s) if (live or args.log) else None
    wall_ref = {"ts": None}
    started = time.time()

    def on_event(ev):
        ts = wall_ref["ts"] if wall_ref["ts"] is not None else time.time()
        print(f"[puff] t={ev.t_start:.1f}s dwell={ev.dwell_s}s exhale={ev.exhale} grip={ev.grip} score={ev.score}", flush=True)
        if coach is not None:
            out = coach.record_hit(ts=ts, source="camera" if live else "replay", confidence=ev.score,
                                   dwell_s=ev.dwell_s, exhale=ev.exhale, meta={"grip": ev.grip})
            _print_intervention(out)

    det = Detector(lms, idle_fps=args.idle_fps or s.idle_fps, active_fps=args.active_fps or s.active_fps,
                   on_event=on_event, always_active=args.always_active)
    if args.trace:
        det.trace = []
    reps = AutoReps(coach, lms) if (live and coach is not None) else None
    disp = Display(args.debug)
    writer = None
    cpu0, wall0 = time.process_time(), time.perf_counter()
    last_beat, last_check, t = 0.0, 0.0, 0.0
    if live and coach is not None:
        coach.store.put("watcher", {"pid": os.getpid(), "beat": time.time(), "mode": "starting",
                                    "fsm": "idle", "puffs": 0, "started": started})
    try:
        for frame, t, wall in _frames(args, s, args.duration):
            wall_ref["ts"] = wall
            if reps is not None and time.time() - last_check > CHECK_EVERY_S:
                last_check = time.time()
                if _stop_requested(coach, started):
                    print("stop requested from the Vape Coach pane.", flush=True)
                    break
                msg = reps.sync()
                if msg:
                    print(msg, flush=True)
            if reps is not None and reps.active:
                msg = reps.feed(frame, t)
                if msg:
                    print(msg, flush=True)
                    if not reps.active:  # just finished: start puff detection from a clean slate
                        det.fsm = HandToMouthFSM(det.fsm.cfg)
            if reps is None or not reps.recently_moving(t):
                det.feed(frame, t)
            if args.debug or args.debug_out:
                over = draw_overlay(frame.copy(), det, reps.counter if reps is not None else None)
                if args.debug_out and not live:
                    writer = writer or _writer(args.debug_out, over, args.replay)
                    writer.write(over)
                if not disp.show(over):
                    break
            if live and coach is not None and time.time() - last_beat > 5:
                last_beat = time.time()
                mode = "reps" if (reps is not None and reps.active) else det.last["mode"]
                coach.store.put("watcher", {"pid": os.getpid(), "beat": last_beat, "mode": mode,
                                            "fsm": det.fsm.state, "puffs": len(det.events), "started": started})
    except KeyboardInterrupt:
        pass
    finally:
        det.finish(t)
        det.metrics.cpu_s = time.process_time() - cpu0
        det.metrics.wall_s = time.perf_counter() - wall0
        det.metrics.media_s = t
        if writer is not None:
            writer.release()
        disp.close()
        lms.close()
        if live and coach is not None:
            coach.store.put("watcher", {"pid": None, "beat": time.time(), "stopped": True})
            if reps is not None and reps.active:
                reps.counter = None
                reps._publish(active=False)
    result = {"source": str(args.replay) if args.replay else "camera",
              "events": [e.as_dict() for e in det.events],
              "rejected": det.fsm.rejected, "metrics": det.metrics.summary()}
    if args.out:
        args.out.write_text(json.dumps(result, indent=2))
    if args.trace and det.trace is not None:
        import csv
        with open(args.trace, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "mode", "source", "hand_mouth", "grip", "exhale", "jaw_open", "fsm", "event", "ms", "pitch", "hint"])
            w.writerows(det.trace)
    print(json.dumps({"puffs": len(det.events), "metrics": result["metrics"]}, indent=2))
    return 0


def _writer(path: Path, frame, src: Optional[Path] = None):
    """Overlay video at the source's frame rate (so its clock matches the replayed file)."""
    import cv2
    h, w = frame.shape[:2]
    fps = 15.0
    if src is not None:
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS) or fps
        cap.release()
    return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))


def cmd_reps(args, s: Settings) -> int:
    import mediapipe as mp
    import cv2
    from .pipeline import Display, draw_overlay
    from .reps import RepCounter
    exercise = args.exercise or s.exercise
    coach = _coach(s)
    owed = [c for c in coach.store.open_challenges() if c["exercise"] == exercise]
    target = args.target or (sum(c["target"] - c["done"] for c in owed) if owed else 10)
    lms = _landmarkers(s)
    counter = RepCounter(exercise)
    disp = Display(args.debug)
    writer = None
    print(f"Counting {exercise}: target {target}. Step back so your head, arms"
          f"{' and feet' if exercise != 'pushups' else ''} are in frame.", flush=True)
    last_proc, credited, t = -1.0, 0, 0.0
    try:
        for frame, t, _ in _frames(args, s, args.duration):
            if t - last_proc < 1 / 15:
                continue
            last_proc = t
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            r = lms.run(rgb, t, ("pose",))
            if counter.update(r["pose"], t):
                print(f"  rep {counter.count}/{target}", flush=True)
                if args.replay is None:
                    coach.store.put("live_reps", {"exercise": exercise, "count": counter.count, "target": target,
                                                  "ts": time.time(), "active": True})
            if args.debug or (args.debug_out and args.replay):
                class _D:  # minimal adapter for the overlay
                    last = {"lm": r, "obs": None, "mode": "reps"}
                    fsm = type("F", (), {"state": "-"})()
                    events: list = []
                over = draw_overlay(frame.copy(), _D(), counter)
                if args.debug_out and args.replay:
                    writer = writer or _writer(args.debug_out, over)
                    writer.write(over)   # one frame per processed frame (15 fps)
                if not disp.show(over):
                    break
            if counter.count >= target:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:
            writer.release()
        disp.close()
        lms.close()
    if args.replay is None:
        coach.store.put("live_reps", {"exercise": exercise, "count": counter.count, "target": target,
                                      "ts": time.time(), "active": False})
    if counter.count and args.replay is None:
        out = coach.credit_reps(exercise, counter.count, "camera")
        if out["message"]:
            print(out["message"])
    elif counter.count:
        print("(replay: reps not credited to your challenges)")
    print(f"{counter.count} {exercise} counted.")
    return 0


# -- bookkeeping --------------------------------------------------------------------

def cmd_hit(args, s: Settings) -> int:
    out = _coach(s).record_puffs(max(1, args.count), ts=time.time() - args.ago * 60)
    print(f"logged {out['puffs']} puff(s).")
    _print_intervention(out)
    return 0


def cmd_undo(args, s: Settings) -> int:
    ev = _coach(s).undo_last()
    print("voided last hit." if ev else "nothing to undo.")
    return 0


def cmd_status(args, s: Settings) -> int:
    st = _coach(s).status()
    if args.json:
        print(json.dumps(st, indent=2, default=str))
        return 0
    print(format_status(st))
    return 0


def format_status(st: dict) -> str:
    lines = [f"phase: {st['phase']}" + (f" (day {st['day_index']})" if st["phase"] == "coaching" else "")]
    if st["phase"] == "baseline":
        lines.append(f"baseline: observe-only until {time.strftime('%a %H:%M', time.localtime(st['baseline_ends_ts']))}")
    if st["phase"] == "coaching":
        lines.append(f"baseline gap {st['baseline_gap']} -> today's target gap {st['target_gap']}")
        lines.append(f"next hit allowed in: {st['until_allowed']}" if st["seconds_until_allowed"] else "next hit: allowed (but you could wait)")
    if st.get("since_last"):
        lines.append(f"since last session: {st['since_last']}")
    t = st["today"]
    lines.append(f"today: {t['puffs']} puffs in {t['sessions']} sessions; on-time streak {st['streak']}")
    for c in st["open_challenges"]:
        lines.append(f"owed: {c['target'] - c['done']} {c['exercise']} (challenge #{c['id']})")
    w = st.get("watcher") or {}
    alive = w.get("pid") and time.time() - w.get("beat", 0) < 20
    lines.append(f"watcher: {'on (' + str(w.get('mode')) + ')' if alive else 'off'}")
    return "\n".join(lines)


def cmd_stats(args, s: Settings) -> int:
    st = _coach(s).stats(days=args.days)
    if args.json:
        print(json.dumps(st, indent=2))
        return 0
    print(format_stats(st))
    return 0


def format_stats(st: dict) -> str:
    rows = ["day   puffs  sessions  mean-gap  longest"]
    for d in st["daily"]:
        rows.append(f"{d['day_offset']:>4}  {d['puffs']:>5}  {d['sessions']:>8}  "
                    f"{d['mean_gap_min'] if d['mean_gap_min'] is not None else '-':>8}  "
                    f"{d['longest_gap_min'] if d['longest_gap_min'] is not None else '-':>7}")
    rows.append(f"total {st['total_puffs']} puffs ({st['camera_puffs']} camera, {st['manual_puffs']} manual), "
                f"on-time rate {st['on_time_rate']}, reps {st['reps']}, streak {st['streak']}")
    return "\n".join(rows)


def cmd_done(args, s: Settings) -> int:
    out = _coach(s).honor_done()
    print(out["message"])
    return 0 if out["ok"] else 1


def cmd_waive(args, s: Settings) -> int:
    print(f"waived {_coach(s).store.waive_challenges()} challenge(s).")
    return 0


def cmd_eval(args, s: Settings) -> int:
    from .evaluate import evaluate_files
    rep = evaluate_files(args.labels, args.predictions, tolerance=args.tolerance, do_sweep=args.sweep)
    if args.brief:
        rep.pop("per_clip", None)
    print(json.dumps(rep, indent=2))
    return 0


def cmd_reset(args, s: Settings) -> int:
    if not args.yes:
        print("refusing without --yes")
        return 2
    _coach(s).store.reset()
    print("all vape-coach data deleted.")
    return 0


_DISPATCH = {"setup": cmd_setup, "watch": cmd_watch, "reps": cmd_reps, "hit": cmd_hit, "undo": cmd_undo,
             "status": cmd_status, "stats": cmd_stats, "done": cmd_done, "waive": cmd_waive,
             "eval": cmd_eval, "reset": cmd_reset}


def _hermes_config_reader() -> Optional[Callable[[str, Any], Any]]:
    """Read ``plugins.entries.vape-coach.settings`` through Hermes' public config API, if present."""
    try:
        from hermes_cli.config import load_config_readonly  # public plugin API
    except Exception:
        return None
    from . import PLUGIN_NAME

    def get(key: str, default: Any = None) -> Any:
        try:
            entry = ((load_config_readonly() or {}).get("plugins") or {}).get("entries", {}).get(PLUGIN_NAME) or {}
            return (entry.get("settings") or {}).get(key, default)
        except Exception:
            return default
    return get


def main(argv: Optional[list[str]] = None) -> int:
    """Standalone entry (``python -m vapecoach.cli``): the Desktop pane's camera button runs this."""
    ap = argparse.ArgumentParser(prog="vape-coach")
    build_parser(ap)
    return make_handler(_hermes_config_reader())(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
