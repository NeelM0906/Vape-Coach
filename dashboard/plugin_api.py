"""vape-coach backend for the Desktop half, mounted at /api/plugins/vape-coach/.

Runs inside the gateway. The camera watcher is a separate process writing to the same SQLite DB,
so the Desktop half polls ``GET /state`` (every 3 s) for new nudges, status and live reps.
In-gateway actions (``POST /hit`` etc.) also push ``plugin.vape-coach.*`` events immediately.

``POST /camera/start`` runs the same watcher as ``hermes vape watch`` as a child process, only
when the user clicks Start in the pane; ``POST /camera/stop`` asks it to exit. The watcher counts
owed reps by itself, so the user never has to type a command.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter

PLUGIN = "vape-coach"
_ROOT = Path(__file__).resolve().parent.parent / "vapecoach"


def _load_core():
    """Import the plugin's own core package by path (this file is loaded standalone)."""
    name = "vape_coach_core"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _ROOT / "__init__.py",
                                                  submodule_search_locations=[str(_ROOT)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_load_core()
from vape_coach_core.coach import Coach  # noqa: E402
from vape_coach_core.config import load_settings  # noqa: E402
from vape_coach_core.interventions import MEMES, PR_THREATS, SHAKES  # noqa: E402
from vape_coach_core.storage import Store  # noqa: E402

router = APIRouter()
_STORE: Store | None = None
# Bumped whenever routes change. Hermes mounts plugin routes once, when the gateway starts, so the
# Desktop half compares this with its own API_VERSION and asks for a restart after an update.
API_VERSION = 2


def _get_config(key: str, default: Any = None) -> Any:
    try:
        from hermes_cli.config import load_config_readonly  # public plugin API
        entry = ((load_config_readonly() or {}).get("plugins") or {}).get("entries", {}).get(PLUGIN) or {}
        return (entry.get("settings") or {}).get(key, default)
    except Exception:
        return default


def _coach() -> Coach:
    global _STORE
    if _STORE is None:
        _STORE = Store()
    return Coach(_STORE, load_settings(_get_config))


def _broadcast(event: str, payload: dict) -> None:
    try:
        from hermes_cli.plugin_events import broadcast_plugin_event  # public plugin API
        broadcast_plugin_event(PLUGIN, event, payload)
    except Exception:
        pass


@router.get("/state")
async def state(since_id: int = 0):
    c = _coach()
    ivs = [i for i in c.store.interventions() if i["id"] > since_id and i["kind"] != "observe"]
    return {"status": c.status(), "interventions": ivs[-10:], "server_ts": time.time(),
            "exercise": c.s.exercise, "delivery": c.s.delivery,
            "verify_reps": c.s.verify_reps_with_camera, "camera": _camera_state(c), "api": API_VERSION}


@router.get("/stats")
async def stats(days: int = 7):
    return _coach().stats(days=max(1, min(days, 30)))


@router.post("/hit")
async def hit(body: dict | None = None):
    c = _coach()
    n = max(1, min(int((body or {}).get("puffs") or 1), 20))
    out = c.record_puffs(n)
    if out.get("intervention"):
        _broadcast("intervention", out["intervention"])
    _broadcast("status", c.status())
    return {"ok": True, "result": out}


@router.post("/done")
async def done():
    c = _coach()
    out = c.honor_done()
    if out["ok"]:
        _broadcast("status", c.status())
    return out


# -- camera watcher (child process) --------------------------------------------------

_CHILD: subprocess.Popen | None = None
BEAT_FRESH_S = 20


def _camera_state(c: Coach) -> dict:
    w = c.store.get("watcher", {}) or {}
    alive = bool(w.get("pid")) and time.time() - float(w.get("beat") or 0) < BEAT_FRESH_S
    starting = _CHILD is not None and _CHILD.poll() is None and not alive
    return {"on": alive, "starting": starting, "mode": w.get("mode") if alive else None,
            "owned": _CHILD is not None and _CHILD.poll() is None,
            "exit_code": _CHILD.returncode if _CHILD is not None and _CHILD.poll() is not None else None}


@router.post("/camera/start")
async def camera_start():
    """Start the watcher (camera + automatic rep counting). User-initiated only."""
    global _CHILD
    c = _coach()
    cam = _camera_state(c)
    if cam["on"] or cam["starting"]:
        return {"ok": True, "camera": cam}
    root = Path(__file__).resolve().parent.parent
    log_path = c.store.path.parent / "watcher.log"
    env = dict(os.environ, VAPE_COACH_DB=str(c.store.path), PYTHONUNBUFFERED="1")
    kw: dict = {}
    if os.name == "nt":
        kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    else:
        kw["start_new_session"] = True
    c.store.put("watcher_cmd", {"stop": False, "ts": time.time()})
    with open(log_path, "ab") as log:
        _CHILD = subprocess.Popen([sys.executable, "-m", "vapecoach.cli", "watch"], cwd=str(root), env=env,
                                  stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, **kw)
    return {"ok": True, "camera": _camera_state(c)}


@router.post("/camera/stop")
async def camera_stop():
    """Ask the watcher (whoever started it) to exit; it checks the flag every ~2 s."""
    global _CHILD
    c = _coach()
    c.store.put("watcher_cmd", {"stop": True, "ts": time.time()})
    if _CHILD is not None and _CHILD.poll() is None:
        for _ in range(40):              # the watcher polls the flag every ~2 s; don't block the loop
            if _CHILD.poll() is not None:
                break
            await asyncio.sleep(0.2)
        else:
            _CHILD.terminate()
    return {"ok": True, "camera": _camera_state(c)}


@router.get("/roast")
async def roast():
    return {"text": random.choice(MEMES + PR_THREATS + SHAKES)}
