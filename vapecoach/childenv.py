"""Minimal environment for the camera watcher child process.

The Desktop backend runs inside the gateway, whose environment holds every provider key Hermes
uses. The watcher needs none of them: only enough to start Python with Hermes's dependency
environment, find the profile's data, and open the camera. Everything else is dropped.
"""

from __future__ import annotations

import os
from typing import Mapping, Optional

# Names copied from the parent when set. No credentials: the watcher never calls a provider.
_COMMON = (
    "PATH",            # the interpreter's DLL/shared-library lookup
    "PYTHONPATH",      # Hermes's dependency environment (mediapipe, opencv, numpy)
    "HERMES_HOME",     # profile-scoped config and plugin data
    "HOME",
    "LANG", "LC_ALL", "LC_CTYPE", "TZ",
)
_WINDOWS = (
    "USERPROFILE", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "TEMP", "TMP",
    "APPDATA", "LOCALAPPDATA", "PROGRAMDATA", "COMSPEC", "PATHEXT",
    "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "OS",
)
_POSIX = (
    "TMPDIR", "USER", "LOGNAME",
    "XDG_RUNTIME_DIR", "DISPLAY", "WAYLAND_DISPLAY",   # camera/display stack on Linux
    "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH",             # OpenCV's native libraries
)


def watcher_env(db_path: str, parent: Optional[Mapping[str, str]] = None, windows: Optional[bool] = None) -> dict:
    """The allowlisted subset of ``parent`` (default ``os.environ``) plus the watcher's own settings."""
    parent = os.environ if parent is None else parent
    windows = (os.name == "nt") if windows is None else windows
    names = _COMMON + (_WINDOWS if windows else _POSIX)
    upper = {k.upper(): k for k in parent} if windows else None   # Windows env names are case-insensitive
    env: dict = {}
    for name in names:
        key = upper.get(name) if upper is not None else (name if name in parent else None)
        if key is not None and parent.get(key):
            env[name] = parent[key]
    env.update({
        "VAPE_COACH_DB": str(db_path),
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "GLOG_minloglevel": "2",
        "TF_CPP_MIN_LOG_LEVEL": "2",
    })
    return env
