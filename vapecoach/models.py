"""MediaPipe .task models: downloaded at setup from Google's official bucket, SHA256-pinned.

Nothing binary is committed to the repo (catalog rule 17). ``hermes vape setup`` fetches into
``<hermes home>/plugin-data/vape-coach/models/`` and refuses any file whose hash differs.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Iterable, Optional

BASE = "https://storage.googleapis.com/mediapipe-models"
MODELS = {
    "pose_lite": (f"{BASE}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
                  "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a"),
    "pose_full": (f"{BASE}/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
                  "5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1"),
    "face": (f"{BASE}/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
             "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"),
    "hand": (f"{BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
             "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"),
}
DEFAULT_SET = ("pose_lite", "face", "hand")


def models_dir() -> Path:
    from .storage import default_db_path
    d = default_db_path().parent / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_path(key: str, root: Optional[Path] = None) -> Path:
    url, _ = MODELS[key]
    return (root or models_dir()) / url.rsplit("/", 1)[1]


def is_ready(keys: Iterable[str] = DEFAULT_SET, root: Optional[Path] = None) -> bool:
    return all(model_path(k, root).is_file() and sha256(model_path(k, root)) == MODELS[k][1] for k in keys)


def ensure(keys: Iterable[str] = DEFAULT_SET, root: Optional[Path] = None, log=print) -> dict[str, Path]:
    out = {}
    root = root or models_dir()
    for key in keys:
        url, want = MODELS[key]
        dest = model_path(key, root)
        if dest.is_file() and sha256(dest) == want:
            out[key] = dest
            continue
        log(f"downloading {key}: {url}")
        fd, tmp = tempfile.mkstemp(dir=str(root), suffix=".part")
        os.close(fd)
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            got = sha256(Path(tmp))
            if got != want:
                raise RuntimeError(f"{key}: SHA256 mismatch (got {got}, want {want}); refusing to use it")
            os.replace(tmp, dest)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        out[key] = dest
    return out
