"""The Desktop half and its backend agree on the API version (stale-backend detection)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_api_version_matches_between_backend_and_desktop():
    api = re.search(r"^API_VERSION = (\d+)", (ROOT / "dashboard" / "plugin_api.py").read_text(encoding="utf-8"), re.M)
    js = re.search(r"^const API_VERSION = (\d+)", (ROOT / "desktop" / "plugin.js").read_text(encoding="utf-8"), re.M)
    assert api and js and api.group(1) == js.group(1)


def test_desktop_has_no_undo_and_uses_camera_routes():
    js = (ROOT / "desktop" / "plugin.js").read_text(encoding="utf-8")
    api = (ROOT / "dashboard" / "plugin_api.py").read_text(encoding="utf-8")
    assert "/undo" not in js and '"/undo"' not in api
    for route in ("/camera/start", "/camera/stop"):
        assert route in js and f'"{route}"' in api
