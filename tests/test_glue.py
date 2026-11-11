"""register(ctx) against a fake context: declarations match, nag-only, no repo-writing tools."""
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


class FakeCtx:
    def __init__(self, settings=None):
        self.tools, self.hooks, self.commands, self.cli, self.skills, self.posted = {}, {}, {}, {}, {}, []
        self._settings = settings or {}

    def register_tool(self, name, toolset, schema, handler, **kw):
        self.tools[name] = (schema, handler)

    def register_hook(self, name, cb, **kw):
        assert not kw.get("hold_stream")
        self.hooks[name] = cb

    def register_command(self, name, handler, **kw):
        self.commands[name] = handler

    def register_cli_command(self, name, help, setup_fn, handler_fn, **kw):
        self.cli[name] = (setup_fn, handler_fn)

    def register_skill(self, name, path, **kw):
        assert Path(path).is_file()
        self.skills[name] = path

    def get_config(self, key, default=None):
        return self._settings.get(key, default)

    def post_message(self, text):
        self.posted.append(text)
        return type("R", (), {"ok": True})()


@pytest.fixture
def plugin(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    spec = importlib.util.spec_from_file_location("vape_coach_plugin", ROOT / "__init__.py",
                                                  submodule_search_locations=[str(ROOT)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["vape_coach_plugin"] = mod
    spec.loader.exec_module(mod)
    ctx = FakeCtx({"seed": 3})
    mod.register(ctx)
    return ctx


def test_registrations_match_manifest(plugin):
    m = yaml.safe_load((ROOT / "plugin.yaml").read_text())
    assert set(plugin.tools) == set(m["provides_tools"])
    assert set(plugin.hooks) == set(m["provides_hooks"])
    assert "pre_tool_call" not in plugin.hooks and "transform_llm_output" not in plugin.hooks
    assert set(plugin.commands) == {"vape"} and set(plugin.cli) == {"vape"} and "vape-coach" in plugin.skills


def test_tool_schemas_valid(plugin):
    for name, (schema, _) in plugin.tools.items():
        assert schema["name"] == name and schema["parameters"]["type"] == "object"


def test_pre_llm_call_never_blocks(plugin):
    cb = plugin.hooks["pre_llm_call"]
    assert cb(session_id="s", user_message="hi") is None
    plugin.commands["vape"]("hit")
    from_store = plugin.tools["vape_status"][1]({})
    assert json.loads(from_store)["today"]["puffs"] == 1
    # owed challenge -> context, not block
    import vape_coach_plugin.vapecoach.storage as st
    st.Store().add_challenge("jumping_jacks", 15)
    for _ in range(3):
        r = cb(session_id="s", user_message="fix the bug")
        assert isinstance(r, dict) and set(r) == {"context"} and "15 jumping jacks" in r["context"]


def test_slash_and_tools(plugin):
    v = plugin.commands["vape"]
    assert "baseline" in v("status") or "phase" in v("status")
    assert "logged 2 puff" in v("hit 2")
    # Hits can't be removed from chat (only `hermes vape undo` in a terminal, for false alarms).
    assert "can't be undone" in v("undo")
    assert json.loads(plugin.tools["vape_status"][1]({}))["today"]["puffs"] == 2
    # Camera-verified challenges can't be honor-closed from chat.
    import vape_coach_plugin.vapecoach.storage as st
    st.Store().add_challenge("jumping_jacks", 15)
    assert "camera-counted" in v("done")
    assert st.Store().open_challenge()["done"] == 0
    assert "automatically" in v("help")
    assert "day" in v("stats")
    assert v("roast")
    out = json.loads(plugin.tools["vape_log_hit"][1]({"puffs": 2, "minutes_ago": 5}))
    assert out["logged_puffs"] == 2
    assert "daily" in json.loads(plugin.tools["vape_stats"][1]({"days": 3}))


def test_post_llm_call_delivers_queued_nudges(plugin):
    import vape_coach_plugin.vapecoach.storage as st
    st.Store().add_intervention("meme", "camera saw that")
    plugin.hooks["post_llm_call"](session_id="s")
    assert any("camera saw that" in p for p in plugin.posted)
    plugin.posted.clear()
    plugin.hooks["post_llm_call"](session_id="s")
    assert plugin.posted == []


def test_source_has_no_blocking_or_repo_writes():
    src = "\n".join(p.read_text() for p in (ROOT / "vapecoach").glob("*.py"))
    assert not re.search(r'return\s*\{\s*"action"\s*:\s*"block"', src)
    assert not re.search(r'"decision"\s*:\s*"block"', src)
    assert not re.search(r"\bsubprocess\b|os\.system|git push|gh pr", src)
