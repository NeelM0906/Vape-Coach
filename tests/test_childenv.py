"""The camera watcher gets an allowlisted environment, never the gateway's provider keys."""
from vapecoach.childenv import watcher_env

GATEWAY = {
    "PATH": r"C:\Windows\system32", "PYTHONPATH": r"C:\hermes;C:\hermes\venv\Lib\site-packages",
    "HERMES_HOME": r"C:\Users\u\AppData\Local\hermes", "SystemRoot": r"C:\Windows", "TEMP": r"C:\Temp",
    "USERPROFILE": r"C:\Users\u", "LOCALAPPDATA": r"C:\Users\u\AppData\Local",
    "OPENAI_API_KEY": "sk-secret", "ANTHROPIC_API_KEY": "sk-ant-secret", "OPENROUTER_API_KEY": "or-secret",
    "GITHUB_TOKEN": "gho_secret", "AWS_SECRET_ACCESS_KEY": "aws", "HERMES_GATEWAY_TOKEN": "tok",
    "DISPLAY": ":0",
}


def test_no_secrets_reach_the_watcher():
    env = watcher_env("C:/data.db", GATEWAY, windows=True)
    assert not any(k in env for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY",
                                      "GITHUB_TOKEN", "AWS_SECRET_ACCESS_KEY", "HERMES_GATEWAY_TOKEN"))
    assert not any("secret" in v for v in env.values())


def test_keeps_what_the_watcher_needs_on_windows():
    env = watcher_env("C:/data.db", GATEWAY, windows=True)
    assert env["PYTHONPATH"] == GATEWAY["PYTHONPATH"] and env["PATH"] == GATEWAY["PATH"]
    assert env["HERMES_HOME"] == GATEWAY["HERMES_HOME"]
    assert env["SYSTEMROOT"] == r"C:\Windows"          # case-insensitive on Windows
    assert env["VAPE_COACH_DB"] == "C:/data.db" and env["PYTHONUNBUFFERED"] == "1"
    assert "DISPLAY" not in env


def test_posix_keeps_display_and_drops_windows_names():
    env = watcher_env("/data.db", {"PATH": "/usr/bin", "HOME": "/home/u", "DISPLAY": ":0", "SystemRoot": "x",
                                   "OPENAI_API_KEY": "sk"}, windows=False)
    assert env["DISPLAY"] == ":0" and env["HOME"] == "/home/u"
    assert "SYSTEMROOT" not in env and "SystemRoot" not in env and "OPENAI_API_KEY" not in env
