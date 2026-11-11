import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def store(tmp_path):
    from vapecoach.storage import Store
    s = Store(tmp_path / "data.db")
    yield s
    s.close()


@pytest.fixture
def settings():
    from vapecoach.config import load_settings
    return load_settings(seed=7)
