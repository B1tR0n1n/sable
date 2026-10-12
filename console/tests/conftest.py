"""Make `import console...` work from any cwd: the console package lives at
<sable root>/console, and pytest's rootdir is console/ (its pyproject)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # the SABLE checkout
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest


@pytest.fixture(autouse=True)
def _no_console_token(monkeypatch):
    """create_app reads CONSOLE_TOKEN from the environment; a token set on the
    box (the OLS container sets one) must not turn every test route into a 401.
    Tests that exercise auth set it themselves."""
    monkeypatch.delenv("CONSOLE_TOKEN", raising=False)
