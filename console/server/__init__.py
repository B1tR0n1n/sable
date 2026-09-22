"""Phase 7 — the console server: one process that runs the loop and serves
the UI. `Loop` (loop.py) is the orchestration, framework-free and driven by
fakes in tests; `app.py` is the FastAPI translation of console/docs/API.md
over it. `python -m console.server` starts it."""

from .loop import Config, Loop  # noqa: F401
from .app import create_app  # noqa: F401
