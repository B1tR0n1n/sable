"""Find OVERLORD's Python SDK (sdk/overlord_client.py) wherever it is installed.

Order: an importable `overlord_client`; `$OVERLORD_SDK` (a file or a dir);
the installed engine (/usr/local/lib/overlord); a sibling checkout."""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path

_CANDIDATES = (
    "/usr/local/lib/overlord/overlord_client.py",
    str(Path.home() / "overlord" / "sdk" / "overlord_client.py"),
    str(Path.home() / "projects" / "overlord" / "sdk" / "overlord_client.py"),
    "/home/user/overlord/sdk/overlord_client.py",
)


def load_sdk():
    """Return the `overlord_client` module. Raises ImportError with the
    places it looked."""
    try:
        return importlib.import_module("overlord_client")
    except ImportError:
        pass
    looked = []
    env = os.environ.get("OVERLORD_SDK")
    paths = []
    if env:
        p = Path(env)
        paths.append(p / "overlord_client.py" if p.is_dir() else p)
    paths += [Path(c) for c in _CANDIDATES]
    for p in paths:
        looked.append(str(p))
        if p.is_file():
            spec = importlib.util.spec_from_file_location("overlord_client", p)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["overlord_client"] = mod
            spec.loader.exec_module(mod)
            return mod
    raise ImportError("OVERLORD SDK not found; set OVERLORD_SDK. Looked in: " + ", ".join(looked))
