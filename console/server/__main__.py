"""python -m console.server [--port 7780] [--bind 127.0.0.1]

Environment: SABLE_URL, OVERLORD_SOCKET (or OVERLORD_SDK to find the SDK),
CONSOLE_DATA, SITE_ID, LAB_DIR, CONSOLE_LAB=1, CONSOLE_POLICY,
CONSOLE_LLM_PROVIDER, CONSOLE_LLM_MODEL, CONSOLE_DISABLE_ACTIONS."""

import argparse
import sys

import uvicorn

from console.executor import load_sdk
from console.sable_bridge import SableClient

from .app import create_app
from .loop import Config, Loop


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="console.server")
    ap.add_argument("--port", type=int, default=7780)
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--no-overlord", action="store_true", help="serve findings/plans without executing")
    args = ap.parse_args(argv)
    cfg = Config()
    sable = SableClient(cfg.sable_url)
    overlord = None
    if not args.no_overlord:
        try:
            sdk = load_sdk()
            overlord = sdk.OverlordClient(cfg.overlord_socket, timeout=600)
            overlord.ping()
        except Exception as e:                        # noqa: BLE001 — say so and keep serving
            print(f"console: OVERLORD daemon not reachable ({e}); execution disabled", file=sys.stderr)
            overlord = None
    loop = Loop(cfg, sable, overlord)
    app = create_app(loop)
    print(f"console: http://{args.bind}:{args.port}  sable={cfg.sable_url}  "
          f"overlord={'ok' if overlord else 'off'}  lab={'on' if cfg.lab_enabled else 'off'}  "
          f"policy={'default-deny' if loop.policy.is_default_deny else 'custom'}", flush=True)
    uvicorn.run(app, host=args.bind, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
