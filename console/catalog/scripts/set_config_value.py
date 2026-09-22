#!/usr/bin/env python3
"""Set KEY=VALUE in a simple `key=value` config file, idempotently.

    set_config_value.py FILE KEY VALUE

- an existing `KEY=...` line (leading whitespace tolerated) is replaced in place
- otherwise the line is appended
- if the file already holds exactly KEY=VALUE nothing is written (exit 0)
- the file is created if missing; its parent directory must exist

The file lives inside the lab directory, so OVERLORD's transactional session
records the before/after objects and can revert it (catalog: reversible).
No shell is involved: the executor passes argv straight through.
"""
import re
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print("usage: set_config_value.py FILE KEY VALUE", file=sys.stderr)
        return 2
    path, key, value = Path(argv[1]), argv[2], argv[3]
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", key):
        print(f"refusing key {key!r}: not an identifier", file=sys.stderr)
        return 2
    if "\n" in value or "\r" in value:
        print("refusing value: contains a newline", file=sys.stderr)
        return 2
    lines = path.read_text().splitlines() if path.exists() else []
    pattern = re.compile(r"^(\s*)" + re.escape(key) + r"\s*=.*$")
    new_line, replaced = f"{key}={value}", False
    out = []
    for line in lines:
        m = pattern.match(line)
        if m and not replaced:
            out.append(m.group(1) + new_line)
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(new_line)
    text = "\n".join(out) + "\n"
    if path.exists() and path.read_text() == text:
        print(f"{path}: {key} already {value}")
        return 0
    path.write_text(text)
    print(f"{path}: {'set' if replaced else 'added'} {key}={value}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
