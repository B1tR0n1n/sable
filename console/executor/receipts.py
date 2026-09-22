"""The receipt hash chain: append-only, one JSON line per sealed Receipt.

Each receipt's `prev_receipt_hash` is the previous receipt's `receipt_hash`,
and `receipt_hash` covers everything including `prev` — so changing or
removing any receipt breaks every hash after it (verify_chain). Phase 5
re-seals a receipt once verification is written; the chain stores the
FINAL sealed form, so `append` is called exactly once per receipt, after
verification (or after a rollback with no verification)."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Optional

from console.contracts import Receipt
from console.contracts.models import verify_chain


class ReceiptChain:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._receipts: list[Receipt] = []
        if self.path and self.path.is_file():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self._receipts.append(Receipt.model_validate_json(line))

    @property
    def head(self) -> Optional[str]:
        return self._receipts[-1].receipt_hash if self._receipts else None

    def append(self, receipt: Receipt) -> Receipt:
        """Persist `receipt` as the new head. A receipt already sealed against
        the current head (Executor.finalize does this so the audit mirror can
        cite the final hash) is taken as is and re-verified; anything else is
        sealed here."""
        with self._lock:
            if receipt.receipt_hash is None or receipt.prev_receipt_hash != self.head:
                receipt.seal(prev_receipt_hash=self.head)
            elif not receipt.verify_hash():
                raise ValueError("receipt hash does not verify against its content")
            self._receipts.append(receipt)
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_suffix(self.path.suffix + ".tmp")
                with open(tmp, "w") as f:
                    for r in self._receipts:
                        f.write(json.dumps(r.model_dump(mode="json"), sort_keys=True) + "\n")
                os.replace(tmp, self.path)
            return receipt

    def list(self, finding_id: str | None = None) -> list[Receipt]:
        rs = list(self._receipts)
        if finding_id:
            rs = [r for r in rs if r.finding_id == finding_id]
        return list(reversed(rs))

    def get(self, receipt_id: str) -> Optional[Receipt]:
        return next((r for r in self._receipts if r.id == receipt_id), None)

    def verify(self) -> dict:
        ok, broken = verify_chain(self._receipts)
        return {"ok": ok, "broken_at": broken, "count": len(self._receipts)}
