"""In-memory Finding store with the dedup rule, change notifications and
optional JSON persistence.

Dedup (PLAN.md Phase 2): a persisting condition updates its open Finding
rather than creating a new one. "Same condition" is `Finding.dedup_key`
(site, root node, root state); "open" is status open or reopened.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

from console.contracts import Finding, FindingStatus

log = logging.getLogger(__name__)

OPEN_STATUSES = (FindingStatus.open.value, FindingStatus.reopened.value)

Subscriber = Callable[[Finding], None]


class FindingStore:
    def __init__(self, path: Optional[str | os.PathLike] = None):
        self._path = Path(path) if path else None
        self._lock = threading.Lock()
        self._findings: dict[str, Finding] = {}
        self._subscribers: list[Subscriber] = []
        if self._path and self._path.exists():
            self._load()

    # ---------------------------------------------------------------- reads

    def get(self, finding_id: str) -> Optional[Finding]:
        with self._lock:
            return self._findings.get(finding_id)

    def list(self, status: Optional[str | FindingStatus] = None) -> list[Finding]:
        """Newest first (by updated_at, else created_at)."""
        want = _status_value(status)
        with self._lock:
            items = [f for f in self._findings.values() if want is None or f.status == want]
        return sorted(items, key=_recency, reverse=True)

    def find_open(self, dedup_key: tuple[str, str, str]) -> Optional[Finding]:
        with self._lock:
            return self._find_open_locked(dedup_key)

    def __len__(self) -> int:
        with self._lock:
            return len(self._findings)

    # ---------------------------------------------------------------- writes

    def upsert(self, finding: Finding) -> Finding:
        """Store a Finding, applying the dedup rule.

        - an id already stored is replaced as given (the mapper's `previous`
          path has already folded the occurrence in);
        - otherwise an OPEN finding with the same dedup_key absorbs it:
          occurrences+1, updated_at=now, evidence/affected/confidence/severity/
          summary refreshed, id/status/receipts kept;
        - otherwise it is inserted.
        Returns the stored Finding and notifies subscribers."""
        with self._lock:
            if finding.id in self._findings:
                stored = finding
            else:
                existing = self._find_open_locked(finding.dedup_key)
                if existing is not None:
                    stored = existing.model_copy(update={
                        "occurrences": existing.occurrences + 1,
                        "updated_at": _now(),
                        "affected_nodes": finding.affected_nodes,
                        "evidence": finding.evidence,
                        "confidence": finding.confidence,
                        "severity": finding.severity,
                        "summary": finding.summary,
                        "summary_generated": finding.summary_generated,
                    })
                else:
                    stored = finding
            self._findings[stored.id] = stored
            self._persist_locked()
        self._notify(stored)
        return stored

    def close(self, finding_id: str, receipt_id: Optional[str] = None) -> Finding:
        return self._transition(finding_id, FindingStatus.closed, receipt_id)

    def reopen(self, finding_id: str, receipt_id: Optional[str] = None) -> Finding:
        return self._transition(finding_id, FindingStatus.reopened, receipt_id)

    def _transition(self, finding_id: str, status: FindingStatus, receipt_id: Optional[str]) -> Finding:
        with self._lock:
            f = self._findings.get(finding_id)
            if f is None:
                raise KeyError(finding_id)
            receipts = list(f.receipt_ids)
            if receipt_id and receipt_id not in receipts:
                receipts.append(receipt_id)
            f = f.model_copy(update={"status": status, "updated_at": _now(), "receipt_ids": receipts})
            self._findings[finding_id] = f
            self._persist_locked()
        self._notify(f)
        return f

    # ---------------------------------------------------------------- subscribers

    def subscribe(self, callback: Subscriber) -> Subscriber:
        with self._lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)
        return callback

    def unsubscribe(self, callback: Subscriber) -> None:
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def _notify(self, finding: Finding) -> None:
        with self._lock:
            subs = list(self._subscribers)
        for cb in subs:                       # outside the lock: a callback may read the store
            try:
                cb(finding)
            except Exception:                 # a bad subscriber must not break the emitter
                log.exception("finding subscriber %r failed", cb)

    # ---------------------------------------------------------------- persistence

    def _find_open_locked(self, dedup_key: tuple[str, str, str]) -> Optional[Finding]:
        candidates = [f for f in self._findings.values()
                      if f.status in OPEN_STATUSES and f.dedup_key == dedup_key]
        return max(candidates, key=_recency) if candidates else None

    def _persist_locked(self) -> None:
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"findings": [f.model_dump(mode="json") for f in self._findings.values()]}
        fd, tmp = tempfile.mkstemp(dir=str(self._path.parent), prefix=self._path.name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(payload, fh, indent=1, sort_keys=True)
                fh.write("\n")
            os.replace(tmp, self._path)       # atomic on POSIX
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _load(self) -> None:
        data = json.loads(self._path.read_text() or "{}")
        for raw in data.get("findings", []):
            f = Finding.model_validate(raw)
            self._findings[f.id] = f

    def load_findings(self, findings: Iterable[Finding]) -> None:
        """Bulk insert without dedup or notification (tests, migrations)."""
        with self._lock:
            for f in findings:
                self._findings[f.id] = f
            self._persist_locked()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _recency(f: Finding) -> datetime:
    return f.updated_at or f.created_at


def _status_value(status: Optional[str | FindingStatus]) -> Optional[str]:
    if status is None or status == "":
        return None
    return status.value if isinstance(status, FindingStatus) else FindingStatus(status).value
