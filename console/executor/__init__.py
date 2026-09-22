"""Phase 4 — the OVERLORD executor: Plans become Receipts, transactionally.

    Executor(client, catalog, context).execute(plan, finding, approvals) -> Receipt

One OVERLORD session per step, opened with the minimum scope the catalog
binding declares: precondition → action (with `cause` naming the plan step
on provenance) → commit; a failure rolls that session back and runs the
completed steps' compensations in reverse order. Every run — pass or fail —
produces a Receipt, sealed into the hash chain and mirrored onto OVERLORD's
own audit chain.
"""

from .executor import ExecutionError, Executor  # noqa: F401
from .overlord_sdk import load_sdk  # noqa: F401
from .receipts import ReceiptChain  # noqa: F401
