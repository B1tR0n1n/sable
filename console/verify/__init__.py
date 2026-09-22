"""Phase 5 — prove fixes worked.

After a plan executes, wait `verification.window_s`, then have SABLE
re-evaluate the affected nodes from a FRESH tick:

    pass          → the finding is closed with the receipt attached
    fail          → the completed steps are compensated (OVERLORD) and the
                    finding is reopened with the failed attempt attached
    inconclusive  → escalated to a human; it never counts as a pass

The receipt's verification is written, then the receipt is sealed onto the
hash chain in its final form.
"""

from .verifier import Verdict, Verifier, evaluate  # noqa: F401
