import type { Receipt } from "../types";
import { age, shortHash } from "../format";

export function verifyClass(status: string | undefined | null): string {
  return status === "pass" ? "sem-pass" : status === "fail" ? "sem-fail" : "sem-inconclusive";
}

export function stepCounts(r: Receipt): { ok: number; failed: number } {
  let ok = 0;
  let failed = 0;
  for (const s of r.steps) {
    if (s.status === "ok") ok += 1;
    else if (s.status === "failed" || s.status === "timed_out" || s.status === "precondition_failed") failed += 1;
  }
  return { ok, failed };
}

export function ReceiptsList({
  receipts,
  onOpen,
  now,
}: {
  receipts: Receipt[];
  onOpen: (id: string) => void;
  now?: number;
}) {
  return (
    <section className="panel" data-testid="receipts-list">
      <div className="panel-head">
        <span className="label">05 — Receipts</span>
        <span className="meta">{receipts.length}</span>
      </div>
      <div className="panel-body flush">
        {receipts.length === 0 && <p className="empty">No receipts yet.</p>}
        {receipts.map((r) => {
          const c = stepCounts(r);
          const v = r.verification?.status ?? "inconclusive";
          return (
            <div
              key={r.id}
              className="receipt-row"
              onClick={() => onOpen(r.id)}
              role="link"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === "Enter") onOpen(r.id);
              }}
              data-testid={`receipt-${r.id}`}
            >
              <span className="bright">{r.id}</span>
              <span className={verifyClass(v)} data-testid="receipt-status">
                {v.toUpperCase()}
              </span>
              <div className="sub">
                <span>{shortHash(r.receipt_hash)}</span>
                <span>
                  ok {c.ok} · <span className={c.failed ? "sem-fail" : ""}>failed {c.failed}</span>
                </span>
                {r.rollback.performed && <span className="tag tag-red">rolled back</span>}
                <span>{age(r.created_at, now)}</span>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
