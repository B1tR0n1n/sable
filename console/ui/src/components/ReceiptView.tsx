import { useEffect, useState } from "react";
import { api } from "../api";
import type { Receipt, VerifyChainResult } from "../types";
import { shortHash, stamp } from "../format";
import { stepCounts, verifyClass } from "./ReceiptsList";

export interface ReceiptViewProps {
  receipt: Receipt | null;
  chain: VerifyChainResult | null;
  onBack: () => void;
}

/** Route `/receipts/:id`. `ReceiptRoute` loads; `ReceiptView` renders. */
export function ReceiptRoute({
  id,
  cached,
  onBack,
}: {
  id: string;
  cached: Receipt | null;
  onBack: () => void;
}) {
  const [receipt, setReceipt] = useState<Receipt | null>(cached);
  const [chain, setChain] = useState<VerifyChainResult | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    api
      .receipt(id)
      .then((r) => alive && setReceipt(r))
      .catch((e: Error) => alive && !cached && setErr(e.message));
    api
      .verifyChain()
      .then((c) => alive && setChain(c))
      .catch(() => alive && setChain(null));
    return () => {
      alive = false;
    };
  }, [id, cached]);

  if (!receipt) {
    return (
      <div className="receipt-view">
        <button className="quiet" onClick={onBack}>
          ← Console
        </button>
        <p className="empty">{err ? `Receipt ${id}: ${err}` : "Loading receipt…"}</p>
      </div>
    );
  }
  return <ReceiptView receipt={receipt} chain={chain} onBack={onBack} />;
}

export function ReceiptView({ receipt: r, chain, onBack }: ReceiptViewProps) {
  if (!r) return null;
  const v = r.verification;
  const c = stepCounts(r);
  const chainClass = chain ? (chain.ok ? "tag-green" : "tag-red") : "tag-dim";
  return (
    <div className="receipt-view" data-testid="receipt-view">
      <header>
        <div>
          <span className="label">06 — Receipt</span>
          <h1 style={{ marginTop: 4 }}>{r.id}</h1>
          <div className="meta" style={{ marginTop: 4 }}>
            plan {r.plan_id} · finding {r.finding_id ?? "—"} · session {r.session_id ?? "—"} · {stamp(r.created_at)}
          </div>
        </div>
        <div className="row">
          <span className={`tag ${chainClass}`} data-testid="chain-badge" title={chain?.broken_at ? `broken at ${chain.broken_at}` : undefined}>
            {chain ? (chain.ok ? "Chain verified" : `Chain broken${chain.broken_at ? ` @ ${shortHash(chain.broken_at)}` : ""}`) : "Chain unverified"}
          </span>
          <a className="btn" href={api.receiptExportUrl(r.id, "json")} download data-testid="export-json">
            Export JSON
          </a>
          <a className="btn" href={api.receiptExportUrl(r.id, "md")} download data-testid="export-md">
            Export MD
          </a>
          <button className="quiet" onClick={onBack}>
            ← Console
          </button>
        </div>
      </header>

      <div className="section">
        <div className="section-label">
          <span className="label">Verification</span>
        </div>
        <dl className="kv">
          <dt>status</dt>
          <dd className={verifyClass(v?.status)} data-testid="verification-status" style={{ letterSpacing: 3 }}>
            {(v?.status ?? "inconclusive").toUpperCase()}
          </dd>
          <dt>observed</dt>
          <dd>{v?.observed ? JSON.stringify(v.observed) : "—"}</dd>
          <dt>checked at</dt>
          <dd>{stamp(v?.checked_at)}</dd>
          <dt>steps</dt>
          <dd>
            ok {c.ok} · <span className={c.failed ? "sem-fail" : ""}>failed {c.failed}</span> · total {r.steps.length}
          </dd>
          <dt>rollback</dt>
          <dd className={r.rollback.performed ? "sem-fail" : ""} data-testid="rollback-flag">
            {r.rollback.performed ? "PERFORMED" : "not performed"}
          </dd>
        </dl>
      </div>

      <div className="section">
        <div className="section-label">
          <span className="label">Step log</span>
        </div>
        <table className="t" data-testid="receipt-steps">
          <thead>
            <tr>
              <th>step</th>
              <th>status</th>
              <th>started</th>
              <th>ended</th>
              <th>session</th>
              <th>output digest</th>
              <th>error</th>
            </tr>
          </thead>
          <tbody>
            {r.steps.map((s) => (
              <tr key={s.step_id}>
                <td className="bright">{s.step_id}</td>
                <td className={s.status === "ok" ? "sem-pass" : s.status === "failed" || s.status === "timed_out" || s.status === "precondition_failed" ? "sem-fail" : "dim"}>
                  {s.status}
                </td>
                <td>{stamp(s.started_at)}</td>
                <td>{stamp(s.ended_at)}</td>
                <td>{s.session_id ?? "—"}</td>
                <td title={s.output_digest ?? undefined}>{shortHash(s.output_digest, 16)}</td>
                <td className="sem-fail">{s.error ?? ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="two-col">
        <div className="section">
          <div className="section-label">
            <span className="label">Snapshots</span>
          </div>
          {r.snapshots.length === 0 ? (
            <span className="meta">none</span>
          ) : (
            <dl className="kv">
              {r.snapshots.map((s) => (
                <span key={s.node_id} style={{ display: "contents" }}>
                  <dt>{s.node_id}</dt>
                  <dd>{s.snapshot_ref}</dd>
                </span>
              ))}
            </dl>
          )}
        </div>
        <div className="section">
          <div className="section-label">
            <span className="label">Approvals</span>
          </div>
          {r.approvals.length === 0 ? (
            <span className="meta">none</span>
          ) : (
            <table className="t">
              <thead>
                <tr>
                  <th>actor</th>
                  <th>decision</th>
                  <th>at</th>
                </tr>
              </thead>
              <tbody>
                {r.approvals.map((a, i) => (
                  <tr key={i}>
                    <td className="bright">{a.actor}</td>
                    <td>{a.decision}</td>
                    <td>{stamp(a.timestamp)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {r.rollback.steps.length > 0 && (
        <div className="section">
          <div className="section-label">
            <span className="label">Rollback steps</span>
          </div>
          <table className="t">
            <thead>
              <tr>
                <th>step</th>
                <th>action</th>
                <th>status</th>
                <th>session</th>
                <th>error</th>
              </tr>
            </thead>
            <tbody>
              {r.rollback.steps.map((s) => (
                <tr key={s.step_id}>
                  <td>{s.step_id}</td>
                  <td>{s.action_id}</td>
                  <td className={s.status === "ok" ? "sem-pass" : "sem-fail"}>{s.status}</td>
                  <td>{s.session_id ?? "—"}</td>
                  <td className="sem-fail">{s.error ?? ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="section">
        <div className="section-label">
          <span className="label">Hash chain</span>
        </div>
        <div className="hash-chain">
          <span className="dim">prev</span>
          <span className="arrow">→</span>
          <span>{r.prev_receipt_hash ?? "∅ (genesis)"}</span>
          <span className="dim">this</span>
          <span className="arrow">→</span>
          <span className="bright" data-testid="receipt-hash">
            {r.receipt_hash ?? "—"}
          </span>
          <span className="dim">audit</span>
          <span className="arrow">→</span>
          <span>{r.audit_ref ? JSON.stringify(r.audit_ref) : "— (not mirrored)"}</span>
        </div>
      </div>
    </div>
  );
}
