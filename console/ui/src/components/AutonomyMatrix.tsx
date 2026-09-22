import type { Band, Policy, Reversibility } from "../types";
import { BANDS, REVERSIBILITIES } from "../types";
import { DECISION_LABEL } from "../gate";

export interface AutonomyMatrixProps {
  policy: Policy | null;
  hit: { band: Band; reversibility: Reversibility } | null;
  defaultDeny: boolean;
  onLoadTarget: () => void;
  busy?: boolean;
}

export function AutonomyMatrix({ policy, hit, defaultDeny, onLoadTarget, busy }: AutonomyMatrixProps) {
  return (
    <section className="panel" data-testid="autonomy-matrix">
      <div className="panel-head">
        <span className="label">03 — Autonomy</span>
        <span className="meta">
          {policy ? `bands ≥${policy.bands.high} / ≥${policy.bands.medium}` : "—"}
          {defaultDeny ? " · default_deny" : ""}
        </span>
      </div>
      <div className="panel-body">
        {!policy ? (
          <span className="meta">policy not loaded</span>
        ) : (
          <div className="matrix" role="grid" aria-label="confidence band by reversibility">
            <span className="mh row">conf \ rev</span>
            {REVERSIBILITIES.map((r) => (
              <span key={r} className="mh">
                {r.slice(0, 6)}
              </span>
            ))}
            {BANDS.map((b) => (
              <span key={b} style={{ display: "contents" }}>
                <span className="mh row">{b}</span>
                {REVERSIBILITIES.map((r) => {
                  const d = policy.matrix[b]?.[r];
                  const isHit = hit?.band === b && hit?.reversibility === r;
                  return (
                    <span
                      key={r}
                      role="gridcell"
                      className={`cell d-${d}${isHit ? " hit" : ""}`}
                      data-testid={`cell-${b}-${r}`}
                      data-hit={isHit ? "true" : undefined}
                      title={`${b} × ${r} → ${d}`}
                    >
                      {d ? DECISION_LABEL[d] : "?"}
                    </span>
                  );
                })}
              </span>
            ))}
          </div>
        )}
        <div className="row spread" style={{ marginTop: 10 }}>
          <span className="meta">irreversible is never auto (hard rule)</span>
          <button className="tiny quiet" onClick={onLoadTarget} disabled={busy} data-testid="btn-load-target">
            Load target matrix
          </button>
        </div>
      </div>
    </section>
  );
}
