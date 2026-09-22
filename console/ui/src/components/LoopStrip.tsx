import type { ConsoleState, LoopStation } from "../types";
import { STATIONS } from "../types";
import type { WsStatus } from "../api";

export interface LoopStripProps {
  state: ConsoleState | null;
  active: LoopStation | null;
  ws: WsStatus;
  polling: boolean;
  onOpenAnalyst: () => void;
}

export function LoopStrip({ state, active, ws, polling, onOpenAnalyst }: LoopStripProps) {
  const sable = state?.sable;
  const overlord = state?.overlord;
  const counts = state?.counts;
  const defaultDeny = state?.policy?.default_deny ?? false;

  return (
    <header className="strip" data-testid="loop-strip">
      <div className="strip-brand">
        <span className="label">00 — Console</span>
        <h1>
          OVERLORD <span className="gold">×</span> SABLE
        </h1>
        <span className="meta">
          <span className={`dot ${ws === "open" ? "ok" : polling ? "" : "down"}`} />
          {ws === "open" ? "LIVE /ws" : polling ? "POLLING 5s" : "OFFLINE"}
        </span>
      </div>

      <nav className="stations" aria-label="loop state">
        {STATIONS.map((s, i) => (
          <span key={s} style={{ display: "contents" }}>
            {i > 0 && <span className="station-arrow">→</span>}
            <span
              className={`station${active === s ? " lit" : ""}`}
              data-testid={`station-${s}`}
              aria-current={active === s ? "step" : undefined}
            >
              {s}
            </span>
          </span>
        ))}
      </nav>

      <div className="strip-status">
        <div className="stat">
          <span className="label">SABLE</span>
          <span>
            <span className={`dot ${sable ? (sable.ok ? "ok" : "down") : ""}`} />
            {sable ? (sable.ok ? "ok" : "down") : "—"}
            {sable?.provider ? ` · ${sable.provider}` : ""}
            {sable?.device ? ` · ${sable.device}` : ""}
          </span>
        </div>
        <div className="stat">
          <span className="label">OVERLORD</span>
          <span>
            <span className={`dot ${overlord ? (overlord.ok ? "ok" : "down") : ""}`} />
            {overlord ? (overlord.ok ? "ok" : "down") : "—"}
            {overlord?.version ? ` · ${overlord.version}` : ""}
          </span>
        </div>
        <div className="stat">
          <span className="label">Counts</span>
          <span>
            open {counts?.findings_open ?? "—"} · pending {counts?.plans_pending ?? "—"} · receipts{" "}
            {counts?.receipts ?? "—"}
          </span>
        </div>
        {defaultDeny && (
          <span className="default-deny" data-testid="default-deny" title="Shipped policy: every cell is human">
            default_deny
          </span>
        )}
        <button className="quiet" onClick={onOpenAnalyst} data-testid="open-analyst">
          Analyst
        </button>
      </div>
    </header>
  );
}
