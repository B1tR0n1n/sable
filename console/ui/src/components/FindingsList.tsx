import type { Finding, Severity } from "../types";
import { age } from "../format";

const GLYPH: Record<Severity, string> = {
  critical: "◆",
  high: "▲",
  medium: "■",
  low: "·",
};

export function confidenceLabel(f: Finding): string {
  const c = f.confidence;
  const method = c.method === "mc_dropout" && c.samples ? `mc_dropout n=${c.samples}` : c.method;
  return `${c.score.toFixed(2)} ${method}`;
}

export interface FindingsListProps {
  findings: Finding[];
  selectedId: string | null;
  filter: "open" | "closed";
  onFilter: (f: "open" | "closed") => void;
  onSelect: (id: string) => void;
  now?: number;
}

export function FindingsList({ findings, selectedId, filter, onFilter, onSelect, now }: FindingsListProps) {
  return (
    <section className="panel grow" data-testid="findings-list">
      <div className="panel-head">
        <span className="label">01 — Findings</span>
        <div className="filter" role="tablist">
          <button
            role="tab"
            aria-selected={filter === "open"}
            className={filter === "open" ? "on" : ""}
            onClick={() => onFilter("open")}
          >
            Open
          </button>
          <button
            role="tab"
            aria-selected={filter === "closed"}
            className={filter === "closed" ? "on" : ""}
            onClick={() => onFilter("closed")}
          >
            Closed
          </button>
        </div>
      </div>
      <div className="panel-body flush">
        {findings.length === 0 && <p className="empty">No {filter} findings.</p>}
        {findings.map((f) => (
          <FindingRow
            key={f.id}
            finding={f}
            selected={f.id === selectedId}
            onSelect={() => onSelect(f.id)}
            now={now}
          />
        ))}
      </div>
    </section>
  );
}

function FindingRow({
  finding: f,
  selected,
  onSelect,
  now,
}: {
  finding: Finding;
  selected: boolean;
  onSelect: () => void;
  now?: number;
}) {
  const gap = f.detection_mode === "unmonitored_gap";
  return (
    <div
      className={`finding-row${selected ? " selected" : ""}`}
      onClick={onSelect}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") onSelect();
      }}
      data-testid={`finding-${f.id}`}
      aria-pressed={selected}
    >
      <span className={`sev sev-${f.severity}`} title={f.severity} aria-label={`severity ${f.severity}`}>
        {GLYPH[f.severity] ?? "·"}
      </span>
      <span className="finding-title">
        {f.root_cause.node_id}
        <span className="dim"> · {f.root_cause.component_type} · {f.root_cause.state}</span>
      </span>
      <span className="meta">{age(f.created_at, now)}</span>
      <div className="finding-sub">
        <span className={`tag ${gap ? "tag-dim" : "tag-gold"}`} data-testid="mode-tag">
          {gap ? "GAP" : "LIVE"}
        </span>
        <span className="meta" title="confidence · method">
          {confidenceLabel(f)}
        </span>
        <span className="meta">×{f.occurrences}</span>
        {f.summary_generated && (
          <span className="tag generated" data-testid="row-generated-tag" title="summary written by a model">
            GENERATED
          </span>
        )}
        {f.status === "reopened" && <span className="tag tag-red">reopened</span>}
      </div>
    </div>
  );
}
