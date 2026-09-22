import { useEffect, useRef, useState } from "react";
import type { LogLine } from "../types";
import { clock } from "../format";

export function SessionLog({ lines }: { lines: LogLine[] }) {
  const ref = useRef<HTMLDivElement>(null);
  const [paused, setPaused] = useState(false);

  useEffect(() => {
    if (paused) return;
    const el = ref.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [lines, paused]);

  return (
    <section className="panel" data-testid="session-log">
      <div className="panel-head">
        <span className="label">04 — Session log</span>
        <span className="meta">
          {lines.length} lines{paused ? " · PAUSED" : ""}
        </span>
      </div>
      <div
        className="log"
        ref={ref}
        onMouseEnter={() => setPaused(true)}
        onMouseLeave={() => setPaused(false)}
        aria-live="polite"
      >
        {lines.length === 0 && <span className="dim">— no log lines yet —</span>}
        {lines.map((l, i) => (
          <div key={`${l.ts}-${i}`} className={`log-line lv-${(l.level || "info").toLowerCase()}`}>
            <span className="ts">{clock(l.ts)}</span>
            <span className="src" title={l.source}>
              {l.source}
            </span>
            <span className="txt">{l.text}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
