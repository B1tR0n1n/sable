import { useState } from "react";
import { api } from "../api";
import type { AnalystToolCall, AnalystTurn, Finding } from "../types";

interface Exchange {
  role: "user" | "assistant";
  content: string;
  tool_calls?: AnalystToolCall[];
  provider?: string;
  generated?: boolean;
}

/** Split a reply into prose and SABLE data blocks. The server/model quotes
 *  engine data inside fenced blocks tagged `sable` (```sable … ```); plain
 *  fenced blocks are treated the same so raw tool output is never presented
 *  as inference. Everything else is the model's own text. */
export function splitReply(text: string): Array<{ kind: "sable" | "prose"; text: string }> {
  const out: Array<{ kind: "sable" | "prose"; text: string }> = [];
  const re = /```(?:sable|json|text)?[^\n]*\n([\s\S]*?)```/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    const before = text.slice(last, m.index).trim();
    if (before) out.push({ kind: "prose", text: before });
    out.push({ kind: "sable", text: m[1].replace(/\s+$/, "") });
    last = m.index + m[0].length;
  }
  const tail = text.slice(last).trim();
  if (tail) out.push({ kind: "prose", text: tail });
  return out;
}

export function chipLabel(tc: AnalystToolCall): string {
  const args = Object.entries(tc.input ?? {})
    .map(([k, v]) => `${k}=${typeof v === "string" ? v : JSON.stringify(v)}`)
    .join(" ");
  return args ? `${tc.name} ${args}` : tc.name;
}

export interface AnalystPanelProps {
  open: boolean;
  finding: Finding | null;
  onClose: () => void;
}

export function AnalystPanel({ open, finding, onClose }: AnalystPanelProps) {
  const [history, setHistory] = useState<Record<string, Exchange[]>>({});
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [provider, setProvider] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fid = finding?.id ?? null;
  const turns = fid ? (history[fid] ?? []) : [];

  async function send() {
    const message = draft.trim();
    if (!fid || !message || busy) return;
    setBusy(true);
    setError(null);
    const priorTurns: AnalystTurn[] = turns.map((t) => ({ role: t.role, content: t.content }));
    setHistory((h) => ({ ...h, [fid]: [...(h[fid] ?? []), { role: "user", content: message }] }));
    setDraft("");
    try {
      const res = await api.analystChat(fid, message, priorTurns);
      setProvider(res.provider);
      setHistory((h) => ({
        ...h,
        [fid]: [
          ...(h[fid] ?? []),
          {
            role: "assistant",
            content: res.reply,
            tool_calls: res.tool_calls ?? [],
            provider: res.provider,
            generated: res.generated,
          },
        ],
      }));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <aside className={`drawer${open ? " open" : ""}`} aria-hidden={!open} data-testid="analyst-panel">
      <div className="panel-head">
        <div className="row">
          <span className="label">07 — Analyst</span>
          <span className={`tag ${provider === "claude" ? "tag-gold" : "tag-dim"}`} data-testid="provider-indicator">
            {provider === "claude" ? "claude API" : provider === "local" ? "local" : provider ?? "provider —"}
          </span>
        </div>
        <div className="row">
          <span className="meta">{fid ?? "no finding selected"}</span>
          <button className="tiny quiet" onClick={onClose}>
            Close
          </button>
        </div>
      </div>

      <div className="chat" data-testid="analyst-chat">
        {!fid && <p className="empty">Select a finding; the analyst is scoped to it.</p>}
        {turns.map((t, i) => (
          <div key={i} className={`turn ${t.role}`}>
            <span className="who">
              {t.role === "user" ? "Operator" : `Analyst · generated${t.provider ? ` · ${t.provider}` : ""}`}
            </span>
            {t.role === "user" ? (
              <p className="prose">{t.content}</p>
            ) : (
              <>
                {t.tool_calls && t.tool_calls.length > 0 && (
                  <div className="chips" data-testid="tool-chips">
                    {t.tool_calls.map((tc, j) => (
                      <span key={j} className="chip">
                        {chipLabel(tc)}
                      </span>
                    ))}
                  </div>
                )}
                {splitReply(t.content).map((part, j) =>
                  part.kind === "sable" ? (
                    <pre key={j} className="sable-block" data-testid="sable-block" aria-label="SABLE data">
                      {part.text}
                    </pre>
                  ) : (
                    <p key={j} className="prose generated" data-testid="generated-text" aria-label="ANALYST · GENERATED">
                      {part.text}
                    </p>
                  ),
                )}
              </>
            )}
          </div>
        ))}
        {busy && <span className="meta">thinking…</span>}
        {error && <pre className="error-block">{error}</pre>}
      </div>

      <div className="chat-input">
        <textarea
          aria-label="message"
          placeholder={fid ? "Ask about this finding… (Enter to send)" : "Select a finding first"}
          value={draft}
          disabled={!fid || busy}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void send();
            }
          }}
          data-testid="analyst-input"
        />
        <button onClick={() => void send()} disabled={!fid || busy || !draft.trim()} data-testid="analyst-send">
          Send
        </button>
      </div>
    </aside>
  );
}
