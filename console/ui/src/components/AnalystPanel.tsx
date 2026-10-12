import { useState } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { api } from "../api";
import type { AnalystToolCall, AnalystTurn, ConsoleState, Finding } from "../types";

interface Exchange {
  role: "user" | "assistant";
  content: string;
  tool_calls?: AnalystToolCall[];
  provider?: string;
  model?: string | null;
  context?: string[];
  generated?: boolean;
}

/** Split a reply into SABLE's data and the model's text. The server puts
 *  SABLE's facts in ONE ```sable fence at the very start of the reply, before
 *  the model's answer; only that block is data. Everything after it is
 *  generated markdown — a fence the model writes itself (even one tagged
 *  `sable`) stays inside the generated text as a code block, so model output
 *  can never pass for SABLE data. */
export function splitReply(text: string): Array<{ kind: "sable" | "prose"; text: string }> {
  const out: Array<{ kind: "sable" | "prose"; text: string }> = [];
  const m = /^```sable[^\n]*\n([\s\S]*?)```[^\n]*(?:\n|$)/.exec(text);
  if (m) out.push({ kind: "sable", text: m[1].replace(/\s+$/, "") });
  const rest = (m ? text.slice(m[0].length) : text).trim();
  if (rest) out.push({ kind: "prose", text: rest });
  return out;
}

/** Links in generated text leave the console in a new tab, and never carry a referrer. */
const MD_COMPONENTS: Components = {
  a: ({ node: _node, ...props }) => <a {...props} target="_blank" rel="noopener noreferrer" />,
};

export function chipLabel(tc: AnalystToolCall): string {
  const args = Object.entries(tc.input ?? {})
    .map(([k, v]) => `${k}=${typeof v === "string" ? v : JSON.stringify(v)}`)
    .join(" ");
  return args ? `${tc.name} ${args}` : tc.name;
}

export interface AnalystPanelProps {
  open: boolean;
  finding: Finding | null;
  /** the model the analyst runs on (from /api/state) */
  model?: ConsoleState["model"] | null;
  onOpenSettings?: () => void;
  onClose: () => void;
}

export function AnalystPanel({ open, finding, model, onOpenSettings, onClose }: AnalystPanelProps) {
  const [history, setHistory] = useState<Record<string, Exchange[]>>({});
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fid = finding?.id ?? null;
  const noKey = !!model && !model.ready;
  const turns = fid ? (history[fid] ?? []) : [];

  async function send() {
    const message = draft.trim();
    if (!fid || !message || busy || noKey) return;
    setBusy(true);
    setError(null);
    const priorTurns: AnalystTurn[] = turns.map((t) => ({ role: t.role, content: t.content }));
    setHistory((h) => ({ ...h, [fid]: [...(h[fid] ?? []), { role: "user", content: message }] }));
    setDraft("");
    try {
      const res = await api.analystChat(fid, message, priorTurns);
      setHistory((h) => ({
        ...h,
        [fid]: [
          ...(h[fid] ?? []),
          {
            role: "assistant",
            content: res.reply,
            tool_calls: res.tool_calls ?? [],
            provider: res.provider,
            model: res.model,
            context: res.context,
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
          <span className={`tag ${model?.ready ? "tag-gold" : "tag-red"}`} data-testid="provider-indicator">
            {model ? `${model.model}${model.ready ? "" : " · no key"}` : "model —"}
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
        {noKey && (
          <p className="empty" data-testid="analyst-nokey">
            No API key for {model?.provider}.{" "}
            {onOpenSettings && (
              <button className="tiny" onClick={onOpenSettings}>
                Open Settings
              </button>
            )}
          </p>
        )}
        {turns.map((t, i) => (
          <div key={i} className={`turn ${t.role}`}>
            <span className="who">
              {t.role === "user" ? "Operator" : `Analyst · generated${t.model ? ` · ${t.model}` : t.provider ? ` · ${t.provider}` : ""}`}
            </span>
            {t.role === "user" ? (
              <p className="prose">{t.content}</p>
            ) : (
              <>
                {t.context && t.context.length > 0 && (
                  <div className="chips" data-testid="context-chips" title="the records the analyst was given; it has no tools">
                    {t.context.map((c) => (
                      <span key={c} className="chip">
                        {c}
                      </span>
                    ))}
                  </div>
                )}
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
                    <div key={j} className="prose generated md" data-testid="generated-text" aria-label="ANALYST · GENERATED">
                      {/* react-markdown renders no raw HTML: model output cannot inject markup */}
                      <ReactMarkdown remarkPlugins={[remarkGfm]} components={MD_COMPONENTS}>
                        {part.text}
                      </ReactMarkdown>
                    </div>
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
          disabled={!fid || busy || noKey}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void send();
            }
          }}
          data-testid="analyst-input"
        />
        <button onClick={() => void send()} disabled={!fid || busy || noKey || !draft.trim()} data-testid="analyst-send">
          Send
        </button>
      </div>
    </aside>
  );
}
