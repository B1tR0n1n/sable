// Typed fetch helpers for console/docs/API.md plus the /ws client.
// Every path here is the contract; the FastAPI server is written to the same
// document. Errors come back as `{"error": "<message>"}` with a 4xx/5xx.

import type {
  AnalystChatResult,
  AnalystTurn,
  ApproveDecisionBody,
  ApproveResult,
  ConsoleState,
  Finding,
  FindingStatus,
  LogLine,
  Plan,
  Policy,
  Receipt,
  VerifyChainResult,
  WsEvent,
} from "./types";

export class ApiError extends Error {
  status: number;
  body: unknown;
  constructor(status: number, message: string, body: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { "content-type": "application/json", ...(init?.headers ?? {}) },
  });
  const text = await res.text();
  let body: unknown = null;
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = text;
    }
  }
  if (!res.ok) {
    const msg =
      body && typeof body === "object" && "error" in (body as Record<string, unknown>)
        ? String((body as Record<string, unknown>).error)
        : typeof body === "string" && body
          ? body
          : `${res.status} ${res.statusText}`;
    throw new ApiError(res.status, msg, body);
  }
  return body as T;
}

const json = (body: unknown): RequestInit => ({ method: "POST", body: JSON.stringify(body) });

export const api = {
  state: () => request<ConsoleState>("/api/state"),

  findings: (status?: FindingStatus) =>
    request<Finding[]>(`/api/findings${status ? `?status=${status}` : ""}`),
  finding: (id: string) => request<Finding>(`/api/findings/${encodeURIComponent(id)}`),
  findingPlan: (id: string) =>
    request<Plan>(`/api/findings/${encodeURIComponent(id)}/plan`),
  replan: (id: string, planner: "template" | "llm") =>
    request<Plan>(`/api/findings/${encodeURIComponent(id)}/plan`, json({ planner })),

  plan: (id: string) => request<Plan>(`/api/plans/${encodeURIComponent(id)}`),
  approve: (id: string, body: ApproveDecisionBody) =>
    request<ApproveResult>(`/api/plans/${encodeURIComponent(id)}/approve`, json(body)),

  receipts: (findingId?: string) =>
    request<Receipt[]>(
      `/api/receipts${findingId ? `?finding_id=${encodeURIComponent(findingId)}` : ""}`,
    ),
  receipt: (id: string) => request<Receipt>(`/api/receipts/${encodeURIComponent(id)}`),
  receiptExportUrl: (id: string, format: "json" | "md") =>
    `/api/receipts/${encodeURIComponent(id)}/export?format=${format}`,
  verifyChain: () => request<VerifyChainResult>("/api/receipts/verify"),

  policy: () => request<Policy>("/api/policy"),
  putPolicy: (policy: Policy) =>
    request<Policy>("/api/policy", { method: "PUT", body: JSON.stringify(policy) }),

  topology: () => request<unknown>("/api/topology"),
  log: (limit = 200) => request<LogLine[]>(`/api/log?limit=${limit}`),

  analystChat: (finding_id: string, message: string, history: AnalystTurn[]) =>
    request<AnalystChatResult>("/api/analyst/chat", json({ finding_id, message, history })),

  labFault: (name: string) => request<unknown>("/api/lab/fault", json({ name })),
};

// ---------------------------------------------------------------- WebSocket

export type WsStatus = "connecting" | "open" | "closed";

export interface WsClientOptions {
  url?: string;
  onEvent: (ev: WsEvent) => void;
  onStatus?: (s: WsStatus) => void;
  /** first backoff (ms); doubles up to maxBackoffMs */
  backoffMs?: number;
  maxBackoffMs?: number;
}

export function wsUrl(path = "/ws"): string {
  if (typeof window === "undefined" || !window.location) return `ws://127.0.0.1:7780${path}`;
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}${path}`;
}

/** A reconnecting client for `/ws`. Backoff doubles on each failure and
 *  resets after a clean open. `close()` stops reconnecting. */
export class WsClient {
  private ws: WebSocket | null = null;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private attempt = 0;
  private stopped = false;
  private readonly opts: Required<WsClientOptions>;

  constructor(opts: WsClientOptions) {
    this.opts = {
      url: opts.url ?? wsUrl(),
      onEvent: opts.onEvent,
      onStatus: opts.onStatus ?? (() => {}),
      backoffMs: opts.backoffMs ?? 1000,
      maxBackoffMs: opts.maxBackoffMs ?? 30000,
    };
  }

  connect(): void {
    if (this.stopped) return;
    if (typeof WebSocket === "undefined") {
      this.opts.onStatus("closed");
      return;
    }
    this.opts.onStatus("connecting");
    let ws: WebSocket;
    try {
      ws = new WebSocket(this.opts.url);
    } catch {
      this.scheduleReconnect();
      return;
    }
    this.ws = ws;
    ws.onopen = () => {
      this.attempt = 0;
      this.opts.onStatus("open");
    };
    ws.onmessage = (m: MessageEvent) => {
      let ev: WsEvent | null = null;
      try {
        ev = JSON.parse(String(m.data)) as WsEvent;
      } catch {
        return;
      }
      if (ev && typeof ev === "object" && "type" in ev) this.opts.onEvent(ev);
    };
    ws.onerror = () => {
      /* onclose follows */
    };
    ws.onclose = () => {
      this.ws = null;
      this.opts.onStatus("closed");
      this.scheduleReconnect();
    };
  }

  private scheduleReconnect(): void {
    if (this.stopped || this.timer) return;
    const delay = Math.min(this.opts.backoffMs * 2 ** this.attempt, this.opts.maxBackoffMs);
    this.attempt += 1;
    this.timer = setTimeout(() => {
      this.timer = null;
      this.connect();
    }, delay);
  }

  close(): void {
    this.stopped = true;
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
    if (this.ws) {
      try {
        this.ws.close();
      } catch {
        /* ignore */
      }
    }
    this.ws = null;
  }
}
