// A tiny event-driven store: one immutable snapshot, subscribers, and
// `apply(event)` for the /ws stream. No library; React reads it through
// `useStore` (useSyncExternalStore).

import { useSyncExternalStore } from "react";
import type {
  Approval,
  ConsoleState,
  Finding,
  LogLine,
  LoopStation,
  Plan,
  PlanStatus,
  Receipt,
  StepResult,
  VerificationResult,
  WsEvent,
} from "./types";
import { CLOSED_STATUSES } from "./types";

export const LOG_RING_SIZE = 500;

export interface ExecutionProgress {
  steps: Record<string, StepResult>; // step_id -> latest result
  verification?: VerificationResult;
  receipt_id?: string;
}

export interface Snapshot {
  state: ConsoleState | null;
  findings: Record<string, Finding>;
  plans: Record<string, Plan>; // by plan id
  planByFinding: Record<string, string>; // finding id -> newest plan id
  approvals: Record<string, Approval[]>; // plan id -> approvals
  planStatus: Record<string, PlanStatus>; // plan id -> server lifecycle
  progress: Record<string, ExecutionProgress>; // plan id -> live execution
  receipts: Receipt[]; // newest first
  log: LogLine[]; // ring buffer, oldest first
  countdowns: Record<string, number>; // plan id -> remaining_s
  activeStation: LoopStation | null;
  activeStationAt: number;
  lastEventAt: number;
}

export const emptySnapshot = (): Snapshot => ({
  state: null,
  findings: {},
  plans: {},
  planByFinding: {},
  approvals: {},
  planStatus: {},
  progress: {},
  receipts: [],
  log: [],
  countdowns: {},
  activeStation: null,
  activeStationAt: 0,
  lastEventAt: 0,
});

type Listener = () => void;

const byNewest = <T extends { created_at: string }>(a: T, b: T) =>
  a.created_at < b.created_at ? 1 : a.created_at > b.created_at ? -1 : 0;

export class Store {
  private snap: Snapshot = emptySnapshot();
  private listeners = new Set<Listener>();

  get = (): Snapshot => this.snap;

  subscribe = (l: Listener): (() => void) => {
    this.listeners.add(l);
    return () => this.listeners.delete(l);
  };

  private set(patch: Partial<Snapshot>): void {
    this.snap = { ...this.snap, ...patch };
    for (const l of this.listeners) l();
  }

  reset(): void {
    this.set(emptySnapshot());
  }

  // -------------------------------------------------------------- bulk loads

  setState(state: ConsoleState): void {
    this.set({ state });
  }

  setFindings(list: Finding[]): void {
    const findings = { ...this.snap.findings };
    for (const f of list) findings[f.id] = f;
    this.set({ findings });
  }

  setPlan(plan: Plan, status?: PlanStatus): void {
    const s = status ?? plan._status;
    this.set({
      plans: { ...this.snap.plans, [plan.id]: plan },
      planByFinding: { ...this.snap.planByFinding, [plan.finding_id]: plan.id },
      ...(s ? { planStatus: { ...this.snap.planStatus, [plan.id]: s } } : {}),
    });
  }

  setPlanStatus(planId: string, status: PlanStatus): void {
    if (this.snap.planStatus[planId] === status) return;
    this.set({ planStatus: { ...this.snap.planStatus, [planId]: status } });
  }

  setReceipts(list: Receipt[]): void {
    this.set({ receipts: mergeReceipts(this.snap.receipts, list) });
  }

  setLog(lines: LogLine[]): void {
    this.set({ log: lines.slice(-LOG_RING_SIZE) });
  }

  clearCountdown(planId: string): void {
    if (!(planId in this.snap.countdowns)) return;
    const countdowns = { ...this.snap.countdowns };
    delete countdowns[planId];
    this.set({ countdowns });
  }

  // -------------------------------------------------------------- events

  apply(ev: WsEvent): void {
    const now = Date.now();
    switch (ev.type) {
      case "hello":
      case "state":
        this.set({ state: ev.state, lastEventAt: now });
        return;

      case "finding": {
        const f = ev.finding;
        const station: LoopStation = f.status === "closed" ? "RECEIPT" : "DETECT";
        this.set({
          findings: { ...this.snap.findings, [f.id]: f },
          activeStation: station,
          activeStationAt: now,
          lastEventAt: now,
        });
        return;
      }

      case "plan": {
        const p = ev.plan;
        const countdowns = { ...this.snap.countdowns };
        if (p.gate?.decision !== "delay") delete countdowns[p.id];
        const s = ev.status ?? p._status;
        this.set({
          plans: { ...this.snap.plans, [p.id]: p },
          planByFinding: { ...this.snap.planByFinding, [p.finding_id]: p.id },
          ...(s ? { planStatus: { ...this.snap.planStatus, [p.id]: s } } : {}),
          countdowns,
          activeStation: p.gate ? "GATE" : "PROPOSE",
          activeStationAt: now,
          lastEventAt: now,
        });
        return;
      }

      case "plan_status": {
        const countdowns = { ...this.snap.countdowns };
        if (ev.status !== "countdown") delete countdowns[ev.plan_id];
        this.set({
          planStatus: { ...this.snap.planStatus, [ev.plan_id]: ev.status },
          countdowns,
          lastEventAt: now,
        });
        return;
      }

      case "approval": {
        const prev = this.snap.approvals[ev.plan_id] ?? [];
        // the POST response and the /ws echo carry the same approval: keep one
        const a = ev.approval;
        if (prev.some((p) => p.actor === a.actor && p.decision === a.decision && p.timestamp === a.timestamp)) return;
        this.set({
          approvals: { ...this.snap.approvals, [ev.plan_id]: [...prev, a] },
          activeStation: "GATE",
          activeStationAt: now,
          lastEventAt: now,
        });
        return;
      }

      case "countdown":
        this.set({
          countdowns: { ...this.snap.countdowns, [ev.plan_id]: ev.remaining_s },
          activeStation: "GATE",
          activeStationAt: now,
          lastEventAt: now,
        });
        return;

      case "step": {
        const prev = this.snap.progress[ev.plan_id] ?? { steps: {} };
        const countdowns = { ...this.snap.countdowns };
        delete countdowns[ev.plan_id];
        this.set({
          progress: {
            ...this.snap.progress,
            [ev.plan_id]: { ...prev, steps: { ...prev.steps, [ev.step.step_id]: ev.step } },
          },
          countdowns,
          activeStation: "EXECUTE",
          activeStationAt: now,
          lastEventAt: now,
        });
        return;
      }

      case "verification": {
        const prev = this.snap.progress[ev.plan_id] ?? { steps: {} };
        this.set({
          progress: {
            ...this.snap.progress,
            [ev.plan_id]: { ...prev, verification: ev.result, receipt_id: ev.receipt_id },
          },
          activeStation: "VERIFY",
          activeStationAt: now,
          lastEventAt: now,
        });
        return;
      }

      case "receipt":
        this.set({
          receipts: mergeReceipts(this.snap.receipts, [ev.receipt]),
          activeStation: "RECEIPT",
          activeStationAt: now,
          lastEventAt: now,
        });
        return;

      case "log": {
        const log = [...this.snap.log, ev.line];
        if (log.length > LOG_RING_SIZE) log.splice(0, log.length - LOG_RING_SIZE);
        this.set({ log, lastEventAt: now });
        return;
      }

      default:
        // Unknown event types are ignored on purpose: the wire may grow.
        return;
    }
  }
}

function mergeReceipts(existing: Receipt[], incoming: Receipt[]): Receipt[] {
  const byId = new Map<string, Receipt>();
  for (const r of existing) byId.set(r.id, r);
  for (const r of incoming) byId.set(r.id, r);
  return [...byId.values()].sort(byNewest);
}

// -------------------------------------------------------------- selectors

export function openFindings(s: Snapshot): Finding[] {
  return Object.values(s.findings)
    .filter((f) => !CLOSED_STATUSES.has(f.status))
    .sort(byNewest);
}

export function closedFindings(s: Snapshot): Finding[] {
  return Object.values(s.findings)
    .filter((f) => CLOSED_STATUSES.has(f.status))
    .sort(byNewest);
}

export function planForFinding(s: Snapshot, findingId: string | null): Plan | null {
  if (!findingId) return null;
  const pid = s.planByFinding[findingId];
  return pid ? (s.plans[pid] ?? null) : null;
}

// -------------------------------------------------------------- React glue

export const store = new Store();

export function useStore<T>(selector: (s: Snapshot) => T, st: Store = store): T {
  return useSyncExternalStore(st.subscribe, () => selector(st.get()), () => selector(st.get()));
}
