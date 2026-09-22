import { describe, expect, it } from "vitest";
import { Store, LOG_RING_SIZE, openFindings, planForFinding } from "../store";
import type { Finding, Plan, Receipt, ConsoleState } from "../types";
import findingFx from "./fixtures/finding.json";
import planFx from "./fixtures/plan.json";
import receiptFx from "./fixtures/receipt.json";

const finding = findingFx as Finding;
const plan = planFx as Plan;
const receipt = receiptFx as Receipt;
const ts = "2026-09-22T14:00:00Z";

const state: ConsoleState = {
  sable: { url: "http://127.0.0.1:8080", ok: true, provider: "claude", device: "cpu" },
  overlord: { socket: "/tmp/overlord.sock", ok: true, version: "0.9" },
  counts: { findings_open: 1, plans_pending: 1, receipts: 0 },
  policy: {
    bands: { high: 0.85, medium: 0.6 },
    matrix: {
      high: { reversible: "human", compensable: "human", irreversible: "human" },
      medium: { reversible: "human", compensable: "human", irreversible: "human" },
      low: { reversible: "human", compensable: "human", irreversible: "human" },
    },
    default_deny: true,
  },
  lab: true,
};

describe("store.apply", () => {
  it("hello/state events set the strip state", () => {
    const s = new Store();
    s.apply({ type: "hello", ts, state });
    expect(s.get().state?.sable.provider).toBe("claude");
    expect(s.get().state?.policy.default_deny).toBe(true);
    s.apply({ type: "state", ts, state: { ...state, counts: { ...state.counts, receipts: 3 } } });
    expect(s.get().state?.counts.receipts).toBe(3);
  });

  it("finding events upsert by id, newest first, and light DETECT", () => {
    const s = new Store();
    s.apply({ type: "finding", ts, finding });
    expect(openFindings(s.get()).map((f) => f.id)).toEqual([finding.id]);
    expect(s.get().activeStation).toBe("DETECT");

    const newer: Finding = { ...finding, id: "fnd-2", created_at: "2026-09-22T15:00:00Z" };
    s.apply({ type: "finding", ts, finding: newer });
    expect(openFindings(s.get()).map((f) => f.id)).toEqual(["fnd-2", finding.id]);

    s.apply({ type: "finding", ts, finding: { ...finding, status: "closed", occurrences: 2 } });
    expect(openFindings(s.get()).map((f) => f.id)).toEqual(["fnd-2"]);
    expect(s.get().findings[finding.id].occurrences).toBe(2);
    expect(s.get().activeStation).toBe("RECEIPT");
  });

  it("plan events index by finding and light GATE when gated", () => {
    const s = new Store();
    s.apply({ type: "finding", ts, finding });
    s.apply({ type: "plan", ts, plan });
    expect(planForFinding(s.get(), finding.id)?.id).toBe(plan.id);
    expect(s.get().plans[plan.id].gate?.rule_id).toBe("default-deny");
    expect(s.get().activeStation).toBe("GATE");

    s.apply({ type: "plan", ts, plan: { ...plan, id: "pln-2", gate: null } });
    expect(planForFinding(s.get(), finding.id)?.id).toBe("pln-2");
    expect(s.get().activeStation).toBe("PROPOSE");
  });

  it("countdown events track remaining seconds per plan and clear on execution", () => {
    const s = new Store();
    s.apply({ type: "plan", ts, plan: { ...plan, gate: { decision: "delay", rule_id: "matrix:high:compensable", reason: "r", delay_s: 120 } } });
    s.apply({ type: "countdown", ts, plan_id: plan.id, remaining_s: 120 });
    s.apply({ type: "countdown", ts, plan_id: plan.id, remaining_s: 119 });
    expect(s.get().countdowns[plan.id]).toBe(119);
    expect(s.get().activeStation).toBe("GATE");

    s.apply({ type: "step", ts, plan_id: plan.id, step: { step_id: "s1", status: "running" } });
    expect(s.get().countdowns[plan.id]).toBeUndefined();
    expect(s.get().activeStation).toBe("EXECUTE");
  });

  it("step events accumulate per plan by step_id (latest wins)", () => {
    const s = new Store();
    s.apply({ type: "step", ts, plan_id: plan.id, step: { step_id: "s1", status: "running", started_at: ts } });
    s.apply({ type: "step", ts, plan_id: plan.id, step: receipt.steps[0] });
    s.apply({ type: "step", ts, plan_id: plan.id, step: { step_id: "s2", status: "failed", error: "boom" } });
    const prog = s.get().progress[plan.id];
    expect(prog.steps.s1.status).toBe("ok");
    expect(prog.steps.s1.output_digest).toBe(receipt.steps[0].output_digest);
    expect(prog.steps.s2.error).toBe("boom");
  });

  it("verification events attach the result and light VERIFY", () => {
    const s = new Store();
    s.apply({
      type: "verification",
      ts,
      plan_id: plan.id,
      receipt_id: receipt.id,
      result: { status: "fail", observed: { "dns-1": "failed" }, checked_at: ts },
    });
    expect(s.get().progress[plan.id].verification?.status).toBe("fail");
    expect(s.get().progress[plan.id].receipt_id).toBe(receipt.id);
    expect(s.get().activeStation).toBe("VERIFY");
  });

  it("receipt events dedupe by id and keep newest first", () => {
    const s = new Store();
    s.apply({ type: "receipt", ts, receipt });
    s.apply({ type: "receipt", ts, receipt: { ...receipt, id: "rcp-2", created_at: "2026-09-22T16:00:00Z" } });
    s.apply({ type: "receipt", ts, receipt: { ...receipt, rollback: { performed: true, steps: [] } } });
    const list = s.get().receipts;
    expect(list.map((r) => r.id)).toEqual(["rcp-2", receipt.id]);
    expect(list[1].rollback.performed).toBe(true);
    expect(s.get().activeStation).toBe("RECEIPT");
  });

  it("approval events append per plan", () => {
    const s = new Store();
    s.apply({ type: "approval", ts, plan_id: plan.id, approval: { actor: "op", decision: "hold", timestamp: ts } });
    s.apply({ type: "approval", ts, plan_id: plan.id, approval: { actor: "op", decision: "approve", timestamp: ts } });
    expect(s.get().approvals[plan.id].map((a) => a.decision)).toEqual(["hold", "approve"]);
  });

  it("log events fill a ring buffer", () => {
    const s = new Store();
    for (let i = 0; i < LOG_RING_SIZE + 25; i++) {
      s.apply({ type: "log", ts, line: { ts, level: "info", source: "executor", text: `line ${i}` } });
    }
    const log = s.get().log;
    expect(log).toHaveLength(LOG_RING_SIZE);
    expect(log[0].text).toBe("line 25");
    expect(log[log.length - 1].text).toBe(`line ${LOG_RING_SIZE + 24}`);
  });

  it("notifies subscribers once per applied event", () => {
    const s = new Store();
    let n = 0;
    const off = s.subscribe(() => n++);
    s.apply({ type: "finding", ts, finding });
    s.apply({ type: "plan", ts, plan });
    expect(n).toBe(2);
    off();
    s.apply({ type: "receipt", ts, receipt });
    expect(n).toBe(2);
  });

  it("ignores unknown event types", () => {
    const s = new Store();
    // @ts-expect-error — the wire may carry types this build does not know
    s.apply({ type: "telemetry", ts, payload: 1 });
    expect(s.get().lastEventAt).toBe(0);
  });
});
