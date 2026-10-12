import { useState } from "react";
import type { Approval, ApproveDecisionBody, ConsoleState, Finding, Plan, PlanStatus, StepResult } from "../types";
import { PENDING_PLAN_STATUSES } from "../types";
import { DECISION_LABEL, planReversibility } from "../gate";
import { fmtNum, fmtParams, mmss } from "../format";
import type { ExecutionProgress } from "../store";

export interface PlanPanelProps {
  finding: Finding | null;
  plan: Plan | null;
  /** server lifecycle; null = not known yet (treated as pending) */
  planStatus?: PlanStatus | null;
  /** the model the LLM planner uses (from /api/state) */
  llm?: ConsoleState["model"] | null;
  progress: ExecutionProgress | null;
  approvals: Approval[];
  countdown: number | null;
  operator: string;
  onOperator: (v: string) => void;
  onDecision: (planId: string, decision: ApproveDecisionBody["decision"]) => Promise<void>;
  onReplan: (findingId: string, planner: "template" | "llm") => Promise<void>;
  /** A 422 (or any error) from the planner, shown verbatim. */
  replanError: string | null;
  busy: boolean;
}

export function PlanPanel(p: PlanPanelProps) {
  const { finding, plan } = p;
  if (!finding) {
    return (
      <section className="panel grow">
        <div className="panel-head">
          <span className="label">02 — Finding · Plan</span>
        </div>
        <p className="empty">Select a finding to see its evidence, plan and gate.</p>
      </section>
    );
  }
  return (
    <section className="panel grow" data-testid="plan-panel">
      <div className="panel-head">
        <span className="label">02 — Finding · Plan</span>
        <span className="meta">
          {finding.id} · {finding.site_id} · {finding.engine_version}
        </span>
      </div>
      <div className="panel-body">
        <FindingDetail finding={finding} />
        <hr className="rule" style={{ margin: "14px 0 18px" }} />
        {plan ? (
          <PlanBody {...p} plan={plan} finding={finding} />
        ) : (
          <div className="section">
            <div className="section-label">
              <span className="label">Plan</span>
            </div>
            <p className="empty" style={{ padding: "8px 0" }}>
              No plan yet for this finding.
            </p>
            <ReplanControls {...p} findingId={finding.id} />
          </div>
        )}
      </div>
    </section>
  );
}

function FindingDetail({ finding: f }: { finding: Finding }) {
  return (
    <>
      <div className="section">
        <div className="row" style={{ marginBottom: 8 }}>
          <span className={`tag ${f.severity === "critical" || f.severity === "high" ? "tag-red" : ""}`}>
            {f.severity}
          </span>
          <span className="tag">{f.status}</span>
          <span className={`tag ${f.detection_mode === "unmonitored_gap" ? "tag-dim" : "tag-gold"}`}>
            {f.detection_mode === "unmonitored_gap" ? "GAP" : "LIVE"}
          </span>
          {f.summary_generated && (
            <span className="tag generated" data-testid="generated-tag" title="This text was written by a model">
              GENERATED
            </span>
          )}
        </div>
        <p className="summary" data-testid="finding-summary">
          {f.summary || <span className="dim">(no summary)</span>}
        </p>
      </div>

      <div className="section">
        <div className="section-label">
          <span className="label">Root cause · Affected</span>
        </div>
        <dl className="kv">
          <dt>root</dt>
          <dd className="bright">
            {f.root_cause.node_id} · {f.root_cause.component_type} · {f.root_cause.state}
          </dd>
          <dt>affected</dt>
          <dd>
            {f.affected_nodes.length === 0
              ? "—"
              : f.affected_nodes.map((n) => (
                  <div key={n.node_id}>
                    {n.node_id} <span className="dim">· {n.component_type} · {n.state}</span>
                  </div>
                ))}
          </dd>
          <dt>confidence</dt>
          <dd>
            {f.confidence.score.toFixed(2)} · {f.confidence.method}
            {f.confidence.samples ? ` n=${f.confidence.samples}` : ""}
          </dd>
        </dl>
      </div>

      <div className="section">
        <div className="section-label">
          <span className="label">Evidence</span>
        </div>
        {f.evidence.length === 0 ? (
          <span className="meta">none attached</span>
        ) : (
          <table className="t" data-testid="evidence-table">
            <thead>
              <tr>
                <th>metric</th>
                <th>value</th>
                <th>unit</th>
                <th>threshold</th>
              </tr>
            </thead>
            <tbody>
              {f.evidence.map((e, i) => (
                <tr key={`${e.metric}-${i}`}>
                  <td>{e.metric}</td>
                  <td className="num">{fmtNum(e.value)}</td>
                  <td>{e.unit || "—"}</td>
                  <td className="num">{fmtNum(e.threshold)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </>
  );
}

function stepStatusClass(s: StepResult["status"] | undefined): string {
  if (!s) return "dim";
  if (s === "ok") return "sem-pass";
  if (s === "failed" || s === "timed_out" || s === "precondition_failed") return "sem-fail";
  if (s === "running") return "gold";
  return "dim";
}

function PlanBody(p: PlanPanelProps & { plan: Plan; finding: Finding }) {
  const { plan, planStatus, progress, approvals, countdown, operator, onOperator, onDecision, busy } = p;
  const gate = plan.gate;
  const decision = gate?.decision ?? null;
  const worst = planReversibility(plan);
  const pending = !planStatus || PENDING_PLAN_STATUSES.has(planStatus);
  const canApprove = pending && (decision === "human" || decision === "human_plus" || decision === "delay");
  const isDelay = decision === "delay";
  const inCountdown = planStatus === "countdown" || (!planStatus && countdown !== null);
  const noOperator = operator.trim().length === 0;

  return (
    <>
      <div className="section">
        <div className="section-label">
          <span className="label">Plan {plan.id}</span>
          {planStatus && (
            <span className={`tag ${pending ? "tag-gold" : "tag-dim"}`} data-testid="plan-status">
              {planStatus}
            </span>
          )}
          <span className="meta">
            {plan.planner ? `${plan.planner.kind}${plan.planner.template_id ? ` · ${plan.planner.template_id}` : ""}${plan.planner.provider ? ` · ${plan.planner.provider}` : ""}${plan.planner.model ? ` · ${plan.planner.model}` : ""}` : ""}
          </span>
        </div>
        <table className="t" data-testid="steps-table">
          <thead>
            <tr>
              <th>#</th>
              <th>action</th>
              <th>target</th>
              <th>params</th>
              <th>rev.</th>
              <th>compensation</th>
              <th>precondition</th>
              <th>t/o</th>
              <th>status</th>
            </tr>
          </thead>
          <tbody>
            {plan.steps.map((s) => {
              const r = progress?.steps[s.step_id];
              return (
                <tr key={s.step_id}>
                  <td className="dim">{s.step_id}</td>
                  <td className="bright">{s.action_id}</td>
                  <td>{s.target_node}</td>
                  <td>{fmtParams(s.params)}</td>
                  <td className={s.reversibility === "irreversible" ? "sem-fail" : ""}>{s.reversibility}</td>
                  <td>
                    {s.compensation ? `${s.compensation.action_id} ${fmtParams(s.compensation.params)}` : "—"}
                  </td>
                  <td>{s.precondition}</td>
                  <td className="num">{s.timeout_s}s</td>
                  <td className={`step-status ${stepStatusClass(r?.status)}`}>
                    {r?.status ?? "—"}
                    {r?.error ? <div className="sem-fail">{r.error}</div> : null}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="section">
        <dl className="kv">
          <dt>reversibility</dt>
          <dd className={worst === "irreversible" ? "sem-fail" : "bright"} data-testid="plan-reversibility">
            {worst} <span className="dim">(worst step)</span>
          </dd>
          <dt>blast radius</dt>
          <dd data-testid="blast-radius">
            {plan.blast_radius.count} <span className="dim">· {plan.blast_radius.nodes.join(", ") || "—"}</span>
          </dd>
          <dt>verification</dt>
          <dd>
            {plan.verification.predicate} <span className="dim">· window {plan.verification.window_s}s</span>
          </dd>
          {progress?.verification && (
            <>
              <dt>observed</dt>
              <dd className={verifyClass(progress.verification.status)}>
                {progress.verification.status}
                {progress.verification.observed ? ` · ${JSON.stringify(progress.verification.observed)}` : ""}
              </dd>
            </>
          )}
        </dl>
      </div>

      <div className="section">
        <div className="section-label">
          <span className="label">Gate</span>
        </div>
        {gate ? (
          <div className="gate" data-testid="gate">
            <span className={`gate-decision ${gate.decision}`} data-testid="gate-decision">
              {DECISION_LABEL[gate.decision]}
            </span>
            <span className="mono" data-testid="gate-rule">
              <span className="dim">rule </span>
              {gate.rule_id}
            </span>
            <p className="gate-reason">{gate.reason}</p>
          </div>
        ) : (
          <span className="meta">not yet gated</span>
        )}
      </div>

      <div className="section">
        <div className="section-label">
          <span className="label">Operator</span>
        </div>
        <div className="controls">
          <input
            aria-label="operator"
            placeholder="operator@site"
            value={operator}
            onChange={(e) => onOperator(e.target.value)}
            style={{ width: 160 }}
            data-testid="operator-input"
          />
          {isDelay && countdown !== null && (
            <span className="countdown" data-testid="countdown" aria-live="polite">
              {mmss(countdown)}
            </span>
          )}
          {isDelay && (
            <button
              disabled={busy || noOperator || !pending}
              onClick={() => onDecision(plan.id, "execute_now")}
              data-testid="btn-execute-now"
            >
              Execute now
            </button>
          )}
          <button
            disabled={busy || noOperator || !canApprove || isDelay}
            onClick={() => onDecision(plan.id, "approve")}
            data-testid="btn-approve"
            title={decision === "report_only" ? "report_only: this plan never executes" : undefined}
          >
            Approve
          </button>
          <button
            className="quiet"
            disabled={busy || noOperator || !canApprove || !inCountdown}
            title={!inCountdown ? "hold stops a running countdown" : undefined}
            onClick={() => onDecision(plan.id, "hold")}
            data-testid="btn-hold"
          >
            Hold
          </button>
          <button
            className="danger"
            disabled={busy || noOperator || !canApprove}
            onClick={() => onDecision(plan.id, "reject")}
            data-testid="btn-reject"
          >
            Reject
          </button>
        </div>
        {pending && canApprove && noOperator && (
          <p className="meta" style={{ marginTop: 6 }} data-testid="operator-hint">
            Name the operator to enable the decision buttons — every decision is recorded against it.
          </p>
        )}
        {!pending && (
          <p className="meta" style={{ marginTop: 6 }} data-testid="plan-closed-hint">
            This plan is {planStatus}; decisions no longer apply.
            {planStatus === "done" && plan._receipt_id ? ` Receipt ${plan._receipt_id}.` : ""}
          </p>
        )}
        {decision === "report_only" && (
          <p className="meta" style={{ marginTop: 6 }}>
            report_only — the gate never executes this plan.
          </p>
        )}
        {decision === "auto" && (
          <p className="meta" style={{ marginTop: 6 }}>
            auto — executes without an operator.
          </p>
        )}
        {approvals.length > 0 && (
          <ul className="meta" style={{ marginTop: 8, paddingLeft: 0, listStyle: "none" }}>
            {approvals.map((a, i) => (
              <li key={i}>
                {a.timestamp ?? ""} {a.actor} → {a.decision}
              </li>
            ))}
          </ul>
        )}
      </div>

      <ReplanControls {...p} findingId={plan.finding_id} />
    </>
  );
}

function verifyClass(s: string): string {
  return s === "pass" ? "sem-pass" : s === "fail" ? "sem-fail" : "sem-inconclusive";
}

function ReplanControls({
  findingId,
  onReplan,
  replanError,
  busy,
  llm,
}: Pick<PlanPanelProps, "onReplan" | "replanError" | "busy" | "llm"> & { findingId: string }) {
  const [pending, setPending] = useState<null | "template" | "llm">(null);
  const run = async (planner: "template" | "llm") => {
    setPending(planner);
    try {
      await onReplan(findingId, planner);
    } finally {
      setPending(null);
    }
  };
  const llmLabel = llm ? llm.model : "LLM";
  const noKey = !!llm && !llm.ready;
  return (
    <div className="section">
      <div className="section-label">
        <span className="label">Planner</span>
      </div>
      <div className="controls">
        <button className="quiet" disabled={busy || pending !== null} onClick={() => void run("template")} data-testid="btn-plan-template">
          {pending === "template" ? "Planning…" : "Plan (template)"}
        </button>
        <button
          className="quiet"
          disabled={busy || pending !== null || noKey}
          onClick={() => void run("llm")}
          data-testid="btn-replan"
          title={noKey ? "no API key — set one under Settings" : `LLM planner: ${llm?.provider ?? ""} · ${llmLabel}`}
        >
          {pending === "llm" ? "Planning…" : `Re-plan (${llmLabel})`}
        </button>
        <span className="meta">a new plan supersedes a pending one; invalid LLM output is rejected, never repaired</span>
      </div>
      {replanError && (
        <pre className="error-block" style={{ marginTop: 8 }} data-testid="replan-error">
          {replanError}
        </pre>
      )}
    </div>
  );
}
