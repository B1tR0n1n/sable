import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent, within } from "@testing-library/react";
import { FindingsList } from "../components/FindingsList";
import { PlanPanel } from "../components/PlanPanel";
import type { Finding, Plan } from "../types";
import findingFx from "./fixtures/finding.json";
import planFx from "./fixtures/plan.json";

const finding = findingFx as Finding;
const plan = planFx as Plan;

describe("FindingsList", () => {
  it("renders the fixture finding with LIVE tag, confidence method, occurrences", () => {
    const onSelect = vi.fn();
    render(
      <FindingsList findings={[finding]} selectedId={null} filter="open" onFilter={() => {}} onSelect={onSelect} />,
    );
    const row = screen.getByTestId(`finding-${finding.id}`);
    expect(within(row).getByText("dns-1")).toBeInTheDocument();
    expect(within(row).getByText(/DNS_SERVER/)).toBeInTheDocument();
    expect(within(row).getByTestId("mode-tag")).toHaveTextContent("LIVE");
    expect(within(row).getByTestId("row-generated-tag")).toHaveTextContent("GENERATED");
    expect(within(row).getByText("0.91 engine_native")).toBeInTheDocument();
    expect(within(row).getByText("×1")).toBeInTheDocument();
    expect(within(row).getByLabelText("severity high")).toHaveClass("sev-high");
    fireEvent.click(row);
    expect(onSelect).toHaveBeenCalledWith(finding.id);
  });

  it("renders GAP and mc_dropout n=… for an unmonitored-gap finding", () => {
    const gap: Finding = {
      ...finding,
      id: "fnd-gap",
      detection_mode: "unmonitored_gap",
      confidence: { score: 0.7, method: "mc_dropout", samples: 20 },
      severity: "low",
      summary_generated: false,
    };
    render(<FindingsList findings={[gap]} selectedId="fnd-gap" filter="open" onFilter={() => {}} onSelect={() => {}} />);
    const row = screen.getByTestId("finding-fnd-gap");
    expect(within(row).getByTestId("mode-tag")).toHaveTextContent("GAP");
    expect(within(row).getByText("0.70 mc_dropout n=20")).toBeInTheDocument();
    expect(within(row).queryByTestId("row-generated-tag")).toBeNull();
    expect(row).toHaveClass("selected");
    expect(within(row).getByLabelText("severity low")).not.toHaveClass("sev-high");
  });

  it("switches the open/closed filter", () => {
    const onFilter = vi.fn();
    render(<FindingsList findings={[]} selectedId={null} filter="open" onFilter={onFilter} onSelect={() => {}} />);
    fireEvent.click(screen.getByRole("tab", { name: "Closed" }));
    expect(onFilter).toHaveBeenCalledWith("closed");
    expect(screen.getByText("No open findings.")).toBeInTheDocument();
  });
});

const baseProps = {
  progress: null,
  approvals: [],
  countdown: null,
  operator: "operator@lab",
  onOperator: () => {},
  onDecision: vi.fn(async () => {}),
  onReplan: vi.fn(async () => {}),
  replanError: null,
  busy: false,
};

describe("PlanPanel", () => {
  it("shows the GENERATED tag, evidence, and the gate rule_id", () => {
    render(<PlanPanel {...baseProps} finding={finding} plan={plan} />);
    expect(screen.getByTestId("generated-tag")).toHaveTextContent("GENERATED");
    expect(screen.getByTestId("finding-summary")).toHaveTextContent(finding.summary);
    const ev = screen.getByTestId("evidence-table");
    expect(within(ev).getByText("dns_query_latency_ms")).toBeInTheDocument();
    expect(within(ev).getByText("ms")).toBeInTheDocument();
    expect(screen.getByTestId("gate-rule")).toHaveTextContent("default-deny");
    expect(screen.getByTestId("gate-decision")).toHaveTextContent("HUMAN");
    expect(screen.getByText(plan.gate!.reason)).toBeInTheDocument();
    expect(screen.getByTestId("plan-reversibility")).toHaveTextContent("compensable");
    expect(screen.getByTestId("blast-radius")).toHaveTextContent("3");
    expect(screen.getByTestId("blast-radius")).toHaveTextContent("dns-1, app-erp, vdi-broker");
    const steps = screen.getByTestId("steps-table");
    expect(within(steps).getByText("restart_service")).toBeInTheDocument();
    expect(within(steps).getByText("service_exists")).toBeInTheDocument();
    expect(within(steps).getByText("60s")).toBeInTheDocument();
    expect(screen.getByTestId("btn-approve")).toBeEnabled();
  });

  it("hides GENERATED when summary_generated is false", () => {
    render(<PlanPanel {...baseProps} finding={{ ...finding, summary_generated: false }} plan={plan} />);
    expect(screen.queryByTestId("generated-tag")).toBeNull();
  });

  it("disables Approve/Hold/Reject when the gate is report_only", () => {
    const ro: Plan = { ...plan, gate: { decision: "report_only", rule_id: "matrix:low:compensable", reason: "low band" } };
    render(<PlanPanel {...baseProps} finding={finding} plan={ro} />);
    expect(screen.getByTestId("gate-rule")).toHaveTextContent("matrix:low:compensable");
    expect(screen.getByTestId("btn-approve")).toBeDisabled();
    expect(screen.getByTestId("btn-hold")).toBeDisabled();
    expect(screen.getByTestId("btn-reject")).toBeDisabled();
    expect(screen.queryByTestId("btn-execute-now")).toBeNull();
  });

  it("disables decisions without an operator name", () => {
    render(<PlanPanel {...baseProps} operator="" finding={finding} plan={plan} />);
    expect(screen.getByTestId("btn-approve")).toBeDisabled();
  });

  it("posts the decision with the plan id", async () => {
    const onDecision = vi.fn(async () => {});
    render(<PlanPanel {...baseProps} onDecision={onDecision} finding={finding} plan={plan} planStatus="proposed" />);
    fireEvent.click(screen.getByTestId("btn-approve"));
    expect(onDecision).toHaveBeenCalledWith(plan.id, "approve");
    fireEvent.click(screen.getByTestId("btn-reject"));
    expect(onDecision).toHaveBeenCalledWith(plan.id, "reject");
  });

  it("offers Hold only while a countdown runs", () => {
    const onDecision = vi.fn(async () => {});
    const { unmount } = render(<PlanPanel {...baseProps} finding={finding} plan={plan} planStatus="proposed" />);
    expect(screen.getByTestId("btn-hold")).toBeDisabled();
    unmount();
    const delay: Plan = { ...plan, gate: { decision: "delay", rule_id: "matrix:high:compensable", reason: "wait", delay_s: 120 } };
    render(<PlanPanel {...baseProps} onDecision={onDecision} finding={finding} plan={delay} planStatus="countdown" countdown={30} />);
    fireEvent.click(screen.getByTestId("btn-hold"));
    expect(onDecision).toHaveBeenCalledWith(plan.id, "hold");
  });

  it.each(["executing", "done", "rejected", "superseded", "withdrawn"] as const)(
    "disables every decision on a %s plan and says why",
    (status) => {
      render(<PlanPanel {...baseProps} finding={finding} plan={plan} planStatus={status} />);
      expect(screen.getByTestId("plan-status")).toHaveTextContent(status);
      expect(screen.getByTestId("btn-approve")).toBeDisabled();
      expect(screen.getByTestId("btn-hold")).toBeDisabled();
      expect(screen.getByTestId("btn-reject")).toBeDisabled();
      expect(screen.getByTestId("plan-closed-hint")).toHaveTextContent(status);
    },
  );

  it("explains the disabled decisions when no operator is named", () => {
    render(<PlanPanel {...baseProps} operator="" finding={finding} plan={plan} planStatus="proposed" />);
    expect(screen.getByTestId("operator-hint")).toBeInTheDocument();
  });

  it("offers both planners and passes the choice", () => {
    const onReplan = vi.fn(async () => {});
    render(
      <PlanPanel {...baseProps} onReplan={onReplan} finding={finding} plan={null}
        llm={{ provider: "openai", model: "gpt-5.5", ready: true }} />,
    );
    expect(screen.getByTestId("btn-replan")).toHaveTextContent("gpt-5.5");
    expect(screen.getByTestId("btn-replan")).toBeEnabled();
    fireEvent.click(screen.getByTestId("btn-plan-template"));
    expect(onReplan).toHaveBeenCalledWith(finding.id, "template");
  });

  it("disables the LLM planner without a key but keeps the template planner", () => {
    render(<PlanPanel {...baseProps} finding={finding} plan={null} llm={{ provider: "anthropic", model: "claude-sonnet-5", ready: false }} />);
    expect(screen.getByTestId("btn-replan")).toBeDisabled();
    expect(screen.getByTestId("btn-replan")).toHaveAttribute("title", expect.stringContaining("Settings"));
    expect(screen.getByTestId("btn-plan-template")).toBeEnabled();
  });

  it("shows the countdown and Execute now for a delay gate", () => {
    const delay: Plan = { ...plan, gate: { decision: "delay", rule_id: "matrix:high:compensable", reason: "wait", delay_s: 120 } };
    const onDecision = vi.fn(async () => {});
    render(<PlanPanel {...baseProps} onDecision={onDecision} finding={finding} plan={delay} countdown={65} />);
    expect(screen.getByTestId("countdown")).toHaveTextContent("01:05");
    fireEvent.click(screen.getByTestId("btn-execute-now"));
    expect(onDecision).toHaveBeenCalledWith(plan.id, "execute_now");
  });

  it("shows a planner validation failure verbatim", () => {
    const err = '422 planner output failed validation\n{\n  "error": "steps.0.action_id: not in catalog"\n}';
    render(<PlanPanel {...baseProps} replanError={err} finding={finding} plan={plan} />);
    expect(screen.getByTestId("replan-error")).toHaveTextContent("steps.0.action_id: not in catalog");
    expect(screen.getByTestId("replan-error").textContent).toBe(err);
  });
});
