import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { AutonomyMatrix } from "../components/AutonomyMatrix";
import { gateCell, planReversibility, isDefaultDeny, TARGET_MATRIX } from "../gate";
import type { Finding, Plan, Policy } from "../types";
import findingFx from "./fixtures/finding.json";
import planFx from "./fixtures/plan.json";

const finding = findingFx as Finding;
const plan = planFx as Plan;

const target: Policy = { bands: { high: 0.85, medium: 0.6 }, matrix: TARGET_MATRIX, unmonitored_gap_cap: 1 };

describe("gate helpers", () => {
  it("plan reversibility is the worst step", () => {
    expect(planReversibility(plan)).toBe("compensable");
    expect(
      planReversibility({ ...plan, steps: [plan.steps[0], { ...plan.steps[0], step_id: "s2", reversibility: "irreversible" }] }),
    ).toBe("irreversible");
  });

  it("parses the cell from rule_id, including the default-deny prefix", () => {
    expect(gateCell(finding, { ...plan, gate: { decision: "human", rule_id: "default-deny:matrix:high:compensable", reason: "" } }, target)).toEqual({
      band: "high",
      reversibility: "compensable",
    });
    expect(gateCell(finding, { ...plan, gate: { decision: "human", rule_id: "cap:unmonitored_gap→medium", reason: "" } }, target)).toEqual({
      band: "medium",
      reversibility: "compensable",
    });
  });

  it("recomputes the cell from score, gap cap and worst step when rule_id is opaque", () => {
    expect(gateCell(finding, plan, target)).toEqual({ band: "high", reversibility: "compensable" });
    const gap: Finding = { ...finding, detection_mode: "unmonitored_gap" };
    expect(gateCell(gap, plan, target)).toEqual({ band: "medium", reversibility: "compensable" });
    const low: Finding = { ...finding, confidence: { score: 0.2, method: "heuristic" } };
    expect(gateCell(low, plan, target)).toEqual({ band: "low", reversibility: "compensable" });
  });

  it("detects the shipped default-deny matrix", () => {
    expect(isDefaultDeny(TARGET_MATRIX)).toBe(false);
    expect(
      isDefaultDeny({
        high: { reversible: "human", compensable: "human", irreversible: "human" },
        medium: { reversible: "human", compensable: "human", irreversible: "human" },
        low: { reversible: "human", compensable: "human", irreversible: "human" },
      }),
    ).toBe(true);
  });
});

describe("AutonomyMatrix", () => {
  it("renders 9 cells with their decisions and highlights the gate cell", () => {
    render(
      <AutonomyMatrix policy={target} hit={{ band: "high", reversibility: "compensable" }} defaultDeny={false} onLoadTarget={() => {}} />,
    );
    expect(screen.getAllByRole("gridcell")).toHaveLength(9);
    const hit = screen.getByTestId("cell-high-compensable");
    expect(hit).toHaveClass("hit");
    expect(hit).toHaveTextContent("DELAY");
    expect(screen.getByTestId("cell-high-reversible")).not.toHaveClass("hit");
    expect(screen.getByTestId("cell-high-reversible")).toHaveTextContent("AUTO");
    expect(screen.getByTestId("cell-low-irreversible")).toHaveTextContent("REPORT ONLY");
    expect(screen.getByTestId("cell-medium-irreversible")).toHaveTextContent("HUMAN+");
    expect(screen.getAllByRole("gridcell").filter((c) => c.classList.contains("hit"))).toHaveLength(1);
  });

  it("shows nothing highlighted without a plan", () => {
    render(<AutonomyMatrix policy={target} hit={null} defaultDeny={true} onLoadTarget={() => {}} />);
    expect(screen.getAllByRole("gridcell").filter((c) => c.classList.contains("hit"))).toHaveLength(0);
    expect(screen.getByText(/default_deny/)).toBeInTheDocument();
  });
});
