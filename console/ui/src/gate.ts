// Pure helpers that mirror console/policy/gate.py so the UI can point at the
// matrix cell that produced a plan's gate (and show a plan's worst step).

import type { Band, Finding, GateDecision, Plan, Policy, Reversibility } from "./types";
import { BANDS, REVERSIBILITIES } from "./types";

const RANK: Record<Reversibility, number> = { reversible: 0, compensable: 1, irreversible: 2 };

/** A plan's reversibility is its WORST step. */
export function planReversibility(plan: Plan): Reversibility {
  let worst: Reversibility = "reversible";
  for (const s of plan.steps) if (RANK[s.reversibility] > RANK[worst]) worst = s.reversibility;
  return worst;
}

export function bandFor(score: number, bands: Policy["bands"]): Band {
  if (score >= bands.high) return "high";
  if (score >= bands.medium) return "medium";
  return "low";
}

export function lowerBand(band: Band, by: number): Band {
  return BANDS[Math.min(BANDS.indexOf(band) + by, BANDS.length - 1)];
}

/** The (band, reversibility) cell the gate looked up for this finding+plan.
 *  Prefers parsing the gate's rule_id (`matrix:<band>:<rev>`); falls back to
 *  recomputing from the score, the gap cap and the plan's worst step. */
export function gateCell(
  finding: Finding | null,
  plan: Plan | null,
  policy: Pick<Policy, "bands" | "unmonitored_gap_cap"> | null,
): { band: Band; reversibility: Reversibility } | null {
  if (!plan) return null;
  const reversibility = planReversibility(plan);
  const rule = plan.gate?.rule_id ?? "";
  const m = /(?:^|:)matrix:(high|medium|low):(reversible|compensable|irreversible)$/.exec(rule);
  if (m) return { band: m[1] as Band, reversibility: m[2] as Reversibility };
  const cap = /cap:unmonitored_gap→(high|medium|low)$/.exec(rule);
  if (cap) return { band: cap[1] as Band, reversibility };
  if (!finding || !policy) return null;
  let band = bandFor(finding.confidence.score, policy.bands);
  if (finding.detection_mode === "unmonitored_gap" && (policy.unmonitored_gap_cap ?? 0) > 0) {
    band = lowerBand(band, policy.unmonitored_gap_cap ?? 0);
  }
  return { band, reversibility };
}

export function isDefaultDeny(matrix: Policy["matrix"] | undefined): boolean {
  if (!matrix) return false;
  return BANDS.every((b) => REVERSIBILITIES.every((r) => matrix[b]?.[r] === "human"));
}

/** The plan's target matrix (console/policy/policy.target.yaml), offered by
 *  the "Load target matrix" affordance. Confirmed by the operator before PUT. */
export const TARGET_MATRIX: Policy["matrix"] = {
  high: { reversible: "auto", compensable: "delay", irreversible: "human" },
  medium: { reversible: "human", compensable: "human", irreversible: "human_plus" },
  low: { reversible: "report_only", compensable: "report_only", irreversible: "report_only" },
};

export const DECISION_LABEL: Record<GateDecision, string> = {
  auto: "AUTO",
  delay: "DELAY",
  human: "HUMAN",
  human_plus: "HUMAN+",
  report_only: "REPORT ONLY",
};
