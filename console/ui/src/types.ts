// TypeScript mirrors of the Phase 1 contracts (console/contracts/schema/*.json)
// and the Phase 7 API surface (console/docs/API.md). Keep these in lock-step
// with the schemas; the server serialises Pydantic models with
// model_dump(mode="json"), so every date is an ISO string.

// ---------------------------------------------------------------- Finding

export type Severity = "low" | "medium" | "high" | "critical";
export type DetectionMode = "live_feed" | "unmonitored_gap";
export type ConfidenceMethod = "mc_dropout" | "engine_native" | "heuristic";
export type FindingStatus = "open" | "closed" | "reopened";

export interface NodeRef {
  node_id: string;
  component_type: string;
  state: string;
}

export interface Confidence {
  score: number;
  method: ConfidenceMethod;
  samples?: number | null;
}

export interface Evidence {
  metric: string;
  value: number;
  unit?: string;
  threshold?: number | null;
  timestamp: string;
}

export interface Finding {
  id: string;
  site_id: string;
  detection_mode: DetectionMode;
  root_cause: NodeRef;
  affected_nodes: NodeRef[];
  evidence: Evidence[];
  confidence: Confidence;
  severity: Severity;
  summary: string;
  summary_generated: boolean;
  occurrences: number;
  status: FindingStatus;
  engine_version: string;
  receipt_ids: string[];
  created_at: string;
  updated_at?: string | null;
}

// ---------------------------------------------------------------- Plan

export type Reversibility = "reversible" | "compensable" | "irreversible";
export type GateDecision = "auto" | "delay" | "human" | "human_plus" | "report_only";

export interface Compensation {
  action_id: string;
  params?: Record<string, unknown>;
}

export interface Step {
  step_id: string;
  action_id: string;
  target_node: string;
  params?: Record<string, unknown>;
  reversibility: Reversibility;
  compensation?: Compensation | null;
  precondition: string;
  timeout_s: number;
}

export interface BlastRadius {
  count: number;
  nodes: string[];
}

export interface Verification {
  predicate: string;
  window_s: number;
}

export interface Gate {
  decision: GateDecision;
  rule_id: string;
  reason: string;
  delay_s?: number | null;
}

export interface PlannerProvenance {
  kind: string;
  template_id?: string | null;
  provider?: string | null;
  model?: string | null;
  prompt_sha256?: string | null;
  output_sha256?: string | null;
}

export interface Plan {
  id: string;
  finding_id: string;
  steps: Step[];
  blast_radius: BlastRadius;
  verification: Verification;
  gate?: Gate | null;
  planner?: PlannerProvenance | null;
  created_at: string;
}

// ---------------------------------------------------------------- Receipt

export type StepStatus =
  | "pending"
  | "running"
  | "ok"
  | "failed"
  | "skipped"
  | "precondition_failed"
  | "timed_out";
export type VerificationStatus = "pass" | "fail" | "inconclusive";

export interface StepResult {
  step_id: string;
  status: StepStatus;
  started_at?: string | null;
  ended_at?: string | null;
  output_digest?: string | null;
  session_id?: string | null;
  error?: string | null;
}

export interface Snapshot {
  node_id: string;
  snapshot_ref: string;
}

export interface Approval {
  actor: string;
  decision: string;
  timestamp?: string;
}

export interface VerificationResult {
  status: VerificationStatus;
  observed?: Record<string, unknown>;
  checked_at?: string;
}

export interface RollbackStep {
  step_id: string;
  action_id: string;
  status: StepStatus;
  session_id?: string | null;
  error?: string | null;
}

export interface Rollback {
  performed: boolean;
  steps: RollbackStep[];
}

export interface Receipt {
  id: string;
  plan_id: string;
  finding_id?: string | null;
  session_id?: string | null;
  steps: StepResult[];
  snapshots: Snapshot[];
  approvals: Approval[];
  verification?: VerificationResult | null;
  rollback: Rollback;
  prev_receipt_hash?: string | null;
  receipt_hash?: string | null;
  audit_ref?: Record<string, unknown> | null;
  created_at: string;
}

// ---------------------------------------------------------------- Policy

export type Band = "high" | "medium" | "low";
export const BANDS: readonly Band[] = ["high", "medium", "low"];
export const REVERSIBILITIES: readonly Reversibility[] = [
  "reversible",
  "compensable",
  "irreversible",
];

export type MatrixRow = Record<Reversibility, GateDecision>;
export type Matrix = Record<Band, MatrixRow>;

export interface Bands {
  high: number;
  medium: number;
}

export interface Policy {
  version?: number;
  bands: Bands;
  matrix: Matrix;
  delay_s?: number;
  unmonitored_gap_cap?: number;
  default_deny?: boolean;
  hard_rules?: { irreversible_never_auto: boolean };
}

// ---------------------------------------------------------------- State

export interface SableStatus {
  url: string;
  ok: boolean;
  provider: "claude" | "nemotron" | string | null;
  device?: string | null;
}

export interface OverlordStatus {
  socket: string;
  ok: boolean;
  version?: string | null;
}

export interface Counts {
  findings_open: number;
  plans_pending: number;
  receipts: number;
}

export interface ConsoleState {
  sable: SableStatus;
  overlord: OverlordStatus;
  counts: Counts;
  policy: { matrix: Matrix; bands: Bands; default_deny: boolean };
  lab?: boolean;
}

// ---------------------------------------------------------------- Log

export interface LogLine {
  ts: string;
  level: string;
  source: string;
  text: string;
}

// ---------------------------------------------------------------- API results

export interface ApproveDecisionBody {
  actor: string;
  decision: "approve" | "reject" | "hold" | "execute_now";
}

export interface ApproveResult {
  approval: Approval;
  plan: Plan;
  started: boolean;
}

export interface VerifyChainResult {
  ok: boolean;
  broken_at?: string | null;
}

export interface AnalystToolCall {
  name: string;
  input: Record<string, unknown>;
}

export interface AnalystTurn {
  role: "user" | "assistant";
  content: string;
}

export interface AnalystChatResult {
  reply: string;
  tool_calls: AnalystToolCall[];
  provider: "claude" | "local" | string;
  generated: true;
}

// ---------------------------------------------------------------- WS events

export type LoopStation = "DETECT" | "PROPOSE" | "GATE" | "EXECUTE" | "VERIFY" | "RECEIPT";
export const STATIONS: readonly LoopStation[] = [
  "DETECT",
  "PROPOSE",
  "GATE",
  "EXECUTE",
  "VERIFY",
  "RECEIPT",
];

interface EventBase {
  ts: string;
}

export type WsEvent =
  | (EventBase & { type: "hello"; state: ConsoleState })
  | (EventBase & { type: "finding"; finding: Finding })
  | (EventBase & { type: "plan"; plan: Plan })
  | (EventBase & { type: "approval"; plan_id: string; approval: Approval })
  | (EventBase & { type: "countdown"; plan_id: string; remaining_s: number })
  | (EventBase & { type: "step"; plan_id: string; step: StepResult })
  | (EventBase & {
      type: "verification";
      plan_id: string;
      receipt_id: string;
      result: VerificationResult;
    })
  | (EventBase & { type: "receipt"; receipt: Receipt })
  | (EventBase & { type: "log"; line: LogLine })
  | (EventBase & { type: "state"; state: ConsoleState });

export type WsEventType = WsEvent["type"];
