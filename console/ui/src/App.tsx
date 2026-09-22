import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, ApiError, WsClient, type WsStatus } from "./api";
import { closedFindings, openFindings, planForFinding, store, useStore } from "./store";
import { gateCell, isDefaultDeny, TARGET_MATRIX } from "./gate";
import type { ApproveDecisionBody, Policy } from "./types";
import { LoopStrip } from "./components/LoopStrip";
import { FindingsList } from "./components/FindingsList";
import { PlanPanel } from "./components/PlanPanel";
import { AutonomyMatrix } from "./components/AutonomyMatrix";
import { SessionLog } from "./components/SessionLog";
import { ReceiptsList } from "./components/ReceiptsList";
import { ReceiptRoute } from "./components/ReceiptView";
import { AnalystPanel } from "./components/AnalystPanel";
import { LabControls } from "./components/LabControls";

const OPERATOR_KEY = "console.operator";
const POLL_MS = 5000;

function readOperator(): string {
  try {
    return localStorage.getItem(OPERATOR_KEY) ?? "";
  } catch {
    return "";
  }
}

// ---------------------------------------------------------------- routing
// Two routes: `/` (console) and `/receipts/:id`. Plain pushState; the server
// must fall back to index.html for unknown paths (or use `#/receipts/:id`).

type Route = { name: "console" } | { name: "receipt"; id: string };

function parseRoute(): Route {
  const path = window.location.hash.startsWith("#/")
    ? window.location.hash.slice(1)
    : window.location.pathname;
  const m = /^\/receipts\/([^/?#]+)/.exec(path);
  return m ? { name: "receipt", id: decodeURIComponent(m[1]) } : { name: "console" };
}

function useRoute(): [Route, (r: Route) => void] {
  const [route, setRoute] = useState<Route>(() => parseRoute());
  useEffect(() => {
    const on = () => setRoute(parseRoute());
    window.addEventListener("popstate", on);
    window.addEventListener("hashchange", on);
    return () => {
      window.removeEventListener("popstate", on);
      window.removeEventListener("hashchange", on);
    };
  }, []);
  const go = useCallback((r: Route) => {
    const useHash = window.location.hash.startsWith("#/");
    const path = r.name === "receipt" ? `/receipts/${encodeURIComponent(r.id)}` : "/";
    if (useHash) window.location.hash = path === "/" ? "" : `#${path}`;
    else window.history.pushState(null, "", path);
    setRoute(r);
  }, []);
  return [route, go];
}

// ---------------------------------------------------------------- app

export default function App() {
  const snap = useStore((s) => s);
  const [route, go] = useRoute();
  const [ws, setWs] = useState<WsStatus>("connecting");
  const [polling, setPolling] = useState(false);
  const [filter, setFilter] = useState<"open" | "closed">("open");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [operator, setOperatorState] = useState<string>(() => readOperator());
  const [policy, setPolicy] = useState<Policy | null>(null);
  const [replanError, setReplanError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [analystOpen, setAnalystOpen] = useState(false);
  const [now, setNow] = useState(() => Date.now());
  const wsRef = useRef<WsStatus>("connecting");

  const setOperator = (v: string) => {
    setOperatorState(v);
    try {
      localStorage.setItem(OPERATOR_KEY, v);
    } catch {
      /* ignore */
    }
  };

  // -------------------------------------------------------------- bootstrap
  const refresh = useCallback(async () => {
    const results = await Promise.allSettled([
      api.state(),
      api.findings("open"),
      api.findings("closed"),
      api.receipts(),
      api.log(200),
      api.policy(),
    ]);
    const [st, open, closed, receipts, log, pol] = results;
    if (st.status === "fulfilled") store.setState(st.value);
    if (open.status === "fulfilled") store.setFindings(open.value);
    if (closed.status === "fulfilled") store.setFindings(closed.value);
    if (receipts.status === "fulfilled") store.setReceipts(receipts.value);
    if (log.status === "fulfilled" && store.get().log.length === 0) store.setLog(log.value);
    if (pol.status === "fulfilled") setPolicy(pol.value);
  }, []);

  useEffect(() => {
    void refresh();
    const client = new WsClient({
      onEvent: (ev) => store.apply(ev),
      onStatus: (s) => {
        wsRef.current = s;
        setWs(s);
        if (s === "open") void refresh();
      },
    });
    client.connect();
    return () => client.close();
  }, [refresh]);

  // Polling fallback while the socket is down.
  useEffect(() => {
    const t = setInterval(() => {
      if (wsRef.current !== "open") {
        setPolling(true);
        void refresh();
      } else {
        setPolling(false);
      }
    }, POLL_MS);
    return () => clearInterval(t);
  }, [refresh]);

  // Clock for ages.
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 10000);
    return () => clearInterval(t);
  }, []);

  // Keep the policy in sync with the strip's copy.
  useEffect(() => {
    if (snap.state?.policy && !policy) {
      setPolicy({ bands: snap.state.policy.bands, matrix: snap.state.policy.matrix });
    }
  }, [snap.state, policy]);

  // -------------------------------------------------------------- selection
  const open = useMemo(() => openFindings(snap), [snap]);
  const closed = useMemo(() => closedFindings(snap), [snap]);
  const list = filter === "open" ? open : closed;

  useEffect(() => {
    if (selectedId && snap.findings[selectedId]) return;
    if (list.length > 0) setSelectedId(list[0].id);
  }, [list, selectedId, snap.findings]);

  const finding = selectedId ? (snap.findings[selectedId] ?? null) : null;
  const plan = planForFinding(snap, selectedId);

  // Fetch the plan for the selection when the stream has not delivered it.
  useEffect(() => {
    if (!selectedId || plan) return;
    let alive = true;
    api
      .findingPlan(selectedId)
      .then((p) => alive && store.setPlan(p))
      .catch(() => {
        /* 404: no plan yet */
      });
    return () => {
      alive = false;
    };
  }, [selectedId, plan]);

  const hit = useMemo(
    () => gateCell(finding, plan, policy ? { bands: policy.bands, unmonitored_gap_cap: policy.unmonitored_gap_cap } : null),
    [finding, plan, policy],
  );
  const defaultDeny = snap.state?.policy?.default_deny ?? isDefaultDeny(policy?.matrix);

  // -------------------------------------------------------------- actions
  const onDecision = async (planId: string, decision: ApproveDecisionBody["decision"]) => {
    setBusy(true);
    setActionError(null);
    try {
      const res = await api.approve(planId, { actor: operator.trim(), decision });
      store.setPlan(res.plan);
      store.apply({ type: "approval", ts: new Date().toISOString(), plan_id: planId, approval: res.approval });
      if (decision !== "execute_now" && decision !== "approve") store.clearCountdown(planId);
    } catch (e) {
      setActionError(`${decision}: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  const onReplan = async (findingId: string) => {
    setReplanError(null);
    try {
      const p = await api.replan(findingId, "llm");
      store.setPlan(p);
    } catch (e) {
      const err = e as ApiError;
      const detail = err.body && typeof err.body === "object" ? JSON.stringify(err.body, null, 2) : err.message;
      setReplanError(`${err.status ? `${err.status} ` : ""}${err.message}${detail !== err.message ? `\n${detail}` : ""}`);
    }
  };

  const onLoadTarget = async () => {
    if (!policy) return;
    const ok = window.confirm(
      "Replace the autonomy matrix with the target matrix?\n\n" +
        "high:   reversible=auto  compensable=delay  irreversible=human\n" +
        "medium: human / human / human_plus\n" +
        "low:    report_only / report_only / report_only\n\n" +
        "This is audited and lifts the shipped default-deny.",
    );
    if (!ok) return;
    setBusy(true);
    setActionError(null);
    try {
      const next = await api.putPolicy({ ...policy, matrix: TARGET_MATRIX });
      setPolicy(next);
    } catch (e) {
      setActionError(`policy: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  // -------------------------------------------------------------- render
  if (route.name === "receipt") {
    const cached = snap.receipts.find((r) => r.id === route.id) ?? null;
    return <ReceiptRoute id={route.id} cached={cached} onBack={() => go({ name: "console" })} />;
  }

  return (
    <div className="app">
      <LoopStrip
        state={snap.state}
        active={snap.activeStation}
        ws={ws}
        polling={polling}
        onOpenAnalyst={() => setAnalystOpen(true)}
      />
      {snap.state?.lab && <LabControls />}
      {actionError && (
        <div className="notice down" role="alert">
          {actionError}
        </div>
      )}
      <div className="main">
        <div className="col">
          <FindingsList
            findings={list}
            selectedId={selectedId}
            filter={filter}
            onFilter={setFilter}
            onSelect={setSelectedId}
            now={now}
          />
        </div>
        <div className="col">
          <PlanPanel
            finding={finding}
            plan={plan}
            progress={plan ? (snap.progress[plan.id] ?? null) : null}
            approvals={plan ? (snap.approvals[plan.id] ?? []) : []}
            countdown={plan ? (snap.countdowns[plan.id] ?? null) : null}
            operator={operator}
            onOperator={setOperator}
            onDecision={onDecision}
            onReplan={onReplan}
            replanError={replanError}
            busy={busy}
          />
        </div>
        <div className="col col-right">
          <AutonomyMatrix policy={policy} hit={hit} defaultDeny={defaultDeny} onLoadTarget={onLoadTarget} busy={busy} />
          <SessionLog lines={snap.log} />
          <ReceiptsList receipts={snap.receipts} onOpen={(id) => go({ name: "receipt", id })} now={now} />
        </div>
      </div>
      <AnalystPanel open={analystOpen} finding={finding} onClose={() => setAnalystOpen(false)} />
    </div>
  );
}
