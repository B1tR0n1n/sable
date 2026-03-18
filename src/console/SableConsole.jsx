import { useState, useEffect, useRef, useCallback } from "react";

// ============================================================================
// SABLE OPERATIONS CONSOLE
// Interactive infrastructure failure simulation + AI analysis
// ============================================================================

// --- Type/color mappings from the simulator ---
const TYPE_COLORS = {
  CORE_SWITCH: "#2196F3", ACCESS_SWITCH: "#64B5F6", FIREWALL: "#FF5722",
  ROUTER: "#FF7043", LOAD_BALANCER: "#AB47BC", SERVER_PHYSICAL: "#66BB6A",
  SERVER_VIRTUAL: "#81C784", HYPERVISOR: "#43A047", STORAGE_ARRAY: "#FFA726",
  STORAGE_TARGET: "#FFB74D", VDI_BROKER: "#5C6BC0", VDI_HOST: "#7986CB",
  DNS_SERVER: "#26A69A", DHCP_SERVER: "#4DB6AC", DOMAIN_CONTROLLER: "#EC407A",
  CERTIFICATE_AUTHORITY: "#F48FB1", MONITORING_SERVER: "#78909C",
  WAN_LINK: "#8D6E63", INTERNET_GATEWAY: "#EF5350", APPLICATION_SERVICE: "#9CCC65",
};

const STATE_COLORS = {
  healthy: "#4CAF50", degraded: "#FF9800", failed: "#F44336", unreachable: "#9E9E9E",
};

const TYPE_ICONS = {
  CORE_SWITCH: "⬡", ACCESS_SWITCH: "⬡", FIREWALL: "🛡", ROUTER: "◈",
  LOAD_BALANCER: "⚖", SERVER_PHYSICAL: "▪", SERVER_VIRTUAL: "□",
  HYPERVISOR: "▣", STORAGE_ARRAY: "▤", STORAGE_TARGET: "▥",
  VDI_BROKER: "◧", VDI_HOST: "◨", DNS_SERVER: "◉", DHCP_SERVER: "◎",
  DOMAIN_CONTROLLER: "⬢", CERTIFICATE_AUTHORITY: "⊛", MONITORING_SERVER: "◉",
  WAN_LINK: "━", INTERNET_GATEWAY: "⊕", APPLICATION_SERVICE: "◆",
};

const FAILURE_CATEGORIES = [
  "SINGLE_COMPONENT_FAILURE", "CASCADING_OVERLOAD", "SILENT_DEGRADATION",
  "NETWORK_PARTITION", "DEPENDENCY_CHAIN", "CORRELATED_FAILURE",
  "INTERMITTENT_FAILURE", "CONFIGURATION_DRIFT",
];

// --- Topology Templates (simplified JS versions) ---
function generateSmallOffice(seed) {
  const rng = mulberry32(seed);
  const nodes = [];
  const edges = [];
  const add = (id, type) => { nodes.push({ id, type, state: "healthy", health: 1.0, props: {} }); };
  const dep = (src, tgt, type, crit) => { edges.push({ source_id: src, target_id: tgt, type, criticality: crit }); };

  add("fw-01", "FIREWALL"); add("igw-01", "INTERNET_GATEWAY"); add("wan-01", "WAN_LINK");
  dep("wan-01", "igw-01", "NETWORK_PATH", "HARD");
  dep("igw-01", "fw-01", "NETWORK_PATH", "HARD");

  add("sw-core-01", "CORE_SWITCH");
  dep("fw-01", "sw-core-01", "NETWORK_PATH", "HARD");

  add("sw-acc-01", "ACCESS_SWITCH"); add("sw-acc-02", "ACCESS_SWITCH");
  dep("sw-acc-01", "sw-core-01", "NETWORK_PATH", "HARD");
  dep("sw-acc-02", "sw-core-01", "NETWORK_PATH", "HARD");

  add("srv-phys-01", "SERVER_PHYSICAL");
  dep("srv-phys-01", "sw-acc-01", "NETWORK_PATH", "HARD");

  add("dns-01", "DNS_SERVER"); add("dhcp-01", "DHCP_SERVER"); add("dc-01", "DOMAIN_CONTROLLER");
  ["dns-01", "dhcp-01", "dc-01"].forEach(id => {
    dep(id, "srv-phys-01", "HOSTING_DEPENDENCY", "HARD");
    dep(id, "sw-acc-01", "NETWORK_PATH", "HARD");
  });

  add("mon-01", "MONITORING_SERVER");
  dep("mon-01", "srv-phys-01", "HOSTING_DEPENDENCY", "HARD");
  dep("mon-01", "sw-acc-01", "NETWORK_PATH", "HARD");

  // Service deps
  nodes.filter(n => !["CORE_SWITCH","ACCESS_SWITCH","WAN_LINK","INTERNET_GATEWAY","DNS_SERVER"].includes(n.type)).forEach(n => {
    dep(n.id, "dns-01", "DNS_DEPENDENCY", "SOFT");
  });
  nodes.filter(n => ["SERVER_PHYSICAL","SERVER_VIRTUAL","HYPERVISOR","APPLICATION_SERVICE"].includes(n.type)).forEach(n => {
    dep(n.id, "dc-01", "AUTHENTICATION_DEPENDENCY", "SOFT");
  });

  return { nodes, edges, name: "Small Office" };
}

function generateEnterpriseCampus(seed) {
  const rng = mulberry32(seed);
  const nodes = []; const edges = [];
  const add = (id, type) => { nodes.push({ id, type, state: "healthy", health: 1.0, props: {} }); };
  const dep = (s, t, ty, c) => { edges.push({ source_id: s, target_id: t, type: ty, criticality: c }); };

  // Perimeter
  add("fw-01", "FIREWALL"); add("fw-02", "FIREWALL");
  add("rtr-01", "ROUTER"); add("rtr-02", "ROUTER");
  add("igw-01", "INTERNET_GATEWAY"); add("wan-01", "WAN_LINK"); add("wan-02", "WAN_LINK");
  dep("wan-01", "igw-01", "NETWORK_PATH", "REDUNDANT"); dep("wan-02", "igw-01", "NETWORK_PATH", "REDUNDANT");
  dep("igw-01", "rtr-01", "NETWORK_PATH", "REDUNDANT"); dep("igw-01", "rtr-02", "NETWORK_PATH", "REDUNDANT");
  dep("rtr-01", "fw-01", "NETWORK_PATH", "HARD"); dep("rtr-02", "fw-02", "NETWORK_PATH", "HARD");

  // Core
  add("sw-core-01", "CORE_SWITCH"); add("sw-core-02", "CORE_SWITCH");
  dep("sw-core-01", "sw-core-02", "NETWORK_PATH", "REDUNDANT");
  dep("sw-core-02", "sw-core-01", "NETWORK_PATH", "REDUNDANT");
  dep("fw-01", "sw-core-01", "NETWORK_PATH", "HARD"); dep("fw-02", "sw-core-02", "NETWORK_PATH", "HARD");

  // Access
  for (let i = 1; i <= 4; i++) {
    add(`sw-acc-${String(i).padStart(2,"0")}`, "ACCESS_SWITCH");
    dep(`sw-acc-${String(i).padStart(2,"0")}`, "sw-core-01", "NETWORK_PATH", "REDUNDANT");
    dep(`sw-acc-${String(i).padStart(2,"0")}`, "sw-core-02", "NETWORK_PATH", "REDUNDANT");
  }

  // Servers
  for (let i = 1; i <= 4; i++) {
    add(`srv-phys-${String(i).padStart(2,"0")}`, "SERVER_PHYSICAL");
    dep(`srv-phys-${String(i).padStart(2,"0")}`, `sw-acc-${String(((i-1)%4)+1).padStart(2,"0")}`, "NETWORK_PATH", "HARD");
  }

  // Hypervisors + VMs
  add("hv-01", "HYPERVISOR"); add("hv-02", "HYPERVISOR");
  dep("hv-01", "srv-phys-01", "HOSTING_DEPENDENCY", "HARD");
  dep("hv-02", "srv-phys-02", "HOSTING_DEPENDENCY", "HARD");

  for (let i = 1; i <= 4; i++) {
    add(`srv-vm-${String(i).padStart(2,"0")}`, "SERVER_VIRTUAL");
    dep(`srv-vm-${String(i).padStart(2,"0")}`, i <= 2 ? "hv-01" : "hv-02", "HOSTING_DEPENDENCY", "HARD");
  }

  // Storage
  add("stor-arr-01", "STORAGE_ARRAY"); add("stor-arr-02", "STORAGE_ARRAY");
  dep("stor-arr-01", "sw-acc-01", "NETWORK_PATH", "HARD");
  dep("stor-arr-02", "sw-acc-02", "NETWORK_PATH", "HARD");
  add("stor-tgt-01", "STORAGE_TARGET"); add("stor-tgt-02", "STORAGE_TARGET");
  dep("stor-tgt-01", "stor-arr-01", "STORAGE_DEPENDENCY", "HARD");
  dep("stor-tgt-02", "stor-arr-02", "STORAGE_DEPENDENCY", "HARD");
  dep("hv-01", "stor-arr-01", "STORAGE_DEPENDENCY", "HARD");
  dep("hv-02", "stor-arr-02", "STORAGE_DEPENDENCY", "HARD");

  // Services
  add("dns-01", "DNS_SERVER"); add("dns-02", "DNS_SERVER");
  add("dhcp-01", "DHCP_SERVER"); add("dc-01", "DOMAIN_CONTROLLER"); add("dc-02", "DOMAIN_CONTROLLER");
  add("ca-01", "CERTIFICATE_AUTHORITY"); add("mon-01", "MONITORING_SERVER");
  add("lb-01", "LOAD_BALANCER"); add("lb-02", "LOAD_BALANCER");

  ["dns-01","dns-02","dhcp-01","dc-01","dc-02","ca-01","mon-01"].forEach((id, i) => {
    dep(id, `srv-vm-${String((i%4)+1).padStart(2,"0")}`, "HOSTING_DEPENDENCY", "HARD");
    dep(id, `sw-acc-${String((i%4)+1).padStart(2,"0")}`, "NETWORK_PATH", "HARD");
  });
  dep("dc-01", "dc-02", "REPLICATION_DEPENDENCY", "SOFT");
  dep("lb-01", "sw-acc-01", "NETWORK_PATH", "HARD"); dep("lb-02", "sw-acc-02", "NETWORK_PATH", "HARD");

  // Apps
  add("app-01", "APPLICATION_SERVICE"); add("app-02", "APPLICATION_SERVICE");
  dep("app-01", "srv-vm-01", "HOSTING_DEPENDENCY", "HARD"); dep("app-01", "lb-01", "SERVICE_DEPENDENCY", "SOFT");
  dep("app-02", "srv-vm-02", "HOSTING_DEPENDENCY", "HARD"); dep("app-02", "lb-02", "SERVICE_DEPENDENCY", "SOFT");

  // Service deps
  nodes.filter(n => !["CORE_SWITCH","ACCESS_SWITCH","WAN_LINK","INTERNET_GATEWAY","DNS_SERVER"].includes(n.type)).forEach(n => {
    dep(n.id, rngChoice(rng, ["dns-01","dns-02"]), "DNS_DEPENDENCY", "REDUNDANT");
  });
  nodes.filter(n => ["SERVER_PHYSICAL","SERVER_VIRTUAL","HYPERVISOR","APPLICATION_SERVICE"].includes(n.type)).forEach(n => {
    dep(n.id, rngChoice(rng, ["dc-01","dc-02"]), "AUTHENTICATION_DEPENDENCY", "REDUNDANT");
  });

  return { nodes, edges, name: "Enterprise Campus" };
}

function generateVDI(seed) {
  const base = generateEnterpriseCampus(seed);
  const { nodes, edges } = base;
  const add = (id, type) => { nodes.push({ id, type, state: "healthy", health: 1.0, props: {} }); };
  const dep = (s, t, ty, c) => { edges.push({ source_id: s, target_id: t, type: ty, criticality: c }); };

  add("vdi-broker-01", "VDI_BROKER"); add("vdi-broker-02", "VDI_BROKER");
  dep("vdi-broker-01", "srv-vm-01", "HOSTING_DEPENDENCY", "HARD");
  dep("vdi-broker-02", "srv-vm-02", "HOSTING_DEPENDENCY", "HARD");
  dep("vdi-broker-01", "sw-acc-01", "NETWORK_PATH", "HARD");
  dep("vdi-broker-02", "sw-acc-02", "NETWORK_PATH", "HARD");

  for (let i = 1; i <= 8; i++) {
    const id = `vdi-host-${String(i).padStart(2,"0")}`;
    add(id, "VDI_HOST");
    dep(id, i <= 4 ? "hv-01" : "hv-02", "HOSTING_DEPENDENCY", "HARD");
    dep(id, i <= 4 ? "vdi-broker-01" : "vdi-broker-02", "SERVICE_DEPENDENCY", "HARD");
    dep(id, i <= 4 ? "stor-arr-01" : "stor-arr-02", "STORAGE_DEPENDENCY", "HARD");
  }

  return { nodes, edges, name: "VDI Environment" };
}

function generateHealthcare(seed) {
  const base = generateEnterpriseCampus(seed);
  const { nodes, edges } = base;
  const rng = mulberry32(seed + 100);
  const add = (id, type) => { nodes.push({ id, type, state: "healthy", health: 1.0, props: {} }); };
  const dep = (s, t, ty, c) => { edges.push({ source_id: s, target_id: t, type: ty, criticality: c }); };

  ["ehr","pacs","pharmacy","lab","radiology"].forEach((name, i) => {
    const id = `app-${name}-01`;
    add(id, "APPLICATION_SERVICE");
    dep(id, `srv-vm-${String((i%4)+1).padStart(2,"0")}`, "HOSTING_DEPENDENCY", "HARD");
    dep(id, `sw-acc-${String((i%4)+1).padStart(2,"0")}`, "NETWORK_PATH", "HARD");
    dep(id, rngChoice(rng, ["dns-01","dns-02"]), "DNS_DEPENDENCY", "REDUNDANT");
  });
  dep("app-pacs-01", "stor-arr-01", "STORAGE_DEPENDENCY", "HARD");

  add("stor-arr-03", "STORAGE_ARRAY");
  dep("stor-arr-03", "sw-acc-03", "NETWORK_PATH", "HARD");

  return { nodes, edges, name: "Healthcare Network" };
}

// --- Seeded RNG ---
function mulberry32(a) {
  return function() {
    let t = a += 0x6D2B79F5;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
function rngChoice(rng, arr) { return arr[Math.floor(rng() * arr.length)]; }

// --- Cascade Propagation Engine (simplified JS port) ---
function simulateCascade(topology, injectedNodeId, maxTicks = 30) {
  const { nodes, edges } = topology;
  const state = {};
  nodes.forEach(n => { state[n.id] = { state: "healthy", health: 1.0 }; });
  const trace = [];

  // Inject failure
  state[injectedNodeId] = { state: "failed", health: 0.0 };
  trace.push({ tick: 0, component_id: injectedNodeId, prev_state: "healthy", new_state: "failed", prev_health: 1.0, new_health: 0.0, cause: "injected_failure", cause_component: null });

  // Build dependency lookup: edges go source→target meaning "source depends on target"
  const dependentsOf = {}; // target → [sources that depend on it]
  edges.forEach(e => {
    if (!dependentsOf[e.target_id]) dependentsOf[e.target_id] = [];
    dependentsOf[e.target_id].push(e);
  });

  let changedThisTick = new Set([injectedNodeId]);

  for (let tick = 1; tick <= maxTicks; tick++) {
    const newChanges = [];
    const alreadyChanged = new Set();

    changedThisTick.forEach(changedId => {
      const changedState = state[changedId];
      const deps = dependentsOf[changedId] || [];

      deps.forEach(edge => {
        const depId = edge.source_id; // the component that depends on changedId
        if (alreadyChanged.has(depId)) return;
        const depState = state[depId];
        if (depState.state === "failed") return;

        const prevState = depState.state;
        const prevHealth = depState.health;

        // Monitoring loss - no real state change
        if (edge.type === "MONITORING_DEPENDENCY") return;

        // DNS/Auth delayed (simplified - skip for now, treat as soft)
        if (edge.type === "DNS_DEPENDENCY" || edge.type === "AUTHENTICATION_DEPENDENCY") {
          if (changedState.state === "failed" && edge.criticality !== "REDUNDANT") {
            depState.health = Math.max(0, depState.health - 0.3);
            if (depState.health <= 0.2) depState.state = "failed";
            else if (depState.health <= 0.5) depState.state = "degraded";
          }
          if (depState.state !== prevState || depState.health !== prevHealth) {
            newChanges.push({ tick, component_id: depId, prev_state: prevState, new_state: depState.state, prev_health: prevHealth, new_health: depState.health, cause: "dns_auth_cascade", cause_component: changedId });
            alreadyChanged.add(depId);
          }
          return;
        }

        // Hosting dependency - hard cascade
        if (edge.type === "HOSTING_DEPENDENCY") {
          if (changedState.state === "failed") {
            depState.state = "failed"; depState.health = 0;
            newChanges.push({ tick, component_id: depId, prev_state: prevState, new_state: "failed", prev_health: prevHealth, new_health: 0, cause: "host_failure", cause_component: changedId });
            alreadyChanged.add(depId);
          } else if (changedState.state === "degraded") {
            depState.health = Math.max(0, depState.health - 0.4);
            if (depState.health <= 0.2) depState.state = "failed";
            else if (depState.health <= 0.5) depState.state = "degraded";
            if (depState.state !== prevState) {
              newChanges.push({ tick, component_id: depId, prev_state: prevState, new_state: depState.state, prev_health: prevHealth, new_health: depState.health, cause: "cascade_degradation", cause_component: changedId });
              alreadyChanged.add(depId);
            }
          }
          return;
        }

        // Network path
        if (edge.type === "NETWORK_PATH" && changedState.state === "failed") {
          if (edge.criticality === "HARD") {
            depState.state = "unreachable"; depState.health = 0;
            newChanges.push({ tick, component_id: depId, prev_state: prevState, new_state: "unreachable", prev_health: prevHealth, new_health: 0, cause: "network_unreachable", cause_component: changedId });
            alreadyChanged.add(depId);
          } else if (edge.criticality === "REDUNDANT") {
            depState.health = Math.max(0, depState.health - 0.15);
            if (depState.health <= 0.5) depState.state = "degraded";
            if (depState.state !== prevState) {
              newChanges.push({ tick, component_id: depId, prev_state: prevState, new_state: depState.state, prev_health: prevHealth, new_health: depState.health, cause: "redundant_path_loss", cause_component: changedId });
              alreadyChanged.add(depId);
            }
          }
          return;
        }

        // Storage, Service - criticality-based
        if (changedState.state === "failed") {
          if (edge.criticality === "HARD") {
            depState.state = "failed"; depState.health = 0;
            newChanges.push({ tick, component_id: depId, prev_state: prevState, new_state: "failed", prev_health: prevHealth, new_health: 0, cause: "hard_dependency_cascade", cause_component: changedId });
            alreadyChanged.add(depId);
          } else if (edge.criticality === "SOFT") {
            depState.health = Math.max(0, depState.health - 0.5);
            if (depState.health <= 0.2) depState.state = "failed";
            else if (depState.health <= 0.5) depState.state = "degraded";
            if (depState.state !== prevState) {
              newChanges.push({ tick, component_id: depId, prev_state: prevState, new_state: depState.state, prev_health: prevHealth, new_health: depState.health, cause: "soft_dependency_cascade", cause_component: changedId });
              alreadyChanged.add(depId);
            }
          }
        }
      });
    });

    if (newChanges.length === 0) break;
    newChanges.forEach(c => trace.push(c));
    changedThisTick = new Set(newChanges.map(c => c.component_id));
  }

  return { state, trace };
}

// --- Force-directed layout ---
function computeLayout(nodes, edges, width, height) {
  const positions = {};
  const N = nodes.length;

  // Initial positions by type tier
  const tiers = {
    WAN_LINK: 0, INTERNET_GATEWAY: 0.1, FIREWALL: 0.2, ROUTER: 0.2,
    CORE_SWITCH: 0.35, ACCESS_SWITCH: 0.5, LOAD_BALANCER: 0.5,
    SERVER_PHYSICAL: 0.65, HYPERVISOR: 0.7, STORAGE_ARRAY: 0.65, STORAGE_TARGET: 0.75,
    SERVER_VIRTUAL: 0.8, DNS_SERVER: 0.85, DHCP_SERVER: 0.85,
    DOMAIN_CONTROLLER: 0.85, CERTIFICATE_AUTHORITY: 0.85, MONITORING_SERVER: 0.9,
    VDI_BROKER: 0.8, VDI_HOST: 0.9, APPLICATION_SERVICE: 0.95,
  };

  nodes.forEach((n, i) => {
    const tier = tiers[n.type] ?? 0.5;
    const jitter = (Math.sin(i * 7.3 + 1.7) * 0.15);
    positions[n.id] = {
      x: (0.15 + jitter + (i % 5) * 0.15) * width,
      y: (0.05 + tier * 0.9) * height,
      vx: 0, vy: 0,
    };
  });

  // Simple force simulation
  const edgeSet = new Set(edges.map(e => `${e.source_id}-${e.target_id}`));
  for (let iter = 0; iter < 120; iter++) {
    // Repulsion
    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) {
        const a = positions[nodes[i].id], b = positions[nodes[j].id];
        let dx = a.x - b.x, dy = a.y - b.y;
        const dist = Math.max(Math.sqrt(dx*dx + dy*dy), 1);
        const force = 8000 / (dist * dist);
        const fx = (dx / dist) * force, fy = (dy / dist) * force;
        a.vx += fx; a.vy += fy; b.vx -= fx; b.vy -= fy;
      }
    }
    // Attraction
    edges.forEach(e => {
      const a = positions[e.source_id], b = positions[e.target_id];
      if (!a || !b) return;
      let dx = b.x - a.x, dy = b.y - a.y;
      const dist = Math.max(Math.sqrt(dx*dx + dy*dy), 1);
      const force = (dist - 80) * 0.01;
      const fx = (dx / dist) * force, fy = (dy / dist) * force;
      a.vx += fx; a.vy += fy; b.vx -= fx; b.vy -= fy;
    });
    // Tier gravity
    nodes.forEach(n => {
      const p = positions[n.id];
      const tier = tiers[n.type] ?? 0.5;
      const targetY = (0.05 + tier * 0.9) * height;
      p.vy += (targetY - p.y) * 0.05;
    });
    // Apply + dampen
    nodes.forEach(n => {
      const p = positions[n.id];
      p.x += p.vx * 0.3; p.y += p.vy * 0.3;
      p.vx *= 0.7; p.vy *= 0.7;
      p.x = Math.max(40, Math.min(width - 40, p.x));
      p.y = Math.max(30, Math.min(height - 30, p.y));
    });
  }
  return positions;
}

// --- Build LLM prompt from operator view ---
function buildOperatorPrompt(topology, simState, trace, injectedId) {
  const { nodes, edges } = topology;
  const failed = [], degraded = [], healthy = [];
  nodes.forEach(n => {
    const s = simState[n.id];
    if (s.state === "failed" || s.state === "unreachable") failed.push(n);
    else if (s.state === "degraded") degraded.push(n);
    else healthy.push(n);
  });

  // Build fog-of-war view (hide root cause, show symptoms)
  let prompt = `Answer in 2-3 concise paragraphs. No bullet lists. No tick-by-tick traces.\n\n`;
  prompt += `You are an infrastructure operations AI. An incident is in progress.\n\n`;
  prompt += `TOPOLOGY: ${nodes.length} components, ${edges.length} dependencies.\n\n`;
  prompt += `ALERTS:\n`;
  failed.forEach(n => { prompt += `- CRITICAL: ${n.id} (${n.type}) is ${simState[n.id].state}\n`; });
  degraded.forEach(n => { prompt += `- WARNING: ${n.id} (${n.type}) degraded (health: ${(simState[n.id].health*100).toFixed(0)}%)\n`; });
  prompt += `\nHealthy components: ${healthy.length} of ${nodes.length}\n`;
  prompt += `\nWhat is the most likely root cause? What should the operator investigate first?`;
  return prompt;
}

// --- Components ---
function TopologyGraph({ topology, simState, positions, selectedNode, onNodeClick, hoveredNode, onHover, currentTick, trace }) {
  const svgRef = useRef(null);
  if (!topology || !positions) return null;
  const { nodes, edges } = topology;
  const W = 900, H = 600;

  // Find edges that are part of the cascade at current tick
  const cascadeEdges = new Set();
  trace.filter(t => t.tick <= currentTick && t.cause_component).forEach(t => {
    cascadeEdges.add(`${t.component_id}-${t.cause_component}`);
  });

  return (
    <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "100%", background: "#0a0a0f" }}>
      <defs>
        <filter id="glow"><feGaussianBlur stdDeviation="3" result="blur" /><feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
        <filter id="failGlow"><feGaussianBlur stdDeviation="6" result="blur" /><feFlood floodColor="#F44336" floodOpacity="0.6" result="color"/><feComposite in="color" in2="blur" operator="in" result="shadow"/><feMerge><feMergeNode in="shadow"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
        <marker id="arrow" viewBox="0 0 10 6" refX="10" refY="3" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 3 L 0 6 z" fill="#333" /></marker>
        <marker id="arrowRed" viewBox="0 0 10 6" refX="10" refY="3" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 3 L 0 6 z" fill="#F44336" /></marker>
      </defs>

      {/* Edges */}
      {edges.map((e, i) => {
        const a = positions[e.source_id], b = positions[e.target_id];
        if (!a || !b) return null;
        const isCascade = cascadeEdges.has(`${e.source_id}-${e.target_id}`) || cascadeEdges.has(`${e.target_id}-${e.source_id}`);
        return (
          <line key={i} x1={a.x} y1={a.y} x2={b.x} y2={b.y}
            stroke={isCascade ? "#F44336" : "#1a1a2e"} strokeWidth={isCascade ? 2 : 0.8}
            opacity={isCascade ? 0.9 : 0.4} markerEnd={isCascade ? "url(#arrowRed)" : "url(#arrow)"}
          />
        );
      })}

      {/* Nodes */}
      {nodes.map(n => {
        const p = positions[n.id];
        if (!p) return null;
        const s = simState ? simState[n.id] : { state: "healthy", health: 1.0 };
        const color = TYPE_COLORS[n.type] || "#888";
        const stateColor = STATE_COLORS[s.state] || "#4CAF50";
        const isFailed = s.state === "failed" || s.state === "unreachable";
        const isSelected = selectedNode === n.id;
        const isHovered = hoveredNode === n.id;
        const r = isFailed ? 14 : (isSelected ? 16 : 12);

        return (
          <g key={n.id} onClick={() => onNodeClick(n.id)} onMouseEnter={() => onHover(n.id)} onMouseLeave={() => onHover(null)} style={{ cursor: "pointer" }}>
            {isFailed && <circle cx={p.x} cy={p.y} r={r + 8} fill={stateColor} opacity={0.15 + Math.sin(Date.now() / 300) * 0.05} />}
            <circle cx={p.x} cy={p.y} r={r} fill={color} stroke={stateColor} strokeWidth={isSelected ? 3 : 2}
              opacity={s.state === "unreachable" ? 0.4 : 1} filter={isFailed ? "url(#failGlow)" : (isHovered ? "url(#glow)" : undefined)}
            />
            {s.health < 1.0 && s.state !== "failed" && (
              <text x={p.x} y={p.y + 3} textAnchor="middle" fill="#fff" fontSize="8" fontWeight="bold">{(s.health * 100).toFixed(0)}%</text>
            )}
            <text x={p.x} y={p.y + r + 12} textAnchor="middle" fill="#8888aa" fontSize="7" fontFamily="'JetBrains Mono', monospace">
              {n.id}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// --- Main App ---
export default function SableConsole() {
  const [template, setTemplate] = useState("small_office");
  const [seed, setSeed] = useState(42);
  const [topology, setTopology] = useState(null);
  const [positions, setPositions] = useState(null);
  const [simState, setSimState] = useState(null);
  const [trace, setTrace] = useState([]);
  const [selectedNode, setSelectedNode] = useState(null);
  const [hoveredNode, setHoveredNode] = useState(null);
  const [injectedNode, setInjectedNode] = useState(null);
  const [currentTick, setCurrentTick] = useState(0);
  const [maxTick, setMaxTick] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [llmResponse, setLlmResponse] = useState(null);
  const [llmLoading, setLlmLoading] = useState(false);
  const [llmEndpoint, setLlmEndpoint] = useState("http://127.0.0.1:8080");
  const [showConfig, setShowConfig] = useState(false);
  const [view, setView] = useState("topology"); // topology | trace | decisions
  const tickRef = useRef(null);
  const fullSimRef = useRef(null);

  const templates = {
    small_office: generateSmallOffice,
    enterprise_campus: generateEnterpriseCampus,
    vdi_environment: generateVDI,
    healthcare_network: generateHealthcare,
  };

  // Generate topology
  const generate = useCallback(() => {
    const gen = templates[template];
    if (!gen) return;
    const topo = gen(seed);
    setTopology(topo);
    setPositions(computeLayout(topo.nodes, topo.edges, 900, 600));
    const initState = {};
    topo.nodes.forEach(n => { initState[n.id] = { state: "healthy", health: 1.0 }; });
    setSimState(initState);
    setTrace([]); setInjectedNode(null); setCurrentTick(0); setMaxTick(0);
    setSelectedNode(null); setLlmResponse(null); setPlaying(false);
    fullSimRef.current = null;
  }, [template, seed]);

  useEffect(() => { generate(); }, [generate]);

  // Inject failure on selected node
  const injectFailure = useCallback(() => {
    if (!selectedNode || !topology) return;
    const result = simulateCascade(topology, selectedNode);
    fullSimRef.current = result;
    setInjectedNode(selectedNode);
    setTrace(result.trace);
    setMaxTick(Math.max(...result.trace.map(t => t.tick), 0));
    setCurrentTick(0);

    // Set initial state (only injected failure visible)
    const initState = {};
    topology.nodes.forEach(n => { initState[n.id] = { state: "healthy", health: 1.0 }; });
    initState[selectedNode] = { state: "failed", health: 0 };
    setSimState(initState);
    setLlmResponse(null);
  }, [selectedNode, topology]);

  // Tick playback
  useEffect(() => {
    if (!playing || !fullSimRef.current) return;
    tickRef.current = setInterval(() => {
      setCurrentTick(prev => {
        const next = prev + 1;
        if (next > maxTick) { setPlaying(false); return prev; }
        // Update state to match tick
        const newState = {};
        topology.nodes.forEach(n => { newState[n.id] = { state: "healthy", health: 1.0 }; });
        fullSimRef.current.trace.filter(t => t.tick <= next).forEach(t => {
          newState[t.component_id] = { state: t.new_state, health: t.new_health };
        });
        setSimState(newState);
        return next;
      });
    }, 800);
    return () => clearInterval(tickRef.current);
  }, [playing, maxTick, topology]);

  // Jump to tick
  const jumpToTick = (tick) => {
    setCurrentTick(tick);
    if (!fullSimRef.current) return;
    const newState = {};
    topology.nodes.forEach(n => { newState[n.id] = { state: "healthy", health: 1.0 }; });
    fullSimRef.current.trace.filter(t => t.tick <= tick).forEach(t => {
      newState[t.component_id] = { state: t.new_state, health: t.new_health };
    });
    setSimState(newState);
  };

  // Ask LLM
  const askLLM = async () => {
    if (!topology || !simState) return;
    setLlmLoading(true); setLlmResponse(null);
    const prompt = buildOperatorPrompt(topology, simState, trace, injectedNode);
    try {
      const res = await fetch(`${llmEndpoint}/v1/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "local", messages: [{ role: "user", content: prompt }],
          max_tokens: 500, temperature: 0.6, top_p: 0.95,
        }),
      });
      const data = await res.json();
      setLlmResponse(data.choices?.[0]?.message?.content || "No response");
    } catch (err) {
      setLlmResponse(`Connection failed: ${err.message}\n\nMake sure llama-server is running at ${llmEndpoint}`);
    }
    setLlmLoading(false);
  };

  // Stats
  const stats = simState ? {
    healthy: Object.values(simState).filter(s => s.state === "healthy").length,
    degraded: Object.values(simState).filter(s => s.state === "degraded").length,
    failed: Object.values(simState).filter(s => s.state === "failed").length,
    unreachable: Object.values(simState).filter(s => s.state === "unreachable").length,
  } : { healthy: 0, degraded: 0, failed: 0, unreachable: 0 };

  const nodeInfo = selectedNode && topology ? topology.nodes.find(n => n.id === selectedNode) : null;
  const nodeState = selectedNode && simState ? simState[selectedNode] : null;

  return (
    <div style={{ fontFamily: "'JetBrains Mono', 'Fira Code', monospace", background: "#08080d", color: "#c8c8d8", minHeight: "100vh", padding: 0, display: "flex", flexDirection: "column" }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "12px 20px", borderBottom: "1px solid #1a1a2e", background: "#0d0d15" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{ fontSize: 18, fontWeight: 700, color: "#e8e8f0", letterSpacing: 3 }}>SABLE</div>
          <div style={{ fontSize: 10, color: "#555", letterSpacing: 1 }}>OPERATIONS CONSOLE</div>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <select value={template} onChange={e => setTemplate(e.target.value)}
            style={{ background: "#12121e", color: "#aaa", border: "1px solid #222", padding: "6px 10px", fontSize: 11, borderRadius: 3 }}>
            <option value="small_office">Small Office</option>
            <option value="enterprise_campus">Enterprise Campus</option>
            <option value="vdi_environment">VDI Environment</option>
            <option value="healthcare_network">Healthcare Network</option>
          </select>
          <input type="number" value={seed} onChange={e => setSeed(parseInt(e.target.value) || 0)}
            style={{ background: "#12121e", color: "#aaa", border: "1px solid #222", padding: "6px 8px", fontSize: 11, width: 60, borderRadius: 3 }} placeholder="seed" />
          <button onClick={generate} style={{ background: "#1a1a2e", color: "#8888cc", border: "1px solid #2a2a4e", padding: "6px 14px", fontSize: 11, cursor: "pointer", borderRadius: 3, letterSpacing: 1 }}>GENERATE</button>
          <button onClick={() => setShowConfig(!showConfig)} style={{ background: "none", color: "#555", border: "1px solid #222", padding: "6px 8px", fontSize: 11, cursor: "pointer", borderRadius: 3 }}>⚙</button>
        </div>
      </div>

      {/* Config panel */}
      {showConfig && (
        <div style={{ padding: "10px 20px", background: "#0a0a12", borderBottom: "1px solid #1a1a2e", display: "flex", gap: 12, alignItems: "center" }}>
          <span style={{ fontSize: 10, color: "#555" }}>LLM ENDPOINT:</span>
          <input value={llmEndpoint} onChange={e => setLlmEndpoint(e.target.value)}
            style={{ background: "#12121e", color: "#aaa", border: "1px solid #222", padding: "5px 8px", fontSize: 11, width: 280, borderRadius: 3 }} />
          <span style={{ fontSize: 9, color: "#444" }}>Start llama-server with --port 8080</span>
        </div>
      )}

      {/* Main content */}
      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
        {/* Graph area */}
        <div style={{ flex: 1, position: "relative", borderRight: "1px solid #1a1a2e" }}>
          {/* Status bar */}
          <div style={{ display: "flex", gap: 16, padding: "8px 16px", borderBottom: "1px solid #111", fontSize: 10, background: "#0a0a10" }}>
            <span style={{ color: STATE_COLORS.healthy }}>● {stats.healthy} healthy</span>
            <span style={{ color: STATE_COLORS.degraded }}>● {stats.degraded} degraded</span>
            <span style={{ color: STATE_COLORS.failed }}>● {stats.failed} failed</span>
            <span style={{ color: STATE_COLORS.unreachable }}>● {stats.unreachable} unreachable</span>
            {injectedNode && <span style={{ color: "#F44336", marginLeft: "auto" }}>INCIDENT ACTIVE — tick {currentTick}/{maxTick}</span>}
          </div>

          {/* Graph */}
          <TopologyGraph topology={topology} simState={simState} positions={positions}
            selectedNode={selectedNode} onNodeClick={setSelectedNode} hoveredNode={hoveredNode} onHover={setHoveredNode}
            currentTick={currentTick} trace={trace} />

          {/* Playback controls */}
          {injectedNode && (
            <div style={{ position: "absolute", bottom: 16, left: 16, right: 16, display: "flex", alignItems: "center", gap: 10, background: "#0d0d18ee", padding: "8px 14px", borderRadius: 4, border: "1px solid #1a1a2e" }}>
              <button onClick={() => setPlaying(!playing)} style={{ background: playing ? "#F44336" : "#1a1a2e", color: "#ddd", border: "1px solid #2a2a4e", padding: "4px 12px", fontSize: 11, cursor: "pointer", borderRadius: 3 }}>
                {playing ? "⏸" : "▶"}
              </button>
              <input type="range" min={0} max={maxTick} value={currentTick} onChange={e => { setPlaying(false); jumpToTick(parseInt(e.target.value)); }}
                style={{ flex: 1, accentColor: "#F44336" }} />
              <button onClick={() => { setPlaying(false); jumpToTick(0); }} style={{ background: "#1a1a2e", color: "#888", border: "1px solid #222", padding: "4px 8px", fontSize: 10, cursor: "pointer", borderRadius: 3 }}>⏮</button>
              <button onClick={() => { setPlaying(false); jumpToTick(maxTick); }} style={{ background: "#1a1a2e", color: "#888", border: "1px solid #222", padding: "4px 8px", fontSize: 10, cursor: "pointer", borderRadius: 3 }}>⏭</button>
            </div>
          )}
        </div>

        {/* Right panel */}
        <div style={{ width: 340, display: "flex", flexDirection: "column", background: "#0a0a12", overflow: "auto" }}>
          {/* Node inspector */}
          <div style={{ padding: 14, borderBottom: "1px solid #1a1a2e" }}>
            <div style={{ fontSize: 10, color: "#555", letterSpacing: 1, marginBottom: 8 }}>COMPONENT INSPECTOR</div>
            {nodeInfo ? (
              <div>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                  <span style={{ fontSize: 18 }}>{TYPE_ICONS[nodeInfo.type] || "?"}</span>
                  <div>
                    <div style={{ fontSize: 13, color: "#e8e8f0", fontWeight: 600 }}>{nodeInfo.id}</div>
                    <div style={{ fontSize: 10, color: TYPE_COLORS[nodeInfo.type] || "#888" }}>{nodeInfo.type.replace(/_/g, " ")}</div>
                  </div>
                </div>
                {nodeState && (
                  <div style={{ display: "flex", gap: 12, fontSize: 11 }}>
                    <span>State: <span style={{ color: STATE_COLORS[nodeState.state] }}>{nodeState.state}</span></span>
                    <span>Health: <span style={{ color: nodeState.health > 0.5 ? "#4CAF50" : nodeState.health > 0.2 ? "#FF9800" : "#F44336" }}>{(nodeState.health * 100).toFixed(0)}%</span></span>
                  </div>
                )}
                {!injectedNode && (
                  <button onClick={injectFailure} style={{ marginTop: 10, background: "#2a1010", color: "#F44336", border: "1px solid #442222", padding: "6px 14px", fontSize: 11, cursor: "pointer", borderRadius: 3, width: "100%", letterSpacing: 1 }}>
                    ⚡ INJECT FAILURE
                  </button>
                )}
              </div>
            ) : (
              <div style={{ fontSize: 11, color: "#444" }}>Click a node to inspect</div>
            )}
          </div>

          {/* Cascade trace */}
          {trace.length > 0 && (
            <div style={{ padding: 14, borderBottom: "1px solid #1a1a2e", flex: 1, overflow: "auto" }}>
              <div style={{ fontSize: 10, color: "#555", letterSpacing: 1, marginBottom: 8 }}>CASCADE TRACE</div>
              <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                {trace.filter(t => t.tick <= currentTick).map((t, i) => (
                  <div key={i} style={{ fontSize: 10, padding: "4px 6px", background: t.new_state === "failed" ? "#1a0a0a" : "#0d0d18", borderRadius: 2, borderLeft: `2px solid ${STATE_COLORS[t.new_state]}` }}>
                    <span style={{ color: "#555" }}>t={t.tick}</span>{" "}
                    <span style={{ color: "#ccc" }}>{t.component_id}</span>{" "}
                    <span style={{ color: STATE_COLORS[t.new_state] }}>{t.new_state}</span>
                    {t.cause_component && <span style={{ color: "#555" }}> ← {t.cause_component}</span>}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* AI Analysis */}
          {injectedNode && currentTick >= maxTick && (
            <div style={{ padding: 14 }}>
              <div style={{ fontSize: 10, color: "#555", letterSpacing: 1, marginBottom: 8 }}>AI ANALYSIS</div>
              {!llmResponse && !llmLoading && (
                <button onClick={askLLM} style={{ background: "#101025", color: "#6666cc", border: "1px solid #2a2a4e", padding: "8px 14px", fontSize: 11, cursor: "pointer", borderRadius: 3, width: "100%", letterSpacing: 1 }}>
                  🧠 ASK NEMOTRON
                </button>
              )}
              {llmLoading && <div style={{ fontSize: 11, color: "#555" }}>Analyzing...</div>}
              {llmResponse && (
                <div style={{ fontSize: 11, color: "#aaa", lineHeight: 1.5, whiteSpace: "pre-wrap", maxHeight: 300, overflow: "auto" }}>
                  {llmResponse}
                </div>
              )}
            </div>
          )}

          {/* Reset */}
          {injectedNode && (
            <div style={{ padding: 14, borderTop: "1px solid #1a1a2e" }}>
              <button onClick={generate} style={{ background: "#1a1a2e", color: "#888", border: "1px solid #222", padding: "6px 14px", fontSize: 11, cursor: "pointer", borderRadius: 3, width: "100%", letterSpacing: 1 }}>↺ RESET</button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
