import { useState } from "react";
import { api } from "../api";

/** The four lab faults (PLAN.md Phase 8): scripts under console/lab/faults/<name>.sh */
export const LAB_FAULTS: Array<{ name: string; label: string }> = [
  { name: "stop_service", label: "Stop service" },
  { name: "corrupt_config", label: "Corrupt config" },
  { name: "kill_primary", label: "Kill primary" },
  { name: "poison_dns", label: "Poison DNS" },
];

/** Puts the whole lab back to pristine outside the loop (faults/heal_all.sh). */
export const LAB_HEAL = { name: "heal_all", label: "Heal all" };

interface FaultResult {
  fault?: string;
  exit_code?: number;
  output?: string;
}

export function LabControls() {
  const [busy, setBusy] = useState<string | null>(null);
  const [msg, setMsg] = useState<string | null>(null);

  async function inject(name: string) {
    setBusy(name);
    setMsg(null);
    try {
      const res = (await api.labFault(name)) as FaultResult;
      if (res && typeof res.exit_code === "number" && res.exit_code !== 0) {
        const last = (res.output ?? "").trim().split("\n").pop() ?? "";
        setMsg(`${name} failed (exit ${res.exit_code})${last ? `: ${last}` : ""}`);
      } else {
        setMsg(name === LAB_HEAL.name ? "lab healed" : `injected ${name}`);
      }
    } catch (e) {
      setMsg(`${name}: ${(e as Error).message}`);
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="lab" data-testid="lab-controls">
      <span className="label">Lab</span>
      {LAB_FAULTS.map((f) => (
        <button
          key={f.name}
          className="tiny danger"
          disabled={busy !== null}
          onClick={() => void inject(f.name)}
          data-testid={`fault-${f.name}`}
        >
          {f.label}
        </button>
      ))}
      <button
        className="tiny quiet"
        disabled={busy !== null}
        onClick={() => void inject(LAB_HEAL.name)}
        data-testid={`fault-${LAB_HEAL.name}`}
        title="restore every lab service and config outside the loop"
      >
        {LAB_HEAL.label}
      </button>
      {busy && <span className="meta">running {busy}…</span>}
      {!busy && msg && <span className="meta" data-testid="lab-msg">{msg}</span>}
    </div>
  );
}
