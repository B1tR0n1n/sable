import { useState } from "react";
import { api } from "../api";

/** The four lab faults (PLAN.md Phase 8): scripts under console/lab/faults/<name>.sh */
export const LAB_FAULTS: Array<{ name: string; label: string }> = [
  { name: "stop_service", label: "Stop service" },
  { name: "corrupt_config", label: "Corrupt config" },
  { name: "kill_primary", label: "Kill primary" },
  { name: "poison_dns", label: "Poison DNS" },
];

export function LabControls() {
  const [busy, setBusy] = useState<string | null>(null);
  const [msg, setMsg] = useState<string | null>(null);

  async function inject(name: string) {
    setBusy(name);
    setMsg(null);
    try {
      await api.labFault(name);
      setMsg(`injected ${name}`);
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
      {msg && <span className="meta">{msg}</span>}
    </div>
  );
}
