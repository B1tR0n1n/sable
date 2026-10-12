import { useEffect, useState } from "react";
import { api } from "../api";
import type { ModelProvider, ModelSettings, ModelTestResult } from "../types";

export interface SettingsPanelProps {
  open: boolean;
  onClose: () => void;
  /** after a save: the strip's model line changes */
  onSaved?: (s: ModelSettings) => void;
}

const KEY_STATE: Record<string, string> = {
  store: "key set · stored",
  env: "key from environment",
  none: "no key",
};

/** Settings → Model: Claude or GPT, which model, and the API key. The key
 *  is write-only — pasted here, stored by the server in OVERLORD's 0600 key
 *  store, never sent back. One choice drives the analyst and the LLM planner. */
export function SettingsPanel({ open, onClose, onSaved }: SettingsPanelProps) {
  const [saved, setSaved] = useState<ModelSettings | null>(null);
  const [provider, setProvider] = useState<ModelProvider>("anthropic");
  const [model, setModel] = useState("");
  const [key, setKey] = useState("");
  const [models, setModels] = useState<Array<{ id: string; name: string }>>([]);
  const [modelsNote, setModelsNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [test, setTest] = useState<ModelTestResult | null>(null);

  useEffect(() => {
    if (!open) return;
    setMsg(null);
    setTest(null);
    setKey("");
    api
      .modelSettings()
      .then((s) => {
        setSaved(s);
        setProvider(s.provider);
        setModel(s.model);
      })
      .catch((e: Error) => setMsg({ ok: false, text: `settings: ${e.message}` }));
  }, [open]);

  const info = saved?.providers.find((p) => p.id === provider) ?? null;

  async function loadModels(p: ModelProvider) {
    setModels([]);
    setModelsNote("listing models…");
    try {
      const r = await api.modelList(p);
      setModels(r.models);
      setModelsNote(`${r.models.length} models available`);
    } catch (e) {
      setModelsNote(`could not list models: ${(e as Error).message}`);
    }
  }

  useEffect(() => {
    if (open && info && info.key !== "none") void loadModels(provider);
    else setModelsNote(open ? "set a key to list this provider's models" : "");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, provider, info?.key]);

  async function save(extra: { clear_key?: boolean } = {}) {
    setBusy(true);
    setMsg(null);
    setTest(null);
    try {
      const s = await api.putModelSettings({
        provider,
        model: model.trim(),
        ...(key.trim() ? { key: key.trim() } : {}),
        ...extra,
      });
      setSaved(s);
      setModel(s.model);
      setKey("");
      setMsg({ ok: true, text: extra.clear_key ? "key cleared" : `saved · ${s.provider} · ${s.effective_model}` });
      onSaved?.(s);
    } catch (e) {
      setMsg({ ok: false, text: (e as Error).message });
    } finally {
      setBusy(false);
    }
  }

  async function runTest() {
    setBusy(true);
    setTest(null);
    try {
      setTest(await api.modelTest());
    } catch (e) {
      const body = (e as { body?: ModelTestResult }).body;
      setTest(body && typeof body === "object" ? body : { ok: false, provider, model, error: (e as Error).message });
    } finally {
      setBusy(false);
    }
  }

  if (!open) return null;
  const dirty = !!saved && (provider !== saved.provider || model.trim() !== saved.model || key.trim() !== "");

  return (
    <div className="modal-back" onClick={onClose} data-testid="settings-modal">
      <div className="modal" role="dialog" aria-label="Settings" onClick={(e) => e.stopPropagation()}>
        <div className="panel-head">
          <span className="label">08 — Settings · Model</span>
          <button className="tiny quiet" onClick={onClose} data-testid="settings-close">
            Close
          </button>
        </div>
        <div className="panel-body">
          <p className="meta" style={{ marginTop: 0 }}>
            One model for the analyst and the LLM planner. It proposes and explains; OVERLORD still executes only
            approved catalog actions.
          </p>

          <div className="field">
            <span className="label">Provider</span>
            <select
              value={provider}
              onChange={(e) => {
                setProvider(e.target.value as ModelProvider);
                setModel("");
                setTest(null);
              }}
              data-testid="settings-provider"
            >
              {(saved?.providers ?? []).map((p) => (
                <option key={p.id} value={p.id}>
                  {p.label}
                </option>
              ))}
            </select>
          </div>

          <div className="field">
            <span className="label">
              API key{" "}
              <span className={`tag ${info?.key === "none" ? "tag-red" : "tag-green"}`} data-testid="settings-keystate">
                {info ? KEY_STATE[info.key] : "—"}
              </span>
            </span>
            <input
              type="password"
              autoComplete="off"
              placeholder={info?.key === "none" ? `paste your ${info?.label ?? ""} key` : "paste to replace — never shown again"}
              value={key}
              onChange={(e) => setKey(e.target.value)}
              data-testid="settings-key"
            />
            <span className="meta">
              {info?.key === "env"
                ? `${info.key_env} is set in the environment and wins over a stored key.`
                : "Stored on this machine only, in OVERLORD's key store (mode 600)."}
            </span>
          </div>

          <div className="field">
            <span className="label">
              Model{" "}
              <button className="tiny quiet" type="button" onClick={() => void loadModels(provider)} disabled={!info || info.key === "none"} data-testid="settings-refresh">
                refresh list
              </button>
            </span>
            <input
              list="settings-models"
              placeholder={`default · ${info?.default_model ?? ""}`}
              value={model}
              onChange={(e) => setModel(e.target.value)}
              data-testid="settings-model"
            />
            <datalist id="settings-models">
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name !== m.id ? m.name : undefined}
                </option>
              ))}
            </datalist>
            <span className="meta" data-testid="settings-models-note">
              {modelsNote}
            </span>
          </div>

          <div className="controls" style={{ marginTop: 14 }}>
            <button onClick={() => void save()} disabled={busy || !dirty} data-testid="settings-save">
              Save
            </button>
            <button className="quiet" onClick={() => void runTest()} disabled={busy || dirty || !saved?.ready} data-testid="settings-test"
              title={dirty ? "save first" : !saved?.ready ? "set a key first" : "one tiny call on the saved model"}>
              Test
            </button>
            {info?.key === "store" && (
              <button className="danger tiny" onClick={() => void save({ clear_key: true })} disabled={busy} data-testid="settings-clear">
                Clear key
              </button>
            )}
          </div>
          {msg && (
            <p className={`meta ${msg.ok ? "sem-pass" : "sem-fail"}`} data-testid="settings-msg">
              {msg.text}
            </p>
          )}
          {test && (
            <p className={`meta ${test.ok ? "sem-pass" : "sem-fail"}`} data-testid="settings-test-result">
              {test.ok ? `ok · ${test.provider} · ${test.model} answered "${test.reply}"` : `failed · ${test.error}`}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
