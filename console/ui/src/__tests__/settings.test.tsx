import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { SettingsPanel } from "../components/SettingsPanel";
import type { ModelSettings } from "../types";

const base = (over: Partial<ModelSettings> = {}): ModelSettings => ({
  provider: "anthropic",
  model: "",
  effective_model: "claude-sonnet-5",
  ready: false,
  providers: [
    { id: "anthropic", label: "Anthropic (Claude)", key_env: "ANTHROPIC_API_KEY", default_model: "claude-sonnet-5", key: "none" },
    { id: "openai", label: "OpenAI (GPT)", key_env: "OPENAI_API_KEY", default_model: "gpt-5.5", key: "none" },
  ],
  ...over,
});

function mockServer(state: ModelSettings) {
  const calls: Array<{ url: string; method: string; body: unknown }> = [];
  let cur = state;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init?: RequestInit) => {
      const method = init?.method ?? "GET";
      const body = init?.body ? JSON.parse(String(init.body)) : null;
      calls.push({ url, method, body });
      if (url === "/api/settings/model" && method === "PUT") {
        const b = body as { provider: "anthropic" | "openai"; model: string; key?: string };
        cur = base({
          provider: b.provider,
          model: b.model,
          effective_model: b.model || (b.provider === "openai" ? "gpt-5.5" : "claude-sonnet-5"),
          ready: true,
          providers: cur.providers.map((p) => (p.id === b.provider && b.key ? { ...p, key: "store" } : p)),
        });
        return new Response(JSON.stringify(cur), { status: 200 });
      }
      if (url === "/api/settings/model") return new Response(JSON.stringify(cur), { status: 200 });
      if (url.startsWith("/api/settings/models"))
        return new Response(JSON.stringify({ provider: "openai", models: [{ id: "gpt-5.5", name: "gpt-5.5" }] }), { status: 200 });
      if (url === "/api/settings/model/test")
        return new Response(JSON.stringify({ ok: true, provider: "openai", model: "gpt-5.5", reply: "ready" }), { status: 200 });
      return new Response("{}", { status: 404 });
    }),
  );
  return calls;
}

afterEach(() => vi.unstubAllGlobals());

describe("SettingsPanel", () => {
  it("switches to GPT, saves the key write-only, and never shows it back", async () => {
    const calls = mockServer(base());
    const onSaved = vi.fn();
    render(<SettingsPanel open onClose={() => {}} onSaved={onSaved} />);
    await waitFor(() => expect(screen.getByTestId("settings-keystate")).toHaveTextContent("no key"));
    expect(screen.getByTestId("settings-test")).toBeDisabled();

    fireEvent.change(screen.getByTestId("settings-provider"), { target: { value: "openai" } });
    fireEvent.change(screen.getByTestId("settings-key"), { target: { value: "sk-live-abc" } });
    fireEvent.click(screen.getByTestId("settings-save"));

    await waitFor(() => expect(screen.getByTestId("settings-msg")).toHaveTextContent("saved · openai · gpt-5.5"));
    const put = calls.find((c) => c.method === "PUT")!;
    expect(put.body).toEqual({ provider: "openai", model: "", key: "sk-live-abc" });
    expect((screen.getByTestId("settings-key") as HTMLInputElement).value).toBe("");
    expect(screen.getByTestId("settings-key")).toHaveAttribute("type", "password");
    expect(screen.getByTestId("settings-keystate")).toHaveTextContent("key set · stored");
    expect(onSaved).toHaveBeenCalled();
    // with a key, the live model list loads
    await waitFor(() => expect(screen.getByTestId("settings-models-note")).toHaveTextContent("1 models available"));

    fireEvent.click(screen.getByTestId("settings-test"));
    await waitFor(() => expect(screen.getByTestId("settings-test-result")).toHaveTextContent('ok · openai · gpt-5.5 answered "ready"'));
  });

  it("says when the key comes from the environment", async () => {
    mockServer(base({ ready: true, providers: base().providers.map((p) => (p.id === "anthropic" ? { ...p, key: "env" } : p)) }));
    render(<SettingsPanel open onClose={() => {}} />);
    await waitFor(() => expect(screen.getByTestId("settings-keystate")).toHaveTextContent("key from environment"));
    expect(screen.getByText(/ANTHROPIC_API_KEY is set in the environment/)).toBeInTheDocument();
  });
});
