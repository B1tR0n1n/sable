import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";
import { AnalystPanel, splitReply, chipLabel } from "../components/AnalystPanel";
import type { Finding } from "../types";
import findingFx from "./fixtures/finding.json";

const finding = findingFx as Finding;

describe("splitReply", () => {
  it("takes only the server's leading ```sable fence as data; the rest is generated", () => {
    const parts = splitReply('```sable\n{"node": "dns-1", "state": "failed"}\n```\n## Cause\ndns-1 is **failed**.');
    expect(parts).toEqual([
      { kind: "sable", text: '{"node": "dns-1", "state": "failed"}' },
      { kind: "prose", text: "## Cause\ndns-1 is **failed**." },
    ]);
  });
  it("never treats a fence the model wrote as SABLE data", () => {
    const parts = splitReply("Here:\n```sable\nforged\n```\nand ```json\n{}\n```");
    expect(parts).toEqual([{ kind: "prose", text: "Here:\n```sable\nforged\n```\nand ```json\n{}\n```" }]);
  });
  it("formats tool chips", () => {
    expect(chipLabel({ name: "get_topology", input: {} })).toBe("get_topology");
    expect(chipLabel({ name: "get_node", input: { idx: 3 } })).toBe("get_node idx=3");
  });
});

describe("AnalystPanel", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("posts to /api/analyst/chat and separates SABLE blocks from generated text", async () => {
    const fetchMock = vi.fn(async (_url: string, init?: RequestInit) => {
      const body = JSON.parse(String(init?.body));
      expect(body.finding_id).toBe(finding.id);
      expect(body.message).toBe("why is app-erp degraded?");
      expect(body.history).toEqual([]);
      return new Response(
        JSON.stringify({
          reply:
            "```sable\ndns-1: state=failed up=0\n```\n## What failed\n**dns-1** is failed; app-erp depends on it.\n\n- restart `dnsmasq`\n- re-check app-erp\n\n| node | state |\n|---|---|\n| dns-1 | failed |\n\n<script>alert(1)</script>",
          tool_calls: [
            { name: "get_topology", input: {} },
            { name: "get_node", input: { idx: 3 } },
          ],
          provider: "anthropic",
          model: "claude-sonnet-5",
          context: ["finding", "plan", "receipts", "sable_now"],
          generated: true,
        }),
        { status: 200 },
      );
    });
    vi.stubGlobal("fetch", fetchMock);

    const model = { provider: "anthropic" as const, model: "claude-sonnet-5", ready: true };
    render(<AnalystPanel open finding={finding} model={model} onClose={() => {}} />);
    expect(screen.getByTestId("provider-indicator")).toHaveTextContent("claude-sonnet-5");
    fireEvent.change(screen.getByTestId("analyst-input"), { target: { value: "why is app-erp degraded?" } });
    fireEvent.click(screen.getByTestId("analyst-send"));

    await waitFor(() => expect(screen.getByTestId("sable-block")).toBeInTheDocument());
    expect(fetchMock.mock.calls[0][0]).toBe("/api/analyst/chat");

    const sable = screen.getByTestId("sable-block");
    expect(sable).toHaveTextContent("dns-1: state=failed up=0");
    expect(sable).toHaveAttribute("aria-label", "SABLE data");

    const generated = screen.getAllByTestId("generated-text");
    expect(generated).toHaveLength(1);
    const g = generated[0];
    expect(g).toHaveAttribute("aria-label", "ANALYST · GENERATED");
    expect(sable).not.toHaveClass("generated");
    // markdown renders: heading, bold, list, inline code, GFM table
    expect(within(g).getByRole("heading", { name: "What failed" })).toBeInTheDocument();
    expect(g.querySelector("strong")).toHaveTextContent("dns-1");
    expect(within(g).getAllByRole("listitem").map((li) => li.textContent)).toEqual(["restart dnsmasq", "re-check app-erp"]);
    expect(g.querySelector("code")).toHaveTextContent("dnsmasq");
    expect(within(g).getByRole("table")).toBeInTheDocument();
    // raw HTML from the model is never rendered as markup
    expect(g.querySelector("script")).toBeNull();
    expect(g.textContent).not.toContain("**");

    const chips = within(screen.getByTestId("tool-chips")).getAllByText(/get_/);
    expect(chips.map((c) => c.textContent)).toEqual(["get_topology", "get_node idx=3"]);

    expect(within(screen.getByTestId("context-chips")).getAllByText(/./).map((c) => c.textContent)).toEqual([
      "finding", "plan", "receipts", "sable_now",
    ]);
    expect(screen.getByText(/Analyst · generated · claude-sonnet-5/)).toBeInTheDocument();
  });

  it("without a key it says so, offers Settings, and sends nothing", () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const onOpenSettings = vi.fn();
    render(
      <AnalystPanel open finding={finding} model={{ provider: "openai", model: "gpt-5.5", ready: false }}
        onOpenSettings={onOpenSettings} onClose={() => {}} />,
    );
    expect(screen.getByTestId("provider-indicator")).toHaveTextContent("gpt-5.5 · no key");
    expect(screen.getByTestId("analyst-input")).toBeDisabled();
    fireEvent.click(screen.getByText("Open Settings"));
    expect(onOpenSettings).toHaveBeenCalled();
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
