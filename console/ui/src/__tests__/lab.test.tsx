import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { LabControls } from "../components/LabControls";
import { api } from "../api";

afterEach(() => vi.restoreAllMocks());

describe("LabControls", () => {
  it("reports a fault script that exits non-zero as a failure, not an injection", async () => {
    vi.spyOn(api, "labFault").mockResolvedValue({ fault: "kill_primary", exit_code: 1, output: "x\ndb is not running" });
    render(<LabControls />);
    fireEvent.click(screen.getByTestId("fault-kill_primary"));
    await waitFor(() => expect(screen.getByTestId("lab-msg")).toHaveTextContent("kill_primary failed (exit 1): db is not running"));
  });

  it("offers Heal all and runs heal_all", async () => {
    const spy = vi.spyOn(api, "labFault").mockResolvedValue({ fault: "heal_all", exit_code: 0, output: "" });
    render(<LabControls />);
    fireEvent.click(screen.getByTestId("fault-heal_all"));
    expect(spy).toHaveBeenCalledWith("heal_all");
    await waitFor(() => expect(screen.getByTestId("lab-msg")).toHaveTextContent("lab healed"));
  });
});
