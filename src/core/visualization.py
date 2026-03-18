"""Topology visualization using matplotlib and networkx."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

from sable_sim.core.graph import InfrastructureGraph
from sable_sim.core.component import ComponentType, ComponentState

logger = logging.getLogger(__name__)

STATE_COLORS = {
    ComponentState.HEALTHY: "#4CAF50",
    ComponentState.DEGRADED: "#FF9800",
    ComponentState.FAILED: "#F44336",
    ComponentState.UNREACHABLE: "#9E9E9E",
}

TYPE_COLORS = {
    ComponentType.CORE_SWITCH: "#2196F3",
    ComponentType.ACCESS_SWITCH: "#64B5F6",
    ComponentType.FIREWALL: "#FF5722",
    ComponentType.ROUTER: "#FF7043",
    ComponentType.LOAD_BALANCER: "#AB47BC",
    ComponentType.SERVER_PHYSICAL: "#66BB6A",
    ComponentType.SERVER_VIRTUAL: "#81C784",
    ComponentType.HYPERVISOR: "#43A047",
    ComponentType.STORAGE_ARRAY: "#FFA726",
    ComponentType.STORAGE_TARGET: "#FFB74D",
    ComponentType.VDI_BROKER: "#5C6BC0",
    ComponentType.VDI_HOST: "#7986CB",
    ComponentType.DNS_SERVER: "#26A69A",
    ComponentType.DHCP_SERVER: "#4DB6AC",
    ComponentType.DOMAIN_CONTROLLER: "#EC407A",
    ComponentType.CERTIFICATE_AUTHORITY: "#F48FB1",
    ComponentType.MONITORING_SERVER: "#78909C",
    ComponentType.WAN_LINK: "#8D6E63",
    ComponentType.INTERNET_GATEWAY: "#EF5350",
    ComponentType.APPLICATION_SERVICE: "#9CCC65",
}


class TopologyVisualizer:
    """Renders infrastructure topologies as images."""

    def render(
        self,
        graph: InfrastructureGraph,
        output_path: str | Path,
        title: str = "Infrastructure Topology",
        show_health: bool = True,
        highlight_failed: bool = True,
        figsize: tuple[int, int] = (20, 14),
    ) -> str:
        """Render topology to PNG. Returns output file path."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_title(title, fontsize=16, fontweight="bold")

        # Build networkx graph for layout
        G = nx.DiGraph()
        components = graph.get_all_components()
        for c in components:
            G.add_node(c.id)
        for d in graph.get_all_dependencies():
            G.add_edge(d.source_id, d.target_id)

        # Layout
        try:
            pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42)
        except Exception:
            pos = nx.kamada_kawai_layout(G)

        # Draw edges
        nx.draw_networkx_edges(
            G, pos, ax=ax, alpha=0.3, edge_color="#888888",
            arrows=True, arrowsize=8, connectionstyle="arc3,rad=0.05",
        )

        # Draw nodes
        for comp in components:
            if comp.id not in pos:
                continue
            x, y = pos[comp.id]
            color = TYPE_COLORS.get(comp.type, "#CCCCCC")
            edge_color = STATE_COLORS.get(comp.state, "#000000")
            size = 300 + len(graph.get_dependents(comp.id)) * 50

            if highlight_failed and comp.state in (
                ComponentState.FAILED,
                ComponentState.UNREACHABLE,
            ):
                edge_color = STATE_COLORS[comp.state]
                ax.scatter(
                    [x], [y], s=size * 1.5, c=edge_color, alpha=0.3, zorder=1
                )

            ax.scatter(
                [x], [y], s=size, c=color, edgecolors=edge_color,
                linewidths=2.5, zorder=2,
            )

            label = comp.id
            if show_health and comp.health < 1.0:
                label += f"\n{comp.health:.0%}"
            ax.annotate(
                label, (x, y), fontsize=6, ha="center", va="bottom",
                xytext=(0, 8), textcoords="offset points",
            )

        # Legend
        type_patches = [
            mpatches.Patch(color=TYPE_COLORS.get(t, "#CCC"), label=t.value)
            for t in ComponentType
            if any(c.type == t for c in components)
        ]
        state_patches = [
            mpatches.Patch(color=STATE_COLORS[s], label=f"Border: {s.value}")
            for s in ComponentState
        ]
        ax.legend(
            handles=type_patches + state_patches,
            loc="upper left", fontsize=7, ncol=2, framealpha=0.9,
        )

        ax.axis("off")
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info("Rendered topology to %s", output_path)
        return str(output_path)

    def render_cascade(
        self,
        graph: InfrastructureGraph,
        cascade_trace: list[dict[str, Any]],
        output_dir: str | Path,
        title: str = "Cascade Propagation",
    ) -> list[str]:
        """Render one image per tick showing cascade progression.

        Returns list of output file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Group changes by tick
        ticks: dict[int, list[dict]] = {}
        for change in cascade_trace:
            t = change.get("tick", 0)
            ticks.setdefault(t, []).append(change)

        paths: list[str] = []
        max_tick = max(ticks.keys()) if ticks else 0

        # Build a working copy of component states
        states: dict[str, str] = {
            c.id: str(c.state) for c in graph.get_all_components()
        }
        healths: dict[str, float] = {
            c.id: c.health for c in graph.get_all_components()
        }

        for tick in range(max_tick + 1):
            # Apply changes for this tick
            for change in ticks.get(tick, []):
                cid = change.get("component_id", "")
                states[cid] = change.get("new_state", states.get(cid, "healthy"))
                healths[cid] = change.get("new_health", healths.get(cid, 1.0))

            # Temporarily update component states for rendering
            for comp in graph.get_all_components():
                comp.state = ComponentState(states.get(comp.id, "healthy"))
                comp.health = healths.get(comp.id, 1.0)

            out = output_dir / f"tick_{tick:03d}.png"
            self.render(
                graph, out,
                title=f"{title} — Tick {tick}",
                figsize=(16, 11),
            )
            paths.append(str(out))

        return paths
