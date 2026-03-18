#!/usr/bin/env python3
"""CLI tool to visualize SABLE topologies."""

import click
import logging

from sable_sim.generation.templates import TemplateGenerator
from sable_sim.export.visualization import TopologyVisualizer
from sable_sim.utils.random import SeededRandom
from sable_sim.utils.logging import setup_logging


@click.command()
@click.option("--template", "-t", default="small_office", help="Topology template")
@click.option("--output", "-o", default="./topology.png", help="Output PNG path")
@click.option("--seed", "-s", default=42, type=int, help="Random seed")
@click.option("--show-health", is_flag=True, default=True, help="Show health values")
@click.option("--figsize", default="20x14", help="Figure size (WxH)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
def main(
    template: str,
    output: str,
    seed: int,
    show_health: bool,
    figsize: str,
    verbose: bool,
) -> None:
    """Render an infrastructure topology to PNG."""
    setup_logging("DEBUG" if verbose else "INFO")
    logger = logging.getLogger(__name__)

    w, h = (int(x) for x in figsize.split("x"))

    rng = SeededRandom(seed)
    tmpl_gen = TemplateGenerator(rng)
    graph = tmpl_gen.generate(template)

    logger.info("Rendering %s topology (%d nodes)...", template, graph.node_count)

    viz = TopologyVisualizer()
    out_path = viz.render(
        graph,
        output_path=output,
        title=f"SABLE - {template}",
        show_health=show_health,
        figsize=(w, h),
    )
    click.echo(f"Rendered to: {out_path}")


if __name__ == "__main__":
    main()
