#!/usr/bin/env python3
"""CLI tool to run a single SABLE scenario and inspect results."""

from pathlib import Path

import click
import logging

from sable_sim.simulation.engine import SimulationEngine
from sable_sim.simulation.failure_injection import FailureCategory
from sable_sim.generation.templates import TemplateGenerator
from sable_sim.export.training_data import TrainingDataExporter
from sable_sim.utils.random import SeededRandom
from sable_sim.utils.logging import setup_logging


@click.command()
@click.option("--template", "-t", default="small_office", help="Topology template")
@click.option("--failure", "-f", default=None, help="Failure category name")
@click.option("--difficulty", "-d", default="medium", help="easy / medium / hard")
@click.option("--output", "-o", default="./data/debug/", help="Output directory")
@click.option("--seed", "-s", default=42, type=int, help="Random seed")
@click.option("--max-ticks", default=50, type=int, help="Maximum simulation ticks")
@click.option("--verbose", "-v", is_flag=True, help="DEBUG logging")
@click.option("--print-trace", is_flag=True, help="Print cascade trace to stdout")
def main(
    template: str,
    failure: str | None,
    difficulty: str,
    output: str,
    seed: int,
    max_ticks: int,
    verbose: bool,
    print_trace: bool,
) -> None:
    """Run a single SABLE scenario for inspection."""
    setup_logging("DEBUG" if verbose else "INFO")
    logger = logging.getLogger(__name__)

    rng = SeededRandom(seed)

    logger.info("Generating %s topology...", template)
    tmpl_gen = TemplateGenerator(rng)
    graph = tmpl_gen.generate(template)
    logger.info("Topology: %d nodes, %d edges", graph.node_count, graph.edge_count)

    categories = None
    if failure:
        categories = [FailureCategory(failure.upper())]

    logger.info("Running simulation...")
    engine = SimulationEngine(seed=seed, max_ticks=max_ticks)
    result = engine.run(graph, failure_categories=categories, difficulty=difficulty)

    click.echo(f"\n{'=' * 60}")
    click.echo(f"Scenario: {result.scenario_id}")
    click.echo(f"Template: {template}")
    click.echo(f"Nodes: {result.topology.node_count}, Edges: {result.topology.edge_count}")
    click.echo(f"Injected failures: {len(result.failure_injections)}")
    click.echo(f"Cascade steps: {len(result.cascade_trace)}")
    click.echo(f"Decision points: {len(result.decision_points)}")
    click.echo(f"{'=' * 60}")

    if print_trace:
        click.echo("\nCascade Trace:")
        for change in result.cascade_trace:
            cause_str = f" <- {change.cause_component}" if change.cause_component else ""
            click.echo(
                f"  [t={change.tick}] {change.component_id}: "
                f"{change.previous_state} -> {change.new_state} "
                f"(health: {change.previous_health:.2f} -> {change.new_health:.2f}) "
                f"cause: {change.cause}{cause_str}"
            )

    Path(output).mkdir(parents=True, exist_ok=True)
    exporter = TrainingDataExporter()
    out_path = exporter.export(result, Path(output) / "scenario.json")
    click.echo(f"\nExported to: {out_path}")


if __name__ == "__main__":
    main()
