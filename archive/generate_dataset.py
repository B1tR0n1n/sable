#!/usr/bin/env python3
"""CLI tool to generate training datasets for SABLE."""

import os
import time
from pathlib import Path

import click
from tqdm import tqdm

from sable_sim.simulation.engine import SimulationEngine
from sable_sim.generation.templates import TemplateGenerator
from sable_sim.export.training_data import TrainingDataExporter
from sable_sim.export.graph_format import GraphFormatExporter
from sable_sim.utils.random import SeededRandom
from sable_sim.utils.logging import setup_logging
import logging


def generate_single(
    idx: int,
    seed: int,
    template: str,
    difficulty: str,
    output_dir: str,
    export_pyg: bool,
) -> str:
    """Generate a single training example."""
    rng = SeededRandom(seed)
    template_gen = TemplateGenerator(rng)
    graph = template_gen.generate(template)

    engine = SimulationEngine(seed=seed)
    result = engine.run(graph, difficulty=difficulty)

    exporter = TrainingDataExporter()
    out_path = exporter.export(result, Path(output_dir) / f"scenario_{idx:06d}.json")

    if export_pyg:
        gfx = GraphFormatExporter()
        gfx.export_adjacency(result.topology, Path(output_dir) / f"graph_{idx:06d}.json")

    return out_path


@click.command()
@click.option("--count", "-n", default=100, help="Number of training examples")
@click.option("--output", "-o", default="./data/training/", help="Output directory")
@click.option(
    "--templates", "-t", default="small_office,enterprise_campus",
    help="Comma-separated template names",
)
@click.option(
    "--difficulty-mix", "-d", default="easy:0.2,medium:0.5,hard:0.3",
    help="Difficulty distribution (level:weight,...)",
)
@click.option("--seed", "-s", default=42, type=int, help="Random seed")
@click.option("--export-pyg", is_flag=True, help="Also export GNN graph format")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
def main(
    count: int,
    output: str,
    templates: str,
    difficulty_mix: str,
    seed: int,
    export_pyg: bool,
    verbose: bool,
) -> None:
    """Generate SABLE training dataset."""
    setup_logging("DEBUG" if verbose else "INFO")
    logger = logging.getLogger(__name__)

    template_list = [t.strip() for t in templates.split(",")]

    difficulties: dict[str, float] = {}
    for item in difficulty_mix.split(","):
        level, weight = item.split(":")
        difficulties[level.strip()] = float(weight.strip())

    os.makedirs(output, exist_ok=True)

    rng = SeededRandom(seed)

    logger.info("Generating %d scenarios...", count)
    start_time = time.time()

    for i in tqdm(range(count), desc="Generating"):
        tmpl = rng.choice(template_list)
        diff = rng.weighted_choice(
            list(difficulties.keys()), list(difficulties.values())
        )
        generate_single(i, seed + i, tmpl, diff, output, export_pyg)

    elapsed = time.time() - start_time
    logger.info(
        "Generated %d scenarios in %.1fs (%.1f/sec)", count, elapsed, count / max(elapsed, 0.01)
    )
    logger.info("Output directory: %s", output)


if __name__ == "__main__":
    main()
