#!/usr/bin/env python3
"""SABLE Evaluation Harness — Generate scenarios, query Nemotron, score responses."""

import json
import re
import time
import logging
from pathlib import Path
from datetime import datetime

import click
import requests

from sable_sim.simulation.engine import SimulationEngine
from sable_sim.generation.templates import TemplateGenerator
from sable_sim.utils.random import SeededRandom
from sable_sim.utils.logging import setup_logging


# --- Prompt Construction ---

SYSTEM_PROMPT = """You are SABLE, a cognitive engine for infrastructure operations and failure analysis.

You will receive a topology description and observable symptoms from a failure event. Your task is to:
1. Identify the most likely root cause
2. Trace the propagation path
3. Recommend immediate remediation steps

Be concise. No more than 300 words. Focus on accuracy over verbosity.
Do not hallucinate components that are not in the topology.
If information is insufficient, say so."""


def build_topology_summary(result) -> str:
    """Build a concise text summary of the topology for the prompt."""
    topo = result.topology
    lines = []
    lines.append(f"TOPOLOGY: {topo.node_count} nodes, {topo.edge_count} edges")
    lines.append("")

    nodes_by_type = {}
    for comp in topo.get_all_components():
        ntype = comp.type.value if hasattr(comp.type, 'value') else str(comp.type)
        nodes_by_type.setdefault(ntype, []).append(comp.id)

    lines.append("COMPONENTS:")
    for ntype, ids in sorted(nodes_by_type.items()):
        lines.append(f"  {ntype}: {', '.join(ids)}")

    lines.append("")
    lines.append("DEPENDENCIES:")
    for dep in topo.get_all_dependencies():
        dep_type = dep.type.value if hasattr(dep.type, 'value') else str(dep.type)
        crit = dep.criticality.value if hasattr(dep.criticality, 'value') else str(dep.criticality)
        lines.append(f"  {dep.source_id} -> {dep.target_id} ({dep_type}, {crit})")

    return "\n".join(lines)


def build_observable_symptoms(result) -> str:
    """Build the operator-visible symptoms (fog of war applied)."""
    lines = []
    lines.append("OBSERVED SYMPTOMS:")

    if result.operator_view and result.operator_view.get("observations"):
        for obs in result.operator_view["observations"]:
            lines.append(f"  - {obs}")
    else:
        lines.append("  Initial failure detected. Cascade in progress.")
        for change in result.cascade_trace[:5]:
            cause_str = f" <- {change.cause_component}" if change.cause_component else ""
            lines.append(
                f"  - [t={change.tick}] {change.component_id}: "
                f"health {change.previous_health:.2f} -> {change.new_health:.2f} "
                f"cause: {change.cause}{cause_str}"
            )
        if len(result.cascade_trace) > 5:
            lines.append(f"  ... and {len(result.cascade_trace) - 5} more state changes")

    return "\n".join(lines)


def build_prompt(result) -> str:
    """Build the full user prompt from a simulation result."""
    topology = build_topology_summary(result)
    symptoms = build_observable_symptoms(result)

    return f"""{topology}

{symptoms}

Analyze this failure. Identify the root cause, trace the propagation path, and recommend remediation."""


# --- LLM Query ---

def query_nemotron(system_prompt: str, user_prompt: str, endpoint: str, timeout: int = 120) -> dict:
    """Send a prompt to the llama-server and return the response with timing."""
    start = time.time()

    try:
        resp = requests.post(
            f"{endpoint}/v1/chat/completions",
            json={
                "model": "nemotron",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.6,
                "top_p": 0.95,
                "max_tokens": 2048,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        elapsed = time.time() - start

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        return {
            "content": content,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "elapsed_seconds": round(elapsed, 2),
            "tokens_per_second": round(
                usage.get("completion_tokens", 0) / elapsed, 1
            ) if elapsed > 0 else 0,
            "error": None,
        }
    except Exception as e:
        return {
            "content": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "elapsed_seconds": round(time.time() - start, 2),
            "tokens_per_second": 0,
            "error": str(e),
        }


# --- Scoring ---

def score_response(response_text: str, result) -> dict:
    """Score a response against ground truth."""
    scores = {}
    text_lower = response_text.lower()

    # 1. Root cause — did it name the injected component?
    injected_ids = [inj.component_id for inj in result.failure_injections]
    root_cause_hits = sum(1 for cid in injected_ids if cid.lower() in text_lower)
    scores["root_cause"] = min(1.0, root_cause_hits / max(len(injected_ids), 1))

    # 2. Propagation coverage — did it mention affected components?
    affected_ids = set()
    for change in result.cascade_trace:
        affected_ids.add(change.component_id)
    mentioned = sum(1 for cid in affected_ids if cid.lower() in text_lower)
    scores["propagation_coverage"] = round(mentioned / max(len(affected_ids), 1), 2)

    # 3. Hallucination — component IDs mentioned that don't exist
    mentioned_ids = set(re.findall(r'[a-z][\w-]*-\d+', text_lower))
    valid_ids = set(c.id.lower() for c in result.topology.get_all_components())
    hallucinated = mentioned_ids - valid_ids
    scores["hallucination_rate"] = round(len(hallucinated) / max(len(mentioned_ids), 1), 2)
    scores["hallucinated_ids"] = list(hallucinated)

    # 4. Conciseness — word count relative to 300 word target
    word_count = len(response_text.split())
    if word_count <= 300:
        scores["conciseness"] = 1.0
    elif word_count <= 600:
        scores["conciseness"] = round(1.0 - (word_count - 300) / 300, 2)
    else:
        scores["conciseness"] = 0.0
    scores["word_count"] = word_count

    # 5. Remediation — did it include actionable steps?
    remediation_keywords = ["remediat", "recommend", "action", "fix", "restart",
                           "failover", "isolate", "migrate", "restore", "redeploy"]
    remediation_hits = sum(1 for kw in remediation_keywords if kw in text_lower)
    scores["has_remediation"] = min(1.0, remediation_hits / 3)

    # 6. Composite
    scores["composite"] = round(
        (scores["root_cause"] * 0.3 +
         scores["propagation_coverage"] * 0.2 +
         (1 - scores["hallucination_rate"]) * 0.2 +
         scores["conciseness"] * 0.15 +
         scores["has_remediation"] * 0.15),
        3
    )

    return scores


# --- Main ---

@click.command()
@click.option("--count", "-n", default=10, type=int, help="Number of scenarios to evaluate")
@click.option("--templates", default="small_office,enterprise_campus",
              help="Comma-separated topology templates")
@click.option("--difficulty", "-d", default="medium", help="easy / medium / hard")
@click.option("--endpoint", default="http://127.0.0.1:8080", help="llama-server endpoint")
@click.option("--output", "-o", default="./data/eval/", help="Output directory for results")
@click.option("--seed", "-s", default=42, type=int, help="Random seed")
@click.option("--verbose", "-v", is_flag=True, help="Print each response")
def main(count, templates, difficulty, endpoint, output, seed, verbose):
    """Run SABLE evaluation harness."""
    setup_logging("DEBUG" if verbose else "INFO")
    logger = logging.getLogger(__name__)

    template_list = [t.strip() for t in templates.split(",")]
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check server health
    try:
        health = requests.get(f"{endpoint}/health", timeout=5)
        if health.status_code != 200:
            click.echo(f"ERROR: llama-server not healthy at {endpoint}")
            return
    except requests.ConnectionError:
        click.echo(f"ERROR: Cannot connect to llama-server at {endpoint}")
        click.echo("Start it with: cd ~/llama.cpp && ./build/bin/llama-server -m /mnt/vault/models/nemotron-nano/*.gguf --ctx-size 75000 -ngl 999 --port 8080 --jinja")
        return

    click.echo(f"SABLE Evaluation Harness")
    click.echo(f"Scenarios: {count} | Templates: {template_list} | Difficulty: {difficulty}")
    click.echo(f"Endpoint: {endpoint}")
    click.echo(f"{'=' * 60}")

    results = []

    for i in range(count):
        template = template_list[i % len(template_list)]
        scenario_seed = seed + i

        # Generate scenario
        tmpl_gen = TemplateGenerator(SeededRandom(scenario_seed))
        graph = tmpl_gen.generate(template)
        engine = SimulationEngine(seed=scenario_seed, max_ticks=50)
        sim_result = engine.run(graph, difficulty=difficulty)

        # Build prompt
        user_prompt = build_prompt(sim_result)
        prompt_tokens_est = len(user_prompt.split()) * 1.3

        click.echo(f"\n[{i+1}/{count}] {template} (seed={scenario_seed}) "
                    f"| nodes={sim_result.topology.node_count} "
                    f"| cascade_steps={len(sim_result.cascade_trace)} "
                    f"| ~{int(prompt_tokens_est)} prompt tokens")

        # Query Nemotron
        llm_response = query_nemotron(SYSTEM_PROMPT, user_prompt, endpoint)

        if llm_response["error"]:
            click.echo(f"  ERROR: {llm_response['error']}")
            results.append({
                "scenario_index": i,
                "template": template,
                "seed": scenario_seed,
                "error": llm_response["error"],
            })
            continue

        # Score
        scores = score_response(llm_response["content"], sim_result)

        click.echo(f"  Composite: {scores['composite']:.3f} | "
                    f"Root cause: {scores['root_cause']:.1f} | "
                    f"Coverage: {scores['propagation_coverage']:.2f} | "
                    f"Hallucination: {scores['hallucination_rate']:.2f} | "
                    f"Concise: {scores['conciseness']:.1f} | "
                    f"Remediation: {scores['has_remediation']:.1f} | "
                    f"{llm_response['tokens_per_second']} t/s | "
                    f"{scores['word_count']} words")

        if verbose:
            click.echo(f"\n  --- Response ---\n{llm_response['content']}\n  --- End ---\n")

        results.append({
            "scenario_index": i,
            "template": template,
            "seed": scenario_seed,
            "node_count": sim_result.topology.node_count,
            "cascade_steps": len(sim_result.cascade_trace),
            "injected_failures": [inj.component_id for inj in sim_result.failure_injections],
            "scores": scores,
            "llm": {
                "prompt_tokens": llm_response["prompt_tokens"],
                "completion_tokens": llm_response["completion_tokens"],
                "elapsed_seconds": llm_response["elapsed_seconds"],
                "tokens_per_second": llm_response["tokens_per_second"],
            },
            "response": llm_response["content"],
        })

    # Summary
    scored = [r for r in results if "scores" in r]
    if scored:
        avg = lambda key: sum(r["scores"][key] for r in scored) / len(scored)
        avg_tps = sum(r["llm"]["tokens_per_second"] for r in scored) / len(scored)
        avg_words = sum(r["scores"]["word_count"] for r in scored) / len(scored)

        click.echo(f"\n{'=' * 60}")
        click.echo(f"EVALUATION SUMMARY — {len(scored)} scenarios scored")
        click.echo(f"{'=' * 60}")
        click.echo(f"  Composite Score:      {avg('composite'):.3f}")
        click.echo(f"  Root Cause Accuracy:  {avg('root_cause'):.3f}")
        click.echo(f"  Propagation Coverage: {avg('propagation_coverage'):.3f}")
        click.echo(f"  Hallucination Rate:   {avg('hallucination_rate'):.3f}")
        click.echo(f"  Conciseness:          {avg('conciseness'):.3f}")
        click.echo(f"  Remediation Quality:  {avg('has_remediation'):.3f}")
        click.echo(f"  Avg Tokens/sec:       {avg_tps:.1f}")
        click.echo(f"  Avg Word Count:       {avg_words:.0f}")
        click.echo(f"{'=' * 60}")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = output_dir / f"eval_{timestamp}.json"
        with open(results_path, "w") as f:
            json.dump({
                "config": {
                    "count": count,
                    "templates": template_list,
                    "difficulty": difficulty,
                    "seed": seed,
                    "endpoint": endpoint,
                    "system_prompt": SYSTEM_PROMPT,
                },
                "summary": {
                    "total": len(results),
                    "scored": len(scored),
                    "errors": len(results) - len(scored),
                    "avg_composite": round(avg('composite'), 3),
                },
                "results": results,
            }, f, indent=2, default=str)

        click.echo(f"\nFull results saved to: {results_path}")
    else:
        click.echo("\nNo scenarios were scored successfully.")


if __name__ == "__main__":
    main()
