"""Export simulation results as ML training data in JSON format."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from sable_sim.simulation.engine import SimulationResult

logger = logging.getLogger(__name__)


class TrainingDataExporter:
    """Exports simulation results as JSON training examples."""

    def export(self, result: SimulationResult, output_path: str | Path) -> str:
        """Export a single simulation result to JSON.

        Returns the output file path as a string.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = self._serialize_result(result)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        logger.debug("Exported scenario %s to %s", result.scenario_id, output_path)
        return str(output_path)

    def export_batch(
        self, results: list[SimulationResult], output_dir: str | Path
    ) -> list[str]:
        """Export multiple results. Returns list of output file paths."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for i, result in enumerate(results):
            path = output_dir / f"scenario_{i:06d}.json"
            paths.append(self.export(result, path))
        return paths

    def _serialize_result(self, result: SimulationResult) -> dict[str, Any]:
        """Convert SimulationResult to the training data JSON format."""
        return {
            "scenario_id": result.scenario_id,
            "metadata": {
                "node_count": result.metadata.get("node_count"),
                "edge_count": result.metadata.get("edge_count"),
                "failure_categories": result.metadata.get("failure_categories", []),
                "difficulty": result.metadata.get("difficulty", "medium"),
                "total_ticks": result.metadata.get("total_ticks"),
                "total_state_changes": result.metadata.get("total_state_changes"),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
            "topology": result.topology.to_dict(),
            "ground_truth": {
                "failure_injections": [
                    inj.to_dict() for inj in result.failure_injections
                ],
                "cascade_trace": [
                    sc.to_dict() for sc in result.cascade_trace
                ],
                "final_state": result.final_state,
            },
            "operator_view": result.operator_view,
            "decision_points": result.decision_points,
        }
