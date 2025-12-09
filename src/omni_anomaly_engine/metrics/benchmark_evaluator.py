"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""
from __future__ import annotations

"""
Benchmark evaluation framework.

Provides standardized evaluation across benchmark datasets.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from omni_anomaly_engine.metrics.anomaly_metrics import AnomalyMetrics

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results.

    Attributes:
        detector_name: Name of the detector evaluated
        dataset_name: Dataset used for evaluation
        metrics: Dict of metric values
        per_category: Per-category metrics if applicable
        timestamp: Evaluation timestamp
        config: Detector configuration used
    """

    detector_name: str
    dataset_name: str
    metrics: dict[str, float]
    per_category: dict[str, dict[str, float]] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "detector_name": self.detector_name,
            "dataset_name": self.dataset_name,
            "metrics": self.metrics,
            "per_category": self.per_category,
            "timestamp": self.timestamp,
            "config": self.config,
        }

    def save(self, path: str | Path) -> None:
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "EvaluationResult":
        """Load results from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def __str__(self) -> str:
        """String representation."""
        lines = [
            f"Evaluation: {self.detector_name} on {self.dataset_name}",
            f"Timestamp: {self.timestamp}",
            "Metrics:",
        ]
        for name, value in sorted(self.metrics.items()):
            lines.append(f"  {name}: {value:.4f}")
        return "\n".join(lines)


class BenchmarkEvaluator:
    """Benchmark evaluation framework.

    Provides standardized evaluation workflow for anomaly detectors
    on benchmark datasets.

    Example:
        >>> evaluator = BenchmarkEvaluator()
        >>> result = evaluator.evaluate(detector, dataset)
        >>> print(result)
        >>> evaluator.compare([result1, result2])
    """

    def __init__(
        self,
        output_dir: str | Path = "./evaluation_results",
        save_predictions: bool = False,
    ):
        """Initialize evaluator.

        Args:
            output_dir: Directory for saving results
            save_predictions: Whether to save raw predictions
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_predictions = save_predictions

    def evaluate(
        self,
        detector: Any,
        dataset: Any,
        detector_name: str | None = None,
        dataset_name: str | None = None,
    ) -> EvaluationResult:
        """Evaluate detector on dataset.

        Args:
            detector: Anomaly detector with detect() method
            dataset: Dataset with samples
            detector_name: Optional detector name
            dataset_name: Optional dataset name

        Returns:
            Evaluation result
        """
        detector_name = detector_name or detector.__class__.__name__
        dataset_name = dataset_name or dataset.__class__.__name__

        logger.info(f"Evaluating {detector_name} on {dataset_name}")

        # Collect predictions
        all_scores = []
        all_labels = []
        all_masks_pred = []
        all_masks_true = []
        all_categories = []

        for i, sample in enumerate(dataset):
            # Get image/video
            if "image" in sample:
                data = sample["image"].unsqueeze(0)
            elif "video" in sample:
                data = sample["video"]
            else:
                continue

            # Detect
            try:
                result = detector.detect(data)
                score = result.get("scores", result.get("score", 0.0))
                if isinstance(score, np.ndarray[Any, Any]):
                    score = score.mean()
                elif isinstance(score, torch.Tensor):
                    score = score.mean().item()

                all_scores.append(float(score))
                all_labels.append(sample["label"])
                all_categories.append(sample.get("category", "default"))

                # Collect masks if available
                if sample.get("mask") is not None and "anomaly_maps" in result:
                    all_masks_true.append(sample["mask"].numpy())
                    all_masks_pred.append(result["anomaly_maps"])

            except Exception as e:
                logger.warning(f"Detection failed for sample {i}: {e}")
                continue

            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(dataset)} samples")

        # Convert to arrays
        scores = np.array(all_scores)
        labels = np.array(all_labels)

        # Compute metrics
        metrics = AnomalyMetrics.compute_all(
            labels,
            scores,
            masks_true=np.stack(all_masks_true) if all_masks_true else None,
            masks_score=np.stack(all_masks_pred) if all_masks_pred else None,
        )

        # Per-category metrics
        per_category = {}
        if len(set(all_categories)) > 1:
            per_category = AnomalyMetrics.compute_per_category(labels, scores, all_categories)

        # Create result
        result = EvaluationResult(
            detector_name=detector_name,
            dataset_name=dataset_name,
            metrics=metrics,
            per_category=per_category,
            config=getattr(detector, "config", {}),
        )

        # Save
        result_path = (
            self.output_dir
            / f"{detector_name}_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        result.save(result_path)
        logger.info(f"Results saved to {result_path}")

        # Save predictions if requested
        if self.save_predictions:
            pred_path = result_path.with_suffix(".npz")
            np.savez(
                pred_path,
                scores=scores,
                labels=labels,
                categories=all_categories,
            )

        return result

    def compare(
        self,
        results: list[EvaluationResult],
        metric: str = "auroc",
    ) -> str:
        """Compare multiple evaluation results.

        Args:
            results: List of evaluation results
            metric: Primary metric for comparison

        Returns:
            Comparison table as string
        """
        if not results:
            return "No results to compare"

        # Build comparison table
        lines = ["Comparison Results", "=" * 60]

        # Header
        datasets = sorted({r.dataset_name for r in results})
        header = f"{'Detector':<20} " + " ".join(f"{d[:10]:>12}" for d in datasets)
        lines.append(header)
        lines.append("-" * len(header))

        # Group by detector
        detectors = sorted({r.detector_name for r in results})
        for detector in detectors:
            row = f"{detector[:20]:<20}"
            for dataset in datasets:
                # Find matching result
                matching = [
                    r for r in results if r.detector_name == detector and r.dataset_name == dataset
                ]
                if matching:
                    value = matching[0].metrics.get(metric, 0.0)
                    row += f" {value:>12.4f}"
                else:
                    row += f" {'N/A':>12}"
            lines.append(row)

        lines.append("=" * 60)

        # Summary stats
        all_values = [r.metrics.get(metric, 0.0) for r in results]
        lines.append(f"Best {metric}: {max(all_values):.4f}")
        lines.append(f"Mean {metric}: {np.mean(all_values):.4f}")

        return "\n".join(lines)

    def generate_report(
        self,
        results: list[EvaluationResult],
        output_path: str | Path,
    ) -> None:
        """Generate comprehensive evaluation report.

        Args:
            results: List of evaluation results
            output_path: Output file path (markdown)
        """
        output_path = Path(output_path)

        lines = [
            "# Anomaly Detection Evaluation Report",
            f"\nGenerated: {datetime.now().isoformat()}",
            "\n## Summary",
            f"\n- Detectors evaluated: {len({r.detector_name for r in results})}",
            f"- Datasets used: {len({r.dataset_name for r in results})}",
            f"- Total evaluations: {len(results)}",
        ]

        # Overall comparison
        lines.append("\n## Overall Comparison")
        lines.append("\n```")
        lines.append(self.compare(results))
        lines.append("```")

        # Per-dataset details
        lines.append("\n## Detailed Results")

        for dataset in sorted({r.dataset_name for r in results}):
            lines.append(f"\n### {dataset}")

            dataset_results = [r for r in results if r.dataset_name == dataset]

            for result in dataset_results:
                lines.append(f"\n#### {result.detector_name}")
                lines.append("\n| Metric | Value |")
                lines.append("|--------|-------|")

                for metric, value in sorted(result.metrics.items()):
                    lines.append(f"| {metric} | {value:.4f} |")

        # Save report
        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        logger.info(f"Report saved to {output_path}")
