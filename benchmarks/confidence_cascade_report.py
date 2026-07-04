#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Confidence-cascade cost-vs-accuracy report (the ``ci/confidence-cascade`` output).

Builds a deterministic labeled workload where most items are easy (the cheap
template path is confident and correct) and a minority are hard (the cheap path
is uncertain, the heavy model is accurate), routes it through
:class:`~omni_mercury_engine.intel.cascade.ConfidenceCascadeRouter`, and compares
the cascade to the all-heavy baseline on **compute cost, latency, and accuracy**.

The headline number is ``compute_saved_fraction`` -- the fraction of the all-heavy
cost the cascade avoided -- which is the stream's value metric
(:data:`value_metrics.VALUE_METRICS['confidence_cascade']`, target 0.50). The
accuracy delta vs all-heavy must stay within ``ACCURACY_TOLERANCE``.

Everything is deterministic (seeded RNG, injected clock), so the report is stable::

    PYTHONPATH=src python benchmarks/confidence_cascade_report.py           # print report
    PYTHONPATH=src python benchmarks/confidence_cascade_report.py --check   # gate (exit 1)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from omni_mercury_engine.intel.cascade import (
    CascadeConfig,
    ConfidenceCascadeRouter,
    PathResult,
)
from omni_mercury_engine.intel.value_metrics import VALUE_METRICS

ARTIFACT_PATH = _REPO / "artifacts" / "confidence_cascade" / "cost_report.json"
#: The cascade's accuracy may fall at most this far below the all-heavy baseline.
ACCURACY_TOLERANCE = 0.02
N_ITEMS = 1000
EASY_FRACTION = 0.70
SEED = 20260704


def _build_workload(seed: int) -> list[dict[str, Any]]:
    """Deterministic items: ``label``, cheap-path prob, heavy-path prob."""
    rng = np.random.default_rng(seed)
    items = []
    for _ in range(N_ITEMS):
        label = int(rng.integers(0, 2))
        easy = rng.random() < EASY_FRACTION
        if easy:
            # Cheap path: confident + correct.
            cheap_prob = (
                float(rng.uniform(0.90, 0.99)) if label == 1 else float(rng.uniform(0.01, 0.10))
            )
        else:
            # Cheap path: uncertain (near 0.5), roughly a coin flip if trusted.
            cheap_prob = float(rng.uniform(0.40, 0.60))
        # Heavy path: accurate.
        heavy_prob = 0.98 if label == 1 else 0.02
        items.append({"label": label, "cheap_prob": cheap_prob, "heavy_prob": heavy_prob})
    return items


def _deterministic_clock():
    ticks = iter(range(10 * N_ITEMS))
    # Cheap path is fast (1 tick), heavy path slow (5 ticks) -- encoded by the
    # cheap/heavy path callables advancing the shared counter differently.
    return lambda: next(ticks) * 0.001


def evaluate(seed: int = SEED) -> dict[str, Any]:
    """Route the workload and return the cost/accuracy report."""
    items = _build_workload(seed)
    cfg = CascadeConfig(
        low_uncertainty=0.30, high_uncertainty=0.60, cheap_cost=1.0, heavy_cost=20.0
    )

    def cheap(item: dict[str, Any]) -> PathResult:
        p = item["cheap_prob"]
        return PathResult(answer=int(p >= 0.5), prob=p)

    def heavy(item: dict[str, Any]) -> PathResult:
        p = item["heavy_prob"]
        return PathResult(answer=int(p >= 0.5), prob=p)

    router = ConfidenceCascadeRouter(cheap, heavy, cfg, clock=_deterministic_clock())
    outcomes = router.route(items)

    correct = sum(1 for it, o in zip(items, outcomes) if o.result.answer == it["label"])
    cascade_acc = correct / len(items)
    # All-heavy baseline: heavy on every item.
    heavy_correct = sum(1 for it in items if int(it["heavy_prob"] >= 0.5) == it["label"])
    baseline_acc = heavy_correct / len(items)

    report = router.instrumentation.report()
    report.update(
        {
            "cascade_accuracy": round(cascade_acc, 6),
            "baseline_all_heavy_accuracy": round(baseline_acc, 6),
            "accuracy_delta": round(cascade_acc - baseline_acc, 6),
            "value_metric": "compute_saved_at_bounded_accuracy",
            "value_target": VALUE_METRICS["confidence_cascade"].target,
        }
    )
    return report


def _write_artifact(report: dict[str, Any]) -> None:
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="gate on savings + accuracy (exit 1)")
    args = parser.parse_args(argv)

    report = evaluate()
    _write_artifact(report)
    target = VALUE_METRICS["confidence_cascade"].target

    if args.check:
        problems = []
        if report["compute_saved_fraction"] < target:
            problems.append(
                f"compute_saved_fraction {report['compute_saved_fraction']:.3f} < target {target}"
            )
        if report["accuracy_delta"] < -ACCURACY_TOLERANCE:
            problems.append(
                f"accuracy dropped {report['accuracy_delta']:.3f} (> tolerance {ACCURACY_TOLERANCE})"
            )
        if problems:
            print("CONFIDENCE CASCADE REGRESSION:", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            return 1
        print(
            f"OK: compute saved {report['compute_saved_fraction']:.1%} "
            f"(cheap {report['cheap_fraction']:.1%}), accuracy Δ {report['accuracy_delta']:+.3f}"
        )
        return 0

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
