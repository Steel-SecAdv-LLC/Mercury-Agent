#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Self-consistency signal report: does disagreement predict error?

Draws a deterministic held-out set of items with a latent per-item reliability
``q`` (how often a sampled reasoning path is correct), samples ``N`` paths per
item, takes the plurality answer, and measures whether the **disagreement** among
paths ranks the *errored* plurality answers above the correct ones. That ranking
quality is :func:`~omni_mercury_engine.intel.self_consistency.disagreement_error_auroc`
-- the stream's value metric
(:data:`value_metrics.VALUE_METRICS['self_consistency']`, target AUROC 0.70).

Nothing is hand-crafted: disagreement and error both fall out of the same sampled
votes, so a high AUROC is real evidence the signal is usable, not a tautology.

    PYTHONPATH=src python benchmarks/self_consistency_signal_report.py           # print
    PYTHONPATH=src python benchmarks/self_consistency_signal_report.py --check   # gate (exit 1)
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

from omni_mercury_engine.intel.self_consistency import (
    disagreement_error_auroc,
    self_consistency,
    self_consistency_decision,
)
from omni_mercury_engine.intel.value_metrics import VALUE_METRICS

ARTIFACT_PATH = _REPO / "artifacts" / "self_consistency" / "signal_report.json"
N_ITEMS = 600
N_SAMPLES = 7
SEED = 20260704


def evaluate(seed: int = SEED) -> dict[str, Any]:
    """Sample paths per item; measure disagreement-vs-error AUROC + abstention lift."""
    rng = np.random.default_rng(seed)
    disagreements: list[float] = []
    errors: list[int] = []
    committed_errors: list[int] = []
    abstained = 0

    for i in range(N_ITEMS):
        true_label = int(rng.integers(0, 2))
        q = float(rng.uniform(0.55, 0.97))  # per-item path reliability

        def sampler(r: np.random.Generator, _q: float = q, _lbl: int = true_label) -> int:
            return _lbl if r.random() < _q else 1 - _lbl

        result = self_consistency(sampler, n_samples=N_SAMPLES, seed=seed + i)
        error = int(result.answer != true_label)
        disagreements.append(result.disagreement)
        errors.append(error)

        # Decision-rule lift: abstain on high disagreement; measure error only on
        # the items the rule actually committed to.
        # With N=7 the disagreement lands in {0, 1/7, 2/7, 3/7}; abstain on the
        # most-split (>= ~3/7) votes, which carry the most error.
        decision = self_consistency_decision(
            result.support, result.disagreement, abstain_above=0.40
        )
        if decision.abstained:
            abstained += 1
        else:
            committed_errors.append(error)

    auroc = disagreement_error_auroc(disagreements, errors)
    overall_error = float(np.mean(errors))
    committed_error = float(np.mean(committed_errors)) if committed_errors else float("nan")
    return {
        "n_items": N_ITEMS,
        "n_samples": N_SAMPLES,
        "disagreement_error_auroc": round(float(auroc), 6),
        "overall_error_rate": round(overall_error, 6),
        "committed_error_rate": round(committed_error, 6),
        "abstained_fraction": round(abstained / N_ITEMS, 6),
        "value_metric": "disagreement_error_auroc",
        "value_target": VALUE_METRICS["self_consistency"].target,
    }


def _write_artifact(report: dict[str, Any]) -> None:
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="gate on AUROC >= target (exit 1)")
    args = parser.parse_args(argv)

    report = evaluate()
    _write_artifact(report)
    target = VALUE_METRICS["self_consistency"].target

    if args.check:
        auroc = report["disagreement_error_auroc"]
        if auroc < target:
            print(
                f"SELF-CONSISTENCY REGRESSION: disagreement AUROC {auroc:.3f} < target {target}",
                file=sys.stderr,
            )
            return 1
        print(
            f"OK: disagreement AUROC {auroc:.3f} (>= {target}); "
            f"abstention cut error {report['overall_error_rate']:.3f} -> "
            f"{report['committed_error_rate']:.3f}"
        )
        return 0

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
