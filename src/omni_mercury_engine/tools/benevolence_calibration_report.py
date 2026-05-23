"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

------------------------------------------------------------------------

Operator tool: benevolence calibration report.

The existing :mod:`benevolence_certifier` answers "is the floor met"
(binary ≥ 0.99).  This tool computes a 10-bin reliability diagram and
Expected Calibration Error (ECE) over the operator-supplied probe set
so the auditor sees *how* well-calibrated the benevolence score is,
not just whether it clears the line.

Inputs:

* ``--scores``: ``.npy`` of shape (N,) with benevolence scores in [0, 1];
* ``--labels``: ``.npy`` of shape (N,) with ground-truth ethical
  outcomes (1 = ethical, 0 = not).

Outputs the per-bin counts/accuracy and the ECE.  Fails when ECE
exceeds ``--ece-max`` (default 0.05 — operator-supplied threshold).
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.benevolence_calibration_report/v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.benevolence_calibration_report",
        description="Compute the reliability diagram + ECE for benevolence scores.",
    )
    parser.add_argument("--scores", required=True, help=".npy of (N,) benevolence scores in [0,1].")
    parser.add_argument(
        "--labels", required=True, help=".npy of (N,) ground-truth ethical labels (0/1)."
    )
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--ece-max", type=float, default=0.05)
    return parser


def _collect(args: argparse.Namespace) -> Certificate:
    scores = np.load(args.scores, allow_pickle=False).astype(np.float64).ravel()
    labels = np.load(args.labels, allow_pickle=False).astype(np.float64).ravel()
    if scores.shape != labels.shape:
        return Certificate(
            tool="benevolence_calibration_report",
            schema=_SCHEMA,
            status="fail",
            body={"error": f"shape mismatch: scores {scores.shape} != labels {labels.shape}"},
        )
    if scores.size == 0:
        return Certificate(
            tool="benevolence_calibration_report",
            schema=_SCHEMA,
            status="fail",
            body={"error": "empty probe set"},
        )
    scores = np.clip(scores, 0.0, 1.0)

    bins = np.linspace(0.0, 1.0, int(args.bins) + 1)
    diagram: list[dict[str, float]] = []
    ece = 0.0
    n_total = float(scores.size)
    for i in range(int(args.bins)):
        lo, hi = float(bins[i]), float(bins[i + 1])
        # Last bin is closed at the top.
        if i == int(args.bins) - 1:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)
        n = int(mask.sum())
        if n == 0:
            diagram.append({"lo": lo, "hi": hi, "n": 0.0, "mean_score": 0.0, "accuracy": 0.0})
            continue
        mean_score = float(scores[mask].mean())
        accuracy = float(labels[mask].mean())
        diagram.append(
            {
                "lo": lo,
                "hi": hi,
                "n": float(n),
                "mean_score": mean_score,
                "accuracy": accuracy,
            }
        )
        ece += (n / n_total) * abs(mean_score - accuracy)

    body: dict[str, Any] = {
        "n": int(scores.size),
        "bins": int(args.bins),
        "ece": ece,
        "ece_max": float(args.ece_max),
        "diagram": diagram,
    }
    if ece > float(args.ece_max):
        return Certificate(
            tool="benevolence_calibration_report",
            schema=_SCHEMA,
            status="fail",
            body=body,
            warnings=[f"ECE {ece:.4f} > limit {args.ece_max}"],
        )
    return Certificate(
        tool="benevolence_calibration_report",
        schema=_SCHEMA,
        status="ok",
        body=body,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
