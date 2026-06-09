# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Operator tool: empirical adversarial-perturbation probe.

Lyapunov gives a theoretical Lipschitz bound on score variation under
input perturbation; this tool gives the empirical companion.  For
every example in the input batch we apply N small Gaussian
perturbations and measure the maximum |Δscore| / ||Δx||₂.  The result
is a per-example "empirical Lipschitz" that should stay close to the
theoretical bound; a large outlier signals adversarial brittleness.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import numpy.typing as npt

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.adversarial_probe/v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.adversarial_probe",
        description=(
            "Empirically bound a detector's response to small input "
            "perturbations and compare against the Lyapunov claim."
        ),
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to a .npy file with the (N, D) feature matrix to probe.",
    )
    parser.add_argument(
        "--detector",
        default="mathmercury",
        choices=["mathmercury"],
        help=(
            "Detector to probe (default mathmercury). Mercury Agent "
            "retired IsolationForest as a live anomaly path; the "
            "AnomalyMathArrest ensemble is the sole baseline."
        ),
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-3,
        help="Perturbation std (default 1e-3).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=32,
        help="Perturbation samples per example (default 32).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default 0).")
    parser.add_argument(
        "--bound",
        type=float,
        default=10.0,
        help="Maximum acceptable empirical Lipschitz (default 10.0).",
    )
    return parser


def _build_scorer(name: str, X: npt.NDArray[np.float64]) -> Any:
    if name == "mathmercury":
        # AnomalyMathArrest is the sole live anomaly path — enforced by
        # ``tests/detectors/test_math_arrest_dominant_path.py``.
        from omni_mercury_engine.detectors.math_arrest.arrest import AnomalyMathArrest

        m = AnomalyMathArrest()
        m.fit(X)
        return lambda batch: m.detect(batch)
    raise ValueError(f"unknown detector: {name}")


def _collect(args: argparse.Namespace) -> Certificate:
    X = np.load(args.data, allow_pickle=False)
    if X.ndim != 2:
        raise ValueError(f"--data must be (N, D); got {X.shape}")
    rng = np.random.default_rng(args.seed)
    scorer = _build_scorer(args.detector, X)

    base = np.asarray(scorer(X), dtype=np.float64).ravel()
    n = X.shape[0]
    max_emp_lip = np.zeros(n, dtype=np.float64)

    for s in range(args.samples):
        # Independent perturbation per example.
        delta = rng.standard_normal(X.shape).astype(np.float64) * args.epsilon
        perturbed = X + delta
        sc = np.asarray(scorer(perturbed), dtype=np.float64).ravel()
        delta_norm = np.linalg.norm(delta, axis=1)
        # Avoid div-by-zero: clamp the tiniest norms to epsilon.
        delta_norm = np.maximum(delta_norm, 1e-12)
        emp = np.abs(sc - base) / delta_norm
        max_emp_lip = np.maximum(max_emp_lip, emp)

    body: dict[str, Any] = {
        "detector": args.detector,
        "n": int(n),
        "d": int(X.shape[1]),
        "epsilon": args.epsilon,
        "samples": args.samples,
        "bound": args.bound,
        "empirical_lipschitz_summary": {
            "min": float(max_emp_lip.min()),
            "median": float(np.median(max_emp_lip)),
            "p95": float(np.percentile(max_emp_lip, 95)),
            "max": float(max_emp_lip.max()),
            "mean": float(max_emp_lip.mean()),
        },
        "exceedances": int(np.sum(max_emp_lip > args.bound)),
    }
    warnings: list[str] = []
    if body["exceedances"] > 0:
        warnings.append(
            f"{body['exceedances']} examples exceed empirical Lipschitz bound {args.bound}"
        )
    status = "ok" if not warnings else "warn"
    return Certificate(
        tool="adversarial_probe",
        schema=_SCHEMA,
        status=status,
        body=body,
        warnings=warnings,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
