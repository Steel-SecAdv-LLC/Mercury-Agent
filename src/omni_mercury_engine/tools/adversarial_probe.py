"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

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

------------------------------------------------------------------------

Operator tool: empirical adversarial-perturbation probe.

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
        default="isoforest",
        choices=["isoforest"],
        help="Detector to probe (default isoforest).",
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
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed (default 0)."
    )
    parser.add_argument(
        "--bound",
        type=float,
        default=10.0,
        help="Maximum acceptable empirical Lipschitz (default 10.0).",
    )
    return parser


def _build_scorer(name: str, X: np.ndarray) -> Any:
    if name == "isoforest":
        from sklearn.ensemble import IsolationForest

        m = IsolationForest(random_state=0, contamination="auto").fit(X)
        return lambda batch: -m.score_samples(batch)
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
