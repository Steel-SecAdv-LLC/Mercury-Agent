# Copyright (C) 2025 Steel Security Advisors LLC
"""(at your option) any later version."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import numpy.typing as npt

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.oae_eigen_monitor/v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.oae_eigen_monitor",
        description="Monitor the OAE fusion-matrix eigenvalues at runtime.",
    )
    parser.add_argument(
        "--A",
        default=None,
        help=".npy of the realised A matrix.  Defaults to the canonical -λI linearisation.",
    )
    parser.add_argument(
        "--P",
        default=None,
        help=".npy of the realised P matrix.  Defaults to I_n.",
    )
    parser.add_argument(
        "--margin-floor",
        type=float,
        default=0.05,
        help="Minimum eigenvalue of -Q permitted; below this the monitor fails.",
    )
    return parser


def _canonical_A() -> npt.NDArray[np.float64]:
    from omni_mercury_engine.core.centralized_constants import LYAPUNOV

    lam = float(LYAPUNOV.LAMBDA_CONVERGENCE)
    return -lam * np.eye(3, dtype=np.float64)


def _collect(args: argparse.Namespace) -> Certificate:
    A = np.load(args.A, allow_pickle=False) if args.A else _canonical_A()
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return Certificate(
            tool="oae_eigen_monitor",
            schema=_SCHEMA,
            status="fail",
            body={"error": f"A must be square; got shape {A.shape}"},
        )
    n = A.shape[0]
    P = np.load(args.P, allow_pickle=False) if args.P else np.eye(n, dtype=np.float64)
    if P.shape != A.shape:
        return Certificate(
            tool="oae_eigen_monitor",
            schema=_SCHEMA,
            status="fail",
            body={"error": f"P shape {P.shape} != A shape {A.shape}"},
        )
    Q = A.T @ P + P @ A
    neg_Q = -Q
    eigs = np.linalg.eigvalsh((neg_Q + neg_Q.T) / 2.0)
    min_eig = float(np.min(eigs))
    body: dict[str, Any] = {
        "n": int(n),
        "eigenvalues_of_minus_Q": [float(e) for e in eigs.tolist()],
        "min_eigenvalue_of_minus_Q": min_eig,
        "margin_floor": float(args.margin_floor),
    }
    if min_eig < float(args.margin_floor):
        return Certificate(
            tool="oae_eigen_monitor",
            schema=_SCHEMA,
            status="fail",
            body=body,
            warnings=[f"Lyapunov margin {min_eig:.6f} < floor {args.margin_floor}"],
        )
    return Certificate(
        tool="oae_eigen_monitor",
        schema=_SCHEMA,
        status="ok",
        body=body,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
