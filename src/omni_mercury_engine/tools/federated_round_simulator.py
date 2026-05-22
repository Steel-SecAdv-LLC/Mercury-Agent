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

Operator tool: federated-round simulator.

Drives the ``FederatedAggregator`` through a synthetic 3-node round,
verifies MLE-style aggregation and the differential-privacy noise
injection.  The federated-learning capability is structural in the
repo; this gives operators a runtime probe so they can answer "did
the aggregation actually run, and was DP noise injected?" with a JSON
certificate rather than a code review.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from omni_mercury_engine.tools._base import Certificate, DependencyMissing, run_tool

_SCHEMA = "mercury.tools.federated_round_simulator/v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.federated_round_simulator",
        description=(
            "Run a synthetic 3-node federated round and report aggregated "
            "weight vector + DP noise statistics."
        ),
    )
    parser.add_argument("--nodes", type=int, default=3, help="Number of synthetic nodes.")
    parser.add_argument("--dim", type=int, default=64, help="Weight-vector dimension.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed.")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
        help="DP epsilon (default 1.0); lower = more noise.",
    )
    return parser


def _collect(args: argparse.Namespace) -> Certificate:
    try:
        from omni_mercury_engine.federated.aggregator import FederatedAggregator
    except ImportError as exc:
        raise DependencyMissing(
            f"FederatedAggregator import failed (federated extra missing?): {exc}"
        ) from exc

    rng = np.random.default_rng(args.seed)
    # Each node's local update: a Gaussian centred on a node-specific mean
    # so the aggregation has a non-trivial outcome (otherwise the average
    # of identical inputs is uninteresting).
    node_updates = [
        rng.standard_normal(args.dim).astype(np.float64) + 0.5 * (i - args.nodes / 2)
        for i in range(args.nodes)
    ]

    try:
        agg = FederatedAggregator(epsilon=args.epsilon)
    except TypeError:
        # Older signature without epsilon — try the no-arg ctor.
        agg = FederatedAggregator()

    # Try a sequence of common aggregator method names.  Each method, if
    # present, is wrapped to capture the aggregated result.
    aggregated: np.ndarray | None = None
    method_used: str | None = None
    for method in ("aggregate", "aggregate_round", "federated_average", "fed_avg"):
        fn = getattr(agg, method, None)
        if fn is None:
            continue
        try:
            aggregated = np.asarray(fn(node_updates), dtype=np.float64)
            method_used = method
            break
        except TypeError:
            try:
                aggregated = np.asarray(fn(*node_updates), dtype=np.float64)
                method_used = method
                break
            except Exception:
                continue
        except Exception:
            continue

    if aggregated is None:
        # Fallback: compute the unweighted mean here.  This gives the
        # operator a known-good reference even when the aggregator API
        # has drifted, and surfaces the API drift as a warning.
        aggregated = np.mean(np.stack(node_updates, axis=0), axis=0)
        method_used = "fallback:numpy.mean"

    # Pure unweighted mean for the noise-injection check.
    noiseless_mean = np.mean(np.stack(node_updates, axis=0), axis=0)
    noise_delta = aggregated - noiseless_mean
    body: dict[str, Any] = {
        "nodes": args.nodes,
        "dim": args.dim,
        "seed": args.seed,
        "epsilon": args.epsilon,
        "method_used": method_used,
        "aggregated_summary": {
            "mean": float(aggregated.mean()),
            "std": float(aggregated.std()),
            "min": float(aggregated.min()),
            "max": float(aggregated.max()),
        },
        "noise_delta_summary": {
            "mean_abs": float(np.mean(np.abs(noise_delta))),
            "std": float(noise_delta.std()),
            "max_abs": float(np.max(np.abs(noise_delta))),
            "nonzero_fraction": float(np.mean(noise_delta != 0.0)),
        },
    }
    warnings: list[str] = []
    if method_used == "fallback:numpy.mean":
        warnings.append(
            "FederatedAggregator API method not detected — used numpy.mean fallback; "
            "verify the aggregator class still exposes a public aggregate() entry"
        )
    if body["noise_delta_summary"]["mean_abs"] == 0.0:
        warnings.append(
            "no DP noise detected (mean_abs delta == 0.0); aggregator may have "
            "DP disabled or the epsilon is ignored"
        )
    status = "ok" if not warnings else "warn"
    return Certificate(
        tool="federated_round_simulator",
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
