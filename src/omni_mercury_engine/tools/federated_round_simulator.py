# Copyright (C) 2025 Steel Security Advisors LLC
"""(at your option) any later version."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import numpy.typing as npt

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
    parser.add_argument(
        "--adversarial",
        choices=("none", "byzantine", "gradient_inversion"),
        default="none",
        help=(
            "Inject an adversarial peer into the round.  ``byzantine`` flips "
            "one node's gradient sign and scales it 10x; "
            "``gradient_inversion`` attempts to recover the original input "
            "from the aggregated gradient and reports residual recovery "
            "error.  The MLE aggregator + DP noise must still hold."
        ),
    )
    parser.add_argument(
        "--byzantine-scale",
        type=float,
        default=10.0,
        help="Multiplier applied to the flipped-sign Byzantine node's update.",
    )
    return parser


def _collect(args: argparse.Namespace) -> Certificate:
    try:
        # The first-party aggregator lives under
        # ``omni_mercury_engine.federation`` (not ``.federated``); the
        # original import path was a typo that made this tool always
        # raise ``DependencyMissing`` even when the capability is
        # present.  Try the real path first, then keep the legacy
        # path as a fallback for downstream forks that may have
        # renamed the package.
        try:
            from omni_mercury_engine.federation.aggregator import (
                FederatedAggregator,
            )
        except ImportError:
            from omni_mercury_engine.federated.aggregator import (  # type: ignore[no-redef]
                FederatedAggregator,
            )
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
    honest_mean = np.mean(np.stack(node_updates, axis=0), axis=0)
    secret_input: npt.NDArray[np.float64] | None = None
    if args.adversarial == "byzantine":
        # Peer 0 is the adversary: flip the sign and scale.  The MLE
        # aggregator should clip or down-weight the outlier so the
        # aggregate stays close to the honest mean.
        node_updates[0] = -float(args.byzantine_scale) * node_updates[0]
    elif args.adversarial == "gradient_inversion":
        # A naive gradient-inversion attack: the adversary observes the
        # aggregate and tries to recover one peer's input.  We capture
        # peer 0's input as the ``secret`` and report how close the
        # inversion estimate gets to it after aggregation + DP noise.
        secret_input = node_updates[0].copy()

    # ``FederatedAggregator`` is structural across forks — first-party
    # currently has a no-arg / (min_nodes, max_age_seconds) ctor while
    # downstream privacy-extended forks accept ``epsilon=``.  Probe with
    # ``epsilon`` first via a generic ``Any``-typed call (mypy correctly
    # rejects passing ``epsilon`` to the in-tree class otherwise) and
    # fall back to the no-arg ctor; this preserves operator ergonomics
    # without weakening the type gate on the in-tree class.
    _agg_factory: Any = FederatedAggregator
    try:
        agg = _agg_factory(epsilon=args.epsilon)
    except TypeError:
        # In-tree signature does not accept ``epsilon``.
        agg = FederatedAggregator()

    # Try a sequence of common aggregator method names.  Each method, if
    # present, is wrapped to capture the aggregated result.
    aggregated: npt.NDArray[np.float64] | None = None
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
    adversarial_summary: dict[str, Any] = {"mode": args.adversarial}
    if args.adversarial == "byzantine":
        # The honest reference (without the adversary) is held aside.
        # The aggregator's job is to stay near it despite the flipped peer.
        deviation = float(np.linalg.norm(aggregated - honest_mean))
        baseline = float(np.linalg.norm(honest_mean) + 1e-12)
        adversarial_summary["deviation_from_honest"] = deviation
        adversarial_summary["deviation_ratio"] = deviation / baseline
        # If the aggregator forwards the adversary's contribution un-clipped,
        # the deviation explodes (≥ byzantine_scale).  Permit up to 0.5 of
        # the honest baseline.
        adversarial_summary["survived"] = adversarial_summary["deviation_ratio"] <= 0.5
    elif args.adversarial == "gradient_inversion" and secret_input is not None:
        # Naive inversion: subtract every other peer's update from the
        # aggregate and check whether the residual matches the secret.
        others = np.stack(node_updates[1:], axis=0).sum(axis=0)
        recovered = float(args.nodes) * aggregated - others
        residual = float(np.linalg.norm(recovered - secret_input))
        baseline = float(np.linalg.norm(secret_input) + 1e-12)
        adversarial_summary["residual_l2"] = residual
        adversarial_summary["residual_ratio"] = residual / baseline
        # DP noise should make recovery hard: residual_ratio >= 0.25 is
        # the operator-visible privacy guarantee.
        adversarial_summary["privacy_held"] = adversarial_summary["residual_ratio"] >= 0.25
    body: dict[str, Any] = {
        "nodes": args.nodes,
        "dim": args.dim,
        "seed": args.seed,
        "epsilon": args.epsilon,
        "method_used": method_used,
        "adversarial": adversarial_summary,
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
    if args.adversarial == "byzantine" and not adversarial_summary.get("survived", True):
        warnings.append(
            f"Byzantine peer survived aggregation: deviation_ratio={adversarial_summary['deviation_ratio']:.3f}"
        )
    if args.adversarial == "gradient_inversion" and not adversarial_summary.get(
        "privacy_held", True
    ):
        warnings.append(
            f"Gradient inversion recovered too much: residual_ratio={adversarial_summary['residual_ratio']:.3f}"
        )
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
