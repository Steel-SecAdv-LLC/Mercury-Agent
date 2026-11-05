# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Operator tool: verify the shipped OAE fuses its three streams as published.

:mod:`oae_weight_certifier` proves the declared ``(w_R, w_H, w_O)`` sum to 1.0
and match the golden-ratio derivation. Declared weights are not served
behaviour, so this tool measures what
:class:`~omni_mercury_engine.core.three_r.fusion.OmniAvaEquation` actually
does and compares it against them:

* **Measured influence.** Sweeping one stream over ``[0, 1]`` against random
  backgrounds must move the fused score by exactly that stream's declared
  weight. A stream whose measured influence is zero is dead weight on the
  equation, and one whose influence disagrees with its declared weight means
  the served arithmetic has drifted from the published table.
* **eta decoupling.** ``eta ** Phi`` is documented as not scaling the fused
  score. Measured across the permitted eta range, the fused score must be
  invariant -- the property that lets a batch be ranked at all, since a
  per-call scalar cannot reorder it.
* **Convexity.** Streams in ``[0, 1]`` must fuse into ``[0, 1]``, and three
  equal streams must fuse to their common value.

Together with the certifier this is a chain of custody: the certifier holds the
declared table to the published derivation, and this tool holds served
behaviour to the declared table. Neither alone is sufficient, and this half
cannot see a table that moved in lock-step with the arithmetic -- that case is
the certifier's.

A previous version of this tool drove random Gaussian tensors through an
inline ``w_R * a + w_H * b + w_O * c`` and asserted the result kept its shape,
the fused L2 norm was bounded by the largest input's, and equal inputs fused
to themselves. Those are convex-combination identities that hold for any
non-negative weights summing to 1: it returned ``ok`` for ``(1, 0, 0)``,
``(0, 0, 1)`` and uniform alike, failing only on a negative weight, which the
weight certifier already covers. It never imported the OAE, so no drift in the
served equation could have reached it.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.oae_dimensionality_probe/v1"

#: Streams, in the order the published weight table names them.
_STREAMS = (("recursion_score", "w_R"), ("resonance_score", "w_H"), ("optimization_score", "w_O"))

#: Exact float arithmetic on a weighted sum; the tolerance covers rounding only.
_TOL = 1e-9

#: Permitted eta range (``OmniAvaEquation`` clamps outside it and warns).
_ETA_SWEEP = (0.93, 0.94, 0.95, 0.96)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.oae_dimensionality_probe",
        description="Measure the shipped OAE's per-stream influence against its declared weights.",
    )
    parser.add_argument("--batch", type=int, default=8, help="Accepted for compatibility.")
    parser.add_argument("--dim", type=int, default=16, help="Accepted for compatibility.")
    parser.add_argument("--trials", type=int, default=32, help="Random backgrounds per stream.")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _collect(args: argparse.Namespace) -> Certificate:
    from omni_mercury_engine.core.three_r.fusion import OmniAvaEquation

    rng = np.random.default_rng(args.seed)
    equation = OmniAvaEquation()
    weights: dict[str, float] = {k: float(v) for k, v in equation.weights.items()}

    def fuse(triple: dict[str, float], eq: OmniAvaEquation = equation) -> float:
        return float(eq.compute(**triple).fusion_score)

    failures: list[str] = []
    influence: dict[str, dict[str, float]] = {}

    for stream, weight_key in _STREAMS:
        declared = weights[weight_key]
        deltas: list[float] = []
        for _ in range(max(1, int(args.trials))):
            background = {s: float(rng.random()) for s, _ in _STREAMS}
            low = fuse({**background, stream: 0.0})
            high = fuse({**background, stream: 1.0})
            deltas.append(high - low)
        measured = float(np.mean(deltas))
        spread = float(np.max(deltas) - np.min(deltas))
        influence[stream] = {
            "declared_weight": declared,
            "measured_influence": measured,
            "across_backgrounds_spread": spread,
            "abs_error": abs(measured - declared),
        }
        if measured <= _TOL:
            failures.append(
                f"{stream} is a dead stream: sweeping it over [0, 1] moves the fused "
                f"score by {measured:.3e}, though it is declared weight {declared:.6f}"
            )
        elif abs(measured - declared) > _TOL:
            failures.append(
                f"{stream} influence {measured:.9f} != declared {weight_key} "
                f"{declared:.9f}; the served fusion has drifted from the published weights"
            )
        if spread > _TOL:
            failures.append(
                f"{stream} influence varies by {spread:.3e} across backgrounds; the "
                "fusion is not the additive weighted sum the equation publishes"
            )

    # eta decoupling: a per-call scalar must not move the fused score.
    probe = {s: float(rng.random()) for s, _ in _STREAMS}
    eta_scores = {
        str(eta): fuse(probe, OmniAvaEquation(ethical_compliance_threshold=eta))
        for eta in _ETA_SWEEP
    }
    eta_span = float(max(eta_scores.values()) - min(eta_scores.values()))
    if eta_span > _TOL:
        failures.append(
            f"the fused score moved {eta_span:.3e} across the permitted eta range; "
            "eta ** Phi is documented as decoupled from it"
        )

    # Convexity: the unit cube maps into the unit interval, and the diagonal is fixed.
    out_of_range: list[dict[str, float]] = []
    for _ in range(max(1, int(args.trials))):
        triple = {s: float(rng.random()) for s, _ in _STREAMS}
        fused = fuse(triple)
        if not 0.0 <= fused <= 1.0:
            out_of_range.append({**triple, "fused": fused})
    if out_of_range:
        failures.append(
            f"{len(out_of_range)} of {args.trials} unit-cube inputs fused outside [0, 1]"
        )
    diagonal: dict[str, float] = {}
    for value in (0.0, 0.25, 0.5, 0.75, 1.0):
        fused = fuse(dict.fromkeys((s for s, _ in _STREAMS), value))
        diagonal[str(value)] = fused
        if abs(fused - value) > _TOL:
            failures.append(
                f"three equal streams at {value} fused to {fused:.9f}; weights summing "
                "to 1 must leave the diagonal fixed"
            )

    body: dict[str, Any] = {
        "trials": int(args.trials),
        "weights": {**weights, "sum": float(sum(weights.values()))},
        "influence": influence,
        "eta_sweep": eta_scores,
        "eta_span": eta_span,
        "diagonal": diagonal,
        "out_of_range": out_of_range[:8],
        "failures": failures,
    }
    if failures:
        return Certificate(
            tool="oae_dimensionality_probe",
            schema=_SCHEMA,
            status="fail",
            body=body,
            warnings=failures,
        )
    return Certificate(tool="oae_dimensionality_probe", schema=_SCHEMA, status="ok", body=body)


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
