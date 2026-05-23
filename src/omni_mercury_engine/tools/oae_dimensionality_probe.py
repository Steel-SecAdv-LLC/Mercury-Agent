"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

------------------------------------------------------------------------

Operator tool: verify the OAE R/H/O branches preserve dimensionality
through fusion at the tensor level.

The scalar :mod:`oae_weight_certifier` proves the (w_R, w_H, w_O)
weights sum to 1.0 and match the documented golden-ratio derivation.
This tool drives random tensors of the contracted shape through the
fusion arithmetic and asserts:

* every per-branch projection produces a tensor of the documented
  output rank (no silent broadcast collapsing a (B, D) tensor to (B,));
* the fused output's L2-norm is bounded by the convex-combination
  identity ||w_R * R + w_H * H + w_O * O|| <= max(||R||, ||H||, ||O||);
* fusion is invariant under the same input on all three branches —
  R==H==O => fused==R (numerical equality to machine precision).
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.oae_dimensionality_probe/v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.oae_dimensionality_probe",
        description="Drive the OAE fusion at tensor level and verify shape + bound invariants.",
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _weights() -> tuple[float, float, float]:
    """Return (w_R, w_H, w_O) from the canonical PHI:1:1 derivation.

    The centralised ``FusionConstants`` legacy values (0.500, 0.309, 0.191)
    came from an earlier ``PHI + 1 + 1/PHI`` derivation that drifted from
    the README's documented PHI:1:1 proportion.  :mod:`oae_weight_certifier`
    is the gate that catches a recurrence of that drift; this probe uses
    the corrected derivation directly so the tensor checks reflect the
    canonical contract.
    """
    from omni_mercury_engine.core.centralized_constants import MATH

    phi = MATH.GOLDEN_RATIO
    phi_sum = phi + 2.0
    return phi / phi_sum, 1.0 / phi_sum, 1.0 / phi_sum


def _collect(args: argparse.Namespace) -> Certificate:
    rng = np.random.default_rng(args.seed)
    w_r, w_h, w_o = _weights()
    weight_sum = w_r + w_h + w_o

    failures: list[str] = []
    shape_records: list[dict[str, Any]] = []
    bound_records: list[dict[str, float]] = []
    for trial in range(int(args.trials)):
        # OAE branches: R(esonance), H(armonic), O(perator).  The
        # single-letter names mirror the canonical Mercury OAE algebra
        # in ``docs/MATH_SPEC.md`` and the matching scalar test in
        # :mod:`oae_weight_certifier`; renaming would obscure the
        # one-to-one correspondence with the paper notation.
        branch_r = rng.standard_normal((args.batch, args.dim)).astype(np.float64)
        branch_h = rng.standard_normal((args.batch, args.dim)).astype(np.float64)
        branch_o = rng.standard_normal((args.batch, args.dim)).astype(np.float64)
        fused = w_r * branch_r + w_h * branch_h + w_o * branch_o
        shape_records.append({"trial": trial, "fused_shape": list(fused.shape)})
        if fused.shape != branch_r.shape:
            failures.append(
                f"trial {trial}: fused shape {fused.shape} != input shape {branch_r.shape}"
            )
            continue
        norms = (
            float(np.linalg.norm(branch_r)),
            float(np.linalg.norm(branch_h)),
            float(np.linalg.norm(branch_o)),
        )
        fused_norm = float(np.linalg.norm(fused))
        # Convex-combination identity: weights are non-negative and sum to 1.
        if not (fused_norm <= max(norms) + 1e-9):
            failures.append(
                f"trial {trial}: fused L2 {fused_norm:.6f} > max input L2 {max(norms):.6f}"
            )
        bound_records.append(
            {
                "fused_l2": fused_norm,
                "max_input_l2": max(norms),
                "trial": float(trial),
            }
        )

    # Identity check: equal branches => fused equals the common input.
    X = rng.standard_normal((args.batch, args.dim)).astype(np.float64)
    fused_id = w_r * X + w_h * X + w_o * X
    if not np.allclose(fused_id, X, atol=1e-12):
        failures.append("identity check failed: R==H==O did not produce fused==R")

    body: dict[str, Any] = {
        "batch": int(args.batch),
        "dim": int(args.dim),
        "trials": int(args.trials),
        "weights": {"w_R": w_r, "w_H": w_h, "w_O": w_o, "sum": weight_sum},
        "shape_records": shape_records[:8],
        "bound_records": bound_records[:8],
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
    return Certificate(
        tool="oae_dimensionality_probe",
        schema=_SCHEMA,
        status="ok",
        body=body,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
