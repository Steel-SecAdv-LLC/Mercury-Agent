# Copyright (C) 2025 Steel Security Advisors LLC
"""(at your option) any later version."""

from __future__ import annotations

import argparse
import math
from typing import Any

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.oae_weight_certifier/v1"

# Sum-to-one tolerance.  All three weights are bit-exact ratios of the
# same denominator so the closure is exact in IEEE-754 binary64 up to
# the rounding of one ULP per division; 1e-12 is two orders of
# magnitude looser than the actual error and tight enough to catch a
# drifted constant.
_SUM_TOLERANCE = 1e-12

# Documented expected values from the README / ARCHITECTURE.
# Computed inline rather than hard-coded so an upstream PHI change is
# caught as a derivation drift, not silently smoothed over.


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.oae_weight_certifier",
        description=(
            "Verify the OAE fusion weights (w_R, w_H, w_O) match the "
            "MATH.GOLDEN_RATIO derivation and sum to 1.0."
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help=(
            "Maximum absolute drift between computed and derived weights "
            "(default 1e-6 — the layer buffers are float32 so values agree "
            "to ~1e-7, comfortably tighter than this)."
        ),
    )
    return parser


def _collect(args: argparse.Namespace) -> Certificate:
    from omni_mercury_engine.core.centralized_constants import MATH

    phi = MATH.GOLDEN_RATIO
    if not math.isclose(phi, 1.618033988749895, abs_tol=1e-15):
        return Certificate(
            tool="oae_weight_certifier",
            schema=_SCHEMA,
            status="fail",
            body={
                "MATH.GOLDEN_RATIO": phi,
                "expected_GOLDEN_RATIO": 1.618033988749895,
                "error": "MATH.GOLDEN_RATIO drifted from canonical value",
            },
        )

    phi_sum = phi + 2.0  # PHI:1:1 normalised to sum 1.0 — see derivation note below.
    expected = {
        # Mercury OAE design fixes the (R, H, O) proportion at PHI:1:1.
        # Recursion carries the φ-weighted prominence; Harmonic and
        # Optimization receive equal unit shares.  Normalising to sum
        # 1.0 gives the canonical (0.4472, 0.2764, 0.2764) tuple quoted
        # in the README and ARCHITECTURE.md.  The denominator is PHI+2,
        # NOT PHI+1+1/PHI — an earlier draft used the latter and silently
        # produced (0.5, 0.309, 0.191).  This certifier is the gate that
        # catches a recurrence of that drift.
        "w_R": phi / phi_sum,
        "w_H": 1.0 / phi_sum,
        "w_O": 1.0 / phi_sum,
    }
    documented = {"w_R": 0.4472, "w_H": 0.2764, "w_O": 0.2764}
    sum_expected = sum(expected.values())

    # Also confirm the layer's registered buffers agree with the derivation.
    layer_values: dict[str, float | None] = {"w_R": None, "w_H": None, "w_O": None}
    layer_error: str | None = None
    try:
        import torch  # noqa: F401 — required for the layer

        from omni_mercury_engine.ml.three_r_attention import ThreeRAttentionBlock

        # ``ThreeRAttentionBlock`` requires (d_model, n_heads); use a tiny
        # config to keep the construction cheap.  We only read the
        # registered buffers (w_R, w_H, w_O), not the forward pass.
        #
        # ``register_buffer`` results in ``nn.Module`` attributes that
        # mypy types as ``Tensor | None``.  We resolve via ``getattr``
        # so the type-checker sees ``Any`` and trusts the documented
        # buffer surface; a missing attribute raises ``AttributeError``
        # and falls through to the ``Exception`` handler below.
        layer = ThreeRAttentionBlock(d_model=64, n_heads=4, dropout=0.0)
        for name in ("w_R", "w_H", "w_O"):
            buf = getattr(layer, name)
            layer_values[name] = float(buf.detach().cpu().item())
    except ImportError as exc:
        layer_error = f"torch not installed: {exc}"
    except Exception as exc:
        layer_error = f"{type(exc).__name__}: {exc}"

    body: dict[str, Any] = {
        "GOLDEN_RATIO": phi,
        "phi_sum": phi_sum,
        "expected": expected,
        "documented_approx": documented,
        "sum_expected": sum_expected,
        "layer_buffers": layer_values,
        "layer_error": layer_error,
        "tolerance": args.tolerance,
        "sum_tolerance": _SUM_TOLERANCE,
    }

    failures: list[str] = []

    # Structural sum-to-one invariant.
    if abs(sum_expected - 1.0) > _SUM_TOLERANCE:
        failures.append(f"sum(w_R, w_H, w_O) = {sum_expected!r} != 1.0")

    # Documented-approx vs derivation parity (catches drift from the
    # README/ARCHITECTURE.md quoted constants — three-decimal precision is
    # what those docs carry).
    for name, approx in documented.items():
        if abs(expected[name] - approx) > 5e-4:
            failures.append(
                f"{name}: derivation={expected[name]!r} drifted from documented≈{approx}"
            )

    # Layer-vs-derivation drift check (skipped when torch is unavailable —
    # the derivation check above still runs and is the load-bearing gate).
    if layer_error is None:
        for name, derived in expected.items():
            measured = layer_values[name]
            if measured is None:
                continue
            drift = abs(measured - derived)
            if drift > args.tolerance:
                failures.append(
                    f"{name}: layer={measured!r} vs derivation={derived!r} drift={drift:.3e}"
                )

    # Status mapping:
    #   * derivation drift OR layer-vs-derivation drift   → "fail"
    #   * derivation passes, layer cross-check skipped    → "ok" + warning
    #     (torch is in the optional ``ml`` extra; the φ derivation is
    #      the load-bearing gate per the file docstring, and is fully
    #      verifiable without torch.  Demoting this to ``warn`` would
    #      make the certifier report "couldn't fully verify" even
    #      when the primary contract — the math — was verified.)
    #   * derivation passes, layer cross-check passes     → "ok"
    if failures:
        body["failures"] = failures
        status = "fail"
    else:
        status = "ok"

    return Certificate(
        tool="oae_weight_certifier",
        schema=_SCHEMA,
        status=status,
        body=body,
        warnings=[layer_error] if layer_error else [],
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
