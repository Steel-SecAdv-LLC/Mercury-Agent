# Copyright (C) 2025 Steel Security Advisors LLC
"""(at your option) any later version."""

from __future__ import annotations

import argparse
import math
from typing import Any

import numpy as np
import numpy.typing as npt

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.lyapunov_validator/v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.lyapunov_validator",
        description=(
            "Validate a Lyapunov stability trajectory V(t) against the Mercury "
            "exponential-decay bound V(t) <= eps * exp(-lambda * t)."
        ),
    )
    parser.add_argument(
        "--trajectory",
        default=None,
        help=(
            "Optional path to a .npy file with a 1-D array of V(t) values "
            "sampled at unit-time intervals. If omitted, a known-stable "
            "synthetic trajectory is generated for self-test."
        ),
    )
    parser.add_argument(
        "--lambda",
        dest="lambda_",
        type=float,
        default=None,
        help="Override LYAPUNOV.LAMBDA_CONVERGENCE (default: read from constants).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Override LYAPUNOV.EPSILON_INITIAL (default: read from constants).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=None,
        help="Stability window in samples (default: LYAPUNOV.STABILITY_WINDOW).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Length of the synthetic trajectory when --trajectory is omitted.",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help=(
            "Std-dev of additive Gaussian noise on the synthetic trajectory. "
            "Use to confirm the bound is robust to small perturbations."
        ),
    )
    return parser


def _load_trajectory(path: str) -> npt.NDArray[np.float64]:
    arr = np.load(path, allow_pickle=False)
    if arr.ndim != 1:
        raise ValueError(f"trajectory must be 1-D, got shape {arr.shape}")
    return np.asarray(arr, dtype=np.float64)


def _synth_trajectory(eps: float, lam: float, n: int, noise: float) -> npt.NDArray[np.float64]:
    # ``lam_actual`` is intentionally slightly larger than the contractual
    # ``lam`` so the synthetic trajectory comfortably sits under the
    # bound — a real training run is expected to decay no slower than
    # the contract, not exactly at it.
    lam_actual = lam * 1.05
    t = np.arange(n, dtype=np.float64)
    v = eps * np.exp(-lam_actual * t)
    if noise > 0.0:
        rng = np.random.default_rng(seed=0)
        # Clip below 0 — a Lyapunov function must be non-negative.
        v = np.maximum(0.0, v + rng.normal(0.0, noise, size=v.shape))
    return v


def _collect(args: argparse.Namespace) -> Certificate:
    from omni_mercury_engine.core.centralized_constants import LYAPUNOV

    lam = float(args.lambda_) if args.lambda_ is not None else float(LYAPUNOV.LAMBDA_CONVERGENCE)
    eps = float(args.epsilon) if args.epsilon is not None else float(LYAPUNOV.EPSILON_INITIAL)
    window = int(args.window) if args.window is not None else int(LYAPUNOV.STABILITY_WINDOW)

    if args.trajectory:
        v = _load_trajectory(args.trajectory)
        source = f"file:{args.trajectory}"
    else:
        v = _synth_trajectory(eps, lam, args.samples, args.noise)
        source = f"synthetic(n={args.samples}, noise={args.noise})"

    t = np.arange(v.shape[0], dtype=np.float64)
    bound = eps * np.exp(-lam * t)
    violations = np.where(v > bound)[0]
    violation_count = int(violations.size)
    max_excess = float(np.max(v - bound)) if v.size else 0.0
    last_violation = int(violations[-1]) if violation_count else -1

    # Restabilised: at least ``window`` consecutive samples under the bound
    # after the last violation.
    re_stabilised = (v.size - 1 - last_violation) >= window if v.size else True

    # Empirical decay rate estimate (least-squares fit on log V):
    # ``log V(t) ≈ log V0 - lam_hat * t``.  Operators use this to spot a
    # trajectory whose long-run rate is shallower than the contractual
    # bound even if the per-sample check passes.
    finite = v > 0
    if finite.sum() >= 2:
        ts = t[finite]
        lvs = np.log(v[finite])
        a, b = np.polyfit(ts, lvs, 1)
        lam_hat = float(-a)
        v0_hat = float(math.exp(b))
    else:
        lam_hat = float("nan")
        v0_hat = float("nan")

    body: dict[str, Any] = {
        "source": source,
        "samples": int(v.size),
        "lambda_contract": lam,
        "epsilon_contract": eps,
        "stability_window": window,
        "violation_count": violation_count,
        "max_excess_over_bound": max_excess,
        "last_violation_index": last_violation,
        "restabilised": bool(re_stabilised),
        "empirical_lambda": lam_hat,
        "empirical_epsilon": v0_hat,
    }

    failures: list[str] = []
    if violation_count > 0 and not re_stabilised:
        failures.append(
            f"trajectory violates V(t) <= eps*exp(-lambda*t) at {violation_count} samples "
            f"and is not restabilised within {window} samples"
        )
    if not math.isnan(lam_hat) and lam_hat + 1e-6 < lam:
        failures.append(
            f"empirical decay rate lambda_hat={lam_hat:.4f} is shallower than the "
            f"contractual lambda={lam:.4f}"
        )

    if failures:
        body["failures"] = failures
        status = "fail"
    elif violation_count > 0 and re_stabilised:
        status = "warn"
    else:
        status = "ok"

    warnings: list[str] = []
    if violation_count > 0 and status != "fail":
        warnings.append(
            f"{violation_count} transient violation(s) restabilised within {window} samples"
        )

    return Certificate(
        tool="lyapunov_validator",
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
