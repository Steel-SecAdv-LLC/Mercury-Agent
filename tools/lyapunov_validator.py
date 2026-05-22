"""Deterministic Lyapunov validator for Mercury Agent stability claims.

This module provides an *executable* validator for the Lyapunov decay
condition

    dot{V}(x) <= -lambda * V(x)

used throughout Mercury Agent's stability proofs (see ``docs/MATH_SPEC.md``
Section 2.2 and the canonical config ``configs/lyapunov_canonical.yaml``).
Two modes are supported:

1. **Certified quadratic mode.**  Given a linear system ``A`` and a
   symmetric matrix ``P`` defining a quadratic Lyapunov candidate
   ``V(x) = x^T P x``, the validator computes the *largest* lambda for
   which the decay condition is provably satisfied for *all* states
   ``x``.  This is done by solving the symmetric--definite generalized
   eigenvalue problem ``Q v = mu P v`` where ``Q = A^T P + P A``; the
   certified rate is ``lambda* = -mu_max``.  No sampling, no Monte
   Carlo: the result is a numerical certificate up to floating-point
   precision.

2. **Sample-based fallback.**  For general (potentially non-quadratic)
   ``V`` and dynamics ``f``, the caller supplies a list of evaluated
   samples ``[{"x": ..., "V": ..., "Vdot": ...}, ...]`` and the
   validator reports the worst observed ratio
   ``lambda_hat = inf_s ( -Vdot_s / V_s )``.  This is *not* a proof but
   a measurable empirical lower bound suitable for regression gating.

The module deliberately depends only on :mod:`numpy` (which is already a
core install requirement) and :mod:`yaml` for config loading; no
:mod:`scipy` dependency is introduced.  The generalized eigenvalue
solver is implemented via Cholesky decomposition of ``P`` followed by
``numpy.linalg.eigvalsh`` on the resulting symmetric similar matrix --
mathematically equivalent to :func:`scipy.linalg.eigh` on the pencil
``(Q, P)`` for symmetric--definite pencils.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "canonical_lambda_for_linear_system",
    "generalized_eigvals_sym_pd",
    "is_positive_definite",
    "validate_lyapunov_from_config",
    "validate_quadratic",
    "validate_samples",
]

# Floating-point tolerances.  ``_PD_TOL`` gates "P is positive definite"
# (eigenvalue floor); ``_LAMBDA_TOL`` gates the agreement between the
# claimed and computed lambda.  Both are chosen >> typical double-
# precision round-off for the matrix sizes encountered in Mercury Agent
# Lyapunov proofs (n <= 32) but small enough to flag genuine drift.
_PD_TOL: float = 1e-9
_LAMBDA_TOL: float = 1e-8


def _as_float_matrix(name: str, data: Any) -> NDArray[np.float64]:
    """Coerce ``data`` to a 2D float64 array, raising :class:`ValueError`."""
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix, got ndim={arr.ndim}")
    if arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be square, got shape={arr.shape}")
    return arr


def is_positive_definite(M: NDArray[np.float64], tol: float = _PD_TOL) -> bool:
    """Return True iff ``M`` is symmetric positive-definite within ``tol``.

    Symmetry is enforced by comparing ``M`` to ``M.T`` with the same
    tolerance.  Positive-definiteness is established via the smallest
    eigenvalue of the symmetrised matrix ``(M + M.T) / 2``.
    """
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        return False
    sym = 0.5 * (M + M.T)
    if not np.allclose(M, sym, atol=max(tol, 1e-12)):
        return False
    vals = np.linalg.eigvalsh(sym)
    return bool(float(vals.min()) > tol)


def generalized_eigvals_sym_pd(
    Q: NDArray[np.float64], P: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Eigenvalues of the symmetric--definite pencil ``(Q, P)``.

    Solves ``Q v = mu P v`` for symmetric ``Q`` and symmetric
    positive-definite ``P`` using the Cholesky factorisation
    ``P = L L^T`` followed by ``numpy.linalg.eigvalsh`` on the
    congruent symmetric matrix ``L^{-1} Q L^{-T}``.

    Raises :class:`numpy.linalg.LinAlgError` if ``P`` is not positive
    definite.
    """
    Q_sym = 0.5 * (Q + Q.T)
    P_sym = 0.5 * (P + P.T)
    L = np.linalg.cholesky(P_sym)
    # Solve L Y = Q_sym  =>  Y = L^{-1} Q_sym
    Y = np.linalg.solve(L, Q_sym)
    # Solve L Z^T = Y^T => Z = (L^{-1} Y^T)^T = Y L^{-T}; combined yields
    # ``L^{-1} Q_sym L^{-T}``.
    M = np.linalg.solve(L, Y.T).T
    M = 0.5 * (M + M.T)
    return np.linalg.eigvalsh(M).astype(np.float64, copy=False)


def canonical_lambda_for_linear_system(A: NDArray[np.float64], P: NDArray[np.float64]) -> float:
    """Compute the *largest* ``lambda`` certifying ``dot V <= -lambda V``.

    For ``V(x) = x^T P x`` with symmetric positive-definite ``P`` and
    linear dynamics ``dot x = A x``, the time derivative is
    ``dot V = x^T (A^T P + P A) x``.  The supremum of
    ``(-dot V) / V`` over non-zero ``x`` equals ``-mu_max`` where
    ``mu_max`` is the largest generalized eigenvalue of the pencil
    ``(A^T P + P A, P)``.  This function returns that quantity.
    """
    A_arr = _as_float_matrix("A", A)
    P_arr = _as_float_matrix("P", P)
    if A_arr.shape != P_arr.shape:
        raise ValueError(f"A and P must have matching shape; got {A_arr.shape} vs {P_arr.shape}")
    Q = A_arr.T @ P_arr + P_arr @ A_arr
    mu = generalized_eigvals_sym_pd(Q, P_arr)
    return float(-mu.max())


def validate_quadratic(
    A: NDArray[np.float64],
    P: NDArray[np.float64],
    claimed_lambda: float,
    tol: float = _LAMBDA_TOL,
) -> tuple[bool, dict[str, Any]]:
    """Verify the claim ``dot V <= -claimed_lambda * V`` for ``V = x^T P x``.

    Returns a ``(ok, details)`` tuple where ``details`` always contains
    ``computed_lambda``, ``claimed_lambda``, ``max_generalized_eig``,
    and ``ok``.  ``ok`` is True iff ``computed_lambda >= claimed_lambda
    - tol`` *and* ``P`` is positive definite.
    """
    A_arr = _as_float_matrix("A", A)
    P_arr = _as_float_matrix("P", P)
    if not is_positive_definite(P_arr):
        return False, {
            "error": "P not positive definite",
            "claimed_lambda": float(claimed_lambda),
        }
    Q = A_arr.T @ P_arr + P_arr @ A_arr
    mu = generalized_eigvals_sym_pd(Q, P_arr)
    max_mu = float(mu.max())
    computed_lambda = -max_mu
    ok = computed_lambda >= (claimed_lambda - tol)
    return ok, {
        "max_generalized_eig": max_mu,
        "computed_lambda": computed_lambda,
        "claimed_lambda": float(claimed_lambda),
        "tol": float(tol),
        "ok": bool(ok),
    }


def validate_samples(
    samples: Sequence[Mapping[str, Any]],
    claimed_lambda: float,
    tol: float = _LAMBDA_TOL,
) -> tuple[bool, dict[str, Any]]:
    """Sample-based decay-ratio fallback for general ``V`` and ``f``.

    Each entry of ``samples`` must provide numeric ``V`` (> 0) and
    ``Vdot``.  Returns ``(ok, details)`` with ``computed_lambda`` set to
    the worst observed ratio ``inf_s ( -Vdot_s / V_s )``.
    """
    if not samples:
        return False, {"error": "no samples provided"}
    min_ratio = math.inf
    for idx, s in enumerate(samples):
        try:
            V = float(s["V"])
            Vdot = float(s["Vdot"])
        except (KeyError, TypeError, ValueError) as exc:
            return False, {"error": f"sample {idx}: {exc}"}
        if V <= 0:
            return False, {"error": f"sample {idx}: V must be > 0, got {V}"}
        ratio = -Vdot / V
        min_ratio = min(min_ratio, ratio)
    computed_lambda = float(min_ratio)
    ok = computed_lambda >= (claimed_lambda - tol)
    return ok, {
        "computed_lambda": computed_lambda,
        "claimed_lambda": float(claimed_lambda),
        "tol": float(tol),
        "num_samples": len(samples),
        "ok": bool(ok),
    }


def _extract_lyapunov_block(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    """Locate the Lyapunov certificate inside a YAML config.

    Accepts three forms so a single validator can serve both single-purpose
    Lyapunov configs (``configs/lyapunov_canonical.yaml``) and multi-variant
    experiment configs (``configs/ablation_3r_lyapunov.yaml``) without
    conflating ablation parameters with the stability certificate:

    1. Flat: top-level ``A``/``P``/``lambda`` or ``lyapunov_samples``.
    2. Nested: ``lyapunov: {A, P, lambda}`` (or with ``lyapunov_samples``).
    3. Multi-variant with a sibling ``lyapunov:`` block (the runner gates the
       whole experiment on a single certificate; individual variants share it).
    """
    flat_has_matrices = "A" in cfg and "P" in cfg
    flat_has_samples = isinstance(cfg.get("lyapunov_samples"), list)
    if flat_has_matrices or flat_has_samples:
        return cfg
    nested = cfg.get("lyapunov")
    if isinstance(nested, Mapping):
        return nested
    return {}


def validate_lyapunov_from_config(cfg_path: Path) -> tuple[bool, dict[str, Any]]:
    """Validate a Lyapunov claim defined in a YAML config file.

    The config may declare the certificate at the top level or under a nested
    ``lyapunov:`` block (so an ablation config can carry both experiment
    parameters and the stability certificate without conflating them).
    Inside whichever block is located, the validator accepts:

    * ``A`` (n x n) and ``P`` (n x n) matrices plus a ``lambda`` scalar
      -- triggers :func:`validate_quadratic`; or
    * a ``lyapunov_samples`` list of ``{"x": ..., "V": ..., "Vdot":
      ...}`` records plus a ``lambda`` scalar -- triggers
      :func:`validate_samples`.

    The ``lambda`` field is **required** and must be strictly positive:
    a missing or non-positive ``lambda`` is a config error (not a
    vacuously-satisfied claim), and the validator returns
    ``(False, {"error": "..."})`` rather than coercing it to ``0.0``.

    Returns ``(ok, details)`` with ``details["mode"]`` indicating which
    path was taken.  Missing/invalid config returns ``(False, {...})``
    rather than raising, to keep callers (CI gates) simple.  PyYAML is
    declared as a core dependency in ``pyproject.toml``; if it is none
    the less missing (e.g. a deeply-minimal install), the validator
    returns a structured error rather than letting the ``ImportError``
    propagate.
    """
    try:
        import yaml  # local import keeps ``tools.lyapunov_validator`` lazy
    except ImportError as exc:
        return False, {
            "error": (
                "PyYAML is required to load Lyapunov config files but is "
                f"not installed in this environment ({exc}). Install with "
                "`pip install pyyaml>=6.0` or `pip install mercury-agent`."
            )
        }

    if not cfg_path.exists():
        return False, {"error": f"config not found: {cfg_path}"}
    try:
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
    except yaml.YAMLError as exc:
        return False, {"error": f"invalid yaml: {exc}"}
    if not isinstance(cfg, Mapping):
        return False, {"error": "config root must be a mapping"}

    block = _extract_lyapunov_block(cfg)
    if not block:
        return False, {
            "error": (
                "config provides neither top-level (A, P) matrices, "
                "lyapunov_samples, nor a nested `lyapunov:` block"
            ),
            "mode": "unknown",
        }

    lambda_raw = block.get("lambda")
    if lambda_raw is None:
        return False, {
            "error": (
                "Lyapunov block is missing the required `lambda` field. "
                "Declare a strictly-positive decay rate or remove the "
                "block entirely; a missing claim is not a vacuous pass."
            ),
            "mode": "unknown",
        }
    try:
        claimed_lambda = float(lambda_raw)
    except (TypeError, ValueError) as exc:
        return False, {"error": f"lambda is not numeric ({lambda_raw!r}): {exc}"}
    if not math.isfinite(claimed_lambda) or claimed_lambda <= 0.0:
        return False, {
            "error": (
                f"lambda must be strictly positive and finite (got "
                f"{claimed_lambda!r}); a zero or negative claim certifies "
                "any stable system vacuously."
            ),
        }

    A_raw = block.get("A")
    P_raw = block.get("P")

    if A_raw is not None and P_raw is not None:
        try:
            A = _as_float_matrix("A", A_raw)
            P = _as_float_matrix("P", P_raw)
        except ValueError as exc:
            return False, {"error": str(exc)}
        ok, details = validate_quadratic(A, P, claimed_lambda)
        details["mode"] = "quadratic"
        return ok, details

    samples_raw = block.get("lyapunov_samples")
    if isinstance(samples_raw, list):
        ok, details = validate_samples(samples_raw, claimed_lambda)
        details["mode"] = "samples"
        return ok, details

    return False, {
        "error": ("Lyapunov block missing required (A, P) matrices or lyapunov_samples"),
        "mode": "unknown",
    }


def _cli(argv: Sequence[str] | None = None) -> int:
    """Module-level CLI entry-point: ``python -m tools.lyapunov_validator``.

    Exits 0 on certified pass, 1 on a numerical failure (the claim does not
    hold), and 2 on a usage/config error.  The full details dictionary is
    written to stdout as JSON so downstream tooling can consume it.
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="python -m tools.lyapunov_validator",
        description=(
            "Certify the Lyapunov decay claim in a YAML config. "
            "Accepts top-level (A, P, lambda) or a nested `lyapunov:` block."
        ),
    )
    parser.add_argument("config", type=Path, help="Path to YAML config")
    args = parser.parse_args(argv)

    ok, details = validate_lyapunov_from_config(args.config)
    print(json.dumps(details, indent=2, sort_keys=True))
    if "error" in details and "ok" not in details:
        return 2
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry-point
    raise SystemExit(_cli())
