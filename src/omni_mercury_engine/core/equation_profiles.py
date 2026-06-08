# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Runtime equation profiles for Mercury fusion scores.

Profiles preserve ``baseline_original_v1`` as the frozen reference while
allowing candidate profiles to be selected explicitly at runtime. The
``phi_fibring_v1`` candidate harmonises the neural/equation blend with
Mercury's canonical golden-ratio fibring fusion
(:mod:`omni_mercury_engine.core.fibring_fusion`): a phi-weighted base split
(``phi/(1+phi) : 1/(1+phi)``) plus correlation-aware decorrelation, so a
runtime equation signal that merely echoes the neural score cannot
double-count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from omni_mercury_engine.core.fibring_fusion import REDUNDANCY_THRESHOLD
from omni_mercury_engine.core.three_r.types import GOLDEN_RATIO_CONSTANT

_EPS = 1e-8
BASELINE_PROFILE_ID = "baseline_original_v1"
QUIET_HORIZON_PROFILE_ID = "quiet_horizon_v1"
PHI_FIBRING_PROFILE_ID = "phi_fibring_v1"


@dataclass(frozen=True)
class RuntimeEquationProfile:
    """Runtime profile metadata and scoring weights."""

    profile_id: str
    description: str
    neural_weight: float
    equation_weight: float
    ethical_exponent: float
    formula: str
    # When True, ``score_runtime_equation_profile`` treats ``neural_weight`` /
    # ``equation_weight`` as the *base* split and applies the canonical
    # correlation-aware decorrelation (see ``core.fibring_fusion``) before the
    # blend; static profiles keep their fixed weights.
    decorrelate: bool = False


_PROFILES = {
    BASELINE_PROFILE_ID: RuntimeEquationProfile(
        profile_id=BASELINE_PROFILE_ID,
        description="Frozen original OAE baseline blended with the runtime neural score.",
        neural_weight=0.70,
        equation_weight=0.30,
        ethical_exponent=GOLDEN_RATIO_CONSTANT,
        formula="S=0.70*N+0.30*((w_R R+w_H H+w_O O)*eta^Phi)",
    ),
    QUIET_HORIZON_PROFILE_ID: RuntimeEquationProfile(
        profile_id=QUIET_HORIZON_PROFILE_ID,
        description=(
            "Original Mercury-derived quiet horizon candidate: a gentle agreement-weighted "
            "geometric horizon around R/H/O, never replacing the frozen baseline."
        ),
        neural_weight=0.70,
        equation_weight=0.30,
        ethical_exponent=GOLDEN_RATIO_CONSTANT**0.5,
        formula=(
            "S=0.70*N+0.30*(eta^sqrt(Phi)*(0.5+0.5*A_RHO)*"
            "(Phi*R+sqrt(HO)+cuberoot(RHO))/(Phi+2))"
        ),
    ),
    PHI_FIBRING_PROFILE_ID: RuntimeEquationProfile(
        profile_id=PHI_FIBRING_PROFILE_ID,
        description=(
            "Golden-ratio fibring candidate: phi-weighted neural/equation base "
            "split (phi/(1+phi) : 1/(1+phi)) with correlation-aware "
            "decorrelation, harmonised with core.fibring_fusion so a redundant "
            "equation signal cannot double-count the neural score."
        ),
        neural_weight=GOLDEN_RATIO_CONSTANT / (1.0 + GOLDEN_RATIO_CONSTANT),
        equation_weight=1.0 / (1.0 + GOLDEN_RATIO_CONSTANT),
        ethical_exponent=GOLDEN_RATIO_CONSTANT,
        formula=(
            "S=w_N*N+w_E*((w_R R+w_H H+w_O O)*eta^Phi); "
            "(w_N,w_E)=fibring(phi/(1+phi), 1/(1+phi)) decorrelated@|r|>=0.85"
        ),
        decorrelate=True,
    ),
}


def available_equation_profiles() -> tuple[str, ...]:
    """Return supported runtime profile identifiers."""
    return tuple(_PROFILES)


def get_equation_profile(profile_id: str | None) -> RuntimeEquationProfile | None:
    """Return a profile by id, or ``None`` when no runtime profile is selected."""
    if profile_id is None:
        return None
    try:
        return _PROFILES[profile_id]
    except KeyError as exc:
        supported = ", ".join(available_equation_profiles())
        raise ValueError(
            f"unknown equation profile {profile_id!r}; supported: {supported}"
        ) from exc


def score_runtime_equation_profile(
    raw_scores: np.ndarray[Any, Any],
    recursion_scores: np.ndarray[Any, Any],
    resonance_scores: np.ndarray[Any, Any],
    optimization_scores: np.ndarray[Any, Any],
    *,
    eta: float | np.ndarray[Any, Any] = 0.96,
    profile_id: str | None = BASELINE_PROFILE_ID,
) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
    """Apply a runtime equation profile to calibrated neural fusion scores.

    Args:
        raw_scores: Existing runtime anomaly probabilities.
        recursion_scores: R(x) signal in ``[0, 1]``.
        resonance_scores: H(ω) signal in ``[0, 1]``.
        optimization_scores: O(θ) signal in ``[0, 1]``.
        eta: Ethical gate value, scalar or per-sample array.
        profile_id: Profile to apply. ``None`` leaves raw scores unchanged.

    Returns:
        ``(scores, metadata)`` where scores are clipped to ``[0, 1]``.
    """
    profile = get_equation_profile(profile_id)
    raw = np.asarray(raw_scores, dtype=np.float64).reshape(-1)
    if profile is None:
        return np.clip(raw, 0.0, 1.0), {"profile_id": None, "applied": False}

    r = _align_component(recursion_scores, raw.shape[0])
    h = _align_component(resonance_scores, raw.shape[0])
    o = _align_component(optimization_scores, raw.shape[0])
    eta_arr = _align_component(np.asarray(eta, dtype=np.float64), raw.shape[0])

    phi = GOLDEN_RATIO_CONSTANT
    w_r = phi / (phi + 2.0)
    w_h = 1.0 / (phi + 2.0)
    w_o = 1.0 / (phi + 2.0)

    baseline_signal = (w_r * r + w_h * h + w_o * o) * np.power(eta_arr, profile.ethical_exponent)

    if profile.profile_id in (BASELINE_PROFILE_ID, PHI_FIBRING_PROFILE_ID):
        equation_signal = baseline_signal
    elif profile.profile_id == QUIET_HORIZON_PROFILE_ID:
        pairwise_spread = np.sqrt(((r - h) ** 2 + (h - o) ** 2 + (o - r) ** 2) / 3.0)
        agreement = np.clip(1.0 - pairwise_spread, 0.0, 1.0)
        geometric_horizon = np.cbrt(np.maximum(_EPS, r * h * o))
        harmonic_bridge = np.sqrt(np.maximum(_EPS, h * o))
        quiet_signal = (phi * r + harmonic_bridge + geometric_horizon) / (phi + 2.0)
        equation_signal = np.power(eta_arr, profile.ethical_exponent) * (0.5 + 0.5 * agreement)
        equation_signal = equation_signal * quiet_signal
    else:  # pragma: no cover - get_equation_profile validates profile ids
        equation_signal = baseline_signal

    raw_clipped = np.clip(raw, 0.0, 1.0)
    eq_clipped = np.clip(equation_signal, 0.0, 1.0)
    w_neural, w_equation, correlation, decorrelation_applied = _resolve_blend_weights(
        raw_clipped, eq_clipped, profile
    )
    scored = w_neural * raw_clipped + w_equation * eq_clipped
    metadata = {
        "profile_id": profile.profile_id,
        "applied": True,
        "formula": profile.formula,
        "description": profile.description,
        "neural_weight": float(w_neural),
        "equation_weight": float(w_equation),
        "correlation": correlation,
        "decorrelation_applied": decorrelation_applied,
        "component_means": {
            "recursion": float(np.mean(r)) if r.size else 0.0,
            "resonance": float(np.mean(h)) if h.size else 0.0,
            "optimization": float(np.mean(o)) if o.size else 0.0,
            "eta": float(np.mean(eta_arr)) if eta_arr.size else 0.0,
            "equation_signal": float(np.mean(equation_signal)) if equation_signal.size else 0.0,
        },
    }
    return np.clip(scored, 0.0, 1.0), metadata


def _pearson_correlation(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> float | None:
    """Pearson correlation of two equal-length series, or ``None`` if undefined.

    Mirrors ``fibring_fusion.FibringComposer._compute_correlation``: a series
    with (near-)zero variance has an undefined correlation, returned as
    ``None`` so no decorrelation fires.
    """
    if a.shape[0] < 2:
        return None
    if float(np.var(a)) <= _EPS or float(np.var(b)) <= _EPS:
        return None
    r = float(np.corrcoef(a, b)[0, 1])
    return r if np.isfinite(r) else None


def _resolve_blend_weights(
    neural: np.ndarray[Any, Any],
    equation: np.ndarray[Any, Any],
    profile: RuntimeEquationProfile,
) -> tuple[float, float, float | None, bool]:
    """Resolve the ``(neural, equation)`` blend weights for a profile.

    Static profiles return their fixed ``(neural_weight, equation_weight)``.
    Profiles with ``decorrelate=True`` apply the canonical correlation-aware
    decorrelation from :mod:`omni_mercury_engine.core.fibring_fusion`: when the
    two signals are redundant (``|Pearson r| >= REDUNDANCY_THRESHOLD``) the
    lower-variance (less informative) stream's weight is shrunk by
    ``1/(1+|r|)`` and the pair is renormalised to sum to 1. The blend therefore
    stays a convex combination of two ``[0, 1]`` signals (output bounded).
    """
    base_n = float(profile.neural_weight)
    base_e = float(profile.equation_weight)
    if not profile.decorrelate:
        return base_n, base_e, None, False

    correlation = _pearson_correlation(neural, equation)
    if correlation is None or abs(correlation) < REDUNDANCY_THRESHOLD:
        return base_n, base_e, correlation, False

    shrink = 1.0 / (1.0 + abs(correlation))
    if float(np.var(neural)) <= float(np.var(equation)):
        base_n *= shrink
    else:
        base_e *= shrink
    total = base_n + base_e
    if total <= 0.0:  # pragma: no cover - phi base weights are strictly positive
        return 0.5, 0.5, correlation, True
    return base_n / total, base_e / total, correlation, True


def components_from_score_channels(
    score_channels: dict[str, Any],
    *,
    raw_scores: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Map detector/model score channels onto runtime R/H/O components."""
    raw = np.asarray(raw_scores, dtype=np.float64).reshape(-1)
    n = raw.shape[0]
    recursion: list[np.ndarray[Any, Any]] = []
    resonance: list[np.ndarray[Any, Any]] = []
    optimization: list[np.ndarray[Any, Any]] = [raw]

    for name, values in score_channels.items():
        arr = _align_component(np.asarray(values, dtype=np.float64), n)
        key = name.lower()
        if any(token in key for token in ("statistical", "temporal", "resonance", "schumann")):
            resonance.append(arr)
        elif any(token in key for token in ("neural", "symbolic", "intelligence", "directive")):
            optimization.append(arr)
        else:
            recursion.append(arr)

    fallback = raw
    return (
        _mean_or_fallback(recursion, fallback),
        _mean_or_fallback(resonance, fallback),
        _mean_or_fallback(optimization, fallback),
    )


def _align_component(values: np.ndarray[Any, Any], n: int) -> np.ndarray[Any, Any]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == n:
        return np.clip(arr, 0.0, 1.0)
    if arr.size == 1:
        return np.full(n, float(np.clip(arr[0], 0.0, 1.0)), dtype=np.float64)
    if arr.size == 0:
        return np.zeros(n, dtype=np.float64)
    return np.resize(np.clip(arr, 0.0, 1.0), n)


def _mean_or_fallback(
    arrays: list[np.ndarray[Any, Any]], fallback: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    if not arrays:
        return cast("np.ndarray[Any, Any]", np.clip(fallback, 0.0, 1.0))
    return cast("np.ndarray[Any, Any]", np.clip(np.mean(np.vstack(arrays), axis=0), 0.0, 1.0))
