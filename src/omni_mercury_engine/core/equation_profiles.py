"""Runtime equation profiles for Mercury fusion scores.

Profiles preserve ``baseline_original_v1`` as the frozen reference while
allowing candidate profiles to be selected explicitly at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from omni_mercury_engine.core.three_r.types import GOLDEN_RATIO_CONSTANT

_EPS = 1e-8
BASELINE_PROFILE_ID = "baseline_original_v1"
QUIET_HORIZON_PROFILE_ID = "quiet_horizon_v1"


@dataclass(frozen=True)
class RuntimeEquationProfile:
    """Runtime profile metadata and scoring weights."""

    profile_id: str
    description: str
    neural_weight: float
    equation_weight: float
    ethical_exponent: float
    formula: str


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

    if profile.profile_id == BASELINE_PROFILE_ID:
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

    scored = profile.neural_weight * np.clip(raw, 0.0, 1.0) + profile.equation_weight * np.clip(
        equation_signal, 0.0, 1.0
    )
    metadata = {
        "profile_id": profile.profile_id,
        "applied": True,
        "formula": profile.formula,
        "description": profile.description,
        "component_means": {
            "recursion": float(np.mean(r)) if r.size else 0.0,
            "resonance": float(np.mean(h)) if h.size else 0.0,
            "optimization": float(np.mean(o)) if o.size else 0.0,
            "eta": float(np.mean(eta_arr)) if eta_arr.size else 0.0,
            "equation_signal": float(np.mean(equation_signal)) if equation_signal.size else 0.0,
        },
    }
    return np.clip(scored, 0.0, 1.0), metadata


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
        return np.clip(fallback, 0.0, 1.0)
    return np.clip(np.mean(np.vstack(arrays), axis=0), 0.0, 1.0)
