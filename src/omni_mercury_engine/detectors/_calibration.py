# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared, edge-case-hardened calibration helpers for the streaming detector tier.

Every detector in the streaming / statistical / state-space tier squashes a raw
non-negative anomaly signal ``r`` into a ``[0, 1]`` score with the same monotone
map ``1 - exp(-r / scale)``, where ``scale`` anchors a high training quantile at
the 0.5 boundary for a controlled false-positive rate. Before this module the two
pieces of that contract — computing the squash scale and finalising the score
vector — were copy-pasted, byte-for-byte, into fourteen detectors. That
duplication also copied two latent robustness defects into all of them:

1. **Empty-input crash.** ``float(np.quantile(raw, q))`` raises ``IndexError`` on
   a zero-length array, so ``fit([])`` (and an *unfitted* ``detect([])``) crashed
   instead of degrading gracefully the way the rest of the tier does.
2. **Non-finite score escape.** The input coercion sanitises NaN via
   ``np.nan_to_num``, but that maps ``±inf`` to ``±1.8e308`` — a value that
   overflows in the detectors' downstream FFTs, matmuls, cumulative sums and
   delay embeddings, re-introducing ``inf``/``NaN``. The final ``np.clip(s, 0, 1)``
   does **not** remove ``NaN`` (``np.clip(np.nan, 0, 1) is nan``), so a single
   non-finite input sample could make a detector emit ``NaN`` anomaly scores that
   then poison the ensemble mean, the stacking meta-learner, the Prometheus
   score histogram, and the alerting path.

Centralising the two operations here fixes both at the root, once, with dedicated
tests (``tests/detectors/test_calibration_helpers.py``), and keeps every
detector's behaviour on *ordinary finite* input byte-identical to before (the
non-empty, all-finite path is unchanged). The module is pure NumPy, so the tier
stays always-importable with no optional dependency.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from omni_mercury_engine.detectors.detection_config import (
    DEFAULT_MAX_MAGNITUDE,
    NaNPolicy,
    NonFinitePolicyError,
    active_max_magnitude,
    active_nan_policy,
    record_nonfinite_correction,
)

logger = logging.getLogger(__name__)

__all__ = [
    "FINITE_CAP",
    "LN2",
    "bound_finite",
    "bound_finite_config",
    "finite_features",
    "finite_scores",
    "squash_scale",
]

#: ``1 - exp(-s / scale) = 0.5`` at ``s = scale * ln 2``; anchoring ``scale`` to a
#: high training quantile places the 0.5 anomaly boundary at that quantile for a
#: controlled ``1 - calibration_quantile`` false-positive rate.
LN2 = float(np.log(2.0))

#: Symmetric magnitude cap applied to sanitised detector input. It exists to
#: neutralise the ``np.nan_to_num`` sentinel (``±inf`` → ``±1.7977e308``): that
#: near-max-float value overflows to ``inf`` the moment it is squared or summed
#: inside a detector (FFT power, covariance, Gram matrix, cumulative sum), and a
#: subsequent ``inf - inf`` yields ``NaN`` — the exact chain that let a single
#: non-finite input sample corrupt scores or, worse, permanently poison a
#: detector's ``fit``-time statistics. ``1e100`` is chosen so that (a) it is
#: astronomically larger than any physical/operational signal Mercury ingests
#: (nanosecond-epoch timestamps are ~1.7e18, financial magnitudes ~1e12), so
#: realistic data is *never* clipped and behaviour is unchanged; and (b) its
#: square (``1e200``) stays comfortably within float64 range, so the arithmetic
#: that overflowed at ``1.8e308`` no longer can. ``finite_scores`` /
#: ``finite_features`` remain the guaranteed output backstop for any residual
#: higher-order overflow.
#:
#: Unified with the NaN-policy layer: this *is*
#: ``detection_config.DEFAULT_MAX_MAGNITUDE`` (same ``1e100`` literal, defined in
#: one place), so ``bound_finite`` and ``apply_nan_policy`` share one magnitude
#: regime. It is the *default* cap: :func:`bound_finite` resolves the effective cap
#: from ``OMNI_DETECTOR_MAX_MAGNITUDE`` (via
#: :func:`~omni_mercury_engine.detectors.detection_config.active_max_magnitude`) and
#: falls back to this constant, so an operator can tighten input sanitisation
#: without editing code.
FINITE_CAP = DEFAULT_MAX_MAGNITUDE


def _count_nonfinite(arr: np.ndarray[Any, Any]) -> tuple[int, int]:
    """Return ``(n_nan, n_inf)`` in ``arr`` (a float array)."""
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    return n_nan, n_inf


def bound_finite(
    arr: np.ndarray[Any, Any],
    *,
    detector: str = "tier",
    policy: NaNPolicy | str | None = None,
    max_magnitude: float | None = None,
) -> np.ndarray[Any, Any]:
    """Map NaN→0 and ``±inf``/out-of-range magnitudes to ``±max_magnitude`` (observably).

    The overflow-safe replacement for ``np.nan_to_num`` the tier detectors use for
    input sanitisation. A bare ``np.nan_to_num`` maps ``±inf`` to ``±1.7977e308``;
    that near-max-float value overflows to ``inf`` the moment it is squared or
    summed inside a detector, and the subsequent ``inf - inf`` produces ``NaN`` —
    corrupting the score or, at ``fit`` time, permanently poisoning the detector's
    calibration statistics. Bounding into ``[-max_magnitude, max_magnitude]`` makes
    that impossible while leaving realistic, in-range data unchanged.

    This is the tier's configurable, *observable* NaN/Inf guard: every correction
    it makes is metered on ``omni_detector_nonfinite_corrected`` and logged (see
    :func:`~omni_mercury_engine.detectors.detection_config.record_nonfinite_correction`),
    and the active policy (``OMNI_DETECTOR_NAN_POLICY``, or the explicit ``policy``
    argument) selects the behaviour: ``neutral`` (default, the historical mapping),
    ``impute``, ``flag``, or ``raise`` (fail closed on any non-finite value). On
    finite input within the cap it is a no-op with no metric/log emitted, so
    behaviour on ordinary data is unchanged.

    The magnitude cap is **configurable** and honours the same knob the rest of the
    tier documents: when ``max_magnitude`` is ``None`` (the usual detector call) it
    resolves from ``OMNI_DETECTOR_MAX_MAGNITUDE`` via
    :func:`~omni_mercury_engine.detectors.detection_config.active_max_magnitude`
    (default :data:`FINITE_CAP`), so lowering that env var actually tightens input
    sanitisation instead of being silently ignored. A caller holding a resolved
    :class:`~omni_mercury_engine.detectors.detection_config.DetectionConfig` (e.g.
    one built from a config file or a per-detector ``config`` dict) may pass
    ``max_magnitude=cfg.max_magnitude`` to apply that layered value directly.

    Args:
        arr: An already-``np.asarray``'d numeric array.
        detector: Detector name for the metric ``detector`` label / log.
        policy: Override the active NaN policy for this call (``None`` = resolve
            from the environment).
        max_magnitude: Explicit magnitude cap for this call (``None`` = resolve
            from ``OMNI_DETECTOR_MAX_MAGNITUDE`` / :data:`FINITE_CAP`).

    Returns:
        The same-shaped array with every element finite and ``|x| <= max_magnitude``.

    Raises:
        NonFinitePolicyError: If the active policy is ``raise`` and ``arr`` has a
            non-finite value.
    """
    from omni_mercury_engine.detectors.detection_config import apply_nan_policy

    resolved = policy if policy is not None else active_nan_policy()
    cap = active_max_magnitude() if max_magnitude is None else float(max_magnitude)
    sanitized, _ = apply_nan_policy(
        arr,
        policy=resolved,
        detector=detector,
        field="input",
        max_magnitude=cap,
    )
    return sanitized


def bound_finite_config(
    detector_obj: Any,
    arr: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """``bound_finite`` honouring a detector's fully-resolved ``DetectionConfig``.

    The input-sanitisation counterpart that makes the documented precedence
    (``defaults < config file < env < per-detector config`` dict) hold *tier-wide*,
    not only for the detectors that separately resolve a ``DetectionConfig``. It
    resolves the detector's config from its ``_config`` dict once and caches it on
    ``detector_obj._detection_config`` (reusing one an ``__init__`` already set, e.g.
    SPOT / digital-twin), then threads that config's ``nan_policy`` / ``max_magnitude``
    into :func:`bound_finite`.

    Behaviour is byte-identical to the plain env-only guard whenever the detector's
    ``config`` sets no ``nan_policy`` / ``max_magnitude`` (the resolved values then
    equal the env defaults), so ordinary detectors and the committed benchmarks are
    unaffected; only a detector explicitly given those overrides now applies them on
    its input path. Resolution failure falls back to the env-only guard so this can
    never make a detector unusable.

    Args:
        detector_obj: The detector instance (uses its ``.name`` and ``._config``).
        arr: An already-``np.asarray``'d numeric array.

    Returns:
        The finite, bounded array (see :func:`bound_finite`).
    """
    name = getattr(detector_obj, "name", "tier")
    cfg = getattr(detector_obj, "_detection_config", None)
    if cfg is None:
        try:
            from omni_mercury_engine.detectors.detection_config import DetectionConfig

            cfg = DetectionConfig.resolve(getattr(detector_obj, "_config", None))
            detector_obj._detection_config = cfg
        except Exception as exc:
            # Resolution failed (e.g. a typo in an unrelated config key). Do not
            # silently revert to the permissive env default: if the operator
            # explicitly set a fail-closed nan_policy, dropping it would void
            # their contract on the data-sanitisation path with zero signal.
            raw_cfg = getattr(detector_obj, "_config", None)
            requested_policy = raw_cfg.get("nan_policy") if isinstance(raw_cfg, dict) else None
            logger.warning(
                "Detection config resolution failed for detector %s (%s); "
                "falling back to the env-only non-finite guard%s.",
                name,
                exc,
                (
                    f" while honouring the explicitly configured nan_policy={requested_policy!r}"
                    if requested_policy
                    else ""
                ),
                exc_info=True,
            )
            if requested_policy is not None:
                # Honour the operator's explicit policy directly; an invalid
                # value fails loudly in NaNPolicy.coerce rather than degrading.
                return bound_finite(arr, detector=name, policy=requested_policy)
            return bound_finite(arr, detector=name)
    return bound_finite(arr, detector=name, policy=cfg.nan_policy, max_magnitude=cfg.max_magnitude)


def finite_features(
    values: np.ndarray[Any, Any],
    *,
    detector: str = "tier",
    policy: NaNPolicy | str | None = None,
) -> np.ndarray[Any, Any]:
    """Coerce a fusion-feature array to guaranteed-finite ``float32`` (observably).

    ``extract_features`` returns *raw* (un-squashed) features cast to ``float32``
    for the fusion network. A large-but-finite float64 feature (e.g. a residual
    amplified by a floored ``1e-9`` denominator) can exceed ``float32``'s
    ``~3.4e38`` range and become ``inf`` on the cast, poisoning the feature store
    and the fusion input. This maps every non-finite entry to the finite float32
    bound so the feature vector is always usable, then returns float32.

    Like :func:`bound_finite`, every correction is metered + logged, and a
    ``raise`` policy fails closed instead of correcting.
    """
    resolved = NaNPolicy.coerce(policy if policy is not None else active_nan_policy())
    f32_max = float(np.finfo(np.float32).max)
    arr = np.asarray(values, dtype=np.float64)
    n_nan, n_inf = _count_nonfinite(arr)
    if n_nan or n_inf:
        if resolved is NaNPolicy.RAISE:
            raise NonFinitePolicyError(
                f"{detector}: {n_nan + n_inf} non-finite value(s) in features "
                f"({n_nan} NaN, {n_inf} Inf) and NaN policy is 'raise'"
            )
        record_nonfinite_correction(detector, resolved, "features", n_nan=n_nan, n_inf=n_inf)
    arr = np.nan_to_num(arr, nan=0.0, posinf=f32_max, neginf=-f32_max)
    return np.clip(arr, -f32_max, f32_max).astype(np.float32)


def squash_scale(raw: np.ndarray[Any, Any], calibration_quantile: float) -> float:
    """Squash scale anchoring ``calibration_quantile`` of ``raw`` at score 0.5.

    Returns the ``scale`` such that ``1 - exp(-r / scale) = 0.5`` when ``r`` equals
    the ``calibration_quantile`` of the (finite) training signal ``raw`` — i.e.
    ``scale = quantile / ln 2``. When that quantile is ~0 (a degenerate,
    near-constant training signal) the mean is used as a fallback location, and
    the result is floored at ``1e-9`` so the later division is always safe.

    Hardening over the historical inline copies (behaviour on ordinary,
    non-empty, all-finite input is unchanged):

    * An **empty** ``raw`` returns ``1.0`` instead of letting ``np.quantile``
      raise ``IndexError`` — so ``fit([])`` / unfitted ``detect([])`` degrade
      gracefully like the rest of the tier.
    * Non-finite entries in ``raw`` (which ``np.nan_to_num`` can turn ``±inf``
      into as ``±1.8e308`` and downstream overflow can turn into ``NaN``) are
      dropped before the quantile, so a poisoned calibration sample cannot make
      the scale ``inf``/``NaN`` and silently flatten every score to 0.

    Args:
        raw: The non-negative raw anomaly signal on the training data.
        calibration_quantile: Quantile in ``(0, 1)`` placed at the 0.5 boundary.

    Returns:
        A strictly positive, finite squash scale (>= ``1e-9``).
    """
    arr = np.asarray(raw, dtype=np.float64).ravel()
    if arr.size:
        arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    q = float(np.quantile(arr, calibration_quantile))
    if q < 1e-9:
        q = float(np.mean(arr)) + 1e-9
    return max(q / LN2, 1e-9)


def finite_scores(
    values: np.ndarray[Any, Any],
    *,
    lo: float = 0.0,
    hi: float = 1.0,
    detector: str = "tier",
    policy: NaNPolicy | str | None = None,
) -> np.ndarray[Any, Any]:
    """Coerce a score vector to guaranteed-finite values in ``[lo, hi]`` (observably).

    This is the single choke point that enforces the ``BaseDetector`` contract's
    "all scores must be finite and in ``[0, 1]``" invariant. Unlike a bare
    ``np.clip`` — which passes ``NaN`` straight through — this maps every
    non-finite entry to a defined bound before clipping:

    * ``NaN`` → ``lo`` (an undefined score is treated as *non*-anomalous, the
      conservative choice that never fabricates an alert);
    * ``+inf`` → ``hi`` (a saturating positive score is maximally anomalous);
    * ``-inf`` → ``lo``.

    Like :func:`bound_finite`, every correction is metered + logged, and a
    ``raise`` policy fails closed instead of correcting. The ``lo``/``hi`` mapping
    is score-specific (asymmetric), so this is *not* the symmetric
    ``apply_nan_policy`` clamp — the two are consistent on the ``neutral`` /
    ``raise`` intent while keeping the score contract's ``[lo, hi]`` semantics.

    Args:
        values: Raw score array (any shape / dtype).
        lo: Lower bound (default ``0.0``).
        hi: Upper bound (default ``1.0``).
        detector: Detector name for the metric ``detector`` label / log.
        policy: Override the active NaN policy for this call (``None`` = resolve
            from the environment).

    Returns:
        A float64 array of the same shape, every element finite and in
        ``[lo, hi]``.

    Raises:
        NonFinitePolicyError: If the active policy is ``raise`` and ``values`` has
            a non-finite entry.
    """
    resolved = NaNPolicy.coerce(policy if policy is not None else active_nan_policy())
    arr = np.asarray(values, dtype=np.float64)
    n_nan, n_inf = _count_nonfinite(arr)
    if n_nan or n_inf:
        if resolved is NaNPolicy.RAISE:
            raise NonFinitePolicyError(
                f"{detector}: {n_nan + n_inf} non-finite score(s) "
                f"({n_nan} NaN, {n_inf} Inf) and NaN policy is 'raise'"
            )
        record_nonfinite_correction(detector, resolved, "scores", n_nan=n_nan, n_inf=n_inf)
    arr = np.nan_to_num(arr, nan=lo, posinf=hi, neginf=lo)
    return np.clip(arr, lo, hi)


#: Training quantile of the PCA reconstruction residuals that sets the
#: micro-anomaly cut. The rate of residuals above it is the micro score.
MICRO_RESIDUAL_QUANTILE = 95.0


def fit_pca_basis(
    data_2d: np.ndarray[Any, Any], target_dim: int
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Mean and the top ``target_dim`` principal axes of the training rows.

    The basis is fitted once, on training data, and then applied unchanged to
    every batch. Refitting it on the scored batch lets the batch's own axes
    absorb whatever the batch contains, so its residuals cannot show a departure
    from the normal regime.
    """
    mean = data_2d.mean(axis=0)
    _, _, vt = np.linalg.svd(data_2d - mean, full_matrices=False)
    return mean, vt[:target_dim]


def pca_residuals(
    data_2d: np.ndarray[Any, Any],
    basis: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]],
) -> tuple[np.ndarray[Any, Any], float]:
    """Absolute residuals of ``data_2d`` against a fitted basis.

    Returns the residual matrix (the shape of ``data_2d``) and the batch's
    residual energy share: the fraction of its centred energy the training
    subspace does not reproduce, in ``[0, 1]``.
    """
    mean, components = basis
    centred = data_2d - mean
    residual = centred - (centred @ components.T) @ components
    total = float(np.sum(centred**2))
    energy_share = float(np.sum(residual**2)) / total if total > 0.0 else 0.0
    return np.abs(residual), energy_share


def micro_residual_cut(
    data_2d: np.ndarray[Any, Any],
    basis: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]],
) -> float:
    """The micro-anomaly cut: the training residuals' :data:`MICRO_RESIDUAL_QUANTILE`.

    ``data_2d`` must be the rows the ``basis`` was fitted on. A cut taken from
    the scored batch's own residuals makes the rate about 0.05 for any input.
    """
    residuals, _ = pca_residuals(data_2d, basis)
    return float(np.percentile(residuals.flatten(), MICRO_RESIDUAL_QUANTILE))


# ---------------------------------------------------------------------------
# Conformal evidence against a fit-time reference
# ---------------------------------------------------------------------------


def conformal_cut(reference_sorted: np.ndarray[Any, Any] | None, alpha: float) -> float | None:
    """One-class split-conformal cut: the ``ceil((n + 1)(1 - alpha))``-th smallest score.

    Under exchangeability of a new normal row with the ``n`` reference rows it
    exceeds the cut with probability at most ``alpha`` (Vovk, Gammerman &
    Shafer 2005). ``None`` for an empty or missing reference.
    """
    if reference_sorted is None or len(reference_sorted) == 0:
        return None
    n = len(reference_sorted)
    k = min(max(int(np.ceil((n + 1) * (1.0 - alpha))), 1), n)
    return float(reference_sorted[k - 1])


#: Share of the reference used as the peaks-over-threshold tail, and the
#: fewest excesses a generalised-Pareto fit is attempted on.
_TAIL_FRACTION = 0.10
_TAIL_MIN_EXCESSES = 20


def _gpd_moments(excesses: np.ndarray[Any, Any]) -> tuple[float, float]:
    """Generalised-Pareto ``(shape, scale)`` by the method of moments.

    For excesses with mean ``m`` and variance ``v``: shape ``0.5 * (1 - m^2 /
    v)`` and scale ``0.5 * m * (m^2 / v + 1)`` (Hosking & Wallis 1987), the
    estimator ``spot_evt`` uses. An exponential tail when the variance is
    degenerate.
    """
    m = float(np.mean(excesses))
    v = float(np.var(excesses))
    if v < 1e-18 or m <= 0.0:
        return 0.0, max(m, 1e-12)
    ratio = m * m / v
    return 0.5 * (1.0 - ratio), max(0.5 * m * (ratio + 1.0), 1e-12)


def conformal_p_values(
    reference_sorted: np.ndarray[Any, Any], scores: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    """``(1 + #{reference >= s}) / (n + 1)`` per score, with a fitted tail past the reference.

    Within the reference's range this is the conformal p-value, super-uniform
    under exchangeability with the reference rows (``P(p <= a) <= a``).
    Against one shared reference a batch's p-values are PRDS, so
    Benjamini-Hochberg over them controls the false discovery rate (Bates,
    Candes, Lei, Romano & Sesia 2023, *Ann. Statist.* 51(1)).

    A score above every reference score gets the floor ``1 / (n + 1)`` from
    the rank alone, which caps any batch-level test: Benjamini-Hochberg over
    ``m`` rows at level ``q`` needs ``p <= q / m``, beyond a 1,000-row
    reference's reach at ``m = 100``, ``q = 0.01``. Past the reference
    maximum the p-value is therefore the generalised-Pareto tail fitted to the
    top ``_TAIL_FRACTION`` of the reference (peaks over threshold;
    Pickands-Balkema-de Haan; Siffer et al. 2017, KDD) with its shape held at
    or above 0, never above the floor.
    That extrapolation is asymptotic rather than finite-sample. Without
    ``_TAIL_MIN_EXCESSES`` excesses the floor stands. Non-finite scores get
    p = 1 (no evidence).
    """
    ref = np.asarray(reference_sorted, dtype=np.float64).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    n = ref.size
    finite = np.isfinite(s)
    at_least = n - np.searchsorted(ref, np.where(finite, s, -np.inf), side="left")
    p = (1.0 + at_least) / (n + 1.0)
    n_tail = int(np.ceil(_TAIL_FRACTION * n))
    beyond = finite & (s > ref[-1]) if n else np.zeros(s.shape, dtype=bool)
    if n_tail >= _TAIL_MIN_EXCESSES and n_tail < n and bool(beyond.any()):
        threshold = float(ref[n - n_tail - 1])
        excesses = ref[n - n_tail :] - threshold
        if float(excesses.max()) > 0.0:
            shape, scale = _gpd_moments(excesses)
            # No finite endpoint: these scores are bounded or saturating, so
            # the moment fit returns a negative shape whose endpoint sits just
            # past the reference maximum, and a normal row landing beyond it
            # got p = 0 (0.17 of all-normal 100-row batches called anomalous
            # by the engine at a 0.05 budget). An exponential-or-heavier tail
            # is conservative whenever the true tail is lighter.
            shape = max(shape, 0.0)
            z = (s[beyond] - threshold) / scale
            if abs(shape) < 1e-8:
                survival = np.exp(-z)
            else:
                inner = 1.0 + shape * z
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    survival = np.where(inner > 0.0, inner ** (-1.0 / shape), 0.0)
            tail_p = (n_tail / (n + 1.0)) * np.nan_to_num(survival, nan=0.0)
            p[beyond] = np.minimum(1.0 / (n + 1.0), tail_p)
    return np.asarray(np.where(finite, p, 1.0), dtype=np.float64)


def benjamini_hochberg(p_values: np.ndarray[Any, Any], level: float) -> np.ndarray[Any, Any]:
    """Benjamini-Hochberg (1995) step-up rejections at false discovery rate ``level``.

    Under the global null (every row normal) with PRDS p-values the
    probability of rejecting anything is at most ``level``.
    """
    p = np.asarray(p_values, dtype=np.float64).reshape(-1)
    m = p.size
    flags = np.zeros(m, dtype=bool)
    if m == 0:
        return flags
    order = np.argsort(p, kind="stable")
    passed = p[order] <= level * np.arange(1, m + 1) / m
    if passed.any():
        flags[order[: int(np.flatnonzero(passed)[-1]) + 1]] = True
    return flags


def batch_evidence(
    reference_sorted: np.ndarray[Any, Any] | None,
    scores: np.ndarray[Any, Any],
    level: float = 0.05,
) -> dict[str, Any]:
    """``p_values`` / ``fdr_flags`` / ``fdr_level`` / ``batch_anomaly`` for a result.

    Empty when no reference exists (a legacy state), so callers keep their
    per-row verdict only. ``batch_anomaly`` is the batch-level test: OR-ing
    per-row flags at level alpha over ``n`` normal rows fires with
    probability ``1 - (1 - alpha)^n``; this fires with probability at most
    ``level``.
    """
    if reference_sorted is None or len(reference_sorted) == 0:
        return {}
    p = conformal_p_values(reference_sorted, scores)
    flags = benjamini_hochberg(p, level)
    n = len(reference_sorted)
    return {
        "p_values": p,
        "fdr_flags": flags,
        "fdr_level": float(level),
        "batch_anomaly": bool(flags.any()),
        # The smallest p-value a row beyond the reference can receive: 0 once
        # a tail is fitted (it decays without bound), else the rank floor.
        "p_value_floor": (
            0.0 if int(np.ceil(_TAIL_FRACTION * n)) >= _TAIL_MIN_EXCESSES else 1.0 / (n + 1.0)
        ),
    }
