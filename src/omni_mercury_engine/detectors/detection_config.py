# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Configuration and non-finite (NaN/Inf) policy for the detector tier.

This module centralises three things the streaming / statistical / state-space
detector tier previously handled implicitly and inconsistently:

1. **An explicit, configurable NaN/Inf policy** (:class:`NaNPolicy`).  Every
   guard that rescues a non-finite value now does so under a *named* policy the
   operator chooses, instead of an undocumented ``np.nan_to_num`` default buried
   in each detector's input coercion.  The four policies are:

   * ``neutral`` (**default**) -- the historical conservative behaviour: a
     non-finite value is replaced with a neutral, non-anomalous constant
     (``0.0``) and ``±inf`` is clamped to ``±max_magnitude``.  Chosen as the
     default because it never *invents* anomaly signal from corrupt data and
     never aborts a streaming pipeline: a bad point scores neutral and the
     stream keeps flowing.  This is the safest behaviour for an always-on
     monitoring system where a hard failure is worse than a single dropped read.
   * ``impute`` -- replace non-finite values with the median of the finite
     values in the same vector (falling back to the neutral constant when the
     whole vector is non-finite).  More informative than ``neutral`` when a few
     reads in an otherwise-healthy window are corrupt.
   * ``flag`` -- replace as in ``neutral`` **and** return a boolean mask marking
     which positions were non-finite, so a caller can treat those points
     specially (e.g. exclude them from a metric or surface them to an operator).
   * ``raise`` -- refuse to continue: raise :class:`NonFinitePolicyError` on the
     first non-finite value.  Chosen by operators who would rather fail loudly
     than score corrupt data.

2. **A single, consistent magnitude regime.**  All caps/clamps in the tier use
   one documented ``max_magnitude`` (default :data:`DEFAULT_MAX_MAGNITUDE` =
   ``1e15``, the same ``core.centralized_constants.API.MAX_VALUE`` bound the API
   layer already enforces).  ``±inf`` maps to ``±max_magnitude`` and every
   finite value is clamped into ``[-max_magnitude, max_magnitude]`` so no
   detector can emit a value outside a known, finite envelope -- and the choice
   of bound is one number, documented in one place, not an ad-hoc literal per
   call site.

3. **Runtime configuration** of the tier's tunable knobs (NaN policy, magnitude
   cap, digital-twin ridge factor, ensemble calibration method + warm-up window)
   from **both** environment variables (``OMNI_*`` prefix, matching the
   engine/API convention) **and** an optional config file
   (``OMNI_DETECTION_CONFIG`` -> YAML/JSON), with sensible documented defaults.
   Precedence, lowest to highest: dataclass defaults < config file < environment
   variables < explicit per-detector ``config`` dict overrides.

Observability
=============
Every correction a guard makes is observable.  :func:`apply_nan_policy` and
:func:`guard_finite_scalar` increment the
``omni_detector_nonfinite_corrected{detector,policy,field}`` Prometheus counter
(see :mod:`omni_mercury_engine.core.metrics`) and emit a structured
``logger.warning`` carrying the detector name, the field that changed, the
original value type (how many NaN vs. how many Inf), and the chosen remediation.
The ``raise`` policy never increments the counter -- it aborts instead of
correcting.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np

from omni_mercury_engine.core.centralized_constants import API

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_MAX_MAGNITUDE",
    "DEFAULT_RIDGE_FACTOR",
    "DetectionConfig",
    "NaNPolicy",
    "NonFinitePolicyError",
    "apply_nan_policy",
    "guard_finite_scalar",
]

#: Single documented safe magnitude bound for the whole tier.  Reuses the API
#: layer's ``MAX_VALUE`` (``1e15``) so the detector tier and the serving layer
#: share one magnitude regime rather than each picking its own literal.
DEFAULT_MAX_MAGNITUDE: float = float(API.MAX_VALUE)

#: Default scale-relative Tikhonov ridge factor for the digital-twin solve
#: (``ridge = ridge_factor * trace(gram) / d``).  Small enough to be numerically
#: negligible on a well-conditioned Gram matrix, large enough (relative to the
#: matrix scale) to keep a near-singular solve bounded.  See
#: :mod:`omni_mercury_engine.detectors.digital_twin`.
DEFAULT_RIDGE_FACTOR: float = 1e-6

#: Environment variable holding an optional YAML/JSON config-file path.
CONFIG_PATH_ENV: str = "OMNI_DETECTION_CONFIG"


class NaNPolicy(StrEnum):
    """How the detector tier treats non-finite (NaN/Inf) values.

    A :class:`~enum.StrEnum` so a config value or env var can be either the enum
    member or its plain string name (``"neutral"``), and so it serialises
    transparently into JSON benchmark artefacts.
    """

    NEUTRAL = "neutral"
    IMPUTE = "impute"
    FLAG = "flag"
    RAISE = "raise"

    @classmethod
    def coerce(cls, value: NaNPolicy | str) -> NaNPolicy:
        """Coerce a member or (case-insensitive) string into a :class:`NaNPolicy`.

        Raises:
            ValueError: If ``value`` is not a recognised policy name.
        """
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            valid = ", ".join(p.value for p in cls)
            raise ValueError(f"unknown NaN policy {value!r}; expected one of: {valid}") from exc


class NonFinitePolicyError(ValueError):
    """Raised when :class:`NaNPolicy.RAISE` is active and a non-finite value appears.

    A :class:`ValueError` subclass so existing callers that already broaden their
    ``except`` to ``ValueError`` keep working, while callers that want to
    distinguish "operator asked us to fail closed on bad data" from other value
    errors can catch this class specifically.
    """


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------
_VALID_CALIBRATIONS: frozenset[str] = frozenset({"rank", "ecdf", "isotonic", "platt", "none"})


@dataclass(frozen=True)
class DetectionConfig:
    """Immutable runtime configuration for the detector tier.

    Attributes:
        nan_policy: How guards treat non-finite values (see :class:`NaNPolicy`).
        max_magnitude: Single safe magnitude bound; ``±inf`` maps to
            ``±max_magnitude`` and finite values are clamped into
            ``[-max_magnitude, max_magnitude]``.  Must be > 0.
        ridge_factor: Scale-relative Tikhonov ridge factor for the digital-twin
            least-squares solve. Must be >= 0.
        ensemble_calibration: Per-detector score-calibration transform applied
            before combining in :class:`~omni_mercury_engine.detectors.detection_tier.StreamingScoreEnsemble`
            (``rank`` / ``ecdf`` / ``isotonic`` / ``platt`` / ``none``).
        ensemble_warmup: Warm-up window the ensemble's per-detector calibrators
            are trained on. ``None`` (default) uses the whole training series; an
            ``int`` uses the first N points; a ``float`` in ``(0, 1]`` uses that
            fraction of the training series.
    """

    nan_policy: NaNPolicy = NaNPolicy.NEUTRAL
    max_magnitude: float = DEFAULT_MAX_MAGNITUDE
    ridge_factor: float = DEFAULT_RIDGE_FACTOR
    ensemble_calibration: str = "rank"
    ensemble_warmup: int | float | None = None

    def __post_init__(self) -> None:
        """Validate and normalise fields (frozen-safe via ``object.__setattr__``)."""
        object.__setattr__(self, "nan_policy", NaNPolicy.coerce(self.nan_policy))
        if not (self.max_magnitude > 0.0) or not np.isfinite(self.max_magnitude):
            raise ValueError(
                f"max_magnitude must be a finite positive number, got {self.max_magnitude}"
            )
        if self.ridge_factor < 0.0 or not np.isfinite(self.ridge_factor):
            raise ValueError(f"ridge_factor must be a finite number >= 0, got {self.ridge_factor}")
        calibration = str(self.ensemble_calibration).strip().lower()
        if calibration not in _VALID_CALIBRATIONS:
            valid = ", ".join(sorted(_VALID_CALIBRATIONS))
            raise ValueError(
                f"unknown ensemble_calibration {self.ensemble_calibration!r}; expected one of: {valid}"
            )
        object.__setattr__(self, "ensemble_calibration", calibration)
        warmup = self.ensemble_warmup
        if warmup is not None:
            if isinstance(warmup, bool):  # guard against bool being an int subclass
                raise ValueError("ensemble_warmup must be an int, float, or None (not bool)")
            if isinstance(warmup, float):
                if not (0.0 < warmup <= 1.0):
                    raise ValueError(
                        f"ensemble_warmup as a fraction must be in (0, 1], got {warmup}"
                    )
            elif isinstance(warmup, int):
                if warmup < 2:
                    raise ValueError(f"ensemble_warmup as a count must be >= 2, got {warmup}")
            else:
                raise ValueError(
                    f"ensemble_warmup must be int, float, or None, got {type(warmup).__name__}"
                )

    # -- construction / layering -------------------------------------------------
    @classmethod
    def from_env(cls, base: DetectionConfig | None = None) -> DetectionConfig:
        """Build a config from defaults (or ``base``) overlaid with the environment.

        Reads the ``OMNI_*`` knobs, each falling back to ``base``'s value (or the
        dataclass default) when unset, so an unset environment reproduces the
        defaults exactly.
        """
        base = base or cls()
        return cls(
            nan_policy=NaNPolicy.coerce(
                os.getenv("OMNI_DETECTOR_NAN_POLICY", base.nan_policy.value)
            ),
            max_magnitude=_env_float("OMNI_DETECTOR_MAX_MAGNITUDE", base.max_magnitude),
            ridge_factor=_env_float("OMNI_DETECTOR_RIDGE_FACTOR", base.ridge_factor),
            ensemble_calibration=os.getenv("OMNI_ENSEMBLE_CALIBRATION", base.ensemble_calibration),
            ensemble_warmup=_env_warmup("OMNI_ENSEMBLE_WARMUP", base.ensemble_warmup),
        )

    @classmethod
    def from_file(cls, path: str, base: DetectionConfig | None = None) -> DetectionConfig:
        """Load a config from a YAML/JSON file, overlaying ``base`` (or defaults).

        Only keys present in the file override; unknown keys are ignored so an
        operator can keep tier knobs alongside unrelated settings in one file.
        """
        return (base or cls()).merge(_load_config_file(path))

    @classmethod
    def resolve(
        cls,
        overrides: Mapping[str, Any] | None = None,
        *,
        config_path: str | None = None,
    ) -> DetectionConfig:
        """Resolve the effective config with full precedence layering.

        Order (lowest to highest): dataclass defaults < config file
        (``config_path`` or ``$OMNI_DETECTION_CONFIG``) < environment variables <
        ``overrides`` (typically a detector's ``config`` dict).

        Args:
            overrides: Highest-precedence per-call overrides (only the recognised
                keys ``nan_policy`` / ``max_magnitude`` / ``ridge_factor`` /
                ``ensemble_calibration`` / ``ensemble_warmup`` are consumed).
            config_path: Explicit config-file path; defaults to the
                ``OMNI_DETECTION_CONFIG`` env var when unset.

        Returns:
            The resolved, validated :class:`DetectionConfig`.
        """
        config = cls()
        path = config_path if config_path is not None else os.getenv(CONFIG_PATH_ENV)
        if path:
            config = config.merge(_load_config_file(path))
        config = cls.from_env(config)
        if overrides:
            config = config.merge(overrides)
        return config

    def merge(self, overrides: Mapping[str, Any] | None) -> DetectionConfig:
        """Return a copy with recognised keys from ``overrides`` applied."""
        if not overrides:
            return self
        recognised = {
            "nan_policy",
            "max_magnitude",
            "ridge_factor",
            "ensemble_calibration",
            "ensemble_warmup",
        }
        patch = {k: overrides[k] for k in recognised if k in overrides and overrides[k] is not None}
        return replace(self, **patch) if patch else self

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view (for benchmark artefacts / logging)."""
        return {
            "nan_policy": self.nan_policy.value,
            "max_magnitude": self.max_magnitude,
            "ridge_factor": self.ridge_factor,
            "ensemble_calibration": self.ensemble_calibration,
            "ensemble_warmup": self.ensemble_warmup,
        }


def _env_float(name: str, default: float) -> float:
    """Read a float env var, falling back to ``default`` on unset/unparseable."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("ignoring non-numeric %s=%r; using default %s", name, raw, default)
        return default


def _env_warmup(name: str, default: int | float | None) -> int | float | None:
    """Read a warm-up env var as int (count) or float (fraction), else ``default``."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    text = raw.strip()
    try:
        if "." in text or "e" in text.lower():
            return float(text)
        return int(text)
    except ValueError:
        logger.warning("ignoring non-numeric %s=%r; using default %s", name, raw, default)
        return default


def _load_config_file(path: str) -> dict[str, Any]:
    """Load a YAML or JSON config file into a dict (empty dict when absent)."""
    if not path or not os.path.exists(path):
        if path:
            logger.warning("detection config file %s not found; using defaults/env", path)
        return {}
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    if path.endswith((".yaml", ".yml")):
        import yaml

        data = yaml.safe_load(text)
    else:
        import json

        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"detection config file {path} must contain a mapping at top level")
    # Support an optional ``detection`` section so the knobs can live alongside
    # unrelated settings in a shared config file.
    section = data.get("detection")
    return dict(section) if isinstance(section, dict) else dict(data)


# ---------------------------------------------------------------------------
# Non-finite guards (the only place the tier corrects NaN/Inf)
# ---------------------------------------------------------------------------
def _record_correction(detector: str, policy: NaNPolicy, field: str, count: int) -> None:
    """Increment the Prometheus correction counter (fail-safe)."""
    if count <= 0:
        return
    try:
        from omni_mercury_engine.core.metrics import DETECTOR_NONFINITE_CORRECTED

        DETECTOR_NONFINITE_CORRECTED.labels(
            detector=detector, policy=policy.value, field=field
        ).inc(count)
    except Exception:
        logger.debug("failed to record non-finite correction metric", exc_info=True)


def apply_nan_policy(
    values: np.ndarray[Any, Any],
    *,
    policy: NaNPolicy | str = NaNPolicy.NEUTRAL,
    detector: str = "unknown",
    field: str = "scores",
    max_magnitude: float = DEFAULT_MAX_MAGNITUDE,
    neutral_value: float = 0.0,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Apply the NaN/Inf policy to a numeric array, returning ``(sanitized, flags)``.

    The single entry point through which the tier rescues non-finite values in a
    vector.  Under every non-``raise`` policy the returned array is guaranteed
    finite and clamped into ``[-max_magnitude, max_magnitude]`` (the unified
    magnitude regime).  ``flags`` is a boolean array, ``True`` where the input
    position was non-finite -- always returned (not just for ``flag``) so callers
    can inspect it uniformly, though only the ``flag`` policy is *expected* to act
    on it.

    A correction (any non-finite value replaced) increments the
    ``omni_detector_nonfinite_corrected`` counter and emits a structured warning.

    Args:
        values: Input array (any shape/dtype coercible to float64).
        policy: The active :class:`NaNPolicy` (or its string name).
        detector: Detector name for the metric label / log.
        field: What is being guarded (``"scores"``, ``"input"``, ``"gram"`` ...).
        max_magnitude: Single magnitude bound (see module docstring).
        neutral_value: Replacement for NaN under ``neutral`` / ``flag`` (and the
            fallback for ``impute`` when the whole vector is non-finite).

    Returns:
        ``(sanitized, flags)`` -- a finite, clamped float64 array and the
        non-finite mask.

    Raises:
        NonFinitePolicyError: If ``policy`` is ``raise`` and any value is
            non-finite.
    """
    policy = NaNPolicy.coerce(policy)
    arr = np.asarray(values, dtype=np.float64)
    finite_mask = np.isfinite(arr)
    nonfinite = ~finite_mask
    n_bad = int(nonfinite.sum())

    if n_bad == 0:
        # No NaN/Inf, but still enforce the single magnitude regime so an
        # in-range-but-huge finite value cannot escape the envelope.
        return np.clip(arr, -max_magnitude, max_magnitude), nonfinite

    if policy is NaNPolicy.RAISE:
        n_nan = int(np.isnan(arr).sum())
        n_inf = n_bad - n_nan
        raise NonFinitePolicyError(
            f"{detector}: {n_bad} non-finite value(s) in field {field!r} "
            f"({n_nan} NaN, {n_inf} Inf) and NaN policy is 'raise'"
        )

    sanitized = np.array(arr, dtype=np.float64, copy=True)
    if policy is NaNPolicy.IMPUTE:
        finite_vals = arr[finite_mask]
        fill = float(np.median(finite_vals)) if finite_vals.size else float(neutral_value)
        # NaN -> imputed fill; ±inf -> ±max_magnitude (direction preserved).
        pos_inf = np.isposinf(arr)
        neg_inf = np.isneginf(arr)
        is_nan = np.isnan(arr)
        sanitized[is_nan] = fill
        sanitized[pos_inf] = max_magnitude
        sanitized[neg_inf] = -max_magnitude
    else:
        # NEUTRAL / FLAG: NaN -> neutral constant; ±inf -> ±max_magnitude.
        sanitized = np.nan_to_num(
            arr, nan=float(neutral_value), posinf=max_magnitude, neginf=-max_magnitude
        )

    sanitized = np.clip(sanitized, -max_magnitude, max_magnitude)

    n_nan = int(np.isnan(arr).sum())
    n_inf = n_bad - n_nan
    _record_correction(detector, policy, field, n_bad)
    logger.warning(
        "non-finite values rescued in detector tier",
        extra={
            "detector": detector,
            "field": field,
            "policy": policy.value,
            "n_corrected": n_bad,
            "n_nan": n_nan,
            "n_inf": n_inf,
            "remediation": _remediation(policy),
        },
    )
    return sanitized, nonfinite


def guard_finite_scalar(
    value: float,
    *,
    policy: NaNPolicy | str = NaNPolicy.NEUTRAL,
    detector: str = "unknown",
    field: str = "metadata",
    max_magnitude: float = DEFAULT_MAX_MAGNITUDE,
    neutral_value: float = 0.0,
) -> float:
    """Guard a single scalar metadata field (e.g. ``z_q`` / ``gamma``) for finiteness.

    Applies the same policy and magnitude regime as :func:`apply_nan_policy` but
    for one scalar, so metadata carries the same finite guarantees as scores.
    Returns a finite float clamped into ``[-max_magnitude, max_magnitude]``.

    Raises:
        NonFinitePolicyError: If ``policy`` is ``raise`` and ``value`` is
            non-finite.
    """
    policy = NaNPolicy.coerce(policy)
    scalar = float(value)
    if np.isfinite(scalar):
        return float(np.clip(scalar, -max_magnitude, max_magnitude))

    if policy is NaNPolicy.RAISE:
        kind = "NaN" if np.isnan(scalar) else "Inf"
        raise NonFinitePolicyError(
            f"{detector}: non-finite metadata field {field!r} ({kind}) and NaN policy is 'raise'"
        )

    if np.isposinf(scalar):
        result = max_magnitude
    elif np.isneginf(scalar):
        result = -max_magnitude
    else:  # NaN
        result = float(neutral_value)
    _record_correction(detector, policy, field, 1)
    logger.warning(
        "non-finite metadata rescued in detector tier",
        extra={
            "detector": detector,
            "field": field,
            "policy": policy.value,
            "n_corrected": 1,
            "n_nan": int(np.isnan(scalar)),
            "n_inf": int(np.isinf(scalar)),
            "remediation": _remediation(policy),
        },
    )
    return float(np.clip(result, -max_magnitude, max_magnitude))


def _remediation(policy: NaNPolicy) -> str:
    """Human-readable remediation string for the structured log."""
    return {
        NaNPolicy.NEUTRAL: "replaced with neutral 0.0 / clamped inf to +-max_magnitude",
        NaNPolicy.IMPUTE: "imputed NaN with finite median / clamped inf to +-max_magnitude",
        NaNPolicy.FLAG: "replaced with neutral 0.0 and flagged position via returned mask",
    }.get(policy, "no correction")
