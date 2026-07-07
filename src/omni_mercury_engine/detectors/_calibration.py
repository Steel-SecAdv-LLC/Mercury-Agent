# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
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

from typing import Any

import numpy as np

__all__ = [
    "FINITE_CAP",
    "LN2",
    "bound_finite",
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
FINITE_CAP = 1e100


def bound_finite(arr: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Map NaN→0 and ``±inf``/out-of-range magnitudes to ``±FINITE_CAP``.

    This is the drop-in, overflow-safe replacement for the
    ``np.nan_to_num(np.asarray(data, dtype=np.float64))`` expression the tier
    detectors used for input sanitisation. A bare ``np.nan_to_num`` maps ``±inf``
    to ``±1.7977e308``; that near-max-float value overflows to ``inf`` the moment
    it is squared or summed inside a detector, and the subsequent ``inf - inf``
    produces ``NaN`` — which then corrupts the anomaly score or, when it happens
    at ``fit`` time, permanently poisons the detector's calibration statistics.
    Bounding into ``[-FINITE_CAP, FINITE_CAP]`` (see :data:`FINITE_CAP`) makes
    that overflow impossible while leaving all realistic, in-range data unchanged
    (it is applied to the same array the old expression produced, so the coercion
    to ``float64`` and any surrounding ``.ravel()`` / reshape stay in place).

    Args:
        arr: An already-``np.asarray``'d numeric array.

    Returns:
        The same-shaped array with every element finite and ``|x| <= FINITE_CAP``.
    """
    arr = np.nan_to_num(arr, nan=0.0, posinf=FINITE_CAP, neginf=-FINITE_CAP)
    return np.clip(arr, -FINITE_CAP, FINITE_CAP)


def finite_features(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Coerce a fusion-feature array to guaranteed-finite ``float32``.

    ``extract_features`` returns *raw* (un-squashed) features cast to ``float32``
    for the fusion network. A large-but-finite float64 feature (e.g. a residual
    amplified by a floored ``1e-9`` denominator) can exceed ``float32``'s
    ``~3.4e38`` range and become ``inf`` on the cast, poisoning the feature store
    and the fusion input. This maps every non-finite entry to the finite float32
    bound so the feature vector is always usable, then returns float32.
    """
    f32_max = float(np.finfo(np.float32).max)
    arr = np.asarray(values, dtype=np.float64)
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
) -> np.ndarray[Any, Any]:
    """Coerce a score vector to guaranteed-finite values in ``[lo, hi]``.

    This is the single choke point that enforces the ``BaseDetector`` contract's
    "all scores must be finite and in ``[0, 1]``" invariant. Unlike a bare
    ``np.clip`` — which passes ``NaN`` straight through — this maps every
    non-finite entry to a defined bound before clipping:

    * ``NaN`` → ``lo`` (an undefined score is treated as *non*-anomalous, the
      conservative choice that never fabricates an alert);
    * ``+inf`` → ``hi`` (a saturating positive score is maximally anomalous);
    * ``-inf`` → ``lo``.

    Args:
        values: Raw score array (any shape / dtype).
        lo: Lower bound (default ``0.0``).
        hi: Upper bound (default ``1.0``).

    Returns:
        A float64 array of the same shape, every element finite and in
        ``[lo, hi]``.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=lo, posinf=hi, neginf=lo)
    return np.clip(arr, lo, hi)
