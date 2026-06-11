# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exact incremental single-sample serving for :class:`DetectorAgent`.

The agent's single-sample contract is *score the sample as the last row of
``vstack([reference, row])``* — the detectors normalize within a batch, so a
lone sample needs company to be scored meaningfully. The reference batch is
fixed at fit time, yet the full path re-scores all of it through every
detector protocol on every call (e.g. the directive detector's O(n²·d)
stability term), which a 2026-06-11 profile measured at ~54 ms per served
sample across the five agents — dominated by directive (28 ms), spatial
(11 ms) and temporal (10 ms) per-batch recomputation.

This module is the *compiled form* of that reference semantics for the three
profile-dominant detectors: every cache below reproduces, bit-for-bit, the
score the full batch path assigns to the appended row, by separating

* row-independent terms (cached once per reference: spatial raw
  distance/LOF scores, the directive reference-magnitude maximum, the
  recursive-memory tail), from
* terms the appended row genuinely participates in (batch-global statistics,
  the directive batch-level blend scalars), which are recomputed each call
  with the *same operations on the same arrays* as the reference path.

Honesty contract (anti-theater):

* **Never an approximation.** Every serve either returns the bit-identical
  score or ``None``, in which case the caller falls back to the verbatim
  full-batch path. Unknown detector subclasses, auto-calibration, non-finite
  inputs, and re-fit detectors (stale caches) all fall back — fail-closed to
  the slow exact path.
* **Exact-type dispatch.** A subclass may override scoring internals, so the
  fast path engages only for the exact detector classes whose reference
  semantics are pinned by ``tests/test_native_acceleration.py``.
* **Purity prerequisite.** Bit-equivalence is defined against the agent's
  pure batch contract (transient detector state reset per scoring call, see
  :meth:`DetectorAgent.score_batch`); the caches reproduce the
  reset-then-score semantics, not any cross-call memory.

Equivalence is pinned by ``tests/test_native_acceleration.py``
(``TestIncrementalServingParity``): incremental vs full-path scores are
asserted bit-identical across datasets, seeds, and detector configurations.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["build_serving_cache"]


def _finite_2d(array: np.ndarray[Any, Any]) -> bool:
    """Whether ``array`` is a non-empty 2-D float array of finite values."""
    return (
        isinstance(array, np.ndarray)
        and array.ndim == 2
        and array.shape[0] > 0
        and array.shape[1] > 0
        and bool(np.all(np.isfinite(array)))
    )


def _admissible_score(score: float) -> float | None:
    """Runtime equivalence guard for the agent's score contract.

    The supported detectors emit clipped / soft-normalized scores, so the
    agent's batch range-normalization is the identity for them. A score
    outside [0, 1] (or non-finite) would mean that assumption broke —
    fall back to the full path rather than serve a wrong-contract value.
    """
    if np.isfinite(score) and 0.0 <= score <= 1.0:
        return float(score)
    return None


class _ServingCache:
    """Base class: serve one row against the fixed reference, or refuse."""

    def __init__(self, detector: Any, reference: np.ndarray[Any, Any]) -> None:
        self._detector = detector
        self._reference = reference
        self._lock = threading.Lock()

    def serve(self, row: np.ndarray[Any, Any]) -> float | None:
        """Score ``row`` as the appended last row, or ``None`` to fall back."""
        if row.ndim != 1 or row.shape[0] != self._reference.shape[1]:
            return None
        if not np.all(np.isfinite(row)):
            return None
        with self._lock:
            if not self._still_valid():
                return None
            score = self._serve_locked(row.astype(np.float64, copy=False))
        return None if score is None else _admissible_score(score)

    def _still_valid(self) -> bool:
        """Whether the detector's fitted state matches what was cached."""
        raise NotImplementedError

    def _serve_locked(self, row: np.ndarray[Any, Any]) -> float | None:
        raise NotImplementedError


class _DirectiveServingCache(_ServingCache):
    """Incremental form of ``SigmaDirectiveDetector.detect`` for the last row.

    Reference semantics per appended row ``x`` over batch ``B = [R; x]``:

    * PCP / GSIS / RMD / EOA protocol scores for ``x``'s row only — the
      combined score has no cross-row coupling (scores are clipped to [0, 1]
      before return, so the agent's batch renormalization never engages).
    * GSIS is the O(n²·d) term: ``x``'s row needs only its own distance
      vector ``[‖x − R_j‖.., 0.0]`` (self-distance last, exactly as in the
      reference where ``x`` is the final row), its 20th percentile, and the
      strict-below count.
    * RMD replays the reference's memory-buffer evolution: after a reset
      (the agent's pure-batch contract) and a pass over ``R``, the buffer
      holds the last ``memory_depth`` reference rows; appending ``x`` and
      scoring against the buffer mean reproduces the loop's final
      iteration.
    * The quantum/nano/harmonic blend scalars are *batch-global* — they are
      recomputed on the assembled ``[R; x]`` array through the detector's
      own private methods (same arrays, same code), which is O(n·d).
    """

    def __init__(self, detector: Any, reference: np.ndarray[Any, Any]) -> None:
        super().__init__(detector, reference)
        n, d = reference.shape
        # Preallocated [R; x] batch for the batch-global blend scalars;
        # row -1 is overwritten per serve under the cache lock.
        self._batch = np.empty((n + 1, d), dtype=np.float64)
        self._batch[:n] = reference
        self._baseline_ref = detector.baseline_pattern
        # EOA: np.max over the batch magnitudes == max(reference max, |x|)
        # exactly (comparison-based, no rounding).
        self._ref_magnitude_max = float(np.max(np.linalg.norm(reference, axis=1)))
        # RMD: buffer contents after a reset and a full pass over R — the
        # loop appends `row.astype(np.float64)` (a copy) per row.
        depth = int(detector.memory_depth)
        tail = reference[max(0, n - depth) :]
        self._rmd_tail: list[np.ndarray[Any, Any]] = [r.astype(np.float64) for r in tail]

    def _still_valid(self) -> bool:
        det = self._detector
        return (
            det._is_fitted
            and not det._auto_calibrate
            and det.baseline_pattern is self._baseline_ref
            and len(self._rmd_tail) == min(int(det.memory_depth), self._reference.shape[0])
        )

    def _serve_locked(self, row: np.ndarray[Any, Any]) -> float | None:
        det = self._detector
        n = self._reference.shape[0]
        n_total = n + 1

        # --- PCP (row-independent) ---------------------------------------
        baseline = det.baseline_pattern
        diff = float(np.linalg.norm(row[np.newaxis, :] - baseline, axis=1)[0])
        normalized = diff / (float(np.linalg.norm(baseline)) + 1e-6)
        pcp = normalized / (det.convergence_threshold + normalized)

        # --- GSIS (x's row of the pairwise term) -------------------------
        # The reference computes ‖block_row − batch_row‖ with the same
        # subtract → square → pairwise-sum → sqrt pipeline; the sign flip
        # (R − x vs x − R) squares away bit-exactly, and x's self-distance
        # is exactly 0.0 at the final position.
        if n_total < 2:
            gsis = 0.0
        else:
            dist_vec = np.empty(n_total, dtype=np.float64)
            dist_vec[:n] = np.linalg.norm(self._reference - row, axis=1)
            dist_vec[n] = 0.0
            threshold = np.percentile(dist_vec, 20)
            local_density = int(np.sum(dist_vec < threshold))
            gsis = (1.0 - local_density / n_total) * det.stability_factor

        # --- RMD (memory-buffer replay) -----------------------------------
        buffer_rows = [*self._rmd_tail, row.astype(np.float64)]
        depth = int(det.memory_depth)
        buffer_rows = buffer_rows[-depth:]
        if len(buffer_rows) > 1:
            memory_mean = np.mean(np.array(buffer_rows), axis=0)
            deviation = float(np.linalg.norm(row - memory_mean))
            rmd = deviation / (1.0 + deviation)
        else:
            rmd = 0.0

        # --- EOA (cached reference maximum) --------------------------------
        magnitude = float(np.linalg.norm(row[np.newaxis, :], axis=1)[0])
        eoa = magnitude / (max(self._ref_magnitude_max, magnitude) + 1e-6)

        weights = det.weights
        combined = (
            pcp * weights.pcp_weight
            + gsis * weights.gsis_weight
            + rmd * weights.rmd_weight
            + eoa * weights.eoa_weight
        )

        # --- Batch-global blend scalars (recomputed on [R; x]) -------------
        self._batch[-1] = row
        if det.use_quantum_enhanced:
            quantum_scores = det._quantum_pattern_containment(self._batch)
            if quantum_scores and weights.quantum_blend > 0:
                quantum_avg = np.mean(list(quantum_scores.values()))
                blend = weights.quantum_blend
                combined = combined * (1 - blend) + quantum_avg * blend
        if det.use_nano_detection:
            nano_scores = det._nano_scale_detection(self._batch)
            if nano_scores and weights.nano_blend > 0:
                nano_avg = np.mean(list(nano_scores.values()))
                blend = weights.nano_blend
                combined = combined * (1 - blend) + nano_avg * blend
        if det.use_harmonic_detection:
            harmonic_score = det._harmonic_anomaly_detection(self._batch)
            if harmonic_score > 0 and weights.harmonic_blend > 0:
                blend = weights.harmonic_blend
                combined = combined * (1 - blend) + harmonic_score * blend

        if not np.isfinite(combined):
            return None  # the reference nan_to_num path; fall back instead
        return float(np.clip(combined, 0.0, 1.0))


class _SpatialServingCache(_ServingCache):
    """Incremental form of ``SpatialAnomalyDetector.detect`` for the last row.

    Both raw scorers are row-independent against fitted state (center
    distance; LOF k-NN against the fit-time KDTree), so the reference batch's
    raw scores are cached and only the batch min/max — exact under
    comparison-based ``np.min``/``np.max`` — is updated with the new row
    before replaying ``_safe_normalize``'s scalar arithmetic.
    """

    def __init__(self, detector: Any, reference: np.ndarray[Any, Any]) -> None:
        super().__init__(detector, reference)
        self._center_ref = detector.center
        self._radius = float(detector.radius_threshold)
        self._lof = detector.lof
        self._tree_ref = detector.lof._tree
        ref_distance = np.asarray(detector._compute_distance_scores(reference), dtype=np.float64)
        ref_neg_lof = -np.asarray(detector.lof.decision_function(reference), dtype=np.float64)
        if not (np.all(np.isfinite(ref_distance)) and np.all(np.isfinite(ref_neg_lof))):
            raise ValueError("non-finite reference scores; incremental serving unavailable")
        self._distance_min = float(ref_distance.min())
        self._distance_max = float(ref_distance.max())
        self._neg_lof_min = float(ref_neg_lof.min())
        self._neg_lof_max = float(ref_neg_lof.max())

    def _still_valid(self) -> bool:
        det = self._detector
        return (
            det._is_fitted
            and not det._auto_calibrate
            and det.center is self._center_ref
            and det.lof is self._lof
            and det.lof._tree is self._tree_ref
            and float(det.radius_threshold) == self._radius
        )

    @staticmethod
    def _normalize_one(cached_min: float, cached_max: float, value: float) -> float | None:
        """Replay ``_safe_normalize`` for the appended row's element."""
        if not np.isfinite(value):
            return None
        score_min = min(cached_min, value)
        score_max = max(cached_max, value)
        score_range = score_max - score_min
        if score_range < 1e-10:
            return 0.5
        return float(np.clip((value - score_min) / score_range, 0.0, 1.0))

    def _serve_locked(self, row: np.ndarray[Any, Any]) -> float | None:
        det = self._detector
        row_2d = row[np.newaxis, :]
        distance_raw = float(det._compute_distance_scores(row_2d)[0])
        neg_lof_raw = float(-det.lof.decision_function(row_2d)[0])
        distance_norm = self._normalize_one(self._distance_min, self._distance_max, distance_raw)
        neg_lof_norm = self._normalize_one(self._neg_lof_min, self._neg_lof_max, neg_lof_raw)
        if distance_norm is None or neg_lof_norm is None:
            return None
        return (distance_norm + neg_lof_norm) / 2.0


class _TemporalServingCache(_ServingCache):
    """Incremental form of ``TemporalAnomalyDetector.detect`` for the last row.

    The trend term reads only the ``window_size`` rows preceding the appended
    row (a fixed reference slice); the sudden-change term's batch statistics
    (``np.mean(|diffs|)`` / ``np.std(diffs)`` over all rows) are recomputed
    on the assembled diff buffers — same arrays, same reductions, bit-equal —
    at O(n·d) instead of re-walking the trend window per reference row.
    """

    def __init__(self, detector: Any, reference: np.ndarray[Any, Any]) -> None:
        super().__init__(detector, reference)
        n, d = reference.shape
        diffs_ref = np.diff(reference, axis=0, prepend=reference[0:1])
        self._diffs = np.empty((n + 1, d), dtype=np.float64)
        self._diffs[:n] = diffs_ref
        self._abs_diffs = np.empty((n + 1, d), dtype=np.float64)
        self._abs_diffs[:n] = np.abs(diffs_ref)
        self._last_reference_row = reference[n - 1]
        # Trend-window statistics keyed by window size (re-derived if the
        # detector's configuration changes between serves).
        self._trend_stats: dict[int, tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]] = {}

    def _still_valid(self) -> bool:
        det = self._detector
        return det._is_fitted and not det._auto_calibrate

    def _serve_locked(self, row: np.ndarray[Any, Any]) -> float | None:
        det = self._detector
        n = self._reference.shape[0]

        # --- Trend (window = the last `window_size` reference rows) -------
        window_size = int(det.window_size)
        if n >= window_size > 0:
            stats = self._trend_stats.get(window_size)
            if stats is None:
                window = self._reference[n - window_size : n]
                stats = (
                    np.mean(window, axis=0),
                    np.std(window, axis=0) + 1e-6,
                )
                self._trend_stats[window_size] = stats
            window_mean, window_std = stats
            z_scores = np.abs((row - window_mean) / window_std)
            z_score = float(np.max(z_scores))
            trend = z_score / (3.0 + z_score)
        else:
            trend = 0.0

        # --- Sudden change (batch statistics over the assembled diffs) ----
        self._diffs[n] = row - self._last_reference_row
        self._abs_diffs[n] = np.abs(self._diffs[n])
        diff_mean = np.mean(self._abs_diffs, axis=0)
        diff_std = np.std(self._diffs, axis=0) + 1e-6
        z_vec = np.abs(self._diffs[n] - diff_mean) / diff_std
        max_z = float(np.max(z_vec))
        change = max_z / (det.change_threshold + max_z)

        return float((trend + change) / 2.0)


def build_serving_cache(
    detector: Any, reference: np.ndarray[Any, Any] | None
) -> _ServingCache | None:
    """Build the exact incremental serving cache for a supported detector.

    Returns ``None`` (callers fall back to the verbatim full-batch path) for
    unsupported detector types, subclasses, auto-calibrating instances,
    unfitted detectors, or non-finite references. Never raises into the
    serving path.
    """
    if reference is None or not _finite_2d(np.asarray(reference)):
        return None

    try:
        from omni_mercury_engine.detectors.directive import SigmaDirectiveDetector
        from omni_mercury_engine.detectors.spatial import SpatialAnomalyDetector
        from omni_mercury_engine.detectors.temporal import TemporalAnomalyDetector
    except Exception as exc:  # pragma: no cover - optional detector imports
        logger.debug("Incremental serving unavailable (detector imports): %s", exc)
        return None

    detector_type = type(detector)
    try:
        if getattr(detector, "_auto_calibrate", True) or not getattr(detector, "_is_fitted", False):
            return None
        if detector_type is SigmaDirectiveDetector:
            if detector.baseline_pattern is None:
                return None
            return _DirectiveServingCache(detector, reference)
        if detector_type is SpatialAnomalyDetector:
            if detector.center is None or detector.radius_threshold is None:
                return None
            if detector.lof._tree is None or detector.lof._lrd is None:
                return None
            return _SpatialServingCache(detector, reference)
        if detector_type is TemporalAnomalyDetector:
            return _TemporalServingCache(detector, reference)
    except Exception as exc:
        logger.debug("Incremental serving cache build failed; using full path: %s", exc)
        return None
    return None
