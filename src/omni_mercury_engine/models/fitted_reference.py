# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fitted-reference anomaly scoring shared by the exotic model family.

Measurement on real labeled data (2026-10-27 sweep) showed the exotic
models scored with *fixed, unfitted directions*: ``astrophysical`` graded
proximity-to-horizon (``1/norm``), so a 12-sigma burst scored as maximally
safe (AUROC 0.000 against clean windows); ``quantum`` graded raw feature
spread, landing below chance (0.33); ``consciousness`` thresholded
coherence against a constant and separated nothing (0.644, means 0.228 vs
0.224). None of them ever learned what *normal* looks like, so "anomalous"
pointed in an arbitrary direction.

This mixin replaces direction-guessing with deviation from a fitted
reference, computed in each model's own feature space:

* :meth:`fit` learns a robust per-feature center (median) and scale
  (1.4826 x MAD, with a std fallback) from reference data, and stores the
  reference deviations for empirical calibration.
* :meth:`reference_scores` scores rows by mean absolute robust z across
  informative features, calibrated to [0, 1] through the empirical CDF of
  the reference deviations - so any departure from the fitted reference,
  high-energy or low-energy, raises the score, and scores are comparable
  across models.

Scoring before :meth:`fit` raises - fail closed, never a silent heuristic.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: Minimum reference rows accepted by :meth:`FittedReferenceScorer.fit`.
#: Below this a median/MAD estimate is noise dressed as a baseline.
MIN_REFERENCE_ROWS = 8

#: Scale floor: features whose robust scale falls below this are constant
#: in the reference and carry no deviation information; they are excluded
#: from scoring rather than amplified by a near-zero denominator.
_SCALE_FLOOR = 1e-9

#: MAD-to-sigma consistency constant for normal data.
_MAD_SIGMA = 1.4826


class FittedReferenceScorer:
    """Mixin: robust fitted-reference scoring over ``extract_features``.

    The host class must provide ``extract_features(data) -> (n, d)``; the
    mixin adds ``fit`` / ``fitted`` / ``reference_scores`` and makes the
    anomaly score a calibrated deviation from the fitted reference instead
    of a hand-picked direction.
    """

    _ref_center: np.ndarray[Any, Any] | None = None
    _ref_scale: np.ndarray[Any, Any] | None = None
    _ref_mask: np.ndarray[Any, Any] | None = None
    _ref_deviations: np.ndarray[Any, Any] | None = None

    @property
    def fitted(self) -> bool:
        """True once :meth:`fit` has stored a usable reference."""
        return self._ref_center is not None

    def fit(self, data: np.ndarray[Any, Any] | dict[str, Any]) -> FittedReferenceScorer:
        """Fit the robust reference from normal (or mostly-normal) data.

        Args:
            data: Reference input in any form ``extract_features`` accepts;
                must yield at least :data:`MIN_REFERENCE_ROWS` feature rows.

        Returns:
            self, for chaining.

        Raises:
            ValueError: Fewer than :data:`MIN_REFERENCE_ROWS` reference rows,
                or no feature carries any spread (nothing to score against).
        """
        feats = np.asarray(self.extract_features(data), dtype=np.float64)  # type: ignore[attr-defined]
        feats = np.where(np.isfinite(feats), feats, np.nan)
        if feats.shape[0] < MIN_REFERENCE_ROWS:
            raise ValueError(
                f"fitted-reference scoring needs >= {MIN_REFERENCE_ROWS} reference rows, "
                f"got {feats.shape[0]}; a smaller reference is noise dressed as a baseline"
            )
        center = np.nanmedian(feats, axis=0)
        mad = np.nanmedian(np.abs(feats - center), axis=0)
        scale = _MAD_SIGMA * mad
        std = np.nanstd(feats, axis=0)
        scale = np.where(scale > _SCALE_FLOOR, scale, std)
        mask = scale > _SCALE_FLOOR
        if not bool(mask.any()):
            raise ValueError("every feature is constant in the reference; nothing to score against")
        self._ref_center = center
        self._ref_scale = np.where(mask, scale, 1.0)
        self._ref_mask = mask
        self._ref_deviations = np.sort(self._deviation(feats))
        return self

    def _deviation(self, feats: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Mean absolute robust z per row over the informative features."""
        assert self._ref_center is not None
        z = np.abs(feats - self._ref_center) / self._ref_scale
        z = np.where(np.isfinite(z), z, 0.0)
        return np.asarray(z[:, self._ref_mask].mean(axis=1), dtype=np.float64)

    def reference_scores(self, data: np.ndarray[Any, Any] | dict[str, Any]) -> dict[str, Any]:
        """Score rows by calibrated deviation from the fitted reference.

        Args:
            data: Input in any form ``extract_features`` accepts.

        Returns:
            ``anomaly_scores``: per-row scores in [0, 1) - the empirical-CDF
            rank of each row's deviation among the reference deviations
            (reference-typical rows spread uniformly below ~1; departures in
            *either* direction rise), with a strictly monotone tail beyond
            the reference maximum so magnitudes past the reference are never
            flattened into ties; ``reference_deviation``: the raw mean |z|
            per row; ``confidence``: fraction of features that were
            informative in the reference, discounted while the reference is
            small (n/64 cap) - low confidence means the reference barely
            constrains the score.

        Raises:
            RuntimeError: Called before :meth:`fit` (fail closed; the
                pre-fit heuristics this family used to ship scored in
                arbitrary directions and are not served as a fallback).
        """
        feats = np.asarray(self.extract_features(data), dtype=np.float64)  # type: ignore[attr-defined]
        return self.score_features(feats)

    def score_features(self, feats: np.ndarray[Any, Any]) -> dict[str, Any]:
        """Score an already-extracted ``(n, d)`` feature matrix.

        Same contract as :meth:`reference_scores`; exists so callers that
        transform features after extraction (e.g. noise-model paths) score
        the transformed matrix rather than silently re-extracting.
        """
        if not self.fitted:
            raise RuntimeError(
                f"{type(self).__name__} is unfitted: call fit(reference) first; "
                "unfitted heuristic scores are refused rather than served"
            )
        assert self._ref_deviations is not None and self._ref_mask is not None
        feats = np.where(np.isfinite(feats), np.asarray(feats, dtype=np.float64), np.nan)
        dev = self._deviation(feats)
        n = self._ref_deviations.size
        ranks = np.searchsorted(self._ref_deviations, dev, side="right")
        scores = (ranks + 0.5) / (n + 1.0)
        # Beyond the reference maximum the ECDF is flat; extend it strictly
        # monotonically in the deviation so a 20-sigma row outranks a
        # 12-sigma one instead of tying with it.
        ref_max = float(self._ref_deviations[-1])
        over = dev > ref_max
        if ref_max > 0.0 and bool(over.any()):
            scores[over] = (n + 0.5) / (n + 1.0) + (0.5 / (n + 1.0)) * (1.0 - ref_max / dev[over])
        n_ref = float(n)
        confidence = float(self._ref_mask.mean()) * min(1.0, n_ref / 64.0)
        return {
            "anomaly_scores": np.clip(scores, 0.0, 1.0).astype(np.float32),
            "reference_deviation": dev.astype(np.float32),
            "confidence": confidence,
        }
