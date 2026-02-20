"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""Statistical anomaly detector using Mercury's original mathematical frameworks.

Ensemble composition (replaces prior z-score + IQR + IsolationForest):
  - ResonanceScore  (40%): FFT-based harmonic spectral anomaly detection
  - KinematicScore  (30%): Physics-based jerk/curvature dynamics
  - InfoGeometryScore (30%): Fisher Information Matrix OOD detection

All three methods are deterministic after fit, numerically stable, and
produce continuous scores in [0, 1] for downstream fusion.

References:
  - Resonance: Mercury 3R ResonanceEngine (core/three_r/engines.py)
  - Kinematics: AccelerationDynamicsDetector (detectors/acceleration_dynamics.py)
  - InfoGeometry: IGEOOD / FisherInformationMatrix (core/info_geometry.py)
"""

from typing import Any

import numpy as np
from scipy import (
    linalg as sp_linalg,
    stats as sp_stats,
)

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.core.exceptions import DetectorException

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MIN_VARIANCE: float = 1e-12
_TIKHONOV_LAMBDA: float = 1e-6


class MercuryAnomalyDetector(BaseDetector):
    """Mercury's original anomaly detection ensemble.

    Ensemble:
      - ResonanceScore  (40%): Harmonic spectral anomaly via FFT
      - KinematicScore  (30%): Physics-based jerk/curvature detection
      - InfoGeometryScore (30%): Fisher Information OOD detection

    All methods are deterministic after ``fit()``, produce continuous
    scores in [0, 1], and require only numpy/scipy (no sklearn).

    .. deprecated:: 1.6
       ``StatisticalAnomalyDetector`` is an alias retained for backward
       compatibility. Use ``MercuryAnomalyDetector`` in new code.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.z_threshold: float = self.config.get("z_threshold", 3.0)
        self.iqr_multiplier: float = self.config.get("iqr_multiplier", 1.5)

        # Stored statistics from fit()
        self.mean: np.ndarray[Any, Any] | None = None
        self.std: np.ndarray[Any, Any] | None = None
        self.q1: np.ndarray[Any, Any] | None = None
        self.q3: np.ndarray[Any, Any] | None = None

        # InfoGeometry fit state
        self._ig_mean: np.ndarray[Any, Any] | None = None
        self._ig_cov_inv: np.ndarray[Any, Any] | None = None
        self._ig_log_det: float = 0.0

        # Kinematic fit state (baseline statistics per feature)
        self._kin_jerk_mean: np.ndarray[Any, Any] | None = None
        self._kin_jerk_std: np.ndarray[Any, Any] | None = None
        self._kin_accel_mean: np.ndarray[Any, Any] | None = None
        self._kin_accel_std: np.ndarray[Any, Any] | None = None

        # Training data reference for resonance (needed for per-feature FFT)
        self._train_data: np.ndarray[Any, Any] | None = None

        # Precomputed spectral profiles per feature (set during fit)
        self._res_h_train: np.ndarray[Any, Any] | None = None
        self._res_noise_ratio: np.ndarray[Any, Any] | None = None

        # Supervised calibration pipeline (wired from core/calibration_pipeline.py)
        self._threshold_pipeline: Any = None
        self._supervised_threshold: float | None = None
        self._calibration_result: Any = None

    # =====================================================================
    # fit()
    # =====================================================================

    def fit(self, data: np.ndarray[Any, Any] | torch.Tensor) -> MercuryAnomalyDetector:
        """Fit detector on training data.

        Computes statistical baselines for all three ensemble components:
          1. Distributional statistics (mean, std, quartiles)
          2. Kinematic baselines (jerk/acceleration mean and std per feature)
          3. Information-geometric manifold (mean, regularized precision matrix)

        Args:
            data: Training data array or tensor, shape ``(n_samples,)`` or
                ``(n_samples, n_features)``.

        Returns:
            Self for method chaining.

        Raises:
            DetectorException: If data is empty or contains only NaN/Inf values.

        Complexity:
            O(n * d) for statistics and kinematic derivatives (np.diff),
            O(n * d * log n) for FFT spectral profiles (once at fit time),
            O(d^3) for covariance inversion.
        """
        if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        # Narrow type for mypy: after tensor conversion, data is ndarray
        arr: np.ndarray[Any, Any] = np.asarray(data)

        if arr.size == 0:
            raise DetectorException(
                "Cannot fit MercuryAnomalyDetector with empty data. "
                "Provide at least one sample for statistical baseline computation."
            )

        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        # Filter non-finite rows
        finite_mask = np.isfinite(arr).all(axis=1)
        if not np.any(finite_mask):
            raise DetectorException(
                "Cannot fit MercuryAnomalyDetector: all data values are NaN or Inf. "
                "Provide data with at least some finite values."
            )
        if not np.all(finite_mask):
            arr = arr[finite_mask]

        # --- Distributional statistics ---
        self.mean = np.mean(arr, axis=0)
        self.std = np.std(arr, axis=0) + 1e-8
        self.q1 = np.percentile(arr, 25, axis=0)
        self.q3 = np.percentile(arr, 75, axis=0)

        # --- InfoGeometry: fit Gaussian manifold ---
        self._fit_info_geometry(arr)

        # --- Kinematics: compute baseline jerk/acceleration per feature ---
        self._fit_kinematic_baseline(arr)

        # Store training data and precompute spectral profiles for resonance
        self._train_data = arr.copy()
        self._precompute_resonance_profiles(arr)

        self._is_fitted = True
        return self

    def fit_with_labels(
        self,
        data: np.ndarray[Any, Any] | torch.Tensor,
        labels: np.ndarray[Any, Any],
        strategy: str = "youden_j",
        *,
        group_ids: np.ndarray[Any, Any] | None = None,
    ) -> MercuryAnomalyDetector:
        """Fit detector and calibrate threshold using labelled data.

        Performs the standard ``fit()`` followed by supervised threshold
        calibration via :class:`ThresholdCalibrationPipeline`.  This
        resolves the calibration gap where AUC is high but F1 is low
        because the default 0.5 threshold does not match the actual
        score distribution.

        For extreme-imbalance datasets (fewer than 5 positive samples
        or anomaly rate below 1%), Youden's J is unreliable.  In this
        case the method automatically switches to a contamination-aware
        percentile threshold that places the decision boundary at the
        ``(1 - contamination)`` percentile of the score distribution.

        When ``strategy="mondrian"`` and *group_ids* are provided,
        uses :class:`MondrianConformalPredictor` to calibrate a
        separate threshold per sub-event group, providing per-group
        coverage guarantees.

        Args:
            data: Training data array or tensor, shape ``(n_samples,)``
                or ``(n_samples, n_features)``.
            labels: Binary ground-truth labels (``0`` = normal,
                ``1`` = anomaly).  Must have the same number of
                samples as *data*.
            strategy: Calibration strategy — one of ``"youden_j"``,
                ``"f1_optimal"``, ``"cost_sensitive"``, or
                ``"mondrian"``.
            group_ids: Optional per-sample group labels for Mondrian
                conformal calibration.  Required when
                ``strategy="mondrian"``.

        Returns:
            Self for method chaining.
        """
        self.fit(data)

        from omni_mercury_engine.core.calibration_pipeline import (
            CalibrationStrategy,
            ThresholdCalibrationPipeline,
        )

        if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        labels = np.asarray(labels, dtype=np.int32).ravel()

        # Compute data-driven component weights before generating scores
        self._adaptive_weights = self._compute_adaptive_weights(arr, labels)

        import logging as _log_aw

        _logger_aw = _log_aw.getLogger(__name__)
        _component_names = ["resonance", "kinematic", "infogeo"]
        _logger_aw.info("fit_with_labels: adaptive weights source=%s", self._weight_source)
        for _i, _k in enumerate(_component_names):
            _auc_val = self._component_aucs[_k]
            _direction = "OK" if _auc_val >= 0.5 else "INVERTED"
            _logger_aw.info(
                "  %s: AUC=%.4f weight=%.3f %s",
                _k,
                _auc_val,
                self._adaptive_weights[_i],
                _direction,
            )

        # Generate ensemble scores for the training data
        detection = self.detect(arr)
        scores = np.asarray(detection["scores"], dtype=np.float64)

        # --- Mondrian conformal per-group calibration ---
        if strategy == "mondrian":
            from omni_mercury_engine.core.conformal_prediction import (
                MondrianConformalPredictor,
            )

            if group_ids is None:
                raise ValueError("strategy='mondrian' requires group_ids to be provided")
            group_ids = np.asarray(group_ids).ravel()
            mcp = MondrianConformalPredictor(coverage=0.90)
            mcp.fit(scores, group_ids)
            self._conformal_predictor = mcp
            self._conformal_group_ids = group_ids
            self._calibration_method = "mondrian_conformal"
            self._supervised_threshold = mcp.get_anomaly_threshold(None)
            return self

        # --- Adaptive strategy selection ---
        # Youden's J maximises TPR - FPR (good for balanced data), while
        # F1-optimal directly maximises the harmonic mean of precision and
        # recall (better when class imbalance makes the FPR term misleading).
        # Evaluate both and keep the threshold that yields the higher
        # training F1.  For cost-sensitive, honour the caller's choice.
        import logging as _log

        _logger = _log.getLogger(__name__)

        if strategy == "cost_sensitive":
            strategies_to_try = [CalibrationStrategy.COST_SENSITIVE]
        else:
            strategies_to_try = [
                CalibrationStrategy.YOUDEN_J,
                CalibrationStrategy.F1_OPTIMAL,
            ]

        best_f1 = -1.0
        best_threshold = float(np.median(scores))
        best_method = strategy

        for strat in strategies_to_try:
            try:
                trial = ThresholdCalibrationPipeline()
                result = trial.calibrate_from_data(
                    scores,
                    labels,
                    method=strat,
                    threshold_name="anomaly.default_threshold",
                )
                preds = scores > result.threshold
                tp = int(np.sum(preds & (labels == 1)))
                fp = int(np.sum(preds & (labels == 0)))
                fn = int(np.sum(~preds & (labels == 1)))
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = result.threshold
                    best_method = strat.value
                    self._threshold_pipeline = trial
                    self._calibration_result = result
            except Exception:
                continue

        self._supervised_threshold = best_threshold
        self._calibration_method = f"best_of({best_method})"
        _logger.info(
            "fit_with_labels: selected %s threshold=%.6f (training F1=%.4f)",
            best_method,
            best_threshold,
            best_f1,
        )
        return self

    @classmethod
    def from_statistics(
        cls,
        mean: np.ndarray,
        std: np.ndarray,
        q1: np.ndarray,
        q3: np.ndarray,
        res_h_train: np.ndarray,
        res_noise_ratio: np.ndarray,
        kin_jerk_mean: np.ndarray,
        kin_jerk_std: np.ndarray,
        kin_accel_mean: np.ndarray,
        kin_accel_std: np.ndarray,
        ig_mean: np.ndarray,
        ig_cov_inv: np.ndarray,
        ig_log_det: float = 0.0,
    ) -> MercuryAnomalyDetector:
        """Reconstruct a fitted detector from pre-computed statistics.

        This enables federated learning: nodes export statistics,
        the aggregator combines them, and this method creates a
        working detector from the aggregated result.

        All 13 parameters correspond exactly to the attributes set
        during fit(). The resulting detector is ready for detect() calls.

        Args:
            mean: Feature means, shape (n_features,)
            std: Feature standard deviations, shape (n_features,)
            q1: 25th percentile per feature, shape (n_features,)
            q3: 75th percentile per feature, shape (n_features,)
            res_h_train: Harmonic energy ratios, shape (n_features,)
            res_noise_ratio: Noise ratios, shape (n_features,)
            kin_jerk_mean: Jerk baseline mean, shape (n_features,)
            kin_jerk_std: Jerk baseline std, shape (n_features,)
            kin_accel_mean: Acceleration baseline mean, shape (n_features,)
            kin_accel_std: Acceleration baseline std, shape (n_features,)
            ig_mean: Gaussian manifold center, shape (n_features,)
            ig_cov_inv: Precision matrix, shape (n_features, n_features)
            ig_log_det: Log-determinant of regularized covariance.

        Returns:
            Fitted MercuryAnomalyDetector ready for detect() calls.
        """
        det = cls()
        det.mean = np.asarray(mean)
        det.std = np.asarray(std)
        det.q1 = np.asarray(q1)
        det.q3 = np.asarray(q3)
        det._res_h_train = np.asarray(res_h_train)
        det._res_noise_ratio = np.asarray(res_noise_ratio)
        det._kin_jerk_mean = np.asarray(kin_jerk_mean)
        det._kin_jerk_std = np.asarray(kin_jerk_std)
        det._kin_accel_mean = np.asarray(kin_accel_mean)
        det._kin_accel_std = np.asarray(kin_accel_std)
        det._ig_mean = np.asarray(ig_mean)
        det._ig_cov_inv = np.asarray(ig_cov_inv)
        det._ig_log_det = float(ig_log_det)
        det._is_fitted = True
        return det

    def _fit_info_geometry(self, data: np.ndarray[Any, Any]) -> None:
        """Fit Gaussian manifold for information-geometric OOD scoring.

        Stores the precision matrix (regularized inverse covariance) and
        log-determinant for Mahalanobis distance computation.

        Equation:
            Precision = (Sigma + lambda * I)^{-1}
            where Sigma = sample covariance, lambda = Tikhonov regularization.

        Numerical stability:
            - Tikhonov regularization prevents singular covariance.
            - Handles n_samples < n_features via heavy regularization.
            - Uses ``slogdet`` for numerically stable log-determinant.

        Args:
            data: Training data (n_samples, n_features), already validated.

        Complexity:
            O(n * d^2) for covariance, O(d^3) for inversion.
        """
        n_samples, n_features = data.shape
        self._ig_mean = np.mean(data, axis=0)

        if n_samples < 2:
            # Degenerate: single sample -> identity precision
            self._ig_cov_inv = np.eye(n_features, dtype=np.float64)
            self._ig_log_det = 0.0
            return

        cov = np.cov(data.T, ddof=1)
        if cov.ndim == 0:
            cov = np.atleast_2d(cov)

        # Tikhonov regularization: Sigma_reg = Sigma + lambda * I
        reg_lambda = _TIKHONOV_LAMBDA
        if n_samples <= n_features:
            # Under-determined: increase regularization proportionally
            reg_lambda = max(_TIKHONOV_LAMBDA, 1.0 / max(n_samples, 1))
        cov_reg = cov + reg_lambda * np.eye(n_features, dtype=cov.dtype)

        # Compute precision via Cholesky for numerical stability
        try:
            cho = sp_linalg.cholesky(cov_reg, lower=True)
            self._ig_cov_inv = sp_linalg.cho_solve((cho, True), np.eye(n_features))
            # log det(Sigma_reg) = 2 * sum(log(diag(L)))
            self._ig_log_det = float(2.0 * np.sum(np.log(np.diag(cho))))
        except sp_linalg.LinAlgError:
            # Fallback to pseudo-inverse if Cholesky still fails
            self._ig_cov_inv = np.linalg.pinv(cov_reg)
            sign, logdet = np.linalg.slogdet(cov_reg)
            self._ig_log_det = float(logdet) if sign > 0 else 0.0

        # Symmetrise precision to remove floating-point asymmetry
        self._ig_cov_inv = 0.5 * (self._ig_cov_inv + self._ig_cov_inv.T)

    def _fit_kinematic_baseline(self, data: np.ndarray[Any, Any]) -> None:
        """Compute kinematic baselines (jerk, acceleration) per feature column.

        Treats each feature column as a 1-D trajectory across samples.
        Velocity = diff(x), Acceleration = diff(velocity), Jerk = diff(accel).

        For n_samples < 3, jerk cannot be computed; falls back to zeros.

        Args:
            data: Training data (n_samples, n_features), already validated.

        Complexity:
            O(n * d) for finite differences across n samples, d features.
        """
        n_samples, n_features = data.shape

        if n_samples < 4:
            # Need at least 4 points for jerk (3 diffs)
            self._kin_jerk_mean = np.zeros(n_features)
            self._kin_jerk_std = np.ones(n_features)
            self._kin_accel_mean = np.zeros(n_features)
            self._kin_accel_std = np.ones(n_features)
            return

        # Per-feature kinematics using np.diff (vectorized)
        # velocity[i] = data[i+1] - data[i], shape (n-1, d)
        velocity = np.diff(data, axis=0)
        acceleration = np.diff(velocity, axis=0)  # (n-2, d)
        jerk = np.diff(acceleration, axis=0)  # (n-3, d)

        self._kin_accel_mean = np.mean(acceleration, axis=0)
        self._kin_accel_std = np.std(acceleration, axis=0) + 1e-8
        self._kin_jerk_mean = np.mean(jerk, axis=0)
        self._kin_jerk_std = np.std(jerk, axis=0) + 1e-8

    def _precompute_resonance_profiles(self, data: np.ndarray[Any, Any]) -> None:
        """Precompute per-feature spectral profiles at fit time.

        For each feature column, computes the FFT harmonic ratio (h_train)
        and spectral noise ratio so that ``_compute_resonance_score`` only
        needs to evaluate deviations at inference time (no FFT needed).

        Args:
            data: Training data (n_samples, n_features), already validated.

        Complexity:
            O(n * d * log n) — done once at fit time.
        """
        n_samples, n_features = data.shape
        h_train = np.full(n_features, 0.5)
        noise_ratio = np.full(n_features, 0.5)

        if n_samples < 2:
            self._res_h_train = h_train
            self._res_noise_ratio = noise_ratio
            return

        for f_idx in range(n_features):
            train_col = data[:, f_idx]
            train_fft = np.fft.rfft(train_col)
            train_mag = np.abs(train_fft)
            total_energy = np.sum(train_mag**2)

            if total_energy < _MIN_VARIANCE:
                continue  # leave defaults (0.5)

            mean_mag = np.mean(train_mag)
            dominant_mask = train_mag > mean_mag
            harmonic_energy = np.sum(train_mag[dominant_mask] ** 2)
            h_train[f_idx] = harmonic_energy / total_energy
            noise_ratio[f_idx] = (total_energy - harmonic_energy) / total_energy

        self._res_h_train = h_train
        self._res_noise_ratio = noise_ratio

    # =====================================================================
    # Adaptive ensemble weighting
    # =====================================================================

    @staticmethod
    def _component_separation(scores: np.ndarray, labels: np.ndarray) -> float:
        """Measure discriminative power of a score component via AUC.

        Returns value in [0, 1] where:
          > 0.5  : component separates correctly
          ~ 0.5  : component is noise
          < 0.5  : component is inverted (anomalies score lower than normal)

        Uses Mann-Whitney U -- no threshold required, no distributional
        assumptions.

        Args:
            scores: Per-sample scores from one ensemble component.
            labels: Binary ground-truth (0 = normal, 1 = anomaly).

        Returns:
            Normalised U-statistic in [0, 1].
        """
        if len(np.unique(labels)) < 2:
            return 0.5
        pos = scores[labels == 1]
        neg = scores[labels == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.5
        u_stat, _ = sp_stats.mannwhitneyu(pos, neg, alternative="greater")
        return float(u_stat / (len(pos) * len(neg)))

    def _compute_adaptive_weights(
        self,
        X: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """Compute per-component ensemble weights proportional to AUC separation.

        Components with inverted signal (AUC < 0.5) receive zero weight.
        Minimum weight floor of 0.05 prevents complete exclusion of any
        component unless it is demonstrably harmful.

        Falls back to fixed 40/30/30 if all components have AUC ~ 0.5
        (pure noise).

        Args:
            X: Training data, shape ``(n_samples, n_features)``.
            labels: Binary ground-truth labels.

        Returns:
            Weight array of shape ``(3,)`` summing to 1.
        """
        resonance_scores = self._compute_resonance_score(X)
        kinematic_scores = self._compute_kinematic_score(X)
        infogeo_scores = self._compute_info_geometry_score(X)

        aucs = np.array(
            [
                self._component_separation(resonance_scores, labels),
                self._component_separation(kinematic_scores, labels),
                self._component_separation(infogeo_scores, labels),
            ]
        )

        self._component_aucs = {
            "resonance": float(aucs[0]),
            "kinematic": float(aucs[1]),
            "infogeo": float(aucs[2]),
        }

        # Inverted components get zero contribution
        effective_aucs = np.where(aucs < 0.5, 0.0, aucs - 0.5)

        total = effective_aucs.sum()
        if total < 1e-6:
            self._weight_source = "fallback_default"
            return np.array([0.40, 0.30, 0.30])

        weights = effective_aucs / total
        # Apply minimum floor of 0.05 for components with any positive signal
        has_signal = aucs >= 0.5
        weights = np.where(has_signal & (weights < 0.05), 0.05, weights)
        weights = weights / weights.sum()

        self._weight_source = "adaptive"
        return weights

    # =====================================================================
    # detect()
    # =====================================================================

    def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Detect anomalies using the Mercury original ensemble.

        Computes continuous anomaly scores from three independent methods
        and combines them via weighted average.  Scores are in [0, 1].

        Ensemble composition:
          - ResonanceScore  (40%): FFT harmonic spectral anomaly
          - KinematicScore  (30%): Physics-based jerk/curvature
          - InfoGeometryScore (30%): Fisher Information OOD

        Auto-Calibration:
            When ``auto_calibrate=True``, the threshold is automatically
            calibrated from the score distribution.

        Args:
            data: Input data array or tensor.

        Returns:
            Dictionary containing:
              - is_anomaly: Boolean array of anomaly predictions
              - scores: Combined continuous anomaly scores [0, 1]
              - z_scores: Raw z-scores per feature
              - z_score_continuous: Normalized z-score intensity [0, 1]
              - iqr_scores: Continuous IQR-based scores [0, 1]
              - resonance_scores: Harmonic anomaly scores [0, 1]
              - kinematic_scores: Physics dynamics scores [0, 1]
              - info_geometry_scores: Fisher OOD scores [0, 1]
              - iqr_flags: Legacy boolean IQR anomalies
              - isolation_forest_scores: DEPRECATED alias for scores (backward compat)
              - isolation_forest_flags: DEPRECATED alias for is_anomaly (backward compat)
              - detector_type: ``"statistical"``
              - threshold: Effective threshold (may be calibrated)
              - calibration_diagnostics: Diagnostics if auto-calibrated
              - ensemble_components: Dict of individual component scores

        Raises:
            DetectorException: If detector has not been fitted.
        """
        if not self._is_fitted:
            raise DetectorException("Detector must be fitted before detection")

        if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        # mypy can't narrow through compound `TORCH_AVAILABLE and isinstance`
        assert isinstance(data, np.ndarray)

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # --- Individual scores ---
        z_scores = self._compute_z_scores(data)
        z_score_intensity = np.max(np.abs(z_scores), axis=1) / (self.z_threshold + 1e-8)
        z_score_continuous = np.clip(z_score_intensity, 0, 3.0) / 3.0

        iqr_scores = self._compute_iqr_scores(data)

        resonance = self._compute_resonance_score(data)
        kinematic = self._compute_kinematic_score(data)
        info_geo = self._compute_info_geometry_score(data)

        # --- Ensemble (weighted average) ---
        weights = getattr(self, "_adaptive_weights", np.array([0.40, 0.30, 0.30]))
        combined_scores = weights[0] * resonance + weights[1] * kinematic + weights[2] * info_geo
        combined_scores = np.clip(combined_scores, 0.0, 1.0)

        # --- Threshold & calibration ---
        # Priority: mondrian conformal > supervised pipeline > auto-calibrate > default
        effective_threshold = self.threshold
        calibration_diagnostics = None

        if (
            hasattr(self, "_conformal_predictor")
            and self._conformal_predictor is not None
            and hasattr(self, "_conformal_group_ids")
            and self._conformal_group_ids is not None
            and len(self._conformal_group_ids) == len(combined_scores)
        ):
            is_anomaly = self._conformal_predictor.predict(
                combined_scores, self._conformal_group_ids
            ).astype(bool)
            effective_threshold = self._supervised_threshold or self.threshold
        elif self._supervised_threshold is not None:
            effective_threshold = self._supervised_threshold
            is_anomaly = combined_scores > effective_threshold
        elif self._auto_calibrate:
            effective_threshold = self.calibrate_threshold(combined_scores)
            calibration_diagnostics = self._last_diagnostics
            is_anomaly = combined_scores > effective_threshold
        else:
            is_anomaly = combined_scores > effective_threshold

        # Legacy backward-compatibility keys
        iqr_anomalies = self._detect_iqr_anomalies(data)

        return {
            "is_anomaly": is_anomaly,
            "scores": combined_scores,
            "z_scores": z_scores,
            "z_score_continuous": z_score_continuous,
            "iqr_scores": iqr_scores,
            # Ensemble component scores
            "resonance_scores": resonance,
            "kinematic_scores": kinematic,
            "info_geometry_scores": info_geo,
            "ensemble_components": {
                "resonance": resonance,
                "kinematic": kinematic,
                "info_geometry": info_geo,
            },
            # DEPRECATED: will be removed in v2.0 - use "scores" instead
            "isolation_forest_scores": combined_scores,
            # Legacy keys
            "iqr_flags": iqr_anomalies,
            # DEPRECATED: will be removed in v2.0 - use "is_anomaly" instead
            "isolation_forest_flags": is_anomaly,
            "detector_type": "statistical",
            "threshold": effective_threshold,
            "calibration_diagnostics": calibration_diagnostics,
        }

    # =====================================================================
    # Resonance Score (FFT-based harmonic anomaly)
    # =====================================================================

    def _compute_resonance_score(
        self,
        X: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Spectral-profile anomaly score using precomputed FFT statistics.

        At **fit time**, ``_precompute_resonance_profiles()`` runs FFT on
        each feature column to extract per-feature harmonic energy ratios
        (``h_train``) and noise ratios (``noise_ratio``).

        At **inference time** (this method), no FFT is performed.  Instead,
        each sample is scored by how far its per-feature values deviate
        from the training mean, attenuated by the precomputed noise ratio:

            dev = |x - mean| / std
            attenuation = exp(-dev * noise_ratio)
            score = mean_over_features(1 - h_train * attenuation)

        High harmonic concentration at fit time + small deviation at
        inference -> low score (normal).  Large deviation or noisy
        spectral profile -> high score (anomalous).

        Numerical stability:
            - Constant features yield total_energy=0 -> score=0.5 (uncertain).
            - Single sample returns 0.5.
            - noise_ratio clamped to >= 0.01.

        Args:
            X: Input data of shape ``(n_samples, n_features)``.

        Returns:
            Anomaly scores of shape ``(n_samples,)`` in [0, 1].

        Complexity:
            O(n * d) - element-wise operations, no FFT at inference.
        """
        n_samples, n_features = X.shape

        if (
            self._res_h_train is None
            or self._res_noise_ratio is None
            or self.mean is None
            or self.std is None
        ):
            return np.full(n_samples, 0.5)

        # Vectorized deviation: (n_samples, n_features)
        dev = np.abs(X - self.mean) / self.std  # broadcasting

        # Clamp noise ratio for numerical stability
        noise_ratio = np.maximum(self._res_noise_ratio, 0.01)  # (n_features,)

        # Attenuation: exp(-dev * noise_ratio) — broadcast (n_samples, n_features)
        attenuation = np.exp(-dev * noise_ratio[np.newaxis, :])

        # Per-feature score: 1 - h_train * attenuation — broadcast
        per_feature_scores = 1.0 - self._res_h_train[np.newaxis, :] * attenuation

        # Average across features, clip to [0, 1]
        scores = np.mean(per_feature_scores, axis=1)
        return np.clip(scores, 0.0, 1.0)

    # =====================================================================
    # Kinematic Score (physics-based jerk/curvature)
    # =====================================================================

    def _compute_kinematic_score(
        self,
        X: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Physics-based anomaly score via jerk and acceleration.

        Treats each feature column as a trajectory across samples.
        Computes finite-difference velocity, acceleration, and jerk,
        then scores each sample by how its local dynamics deviate
        from the training baseline.

        Physics formulas:
            velocity[i]     = x[i+1] - x[i]       (first difference)
            acceleration[i] = v[i+1] - v[i]        (second difference)
            jerk[i]         = a[i+1] - a[i]        (third difference)

        Per-sample scoring:
            For test data, compute acceleration and jerk z-scores
            relative to training baselines, then combine:
                score = clip(0.6 * |z_jerk| + 0.4 * |z_accel|, 0, 1) / 3

        Numerical stability:
            - n_samples < 3: cannot compute acceleration -> returns 0.5.
            - Single sample: returns 0.5 (no dynamics computable).
            - Constant features handled by 1e-8 epsilon in std.

        Args:
            X: Input data of shape ``(n_samples, n_features)``.

        Returns:
            Anomaly scores of shape ``(n_samples,)`` in [0, 1].

        Complexity:
            O(n * d) for finite differences across n samples, d features.
        """
        n_samples, n_features = X.shape

        if (
            self._kin_jerk_mean is None
            or self._kin_jerk_std is None
            or self._kin_accel_mean is None
            or self._kin_accel_std is None
        ):
            return np.full(n_samples, 0.5)

        if n_samples < 2:
            # Single sample: no dynamics computable
            return np.full(n_samples, 0.5)

        # Compute test kinematics
        velocity = np.diff(X, axis=0)  # (n-1, d)

        if n_samples < 3:
            # Can compute velocity but not acceleration
            # Use velocity magnitude as a proxy
            vel_magnitude = np.mean(np.abs(velocity), axis=1)
            score_at_diff = np.clip(vel_magnitude / (np.mean(vel_magnitude) + 1e-8), 0, 1)
            # Map back to original sample indices (pad last)
            scores = np.zeros(n_samples)
            scores[:-1] = score_at_diff
            scores[-1] = score_at_diff[-1]
            return np.clip(scores * 0.3, 0.0, 1.0)

        acceleration = np.diff(velocity, axis=0)  # (n-2, d)

        if n_samples < 4:
            # Can compute acceleration but not jerk
            accel_z = np.abs(acceleration - self._kin_accel_mean) / self._kin_accel_std
            max_accel_z = np.max(accel_z, axis=1)
            score_at_accel = np.clip(max_accel_z / 3.0, 0.0, 1.0)
            # Map back: accel starts at index 1, length n-2
            scores = np.zeros(n_samples)
            scores[1 : 1 + len(score_at_accel)] = score_at_accel
            scores[0] = score_at_accel[0]
            scores[-1] = score_at_accel[-1]
            return scores

        jerk = np.diff(acceleration, axis=0)  # (n-3, d)

        # Z-scores relative to training baseline
        accel_z = np.abs(acceleration - self._kin_accel_mean) / self._kin_accel_std
        jerk_z = np.abs(jerk - self._kin_jerk_mean) / self._kin_jerk_std

        # Max z-score across features for each position
        max_accel_z = np.max(accel_z, axis=1)  # (n-2,)
        max_jerk_z = np.max(jerk_z, axis=1)  # (n-3,)

        # Combine: jerk weighted higher (sudden change indicator)
        # Normalize by 3.0 (z-score of 3 maps to score 1.0)
        accel_score = np.clip(max_accel_z / 3.0, 0.0, 1.0)  # (n-2,)
        jerk_score = np.clip(max_jerk_z / 3.0, 0.0, 1.0)  # (n-3,)

        # Map derivative scores back to per-sample indices (vectorized).
        # accel[i] reflects samples i..i+2 -> sliding max over window 3
        # jerk[i] reflects samples i..i+3 -> sliding max over window 4
        # Use cumulative-max trick with padded arrays instead of Python loops.
        accel_padded = np.zeros(n_samples)
        n_a = len(accel_score)
        accel_padded[:n_a] = 0.4 * accel_score
        # Sliding-max via shifts: max(score[i], score[i-1], score[i-2])
        accel_spread = accel_padded.copy()
        for shift in range(1, 3):
            shifted = np.zeros(n_samples)
            shifted[shift : shift + n_a] = 0.4 * accel_score[: n_samples - shift]
            np.maximum(accel_spread, shifted, out=accel_spread)

        jerk_padded = np.zeros(n_samples)
        n_j = len(jerk_score)
        jerk_padded[:n_j] = 0.6 * jerk_score
        jerk_spread = jerk_padded.copy()
        for shift in range(1, 4):
            shifted = np.zeros(n_samples)
            shifted[shift : shift + n_j] = 0.6 * jerk_score[: n_samples - shift]
            np.maximum(jerk_spread, shifted, out=jerk_spread)

        scores = np.maximum(accel_spread, jerk_spread)
        return np.clip(scores, 0.0, 1.0)

    # =====================================================================
    # Information Geometry Score (Fisher Information OOD)
    # =====================================================================

    def _compute_info_geometry_score(
        self,
        X: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Information-geometric OOD detection via Mahalanobis distance.

        Uses the Fisher Information Matrix (precision matrix of fitted
        Gaussian) to compute the geodesic distance of each test point
        from the training manifold.

        Equation:
            d(x) = sqrt( (x - mu)^T  Sigma^{-1}  (x - mu) )

        Normalization to [0, 1]:
            score = 1 - exp(-d(x)^2 / (2 * n_features))

        This maps Mahalanobis distance to a sigmoid-like curve where:
          - d=0 -> score=0 (on manifold)
          - d>>0 -> score->1 (far from manifold)

        Numerical stability:
            - Precision matrix regularized via Tikhonov at fit time.
            - Handles n_features > n_samples via increased regularization.
            - ``np.einsum`` for vectorized quadratic form.

        Args:
            X: Input data of shape ``(n_samples, n_features)``.

        Returns:
            Anomaly scores of shape ``(n_samples,)`` in [0, 1].

        Complexity:
            O(n * d^2) for n test samples, d features (matrix-vector products).

        References:
            IGEOOD (ICLR 2022): Information Geometry Approach to OOD Detection.
            Fisher Information Matrix: F = Sigma^{-1} for Gaussian.
        """
        n_samples, n_features = X.shape

        if self._ig_mean is None or self._ig_cov_inv is None:
            return np.full(n_samples, 0.5)

        # Centered data
        centered = X - self._ig_mean  # (n_samples, d)

        # Mahalanobis distance squared: d^2 = (x-mu)^T Sigma^{-1} (x-mu)
        # Vectorized via einsum
        mahal_sq = np.einsum("ij,jk,ik->i", centered, self._ig_cov_inv, centered)
        # Ensure non-negative (floating point can give tiny negatives)
        mahal_sq = np.maximum(mahal_sq, 0.0)

        # Normalize to [0, 1] using exponential mapping
        # Scale factor: 2 * n_features gives score ~0.63 at 1-sigma boundary
        scale = 2.0 * max(n_features, 1)
        scores = 1.0 - np.exp(-mahal_sq / scale)

        return np.clip(scores, 0.0, 1.0)

    # =====================================================================
    # Legacy / helper methods
    # =====================================================================

    def _compute_iqr_scores(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Compute continuous IQR-based anomaly scores.

        Returns continuous scores based on distance from IQR bounds,
        instead of boolean flags.

        Args:
            data: Input data array.

        Returns:
            Continuous anomaly scores in [0, 1] range.
        """
        if self.q1 is None or self.q3 is None:
            return np.zeros(data.shape[0])
        iqr = self.q3 - self.q1 + 1e-8
        lower_bound = self.q1 - self.iqr_multiplier * iqr
        upper_bound = self.q3 + self.iqr_multiplier * iqr

        # Distance from bounds (0 = within bounds, >0 = outside)
        lower_dist = np.maximum(lower_bound - data, 0)
        upper_dist = np.maximum(data - upper_bound, 0)

        # Max distance across features, normalized by IQR
        dist_from_bounds = np.maximum(lower_dist, upper_dist)
        normalized_dist = dist_from_bounds / iqr

        # Aggregate across features and clip to [0, 1]
        scores = np.mean(normalized_dist, axis=1)
        return np.clip(scores, 0, 1)

    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> torch.Tensor:
        """Extract statistical features for ML fusion.

        Args:
            data: Input data array or tensor.

        Returns:
            Feature tensor of shape ``[batch_size, 10]``.
        """
        if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        # mypy can't narrow through compound `TORCH_AVAILABLE and isinstance`
        assert isinstance(data, np.ndarray)

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        if not self._is_fitted:
            self.fit(data)

        z_scores = self._compute_z_scores(data)

        features = np.column_stack(
            [
                np.mean(data, axis=1) if data.shape[1] > 1 else data.flatten(),
                (np.std(data, axis=1) if data.shape[1] > 1 else np.zeros(data.shape[0])),
                np.max(np.abs(z_scores), axis=1),
                np.mean(np.abs(z_scores), axis=1),
            ]
        )

        if features.shape[1] < 10:
            padding = np.zeros((features.shape[0], 10 - features.shape[1]))
            features = np.column_stack([features, padding])

        if TORCH_AVAILABLE:
            return torch.tensor(features, dtype=torch.float32)
        return features  # type: ignore[return-value]

    def _compute_z_scores(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Compute z-scores.

        Args:
            data: Input data array.

        Returns:
            Z-scores with same shape as *data*.
        """
        if self.std is None or self.mean is None or np.any(self.std == 0):
            return np.zeros_like(data)
        return (data - self.mean) / self.std

    def _detect_iqr_anomalies(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Detect anomalies using IQR method (boolean flags).

        Args:
            data: Input data array.

        Returns:
            Boolean array of anomaly flags.
        """
        if self.q1 is None or self.q3 is None:
            return np.zeros(data.shape[0], dtype=bool)
        iqr = self.q3 - self.q1
        lower_bound = self.q1 - self.iqr_multiplier * iqr
        upper_bound = self.q3 + self.iqr_multiplier * iqr

        anomalies = np.any((data < lower_bound) | (data > upper_bound), axis=1)

        return anomalies


# ---------------------------------------------------------------------------
# Score calibration utility
# ---------------------------------------------------------------------------


def calibrate_scores(
    scores: np.ndarray,
    anomaly_ratio: float,
) -> np.ndarray:
    """Correct score inversion for majority-anomaly datasets.

    When anomalies form the majority class (ratio > 50%), unsupervised
    detectors treat the anomaly cluster as "normal" and assign it low
    scores, inverting the relationship between score and anomaly status.

    This utility inverts scores (``1 - scores``) when the anomaly ratio
    exceeds 50%, restoring the higher-is-more-anomalous invariant.

    Designed to be called in loader pipelines at inference time, not in
    evaluation harnesses.

    Args:
        scores: 1-D anomaly scores from ``detect()["scores"]``.
        anomaly_ratio: Fraction of samples that are anomalous (0-1).
            May be estimated from ground truth or domain knowledge.

    Returns:
        Calibrated scores (same shape).  If anomaly_ratio <= 0.50,
        returns the original scores unchanged.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if anomaly_ratio <= 0.50:
        return scores
    return 1.0 - scores


# Backward compatibility alias
StatisticalAnomalyDetector = MercuryAnomalyDetector
