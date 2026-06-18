# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Statistical anomaly detector using Mercury's original mathematical frameworks.

Ensemble composition — three complementary, deterministic detectors. The
percentages below are the **default (fallback) weights**; after ``fit()`` they
are re-weighted proportional to each component's measured AUC separation. A
component whose signal is anti-correlated with anomalies (AUC < 0.5) is given
zero weight — its own scores are not flipped; whole-ensemble score inversion
is a separate safeguard applied during validation and detection. The ensemble
falls back to the defaults only when every component is near-random, so it is
adaptive, not static:
  - ResonanceScore  (default 40%): FFT-based harmonic spectral anomaly detection
  - KinematicScore  (default 30%): Physics-based jerk/curvature dynamics
  - InfoGeometryScore (default 30%): Fisher Information Matrix OOD detection

All three methods are deterministic after fit, numerically stable, and
produce continuous scores in [0, 1] for downstream fusion.

References:
  - Resonance: Mercury 3R ResonanceEngine (core/three_r/engines.py)
  - Kinematics: AccelerationDynamicsDetector (detectors/acceleration_dynamics.py)
  - InfoGeometry: IGEOOD / FisherInformationMatrix (core/info_geometry.py)
"""

from __future__ import annotations

import logging
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

try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.core.config import COMPONENT_COMPATIBILITY, DataCharacteristics
from omni_mercury_engine.core.exceptions import DetectorException
from omni_mercury_engine.core.governed_fusion import (
    InfoGeometryCertificate,
    mahalanobis_score_to_price_threshold,
)

logger = logging.getLogger(__name__)

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

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        auto_validate: bool = False,
        auto_tune: bool = False,
    ) -> None:
        """Initialize the instance."""
        super().__init__(config)
        self.z_threshold: float = self.config.get("z_threshold", 3.0)
        self.iqr_multiplier: float = self.config.get("iqr_multiplier", 1.5)

        # Whether the caller pinned an explicit decision threshold. When they
        # did not, the unsupervised detect() path derives a distribution-
        # adaptive operating point instead of the arbitrary fixed 0.5 cut
        # (see _adaptive_operating_point). An explicit threshold is always
        # honoured exactly, preserving backward compatibility.
        self._user_set_threshold: bool = isinstance(config, dict) and "threshold" in config

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

        # Data type detection (Task 2)
        self._data_type: DataCharacteristics = DataCharacteristics.UNKNOWN

        # Ensemble diversity metrics (Task 6)
        self._ensemble_diversity: dict[str, float] | None = None

        # Tuned hyperparameters (Task 5)
        self._tuned_lambda: float | None = None
        self._tuned_weights: np.ndarray[Any, Any] | None = None

        # Validation diagnostics (Task 4)
        self._validation_diagnostics: dict[str, Any] | None = None

        # Constructor options
        self._auto_validate = auto_validate
        self._auto_tune = auto_tune

        # Score flip for ensemble inversion (Task 4, set during fit if auto_validate=True)
        self._score_flip: bool = False

        # Adaptive component weights (set during fit)
        self._adaptive_weights: np.ndarray[Any, Any] = np.array([0.40, 0.30, 0.30])
        self._weight_source: str = "default"
        # Post-hoc info-geometry component certificate. Optional and DEFAULT-OFF
        # (Invariant I2): when off the detector behaves exactly as before this
        # feature existed — no ``info_geometry_certificate`` key is emitted and
        # nothing about scores/threshold/verdict changes. The legacy
        # ``fusion_certificates_enabled`` spelling is still honoured as a
        # deprecated alias so existing opt-in callers keep working.
        self._info_geometry_certificate_enabled: bool = bool(
            self.config.get(
                "info_geometry_certificate_enabled",
                self.config.get("fusion_certificates_enabled", False),
            )
        )

        # Conformal split-calibrated operating point (Item 4). Optional and
        # DEFAULT-OFF (Invariant I2): when off, supervised threshold calibration
        # is byte-for-byte the existing Youden/F1 pipeline. When on AND labels
        # are supplied, the decision threshold is the class-1 LAC conformal
        # quantile from a strict calibration split (no peeking at eval).
        self._conformal_operating_point_enabled: bool = bool(
            self.config.get("conformal_operating_point", False)
        )
        self._conformal_coverage: float = float(self.config.get("conformal_coverage", 0.90))

        # Beta-MCA monotone calibration (Stage 2, R1). Optional and DEFAULT-OFF
        # (Invariant I2): default "identity" -> detect() output is byte-identical.
        # When "mca" AND labels are supplied to fit_with_calibration_subset, an
        # accept-gated monotone beta map is fit; detect() then ADDS a
        # "calibrated_probabilities" key (rank-preserving -> AUROC exact tie),
        # leaving "scores" / "is_anomaly" untouched (exact-reducing).
        self._calibration_map: str = str(self.config.get("calibration_map", "identity"))
        self._mca_calibrator: Any = None

        # Oracle detector (set during fit if data is temporal)
        self._oracle_detector: Any = None
        self._oracle_metadata: dict[str, Any] = {"active": False}

    # =====================================================================
    # fit()
    # =====================================================================

    def fit(
        self,
        data: np.ndarray[Any, Any] | torch.Tensor,
        calibration_labels: np.ndarray[Any, Any] | None = None,
    ) -> MercuryAnomalyDetector:
        """Fit detector on training data.

        Computes statistical baselines for all three ensemble components:
          1. Distributional statistics (mean, std, quartiles)
          2. Kinematic baselines (jerk/acceleration mean and std per feature)
          3. Information-geometric manifold (mean, regularized precision matrix)

        If *calibration_labels* is provided (even for a subset of training
        data), the supervised threshold calibration pipeline is automatically
        invoked to set ``self._supervised_threshold``.

        Args:
            data: Training data array or tensor, shape ``(n_samples,)`` or
                ``(n_samples, n_features)``.
            calibration_labels: Optional binary labels (0=normal, 1=anomaly)
                with the same length as *data*.  When provided, the
                supervised adaptive weighting and threshold calibration
                pipeline are automatically invoked.

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

        # --- Data type detection (Task 2) ---
        self._data_type = self._detect_data_characteristics(arr)
        logger.info("fit: detected data type=%s", self._data_type.value)

        # --- InfoGeometry: fit Gaussian manifold ---
        self._fit_info_geometry(arr)

        # --- Kinematics: compute baseline jerk/acceleration per feature ---
        self._fit_kinematic_baseline(arr)

        # Store training data and precompute spectral profiles for resonance
        self._train_data = arr.copy()
        self._precompute_resonance_profiles(arr)

        self._is_fitted = True

        # --- Oracle initialization (for temporal data) ---
        self._oracle_detector = None
        self._oracle_metadata = {"active": False}

        if self._data_type == DataCharacteristics.TEMPORAL:
            try:
                from omni_mercury_engine.core.config import (
                    ORACLE_DOMAIN_POLICY,
                    OracleActivation,
                )
                from omni_mercury_engine.detectors.spectral_domain_frequency import (
                    SpectralDomainFrequency,
                )

                oracle_mode = OracleActivation.AUTO
                oracle_domain = self._infer_oracle_domain(arr, self._data_type)

                should_init = oracle_mode == OracleActivation.ENABLED or (
                    oracle_mode == OracleActivation.AUTO
                    and ORACLE_DOMAIN_POLICY.get(oracle_domain, "disabled") != "disabled"
                )

                if should_init:
                    oracle_cfg = {"domain": oracle_domain}
                    self._oracle_detector = SpectralDomainFrequency(oracle_cfg)
                    self._oracle_detector.fit(arr)
                    logger.info("Oracle fitted: domain=%s", oracle_domain)
            except Exception as exc:
                logger.debug("Oracle init skipped: %s", exc)
                self._oracle_detector = None

        # --- Unsupervised adaptive weighting (Task 1) ---
        self._adaptive_weights = self._compute_unsupervised_adaptive_weights(arr)
        logger.info(
            "fit: unsupervised adaptive weights=[%.3f, %.3f, %.3f] source=%s",
            self._adaptive_weights[0],
            self._adaptive_weights[1],
            self._adaptive_weights[2],
            self._weight_source,
        )

        # --- Ensemble diversity metrics (Task 6) ---
        self._ensemble_diversity = self._compute_ensemble_diversity(arr)
        if self._ensemble_diversity["mean_correlation"] > 0.9:
            logger.warning(
                "fit: high mean component correlation (%.3f) — ensemble diversity is low",
                self._ensemble_diversity["mean_correlation"],
            )

        # --- Optional auto-tuning (Task 5) ---
        if self._auto_tune:
            self.auto_tune(arr)

        # --- Optional auto-validation (Task 4) ---
        if self._auto_validate:
            diag = self.validate()
            if diag["is_inverted"]:
                logger.warning(
                    "fit: ensemble inversion detected (AUC=%.3f). "
                    "Applying score flip as recommended action.",
                    diag["ensemble_auc"],
                )
                self._score_flip = True
            else:
                self._score_flip = False

        # --- Calibration labels support (Task 3) ---
        if calibration_labels is not None:
            cal_labels = np.asarray(calibration_labels, dtype=np.int32).ravel()
            if len(cal_labels) == len(arr):
                # Full labeling: use supervised adaptive weights + threshold
                self._adaptive_weights = self._compute_adaptive_weights(arr, cal_labels)
                self._weight_source = "supervised_calibration"
                logger.info(
                    "fit: calibration_labels provided (n=%d). "
                    "Supervised adaptive weights=[%.3f, %.3f, %.3f]",
                    len(cal_labels),
                    self._adaptive_weights[0],
                    self._adaptive_weights[1],
                    self._adaptive_weights[2],
                )
                # Compute supervised threshold
                detection = self.detect(arr)
                scores = np.asarray(detection["scores"], dtype=np.float64)
                try:
                    from omni_mercury_engine.core.calibration_pipeline import (
                        CalibrationStrategy,
                        ThresholdCalibrationPipeline,
                    )

                    best_f1 = -1.0
                    best_threshold = float(np.median(scores))
                    for strat in [
                        CalibrationStrategy.YOUDEN_J,
                        CalibrationStrategy.F1_OPTIMAL,
                    ]:
                        try:
                            trial = ThresholdCalibrationPipeline()
                            result = trial.calibrate_from_data(
                                scores,
                                cal_labels,
                                method=strat,
                                threshold_name="anomaly.default_threshold",
                            )
                            preds = scores > result.threshold
                            tp = int(np.sum(preds & (cal_labels == 1)))
                            fp = int(np.sum(preds & (cal_labels == 0)))
                            fn = int(np.sum(~preds & (cal_labels == 1)))
                            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                            if f1 > best_f1:
                                best_f1 = f1
                                best_threshold = result.threshold
                                self._threshold_pipeline = trial
                                self._calibration_result = result
                        except Exception:
                            continue
                    self._supervised_threshold = best_threshold
                    logger.info(
                        "fit: supervised threshold=%.6f (F1=%.4f)",
                        best_threshold,
                        best_f1,
                    )
                except ImportError:
                    logger.debug("fit: calibration_pipeline not available")
            else:
                logger.warning(
                    "fit: calibration_labels length (%d) != data length (%d), ignoring",
                    len(cal_labels),
                    len(arr),
                )

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

        # --- Conformal split operating point (Item 4, opt-in DEFAULT-OFF) ---
        # When enabled, the decision threshold is the class-1 LAC conformal
        # quantile of the calibration labels (a distribution-free operating
        # point), instead of the Youden/F1 search. The detector was fit
        # unsupervised, so these labels are a valid calibration split.
        if self._conformal_operating_point_enabled:
            tau = self._conformal_operating_threshold(scores, labels)
            if tau is not None:
                self._supervised_threshold = tau
                self._calibration_method = "conformal_lac"
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
                _logger.debug("Calibration strategy %s failed, skipping", strat.value)
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

    def get_oracle_statistics(self) -> dict[str, Any] | None:
        """Export Oracle reference statistics for federation.

        Returns a serializable dict containing the Oracle domain,
        fitted reference stats, and configuration so that a receiving
        node can reconstruct Oracle state without re-fitting.

        Returns:
            Dict with Oracle state, or ``None`` if Oracle is not active.
        """
        if self._oracle_detector is None:
            return None
        try:
            oracle = self._oracle_detector
            oc = oracle._oracle_config
            ref_stats: dict[str, Any] = {
                "domain": getattr(oc, "domain", "environmental"),
                "threshold": getattr(oracle, "threshold", 0.5),
                "is_fitted": getattr(oracle, "_is_fitted", False),
            }
            # Export per-band reference statistics
            if hasattr(oracle, "_ref_band_means") and oracle._ref_band_means:
                ref_stats["ref_band_means"] = {
                    k: float(v) for k, v in oracle._ref_band_means.items()
                }
            if hasattr(oracle, "_ref_band_stds") and oracle._ref_band_stds:
                ref_stats["ref_band_stds"] = {k: float(v) for k, v in oracle._ref_band_stds.items()}
            # Export full-spectrum reference
            if hasattr(oracle, "_ref_full_spectrum_mean"):
                v = oracle._ref_full_spectrum_mean
                if isinstance(v, np.ndarray):
                    ref_stats["ref_full_spectrum_mean"] = v.tolist()
            if hasattr(oracle, "_ref_full_spectrum_std"):
                v = oracle._ref_full_spectrum_std
                if isinstance(v, np.ndarray):
                    ref_stats["ref_full_spectrum_std"] = v.tolist()
            if hasattr(oracle, "_ref_spectral_entropy_mean"):
                ref_stats["ref_spectral_entropy_mean"] = float(oracle._ref_spectral_entropy_mean)
            if hasattr(oracle, "_ref_spectral_entropy_std"):
                ref_stats["ref_spectral_entropy_std"] = float(oracle._ref_spectral_entropy_std)
            # Export noise color estimation (F1 Precision Directive)
            if hasattr(oracle, "_noise_beta"):
                ref_stats["noise_beta"] = float(oracle._noise_beta)
            if hasattr(oracle, "_noise_color"):
                ref_stats["noise_color"] = str(oracle._noise_color)
            if hasattr(oracle, "_noise_fit_r2"):
                ref_stats["noise_fit_r2"] = float(oracle._noise_fit_r2)
            return ref_stats
        except Exception as exc:
            logger.debug("Failed to export Oracle statistics: %s", exc)
            return None

    @classmethod
    def from_statistics(
        cls,
        mean: np.ndarray[Any, Any],
        std: np.ndarray[Any, Any],
        q1: np.ndarray[Any, Any],
        q3: np.ndarray[Any, Any],
        res_h_train: np.ndarray[Any, Any],
        res_noise_ratio: np.ndarray[Any, Any],
        kin_jerk_mean: np.ndarray[Any, Any],
        kin_jerk_std: np.ndarray[Any, Any],
        kin_accel_mean: np.ndarray[Any, Any],
        kin_accel_std: np.ndarray[Any, Any],
        ig_mean: np.ndarray[Any, Any],
        ig_cov_inv: np.ndarray[Any, Any],
        ig_log_det: float = 0.0,
        adaptive_weights: np.ndarray[Any, Any] | None = None,
        data_type: str | None = None,
        oracle_ref_stats: dict[str, Any] | None = None,
    ) -> MercuryAnomalyDetector:
        """Reconstruct a fitted detector from pre-computed statistics.

        This enables federated learning: nodes export statistics,
        the aggregator combines them, and this method creates a
        working detector from the aggregated result.

        All 13 core parameters correspond exactly to the attributes set
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
            adaptive_weights: Optional component weights from unsupervised
                adaptive weighting, shape (3,). If provided, the
                reconstructed detector uses these weights instead of the
                default [0.40, 0.30, 0.30].
            data_type: Optional detected data type ("temporal", "tabular",
                "image", "unknown"). Preserves the originating node's
                data-type classification.
            oracle_ref_stats: Optional Oracle reference statistics exported
                by :meth:`get_oracle_statistics`. When provided, the
                receiving node restores Oracle state so that frequency-
                domain scoring activates without re-fitting on local data.

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
        if adaptive_weights is not None:
            det._adaptive_weights = np.asarray(adaptive_weights)
        if data_type is not None:
            from omni_mercury_engine.core.config import DataCharacteristics

            det._data_type = DataCharacteristics(data_type)

        # Restore Oracle from federation stats
        det._restore_oracle_from_ref_stats(oracle_ref_stats)

        det._is_fitted = True
        return det

    def _restore_oracle_from_ref_stats(self, oracle_ref_stats: dict[str, Any] | None) -> None:
        """Re-arm the Oracle from exported reference statistics, or clear it.

        Shared by :meth:`from_statistics` (federation) and
        :meth:`set_fitted_state` (checkpoint round-trip): the Oracle is
        reconstructed from the stats :meth:`get_oracle_statistics` exports,
        without re-fitting on local data.
        """
        self._oracle_detector = None
        self._oracle_metadata = {"active": False}
        if oracle_ref_stats is None:
            return
        try:
            from omni_mercury_engine.detectors.spectral_domain_frequency import (
                SpectralDomainFrequency,
            )

            domain = oracle_ref_stats.get("domain", "environmental")
            oracle = SpectralDomainFrequency({"domain": domain})
            # Restore per-band reference statistics
            if "ref_band_means" in oracle_ref_stats:
                oracle._ref_band_means = oracle_ref_stats["ref_band_means"]
            if "ref_band_stds" in oracle_ref_stats:
                oracle._ref_band_stds = oracle_ref_stats["ref_band_stds"]
            if "ref_full_spectrum_mean" in oracle_ref_stats:
                oracle._ref_full_spectrum_mean = np.asarray(
                    oracle_ref_stats["ref_full_spectrum_mean"]
                )
            if "ref_full_spectrum_std" in oracle_ref_stats:
                oracle._ref_full_spectrum_std = np.asarray(
                    oracle_ref_stats["ref_full_spectrum_std"]
                )
            if "ref_spectral_entropy_mean" in oracle_ref_stats:
                oracle._ref_spectral_entropy_mean = float(
                    oracle_ref_stats["ref_spectral_entropy_mean"]
                )
            if "ref_spectral_entropy_std" in oracle_ref_stats:
                oracle._ref_spectral_entropy_std = float(
                    oracle_ref_stats["ref_spectral_entropy_std"]
                )
            # Restore noise color (F1 Precision Directive)
            if "noise_beta" in oracle_ref_stats:
                oracle._noise_beta = float(oracle_ref_stats["noise_beta"])
            if "noise_color" in oracle_ref_stats:
                oracle._noise_color = str(oracle_ref_stats["noise_color"])
            if "noise_fit_r2" in oracle_ref_stats:
                oracle._noise_fit_r2 = float(oracle_ref_stats["noise_fit_r2"])
            oracle._is_fitted = True
            self._oracle_detector = oracle
            self._oracle_metadata = {"active": True, "domain": domain}
            logger.info("Oracle restored from federation stats: domain=%s", domain)
        except Exception as exc:
            logger.debug("Failed to restore Oracle from federation: %s", exc)

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
    def _component_separation(scores: np.ndarray[Any, Any], labels: np.ndarray[Any, Any]) -> float:
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
        X: np.ndarray[Any, Any],
        labels: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
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
    # Data type detection (Task 2)
    # =====================================================================

    def _detect_data_characteristics(self, X: np.ndarray[Any, Any]) -> DataCharacteristics:
        """Automatically detect whether data is temporal, tabular, or image-like.

        Detection heuristics (applied in order):

        1. **Temporal autocorrelation**: Compute lag-1 autocorrelation for each
           feature. If mean |autocorrelation| > 0.3, classify as ``TEMPORAL``.
           Rationale: temporally ordered data exhibits serial dependence.

        2. **Row shuffling test**: Compute mean absolute correlation between
           adjacent rows. If mean correlation < 0.1, classify as ``TABULAR``
           (shuffled). Rationale: shuffled tabular rows have near-zero
           adjacent-row correlation.

        3. **Dimensionality heuristic**: If ``n_features > 100`` and
           ``n_features`` is approximately ``sqrt(n_samples)``, classify as
           ``IMAGE``. Rationale: image-like datasets have many features
           (pixels) and the feature count often relates to image dimensions.

        Falls back to ``UNKNOWN`` if no heuristic triggers.

        Args:
            X: Training data of shape ``(n_samples, n_features)``.

        Returns:
            Detected :class:`DataCharacteristics` enum value.
        """
        n_samples, n_features = X.shape

        if n_samples < 5:
            return DataCharacteristics.UNKNOWN

        # --- Heuristic 1: Temporal autocorrelation ---
        # Use *median* autocorrelation (robust to outlier spikes that
        # destroy mean autocorrelation in anomaly-injected temporal data).
        try:
            autocorrs: list[float] = []
            n_cols = min(n_features, 50)  # Cap for efficiency
            for f_idx in range(n_cols):
                col = X[:, f_idx]
                col_centered = col - np.mean(col)
                var = np.var(col)
                if var < _MIN_VARIANCE:
                    continue
                lag1_cov = np.mean(col_centered[:-1] * col_centered[1:])
                autocorrs.append(abs(lag1_cov / var))
            if autocorrs and np.median(autocorrs) > 0.3:
                return DataCharacteristics.TEMPORAL
        except (ValueError, TypeError, FloatingPointError, IndexError) as exc:
            logger.debug(
                "Data type detection: autocorrelation heuristic failed (%s), "
                "falling through to next heuristic.",
                exc,
            )

        # --- Heuristic 2: Adjacent row correlation ---
        # High adjacent-row correlation (> 0.3) also indicates temporal
        # ordering (rows are related to their neighbours).
        # Low adjacent-row correlation (< 0.1) indicates shuffled tabular.
        adj_row_corr: float | None = None
        try:
            if n_samples > 10 and n_features >= 2:
                row_corrs: list[float] = []
                n_check = min(n_samples - 1, 200)  # Cap for efficiency
                for i in range(n_check):
                    row_a = X[i, :]
                    row_b = X[i + 1, :]
                    std_a = np.std(row_a)
                    std_b = np.std(row_b)
                    if std_a < _MIN_VARIANCE or std_b < _MIN_VARIANCE:
                        continue
                    corr = np.corrcoef(row_a, row_b)[0, 1]
                    if np.isfinite(corr):
                        row_corrs.append(abs(corr))
                if row_corrs:
                    adj_row_corr = float(np.median(row_corrs))
                    if adj_row_corr > 0.3:
                        # High adjacent-row correlation: temporal ordering
                        return DataCharacteristics.TEMPORAL
                    if adj_row_corr < 0.1:
                        # Low adjacent-row correlation: shuffled tabular
                        return DataCharacteristics.TABULAR
        except (ValueError, TypeError, FloatingPointError, IndexError) as exc:
            logger.debug(
                "Data type detection: adjacent-row correlation heuristic failed (%s), "
                "falling through to next heuristic.",
                exc,
            )

        # --- Heuristic 3: Image dimensionality ---
        if n_features > 100:
            sqrt_n = np.sqrt(n_samples)
            if 0.3 * sqrt_n <= n_features <= 3.0 * sqrt_n:
                return DataCharacteristics.IMAGE

        # Default: if not temporal and not obviously image, assume tabular.
        # This is the conservative choice — tabular is the most common case
        # in ADBench-style datasets, and KinematicScore underperforms on it.
        if n_features <= 100:
            return DataCharacteristics.TABULAR

        return DataCharacteristics.UNKNOWN

    # =====================================================================
    # Oracle domain auto-selection (Phase 11)
    # =====================================================================

    @staticmethod
    def _infer_oracle_domain(
        X: np.ndarray[Any, Any],
        detected_type: DataCharacteristics,
    ) -> str:
        """Infer the most appropriate Oracle domain from data characteristics.

        Heuristics (applied in order):
        1. **Sample rate estimation**: If inter-sample intervals suggest
           < 1 Hz effective rate → ``environmental``, 100-500 Hz →
           ``medical``, > 500 Hz → ``infrastructure``.
        2. **Dominant FFT frequency**: Cross-reference against
           ``DOMAIN_FREQUENCY_BANDS`` to find best domain match.
        3. **Feature count heuristic**: 1-3 features → single-sensor
           (``environmental``), 20-100 → network (``security``).
        4. **Fallback**: ``environmental`` (broadest bands, safest default).

        User-specified domain always overrides this method.

        Args:
            X: Training data, shape ``(n_samples, n_features)``.
            detected_type: Result of ``_detect_data_characteristics()``.

        Returns:
            Domain string (e.g., ``"environmental"``, ``"medical"``).
        """
        n_samples, n_features = X.shape

        # Heuristic 3: Feature count
        if n_features >= 20:
            return "security"

        # Heuristic 1: Estimate effective sample rate from autocorrelation decay
        # (proxy for actual sample rate when timestamps aren't available)
        if detected_type == DataCharacteristics.TEMPORAL and n_samples >= 64:
            try:
                col = X[:, 0] if X.ndim > 1 else X
                col = col - np.mean(col)
                n = len(col)
                fft_vals = np.abs(np.fft.rfft(col))
                freqs = np.fft.rfftfreq(n)

                if len(fft_vals) > 1:
                    # Skip DC component
                    fft_vals = fft_vals[1:]
                    freqs = freqs[1:]
                    if len(fft_vals) > 0:
                        dominant_idx = int(np.argmax(fft_vals))
                        dominant_freq = float(freqs[dominant_idx])

                        # Map dominant normalised frequency to domain
                        # Very low freq -> environmental/climate
                        # Mid freq -> medical
                        # High freq -> infrastructure/security
                        if dominant_freq < 0.05:
                            return "environmental"
                        elif dominant_freq < 0.2:
                            return "medical"
                        elif dominant_freq < 0.4:
                            return "infrastructure"
                        else:
                            return "security"
            except (ValueError, IndexError):
                pass

        # Heuristic 3 (continued): Low feature count
        if n_features <= 3:
            return "environmental"
        if n_features <= 10:
            return "space"

        # Fallback to environmental (broadest bands)
        return "environmental"

    # =====================================================================
    # Unsupervised adaptive weighting (Task 1)
    # =====================================================================

    def _compute_unsupervised_adaptive_weights(
        self,
        X: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Compute adaptive ensemble weights without labels via self-supervised anomaly injection.

        Strategy:
          1. Split training data into K=5 folds.
          2. For each fold, fit on K-1 folds and score the held-out fold.
          3. Inject synthetic anomalies into the held-out fold (Gaussian noise
             with sigma = 3 * std per feature) to create pseudo-labels.
          4. Compute per-component AUC using pseudo-labels.
          5. Apply the same logic as ``_compute_adaptive_weights()``: zero out
             components with AUC < 0.5, apply 0.05 minimum floor.

        Additionally applies data-type-aware weight adjustment using
        :data:`COMPONENT_COMPATIBILITY` from ``config.py``.

        Args:
            X: Training data, shape ``(n_samples, n_features)``.

        Returns:
            Weight array of shape ``(3,)`` summing to 1.
        """
        n_samples, n_features = X.shape

        # For very small datasets, fall back to data-type-based defaults
        if n_samples < 20:
            return self._data_type_default_weights()

        try:
            k_folds = min(5, n_samples // 4)
            if k_folds < 2:
                return self._data_type_default_weights()

            rng = np.random.RandomState(42)
            indices = np.arange(n_samples)
            rng.shuffle(indices)
            fold_size = n_samples // k_folds

            component_aucs_accum: list[list[float]] = [[], [], []]

            for k in range(k_folds):
                val_start = k * fold_size
                val_end = val_start + fold_size if k < k_folds - 1 else n_samples
                val_idx = indices[val_start:val_end]
                train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

                X_train_fold = X[train_idx]
                X_val_fold = X[val_idx]

                if len(X_train_fold) < 4 or len(X_val_fold) < 2:
                    continue

                # Fit a temporary detector on the fold
                fold_det = MercuryAnomalyDetector()
                fold_det._data_type = self._data_type
                # Manually fit components without recursion
                fold_det.mean = np.mean(X_train_fold, axis=0)
                fold_det.std = np.std(X_train_fold, axis=0) + 1e-8
                fold_det.q1 = np.percentile(X_train_fold, 25, axis=0)
                fold_det.q3 = np.percentile(X_train_fold, 75, axis=0)
                fold_det._fit_info_geometry(X_train_fold)
                fold_det._fit_kinematic_baseline(X_train_fold)
                fold_det._train_data = X_train_fold.copy()
                fold_det._precompute_resonance_profiles(X_train_fold)
                fold_det._is_fitted = True

                # Generate synthetic anomalies via Gaussian noise injection
                n_anomalies = max(len(X_val_fold), 10)
                noise = rng.randn(n_anomalies, n_features) * 3.0 * fold_det.std
                synthetic_anomalies = fold_det.mean + noise

                # Combine normal (val) + synthetic anomalies
                X_combined = np.vstack([X_val_fold, synthetic_anomalies])
                pseudo_labels = np.concatenate(
                    [
                        np.zeros(len(X_val_fold), dtype=np.int32),
                        np.ones(n_anomalies, dtype=np.int32),
                    ]
                )

                # Score each component
                res_scores = fold_det._compute_resonance_score(X_combined)
                kin_scores = fold_det._compute_kinematic_score(X_combined)
                ig_scores = fold_det._compute_info_geometry_score(X_combined)

                for comp_idx, comp_scores in enumerate([res_scores, kin_scores, ig_scores]):
                    auc = self._component_separation(comp_scores, pseudo_labels)
                    component_aucs_accum[comp_idx].append(auc)

            # Aggregate AUCs across folds
            if not component_aucs_accum[0]:
                return self._data_type_default_weights()

            mean_aucs = np.array(
                [float(np.mean(aucs)) if aucs else 0.5 for aucs in component_aucs_accum]
            )

            self._component_aucs = {
                "resonance": float(mean_aucs[0]),
                "kinematic": float(mean_aucs[1]),
                "infogeo": float(mean_aucs[2]),
            }

            # Apply data-type compatibility modifiers
            compat = COMPONENT_COMPATIBILITY.get(
                self._data_type, COMPONENT_COMPATIBILITY[DataCharacteristics.UNKNOWN]
            )
            # If TABULAR, force kinematic weight to near-zero
            if self._data_type == DataCharacteristics.TABULAR:
                mean_aucs[1] = min(mean_aucs[1], 0.50)  # Cap at random

            # Standard adaptive weight logic: zero out inverted, normalize
            effective_aucs = np.where(mean_aucs < 0.5, 0.0, mean_aucs - 0.5)

            # Apply compatibility multipliers
            effective_aucs[0] *= compat["resonance"]
            effective_aucs[1] *= compat["kinematic"]
            effective_aucs[2] *= compat["infogeo"]

            total = effective_aucs.sum()
            if total < 1e-6:
                self._weight_source = "fallback_data_type"
                return self._data_type_default_weights()

            weights = effective_aucs / total
            # Apply minimum floor of 0.05 for components with positive signal
            has_signal = mean_aucs >= 0.5
            weights = np.where(has_signal & (weights < 0.05), 0.05, weights)
            # Zero out kinematic on tabular data explicitly
            if self._data_type == DataCharacteristics.TABULAR:
                weights[1] = 0.0

            wsum = weights.sum()
            if wsum > 0:
                weights = weights / wsum
            else:
                return self._data_type_default_weights()

            self._weight_source = "unsupervised_adaptive"
            return weights

        except Exception as exc:
            logger.debug(
                "Unsupervised adaptive weighting failed (%s), using data-type defaults",
                exc,
            )
            return self._data_type_default_weights()

    def _data_type_default_weights(self) -> np.ndarray[Any, Any]:
        """Return default component weights adjusted for detected data type.

        Uses :data:`COMPONENT_COMPATIBILITY` to compute data-type-aware
        default weights.  When data type is ``TABULAR``, kinematic weight
        is set to zero.

        Returns:
            Weight array of shape ``(3,)`` summing to 1.
        """
        compat = COMPONENT_COMPATIBILITY.get(
            self._data_type, COMPONENT_COMPATIBILITY[DataCharacteristics.UNKNOWN]
        )
        raw = np.array(
            [
                0.40 * compat["resonance"],
                0.30 * compat["kinematic"],
                0.30 * compat["infogeo"],
            ]
        )
        # Force kinematic to zero for tabular
        if self._data_type == DataCharacteristics.TABULAR:
            raw[1] = 0.0
        total = raw.sum()
        if total < 1e-6:
            self._weight_source = "fallback_default"
            return np.array([0.40, 0.30, 0.30])
        self._weight_source = "data_type_default"
        return raw / total

    # =====================================================================
    # Ensemble diversity metrics (Task 6)
    # =====================================================================

    def _compute_ensemble_diversity(
        self,
        X: np.ndarray[Any, Any],
    ) -> dict[str, float]:
        """Measure pairwise correlation between ensemble components.

        High correlation (>0.9) between two components indicates redundancy.
        When detected, the lower-AUC component's weight is reduced by 50%
        in the adaptive weighting.

        Args:
            X: Data to score, shape ``(n_samples, n_features)``.

        Returns:
            Dict with pairwise correlations and mean correlation:
            ``{"resonance_kinematic": float, "resonance_infogeo": float,
            "kinematic_infogeo": float, "mean_correlation": float}``
        """
        try:
            res = self._compute_resonance_score(X)
            kin = self._compute_kinematic_score(X)
            ig = self._compute_info_geometry_score(X)

            def _safe_corr(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> float:
                if np.std(a) < 1e-10 or np.std(b) < 1e-10:
                    return 0.0
                corr = np.corrcoef(a, b)[0, 1]
                return float(corr) if np.isfinite(corr) else 0.0

            rk = _safe_corr(res, kin)
            ri = _safe_corr(res, ig)
            ki = _safe_corr(kin, ig)
            mean_corr = (abs(rk) + abs(ri) + abs(ki)) / 3.0

            diversity = {
                "resonance_kinematic": rk,
                "resonance_infogeo": ri,
                "kinematic_infogeo": ki,
                "mean_correlation": mean_corr,
            }

            # Apply redundancy penalty to adaptive weights if they exist
            if hasattr(self, "_adaptive_weights") and hasattr(self, "_component_aucs"):
                aucs = self._component_aucs
                pairs = [
                    (0, 1, abs(rk), "resonance", "kinematic"),
                    (0, 2, abs(ri), "resonance", "infogeo"),
                    (1, 2, abs(ki), "kinematic", "infogeo"),
                ]
                modified = False
                for idx_a, idx_b, corr_val, name_a, name_b in pairs:
                    if corr_val > 0.9:
                        # Reduce weight of lower-AUC component by 50%
                        auc_a = aucs.get(name_a, 0.5)
                        auc_b = aucs.get(name_b, 0.5)
                        lower_idx = idx_a if auc_a < auc_b else idx_b
                        self._adaptive_weights[lower_idx] *= 0.5
                        modified = True
                        logger.info(
                            "Diversity: %s-%s correlation=%.3f, reducing %s weight by 50%%",
                            name_a,
                            name_b,
                            corr_val,
                            name_a if lower_idx == idx_a else name_b,
                        )
                if modified:
                    wsum = self._adaptive_weights.sum()
                    if wsum > 0:
                        self._adaptive_weights = self._adaptive_weights / wsum

            return diversity

        except Exception as exc:
            logger.debug("Ensemble diversity computation failed: %s", exc)
            return {
                "resonance_kinematic": 0.0,
                "resonance_infogeo": 0.0,
                "kinematic_infogeo": 0.0,
                "mean_correlation": 0.0,
            }

    # =====================================================================
    # Per-component validation and diagnostics (Task 4)
    # =====================================================================

    def validate(
        self,
        X: np.ndarray[Any, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate ensemble behaviour and detect inversion.

        If *X* is ``None``, generates synthetic anomalies from the training
        distribution (uniform samples over feature ranges) and compares
        against training data scores.

        Args:
            X: Optional test data. If ``None``, uses synthetic anomalies
               generated from the training distribution.

        Returns:
            Diagnostics dict with keys:
              - ``ensemble_auc``: float — AUC of ensemble on normal vs synthetic.
              - ``component_aucs``: dict — Per-component AUC.
              - ``is_inverted``: bool — True if ensemble AUC < 0.5.
              - ``recommended_action``: str — Suggested fix if inverted.
              - ``data_type``: str — Detected data characteristics.
              - ``weights``: list — Current adaptive weights.
        """
        if not self._is_fitted or self._train_data is None:
            return {
                "ensemble_auc": 0.5,
                "component_aucs": {},
                "is_inverted": False,
                "recommended_action": "Fit detector first",
                "data_type": self._data_type.value,
                "weights": [0.40, 0.30, 0.30],
            }

        train_data = self._train_data
        n_samples, n_features = train_data.shape

        # Generate synthetic anomalies from uniform distribution over feature ranges
        rng = np.random.RandomState(42)
        n_synthetic = min(n_samples, 500)
        feature_min = np.min(train_data, axis=0)
        feature_max = np.max(train_data, axis=0)
        feature_range = feature_max - feature_min + 1e-8
        # Generate OOD samples: extend beyond training range
        synthetic = (
            feature_min
            - 0.5 * feature_range
            + rng.rand(n_synthetic, n_features) * 2.0 * feature_range
        )

        if X is not None:
            normal_data = X
        else:
            normal_data = train_data

        # Combine and create pseudo-labels
        X_eval = np.vstack([normal_data[:n_synthetic], synthetic])
        y_eval = np.concatenate(
            [
                np.zeros(min(len(normal_data), n_synthetic), dtype=np.int32),
                np.ones(n_synthetic, dtype=np.int32),
            ]
        )

        # Score with current detector
        detection = self.detect(X_eval)
        scores = np.asarray(detection["scores"])

        # Per-component AUCs
        res_scores = np.asarray(detection["resonance_scores"])
        kin_scores = np.asarray(detection["kinematic_scores"])
        ig_scores = np.asarray(detection["info_geometry_scores"])

        ensemble_auc = self._component_separation(scores, y_eval)
        component_aucs = {
            "resonance": self._component_separation(res_scores, y_eval),
            "kinematic": self._component_separation(kin_scores, y_eval),
            "infogeo": self._component_separation(ig_scores, y_eval),
        }

        is_inverted = ensemble_auc < 0.5

        if is_inverted:
            recommended = (
                "Ensemble inversion detected. Options: "
                "(1) Flip scores: scores = 1.0 - scores; "
                "(2) Fall back to single best component; "
                "(3) Use supervised calibration if labels available."
            )
        else:
            recommended = "No action needed — ensemble is performing correctly."

        weights = self._adaptive_weights

        diagnostics: dict[str, Any] = {
            "ensemble_auc": float(ensemble_auc),
            "component_aucs": component_aucs,
            "is_inverted": is_inverted,
            "recommended_action": recommended,
            "data_type": self._data_type.value,
            "weights": weights.tolist(),
        }

        self._validation_diagnostics = diagnostics
        return diagnostics

    # =====================================================================
    # Enhanced supervised calibration (Task 3)
    # =====================================================================

    def fit_with_calibration_subset(
        self,
        X: np.ndarray[Any, Any],
        calibration_indices: np.ndarray[Any, Any],
        calibration_labels: np.ndarray[Any, Any],
    ) -> MercuryAnomalyDetector:
        """Fit on full data and calibrate threshold using a labeled subset.

        Convenience method that:
          1. Fits the detector on the full dataset *X* via ``fit()``.
          2. Uses the labeled calibration subset to compute a supervised
             threshold via :class:`ThresholdCalibrationPipeline`.
          3. Stores the threshold in ``self._supervised_threshold``.

        This is useful when only a small fraction of data has labels
        (e.g., active learning or partial labeling scenarios).

        Args:
            X: Full training data, shape ``(n_samples, n_features)``.
            calibration_indices: Integer indices into *X* for the labeled
                calibration subset.
            calibration_labels: Binary labels (0=normal, 1=anomaly) for the
                calibration subset. Must have the same length as
                *calibration_indices*.

        Returns:
            Self for method chaining.
        """
        self.fit(X)

        cal_indices = np.asarray(calibration_indices).ravel()
        cal_labels = np.asarray(calibration_labels, dtype=np.int32).ravel()

        if len(cal_indices) != len(cal_labels):
            raise ValueError(
                f"calibration_indices ({len(cal_indices)}) and "
                f"calibration_labels ({len(cal_labels)}) must have the same length"
            )

        X_cal = X[cal_indices]

        # Compute adaptive weights using labeled subset
        self._adaptive_weights = self._compute_adaptive_weights(X_cal, cal_labels)

        # Score the calibration subset
        detection = self.detect(X_cal)
        scores = np.asarray(detection["scores"], dtype=np.float64)

        # Beta-MCA monotone calibration (Stage 2, R1), opt-in DEFAULT-OFF. Fit the
        # accept-gated map on the calibration scores; it can never regress
        # Brier/ECE (else it falls back to identity). detect() exposes it as an
        # additive "calibrated_probabilities" key without touching scores/verdict.
        if self._calibration_map == "mca":
            from omni_mercury_engine.core.calibration import fit_accept_gated_mca

            self._mca_calibrator, _ = fit_accept_gated_mca(scores, cal_labels)

        # Conformal split operating point (Item 4, opt-in DEFAULT-OFF).
        if self._conformal_operating_point_enabled:
            tau = self._conformal_operating_threshold(scores, cal_labels)
            if tau is not None:
                self._supervised_threshold = tau
                self._calibration_method = "conformal_lac"
                logger.info(
                    "fit_with_calibration_subset: conformal LAC threshold=%.6f (n_cal=%d)",
                    tau,
                    len(cal_indices),
                )
                return self

        # Calibrate threshold
        from omni_mercury_engine.core.calibration_pipeline import (
            CalibrationStrategy,
            ThresholdCalibrationPipeline,
        )

        best_f1 = -1.0
        best_threshold = float(np.median(scores))

        for strat in [CalibrationStrategy.YOUDEN_J, CalibrationStrategy.F1_OPTIMAL]:
            try:
                trial = ThresholdCalibrationPipeline()
                result = trial.calibrate_from_data(
                    scores,
                    cal_labels,
                    method=strat,
                    threshold_name="anomaly.default_threshold",
                )
                preds = scores > result.threshold
                tp = int(np.sum(preds & (cal_labels == 1)))
                fp = int(np.sum(preds & (cal_labels == 0)))
                fn = int(np.sum(~preds & (cal_labels == 1)))
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = result.threshold
                    self._threshold_pipeline = trial
                    self._calibration_result = result
            except Exception:
                continue

        self._supervised_threshold = best_threshold
        logger.info(
            "fit_with_calibration_subset: threshold=%.6f (cal F1=%.4f, n_cal=%d)",
            best_threshold,
            best_f1,
            len(cal_indices),
        )
        return self

    def _conformal_operating_threshold(
        self,
        scores: np.ndarray[Any, Any],
        labels: np.ndarray[Any, Any],
    ) -> float | None:
        """Class-1 LAC conformal operating point on a labelled calibration set.

        Returns the score threshold ``1 - q_1`` such that ``score >= threshold``
        flags an anomaly with the conformal class-1 coverage guarantee, or
        ``None`` when calibration is impossible (only one class present, or the
        class-1 quantile is degenerate). Returning ``None`` lets the caller fall
        back to the existing Youden/F1 path rather than mis-calibrate.
        """
        from omni_mercury_engine.core.conformal_prediction import (
            BinaryConformalClassifier,
        )

        s = np.asarray(scores, dtype=np.float64).reshape(-1)
        y = np.asarray(labels, dtype=int).reshape(-1)
        if s.size != y.size or np.unique(y).size < 2:
            return None
        clf = BinaryConformalClassifier(coverage=self._conformal_coverage, seed=42)
        clf.fit(s, y)
        tau = clf.anomaly_score_threshold()
        if not np.isfinite(tau):
            return None
        return float(tau)

    # =====================================================================
    # Automated hyperparameter tuning (Task 5)
    # =====================================================================

    def auto_tune(
        self,
        X: np.ndarray[Any, Any],
        labels: np.ndarray[Any, Any] | None = None,
        n_trials: int = 50,
    ) -> MercuryAnomalyDetector:
        """Optimize hyperparameters using Optuna (optional dependency).

        When labels are provided, maximizes AUC. When unsupervised,
        maximizes ensemble diversity (minimizes mean pairwise component
        correlation).

        Tunable parameters:
          - Tikhonov regularization lambda
          - Component weights (if labels available)
          - MAD threshold multiplier (if using MAD calibration)

        Args:
            X: Training data, shape ``(n_samples, n_features)``.
            labels: Optional binary ground-truth labels.
            n_trials: Number of Optuna optimization trials (default 50).

        Returns:
            Self for method chaining.

        Raises:
            ImportError: If optuna is not installed.
        """
        if not OPTUNA_AVAILABLE:
            logger.warning("auto_tune: optuna not installed. Install with: pip install optuna")
            return self

        if not self._is_fitted:
            logger.warning("auto_tune: detector must be fitted first")
            return self

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            # Tune Tikhonov lambda
            trial_lambda = trial.suggest_float("tikhonov_lambda", 1e-8, 1e-2, log=True)

            # Re-fit info geometry with trial lambda
            n_samples_ig, n_features_ig = X.shape
            ig_mean = np.mean(X, axis=0)
            cov = np.cov(X.T, ddof=1) if n_samples_ig > 1 else np.eye(n_features_ig)
            if cov.ndim == 0:
                cov = np.atleast_2d(cov)
            reg_lambda = trial_lambda
            if n_samples_ig <= n_features_ig:
                reg_lambda = max(trial_lambda, 1.0 / max(n_samples_ig, 1))
            cov_reg = cov + reg_lambda * np.eye(n_features_ig, dtype=cov.dtype)

            try:
                cho = sp_linalg.cholesky(cov_reg, lower=True)
                ig_cov_inv = sp_linalg.cho_solve((cho, True), np.eye(n_features_ig))
            except sp_linalg.LinAlgError:
                ig_cov_inv = np.linalg.pinv(cov_reg)
            ig_cov_inv = 0.5 * (ig_cov_inv + ig_cov_inv.T)

            # Temporarily override info geometry parameters
            orig_ig_mean = self._ig_mean
            orig_ig_cov_inv = self._ig_cov_inv
            self._ig_mean = ig_mean
            self._ig_cov_inv = ig_cov_inv

            if labels is not None:
                # Supervised: maximize AUC
                w0 = trial.suggest_float("w_resonance", 0.1, 0.8)
                w1 = trial.suggest_float("w_kinematic", 0.0, 0.5)
                w2 = 1.0 - w0 - w1
                if w2 < 0.0:
                    w2 = 0.0
                    w_total = w0 + w1
                    w0 /= w_total
                    w1 /= w_total
                trial_weights = np.array([w0, w1, w2])
                trial_weights = trial_weights / trial_weights.sum()

                orig_weights = self._adaptive_weights
                self._adaptive_weights = trial_weights

                detection = self.detect(X)
                scores = np.asarray(detection["scores"])
                lab = np.asarray(labels, dtype=np.int32).ravel()
                auc_val = self._component_separation(scores, lab)

                self._adaptive_weights = orig_weights
                self._ig_mean = orig_ig_mean
                self._ig_cov_inv = orig_ig_cov_inv
                return auc_val
            else:
                # Unsupervised: maximize diversity (minimize correlation)
                res = self._compute_resonance_score(X)
                ig = self._compute_info_geometry_score(X)

                self._ig_mean = orig_ig_mean
                self._ig_cov_inv = orig_ig_cov_inv

                def _corr(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> float:
                    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
                        return 0.0
                    c = np.corrcoef(a, b)[0, 1]
                    return float(c) if np.isfinite(c) else 0.0

                # Lower correlation = better diversity = higher value
                mean_corr = abs(_corr(res, ig))
                return 1.0 - mean_corr

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best = study.best_params
        self._tuned_lambda = best.get("tikhonov_lambda", _TIKHONOV_LAMBDA)

        if labels is not None and "w_resonance" in best:
            w0 = best["w_resonance"]
            w1 = best["w_kinematic"]
            w2 = max(0.0, 1.0 - w0 - w1)
            self._tuned_weights = np.array([w0, w1, w2])
            self._tuned_weights = self._tuned_weights / self._tuned_weights.sum()
            self._adaptive_weights = self._tuned_weights
            self._weight_source = "auto_tuned"

        # Re-fit info geometry with tuned lambda
        self._fit_info_geometry_with_lambda(X, self._tuned_lambda)

        logger.info(
            "auto_tune: best_lambda=%.2e, best_value=%.4f",
            self._tuned_lambda,
            study.best_value,
        )
        return self

    def _fit_info_geometry_with_lambda(
        self,
        data: np.ndarray[Any, Any],
        reg_lambda: float,
    ) -> None:
        """Re-fit info geometry with a specific Tikhonov lambda value.

        Args:
            data: Training data (n_samples, n_features).
            reg_lambda: Tikhonov regularization parameter.
        """
        n_samples, n_features = data.shape
        self._ig_mean = np.mean(data, axis=0)
        if n_samples < 2:
            self._ig_cov_inv = np.eye(n_features, dtype=np.float64)
            self._ig_log_det = 0.0
            return
        cov = np.cov(data.T, ddof=1)
        if cov.ndim == 0:
            cov = np.atleast_2d(cov)
        if n_samples <= n_features:
            reg_lambda = max(reg_lambda, 1.0 / max(n_samples, 1))
        cov_reg = cov + reg_lambda * np.eye(n_features, dtype=cov.dtype)
        try:
            cho = sp_linalg.cholesky(cov_reg, lower=True)
            self._ig_cov_inv = sp_linalg.cho_solve((cho, True), np.eye(n_features))
            self._ig_log_det = float(2.0 * np.sum(np.log(np.diag(cho))))
        except sp_linalg.LinAlgError:
            self._ig_cov_inv = np.linalg.pinv(cov_reg)
            sign, logdet = np.linalg.slogdet(cov_reg)
            self._ig_log_det = float(logdet) if sign > 0 else 0.0
        self._ig_cov_inv = 0.5 * (self._ig_cov_inv + self._ig_cov_inv.T)

    # =====================================================================
    # Conformal prediction uncertainty bands (Task 10)
    # =====================================================================

    def predict_with_uncertainty(
        self,
        data: np.ndarray[Any, Any],
        alpha: float = 0.1,
    ) -> dict[str, Any]:
        """Detect anomalies and provide confidence intervals on scores.

        Extends ``detect()`` with uncertainty bands computed via conformal
        prediction on the training score distribution.

        When a conformal predictor is fitted (via ``fit_with_labels()``
        with ``strategy="mondrian"``), uses per-group intervals. Otherwise,
        bootstraps from the training score variance.

        Args:
            data: Input data array, shape ``(n_samples, n_features)``.
            alpha: Significance level (default 0.1 for 90% confidence).

        Returns:
            Standard ``detect()`` dict augmented with:
              - ``uncertainty_lower``: Lower confidence bound on scores.
              - ``uncertainty_upper``: Upper confidence bound on scores.
              - ``uncertainty_width``: Width of confidence interval.
        """
        result = self.detect(data)
        scores = np.asarray(result["scores"], dtype=np.float64)

        # Compute uncertainty from training score distribution
        if self._train_data is not None and self._is_fitted:
            train_detection = self.detect(self._train_data)
            train_scores = np.asarray(train_detection["scores"], dtype=np.float64)
            score_std = np.std(train_scores)
            z_val = sp_stats.norm.ppf(1 - alpha / 2)
            half_width = float(z_val * score_std)
        else:
            half_width = 0.1  # Fallback

        lower = np.clip(scores - half_width, 0.0, 1.0)
        upper = np.clip(scores + half_width, 0.0, 1.0)
        width = upper - lower

        # Warn about high uncertainty
        high_uncertainty_frac = float(np.mean(width > 0.3))
        if high_uncertainty_frac > 0.1:
            logger.warning(
                "predict_with_uncertainty: %.1f%% of predictions have "
                "uncertainty width > 0.3 (borderline predictions)",
                high_uncertainty_frac * 100,
            )

        result["uncertainty_lower"] = lower
        result["uncertainty_upper"] = upper
        result["uncertainty_width"] = width
        return result

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
        weights = self._adaptive_weights.copy()

        # Domain preset blending (F1 Precision Directive, Phase 3):
        # Blend adaptive weights with domain prior (60% data-driven, 40% domain)
        domain = getattr(self, "_benchmark_domain", None)
        if domain:
            from omni_mercury_engine.core.domain_weight_presets import get_domain_preset

            prior_weights = np.array(get_domain_preset(domain))
            weights = 0.6 * weights + 0.4 * prior_weights
            weights = weights / weights.sum()

        # Pairwise inversion guard (F1 Precision Directive, Phase 2):
        # If a component is anti-correlated with BOTH other components (Spearman rho < -0.2),
        # zero its weight. This catches inversions the self-supervised approach misses.
        if len(data) >= 30:
            components = [
                (resonance, "resonance", 0),
                (kinematic, "kinematic", 1),
                (info_geo, "info_geometry", 2),
            ]
            active_weights = weights.copy()
            for idx, (comp_scores, comp_name, wi) in enumerate(components):
                if active_weights[wi] < 0.01:
                    continue
                other_idxs = [j for j in range(3) if j != idx]
                other_scores = [components[j][0] for j in other_idxs]
                rho_vals = []
                for other in other_scores:
                    try:
                        # ``spearmanr`` on a constant array emits a
                        # ``ConstantInputWarning`` and returns NaN —
                        # which the next branch already maps to 0.0.
                        # Detect the degenerate case ahead of time so
                        # the warning is never emitted at all.
                        if np.ptp(comp_scores) == 0.0 or np.ptp(other) == 0.0:
                            rho = 0.0
                        else:
                            rho_result, _ = sp_stats.spearmanr(comp_scores, other)
                            rho = 0.0 if np.isnan(rho_result) else float(rho_result)
                    except Exception:
                        rho = 0.0
                    rho_vals.append(rho)
                if all(r < -0.2 for r in rho_vals):
                    logger.info(
                        "Inversion guard: %s rhos=[%.3f, %.3f] — zeroing",
                        comp_name,
                        rho_vals[0],
                        rho_vals[1],
                    )
                    active_weights[wi] = 0.0
            wsum = active_weights.sum()
            if wsum > 0:
                active_weights = active_weights / wsum
            else:
                active_weights = np.array([0.4, 0.0, 0.6])
            weights = active_weights

        combined_scores = weights[0] * resonance + weights[1] * kinematic + weights[2] * info_geo
        combined_scores = np.clip(combined_scores, 0.0, 1.0)

        # Unsupervised ensemble flip (F1 Precision Directive, Phase 2):
        # If median score > 0.80, scores are likely inverted (most points shouldn't be anomalous).
        if len(combined_scores) >= 50:
            median_score = float(np.median(combined_scores))
            if median_score > 0.80:
                combined_scores = 1.0 - combined_scores
                logger.info("Ensemble flip: median=%.3f, inverting scores", median_score)

        # --- Oracle spectral influence ---
        oracle_meta: dict[str, Any] = {"active": False}
        if self._oracle_detector is not None:
            try:
                # Dynamic Oracle sensitivity (F1 Precision Directive, Phase 9):
                # High initial severity → look harder for spectral confirmation
                initial_severity = float(np.mean(combined_scores))
                severity_factor = 1.0 + (initial_severity - 0.3) * 2.0
                severity_factor = float(np.clip(severity_factor, 0.5, 3.0))
                self._oracle_detector._dynamic_alpha_factor = severity_factor

                oracle_result = self._oracle_detector.detect(data)
                # Extract influence multiplier from the influence_vector object
                # (Oracle returns per-signal analysis, not per-sample)
                iv = oracle_result.get("influence_vector")
                if iv is not None and hasattr(iv, "influence_multiplier"):
                    scalar = float(iv.influence_multiplier)
                    multiplier = np.full(len(data), scalar)
                else:
                    multiplier = np.ones(len(data))
                combined_scores = combined_scores * multiplier
                oracle_meta = {
                    "active": True,
                    "domain": getattr(
                        getattr(self._oracle_detector, "_oracle_config", None),
                        "domain",
                        "unknown",
                    ),
                    "mean_multiplier": float(np.mean(multiplier)),
                    "significant_bands": [
                        b.band_label
                        for b in oracle_result.get("band_results", [])
                        if getattr(b, "is_significant", False)
                    ],
                    "change_points": oracle_result.get("n_change_points", 0),
                }
            except Exception as exc:
                logger.debug("Oracle detect failed: %s", exc)
        self._oracle_metadata = oracle_meta

        combined_scores = np.clip(combined_scores, 0.0, 1.0)

        # Residual frequency filter (F1 Precision Directive, Phase 7):
        # Only apply to temporal-like data where FFT filtering is meaningful.
        is_temporal_like = data.shape[0] >= max(50, 10 * data.shape[1])
        if is_temporal_like and len(combined_scores) >= 32:
            combined_scores = self._residual_frequency_filter(combined_scores)

        # Score flip for detected ensemble inversion (Task 4)
        if self._score_flip:
            combined_scores = 1.0 - combined_scores

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
        elif not self._user_set_threshold:
            # No conformal predictor, supervised label, or explicit auto-
            # calibration was configured, and the caller did not pin a
            # threshold. A fixed 0.5 cut on the ensemble's compressed [0, 1]
            # scores is an arbitrary operating point (strong ranking, broken
            # threshold: high AUROC / near-zero F1). Derive the operating
            # point from the score distribution itself. This is rank-
            # preserving (AUROC/AUPRC unchanged) — only the cut location moves.
            effective_threshold, calibration_diagnostics = self._adaptive_operating_point(
                combined_scores
            )
            is_anomaly = combined_scores > effective_threshold
        else:
            is_anomaly = combined_scores > effective_threshold

        # Legacy backward-compatibility keys
        iqr_anomalies = self._detect_iqr_anomalies(data)

        result = {
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
            "oracle_metadata": oracle_meta,
        }
        if self._info_geometry_certificate_enabled:
            certificate = self._info_geometry_certificate_payload(data, info_geo)
            if certificate is not None:
                result["info_geometry_certificate"] = certificate
        # Beta-MCA calibrated probabilities (Stage 2, R1) — additive, opt-in.
        # Rank-preserving (AUROC exact tie); scores/is_anomaly untouched.
        if self._calibration_map == "mca" and self._mca_calibrator is not None:
            result["calibrated_probabilities"] = self._mca_calibrator.calibrate(combined_scores)
        return result

    def _info_geometry_certificate_payload(
        self,
        data: np.ndarray[Any, Any],
        info_geo_scores: np.ndarray[Any, Any],
    ) -> dict[str, Any] | None:
        """Post-hoc certificate for the **info-geometry component** boundary.

        ``p_tau`` is derived from the info-geometry component's *own* operating
        threshold (the adaptive cut on that component's score distribution),
        inverted through the component's real score map ``g`` — not from the
        ensemble operating point. The certified radius is therefore the sound
        L2 radius within which this component's price cannot cross its own
        boundary. It certifies that component's price level-set only, never the
        fused or gated verdict.
        """
        if self._ig_mean is None or self._ig_cov_inv is None:
            return None
        x = np.asarray(data, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.shape[1] != len(self._ig_mean):
            return None
        # The component's own operating point, in the component's score space.
        component_threshold, _ = self._adaptive_operating_point(
            np.asarray(info_geo_scores, dtype=np.float64).reshape(-1)
        )
        p_tau = mahalanobis_score_to_price_threshold(component_threshold, x.shape[1])
        cert = InfoGeometryCertificate(self._ig_mean, self._ig_cov_inv, p_tau)
        payload = cert.certify(x)
        info_geo = np.asarray(info_geo_scores, dtype=np.float64).reshape(-1)
        return {
            "model": "information_geometry_mahalanobis",
            "certifies": (
                "info_geometry component price level-set; " "NOT the fused/gated verdict"
            ),
            "component_threshold_score": float(component_threshold),
            "threshold_price": float(p_tau),
            "component_verdict": info_geo > component_threshold,
            "price": payload["price"],
            "certified_l2_radius": payload["certified_l2_radius"],
            "witness": payload["witness"],
            "witness_channel": payload["witness_channel"],
        }

    # =====================================================================
    # Unsupervised operating-point calibration
    # =====================================================================

    # Histogram valley depth (1 - density_at_split / peak_density) above which
    # the score distribution is treated as bimodal/higher-contamination and the
    # Otsu split is used; below it the anomalies are a low-contamination upper
    # tail and a robust MAD tail cut is used. Selected on the live benchmark
    # suite (best full-suite F1 on a 0.55-0.60 plateau, no fragile boundary).
    _ADAPTIVE_VALLEY_DEPTH: float = 0.55

    def _adaptive_operating_point(
        self,
        scores: np.ndarray[Any, Any],
    ) -> tuple[float, dict[str, Any]]:
        """Derive an unsupervised decision threshold from the score distribution.

        Mercury's ensemble emits a compressed [0, 1] score distribution whose
        normal-cluster location is data-dependent (the resonance component alone
        contributes a ~``0.4 * (1 - h_train)`` baseline). A fixed 0.5 cut
        therefore lands almost arbitrarily relative to that cluster, yielding
        strong ranking (high AUROC) but a broken operating point (near-zero F1).

        This picks the cut from the scores themselves:

          * **High-contamination / bimodal** — if the Otsu between-class split
            sits in a deep histogram valley, a distinct high-score mode exists;
            use the Otsu threshold.
          * **Low-contamination / upper tail** — otherwise cut at a robust
            number of MADs above the median (``median + 2 * 1.4826 * MAD``),
            which adapts to the bulk's spread without assuming a fixed rate.

        The transform is rank-preserving: only the location of the cut changes,
        not the scores, so AUROC/AUPRC are unaffected.

        Args:
            scores: Ensemble anomaly scores in [0, 1], shape ``(n_samples,)``.

        Returns:
            ``(threshold, diagnostics)`` — the chosen cut and a small dict
            describing the regime and threshold for transparency.
        """
        s = np.asarray(scores, float).reshape(-1)
        n = s.size
        rng = float(np.ptp(s)) if n else 0.0

        # Too few points or no spread to calibrate against: fall back to a
        # conservative high-quantile cut (or the configured default if empty).
        if n < 8 or rng < 1e-9:
            thr = float(np.percentile(s, 95)) if n else float(self.threshold)
            return thr, {"method": "adaptive", "regime": "degenerate", "threshold": thr}

        t_otsu = self._otsu_threshold(s)
        valley = self._score_valley_depth(s, t_otsu)
        if valley >= self._ADAPTIVE_VALLEY_DEPTH:
            thr = t_otsu
            regime = "bimodal_otsu"
        else:
            thr = self._robust_tail_threshold(s)
            regime = "robust_tail"

        # Keep the cut inside the observed score range.
        thr = float(np.clip(thr, float(s.min()), float(s.max())))
        return thr, {
            "method": "adaptive",
            "regime": regime,
            "valley_depth": round(float(valley), 4),
            "threshold": thr,
            "flagged_fraction": round(float(np.mean(s > thr)), 4),
        }

    @staticmethod
    def _otsu_threshold(s: np.ndarray[Any, Any]) -> float:
        """Otsu between-class-variance threshold on a 256-bin score histogram."""
        lo = float(s.min())
        hi = float(s.max())
        if hi - lo < 1e-12:
            return hi
        norm = ((s - lo) / (hi - lo) * 255).astype(int)
        hist = np.bincount(norm, minlength=256).astype(float)
        total = hist.sum()
        sum_total = float(np.dot(np.arange(256), hist))
        w_b = 0.0
        sum_b = 0.0
        best_var = 0.0
        best_bin = 0
        for t in range(256):
            w_b += hist[t]
            if w_b == 0:
                continue
            w_f = total - w_b
            if w_f == 0:
                break
            sum_b += t * hist[t]
            m_b = sum_b / w_b
            m_f = (sum_total - sum_b) / w_f
            var = w_b * w_f * (m_b - m_f) ** 2
            if var > best_var:
                best_var = var
                best_bin = t
        return lo + (best_bin / 255.0) * (hi - lo)

    @staticmethod
    def _score_valley_depth(s: np.ndarray[Any, Any], threshold: float) -> float:
        """Histogram valley depth at ``threshold``: ``1 - density/peak_density``.

        ~1.0 => the split sits in a deep valley between two modes (bimodal);
        ~0.0 => the split sits inside a single dense mode (unimodal).
        """
        n_bins = int(np.clip(np.sqrt(s.size), 10, 40))
        hist, edges = np.histogram(s, bins=n_bins)
        peak = hist.max()
        if peak == 0:
            return 0.0
        bin_idx = int(np.clip(np.searchsorted(edges, threshold) - 1, 0, len(hist) - 1))
        return 1.0 - float(hist[bin_idx]) / float(peak)

    @staticmethod
    def _robust_tail_threshold(s: np.ndarray[Any, Any], k: float = 2.0) -> float:
        """Robust upper-tail cut: ``median + k * 1.4826 * MAD``.

        MAD (median absolute deviation) scaled by 1.4826 is a robust estimate of
        the bulk's standard deviation; the cut sits ``k`` such deviations above
        the median. Falls back to the 97th percentile when MAD is degenerate
        (near-constant scores).
        """
        med = float(np.median(s))
        mad = float(np.median(np.abs(s - med)))
        if mad < 1e-12:
            return float(np.percentile(s, 97))
        return med + k * 1.4826 * mad

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

        .. warning::

            **Temporal ordering assumption**: This method assumes that rows
            in *X* are temporally ordered (i.e., adjacent rows represent
            consecutive time steps).  On **shuffled tabular data**,
            derivatives computed via ``np.diff`` are meaningless noise and
            will produce near-random AUC (~0.60).  Use data type detection
            (``_detect_data_characteristics()``) to automatically disable
            this component on non-temporal data.

            **Ideal use cases**: Time-series data, sequential sensor
            readings, trajectory data.

            **Poor use cases**: Shuffled tabular data, cross-sectional data,
            unordered features.

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

    @staticmethod
    def _residual_frequency_filter(
        scores: np.ndarray[Any, Any], cutoff_quantile: float = 0.75
    ) -> np.ndarray[Any, Any]:
        """Apply frequency-domain filtering to the score residual.

        Computes the score residual (deviation from moving average),
        applies a bandpass filter to isolate anomaly-relevant frequencies,
        and blends the filtered signal back into the scores.

        Args:
            scores: Raw anomaly scores, shape (n_samples,).
            cutoff_quantile: Fraction of frequency spectrum to preserve.

        Returns:
            Filtered scores with noise-suppressed anomaly signal.
        """
        if len(scores) < 16:
            return scores

        kernel_size = max(5, len(scores) // 20)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones(kernel_size) / kernel_size
        baseline = np.convolve(scores, kernel, mode="same")
        residual = scores - baseline

        fft_res = np.fft.rfft(residual)
        power = np.abs(fft_res) ** 2

        if power.sum() < 1e-10:
            return scores

        cumulative = np.cumsum(power) / power.sum()
        low_cut = int(np.searchsorted(cumulative, 0.05))
        high_cut = int(np.searchsorted(cumulative, cutoff_quantile))

        filtered_fft = np.zeros_like(fft_res)
        filtered_fft[low_cut:high_cut] = fft_res[low_cut:high_cut]

        filtered_residual = np.fft.irfft(filtered_fft, n=len(residual))

        blended = 0.7 * scores + 0.3 * (baseline + filtered_residual)
        return np.clip(blended, 0.0, 1.0)

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

    def extract_features(
        self, data: np.ndarray[Any, Any] | torch.Tensor
    ) -> np.ndarray[Any, Any] | torch.Tensor:
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
        return features

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

    def get_fitted_state(self) -> dict[str, Any] | None:
        """Export the fitted state for checkpoint round-tripping.

        Mirrors the attribute set :meth:`from_statistics` (the federated
        reconstruction path) already restores, plus the stored training
        sample that the ensemble-health introspection reads (ROADMAP row
        16). An active Oracle is exported through the same reference
        statistics the federation path uses (``oracle_ref_stats`` from
        :meth:`get_oracle_statistics`); :meth:`set_fitted_state` re-arms it
        via :meth:`_restore_oracle_from_ref_stats` without re-fitting.

        Returns:
            JSON/tensor-safe mapping, or ``None`` when unfitted.
        """
        if not self._is_fitted:
            return None

        def _arr(value: np.ndarray[Any, Any] | None) -> np.ndarray[Any, Any] | None:
            return None if value is None else np.asarray(value, dtype=np.float64)

        return {
            "mean": _arr(self.mean),
            "std": _arr(self.std),
            "q1": _arr(self.q1),
            "q3": _arr(self.q3),
            "res_h_train": _arr(self._res_h_train),
            "res_noise_ratio": _arr(self._res_noise_ratio),
            "kin_jerk_mean": _arr(self._kin_jerk_mean),
            "kin_jerk_std": _arr(self._kin_jerk_std),
            "kin_accel_mean": _arr(self._kin_accel_mean),
            "kin_accel_std": _arr(self._kin_accel_std),
            "ig_mean": _arr(self._ig_mean),
            "ig_cov_inv": _arr(self._ig_cov_inv),
            "ig_log_det": float(self._ig_log_det),
            "adaptive_weights": _arr(self._adaptive_weights),
            "data_type": self._data_type.value,
            "train_data": _arr(self._train_data),
            "supervised_threshold": (
                float(self._supervised_threshold)
                if self._supervised_threshold is not None
                else None
            ),
            "oracle_ref_stats": self.get_oracle_statistics(),
        }

    def set_fitted_state(self, state: dict[str, Any]) -> None:
        """Restore a state produced by :meth:`get_fitted_state`."""

        def _arr(value: Any) -> np.ndarray[Any, Any] | None:
            return None if value is None else np.asarray(value, dtype=np.float64)

        self.mean = _arr(state["mean"])
        self.std = _arr(state["std"])
        self.q1 = _arr(state["q1"])
        self.q3 = _arr(state["q3"])
        self._res_h_train = _arr(state["res_h_train"])
        self._res_noise_ratio = _arr(state["res_noise_ratio"])
        self._kin_jerk_mean = _arr(state["kin_jerk_mean"])
        self._kin_jerk_std = _arr(state["kin_jerk_std"])
        self._kin_accel_mean = _arr(state["kin_accel_mean"])
        self._kin_accel_std = _arr(state["kin_accel_std"])
        self._ig_mean = _arr(state["ig_mean"])
        self._ig_cov_inv = _arr(state["ig_cov_inv"])
        self._ig_log_det = float(state["ig_log_det"])
        weights = _arr(state["adaptive_weights"])
        if weights is not None:
            self._adaptive_weights = weights
        self._data_type = DataCharacteristics(str(state["data_type"]))
        self._train_data = _arr(state["train_data"])
        supervised = state.get("supervised_threshold")
        self._supervised_threshold = float(supervised) if supervised is not None else None
        oracle_ref_stats = state.get("oracle_ref_stats")
        self._restore_oracle_from_ref_stats(
            dict(oracle_ref_stats) if oracle_ref_stats is not None else None
        )
        self._is_fitted = True


def calibrate_scores(
    scores: np.ndarray[Any, Any],
    anomaly_ratio: float,
) -> np.ndarray[Any, Any]:
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
