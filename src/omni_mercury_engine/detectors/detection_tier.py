# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""End-to-end integration for the streaming / statistical / state-space detector tier.

The detectors added under this tier (spectral-residual, BOCPD, SPOT/DSPOT,
Hawkes, particle-filter, IMM, Gaussian-process, echo-state, spiking, digital-twin,
survival, energy-based, Deep-SVDD, RCA, DeepLog, frequent-pattern, and the
torch-gated SR-CNN / diffusion detectors) each implement the
:class:`~omni_mercury_engine.core.base.BaseDetector` contract and are registered
for auto-discovery. That makes every one of them available to Mercury's fusion
pipeline individually, but the operational value comes from combining them: a
streaming point is anomalous when a *calibrated ensemble* of these complementary
views says so, with a bounded false-positive rate and an attribution back to the
originating node.

This module supplies that wiring on top of existing Mercury infrastructure rather
than reinventing it:

* :class:`StreamingScoreEnsemble` stacks the per-point scores of several tier
  detectors through Mercury's own logistic meta-learner
  (:class:`omni_mercury_engine.ml.mercury_ml.LogisticRegression`) for calibrated
  stacked generalisation, or combines them by Bayesian Model Averaging (BIC
  posterior weights with bootstrap uncertainty), and reports per-point ensemble
  uncertainty as cross-detector disagreement. Final thresholds come from the
  shared score-calibration layer
  (:func:`omni_mercury_engine.core.score_calibration.calibrate_scores`).
* :func:`conformal_threshold` / :func:`conformal_flags` bound the streaming
  false-positive rate distribution-free via split conformal prediction
  (:class:`omni_mercury_engine.core.conformal_prediction.SplitConformalPredictor`).
* :func:`rca_localize` turns a multivariate anomaly into a ranked list of root
  causes over a causal / service graph using the tier's
  :class:`~omni_mercury_engine.detectors.rca.RootCauseGraphDetector`.

Everything here is pure NumPy plus the first-party helpers above, so the module is
always importable (no PyTorch gate) and deterministic under a fixed seed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch

    from omni_mercury_engine.core.base import BaseDetector

logger = logging.getLogger(__name__)

__all__ = [
    "STREAMING_TIER",
    "TIER_PARADIGMS",
    "TORCH_TIER",
    "StreamingScoreEnsemble",
    "align_point_scores",
    "build_tier_detectors",
    "conformal_flags",
    "conformal_threshold",
    "rca_localize",
]

# ---------------------------------------------------------------------------
# Canonical tier -- grouped by detection paradigm. Kept in lockstep with the
# DetectorManifestEntry records in ``core.detector_registry.DETECTOR_MANIFEST``;
# ``build_tier_detectors`` resolves module/class from that manifest so the two
# never drift.
# ---------------------------------------------------------------------------
TIER_PARADIGMS: dict[str, tuple[str, ...]] = {
    "temporal_streaming": ("spectral_residual", "bocpd", "hawkes"),
    "state_space": ("particle_filter", "imm", "digital_twin"),
    "probabilistic": ("spot_evt", "gaussian_process", "survival"),
    "generative": ("energy_based", "deep_svdd"),
    "neuromorphic": ("echo_state", "spiking"),
    "systems": ("rca", "deeplog_sequence", "frequent_pattern"),
}

#: Flat, de-duplicated tuple of every pure-NumPy tier detector name. Always
#: buildable (no optional dependency), so this is the default tier.
STREAMING_TIER: tuple[str, ...] = tuple(name for group in TIER_PARADIGMS.values() for name in group)

#: Torch-gated tier members (SR-CNN, DDPM-AD). Not in :data:`STREAMING_TIER` so
#: the default tier stays importable without PyTorch; pass these explicitly to
#: :func:`build_tier_detectors` when the ML extra is installed.
TORCH_TIER: tuple[str, ...] = ("srcnn", "diffusion_ad")

_ENSEMBLE_METHODS = ("stacking", "bma", "average")


def build_tier_detectors(
    subset: list[str] | tuple[str, ...] | None = None,
) -> dict[str, BaseDetector]:
    """Instantiate tier detectors, resolving module/class from the manifest.

    Args:
        subset: Optional detector names to build (defaults to the full
            :data:`STREAMING_TIER`). Unknown names raise ``KeyError`` so a typo
            fails loudly rather than silently dropping a detector.

    Returns:
        Mapping ``name -> detector instance``, insertion-ordered by ``subset``.

    Raises:
        KeyError: If a requested name is not a registered tier detector.
    """
    from omni_mercury_engine.core.detector_registry import DETECTOR_MANIFEST

    manifest = {entry.name: entry for entry in DETECTOR_MANIFEST}
    names = tuple(subset) if subset is not None else STREAMING_TIER
    built: dict[str, BaseDetector] = {}
    for name in names:
        if name not in manifest:
            raise KeyError(f"'{name}' is not a registered detector")
        entry = manifest[name]
        module = __import__(entry.module_path, fromlist=[entry.class_name])
        cls = getattr(module, entry.class_name)
        built[name] = cls()
    return built


def _to_1d_series(data: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
    """Coerce input to a finite 1-D float series (flattening extra dimensions)."""
    detach = getattr(data, "detach", None)
    if callable(detach):
        data = detach().cpu().numpy()
    arr = np.nan_to_num(np.asarray(data, dtype=np.float64)).ravel()
    if arr.size == 0:
        raise ValueError("input series is empty")
    return arr


def align_point_scores(
    detector: BaseDetector,
    series: np.ndarray[Any, Any] | torch.Tensor,
) -> np.ndarray[Any, Any]:
    """Return a detector's per-point anomaly scores aligned to ``len(series)``.

    Tier detectors emit either a per-point ``scores`` vector or a scalar
    ``anomaly_score``; windowed detectors may emit a shorter vector. This
    normalises all of them to one score in ``[0, 1]`` per input point via linear
    interpolation, so heterogeneous detectors can be stacked column-wise.

    Args:
        detector: A fitted tier detector.
        series: 1-D input series.

    Returns:
        ``(n_points,)`` float64 scores in ``[0, 1]``.
    """
    arr = _to_1d_series(series)
    result = detector.detect(arr)
    raw = result.get("scores")
    if raw is None:
        fallback = float(result.get("anomaly_score", result.get("anomaly_prob", 0.0)))
        return np.full(arr.size, fallback, dtype=np.float64)
    scores = np.asarray(raw, dtype=np.float64).ravel()
    if scores.size == 0:
        return np.zeros(arr.size, dtype=np.float64)
    if scores.size != arr.size:
        src = np.linspace(0.0, 1.0, scores.size)
        dst = np.linspace(0.0, 1.0, arr.size)
        scores = np.interp(dst, src, scores)
    return np.clip(scores, 0.0, 1.0)


class StreamingScoreEnsemble:
    """Calibrated stacking / BMA ensemble over tier detectors' per-point scores.

    The ensemble fits each detector on (the normal part of) a training series,
    collects their per-point scores into a ``(n_points, n_detectors)`` matrix,
    and combines them by one of:

    * ``"stacking"`` -- a logistic meta-learner
      (:class:`omni_mercury_engine.ml.mercury_ml.LogisticRegression`) trained on
      point labels (stacked generalisation, Wolpert 1992);
    * ``"bma"`` -- Bayesian Model Averaging with BIC posterior weights and
      bootstrap weight uncertainty;
    * ``"average"`` -- the unweighted score mean (label-free baseline).

    The combined score is thresholded through the shared score-calibration layer,
    and :meth:`ensemble_uncertainty` exposes per-point cross-detector disagreement
    for downstream fusion weighting.
    """

    def __init__(
        self,
        detectors: dict[str, BaseDetector],
        method: str = "stacking",
        contamination: float = 0.05,
        seed: int = 0,
    ) -> None:
        """Initialize the ensemble.

        Args:
            detectors: Mapping ``name -> BaseDetector`` to combine.
            method: One of ``"stacking"``, ``"bma"``, ``"average"``.
            contamination: Expected anomaly fraction for threshold calibration.
            seed: RNG seed for the BMA bootstrap (reproducibility).

        Raises:
            ValueError: If ``detectors`` is empty or ``method`` is unknown.
        """
        if not detectors:
            raise ValueError("StreamingScoreEnsemble needs at least one detector")
        if method not in _ENSEMBLE_METHODS:
            raise ValueError(f"method must be one of {_ENSEMBLE_METHODS}, got {method!r}")
        self.detectors = dict(detectors)
        self.method = method
        self.contamination = float(contamination)
        self.seed = int(seed)
        self._names = list(self.detectors)
        self._meta: Any = None
        self._weights: np.ndarray[Any, Any] = np.full(
            len(self._names), 1.0 / len(self._names), dtype=np.float64
        )
        self._weight_std: np.ndarray[Any, Any] = np.zeros(len(self._names), dtype=np.float64)
        self._threshold: float = 0.5
        self._fitted = False

    def _score_matrix(self, series: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Stack every detector's aligned per-point scores column-wise."""
        columns = [align_point_scores(self.detectors[name], series) for name in self._names]
        return np.column_stack(columns)

    @staticmethod
    def _bic_weight(scores: np.ndarray[Any, Any], labels: np.ndarray[Any, Any]) -> float:
        """Unnormalised BMA weight ``exp(-BIC/2)`` from a univariate logistic fit.

        The BIC of ``label ~ sigmoid(a * score + b)`` rewards a detector whose
        scores separate the labelled anomalies with few parameters (k=2).
        """
        from omni_mercury_engine.ml.mercury_ml import LogisticRegression

        n = labels.size
        if len(np.unique(labels)) < 2:
            return 0.0
        model = LogisticRegression()
        feats = scores.reshape(-1, 1)
        model.fit(feats, labels)
        proba = np.clip(np.asarray(model.predict_proba(feats))[:, 1], 1e-9, 1 - 1e-9)
        log_lik = float(np.sum(labels * np.log(proba) + (1 - labels) * np.log(1 - proba)))
        bic = 2.0 * np.log(max(n, 2)) - 2.0 * log_lik
        return float(np.exp(-0.5 * bic))

    def _fit_bma(self, score_matrix: np.ndarray[Any, Any], labels: np.ndarray[Any, Any]) -> None:
        """Fit BMA posterior weights with bootstrap uncertainty."""
        rng = np.random.default_rng(self.seed)
        n = labels.size

        def weights_from(idx: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            raw = np.array(
                [
                    self._bic_weight(score_matrix[idx, j], labels[idx])
                    for j in range(len(self._names))
                ]
            )
            total = raw.sum()
            if total <= 0.0:
                return np.full(len(self._names), 1.0 / len(self._names))
            return raw / total

        self._weights = weights_from(np.arange(n))
        boots = np.array([weights_from(rng.integers(0, n, size=n)) for _ in range(20)])
        self._weight_std = boots.std(axis=0)

    def fit(
        self,
        series: np.ndarray[Any, Any] | torch.Tensor,
        labels: np.ndarray[Any, Any] | None = None,
    ) -> StreamingScoreEnsemble:
        """Fit member detectors, the combiner, and the calibrated threshold.

        Args:
            series: Training series (1-D). When ``labels`` are supplied the
                detectors are fitted on the normal (label 0) points only.
            labels: Optional per-point 0/1 labels. Required for ``stacking`` and
                strongly recommended for ``bma``; ``average`` ignores them.

        Returns:
            ``self``.
        """
        arr = _to_1d_series(series)
        lab = None if labels is None else np.asarray(labels).astype(int).ravel()
        train = arr if lab is None or not (lab == 0).any() else arr[lab == 0]
        for detector in self.detectors.values():
            detector.fit(train)

        score_matrix = self._score_matrix(arr)
        if self.method == "stacking":
            if lab is None:
                raise ValueError("stacking requires per-point labels")
            from omni_mercury_engine.ml.mercury_ml import LogisticRegression

            self._meta = LogisticRegression()
            self._meta.fit(score_matrix, lab)
        elif self.method == "bma" and lab is not None:
            self._fit_bma(score_matrix, lab)

        combined = self._combine(score_matrix)
        from omni_mercury_engine.core.score_calibration import calibrate_scores

        threshold, _, _ = calibrate_scores(combined, contamination=self.contamination, labels=lab)
        self._threshold = float(threshold)
        self._fitted = True
        return self

    def _combine(self, score_matrix: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Combine the per-detector score matrix into one calibrated score."""
        if self.method == "stacking" and self._meta is not None:
            proba = np.asarray(self._meta.predict_proba(score_matrix))[:, 1]
        elif self.method == "bma":
            proba = score_matrix @ self._weights
        else:
            proba = score_matrix.mean(axis=1)
        return np.clip(proba, 0.0, 1.0)

    def score(self, series: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Per-point ensemble anomaly probability in ``[0, 1]``."""
        if not self._fitted:
            raise RuntimeError("call fit() before score()")
        return self._combine(self._score_matrix(_to_1d_series(series)))

    def predict(self, series: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Per-point 0/1 anomaly flags at the calibrated threshold."""
        return (self.score(series) > self._threshold).astype(int)

    def ensemble_uncertainty(
        self, series: np.ndarray[Any, Any] | torch.Tensor
    ) -> np.ndarray[Any, Any]:
        """Per-point cross-detector disagreement (score std) as an uncertainty."""
        if not self._fitted:
            raise RuntimeError("call fit() before ensemble_uncertainty()")
        return self._score_matrix(_to_1d_series(series)).std(axis=1)

    def bma_weights(self) -> dict[str, tuple[float, float]]:
        """BMA weight ± bootstrap-uncertainty per detector (``{}`` if not BMA)."""
        if self.method != "bma":
            return {}
        return {
            name: (float(self._weights[i]), float(self._weight_std[i]))
            for i, name in enumerate(self._names)
        }

    @property
    def threshold(self) -> float:
        """The calibrated decision threshold on the combined score."""
        return self._threshold


def conformal_threshold(
    calibration_scores: np.ndarray[Any, Any],
    alpha: float = 0.05,
) -> float:
    """Split-conformal anomaly threshold bounding the false-positive rate at ``alpha``.

    Given anomaly scores from an exchangeable *normal* calibration stream, the
    returned threshold guarantees (finite-sample, distribution-free) that a normal
    point exceeds it with probability at most ``alpha`` -- i.e. FPR ``<= alpha``.
    Delegates to
    :class:`omni_mercury_engine.core.conformal_prediction.SplitConformalPredictor`.

    Args:
        calibration_scores: Anomaly scores of known-normal points.
        alpha: Target false-positive rate in ``(0, 1)``.

    Returns:
        The conformal score threshold.

    Raises:
        ValueError: If ``alpha`` is not in ``(0, 1)`` or the calibration set is empty.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    scores = np.asarray(calibration_scores, dtype=np.float64).ravel()
    if scores.size == 0:
        raise ValueError("calibration_scores is empty")
    from omni_mercury_engine.core.conformal_prediction import SplitConformalPredictor

    predictor = SplitConformalPredictor(coverage=1.0 - alpha)
    predictor.fit(scores)
    return float(predictor.get_anomaly_threshold())


def conformal_flags(
    scores: np.ndarray[Any, Any],
    calibration_scores: np.ndarray[Any, Any],
    alpha: float = 0.05,
) -> np.ndarray[Any, Any]:
    """Flag points whose score exceeds the conformal threshold (FPR ``<= alpha``).

    Args:
        scores: Anomaly scores to threshold.
        calibration_scores: Known-normal calibration scores.
        alpha: Target false-positive rate.

    Returns:
        Boolean array, ``True`` where a point is flagged anomalous.
    """
    threshold = conformal_threshold(calibration_scores, alpha=alpha)
    return np.asarray(scores, dtype=np.float64).ravel() > threshold


def rca_localize(
    observations: np.ndarray[Any, Any] | torch.Tensor,
    adjacency: np.ndarray[Any, Any] | None = None,
    train: np.ndarray[Any, Any] | torch.Tensor | None = None,
    top_k: int | None = None,
) -> list[tuple[int, float]]:
    """Rank candidate root-cause nodes for a multivariate anomaly.

    Wraps the tier's
    :class:`~omni_mercury_engine.detectors.rca.RootCauseGraphDetector`: it learns
    per-node baselines (from ``train`` if given, else ``observations``), then runs
    a reverse personalised random walk over the causal / service ``adjacency`` to
    attribute the final observation's anomaly to upstream nodes.

    Args:
        observations: ``(n_rows, n_nodes)`` signal; the last row is localised.
        adjacency: Optional ``(n_nodes, n_nodes)`` non-negative causal adjacency
            (``A[i, j] > 0`` ⇒ ``i`` influences ``j``); inferred from training
            correlations when ``None``.
        train: Optional normal-behaviour rows for the baselines (defaults to
            ``observations``).
        top_k: If given, return only the ``top_k`` highest-attribution nodes.

    Returns:
        ``(node_index, attribution)`` pairs, descending by attribution.
    """
    from omni_mercury_engine.detectors.rca import RootCauseGraphDetector

    detector = RootCauseGraphDetector(adjacency=adjacency)
    detector.fit(observations if train is None else train)
    ranked = detector.rank_root_causes(observations)
    return ranked if top_k is None else ranked[:top_k]
