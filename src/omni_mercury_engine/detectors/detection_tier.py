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
from collections import deque
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

from omni_mercury_engine.detectors._calibration import finite_scores
from omni_mercury_engine.detectors.detection_config import DetectionConfig

if TYPE_CHECKING:
    import torch

    from omni_mercury_engine.core.base import BaseDetector
    from omni_mercury_engine.core.feature_pipeline import FeatureSchema, FeatureStore

logger = logging.getLogger(__name__)

__all__ = [
    "STREAMING_TIER",
    "TIER_PARADIGMS",
    "TORCH_TIER",
    "StreamingScoreEnsemble",
    "TierStreamingScorer",
    "align_point_scores",
    "build_tier_detectors",
    "conformal_flags",
    "conformal_threshold",
    "rca_localize",
    "store_tier_features",
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

_ENSEMBLE_METHODS = ("stacking", "bma", "average", "consensus")

#: Default cross-detector quantile for the label-free ``"consensus"`` combiner. A
#: high quantile (~0.9) makes the combined score "a point most calibrated
#: detectors rank in their tail", which is robust to the uninformative members a
#: plain mean is dragged down by.
_DEFAULT_CONSENSUS_QUANTILE = 0.9

#: Per-detector score-calibration transforms selectable in
#: :class:`StreamingScoreEnsemble`. ``rank``/``ecdf`` are label-free empirical-CDF
#: transforms (the default); ``isotonic``/``platt`` are supervised monotone maps
#: trained on the warm-up window; ``none`` disables per-detector calibration.
_CALIBRATION_METHODS = ("rank", "ecdf", "isotonic", "platt", "none")


def _pool_adjacent_violators(y: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Isotonic (non-decreasing) fit of ``y`` via the pool-adjacent-violators algorithm.

    Returns a non-decreasing vector minimising the squared error to ``y`` (unit
    weights). Pure NumPy; ``O(n)`` amortised.
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n == 0:
        return y
    # Stack of (mean, weight, length) blocks; merge while the previous block's
    # mean exceeds the current (a monotonicity violation).
    values = np.empty(n, dtype=np.float64)
    weights = np.empty(n, dtype=np.float64)
    lengths = np.empty(n, dtype=np.int64)
    top = -1
    for value in y:
        top += 1
        values[top] = value
        weights[top] = 1.0
        lengths[top] = 1
        while top > 0 and values[top - 1] > values[top]:
            w = weights[top - 1] + weights[top]
            values[top - 1] = (weights[top - 1] * values[top - 1] + weights[top] * values[top]) / w
            weights[top - 1] = w
            lengths[top - 1] += lengths[top]
            top -= 1
    out = np.empty(n, dtype=np.float64)
    idx = 0
    for b in range(top + 1):
        out[idx : idx + lengths[b]] = values[b]
        idx += int(lengths[b])
    return out


class _ScoreCalibrator:
    """Per-detector monotone map from a raw score column into a calibrated ``[0, 1]``.

    Concrete subclasses implement :meth:`transform`. The point of calibration is
    that heterogeneous detectors emit scores on incomparable scales; mapping each
    detector's scores onto a common ``[0, 1]`` scale *before* combining stops a
    detector with a systematically inflated score range from dominating an
    unweighted average.
    """

    def transform(self, scores: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Map raw scores to calibrated ``[0, 1]`` scores."""
        raise NotImplementedError


class _IdentityCalibrator(_ScoreCalibrator):
    """No-op calibrator (``calibration='none'``): clip into ``[0, 1]`` only."""

    def transform(self, scores: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Return the scores clipped into ``[0, 1]`` unchanged otherwise."""
        return np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)


class _EcdfCalibrator(_ScoreCalibrator):
    """Rank / empirical-CDF calibrator: map a score to ``P(reference <= score)``.

    Fit on a reference (warm-up) window, it transforms any score to the fraction
    of reference points at or below it -- the empirical CDF, which is uniform on
    ``[0, 1]`` under the reference distribution. Monotone, label-free, and robust
    to each detector's arbitrary score scale.
    """

    def __init__(self, reference: np.ndarray[Any, Any]) -> None:
        """Store the sorted reference distribution."""
        ref = np.asarray(reference, dtype=np.float64)
        ref = ref[np.isfinite(ref)]
        self._reference = np.sort(ref)

    def transform(self, scores: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Empirical-CDF transform against the fitted reference."""
        arr = np.asarray(scores, dtype=np.float64)
        if self._reference.size == 0:
            return np.clip(arr, 0.0, 1.0)
        ranks = np.searchsorted(self._reference, arr, side="right")
        return ranks.astype(np.float64) / float(self._reference.size)


class _IsotonicCalibrator(_ScoreCalibrator):
    """Isotonic-regression calibrator: monotone score->P(anomaly) from labels.

    Trained on the (warm-up) window's ``(score, label)`` pairs via
    pool-adjacent-violators; transforms via monotone interpolation. Falls back to
    the fitted step function's endpoints outside the training range.
    """

    def __init__(self, scores: np.ndarray[Any, Any], labels: np.ndarray[Any, Any]) -> None:
        """Fit the isotonic map on ``(scores, labels)``."""
        x = np.asarray(scores, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        order = np.argsort(x, kind="mergesort")
        self._x = x[order]
        self._y = _pool_adjacent_violators(y[order])

    def transform(self, scores: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Monotone-interpolate calibrated probabilities for ``scores``."""
        arr = np.asarray(scores, dtype=np.float64)
        if self._x.size == 0:
            return np.clip(arr, 0.0, 1.0)
        calibrated = np.interp(arr, self._x, self._y, left=self._y[0], right=self._y[-1])
        return np.clip(calibrated, 0.0, 1.0)


class _PlattCalibrator(_ScoreCalibrator):
    """Platt-scaling calibrator: logistic score->P(anomaly) from labels.

    Fits Mercury's own logistic regression on the (warm-up) ``(score, label)``
    pairs and transforms via the fitted sigmoid probability.
    """

    def __init__(self, scores: np.ndarray[Any, Any], labels: np.ndarray[Any, Any]) -> None:
        """Fit the 1-D logistic map on ``(scores, labels)``."""
        from omni_mercury_engine.ml.mercury_ml import LogisticRegression

        self._model = LogisticRegression()
        self._model.fit(np.asarray(scores, dtype=np.float64).reshape(-1, 1), np.asarray(labels))

    def transform(self, scores: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Sigmoid-calibrated probabilities for ``scores``."""
        arr = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
        proba = np.asarray(self._model.predict_proba(arr))[:, 1]
        return np.clip(proba, 0.0, 1.0)


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
    # Attribute any non-finite correction to the *member* that produced it (not a
    # generic "align" label), so ``omni_detector_nonfinite_corrected`` and the
    # structured log name the misbehaving detector and stay actionable.
    member = getattr(detector, "name", "align")
    result = detector.detect(arr)
    raw = result.get("scores")
    if raw is None:
        fallback = float(result.get("anomaly_score", result.get("anomaly_prob", 0.0)))
        return finite_scores(np.full(arr.size, fallback, dtype=np.float64), detector=member)
    scores = np.asarray(raw, dtype=np.float64).ravel()
    if scores.size == 0:
        return np.zeros(arr.size, dtype=np.float64)
    if scores.size != arr.size:
        src = np.linspace(0.0, 1.0, scores.size)
        dst = np.linspace(0.0, 1.0, arr.size)
        scores = np.interp(dst, src, scores)
    # Defence in depth: a member that emits NaN/inf (np.clip does not scrub NaN)
    # would otherwise poison the ensemble mean / stacking / BMA and crash
    # calibrate_scores' np.histogram. finite_scores guarantees a finite [0, 1]
    # column regardless of member behaviour.
    return finite_scores(scores, detector=member)


class StreamingScoreEnsemble:
    """Calibrated stacking / BMA ensemble over tier detectors' per-point scores.

    The ensemble fits each detector on (the normal part of) a training series,
    collects their per-point scores into a ``(n_points, n_detectors)`` matrix,
    **calibrates each detector's column onto a common scale** (see below), and
    combines the calibrated columns by one of:

    * ``"stacking"`` -- a logistic meta-learner
      (:class:`omni_mercury_engine.ml.mercury_ml.LogisticRegression`) trained on
      point labels (stacked generalisation, Wolpert 1992);
    * ``"bma"`` -- Bayesian Model Averaging with BIC posterior weights and
      bootstrap weight uncertainty;
    * ``"average"`` -- the unweighted mean of the calibrated columns (label-free
      baseline);
    * ``"consensus"`` -- a label-free high-quantile consensus of the calibrated
      columns (the recommended unsupervised combiner): the per-point
      ``consensus_quantile`` across detectors, which is robust to uninformative
      members that drag a plain mean toward 0.5. On real NAB this is what lets the
      unsupervised ensemble beat the best single detector.

    Per-detector calibration
    ------------------------
    Heterogeneous detectors emit scores on incomparable scales, so a raw average
    lets a detector with a systematically inflated score range dominate. Before
    combining, each detector's score column is mapped through a per-detector
    calibrator fitted on a warm-up window:

    * ``"rank"`` / ``"ecdf"`` (default) -- the empirical-CDF transform, which is
      label-free and maps each detector's scores to a uniform ``[0, 1]`` scale;
    * ``"isotonic"`` / ``"platt"`` -- supervised monotone maps (isotonic
      regression / logistic scaling) from score to ``P(anomaly)`` trained on the
      warm-up window's labels. When the warm-up window is single-class (the
      common all-normal case) or labels are absent, these fall back to the ECDF
      transform so calibration never fails closed;
    * ``"none"`` -- disable per-detector calibration (raw scores, clipped).

    The combined score is thresholded through the shared score-calibration layer,
    and :meth:`ensemble_uncertainty` exposes per-point cross-detector disagreement
    (on the calibrated scale) for downstream fusion weighting.
    """

    def __init__(
        self,
        detectors: dict[str, BaseDetector],
        method: str = "stacking",
        contamination: float = 0.05,
        seed: int = 0,
        calibration: str | None = None,
        warmup: int | float | None = None,
        consensus_quantile: float = _DEFAULT_CONSENSUS_QUANTILE,
    ) -> None:
        """Initialize the ensemble.

        Args:
            detectors: Mapping ``name -> BaseDetector`` to combine.
            method: The *combiner* -- one of ``"stacking"``, ``"bma"``,
                ``"average"``, ``"consensus"``. ``"consensus"`` is a label-free
                combiner that takes the per-point ``consensus_quantile`` across the
                calibrated detector scores (robust to uninformative members that
                drag a plain ``"average"`` down); ``"average"`` is the plain mean
                of the calibrated scores.
            contamination: Expected anomaly fraction for threshold calibration.
            seed: RNG seed for the BMA bootstrap (reproducibility).
            calibration: The per-detector score *transform* applied before
                combining -- one of ``"rank"``/``"ecdf"`` (empirical-CDF, label-
                free, the default), ``"isotonic"``/``"platt"`` (supervised monotone
                maps trained on the warm-up window), or ``"none"``. ``None``
                resolves it from the tier config (env ``OMNI_ENSEMBLE_CALIBRATION``
                / config file, default ``"rank"``). Rank/ECDF calibration replaces
                the old raw-score averaging: it maps every detector onto a common
                uniform scale so no single detector's score range dominates the
                combination.
            warmup: Window the per-detector calibrators are trained on. ``None``
                (resolved from ``OMNI_ENSEMBLE_WARMUP``, default the whole training
                series) uses all training points; an ``int`` uses the first N; a
                ``float`` in ``(0, 1]`` uses that fraction.
            consensus_quantile: Cross-detector quantile for the ``"consensus"``
                combiner, in ``(0, 1]`` (default ~0.9).

        Raises:
            ValueError: If ``detectors`` is empty, ``method`` / ``calibration`` is
                unknown, ``consensus_quantile`` is out of ``(0, 1]``, or an
                explicit ``warmup`` fails the same validation as
                ``DetectionConfig.ensemble_warmup`` (bool, a float outside
                ``(0, 1]``, or an int ``< 2``).
        """
        if not detectors:
            raise ValueError("StreamingScoreEnsemble needs at least one detector")
        if method not in _ENSEMBLE_METHODS:
            raise ValueError(f"method must be one of {_ENSEMBLE_METHODS}, got {method!r}")
        cfg = DetectionConfig.resolve()
        resolved_calibration = (
            cfg.ensemble_calibration if calibration is None else str(calibration).strip().lower()
        )
        if resolved_calibration not in _CALIBRATION_METHODS:
            raise ValueError(
                f"calibration must be one of {_CALIBRATION_METHODS}, got {calibration!r}"
            )
        if not 0.0 < consensus_quantile <= 1.0:
            raise ValueError(f"consensus_quantile must be in (0, 1], got {consensus_quantile}")
        self.detectors = dict(detectors)
        self.method = method
        self.calibration = resolved_calibration
        # An explicitly-passed ``warmup`` must clear the *same* validation
        # ``DetectionConfig.ensemble_warmup`` enforces (reject bool, require a float
        # in (0, 1] or an int >= 2), so the constructor argument cannot smuggle in a
        # value the config layer would refuse. Re-validate by round-tripping through
        # the frozen dataclass, which raises ValueError with the identical message.
        if warmup is None:
            self.warmup = cfg.ensemble_warmup
        else:
            self.warmup = replace(cfg, ensemble_warmup=warmup).ensemble_warmup
        self.consensus_quantile = float(consensus_quantile)
        self.contamination = float(contamination)
        self.seed = int(seed)
        self._names = list(self.detectors)
        self._meta: Any = None
        self._weights: np.ndarray[Any, Any] = np.full(
            len(self._names), 1.0 / len(self._names), dtype=np.float64
        )
        self._weight_std: np.ndarray[Any, Any] = np.zeros(len(self._names), dtype=np.float64)
        self._calibrators: list[_ScoreCalibrator] = []
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
        # Fit the per-detector calibrators on the warm-up window, then combine on
        # the calibrated columns (the combiner -- stacking/bma -- is fit on the
        # calibrated matrix so training and scoring see the same transform).
        self._fit_calibrators(score_matrix, lab)
        calibrated = self._calibrate(score_matrix)
        if self.method == "stacking":
            if lab is None:
                raise ValueError("stacking requires per-point labels")
            from omni_mercury_engine.ml.mercury_ml import LogisticRegression

            self._meta = LogisticRegression()
            self._meta.fit(calibrated, lab)
        elif self.method == "bma" and lab is not None:
            self._fit_bma(calibrated, lab)

        combined = self._combine_calibrated(calibrated)
        from omni_mercury_engine.core.score_calibration import calibrate_scores

        threshold, _, _ = calibrate_scores(combined, contamination=self.contamination, labels=lab)
        self._threshold = float(threshold)
        self._fitted = True
        return self

    def _resolve_warmup(self, n: int) -> int:
        """Resolve the calibrator warm-up length for a training series of ``n`` points."""
        warm = self.warmup
        if warm is None:
            return n
        if isinstance(warm, float):
            return max(2, min(n, round(warm * n)))
        return max(2, min(n, int(warm)))

    def _make_calibrator(
        self, column: np.ndarray[Any, Any], labels: np.ndarray[Any, Any] | None
    ) -> _ScoreCalibrator:
        """Build one per-detector calibrator from a warm-up score column (+labels)."""
        method = self.calibration
        if method == "none":
            return _IdentityCalibrator()
        if method in ("rank", "ecdf"):
            return _EcdfCalibrator(column)
        # Supervised isotonic/platt need both classes present in the warm-up
        # window; otherwise fall back to the label-free ECDF transform so
        # calibration degrades gracefully instead of failing.
        if labels is None or np.unique(labels).size < 2:
            return _EcdfCalibrator(column)
        if method == "isotonic":
            return _IsotonicCalibrator(column, labels)
        return _PlattCalibrator(column, labels)

    def _fit_calibrators(
        self, score_matrix: np.ndarray[Any, Any], labels: np.ndarray[Any, Any] | None
    ) -> None:
        """Fit one calibrator per detector on the warm-up slice of the score matrix."""
        k = self._resolve_warmup(score_matrix.shape[0])
        warm_scores = score_matrix[:k]
        warm_labels = None if labels is None else labels[:k]
        self._calibrators = [
            self._make_calibrator(warm_scores[:, j], warm_labels) for j in range(len(self._names))
        ]

    def _calibrate(self, score_matrix: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Apply the fitted per-detector calibrators column-wise."""
        if not self._calibrators:
            return np.clip(score_matrix, 0.0, 1.0)
        columns = [
            self._calibrators[j].transform(score_matrix[:, j]) for j in range(len(self._names))
        ]
        return np.clip(np.column_stack(columns), 0.0, 1.0)

    def _combine_calibrated(self, calibrated: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Combine an already-calibrated score matrix into one score in ``[0, 1]``."""
        if self.method == "stacking" and self._meta is not None:
            proba = np.asarray(self._meta.predict_proba(calibrated))[:, 1]
        elif self.method == "bma":
            proba = calibrated @ self._weights
        elif self.method == "consensus":
            # Robust label-free consensus: the point's score is the level most
            # calibrated detectors agree it exceeds (a high cross-detector
            # quantile), which -- unlike the mean -- is not dragged toward 0.5 by
            # uninformative members.
            proba = np.quantile(calibrated, self.consensus_quantile, axis=1)
        else:  # "average": plain mean of the calibrated per-detector scores
            proba = calibrated.mean(axis=1)
        return np.clip(proba, 0.0, 1.0)

    def _combine(self, score_matrix: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Calibrate the per-detector score matrix, then combine into one score."""
        return self._combine_calibrated(self._calibrate(score_matrix))

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
        """Per-point cross-detector disagreement (calibrated score std) as an uncertainty."""
        if not self._fitted:
            raise RuntimeError("call fit() before ensemble_uncertainty()")
        return self._calibrate(self._score_matrix(_to_1d_series(series))).std(axis=1)

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


class TierStreamingScorer:
    """Adapt a tier detector to the streaming pipeline's ``dict -> dict`` callable.

    :class:`~omni_mercury_engine.infrastructure.streaming.StreamingAnomalyPipeline`
    consumes messages and calls ``detector(message) -> result``. Tier detectors
    are batch/series detectors, so this scorer keeps a rolling window of the most
    recent numeric values, (re)fits the wrapped detector every ``refit_interval``
    points to track drift, scores the newest point, and emits the score to the
    per-detector Prometheus histogram. It is directly usable as
    ``StreamingAnomalyPipeline(detector=TierStreamingScorer(det))``.
    """

    def __init__(
        self,
        detector: BaseDetector,
        *,
        name: str | None = None,
        window_size: int = 200,
        min_samples: int = 32,
        refit_interval: int = 64,
        value_key: str | None = None,
        threshold: float = 0.5,
    ) -> None:
        """Initialize the streaming scorer.

        Args:
            detector: The tier detector to drive online.
            name: Metric label for the detector (defaults to its class name).
            window_size: Rolling-window length scored on each point. Must be >= 2.
            min_samples: Points to buffer before scoring (warm-up). Must be >= 2.
            refit_interval: Points between detector refits. Must be >= 1.
            value_key: Message key to read the scalar from; if ``None`` the first
                finite numeric value in the message is used.
            threshold: Score above which a point is flagged anomalous.

        Raises:
            ValueError: If a window/sample/interval parameter is out of range.
        """
        if window_size < 2:
            raise ValueError(f"window_size must be >= 2, got {window_size}")
        if min_samples < 2:
            raise ValueError(f"min_samples must be >= 2, got {min_samples}")
        if refit_interval < 1:
            raise ValueError(f"refit_interval must be >= 1, got {refit_interval}")
        self.detector = detector
        self.name = name or type(detector).__name__
        self.window_size = int(window_size)
        self.min_samples = int(min_samples)
        self.refit_interval = int(refit_interval)
        self.value_key = value_key
        self.threshold = float(threshold)
        self._buffer: deque[float] = deque(maxlen=self.window_size)
        self._since_fit = 0

    def _extract_value(self, message: dict[str, Any]) -> float | None:
        """Pull the scalar to score from a stream message."""
        if self.value_key is not None:
            raw = message.get(self.value_key)
            candidates: list[Any] = [raw]
        else:
            candidates = list(message.values())
        for value in candidates:
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and np.isfinite(value):
                return float(value)
        return None

    def __call__(self, message: dict[str, Any]) -> dict[str, Any]:
        """Score one stream message; returns a detector-result dict."""
        value = self._extract_value(message)
        if value is None:
            # No finite numeric value to score in this message. Report warm-up
            # from the accumulated buffer state, not unconditionally: a scorer
            # that has already buffered ``min_samples`` points is *ready*, and a
            # single value-less message must not make downstream alerting think
            # it regressed to warm-up and suppress results.
            return {
                "is_anomaly": False,
                "anomaly_score": 0.0,
                "score": 0.0,
                "warmup": len(self._buffer) < self.min_samples,
            }
        self._buffer.append(value)
        if len(self._buffer) < self.min_samples:
            return {"is_anomaly": False, "anomaly_score": 0.0, "score": 0.0, "warmup": True}

        window = np.asarray(self._buffer, dtype=np.float64)
        self._since_fit += 1
        if not self.detector.is_fitted() or self._since_fit >= self.refit_interval:
            self.detector.fit(window)
            self._since_fit = 0

        score = float(align_point_scores(self.detector, window)[-1])
        from omni_mercury_engine.core.metrics import record_detector_score

        record_detector_score(self.name, score)
        return {
            "is_anomaly": bool(score > self.threshold),
            "anomaly_score": score,
            "score": score,
            "warmup": False,
        }


def store_tier_features(
    store: FeatureStore,
    detector: BaseDetector,
    name: str,
    data: np.ndarray[Any, Any] | torch.Tensor,
    *,
    version_manager: Any | None = None,
    schema_version: str = "1.0.0",
) -> tuple[np.ndarray[Any, Any], FeatureSchema]:
    """Extract, store, and provenance a tier detector's fusion features.

    Persists the detector's ``extract_features`` output into the shared
    :class:`~omni_mercury_engine.core.feature_pipeline.FeatureStore` (per-detector,
    data-hashed key) and builds a
    :class:`~omni_mercury_engine.core.feature_pipeline.FeatureSchema` recording the
    feature count, dtypes, and value ranges. When a ``version_manager`` is given
    the schema is registered for validation/versioning.

    Args:
        store: The feature store to write into.
        detector: A fitted tier detector.
        name: Detector name (feature-store key + schema name).
        data: Input the features are extracted from.
        version_manager: Optional ``FeatureVersionManager`` to register the schema.
        schema_version: Schema version string.

    Returns:
        ``(features, schema)`` — the stored features and their provenance schema.
    """
    from omni_mercury_engine.core.feature_pipeline import FeatureSchema

    features = np.asarray(detector.extract_features(data), dtype=np.float64)
    flat = features.reshape(features.shape[0], -1) if features.ndim > 1 else features.reshape(1, -1)
    n_features = int(flat.shape[-1])
    store.store(name, np.asarray(data, dtype=np.float64), features)
    schema = FeatureSchema(
        name=name,
        version=schema_version,
        n_features=n_features,
        feature_names=[f"{name}_{i}" for i in range(n_features)],
        dtypes=["float64"] * n_features,
        min_values=flat.min(axis=0).astype(float).tolist(),
        max_values=flat.max(axis=0).astype(float).tolist(),
    )
    if version_manager is not None:
        version_manager.register_schema(schema)
    return features, schema
