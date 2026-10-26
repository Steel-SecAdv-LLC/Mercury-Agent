# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Temporal anomaly detector for time series analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.core.exceptions import DetectorException
from omni_mercury_engine.detectors.provenance import DecisionSource

# Fixed initialization seed for the untrained LSTM feature projector. The
# value is arbitrary but immutable: every TemporalAnomalyDetector instance
# must project with identical weights so checkpoint-free engines agree
# across processes. Changing it changes what fresh instances extract;
# checkpoints are unaffected (they persist the projector weights).
_LSTM_INIT_SEED: int = 1_726_021

#: Elements of the window view scored per block in ``_detect_trend_anomalies``.
_TREND_BLOCK_ELEMENTS: int = 1 << 21


def _reduce_feature_z(z_matrix: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Reduce a per-feature z matrix [n, d] to one comparable z per row.

    The max over features sees a single-feature spike but throws away a
    shift spread over every feature: with d features, the max of d null
    z's runs about sqrt(2 ln d) (~2.0 at d=8), so a simultaneous 1.8-sigma
    move on all features ranks BELOW ordinary noise. The RMS sees the
    spread shift but dilutes a single-feature spike by sqrt(d). Each
    reduction is therefore normalized by its own null scale — sqrt(2 ln d)
    for the max, 1 for the RMS — and the larger one is the row's z, so
    both anomaly geometries keep their full resolution.
    """
    n_features = z_matrix.shape[1]
    max_null = float(np.sqrt(2.0 * np.log(max(n_features, 2))))
    max_z = z_matrix.max(axis=1) / max_null
    rms_z = np.sqrt(np.mean(np.square(z_matrix), axis=1))
    return np.asarray(np.maximum(max_z, rms_z))


class TemporalAnomalyDetector(BaseDetector):
    """Time series anomaly detection using:.

    - Trend analysis
    - Sudden changes
    - Seasonality detection
    - LSTM-based forecasting
    """

    decision_source = DecisionSource(
        kind="experimental",
        description=(
            "In-house heuristic: the score is the mean of a windowed trend "
            "z-score channel and a sudden-change channel; is_anomaly fires when "
            "that combined score exceeds the configured threshold (default 0.5 "
            "on the normalized score) or, when auto_calibrate is set, a "
            "threshold auto-calibrated from the batch score distribution at the "
            "configured contamination. No published threshold exists for this "
            "rule and no labelled corpus calibrates it."
        ),
    )

    #: Verified free of process-global mutable state at inference (bit-identical
    #: concurrent vs sequential: tests/core/test_parallel_detectors.py).
    parallel_safe = True

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the instance."""
        super().__init__(config)
        self.window_size = self.config.get("window_size", 10)
        self.change_threshold = self.config.get("change_threshold", 2.0)

        # LSTM is optional: it powers the documented "LSTM-based
        # forecasting" feature but trend / change / seasonality
        # analysis run on pure NumPy.  Previously ``__init__``
        # unconditionally constructed ``nn.LSTM``, which raised
        # ``NameError: name 'nn' is not defined`` when the optional
        # ``ml`` extra was absent (``nn`` was conditionally imported
        # but used unconditionally — a real bug).  Construct lazily:
        # callers without torch can still use the non-LSTM detection
        # paths, and callers that DO try the LSTM path receive a clear
        # ImportError from :meth:`_lstm` rather than a confusing
        # NameError leaking the implementation detail.
        # The LSTM is an *untrained, fixed* feature projector: its weights are
        # never fit, so they must be a constant of the architecture rather
        # than a draw from the process-global RNG. Constructing it under a
        # forked, fixed-seed RNG makes every instance bit-identical (and
        # leaves the caller's global RNG stream untouched). Checkpoints
        # already persist the projector weights (get_fitted_state, ROADMAP
        # row 16); this makes checkpoint-free instances agree across
        # processes too.
        self.lstm: Any | None = None
        if TORCH_AVAILABLE:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(_LSTM_INIT_SEED)
                self.lstm = nn.LSTM(
                    input_size=1,
                    hidden_size=32,
                    num_layers=2,
                    batch_first=True,
                )

        self.baseline_mean: float | None = None
        self.baseline_std: float | None = None
        # Per-feature fitted references (see fit()). None until fitted, and
        # None on a state restored from a legacy checkpoint, in which case
        # detect() falls back to the scored batch's own statistics exactly
        # as before the references existed.
        self.feature_mean_: np.ndarray[Any, Any] | None = None
        self.feature_std_: np.ndarray[Any, Any] | None = None
        self.diff_mean_: np.ndarray[Any, Any] | None = None
        self.diff_std_: np.ndarray[Any, Any] | None = None

    def _lstm(self) -> Any:
        """Return the LSTM, raising a clear error when torch is absent.

        Trend / change / seasonality detection paths do not need the
        LSTM; only the forecasting branch consults it.  Surface the
        missing-torch condition with an actionable message instead of
        the historical ``NameError: name 'nn' is not defined`` cascade.
        """
        if self.lstm is None:
            raise ImportError(
                "TemporalAnomalyDetector LSTM forecasting requires PyTorch. "
                "Install with `pip install 'mercury-agent-global[ml]'` or use the "
                "trend / change / seasonality detection paths which run on "
                "NumPy only."
            )
        return self.lstm

    def fit(self, data: np.ndarray[Any, Any] | torch.Tensor) -> TemporalAnomalyDetector:
        """Fit the temporal reference: per-feature level and innovation statistics.

        The fitted state is what detect() scores against. Before this, fit()
        stored a global mean/std that detect() never read, so every score was
        computed from the scored batch's own statistics: the anomalies
        contaminated their own reference (a batch with large spikes inflated
        the diff scale that was supposed to expose them), and the first
        ``window_size`` points scored a hard 0.

        Learns, per feature: the level mean/std (the trend channel's reference
        for points too early to have a full window) and the first-difference
        mean/std (the change channel's innovation scale). The scalar
        ``baseline_mean``/``baseline_std`` stay for checkpoint compatibility.
        """
        if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        data_np = np.asarray(data, dtype=np.float64)
        if data_np.ndim == 1:
            data_np = data_np.reshape(-1, 1)

        self.baseline_mean = float(np.mean(data_np))
        self.baseline_std = float(np.std(data_np))
        self.feature_mean_ = data_np.mean(axis=0)
        self.feature_std_ = data_np.std(axis=0) + 1e-6
        if data_np.shape[0] >= 2:
            diffs = np.diff(data_np, axis=0)
            self.diff_mean_ = np.abs(diffs).mean(axis=0)
            # Floor the innovation scale at 1% of the level scale: a
            # noiseless training series (e.g. an exact ramp) measures a
            # near-zero diff std, under which EVERY later innovation scores
            # as astronomically surprising and the whole batch saturates to
            # the same score. The floor caps the channel's sensitivity at
            # 100x the level scale instead of infinity.
            self.diff_std_ = np.maximum(diffs.std(axis=0) + 1e-6, 0.01 * self.feature_std_)
        else:
            self.diff_mean_ = None
            self.diff_std_ = None

        self._is_fitted = True
        self._calibrate_threshold_on_training_scores(data_np)
        return self

    #: Training-score quantile the decision threshold is cut at when the
    #: caller sets no explicit threshold: alarms on training-like data are
    #: bounded near 1 - this quantile.
    _THRESHOLD_CALIBRATION_QUANTILE = 0.995

    def _calibrate_threshold_on_training_scores(self, data_np: np.ndarray[Any, Any]) -> None:
        """Cut the alarm threshold from the training data's own scores.

        The combined score's normal regime is the squash of the max of three
        null z-channels, which sits nowhere near any fixed constant (and moved
        when the channels were rebuilt), so a fixed default threshold either
        never fires or fires on everything. The cut at the
        ``_THRESHOLD_CALIBRATION_QUANTILE`` of the training batch's own
        combined scores bounds the alarm rate on training-like data. An
        explicitly configured ``threshold``, or ``enable_auto_calibration``,
        is the caller's decision and is left alone (the same contract as
        AccelerationDynamicsDetector's fit-time calibration).
        """
        if "threshold" in self.config or self._auto_calibrate:
            return
        if data_np.shape[0] < 20:
            return
        trend = self._detect_trend_anomalies(data_np)
        change = self._detect_sudden_changes(data_np)
        prediction = self._detect_prediction_residuals(data_np)
        combined_z = np.maximum(np.maximum(trend, change), prediction)
        scores = combined_z / (3.0 + combined_z)
        self.threshold = float(np.quantile(scores, self._THRESHOLD_CALIBRATION_QUANTILE))

    def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Detect temporal anomalies with optional auto-calibration.

        Auto-Calibration:
            When auto_calibrate=True (via enable_auto_calibration()), the
            threshold is automatically calibrated based on the score
            distribution, solving the F1=0 problem.

        Returns:
            Dictionary containing:
                - is_anomaly: Boolean array of anomaly predictions
                - scores: Combined anomaly scores [0, 1]
                - trend_flags: Boolean trend anomaly indicators
                - change_flags: Boolean sudden change indicators
                - detector_type: "temporal"
                - threshold: Effective threshold (may be calibrated)
                - calibration_diagnostics: Diagnostics if auto-calibrated
        """
        if not self._is_fitted:
            raise DetectorException("Detector must be fitted before detection")

        data_np = data.cpu().numpy() if TORCH_AVAILABLE and isinstance(data, torch.Tensor) else data
        assert isinstance(data_np, np.ndarray)

        # Sanitise non-finite inputs at the boundary.  The detector
        # contract is "output scores are always finite" (asserted by
        # ``test_inf_in_data_sanitized``/``test_nan_in_data_sanitized``);
        # cleansing here pushes that guarantee in front of the
        # intermediate z-score / soft-normalization ops so they cannot
        # propagate inf or NaN, which would otherwise surface as
        # noisy RuntimeWarnings even though the final ``nan_to_num``
        # already cleans the output.
        if data_np.dtype.kind == "f" and not np.all(np.isfinite(data_np)):
            data_np = np.nan_to_num(data_np, nan=0.0, posinf=0.0, neginf=0.0)

        trend_anomalies = self._detect_trend_anomalies(data_np)
        change_anomalies = self._detect_sudden_changes(data_np)
        prediction_anomalies = self._detect_prediction_residuals(data_np)

        # Fix for P0: Validate and sanitize NaN/Inf before combining scores
        # Silent NaN propagation can corrupt downstream fusion
        if np.any(~np.isfinite(trend_anomalies)):
            trend_anomalies = np.nan_to_num(trend_anomalies, nan=0.0, posinf=1.0, neginf=0.0)
        if np.any(~np.isfinite(change_anomalies)):
            change_anomalies = np.nan_to_num(change_anomalies, nan=0.0, posinf=1.0, neginf=0.0)

        # The channels are specialized — trend fires on sustained level
        # shifts, prediction on point departures from the local level, change
        # on sharp spikes in the first differences — so a point is anomalous
        # when ANY fires; averaging halved whichever channel fired alone.
        # Channels combine on the raw z scale and are squashed once: squashing
        # each channel with its own scale and then taking the max let the
        # channel with the highest baseline bury another channel's signal.
        combined_z = np.maximum(np.maximum(trend_anomalies, change_anomalies), prediction_anomalies)
        combined_scores = combined_z / (3.0 + combined_z)

        # Final validation to ensure output is clean
        if np.any(~np.isfinite(combined_scores)):
            combined_scores = np.nan_to_num(combined_scores, nan=0.5, posinf=1.0, neginf=0.0)

        # Auto-calibration: compute optimal threshold from score distribution
        effective_threshold = self.threshold
        calibration_diagnostics = None

        if self._auto_calibrate:
            effective_threshold = self.calibrate_threshold(combined_scores)
            calibration_diagnostics = self._last_diagnostics

        is_anomaly = combined_scores > effective_threshold

        return {
            "is_anomaly": is_anomaly,
            "scores": combined_scores,
            "trend_flags": trend_anomalies > 3.0,
            "change_flags": change_anomalies > 3.0,
            "detector_type": "temporal",
            "threshold": effective_threshold,
            "calibration_diagnostics": calibration_diagnostics,
        }

    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> torch.Tensor:
        """Extract temporal features for ML fusion.

        Returns LSTM hidden-state features; requires the optional ``ml``
        extra (torch).  When torch is absent every reference in this
        method body (``torch.tensor``, ``torch.no_grad``) would raise
        ``NameError: name 'torch' is not defined`` before reaching
        :meth:`_lstm`.  Guard at the entry so callers receive the same
        actionable ImportError surface as ``_lstm()``.
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "TemporalAnomalyDetector.extract_features requires PyTorch. "
                "Install with `pip install 'mercury-agent-global[ml]'`, or use "
                "detect()/fit() which run on NumPy only."
            )

        data_np = data.cpu().numpy() if isinstance(data, torch.Tensor) else data

        if not self._is_fitted:
            self.fit(data_np)

        if data_np.ndim == 1:
            data_tensor = torch.tensor(data_np, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        else:
            data_tensor = torch.tensor(data_np, dtype=torch.float32).unsqueeze(-1)

        with torch.no_grad():
            _, (hidden, _) = self._lstm()(data_tensor)
            lstm_features = hidden[-1]

        return lstm_features

    def _detect_trend_anomalies(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Detect sustained deviations of the local level from the fitted level.

        Fitted path: the statistic for point ``t`` is a z-test of the trailing
        ``window_size`` MEDIAN (over a growing prefix before a full window
        exists) against the per-feature level reference learned by fit(),
        ``|rolling_median - fitted_mean| / (1.2533 * fitted_std /
        sqrt(window))`` (1.2533 is the median's efficiency factor at the
        Gaussian), reduced over features. The median, not the mean: an
        isolated spike inside the trailing window drags a window MEAN for the
        next ``window_size`` points, handing every neighbour of a point
        anomaly a raised trend score that buries the other channels' signal
        in the max-combine, while a median shrugs off up to half the window.
        A sustained shift moves the median fully, so shift detection keeps
        its power. A rolling statistic compared with the batch's own local
        window — the previous behaviour, kept below for states restored from
        legacy checkpoints — adapts INTO a level shift within one window
        length and then reports the shifted regime as normal, so it flagged
        only a shift's first few points. Point spikes are the prediction and
        change channels' case.

        Returns raw z per row; detect() squashes the combined channels once.
        """
        if len(data) == 0:
            return np.zeros(0)
        data_2d = data.reshape(len(data), -1)
        if (
            self.feature_mean_ is not None
            and self.feature_std_ is not None
            and data_2d.shape[1] == self.feature_mean_.shape[0]
        ):
            width = min(self.window_size, len(data_2d))
            n_rows = len(data_2d)
            rolling_median = np.empty_like(data_2d, dtype=np.float64)
            # Growing prefixes for the first width-1 rows, then full windows.
            for t in range(min(width - 1, n_rows)):
                rolling_median[t] = np.median(data_2d[: t + 1], axis=0)
            if n_rows >= width:
                windows = sliding_window_view(data_2d, width, axis=0)
                rolling_median[width - 1 :] = np.median(windows, axis=-1)
            counts = np.minimum(np.arange(1, n_rows + 1), width).astype(np.float64)
            median_efficiency = 1.2533
            z = np.abs(rolling_median - self.feature_mean_) / (
                median_efficiency * self.feature_std_ / np.sqrt(counts)[:, None]
            )
            return _reduce_feature_z(z)

        # Legacy fallback (unfitted per-feature reference, or a different
        # width): the original batch-local windowed z-score.
        scores = np.zeros(len(data))
        if len(data) < self.window_size:
            return scores

        # Score every point against the ``window_size`` points before it, a
        # block of points at a time so the window views stay small.
        width = self.window_size
        per_point = max(1, width * (data.shape[1] if data.ndim > 1 else 1))
        block = max(1, _TREND_BLOCK_ELEMENTS // per_point)
        for start in range(width, len(data), block):
            stop = min(start + block, len(data))
            # Window k of this view is ``data[start - width + k : start + k]``,
            # the history of point ``start + k``; the last window is unused.
            windows = sliding_window_view(data[start - width : stop], width, axis=0)[:-1]
            current = data[start:stop]
            window_mean = np.mean(windows, axis=-1)
            window_std = np.std(windows, axis=-1) + 1e-6
            z_score = np.abs((current - window_mean) / window_std)
            if data.ndim > 1:
                z_score = z_score.reshape(len(z_score), -1).max(axis=1)

            # Continuous score: sigmoid-like transformation for soft capping
            # Preserves ordering while keeping scores in reasonable range
            scores[start:stop] = z_score

        return scores

    def _detect_prediction_residuals(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """One-step-ahead prediction residuals under a local-level model.

        Point ``t`` is predicted by the mean of its trailing window (the
        fitted level reference for ``t = 0``) and the residual is scored
        against the FITTED noise scale, inflated by the predictor's own
        standard error: ``|x_t - mean(window)| / (sigma_fit * sqrt(1 + 1/w))``.
        This is the point-anomaly channel the differences cannot provide: a
        first difference compares two noisy samples (a sqrt(2) noise tax) and
        the spike-isolating min rule loses adjacent anomalies, where the step
        between two shifted points is ordinary. Scoring against the local
        prediction pays only the standard-error tax and sees each point on
        its own. Zeros when no per-feature reference is fitted (legacy
        checkpoints), matching their pre-upgrade channel set.
        """
        if len(data) == 0:
            return np.zeros(0)
        data_2d = data.reshape(len(data), -1)
        if (
            self.feature_mean_ is None
            or self.feature_std_ is None
            or data_2d.shape[1] != self.feature_mean_.shape[0]
            or len(data_2d) == 0
        ):
            return np.zeros(len(data))

        width = min(self.window_size, max(len(data_2d) - 1, 1))
        csum = np.vstack([np.zeros((1, data_2d.shape[1])), np.cumsum(data_2d, axis=0)])
        positions = np.arange(len(data_2d))
        counts = np.minimum(positions, width)
        tail = positions - counts
        with np.errstate(invalid="ignore", divide="ignore"):
            predicted = (csum[positions] - csum[tail]) / np.maximum(counts, 1)[:, None]
        # No history at t=0: predict from the fitted level, whose standard
        # error term vanishes (1/count -> 0 for the reference mean).
        predicted[counts == 0] = self.feature_mean_
        stderr = np.sqrt(1.0 + 1.0 / np.maximum(counts, 1).astype(np.float64))
        stderr[counts == 0] = 1.0
        z = np.abs(data_2d - predicted) / (self.feature_std_ * stderr[:, None])
        return _reduce_feature_z(z)

    def _detect_sudden_changes(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Detect sudden changes in values.

        Returns continuous scores without hard clipping.

        Fix for Issue #7: Uses soft normalization instead of np.minimum().
        """
        if len(data) < 2:
            return np.zeros(len(data))

        if len(data) == 0:
            return np.zeros(0)
        data_2d = data.reshape(len(data), -1)
        diffs = np.diff(data_2d, axis=0, prepend=data_2d[0:1])

        # Score innovations against the FITTED diff statistics when they
        # exist for this width. The batch's own statistics (the previous
        # reference, kept as the fallback) are contaminated by the very
        # anomalies being scored: one large spike inflates the scale that
        # should expose it and every other spike in the batch.
        if self.diff_std_ is not None and self.diff_std_.shape[0] == data_2d.shape[1]:
            diff_mean = self.diff_mean_
            diff_std = self.diff_std_
        else:
            diff_mean = np.mean(np.abs(diffs), axis=0)
            diff_std = np.std(diffs, axis=0) + 1e-6

        z_scores = np.abs(diffs - diff_mean) / diff_std
        row_z = _reduce_feature_z(z_scores)

        # A point anomaly at t produces a large step INTO t and a large step
        # OUT of it, while t's normal successor has only the large step in,
        # so the per-point score is min(step in, step out): the spike keeps
        # its score and stops bleeding onto the next point as a false alarm.
        # The last point has no outgoing step and keeps its incoming one, and
        # a level shift (large in, small out) is the trend channel's case.
        z_out = np.empty_like(row_z)
        z_out[:-1] = row_z[1:]
        z_out[-1] = row_z[-1]
        point_z = np.minimum(row_z, z_out)

        # change_threshold keeps its sensitivity meaning on the z scale: the
        # default 2.0 is neutral, a lower value amplifies this channel's z.
        result: np.ndarray[Any, Any] = point_z * (2.0 / self.change_threshold)
        return result

    def get_fitted_state(self) -> dict[str, Any] | None:
        """Export the fitted state for checkpoint round-tripping.

        Covers everything the detect/extract paths read: the fit-time
        baselines and the LSTM feature-extractor weights. The LSTM is now
        constructed under a fixed seed, but checkpoints written before that
        change carry per-process random weights, so persisting them keeps
        every existing checkpoint exact (ROADMAP row 16).

        Returns:
            JSON/tensor-safe mapping, or ``None`` when unfitted.
        """
        if not self._is_fitted:
            return None
        return {
            "baseline_mean": self.baseline_mean,
            "baseline_std": self.baseline_std,
            "threshold": float(self.threshold),
            "feature_mean": self.feature_mean_,
            "feature_std": self.feature_std_,
            "diff_mean": self.diff_mean_,
            "diff_std": self.diff_std_,
            "lstm_state_dict": dict(self.lstm.state_dict()) if self.lstm is not None else None,
        }

    def set_fitted_state(self, state: dict[str, Any]) -> None:
        """Restore a state produced by :meth:`get_fitted_state`."""
        self.baseline_mean = (
            float(state["baseline_mean"]) if state["baseline_mean"] is not None else None
        )
        self.baseline_std = (
            float(state["baseline_std"]) if state["baseline_std"] is not None else None
        )
        # Legacy checkpoints carry no per-feature references: detect() then
        # scores from the batch's own statistics (the pre-upgrade reference
        # for each channel; the channel combiner applies to every state).
        for attr, key in (
            ("feature_mean_", "feature_mean"),
            ("feature_std_", "feature_std"),
            ("diff_mean_", "diff_mean"),
            ("diff_std_", "diff_std"),
        ):
            value = state.get(key)
            setattr(self, attr, np.asarray(value, dtype=np.float64) if value is not None else None)
        saved_threshold = state.get("threshold")
        if saved_threshold is not None and "threshold" not in self.config:
            self.threshold = float(saved_threshold)
        lstm_state = state.get("lstm_state_dict")
        if lstm_state is not None and self.lstm is not None:
            self.lstm.load_state_dict(lstm_state)
        self._is_fitted = True
