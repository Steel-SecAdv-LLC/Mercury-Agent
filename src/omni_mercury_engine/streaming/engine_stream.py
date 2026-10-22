# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""One-sample-at-a-time detection through the engine.

Before this module, nothing took live data one sample at a time *through
the engine*: the streaming components (``streaming.streaming_detector``,
``detectors.detection_tier.TierStreamingScorer``) each drive a single
detector, while the engine's own surfaces (``detect``,
``detect_with_fusion``) are batch calls with no state between them.

:class:`EngineStreamSession` closes that gap: it holds a rolling window,
fits the engine's enabled detectors once at warm-up (and again every
``refit_interval`` samples, to track drift), and routes every new sample
through :meth:`OmniMercuryEngine.detect` -- the gated boundary, so the
harm-uplift gate applies to streaming verdicts exactly as to batch ones.
Detector state persists across samples; nothing is re-fit per record.

Example::

    engine = OmniMercuryEngine()
    session = engine.stream_session(detector_types=["statistical"])
    for value in live_feed:
        verdict = session.observe(value)
        if verdict["is_anomaly"]:
            alert(verdict)
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from omni_mercury_engine.engine import OmniMercuryEngine

logger = logging.getLogger(__name__)


class EngineStreamSession:
    """Stateful one-sample-at-a-time detection over an engine's detectors.

    The session keeps the most recent ``window_size`` samples. Each
    :meth:`observe` call appends one sample, and once ``min_samples`` have
    arrived, scores the window through ``engine.detect`` and reports the
    newest sample's verdict. Detectors are fitted on the warm-up window
    once, then re-fitted every ``refit_interval`` samples; between refits
    each call only scores, so per-sample cost is one detect pass, never a
    retrain.

    A non-finite sample is treated as unmeasured: it is excluded from the
    window, counted in ``samples_unmeasured``, and the returned verdict is
    marked ``partial`` -- it never poisons the window and never becomes a
    silent all-clear.
    """

    def __init__(
        self,
        engine: OmniMercuryEngine,
        *,
        detector_types: list[str] | None = None,
        window_size: int = 256,
        min_samples: int = 32,
        refit_interval: int = 128,
    ) -> None:
        """Initialize the session.

        Args:
            engine: The engine whose enabled detectors score the stream.
            detector_types: Detector names to run (default: all enabled).
            window_size: Rolling-window length scored on each sample (>= 2).
            min_samples: Samples to buffer before the first verdict (>= 2,
                <= window_size).
            refit_interval: Samples between detector refits (>= 1).

        Raises:
            ValueError: If a window/sample/interval parameter is out of range.
        """
        if window_size < 2:
            raise ValueError(f"window_size must be >= 2, got {window_size}")
        if not 2 <= min_samples <= window_size:
            raise ValueError(
                f"min_samples must be in [2, window_size={window_size}], got {min_samples}"
            )
        if refit_interval < 1:
            raise ValueError(f"refit_interval must be >= 1, got {refit_interval}")
        self._engine = engine
        self._detector_types = list(detector_types) if detector_types is not None else None
        self._window: deque[np.ndarray[Any, Any]] = deque(maxlen=window_size)
        self._min_samples = min_samples
        self._refit_interval = refit_interval
        self._samples_seen = 0
        self._samples_unmeasured = 0
        self._samples_since_fit = 0
        self._fitted = False
        self._feature_dim: int | None = None

    @property
    def samples_seen(self) -> int:
        """Total samples offered to :meth:`observe` (measured or not)."""
        return self._samples_seen

    @property
    def samples_unmeasured(self) -> int:
        """Samples excluded as non-finite (reported, never scored)."""
        return self._samples_unmeasured

    def _coerce(self, sample: Any) -> np.ndarray[Any, Any] | None:
        """One sample as a finite 1-D float vector, or None if unmeasured."""
        array = np.asarray(sample, dtype=np.float64).reshape(-1)
        if array.size == 0 or not np.all(np.isfinite(array)):
            return None
        if self._feature_dim is None:
            self._feature_dim = int(array.size)
        elif array.size != self._feature_dim:
            raise ValueError(
                f"sample has {array.size} features; this stream carries {self._feature_dim}"
            )
        return array

    def _fit_window(self, window: np.ndarray[Any, Any]) -> None:
        """Fit the session's detectors on the current window."""
        names = (
            self._detector_types
            if self._detector_types is not None
            else list(self._engine.detectors)
        )
        for name in names:
            detector = self._engine.detectors.get(name)
            if detector is None:
                raise KeyError(f"engine has no detector named {name!r}")
            detector.fit(window)
        self._fitted = True
        self._samples_since_fit = 0

    def observe(self, sample: Any, timestamp: Any = None) -> dict[str, Any]:
        """Feed one sample; return the newest sample's verdict.

        Args:
            sample: One observation -- a scalar or a fixed-length vector.
            timestamp: Optional timestamp echoed back in the verdict.

        Returns:
            Verdict dict: ``status`` ("warming_up", "ok" or "unmeasured"),
            ``partial`` (True while warming up or when this sample was
            unmeasured), ``is_anomaly``/``score``/``per_detector``/
            ``uncertainty`` once warmed up, plus the session counters.

        Raises:
            ValueError: On a sample whose dimensionality contradicts the
                stream's.
        """
        from omni_mercury_engine.core.base import BaseDetector

        self._samples_seen += 1
        array = self._coerce(sample)
        base: dict[str, Any] = {
            "timestamp": timestamp,
            "samples_seen": self._samples_seen,
            "samples_unmeasured": self._samples_unmeasured,
            "window_fill": len(self._window),
        }
        if array is None:
            self._samples_unmeasured += 1
            base["samples_unmeasured"] = self._samples_unmeasured
            # Unmeasured is reported, never scored and never an all-clear.
            return {**base, "status": "unmeasured", "partial": True, "is_anomaly": None}

        self._window.append(array)
        base["window_fill"] = len(self._window)
        if len(self._window) < self._min_samples:
            return {**base, "status": "warming_up", "partial": True, "is_anomaly": None}

        window = np.vstack(self._window)
        if not self._fitted or self._samples_since_fit >= self._refit_interval:
            self._fit_window(window)
        self._samples_since_fit += 1

        detection = self._engine.detect(window, detector_types=self._detector_types)
        per_detector: dict[str, dict[str, Any]] = {}
        flags: list[bool] = []
        scores: list[float] = []
        for name, result in detection["detectors"].items():
            raw_scores = result.get("scores")
            if raw_scores is not None and len(raw_scores) == window.shape[0]:
                last_score = float(np.clip(raw_scores[-1], 0.0, 1.0))
            else:
                last_score = float(
                    np.clip(result.get("anomaly_score", result.get("anomaly_prob", 0.0)), 0.0, 1.0)
                )
            is_anomaly = result.get("is_anomaly", False)
            if not isinstance(is_anomaly, bool):
                seq = np.asarray(is_anomaly).reshape(-1)
                is_anomaly = bool(seq[-1]) if seq.size == window.shape[0] else bool(seq.any())
            threshold = self._engine.detectors[name].get_effective_threshold()
            uncertainty = BaseDetector.uncertainty_from_result(
                {**result, "scores": [last_score]}, threshold=threshold
            )
            per_detector[name] = {
                "score": last_score,
                "is_anomaly": bool(is_anomaly),
                "uncertainty": float(uncertainty),
            }
            flags.append(bool(is_anomaly))
            scores.append(last_score)

        score = float(np.mean(scores)) if scores else 0.0
        certainty_weights = [1.0 - per_detector[n]["uncertainty"] for n in per_detector]
        if scores and sum(certainty_weights) > 0.0:
            # Certainty-weighted mean: a detector shrugging at its decision
            # boundary moves the stream verdict less than a certain one.
            score = float(np.average(scores, weights=np.asarray(certainty_weights) + 1e-9))
        verdict = {
            **base,
            "status": "ok",
            "partial": False,
            "is_anomaly": any(flags),
            "score": score,
            "per_detector": per_detector,
            "uncertainty": (
                float(np.mean([per_detector[n]["uncertainty"] for n in per_detector]))
                if per_detector
                else 1.0
            ),
        }
        if math.isnan(score):
            raise RuntimeError("stream verdict produced a NaN score; refusing to serve it")
        return verdict
