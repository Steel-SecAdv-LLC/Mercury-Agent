# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Affective computing anomaly detection model."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from omni_mercury_engine.utils.rng import DeterministicRNG, get_global_rng

logger = logging.getLogger(__name__)

#: Dict key under which a caller declares a genuine affective modality: an
#: emotion-probability time series of shape ``(time, 6)`` (one sample) or
#: ``(batch, time, 6)``, ordered per
#: :attr:`~omni_mercury_engine.core.model_domains.AffectiveStateModel.emotion_labels`
#: — ``["neutral", "happy", "sad", "angry", "fearful", "surprised"]``.
AFFECTIVE_EMOTIONS_KEY = "emotions"


class AffectiveAnomalyModel:
    """Affective computing model for emotional state anomaly detection.

    Modality contract:

    * **Declared affective input** — a dict carrying an emotion-probability
      time series under :data:`AFFECTIVE_EMOTIONS_KEY` — is analysed with the
      deterministic in-repo affective pipeline
      (:class:`~omni_mercury_engine.core.model_domains.AffectiveStateModel`):
      per-sample temporal emotion aggregation, entropy/negative-affect
      distress scoring (:meth:`~..AffectiveStateModel.detect_distress`),
      and a distress-driven anomaly score. Nothing is learned or fabricated;
      the pipeline is a documented deterministic heuristic over genuinely
      declared emotion distributions, and malformed declared input fails
      loud with ``ValueError``.

    * **Generic input** (arrays / other dicts) carries no affective modality,
      so the model emits a deterministic *neutral* output — zero features,
      0.5 (uninformative) scores — with a one-time warning.  Quarantine
      history (2026-06-11): this path previously emitted *fresh RNG noise
      per call*, feeding 64 columns of fabricated, nondeterministic signal
      into the fusion feature set at train and serve time — repeated
      ``detect_with_fusion`` calls on identical data moved fused
      probabilities by up to ±0.08, and no checkpoint could reproduce the
      engine that wrote it (ROADMAP row 16). Per the repo's anti-theater
      quarantine pattern (``models/parapsychology.py``,
      ``space/schumann_resonance.py``), off-modality input fabricates
      nothing.

    ``extract_features`` stays neutral (zeros) for **all** inputs: it is the
    fusion feature-group contract, and the shipped ``default_fusion.pt``
    checkpoint was trained against exactly that constant group — silently
    changing the fused feature distribution would invalidate the shipped
    checkpoint. A learned extractor requires a real labelled affect corpus
    (see docs/DORMANCY_LEDGER.md).
    """

    _quarantine_warned = False

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        rng: DeterministicRNG | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the instance."""
        self.config = config or {}
        self._rng = rng or get_global_rng()

    @classmethod
    def _warn_once(cls) -> None:
        if not cls._quarantine_warned:
            cls._quarantine_warned = True
            logger.warning(
                "AffectiveAnomalyModel has no learned feature extractor: "
                "extract_features returns zero features and predict on "
                "non-declared input returns the neutral 0.5 prior (never "
                "fabricated noise). Declared affective input "
                "(AFFECTIVE_EMOTIONS_KEY) is scored by the deterministic "
                "distress pipeline, not neutralised."
            )

    @staticmethod
    def _batch_size(data: np.ndarray[Any, Any] | dict[str, Any]) -> int:
        if isinstance(data, dict):
            data = np.array(next(iter(data.values())))
        elif not isinstance(data, np.ndarray):
            data = np.array(data)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return int(data.shape[0])

    @staticmethod
    def _declared_emotions(
        data: np.ndarray[Any, Any] | dict[str, Any],
    ) -> np.ndarray[Any, Any] | None:
        """Return the declared emotion series as ``(batch, time, 6)``, or None.

        Only a dict input carrying :data:`AFFECTIVE_EMOTIONS_KEY` declares the
        affective modality.  Declared-but-malformed input fails loud instead
        of being silently coerced.

        Raises:
            ValueError: If the declared series is not ``(time, 6)`` /
                ``(batch, time, 6)``, has an empty time axis, is non-finite,
                or is negative.
        """
        if not isinstance(data, dict) or AFFECTIVE_EMOTIONS_KEY not in data:
            return None
        emotions = np.asarray(data[AFFECTIVE_EMOTIONS_KEY], dtype=np.float64)
        if emotions.ndim == 2:
            emotions = emotions[np.newaxis, ...]
        if emotions.ndim != 3 or emotions.shape[-1] != 6:
            raise ValueError(
                f"'{AFFECTIVE_EMOTIONS_KEY}' must be an emotion-probability time "
                f"series of shape (time, 6) or (batch, time, 6); got shape "
                f"{np.asarray(data[AFFECTIVE_EMOTIONS_KEY]).shape}"
            )
        if emotions.shape[1] == 0:
            # An empty time axis would make the temporal mean NaN and poison
            # every downstream score; declared-but-empty input fails loud.
            raise ValueError(
                f"'{AFFECTIVE_EMOTIONS_KEY}' declares an empty emotion time "
                f"series (shape {emotions.shape}); at least one timestep is "
                "required"
            )
        if not np.all(np.isfinite(emotions)):
            raise ValueError(f"'{AFFECTIVE_EMOTIONS_KEY}' contains non-finite values")
        if np.any(emotions < 0.0):
            raise ValueError(
                f"'{AFFECTIVE_EMOTIONS_KEY}' must be non-negative emotion "
                "probabilities/intensities"
            )
        return emotions

    def extract_features(self, data: np.ndarray[Any, Any] | dict[str, Any]) -> np.ndarray[Any, Any]:
        """Extract affective features from data (neutral zeros; see class docstring)."""
        self._warn_once()
        return np.zeros((self._batch_size(data), 64), dtype=np.float32)

    def predict(self, data: np.ndarray[Any, Any] | dict[str, Any]) -> dict[str, Any]:
        """Predict emotional state anomalies.

        Declared affective input (see :data:`AFFECTIVE_EMOTIONS_KEY`) is
        analysed with the deterministic in-repo affective pipeline; any other
        input carries no affective modality and receives the neutral prior
        (see class docstring).
        """
        emotions = self._declared_emotions(data)
        if emotions is None:
            self._warn_once()
            batch_size = self._batch_size(data)
            return {
                "anomaly_scores": np.full(batch_size, 0.5, dtype=np.float32),
                "emotion_scores": np.zeros((batch_size, 6), dtype=np.float32),
                "distress_levels": np.full(batch_size, 0.5, dtype=np.float32),
            }

        # Deterministic declared-modality path: normalise each timestep to a
        # probability distribution, aggregate temporally, and score distress
        # with the documented entropy/negative-affect heuristic.
        from omni_mercury_engine.core.model_domains import AffectiveStateModel

        analyzer = AffectiveStateModel(n_emotions=6)
        batch = emotions.shape[0]
        emotion_scores = np.zeros((batch, 6), dtype=np.float32)
        distress_levels = np.zeros(batch, dtype=np.float32)
        for i in range(batch):
            step_sums = emotions[i].sum(axis=1, keepdims=True)
            normalised = np.where(
                step_sums > 0.0, emotions[i] / np.maximum(step_sums, 1e-12), 1.0 / 6.0
            )
            emotion_scores[i] = normalised.mean(axis=0).astype(np.float32)
            distress = analyzer.detect_distress(normalised)
            distress_levels[i] = np.float32(np.clip(distress["distress_level"], 0.0, 1.0))

        return {
            # Distress is the anomaly signal for a declared emotion series: a
            # sustained negative / high-entropy affect state is the anomalous
            # condition this model exists to flag.
            "anomaly_scores": distress_levels.copy(),
            "emotion_scores": emotion_scores,
            "distress_levels": distress_levels,
        }
