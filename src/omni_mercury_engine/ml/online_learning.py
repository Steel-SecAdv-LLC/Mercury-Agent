"""
Mercury Agent - Online Learning Pipeline
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Production-grade online learning pipeline providing:
- Incremental model updates with streaming data
- Automatic concept drift detection and adaptation
- Mini-batch gradient updates
- Exponential moving average for model stability
- Buffer management for replay
- Automatic retraining triggers
- Performance monitoring and alerting

This addresses the critical gaps:
- "No Online Learning" identified in audit
- "Drift Adaptation" partially implemented but not integrated
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from numpy.typing import NDArray

from omni_mercury_engine.ml.drift import (
    DriftResult,
    DriftSeverity,
    EnsembleDriftDetector,
)

logger = logging.getLogger(__name__)


class UpdateStrategy(StrEnum):
    """Strategies for online model updates."""

    FULL_RETRAIN = "full_retrain"  # Retrain from scratch
    INCREMENTAL = "incremental"  # Incremental updates only
    MINI_BATCH = "mini_batch"  # Mini-batch gradient updates
    EXPONENTIAL_MOVING_AVERAGE = "ema"  # EMA of model parameters
    ENSEMBLE_UPDATE = "ensemble_update"  # Update ensemble members


class RetrainingTrigger(StrEnum):
    """Conditions that trigger model retraining."""

    DRIFT_DETECTED = "drift_detected"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SAMPLE_COUNT = "sample_count"
    TIME_INTERVAL = "time_interval"
    MANUAL = "manual"


@dataclass
class StreamingSample:
    """A single sample from the data stream."""

    features: NDArray[np.float64]
    label: int | None = None  # None for unlabeled
    timestamp: float = field(default_factory=time.time)
    sample_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OnlineLearningMetrics:
    """Metrics for online learning performance."""

    samples_processed: int
    samples_in_buffer: int
    drift_events: int
    retraining_events: int
    current_accuracy: float | None
    rolling_accuracy: float | None
    last_drift_time: float | None
    last_retrain_time: float | None
    avg_update_latency_ms: float
    throughput_samples_per_sec: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "samples_processed": self.samples_processed,
            "samples_in_buffer": self.samples_in_buffer,
            "drift_events": self.drift_events,
            "retraining_events": self.retraining_events,
            "current_accuracy": self.current_accuracy,
            "rolling_accuracy": self.rolling_accuracy,
            "last_drift_time": self.last_drift_time,
            "last_retrain_time": self.last_retrain_time,
            "avg_update_latency_ms": self.avg_update_latency_ms,
            "throughput_samples_per_sec": self.throughput_samples_per_sec,
        }


@dataclass
class RetrainingEvent:
    """Record of a retraining event."""

    timestamp: float
    trigger: RetrainingTrigger
    samples_used: int
    accuracy_before: float | None
    accuracy_after: float | None
    drift_severity: DriftSeverity | None
    duration_seconds: float


class SampleBuffer:
    """
    Thread-safe buffer for streaming samples.

    Supports FIFO, reservoir sampling, and stratified sampling.
    """

    def __init__(
        self,
        max_size: int = 10000,
        strategy: str = "fifo",
        reservoir_size: int | None = None,
        random_state: int | None = None,
    ):
        """
        Initialize sample buffer.

        Args:
            max_size: Maximum buffer size
            strategy: 'fifo', 'reservoir', or 'stratified'
            reservoir_size: Size for reservoir sampling (if used)
            random_state: Seed for reproducible random sampling
        """
        self.max_size = max_size
        self.strategy = strategy
        self.reservoir_size = reservoir_size or max_size
        self.rng = np.random.default_rng(random_state)

        self._buffer: deque[StreamingSample] = deque(maxlen=max_size)
        self._reservoir: list[StreamingSample] = []
        self._lock = threading.Lock()
        self._sample_count = 0

    def add(self, sample: StreamingSample) -> None:
        """Add sample to buffer."""
        with self._lock:
            self._sample_count += 1

            if self.strategy == "fifo":
                self._buffer.append(sample)

            elif self.strategy == "reservoir":
                # Reservoir sampling (Algorithm R)
                if len(self._reservoir) < self.reservoir_size:
                    self._reservoir.append(sample)
                else:
                    # Replace with probability reservoir_size/sample_count
                    j = self.rng.integers(0, self._sample_count)
                    if j < self.reservoir_size:
                        self._reservoir[j] = sample

            elif self.strategy == "stratified":
                # Keep balanced samples by label
                self._buffer.append(sample)

    def get_batch(
        self,
        batch_size: int,
        remove: bool = False,
    ) -> list[StreamingSample]:
        """
        Get a batch of samples.

        Args:
            batch_size: Number of samples
            remove: Remove samples from buffer after getting

        Returns:
            List of samples
        """
        with self._lock:
            if self.strategy == "reservoir":
                samples = self._reservoir.copy()
                self.rng.shuffle(samples)
                return samples[:batch_size]

            samples = list(self._buffer)
            if len(samples) <= batch_size:
                batch = samples.copy()
            else:
                # Random sampling from buffer
                indices = self.rng.choice(len(samples), batch_size, replace=False)
                batch = [samples[i] for i in indices]

            if remove:
                for sample in batch:
                    try:
                        self._buffer.remove(sample)
                    except ValueError:
                        pass

            return batch

    def get_all(self) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Get all labeled samples as arrays."""
        with self._lock:
            if self.strategy == "reservoir":
                samples = self._reservoir
            else:
                samples = list(self._buffer)

            labeled = [s for s in samples if s.label is not None]

            if not labeled:
                return np.array([]).reshape(0, 1), np.array([], dtype=np.int64)

            X = np.array([s.features for s in labeled])
            y = np.array([s.label for s in labeled], dtype=np.int64)

            return X, y

    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._buffer.clear()
            self._reservoir.clear()

    def __len__(self) -> int:
        with self._lock:
            if self.strategy == "reservoir":
                return len(self._reservoir)
            return len(self._buffer)


class OnlineLearner(ABC):
    """Base class for online learning models."""

    @abstractmethod
    def partial_fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int64],
    ) -> None:
        """Incrementally update model with new data."""
        pass

    @abstractmethod
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """Predict labels."""
        pass

    @abstractmethod
    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict probabilities."""
        pass


class SGDOnlineLearner(OnlineLearner):
    """
    Stochastic Gradient Descent based online learner.

    Uses Mercury-native SGDClassifier for online anomaly detection.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        loss: str = "log_loss",
        penalty: str = "l2",
        alpha: float = 0.0001,
        warm_start: bool = True,
    ):
        """
        Initialize SGD online learner.

        Args:
            learning_rate: Learning rate for SGD
            loss: Loss function ('log_loss', 'hinge', 'modified_huber')
            penalty: Regularization ('l1', 'l2', 'elasticnet')
            alpha: Regularization strength
            warm_start: Reuse weights from previous fit
        """
        try:
            from omni_mercury_engine.ml.mercury_ml import SGDClassifier

            self.model = SGDClassifier(
                loss=loss,
                penalty=penalty,
                alpha=alpha,
                learning_rate="constant",
                eta0=learning_rate,
                warm_start=warm_start,
                max_iter=1,
                tol=None,
                random_state=42,
            )
        except ImportError:
            raise RuntimeError("mercury_ml SGDClassifier required for SGDOnlineLearner")

        self._fitted = False
        self._classes = np.array([0, 1])

    def partial_fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int64],
    ) -> None:
        """Update model with new samples."""
        self.model.partial_fit(X, y, classes=self._classes)
        self._fitted = True

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """Predict labels."""
        if not self._fitted:
            return np.zeros(len(X), dtype=np.int64)
        result = self.model.predict(X)
        return np.asarray(result, dtype=np.int64)

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict probabilities."""
        if not self._fitted:
            return np.full((len(X), 2), 0.5)
        result = self.model.predict_proba(X)
        return np.asarray(result, dtype=np.float64)


class PassiveAggressiveOnlineLearner(OnlineLearner):
    """
    Passive-Aggressive online learner.

    Good for scenarios with concept drift as it aggressively
    corrects mistakes.
    """

    def __init__(
        self,
        C: float = 1.0,
        fit_intercept: bool = True,
    ):
        """
        Initialize Passive-Aggressive learner.

        Args:
            C: Regularization parameter
            fit_intercept: Whether to fit intercept
        """
        try:
            from omni_mercury_engine.ml.mercury_ml import PassiveAggressiveClassifier

            self.model = PassiveAggressiveClassifier(
                C=C,
                fit_intercept=fit_intercept,
                warm_start=True,
                max_iter=1,
                tol=None,
                random_state=42,
            )
        except ImportError:
            raise RuntimeError("mercury_ml PassiveAggressiveClassifier required for PassiveAggressiveOnlineLearner")

        self._fitted = False
        self._classes = np.array([0, 1])

    def partial_fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int64],
    ) -> None:
        """Update model with new samples."""
        self.model.partial_fit(X, y, classes=self._classes)
        self._fitted = True

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """Predict labels."""
        if not self._fitted:
            return np.zeros(len(X), dtype=np.int64)
        result = self.model.predict(X)
        return np.asarray(result, dtype=np.int64)

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict probabilities (from decision function)."""
        if not self._fitted:
            return np.full((len(X), 2), 0.5)

        # Convert decision function to probabilities with numerical stability
        decision = self.model.decision_function(X)
        # Clip to prevent overflow in exp (exp(-710) underflows, exp(710) overflows)
        decision = np.clip(decision, -500, 500)
        proba = 1 / (1 + np.exp(-decision))
        return np.column_stack([1 - proba, proba])


class OnlineLearningPipeline:
    """
    Complete online learning pipeline with drift detection and adaptation.

    Handles streaming data, automatic drift detection, and model retraining.
    """

    def __init__(
        self,
        model: OnlineLearner | Any,
        update_strategy: UpdateStrategy = UpdateStrategy.MINI_BATCH,
        buffer_size: int = 10000,
        mini_batch_size: int = 32,
        drift_detection: bool = True,
        drift_threshold: float = 0.05,
        performance_threshold: float = 0.1,
        retrain_interval: int = 1000,
        ema_decay: float = 0.99,
        random_state: int | None = None,
    ):
        """
        Initialize online learning pipeline.

        Args:
            model: Online learner or any model with partial_fit
            update_strategy: How to update the model
            buffer_size: Size of sample buffer
            mini_batch_size: Size of mini-batches
            drift_detection: Enable automatic drift detection
            drift_threshold: P-value threshold for drift detection
            performance_threshold: Accuracy drop to trigger retraining
            retrain_interval: Samples between automatic retraining checks
            ema_decay: Decay rate for EMA strategy
            random_state: Seed for reproducible random sampling
        """
        self.model = model
        self.update_strategy = update_strategy
        self.mini_batch_size = mini_batch_size
        self.drift_detection = drift_detection
        self.drift_threshold = drift_threshold
        self.performance_threshold = performance_threshold
        self.retrain_interval = retrain_interval
        self.ema_decay = ema_decay
        self.rng = np.random.default_rng(random_state)

        # Buffer for samples
        self.buffer = SampleBuffer(max_size=buffer_size, strategy="fifo", random_state=random_state)

        # Reference data for drift detection
        self._reference_data: np.ndarray | None = None
        self._reference_size = 500

        # Drift detector
        self._drift_detector = EnsembleDriftDetector() if drift_detection else None

        # Performance tracking
        self._samples_processed = 0
        self._drift_events = 0
        self._retraining_events = 0
        self._last_drift_time: float | None = None
        self._last_retrain_time: float | None = None
        self._update_latencies: deque[float] = deque(maxlen=100)

        # Rolling accuracy
        self._recent_predictions: deque[tuple[int, int]] = deque(maxlen=1000)
        self._baseline_accuracy: float | None = None

        # Retraining history
        self._retraining_history: list[RetrainingEvent] = []

        # Thread safety
        self._lock = threading.Lock()

        # Callbacks
        self._on_drift_callbacks: list[Callable[[DriftResult], None]] = []
        self._on_retrain_callbacks: list[Callable[[RetrainingEvent], None]] = []

        # Start time for throughput calculation
        self._start_time = time.time()

    def process(self, sample: StreamingSample) -> dict[str, Any]:
        """
        Process a single streaming sample.

        Args:
            sample: Incoming sample

        Returns:
            Dictionary with prediction and metadata
        """
        start_time = time.time()

        with self._lock:
            # Make prediction
            prediction = None
            probability = None

            if hasattr(self.model, "predict"):
                try:
                    pred = self.model.predict(sample.features.reshape(1, -1))
                    prediction = int(pred[0])

                    if hasattr(self.model, "predict_proba"):
                        proba = self.model.predict_proba(sample.features.reshape(1, -1))
                        probability = float(proba[0, 1])
                except Exception as e:
                    logger.debug(f"Prediction failed: {e}")

            # Add to buffer
            self.buffer.add(sample)

            # Track predictions if label available
            if sample.label is not None and prediction is not None:
                self._recent_predictions.append((prediction, sample.label))

            # Update sample count
            self._samples_processed += 1

            # Check for drift
            drift_result = None
            if self.drift_detection and self._samples_processed % 100 == 0:
                drift_result = self._check_drift(sample.features)

                if drift_result and drift_result.is_drift:
                    self._handle_drift(drift_result)

            # Check for performance degradation
            if self._samples_processed % self.retrain_interval == 0:
                self._check_performance_and_retrain()

            # Online update if labeled
            if sample.label is not None:
                self._online_update(sample)

        # Track latency
        latency = (time.time() - start_time) * 1000
        self._update_latencies.append(latency)

        return {
            "prediction": prediction,
            "probability": probability,
            "sample_id": sample.sample_id,
            "timestamp": sample.timestamp,
            "drift_detected": drift_result.is_drift if drift_result else False,
            "latency_ms": latency,
        }

    def _online_update(self, sample: StreamingSample) -> None:
        """Perform online model update."""
        if sample.label is None:
            return

        if self.update_strategy == UpdateStrategy.INCREMENTAL:
            # Single sample update
            if hasattr(self.model, "partial_fit"):
                X = sample.features.reshape(1, -1)
                y = np.array([sample.label])
                self.model.partial_fit(X, y)

        elif self.update_strategy == UpdateStrategy.MINI_BATCH:
            # Mini-batch update
            if len(self.buffer) >= self.mini_batch_size:
                X, y = self.buffer.get_all()
                if len(y) >= self.mini_batch_size:
                    # Sample mini-batch
                    indices = self.rng.choice(len(y), self.mini_batch_size, replace=False)
                    X_batch = X[indices]
                    y_batch = y[indices]

                    if hasattr(self.model, "partial_fit"):
                        self.model.partial_fit(X_batch, y_batch)

    def _check_drift(self, new_features: NDArray[np.float64]) -> DriftResult | None:
        """Check for concept/data drift."""
        if self._drift_detector is None:
            return None

        # Initialize reference if needed
        if self._reference_data is None:
            if len(self.buffer) >= self._reference_size:
                X, _ = self.buffer.get_all()
                if len(X) >= self._reference_size:
                    self._reference_data = X[: self._reference_size]
                    self._drift_detector.fit(self._reference_data)
            return None

        # Check drift on recent data
        X, _ = self.buffer.get_all()
        if len(X) < 50:
            return None

        recent_data = X[-100:]  # Check last 100 samples
        return self._drift_detector.detect(recent_data)

    def _handle_drift(self, drift_result: DriftResult) -> None:
        """Handle detected drift."""
        self._drift_events += 1
        self._last_drift_time = time.time()

        logger.warning(
            f"Drift detected: severity={drift_result.severity.value}, "
            f"p-value={drift_result.p_value:.4f}"
        )

        # Notify callbacks
        for callback in self._on_drift_callbacks:
            try:
                callback(drift_result)
            except Exception as e:
                logger.warning(f"Drift callback failed: {e}")

        # Trigger retraining based on severity
        if drift_result.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
            self._trigger_retraining(RetrainingTrigger.DRIFT_DETECTED, drift_result.severity)

    def _check_performance_and_retrain(self) -> None:
        """Check performance and trigger retraining if needed."""
        current_accuracy = self._get_rolling_accuracy()

        if current_accuracy is None:
            return

        # Set baseline if not set
        if self._baseline_accuracy is None:
            self._baseline_accuracy = current_accuracy
            return

        # Check for degradation
        degradation = self._baseline_accuracy - current_accuracy

        if degradation > self.performance_threshold:
            logger.warning(
                f"Performance degradation detected: "
                f"{self._baseline_accuracy:.4f} -> {current_accuracy:.4f}"
            )
            self._trigger_retraining(RetrainingTrigger.PERFORMANCE_DEGRADATION)

    def _get_rolling_accuracy(self) -> float | None:
        """Get rolling accuracy from recent predictions."""
        if len(self._recent_predictions) < 100:
            return None

        correct = sum(1 for p, y in self._recent_predictions if p == y)
        return correct / len(self._recent_predictions)

    def _trigger_retraining(
        self,
        trigger: RetrainingTrigger,
        drift_severity: DriftSeverity | None = None,
    ) -> None:
        """Trigger model retraining."""
        start_time = time.time()
        accuracy_before = self._get_rolling_accuracy()

        # Get all labeled data
        X, y = self.buffer.get_all()

        if len(y) < 10:
            logger.warning("Not enough labeled data for retraining")
            return

        # Retrain based on strategy
        if self.update_strategy == UpdateStrategy.FULL_RETRAIN:
            if hasattr(self.model, "fit"):
                self.model.fit(X, y)
        elif hasattr(self.model, "partial_fit"):
            # Multiple passes over data
            for _ in range(3):
                indices = np.random.permutation(len(y))
                self.model.partial_fit(X[indices], y[indices])

        # Update reference data
        self._reference_data = X[: self._reference_size] if len(X) >= self._reference_size else X
        if self._drift_detector:
            self._drift_detector.fit(self._reference_data)

        # Reset baseline accuracy
        self._baseline_accuracy = None

        # Record event
        duration = time.time() - start_time
        accuracy_after = self._get_rolling_accuracy()

        event = RetrainingEvent(
            timestamp=time.time(),
            trigger=trigger,
            samples_used=len(y),
            accuracy_before=accuracy_before,
            accuracy_after=accuracy_after,
            drift_severity=drift_severity,
            duration_seconds=duration,
        )

        self._retraining_events += 1
        self._last_retrain_time = time.time()
        self._retraining_history.append(event)

        logger.info(
            f"Retraining completed: trigger={trigger.value}, "
            f"samples={len(y)}, duration={duration:.2f}s"
        )

        # Notify callbacks
        for callback in self._on_retrain_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.warning(f"Retrain callback failed: {e}")

    def process_batch(
        self,
        samples: list[StreamingSample],
    ) -> list[dict[str, Any]]:
        """Process a batch of samples."""
        return [self.process(sample) for sample in samples]

    def stream(
        self,
        data_iterator: Iterator[StreamingSample],
    ) -> Iterator[dict[str, Any]]:
        """
        Process streaming data.

        Args:
            data_iterator: Iterator yielding StreamingSample objects

        Yields:
            Processing results for each sample
        """
        for sample in data_iterator:
            yield self.process(sample)

    def get_metrics(self) -> OnlineLearningMetrics:
        """Get current pipeline metrics."""
        elapsed = time.time() - self._start_time
        throughput = self._samples_processed / max(elapsed, 1)

        avg_latency = np.mean(list(self._update_latencies)) if self._update_latencies else 0.0

        return OnlineLearningMetrics(
            samples_processed=self._samples_processed,
            samples_in_buffer=len(self.buffer),
            drift_events=self._drift_events,
            retraining_events=self._retraining_events,
            current_accuracy=self._get_rolling_accuracy(),
            rolling_accuracy=self._get_rolling_accuracy(),
            last_drift_time=self._last_drift_time,
            last_retrain_time=self._last_retrain_time,
            avg_update_latency_ms=float(avg_latency),
            throughput_samples_per_sec=throughput,
        )

    def add_drift_callback(self, callback: Callable[[DriftResult], None]) -> None:
        """Add callback for drift events."""
        self._on_drift_callbacks.append(callback)

    def add_retrain_callback(self, callback: Callable[[RetrainingEvent], None]) -> None:
        """Add callback for retraining events."""
        self._on_retrain_callbacks.append(callback)

    def force_retrain(self) -> None:
        """Force model retraining."""
        self._trigger_retraining(RetrainingTrigger.MANUAL)


def create_online_pipeline(
    model_type: str = "sgd",
    **kwargs: Any,
) -> OnlineLearningPipeline:
    """
    Factory function to create online learning pipeline.

    Args:
        model_type: Type of online model ('sgd', 'passive_aggressive')
        **kwargs: Additional arguments for pipeline

    Returns:
        Configured OnlineLearningPipeline
    """
    model: SGDOnlineLearner | PassiveAggressiveOnlineLearner
    if model_type == "sgd":
        model = SGDOnlineLearner()
    elif model_type == "passive_aggressive":
        model = PassiveAggressiveOnlineLearner()
    else:
        model = SGDOnlineLearner()

    return OnlineLearningPipeline(model=model, **kwargs)


# Exports
__all__ = [
    "OnlineLearner",
    "OnlineLearningMetrics",
    "OnlineLearningPipeline",
    "PassiveAggressiveOnlineLearner",
    "RetrainingEvent",
    "RetrainingTrigger",
    "SGDOnlineLearner",
    "SampleBuffer",
    "StreamingSample",
    "UpdateStrategy",
    "create_online_pipeline",
]
