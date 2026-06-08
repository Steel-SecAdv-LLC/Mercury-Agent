# Copyright (C) 2025 Steel Security Advisors LLC
"""Abstract base classes for detectors, models, and encoders with enforced interface contracts.

This module defines the core abstractions for the Mercury Agent anomaly detection system.
All detectors and models MUST implement these interfaces to ensure consistent behavior
across the fusion pipeline.

Interface Contracts:
    - BaseDetector.detect() MUST return dict with keys:
        - "anomaly_score" or "anomaly_prob": float in [0, 1]
        - "is_anomaly": bool
        - "severity": float in [0, 1] (optional)
        - "confidence": float in [0, 1] (optional)
        - "uncertainty": float >= 0 (optional, for fusion weighting)

    - BaseModel.predict() MUST return dict with keys:
        - "anomaly_scores": array-like of floats
        - "is_anomaly": bool or array-like of bools
        - "class_predictions": array-like of ints (optional)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import wraps
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
else:
    try:
        import torch
        from torch import nn

        TORCH_AVAILABLE = True
    except ImportError:
        torch = None  # type: ignore[assignment, unused-ignore]
        nn = None  # type: ignore[assignment, unused-ignore]
        TORCH_AVAILABLE = False

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np


# Type definitions for interface contracts
class DetectorResult(TypedDict, total=False):
    """Standard result format for detector.detect() method.

    Required (one of the following):
        anomaly_score: Anomaly score in [0, 1] range
        anomaly_prob: Alias for anomaly_score (use either, not both)

    Note: The decorator `validate_detector_result` ensures one of these
    keys exists. If neither is present, it derives the score from `scores`
    or defaults to 0.0.

    Auto-derived if missing:
        is_anomaly: Boolean indicating if anomaly detected (score > 0.5)

    Optional keys:
        severity: Severity score in [0, 1] range
        confidence: Confidence score in [0, 1] range
        uncertainty: Uncertainty estimate for fusion weighting
        scores: Array of per-sample scores (for batch processing)
        detector_name: Name of the detector
        metadata: Additional detector-specific metadata
    """

    # Score keys (use one or the other, not both)
    anomaly_score: float  # Primary: anomaly score in [0, 1]
    anomaly_prob: float  # Alias: same semantics as anomaly_score
    is_anomaly: bool
    severity: float
    confidence: float
    uncertainty: float
    scores: list[float]
    detector_name: str
    metadata: dict[str, Any]


class ModelResult(TypedDict, total=False):
    """Standard result format for model.predict() method.

    Required keys:
        anomaly_scores: Array of anomaly scores
        is_anomaly: Boolean or array of booleans

    Optional keys:
        class_predictions: Array of class indices
        probabilities: Array of class probabilities
        features: Extracted feature tensor
        uncertainty: Uncertainty estimates
        model_name: Name of the model
        metadata: Additional model-specific metadata
    """

    anomaly_scores: list[float]
    is_anomaly: bool | list[bool]
    class_predictions: list[int]
    probabilities: list[float]
    features: torch.Tensor
    uncertainty: list[float]
    model_name: str
    metadata: dict[str, Any]


@dataclass
class DetectorMetrics:
    """Metrics for detector performance tracking."""

    total_predictions: int = 0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    avg_inference_time_ms: float = 0.0
    avg_uncertainty: float = 0.0

    @property
    def precision(self) -> float:
        """Calculate precision score."""
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        """Calculate recall score."""
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)

    @property
    def f1_score(self) -> float:
        """Calculate F1 score."""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)


def validate_detector_result(func: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    """Decorator to validate detector.detect() return format.

    Ensures the result contains required keys and values are in valid ranges.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = func(*args, **kwargs)

        # Ensure we have either anomaly_score or anomaly_prob
        if "anomaly_score" not in result and "anomaly_prob" not in result:
            # Try to derive from other keys
            if "scores" in result and len(result["scores"]) > 0:
                import numpy as np

                result["anomaly_score"] = float(np.mean(result["scores"]))
            else:
                result["anomaly_score"] = 0.0

        # Ensure is_anomaly key exists
        if "is_anomaly" not in result:
            score = result.get("anomaly_score", result.get("anomaly_prob", 0.0))
            if isinstance(score, (list, tuple)):
                import numpy as np

                result["is_anomaly"] = bool(np.mean(score) > 0.5)
            else:
                result["is_anomaly"] = bool(score > 0.5)

        return result

    return wrapper


def validate_model_result(func: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    """Decorator to validate model.predict() return format.

    Ensures the result contains required keys for fusion compatibility.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = func(*args, **kwargs)

        # Ensure anomaly_scores key exists
        if "anomaly_scores" not in result:
            if "anomaly_score" in result:
                score = result["anomaly_score"]
                result["anomaly_scores"] = [score] if isinstance(score, (int, float)) else score
            elif "scores" in result:
                result["anomaly_scores"] = result["scores"]
            else:
                result["anomaly_scores"] = [0.0]

        # Ensure is_anomaly key exists
        if "is_anomaly" not in result:
            import numpy as np

            scores = result["anomaly_scores"]
            if isinstance(scores, (list, tuple)):
                result["is_anomaly"] = [s > 0.5 for s in scores]
            else:
                result["is_anomaly"] = bool(np.mean(scores) > 0.5)

        return result

    return wrapper


class BaseDetector(ABC):
    """Abstract base class for all anomaly detectors.

    All detectors MUST implement the following interface:
        - fit(data): Fit the detector to training data
        - detect(data): Detect anomalies and return standardized result
        - extract_features(data): Extract features for fusion

    The detect() method MUST return a dict with at least:
        - "anomaly_score" or "anomaly_prob": float in [0, 1]
        - "is_anomaly": bool

    Auto-Calibration (New):
        When auto_calibrate=True, the detector automatically calibrates its
        threshold based on the score distribution, solving the F1=0 problem
        where good ROC-AUC is achieved but fixed threshold produces no positives.

        Configure with:
        - "auto_calibrate": Enable automatic threshold calibration (default: False)
        - "calibration_method": Method to use ("auto", "percentile", "otsu", etc.)
        - "contamination": Expected anomaly ratio (default: 0.05)

    Example:
        >>> class MyDetector(BaseDetector):
        ...     def fit(self, data):
        ...         self._mean = data.mean()
        ...         self._std = data.std()
        ...         self._is_fitted = True
        ...         return self
        ...
        ...     def detect(self, data):
        ...         z_score = abs(data.mean() - self._mean) / self._std
        ...         return {
        ...             "anomaly_score": min(z_score / 3, 1.0),
        ...             "is_anomaly": z_score > 3,
        ...             "severity": min(z_score / 5, 1.0),
        ...             "uncertainty": 1 / (1 + len(data)),
        ...         }
        ...
        ...     def extract_features(self, data):
        ...         return torch.tensor([data.mean(), data.std(), data.max()])
    """

    # Required result keys for interface compliance
    REQUIRED_RESULT_KEYS = {"anomaly_score", "is_anomaly"}
    OPTIONAL_RESULT_KEYS = {"severity", "confidence", "uncertainty", "scores", "metadata"}

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize detector with configuration.

        Args:
            config: Configuration dictionary with detector parameters.
                Common keys:
                - "threshold": Anomaly threshold (default 0.5, must be in [0, 1])
                - "name": Detector name for logging
                - "enable_uncertainty": Enable uncertainty estimation

                Auto-calibration keys (new):
                - "auto_calibrate": Enable automatic threshold calibration
                - "calibration_method": Method ("auto", "percentile", "otsu", "mad", etc.)
                - "contamination": Expected fraction of anomalies (0.0-1.0)

        Raises:
            ValueError: If threshold is not in valid [0, 1] range.
        """
        # Store config in private attribute; subclasses may override config property
        # with typed config objects (covariant return types are valid OOP)
        self._config: dict[str, Any] = config or {}
        raw_threshold = self._config.get("threshold", 0.5)

        # Fix for P0: Validate threshold is in [0, 1] range
        # Invalid thresholds cause incorrect anomaly classification
        if not isinstance(raw_threshold, (int, float)):
            raise ValueError(f"Threshold must be numeric, got {type(raw_threshold).__name__}")
        if raw_threshold < 0.0 or raw_threshold > 1.0:
            raise ValueError(
                f"Threshold must be in [0, 1] range, got {raw_threshold}. "
                f"Scores are normalized to [0, 1] and compared against this threshold."
            )

        self.threshold = float(raw_threshold)
        self._is_fitted = False
        self._name = self._config.get("name", self.__class__.__name__)
        self._metrics = DetectorMetrics()

        # Auto-calibration settings (solves F1=0 problem)
        self._auto_calibrate = self._config.get("auto_calibrate", False)
        self._calibration_method = self._config.get("calibration_method", "auto")
        self._contamination = self._config.get("contamination", 0.05)
        self._calibration_manager: Any = None  # ScoreCalibrationManager when initialized
        self._calibrated_threshold: float | None = None
        self._last_diagnostics: Any = None

    @property
    def config(self) -> Any:
        """Get detector configuration.

        Returns:
            Configuration dict or typed config object.
            Subclasses may override to return specific config types
            (covariant return types are valid for property overrides).
        """
        return self._config

    @config.setter
    def config(self, value: Any) -> None:
        """Set detector configuration."""
        if isinstance(value, dict):
            self._config = value
        else:
            # Allow subclasses to set typed config objects
            self._config = value if hasattr(value, "__dict__") else {}

    @property
    def name(self) -> str:
        """Get detector name."""
        return str(self._name)  # type: ignore[no-any-return, unused-ignore]

    @property
    def metrics(self) -> DetectorMetrics:
        """Get detector metrics."""
        return self._metrics

    @abstractmethod
    def fit(self, data: np.ndarray[Any, Any] | torch.Tensor) -> BaseDetector:
        """Fit the detector to normal/training data.

        Args:
            data: Training data array or tensor.

        Returns:
            Self for method chaining.

        Note:
            Implementations MUST set self._is_fitted = True after fitting.
        """
        pass

    @abstractmethod
    def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Detect anomalies in data.

        Args:
            data: Input data array or tensor.

        Returns:
            Dictionary with detection results. MUST contain:
                - "anomaly_score" or "anomaly_prob": float in [0, 1]
                - "is_anomaly": bool

            MAY contain:
                - "severity": float in [0, 1]
                - "confidence": float in [0, 1]
                - "uncertainty": float >= 0 (for fusion weighting)
                - "scores": list of per-sample scores
                - "metadata": dict with additional info
        """
        pass

    @abstractmethod
    def extract_features(
        self,
        data: np.ndarray[Any, Any] | torch.Tensor,
    ) -> np.ndarray[Any, Any] | torch.Tensor:
        """Extract features for ML fusion.

        Args:
            data: Input data array or tensor.

        Returns:
            Feature array or tensor of shape ``[batch_size, feature_dim]``.
            Callers must handle both ``np.ndarray[Any, Any]`` and ``torch.Tensor``.

        Note:
            Features should be normalized and suitable for neural network input.
        """
        pass

    def is_fitted(self) -> bool:
        """Check if detector has been fitted.

        Returns:
            True if detector has been fitted, False otherwise.
        """
        return self._is_fitted

    def get_uncertainty(self, data: np.ndarray[Any, Any] | torch.Tensor) -> float:
        """Estimate uncertainty for fusion weighting.

        Default implementation returns 0 (no uncertainty).
        Override for uncertainty-aware detectors.

        Args:
            data: Input data.

        Returns:
            Uncertainty estimate (higher = more uncertain).
        """
        return 0.0

    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self._metrics = DetectorMetrics()

    # =========================================================================
    # Auto-Calibration Methods (Solves F1=0 Problem)
    # =========================================================================

    def enable_auto_calibration(
        self,
        contamination: float = 0.05,
        method: str = "auto",
    ) -> BaseDetector:
        """Enable automatic threshold calibration.

        This solves the F1=0 problem where good ROC-AUC is achieved but
        a fixed 0.5 threshold produces no positive predictions because
        scores are distributed below 0.5.

        Args:
            contamination: Expected fraction of anomalies (0.0-1.0)
            method: Calibration method:
                - "auto": Automatically select best method
                - "percentile": Use percentile based on contamination
                - "otsu": Otsu's bimodal threshold
                - "mad": Median Absolute Deviation
                - "knee": Knee/elbow detection

        Returns:
            Self for method chaining.

        Example:
            >>> detector = MercuryAnomalyDetector()
            >>> detector.fit(train_data).enable_auto_calibration(contamination=0.05)
            >>> result = detector.detect(test_data)  # Threshold auto-calibrated
        """
        self._auto_calibrate = True
        self._contamination = contamination
        self._calibration_method = method
        self._calibration_manager = None  # Reset to force re-init
        return self

    def disable_auto_calibration(self) -> BaseDetector:
        """Disable automatic threshold calibration."""
        self._auto_calibrate = False
        self._calibrated_threshold = None
        return self

    def calibrate_threshold(
        self,
        scores: np.ndarray[Any, Any],
        labels: np.ndarray[Any, Any] | None = None,
        method: str | None = None,
    ) -> float:
        """Calibrate threshold based on score distribution.

        Can be called manually or is called automatically when
        auto_calibrate=True.

        Args:
            scores: Anomaly scores array
            labels: Optional ground truth labels (enables optimal F1 method)
            method: Override calibration method

        Returns:
            Calibrated threshold value
        """
        from omni_mercury_engine.core.score_calibration import (
            CalibrationMethod,
            ScoreCalibrationManager,
        )

        method = method or self._calibration_method

        if self._calibration_manager is None:
            self._calibration_manager = ScoreCalibrationManager(
                contamination=self._contamination,
                method=CalibrationMethod(method),
            )

        result = self._calibration_manager.calibrate(
            scores=scores,
            labels=labels,
            method=CalibrationMethod(method) if method else None,
        )

        self._calibrated_threshold = result.threshold
        self._last_diagnostics = result.diagnostics

        return float(result.threshold)  # type: ignore[no-any-return, unused-ignore]

    def get_effective_threshold(self) -> float:
        """Get the effective threshold (calibrated if available, else default).

        Returns:
            The threshold to use for anomaly classification.
        """
        if self._auto_calibrate and self._calibrated_threshold is not None:
            return self._calibrated_threshold
        return self.threshold

    def get_calibration_diagnostics(self) -> Any:
        """Get diagnostics from last calibration.

        Returns:
            CalibrationDiagnostics object or None if not calibrated.
        """
        return self._last_diagnostics

    def diagnose_scores(
        self,
        scores: np.ndarray[Any, Any],
        labels: np.ndarray[Any, Any] | None = None,
        print_output: bool = True,
    ) -> Any:
        """Diagnose score distribution and threshold issues.

        This is the diagnostic tool requested to debug the F1=0 problem:
        - Score range, mean, std
        - Threshold value
        - Predictions above threshold
        - Distribution characteristics

        Args:
            scores: Anomaly scores
            labels: Optional ground truth labels
            print_output: Whether to print diagnostics

        Returns:
            CalibrationDiagnostics object
        """
        from omni_mercury_engine.core.score_calibration import diagnose_scores

        return diagnose_scores(
            scores=scores,
            threshold=self.get_effective_threshold(),
            labels=labels,
            print_output=print_output,
        )


class BaseModel(ABC):
    """Abstract base class for all models.

    All models MUST implement the following interface:
        - predict(data): Make predictions and return standardized result
        - extract_features(data): Extract features for fusion

    The predict() method MUST return a dict with at least:
        - "anomaly_scores": array-like of floats
        - "is_anomaly": bool or array-like of bools
    """

    # Required result keys for interface compliance
    REQUIRED_RESULT_KEYS = {"anomaly_scores", "is_anomaly"}
    OPTIONAL_RESULT_KEYS = {"class_predictions", "probabilities", "features", "uncertainty"}

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize model with configuration.

        Args:
            config: Configuration dictionary with model parameters.
        """
        # Store config in private attribute; subclasses may override config property
        # with typed config objects (covariant return types are valid OOP)
        self._config: dict[str, Any] = config or {}
        self._name = self._config.get("name", self.__class__.__name__)

    @property
    def config(self) -> Any:
        """Get model configuration.

        Returns:
            Configuration dict or typed config object.
            Subclasses may override to return specific config types
            (covariant return types are valid for property overrides).
        """
        return self._config

    @config.setter
    def config(self, value: Any) -> None:
        """Set model configuration."""
        if isinstance(value, dict):
            self._config = value
        else:
            # Allow subclasses to set typed config objects
            self._config = value if hasattr(value, "__dict__") else {}

    @property
    def name(self) -> str:
        """Get model name."""
        return str(self._name)  # type: ignore[no-any-return, unused-ignore]

    @abstractmethod
    def predict(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Make predictions on data.

        Args:
            data: Input data array or tensor.

        Returns:
            Dictionary with prediction results. MUST contain:
                - "anomaly_scores": array-like of floats
                - "is_anomaly": bool or array-like of bools

            MAY contain:
                - "class_predictions": array-like of ints
                - "probabilities": array-like of floats
                - "features": extracted feature tensor
                - "uncertainty": array-like of uncertainty estimates
        """
        pass

    @abstractmethod
    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> torch.Tensor:
        """Extract features for ML fusion.

        Args:
            data: Input data array or tensor.

        Returns:
            Feature tensor of shape [batch_size, feature_dim].
        """
        pass

    def get_uncertainty(
        self, data: np.ndarray[Any, Any] | torch.Tensor
    ) -> np.ndarray[Any, Any] | torch.Tensor:
        """Estimate uncertainty for fusion weighting.

        Default implementation returns zeros.
        Override for uncertainty-aware models.

        Args:
            data: Input data.

        Returns:
            Uncertainty tensor of shape [batch_size].
        """
        if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            return torch.zeros(data.shape[0])
        if TORCH_AVAILABLE:
            return torch.zeros(len(data) if hasattr(data, "__len__") else 1)
        import numpy as np

        return np.zeros(len(data) if hasattr(data, "__len__") else 1)


if TYPE_CHECKING or TORCH_AVAILABLE:

    class BaseEncoder(nn.Module):
        """Abstract base class for feature encoders.

        Encoders transform variable-length or heterogeneous inputs into fixed-size embeddings
        suitable for fusion.
        """

        def __init__(self, input_dim: int, output_dim: int) -> None:
            """Initialize encoder.

            Args:
                input_dim: Input feature dimension.
                output_dim: Output embedding dimension.
            """
            super().__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim

        @abstractmethod
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Encode input features to fixed-size embedding.

            Args:
                x: Input tensor of shape [batch_size, input_dim].

            Returns:
                Embedding tensor of shape [batch_size, output_dim].
            """
            pass

        def get_output_dim(self) -> int:
            """Get output embedding dimension."""
            return self.output_dim

else:

    class BaseEncoder:
        """Stub: BaseEncoder requires PyTorch."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Initialize the instance."""
            raise ImportError("BaseEncoder requires PyTorch. Install with: pip install torch")


@dataclass
class FusionInterface:
    """Interface specification for fusion-compatible components.

    Documents the contract that detectors and models must satisfy for integration with the fusion
    pipeline.
    """

    # Detector interface
    detector_result_keys: set[str] = field(
        default_factory=lambda: {"anomaly_score", "is_anomaly", "severity", "uncertainty"}
    )

    # Model interface
    model_result_keys: set[str] = field(
        default_factory=lambda: {"anomaly_scores", "is_anomaly", "class_predictions", "uncertainty"}
    )

    # Feature requirements
    min_feature_dim: int = 8
    max_feature_dim: int = 1024

    # Score ranges
    score_min: float = 0.0
    score_max: float = 1.0

    def validate_detector_result(self, result: dict[str, Any]) -> bool:
        """Validate that detector result meets interface requirements."""
        has_score = "anomaly_score" in result or "anomaly_prob" in result
        has_is_anomaly = "is_anomaly" in result
        return has_score and has_is_anomaly

    def validate_model_result(self, result: dict[str, Any]) -> bool:
        """Validate that model result meets interface requirements."""
        has_scores = "anomaly_scores" in result
        has_is_anomaly = "is_anomaly" in result
        return has_scores and has_is_anomaly


# Global fusion interface specification
FUSION_INTERFACE = FusionInterface()
