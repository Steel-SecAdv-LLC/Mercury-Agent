# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
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

import math
import weakref
from abc import ABC, ABCMeta, abstractmethod
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


def _serializable_state(value: Any, *, owner: str, attr: str) -> Any:
    """``value`` as a primitives-only payload, or raise ``NotImplementedError``.

    Accepted: None, bool, int, float, str, bytes, numpy scalars/arrays,
    torch tensors, and dict/list/tuple/set nests thereof (sets become sorted
    lists; dict keys must be primitives).  An ``nn.Module`` is captured as
    ``{"__module_state__": state_dict}``.  Anything else is a sign the
    detector holds richer learned objects and must override
    ``_learned_state`` itself -- half-saving silently is worse than failing.
    """
    import numpy as np

    # numpy objects are packed as raw bytes + dtype: the weights_only
    # unpickler refuses numpy's reconstruction globals, so a raw ndarray /
    # numpy scalar in the payload would make the checkpoint unloadable.
    if isinstance(value, np.generic):
        if value.dtype.hasobject:
            raise NotImplementedError(
                f"{owner}.{attr} holds an object-dtype numpy scalar, which the "
                "shared detector checkpoint cannot serialize safely"
            )
        return {"__npgeneric__": value.tobytes(), "dtype": value.dtype.str}
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject or value.dtype.names:
            raise NotImplementedError(
                f"{owner}.{attr} holds an object/structured-dtype numpy array, "
                "which the shared detector checkpoint cannot serialize safely; "
                "override _learned_state/_restore_learned_state for this detector"
            )
        return {
            "__ndarray__": np.ascontiguousarray(value).tobytes(),
            "dtype": value.dtype.str,
            "shape": tuple(int(s) for s in value.shape),
        }
    if TORCH_AVAILABLE and isinstance(value, torch.Tensor):
        return value
    if TORCH_AVAILABLE and isinstance(value, nn.Module):
        return {"__module_state__": value.state_dict()}
    if isinstance(value, dict):
        return {
            k: _serializable_state(v, owner=owner, attr=f"{attr}[{k!r}]") for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        items = [
            _serializable_state(v, owner=owner, attr=f"{attr}[{i}]") for i, v in enumerate(value)
        ]
        return tuple(items) if isinstance(value, tuple) else items
    if isinstance(value, (set, frozenset)):
        return sorted(
            (_serializable_state(v, owner=owner, attr=f"{attr}{{}}") for v in value),
            key=repr,
        )
    raise NotImplementedError(
        f"{owner}.{attr} holds a {type(value).__name__}, which the shared "
        "detector checkpoint cannot serialize safely; override _learned_state/"
        "_restore_learned_state for this detector"
    )


def _deserialized_state(value: Any) -> Any:
    """Invert :func:`_serializable_state`'s numpy packing on a loaded payload.

    Walks dict/list/tuple nests, turning ``{"__ndarray__": bytes, ...}`` and
    ``{"__npgeneric__": bytes, ...}`` wrappers back into the numpy objects
    they were saved from; ``{"__module_state__": ...}`` wrappers are left for
    :meth:`BaseDetector._restore_learned_state` to feed into ``load_state_dict``.
    """
    import numpy as np

    if isinstance(value, dict):
        keys = set(value)
        if keys == {"__npgeneric__", "dtype"}:
            return np.frombuffer(value["__npgeneric__"], dtype=value["dtype"])[0]
        if keys == {"__ndarray__", "dtype", "shape"}:
            flat = np.frombuffer(value["__ndarray__"], dtype=value["dtype"])
            return flat.reshape(tuple(value["shape"])).copy()
        if "__module_state__" in value:
            return value
        return {k: _deserialized_state(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deserialized_state(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_deserialized_state(v) for v in value)
    return value


def _finite_sample_mask(data: Any) -> Any:
    """Per-sample all-finite mask for a float array/tensor, else ``None``.

    Samples are axis-0 entries (rows of a matrix, elements of a series,
    frames of a stack). Integer and boolean arrays cannot hold NaN and map
    to ``None`` (nothing to mask); non-array inputs (mappings, lists of
    records) are left to the detector's own validators.
    """
    if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
        if not data.is_floating_point() or data.ndim == 0 or data.shape[0] == 0:
            return None
        flat_tensor = data.reshape(data.shape[0], -1)
        return torch.isfinite(flat_tensor).all(dim=1)
    import numpy as np

    if isinstance(data, np.ndarray):
        if data.dtype.kind != "f" or data.ndim == 0 or data.shape[0] == 0:
            return None
        flat_array = data.reshape(data.shape[0], -1)
        return np.isfinite(flat_array).all(axis=1)
    return None


#: Reinserted per-sample score for an unmeasured position: the maximally
#: uncertain mid-point, never the confident all-clear 0.0.  Unmeasured
#: positions are additionally named in ``nan_rows``/``unmeasured`` and the
#: verdict is marked ``partial``.
_UNMEASURED_NEUTRAL_SCORE = 0.5


def _check_finite_scores(detector: Any, result: dict[str, Any]) -> dict[str, Any]:
    """Refuse to serve a non-finite anomaly score (B9: never a NaN score)."""
    import numpy as np

    for key in ("anomaly_score", "anomaly_prob"):
        value = result.get(key)
        if value is not None and not math.isfinite(float(value)):
            raise RuntimeError(
                f"{type(detector).__name__} produced a non-finite {key}; "
                "refusing to serve a NaN score"
            )
    scores = result.get("scores")
    if scores is not None:
        score_array = np.asarray(scores, dtype=np.float64)
        if score_array.size and not np.all(np.isfinite(score_array)):
            raise RuntimeError(
                f"{type(detector).__name__} produced non-finite per-sample "
                "scores; refusing to serve NaN scores"
            )
    return result


def _nan_guard(func: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    """Treat non-finite samples as unmeasured; never serve a NaN score.

    Pre: non-finite samples (rows) of an array/tensor input are set aside
    before the wrapped ``detect`` runs, scored as the neutral mid-point at
    their original positions (alignment is preserved), counted and indexed
    in the result (``unmeasured`` + the tier's ``nan_rows`` convention),
    and the verdict is marked ``partial`` so it can never read as a full
    all-clear. An input with no finite sample at all raises ``ValueError``:
    an unmeasured stream is not a quiet one. Sanitization is metered
    through the tier's single entry point
    (:func:`~omni_mercury_engine.detectors.detection_config.apply_nan_policy`).

    A detector that implements its own non-finite contract declares
    ``handles_unmeasured = True`` and runs untouched except for the
    never-serve-a-NaN-score post-check.

    Post: a non-finite headline or per-sample score raises ``RuntimeError``
    instead of being served.
    """

    @wraps(func)
    def guarded(self: Any, data: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        import numpy as np

        if getattr(self, "handles_unmeasured", False):
            return _check_finite_scores(self, func(self, data, *args, **kwargs))

        mask = _finite_sample_mask(data)
        unmeasured_indices: list[int] = []
        cleaned = data
        total = 0
        if mask is not None:
            total = int(mask.shape[0])
            if TORCH_AVAILABLE and isinstance(mask, torch.Tensor):
                all_finite = bool(mask.all())
                unmeasured_indices = (~mask).nonzero(as_tuple=False).reshape(-1).tolist()
            else:
                all_finite = bool(np.all(mask))
                unmeasured_indices = np.flatnonzero(~mask).tolist()
            if not all_finite:
                if len(unmeasured_indices) == total:
                    raise ValueError(
                        f"{type(self).__name__}: every sample carries a non-finite "
                        "value; nothing is measurable in this input"
                    )
                # Meter through the tier's single sanitization entry point so
                # the omni_detector_nonfinite_corrected counter stays the one
                # place operators watch for dirty feeds.
                from omni_mercury_engine.detectors.detection_config import apply_nan_policy

                apply_nan_policy(
                    np.asarray(
                        (
                            data.detach().cpu().numpy()
                            if TORCH_AVAILABLE and isinstance(data, torch.Tensor)
                            else data
                        ),
                        dtype=np.float64,
                    ),
                    policy="neutral",
                    detector=getattr(self, "name", type(self).__name__),
                    field="input",
                )
                cleaned = data[mask]

        result = func(self, cleaned, *args, **kwargs)

        if unmeasured_indices:
            result = dict(result)
            unmeasured_set = set(unmeasured_indices)
            measured_indices = [i for i in range(total) if i not in unmeasured_set]
            # Re-expand per-sample arrays to the input's length so callers'
            # row alignment survives; unmeasured rows carry the neutral
            # mid-point and False, and are named below.
            # A scalar (whole-input) score has no rows to re-expand.
            scores = result.get("scores")
            if scores is not None and np.ndim(scores) == 1 and len(scores) == len(measured_indices):
                expanded = np.full(total, _UNMEASURED_NEUTRAL_SCORE, dtype=np.float64)
                expanded[measured_indices] = np.asarray(scores, dtype=np.float64).reshape(-1)
                result["scores"] = expanded if isinstance(scores, np.ndarray) else expanded.tolist()
            flags = result.get("is_anomaly")
            if not isinstance(flags, bool) and flags is not None:
                flag_array = np.asarray(flags).reshape(-1)
                if flag_array.size == len(measured_indices):
                    expanded_flags = np.zeros(total, dtype=bool)
                    expanded_flags[measured_indices] = flag_array.astype(bool)
                    result["is_anomaly"] = (
                        expanded_flags if isinstance(flags, np.ndarray) else expanded_flags.tolist()
                    )
            nan_rows = np.zeros(total, dtype=bool)
            nan_rows[unmeasured_indices] = True
            result["nan_rows"] = nan_rows.tolist()
            result["partial"] = True
            result["unmeasured"] = {
                "count": len(unmeasured_indices),
                "indices": [int(i) for i in unmeasured_indices],
                "reason": "non-finite values treated as unmeasured, set aside before scoring",
            }

        return _check_finite_scores(self, result)

    _NAN_GUARDED.add(guarded)
    return guarded


#: Every ``detect`` the guard has produced, so a guarded method inherited into
#: a subclass namespace is never wrapped twice.
_NAN_GUARDED: weakref.WeakSet[Callable[..., Any]] = weakref.WeakSet()


class _GuardedDetectorMeta(ABCMeta):
    """Builds each detector class with its own ``detect`` already guarded.

    Wrapping in the class namespace, before the class exists, replaces the
    former post-hoc ``cls.detect = ...`` reassignment.  ``BaseDetector``'s own
    ``detect`` (the abstract interface) is left as written: only classes
    derived from it are wrapped.
    """

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        /,
        **kwargs: Any,
    ) -> _GuardedDetectorMeta:
        detect = namespace.get("detect")
        derived = any(isinstance(base, _GuardedDetectorMeta) for base in bases)
        if derived and detect is not None and detect not in _NAN_GUARDED:
            namespace["detect"] = _nan_guard(detect)
        return super().__new__(mcs, name, bases, namespace, **kwargs)


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


class BaseDetector(ABC, metaclass=_GuardedDetectorMeta):
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

    #: True when ``extract_features`` and ``detect`` on a fitted instance touch
    #: no process-global mutable state (the shared random generator, the global
    #: torch RNG, unlocked module-level caches). Only such detectors are run
    #: concurrently by the engine (see ``core/parallel_detectors.py``). The
    #: default is the conservative ``False``: a detector runs sequentially, in
    #: order, until it is verified and opts in.
    parallel_safe: bool = False

    #: True when ``detect`` keeps streaming state in thread-local storage, so
    #: each calling thread owns its own stream. Such a detector is still run
    #: concurrently on the feature path (which resets transient state first),
    #: but plain ``detect()`` calls run it in the caller's thread so the
    #: caller's stream is the one that advances.
    thread_bound_state: bool = False

    # Each concrete ``detect`` is wrapped with the shared non-finite guard by
    # ``_GuardedDetectorMeta`` as the subclass is built.  Measured over the
    # manifest's engine-seam detectors before this guard: 25 accepted a NaN
    # sample silently (it failed every comparison and biased every statistic
    # it reached) and 6 returned a NaN anomaly score. The guard applies
    # uniformly to every subclass, so no detector built on this base can take
    # a NaN without saying so or serve one as a score.

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

    # =========================================================================
    # Learned-state persistence (shared save/load)
    # =========================================================================
    # 33 manifest detectors learned from data (fit) but could not save what
    # they learned: every deployment refit from scratch.  The shared
    # mechanism below serializes the learned instance state as a
    # primitives-only payload (torch.save), loaded back exclusively through
    # security.safe_torch.safe_torch_load (weights_only=True), mirroring the
    # visual detectors' checkpoint convention: no pickle execution surface.

    #: Instance attributes that are infrastructure, never learned state.
    _UNSAVED_ATTRS: frozenset[str] = frozenset(
        {
            "_config",
            "_metrics",
            "logger",
            "_calibration_manager",
            "_last_diagnostics",
            "_name",
        }
    )

    @staticmethod
    def _is_unsaved_infrastructure(value: Any) -> bool:
        """Construction-time infrastructure that never belongs in a checkpoint.

        Configuration objects travel with the constructor (the caller builds
        the detector with its config, then loads learned state -- the
        sklearn convention); RNGs, loggers, locks and thread-local storage
        are process state, not knowledge.
        """
        import logging
        import random
        import threading

        import numpy as np

        if isinstance(
            value,
            (
                logging.Logger,
                random.Random,
                np.random.Generator,
                np.random.RandomState,
                threading.local,
            ),
        ):
            return True
        if isinstance(value, (threading.Lock().__class__, threading.RLock().__class__)):
            return True
        # A torch.device is compute placement, decided by the constructor /
        # deployment config, never knowledge learned from data.
        if TORCH_AVAILABLE and isinstance(value, torch.device):
            return True
        type_name = type(value).__name__
        return type_name.endswith(("Config", "Configuration"))

    def _learned_state(self) -> dict[str, Any]:
        """The instance attributes that constitute what this detector learned.

        The default captures every instance attribute except the
        infrastructure names in :attr:`_UNSAVED_ATTRS`, and refuses (raises
        ``NotImplementedError``) when an attribute is not safely
        serializable as primitives / numpy / torch state -- a detector
        holding richer objects must override ``_learned_state`` /
        ``_restore_learned_state`` rather than silently half-saving.

        Returns:
            A primitives-only mapping (nested dict/list/tuple of numbers,
            strings, booleans, None, numpy arrays, torch tensors; an
            ``nn.Module`` attribute is stored as its ``state_dict`` under a
            ``{"__module_state__": ...}`` wrapper).
        """
        state: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if key in self._UNSAVED_ATTRS or self._is_unsaved_infrastructure(value):
                continue
            state[key] = _serializable_state(value, owner=type(self).__name__, attr=key)
        return state

    def _restore_learned_state(self, state: dict[str, Any]) -> None:
        """Reinstate the attributes captured by :meth:`_learned_state`."""
        for key, value in state.items():
            if isinstance(value, dict) and "__module_state__" in value:
                target = getattr(self, key, None)
                if not (TORCH_AVAILABLE and isinstance(target, nn.Module)):
                    raise ValueError(
                        f"checkpoint carries module weights for {type(self).__name__}.{key}, "
                        "but this instance has no constructed module there to load them "
                        "into; the detector must override _restore_learned_state to build it"
                    )
                target.load_state_dict(value["__module_state__"])
                continue
            setattr(self, key, _deserialized_state(value))

    def save(self, path: str) -> None:
        """Save the fitted detector's learned state to ``path``.

        The payload is primitives-only so it round-trips under
        ``weights_only=True`` on load; the detector's class is recorded and
        checked on load so a checkpoint can never silently restore into a
        different detector.

        Raises:
            ImportError: When torch is unavailable (persistence ships with
                the ``[ml]`` extra, matching the visual checkpoint path).
            NotImplementedError: When an attribute is not safely
                serializable and the subclass has not overridden
                ``_learned_state``.
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "BaseDetector.save requires torch (the [ml] extra): checkpoints "
                "are torch payloads loaded under weights_only=True"
            )
        torch.save(
            {
                "detector_class": f"{type(self).__module__}.{type(self).__qualname__}",
                "schema": 1,
                "threshold": float(self.threshold),
                "is_fitted": bool(self._is_fitted),
                "name": str(self._name),
                "auto_calibrate": bool(self._auto_calibrate),
                "calibrated_threshold": self._calibrated_threshold,
                "learned": self._learned_state(),
            },
            path,
        )

    def load(self, path: str) -> BaseDetector:
        """Restore a detector saved by :meth:`save` into this instance.

        Loading is hard-pinned to ``weights_only=True`` via
        ``security.safe_torch.safe_torch_load``; a checkpoint written for a
        different detector class is refused.

        Args:
            path: The checkpoint written by :meth:`save`.

        Returns:
            Self, fitted with the restored state.

        Raises:
            ValueError: When the checkpoint belongs to another detector
                class or carries an unknown schema.
        """
        from omni_mercury_engine.security.safe_torch import safe_torch_load

        payload = safe_torch_load(path, map_location="cpu")
        expected = f"{type(self).__module__}.{type(self).__qualname__}"
        if payload.get("detector_class") != expected:
            raise ValueError(
                f"checkpoint at {path!r} was saved by "
                f"{payload.get('detector_class')!r}, not {expected!r}; refusing "
                "to restore another detector's state"
            )
        if payload.get("schema") != 1:
            raise ValueError(f"unknown detector checkpoint schema {payload.get('schema')!r}")
        self._restore_learned_state(payload["learned"])
        self.threshold = float(payload["threshold"])
        self._is_fitted = bool(payload["is_fitted"])
        self._name = payload.get("name", self._name)
        self._auto_calibrate = bool(payload.get("auto_calibrate", self._auto_calibrate))
        self._calibrated_threshold = payload.get("calibrated_threshold")
        return self

    def is_fitted(self) -> bool:
        """Check if detector has been fitted.

        Returns:
            True if detector has been fitted, False otherwise.
        """
        return self._is_fitted

    @staticmethod
    def uncertainty_from_result(result: dict[str, Any], threshold: float = 0.5) -> float:
        """Margin-based decision uncertainty of one ``detect()`` result.

        Uncertainty is maximal (1.0) when the score sits exactly on the
        decision threshold and falls linearly to 0.0 at the score range's
        ends -- the standard least-confidence/margin uncertainty of a
        thresholded score (Settles 2009, Active Learning Literature Survey,
        Univ. of Wisconsin-Madison CS TR-1648, sec. 3.1).  For a batch
        result the per-sample margins are averaged.  A result that already
        carries an explicit ``uncertainty`` (a detector's own probabilistic
        estimate, e.g. predictive variance) is returned unchanged.

        Args:
            result: A ``detect()`` result dict.
            threshold: The decision threshold the scores are compared to.

        Returns:
            Uncertainty in [0, 1] (higher = closer to the decision boundary).
        """
        own = result.get("uncertainty")
        if own is not None:
            return float(own)
        scores = result.get("scores")
        if scores is None:
            scores = [result.get("anomaly_score", result.get("anomaly_prob", 0.0))]
        import numpy as np

        values = np.clip(np.asarray(scores, dtype=np.float64).reshape(-1), 0.0, 1.0)
        if values.size == 0:
            return 1.0
        span = max(threshold, 1.0 - threshold)
        if span <= 0.0:
            return 1.0
        margins = 1.0 - np.abs(values - threshold) / span
        return float(np.clip(np.mean(margins), 0.0, 1.0))

    def get_uncertainty(self, data: np.ndarray[Any, Any] | torch.Tensor) -> float:
        """Estimate uncertainty for fusion weighting.

        The default scores ``data`` and returns the margin-based decision
        uncertainty of the result (see :meth:`uncertainty_from_result`):
        1.0 at the decision boundary, 0.0 at the score extremes.  This
        replaces the former constant 0.0, which told fusion every detector
        was perfectly certain of every verdict.  Detectors with a genuine
        probabilistic uncertainty (predictive variance, posterior mass)
        should override this; detectors whose ``detect`` advances streaming
        state should prefer ``uncertainty_from_result`` on an already
        computed result.

        Args:
            data: Input data.

        Returns:
            Uncertainty estimate in [0, 1] (higher = more uncertain).
        """
        return self.uncertainty_from_result(
            self.detect(data), threshold=self.get_effective_threshold()
        )

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
