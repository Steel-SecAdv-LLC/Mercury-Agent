"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Advanced Anomaly Detection Module

State-of-the-art detectors addressing identified performance gaps:
- Time-Series: Multi-scale transformer with point-adjustment evaluation
- Industrial Control: Adversarial autoencoder with covariance modeling
- Contrastive Learning: SimCLR-style representation learning
- Copula-Based: Multivariate dependency modeling (COPOD)
- GWO-Enhanced: Grey Wolf Optimizer for ensemble fusion

Target Improvements:
- Time-series (SMD): F1 0.15-0.25 → 0.70+
- Industrial (BATADAL): F1 0.30-0.45 → 0.80+

Quick Start:
    >>> from omni_mercury_engine.detectors.advanced import create_detector
    >>> detector = create_detector("timeseries", input_dim=38)
    >>> detector.fit(X_train)
    >>> scores = detector.predict(X_test)
"""

from __future__ import annotations

from typing import Any, Literal

from omni_mercury_engine.detectors.advanced.adversarial_ae import (
    AdversarialAEConfig,
    AdversarialAutoencoderDetector,
)
from omni_mercury_engine.detectors.advanced.contrastive_detector import (
    ContrastiveConfig,
    ContrastiveLearningDetector,
)
from omni_mercury_engine.detectors.advanced.copod_detector import (
    COPODConfig,
    COPODDetector,
)
from omni_mercury_engine.detectors.advanced.gwo_ensemble import (
    GWOEnsembleConfig,
    GWOEnsembleDetector,
)
from omni_mercury_engine.detectors.advanced.multi_scale_transformer import (
    MultiScaleTransformerConfig,
    MultiScaleTransformerDetector,
)
from omni_mercury_engine.detectors.advanced.point_adjustment import (
    PointAdjustmentEvaluator,
    SegmentInfo,
    adjust_predictions,
    compute_adjusted_metrics,
    find_anomaly_segments,
)


# Type alias for detector types
DetectorType = Literal["timeseries", "industrial", "contrastive", "copod", "ensemble", "fast"]


def create_detector(
    detector_type: DetectorType,
    input_dim: int = 38,
    **kwargs: Any,
) -> (
    MultiScaleTransformerDetector
    | AdversarialAutoencoderDetector
    | ContrastiveLearningDetector
    | COPODDetector
    | GWOEnsembleDetector
):
    """
    Factory function to create optimized detectors for specific use cases.

    Args:
        detector_type: Type of detector to create
            - "timeseries": MultiScaleTransformerDetector (best for SMD, SMAP, MSL)
            - "industrial": AdversarialAutoencoderDetector (best for SWaT, BATADAL)
            - "contrastive": ContrastiveLearningDetector (representation learning)
            - "copod": COPODDetector (fast, parameter-free)
            - "ensemble": GWOEnsembleDetector (optimized ensemble)
            - "fast": COPODDetector (alias for speed)
        input_dim: Number of input features
        **kwargs: Additional arguments passed to detector constructor

    Returns:
        Configured detector instance

    Example:
        >>> detector = create_detector("timeseries", input_dim=38, epochs=50)
        >>> detector.fit(X_train)
        >>> result = detector.detect(X_test)
    """
    if detector_type == "timeseries":
        return MultiScaleTransformerDetector(input_dim=input_dim, **kwargs)
    elif detector_type == "industrial":
        return AdversarialAutoencoderDetector(input_dim=input_dim, **kwargs)
    elif detector_type == "contrastive":
        return ContrastiveLearningDetector(input_dim=input_dim, **kwargs)
    elif detector_type in ("copod", "fast"):
        return COPODDetector(**kwargs)
    elif detector_type == "ensemble":
        return GWOEnsembleDetector(**kwargs)
    else:
        raise ValueError(
            f"Unknown detector type: {detector_type}. "
            f"Valid types: timeseries, industrial, contrastive, copod, ensemble, fast"
        )


def list_detectors() -> dict[str, str]:
    """
    List available advanced detectors with descriptions.

    Returns:
        Dictionary mapping detector names to descriptions
    """
    return {
        "MultiScaleTransformerDetector": "Time-series anomaly detection with multi-scale attention",
        "AdversarialAutoencoderDetector": "Industrial control system anomaly detection",
        "ContrastiveLearningDetector": "Representation learning for anomaly detection",
        "COPODDetector": "Fast, parameter-free copula-based detection",
        "GWOEnsembleDetector": "Grey Wolf Optimizer enhanced ensemble",
        "PointAdjustmentEvaluator": "Fair evaluation protocol for time-series",
    }


__all__ = [
    # Configs
    "AdversarialAEConfig",
    # Detectors
    "AdversarialAutoencoderDetector",
    "COPODConfig",
    "COPODDetector",
    "ContrastiveConfig",
    "ContrastiveLearningDetector",
    # Type aliases
    "DetectorType",
    "GWOEnsembleConfig",
    "GWOEnsembleDetector",
    "MultiScaleTransformerConfig",
    "MultiScaleTransformerDetector",
    # Evaluation
    "PointAdjustmentEvaluator",
    "SegmentInfo",
    "adjust_predictions",
    "compute_adjusted_metrics",
    # Factory functions
    "create_detector",
    "find_anomaly_segments",
    "list_detectors",
]
