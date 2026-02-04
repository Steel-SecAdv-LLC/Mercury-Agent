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
"""

from omni_mercury_engine.detectors.advanced.adversarial_ae import (
    AdversarialAutoencoderDetector,
)
from omni_mercury_engine.detectors.advanced.contrastive_detector import (
    ContrastiveLearningDetector,
)
from omni_mercury_engine.detectors.advanced.copod_detector import COPODDetector
from omni_mercury_engine.detectors.advanced.gwo_ensemble import GWOEnsembleDetector
from omni_mercury_engine.detectors.advanced.multi_scale_transformer import (
    MultiScaleTransformerDetector,
)
from omni_mercury_engine.detectors.advanced.point_adjustment import (
    PointAdjustmentEvaluator,
    adjust_predictions,
)

__all__ = [
    "AdversarialAutoencoderDetector",
    "COPODDetector",
    "ContrastiveLearningDetector",
    "GWOEnsembleDetector",
    "MultiScaleTransformerDetector",
    "PointAdjustmentEvaluator",
    "adjust_predictions",
]
