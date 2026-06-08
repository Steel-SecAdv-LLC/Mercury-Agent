# Copyright (C) 2025 Steel Security Advisors LLC
"""State-of-the-Art Anomaly Detection Models."""

from __future__ import annotations

from omni_mercury_engine.models.sota.association_discrepancy import (
    AnomalyTransformerEncoder,
    AssociationDiscrepancyModule,
    PriorAssociation,
    SeriesAssociation,
)
from omni_mercury_engine.models.sota.maat import (
    GatedFeatureFusion,
    MAATModel,
    MambaSSM,
    SparseAttention,
)
from omni_mercury_engine.models.sota.tranad import (
    AdversarialTrainer,
    FocusScoreConditioning,
    MAMLOptimizer,
    TranADModel,
)

__all__ = [
    "AdversarialTrainer",
    "AnomalyTransformerEncoder",
    "AssociationDiscrepancyModule",
    "FocusScoreConditioning",
    "GatedFeatureFusion",
    "MAATModel",
    "MAMLOptimizer",
    "MambaSSM",
    "PriorAssociation",
    "SeriesAssociation",
    "SparseAttention",
    "TranADModel",
]

__version__ = "1.7.0"
__author__ = "Steel Security Advisors LLC"
