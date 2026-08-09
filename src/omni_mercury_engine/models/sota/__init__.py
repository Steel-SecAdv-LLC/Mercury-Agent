# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Third-party-derived anomaly detection architectures.

This module integrates architectures published in academic research:
- Association Discrepancy (Anomaly Transformer, ICLR 2022)
- TranAD (VLDB 2022)
- MAAT (Mamba Adaptive Anomaly Transformer, 2025)

All implementations respect Mercury-Agent's ethical scalars and its
civilization-first mission.

References:
    - Xu et al., "Anomaly Transformer: Time Series Anomaly Detection with
      Association Discrepancy", ICLR 2022
    - Tuli et al., "TranAD: Deep Transformer Networks for Anomaly Detection
      in Multivariate Time Series Data", VLDB 2022
    - Benaissa et al., "MAAT: Mamba Adaptive Anomaly Transformer", arXiv 2025
"""

from __future__ import annotations

from omni_mercury_engine._version import get_version as _get_version
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

__version__ = _get_version()
__author__ = "Steel Security Advisors LLC"
