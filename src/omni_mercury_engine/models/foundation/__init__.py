"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""
Foundation Model Adapters for Time-Series Anomaly Detection

Integrates state-of-the-art foundation models for time-series:
- TimeGPT: Nixtla's 100B+ parameter pre-trained model
- Chronos: Amazon's local inference model
- MOMENT: CMU's multi-task foundation model

Key Features:
    - Zero-shot anomaly detection
    - Fine-tuning on domain data
    - Ensemble predictions across models
    - Seamless integration with Mercury-Agent fusion pipeline

Example:
    Basic usage with TimeGPT::

        from omni_mercury_engine.models.foundation import TimeGPTAdapter

        adapter = TimeGPTAdapter(api_key="your_key")
        anomalies = adapter.detect_anomalies(time_series_data)

    Ensemble usage::

        from omni_mercury_engine.models.foundation import FoundationEnsemble

        ensemble = FoundationEnsemble(models=['timegpt', 'chronos'])
        results = ensemble.detect(time_series_data)
"""

from omni_mercury_engine.models.foundation.base_foundation import (
    BaseFoundationAdapter,
    BaseFoundationModel,
    ForecastResult,
    FoundationModelConfig,
)
from omni_mercury_engine.models.foundation.chronos_adapter import ChronosAdapter
from omni_mercury_engine.models.foundation.ensemble import FoundationEnsemble
from omni_mercury_engine.models.foundation.matrix_profile import MatrixProfileDetector
from omni_mercury_engine.models.foundation.timegpt_adapter import TimeGPTAdapter

# Compatibility aliases for tests
MatrixProfileAdapter = MatrixProfileDetector

__all__ = [
    # Base classes
    "BaseFoundationAdapter",
    "BaseFoundationModel",
    "ChronosAdapter",
    "ForecastResult",
    # Ensemble
    "FoundationEnsemble",
    "FoundationModelConfig",
    "MatrixProfileAdapter",  # Compatibility alias
    "MatrixProfileDetector",
    # Adapters
    "TimeGPTAdapter",
]
