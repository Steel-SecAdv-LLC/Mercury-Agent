# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Real-World Dataset Validation Pipeline.

Provides data loaders and validation utilities for:
- NSL-KDD (Security/Network intrusion detection)
- USGS Earthquake Data (Geological/Disaster detection)
- MIMIC-III (Medical - with IRB placeholder simulation)

All datasets are publicly available or simulated for research purposes.
"""

from __future__ import annotations

from omni_mercury_engine.validation.data_loaders import (
    DatasetLoader,
    MIMICLoader,
    NSLKDDLoader,
    USGSEarthquakeLoader,
)
from omni_mercury_engine.validation.pipeline import (
    ABTestResult,
    QualityCheckResult,
    ValidationPipeline,
    ValidationResult,
)

__all__ = [
    "ABTestResult",
    "DatasetLoader",
    "MIMICLoader",
    "NSLKDDLoader",
    "QualityCheckResult",
    "USGSEarthquakeLoader",
    "ValidationPipeline",
    "ValidationResult",
]
