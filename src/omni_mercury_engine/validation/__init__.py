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
Real-World Dataset Validation Pipeline

Provides data loaders and validation utilities for:
- NSL-KDD (Security/Network intrusion detection)
- USGS Earthquake Data (Geological/Disaster detection)
- MIMIC-III (Medical - with IRB placeholder simulation)

All datasets are publicly available or simulated for research purposes.
"""

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
