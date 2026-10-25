# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Real-World Dataset Validation Pipeline.

Provides data loaders and validation utilities for:
- NSL-KDD (Security/Network intrusion detection)
- USGS Earthquake Data (Geological/Disaster detection)
- MIMIC-III (Medical - with IRB placeholder simulation)

All datasets are publicly available or simulated for research purposes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from omni_mercury_engine.validation.pipeline import (
    ABTestResult,
    QualityCheckResult,
    ValidationPipeline,
    ValidationResult,
)

if TYPE_CHECKING:
    from omni_mercury_engine.validation.data_loaders import (
        DatasetLoader,
        MIMICLoader,
        NSLKDDLoader,
        USGSEarthquakeLoader,
    )

#: Loader names served from the deprecated ``data_loaders`` module. Importing that module
#: emits a DeprecationWarning, so it loads only when one of these is first used.
_LAZY_LOADER_NAMES = frozenset(
    {"DatasetLoader", "MIMICLoader", "NSLKDDLoader", "USGSEarthquakeLoader"}
)


def __getattr__(name: str) -> Any:
    if name in _LAZY_LOADER_NAMES:
        from omni_mercury_engine.validation import data_loaders

        return getattr(data_loaders, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
