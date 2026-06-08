# Copyright (C) 2025 Steel Security Advisors LLC
"""Backwards-compatibility shim — canonical code lives in datasets.benchmarks."""

from omni_mercury_engine.datasets.benchmarks.shanghai_tech import (
    ShanghaiTechConfig,
    ShanghaiTechDataset,
)

__all__ = ["ShanghaiTechConfig", "ShanghaiTechDataset"]
