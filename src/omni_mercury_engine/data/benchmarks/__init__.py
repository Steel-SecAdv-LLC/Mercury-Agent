# Copyright (C) 2025 Steel Security Advisors LLC
"""Backwards-compatibility shim — canonical code lives in datasets.benchmarks."""

from omni_mercury_engine.datasets.benchmarks import (
    BaseDatasetConfig,
    BaseImageDataset,
    BaseVideoDataset,
    MVTecADConfig,
    MVTecADDataset,
    ShanghaiTechConfig,
    ShanghaiTechDataset,
    UCFCrimeConfig,
    UCFCrimeDataset,
    get_default_transforms,
)

__all__ = [
    "BaseDatasetConfig",
    "BaseImageDataset",
    "BaseVideoDataset",
    "MVTecADConfig",
    "MVTecADDataset",
    "ShanghaiTechConfig",
    "ShanghaiTechDataset",
    "UCFCrimeConfig",
    "UCFCrimeDataset",
    "get_default_transforms",
]
