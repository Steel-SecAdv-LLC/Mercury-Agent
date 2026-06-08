# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Backwards-compatibility shim — canonical code lives in datasets.benchmarks."""

from omni_mercury_engine.datasets.benchmarks.base_dataset import (
    BaseDatasetConfig,
    BaseImageDataset,
    BaseVideoDataset,
    get_default_transforms,
)

__all__ = [
    "BaseDatasetConfig",
    "BaseImageDataset",
    "BaseVideoDataset",
    "get_default_transforms",
]
