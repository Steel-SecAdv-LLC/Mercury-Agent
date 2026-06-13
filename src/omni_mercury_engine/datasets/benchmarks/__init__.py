# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Benchmark Datasets and Evaluation Suite for Mercury Agent.

This module provides:

1. Visual Anomaly Detection Datasets:
   - MVTec AD: Industrial defect detection
   - UCF-Crime: Video anomaly detection in surveillance
   - Shanghai Tech Campus: Campus surveillance anomaly detection

2. Real-World Benchmark Suite:
   - BenchmarkResult: Results from benchmark runs
   - RealWorldBenchmarkSuite: Comprehensive benchmarking across datasets

All datasets follow standard interfaces for easy integration with
visual anomaly detection models.
"""

from __future__ import annotations

# Visual anomaly detection datasets (from data/benchmarks)
from .base_dataset import (
    BaseDatasetConfig,
    BaseImageDataset,
    BaseVideoDataset,
    get_default_transforms,
)
from .mvtec import MVTecADConfig, MVTecADDataset
from .shanghai_tech import ShanghaiTechConfig, ShanghaiTechDataset

# Real-world benchmark suite (from datasets/benchmarks.py)
from .suite import (
    BenchmarkComparison,
    BenchmarkResult,
    RealWorldBenchmarkSuite,
    mercury_baseline,
    random_baseline,
)
from .ucf_crime import UCFCrimeConfig, UCFCrimeDataset

__all__ = [
    # Visual anomaly detection base classes
    "BaseDatasetConfig",
    "BaseImageDataset",
    "BaseVideoDataset",
    # Benchmark suite
    "BenchmarkComparison",
    "BenchmarkResult",
    # MVTec AD
    "MVTecADConfig",
    "MVTecADDataset",
    "RealWorldBenchmarkSuite",
    # Shanghai Tech Campus
    "ShanghaiTechConfig",
    "ShanghaiTechDataset",
    # UCF-Crime
    "UCFCrimeConfig",
    "UCFCrimeDataset",
    "get_default_transforms",
    "mercury_baseline",
    "random_baseline",
]
