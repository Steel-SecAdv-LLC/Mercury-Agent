# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Backwards-compatibility shim — canonical code lives in datasets.benchmarks."""

from omni_mercury_engine.datasets.benchmarks.mvtec import MVTecADConfig, MVTecADDataset

__all__ = ["MVTecADConfig", "MVTecADDataset"]
