"""
OMNI AVA (O+A)
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

"""
Benchmark Dataset Loaders for OMNI AVA.

Provides unified access to visual anomaly detection benchmark datasets:
- MVTec AD: Industrial defect detection
- UCF-Crime: Video anomaly detection in surveillance
- Shanghai Tech Campus: Campus surveillance anomaly detection

All datasets follow standard interfaces for easy integration with
visual anomaly detection models.
"""

from .base_dataset import (
    BaseDatasetConfig,
    BaseImageDataset,
    BaseVideoDataset,
    get_default_transforms,
)
from .mvtec import MVTecADConfig, MVTecADDataset
from .shanghai_tech import ShanghaiTechConfig, ShanghaiTechDataset
from .ucf_crime import UCFCrimeConfig, UCFCrimeDataset

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
