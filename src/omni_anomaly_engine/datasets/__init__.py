"""
OMNI ♱ AVA (O♱A)
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
Real-World Dataset Loaders for OMNI ♱ AVA

Provides unified access to real-world datasets for benchmarking:
- Medical: MIMIC-III, MIMIC-IV, PhysioNet
- Space: SETI signal archives, NASA exoplanet data
- Environmental: USGS earthquake, NOAA weather
- Security: NSL-KDD, CICIDS network intrusion

All loaders follow PhysioNet and source licensing requirements.
"""

from .base import DatasetConfig, DatasetLoader, DatasetRegistry, DatasetSplit
from .benchmarks import BenchmarkResult, RealWorldBenchmarkSuite
from .environmental import NOAAWeatherLoader, USGSEarthquakeLoader, WildfireDataLoader
from .medical import CardiologyDataset, MIMICLoader, PhysioNetLoader, SepsisDataset
from .security import CICIDSLoader, NSLKDDLoader, ThreatIntelLoader
from .space import NASAExoplanetLoader, SETILoader, SolarDynamicsLoader

__all__ = [
    # Base
    "DatasetLoader",
    "DatasetConfig",
    "DatasetSplit",
    "DatasetRegistry",
    # Medical
    "MIMICLoader",
    "PhysioNetLoader",
    "SepsisDataset",
    "CardiologyDataset",
    # Space
    "SETILoader",
    "NASAExoplanetLoader",
    "SolarDynamicsLoader",
    # Environmental
    "USGSEarthquakeLoader",
    "NOAAWeatherLoader",
    "WildfireDataLoader",
    # Security
    "NSLKDDLoader",
    "CICIDSLoader",
    "ThreatIntelLoader",
    # Benchmarks
    "RealWorldBenchmarkSuite",
    "BenchmarkResult",
]
