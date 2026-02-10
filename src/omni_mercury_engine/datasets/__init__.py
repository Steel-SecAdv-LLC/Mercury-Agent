"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisors LLC

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
Real-World Dataset Loaders for Mercury Agent

All loaders fetch REAL DATA from official sources with synthetic fallbacks.

Provides unified access to real-world datasets for benchmarking:
- ADRepository: 21+ real-world anomaly detection datasets (fraud, backdoor, thyroid, etc.)
- Time-Series: NAB, SMD, SMAP/MSL (standard anomaly detection benchmarks)
- Medical: MIMIC-III, MIMIC-IV, PhysioNet (credentialed access)
- Space: SETI signal archives, NASA exoplanet data
- Environmental: USGS earthquake API, NOAA storm events, NASA FIRMS
- Security: NSL-KDD, CICIDS-2017 network intrusion
- Ocean: NOAA buoy, Simons CMAP, World Ocean Database, Copernicus sea level
- Climate: Satellite altimetry, ocean biogeochemistry
- Disaster: FEMA disaster declarations, hazard mitigation programs

Quick Start:
    >>> from omni_mercury_engine.datasets import load_dataset, list_available_datasets
    >>> print(list_available_datasets())  # See all ADRepository datasets
    >>> X, y, meta = load_dataset('fraud')  # Load credit card fraud dataset

All loaders follow official source licensing requirements.
"""

from .adrepository import (
    ADREPOSITORY_DATASETS,
    ADRepositoryLoader,
    list_available_datasets,
    load_dataset,
)
from .base import DatasetConfig, DatasetLoader, DatasetRegistry, DatasetSplit
from .benchmarks import (
    BaseDatasetConfig,
    BaseImageDataset,
    BaseVideoDataset,
    BenchmarkComparison,
    BenchmarkResult,
    MVTecADConfig,
    MVTecADDataset,
    RealWorldBenchmarkSuite,
    ShanghaiTechConfig,
    ShanghaiTechDataset,
    UCFCrimeConfig,
    UCFCrimeDataset,
    get_default_transforms,
    isolation_forest_baseline,
    one_class_svm_baseline,
    random_baseline,
)
from .climate import (
    CopernicusSeaLevelLoader,
    SimonsCMAPLoader,
    WorldOceanDatabaseLoader,
)
from .disaster import (
    FEMADisasterLoader,
    FEMAHazardMitigationLoader,
)
from .environmental import (
    NOAAWeatherLoader,
    USGSEarthquakeLoader,
    USGSGeochemistryLoader,
    WildfireDataLoader,
)
from .industrial import BATADALLoader, SWaTLoader, WADILoader
from .medical import CardiologyDataset, MIMICLoader, PhysioNetLoader, SepsisDataset
from .ocean import NOAABuoyLoader
from .security import CICIDSLoader, NSLKDDLoader, ThreatIntelLoader
from .space import NASAExoplanetLoader, SETILoader, SolarDynamicsLoader
from .timeseries import NABLoader, SMAPMSLLoader, SMDLoader
from .ucr_archive import CWRUBearingLoader, MBALoader, MSDSLoader, UCRLoader

__all__ = [
    "ADREPOSITORY_DATASETS",
    "ADRepositoryLoader",
    "BATADALLoader",
    # Visual anomaly detection base classes (from data/benchmarks)
    "BaseDatasetConfig",
    "BaseImageDataset",
    "BaseVideoDataset",
    "BenchmarkComparison",
    "BenchmarkResult",
    "CICIDSLoader",
    "CWRUBearingLoader",
    "CardiologyDataset",
    "CopernicusSeaLevelLoader",
    "DatasetConfig",
    "DatasetLoader",
    "DatasetRegistry",
    "DatasetSplit",
    "FEMADisasterLoader",
    "FEMAHazardMitigationLoader",
    "MBALoader",
    "MIMICLoader",
    "MSDSLoader",
    # MVTec AD dataset
    "MVTecADConfig",
    "MVTecADDataset",
    "NABLoader",
    "NASAExoplanetLoader",
    "NOAABuoyLoader",
    "NOAAWeatherLoader",
    "NSLKDDLoader",
    "PhysioNetLoader",
    "RealWorldBenchmarkSuite",
    "SETILoader",
    "SMAPMSLLoader",
    "SMDLoader",
    "SWaTLoader",
    "SepsisDataset",
    # Shanghai Tech Campus dataset
    "ShanghaiTechConfig",
    "ShanghaiTechDataset",
    "SimonsCMAPLoader",
    "SolarDynamicsLoader",
    "ThreatIntelLoader",
    # UCF-Crime dataset
    "UCFCrimeConfig",
    "UCFCrimeDataset",
    "UCRLoader",
    "USGSEarthquakeLoader",
    "USGSGeochemistryLoader",
    "WADILoader",
    "WildfireDataLoader",
    "WorldOceanDatabaseLoader",
    # Benchmark utilities
    "get_default_transforms",
    "isolation_forest_baseline",
    "list_available_datasets",
    "load_dataset",
    "one_class_svm_baseline",
    "random_baseline",
]
