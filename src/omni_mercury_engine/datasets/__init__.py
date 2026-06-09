# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Real-World Dataset Loaders for Mercury Agent.

All loaders fetch REAL DATA from official sources. Synthetic fallbacks are
disabled by default (set MERCURY_ALLOW_SYNTHETIC=1 to permit). Every loader
either returns real data with verified metadata or raises DataSourceUnavailableError.

Provides unified access to real-world datasets for benchmarking:
- ADBench: 47 tabular anomaly detection datasets from GitHub (fraud, thyroid, etc.)
- ADRepository: 21+ real-world anomaly detection datasets
- Time-Series: NAB, SMD, SMAP/MSL (standard anomaly detection benchmarks)
- Medical: MIT-BIH Arrhythmia (open access), MIMIC-III (credentialed)
- Space: NASA Exoplanet Archive (TAP API)
- Environmental: USGS earthquake, NOAA GSOD, EPA Air Quality, NASA FIRMS
- Security: NSL-KDD, CICIDS-2017 network intrusion
- Ocean: NOAA Buoy, NOAA ERDDAP (replaces Copernicus/SimonsCMAP/WorldOcean)
- Disaster: FEMA disaster declarations, hazard mitigation, NOAA Storm Events
- Industrial: BATADAL, SWaT/WADI (credential-gated stubs)

Quick Start:
    >>> from omni_mercury_engine.datasets import load_dataset, list_available_datasets
    >>> print(list_available_datasets())
    >>> X, y, meta = load_dataset('fraud')

All loaders follow official source licensing requirements.
"""

from __future__ import annotations

from .adbench import ADBENCH_CATALOG, ADBenchLoader
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
    mercury_baseline,
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
from .epa_air import EPAAirQualityLoader
from .exceptions import ALLOW_SYNTHETIC, DataSourceUnavailableError
from .industrial import BATADALLoader, SWaTLoader, WADILoader
from .medical import CardiologyDataset, MIMICLoader, PhysioNetLoader, SepsisDataset
from .metadata import LoaderDataset, LoaderDatasetMetadata
from .mitbih import MITBIHLoader
from .noaa_erddap import NOAAERDDAPLoader
from .noaa_gsod import NOAAGSODLoader
from .noaa_storm import NOAAStormEventsLoader
from .ocean import NOAABuoyLoader
from .security import CICIDSLoader, NSLKDDLoader, ThreatIntelLoader
from .space import NASAExoplanetLoader, SETILoader, SolarDynamicsLoader
from .timeseries import NABLoader, SMAPMSLLoader, SMDLoader
from .ucr_archive import CWRUBearingLoader, MBALoader, MSDSLoader, UCRLoader

__all__ = [
    # ADBench (47 tabular anomaly detection datasets)
    "ADBENCH_CATALOG",
    "ADREPOSITORY_DATASETS",
    # Core infrastructure
    "ALLOW_SYNTHETIC",
    "ADBenchLoader",
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
    "DataSourceUnavailableError",
    "DatasetConfig",
    "DatasetLoader",
    "DatasetRegistry",
    "DatasetSplit",
    # EPA Air Quality
    "EPAAirQualityLoader",
    "FEMADisasterLoader",
    "FEMAHazardMitigationLoader",
    "LoaderDataset",
    "LoaderDatasetMetadata",
    "MBALoader",
    "MIMICLoader",
    # MIT-BIH Arrhythmia
    "MITBIHLoader",
    "MSDSLoader",
    # MVTec AD dataset
    "MVTecADConfig",
    "MVTecADDataset",
    "NABLoader",
    "NASAExoplanetLoader",
    "NOAABuoyLoader",
    # NOAA ERDDAP (replaces Copernicus/SimonsCMAP/WorldOcean)
    "NOAAERDDAPLoader",
    # NOAA GSOD
    "NOAAGSODLoader",
    # NOAA Storm Events
    "NOAAStormEventsLoader",
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
    "list_available_datasets",
    "load_dataset",
    "mercury_baseline",
    "random_baseline",
]
