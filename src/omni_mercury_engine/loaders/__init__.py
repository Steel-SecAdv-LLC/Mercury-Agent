# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Domain-specific data loaders for Mercury anomaly detection.

Each loader connects to real-world APIs and data sources, fetches historical events with ground
truth labels, and provides data in a format ready for MercuryAnomalyDetector.

All loaders implement the BaseDomainLoader interface.
"""

from __future__ import annotations

from omni_mercury_engine.loaders.base import BaseDomainLoader
from omni_mercury_engine.loaders.drought_loader import DroughtLoader
from omni_mercury_engine.loaders.earthquake_loader import EarthquakeLoader
from omni_mercury_engine.loaders.energy_loader import EnergyLoader
from omni_mercury_engine.loaders.fema_loader import FEMALoader
from omni_mercury_engine.loaders.financial_loader import FinancialLoader
from omni_mercury_engine.loaders.flood_loader import FloodLoader
from omni_mercury_engine.loaders.hail_loader import HailLoader
from omni_mercury_engine.loaders.heatwave_loader import HeatwaveLoader
from omni_mercury_engine.loaders.hurricane_loader import HurricaneLoader
from omni_mercury_engine.loaders.landslide_loader import LandslideLoader
from omni_mercury_engine.loaders.marine_loader import MarineLoader
from omni_mercury_engine.loaders.meteor_loader import MeteorLoader
from omni_mercury_engine.loaders.network_security_loader import NetworkSecurityLoader
from omni_mercury_engine.loaders.pandemic_loader import PandemicLoader
from omni_mercury_engine.loaders.sepsis_loader import SepsisLoader
from omni_mercury_engine.loaders.space_weather_loader import SpaceWeatherLoader
from omni_mercury_engine.loaders.tornado_loader import TornadoLoader
from omni_mercury_engine.loaders.transforms import prepare_for_detector
from omni_mercury_engine.loaders.tsunami_loader import TsunamiLoader
from omni_mercury_engine.loaders.volcanic_loader import VolcanicLoader
from omni_mercury_engine.loaders.wildfire_loader import WildfireLoader

__all__ = [
    "BaseDomainLoader",
    "DroughtLoader",
    "EarthquakeLoader",
    "EnergyLoader",
    "FEMALoader",
    "FinancialLoader",
    "FloodLoader",
    "HailLoader",
    "HeatwaveLoader",
    "HurricaneLoader",
    "LandslideLoader",
    "MarineLoader",
    "MeteorLoader",
    "NetworkSecurityLoader",
    "PandemicLoader",
    "SepsisLoader",
    "SpaceWeatherLoader",
    "TornadoLoader",
    "TsunamiLoader",
    "VolcanicLoader",
    "WildfireLoader",
    "prepare_for_detector",
]
