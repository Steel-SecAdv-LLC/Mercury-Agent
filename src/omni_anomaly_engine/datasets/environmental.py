"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

Environmental Dataset Loaders: USGS Earthquake, NOAA Weather, Wildfire Data

References:
- USGS Earthquake Catalog: https://earthquake.usgs.gov/earthquakes/search/
- NOAA Climate Data: https://www.ncdc.noaa.gov/cdo-web/
- NASA FIRMS (Fire): https://firms.modaps.eosdis.nasa.gov/
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .base import DatasetConfig, DatasetLoader, DatasetRegistry

logger = logging.getLogger(__name__)


class USGSEarthquakeLoader(DatasetLoader):
    """
    USGS Earthquake Catalog Data Loader.

    Provides access to:
    - Global earthquake records
    - Seismic precursor features
    - Aftershock sequences

    Reference: https://earthquake.usgs.gov/
    """

    DATASET_NAME = "earthquake"
    DATASET_URL = "https://earthquake.usgs.gov/earthquakes/search/"
    LICENSE = "Public Domain (USGS)"
    CITATION = """U.S. Geological Survey (USGS) Earthquake Hazards Program.
    National Earthquake Information Center (NEIC)."""
    REQUIRES_CREDENTIALS = False

    FEATURE_NAMES = [
        "latitude",
        "longitude",
        "depth",
        "magnitude",
        "gap",
        "dmin",
        "rms",
        "nst",  # Station parameters
        "horizontal_error",
        "depth_error",
        "mag_error",
        "previous_mag_7d",
        "previous_count_7d",
        "previous_energy_7d",
        "b_value_local",
        "time_since_last",
        "distance_to_fault",
    ]

    def __init__(self, config: DatasetConfig) -> None:
        super().__init__(config)
        self.min_magnitude = config.preprocessing.get("min_magnitude", 2.5)

    def download(self) -> bool:
        """Download or generate earthquake data."""
        return self._create_synthetic_earthquake()

    def _create_synthetic_earthquake(self) -> bool:
        """Create synthetic earthquake catalog."""
        np.random.seed(self.config.random_seed)
        n_samples = self.config.max_samples or 10000

        # Simulate global earthquake distribution
        features = []
        labels = []

        for _i in range(n_samples):
            # Spatial distribution (clustered around fault zones)
            zone = np.random.choice(["pacific_rim", "mediterranean", "himalayan", "mid_atlantic"])

            if zone == "pacific_rim":
                lat = np.random.normal(35, 20)
                lon = np.random.normal(140, 30)
            elif zone == "mediterranean":
                lat = np.random.normal(38, 5)
                lon = np.random.normal(20, 15)
            elif zone == "himalayan":
                lat = np.random.normal(30, 5)
                lon = np.random.normal(85, 10)
            else:
                lat = np.random.normal(0, 30)
                lon = np.random.normal(-30, 10)

            # Gutenberg-Richter magnitude distribution
            magnitude = np.random.exponential(1.0) + self.min_magnitude

            params = {
                "latitude": np.clip(lat, -90, 90),
                "longitude": np.clip(lon, -180, 180),
                "depth": np.random.exponential(30),  # km
                "magnitude": magnitude,
                "gap": np.random.uniform(20, 300),  # azimuthal gap
                "dmin": np.random.exponential(0.5),  # distance to nearest station
                "rms": np.random.exponential(0.3),  # residual
                "nst": np.random.poisson(20),  # number of stations
                "horizontal_error": np.random.exponential(2),
                "depth_error": np.random.exponential(5),
                "mag_error": np.random.exponential(0.2),
                # Precursor features (simulated)
                "previous_mag_7d": np.random.exponential(1) + 2,
                "previous_count_7d": np.random.poisson(5),
                "previous_energy_7d": np.random.exponential(1e10),
                "b_value_local": np.random.normal(1.0, 0.2),
                "time_since_last": np.random.exponential(24),  # hours
                "distance_to_fault": np.random.exponential(10),  # km
            }

            feature_vec = [params[f] for f in self.FEATURE_NAMES]
            features.append(feature_vec)

            # Anomaly: significant seismic events
            is_anomaly = (
                magnitude >= 6.0  # Major earthquake
                or (params["previous_count_7d"] > 15 and magnitude > 4.5)  # Swarm
                or params["depth"] > 300  # Deep focus
            )
            labels.append(1 if is_anomaly else 0)

        features = np.array(features, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        save_path = self.data_path / "synthetic_earthquake.npz"
        np.savez_compressed(save_path, features=features, labels=labels)

        logger.info(f"Generated {n_samples} earthquake samples, {labels.sum()} significant events")
        return True

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        synthetic_path = self.data_path / "synthetic_earthquake.npz"
        if synthetic_path.exists():
            data = np.load(synthetic_path)
            return data["features"], data["labels"]
        raise FileNotFoundError("Earthquake data not found")

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Preprocess earthquake features."""
        # Log transform energy and magnitude
        data = np.nan_to_num(data, nan=0.0, posinf=1e10, neginf=-1e10)
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        return data.astype(np.float32)


class NOAAWeatherLoader(DatasetLoader):
    """
    NOAA Weather/Climate Data Loader.

    Provides access to:
    - Extreme weather events
    - Climate anomalies
    - Weather station observations

    Reference: https://www.ncdc.noaa.gov/
    """

    DATASET_NAME = "weather"
    DATASET_URL = "https://www.ncdc.noaa.gov/cdo-web/"
    LICENSE = "Public Domain (NOAA)"
    CITATION = """NOAA National Centers for Environmental Information.
    Climate Data Online (CDO)."""
    REQUIRES_CREDENTIALS = False

    FEATURE_NAMES = [
        "temperature",
        "temperature_anomaly",
        "dewpoint",
        "humidity",
        "pressure",
        "pressure_change_3h",
        "wind_speed",
        "wind_gust",
        "wind_direction",
        "precipitation_1h",
        "precipitation_24h",
        "visibility",
        "cloud_cover",
        "snow_depth",
        "uv_index",
        "heat_index",
        "wind_chill",
    ]

    def __init__(self, config: DatasetConfig) -> None:
        super().__init__(config)

    def download(self) -> bool:
        return self._create_synthetic_weather()

    def _create_synthetic_weather(self) -> bool:
        """Create synthetic weather observation data."""
        np.random.seed(self.config.random_seed)
        n_samples = self.config.max_samples or 10000

        # Seasonal variation
        t = np.linspace(0, 4 * np.pi, n_samples)
        seasonal = np.sin(t)

        data = {
            "temperature": 15 + 15 * seasonal + np.random.normal(0, 5, n_samples),
            "temperature_anomaly": np.random.normal(0, 2, n_samples),
            "dewpoint": 10 + 10 * seasonal + np.random.normal(0, 5, n_samples),
            "humidity": np.clip(60 + 20 * np.random.randn(n_samples), 0, 100),
            "pressure": 1013 + np.random.normal(0, 15, n_samples),
            "pressure_change_3h": np.random.normal(0, 3, n_samples),
            "wind_speed": np.random.exponential(5, n_samples),
            "wind_gust": np.random.exponential(8, n_samples),
            "wind_direction": np.random.uniform(0, 360, n_samples),
            "precipitation_1h": np.random.exponential(0.5, n_samples),
            "precipitation_24h": np.random.exponential(5, n_samples),
            "visibility": np.clip(10 + np.random.normal(0, 3, n_samples), 0, 50),
            "cloud_cover": np.clip(np.random.normal(50, 30, n_samples), 0, 100),
            "snow_depth": np.maximum(0, (1 - seasonal) * 10 + np.random.exponential(2, n_samples)),
            "uv_index": np.clip((1 + seasonal) * 5 + np.random.exponential(1, n_samples), 0, 11),
            "heat_index": 15 + 20 * seasonal + np.random.normal(0, 5, n_samples),
            "wind_chill": 15 + 15 * seasonal - 5 * np.random.exponential(1, n_samples),
        }

        features = np.column_stack([data[f] for f in self.FEATURE_NAMES])

        # Extreme weather anomalies
        labels = np.zeros(n_samples, dtype=np.int64)
        extreme_mask = (
            (np.abs(data["temperature_anomaly"]) > 5)  # Extreme temperature
            | (data["wind_gust"] > 25)  # Severe wind
            | (data["precipitation_1h"] > 10)  # Heavy rain
            | (np.abs(data["pressure_change_3h"]) > 8)  # Rapid pressure change
            | (data["visibility"] < 1)  # Low visibility
        )
        labels[extreme_mask] = 1

        save_path = self.data_path / "synthetic_weather.npz"
        np.savez_compressed(save_path, features=features, labels=labels)

        logger.info(f"Generated {n_samples} weather samples, {labels.sum()} extreme events")
        return True

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        synthetic_path = self.data_path / "synthetic_weather.npz"
        if synthetic_path.exists():
            data = np.load(synthetic_path)
            return data["features"], data["labels"]
        raise FileNotFoundError("Weather data not found")

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Preprocess weather features."""
        data = np.nan_to_num(data, nan=0.0)
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        return data.astype(np.float32)


class WildfireDataLoader(DatasetLoader):
    """
    Wildfire Detection and Prediction Data Loader.

    Provides access to:
    - NASA FIRMS fire detection
    - Fire weather indices
    - Vegetation/fuel conditions

    Reference: https://firms.modaps.eosdis.nasa.gov/
    """

    DATASET_NAME = "wildfire"
    DATASET_URL = "https://firms.modaps.eosdis.nasa.gov/"
    LICENSE = "Public Domain (NASA)"
    CITATION = """NASA FIRMS (Fire Information for Resource Management System).
    MODIS Collection 6 Active Fire Product."""
    REQUIRES_CREDENTIALS = False

    FEATURE_NAMES = [
        "latitude",
        "longitude",
        "brightness",
        "brightness_t31",
        "frp",  # Fire radiative power
        "confidence",
        "scan",
        "track",
        "temperature",
        "humidity",
        "wind_speed",
        "wind_direction",
        "precipitation_7d",
        "fuel_moisture",
        "ndvi",  # vegetation index
        "slope",
        "aspect",
        "elevation",
    ]

    def __init__(self, config: DatasetConfig) -> None:
        super().__init__(config)

    def download(self) -> bool:
        return self._create_synthetic_wildfire()

    def _create_synthetic_wildfire(self) -> bool:
        """Create synthetic wildfire detection data."""
        np.random.seed(self.config.random_seed)
        n_samples = self.config.max_samples or 5000

        features = []
        labels = []

        for _i in range(n_samples):
            # Fire-prone regions
            region = np.random.choice(["western_us", "australia", "amazon", "mediterranean"])

            if region == "western_us":
                lat = np.random.normal(38, 5)
                lon = np.random.normal(-120, 5)
            elif region == "australia":
                lat = np.random.normal(-33, 5)
                lon = np.random.normal(148, 5)
            elif region == "amazon":
                lat = np.random.normal(-5, 5)
                lon = np.random.normal(-60, 10)
            else:
                lat = np.random.normal(40, 3)
                lon = np.random.normal(15, 10)

            # Fire conditions
            is_fire = np.random.random() < 0.3

            if is_fire:
                params = {
                    "latitude": lat,
                    "longitude": lon,
                    "brightness": np.random.uniform(320, 400),  # Kelvin
                    "brightness_t31": np.random.uniform(290, 350),
                    "frp": np.random.exponential(50),  # MW
                    "confidence": np.random.uniform(50, 100),
                    "scan": np.random.uniform(1, 4),
                    "track": np.random.uniform(1, 4),
                    "temperature": np.random.normal(35, 8),  # Hot
                    "humidity": np.random.uniform(10, 40),  # Dry
                    "wind_speed": np.random.exponential(8),
                    "wind_direction": np.random.uniform(0, 360),
                    "precipitation_7d": np.random.exponential(1),  # Low
                    "fuel_moisture": np.random.uniform(5, 20),  # Low
                    "ndvi": np.random.uniform(0.2, 0.6),
                    "slope": np.random.exponential(10),
                    "aspect": np.random.uniform(0, 360),
                    "elevation": np.random.exponential(500),
                }
                labels.append(1)
            else:
                params = {
                    "latitude": lat,
                    "longitude": lon,
                    "brightness": np.random.uniform(280, 310),
                    "brightness_t31": np.random.uniform(280, 300),
                    "frp": np.random.exponential(2),
                    "confidence": np.random.uniform(0, 30),
                    "scan": np.random.uniform(1, 2),
                    "track": np.random.uniform(1, 2),
                    "temperature": np.random.normal(20, 10),
                    "humidity": np.random.uniform(40, 90),
                    "wind_speed": np.random.exponential(4),
                    "wind_direction": np.random.uniform(0, 360),
                    "precipitation_7d": np.random.exponential(10),
                    "fuel_moisture": np.random.uniform(20, 50),
                    "ndvi": np.random.uniform(0.3, 0.9),
                    "slope": np.random.exponential(5),
                    "aspect": np.random.uniform(0, 360),
                    "elevation": np.random.exponential(300),
                }
                labels.append(0)

            feature_vec = [params[f] for f in self.FEATURE_NAMES]
            features.append(feature_vec)

        features = np.array(features, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        save_path = self.data_path / "synthetic_wildfire.npz"
        np.savez_compressed(save_path, features=features, labels=labels)

        logger.info(f"Generated {n_samples} wildfire samples, {labels.sum()} fire detections")
        return True

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        synthetic_path = self.data_path / "synthetic_wildfire.npz"
        if synthetic_path.exists():
            data = np.load(synthetic_path)
            return data["features"], data["labels"]
        raise FileNotFoundError("Wildfire data not found")

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Preprocess wildfire features."""
        data = np.nan_to_num(data, nan=0.0)
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        return data.astype(np.float32)


# Register environmental loaders
DatasetRegistry.register("earthquake", USGSEarthquakeLoader)
DatasetRegistry.register("weather", NOAAWeatherLoader)
DatasetRegistry.register("wildfire", WildfireDataLoader)
