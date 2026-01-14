"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

Environmental Dataset Loaders: USGS Earthquake, NOAA Weather, Wildfire Data

References:
- USGS Earthquake Catalog: https://earthquake.usgs.gov/earthquakes/search/
- NOAA Climate Data: https://www.ncdc.noaa.gov/cdo-web/
- NASA FIRMS (Fire): https://firms.modaps.eosdis.nasa.gov/
"""

from __future__ import annotations

import io
import json
import logging
from typing import Any

import numpy as np

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False

from .base import DatasetConfig, DatasetLoader, DatasetRegistry

logger = logging.getLogger(__name__)


class USGSEarthquakeLoader(DatasetLoader):
    """
    USGS Earthquake Catalog Data Loader.

    Downloads REAL earthquake data from USGS API including:
    - Global earthquake records with magnitude, depth, location
    - Seismic quality metrics (gap, rms, nst, errors)
    - Computed precursor features

    Data source: https://earthquake.usgs.gov/fdsnws/event/1/
    License: Public Domain (USGS)
    """

    DATASET_NAME = "earthquake"
    DATASET_URL = "https://earthquake.usgs.gov/earthquakes/search/"
    LICENSE = "Public Domain (USGS)"
    CITATION = """U.S. Geological Survey (USGS) Earthquake Hazards Program.
    National Earthquake Information Center (NEIC)."""
    REQUIRES_CREDENTIALS = False

    # USGS API endpoint for GeoJSON earthquake data
    USGS_API_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    FEATURE_NAMES = [
        "latitude",
        "longitude",
        "depth",
        "magnitude",
        "gap",
        "dmin",
        "rms",
        "nst",
        "horizontal_error",
        "depth_error",
        "mag_error",
    ]

    def __init__(self, config: DatasetConfig) -> None:
        super().__init__(config)
        self.min_magnitude = config.preprocessing.get("min_magnitude", 2.5)
        self.days_back = config.preprocessing.get("days_back", 30)
        self._is_real_data = False

    @property
    def is_real_data(self) -> bool:
        """Return True if real data was loaded."""
        return self._is_real_data

    def download(self) -> bool:
        """Download real earthquake data from USGS API.

        Returns:
            True if download successful, False otherwise.
        """
        if self._download_from_usgs():
            return True

        logger.warning("USGS API failed, falling back to SYNTHETIC data.")
        return self._create_synthetic_earthquake()

    def _download_from_usgs(self) -> bool:
        """Download earthquake data from USGS Earthquake Hazards API."""
        import urllib.request
        from datetime import datetime, timedelta

        dataset_dir = self.data_path
        dataset_dir.mkdir(parents=True, exist_ok=True)
        cache_file = dataset_dir / "usgs_earthquake_real.npz"

        if cache_file.exists():
            logger.info(f"USGS earthquake data already cached at {cache_file}")
            self._is_real_data = True
            return True

        try:
            # Build API query for recent earthquakes
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.days_back)

            params = {
                "format": "geojson",
                "starttime": start_date.strftime("%Y-%m-%d"),
                "endtime": end_date.strftime("%Y-%m-%d"),
                "minmagnitude": str(self.min_magnitude),
                "limit": str(min(self.config.max_samples or 20000, 20000)),
                "orderby": "time",
            }

            query_string = "&".join(f"{k}={v}" for k, v in params.items())
            url = f"{self.USGS_API_URL}?{query_string}"

            logger.info(
                f"Downloading earthquake data from USGS API (last {self.days_back} days)..."
            )
            req = urllib.request.Request(  # noqa: S310
                url, headers={"User-Agent": "Mozilla/5.0 Mercury-Agent/1.0"}
            )
            with urllib.request.urlopen(req, timeout=120) as response:  # noqa: S310
                data = json.loads(response.read().decode("utf-8"))

            features_list = data.get("features", [])
            if not features_list:
                logger.warning("No earthquake data returned from USGS API")
                return False

            logger.info(f"Downloaded {len(features_list)} earthquake records from USGS")

            # Extract features from GeoJSON
            features, labels = self._process_usgs_geojson(features_list)

            # Save to cache
            np.savez_compressed(cache_file, features=features, labels=labels)
            self._is_real_data = True

            logger.info(
                f"USGS earthquake data loaded: {len(features)} samples, "
                f"{labels.sum()} significant events (is_real_data=True)"
            )
            return True

        except Exception as e:
            logger.warning(f"USGS API download failed: {e}")
            return False

    def _process_usgs_geojson(
        self, features_list: list[dict[str, Any]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Process USGS GeoJSON earthquake data.

        Args:
            features_list: List of GeoJSON feature objects

        Returns:
            Tuple of (features, labels) numpy arrays
        """
        rows = []

        for feature in features_list:
            props = feature.get("properties", {})
            geom = feature.get("geometry", {})
            coords = geom.get("coordinates", [0, 0, 0])

            row = {
                "longitude": coords[0] if len(coords) > 0 else 0,
                "latitude": coords[1] if len(coords) > 1 else 0,
                "depth": coords[2] if len(coords) > 2 else 0,
                "magnitude": props.get("mag", 0) or 0,
                "gap": props.get("gap", 180) or 180,
                "dmin": props.get("dmin", 1) or 1,
                "rms": props.get("rms", 0.5) or 0.5,
                "nst": props.get("nst", 10) or 10,
                "horizontal_error": props.get("horizontalError", 5) or 5,
                "depth_error": props.get("depthError", 10) or 10,
                "mag_error": props.get("magError", 0.3) or 0.3,
            }
            rows.append(row)

        # Convert to numpy array
        features = np.array(
            [[row[f] for f in self.FEATURE_NAMES] for row in rows], dtype=np.float32
        )

        # Handle NaN/inf
        features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)

        # Label significant events (magnitude >= 5.0 or deep focus)
        magnitudes = features[:, 3]  # magnitude column
        depths = features[:, 2]  # depth column
        labels = ((magnitudes >= 5.0) | (depths > 300)).astype(np.int64)

        # Apply max_samples limit
        if self.config.max_samples and len(features) > self.config.max_samples:
            np.random.seed(self.config.random_seed)
            indices = np.random.choice(len(features), self.config.max_samples, replace=False)
            features = features[indices]
            labels = labels[indices]

        return features, labels

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
        """Load earthquake data from cache (real data first, then synthetic)."""
        # Check for real data first
        real_cache = self.data_path / "usgs_earthquake_real.npz"
        if real_cache.exists():
            data = np.load(real_cache)
            self._is_real_data = True
            logger.info(f"Loaded REAL USGS earthquake data from {real_cache}")
            return data["features"], data["labels"]

        # Fall back to synthetic
        synthetic_path = self.data_path / "synthetic_earthquake.npz"
        if synthetic_path.exists():
            data = np.load(synthetic_path)
            self._is_real_data = False
            logger.info("Loaded SYNTHETIC earthquake data (is_real_data=False)")
            return data["features"], data["labels"]

        raise FileNotFoundError("Earthquake data not found. Run download() first.")

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Preprocess earthquake features."""
        # Log transform energy and magnitude
        data = np.nan_to_num(data, nan=0.0, posinf=1e10, neginf=-1e10)
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        return data.astype(np.float32)


class NOAAWeatherLoader(DatasetLoader):
    """
    Weather/Climate Data Loader using Open-Meteo API.

    Downloads REAL weather data including:
    - Temperature, humidity, pressure
    - Wind speed and direction
    - Precipitation and cloud cover
    - Derived indices (heat index, wind chill)

    Data source: https://open-meteo.com/ (free, no API key required)
    License: CC BY 4.0
    """

    DATASET_NAME = "weather"
    DATASET_URL = "https://open-meteo.com/"
    LICENSE = "CC BY 4.0 (Open-Meteo)"
    CITATION = """Open-Meteo Free Weather API. https://open-meteo.com/"""
    REQUIRES_CREDENTIALS = False

    # Open-Meteo API endpoint
    OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

    # Major cities for diverse weather sampling
    LOCATIONS = [
        {"name": "New York", "lat": 40.71, "lon": -74.01},
        {"name": "Los Angeles", "lat": 34.05, "lon": -118.24},
        {"name": "London", "lat": 51.51, "lon": -0.13},
        {"name": "Tokyo", "lat": 35.68, "lon": 139.69},
        {"name": "Sydney", "lat": -33.87, "lon": 151.21},
        {"name": "Dubai", "lat": 25.20, "lon": 55.27},
        {"name": "Moscow", "lat": 55.75, "lon": 37.62},
        {"name": "Mumbai", "lat": 19.08, "lon": 72.88},
    ]

    FEATURE_NAMES = [
        "temperature",
        "humidity",
        "pressure",
        "wind_speed",
        "wind_direction",
        "precipitation",
        "cloud_cover",
        "apparent_temperature",
    ]

    def __init__(self, config: DatasetConfig) -> None:
        super().__init__(config)
        self.days_back = config.preprocessing.get("days_back", 90)
        self._is_real_data = False

    @property
    def is_real_data(self) -> bool:
        """Return True if real data was loaded."""
        return self._is_real_data

    def download(self) -> bool:
        """Download real weather data from Open-Meteo API.

        Returns:
            True if download successful, False otherwise.
        """
        if self._download_from_open_meteo():
            return True

        logger.warning("Open-Meteo API failed, falling back to SYNTHETIC data.")
        return self._create_synthetic_weather()

    def _download_from_open_meteo(self) -> bool:
        """Download weather data from Open-Meteo Archive API."""
        import urllib.request
        from datetime import datetime, timedelta

        dataset_dir = self.data_path
        dataset_dir.mkdir(parents=True, exist_ok=True)
        cache_file = dataset_dir / "open_meteo_weather_real.npz"

        if cache_file.exists():
            logger.info(f"Weather data already cached at {cache_file}")
            self._is_real_data = True
            return True

        try:
            end_date = datetime.now() - timedelta(days=5)  # Archive has 5-day delay
            start_date = end_date - timedelta(days=self.days_back)

            all_features = []

            for loc in self.LOCATIONS:
                params = {
                    "latitude": loc["lat"],
                    "longitude": loc["lon"],
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d"),
                    "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,"
                    "wind_speed_10m,wind_direction_10m,precipitation,cloud_cover,"
                    "apparent_temperature",
                }

                query_string = "&".join(f"{k}={v}" for k, v in params.items())
                url = f"{self.OPEN_METEO_URL}?{query_string}"

                logger.info(f"Downloading weather data for {loc['name']}...")
                req = urllib.request.Request(  # noqa: S310
                    url, headers={"User-Agent": "Mozilla/5.0 Mercury-Agent/1.0"}
                )
                with urllib.request.urlopen(req, timeout=60) as response:  # noqa: S310
                    data = json.loads(response.read().decode("utf-8"))

                hourly = data.get("hourly", {})
                if not hourly:
                    continue

                n_records = len(hourly.get("time", []))
                for i in range(n_records):
                    row = [
                        hourly.get("temperature_2m", [None] * n_records)[i] or 0,
                        hourly.get("relative_humidity_2m", [None] * n_records)[i] or 50,
                        hourly.get("surface_pressure", [None] * n_records)[i] or 1013,
                        hourly.get("wind_speed_10m", [None] * n_records)[i] or 0,
                        hourly.get("wind_direction_10m", [None] * n_records)[i] or 0,
                        hourly.get("precipitation", [None] * n_records)[i] or 0,
                        hourly.get("cloud_cover", [None] * n_records)[i] or 0,
                        hourly.get("apparent_temperature", [None] * n_records)[i] or 0,
                    ]
                    all_features.append(row)

                logger.info(f"  Downloaded {n_records} hourly records from {loc['name']}")

            if not all_features:
                logger.warning("No weather data downloaded")
                return False

            features = np.array(all_features, dtype=np.float32)
            features = np.nan_to_num(features, nan=0.0)

            # Label extreme weather (statistical outliers)
            z_scores = np.abs((features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8))
            labels = (z_scores.max(axis=1) > 3.0).astype(np.int64)

            # Apply max_samples limit
            if self.config.max_samples and len(features) > self.config.max_samples:
                np.random.seed(self.config.random_seed)
                indices = np.random.choice(len(features), self.config.max_samples, replace=False)
                features = features[indices]
                labels = labels[indices]

            # Save to cache
            np.savez_compressed(cache_file, features=features, labels=labels)
            self._is_real_data = True

            logger.info(
                f"Weather data loaded: {len(features)} samples, "
                f"{labels.sum()} extreme events (is_real_data=True)"
            )
            return True

        except Exception as e:
            logger.warning(f"Open-Meteo API download failed: {e}")
            return False

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
        """Load weather data from cache (real data first, then synthetic)."""
        # Check for real data first
        real_cache = self.data_path / "open_meteo_weather_real.npz"
        if real_cache.exists():
            data = np.load(real_cache)
            self._is_real_data = True
            logger.info(f"Loaded REAL weather data from {real_cache}")
            return data["features"], data["labels"]

        # Fall back to synthetic
        synthetic_path = self.data_path / "synthetic_weather.npz"
        if synthetic_path.exists():
            data = np.load(synthetic_path)
            self._is_real_data = False
            logger.info("Loaded SYNTHETIC weather data (is_real_data=False)")
            return data["features"], data["labels"]

        raise FileNotFoundError("Weather data not found. Run download() first.")

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Preprocess weather features."""
        data = np.nan_to_num(data, nan=0.0)
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        return data.astype(np.float32)


class WildfireDataLoader(DatasetLoader):
    """
    NASA FIRMS Active Fire Data Loader.

    Downloads REAL wildfire detection data from NASA FIRMS including:
    - MODIS and VIIRS satellite fire detections
    - Brightness temperature, fire radiative power
    - Confidence scores and scan/track geometry

    Data source: https://firms.modaps.eosdis.nasa.gov/
    License: Public Domain (NASA)
    """

    DATASET_NAME = "wildfire"
    DATASET_URL = "https://firms.modaps.eosdis.nasa.gov/"
    LICENSE = "Public Domain (NASA)"
    CITATION = """NASA FIRMS (Fire Information for Resource Management System).
    MODIS Collection 6.1 and VIIRS Active Fire Products."""
    REQUIRES_CREDENTIALS = False

    # NASA FIRMS public CSV data URLs (no API key needed)
    FIRMS_URLS = {
        "modis_7d": "https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/MODIS_C6_1_Global_7d.csv",
        "viirs_7d": "https://firms.modaps.eosdis.nasa.gov/data/active_fire/suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_Global_7d.csv",
    }

    FEATURE_NAMES = [
        "latitude",
        "longitude",
        "brightness",
        "scan",
        "track",
        "confidence",
        "frp",
    ]

    def __init__(self, config: DatasetConfig) -> None:
        super().__init__(config)
        self.source = config.preprocessing.get("source", "modis_7d")
        self._is_real_data = False

    @property
    def is_real_data(self) -> bool:
        """Return True if real data was loaded."""
        return self._is_real_data

    def download(self) -> bool:
        """Download real fire data from NASA FIRMS.

        Returns:
            True if download successful, False otherwise.
        """
        if self._download_from_firms():
            return True

        logger.warning("NASA FIRMS download failed, falling back to SYNTHETIC data.")
        return self._create_synthetic_wildfire()

    def _download_from_firms(self) -> bool:
        """Download active fire data from NASA FIRMS public CSV."""
        import urllib.request

        if not PANDAS_AVAILABLE:
            logger.warning("pandas required for FIRMS CSV processing")
            return False

        dataset_dir = self.data_path
        dataset_dir.mkdir(parents=True, exist_ok=True)
        cache_file = dataset_dir / "nasa_firms_real.npz"

        if cache_file.exists():
            logger.info(f"NASA FIRMS data already cached at {cache_file}")
            self._is_real_data = True
            return True

        try:
            url = self.FIRMS_URLS.get(self.source, self.FIRMS_URLS["modis_7d"])
            logger.info(f"Downloading fire data from NASA FIRMS ({self.source})...")

            req = urllib.request.Request(  # noqa: S310
                url, headers={"User-Agent": "Mozilla/5.0 Mercury-Agent/1.0"}
            )
            with urllib.request.urlopen(req, timeout=120) as response:  # noqa: S310
                content = response.read().decode("utf-8")

            # Parse CSV
            df = pd.read_csv(io.StringIO(content), low_memory=False)
            logger.info(f"Downloaded {len(df)} fire detection records")

            if len(df) == 0:
                return False

            # Extract features
            features, labels = self._process_firms_data(df)

            # Save to cache
            np.savez_compressed(cache_file, features=features, labels=labels)
            self._is_real_data = True

            logger.info(
                f"NASA FIRMS data loaded: {len(features)} samples, "
                f"{labels.sum()} high-confidence fires (is_real_data=True)"
            )
            return True

        except Exception as e:
            logger.warning(f"NASA FIRMS download failed: {e}")
            return False

    def _process_firms_data(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Process NASA FIRMS CSV data.

        Args:
            df: Raw FIRMS dataframe

        Returns:
            Tuple of (features, labels) numpy arrays
        """
        # Map column names (MODIS vs VIIRS have slightly different names)
        col_map = {
            "latitude": ["latitude", "lat"],
            "longitude": ["longitude", "lon"],
            "brightness": ["brightness", "bright_ti4"],
            "scan": ["scan"],
            "track": ["track"],
            "confidence": ["confidence"],
            "frp": ["frp"],
        }

        rows = []
        for _, row in df.iterrows():
            data_row = []
            for feature in self.FEATURE_NAMES:
                value = 0
                for col_name in col_map.get(feature, [feature]):
                    if col_name in df.columns:
                        val = row.get(col_name)
                        if pd.notna(val):
                            # Handle confidence which may be string (h/n/l) or numeric
                            if feature == "confidence" and isinstance(val, str):
                                val = {"h": 90, "n": 50, "l": 20}.get(val.lower(), 50)
                            value = float(val)
                            break
                data_row.append(value)
            rows.append(data_row)

        features = np.array(rows, dtype=np.float32)
        features = np.nan_to_num(features, nan=0.0)

        # Label high-confidence fires (confidence > 70)
        confidence_col = self.FEATURE_NAMES.index("confidence")
        labels = (features[:, confidence_col] > 70).astype(np.int64)

        # Apply max_samples limit
        if self.config.max_samples and len(features) > self.config.max_samples:
            np.random.seed(self.config.random_seed)
            indices = np.random.choice(len(features), self.config.max_samples, replace=False)
            features = features[indices]
            labels = labels[indices]

        return features, labels

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
        """Load wildfire data from cache (real data first, then synthetic)."""
        # Check for real data first
        real_cache = self.data_path / "nasa_firms_real.npz"
        if real_cache.exists():
            data = np.load(real_cache)
            self._is_real_data = True
            logger.info(f"Loaded REAL NASA FIRMS data from {real_cache}")
            return data["features"], data["labels"]

        # Fall back to synthetic
        synthetic_path = self.data_path / "synthetic_wildfire.npz"
        if synthetic_path.exists():
            data = np.load(synthetic_path)
            self._is_real_data = False
            logger.info("Loaded SYNTHETIC wildfire data (is_real_data=False)")
            return data["features"], data["labels"]

        raise FileNotFoundError("Wildfire data not found. Run download() first.")

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Preprocess wildfire features."""
        data = np.nan_to_num(data, nan=0.0)
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        return data.astype(np.float32)


# Register environmental loaders
DatasetRegistry.register("earthquake", USGSEarthquakeLoader)
DatasetRegistry.register("weather", NOAAWeatherLoader)
DatasetRegistry.register("wildfire", WildfireDataLoader)
