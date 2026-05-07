"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

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

from omni_mercury_engine.security.input_validation import TrustedEndpoints

from .base import DatasetConfig, DatasetLoader, DatasetRegistry, http_get_with_retry
from .exceptions import ALLOW_SYNTHETIC, DataSourceUnavailableError, check_synthetic_allowed

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

    Note:
        This dataset does not include ground-truth anomaly labels.
        Use for unsupervised deployment only.
    """

    DATASET_NAME = "earthquake"
    DATASET_URL = "https://earthquake.usgs.gov/earthquakes/search/"
    LICENSE = "Public Domain (USGS)"
    CITATION = """U.S. Geological Survey (USGS) Earthquake Hazards Program.
    National Earthquake Information Center (NEIC)."""
    REQUIRES_CREDENTIALS = False

    # USGS API endpoint for GeoJSON earthquake data (via TrustedEndpoints for SSRF prevention)
    USGS_API_URL = TrustedEndpoints.USGS_EARTHQUAKE

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
        """
        Download real earthquake data from USGS API.

        Returns:
            True if download successful, False otherwise.
        """
        if self._download_from_usgs():
            return True

        if ALLOW_SYNTHETIC:
            check_synthetic_allowed("USGSEarthquake", "USGS API failed")
            return self._create_synthetic_earthquake()
        raise DataSourceUnavailableError(
            loader_name="USGSEarthquake",
            source_url=self.USGS_API_URL,
            reason="USGS API failed",
        )

    def _download_from_usgs(self) -> bool:
        """Download earthquake data from USGS Earthquake Hazards API."""
        import urllib.parse
        from datetime import UTC, datetime, timedelta

        dataset_dir = self.data_path
        dataset_dir.mkdir(parents=True, exist_ok=True)
        cache_file = dataset_dir / "usgs_earthquake_real.npz"

        if cache_file.exists():
            logger.info(f"USGS earthquake data already cached at {cache_file}")
            self._is_real_data = True
            return True

        try:
            # USGS fdsnws expects UTC; using naive datetime.now() can drop or
            # double-count records around UTC day boundaries. The API's hard
            # limit per request is 20000.
            end_date = datetime.now(UTC)
            start_date = end_date - timedelta(days=self.days_back)

            params = {
                "format": "geojson",
                "starttime": start_date.strftime("%Y-%m-%d"),
                "endtime": end_date.strftime("%Y-%m-%d"),
                "minmagnitude": str(self.min_magnitude),
                "limit": str(min(self.config.max_samples or 20000, 20000)),
                "orderby": "time",
            }

            url = f"{self.USGS_API_URL}?{urllib.parse.urlencode(params)}"
            TrustedEndpoints.validate_url(self.USGS_API_URL)
            logger.info(
                f"Downloading earthquake data from USGS API (last {self.days_back} days)..."
            )
            content = http_get_with_retry(url, timeout=120)
            data = json.loads(content.decode("utf-8"))

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
        """
        Process USGS GeoJSON earthquake data.

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
            rng = np.random.default_rng(self.config.random_seed)
            indices = rng.choice(len(features), self.config.max_samples, replace=False)
            features = features[indices]
            labels = labels[indices]

        return features, labels

    def _create_synthetic_earthquake(self) -> bool:
        """Create synthetic earthquake catalog."""
        rng = np.random.default_rng(self.config.random_seed)
        n_samples = self.config.max_samples or 10000

        # Simulate global earthquake distribution
        features = []
        labels = []

        for _i in range(n_samples):
            # Spatial distribution (clustered around fault zones)
            zone = rng.choice(["pacific_rim", "mediterranean", "himalayan", "mid_atlantic"])

            if zone == "pacific_rim":
                lat = rng.normal(35, 20)
                lon = rng.normal(140, 30)
            elif zone == "mediterranean":
                lat = rng.normal(38, 5)
                lon = rng.normal(20, 15)
            elif zone == "himalayan":
                lat = rng.normal(30, 5)
                lon = rng.normal(85, 10)
            else:
                lat = rng.normal(0, 30)
                lon = rng.normal(-30, 10)

            # Gutenberg-Richter magnitude distribution
            magnitude = rng.exponential(1.0) + self.min_magnitude

            params = {
                "latitude": np.clip(lat, -90, 90),
                "longitude": np.clip(lon, -180, 180),
                "depth": rng.exponential(30),  # km
                "magnitude": magnitude,
                "gap": rng.uniform(20, 300),  # azimuthal gap
                "dmin": rng.exponential(0.5),  # distance to nearest station
                "rms": rng.exponential(0.3),  # residual
                "nst": rng.poisson(20),  # number of stations
                "horizontal_error": rng.exponential(2),
                "depth_error": rng.exponential(5),
                "mag_error": rng.exponential(0.2),
                # Precursor features (simulated)
                "previous_mag_7d": rng.exponential(1) + 2,
                "previous_count_7d": rng.poisson(5),
                "previous_energy_7d": rng.exponential(1e10),
                "b_value_local": rng.normal(1.0, 0.2),
                "time_since_last": rng.exponential(24),  # hours
                "distance_to_fault": rng.exponential(10),  # km
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

        features = np.array(features, dtype=np.float32)  # type: ignore[assignment, unused-ignore]
        labels = np.array(labels, dtype=np.int64)  # type: ignore[assignment, unused-ignore]

        save_path = self.data_path / "synthetic_earthquake.npz"
        np.savez_compressed(save_path, features=features, labels=labels)

        logger.info(f"Generated {n_samples} earthquake samples, {labels.sum()} significant events")  # type: ignore[attr-defined, unused-ignore]
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

        # Fall back to synthetic (only if allowed)
        synthetic_path = self.data_path / "synthetic_earthquake.npz"
        if synthetic_path.exists() and ALLOW_SYNTHETIC:
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

    Note:
        This dataset does not include ground-truth anomaly labels.
        Use for unsupervised deployment only.
    """

    DATASET_NAME = "weather"
    DATASET_URL = "https://open-meteo.com/"
    LICENSE = "CC BY 4.0 (Open-Meteo)"
    CITATION = """Open-Meteo Free Weather API. https://open-meteo.com/"""
    REQUIRES_CREDENTIALS = False

    # Open-Meteo API endpoint (via TrustedEndpoints for SSRF prevention)
    OPEN_METEO_URL = TrustedEndpoints.OPEN_METEO_ARCHIVE

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
        """
        Download real weather data from Open-Meteo API.

        Returns:
            True if download successful, False otherwise.
        """
        if self._download_from_open_meteo():
            return True

        if ALLOW_SYNTHETIC:
            check_synthetic_allowed("NOAAWeather", "Open-Meteo API failed")
            return self._create_synthetic_weather()
        raise DataSourceUnavailableError(
            loader_name="NOAAWeather",
            source_url=self.OPEN_METEO_URL,
            reason="Open-Meteo API failed",
        )

    def _download_from_open_meteo(self) -> bool:
        """Download weather data from Open-Meteo Archive API."""
        import urllib.parse
        from datetime import UTC, datetime, timedelta

        dataset_dir = self.data_path
        dataset_dir.mkdir(parents=True, exist_ok=True)
        cache_file = dataset_dir / "open_meteo_weather_real.npz"

        if cache_file.exists():
            logger.info(f"Weather data already cached at {cache_file}")
            self._is_real_data = True
            return True

        try:
            # Open-Meteo Archive lags realtime by ~5 days; use UTC for stable
            # day boundaries.
            end_date = datetime.now(UTC) - timedelta(days=5)
            start_date = end_date - timedelta(days=self.days_back)

            all_features = []

            TrustedEndpoints.validate_url(self.OPEN_METEO_URL)
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
                url = f"{self.OPEN_METEO_URL}?{urllib.parse.urlencode(params)}"

                logger.info(f"Downloading weather data for {loc['name']}...")
                try:
                    content = http_get_with_retry(url, timeout=60)
                except Exception as e:  # noqa: BLE001 - per-location tolerant
                    logger.warning(
                        "Open-Meteo location %s failed: %s", loc["name"], e
                    )
                    continue
                data = json.loads(content.decode("utf-8"))

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
                rng = np.random.default_rng(self.config.random_seed)
                indices = rng.choice(len(features), self.config.max_samples, replace=False)
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
        """
        Create synthetic weather observation data.

        Generates data matching FEATURE_NAMES:
        temperature, humidity, pressure, wind_speed, wind_direction,
        precipitation, cloud_cover, apparent_temperature
        """
        rng = np.random.default_rng(self.config.random_seed)
        n_samples = self.config.max_samples or 10000

        # Seasonal variation
        t = np.linspace(0, 4 * np.pi, n_samples)
        seasonal = np.sin(t)

        # Generate features matching FEATURE_NAMES exactly
        data = {
            "temperature": 15 + 15 * seasonal + rng.normal(0, 5, n_samples),
            "humidity": np.clip(60 + 20 * rng.standard_normal(n_samples), 0, 100),
            "pressure": 1013 + rng.normal(0, 15, n_samples),
            "wind_speed": rng.exponential(5, n_samples),
            "wind_direction": rng.uniform(0, 360, n_samples),
            "precipitation": rng.exponential(2, n_samples),
            "cloud_cover": np.clip(rng.normal(50, 30, n_samples), 0, 100),
            "apparent_temperature": 15 + 15 * seasonal + rng.normal(0, 6, n_samples),
        }

        features = np.column_stack([data[f] for f in self.FEATURE_NAMES])

        # Extreme weather anomalies (~10% of samples)
        labels = np.zeros(n_samples, dtype=np.int64)
        extreme_mask = (
            (data["temperature"] > 40)
            | (data["temperature"] < -10)  # Extreme temp
            | (data["wind_speed"] > 20)  # Severe wind
            | (data["precipitation"] > 15)  # Heavy rain
            | (data["pressure"] < 980)
            | (data["pressure"] > 1040)  # Extreme pressure
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

        # Fall back to synthetic (only if allowed)
        synthetic_path = self.data_path / "synthetic_weather.npz"
        if synthetic_path.exists() and ALLOW_SYNTHETIC:
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

    Note:
        This dataset does not include ground-truth anomaly labels.
        Use for unsupervised deployment only.
    """

    DATASET_NAME = "wildfire"
    DATASET_URL = "https://firms.modaps.eosdis.nasa.gov/"
    LICENSE = "Public Domain (NASA)"
    CITATION = """NASA FIRMS (Fire Information for Resource Management System).
    MODIS Collection 6.1 and VIIRS Active Fire Products."""
    REQUIRES_CREDENTIALS = False

    # NASA FIRMS public CSV data URLs (via TrustedEndpoints for SSRF prevention)
    FIRMS_URLS = {
        "modis_7d": TrustedEndpoints.NASA_FIRMS_MODIS_7D,
        "viirs_7d": TrustedEndpoints.NASA_FIRMS_VIIRS_SUOMI_7D,
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
        """
        Download real fire data from NASA FIRMS.

        Returns:
            True if download successful, False otherwise.
        """
        if self._download_from_firms():
            return True

        if ALLOW_SYNTHETIC:
            check_synthetic_allowed("Wildfire", "NASA FIRMS download failed")
            return self._create_synthetic_wildfire()
        raise DataSourceUnavailableError(
            loader_name="Wildfire",
            source_url="https://firms.modaps.eosdis.nasa.gov/",
            reason="NASA FIRMS download failed",
        )

    def _download_from_firms(self) -> bool:
        """Download active fire data from NASA FIRMS public CSV."""
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
            # Try the requested source first, then fall through to the other
            # public 7-day archives. FIRMS occasionally rotates which sensor
            # archive is current; cross-mirroring keeps the loader live even
            # when one CSV path returns 404 mid-rotation.
            preferred = self.FIRMS_URLS.get(self.source, self.FIRMS_URLS["modis_7d"])
            ordered_urls: list[str] = [preferred]
            for alt in self.FIRMS_URLS.values():
                if alt not in ordered_urls:
                    ordered_urls.append(alt)

            content_text: str | None = None
            last_err: Exception | None = None
            for url in ordered_urls:
                try:
                    TrustedEndpoints.validate_url(url)
                    logger.info("Downloading fire data from NASA FIRMS (%s)...", url)
                    body = http_get_with_retry(url, timeout=120)
                    content_text = body.decode("utf-8", errors="replace")
                    break
                except Exception as e:  # noqa: BLE001 - mirror failover
                    last_err = e
                    logger.info("FIRMS source %s failed: %s", url, e)
                    continue

            if content_text is None:
                logger.warning("All NASA FIRMS sources failed: %s", last_err)
                return False

            df = pd.read_csv(io.StringIO(content_text), low_memory=False)
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
        """
        Process NASA FIRMS CSV data.

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
            data_row: list[float] = []
            for feature in self.FEATURE_NAMES:
                value: float = 0.0
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
            rng = np.random.default_rng(self.config.random_seed)
            indices = rng.choice(len(features), self.config.max_samples, replace=False)
            features = features[indices]
            labels = labels[indices]

        return features, labels

    def _create_synthetic_wildfire(self) -> bool:
        """Create synthetic wildfire detection data."""
        rng = np.random.default_rng(self.config.random_seed)
        n_samples = self.config.max_samples or 5000

        features = []
        labels = []

        for _i in range(n_samples):
            # Fire-prone regions
            region = rng.choice(["western_us", "australia", "amazon", "mediterranean"])

            if region == "western_us":
                lat = rng.normal(38, 5)
                lon = rng.normal(-120, 5)
            elif region == "australia":
                lat = rng.normal(-33, 5)
                lon = rng.normal(148, 5)
            elif region == "amazon":
                lat = rng.normal(-5, 5)
                lon = rng.normal(-60, 10)
            else:
                lat = rng.normal(40, 3)
                lon = rng.normal(15, 10)

            # Fire conditions
            is_fire = rng.random() < 0.3

            if is_fire:
                params = {
                    "latitude": lat,
                    "longitude": lon,
                    "brightness": rng.uniform(320, 400),  # Kelvin
                    "brightness_t31": rng.uniform(290, 350),
                    "frp": rng.exponential(50),  # MW
                    "confidence": rng.uniform(50, 100),
                    "scan": rng.uniform(1, 4),
                    "track": rng.uniform(1, 4),
                    "temperature": rng.normal(35, 8),  # Hot
                    "humidity": rng.uniform(10, 40),  # Dry
                    "wind_speed": rng.exponential(8),
                    "wind_direction": rng.uniform(0, 360),
                    "precipitation_7d": rng.exponential(1),  # Low
                    "fuel_moisture": rng.uniform(5, 20),  # Low
                    "ndvi": rng.uniform(0.2, 0.6),
                    "slope": rng.exponential(10),
                    "aspect": rng.uniform(0, 360),
                    "elevation": rng.exponential(500),
                }
                labels.append(1)
            else:
                params = {
                    "latitude": lat,
                    "longitude": lon,
                    "brightness": rng.uniform(280, 310),
                    "brightness_t31": rng.uniform(280, 300),
                    "frp": rng.exponential(2),
                    "confidence": rng.uniform(0, 30),
                    "scan": rng.uniform(1, 2),
                    "track": rng.uniform(1, 2),
                    "temperature": rng.normal(20, 10),
                    "humidity": rng.uniform(40, 90),
                    "wind_speed": rng.exponential(4),
                    "wind_direction": rng.uniform(0, 360),
                    "precipitation_7d": rng.exponential(10),
                    "fuel_moisture": rng.uniform(20, 50),
                    "ndvi": rng.uniform(0.3, 0.9),
                    "slope": rng.exponential(5),
                    "aspect": rng.uniform(0, 360),
                    "elevation": rng.exponential(300),
                }
                labels.append(0)

            feature_vec = [params[f] for f in self.FEATURE_NAMES]
            features.append(feature_vec)

        features = np.array(features, dtype=np.float32)  # type: ignore[assignment, unused-ignore]
        labels = np.array(labels, dtype=np.int64)  # type: ignore[assignment, unused-ignore]

        save_path = self.data_path / "synthetic_wildfire.npz"
        np.savez_compressed(save_path, features=features, labels=labels)

        logger.info(f"Generated {n_samples} wildfire samples, {labels.sum()} fire detections")  # type: ignore[attr-defined, unused-ignore]
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

        # Fall back to synthetic (only if allowed)
        synthetic_path = self.data_path / "synthetic_wildfire.npz"
        if synthetic_path.exists() and ALLOW_SYNTHETIC:
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


class USGSGeochemistryLoader(DatasetLoader):
    """
    USGS Geochemistry Data Loader for Environmental Contamination Detection.

    Downloads REAL soil/sediment geochemistry data from USGS MRData including:
    - Heavy metal concentrations (As, Pb, Hg, Cu, Zn)
    - pH, conductivity, and other water quality indicators
    - Contamination site data for anomaly detection

    Data source: https://mrdata.usgs.gov/geochem/
    License: Public Domain (USGS)
    Citation: USGS Mineral Resources Data System (MRDS).

    Note:
        This dataset does not include ground-truth anomaly labels.
        Use for unsupervised deployment only.
    """

    DATASET_NAME = "geochemistry"
    DATASET_URL = "https://mrdata.usgs.gov/geochem/"
    LICENSE = "Public Domain (USGS)"
    CITATION = """U.S. Geological Survey (USGS). Mineral Resources Data System.
    National Geochemical Survey Database."""
    REQUIRES_CREDENTIALS = False

    # EPA Regional Screening Levels for soil contamination (mg/kg)
    EPA_SCREENING_LEVELS = {
        "arsenic": 0.68,  # Carcinogenic
        "lead": 400,
        "mercury": 11,
        "cadmium": 70,
        "copper": 3100,
        "zinc": 23000,
    }

    FEATURE_NAMES = [
        "latitude",
        "longitude",
        "arsenic",
        "lead",
        "mercury",
        "cadmium",
        "copper",
        "zinc",
        "iron",
        "calcium",
        "ph",
    ]

    def __init__(self, config: DatasetConfig) -> None:
        """
        Initialize USGS Geochemistry loader.

        Args:
            config: Dataset configuration. Preprocessing options:
                - region (dict): Geographic bounds {lat_min, lat_max, lon_min, lon_max}
                - contaminant_focus (list): List of metals to focus on
        """
        super().__init__(config)
        self.region = config.preprocessing.get(
            "region",
            {
                "lat_min": 24,
                "lat_max": 50,
                "lon_min": -125,
                "lon_max": -66,
            },
        )
        self.contaminant_focus = config.preprocessing.get(
            "contaminant_focus", ["arsenic", "lead", "mercury"]
        )
        self._is_real_data = False

    @property
    def is_real_data(self) -> bool:
        """Return True if real data was loaded."""
        return self._is_real_data

    def download(self) -> bool:
        """
        Download geochemistry data from USGS MRData or generate synthetic.

        Returns:
            True if download successful, False otherwise.
        """
        if self._download_from_usgs():
            return True

        if ALLOW_SYNTHETIC:
            check_synthetic_allowed(
                "USGSGeochemistry",
                "USGS MRData API requires bulk download",
            )
            return self._create_synthetic_geochemistry()
        raise DataSourceUnavailableError(
            loader_name="USGSGeochemistry",
            source_url="https://mrdata.usgs.gov/geochem/",
            reason="USGS MRData API requires bulk download",
        )

    def _download_from_usgs(self) -> bool:
        """Attempt to download USGS geochemistry data."""
        # Note: USGS MRData requires complex WFS/WMS queries
        # For production, integrate with their downloadable datasets
        # For now, use synthetic data with realistic distributions
        return False

    def _create_synthetic_geochemistry(self) -> bool:
        """Create synthetic geochemistry data based on realistic distributions."""
        rng = np.random.default_rng(self.config.random_seed)
        n_samples = self.config.max_samples or 5000

        features = []
        labels = []

        for _ in range(n_samples):
            # Random location within continental US
            lat = rng.uniform(self.region["lat_min"], self.region["lat_max"])
            lon = rng.uniform(self.region["lon_min"], self.region["lon_max"])

            # Heavy metals - lognormal distributions (mg/kg in soil)
            # Background levels with occasional contamination hotspots
            is_contaminated = rng.random() < 0.15  # 15% contamination rate

            if is_contaminated:
                # Elevated levels at contamination sites
                arsenic = np.random.lognormal(2.0, 1.0)  # Higher mean
                lead = np.random.lognormal(5.0, 1.5)
                mercury = np.random.lognormal(1.0, 1.2)
                cadmium = np.random.lognormal(1.5, 1.0)
                copper = np.random.lognormal(4.0, 1.0)
                zinc = np.random.lognormal(5.0, 1.0)
            else:
                # Background levels
                arsenic = np.random.lognormal(0.5, 0.8)
                lead = np.random.lognormal(2.5, 0.8)
                mercury = np.random.lognormal(-1.0, 0.8)
                cadmium = np.random.lognormal(0.0, 0.6)
                copper = np.random.lognormal(2.5, 0.6)
                zinc = np.random.lognormal(3.5, 0.6)

            # Major elements (typically stable)
            iron = np.random.lognormal(10.0, 0.3)  # % Fe2O3
            calcium = np.random.lognormal(8.0, 0.5)  # % CaO

            # pH (soil typically 4-9)
            ph = np.clip(rng.normal(6.5, 1.0), 4.0, 9.0)

            feature_vec = [
                lat,
                lon,
                arsenic,
                lead,
                mercury,
                cadmium,
                copper,
                zinc,
                iron,
                calcium,
                ph,
            ]
            features.append(feature_vec)

            # Anomaly: exceeds EPA screening levels
            is_anomaly = (
                arsenic > self.EPA_SCREENING_LEVELS["arsenic"]
                or lead > self.EPA_SCREENING_LEVELS["lead"]
                or mercury > self.EPA_SCREENING_LEVELS["mercury"]
                or cadmium > self.EPA_SCREENING_LEVELS["cadmium"]
            )
            labels.append(1 if is_anomaly else 0)

        features = np.array(features, dtype=np.float32)  # type: ignore[assignment, unused-ignore]
        labels = np.array(labels, dtype=np.int64)  # type: ignore[assignment, unused-ignore]

        save_path = self.data_path / "synthetic_geochemistry.npz"
        np.savez_compressed(save_path, features=features, labels=labels)

        logger.info(
            f"Generated {n_samples} synthetic geochemistry samples, "
            f"{labels.sum()} contamination anomalies (is_real_data=False)"  # type: ignore[attr-defined, unused-ignore]
        )
        return True

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Load geochemistry data from cache."""
        real_cache = self.data_path / "usgs_geochemistry_real.npz"
        if real_cache.exists():
            data = np.load(real_cache)
            self._is_real_data = True
            logger.info(f"Loaded REAL USGS geochemistry data from {real_cache}")
            return data["features"], data["labels"]

        synthetic_path = self.data_path / "synthetic_geochemistry.npz"
        if synthetic_path.exists() and ALLOW_SYNTHETIC:
            data = np.load(synthetic_path)
            self._is_real_data = False
            logger.info("Loaded SYNTHETIC geochemistry data (is_real_data=False)")
            return data["features"], data["labels"]

        raise FileNotFoundError("Geochemistry data not found. Run download() first.")

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Preprocess geochemistry data with log transform for metals."""
        # Log transform metal concentrations (columns 2-9)
        data_processed = data.copy()
        data_processed[:, 2:10] = np.log1p(data_processed[:, 2:10])

        # Z-score normalization
        data_processed = (data_processed - data_processed.mean(axis=0)) / (
            data_processed.std(axis=0) + 1e-8
        )
        return np.asarray(data_processed.astype(np.float32))  # type: ignore[no-any-return, unused-ignore]


# Register environmental loaders
DatasetRegistry.register("earthquake", USGSEarthquakeLoader)
DatasetRegistry.register("weather", NOAAWeatherLoader)
DatasetRegistry.register("wildfire", WildfireDataLoader)
DatasetRegistry.register("geochemistry", USGSGeochemistryLoader)
DatasetRegistry.register("usgs_geochemistry", USGSGeochemistryLoader)  # Alias
