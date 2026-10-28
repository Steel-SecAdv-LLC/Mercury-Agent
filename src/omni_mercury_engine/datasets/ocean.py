# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ocean Dataset Loaders - REAL Marine and Oceanographic Data.

This module provides loaders for real-world ocean and marine datasets
for anomaly detection in oceanographic monitoring:
- NOAA Buoy: Real-time buoy observations (wave height, temperature, wind)
- Argo Floats: Deep ocean temperature/salinity profiles
- Sea Surface Temperature: SST anomalies for climate monitoring

All data sources are free and require no authentication.
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


class NOAABuoyLoader(DatasetLoader):
    """NOAA National Data Buoy Center (NDBC) Real-Time Data Loader.

    Downloads REAL oceanographic data from NOAA buoys including:
    - Wave height (WVHT)
    - Water temperature (WTMP)
    - Air temperature (ATMP)
    - Atmospheric pressure (PRES)
    - Wind speed (WSPD)
    - Wind gust (GST)

    Anomalies include: sensor failures (999.0 values), sudden spikes,
    extreme weather events.

    Data source: https://www.ndbc.noaa.gov/
    License: Public Domain (US Government)

    Note:
        This dataset does not include ground-truth anomaly labels.
        Use for unsupervised deployment only.
    """

    DATASET_NAME = "noaa_buoy"
    LABEL_SOURCE = "statistical"  # no ground-truth labels; manufactured from buoy signal stats
    DATASET_URL = "https://www.ndbc.noaa.gov/"
    LICENSE = "Public Domain (US Government)"
    CITATION = "NOAA National Data Buoy Center. https://www.ndbc.noaa.gov/"
    REQUIRES_CREDENTIALS = False

    # NDBC buoy stations (mix of Pacific and Atlantic)
    BUOY_STATIONS = {
        "46026": "San Francisco, CA",
        "46047": "Tanner Bank, CA",
        "46086": "San Clemente Basin, CA",
        "41047": "Cape Henry, VA",
        "41048": "South Hatteras, NC",
        "44025": "Long Island, NY",
        "46042": "Monterey Bay, CA",
    }

    # Base URL for real-time buoy data (via TrustedEndpoints for SSRF prevention)
    # Uses NOAA_NDBC_REALTIME + /{station}.txt pattern
    BASE_URL = TrustedEndpoints.NOAA_NDBC_REALTIME + "/{station}.txt"

    # Feature columns to extract
    FEATURE_COLS = ["WVHT", "DPD", "APD", "MWD", "WTMP", "ATMP", "PRES", "WSPD", "GST"]

    # Missing value codes used by NOAA NDBC
    # Comprehensive list based on NOAA documentation:
    # https://www.ndbc.noaa.gov/measdes.shtml
    MISSING_VALUES = [
        # Standard NDBC missing codes
        99.0,
        99.00,  # General missing indicator
        999.0,
        999.00,  # Extended missing indicator
        9999.0,
        9999.00,  # Long format missing
        99999.0,  # Very long format
        # Specific sensor missing codes
        -99.9,
        -999.9,  # Negative indicator variants
        -9999.0,  # Negative long format
        # Column-specific codes (WDIR, MWD use 999 for calm/missing)
        0.0,  # Some sensors use 0 for missing wind direction
        # Temperature missing codes (some stations use different scales)
        -99.0,  # Temperature missing
        # String-based (handled separately in processing)
        # "MM", "NA", "N/A" - converted to NaN via pd.to_numeric errors='coerce'
    ]

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize NOAA Buoy loader.

        Args:
            config: Dataset configuration. Preprocessing options:
                - stations (list): Specific station IDs to load
                - anomaly_std (float): Standard deviations for anomaly labeling (default 3)
        """
        super().__init__(config)
        self.stations = config.preprocessing.get("stations", list(self.BUOY_STATIONS.keys())[:5])
        self.anomaly_std = config.preprocessing.get("anomaly_std", 3.0)
        self._features: np.ndarray[Any, Any] | None = None
        self._raw_labels: np.ndarray[Any, Any] | None = None
        self._is_real_data = False

    @property
    def is_real_data(self) -> bool:
        """Return True if real data was loaded."""
        return self._is_real_data

    def download(self) -> bool:
        """Download real-time buoy data from NOAA NDBC.

        Returns:
            True if download successful, False otherwise.
        """
        if not PANDAS_AVAILABLE:
            logger.warning("pandas required for NOAA buoy data processing")
            if ALLOW_SYNTHETIC:
                check_synthetic_allowed("NOAABuoy", "pandas not available for buoy data processing")
                return self._create_synthetic_fallback()
            raise DataSourceUnavailableError(
                loader_name="NOAABuoy",
                source_url="https://www.ndbc.noaa.gov/",
                reason="pandas required for NOAA buoy data processing",
            )

        dataset_dir = self.data_path
        dataset_dir.mkdir(parents=True, exist_ok=True)
        cache_file = dataset_dir / "noaa_buoy_real.npz"

        if cache_file.exists():
            logger.info(f"NOAA buoy data already cached at {cache_file}")
            self._is_real_data = True
            return True

        try:
            all_data = []

            for station in self.stations:
                url = self.BASE_URL.format(station=station)
                try:
                    logger.info(
                        f"Downloading buoy {station} ({self.BUOY_STATIONS.get(station, 'Unknown')})..."
                    )

                    body = http_get_with_retry(url, timeout=60)
                    content = body.decode("utf-8", errors="replace")

                    # Parse the data (space-delimited, first row is header, second is units)
                    df = pd.read_csv(
                        io.StringIO(content),
                        sep=r"\s+",  # Regex whitespace separator (replaces deprecated delim_whitespace)
                        skiprows=[1],  # Skip units row
                    )
                    df["station"] = station
                    all_data.append(df)
                    logger.info(f"  Downloaded {len(df)} records from buoy {station}")

                except Exception as e:
                    logger.warning(f"  Failed to download buoy {station}: {e}")
                    continue

            if not all_data:
                logger.warning("No buoy data downloaded")
                if ALLOW_SYNTHETIC:
                    check_synthetic_allowed("NOAABuoy", "No buoy data downloaded from any station")
                    return self._create_synthetic_fallback()
                raise DataSourceUnavailableError(
                    loader_name="NOAABuoy",
                    source_url="https://www.ndbc.noaa.gov/",
                    reason="No buoy data downloaded from any station",
                )

            combined = pd.concat(all_data, ignore_index=True)
            logger.info(f"Total: {len(combined)} buoy records from {len(all_data)} stations")

            # Process the data
            features, labels = self._process_buoy_data(combined)

            # Save to cache
            np.savez_compressed(cache_file, features=features, labels=labels)
            self._features = features
            self._raw_labels = labels
            self._is_real_data = True

            logger.info(
                f"NOAA Buoy data loaded: {len(features)} samples, "
                f"{labels.sum()} anomalies (is_real_data=True)"
            )
            return True

        except Exception as e:
            logger.warning(f"NOAA Buoy download failed: {e}")
            if ALLOW_SYNTHETIC:
                check_synthetic_allowed("NOAABuoy", f"NOAA Buoy download failed: {e}")
                return self._create_synthetic_fallback()
            raise DataSourceUnavailableError(
                loader_name="NOAABuoy",
                source_url="https://www.ndbc.noaa.gov/",
                reason=f"NOAA Buoy download failed: {e}",
            ) from e

    def _process_buoy_data(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Process buoy data for anomaly detection with comprehensive missing value handling.

        Implements a multi-strategy approach for oceanographic data quality:
        1. Sensor-specific missing value codes identification
        2. Quality flags generation for data provenance
        3. Time-series aware interpolation for short gaps
        4. Physics-based bounds checking
        5. Robust imputation using multiple strategies

        Args:
            df: Raw buoy dataframe

        Returns:
            Tuple of (features, labels) numpy arrays
        """
        # Identify available feature columns
        available_cols = [col for col in self.FEATURE_COLS if col in df.columns]

        if not available_cols:
            raise ValueError(f"No feature columns found. Available: {list(df.columns)}")

        # Extract features
        features_df = df[available_cols].copy()

        # Track original indices for potential debugging
        original_len = len(features_df)

        # ============================================================
        # Phase 1: Replace sensor-specific missing value codes with NaN
        # ============================================================
        for val in self.MISSING_VALUES:
            features_df = features_df.replace(val, np.nan)

        # Also handle common NOAA buoy missing codes: 99, 999, 9999
        for col in features_df.columns:
            features_df[col] = pd.to_numeric(features_df[col], errors="coerce")

        # ============================================================
        # Phase 2: Physics-based bounds checking (flag unrealistic values)
        # ============================================================
        physics_bounds = {
            "WVHT": (0.0, 30.0),  # Wave height in meters (max ~30m for extreme waves)
            "WTMP": (-5.0, 45.0),  # Water temperature in Celsius
            "ATMP": (-60.0, 60.0),  # Air temperature in Celsius
            "PRES": (870.0, 1084.0),  # Pressure in hPa (historical extremes)
            "WSPD": (0.0, 100.0),  # Wind speed in m/s (Cat 5 ~70 m/s)
            "WDIR": (0.0, 360.0),  # Wind direction in degrees
            "DPD": (1.0, 30.0),  # Dominant wave period in seconds
            "MWD": (0.0, 360.0),  # Mean wave direction
            "APD": (1.0, 25.0),  # Average wave period
        }

        for col in features_df.columns:
            if col in physics_bounds:
                low, high = physics_bounds[col]
                mask = (features_df[col] < low) | (features_df[col] > high)
                if mask.any():
                    logger.debug(f"Physics bounds: {mask.sum()} out-of-range values in {col}")
                    features_df.loc[mask, col] = np.nan

        # ============================================================
        # Phase 3: Quality scoring per row (fraction of valid values)
        # ============================================================
        n_cols = len(features_df.columns)
        quality_scores = 1.0 - (features_df.isna().sum(axis=1) / n_cols)

        # Remove rows with very low quality (>80% missing)
        high_quality_mask = quality_scores >= 0.2
        features_df = features_df[high_quality_mask].copy()
        quality_scores = quality_scores[high_quality_mask]

        logger.info(
            f"Quality filtering: {original_len - len(features_df)} rows removed "
            f"({(1 - len(features_df) / original_len):.1%} of data)"
        )

        # ============================================================
        # Phase 4: Time-series aware interpolation for short gaps
        # ============================================================
        # Limit interpolation to gaps of 3 or fewer consecutive NaNs
        max_gap = 3
        for col in features_df.columns:
            # Count consecutive NaNs
            is_nan = features_df[col].isna()
            if is_nan.any():
                # Group consecutive NaNs
                nan_groups = is_nan.ne(is_nan.shift()).cumsum()
                nan_counts = is_nan.groupby(nan_groups).transform("sum")

                # Only interpolate short gaps
                short_gap_mask = (is_nan) & (nan_counts <= max_gap)

                if short_gap_mask.any():
                    # Use linear interpolation for time series continuity
                    interpolated = features_df[col].interpolate(method="linear", limit=max_gap)
                    features_df.loc[short_gap_mask, col] = interpolated.loc[short_gap_mask]

        # ============================================================
        # Phase 5: Multi-strategy imputation for remaining NaNs
        # ============================================================
        # Strategy 1: Seasonal median (for oceanographic patterns)
        # If we have enough data, use rolling window median
        for col in features_df.columns:
            if features_df[col].isna().any():
                # Rolling median with 24-hour window (assuming hourly data)
                window_size = min(24, len(features_df) // 4)
                if window_size >= 3:
                    rolling_median = (
                        features_df[col]
                        .rolling(window=window_size, center=True, min_periods=1)
                        .median()
                    )
                    fill_mask = features_df[col].isna()
                    features_df.loc[fill_mask, col] = rolling_median[fill_mask]

        # Strategy 2: Column median for any remaining NaNs
        for col in features_df.columns:
            if features_df[col].isna().any():
                col_median = features_df[col].median()
                if pd.isna(col_median):
                    # Last resort: use 0 (should rarely happen)
                    col_median = 0.0
                features_df[col] = features_df[col].fillna(col_median)

        # ============================================================
        # Phase 6: Drop rows that still have any NaN (edge cases)
        # ============================================================
        features_df = features_df.dropna(how="any")

        if len(features_df) == 0:
            raise ValueError("All data removed after missing value handling")

        # Convert to numpy
        features = features_df.values.astype(np.float32)

        # ============================================================
        # Phase 7: Advanced anomaly labeling with multiple indicators
        # ============================================================
        # Z-score based anomalies
        z_scores = np.abs(
            (features - np.nanmean(features, axis=0)) / (np.nanstd(features, axis=0) + 1e-8)
        )

        # IQR-based anomalies (more robust to outliers)
        q1 = np.percentile(features, 25, axis=0)
        q3 = np.percentile(features, 75, axis=0)
        iqr = q3 - q1
        iqr_lower = q1 - 1.5 * iqr
        iqr_upper = q3 + 1.5 * iqr
        iqr_anomaly = np.any((features < iqr_lower) | (features > iqr_upper), axis=1)

        # Rate of change anomalies (for time series)
        if len(features) > 1:
            rate_of_change = np.abs(np.diff(features, axis=0, prepend=features[:1]))
            roc_threshold = np.percentile(rate_of_change, 99, axis=0)
            roc_anomaly = np.any(rate_of_change > roc_threshold, axis=1)
        else:
            roc_anomaly = np.zeros(len(features), dtype=bool)

        # Combine anomaly indicators: z-score OR IQR OR rate-of-change
        zscore_anomaly = np.nanmax(z_scores, axis=1) > self.anomaly_std
        labels = (zscore_anomaly | iqr_anomaly | roc_anomaly).astype(np.int64)

        # Log anomaly statistics
        logger.info(
            f"Anomaly detection: {labels.sum()} anomalies found ({labels.mean():.1%} of data). "
            f"Z-score: {zscore_anomaly.sum()}, IQR: {iqr_anomaly.sum()}, RoC: {roc_anomaly.sum()}"
        )

        # Apply max_samples limit if specified
        if self.config.max_samples and len(features) > self.config.max_samples:
            rng = np.random.default_rng(self.config.random_seed)
            indices = rng.choice(len(features), self.config.max_samples, replace=False)
            features = features[indices]
            labels = labels[indices]

        return features, labels

    def _create_synthetic_fallback(self) -> bool:
        """Create synthetic buoy data as fallback."""
        logger.warning(
            "Creating SYNTHETIC NOAA buoy approximation. "
            "Results will NOT reflect real-world oceanographic patterns."
        )

        rng = np.random.default_rng(self.config.random_seed)
        n_samples = self.config.max_samples or 5000
        n_features = len(self.FEATURE_COLS)

        # Generate realistic oceanographic features
        features = np.zeros((n_samples, n_features))

        # WVHT - Wave height (meters), typically 0.5-5m
        features[:, 0] = np.abs(rng.normal(1.5, 0.8, n_samples))

        # DPD - Dominant wave period (seconds), typically 5-15s
        features[:, 1] = rng.normal(8, 2, n_samples)

        # APD - Average wave period
        features[:, 2] = features[:, 1] * rng.uniform(0.7, 0.9, n_samples)

        # MWD - Mean wave direction (degrees)
        features[:, 3] = rng.uniform(0, 360, n_samples)

        # WTMP - Water temperature (°C), typically 10-25°C
        features[:, 4] = rng.normal(18, 4, n_samples)

        # ATMP - Air temperature (°C)
        features[:, 5] = features[:, 4] + rng.normal(0, 2, n_samples)

        # PRES - Atmospheric pressure (hPa), typically 990-1030
        features[:, 6] = rng.normal(1013, 10, n_samples)

        # WSPD - Wind speed (m/s)
        features[:, 7] = np.abs(rng.normal(5, 3, n_samples))

        # GST - Wind gust
        features[:, 8] = features[:, 7] * rng.uniform(1.2, 1.8, n_samples)

        # Inject some anomalies (~5%)
        n_anomalies = int(n_samples * 0.05)
        anomaly_indices = rng.choice(n_samples, n_anomalies, replace=False)

        # Extreme wave heights
        features[anomaly_indices[: n_anomalies // 3], 0] *= 5

        # Temperature spikes
        features[anomaly_indices[n_anomalies // 3 : 2 * n_anomalies // 3], 4] += 10

        # Pressure drops (storms)
        features[anomaly_indices[2 * n_anomalies // 3 :], 6] -= 30

        # Label based on z-scores
        z_scores = np.abs((features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8))
        labels = (z_scores.max(axis=1) > self.anomaly_std).astype(np.int64)

        self._features = features.astype(np.float32)
        self._raw_labels = labels
        self._is_real_data = False

        save_path = self.data_path / "synthetic_noaa_buoy.npz"
        np.savez_compressed(save_path, features=self._features, labels=self._raw_labels)

        logger.info(
            f"Generated SYNTHETIC {n_samples} NOAA buoy samples, "
            f"{self._raw_labels.sum()} anomalies (is_real_data=False)"
        )
        return True

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Load raw buoy data from cached files."""
        real_cache = self.data_path / "noaa_buoy_real.npz"
        if real_cache.exists():
            data = np.load(real_cache, allow_pickle=False)
            self._features = data["features"]
            self._raw_labels = data["labels"]
            self._is_real_data = True
            logger.info(f"Loaded REAL NOAA buoy data from {real_cache}")
            return self._features, self._raw_labels

        synthetic_path = self.data_path / "synthetic_noaa_buoy.npz"
        if synthetic_path.exists() and ALLOW_SYNTHETIC:
            data = np.load(synthetic_path, allow_pickle=False)
            self._features = data["features"]
            self._raw_labels = data["labels"]
            self._is_real_data = False
            logger.info("Loaded SYNTHETIC NOAA buoy data (is_real_data=False)")
            return self._features, self._raw_labels

        raise FileNotFoundError("NOAA buoy data not found. Run download() first.")

    def load_data(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Load NOAA buoy dataset.

        Returns:
            Tuple of (features, labels) numpy arrays
        """
        if self._features is not None and self._raw_labels is not None:
            return self._features, self._raw_labels

        try:
            return self._load_raw()
        except FileNotFoundError:
            self.download()
            return self._load_raw()

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Preprocess oceanographic features."""
        data = np.nan_to_num(data, nan=0.0, posinf=1e10, neginf=0)
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        return data.astype(np.float32)

    def get_metadata(self) -> dict[str, Any]:
        """Get dataset metadata."""
        if self._features is None:
            self.load_data()

        return {
            "name": "NOAA NDBC Buoy",
            "source": "National Data Buoy Center",
            "n_samples": len(self._features) if self._features is not None else 0,
            "n_features": self._features.shape[1] if self._features is not None else 0,
            "feature_names": self.FEATURE_COLS,
            "stations": self.stations,
            "is_real_data": self._is_real_data,
            "url": self.DATASET_URL,
            "citation": self.CITATION,
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about loaded data."""
        if self._features is None:
            self.load_data()

        # Type guards for mypy - load_data() ensures these are not None
        if self._features is None or self._raw_labels is None:
            raise RuntimeError("Failed to load data")

        return {
            "n_samples": len(self._features),
            "n_features": self._features.shape[1],
            "n_anomalies": int(self._raw_labels.sum()),
            "anomaly_ratio": float(self._raw_labels.mean()),
            "feature_means": {
                name: float(self._features[:, i].mean())
                for i, name in enumerate(self.FEATURE_COLS[: self._features.shape[1]])
            },
            "is_real_data": self._is_real_data,
        }


class NOAACoOpsStationLoader(DatasetLoader):
    """NOAA CO-OPS tide-station loader over real per-product observations.

    Canonical port of ``validation.data_loaders.NOAAOceanLoader``, rebuilt on
    the products the CO-OPS API actually serves. The validation-era loader
    fetched one real channel (station water temperature) and generated the
    other seventeen features with a seeded RNG while naming them
    ``salinity_psu``, ``chlorophyll_mg_m3``, ``ph_level`` and so on, under
    ``source="NOAA National Ocean Service"``. This loader fetches each real
    product for one station (water temperature, air temperature, barometric
    pressure, wind, water level), buckets them onto a shared hourly UTC grid,
    and keeps only hours observed by every product — no value is imputed or
    invented.

    Labels mark marine-heatwave-like hours: water temperature at or above
    the per-calendar-month 90th percentile *within the fetched window* (a
    documented in-sample proxy, not a climatological baseline), i.e. a
    threshold on a feature, so ``LABEL_SOURCE`` is ``"statistical"``.

    API reference: https://api.tidesandcurrents.noaa.gov/api/prod/
    (response schemas verified live 2026-10-26: scalar products carry
    ``{"t": ..., "v": ...}`` rows; wind carries ``s``/``d``/``g``;
    water level stays at 6-minute cadence whatever ``interval`` asks).
    """

    DATASET_NAME = "noaa_coops_station"
    LABEL_SOURCE = "statistical"  # labels = SST >= in-window monthly p90 (threshold on a feature)
    DATASET_URL = "https://tidesandcurrents.noaa.gov/"
    LICENSE = "Public Domain (U.S. Government Work)"
    CITATION = """NOAA Center for Operational Oceanographic Products and Services (CO-OPS).
    https://tidesandcurrents.noaa.gov/"""
    REQUIRES_CREDENTIALS = False

    COOPS_API_URL = TrustedEndpoints.NOAA_NOS_API

    #: (product, response value keys) fetched per station. Wind contributes
    #: three channels from one call; the others one channel each.
    #: ``hourly_height`` is the verified hourly water level; the 6-minute
    #: ``water_level`` product refuses ranges beyond 31 days (HTTP 400).
    SCALAR_PRODUCTS = ("water_temperature", "air_temperature", "air_pressure", "hourly_height")

    FEATURE_NAMES = [
        "water_temp_c",
        "air_temp_c",
        "air_pressure_mb",
        "water_level_m",
        "wind_speed_m_s",
        "wind_dir_deg",
        "wind_gust_m_s",
        "hour_of_day",
        "day_of_year",
    ]

    #: In-window percentile defining the heatwave-like label.
    HEATWAVE_PERCENTILE = 90.0

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize the instance."""
        super().__init__(config)
        self.station = str(config.preprocessing.get("station", "8454000"))
        self.days_back = int(config.preprocessing.get("days_back", 60))
        self._is_real_data = False

    @property
    def is_real_data(self) -> bool:
        """Return True if real data was loaded."""
        return self._is_real_data

    def _real_cache_name(self) -> str:
        """Cache name carries the station and window: both shape the matrix."""
        return f"noaa_coops_real_{self.station}_{self.days_back}d.npz"

    def download(self) -> bool:
        """Fetch every product for the station and cache the aligned matrix."""
        if self._download_from_coops():
            return True
        if ALLOW_SYNTHETIC:
            check_synthetic_allowed("NOAACoOpsStation", "CO-OPS API failed")
            return self._create_synthetic()
        raise DataSourceUnavailableError(
            loader_name="NOAACoOpsStation",
            source_url=self.COOPS_API_URL,
            reason="CO-OPS API failed",
        )

    def _fetch_product(self, product: str) -> list[dict[str, Any]]:
        import urllib.parse
        from datetime import UTC, datetime, timedelta

        end = datetime.now(UTC)
        begin = end - timedelta(days=self.days_back)
        params = {
            "begin_date": begin.strftime("%Y%m%d"),
            "end_date": end.strftime("%Y%m%d"),
            "station": self.station,
            "product": product,
            "datum": "MLLW",
            "units": "metric",
            "time_zone": "gmt",
            "interval": "h",
            "application": "Mercury-Agent",
            "format": "json",
        }
        TrustedEndpoints.validate_url(self.COOPS_API_URL)
        url = f"{self.COOPS_API_URL}?{urllib.parse.urlencode(params)}"
        content = http_get_with_retry(url, timeout=120, timeout_per_attempt=30, retries=1)
        payload = json.loads(content.decode("utf-8"))
        rows = payload.get("data")
        if not isinstance(rows, list) or not rows:
            raise DataSourceUnavailableError(
                loader_name="NOAACoOpsStation",
                source_url=self.COOPS_API_URL,
                reason=(
                    f"CO-OPS returned no '{product}' rows for station {self.station}: "
                    f"{str(payload)[:200]}"
                ),
            )
        return rows

    @staticmethod
    def _bucket_scalar(rows: list[dict[str, Any]], key: str) -> dict[int, list[float]]:
        """Hourly-bucketed finite samples of ``key`` keyed on whole UTC hours."""
        from datetime import UTC, datetime

        buckets: dict[int, list[float]] = {}
        for row in rows:
            stamp = row.get("t")
            if not isinstance(stamp, str):
                continue
            try:
                moment = datetime.strptime(stamp, "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
                value = float(row[key])
            except (KeyError, TypeError, ValueError):
                continue
            if not np.isfinite(value):
                continue
            buckets.setdefault(int(moment.timestamp() // 3600), []).append(value)
        return buckets

    def _download_from_coops(self) -> bool:
        cache_file = self.data_path / self._real_cache_name()
        if cache_file.exists():
            logger.info(f"CO-OPS station data already cached at {cache_file}")
            self._is_real_data = True
            return True
        try:
            channels: dict[str, dict[int, list[float]]] = {}
            for product in self.SCALAR_PRODUCTS:
                channels[product] = self._bucket_scalar(self._fetch_product(product), "v")
            wind_rows = self._fetch_product("wind")
            channels["wind_speed"] = self._bucket_scalar(wind_rows, "s")
            channels["wind_dir"] = self._bucket_scalar(wind_rows, "d")
            channels["wind_gust"] = self._bucket_scalar(wind_rows, "g")

            shared = sorted(set.intersection(*(set(b) for b in channels.values())))
            if not shared:
                logger.warning("CO-OPS products fetched but no hour was observed by every product")
                return False
            from datetime import UTC, datetime

            order = (
                "water_temperature",
                "air_temperature",
                "air_pressure",
                "hourly_height",
                "wind_speed",
                "wind_dir",
                "wind_gust",
            )
            rows: list[list[float]] = []
            for bucket in shared:
                moment = datetime.fromtimestamp(bucket * 3600, tz=UTC)
                rows.append(
                    [float(np.mean(channels[name][bucket])) for name in order]
                    + [float(moment.hour), float(moment.timetuple().tm_yday)]
                )
            features = np.asarray(rows, dtype=np.float64)
            labels = self._heatwave_labels(features, shared)
            np.savez_compressed(cache_file, features=features, labels=labels)
            self._is_real_data = True
            logger.info(
                f"CO-OPS station {self.station} loaded: {len(features)} fully-observed "
                f"hours, {int(labels.sum())} heatwave-like (is_real_data=True)"
            )
            return True
        except Exception as e:
            logger.warning(f"CO-OPS download failed: {e}")
            return False

    def _heatwave_labels(
        self, features: np.ndarray[Any, Any], buckets: list[int]
    ) -> np.ndarray[Any, Any]:
        """Water temperature at/above its in-window monthly p90."""
        from datetime import UTC, datetime

        months = np.asarray(
            [datetime.fromtimestamp(b * 3600, tz=UTC).month for b in buckets], dtype=np.int64
        )
        water_temp = features[:, 0]
        labels = np.zeros(len(features), dtype=np.int64)
        for month in np.unique(months):
            mask = months == month
            cut = float(np.percentile(water_temp[mask], self.HEATWAVE_PERCENTILE))
            labels[mask] = (water_temp[mask] >= cut).astype(np.int64)
        return labels

    def _create_synthetic(self) -> bool:
        """Create a labelled synthetic station matrix (policy-gated callers only)."""
        rng = np.random.default_rng(self.config.random_seed)
        n = self.config.max_samples or 2000
        day = rng.integers(1, 366, size=n).astype(np.float64)
        seasonal = 12 + 8 * np.sin(2 * np.pi * (day - 120) / 365.0)
        water_temp = seasonal + rng.normal(0, 1.2, size=n)
        air_temp = water_temp + rng.normal(0, 3.0, size=n)
        pressure = rng.normal(1015, 8, size=n)
        water_level = rng.normal(0.6, 0.5, size=n)
        wind_speed = np.clip(rng.gamma(2.0, 2.0, size=n), 0, 40)
        wind_dir = rng.uniform(0, 360, size=n)
        wind_gust = wind_speed * rng.uniform(1.1, 1.8, size=n)
        hour = rng.integers(0, 24, size=n).astype(np.float64)
        features = np.column_stack(
            [
                water_temp,
                air_temp,
                pressure,
                water_level,
                wind_speed,
                wind_dir,
                wind_gust,
                hour,
                day,
            ]
        ).astype(np.float64)
        cut = float(np.percentile(water_temp, self.HEATWAVE_PERCENTILE))
        labels = (water_temp >= cut).astype(np.int64)
        cache_file = self.data_path / "noaa_coops_synthetic.npz"  # station-independent
        np.savez_compressed(cache_file, features=features, labels=labels)
        self._is_real_data = False
        return True

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Load the cached matrix (real preferred over synthetic)."""
        for name, is_real in (
            (self._real_cache_name(), True),
            ("noaa_coops_synthetic.npz", False),
        ):
            cache_file = self.data_path / name
            if cache_file.exists():
                data = np.load(cache_file)
                self._is_real_data = is_real
                return data["features"], data["labels"]
        raise FileNotFoundError(
            f"No cached CO-OPS data under {self.data_path}; call download() first."
        )

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """No-op: rows are already aligned, fully observed and finite."""
        return data


# Register ocean loaders
DatasetRegistry.register("noaa_buoy", NOAABuoyLoader)
DatasetRegistry.register("ocean-buoy", NOAABuoyLoader)
DatasetRegistry.register("noaa_coops_station", NOAACoOpsStationLoader)
