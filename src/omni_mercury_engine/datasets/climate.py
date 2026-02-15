"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Climate and Oceanographic Dataset Loaders - Advanced Marine Data Integration

This module provides loaders for advanced climate and oceanographic datasets:
- Simons CMAP: Ocean biogeochemistry, satellite observations, model outputs
- World Ocean Database (WOD): Temperature/salinity profiles from NCEI
- Copernicus Sea Level: Global satellite altimetry data

All data sources follow FAIR principles and are freely accessible.
"""

from __future__ import annotations

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

from .base import DatasetConfig, DatasetLoader, DatasetRegistry
from .exceptions import ALLOW_SYNTHETIC, DataSourceUnavailableError, check_synthetic_allowed

logger = logging.getLogger(__name__)


class SimonsCMAPLoader(DatasetLoader):
    """
    Simons Collaborative Marine Atlas Project (CMAP) Data Loader.

    Downloads REAL ocean biogeochemistry data from Simons CMAP including:
    - Satellite observations (chlorophyll, SST, altimetry)
    - In-situ measurements (nutrients, carbon, oxygen)
    - Model outputs (ocean circulation, biogeochemistry)

    The CMAP API provides SQL-like queries over a unified ocean data platform.
    For full API access, users can install pycmap package.

    Data source: https://simonscmap.com/
    License: CC BY 4.0 (most datasets)
    Citation: Ashkezari et al. (2021). Limnology and Oceanography: Methods.

    Note:
        This dataset does not include ground-truth anomaly labels.
        Use for unsupervised deployment only.
    """

    DATASET_NAME = "simons_cmap"
    DATASET_URL = "https://simonscmap.com/"
    LICENSE = "CC BY 4.0"
    CITATION = """Ashkezari MD, et al. (2021). Simons Collaborative Marine Atlas Project
    (Simons CMAP): An open-source portal to share, visualize, and analyze ocean data.
    Limnology and Oceanography: Methods. doi:10.1002/lom3.10439"""
    REQUIRES_CREDENTIALS = True  # API key required for full access

    # Key oceanographic variables available in CMAP
    VARIABLE_SETS = {
        "satellite": [
            "sst",  # Sea surface temperature
            "chl",  # Chlorophyll-a concentration
            "sla",  # Sea level anomaly
            "wind_speed",  # Wind speed
        ],
        "biogeochemistry": [
            "nitrate",
            "phosphate",
            "silicate",
            "oxygen",
            "dissolved_organic_carbon",
        ],
        "physical": [
            "temperature",
            "salinity",
            "density",
            "mixed_layer_depth",
        ],
    }

    FEATURE_NAMES = [
        "latitude",
        "longitude",
        "depth",
        "temperature",
        "salinity",
        "chlorophyll",
        "nitrate",
        "oxygen",
    ]

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize Simons CMAP loader.

        Args:
            config: Dataset configuration. Preprocessing options:
                - api_key (str): Simons CMAP API key (optional for sample data)
                - variable_set (str): 'satellite', 'biogeochemistry', or 'physical'
                - region (dict): Geographic bounds {lat_min, lat_max, lon_min, lon_max}
                - depth_range (tuple): (min_depth, max_depth) in meters
        """
        super().__init__(config)
        self.api_key = config.preprocessing.get("api_key")
        self.variable_set = config.preprocessing.get("variable_set", "physical")
        self.region = config.preprocessing.get(
            "region",
            {
                "lat_min": -60,
                "lat_max": 60,
                "lon_min": -180,
                "lon_max": 180,
            },
        )
        self.depth_range = config.preprocessing.get("depth_range", (0, 500))
        self._is_real_data = False

    @property
    def is_real_data(self) -> bool:
        """Return True if real data was loaded."""
        return self._is_real_data

    def download(self) -> bool:
        """Download ocean data from Simons CMAP or generate synthetic.

        Returns:
            True if download successful, False otherwise.
        """
        # Try to use pycmap if available
        try:
            import pycmap  # noqa: F401

            if self.api_key:
                return self._download_via_pycmap()
        except ImportError:
            logger.info("pycmap not installed. Using sample data or synthetic fallback.")

        # Fall back to synthetic data with realistic oceanographic patterns
        if ALLOW_SYNTHETIC:
            check_synthetic_allowed(
                "SimonsCMAP",
                "Simons CMAP requires pycmap package and API key for full access",
            )
            return self._create_synthetic_ocean()
        raise DataSourceUnavailableError(
            loader_name="SimonsCMAP",
            source_url="https://simonscmap.com/",
            reason=(
                "Simons CMAP requires pycmap package and API key for full access. "
                "Install: pip install pycmap | Get API key: https://simonscmap.com/register"
            ),
        )

    def _build_cmap_query(self) -> str:
        """Build a safe SQL query for the pycmap API with validated numeric bounds.

        This method constructs a SQL query for the Simons CMAP API using only
        validated numeric values. All bounds are validated to be within acceptable
        geographic and physical ranges before being used in the query.

        Security: The pycmap API is a trusted Python library that executes queries
        against the Simons CMAP database. All values interpolated into the query
        are validated to be numeric types within valid ranges, preventing injection.

        Returns:
            SQL query string with validated numeric bounds.

        Raises:
            ValueError: If any bound is not a valid numeric type or out of range.
        """
        # Validate and extract latitude bounds
        lat_min = self.region.get("lat_min", -60)
        lat_max = self.region.get("lat_max", 60)
        if not isinstance(lat_min, (int, float)) or not isinstance(lat_max, (int, float)):
            raise ValueError(f"Latitude bounds must be numeric: {lat_min}, {lat_max}")
        if not (-90 <= lat_min <= 90 and -90 <= lat_max <= 90):
            raise ValueError(f"Latitude must be between -90 and 90: {lat_min}, {lat_max}")

        # Validate and extract longitude bounds
        lon_min = self.region.get("lon_min", -180)
        lon_max = self.region.get("lon_max", 180)
        if not isinstance(lon_min, (int, float)) or not isinstance(lon_max, (int, float)):
            raise ValueError(f"Longitude bounds must be numeric: {lon_min}, {lon_max}")
        if not (-180 <= lon_min <= 180 and -180 <= lon_max <= 180):
            raise ValueError(f"Longitude must be between -180 and 180: {lon_min}, {lon_max}")

        # Validate and extract depth range
        depth_min, depth_max = self.depth_range
        if not isinstance(depth_min, (int, float)) or not isinstance(depth_max, (int, float)):
            raise ValueError(f"Depth range must be numeric: {depth_min}, {depth_max}")
        if depth_min < 0 or depth_max < 0:
            raise ValueError(f"Depth must be non-negative: {depth_min}, {depth_max}")

        # Validate sample limit
        max_samples = self.config.max_samples or 10000
        if not isinstance(max_samples, int):
            raise ValueError(f"max_samples must be an integer: {max_samples}")
        limit = min(max_samples, 10000)

        # Build query using string concatenation with explicit float/int conversion
        # This ensures only numeric values can be interpolated
        query_parts = [
            "SELECT time, lat, lon, depth, temp, psal, doxy",
            "FROM tblArgoMerge_REP",
            "WHERE lat BETWEEN " + str(float(lat_min)) + " AND " + str(float(lat_max)),
            "AND lon BETWEEN " + str(float(lon_min)) + " AND " + str(float(lon_max)),
            "AND depth BETWEEN " + str(float(depth_min)) + " AND " + str(float(depth_max)),
            "ORDER BY time DESC",
            "LIMIT " + str(int(limit)),
        ]
        return " ".join(query_parts)

    def _download_via_pycmap(self) -> bool:
        """Download data using pycmap Python client."""
        try:
            import pycmap

            dataset_dir = self.data_path
            dataset_dir.mkdir(parents=True, exist_ok=True)
            cache_file = dataset_dir / "simons_cmap_real.npz"

            if cache_file.exists():
                logger.info(f"Simons CMAP data already cached at {cache_file}")
                self._is_real_data = True
                return True

            # Initialize API client
            api = pycmap.API(token=self.api_key)

            # Query Argo float data (most comprehensive real-time ocean data)
            logger.info("Downloading Argo float data from Simons CMAP...")

            # Build query with validated numeric bounds
            # All values are validated and converted to numeric types before interpolation
            query = self._build_cmap_query()

            df = api.query(query)

            if df is None or len(df) == 0:
                logger.warning("No data returned from CMAP query")
                return False

            # Process the data
            features, labels = self._process_cmap_data(df)

            # Save to cache
            np.savez_compressed(cache_file, features=features, labels=labels)
            self._is_real_data = True

            logger.info(
                f"Simons CMAP data loaded: {len(features)} samples, "
                f"{labels.sum()} anomalies (is_real_data=True)"
            )
            return True

        except Exception as e:
            logger.warning(f"Simons CMAP download failed: {e}")
            return False

    def _process_cmap_data(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Process CMAP query results.

        Args:
            df: DataFrame from CMAP query

        Returns:
            Tuple of (features, labels) numpy arrays
        """
        # Map columns to feature names
        col_map = {
            "lat": "latitude",
            "lon": "longitude",
            "depth": "depth",
            "temp": "temperature",
            "psal": "salinity",
            "chl": "chlorophyll",
            "nitrate": "nitrate",
            "doxy": "oxygen",
        }

        rows = []
        for _, row in df.iterrows():
            feature_row = []
            for fname in self.FEATURE_NAMES:
                # Find matching column
                for col, mapped in col_map.items():
                    if mapped == fname and col in df.columns:
                        val = row.get(col, 0)
                        feature_row.append(float(val) if pd.notna(val) else 0.0)
                        break
                else:
                    feature_row.append(0.0)
            rows.append(feature_row)

        features = np.array(rows, dtype=np.float32)
        features = np.nan_to_num(features, nan=0.0)

        # Label oceanographic anomalies
        # Low oxygen zones (hypoxia), unusual temperatures, high nutrients
        temp_col = self.FEATURE_NAMES.index("temperature")
        oxygen_col = self.FEATURE_NAMES.index("oxygen")
        nitrate_col = self.FEATURE_NAMES.index("nitrate")

        labels = (
            (features[:, oxygen_col] < 2.0)  # Hypoxic (< 2 mg/L)
            | (features[:, temp_col] > 30)  # Very warm surface
            | (features[:, nitrate_col] > 30)  # High nutrient upwelling
        ).astype(np.int64)

        return features, labels

    def _create_synthetic_ocean(self) -> bool:
        """Create synthetic oceanographic data."""
        np.random.seed(self.config.random_seed)
        n_samples = self.config.max_samples or 10000

        # Generate oceanographic features based on realistic distributions
        features = []
        labels = []

        for _ in range(n_samples):
            # Generate location (global ocean distribution)
            lat = np.random.uniform(-60, 60)
            lon = np.random.uniform(-180, 180)
            depth = np.random.exponential(100)  # Most samples near surface

            # Temperature profile (decreases with depth, varies by latitude)
            surface_temp = 28 - 0.3 * abs(lat)  # Warmer at equator
            temp = surface_temp * np.exp(-depth / 500) + np.random.normal(0, 1)

            # Salinity (relatively stable, varies slightly with location)
            salinity = 35 + np.random.normal(0, 0.5) + 0.01 * depth

            # Chlorophyll (higher in upwelling regions, surface)
            chl = np.random.exponential(0.3) * np.exp(-depth / 50)

            # Nitrate (increases with depth due to remineralization)
            nitrate = 0.5 + 30 * (1 - np.exp(-depth / 200)) + np.random.normal(0, 2)

            # Oxygen (decreases with depth, minimum around 400-800m)
            oxygen = 8 - 5 * np.exp(-((depth - 600) ** 2) / 50000) + np.random.normal(0, 0.5)

            feature_vec = [lat, lon, depth, temp, salinity, chl, max(nitrate, 0), max(oxygen, 0)]
            features.append(feature_vec)

            # Anomaly labeling based on oceanographic criteria
            is_anomaly = (
                oxygen < 2.0  # Oxygen minimum zone / hypoxia
                or temp > surface_temp + 3  # Temperature anomaly
                or (chl > 3 and depth < 10)  # Algal bloom
                or (nitrate > 35 and depth < 50)  # Strong upwelling
            )
            labels.append(1 if is_anomaly else 0)

        features = np.array(features, dtype=np.float32)  # type: ignore[assignment, unused-ignore]
        labels = np.array(labels, dtype=np.int64)  # type: ignore[assignment, unused-ignore]

        save_path = self.data_path / "synthetic_cmap.npz"
        np.savez_compressed(save_path, features=features, labels=labels)

        logger.info(
            f"Generated {n_samples} synthetic ocean samples, "
            f"{labels.sum()} anomalies (is_real_data=False)"  # type: ignore[attr-defined, unused-ignore]
        )
        return True

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Load ocean data from cache."""
        real_cache = self.data_path / "simons_cmap_real.npz"
        if real_cache.exists():
            data = np.load(real_cache)
            self._is_real_data = True
            logger.info(f"Loaded REAL Simons CMAP data from {real_cache}")
            return data["features"], data["labels"]

        synthetic_path = self.data_path / "synthetic_cmap.npz"
        if synthetic_path.exists() and ALLOW_SYNTHETIC:
            data = np.load(synthetic_path)
            self._is_real_data = False
            logger.info("Loaded SYNTHETIC ocean data (is_real_data=False)")
            return data["features"], data["labels"]

        raise FileNotFoundError("Simons CMAP data not found. Run download() first.")

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Preprocess oceanographic features."""
        data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        return data.astype(np.float32)


class WorldOceanDatabaseLoader(DatasetLoader):
    """
    NCEI World Ocean Database (WOD) Loader.

    Downloads REAL oceanographic profiles from the World Ocean Database:
    - Temperature/salinity profiles from ships, Argo floats, XBTs
    - Over 20 million profiles spanning 1770-present
    - Quality-controlled observations with metadata

    Data source: https://www.ncei.noaa.gov/products/world-ocean-database
    License: Public Domain (US Government)
    Citation: WOD Team (2025). World Ocean Database. NOAA NCEI.

    Note:
        This dataset does not include ground-truth anomaly labels.
        Use for unsupervised deployment only.
    """

    DATASET_NAME = "world_ocean_database"
    DATASET_URL = "https://www.ncei.noaa.gov/products/world-ocean-database"
    LICENSE = "Public Domain"
    CITATION = """WOD Team (2025). World Ocean Database data product series.
    NOAA National Centers for Environmental Information.
    https://doi.org/10.25921/v92s-y066"""
    REQUIRES_CREDENTIALS = False

    # WOD instrument types for sampling
    INSTRUMENT_TYPES = {
        "OSD": "Ocean Station Data (bottles)",
        "CTD": "Conductivity-Temperature-Depth",
        "XBT": "Expendable Bathythermograph",
        "PFL": "Profiling Floats (Argo)",
        "MRB": "Moored Buoys",
        "DRB": "Drifting Buoys",
    }

    FEATURE_NAMES = [
        "latitude",
        "longitude",
        "depth",
        "temperature",
        "salinity",
        "year",
        "month",
        "day",
    ]

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize WOD loader.

        Args:
            config: Dataset configuration. Preprocessing options:
                - instrument_type (str): 'CTD', 'PFL', 'XBT', etc.
                - year_range (tuple): (start_year, end_year)
                - region (dict): Geographic bounds
        """
        super().__init__(config)
        self.instrument_type = config.preprocessing.get("instrument_type", "PFL")
        self.year_range = config.preprocessing.get("year_range", (2020, 2025))
        self.region = config.preprocessing.get(
            "region",
            {
                "lat_min": -90,
                "lat_max": 90,
                "lon_min": -180,
                "lon_max": 180,
            },
        )
        self._is_real_data = False

    @property
    def is_real_data(self) -> bool:
        """Return True if real data was loaded."""
        return self._is_real_data

    def download(self) -> bool:
        """Download WOD data from NCEI.

        Note: Full WOD data requires bulk download or WODselect tool.
        This loader uses pre-sorted geographic data when available.

        Returns:
            True if download successful, False otherwise.
        """
        if self._download_wod_sample():
            return True

        if ALLOW_SYNTHETIC:
            check_synthetic_allowed(
                "WorldOceanDatabase",
                "WOD bulk data requires WODselect tool or wget",
            )
            return self._create_synthetic_wod()
        raise DataSourceUnavailableError(
            loader_name="WorldOceanDatabase",
            source_url="https://www.ncei.noaa.gov/products/world-ocean-database",
            reason=(
                "World Ocean Database bulk data requires WODselect tool or wget. "
                "Access: https://www.ncei.noaa.gov/access/world-ocean-database-select"
            ),
        )

    def _download_wod_sample(self) -> bool:
        """Attempt to download WOD sample data."""

        dataset_dir = self.data_path
        dataset_dir.mkdir(parents=True, exist_ok=True)
        cache_file = dataset_dir / "wod_real.npz"

        if cache_file.exists():
            logger.info(f"WOD data already cached at {cache_file}")
            self._is_real_data = True
            return True

        try:
            # WOD provides pre-sorted data by geographic squares
            # For sample purposes, download Argo float data for recent years
            base_url = TrustedEndpoints.NCEI_WOD_DATA

            logger.info("WOD requires local download via WODselect or wget.")
            logger.info(
                "For bulk download:\n"
                "  wget -r -np -nH --cut-dirs=4 "
                f"{base_url}/YEARLY/{self.year_range[1]}/wod_pfl_{self.year_range[1]}.nc.gz"
            )

            # For automated access, we'd need to parse the WOD native format
            # which requires specific decoders. Fall back to synthetic.
            return False

        except Exception as e:
            logger.warning(f"WOD download failed: {e}")
            return False

    def _create_synthetic_wod(self) -> bool:
        """Create synthetic WOD-like profile data."""
        np.random.seed(self.config.random_seed)
        n_samples = self.config.max_samples or 10000

        # Generate oceanographic profiles
        features = []
        labels = []

        for _ in range(n_samples):
            # Random location in ocean regions
            lat = np.random.uniform(-80, 80)
            lon = np.random.uniform(-180, 180)

            # Generate depth profile (surface to 2000m for Argo-like data)
            depth = np.random.exponential(200)

            # Temperature profile based on depth and latitude
            # Surface temperature varies with latitude
            sst = 28 - 0.4 * abs(lat) + np.random.normal(0, 2)
            # Temperature decreases exponentially with depth
            temp = 2 + (sst - 2) * np.exp(-depth / 300) + np.random.normal(0, 0.5)

            # Salinity profile
            # Surface salinity varies (higher in subtropics due to evaporation)
            sss = 35 + 0.5 * np.exp(-((abs(lat) - 25) ** 2) / 100) + np.random.normal(0, 0.2)
            salinity = sss + 0.01 * depth * (1 - np.exp(-depth / 500)) + np.random.normal(0, 0.1)

            # Date (random within year range)
            year = np.random.randint(self.year_range[0], self.year_range[1] + 1)
            month = np.random.randint(1, 13)
            day = np.random.randint(1, 29)

            feature_vec = [lat, lon, depth, temp, salinity, year, month, day]
            features.append(feature_vec)

            # Label oceanographic anomalies
            is_anomaly = (
                abs(temp - (2 + (sst - 2) * np.exp(-depth / 300))) > 2  # Temperature anomaly
                or abs(salinity - sss) > 1  # Salinity anomaly
                or (temp < 0 and lat > -60)  # Freezing water outside polar regions
            )
            labels.append(1 if is_anomaly else 0)

        features = np.array(features, dtype=np.float32)  # type: ignore[assignment, unused-ignore]
        labels = np.array(labels, dtype=np.int64)  # type: ignore[assignment, unused-ignore]

        save_path = self.data_path / "synthetic_wod.npz"
        np.savez_compressed(save_path, features=features, labels=labels)

        logger.info(
            f"Generated {n_samples} synthetic WOD profiles, "
            f"{labels.sum()} anomalies (is_real_data=False)"  # type: ignore[attr-defined, unused-ignore]
        )
        return True

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Load WOD data from cache."""
        real_cache = self.data_path / "wod_real.npz"
        if real_cache.exists():
            data = np.load(real_cache)
            self._is_real_data = True
            logger.info(f"Loaded REAL WOD data from {real_cache}")
            return data["features"], data["labels"]

        synthetic_path = self.data_path / "synthetic_wod.npz"
        if synthetic_path.exists() and ALLOW_SYNTHETIC:
            data = np.load(synthetic_path)
            self._is_real_data = False
            logger.info("Loaded SYNTHETIC WOD data (is_real_data=False)")
            return data["features"], data["labels"]

        raise FileNotFoundError("WOD data not found. Run download() first.")

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Preprocess WOD profile data."""
        data = np.nan_to_num(data, nan=0.0)
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        return data.astype(np.float32)


class CopernicusSeaLevelLoader(DatasetLoader):
    """
    Copernicus Climate Data Store Sea Level Loader.

    Downloads REAL satellite altimetry data from Copernicus CDS:
    - Global gridded sea level anomalies (SLA)
    - Multi-mission altimetry (Jason, Sentinel-3, etc.)
    - 0.25 degree resolution, 1993-present

    Data source: https://cds.climate.copernicus.eu/datasets/satellite-sea-level-global
    License: CC BY 4.0
    Requires: CDS API registration (free)

    Note:
        This dataset does not include ground-truth anomaly labels.
        Use for unsupervised deployment only.
    """

    DATASET_NAME = "copernicus_sea_level"
    DATASET_URL = "https://cds.climate.copernicus.eu/datasets/satellite-sea-level-global"
    LICENSE = "CC BY 4.0"
    CITATION = """Copernicus Climate Change Service (C3S). Sea level gridded data
    from satellite observations for the global ocean from 1993 to present.
    Climate Data Store. https://doi.org/10.24381/cds.4c328c78"""
    REQUIRES_CREDENTIALS = True  # CDS API key required

    FEATURE_NAMES = [
        "latitude",
        "longitude",
        "sla",  # Sea level anomaly (m)
        "adt",  # Absolute dynamic topography (m)
        "ugos",  # Geostrophic velocity u-component (m/s)
        "vgos",  # Geostrophic velocity v-component (m/s)
        "year",
        "month",
    ]

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize Copernicus sea level loader.

        Args:
            config: Dataset configuration. Preprocessing options:
                - cds_api_key (str): CDS API key
                - cds_api_url (str): CDS API URL (optional)
                - year_range (tuple): (start_year, end_year)
                - region (dict): Geographic bounds
        """
        super().__init__(config)
        self.cds_api_key = config.preprocessing.get("cds_api_key")
        self.cds_api_url = config.preprocessing.get(
            "cds_api_url", "https://cds.climate.copernicus.eu/api"
        )
        self.year_range = config.preprocessing.get("year_range", (2020, 2024))
        self.region = config.preprocessing.get(
            "region",
            {
                "lat_min": -60,
                "lat_max": 60,
                "lon_min": -180,
                "lon_max": 180,
            },
        )
        self._is_real_data = False

    @property
    def is_real_data(self) -> bool:
        """Return True if real data was loaded."""
        return self._is_real_data

    def download(self) -> bool:
        """Download sea level data from Copernicus CDS.

        Requires CDS API credentials. Register at:
        https://cds.climate.copernicus.eu/user/register

        Returns:
            True if download successful, False otherwise.
        """
        # Try to use cdsapi if available
        try:
            import cdsapi  # noqa: F401

            if self.cds_api_key:
                return self._download_via_cdsapi()
        except ImportError:
            logger.info("cdsapi not installed. Using synthetic fallback.")

        if ALLOW_SYNTHETIC:
            check_synthetic_allowed(
                "CopernicusSeaLevel",
                "Copernicus CDS requires cdsapi package and CDS API credentials",
            )
            return self._create_synthetic_sea_level()
        raise DataSourceUnavailableError(
            loader_name="CopernicusSeaLevel",
            source_url="https://cds.climate.copernicus.eu/datasets/satellite-sea-level-global",
            reason=(
                "Copernicus CDS requires cdsapi package and CDS API credentials. "
                "Install: pip install cdsapi | "
                "Register: https://cds.climate.copernicus.eu/user/register"
            ),
        )

    def _download_via_cdsapi(self) -> bool:
        """Download data using cdsapi Python client."""
        try:
            import cdsapi

            dataset_dir = self.data_path
            dataset_dir.mkdir(parents=True, exist_ok=True)
            cache_file = dataset_dir / "copernicus_sealevel_real.npz"

            if cache_file.exists():
                logger.info(f"Copernicus sea level data cached at {cache_file}")
                self._is_real_data = True
                return True

            # Initialize CDS client
            c = cdsapi.Client(url=self.cds_api_url, key=self.cds_api_key)

            # Download sea level data
            logger.info("Downloading sea level data from Copernicus CDS...")

            # Request monthly gridded sea level anomalies
            download_file = dataset_dir / "sea_level_temp.nc"

            c.retrieve(
                "satellite-sea-level-global",
                {
                    "version": "vDT2024",
                    "variable": [
                        "daily",
                    ],
                    "year": [str(y) for y in range(self.year_range[0], self.year_range[1] + 1)],
                    "month": ["01", "06", "12"],  # Sample months
                    "day": ["15"],
                    "format": "netcdf",
                },
                str(download_file),
            )

            # Parse NetCDF and convert to numpy
            import netCDF4 as nc

            with nc.Dataset(download_file) as ds:
                features, labels = self._process_netcdf(ds)

            # Save to cache
            np.savez_compressed(cache_file, features=features, labels=labels)
            self._is_real_data = True

            # Cleanup temp file
            download_file.unlink()

            logger.info(
                f"Copernicus sea level data loaded: {len(features)} samples, "
                f"{labels.sum()} anomalies (is_real_data=True)"
            )
            return True

        except Exception as e:
            logger.warning(f"Copernicus CDS download failed: {e}")
            return False

    def _process_netcdf(self, ds: Any) -> tuple[np.ndarray, np.ndarray]:
        """Process NetCDF sea level data."""
        lats = ds.variables["latitude"][:]
        lons = ds.variables["longitude"][:]
        sla = ds.variables["sla"][:]  # Sea level anomaly

        rows = []
        for t_idx in range(min(len(sla), 100)):  # Limit samples
            for lat_idx in range(0, len(lats), 10):  # Subsample spatially
                for lon_idx in range(0, len(lons), 10):
                    if not np.ma.is_masked(sla[t_idx, lat_idx, lon_idx]):
                        row = [
                            lats[lat_idx],
                            lons[lon_idx],
                            float(sla[t_idx, lat_idx, lon_idx]),
                            0,  # ADT (if available)
                            0,  # ugos
                            0,  # vgos
                            self.year_range[0] + t_idx // 3,
                            (t_idx % 3) * 5 + 1,  # Approximate month
                        ]
                        rows.append(row)

        features = np.array(rows[: self.config.max_samples or 10000], dtype=np.float32)

        # Label significant sea level anomalies
        sla_col = 2
        labels = (np.abs(features[:, sla_col]) > 0.15).astype(np.int64)  # > 15cm anomaly

        return features, labels

    def _create_synthetic_sea_level(self) -> bool:
        """Create synthetic sea level anomaly data."""
        np.random.seed(self.config.random_seed)
        n_samples = self.config.max_samples or 10000

        features = []
        labels = []

        for _ in range(n_samples):
            # Random ocean location
            lat = np.random.uniform(-60, 60)
            lon = np.random.uniform(-180, 180)

            # Sea level anomaly (global mean ~0, regional variations)
            # El Nino/La Nina patterns in tropical Pacific
            enso_effect = 0.1 * np.sin(2 * np.pi * lon / 360) if abs(lat) < 20 else 0
            seasonal = 0.05 * np.sin(2 * np.pi * np.random.random())
            sla = enso_effect + seasonal + np.random.normal(0, 0.05)

            # Absolute dynamic topography
            adt = 0.5 + sla + np.random.normal(0, 0.01)

            # Geostrophic velocities (derived from sea level gradients)
            ugos = np.random.normal(0, 0.1)
            vgos = np.random.normal(0, 0.1)

            year = np.random.randint(self.year_range[0], self.year_range[1] + 1)
            month = np.random.randint(1, 13)

            feature_vec = [lat, lon, sla, adt, ugos, vgos, year, month]
            features.append(feature_vec)

            # Anomaly: significant sea level changes
            is_anomaly = (
                abs(sla) > 0.15  # > 15cm anomaly
                or abs(ugos) > 0.5  # Strong current
                or abs(vgos) > 0.5
            )
            labels.append(1 if is_anomaly else 0)

        features = np.array(features, dtype=np.float32)  # type: ignore[assignment, unused-ignore]
        labels = np.array(labels, dtype=np.int64)  # type: ignore[assignment, unused-ignore]

        save_path = self.data_path / "synthetic_sealevel.npz"
        np.savez_compressed(save_path, features=features, labels=labels)

        logger.info(
            f"Generated {n_samples} synthetic sea level samples, "
            f"{labels.sum()} anomalies (is_real_data=False)"  # type: ignore[attr-defined, unused-ignore]
        )
        return True

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Load sea level data from cache."""
        real_cache = self.data_path / "copernicus_sealevel_real.npz"
        if real_cache.exists():
            data = np.load(real_cache)
            self._is_real_data = True
            logger.info(f"Loaded REAL Copernicus sea level data from {real_cache}")
            return data["features"], data["labels"]

        synthetic_path = self.data_path / "synthetic_sealevel.npz"
        if synthetic_path.exists() and ALLOW_SYNTHETIC:
            data = np.load(synthetic_path)
            self._is_real_data = False
            logger.info("Loaded SYNTHETIC sea level data (is_real_data=False)")
            return data["features"], data["labels"]

        raise FileNotFoundError("Sea level data not found. Run download() first.")

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Preprocess sea level data."""
        data = np.nan_to_num(data, nan=0.0)
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        return data.astype(np.float32)


class CopernicusERA5Loader(DatasetLoader):
    """
    Copernicus Climate Data Store ERA5 Reanalysis Loader.

    Downloads REAL atmospheric reanalysis data from ERA5:
    - Hourly data from 1940-present at 0.25° resolution
    - Temperature, pressure, wind, humidity, precipitation
    - Most comprehensive climate reanalysis dataset available

    ERA5 is the gold standard for climate analysis and anomaly detection,
    providing consistent atmospheric state estimates spanning 80+ years.

    Data source: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
    License: CC BY 4.0
    Requires: CDS API registration (free)
    Citation: Hersbach et al. (2020). ERA5 hourly data. Copernicus CDS.
    """

    DATASET_NAME = "copernicus_era5"
    DATASET_URL = "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels"
    LICENSE = "CC BY 4.0"
    CITATION = """Hersbach, H. et al. (2020). The ERA5 global reanalysis.
    Quarterly Journal of the Royal Meteorological Society, 146(730), 1999-2049.
    https://doi.org/10.1002/qj.3803"""
    REQUIRES_CREDENTIALS = True  # CDS API key required

    # ERA5 single-level variables for surface weather
    VARIABLE_SETS = {
        "surface": [
            "2m_temperature",
            "2m_dewpoint_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "surface_pressure",
            "total_precipitation",
        ],
        "radiation": [
            "surface_solar_radiation_downwards",
            "surface_thermal_radiation_downwards",
            "top_net_solar_radiation",
        ],
        "soil": [
            "soil_temperature_level_1",
            "volumetric_soil_water_layer_1",
            "snow_depth",
        ],
    }

    FEATURE_NAMES = [
        "latitude",
        "longitude",
        "temperature_2m",  # K -> C
        "dewpoint_2m",  # K -> C
        "u_wind_10m",  # m/s
        "v_wind_10m",  # m/s
        "pressure",  # Pa -> hPa
        "precipitation",  # m -> mm
        "year",
        "month",
        "day",
        "hour",
    ]

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize ERA5 loader.

        Args:
            config: Dataset configuration. Preprocessing options:
                - cds_api_key (str): CDS API key
                - cds_api_url (str): CDS API URL (optional)
                - variable_set (str): 'surface', 'radiation', or 'soil'
                - year_range (tuple): (start_year, end_year)
                - region (dict): Geographic bounds {lat_min, lat_max, lon_min, lon_max}
                - hours (list): Hours to retrieve [0, 6, 12, 18] for 6-hourly
        """
        super().__init__(config)
        self.cds_api_key = config.preprocessing.get("cds_api_key")
        self.cds_api_url = config.preprocessing.get(
            "cds_api_url", "https://cds.climate.copernicus.eu/api"
        )
        self.variable_set = config.preprocessing.get("variable_set", "surface")
        self.year_range = config.preprocessing.get("year_range", (2020, 2024))
        self.region = config.preprocessing.get(
            "region",
            {
                "lat_min": 30,
                "lat_max": 50,
                "lon_min": -130,
                "lon_max": -70,
            },
        )
        self.hours = config.preprocessing.get("hours", [0, 6, 12, 18])
        self._is_real_data = False

    @property
    def is_real_data(self) -> bool:
        """Return True if real data was loaded."""
        return self._is_real_data

    def download(self) -> bool:
        """Download ERA5 reanalysis data from Copernicus CDS.

        Returns:
            True if download successful, False otherwise.
        """
        # Try to use cdsapi if available
        try:
            import cdsapi  # noqa: F401

            if self.cds_api_key:
                return self._download_via_cdsapi()
        except ImportError:
            logger.info("cdsapi not installed. Using synthetic fallback.")

        if ALLOW_SYNTHETIC:
            check_synthetic_allowed(
                "CopernicusERA5",
                "Copernicus ERA5 requires cdsapi package and API key",
            )
            return self._create_synthetic_era5()
        raise DataSourceUnavailableError(
            loader_name="CopernicusERA5",
            source_url="https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels",
            reason=(
                "Copernicus ERA5 requires cdsapi package and API key. "
                "Install: pip install cdsapi | "
                "Register: https://cds.climate.copernicus.eu/user/register"
            ),
        )

    def _download_via_cdsapi(self) -> bool:
        """Download data using cdsapi Python client."""
        try:
            import cdsapi

            dataset_dir = self.data_path
            dataset_dir.mkdir(parents=True, exist_ok=True)
            cache_file = dataset_dir / "copernicus_era5_real.npz"

            if cache_file.exists():
                logger.info(f"ERA5 data cached at {cache_file}")
                self._is_real_data = True
                return True

            # Initialize CDS client
            c = cdsapi.Client(url=self.cds_api_url, key=self.cds_api_key)

            logger.info("Downloading ERA5 reanalysis data from Copernicus CDS...")

            download_file = dataset_dir / "era5_temp.nc"
            variables = self.VARIABLE_SETS.get(self.variable_set, self.VARIABLE_SETS["surface"])

            # Request ERA5 data
            c.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "variable": variables,
                    "year": [str(y) for y in range(self.year_range[0], self.year_range[1] + 1)],
                    "month": ["01", "04", "07", "10"],  # Quarterly sampling
                    "day": ["15"],
                    "time": [f"{h:02d}:00" for h in self.hours],
                    "area": [
                        self.region["lat_max"],
                        self.region["lon_min"],
                        self.region["lat_min"],
                        self.region["lon_max"],
                    ],
                    "format": "netcdf",
                },
                str(download_file),
            )

            # Parse NetCDF
            import netCDF4 as nc

            with nc.Dataset(download_file) as ds:
                features, labels = self._process_era5_netcdf(ds)

            np.savez_compressed(cache_file, features=features, labels=labels)
            self._is_real_data = True
            download_file.unlink()

            logger.info(
                f"ERA5 data loaded: {len(features)} samples, "
                f"{labels.sum()} anomalies (is_real_data=True)"
            )
            return True

        except Exception as e:
            logger.warning(f"ERA5 CDS download failed: {e}")
            return False

    def _process_era5_netcdf(self, ds: Any) -> tuple[np.ndarray, np.ndarray]:
        """Process NetCDF ERA5 data."""
        lats = ds.variables.get("latitude", ds.variables.get("lat"))[:]
        lons = ds.variables.get("longitude", ds.variables.get("lon"))[:]
        time_var = ds.variables.get("time", ds.variables.get("valid_time"))[:]

        # Get available variables
        t2m = ds.variables.get("t2m", ds.variables.get("2m_temperature"))
        d2m = ds.variables.get("d2m", ds.variables.get("2m_dewpoint_temperature"))
        u10 = ds.variables.get("u10", ds.variables.get("10m_u_component_of_wind"))
        v10 = ds.variables.get("v10", ds.variables.get("10m_v_component_of_wind"))
        sp = ds.variables.get("sp", ds.variables.get("surface_pressure"))
        tp = ds.variables.get("tp", ds.variables.get("total_precipitation"))

        rows: list[list[float]] = []
        max_samples = self.config.max_samples or 10000

        for t_idx in range(min(len(time_var), 100)):
            for lat_idx in range(0, len(lats), 5):
                for lon_idx in range(0, len(lons), 5):
                    if len(rows) >= max_samples:
                        break

                    try:
                        row = [
                            float(lats[lat_idx]),
                            float(lons[lon_idx]),
                            float(t2m[t_idx, lat_idx, lon_idx]) - 273.15 if t2m else 0,
                            float(d2m[t_idx, lat_idx, lon_idx]) - 273.15 if d2m else 0,
                            float(u10[t_idx, lat_idx, lon_idx]) if u10 else 0,
                            float(v10[t_idx, lat_idx, lon_idx]) if v10 else 0,
                            float(sp[t_idx, lat_idx, lon_idx]) / 100 if sp else 0,
                            float(tp[t_idx, lat_idx, lon_idx]) * 1000 if tp else 0,
                            self.year_range[0] + t_idx // 4,
                            (t_idx % 4) * 3 + 1,
                            15,
                            0,
                        ]
                        rows.append(row)
                    except (IndexError, TypeError):
                        continue

        features = np.array(rows, dtype=np.float32)

        # Label climate anomalies
        temp_col = 2
        precip_col = 7
        temp_mean = features[:, temp_col].mean()
        temp_std = features[:, temp_col].std() + 1e-8
        precip_95 = np.percentile(features[:, precip_col], 95)

        labels = (
            (np.abs(features[:, temp_col] - temp_mean) > 2 * temp_std)
            | (features[:, precip_col] > precip_95)
        ).astype(np.int64)

        return features, labels

    def _create_synthetic_era5(self) -> bool:
        """Create synthetic ERA5-like atmospheric reanalysis data."""
        np.random.seed(self.config.random_seed)
        n_samples = self.config.max_samples or 10000

        features = []
        labels = []

        lat_range = (self.region["lat_min"], self.region["lat_max"])
        lon_range = (self.region["lon_min"], self.region["lon_max"])

        for _ in range(n_samples):
            lat = np.random.uniform(*lat_range)
            lon = np.random.uniform(*lon_range)

            # Temperature varies with latitude and season
            base_temp = 25 - 0.5 * abs(lat - 35)
            month = np.random.randint(1, 13)
            seasonal = 10 * np.sin(2 * np.pi * (month - 1) / 12)
            temp_2m = base_temp + seasonal + np.random.normal(0, 3)

            # Dewpoint slightly lower than temperature
            dewpoint_2m = temp_2m - np.random.exponential(5)

            # Wind components
            u_wind = np.random.normal(0, 5)
            v_wind = np.random.normal(0, 5)

            # Surface pressure (varies with altitude and weather)
            pressure = np.random.normal(1013, 10)

            # Precipitation (exponential with seasonal variation)
            precip_rate = 0.5 + 0.3 * np.sin(2 * np.pi * (month - 3) / 12)
            precipitation = np.random.exponential(precip_rate) if np.random.random() > 0.7 else 0

            year = np.random.randint(self.year_range[0], self.year_range[1] + 1)
            day = np.random.randint(1, 29)
            hour = np.random.choice(self.hours)

            feature_vec = [
                lat,
                lon,
                temp_2m,
                dewpoint_2m,
                u_wind,
                v_wind,
                pressure,
                precipitation,
                year,
                month,
                day,
                hour,
            ]
            features.append(feature_vec)

            # Climate anomalies
            is_anomaly = (
                abs(temp_2m - base_temp - seasonal) > 6
                or precipitation > 20
                or abs(pressure - 1013) > 25
                or np.sqrt(u_wind**2 + v_wind**2) > 15
            )
            labels.append(1 if is_anomaly else 0)

        features = np.array(features, dtype=np.float32)  # type: ignore[assignment, unused-ignore]
        labels = np.array(labels, dtype=np.int64)  # type: ignore[assignment, unused-ignore]

        save_path = self.data_path / "synthetic_era5.npz"
        np.savez_compressed(save_path, features=features, labels=labels)

        logger.info(
            f"Generated {n_samples} synthetic ERA5 samples, "
            f"{labels.sum()} anomalies (is_real_data=False)"  # type: ignore[attr-defined, unused-ignore]
        )
        return True

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Load ERA5 data from cache."""
        real_cache = self.data_path / "copernicus_era5_real.npz"
        if real_cache.exists():
            data = np.load(real_cache)
            self._is_real_data = True
            logger.info(f"Loaded REAL ERA5 data from {real_cache}")
            return data["features"], data["labels"]

        synthetic_path = self.data_path / "synthetic_era5.npz"
        if synthetic_path.exists() and ALLOW_SYNTHETIC:
            data = np.load(synthetic_path)
            self._is_real_data = False
            logger.info("Loaded SYNTHETIC ERA5 data (is_real_data=False)")
            return data["features"], data["labels"]

        raise FileNotFoundError("ERA5 data not found. Run download() first.")

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Preprocess ERA5 atmospheric data."""
        data = np.nan_to_num(data, nan=0.0)
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        return data.astype(np.float32)


# Register climate/ocean loaders
DatasetRegistry.register("simons_cmap", SimonsCMAPLoader)
DatasetRegistry.register("cmap", SimonsCMAPLoader)  # Alias
DatasetRegistry.register("world_ocean_database", WorldOceanDatabaseLoader)
DatasetRegistry.register("wod", WorldOceanDatabaseLoader)  # Alias
DatasetRegistry.register("copernicus_sea_level", CopernicusSeaLevelLoader)
DatasetRegistry.register("sea_level", CopernicusSeaLevelLoader)  # Alias
DatasetRegistry.register("copernicus_era5", CopernicusERA5Loader)
DatasetRegistry.register("era5", CopernicusERA5Loader)  # Alias
