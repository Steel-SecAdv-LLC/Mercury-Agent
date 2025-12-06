"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

Environmental Dataset Loaders: USGS Earthquake, NOAA Weather, Wildfire Data

These loaders fetch REAL DATA from public APIs:
- USGS Earthquake Catalog API: https://earthquake.usgs.gov/fdsnws/event/1/
- NOAA Climate Data Online: https://www.ncdc.noaa.gov/cdo-web/api/v2/
- NASA FIRMS (Fire): https://firms.modaps.eosdis.nasa.gov/api/

References:
- USGS Earthquake Catalog: https://earthquake.usgs.gov/earthquakes/search/
- NOAA Climate Data: https://www.ncdc.noaa.gov/cdo-web/
- NASA FIRMS (Fire): https://firms.modaps.eosdis.nasa.gov/
"""

import json
import logging
from datetime import datetime, timedelta

import numpy as np

from .base import DatasetConfig, DatasetLoader, DatasetRegistry

logger = logging.getLogger(__name__)


class USGSEarthquakeLoader(DatasetLoader):
    """
    USGS Earthquake Catalog Data Loader.

    Downloads REAL earthquake data from the USGS Earthquake Hazards Program API.
    API Documentation: https://earthquake.usgs.gov/fdsnws/event/1/

    Features extracted:
    - Location (lat, lon, depth)
    - Magnitude and type
    - Network quality metrics (gap, dmin, rms, nst)
    - Error estimates

    Reference: U.S. Geological Survey Earthquake Hazards Program
    """

    DATASET_NAME = "earthquake"
    DATASET_URL = "https://earthquake.usgs.gov/earthquakes/search/"
    LICENSE = "Public Domain (USGS)"
    CITATION = """U.S. Geological Survey (USGS) Earthquake Hazards Program.
    National Earthquake Information Center (NEIC)."""
    REQUIRES_CREDENTIALS = False

    # USGS Earthquake API endpoint
    API_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

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
        "significance",
        "felt",
        "cdi",
        "mmi",
        "tsunami",
    ]

    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        self.min_magnitude = config.preprocessing.get("min_magnitude", 2.5)
        self.days_back = config.preprocessing.get("days_back", 365)

    def download(self) -> bool:
        """Download REAL earthquake data from USGS API."""
        import urllib.error
        import urllib.request

        logger.info("Downloading REAL earthquake data from USGS API...")

        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=self.days_back)

        # Build API request
        params = {
            "format": "geojson",
            "starttime": start_date.strftime("%Y-%m-%d"),
            "endtime": end_date.strftime("%Y-%m-%d"),
            "minmagnitude": str(self.min_magnitude),
            "orderby": "time",
            "limit": str(min(self.config.max_samples or 20000, 20000)),
        }

        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{self.API_URL}?{query_string}"

        try:
            logger.info(f"  Fetching from: {url}")
            with urllib.request.urlopen(url, timeout=60) as response:
                data = json.loads(response.read().decode())

            # Save raw GeoJSON
            raw_path = self.data_path / "usgs_earthquakes.json"
            with open(raw_path, "w") as f:
                json.dump(data, f)

            n_events = len(data.get("features", []))
            logger.info(f"  Downloaded {n_events} REAL earthquake events")
            return True

        except urllib.error.URLError as e:
            logger.error(f"Failed to download earthquake data: {e}")
            return False
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse earthquake data: {e}")
            return False

    def _load_raw(self) -> tuple[np.ndarray, np.ndarray]:
        """Load REAL earthquake data from downloaded JSON."""
        raw_path = self.data_path / "usgs_earthquakes.json"

        if not raw_path.exists():
            raise FileNotFoundError(
                f"USGS earthquake data not found at {raw_path}. "
                "Run with download=True to fetch real data from USGS API."
            )

        logger.info("Loading REAL USGS earthquake data...")

        with open(raw_path) as f:
            data = json.load(f)

        features = []
        labels = []

        for feature in data.get("features", []):
            props = feature.get("properties", {})
            geom = feature.get("geometry", {})
            coords = geom.get("coordinates", [0, 0, 0])

            # Extract features
            row = [
                coords[1] if len(coords) > 1 else 0,  # latitude
                coords[0] if len(coords) > 0 else 0,  # longitude
                coords[2] if len(coords) > 2 else 0,  # depth
                props.get("mag", 0) or 0,
                props.get("gap", 0) or 0,
                props.get("dmin", 0) or 0,
                props.get("rms", 0) or 0,
                props.get("nst", 0) or 0,
                props.get("horizontalError", 0) or 0,
                props.get("depthError", 0) or 0,
                props.get("magError", 0) or 0,
                props.get("sig", 0) or 0,  # significance
                props.get("felt", 0) or 0,
                props.get("cdi", 0) or 0,  # community intensity
                props.get("mmi", 0) or 0,  # modified Mercalli
                1 if props.get("tsunami", 0) else 0,
            ]
            features.append(row)

            # Label: significant seismic events (magnitude >= 5 or high significance)
            mag = props.get("mag", 0) or 0
            sig = props.get("sig", 0) or 0
            tsunami = props.get("tsunami", 0) or 0
            is_anomaly = mag >= 5.0 or sig >= 500 or tsunami > 0
            labels.append(1 if is_anomaly else 0)

        features = np.array(features, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        logger.info(f"Loaded {len(features)} REAL earthquake events")
        logger.info(f"  Significant events (M>=5 or high sig): {labels.sum()}")

        return features, labels

    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """Preprocess earthquake features."""
        data = np.nan_to_num(data, nan=0.0, posinf=1e10, neginf=-1e10)
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        return data.astype(np.float32)

    def get_dataset_info(self) -> dict:
        """Get information about the loaded dataset."""
        return {
            "name": "USGS Earthquake Catalog",
            "type": "REAL DATA (API)",
            "source": self.API_URL,
            "features": len(self.FEATURE_NAMES),
            "citation": self.CITATION,
        }


class NOAAWeatherLoader(DatasetLoader):
    """
    NOAA Weather/Climate Data Loader.

    Downloads REAL weather data from NOAA's public APIs and datasets.

    Data Sources:
    - Storm Events Database: https://www.ncdc.noaa.gov/stormevents/
    - GHCN (Global Historical Climatology Network): https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily

    Reference: https://www.ncdc.noaa.gov/
    """

    DATASET_NAME = "weather"
    DATASET_URL = "https://www.ncdc.noaa.gov/cdo-web/"
    LICENSE = "Public Domain (NOAA)"
    CITATION = """NOAA National Centers for Environmental Information.
    Climate Data Online (CDO)."""
    REQUIRES_CREDENTIALS = False

    # Storm events CSV download (public, no API key needed)
    STORM_EVENTS_URL = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"

    FEATURE_NAMES = [
        "event_type_code",
        "state_fips",
        "year",
        "month",
        "begin_day",
        "begin_lat",
        "begin_lon",
        "end_lat",
        "end_lon",
        "injuries_direct",
        "injuries_indirect",
        "deaths_direct",
        "deaths_indirect",
        "damage_property",
        "damage_crops",
        "magnitude",
        "magnitude_type",
        "tor_f_scale",
    ]

    # Event type to code mapping
    EVENT_TYPES = {
        "Tornado": 1,
        "Hail": 2,
        "Thunderstorm Wind": 3,
        "Flash Flood": 4,
        "Flood": 5,
        "Winter Storm": 6,
        "Heavy Snow": 7,
        "High Wind": 8,
        "Blizzard": 9,
        "Ice Storm": 10,
        "Drought": 11,
        "Hurricane": 12,
        "Tropical Storm": 13,
        "Heat": 14,
        "Wildfire": 15,
        "Lightning": 16,
    }

    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        self.year = config.preprocessing.get("year", 2023)

    def download(self) -> bool:
        """Download REAL storm events data from NOAA."""
        import gzip
        import io
        import urllib.error
        import urllib.request

        logger.info("Downloading REAL NOAA Storm Events data...")

        # Storm events file naming: StormEvents_details-ftp_v1.0_dYYYY_cYYYYMMDD.csv.gz
        # We'll download the yearly summary file
        year = self.year

        # Try to find and download storm events data
        # List of potential filenames (NOAA naming convention varies)
        potential_files = [
            f"StormEvents_details-ftp_v1.0_d{year}_c{year + 1}0101.csv.gz",
            f"StormEvents_details-ftp_v1.0_d{year}_c{year}1231.csv.gz",
        ]

        downloaded = False
        for filename in potential_files:
            url = f"{self.STORM_EVENTS_URL}{filename}"
            output_path = self.data_path / f"storm_events_{year}.csv"

            try:
                logger.info(f"  Trying: {url}")
                with urllib.request.urlopen(url, timeout=60) as response:
                    # Decompress gzip
                    compressed = response.read()
                    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as gz:
                        data = gz.read().decode("utf-8", errors="ignore")

                with open(output_path, "w") as f:
                    f.write(data)

                logger.info(f"  Downloaded storm events for {year}")
                downloaded = True
                break
            except urllib.error.URLError as e:
                logger.warning(f"  Failed: {e}")
                continue

        if not downloaded:
            # Provide manual download instructions
            logger.warning("Could not auto-download NOAA storm events.")
            logger.warning("Please manually download from:")
            logger.warning(f"  {self.STORM_EVENTS_URL}")
            logger.warning(f"  Place CSV file in: {self.data_path}")

            instructions_path = self.data_path / "DOWNLOAD_INSTRUCTIONS.txt"
            with open(instructions_path, "w") as f:
                f.write("NOAA Storm Events Download Instructions\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"1. Visit: {self.STORM_EVENTS_URL}\n")
                f.write(f"2. Download: StormEvents_details-ftp_v1.0_d{year}_*.csv.gz\n")
                f.write(f"3. Extract and place in: {self.data_path}\n")
                f.write(f"4. Rename to: storm_events_{year}.csv\n")

            return False

        return True

    def _load_raw(self) -> tuple[np.ndarray, np.ndarray]:
        """Load REAL NOAA storm events data."""
        csv_files = list(self.data_path.glob("storm_events_*.csv"))

        if not csv_files:
            raise FileNotFoundError(
                f"NOAA storm events data not found in {self.data_path}. "
                "Run with download=True to fetch real data."
            )

        logger.info(f"Loading REAL NOAA storm events from {len(csv_files)} files...")

        features = []
        labels = []

        for csv_file in csv_files:
            f_data, l_data = self._parse_storm_events(csv_file)
            features.extend(f_data)
            labels.extend(l_data)

        features = np.array(features, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        logger.info(f"Loaded {len(features)} REAL storm events")
        logger.info(f"  Severe events: {labels.sum()}")

        return features, labels

    def _parse_storm_events(self, filepath) -> tuple[list, list]:
        """Parse NOAA storm events CSV."""
        import csv

        features = []
        labels = []

        with open(filepath, newline="", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)

            for row in reader:
                try:
                    event_type = row.get("EVENT_TYPE", "Unknown")
                    event_code = self.EVENT_TYPES.get(event_type, 0)

                    feature_row = [
                        event_code,
                        int(row.get("STATE_FIPS", 0) or 0),
                        int(row.get("YEAR", 0) or 0),
                        int(row.get("MONTH_NAME", "1").split()[0] if row.get("MONTH_NAME") else 1),
                        int(row.get("BEGIN_DAY", 1) or 1),
                        float(row.get("BEGIN_LAT", 0) or 0),
                        float(row.get("BEGIN_LON", 0) or 0),
                        float(row.get("END_LAT", 0) or 0),
                        float(row.get("END_LON", 0) or 0),
                        int(row.get("INJURIES_DIRECT", 0) or 0),
                        int(row.get("INJURIES_INDIRECT", 0) or 0),
                        int(row.get("DEATHS_DIRECT", 0) or 0),
                        int(row.get("DEATHS_INDIRECT", 0) or 0),
                        self._parse_damage(row.get("DAMAGE_PROPERTY", "0")),
                        self._parse_damage(row.get("DAMAGE_CROPS", "0")),
                        float(row.get("MAGNITUDE", 0) or 0),
                        1 if row.get("MAGNITUDE_TYPE") else 0,
                        self._parse_tornado_scale(row.get("TOR_F_SCALE", "")),
                    ]

                    features.append(feature_row)

                    # Label: severe event (deaths, injuries, or significant damage)
                    deaths = int(row.get("DEATHS_DIRECT", 0) or 0) + int(
                        row.get("DEATHS_INDIRECT", 0) or 0
                    )
                    injuries = int(row.get("INJURIES_DIRECT", 0) or 0) + int(
                        row.get("INJURIES_INDIRECT", 0) or 0
                    )
                    damage = self._parse_damage(row.get("DAMAGE_PROPERTY", "0"))
                    is_severe = deaths > 0 or injuries > 5 or damage > 100000
                    labels.append(1 if is_severe else 0)

                except (ValueError, KeyError) as e:
                    continue

        return features, labels

    def _parse_damage(self, damage_str: str) -> float:
        """Parse NOAA damage string (e.g., '10K', '1.5M')."""
        if not damage_str:
            return 0.0
        damage_str = str(damage_str).strip().upper()
        multipliers = {"K": 1000, "M": 1000000, "B": 1000000000}
        for suffix, mult in multipliers.items():
            if damage_str.endswith(suffix):
                try:
                    return float(damage_str[:-1]) * mult
                except ValueError:
                    return 0.0
        try:
            return float(damage_str)
        except ValueError:
            return 0.0

    def _parse_tornado_scale(self, scale_str: str) -> int:
        """Parse tornado F/EF scale."""
        if not scale_str:
            return -1
        scale_str = str(scale_str).upper()
        for i in range(6):
            if f"F{i}" in scale_str or f"EF{i}" in scale_str:
                return i
        return -1

    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """Preprocess weather features."""
        data = np.nan_to_num(data, nan=0.0)
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        return data.astype(np.float32)

    def get_dataset_info(self) -> dict:
        """Get information about the loaded dataset."""
        return {
            "name": "NOAA Storm Events Database",
            "type": "REAL DATA",
            "source": self.STORM_EVENTS_URL,
            "features": len(self.FEATURE_NAMES),
            "citation": self.CITATION,
        }


class WildfireDataLoader(DatasetLoader):
    """
    NASA FIRMS Wildfire Detection Data Loader.

    Downloads REAL active fire detection data from NASA FIRMS (Fire Information
    for Resource Management System).

    Data Source: https://firms.modaps.eosdis.nasa.gov/
    API Documentation: https://firms.modaps.eosdis.nasa.gov/api/

    Features include satellite-detected fire hotspots with:
    - Location (lat, lon)
    - Brightness temperature (Kelvin)
    - Fire Radiative Power (FRP)
    - Confidence level
    - Satellite detection parameters

    Note: Requires free API key from NASA FIRMS for full access.
    Limited data available without key.
    """

    DATASET_NAME = "wildfire"
    DATASET_URL = "https://firms.modaps.eosdis.nasa.gov/"
    LICENSE = "Public Domain (NASA)"
    CITATION = """NASA FIRMS (Fire Information for Resource Management System).
    MODIS Collection 6 and VIIRS Active Fire Products."""
    REQUIRES_CREDENTIALS = False  # API key optional for limited access

    # NASA FIRMS API endpoints
    # Open data (last 24h, limited area): https://firms.modaps.eosdis.nasa.gov/data/active_fire/
    OPEN_DATA_URL = "https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/"

    FEATURE_NAMES = [
        "latitude",
        "longitude",
        "brightness",
        "scan",
        "track",
        "acq_date",
        "acq_time",
        "satellite",
        "confidence",
        "version",
        "bright_t31",
        "frp",
        "daynight",
    ]

    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        self.api_key = config.preprocessing.get("api_key", None)
        self.days_back = config.preprocessing.get("days_back", 7)

    def download(self) -> bool:
        """Download REAL fire detection data from NASA FIRMS."""
        import urllib.error
        import urllib.request

        logger.info("Downloading REAL NASA FIRMS fire detection data...")

        # NASA FIRMS provides open CSV downloads for recent global fire data
        # Format: MODIS_C6_1_Global_24h.csv (last 24 hours, global)
        open_files = [
            "MODIS_C6_1_Global_24h.csv",
            "MODIS_C6_1_Global_48h.csv",
            "MODIS_C6_1_Global_7d.csv",
        ]

        downloaded = False
        for filename in open_files:
            url = f"{self.OPEN_DATA_URL}{filename}"
            output_path = self.data_path / f"firms_{filename}"

            try:
                logger.info(f"  Trying: {url}")
                with urllib.request.urlopen(url, timeout=120) as response:
                    data = response.read().decode("utf-8", errors="ignore")

                with open(output_path, "w") as f:
                    f.write(data)

                # Count lines
                n_fires = data.count("\n") - 1
                logger.info(f"  Downloaded {n_fires} fire detections from {filename}")
                downloaded = True
                break

            except urllib.error.URLError as e:
                logger.warning(f"  Failed: {e}")
                continue

        if not downloaded:
            # Provide instructions for API key access
            logger.warning("Could not download open NASA FIRMS data.")
            logger.warning("For full access, get a free API key from:")
            logger.warning("  https://firms.modaps.eosdis.nasa.gov/api/")

            instructions_path = self.data_path / "DOWNLOAD_INSTRUCTIONS.txt"
            with open(instructions_path, "w") as f:
                f.write("NASA FIRMS Fire Data Download Instructions\n")
                f.write("=" * 50 + "\n\n")
                f.write("Option 1: Open Data (limited)\n")
                f.write(f"  Visit: {self.OPEN_DATA_URL}\n")
                f.write("  Download CSV files and place in this directory\n\n")
                f.write("Option 2: API Access (full, free registration)\n")
                f.write("  1. Register at: https://firms.modaps.eosdis.nasa.gov/api/\n")
                f.write("  2. Get your MAP_KEY\n")
                f.write("  3. Set api_key in preprocessing config\n")

            return False

        return True

    def _load_raw(self) -> tuple[np.ndarray, np.ndarray]:
        """Load REAL NASA FIRMS fire detection data."""
        csv_files = list(self.data_path.glob("firms_*.csv"))

        if not csv_files:
            raise FileNotFoundError(
                f"NASA FIRMS data not found in {self.data_path}. "
                "Run with download=True to fetch real fire detection data."
            )

        logger.info(f"Loading REAL NASA FIRMS fire data from {len(csv_files)} files...")

        all_features = []
        all_labels = []

        for csv_file in csv_files:
            features, labels = self._parse_firms_csv(csv_file)
            all_features.extend(features)
            all_labels.extend(labels)

        features = np.array(all_features, dtype=np.float32)
        labels = np.array(all_labels, dtype=np.int64)

        logger.info(f"Loaded {len(features)} REAL fire detections")
        logger.info(f"  High confidence fires: {labels.sum()}")

        return features, labels

    def _parse_firms_csv(self, filepath) -> tuple[list, list]:
        """Parse NASA FIRMS CSV file."""
        import csv

        features = []
        labels = []

        # Satellite code mapping
        satellite_map = {"Terra": 0, "Aqua": 1, "N": 2, "1": 3}

        with open(filepath, newline="", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)

            for row in reader:
                try:
                    # Parse date as days since epoch
                    acq_date = row.get("acq_date", "2020-01-01")
                    try:
                        from datetime import datetime

                        date_val = (
                            datetime.strptime(acq_date, "%Y-%m-%d") - datetime(2020, 1, 1)
                        ).days
                    except ValueError:
                        date_val = 0

                    # Parse time as minutes since midnight
                    acq_time = row.get("acq_time", "0")
                    try:
                        time_val = (
                            int(acq_time[:2]) * 60 + int(acq_time[2:]) if len(acq_time) >= 4 else 0
                        )
                    except ValueError:
                        time_val = 0

                    feature_row = [
                        float(row.get("latitude", 0) or 0),
                        float(row.get("longitude", 0) or 0),
                        float(row.get("brightness", 0) or 0),
                        float(row.get("scan", 1) or 1),
                        float(row.get("track", 1) or 1),
                        date_val,
                        time_val,
                        satellite_map.get(row.get("satellite", ""), 0),
                        float(row.get("confidence", 0) or 0),
                        float(row.get("version", 6.1) or 6.1),
                        float(row.get("bright_t31", 0) or 0),
                        float(row.get("frp", 0) or 0),
                        1 if row.get("daynight", "D") == "D" else 0,
                    ]

                    features.append(feature_row)

                    # Label: high confidence fire (confidence >= 80 or high FRP)
                    confidence = float(row.get("confidence", 0) or 0)
                    frp = float(row.get("frp", 0) or 0)
                    is_significant = confidence >= 80 or frp >= 50
                    labels.append(1 if is_significant else 0)

                except (ValueError, KeyError) as e:
                    continue

        return features, labels

    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """Preprocess wildfire features."""
        data = np.nan_to_num(data, nan=0.0)
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        return data.astype(np.float32)

    def get_dataset_info(self) -> dict:
        """Get information about the loaded dataset."""
        return {
            "name": "NASA FIRMS Active Fire Detection",
            "type": "REAL DATA (Satellite)",
            "source": self.DATASET_URL,
            "features": len(self.FEATURE_NAMES),
            "citation": self.CITATION,
        }


# Register environmental loaders
DatasetRegistry.register("earthquake", USGSEarthquakeLoader)
DatasetRegistry.register("weather", NOAAWeatherLoader)
DatasetRegistry.register("wildfire", WildfireDataLoader)
