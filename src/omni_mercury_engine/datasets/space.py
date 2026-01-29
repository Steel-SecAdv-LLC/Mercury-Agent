"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

Space & Astronomical Dataset Loaders: SETI, NASA Exoplanets, Solar Dynamics

References:
- SETI@home: https://setiathome.berkeley.edu/
- NASA Exoplanet Archive: https://exoplanetarchive.ipac.caltech.edu/
- Solar Dynamics Observatory: https://sdo.gsfc.nasa.gov/
- Breakthrough Listen: https://breakthroughinitiatives.org/initiative/1
"""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.parse import urlparse

import numpy as np


try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False

from .base import DatasetConfig, DatasetLoader, DatasetRegistry


logger = logging.getLogger(__name__)


# Allowlist of trusted domains for SSRF protection
_ALLOWED_DOMAINS: frozenset[str] = frozenset(
    [
        "exoplanetarchive.ipac.caltech.edu",
        "services.swpc.noaa.gov",
    ]
)


def _sanitize_url(url: str) -> str:
    """Validate and sanitize URL to prevent SSRF attacks.

    Validates that:
    1. URL uses HTTPS scheme
    2. Domain is in the allowlist of trusted data sources

    Args:
        url: URL to validate

    Returns:
        The validated URL if it passes all security checks

    Raises:
        ValueError: If URL fails security validation
    """
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"URL must use HTTPS scheme, got: {parsed.scheme}")
    if parsed.netloc not in _ALLOWED_DOMAINS:
        raise ValueError(f"Domain not in allowlist: {parsed.netloc}")
    return url


class SETILoader(DatasetLoader):
    """
    SETI Signal Dataset Loader.

    Provides access to radio telescope signal data for:
    - Technosignature detection
    - Radio frequency interference (RFI) classification
    - Anomalous signal identification

    Based on:
    - SETI@home data analysis
    - Breakthrough Listen observations
    - Simulated SETI signal dataset (Kaggle)

    Citation:
        Anderson DP, et al. SETI@home: An Experiment in Public-Resource Computing.
        Communications of the ACM, 2002.
    """

    DATASET_NAME = "seti"
    DATASET_URL = "https://seti.berkeley.edu/"
    LICENSE = "Creative Commons Attribution 4.0"
    CITATION = """Anderson DP, Werthimer D, Cobb J. SETI@home: An Experiment in
    Public-Resource Computing. Communications of the ACM. 2002;45(11):56-61."""
    REQUIRES_CREDENTIALS = False

    # Signal classes from SETI classification
    SIGNAL_CLASSES = [
        "squiggle",  # Variable frequency signals
        "narrowband",  # Single frequency
        "narrowbanddrd",  # Narrowband with doppler drift
        "noise",  # Background noise
        "squarepulsednarrowband",  # Pulsed signals
        "squiggle_squarepulsednarrowband",  # Combined
        "brightpixel",  # Bright artifacts
    ]

    def __init__(self, config: DatasetConfig) -> None:
        super().__init__(config)
        self.signal_length = config.preprocessing.get("signal_length", 512)
        self.frequency_bins = config.preprocessing.get("frequency_bins", 256)

    def download(self) -> bool:
        """Generate synthetic SETI signals for development."""
        return self._create_synthetic_seti()

    def _create_synthetic_seti(self) -> bool:
        """Create synthetic SETI-like signal data."""
        np.random.seed(self.config.random_seed)
        n_samples = self.config.max_samples or 5000

        features = []
        labels = []

        for i in range(n_samples):
            # Create frequency-time spectrogram
            spectrogram = self._generate_signal(i % len(self.SIGNAL_CLASSES))
            features.append(spectrogram.flatten())

            # 0 = noise (normal), 1 = potential signal (anomaly)
            label = 0 if (i % len(self.SIGNAL_CLASSES)) == 3 else 1  # noise class
            labels.append(label)

        features = np.array(features, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        save_path = self.data_path / "synthetic_seti.npz"
        np.savez_compressed(save_path, features=features, labels=labels)

        logger.info(f"Generated {n_samples} SETI signal samples")
        return True

    def _generate_signal(self, signal_type: int) -> np.ndarray[Any, Any]:
        """Generate a synthetic SETI signal spectrogram."""
        # Create base noise floor
        spectrogram = np.random.normal(0, 0.1, (self.frequency_bins, self.signal_length))

        if signal_type == 0:  # squiggle
            # Variable frequency signal
            t = np.linspace(0, self.signal_length, self.signal_length)
            for i in range(self.signal_length):
                f_idx = int(
                    self.frequency_bins / 2 + 30 * np.sin(2 * np.pi * i / self.signal_length)
                )
                if 0 <= f_idx < self.frequency_bins:
                    spectrogram[f_idx, i] += float(np.random.uniform(0.5, 1.0))

        elif signal_type == 1:  # narrowband
            # Single frequency line
            f_center = np.random.randint(50, self.frequency_bins - 50)
            amplitude = float(np.random.uniform(0.5, 1.0))
            spectrogram[f_center - 1 : f_center + 2, :] += amplitude

        elif signal_type == 2:  # narrowbanddrd (with doppler drift)
            # Drifting narrowband
            f_start = np.random.randint(50, self.frequency_bins - 100)
            drift_rate = float(np.random.uniform(-0.5, 0.5))
            for t_idx in range(self.signal_length):
                f = int(f_start + drift_rate * t_idx)
                if 0 <= f < self.frequency_bins:
                    spectrogram[f, t_idx] += float(np.random.uniform(0.5, 1.0))

        elif signal_type == 3:  # noise
            # Just background noise - no additional signal
            pass

        elif signal_type == 4:  # squarepulsednarrowband
            # Pulsed signal
            f_center = np.random.randint(50, self.frequency_bins - 50)
            pulse_period = np.random.randint(20, 50)
            amplitude = float(np.random.uniform(0.5, 1.0))
            for t_idx in range(0, self.signal_length, pulse_period):
                end_t = min(t_idx + pulse_period // 2, self.signal_length)
                spectrogram[f_center - 1 : f_center + 2, t_idx:end_t] += amplitude

        elif signal_type == 5:  # combined
            # Squiggle + pulsed
            spectrogram = self._generate_signal(0)
            spectrogram += self._generate_signal(4) * 0.5

        elif signal_type == 6:  # brightpixel
            # Artifact - single bright pixel
            f = np.random.randint(0, self.frequency_bins)
            t = np.random.randint(0, self.signal_length)
            spectrogram[f, t] = 5.0  # Very bright

        return spectrogram

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        synthetic_path = self.data_path / "synthetic_seti.npz"
        if synthetic_path.exists():
            data = np.load(synthetic_path)
            return data["features"], data["labels"]
        raise FileNotFoundError("SETI data not found")

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Preprocess SETI spectrograms."""
        # Normalize spectrograms
        data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)
        return data.astype(np.float32)


class NASAExoplanetLoader(DatasetLoader):
    """
    NASA Exoplanet Archive Data Loader.

    Downloads REAL exoplanet data from NASA Exoplanet Archive including:
    - Confirmed exoplanet parameters (radius, mass, orbital period)
    - Host star properties (mass, radius, temperature)
    - Transit parameters (depth, duration)

    Data source: https://exoplanetarchive.ipac.caltech.edu/TAP/
    License: Public Domain
    """

    DATASET_NAME = "exoplanet"
    DATASET_URL = "https://exoplanetarchive.ipac.caltech.edu/"
    LICENSE = "Public Domain"
    CITATION = """NASA Exoplanet Archive, which is operated by the California Institute
    of Technology, under contract with NASA under the Exoplanet Exploration Program."""
    REQUIRES_CREDENTIALS = False

    # NASA Exoplanet Archive TAP endpoint
    NASA_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

    FEATURE_NAMES = [
        "orbital_period",
        "planet_radius",
        "planet_mass",
        "stellar_mass",
        "stellar_radius",
        "stellar_temp",
        "eccentricity",
        "semi_major_axis",
    ]

    # TAP query column mapping
    TAP_COLUMNS = {
        "orbital_period": "pl_orbper",
        "planet_radius": "pl_rade",
        "planet_mass": "pl_bmasse",
        "stellar_mass": "st_mass",
        "stellar_radius": "st_rad",
        "stellar_temp": "st_teff",
        "eccentricity": "pl_orbeccen",
        "semi_major_axis": "pl_orbsmax",
    }

    def __init__(self, config: DatasetConfig) -> None:
        super().__init__(config)
        self._is_real_data = False

    @property
    def is_real_data(self) -> bool:
        """Return True if real data was loaded."""
        return self._is_real_data

    def download(self) -> bool:
        """Download real exoplanet data from NASA Archive.

        Returns:
            True if download successful, False otherwise.
        """
        if self._download_from_nasa_tap():
            return True

        logger.warning("NASA Exoplanet Archive failed, falling back to SYNTHETIC data.")
        return self._create_synthetic_exoplanet()

    def _download_from_nasa_tap(self) -> bool:
        """Download exoplanet data from NASA TAP service."""
        import urllib.parse
        import urllib.request

        dataset_dir = self.data_path
        dataset_dir.mkdir(parents=True, exist_ok=True)
        cache_file = dataset_dir / "nasa_exoplanet_real.npz"

        if cache_file.exists():
            logger.info(f"NASA exoplanet data already cached at {cache_file}")
            self._is_real_data = True
            return True

        try:
            # Build TAP/ADQL query
            columns = ",".join(self.TAP_COLUMNS.values())
            limit = min(self.config.max_samples or 5000, 5000)
            query = f"select top {limit} {columns} from ps where pl_rade is not null"  # noqa: S608  # nosec B608

            params = {
                "query": query,
                "format": "json",
            }

            url = f"{self.NASA_TAP_URL}?{urllib.parse.urlencode(params)}"

            # Sanitize URL to prevent SSRF attacks - raises ValueError if invalid
            sanitized_url = _sanitize_url(url)

            logger.info("Downloading exoplanet data from NASA Exoplanet Archive...")

            req = urllib.request.Request(  # noqa: S310
                sanitized_url, headers={"User-Agent": "Mozilla/5.0 Mercury-Agent/1.0"}
            )
            with urllib.request.urlopen(req, timeout=60) as response:  # noqa: S310  # nosec B310
                data = json.loads(response.read().decode("utf-8"))

            # Parse TAP response
            records = data.get("data", []) if isinstance(data, dict) else data
            if not records:
                logger.warning("No exoplanet data returned from NASA Archive")
                return False

            logger.info(f"Downloaded {len(records)} exoplanet records")

            # Convert to features array
            features, labels = self._process_tap_data(records)

            # Save to cache
            np.savez_compressed(cache_file, features=features, labels=labels)
            self._is_real_data = True

            logger.info(
                f"NASA exoplanet data loaded: {len(features)} samples, "
                f"{labels.sum()} anomalous planets (is_real_data=True)"
            )
            return True

        except Exception as e:
            logger.warning(f"NASA TAP download failed: {e}")
            return False

    def _process_tap_data(self, records: list[Any]) -> tuple[np.ndarray, np.ndarray]:
        """Process TAP query results.

        Args:
            records: List of records from TAP query

        Returns:
            Tuple of (features, labels) numpy arrays
        """
        rows = []
        col_order = list(self.TAP_COLUMNS.values())

        for record in records:
            if isinstance(record, dict):
                row = [record.get(col, 0) or 0 for col in col_order]
            else:
                row = [
                    record[i] if i < len(record) and record[i] is not None else 0
                    for i in range(len(col_order))
                ]
            rows.append(row)

        features = np.array(rows, dtype=np.float32)
        features = np.nan_to_num(features, nan=0.0)

        # Label anomalous planets (very large, very close, or highly eccentric)
        # Based on columns: orbital_period, planet_radius, ..., eccentricity, semi_major_axis
        orbital_period = features[:, 0]
        planet_radius = features[:, 1]
        eccentricity = features[:, 6]

        labels = (
            (planet_radius > 15)  # Very large (Jupiter+)
            | (orbital_period < 1)  # Ultra-short period
            | (eccentricity > 0.7)  # Highly eccentric
        ).astype(np.int64)

        return features, labels

    def _create_synthetic_exoplanet(self) -> bool:
        """Create synthetic exoplanet detection data."""
        np.random.seed(self.config.random_seed)
        n_samples = self.config.max_samples or 3000

        features = []
        labels = []

        for _i in range(n_samples):
            # Generate realistic exoplanet parameters
            params = {
                "orbital_period": np.random.lognormal(2, 1.5),  # days
                "planet_radius": np.random.lognormal(0, 0.8),  # Earth radii
                "stellar_mass": np.random.lognormal(0, 0.3),  # Solar masses
                "stellar_radius": np.random.lognormal(0, 0.3),  # Solar radii
                "stellar_temp": np.random.normal(5500, 800),  # Kelvin
                "transit_depth": np.random.exponential(0.001),  # fraction
                "transit_duration": np.random.lognormal(1, 0.5),  # hours
                "insolation_flux": np.random.lognormal(0, 1),  # Earth flux
                "equilibrium_temp": np.random.normal(500, 300),  # Kelvin
                "eccentricity": np.random.beta(1, 5),  # 0-1
                "semi_major_axis": np.random.lognormal(-0.5, 1),  # AU
                "inclination": np.random.uniform(80, 90),  # degrees
            }

            feature_vec = [params[f] for f in self.FEATURE_NAMES]
            features.append(feature_vec)

            # Anomalies: unusual planet configurations
            is_anomaly = (
                params["planet_radius"] > 5  # Very large planet
                or params["orbital_period"] < 0.5  # Ultra-short period
                or params["equilibrium_temp"] > 2000  # Very hot
                or params["eccentricity"] > 0.8  # Highly eccentric
            )
            labels.append(1 if is_anomaly else 0)

        features = np.array(features, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        save_path = self.data_path / "synthetic_exoplanet.npz"
        np.savez_compressed(save_path, features=features, labels=labels)

        logger.info(f"Generated {n_samples} exoplanet samples, {labels.sum()} anomalies")
        return True

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Load exoplanet data from cache (real data first, then synthetic)."""
        # Check for real data first
        real_cache = self.data_path / "nasa_exoplanet_real.npz"
        if real_cache.exists():
            data = np.load(real_cache)
            self._is_real_data = True
            logger.info(f"Loaded REAL NASA exoplanet data from {real_cache}")
            return data["features"], data["labels"]

        # Fall back to synthetic
        synthetic_path = self.data_path / "synthetic_exoplanet.npz"
        if synthetic_path.exists():
            data = np.load(synthetic_path)
            self._is_real_data = False
            logger.info("Loaded SYNTHETIC exoplanet data (is_real_data=False)")
            return data["features"], data["labels"]

        raise FileNotFoundError("Exoplanet data not found. Run download() first.")

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Preprocess exoplanet features."""
        # Log transform for skewed features
        data = np.log1p(np.abs(data)) * np.sign(data)
        # Z-score normalize
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        return data.astype(np.float32)


class SolarDynamicsLoader(DatasetLoader):
    """
    NOAA Space Weather Prediction Center Data Loader.

    Downloads REAL solar activity data from NOAA SWPC including:
    - GOES X-ray flux (solar flares)
    - Proton flux (radiation storms)
    - Geomagnetic indices (Kp, Dst)

    Data source: https://services.swpc.noaa.gov/
    License: Public Domain (NOAA)
    """

    DATASET_NAME = "solar"
    DATASET_URL = "https://services.swpc.noaa.gov/"
    LICENSE = "Public Domain (NOAA)"
    CITATION = """NOAA Space Weather Prediction Center. https://www.swpc.noaa.gov/"""
    REQUIRES_CREDENTIALS = False

    # NOAA SWPC JSON data endpoints
    SWPC_URLS = {
        "xrays": "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json",
        "protons": "https://services.swpc.noaa.gov/json/goes/primary/integral-protons-1-day.json",
        "kp": "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json",
    }

    FEATURE_NAMES = [
        "xray_short",
        "xray_long",
    ]

    def __init__(self, config: DatasetConfig) -> None:
        super().__init__(config)
        self._is_real_data = False

    @property
    def is_real_data(self) -> bool:
        """Return True if real data was loaded."""
        return self._is_real_data

    def download(self) -> bool:
        """Download real solar data from NOAA SWPC.

        Returns:
            True if download successful, False otherwise.
        """
        if self._download_from_swpc():
            return True

        logger.warning("NOAA SWPC download failed, falling back to SYNTHETIC data.")
        return self._create_synthetic_solar()

    def _download_from_swpc(self) -> bool:
        """Download solar activity data from NOAA SWPC."""
        import urllib.request

        dataset_dir = self.data_path
        dataset_dir.mkdir(parents=True, exist_ok=True)
        cache_file = dataset_dir / "noaa_solar_real.npz"

        if cache_file.exists():
            logger.info(f"NOAA solar data already cached at {cache_file}")
            self._is_real_data = True
            return True

        try:
            # Download X-ray data (primary solar activity indicator)
            url = self.SWPC_URLS["xrays"]

            # Sanitize URL to prevent SSRF attacks - raises ValueError if invalid
            sanitized_url = _sanitize_url(url)

            logger.info("Downloading solar X-ray data from NOAA SWPC...")

            req = urllib.request.Request(  # noqa: S310
                sanitized_url, headers={"User-Agent": "Mozilla/5.0 Mercury-Agent/1.0"}
            )
            with urllib.request.urlopen(req, timeout=60) as response:  # noqa: S310  # nosec B310
                data = json.loads(response.read().decode("utf-8"))

            if not data:
                logger.warning("No solar data returned from NOAA SWPC")
                return False

            logger.info(f"Downloaded {len(data)} X-ray flux records")

            # Process the data
            features, labels = self._process_swpc_data(data)

            # Save to cache
            np.savez_compressed(cache_file, features=features, labels=labels)
            self._is_real_data = True

            logger.info(
                f"NOAA solar data loaded: {len(features)} samples, "
                f"{labels.sum()} flare events (is_real_data=True)"
            )
            return True

        except Exception as e:
            logger.warning(f"NOAA SWPC download failed: {e}")
            return False

    def _process_swpc_data(self, data: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        """Process NOAA SWPC X-ray flux data.

        Args:
            data: List of X-ray flux records

        Returns:
            Tuple of (features, labels) numpy arrays
        """
        rows = []

        for record in data:
            # Extract flux values (short and long wavelength)
            xray_short = record.get("flux", 0) or 0
            # Some records have both, some have one
            xray_long = record.get("observed_flux", xray_short) or xray_short

            rows.append([xray_short, xray_long])

        features = np.array(rows, dtype=np.float32)
        features = np.nan_to_num(features, nan=0.0)

        # Label solar flare events based on X-ray flux
        # C-class: 1e-6, M-class: 1e-5, X-class: 1e-4
        labels = (features[:, 0] > 1e-5).astype(np.int64)  # M-class or higher

        # Apply max_samples limit with stratified sampling to ensure storm events
        if self.config.max_samples and len(features) > self.config.max_samples:
            np.random.seed(self.config.random_seed)

            # Stratified sampling: ensure we get some storm events
            storm_indices = np.where(labels == 1)[0]
            normal_indices = np.where(labels == 0)[0]

            # Target ~20% storm events (or all available if fewer)
            n_storms = min(len(storm_indices), max(1, self.config.max_samples // 5))
            n_normal = self.config.max_samples - n_storms

            # Sample from each class
            if len(storm_indices) > 0 and n_storms > 0:
                storm_sample = np.random.choice(
                    storm_indices, min(n_storms, len(storm_indices)), replace=False
                )
            else:
                storm_sample = np.array([], dtype=np.int64)

            normal_sample = np.random.choice(
                normal_indices, min(n_normal, len(normal_indices)), replace=False
            )

            # Combine and shuffle
            indices = np.concatenate([storm_sample, normal_sample])
            np.random.shuffle(indices)

            features = features[indices]
            labels = labels[indices]

        return features, labels

    def _create_synthetic_solar(self) -> bool:
        """Create synthetic solar activity data."""
        np.random.seed(self.config.random_seed)
        n_samples = self.config.max_samples or 10000

        # Simulate solar cycle variations
        t = np.linspace(0, 4 * np.pi, n_samples)  # ~2 solar cycles
        cycle = 0.5 + 0.5 * np.sin(t)

        # Generate X-ray flux data matching FEATURE_NAMES (xray_short, xray_long)
        xray_short = np.random.exponential(1e-7, n_samples) * (1 + cycle)
        xray_long = np.random.exponential(1e-6, n_samples) * (1 + cycle)

        # Additional parameters for storm detection
        kp_index = np.random.randint(0, 9, n_samples).astype(float)
        dst_index = np.random.normal(-20, 30, n_samples)
        proton_flux_100mev = np.random.exponential(0.01, n_samples) * (1 + cycle)

        data = {
            "xray_short": xray_short,
            "xray_long": xray_long,
        }

        features = np.column_stack([data[f] for f in self.FEATURE_NAMES])

        # Label anomalies: solar storm events
        # Use thresholds appropriate for the exponential distribution
        labels = np.zeros(n_samples, dtype=np.int64)
        storm_mask = (
            (xray_short > 1e-5)  # X-class flare
            | (kp_index >= 7)  # Geomagnetic storm
            | (dst_index < -100)  # Major storm
            | (proton_flux_100mev > 1)  # SEP event
        )
        labels[storm_mask] = 1

        save_path = self.data_path / "synthetic_solar.npz"
        np.savez_compressed(save_path, features=features, labels=labels)

        logger.info(f"Generated {n_samples} solar samples, {labels.sum()} storm events")
        return True

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Load solar data from cache (real data first, then synthetic)."""
        # Check for real data first
        real_cache = self.data_path / "noaa_solar_real.npz"
        if real_cache.exists():
            data = np.load(real_cache)
            self._is_real_data = True
            logger.info(f"Loaded REAL NOAA solar data from {real_cache}")
            return data["features"], data["labels"]

        # Fall back to synthetic
        synthetic_path = self.data_path / "synthetic_solar.npz"
        if synthetic_path.exists():
            data = np.load(synthetic_path)
            self._is_real_data = False
            logger.info("Loaded SYNTHETIC solar data (is_real_data=False)")
            return data["features"], data["labels"]

        raise FileNotFoundError("Solar data not found. Run download() first.")

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Preprocess solar data."""
        # Log transform for flux measurements
        data = np.log1p(np.abs(data)) * np.sign(data)
        # Z-score normalize
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        return data.astype(np.float32)


# Register space loaders
DatasetRegistry.register("seti", SETILoader)
DatasetRegistry.register("exoplanet", NASAExoplanetLoader)
DatasetRegistry.register("solar", SolarDynamicsLoader)
