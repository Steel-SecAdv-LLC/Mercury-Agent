"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

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
import re
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
        """SETI loader is deprecated — use NASAExoplanetLoader instead."""
        raise DataSourceUnavailableError(
            loader_name="SETI",
            reason=(
                "SETI loader has been deprecated. No auth-free real SETI signal data source exists. "
                "Use NASAExoplanetLoader instead for space anomaly detection."
            ),
        )

    def _create_synthetic_seti(self) -> bool:
        """Create synthetic SETI-like signal data."""
        rng = np.random.default_rng(self.config.random_seed)
        n_samples = self.config.max_samples or 5000

        features = []
        labels = []

        for i in range(n_samples):
            # Create frequency-time spectrogram
            spectrogram = self._generate_signal(i % len(self.SIGNAL_CLASSES), rng=rng)
            features.append(spectrogram.flatten())

            # 0 = noise (normal), 1 = potential signal (anomaly)
            label = 0 if (i % len(self.SIGNAL_CLASSES)) == 3 else 1  # noise class
            labels.append(label)

        features = np.array(features, dtype=np.float32)  # type: ignore[assignment, unused-ignore]
        labels = np.array(labels, dtype=np.int64)  # type: ignore[assignment, unused-ignore]

        save_path = self.data_path / "synthetic_seti.npz"
        np.savez_compressed(save_path, features=features, labels=labels)

        logger.info(f"Generated {n_samples} SETI signal samples")
        return True

    def _generate_signal(
        self,
        signal_type: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray[Any, Any]:
        """
        Generate a synthetic SETI signal spectrogram.

        Args:
            signal_type: SETI signal class index.
            rng: Optional caller-supplied ``Generator``.  ``None``
                creates a fresh per-call ``default_rng()`` so this
                helper never consumes the legacy global ``np.random``
                state.
        """
        if rng is None:
            rng = np.random.default_rng()
        # Create base noise floor
        spectrogram = rng.normal(0, 0.1, (self.frequency_bins, self.signal_length))

        if signal_type == 0:  # squiggle
            # Variable frequency signal
            t = np.linspace(0, self.signal_length, self.signal_length)
            for i in range(self.signal_length):
                f_idx = int(
                    self.frequency_bins / 2 + 30 * np.sin(2 * np.pi * i / self.signal_length)
                )
                if 0 <= f_idx < self.frequency_bins:
                    spectrogram[f_idx, i] += float(rng.uniform(0.5, 1.0))

        elif signal_type == 1:  # narrowband
            # Single frequency line
            f_center = rng.integers(50, self.frequency_bins - 50)
            amplitude = float(rng.uniform(0.5, 1.0))
            spectrogram[f_center - 1 : f_center + 2, :] += amplitude

        elif signal_type == 2:  # narrowbanddrd (with doppler drift)
            # Drifting narrowband
            f_start = rng.integers(50, self.frequency_bins - 100)
            drift_rate = float(rng.uniform(-0.5, 0.5))
            for t_idx in range(self.signal_length):
                f = int(f_start + drift_rate * t_idx)
                if 0 <= f < self.frequency_bins:
                    spectrogram[f, t_idx] += float(rng.uniform(0.5, 1.0))

        elif signal_type == 3:  # noise
            # Just background noise - no additional signal
            pass

        elif signal_type == 4:  # squarepulsednarrowband
            # Pulsed signal
            f_center = rng.integers(50, self.frequency_bins - 50)
            pulse_period = rng.integers(20, 50)
            amplitude = float(rng.uniform(0.5, 1.0))
            for t_idx in range(0, self.signal_length, pulse_period):
                end_t = min(t_idx + pulse_period // 2, self.signal_length)
                spectrogram[f_center - 1 : f_center + 2, t_idx:end_t] += amplitude

        elif signal_type == 5:  # combined
            # Squiggle + pulsed
            spectrogram = self._generate_signal(0, rng=rng)
            spectrogram += self._generate_signal(4, rng=rng) * 0.5

        elif signal_type == 6:  # brightpixel
            # Artifact - single bright pixel
            f = rng.integers(0, self.frequency_bins)
            t = rng.integers(0, self.signal_length)
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

    Note:
        This dataset does not include ground-truth anomaly labels.
        Use for unsupervised deployment only.
    """

    DATASET_NAME = "exoplanet"
    DATASET_URL = "https://exoplanetarchive.ipac.caltech.edu/"
    LICENSE = "Public Domain"
    CITATION = """NASA Exoplanet Archive, which is operated by the California Institute
    of Technology, under contract with NASA under the Exoplanet Exploration Program."""
    REQUIRES_CREDENTIALS = False

    # NASA Exoplanet Archive TAP endpoint (via TrustedEndpoints for SSRF prevention)
    NASA_TAP_URL = TrustedEndpoints.NASA_EXOPLANET_TAP

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
        """
        Download real exoplanet data from NASA Archive.

        Returns:
            True if download successful, False otherwise.
        """
        if self._download_from_nasa_tap():
            return True

        if ALLOW_SYNTHETIC:
            check_synthetic_allowed("NASAExoplanet", "NASA Exoplanet Archive failed")
            return self._create_synthetic_exoplanet()
        raise DataSourceUnavailableError(
            loader_name="NASAExoplanet",
            source_url="https://exoplanetarchive.ipac.caltech.edu/TAP/",
            reason="NASA Exoplanet Archive API failed",
        )

    def _download_from_nasa_tap(self) -> bool:
        """Download exoplanet data from NASA TAP service."""
        import urllib.parse

        dataset_dir = self.data_path
        dataset_dir.mkdir(parents=True, exist_ok=True)
        cache_file = dataset_dir / "nasa_exoplanet_real.npz"

        if cache_file.exists():
            logger.info(f"NASA exoplanet data already cached at {cache_file}")
            self._is_real_data = True
            return True

        try:
            # Build TAP/ADQL query.
            # B608 SAFETY CONTRACT: ``columns`` comes from
            # ``self.TAP_COLUMNS.values()`` — a class-constant column
            # list defined in this module, never user input.  ``limit``
            # is an integer clamped by
            # ``min(self.config.max_samples or 5000, 5000)``.  No
            # caller-controlled string flows into this f-string, so the
            # bandit ``hardcoded_sql_expressions`` finding here is a
            # static-analysis false positive (mirrored by the ``S608``
            # ruff lift in ``[tool.ruff.lint.per-file-ignores]``).
            # B608 SAFETY CONTRACT:
            #   * ``columns`` joined from ``self.TAP_COLUMNS.values()`` (a class
            #     constant dict defined in this module). Each value is also
            #     validated below to be a bare identifier so it cannot escape
            #     the f-string.
            #   * ``limit`` is forced through ``int(...)`` and clamped to
            #     ``[1, 5000]``; a non-integer max_samples would raise here
            #     rather than reach the query string.
            for _col in self.TAP_COLUMNS.values():
                if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", _col):
                    raise RuntimeError(f"Unsafe TAP column literal: {_col!r}")
            columns = ",".join(self.TAP_COLUMNS.values())
            limit = max(1, min(int(self.config.max_samples or 5000), 5000))
            query = f"select top {limit} {columns} from ps where pl_rade is not null"  # noqa: S608  # nosec B608 - columns are class-constant identifiers (regex-checked); limit is int-coerced and clamped

            params = {
                "query": query,
                "format": "json",
            }

            url = f"{self.NASA_TAP_URL}?{urllib.parse.urlencode(params)}"
            logger.info("Downloading exoplanet data from NASA Exoplanet Archive...")

            TrustedEndpoints.validate_url(self.NASA_TAP_URL)
            content = http_get_with_retry(url, timeout=60)
            data = json.loads(content.decode("utf-8"))

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
        """
        Process TAP query results.

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
        rng = np.random.default_rng(self.config.random_seed)
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
                "stellar_temp": rng.normal(5500, 800),  # Kelvin
                "transit_depth": rng.exponential(0.001),  # fraction
                "transit_duration": np.random.lognormal(1, 0.5),  # hours
                "insolation_flux": np.random.lognormal(0, 1),  # Earth flux
                "equilibrium_temp": rng.normal(500, 300),  # Kelvin
                "eccentricity": rng.beta(1, 5),  # 0-1
                "semi_major_axis": np.random.lognormal(-0.5, 1),  # AU
                "inclination": rng.uniform(80, 90),  # degrees
            }
            # Mass-radius relation (Chen & Kipping 2017,
            # arXiv:1603.08614): for sub-Neptunian bodies M ∝ R^3.7
            # is the high-end of the empirical scatter, while gas
            # giants flatten near M ∝ R^0.6.  We use a piecewise
            # power law in Earth-mass / Earth-radius units and add
            # log-normal scatter so the synthetic mass distribution
            # is consistent with NASA Exoplanet Archive ``pl_bmasse``
            # (the column ``TAP_COLUMNS["planet_mass"]`` queries).
            radius_earths = params["planet_radius"]
            if radius_earths < 1.5:
                mass_mean = radius_earths**3.7
            else:
                mass_mean = (1.5**3.7) * (radius_earths / 1.5) ** 0.6
            params["planet_mass"] = float(mass_mean * np.random.lognormal(0.0, 0.25))

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

        features = np.array(features, dtype=np.float32)  # type: ignore[assignment, unused-ignore]
        labels = np.array(labels, dtype=np.int64)  # type: ignore[assignment, unused-ignore]

        save_path = self.data_path / "synthetic_exoplanet.npz"
        np.savez_compressed(save_path, features=features, labels=labels)

        logger.info(f"Generated {n_samples} exoplanet samples, {labels.sum()} anomalies")  # type: ignore[attr-defined, unused-ignore]
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

        # Fall back to synthetic (only if allowed)
        synthetic_path = self.data_path / "synthetic_exoplanet.npz"
        if synthetic_path.exists() and ALLOW_SYNTHETIC:
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

    Note:
        This dataset does not include ground-truth anomaly labels.
        Use for unsupervised deployment only.
    """

    DATASET_NAME = "solar"
    DATASET_URL = "https://services.swpc.noaa.gov/"
    LICENSE = "Public Domain (NOAA)"
    CITATION = """NOAA Space Weather Prediction Center. https://www.swpc.noaa.gov/"""
    REQUIRES_CREDENTIALS = False

    # NOAA SWPC JSON data endpoints (via TrustedEndpoints for SSRF prevention)
    SWPC_URLS = {
        "xrays": TrustedEndpoints.NOAA_SWPC_XRAYS,
        "protons": TrustedEndpoints.NOAA_SWPC_PROTONS,
        "kp": TrustedEndpoints.NOAA_SWPC_KP_PRODUCTS,
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
        """
        Download real solar data from NOAA SWPC.

        Returns:
            True if download successful, False otherwise.
        """
        if self._download_from_swpc():
            return True

        if ALLOW_SYNTHETIC:
            check_synthetic_allowed("SolarDynamics", "NOAA SWPC download failed")
            return self._create_synthetic_solar()
        raise DataSourceUnavailableError(
            loader_name="SolarDynamics",
            source_url="https://services.swpc.noaa.gov/",
            reason="NOAA SWPC download failed",
        )

    def _download_from_swpc(self) -> bool:
        """Download solar activity data from NOAA SWPC."""
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
            logger.info("Downloading solar X-ray data from NOAA SWPC...")

            TrustedEndpoints.validate_url(url)
            content = http_get_with_retry(url, timeout=60)
            data = json.loads(content.decode("utf-8"))

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
        """
        Process NOAA SWPC X-ray flux data.

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

        # Label solar flare events based on X-ray flux.
        # Primary: M-class threshold (1e-5 W/m²) or higher.
        # Fallback: if the observation window has no M-class events (solar quiet),
        # use statistical outlier detection (>3 sigma above mean) so the dataset
        # still contains labeled anomalies for detector evaluation.
        labels = (features[:, 0] > 1e-5).astype(np.int64)  # M-class or higher

        if not np.any(labels == 1):
            # Solar quiet period — use statistical outlier detection
            xray = features[:, 0]
            nonzero = xray[xray > 0]
            if len(nonzero) > 0:
                mu = np.mean(nonzero)
                sigma = np.std(nonzero)
                if sigma > 0:
                    labels = (xray > mu + 3 * sigma).astype(np.int64)
                    logger.info(
                        "Solar quiet period: using 3-sigma outlier labels "
                        "(threshold=%.2e, n_anomalies=%d)",
                        mu + 3 * sigma,
                        labels.sum(),
                    )

        # Apply max_samples limit with stratified sampling to ensure storm events
        if self.config.max_samples and len(features) > self.config.max_samples:
            rng = np.random.default_rng(self.config.random_seed)

            # Stratified sampling: ensure we get some storm events
            storm_indices = np.where(labels == 1)[0]
            normal_indices = np.where(labels == 0)[0]

            # Target ~20% storm events (or all available if fewer)
            n_storms = min(len(storm_indices), max(1, self.config.max_samples // 5))
            n_normal = self.config.max_samples - n_storms

            # Sample from each class
            if len(storm_indices) > 0 and n_storms > 0:
                storm_sample = rng.choice(
                    storm_indices, min(n_storms, len(storm_indices)), replace=False
                )
            else:
                storm_sample = np.array([], dtype=np.int64)

            normal_sample = rng.choice(
                normal_indices, min(n_normal, len(normal_indices)), replace=False
            )

            # Combine and shuffle
            indices = np.concatenate([storm_sample, normal_sample])
            rng.shuffle(indices)

            features = features[indices]
            labels = labels[indices]

        return features, labels

    def _create_synthetic_solar(self) -> bool:
        """Create synthetic solar activity data."""
        rng = np.random.default_rng(self.config.random_seed)
        n_samples = self.config.max_samples or 10000

        # Simulate solar cycle variations
        t = np.linspace(0, 4 * np.pi, n_samples)  # ~2 solar cycles
        cycle = 0.5 + 0.5 * np.sin(t)

        # Generate X-ray flux data matching FEATURE_NAMES (xray_short, xray_long)
        xray_short = rng.exponential(1e-7, n_samples) * (1 + cycle)
        xray_long = rng.exponential(1e-6, n_samples) * (1 + cycle)

        # Additional parameters for storm detection
        kp_index = rng.integers(0, 9, n_samples).astype(float)
        dst_index = rng.normal(-20, 30, n_samples)
        proton_flux_100mev = rng.exponential(0.01, n_samples) * (1 + cycle)

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

        # Fall back to synthetic (only if allowed)
        synthetic_path = self.data_path / "synthetic_solar.npz"
        if synthetic_path.exists() and ALLOW_SYNTHETIC:
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
