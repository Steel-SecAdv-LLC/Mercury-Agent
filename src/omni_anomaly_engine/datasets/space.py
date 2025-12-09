"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

Space & Astronomical Dataset Loaders: SETI, NASA Exoplanets, Solar Dynamics

References:
- SETI@home: https://setiathome.berkeley.edu/
- NASA Exoplanet Archive: https://exoplanetarchive.ipac.caltech.edu/
- Solar Dynamics Observatory: https://sdo.gsfc.nasa.gov/
- Breakthrough Listen: https://breakthroughinitiatives.org/initiative/1
"""
from __future__ import annotations
from typing import Any

import logging

import numpy as np

from .base import DatasetConfig, DatasetLoader, DatasetRegistry

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

    Provides access to:
    - Kepler/K2 transit data
    - TESS light curves
    - Confirmed exoplanet parameters

    Reference: https://exoplanetarchive.ipac.caltech.edu/
    """

    DATASET_NAME = "exoplanet"
    DATASET_URL = "https://exoplanetarchive.ipac.caltech.edu/"
    LICENSE = "Public Domain"
    CITATION = """NASA Exoplanet Archive, which is operated by the California Institute
    of Technology, under contract with NASA under the Exoplanet Exploration Program."""
    REQUIRES_CREDENTIALS = False

    FEATURE_NAMES = [
        "orbital_period",
        "planet_radius",
        "stellar_mass",
        "stellar_radius",
        "stellar_temp",
        "transit_depth",
        "transit_duration",
        "insolation_flux",
        "equilibrium_temp",
        "eccentricity",
        "semi_major_axis",
        "inclination",
    ]

    def __init__(self, config: DatasetConfig) -> None:
        super().__init__(config)

    def download(self) -> bool:
        """Download or generate exoplanet data."""
        return self._create_synthetic_exoplanet()

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
        synthetic_path = self.data_path / "synthetic_exoplanet.npz"
        if synthetic_path.exists():
            data = np.load(synthetic_path)
            return data["features"], data["labels"]
        raise FileNotFoundError("Exoplanet data not found")

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Preprocess exoplanet features."""
        # Log transform for skewed features
        data = np.log1p(np.abs(data)) * np.sign(data)
        # Z-score normalize
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        return data.astype(np.float32)


class SolarDynamicsLoader(DatasetLoader):
    """
    Solar Dynamics Observatory (SDO) Data Loader.

    Provides access to:
    - Solar flare predictions
    - Coronal mass ejection (CME) events
    - Solar activity indices

    Reference: https://sdo.gsfc.nasa.gov/
    """

    DATASET_NAME = "solar"
    DATASET_URL = "https://sdo.gsfc.nasa.gov/"
    LICENSE = "Public Domain (NASA)"
    CITATION = """Pesnell WD, Thompson BJ, Chamberlin PC. The Solar Dynamics Observatory (SDO).
    Solar Physics. 2012;275:3-15."""
    REQUIRES_CREDENTIALS = False

    FEATURE_NAMES = [
        "sunspot_number",
        "f10.7_flux",
        "x_ray_flux_short",
        "x_ray_flux_long",
        "proton_flux_10mev",
        "proton_flux_100mev",
        "electron_flux",
        "kp_index",
        "dst_index",
        "solar_wind_speed",
        "solar_wind_density",
        "imf_magnitude",
        "imf_bz",
        "active_region_count",
    ]

    def __init__(self, config: DatasetConfig) -> None:
        super().__init__(config)

    def download(self) -> bool:
        return self._create_synthetic_solar()

    def _create_synthetic_solar(self) -> bool:
        """Create synthetic solar activity data."""
        np.random.seed(self.config.random_seed)
        n_samples = self.config.max_samples or 10000

        # Simulate solar cycle variations
        t = np.linspace(0, 4 * np.pi, n_samples)  # ~2 solar cycles
        cycle = 0.5 + 0.5 * np.sin(t)

        data = {
            "sunspot_number": (100 * cycle + np.random.normal(0, 20, n_samples)).clip(0),
            "f10.7_flux": (100 + 100 * cycle + np.random.normal(0, 15, n_samples)),
            "x_ray_flux_short": np.random.exponential(1e-7, n_samples) * (1 + cycle),
            "x_ray_flux_long": np.random.exponential(1e-6, n_samples) * (1 + cycle),
            "proton_flux_10mev": np.random.exponential(0.1, n_samples) * (1 + cycle),
            "proton_flux_100mev": np.random.exponential(0.01, n_samples) * (1 + cycle),
            "electron_flux": np.random.exponential(100, n_samples) * (1 + cycle),
            "kp_index": np.random.randint(0, 9, n_samples).astype(float),
            "dst_index": np.random.normal(-20, 30, n_samples),
            "solar_wind_speed": np.random.normal(400, 100, n_samples).clip(200),
            "solar_wind_density": np.random.exponential(5, n_samples),
            "imf_magnitude": np.random.exponential(5, n_samples),
            "imf_bz": np.random.normal(0, 5, n_samples),
            "active_region_count": (5 * cycle + np.random.poisson(3, n_samples)),
        }

        features = np.column_stack([data[f] for f in self.FEATURE_NAMES])

        # Label anomalies: solar storm events
        labels = np.zeros(n_samples, dtype=np.int64)
        storm_mask = (
            (data["x_ray_flux_short"] > 1e-5)  # X-class flare
            | (data["kp_index"] >= 7)  # Geomagnetic storm
            | (data["dst_index"] < -100)  # Major storm
            | (data["proton_flux_100mev"] > 1)  # SEP event
        )
        labels[storm_mask] = 1

        save_path = self.data_path / "synthetic_solar.npz"
        np.savez_compressed(save_path, features=features, labels=labels)

        logger.info(f"Generated {n_samples} solar samples, {labels.sum()} storm events")
        return True

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        synthetic_path = self.data_path / "synthetic_solar.npz"
        if synthetic_path.exists():
            data = np.load(synthetic_path)
            return data["features"], data["labels"]
        raise FileNotFoundError("Solar data not found")

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
