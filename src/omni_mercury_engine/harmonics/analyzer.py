"""
Advanced Harmonic Analyzer for Mercury Agent.

High-level interface for 3D surface analysis and anomaly detection
using spherical harmonic decomposition.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from omni_mercury_engine.harmonics.features import (
    HarmonicFeatureExtractor,
    HarmonicSimilarity,
    RotationInvariantDescriptor,
)
from omni_mercury_engine.harmonics.transform import (
    FastSHTransform,
    HarmonicCoefficients,
    SphericalHarmonicTransform,
)

logger = logging.getLogger(__name__)


@dataclass
class HarmonicAnomalyResult:
    """Result of harmonic anomaly detection."""

    is_anomaly: bool
    anomaly_score: float
    similarity_to_reference: float
    power_spectrum_deviation: float
    complexity_deviation: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class HarmonicSignature:
    """Stored harmonic signature for reference."""

    name: str
    coefficients: HarmonicCoefficients
    descriptor: RotationInvariantDescriptor
    metadata: dict[str, Any] = field(default_factory=dict)


class HarmonicDatabase:
    """
    Database of reference harmonic signatures.

    Stores and retrieves harmonic signatures for comparison.
    """

    def __init__(self) -> None:
        """Initialize the database."""
        self._signatures: dict[str, HarmonicSignature] = {}
        self._extractor = HarmonicFeatureExtractor()
        self._similarity = HarmonicSimilarity()

    def add(
        self,
        name: str,
        coefficients: HarmonicCoefficients,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Add a signature to the database.

        Args:
            name: Unique identifier
            coefficients: Harmonic coefficients
            metadata: Optional metadata
        """
        descriptor = self._extractor.extract(coefficients)
        self._signatures[name] = HarmonicSignature(
            name=name,
            coefficients=coefficients,
            descriptor=descriptor,
            metadata=metadata or {},
        )

    def find_nearest(
        self,
        coefficients: HarmonicCoefficients,
        k: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Find k nearest signatures.

        Args:
            coefficients: Query coefficients
            k: Number of results

        Returns:
            List of (name, similarity) tuples
        """
        descriptor = self._extractor.extract(coefficients)

        similarities = []
        for name, sig in self._signatures.items():
            sim = self._similarity.compute(descriptor, sig.descriptor)
            similarities.append((name, sim))

        similarities.sort(key=lambda x: -x[1])
        return similarities[:k]

    def get(self, name: str) -> HarmonicSignature | None:
        """Get signature by name."""
        return self._signatures.get(name)

    def remove(self, name: str) -> bool:
        """Remove signature by name."""
        if name in self._signatures:
            del self._signatures[name]
            return True
        return False

    def list_all(self) -> list[str]:
        """List all signature names."""
        return list(self._signatures.keys())

    def __len__(self) -> int:
        """Number of signatures."""
        return len(self._signatures)


class AdvancedHarmonicAnalyzer:
    """
    High-level interface for spherical harmonic analysis.

    Provides methods for decomposition, feature extraction, and anomaly detection.

    Example:
        analyzer = AdvancedHarmonicAnalyzer(
            l_max=64,
            backend="numpy",
            precision="float64",
        )

        # Decompose 3D surface
        coefficients = analyzer.decompose(point_cloud)

        # Extract rotation-invariant features
        features = analyzer.extract_features(
            coefficients,
            descriptors=["power_spectrum", "bispectrum"],
        )

        # Detect anomalies
        anomalies = analyzer.detect_anomalies(
            features,
            reference_database=normal_signatures,
        )
    """

    def __init__(
        self,
        l_max: int = 32,
        backend: str = "numpy",
        precision: str = "float64",
    ) -> None:
        """
        Initialize the harmonic analyzer.

        Args:
            l_max: Maximum spherical harmonic degree
            backend: Computation backend ("numpy", "cuda", "torch", "jax")
            precision: Numerical precision ("float32", "float64")
        """
        self._l_max = l_max
        self._backend = backend
        self._precision = precision

        self._transform = SphericalHarmonicTransform(l_max, backend, precision)
        self._fast_transform = FastSHTransform(l_max)
        self._extractor = HarmonicFeatureExtractor(l_max)
        self._similarity = HarmonicSimilarity()

        self._reference_db: HarmonicDatabase | None = None
        self._mean_power_spectrum: np.ndarray | None = None
        self._std_power_spectrum: np.ndarray | None = None
        self._mean_complexity: float = 0.5
        self._std_complexity: float = 0.2

    @property
    def l_max(self) -> int:
        """Maximum spherical harmonic degree."""
        return self._l_max

    def decompose(
        self,
        point_cloud: np.ndarray,
        sampling: str = "healpix",
    ) -> HarmonicCoefficients:
        """
        Decompose 3D point cloud into spherical harmonics.

        Args:
            point_cloud: 3D point cloud (N, 3) or (N, 4) with values
            sampling: Sampling scheme

        Returns:
            HarmonicCoefficients
        """
        return self._transform.decompose(point_cloud, sampling)

    def reconstruct(
        self,
        coefficients: HarmonicCoefficients,
        n_theta: int = 64,
        n_phi: int = 128,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Reconstruct surface from coefficients.

        Args:
            coefficients: Spherical harmonic coefficients
            n_theta: Number of colatitude samples
            n_phi: Number of azimuthal samples

        Returns:
            Tuple of (theta_grid, phi_grid, values)
        """
        return self._transform.reconstruct(coefficients, n_theta, n_phi)

    def extract_features(
        self,
        coefficients: HarmonicCoefficients,
        descriptors: list[str] | None = None,
    ) -> np.ndarray:
        """
        Extract rotation-invariant features.

        Args:
            coefficients: Spherical harmonic coefficients
            descriptors: Which descriptors to include

        Returns:
            Feature vector
        """
        if descriptors is None:
            descriptors = ["power_spectrum", "bispectrum"]

        full_desc = self._extractor.extract(coefficients)

        features = []

        if "power_spectrum" in descriptors:
            features.append(full_desc.power_spectrum.to_feature_vector())

        if "bispectrum" in descriptors and full_desc.bispectrum is not None:
            features.append(full_desc.bispectrum.to_feature_vector())

        if "energy" in descriptors:
            features.append(full_desc.energy_distribution)

        if "complexity" in descriptors:
            features.append(np.array([full_desc.complexity_measure]))

        if not features:
            return full_desc.power_spectrum.to_feature_vector()

        return np.concatenate(features)

    def fit(
        self,
        point_clouds: list[np.ndarray],
        labels: list[str] | None = None,
    ) -> AdvancedHarmonicAnalyzer:
        """
        Fit analyzer on reference data.

        Args:
            point_clouds: List of reference point clouds
            labels: Optional labels for each point cloud

        Returns:
            self for method chaining
        """
        self._reference_db = HarmonicDatabase()

        all_power_spectra = []
        all_complexities = []

        for i, pc in enumerate(point_clouds):
            coeffs = self.decompose(pc)
            desc = self._extractor.extract(coeffs)

            name = labels[i] if labels else f"reference_{i}"
            self._reference_db.add(name, coeffs, {"index": i})

            all_power_spectra.append(desc.power_spectrum.spectrum)
            all_complexities.append(desc.complexity_measure)

        if all_power_spectra:
            ps_array = np.array(all_power_spectra)
            self._mean_power_spectrum = np.mean(ps_array, axis=0)
            self._std_power_spectrum = np.std(ps_array, axis=0) + 1e-10

            self._mean_complexity = np.mean(all_complexities)  # type: ignore[assignment, unused-ignore]
            self._std_complexity = np.std(all_complexities) + 1e-10  # type: ignore[assignment, unused-ignore]

        return self

    def detect_anomalies(
        self,
        point_cloud: np.ndarray,
        threshold: float = 0.5,
    ) -> HarmonicAnomalyResult:
        """
        Detect anomalies in a 3D surface.

        Args:
            point_cloud: 3D point cloud to analyze
            threshold: Anomaly threshold

        Returns:
            HarmonicAnomalyResult
        """
        coefficients = self.decompose(point_cloud)
        descriptor = self._extractor.extract(coefficients)

        similarity_score = 0.5
        if self._reference_db and len(self._reference_db) > 0:
            nearest = self._reference_db.find_nearest(coefficients, k=3)
            if nearest:
                similarity_score = np.mean([s for _, s in nearest])  # type: ignore[assignment, unused-ignore]

        ps_deviation = 0.0
        if self._mean_power_spectrum is not None:
            ps_diff = descriptor.power_spectrum.spectrum - self._mean_power_spectrum
            ps_z_scores = np.abs(ps_diff / self._std_power_spectrum)
            ps_deviation = np.mean(ps_z_scores)

        complexity_deviation = (
            abs(descriptor.complexity_measure - self._mean_complexity) / self._std_complexity
        )

        anomaly_score = (
            0.4 * (1 - similarity_score)
            + 0.4 * min(1.0, ps_deviation / 3.0)
            + 0.2 * min(1.0, complexity_deviation / 3.0)
        )

        is_anomaly = anomaly_score > threshold

        return HarmonicAnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            similarity_to_reference=similarity_score,
            power_spectrum_deviation=ps_deviation,
            complexity_deviation=complexity_deviation,
            details={
                "power_spectrum": descriptor.power_spectrum.spectrum.tolist(),
                "complexity": descriptor.complexity_measure,
                "threshold": threshold,
            },
        )

    def batch_detect(
        self,
        point_clouds: list[np.ndarray],
        threshold: float = 0.5,
    ) -> list[HarmonicAnomalyResult]:
        """
        Detect anomalies in batch.

        Args:
            point_clouds: List of point clouds
            threshold: Anomaly threshold

        Returns:
            List of anomaly results
        """
        return [self.detect_anomalies(pc, threshold) for pc in point_clouds]

    def compare(
        self,
        coefficients1: HarmonicCoefficients,
        coefficients2: HarmonicCoefficients,
    ) -> float:
        """
        Compare two harmonic representations.

        Args:
            coefficients1: First coefficients
            coefficients2: Second coefficients

        Returns:
            Similarity score in [0, 1]
        """
        desc1 = self._extractor.extract(coefficients1)
        desc2 = self._extractor.extract(coefficients2)

        return self._similarity.compute(desc1, desc2)

    def get_dominant_modes(
        self,
        coefficients: HarmonicCoefficients,
        n_modes: int = 10,
    ) -> list[tuple[int, int, complex]]:
        """
        Get the dominant harmonic modes.

        Args:
            coefficients: Spherical harmonic coefficients
            n_modes: Number of modes to return

        Returns:
            List of (l, m, coefficient) tuples
        """
        modes = []
        for degree in range(coefficients.l_max + 1):
            for m in range(-degree, degree + 1):
                c = coefficients.get_coefficient(degree, m)
                modes.append((degree, m, c))

        modes.sort(key=lambda x: -np.abs(x[2]))
        return modes[:n_modes]

    def filter_coefficients(
        self,
        coefficients: HarmonicCoefficients,
        l_min: int = 0,
        l_max: int | None = None,
    ) -> HarmonicCoefficients:
        """
        Filter coefficients to specific degree range.

        Args:
            coefficients: Input coefficients
            l_min: Minimum degree to keep
            l_max: Maximum degree to keep

        Returns:
            Filtered coefficients
        """
        if l_max is None:
            l_max = coefficients.l_max

        new_l_max = l_max
        n_coeffs = (new_l_max + 1) ** 2
        new_coeffs = np.zeros(n_coeffs, dtype=np.complex128)

        for degree in range(l_min, l_max + 1):
            for m in range(-degree, degree + 1):
                idx = degree * (degree + 1) + m
                if degree <= coefficients.l_max:
                    new_coeffs[idx] = coefficients.get_coefficient(degree, m)

        return HarmonicCoefficients(
            l_max=new_l_max,
            coefficients=new_coeffs,
        )
