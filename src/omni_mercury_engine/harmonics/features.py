"""
Harmonic Feature Extraction for Mercury Agent.

Provides rotation-invariant feature extraction from spherical harmonic coefficients.

References:
- Kazhdan et al. (2003): Rotation Invariant Spherical Harmonic Representation
- Funkhouser et al. (2003): A Search Engine for 3D Models
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from omni_mercury_engine.harmonics.transform import HarmonicCoefficients


logger = logging.getLogger(__name__)


@dataclass
class PowerSpectrum:
    """Power spectrum of spherical harmonic decomposition."""

    l_max: int
    spectrum: npt.NDArray[Any]
    normalized: bool = True

    def to_feature_vector(self) -> npt.NDArray[Any]:
        """Convert to feature vector."""
        return self.spectrum.copy()

    def distance_to(self, other: PowerSpectrum) -> float:
        """Compute distance to another power spectrum."""
        return float(np.linalg.norm(self.spectrum - other.spectrum))


@dataclass
class Bispectrum:
    """Bispectrum (third-order statistics) of SH coefficients."""

    l_max: int
    components: npt.NDArray[Any]
    indices: list[tuple[int, int, int]]

    def to_feature_vector(self) -> npt.NDArray[Any]:
        """Convert to feature vector."""
        return self.components.copy()


@dataclass
class RotationInvariantDescriptor:
    """Collection of rotation-invariant shape descriptors."""

    power_spectrum: PowerSpectrum
    bispectrum: Bispectrum | None
    zernike_moments: npt.NDArray[Any] | None
    energy_distribution: npt.NDArray[Any]
    complexity_measure: float


class HarmonicFeatureExtractor:
    """
    Extract rotation-invariant features from spherical harmonic coefficients.

    Provides multiple descriptor types for 3D shape analysis.
    """

    def __init__(
        self,
        l_max: int = 32,
        include_bispectrum: bool = True,
        normalize: bool = True,
    ) -> None:
        """Initialize the feature extractor."""
        self._l_max = l_max
        self._include_bispectrum = include_bispectrum
        self._normalize = normalize

    def extract(
        self,
        coefficients: HarmonicCoefficients,
    ) -> RotationInvariantDescriptor:
        """
        Extract all rotation-invariant features.

        Args:
            coefficients: Spherical harmonic coefficients

        Returns:
            RotationInvariantDescriptor with all features
        """
        power_spectrum = self.compute_power_spectrum(coefficients)

        bispectrum = None
        if self._include_bispectrum:
            bispectrum = self.compute_bispectrum(coefficients)

        energy_dist = self._compute_energy_distribution(coefficients)
        complexity = self._compute_complexity(coefficients)

        return RotationInvariantDescriptor(
            power_spectrum=power_spectrum,
            bispectrum=bispectrum,
            zernike_moments=None,
            energy_distribution=energy_dist,
            complexity_measure=complexity,
        )

    def compute_power_spectrum(
        self,
        coefficients: HarmonicCoefficients,
    ) -> PowerSpectrum:
        """
        Compute the power spectrum (rotation-invariant).

        P_l = sum_m |c_lm|^2

        Args:
            coefficients: SH coefficients

        Returns:
            PowerSpectrum descriptor
        """
        l_max = min(coefficients.l_max, self._l_max)
        spectrum = np.zeros(l_max + 1)

        for degree in range(l_max + 1):
            power = 0.0
            for m in range(-degree, degree + 1):
                c = coefficients.get_coefficient(degree, m)
                power += np.abs(c) ** 2
            spectrum[degree] = power

        if self._normalize:
            total = np.sum(spectrum) + 1e-10
            spectrum = spectrum / total

        return PowerSpectrum(
            l_max=l_max,
            spectrum=spectrum,
            normalized=self._normalize,
        )

    def compute_bispectrum(
        self,
        coefficients: HarmonicCoefficients,
        max_l: int | None = None,
    ) -> Bispectrum:
        """
        Compute the bispectrum (rotation-invariant third-order statistics).

        B_{l1,l2,l3} = sum_{m1,m2,m3} c_{l1,m1} * c_{l2,m2} * c_{l3,m3}^* * G(...)

        Args:
            coefficients: SH coefficients
            max_l: Maximum l for bispectrum (default: min(8, l_max))

        Returns:
            Bispectrum descriptor
        """
        if max_l is None:
            max_l = min(8, coefficients.l_max, self._l_max)

        indices = []
        components = []

        for l1 in range(max_l + 1):
            for l2 in range(l1, max_l + 1):
                for l3 in range(abs(l1 - l2), min(l1 + l2, max_l) + 1):
                    if (l1 + l2 + l3) % 2 != 0:
                        continue

                    b = self._compute_bispectrum_component(coefficients, l1, l2, l3)
                    indices.append((l1, l2, l3))
                    components.append(b)

        return Bispectrum(
            l_max=max_l,
            components=np.array(components),
            indices=indices,
        )

    def _compute_bispectrum_component(
        self,
        coefficients: HarmonicCoefficients,
        l1: int,
        l2: int,
        l3: int,
    ) -> complex:
        """Compute single bispectrum component."""
        total = 0.0j

        for m1 in range(-l1, l1 + 1):
            for m2 in range(-l2, l2 + 1):
                m3 = m1 + m2
                if abs(m3) > l3:
                    continue

                c1 = coefficients.get_coefficient(l1, m1)
                c2 = coefficients.get_coefficient(l2, m2)
                c3 = coefficients.get_coefficient(l3, m3)

                cg = self._clebsch_gordan_approx(l1, m1, l2, m2, l3, m3)

                total += c1 * c2 * np.conj(c3) * cg

        return total

    def _clebsch_gordan_approx(
        self,
        l1: int,
        m1: int,
        l2: int,
        m2: int,
        l3: int,
        m3: int,
    ) -> float:
        """Approximate Clebsch-Gordan coefficient."""
        if m1 + m2 != m3:
            return 0.0
        if l3 < abs(l1 - l2) or l3 > l1 + l2:
            return 0.0

        return 1.0 / np.sqrt(2 * l3 + 1)

    def _compute_energy_distribution(
        self,
        coefficients: HarmonicCoefficients,
    ) -> npt.NDArray[Any]:
        """Compute energy distribution across scales."""
        l_max = coefficients.l_max
        n_bands = min(5, (l_max + 1) // 4)

        n_bands = max(n_bands, 1)

        band_size = (l_max + 1) // n_bands
        energy = np.zeros(n_bands)

        for band in range(n_bands):
            l_start = band * band_size
            l_end = min((band + 1) * band_size, l_max + 1)

            for degree in range(l_start, l_end):
                energy[band] += coefficients.get_power_at_l(degree)

        total = np.sum(energy) + 1e-10
        return energy / total

    def _compute_complexity(
        self,
        coefficients: HarmonicCoefficients,
    ) -> float:
        """Compute shape complexity measure."""
        powers = []
        for degree in range(coefficients.l_max + 1):
            p = coefficients.get_power_at_l(degree)
            if p > 0:
                powers.append(p)

        if not powers:
            return 0.0

        powers = np.array(powers)
        total = np.sum(powers) + 1e-10
        probs = powers / total

        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(powers))

        return entropy / (max_entropy + 1e-10)

    def extract_multi_scale(
        self,
        coefficients: HarmonicCoefficients,
        scales: list[int] | None = None,
    ) -> list[RotationInvariantDescriptor]:
        """
        Extract features at multiple scales.

        Args:
            coefficients: SH coefficients
            scales: List of l_max values for each scale

        Returns:
            List of descriptors at each scale
        """
        if scales is None:
            max_l = coefficients.l_max
            scales = [max_l // 4, max_l // 2, max_l]
            scales = [s for s in scales if s > 0]

        descriptors = []
        for scale in scales:
            truncated = self._truncate_coefficients(coefficients, scale)
            desc = self.extract(truncated)
            descriptors.append(desc)

        return descriptors

    def _truncate_coefficients(
        self,
        coefficients: HarmonicCoefficients,
        new_l_max: int,
    ) -> HarmonicCoefficients:
        """Truncate coefficients to lower l_max."""
        n_coeffs = (new_l_max + 1) ** 2
        new_coeffs = np.zeros(n_coeffs, dtype=np.complex128)

        for degree in range(new_l_max + 1):
            for m in range(-degree, degree + 1):
                idx = degree * (degree + 1) + m
                new_coeffs[idx] = coefficients.get_coefficient(degree, m)

        return HarmonicCoefficients(
            l_max=new_l_max,
            coefficients=new_coeffs,
        )


class HarmonicSimilarity:
    """
    Compute similarity between harmonic representations.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
    ) -> None:
        """Initialize similarity calculator."""
        self._weights = weights or {
            "power_spectrum": 0.5,
            "bispectrum": 0.3,
            "energy": 0.2,
        }

    def compute(
        self,
        desc1: RotationInvariantDescriptor,
        desc2: RotationInvariantDescriptor,
    ) -> float:
        """
        Compute similarity between two descriptors.

        Returns value in [0, 1] where 1 is identical.
        """
        distances = {}

        ps_dist = desc1.power_spectrum.distance_to(desc2.power_spectrum)
        max_ps_dist = np.sqrt(2.0)
        distances["power_spectrum"] = ps_dist / max_ps_dist

        if desc1.bispectrum is not None and desc2.bispectrum is not None:
            bs_dist = np.linalg.norm(desc1.bispectrum.components - desc2.bispectrum.components)
            max_bs = max(
                np.linalg.norm(desc1.bispectrum.components),
                np.linalg.norm(desc2.bispectrum.components),
                1e-10,
            )
            distances["bispectrum"] = bs_dist / max_bs

        energy_dist = np.linalg.norm(desc1.energy_distribution - desc2.energy_distribution)
        distances["energy"] = energy_dist / np.sqrt(2.0)

        total_dist = 0.0
        total_weight = 0.0

        for key, weight in self._weights.items():
            if key in distances:
                total_dist += weight * distances[key]
                total_weight += weight

        if total_weight > 0:
            total_dist /= total_weight

        similarity = 1.0 - min(1.0, total_dist)
        return similarity
