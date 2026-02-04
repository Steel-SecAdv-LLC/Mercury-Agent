"""
Spherical Harmonic Transform Implementation.

Provides numerically stable spherical harmonic transforms for high l_max values.

References:
- Driscoll & Healy (1994): Computing Fourier Transforms on the 2-Sphere
- Healy et al. (2003): FFTs for the 2-Sphere - Improvements and Variations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class HarmonicCoefficients:
    """Spherical harmonic coefficients."""

    l_max: int
    coefficients: np.ndarray
    normalization: str = "ortho"
    real_basis: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_coefficient(self, degree: int, m: int) -> complex:
        """Get coefficient for degree and order m."""
        if degree > self.l_max or abs(m) > degree:
            return 0.0j

        idx = degree * (degree + 1) + m
        return self.coefficients[idx]

    def set_coefficient(self, degree: int, m: int, value: complex) -> None:
        """Set coefficient for degree and order m."""
        if degree <= self.l_max and abs(m) <= degree:
            idx = degree * (degree + 1) + m
            self.coefficients[idx] = value

    def get_power_at_l(self, degree: int) -> float:
        """Get power at degree (sum of |c_lm|^2)."""
        power = 0.0
        for m in range(-degree, degree + 1):
            c = self.get_coefficient(degree, m)
            power += np.abs(c) ** 2
        return power

    def to_array(self) -> np.ndarray:
        """Convert to flat array representation."""
        return self.coefficients.copy()

    @classmethod
    def from_array(cls, array: np.ndarray, l_max: int) -> HarmonicCoefficients:
        """Create from flat array."""
        return cls(l_max=l_max, coefficients=array)


class AssociatedLegendre:
    """
    Compute associated Legendre polynomials with numerical stability.

    Uses recurrence relations with proper normalization for high degrees.
    """

    def __init__(
        self,
        l_max: int,
        normalization: str = "ortho",
        use_float64: bool = True,
    ) -> None:
        """Initialize Legendre polynomial calculator."""
        self._l_max = l_max
        self._normalization = normalization
        self._dtype = np.float64 if use_float64 else np.float32

        self._plm_cache: dict[tuple[int, int], np.ndarray] = {}

    def compute(self, degree: int, m: int, cos_theta: np.ndarray) -> np.ndarray:
        """
        Compute P_l^m(cos(theta)) with proper normalization.

        Args:
            degree: Degree
            m: Order (|m| <= degree)
            cos_theta: Cosine of colatitude angles

        Returns:
            Associated Legendre polynomial values
        """
        m_abs = abs(m)
        if m_abs > degree:
            return np.zeros_like(cos_theta, dtype=self._dtype)

        key = (degree, m_abs)
        if key in self._plm_cache:
            plm = self._plm_cache[key].copy()
        else:
            plm = self._compute_plm(degree, m_abs, cos_theta)
            if len(self._plm_cache) < 10000:
                self._plm_cache[key] = plm.copy()

        if self._normalization == "ortho":
            norm = np.sqrt(
                (2 * degree + 1)
                / (4 * np.pi)
                * np.math.factorial(degree - m_abs)
                / np.math.factorial(degree + m_abs)
            )
            plm = plm * norm

        if m < 0:
            plm = plm * ((-1) ** m_abs)

        return plm

    def _compute_plm(
        self,
        degree: int,
        m: int,
        cos_theta: np.ndarray,
    ) -> np.ndarray:
        """Compute P_l^m using stable recurrence."""
        sin_theta = np.sqrt(1 - cos_theta**2)

        if degree == 0 and m == 0:
            return np.ones_like(cos_theta, dtype=self._dtype)

        if m == degree:
            if degree == 0:
                return np.ones_like(cos_theta, dtype=self._dtype)

            pmm_prev = np.ones_like(cos_theta, dtype=self._dtype)
            for i in range(1, m + 1):
                pmm_prev = pmm_prev * (-(2 * i - 1)) * sin_theta

            return pmm_prev

        pmm = self._compute_plm(m, m, cos_theta)

        if degree == m:
            return pmm

        pmm_plus_1 = cos_theta * (2 * m + 1) * pmm

        if degree == m + 1:
            return pmm_plus_1

        plm_prev_prev = pmm
        plm_prev = pmm_plus_1

        for ll in range(m + 2, degree + 1):
            plm = ((2 * ll - 1) * cos_theta * plm_prev - (ll + m - 1) * plm_prev_prev) / (ll - m)
            plm_prev_prev = plm_prev
            plm_prev = plm

        return plm_prev


class SHBasis:
    """
    Spherical harmonic basis functions.

    Computes Y_l^m(theta, phi) for given angles.
    """

    def __init__(
        self,
        l_max: int,
        real_basis: bool = True,
    ) -> None:
        """Initialize SH basis calculator."""
        self._l_max = l_max
        self._real_basis = real_basis
        self._legendre = AssociatedLegendre(l_max)

    def compute(
        self,
        degree: int,
        m: int,
        theta: np.ndarray,
        phi: np.ndarray,
    ) -> np.ndarray:
        """
        Compute spherical harmonic Y_l^m(theta, phi).

        Args:
            degree: Degree
            m: Order
            theta: Colatitude angles (0 to pi)
            phi: Azimuthal angles (0 to 2*pi)

        Returns:
            Spherical harmonic values
        """
        cos_theta = np.cos(theta)
        plm = self._legendre.compute(degree, m, cos_theta)

        if self._real_basis:
            if m > 0:
                return np.sqrt(2) * plm * np.cos(m * phi)
            elif m < 0:
                return np.sqrt(2) * plm * np.sin(abs(m) * phi)
            else:
                return plm
        else:
            return plm * np.exp(1j * m * phi)

    def compute_all(
        self,
        theta: np.ndarray,
        phi: np.ndarray,
    ) -> np.ndarray:
        """
        Compute all spherical harmonics up to l_max.

        Args:
            theta: Colatitude angles
            phi: Azimuthal angles

        Returns:
            Array of shape (n_coeffs, n_points)
        """
        n_points = len(theta)
        n_coeffs = (self._l_max + 1) ** 2
        result = np.zeros((n_coeffs, n_points), dtype=np.float64)

        for degree in range(self._l_max + 1):
            for m in range(-degree, degree + 1):
                idx = degree * (degree + 1) + m
                result[idx] = self.compute(degree, m, theta, phi)

        return result


class SphericalHarmonicTransform:
    """
    Fast spherical harmonic transform.

    Implements forward and inverse SH transforms with support for
    high l_max values using numerically stable algorithms.
    """

    def __init__(
        self,
        l_max: int = 32,
        backend: str = "numpy",
        precision: str = "float64",
    ) -> None:
        """
        Initialize the SH transform.

        Args:
            l_max: Maximum spherical harmonic degree
            backend: Computation backend ("numpy", "cuda", "torch", "jax")
            precision: Numerical precision ("float32", "float64")
        """
        self._l_max = l_max
        self._backend = backend
        self._precision = precision

        self._dtype = np.float64 if precision == "float64" else np.float32
        self._basis = SHBasis(l_max, real_basis=True)
        self._legendre = AssociatedLegendre(l_max)

        self._quadrature_weights: np.ndarray | None = None
        self._quadrature_points: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def l_max(self) -> int:
        """Maximum spherical harmonic degree."""
        return self._l_max

    @property
    def n_coefficients(self) -> int:
        """Number of coefficients for this l_max."""
        return (self._l_max + 1) ** 2

    def forward(
        self,
        f: np.ndarray,
        theta: np.ndarray,
        phi: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> HarmonicCoefficients:
        """
        Forward spherical harmonic transform (analysis).

        Args:
            f: Function values at sample points
            theta: Colatitude angles
            phi: Azimuthal angles
            weights: Quadrature weights (computed if None)

        Returns:
            HarmonicCoefficients
        """
        if weights is None:
            weights = self._compute_weights(theta)

        n_coeffs = self.n_coefficients
        coefficients = np.zeros(n_coeffs, dtype=np.complex128)

        basis = self._basis.compute_all(theta, phi)

        for idx in range(n_coeffs):
            coefficients[idx] = np.sum(f * basis[idx] * weights)

        return HarmonicCoefficients(
            l_max=self._l_max,
            coefficients=coefficients,
            real_basis=True,
        )

    def inverse(
        self,
        coefficients: HarmonicCoefficients,
        theta: np.ndarray,
        phi: np.ndarray,
    ) -> np.ndarray:
        """
        Inverse spherical harmonic transform (synthesis).

        Args:
            coefficients: Spherical harmonic coefficients
            theta: Colatitude angles for reconstruction
            phi: Azimuthal angles for reconstruction

        Returns:
            Reconstructed function values
        """
        basis = self._basis.compute_all(theta, phi)

        f = np.zeros(len(theta), dtype=self._dtype)

        for degree in range(coefficients.l_max + 1):
            for m in range(-degree, degree + 1):
                idx = degree * (degree + 1) + m
                c = coefficients.coefficients[idx]
                f += np.real(c) * basis[idx]

        return f

    def _compute_weights(self, theta: np.ndarray) -> np.ndarray:
        """Compute quadrature weights for integration."""
        n = len(theta)
        weights = np.ones(n) * (4 * np.pi / n)

        sin_theta = np.sin(theta)
        weights = weights * sin_theta / (np.sum(sin_theta) / n)

        return weights

    def decompose(
        self,
        point_cloud: np.ndarray,
        sampling: str = "healpix",
    ) -> HarmonicCoefficients:
        """
        Decompose a 3D point cloud into spherical harmonics.

        Args:
            point_cloud: 3D point cloud (N, 3) or (N, 4) with values
            sampling: Sampling scheme ("healpix", "equiangular", "fibonacci")

        Returns:
            HarmonicCoefficients
        """
        if point_cloud.shape[1] == 3:
            x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
            r = np.sqrt(x**2 + y**2 + z**2)
            values = r
        else:
            x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
            values = point_cloud[:, 3]
            r = np.sqrt(x**2 + y**2 + z**2)

        theta = np.arccos(z / (r + 1e-10))
        phi = np.arctan2(y, x)

        return self.forward(values, theta, phi)

    def reconstruct(
        self,
        coefficients: HarmonicCoefficients,
        n_theta: int = 64,
        n_phi: int = 128,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Reconstruct surface from spherical harmonic coefficients.

        Args:
            coefficients: Spherical harmonic coefficients
            n_theta: Number of colatitude samples
            n_phi: Number of azimuthal samples

        Returns:
            Tuple of (theta_grid, phi_grid, values)
        """
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi)
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

        theta_flat = theta_grid.flatten()
        phi_flat = phi_grid.flatten()

        values_flat = self.inverse(coefficients, theta_flat, phi_flat)
        values = values_flat.reshape(theta_grid.shape)

        return theta_grid, phi_grid, values


class FastSHTransform:
    """
    Optimized spherical harmonic transform using FFT.

    Provides faster computation for large datasets.
    """

    def __init__(
        self,
        l_max: int = 64,
        n_theta: int | None = None,
        n_phi: int | None = None,
    ) -> None:
        """Initialize fast SH transform."""
        self._l_max = l_max
        self._n_theta = n_theta or (2 * l_max + 2)
        self._n_phi = n_phi or (2 * l_max + 2)

        self._setup_grids()
        self._precompute_basis()

    def _setup_grids(self) -> None:
        """Setup sampling grids."""
        self._theta = np.linspace(0, np.pi, self._n_theta)
        self._phi = np.linspace(0, 2 * np.pi, self._n_phi, endpoint=False)

        self._theta_grid, self._phi_grid = np.meshgrid(self._theta, self._phi, indexing="ij")

        sin_theta = np.sin(self._theta)
        self._weights = np.outer(sin_theta, np.ones(self._n_phi))
        self._weights *= (np.pi / self._n_theta) * (2 * np.pi / self._n_phi)

    def _precompute_basis(self) -> None:
        """Precompute basis functions for speed."""
        self._legendre = AssociatedLegendre(self._l_max)
        self._cos_theta = np.cos(self._theta)

        self._plm_table: dict[tuple[int, int], np.ndarray] = {}
        for degree in range(self._l_max + 1):
            for m in range(degree + 1):
                plm = self._legendre.compute(degree, m, self._cos_theta)
                self._plm_table[(degree, m)] = plm

    def forward(self, f: np.ndarray) -> HarmonicCoefficients:
        """
        Fast forward SH transform using FFT.

        Args:
            f: Function values on grid (n_theta, n_phi)

        Returns:
            HarmonicCoefficients
        """
        f_phi_fft = np.fft.fft(f, axis=1)

        n_coeffs = (self._l_max + 1) ** 2
        coefficients = np.zeros(n_coeffs, dtype=np.complex128)

        for degree in range(self._l_max + 1):
            for m in range(-degree, degree + 1):
                idx = degree * (degree + 1) + m

                plm = self._plm_table.get((degree, abs(m)))
                if plm is None:
                    continue

                if m >= 0:
                    f_m = f_phi_fft[:, m]
                else:
                    f_m = np.conj(f_phi_fft[:, -m])

                integral = np.sum(f_m * plm * np.sin(self._theta)) * np.pi / self._n_theta

                coefficients[idx] = integral

        return HarmonicCoefficients(
            l_max=self._l_max,
            coefficients=coefficients,
        )

    def inverse(self, coefficients: HarmonicCoefficients) -> np.ndarray:
        """
        Fast inverse SH transform using FFT.

        Args:
            coefficients: Spherical harmonic coefficients

        Returns:
            Function values on grid (n_theta, n_phi)
        """
        f_phi_fft = np.zeros((self._n_theta, self._n_phi), dtype=np.complex128)

        for degree in range(coefficients.l_max + 1):
            for m in range(-degree, degree + 1):
                idx = degree * (degree + 1) + m
                c = coefficients.coefficients[idx]

                plm = self._plm_table.get((degree, abs(m)))
                if plm is None:
                    continue

                if m >= 0:
                    f_phi_fft[:, m] += c * plm
                else:
                    f_phi_fft[:, -m] += np.conj(c) * plm

        f = np.fft.ifft(f_phi_fft, axis=1).real

        return f
