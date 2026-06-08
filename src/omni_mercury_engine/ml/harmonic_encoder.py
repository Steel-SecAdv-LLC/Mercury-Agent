# Copyright (C) 2025 Steel Security Advisors LLC
"""Harmonic analysis encoder using spherical harmonics and Fourier analysis.

Provides frequency-domain feature extraction for anomaly detection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
else:
    try:
        import torch
        from torch import nn

        TORCH_AVAILABLE = True
    except ImportError:
        TORCH_AVAILABLE = False

from scipy.fft import fft, ifft


# Handle scipy.special spherical harmonics API deprecation (scipy 1.14+).
# scipy 1.14 introduced ``sph_harm_y(n, m, theta, phi)`` and deprecated the
# legacy ``sph_harm(m, n, theta, phi)``. We resolve the appropriate backend
# exactly once via a closure factory so the public ``_sph_harm`` symbol has a
# single canonical definition (no conditional redefinition / no ``no-redef``
# suppression required).
def _make_sph_harm() -> Any:
    """Build a uniform ``_sph_harm(m, n, theta, phi)`` callable.

    Prefers the modern ``scipy.special.sph_harm_y`` API and falls back to the
    legacy ``scipy.special.sph_harm`` symbol for older SciPy releases. The
    returned callable always takes arguments in the legacy ``(m, n, theta,
    phi)`` order so the rest of this module can be agnostic to the SciPy
    version installed.
    """
    try:
        from scipy.special import sph_harm_y as _sph_harm_y

        def _impl(
            m: int, n: int, theta: np.ndarray[Any, Any], phi: np.ndarray[Any, Any]
        ) -> np.ndarray[Any, Any]:
            # New API takes (n, m, theta, phi)
            result: np.ndarray[Any, Any] = _sph_harm_y(n, m, theta, phi)
            return result

    except ImportError:
        from scipy.special import sph_harm as _sph_harm_legacy

        def _impl(
            m: int, n: int, theta: np.ndarray[Any, Any], phi: np.ndarray[Any, Any]
        ) -> np.ndarray[Any, Any]:
            # Legacy API takes (m, n, theta, phi)
            result: np.ndarray[Any, Any] = _sph_harm_legacy(m, n, theta, phi)
            return result

    return _impl


_sph_harm = _make_sph_harm()


class SphericalHarmonicDecomposer:
    """Spherical harmonic decomposition for 3D surface analysis Provides rotation-invariant feature.

    extraction for facial biometrics.
    """

    def __init__(self, l_max: int = 10) -> None:
        """Initialize the instance."""
        self.l_max = l_max
        self.num_coefficients = (l_max + 1) ** 2

    def decompose_surface(
        self, points: np.ndarray[Any, Any], values: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        """Decompose 3D surface into spherical harmonic coefficients.

        Args:
            points: Array of shape (N, 3) containing (x, y, z) coordinates
            values: Array of shape (N,) containing surface values

        Returns:
            Coefficients array of shape (num_coefficients,)
        """
        theta, phi = self._cartesian_to_spherical(points)

        coefficients = np.zeros(self.num_coefficients, dtype=complex)

        idx = 0
        for degree in range(self.l_max + 1):
            for order in range(-degree, degree + 1):
                Y_lm = _sph_harm(order, degree, theta, phi)
                coefficients[idx] = np.sum(values * Y_lm.conj()) / len(values)
                idx += 1

        return coefficients

    def reconstruct_surface(
        self,
        coefficients: np.ndarray[Any, Any],
        theta: np.ndarray[Any, Any],
        phi: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Reconstruct surface from spherical harmonic coefficients.

        Args:
            coefficients: Spherical harmonic coefficients
            theta: Polar angles
            phi: Azimuthal angles

        Returns:
            Reconstructed surface values
        """
        reconstruction = np.zeros_like(theta, dtype=complex)

        idx = 0
        for degree in range(self.l_max + 1):
            for order in range(-degree, degree + 1):
                Y_lm = _sph_harm(order, degree, theta, phi)
                reconstruction += coefficients[idx] * Y_lm
                idx += 1

        return reconstruction.real

    def compute_rotation_invariant_features(
        self, coefficients: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        """Compute rotation-invariant features from spherical harmonic coefficients Uses power spectrum.

        which is rotation-invariant.

        Args:
            coefficients: Spherical harmonic coefficients

        Returns:
            Rotation-invariant power spectrum features
        """
        power_spectrum = np.zeros(self.l_max + 1)

        idx = 0
        for degree in range(self.l_max + 1):
            power = 0.0
            for _order in range(-degree, degree + 1):
                power += np.abs(coefficients[idx]) ** 2
                idx += 1
            power_spectrum[degree] = power / (2 * degree + 1)

        return power_spectrum

    def _cartesian_to_spherical(
        self, points: np.ndarray[Any, Any]
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Convert Cartesian coordinates to spherical (theta, phi).

        Args:
            points: Array of shape (N, 3) with (x, y, z)

        Returns:
            theta: Polar angles
            phi: Azimuthal angles
        """
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)

        theta = np.arccos(z / (r + 1e-10))
        phi = np.arctan2(y, x)

        return theta, phi


class FourierHarmonicAnalyzer:
    """Fourier harmonic analysis for frequency-domain pattern extraction."""

    def __init__(self, num_harmonics: int = 8) -> None:
        """Initialize the instance."""
        self.num_harmonics = num_harmonics

    def extract_harmonics(self, signal: np.ndarray[Any, Any]) -> dict[str, np.ndarray[Any, Any]]:
        """Extract harmonic components from signal using FFT.

        Args:
            signal: Input signal (1D array)

        Returns:
            Dictionary with harmonic features
        """
        fft_result = fft(signal)
        frequencies = np.fft.fftfreq(len(signal))

        power_spectrum = np.abs(fft_result) ** 2

        sorted_indices = np.argsort(power_spectrum)[::-1]
        top_harmonics = sorted_indices[: self.num_harmonics]

        harmonic_amplitudes = np.abs(fft_result[top_harmonics])
        harmonic_phases = np.angle(fft_result[top_harmonics])
        harmonic_frequencies = frequencies[top_harmonics]

        return {
            "amplitudes": harmonic_amplitudes,
            "phases": harmonic_phases,
            "frequencies": harmonic_frequencies,
            "power_spectrum": power_spectrum,
        }

    def apply_bandpass_filter(
        self, signal: np.ndarray[Any, Any], low_freq: float, high_freq: float
    ) -> np.ndarray[Any, Any]:
        """Apply bandpass filter to signal.

        Args:
            signal: Input signal
            low_freq: Lower cutoff frequency (0-0.5)
            high_freq: Upper cutoff frequency (0-0.5)

        Returns:
            Filtered signal
        """
        fft_result = fft(signal)
        frequencies = np.fft.fftfreq(len(signal))

        mask = (np.abs(frequencies) >= low_freq) & (np.abs(frequencies) <= high_freq)
        fft_filtered = fft_result * mask

        filtered_signal = ifft(fft_filtered).real

        return filtered_signal


class QuantumHarmonicOscillator:
    """Quantum harmonic oscillator model for state evolution.

    Based on quantum mechanics principles for coherent state evolution.
    """

    def __init__(self, mass: float = 1.0, omega: float = 1.0, hbar: float = 1.0) -> None:
        """Initialize the instance."""
        self.mass = mass
        self.omega = omega
        self.hbar = hbar

    def energy_level(self, n: int) -> float:
        """Compute energy of quantum harmonic oscillator at level n.

        Args:
            n: Quantum number (non-negative integer)

        Returns:
            Energy value
        """
        return self.hbar * self.omega * (n + 0.5)

    def wavefunction(self, x: np.ndarray[Any, Any], n: int) -> np.ndarray[Any, Any]:
        """Compute wavefunction for quantum harmonic oscillator.

        Args:
            x: Position array
            n: Quantum number

        Returns:
            Wavefunction values
        """
        import math

        from scipy.special import hermite

        alpha = np.sqrt(self.mass * self.omega / self.hbar)

        normalization = (alpha / np.pi) ** 0.25 / np.sqrt(2**n * math.factorial(n))

        H_n = hermite(n)
        psi = normalization * H_n(alpha * x) * np.exp(-0.5 * alpha**2 * x**2)

        return psi

    def evolve_state(
        self, psi_0: np.ndarray[Any, Any], t: float, n_max: int = 10
    ) -> np.ndarray[Any, Any]:
        """Evolve quantum state in time.

        Args:
            psi_0: Initial state
            t: Time
            n_max: Maximum quantum number for expansion

        Returns:
            Evolved state
        """
        x = np.linspace(-5, 5, len(psi_0))
        psi_t = np.zeros_like(psi_0, dtype=complex)

        for n in range(n_max + 1):
            psi_n = self.wavefunction(x, n)
            c_n = np.sum(psi_0.conj() * psi_n) * (x[1] - x[0])

            E_n = self.energy_level(n)
            phase = np.exp(-1j * E_n * t / self.hbar)

            psi_t += c_n * psi_n * phase

        return psi_t


if TYPE_CHECKING or TORCH_AVAILABLE:

    class HarmonicEncoder(nn.Module):
        """PyTorch module wrapping harmonic analysis for ML fusion."""

        def __init__(
            self,
            l_max: int = 10,
            num_fourier_harmonics: int = 8,
            output_dim: int = 64,
        ):
            """Initialize the instance."""
            super().__init__()

            self.spherical_decomposer = SphericalHarmonicDecomposer(l_max=l_max)
            self.fourier_analyzer = FourierHarmonicAnalyzer(num_harmonics=num_fourier_harmonics)

            self.feature_dim = l_max + 1 + num_fourier_harmonics * 2

            self.projection = nn.Linear(self.feature_dim, output_dim)

        def forward(
            self,
            points: torch.Tensor | None = None,
            values: torch.Tensor | None = None,
            signal: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """Extract harmonic features from 3D surface or 1D signal.

            Args:
                points: 3D surface points (N, 3)
                values: Surface values (N,)
                signal: 1D signal for Fourier analysis (N,)

            Returns:
                Encoded features (output_dim,)
            """
            features = []

            if points is not None and values is not None:
                points_np = points.cpu().numpy()
                values_np = values.cpu().numpy()

                coeffs = self.spherical_decomposer.decompose_surface(points_np, values_np)
                power_spectrum = self.spherical_decomposer.compute_rotation_invariant_features(
                    coeffs
                )
                features.append(torch.tensor(power_spectrum, dtype=torch.float32))

            if signal is not None:
                signal_np = signal.cpu().numpy()
                harmonics = self.fourier_analyzer.extract_harmonics(signal_np)

                fourier_feats = np.concatenate(
                    [
                        harmonics["amplitudes"],
                        harmonics["phases"],
                    ]
                )
                features.append(torch.tensor(fourier_feats, dtype=torch.float32))

            if not features:
                raise ValueError("Must provide either (points, values) or signal")

            combined_features = torch.cat(features, dim=0)

            if combined_features.shape[0] < self.feature_dim:
                padding = torch.zeros(self.feature_dim - combined_features.shape[0])
                combined_features = torch.cat([combined_features, padding], dim=0)
            elif combined_features.shape[0] > self.feature_dim:
                combined_features = combined_features[: self.feature_dim]

            encoded = self.projection(combined_features.unsqueeze(0))

            return encoded.squeeze(0)

else:

    class HarmonicEncoder:
        """Stub: HarmonicEncoder requires PyTorch."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Initialize the instance."""
            raise ImportError("HarmonicEncoder requires PyTorch. Install with: pip install torch")
