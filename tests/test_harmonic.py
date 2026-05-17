"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""
Test harmonic encoder functionality
"""

import numpy as np

from omni_mercury_engine.ml.harmonic_encoder import (
    FourierHarmonicAnalyzer,
    QuantumHarmonicOscillator,
    SphericalHarmonicDecomposer,
)


def test_spherical_harmonic_decomposer_init() -> None:
    """Test spherical harmonic decomposer initialization"""
    decomposer = SphericalHarmonicDecomposer(l_max=5)
    assert decomposer.l_max == 5


def test_spherical_harmonic_decompose() -> None:
    """Test spherical harmonic decomposition"""
    decomposer = SphericalHarmonicDecomposer(l_max=3)

    mesh_points = np.random.randn(50, 3)
    function_values = np.random.randn(50)

    coefficients = decomposer.decompose_surface(mesh_points, function_values)
    assert coefficients is not None
    assert len(coefficients) > 0


def test_spherical_harmonic_features() -> None:
    """Test rotation invariant feature extraction"""
    decomposer = SphericalHarmonicDecomposer(l_max=3)

    mesh_points = np.random.randn(50, 3)
    function_values = np.random.randn(50)

    coefficients = decomposer.decompose_surface(mesh_points, function_values)
    features = decomposer.compute_rotation_invariant_features(coefficients)

    assert features is not None
    assert len(features) == 4


def test_fourier_harmonic_analyzer_init() -> None:
    """Test Fourier harmonic analyzer initialization"""
    analyzer = FourierHarmonicAnalyzer(num_harmonics=10)
    assert analyzer.num_harmonics == 10


def test_fourier_analyze_signal() -> None:
    """Test Fourier analysis of signal"""
    analyzer = FourierHarmonicAnalyzer(num_harmonics=5)

    signal = np.sin(2 * np.pi * np.linspace(0, 1, 100))
    harmonics = analyzer.extract_harmonics(signal)

    assert harmonics is not None
    assert len(harmonics) > 0


def test_fourier_extract_features() -> None:
    """Test Fourier feature extraction"""
    analyzer = FourierHarmonicAnalyzer(num_harmonics=5)

    signal = np.random.randn(100)
    harmonics = analyzer.extract_harmonics(signal)

    assert harmonics is not None
    assert isinstance(harmonics, dict)
    assert "amplitudes" in harmonics


def test_quantum_harmonic_oscillator_init() -> None:
    """Test quantum harmonic oscillator initialization"""
    oscillator = QuantumHarmonicOscillator(mass=1.0, omega=1.0, hbar=1.0)
    assert oscillator.mass == 1.0
    assert oscillator.omega == 1.0


def test_quantum_harmonic_state() -> None:
    """Test quantum harmonic state evolution"""
    oscillator = QuantumHarmonicOscillator(mass=1.0, omega=1.0)

    x = np.linspace(-5, 5, 100)
    initial_state = oscillator.wavefunction(x, n=0)
    evolved = oscillator.evolve_state(initial_state, t=0.1, n_max=5)

    assert evolved is not None
    assert len(evolved) == 100


def test_quantum_harmonic_energy() -> None:
    """Test quantum harmonic energy computation"""
    oscillator = QuantumHarmonicOscillator(mass=1.0, omega=1.0, hbar=1.0)

    energy_0 = oscillator.energy_level(0)
    energy_1 = oscillator.energy_level(1)

    assert isinstance(energy_0, (int, float, np.number))
    assert energy_0 > 0
    assert energy_1 > energy_0
