"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

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

"""Tests for STEM-verifiable mathematical constants."""

from omni_anomaly_engine.core.three_r_mechanism import RefactoringConfig


class TestMathematicalConstants:
    """Tests for mathematical constants in RefactoringConfig."""

    def test_golden_ratio_value(self):
        """Verify golden ratio constant."""
        config = RefactoringConfig()
        assert abs(config.golden_ratio - 0.618033988749) < 1e-10

    def test_catalan_constant_value(self):
        """Verify Catalan's constant."""
        config = RefactoringConfig()
        assert abs(config.catalan_constant - 0.915965594177219) < 1e-10

    def test_euler_mascheroni_value(self):
        """Verify Euler-Mascheroni constant."""
        config = RefactoringConfig()
        assert abs(config.euler_mascheroni - 0.5772156649) < 1e-9

    def test_feigenbaum_delta_value(self):
        """Verify Feigenbaum constant (chaos theory)."""
        config = RefactoringConfig()
        assert abs(config.feigenbaum_delta - 4.6692016091) < 1e-9

    def test_omega_constant_value(self):
        """Verify Omega constant."""
        config = RefactoringConfig()
        assert abs(config.omega_constant - 0.5671432904) < 1e-9

    def test_constants_are_accessible(self):
        """Verify all constants can be accessed from config."""
        config = RefactoringConfig()

        assert hasattr(config, "golden_ratio")
        assert hasattr(config, "catalan_constant")
        assert hasattr(config, "euler_mascheroni")
        assert hasattr(config, "feigenbaum_delta")
        assert hasattr(config, "omega_constant")

    def test_constants_are_positive(self):
        """Verify all mathematical constants are positive."""
        config = RefactoringConfig()

        assert config.golden_ratio > 0
        assert config.catalan_constant > 0
        assert config.euler_mascheroni > 0
        assert config.feigenbaum_delta > 0
        assert config.omega_constant > 0

    def test_golden_ratio_is_less_than_one(self):
        """Verify golden ratio minus one is less than 1."""
        config = RefactoringConfig()
        assert config.golden_ratio < 1

    def test_feigenbaum_delta_greater_than_four(self):
        """Verify Feigenbaum constant is greater than 4."""
        config = RefactoringConfig()
        assert config.feigenbaum_delta > 4

    def test_constants_can_be_customized(self):
        """Verify constants can be overridden if needed."""
        config = RefactoringConfig(golden_ratio=0.5)
        assert config.golden_ratio == 0.5

    def test_spherical_harmonic_degree_default(self):
        """Verify spherical harmonic degree default value."""
        config = RefactoringConfig()
        assert config.spherical_harmonic_degree == 4

    def test_superposition_paths_default(self):
        """Verify superposition paths default value."""
        config = RefactoringConfig()
        assert config.superposition_paths == 3

    def test_all_optional_flags_default_false(self):
        """Verify advanced features are disabled by default."""
        config = RefactoringConfig()

        assert config.enable_spherical_harmonics is False
        assert config.enable_rotation_invariance is False
        assert config.enable_quantum_superposition is False

    def test_config_with_all_features_enabled(self):
        """Verify config can enable all advanced features."""
        config = RefactoringConfig(
            enable_spherical_harmonics=True,
            enable_rotation_invariance=True,
            enable_quantum_superposition=True,
        )

        assert config.enable_spherical_harmonics is True
        assert config.enable_rotation_invariance is True
        assert config.enable_quantum_superposition is True
