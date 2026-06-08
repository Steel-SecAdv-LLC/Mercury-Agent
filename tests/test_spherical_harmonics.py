# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for spherical harmonics analysis feature."""

from __future__ import annotations

from omni_mercury_engine.core.three_r_mechanism import RefactoringConfig, RefactoringEngine


class TestSphericalHarmonics:
    """Tests for spherical harmonics analysis."""

    def test_spherical_harmonics_disabled_by_default(self) -> None:
        """Verify spherical harmonics is disabled by default."""
        config = RefactoringConfig()
        assert config.enable_spherical_harmonics is False

    def test_spherical_harmonics_can_be_enabled(self) -> None:
        """Verify spherical harmonics can be enabled."""
        config = RefactoringConfig(enable_spherical_harmonics=True)
        engine = RefactoringEngine(config)

        def sample_func():
            return 42

        result = engine.analyze_with_spherical_harmonics(sample_func)
        assert result["enabled"] is True
        assert "coefficients" in result
        assert "spherical_coords" in result

    def test_spherical_harmonics_returns_coefficients(self) -> None:
        """Verify Y_l^m coefficients are computed."""
        config = RefactoringConfig(enable_spherical_harmonics=True, spherical_harmonic_degree=2)
        engine = RefactoringEngine(config)

        def sample_func(x, y):
            if x > 0:
                return x + y
            return y

        result = engine.analyze_with_spherical_harmonics(sample_func)

        assert "Y_0_0" in result["coefficients"]
        assert "Y_1_0" in result["coefficients"]
        assert "Y_2_0" in result["coefficients"]

    def test_spherical_harmonics_respects_max_degree(self) -> None:
        """Verify max_degree parameter controls coefficient count."""
        config1 = RefactoringConfig(enable_spherical_harmonics=True, spherical_harmonic_degree=1)
        config2 = RefactoringConfig(enable_spherical_harmonics=True, spherical_harmonic_degree=3)

        engine1 = RefactoringEngine(config1)
        engine2 = RefactoringEngine(config2)

        def sample_func():
            x = 0
            for i in range(10):
                x += i
            return x

        result1 = engine1.analyze_with_spherical_harmonics(sample_func)
        result2 = engine2.analyze_with_spherical_harmonics(sample_func)

        assert len(result1["coefficients"]) < len(result2["coefficients"])

    def test_spherical_harmonics_disabled_returns_message(self) -> None:
        """Verify disabled feature returns appropriate message."""
        config = RefactoringConfig(enable_spherical_harmonics=False)
        engine = RefactoringEngine(config)

        def sample_func():
            return 42

        result = engine.analyze_with_spherical_harmonics(sample_func)

        assert result["enabled"] is False
        assert "message" in result

    def test_spherical_coords_in_valid_range(self) -> None:
        """Verify spherical coordinates are in valid ranges."""
        import math

        config = RefactoringConfig(enable_spherical_harmonics=True)
        engine = RefactoringEngine(config)

        def sample_func(x):
            if x > 10:
                return x * 2
            return x

        result = engine.analyze_with_spherical_harmonics(sample_func)

        theta = result["spherical_coords"]["theta"]
        phi = result["spherical_coords"]["phi"]

        assert 0 <= theta <= math.pi
        assert -math.pi <= phi <= math.pi

    def test_spherical_harmonics_coefficient_structure(self) -> None:
        """Verify coefficient structure contains real, imag, magnitude."""
        config = RefactoringConfig(enable_spherical_harmonics=True)
        engine = RefactoringEngine(config)

        def sample_func():
            return sum(range(5))

        result = engine.analyze_with_spherical_harmonics(sample_func)

        for coeff_name, coeff_data in result["coefficients"].items():
            assert "real" in coeff_data
            assert "imag" in coeff_data
            assert "magnitude" in coeff_data

    def test_orchestrate_with_spherical_strategy(self) -> None:
        """Verify orchestration supports spherical strategy."""
        config = RefactoringConfig(enable_spherical_harmonics=True)
        engine = RefactoringEngine(config)

        def sample_func():
            return 42

        result = engine.orchestrate_refactoring(sample_func, strategies=["spherical"])

        assert "orchestrated_analysis" in result
        assert "spherical" in result["orchestrated_analysis"]
        assert result["orchestrated_analysis"]["spherical"]["enabled"] is True

    def test_rotation_invariance_flag_included(self) -> None:
        """Verify rotation invariance flag is included in results."""
        config = RefactoringConfig(enable_spherical_harmonics=True, enable_rotation_invariance=True)
        engine = RefactoringEngine(config)

        def sample_func():
            return 42

        result = engine.analyze_with_spherical_harmonics(sample_func)

        assert "rotation_invariant" in result
        assert result["rotation_invariant"] is True

    def test_spherical_harmonics_handles_complex_function(self) -> None:
        """Verify spherical harmonics works with complex nested function."""
        config = RefactoringConfig(enable_spherical_harmonics=True)
        engine = RefactoringEngine(config)

        def complex_func(data):
            result = []
            for item in data:
                if item > 5:
                    if item < 10:
                        result.append(item * 2)
                    else:
                        result.append(item + 1)
                else:
                    result.append(item)
            return result

        result = engine.analyze_with_spherical_harmonics(complex_func)

        assert result["enabled"] is True
        assert "coefficients" in result
        assert len(result["coefficients"]) > 0
