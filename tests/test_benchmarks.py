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
from __future__ import annotations

"""Tests for benchmark framework."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.refactoring_benchmarks import RefactoringBenchmark  # noqa: E402
from benchmarks.statistical_validation import statistical_analysis  # noqa: E402


class TestBenchmarkFramework:
    """Tests for benchmark execution."""

    def test_benchmark_initialization(self):
        """Test benchmark can be initialized."""
        repo_paths = [Path.home() / "benchmark_repos" / "requests"]
        benchmark = RefactoringBenchmark(repo_paths)

        assert benchmark.repo_paths == repo_paths
        assert benchmark.engine is not None

    def test_statistical_validation(self):
        """Test statistical validation functions."""
        baseline = np.array([10.0, 11.0, 9.0, 10.5, 9.5] * 20)
        improved = np.array([8.0, 8.5, 7.5, 8.2, 7.8] * 20)

        results = statistical_analysis(baseline, improved)

        assert "t_statistic" in results
        assert "p_value" in results
        assert "improvement_percent" in results
        assert "confidence_interval_95" in results
        assert "cohens_d" in results
        assert results["significant"] is True
        assert results["improvement_percent"] > 15

    def test_improvement_calculation(self):
        """Test improvement percentage calculation."""
        baseline = np.array([100.0] * 50)
        improved = np.array([82.0] * 50)

        results = statistical_analysis(baseline, improved)

        assert abs(results["improvement_percent"] - 18.0) < 0.1
