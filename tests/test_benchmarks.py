# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for benchmark framework."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.refactoring_benchmarks import RefactoringBenchmark
from benchmarks.statistical_validation import statistical_analysis


class TestBenchmarkFramework:
    """Tests for benchmark execution."""

    def test_benchmark_initialization(self) -> None:
        """Test benchmark can be initialized."""
        repo_paths = [Path.home() / "benchmark_repos" / "requests"]
        benchmark = RefactoringBenchmark(repo_paths)

        assert benchmark.repo_paths == repo_paths
        assert benchmark.engine is not None

    def test_statistical_validation(self) -> None:
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

    def test_improvement_calculation(self) -> None:
        """Test improvement percentage calculation."""
        baseline = np.array([100.0] * 50)
        improved = np.array([82.0] * 50)

        results = statistical_analysis(baseline, improved)

        assert abs(results["improvement_percent"] - 18.0) < 0.1
