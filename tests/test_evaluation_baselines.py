"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Tests for evaluation baselines module.
"""

from __future__ import annotations

from typing import cast

import pytest

from omni_mercury_engine.evaluation.baselines import (
    BASELINE_RESULTS,
    BaselineComparison,
    compare_to_baselines,
    get_baseline_citations,
    get_sota_for_dataset,
    list_available_datasets,
    print_baseline_table,
)


class TestBaselineResults:
    """Tests for baseline results data."""

    def test_baseline_results_not_empty(self) -> None:
        """Should have baseline results."""
        assert len(BASELINE_RESULTS) > 0

    def test_expected_datasets_present(self) -> None:
        """Should have results for standard datasets."""
        expected_datasets = ["SMD", "SMAP", "MSL", "NSL-KDD", "NAB"]
        for dataset in expected_datasets:
            assert dataset in BASELINE_RESULTS

    def test_smd_baselines_present(self) -> None:
        """SMD should have expected baseline methods."""
        smd = BASELINE_RESULTS["SMD"]
        expected_methods = ["OmniAnomaly", "DAGMM", "TranAD"]
        for method in expected_methods:
            assert method in smd

    def test_baseline_metrics_format(self) -> None:
        """Each baseline should have precision, recall, f1."""
        # BASELINE_RESULTS values are heterogeneous (float scores + "paper" string),
        # so per-key lookups return object; cast to float for ordered comparison.
        for dataset, baselines in BASELINE_RESULTS.items():
            for method, metrics in baselines.items():
                if "f1" in metrics:  # Some like NAB use nab_score
                    assert 0 <= cast("float", metrics["f1"]) <= 1
                if "precision" in metrics:
                    assert 0 <= cast("float", metrics["precision"]) <= 1
                if "recall" in metrics:
                    assert 0 <= cast("float", metrics["recall"]) <= 1


class TestCompareToBaselines:
    """Tests for baseline comparison function."""

    def test_compare_basic(self) -> None:
        """Should return BaselineComparison object."""
        result = compare_to_baselines(
            dataset="SMD",
            your_precision=0.85,
            your_recall=0.90,
            your_f1=0.87,
        )
        assert isinstance(result, BaselineComparison)

    def test_compare_rank_calculation(self) -> None:
        """Should correctly calculate rank."""
        # TranAD has highest F1 on SMD (~0.9605)
        # Our score of 0.97 should be rank 1 (higher than all baselines)
        result = compare_to_baselines(
            dataset="SMD",
            your_precision=0.97,
            your_recall=0.97,
            your_f1=0.97,
        )
        assert result.rank == 1

    def test_compare_low_score_rank(self) -> None:
        """Low score should have low rank."""
        result = compare_to_baselines(
            dataset="SMD",
            your_precision=0.50,
            your_recall=0.50,
            your_f1=0.50,
        )
        # Should be ranked last (there are 7 baselines + ours = 8)
        assert result.rank > 1

    def test_compare_improvement_over_avg(self) -> None:
        """Should calculate improvement over average."""
        result = compare_to_baselines(
            dataset="SMD",
            your_precision=0.85,
            your_recall=0.90,
            your_f1=0.87,
        )
        assert result.improvement_over_avg is not None

    def test_compare_unknown_dataset_raises(self) -> None:
        """Should raise for unknown dataset."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            compare_to_baselines(
                dataset="UNKNOWN",
                your_precision=0.85,
                your_recall=0.90,
                your_f1=0.87,
            )

    def test_baseline_comparison_str(self) -> None:
        """Should have readable string representation."""
        result = compare_to_baselines(
            dataset="SMD",
            your_precision=0.85,
            your_recall=0.90,
            your_f1=0.87,
        )
        s = str(result)
        assert "SMD" in s
        assert "Your F1" in s


class TestPrintBaselineTable:
    """Tests for baseline table formatting."""

    def test_generates_table(self) -> None:
        """Should generate formatted table string."""
        table = print_baseline_table("SMD")
        assert "SMD" in table
        assert "OmniAnomaly" in table

    def test_includes_your_results(self) -> None:
        """Should include user's results in table."""
        table = print_baseline_table(
            "SMD", your_results={"precision": 0.85, "recall": 0.90, "f1": 0.87}
        )
        assert "YOUR MODEL" in table

    def test_unknown_dataset_message(self) -> None:
        """Should return message for unknown dataset."""
        table = print_baseline_table("UNKNOWN")
        assert "No baselines available" in table

    def test_sorted_by_f1(self) -> None:
        """Results should be sorted by F1 descending."""
        table = print_baseline_table("SMD")
        # TranAD should appear before DAGMM (higher F1)
        tranad_pos = table.find("TranAD")
        dagmm_pos = table.find("DAGMM")
        assert tranad_pos < dagmm_pos


class TestGetBaselineCitations:
    """Tests for citation retrieval."""

    def test_returns_citations_dict(self) -> None:
        """Should return dictionary of citations."""
        citations = get_baseline_citations()
        assert isinstance(citations, dict)

    def test_expected_methods_cited(self) -> None:
        """Should have citations for key methods."""
        citations = get_baseline_citations()
        expected = ["OmniAnomaly", "MSCRED", "DAGMM", "TranAD"]
        for method in expected:
            assert method in citations

    def test_citation_format(self) -> None:
        """Citations should be non-empty strings."""
        citations = get_baseline_citations()
        for method, citation in citations.items():
            assert isinstance(citation, str)
            assert len(citation) > 0


class TestListAvailableDatasets:
    """Tests for dataset listing."""

    def test_returns_list(self) -> None:
        """Should return list of dataset names."""
        datasets = list_available_datasets()
        assert isinstance(datasets, list)

    def test_expected_datasets(self) -> None:
        """Should include expected datasets."""
        datasets = list_available_datasets()
        assert "SMD" in datasets
        assert "SMAP" in datasets
        assert "MSL" in datasets


class TestGetSotaForDataset:
    """Tests for SOTA retrieval."""

    def test_returns_sota(self) -> None:
        """Should return SOTA method and metrics."""
        method, metrics = get_sota_for_dataset("SMD")
        assert method is not None
        assert isinstance(metrics, dict)

    def test_sota_has_highest_f1(self) -> None:
        """SOTA should have highest F1 for dataset."""
        method, metrics = get_sota_for_dataset("SMD")
        smd_baselines = BASELINE_RESULTS["SMD"]

        sota_f1 = cast("float", metrics["f1"])
        for name, m in smd_baselines.items():
            if "f1" in m:
                assert cast("float", m["f1"]) <= sota_f1

    def test_unknown_dataset_raises(self) -> None:
        """Should raise for unknown dataset."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_sota_for_dataset("UNKNOWN")


class TestDatasetSpecificBaselines:
    """Tests for specific dataset baselines."""

    def test_smap_baselines(self) -> None:
        """SMAP dataset should have correct baselines."""
        smap = BASELINE_RESULTS["SMAP"]
        assert "OmniAnomaly" in smap
        assert smap["OmniAnomaly"]["paper"] == "KDD 2019"

    def test_msl_baselines(self) -> None:
        """MSL dataset should have correct baselines."""
        msl = BASELINE_RESULTS["MSL"]
        assert "TranAD" in msl
        assert msl["TranAD"]["paper"] == "VLDB 2022"

    def test_nsl_kdd_baselines(self) -> None:
        """NSL-KDD should have network intrusion baselines."""
        nsl = BASELINE_RESULTS["NSL-KDD"]
        assert "Random_Forest" in nsl
        assert "DNN" in nsl

    def test_nab_baselines(self) -> None:
        """NAB should have HTM and other streaming baselines."""
        nab = BASELINE_RESULTS["NAB"]
        assert "HTM" in nab
        # NAB uses nab_score not f1
        assert "nab_score" in nab["HTM"]
