"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

Tests for ``benchmarks/statistical_significance.py`` — the paired-inference
confirmation of PR #265's sub-threshold neuro-symbolic sweep results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from benchmarks import statistical_significance as ss

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RULEGRAPH = REPO_ROOT / "artifacts" / "symbolic_rulegraph_sweep.json"
SEMANTICS = REPO_ROOT / "artifacts" / "symbolic_semantics_sweep.json"


def test_sign_test_exact_values() -> None:
    assert ss._sign_test_p(0, 0) == 1.0
    # all positive -> strongest two-sided signal
    assert ss._sign_test_p(10, 10) == pytest.approx(2 * 0.5**10)
    # perfectly split -> p = 1.0
    assert ss._sign_test_p(5, 10) == pytest.approx(1.0)


def test_t_sf_matches_scipy_when_available() -> None:
    scipy_stats = pytest.importorskip("scipy.stats")
    for t_abs, df in [(2.0, 10), (1.0, 26), (3.5, 5)]:
        expected = float(2.0 * scipy_stats.t.sf(t_abs, df))
        assert ss._t_sf(t_abs, df) == pytest.approx(expected, rel=1e-3, abs=1e-4)


def test_paired_stats_detects_real_positive_effect() -> None:
    rng = np.random.default_rng(0)
    b = rng.normal(0.8, 0.02, size=200)
    a = b + 0.01 + rng.normal(0.0, 0.002, size=200)  # +0.01 with realistic jitter
    s = ss.paired_stats(a, b, n_boot=2000, seed=0, bar=0.002)
    assert s["mean_diff"] == pytest.approx(0.01, abs=2e-3)
    assert s["clears_bar"] is True
    assert s["ci_excludes_zero"] is True
    assert s["confirmed"] is True
    assert s["p_value_ttest"] < 0.05


def test_paired_stats_rejects_noise() -> None:
    rng = np.random.default_rng(1)
    b = rng.normal(0.8, 0.05, size=30)
    a = b + rng.normal(0.0, 0.05, size=30)  # no real difference
    s = ss.paired_stats(a, b, n_boot=2000, seed=0, bar=0.002)
    assert s["confirmed"] is False


def test_bootstrap_ci_is_seed_deterministic() -> None:
    d = np.linspace(-0.01, 0.02, 27)
    assert ss._bootstrap_ci(d, 5000, 0) == ss._bootstrap_ci(d, 5000, 0)


def test_holm_bonferroni_matches_reference() -> None:
    # Reference values from the step-down definition.
    assert ss.holm_bonferroni([]) == []
    assert ss.holm_bonferroni([0.04]) == pytest.approx([0.04])
    # Monotone, clamped at 1.0; smallest p scaled by m, enforced non-decreasing.
    adj = ss.holm_bonferroni([0.01, 0.02, 0.03])
    assert adj == pytest.approx([0.03, 0.04, 0.04])
    assert all(0.0 <= a <= 1.0 for a in ss.holm_bonferroni([0.5, 0.4, 0.9, 0.045]))
    # Adjusted p-values are never below the raw p-values (correction is upward).
    raw = [0.001, 0.2, 0.045, 0.6, 0.31, 0.69]
    assert all(a >= r for a, r in zip(ss.holm_bonferroni(raw), raw))


@pytest.mark.skipif(
    not (RULEGRAPH.exists() and SEMANTICS.exists()),
    reason="sweep artifacts not present (run benchmarks.symbolic_*_sweep)",
)
def test_real_artifacts_corroborate_keep_decisions() -> None:
    result = ss.analyze(RULEGRAPH, SEMANTICS, n_boot=5000, seed=0)
    # Both headline alternatives must remain unconfirmed on the committed
    # 27-cell sweep — matching PR #265's KEEP-consensus / KEEP-product calls.
    assert result["rulegraph_salience_vs_consensus"]["confirmed"] is False
    assert result["semantics_godel_vs_product"]["confirmed"] is False
    assert result["conclusion"]["salience_beats_consensus"] is False
    assert result["conclusion"]["godel_beats_product"] is False
    # Structure / provenance present for reproducibility.
    assert result["method"]["bootstrap_seed"] == 0
    assert result["rulegraph_salience_vs_consensus"]["n_cells"] == 27


@pytest.mark.skipif(
    not (RULEGRAPH.exists() and SEMANTICS.exists()),
    reason="sweep artifacts not present (run benchmarks.symbolic_*_sweep)",
)
def test_familywise_correction_demotes_marginal_neural_win() -> None:
    """The single comparison that clears the uncorrected bar
    (godel-vs-neural, p≈0.045) must NOT survive Holm-Bonferroni across the
    6-test family — otherwise the report would over-claim 'co-training beats
    neural-only' from a finding that is an artefact of multiple testing."""
    result = ss.analyze(RULEGRAPH, SEMANTICS, n_boot=5000, seed=0)
    god_neu = result["semantics_godel_vs_neural"]
    assert god_neu["significant_at_0.05"] is True  # raw single-test
    assert god_neu["p_value_ttest_holm"] > god_neu["p_value_ttest"]  # correction is upward
    assert god_neu["confirmed_familywise"] is False  # dissolves under correction
    # The report must surface this explicitly and confirm nothing family-wise.
    assert result["conclusion"]["familywise_confirmed"] == []
    assert "semantics_godel_vs_neural" in result["conclusion"]["confirmed_uncorrected_only"]
    assert result["method"]["multiple_comparison_correction"] == "holm-bonferroni"
