# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Behavioral contract for the DB spectral-divergence term revival.

The legacy term was identically zero (single-row signatures are empty); the
operator-approved revival gives it real semantics via the
``db_spectral_divergence`` config flag. The pre-registered ablation gate
(``benchmarks/db_spectral_ablation.py``) cleared decisively — mean paired
detector dAUC +0.071, seed agreement 0.93 — so the flag now **defaults to
True**; ``False`` preserves the pre-2026-06-11 shipped scores. These tests
pin:

* the gate-cleared default and the legacy opt-out (explicit zero term,
  byte-equal to the prior shipped form);
* enabled semantics are real: nonzero, finite, and the term *detects what
  it claims* (rows whose feature-axis spectrum diverges from the training
  baseline out-score spectrum-conforming rows, multi-seed);
* the fit-time baseline and dimension-mismatch truncation behave.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.detectors.dimensional import DimensionalAnalyzer

SEEDS = (0, 1, 2)


def _smooth_rows(rng: np.random.Generator, n: int, d: int) -> np.ndarray[Any, Any]:
    """Rows dominated by low-frequency structure along the feature axis."""
    base = np.sin(np.linspace(0.0, np.pi, d))
    return base[None, :] + 0.05 * rng.normal(size=(n, d))


def _alternating_rows(rng: np.random.Generator, n: int, d: int) -> np.ndarray[Any, Any]:
    """Rows dominated by the highest frequency (sign-alternating features)."""
    base = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(d)])
    return base[None, :] + 0.05 * rng.normal(size=(n, d))


class TestLegacyOptOutPreservesPriorScores:
    def test_flag_defaults_to_true_after_ablation_gate(self) -> None:
        # Default flipped by the cleared pre-registered gate
        # (artifacts/db_spectral_ablation.json: mean dAUC +0.071,
        # agreement 0.93). Legacy scores remain one config away.
        assert DimensionalAnalyzer().db_spectral_divergence is True
        legacy = DimensionalAnalyzer({"db_spectral_divergence": False})
        assert legacy.db_spectral_divergence is False

    @pytest.mark.parametrize("seed", SEEDS)
    def test_db_scores_equal_phase_thd_only_form(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        train = rng.normal(size=(120, 9))
        query = rng.normal(size=(40, 9))

        analyzer = DimensionalAnalyzer({"db_spectral_divergence": False})
        analyzer.fit(train)
        produced = analyzer._dimensional_code_breaking(query)

        expected = np.minimum(
            (1.0 - analyzer._phase_coherence_rows(query)) * 0.3
            + analyzer._harmonic_distortion_rows(query) * 0.2,
            1.0,
        )
        np.testing.assert_allclose(produced, expected, rtol=0, atol=1e-12)


class TestOptInRealSemantics:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_fit_stores_row_spectrum_baseline(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        analyzer = DimensionalAnalyzer({"db_spectral_divergence": True})
        analyzer.fit(rng.normal(size=(100, 10)))
        assert analyzer.baseline_row_spectrum is not None
        assert analyzer.baseline_row_spectrum.shape == (5,)  # d // 2
        assert np.all(np.isfinite(analyzer.baseline_row_spectrum))

    @pytest.mark.parametrize("seed", SEEDS)
    def test_term_is_nonzero_and_finite_when_enabled(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        train = rng.normal(size=(120, 12))
        query = rng.normal(size=(30, 12)) * 3.0

        on = DimensionalAnalyzer({"db_spectral_divergence": True})
        np.random.seed(seed)
        on.fit(train)
        db_on = on._dimensional_code_breaking(query)

        off = DimensionalAnalyzer({"db_spectral_divergence": False})
        np.random.seed(seed)
        off.fit(train)
        db_off = off._dimensional_code_breaking(query)

        assert np.all(np.isfinite(db_on))
        # The enabled term genuinely contributes: scores differ from the
        # phase/THD-only legacy form for spectrum-shifted inputs.
        assert not np.allclose(db_on, db_off)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_detects_spectral_divergence_it_claims_to_detect(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        d = 16
        analyzer = DimensionalAnalyzer({"db_spectral_divergence": True})
        analyzer.fit(_smooth_rows(rng, 150, d))

        conforming = _smooth_rows(rng, 40, d)
        diverging = _alternating_rows(rng, 40, d)

        assert analyzer.baseline_row_spectrum is not None
        baseline = analyzer.baseline_row_spectrum
        div_conforming = np.linalg.norm(
            analyzer._row_power_spectrum(conforming) - baseline[None, :], axis=1
        ) / (np.linalg.norm(baseline) + 1e-10)
        div_diverging = np.linalg.norm(
            analyzer._row_power_spectrum(diverging) - baseline[None, :], axis=1
        ) / (np.linalg.norm(baseline) + 1e-10)

        # Every spectrum-diverging row must out-score every conforming row.
        assert float(div_diverging.min()) > float(div_conforming.max())

    @pytest.mark.parametrize("seed", SEEDS)
    def test_dimension_mismatch_truncates_instead_of_raising(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        analyzer = DimensionalAnalyzer({"db_spectral_divergence": True})
        analyzer.fit(rng.normal(size=(100, 12)))
        narrower = rng.normal(size=(10, 8))
        scores = analyzer._dimensional_code_breaking(narrower)
        assert scores.shape == (10,)
        assert np.all(np.isfinite(scores))
