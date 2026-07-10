# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the consciousness_field REG-deviation training pipeline.

Offline by design: the committed fixture is a REAL day-file excerpt --
provenance: ``https://noosphere.princeton.edu/data/eggsummary/2015/
basketdata-2015-01-01.csv.gz`` as archived by the Internet Archive Wayback
Machine, capture ``20250109082759`` (raw bytes via
``https://web.archive.org/web/20250109082759id_/https://noosphere.princeton
.edu/data/eggsummary/2015/basketdata-2015-01-01.csv.gz``), truncated to the
header records (types 10/11/12) plus the first 3600 data rows (the first
hour of 2015-01-01 UTC, 41 eggs). The full day file has 3,292,812
egg-second trials with mean 99.9993 and sd 7.0732 versus the
Binomial(200, 0.5) theory values 100 and 7.0711. No network, no synthetic
"GCP" data; pipeline-stage network code runs in the training lane, not here.

Covered: the basketdata CSV v2 parser, the Stouffer per-second composite,
window validity rules (>=10 eggs/second, no timestamp gaps), the three
documented fault channels (bias mean-shift expectation, common-mode netvar
inflation, stuck-bit exactness, seeded determinism), temporal-split
enforcement, the wrapped-payload ``load_neural_weights`` contract, and the
differential closed-form-vs-shipped comparison (skipped until a
``reg_deviation_gcp`` checkpoint ships through the merit gate).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from omni_mercury_engine.ml.hazard_training import consciousness_field as cf
from omni_mercury_engine.models.checkpoint_paths import shipped_checkpoint_path
from omni_mercury_engine.models.gcp_ingest import egg_sums_to_z, network_variance
from omni_mercury_engine.models.parapsychology import ParapsychologyDetector

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "hazard_training"
    / "consciousness_field"
    / "basketdata-2015-01-01-excerpt.csv.gz"
)


@pytest.fixture(name="day")
def _day() -> cf.BasketDay:
    """Parse the committed real day-file excerpt (see module docstring)."""
    assert FIXTURE.exists(), f"missing committed fixture {FIXTURE}"
    return cf.parse_basketdata(FIXTURE)


class TestParser:
    """The basketdata CSV v2 parser recovers the real trial matrix."""

    def test_shape_and_ids(self, day: cf.BasketDay) -> None:
        assert day.egg_sums.shape == (3600, 41)
        assert len(day.egg_ids) == 41
        assert day.egg_ids[0] == "1"

    def test_epochs_contiguous_from_day_start(self, day: cf.BasketDay) -> None:
        assert int(day.epochs[0]) == 1420070400  # 2015-01-01 00:00:00 UTC
        assert np.all(np.diff(day.epochs) == 1)

    def test_offline_eggs_are_nan(self, day: cf.BasketDay) -> None:
        """Empty CSV fields (egg offline) must become NaN, not zeros."""
        assert int(np.isnan(day.egg_sums).sum()) > 0
        finite = day.egg_sums[np.isfinite(day.egg_sums)]
        assert float(finite.min()) >= 0.0 and float(finite.max()) <= 200.0

    def test_null_distribution_matches_binomial_theory(self, day: cf.BasketDay) -> None:
        """Real measured trials sit on the Binomial(200, 0.5) null."""
        finite = day.egg_sums[np.isfinite(day.egg_sums)]
        assert abs(float(finite.mean()) - 100.0) < 0.5
        assert abs(float(finite.std()) - 7.0711) < 0.15


class TestCompositeAndWindows:
    """Per-second Stouffer composite and window validity rules."""

    def test_composite_is_standard_normal_ish(self, day: cf.BasketDay) -> None:
        comp, counts = cf.stouffer_composite(day.egg_sums)
        assert comp.shape == (3600,)
        assert int(counts.min()) >= cf.MIN_EGGS_PER_SECOND
        assert abs(float(np.nanmean(comp))) < 0.1
        assert 0.9 < float(np.nanstd(comp)) < 1.1

    def test_valid_windows_cover_the_hour(self, day: cf.BasketDay) -> None:
        comp, counts = cf.stouffer_composite(day.egg_sums)
        starts = cf._valid_window_starts(day.epochs, counts)
        assert list(starts) == list(range(0, 3600, cf.WINDOW_SECONDS))

    def test_timestamp_gap_invalidates_window(self, day: cf.BasketDay) -> None:
        epochs = day.epochs.copy()
        epochs[150:] += 5  # 5 s gap inside the second window
        _, counts = cf.stouffer_composite(day.egg_sums)
        starts = cf._valid_window_starts(epochs, counts)
        assert 100 not in starts and 0 in starts

    def test_low_egg_second_invalidates_window(self, day: cf.BasketDay) -> None:
        sums = day.egg_sums.copy()
        sums[250, cf.MIN_EGGS_PER_SECOND - 1 :] = np.nan  # 9 eggs at second 250
        _, counts = cf.stouffer_composite(sums)
        starts = cf._valid_window_starts(day.epochs, counts)
        assert 200 not in starts and 0 in starts


class TestFaultChannels:
    """The documented hardware-failure channels have their stated math."""

    def test_bias_mean_shift_expectation(self, day: cf.BasketDay) -> None:
        """The bias channel shifts the mean by (200 - E[v]) * q exactly in expectation."""
        q = 0.02
        faulted = cf.apply_bias_fault(day.egg_sums, q, np.random.default_rng(7))
        shift = float(np.nanmean(faulted - day.egg_sums))
        expected = q * float(np.nanmean(200.0 - day.egg_sums))
        # ~131k egg-seconds: the Monte Carlo error of the mean is ~0.005.
        assert abs(shift - expected) < 0.05
        assert np.isnan(faulted[np.isnan(day.egg_sums)]).all()

    def test_bias_is_deterministic_under_seed(self, day: cf.BasketDay) -> None:
        a = cf.apply_bias_fault(day.egg_sums, 0.01, np.random.default_rng(3))
        b = cf.apply_bias_fault(day.egg_sums, 0.01, np.random.default_rng(3))
        c = cf.apply_bias_fault(day.egg_sums, 0.01, np.random.default_rng(4))
        assert np.array_equal(a, b, equal_nan=True)
        assert not np.array_equal(a, c, equal_nan=True)

    def test_common_mode_inflates_network_variance(self, day: cf.BasketDay) -> None:
        """Correlated same-sign bias must inflate the classic GCP statistic."""
        for sign in (-1, 1):
            faulted = cf.apply_common_mode_fault(
                day.egg_sums, 0.05, sign, np.random.default_rng(11)
            )
            nv_null = float(np.nanmean(network_variance(egg_sums_to_z(day.egg_sums))))
            nv_fault = float(np.nanmean(network_variance(egg_sums_to_z(faulted))))
            assert nv_fault > nv_null * 1.2, f"sign={sign}"

    def test_common_mode_sign_direction(self, day: cf.BasketDay) -> None:
        up = cf.apply_common_mode_fault(day.egg_sums, 0.02, +1, np.random.default_rng(5))
        down = cf.apply_common_mode_fault(day.egg_sums, 0.02, -1, np.random.default_rng(5))
        assert float(np.nanmean(up)) > float(np.nanmean(day.egg_sums))
        assert float(np.nanmean(down)) < float(np.nanmean(day.egg_sums))

    def test_stuck_bit_formula_is_exact(self) -> None:
        v = np.array([0.0, 100.0, 200.0])
        out = cf.apply_stuck_bit_fault(v, 10)
        np.testing.assert_allclose(out, np.round(v * 190 / 200) + 10)

    def test_fault_injection_is_seed_deterministic(self, day: cf.BasketDay) -> None:
        window = day.egg_sums[:100]
        a = cf._inject_fault(window, day.egg_ids, np.random.default_rng([1, 2]))
        b = cf._inject_fault(window, day.egg_ids, np.random.default_rng([1, 2]))
        assert a[1:] == b[1:]  # family, param, sign, egg
        assert np.array_equal(a[0], b[0], equal_nan=True)


class TestSplitAndScore:
    """Temporal-split enforcement and the pre-registered closed-form score."""

    def test_split_is_by_year_and_ordered(self) -> None:
        assert cf.SPLIT.train_years == tuple(range(2012, 2019))
        assert cf.SPLIT.val_years == (2019, 2020, 2021)
        assert cf.SPLIT.test_years == (2022, 2023, 2024)
        assert max(cf.SPLIT.train_years) < min(cf.SPLIT.val_years)
        assert max(cf.SPLIT.val_years) < min(cf.SPLIT.test_years)

    def test_closed_form_score_flags_fault_over_null(self, day: cf.BasketDay) -> None:
        comp, _ = cf.stouffer_composite(day.egg_sums)
        null_score = cf.closed_form_score(comp[:100])
        faulted = cf.apply_common_mode_fault(day.egg_sums[:100], 0.05, +1, np.random.default_rng(9))
        fault_comp, _ = cf.stouffer_composite(faulted)
        fault_score = cf.closed_form_score(fault_comp)
        assert np.isfinite(fault_score)
        assert fault_score > null_score + 5.0  # decades of -log10 p apart


class TestDetectorContract:
    """load_neural_weights handles wrapped payloads and the shipped file."""

    def test_wrapped_payload_unwraps(self) -> None:
        det = ParapsychologyDetector(enable_consciousness_field=True)
        assert det.field_analyzer is not None
        payload = {"field_analyzer": det.field_analyzer.state_dict(), "extra": 1}
        det.load_neural_weights(payload)
        assert det._neural_trained is True

    @pytest.mark.skipif(
        not shipped_checkpoint_path(cf.CHECKPOINT_NAME).exists(),
        reason="no shipped reg_deviation_gcp checkpoint (merit gate not passed)",
    )
    def test_shipped_checkpoint_beats_chance_on_fixture_faults(self, day: cf.BasketDay) -> None:
        """Differential: closed-form vs shipped weights on identical windows.

        Both paths score the fixture's real null windows against their
        common-mode-faulted twins; each must separate the classes, and the
        learned scores must go the same direction as the closed-form rule.
        """
        comp, counts = cf.stouffer_composite(day.egg_sums)
        starts = cf._valid_window_starts(day.epochs, counts)
        det = ParapsychologyDetector(enable_consciousness_field=True)
        det.load_neural_weights(None)  # shipped reg_deviation_gcp

        learned_null, learned_fault = [], []
        physics_null, physics_fault = [], []
        for s in starts[:12]:
            window = day.egg_sums[s : s + 100]
            rng = np.random.default_rng([13, int(s)])
            faulted = cf.apply_common_mode_fault(window, 0.05, +1, rng)
            for raw, l_acc, p_acc in (
                (window, learned_null, physics_null),
                (faulted, learned_fault, physics_fault),
            ):
                c, _ = cf.stouffer_composite(raw)
                result = det.detect_psi_anomaly({"reg_output": c.astype(np.float64)})
                assert result.coherence_score is not None
                l_acc.append(float(result.coherence_score))
                p_acc.append(cf.closed_form_score(c))
        assert float(np.mean(learned_fault)) > float(np.mean(learned_null))
        assert float(np.mean(physics_fault)) > float(np.mean(physics_null))
