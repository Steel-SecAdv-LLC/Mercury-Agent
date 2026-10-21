# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""The three observatory instruments measure real mathematics, pinned to literature values."""

from __future__ import annotations

import math

import pytest

import omni_mercury_engine.observatory.zeta_spectrum as zeta_module
from omni_mercury_engine.observatory import (
    CollatzRecordsInstrument,
    GoldbachCometInstrument,
    ZetaSpectrumInstrument,
)
from omni_mercury_engine.observatory.goldbach_comet import TWIN_PRIME_CONSTANT
from omni_mercury_engine.observatory.zeta_spectrum import gue_spacing_cdf


class TestGueSpacingCdf:
    """The closed-form GUE CDF against an independent numeric integration."""

    def test_matches_numeric_integration_of_the_surmise(self) -> None:
        """F(s) equals the trapezoid integral of P(s) to 6 decimals."""
        for s_target in (0.3, 0.7, 1.0, 1.5, 2.3):
            steps = 20000
            total = 0.0
            for i in range(steps):
                s = s_target * (i + 0.5) / steps
                total += (32 / math.pi**2) * s * s * math.exp(-4 * s * s / math.pi)
            integral = total * s_target / steps
            assert gue_spacing_cdf(s_target) == pytest.approx(integral, abs=1e-6)

    def test_boundaries(self) -> None:
        """F(0) = 0, F grows to 1, negative spacing refused."""
        assert gue_spacing_cdf(0.0) == 0.0
        assert gue_spacing_cdf(5.0) == pytest.approx(1.0, abs=1e-9)
        with pytest.raises(ValueError, match="negative"):
            gue_spacing_cdf(-0.1)


class TestZetaSpectrum:
    """Spacing statistics of genuine zeta zeros."""

    def test_first_gap_matches_known_ordinates(self) -> None:
        """The first two zeros are at 14.134725/21.022040; the unfolded gap is pinned."""
        result = ZetaSpectrumInstrument().observe(10, top_k=9)
        assert result["status"] == "grounded"
        pair_one = next(p for p in result["closest_pairs"] if p["lower_zero_index"] == 1)
        assert pair_one["lower_ordinate"] == pytest.approx(14.134725141734693, abs=1e-9)
        assert pair_one["upper_ordinate"] == pytest.approx(21.022039638771554, abs=1e-9)
        gap = 21.022039638771554 - 14.134725141734693
        expected = gap * math.log(14.134725141734693 / (2 * math.pi)) / (2 * math.pi)
        assert pair_one["normalized_spacing"] == pytest.approx(expected, abs=1e-9)

    def test_mean_spacing_near_one_and_ks_bounded(self) -> None:
        """Unfolded spacings have mean near 1 and a KS distance in [0, 1]."""
        result = ZetaSpectrumInstrument().observe(30)
        assert result["mean_normalized_spacing"] == pytest.approx(1.0, abs=0.2)
        assert 0.0 <= result["ks_distance_to_gue"] <= 1.0

    def test_close_pair_threshold_filters(self) -> None:
        """Every reported anomaly is below the threshold; pairs are sorted ascending."""
        result = ZetaSpectrumInstrument().observe(30, close_pair_threshold=0.6, top_k=3)
        spacings = [p["normalized_spacing"] for p in result["closest_pairs"]]
        assert spacings == sorted(spacings)
        assert len(result["closest_pairs"]) == 3
        assert all(p["normalized_spacing"] < 0.6 for p in result["close_pair_anomalies"])

    def test_bounds_refused(self) -> None:
        """num_zeros outside 2..100 and bad knobs are refused."""
        instrument = ZetaSpectrumInstrument()
        with pytest.raises(ValueError, match="num_zeros"):
            instrument.observe(1)
        with pytest.raises(ValueError, match="num_zeros"):
            instrument.observe(101)
        with pytest.raises(ValueError, match="top_k"):
            instrument.observe(10, top_k=0)

    def test_fails_closed_without_mpmath(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No oracle: zero zeros, NaN statistics, honest note -- no fabrication."""
        monkeypatch.setattr(zeta_module, "MPMATH_AVAILABLE", False)
        result = ZetaSpectrumInstrument().observe(10)
        assert result["status"] == "unavailable"
        assert result["num_zeros"] == 0
        assert math.isnan(result["ks_distance_to_gue"])
        assert result["closest_pairs"] == []


class TestGoldbachComet:
    """Partition counts, the comet's bands, and the Hardy--Littlewood collapse."""

    def test_exact_partition_counts(self) -> None:
        """r(10)=2, r(12)=1, r(100)=6 -- textbook values, counted exactly."""
        instrument = GoldbachCometInstrument()
        is_prime = instrument._sieve(100)
        primes = [p for p in range(2, 101) if is_prime[p]]

        def count(n: int) -> int:
            return sum(1 for p in primes if p <= n // 2 and is_prime[n - p])

        assert count(10) == 2  # 3+7, 5+5
        assert count(12) == 1  # 5+7
        assert count(100) == 6  # 3+97, 11+89, 17+83, 29+71, 41+59, 47+53

    def test_hardy_littlewood_singular_series(self) -> None:
        """The prediction carries exactly the odd-prime-divisor product."""
        hl = GoldbachCometInstrument._hardy_littlewood
        assert hl(16) == pytest.approx(TWIN_PRIME_CONSTANT * 16 / math.log(16) ** 2)
        assert hl(12) == pytest.approx(2.0 * TWIN_PRIME_CONSTANT * 12 / math.log(12) ** 2)
        # 30 = 2*3*5: product = 2 * (4/3) = 8/3.
        assert hl(30) == pytest.approx((8 / 3) * TWIN_PRIME_CONSTANT * 30 / math.log(30) ** 2)
        # 50 = 2*5^2: the product counts each odd prime once.
        assert hl(50) == pytest.approx((4 / 3) * TWIN_PRIME_CONSTANT * 50 / math.log(50) ** 2)
        with pytest.raises(ValueError, match="even"):
            hl(15)

    def test_comet_banding_is_measured(self) -> None:
        """The 3|n band sits well above the others -- the comet's visible structure."""
        result = GoldbachCometInstrument().observe(2000)
        bands = result["band_density_by_mod3"]
        assert bands[0] > 1.5 * bands[1]
        assert bands[0] > 1.5 * bands[2]

    def test_no_counterexamples_below_limit(self) -> None:
        """Every even number up to the limit has a partition (verified to 4e18 in the literature)."""
        result = GoldbachCometInstrument().observe(2000)
        assert result["counterexample_candidates"] == []
        assert result["status"] == "grounded"
        assert result["evens_checked"] == (2000 - 4) // 2 + 1

    def test_outliers_sorted_and_bounded(self) -> None:
        """top_k outliers come largest-deviation first."""
        result = GoldbachCometInstrument().observe(1000, top_k=4)
        deviations = [o["abs_log_ratio"] for o in result["ratio_outliers"]]
        assert len(deviations) == 4
        assert deviations == sorted(deviations, reverse=True)

    def test_bounds_refused(self) -> None:
        """Too-small limits and bad knobs are refused."""
        with pytest.raises(ValueError, match="limit"):
            GoldbachCometInstrument().observe(50)
        with pytest.raises(ValueError, match="top_k"):
            GoldbachCometInstrument().observe(1000, top_k=0)


class TestCollatzRecords:
    """Exhaustive trajectories pinned to the classic record ladder."""

    def test_27_record(self) -> None:
        """27 takes 111 steps and peaks at 9,232 -- the textbook record."""
        result = CollatzRecordsInstrument().observe(2000)
        record = next(r for r in result["stopping_time_records"] if r["n"] == 27)
        assert record["stopping_time"] == 111
        assert record["peak"] == 9232
        assert record["anomaly_score"] == pytest.approx(
            111 / (2.0 / math.log(4 / 3) * math.log(27))
        )

    def test_records_strictly_increase(self) -> None:
        """Each stopping-time record beats every earlier one; same for peaks."""
        result = CollatzRecordsInstrument().observe(5000)
        times = [r["stopping_time"] for r in result["stopping_time_records"]]
        peaks = [r["peak"] for r in result["excursion_records"]]
        assert times == sorted(times) and len(set(times)) == len(times)
        assert peaks == sorted(peaks) and len(set(peaks)) == len(peaks)
        assert result["counterexample_candidates"] == []

    def test_budget_exhaustion_is_a_candidate_not_a_crash(self) -> None:
        """A tiny budget turns unfinished trajectories into counterexample candidates."""
        result = CollatzRecordsInstrument().observe(30, step_budget=5)
        assert 27 in result["counterexample_candidates"]
        assert result["status"] == "grounded"

    def test_bounds_refused(self) -> None:
        """Too-small limits and budgets are refused."""
        with pytest.raises(ValueError, match="limit"):
            CollatzRecordsInstrument().observe(5)
        with pytest.raises(ValueError, match="step_budget"):
            CollatzRecordsInstrument().observe(100, step_budget=0)
