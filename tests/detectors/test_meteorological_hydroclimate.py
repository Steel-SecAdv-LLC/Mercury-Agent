# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the hydro-climate meteorological detectors (T4b1).

Drought (SPI/SPEI, McKee 1993 / Vicente-Serrano 2010), heatwave
(Perkins & Alexander 2013 percentile definition + NWS Rothfusz heat index),
atmospheric river (IVT + Ralph et al. 2019 scale), lightning (Schultz et al.
2009 2-sigma jump), and the hurricane->surge->flood cascade. Every numeric
check is anchored to a hand-computable construction or a published worked
example; constructed series are labelled as constructed.
"""

from __future__ import annotations

import datetime as dt
import itertools

import numpy as np
import pytest

from omni_mercury_engine.detectors.meteorological.atmospheric_river_detector import (
    AtmosphericRiverDetector,
    compute_ivt,
)
from omni_mercury_engine.detectors.meteorological.drought_detector import (
    DroughtCategory,
    DroughtDetector,
    classify_usdm,
    compute_spi,
    fit_gamma_thom,
    thornthwaite_pet,
)
from omni_mercury_engine.detectors.meteorological.heatwave_detector import (
    HeatwaveDetector,
    heat_index_f,
)
from omni_mercury_engine.detectors.meteorological.lightning_detector import (
    LightningDetector,
)
from omni_mercury_engine.detectors.meteorological.surge_flood_cascade import (
    CascadeStage,
    SurgeFloodCascade,
)

RNG = np.random.default_rng(42)


# =============================================================================
# Drought
# =============================================================================


class TestGammaFit:
    def test_thom_recovers_known_gamma(self) -> None:
        """Thom's approximation recovers shape/scale of a large gamma sample."""
        sample = RNG.gamma(shape=2.5, scale=30.0, size=20000)
        shape, scale = fit_gamma_thom(sample)
        assert shape == pytest.approx(2.5, rel=0.05)
        assert scale == pytest.approx(30.0, rel=0.05)


class TestSPI:
    def test_spi_is_standard_normal_over_baseline(self) -> None:
        """SPI of the fitting sample itself is ~N(0,1) (McKee's definition)."""
        precip = RNG.gamma(shape=2.0, scale=40.0, size=600)
        spi = compute_spi(precip, window_months=3)
        assert spi.shape == (598,)
        assert float(np.mean(spi)) == pytest.approx(0.0, abs=0.05)
        assert float(np.std(spi)) == pytest.approx(1.0, abs=0.08)

    def test_dry_spell_scores_strongly_negative(self) -> None:
        precip = RNG.gamma(shape=2.0, scale=40.0, size=240)
        precip[-6:] = 0.0  # constructed six-month total drought
        spi = compute_spi(precip, window_months=6)
        assert spi[-1] < -1.6, "an all-dry 6-month window must be extreme drought"

    def test_zero_probability_mass_handled(self) -> None:
        """The H(x)=q+(1-q)G(x) mixed distribution keeps zeros finite."""
        precip = RNG.gamma(shape=2.0, scale=30.0, size=400)
        precip[::7] = 0.0
        spi = compute_spi(precip, window_months=1)
        assert np.all(np.isfinite(spi))

    def test_negative_precip_rejected(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            compute_spi(np.array([1.0, -2.0, 3.0] * 40), window_months=1)

    def test_short_series_fails_loud(self) -> None:
        with pytest.raises(ValueError, match="needs >="):
            compute_spi(np.full(10, 25.0), window_months=3)

    def test_all_dry_climatology_fails_loud(self) -> None:
        with pytest.raises(ValueError, match="all-dry"):
            compute_spi(np.zeros(120), window_months=1)


class TestThornthwaitePET:
    def test_all_frozen_year_is_undefined(self) -> None:
        """Thornthwaite's annual heat index I=0 -> PET undefined, fail loud."""
        temps = np.full(12, -5.0)
        months = np.arange(1, 13)
        with pytest.raises(ValueError, match="undefined"):
            thornthwaite_pet(temps, 45.0, months)

    def test_frozen_months_evaporate_nothing(self) -> None:
        temps = np.array([-5.0, -3.0, 2.0, 8.0, 14.0, 18.0, 21.0, 20.0, 15.0, 9.0, 3.0, -2.0])
        months = np.arange(1, 13)
        pet = thornthwaite_pet(temps, 45.0, months)
        assert pet[0] == 0.0 and pet[1] == 0.0 and pet[11] == 0.0
        assert pet[6] > 0.0

    def test_warm_month_magnitude_plausible(self) -> None:
        """Thornthwaite at 20 degC mid-latitude summer: O(100 mm)/month."""
        temps = np.array([0.0, 1.0, 5.0, 9.0, 14.0, 18.0, 20.0, 19.0, 15.0, 10.0, 5.0, 1.0])
        months = np.arange(1, 13)
        pet = thornthwaite_pet(temps, 40.0, months)
        july = pet[6]
        assert 60.0 < july < 160.0
        assert pet[0] == 0.0  # 0 degC month evaporates nothing in Thornthwaite


class TestUSDMClassification:
    @pytest.mark.parametrize(
        ("spi", "expected"),
        [
            (0.0, DroughtCategory.NONE),
            (-0.5, DroughtCategory.D0_ABNORMALLY_DRY),
            (-0.8, DroughtCategory.D1_MODERATE_DROUGHT),
            (-1.3, DroughtCategory.D2_SEVERE_DROUGHT),
            (-1.6, DroughtCategory.D3_EXTREME_DROUGHT),
            (-2.0, DroughtCategory.D4_EXCEPTIONAL_DROUGHT),
            (-3.5, DroughtCategory.D4_EXCEPTIONAL_DROUGHT),
        ],
    )
    def test_svoboda_thresholds(self, spi: float, expected: DroughtCategory) -> None:
        assert classify_usdm(spi) is expected

    def test_non_finite_rejected(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            classify_usdm(float("nan"))


class TestDroughtDetector:
    def test_assess_flags_imposed_drought(self) -> None:
        n_years = 35  # per-calendar-month strata need ~30 samples each
        months = np.tile(np.arange(1, 13), n_years)
        precip = RNG.gamma(shape=2.0, scale=40.0, size=12 * n_years)
        precip[-12:] *= 0.05  # constructed final-year collapse
        detector = DroughtDetector()
        result = detector.assess(precip, month_numbers=months)
        assert result.drought_detected is True
        order = [c.value for c in DroughtCategory]
        category = (
            result.category.value
            if isinstance(result.category, DroughtCategory)
            else result.category
        )
        assert order.index(category) >= order.index(DroughtCategory.D1_MODERATE_DROUGHT.value)
        assert np.isfinite(result.spi_latest[min(detector.windows_months)])

    def test_spei_requires_latitude_and_months(self) -> None:
        detector = DroughtDetector()
        precip = RNG.gamma(shape=2.0, scale=40.0, size=420)
        with pytest.raises(ValueError, match="latitude_deg"):
            detector.assess(precip, monthly_temp_c=np.full(420, 15.0))


# =============================================================================
# Heatwave
# =============================================================================


class TestHeatIndex:
    def test_nws_worked_example(self) -> None:
        """NWS chart: T=95 degF, RH=55%% -> HI ~ 110 degF."""
        assert heat_index_f(95.0, 55.0) == pytest.approx(110.0, abs=2.0)

    def test_cool_conditions_use_simple_formula(self) -> None:
        hi = heat_index_f(70.0, 50.0)
        assert abs(hi - 70.0) < 5.0

    def test_monotonic_in_humidity_when_hot(self) -> None:
        values = [heat_index_f(96.0, rh) for rh in (40.0, 55.0, 70.0, 85.0)]
        assert all(b > a for a, b in itertools.pairwise(values))

    def test_range_guards(self) -> None:
        with pytest.raises(ValueError):
            heat_index_f(200.0, 50.0)
        with pytest.raises(ValueError):
            heat_index_f(95.0, 130.0)


def _baseline_series(years: int = 5) -> tuple[list[dt.date], np.ndarray, np.ndarray]:
    """Constructed sinusoidal Tmax/Tmin climatology with seeded noise."""
    dates: list[dt.date] = []
    day = dt.date(2015, 1, 1)
    end = dt.date(2015 + years, 1, 1)
    while day < end:
        dates.append(day)
        day += dt.timedelta(days=1)
    doy = np.array([d.timetuple().tm_yday for d in dates], dtype=np.float64)
    tmax = 20.0 + 10.0 * np.sin(2.0 * np.pi * (doy - 105.0) / 365.0)
    tmax = tmax + RNG.normal(0.0, 1.5, size=tmax.size)
    tmin = tmax - 10.0
    return dates, tmax, tmin


class TestHeatwaveDetector:
    def test_detects_imposed_five_day_excursion(self) -> None:
        dates, tmax, tmin = _baseline_series()
        detector = HeatwaveDetector()
        detector.fit_baseline(dates, tmax, tmin_c=tmin)

        test_dates = [dt.date(2021, 7, d) for d in range(1, 21)]
        doy = np.array([d.timetuple().tm_yday for d in test_dates], dtype=np.float64)
        test_tmax = 20.0 + 10.0 * np.sin(2.0 * np.pi * (doy - 105.0) / 365.0)
        test_tmax[7:12] += 12.0  # constructed 5-day extreme excursion
        result = detector.detect_heatwaves(test_dates, test_tmax, tmin_c=test_tmax - 10.0)

        assert len(result.events) == 1
        event = result.events[0]
        assert event.duration_days >= 3
        assert event.max_exceedance_c > 5.0
        assert event.mean_tmax_c > float(np.median(test_tmax))

    def test_quiet_window_has_no_events(self) -> None:
        dates, tmax, tmin = _baseline_series()
        detector = HeatwaveDetector()
        detector.fit_baseline(dates, tmax, tmin_c=tmin)
        test_dates = [dt.date(2021, 7, d) for d in range(1, 21)]
        doy = np.array([d.timetuple().tm_yday for d in test_dates], dtype=np.float64)
        quiet = 20.0 + 10.0 * np.sin(2.0 * np.pi * (doy - 105.0) / 365.0) - 2.0
        result = detector.detect_heatwaves(test_dates, quiet)
        assert result.events == []

    def test_unfitted_fails_loud(self) -> None:
        detector = HeatwaveDetector()
        with pytest.raises((ValueError, RuntimeError)):
            detector.detect_heatwaves([dt.date(2021, 7, 1)], np.array([30.0]))

    def test_short_baseline_rejected(self) -> None:
        detector = HeatwaveDetector()
        dates = [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(200)]
        with pytest.raises(ValueError, match="year"):
            detector.fit_baseline(dates, np.full(200, 20.0))


# =============================================================================
# Atmospheric river
# =============================================================================


class TestIVT:
    def test_constant_profile_matches_hand_integral(self) -> None:
        """Constant q,u over [1000,300] hPa: IVT = q*u*(dp)/g exactly."""
        p = np.array([1000.0, 850.0, 700.0, 500.0, 300.0])
        q = np.full(5, 0.01)
        u = np.full(5, 25.0)
        v = np.zeros(5)
        result = compute_ivt(q, u, v, p)
        expected = 0.01 * 25.0 * (1000.0 - 300.0) * 100.0 / 9.80665
        assert result.ivt[0] == pytest.approx(expected, rel=1e-6)
        assert result.ivt_v[0] == pytest.approx(0.0, abs=1e-9)

    def test_components_keep_flux_sign(self) -> None:
        p = np.array([1000.0, 700.0, 400.0])
        q = np.full(3, 0.008)
        u = np.full(3, -10.0)
        v = np.full(3, 10.0)
        result = compute_ivt(q, u, v, p)
        assert result.ivt_u[0] < 0.0 < result.ivt_v[0]

    def test_non_monotonic_levels_rejected(self) -> None:
        p = np.array([1000.0, 500.0, 700.0])
        with pytest.raises(ValueError):
            compute_ivt(np.full(3, 0.01), np.full(3, 10.0), np.zeros(3), p)


class TestRalphScale:
    def _detector(self) -> AtmosphericRiverDetector:
        return AtmosphericRiverDetector()

    def test_long_duration_promotes_rank(self) -> None:
        """Ralph 2019: same peak IVT, > 48 h promotes one rank."""
        detector = self._detector()
        short = detector.classify_ar_scale(np.full(6, 600.0), dt_hours=3.0)  # 18 h
        long = detector.classify_ar_scale(np.full(20, 600.0), dt_hours=3.0)  # 60 h
        assert long.episodes[0].final_rank > short.episodes[0].final_rank

    def test_sub_threshold_series_yields_no_episode(self) -> None:
        detector = self._detector()
        result = detector.classify_ar_scale(np.full(20, 100.0), dt_hours=3.0)
        assert result.episodes == []

    def test_peak_1300_long_duration_is_ar5(self) -> None:
        detector = self._detector()
        result = detector.classify_ar_scale(np.full(24, 1300.0), dt_hours=3.0)
        assert result.episodes[0].final_rank == 5
        assert result.episodes[0].label == "AR5"

    def test_time_spec_is_exclusive(self) -> None:
        detector = self._detector()
        with pytest.raises(ValueError, match="exactly one"):
            detector.classify_ar_scale(np.full(4, 300.0))


# =============================================================================
# Lightning
# =============================================================================


class TestLightningJump:
    def _flash_series(self, rates_per_2min: list[int]) -> np.ndarray:  # type: ignore[type-arg]
        """Constructed flash times realizing the given per-bin counts."""
        times: list[float] = []
        for i, count in enumerate(rates_per_2min):
            start = i * 120.0
            times.extend(np.linspace(start, start + 119.0, count).tolist())
        return np.asarray(times)

    def test_schultz_ramp_is_detected(self) -> None:
        """Steady 20 flashes/bin then a burst to 80 trips the 2-sigma jump."""
        detector = LightningDetector()
        flashes = self._flash_series([20, 21, 19, 20, 22, 20, 21, 80, 85])
        result = detector.detect_lightning_jumps(flashes)
        assert result.jumps, "the constructed burst must be flagged"
        assert result.jump_detected is True
        assert result.jumps[0].flash_rate_per_min >= detector.activation_rate_per_min
        assert result.severe_weather_precursor is True

    def test_steady_activity_has_no_jump(self) -> None:
        detector = LightningDetector()
        flashes = self._flash_series([20, 21, 19, 20, 22, 20, 21, 20, 19])
        result = detector.detect_lightning_jumps(flashes)
        assert result.jumps == []

    def test_low_rate_burst_below_activation_is_ignored(self) -> None:
        """A relative burst below 10 flashes/min stays sub-severe (paper rule)."""
        detector = LightningDetector()
        flashes = self._flash_series([2, 2, 1, 2, 2, 2, 2, 8, 9])
        result = detector.detect_lightning_jumps(flashes)
        assert result.jumps == []

    def test_short_series_fails_loud(self) -> None:
        detector = LightningDetector()
        with pytest.raises(ValueError, match="sigma history"):
            detector.detect_lightning_jumps(np.linspace(0.0, 200.0, 30))

    def test_cluster_cells_separates_two_storms(self) -> None:
        detector = LightningDetector()
        # Two storms placed at cell centres (0.15 deg grid) with jitter far
        # smaller than the half-cell, so each storm occupies exactly one bin.
        lats = np.concatenate([np.full(30, 35.075), np.full(40, 36.525)])
        lons = np.concatenate([np.full(30, -97.075), np.full(40, -95.475)])
        lats = lats + RNG.normal(0.0, 0.005, size=lats.size)
        lons = lons + RNG.normal(0.0, 0.005, size=lons.size)
        cells = detector.cluster_cells(lats, lons)
        assert len(cells) == 2
        assert sorted(c.flash_count for c in cells) == [30, 40]


# =============================================================================
# Hurricane -> surge -> flood cascade
# =============================================================================


def _coops_payload(values: list[float], key: str) -> dict[str, object]:
    """CO-OPS datagetter payload shape (product=water_level / predictions)."""
    rows = [{"t": f"2026-07-08 {i:02d}:00", "v": f"{v:.3f}"} for i, v in enumerate(values)]
    return {key: rows}


def _hurricane_result(category: str = "category_2") -> object:
    from omni_mercury_engine.detectors.geological.hurricane_detector import (
        HurricanePredictionResult,
    )

    return HurricanePredictionResult(
        cyclone_detected=True,
        confidence=0.9,
        category=category,
        cyclone_type="hurricane",
        max_wind_speed_kt=85.0,
        min_pressure_mb=965.0,
    )


class TestSurgeFloodCascade:
    def test_initial_state_quiet(self) -> None:
        cascade = SurgeFloodCascade()
        assessment = cascade.evaluate()
        assert assessment.stage is CascadeStage.QUIET

    def test_hurricane_evidence_reaches_watch(self) -> None:
        cascade = SurgeFloodCascade()
        record = cascade.update_hurricane_evidence(_hurricane_result())
        assert record.qualifies is True
        assert cascade.evaluate().stage is CascadeStage.WATCH

    def test_sub_watch_category_does_not_qualify(self) -> None:
        cascade = SurgeFloodCascade(min_watch_category="category_1")
        record = cascade.update_hurricane_evidence(_hurricane_result("tropical_storm"))
        assert record.qualifies is False
        assert cascade.evaluate().stage is CascadeStage.QUIET

    def test_surge_residual_is_observed_minus_predicted(self) -> None:
        cascade = SurgeFloodCascade()
        observed = _coops_payload([1.0, 1.1, 2.4, 2.5, 2.3], key="data")
        predicted = _coops_payload([1.0, 1.05, 1.1, 1.15, 1.1], key="predictions")
        series = cascade.compute_surge_residual(observed, predicted)
        assert series.max_residual_m == pytest.approx(2.5 - 1.15, abs=1e-9)
        assert np.allclose(series.residual_m, series.observed_m - series.predicted_m)

    def test_full_cascade_reaches_compound_flood(self) -> None:
        cascade = SurgeFloodCascade()
        cascade.update_hurricane_evidence(_hurricane_result())
        observed = _coops_payload([1.0, 1.5, 2.0, 2.2], key="data")
        predicted = _coops_payload([1.0, 1.0, 1.0, 1.0], key="predictions")
        cascade.update_surge_evidence(observed, predicted)
        cascade.update_river_evidence(
            {
                "lid": "TESTG1",
                "status": {"observed": {"primary": 21.4, "floodCategory": "major"}},
                "flood": {"categories": {"minor": {"stage": 14.0}}},
            }
        )
        assessment = cascade.evaluate()
        assert assessment.stage is CascadeStage.COMPOUND_FLOOD
        kinds = {record.kind for record in assessment.evidence_chain}
        assert kinds == {"hurricane", "surge", "river"}

    def test_surge_without_hurricane_does_not_reach_compound(self) -> None:
        cascade = SurgeFloodCascade()
        observed = _coops_payload([2.0, 2.0], key="data")
        predicted = _coops_payload([1.0, 1.0], key="predictions")
        cascade.update_surge_evidence(observed, predicted)
        assert cascade.evaluate().stage is not CascadeStage.COMPOUND_FLOOD

    def test_reset_returns_to_quiet(self) -> None:
        cascade = SurgeFloodCascade()
        cascade.update_hurricane_evidence(_hurricane_result())
        cascade.reset()
        assert cascade.evaluate().stage is CascadeStage.QUIET

    def test_bad_threshold_rejected(self) -> None:
        with pytest.raises(ValueError, match="surge_threshold_m"):
            SurgeFloodCascade(surge_threshold_m=0.0)

    def test_river_evidence_without_stage_fails_loud(self) -> None:
        cascade = SurgeFloodCascade()
        with pytest.raises(ValueError, match="no observed stage"):
            cascade.update_river_evidence({"lid": "X", "status": {"observed": {}}})
