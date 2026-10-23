# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hazard adapters: the engine contract over the hazard detectors' own APIs.

Each test drives an adapter with physically meaningful rows and asserts the
wrapped detector's OWN outputs come through — scores on the detectors' cited
scales, decisions from their released logic — and that the batch contract
(schema refusal, non-finite refusal, feature alignment) holds. These fail on
the tree before the adapters existed: the hazard detectors registered in the
manifest without ``extract_features``/``predict``, so the registry's feature
extraction skipped them silently.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest

pytest.importorskip("torch")

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.core.exceptions import InputModalityError, InvalidObservationError
from omni_mercury_engine.detectors.hazard_adapter import (
    Assessment,
    FinancialStressAdapter,
    FloodPrecipitationAdapter,
    GICMinuteSeriesAdapter,
    HazardRecordAdapter,
    HazardRowAdapter,
    HazardTextAdapter,
    HazardUnitAdapter,
    HazardWindowAdapter,
    HurricanePressureAdapter,
    SolarWindGeomagAdapter,
    TornadoConditionsAdapter,
    XrayFlareAdapter,
    ordinal_score,
    utc_datetime,
)

QUIET_SOUNDING = [200.0, -50.0, 50.0, 10.0, 1500.0]
SUPERCELL_SOUNDING = [3500.0, -30.0, 450.0, 60.0, 800.0]


class TestBatchContract:
    """The shared row-batch contract, shown on one representative adapter."""

    def test_is_a_base_detector(self) -> None:
        adapter = TornadoConditionsAdapter()
        assert isinstance(adapter, BaseDetector)

    def test_fit_validates_schema_and_trains_nothing(self) -> None:
        adapter = TornadoConditionsAdapter()
        assert adapter.fit(np.array([QUIET_SOUNDING]))._is_fitted is True

    def test_wrong_width_refused(self) -> None:
        with pytest.raises(InvalidObservationError, match="declared columns"):
            TornadoConditionsAdapter().detect(np.zeros((3, 4)))

    def test_non_finite_rows_refused_not_graded(self) -> None:
        bad = np.array([QUIET_SOUNDING, [np.nan] * 5])
        with pytest.raises(InvalidObservationError, match="not measurements"):
            TornadoConditionsAdapter().detect(bad)

    def test_detect_contract_keys_and_alignment(self) -> None:
        result = TornadoConditionsAdapter().detect(np.array([QUIET_SOUNDING, SUPERCELL_SOUNDING]))
        assert set(result) >= {"anomaly_score", "is_anomaly", "scores"}
        assert result["scores"].shape == (2,)
        assert result["anomaly_score"] == result["scores"].max()

    def test_features_align_with_declared_names(self) -> None:
        adapter = TornadoConditionsAdapter()
        features = adapter.extract_features(np.array([QUIET_SOUNDING, SUPERCELL_SOUNDING]))
        assert features.shape == (2, len(adapter.FEATURE_NAMES))
        assert np.all(np.isfinite(features))

    def test_predict_serves_per_row_scores(self) -> None:
        result = TornadoConditionsAdapter().predict(np.array([QUIET_SOUNDING, SUPERCELL_SOUNDING]))
        assert result["anomaly_scores"].shape == (2,)


class TestTornadoConditions:
    def test_supercell_outranks_quiet_with_released_semantics(self) -> None:
        result = TornadoConditionsAdapter().detect(np.array([QUIET_SOUNDING, SUPERCELL_SOUNDING]))
        # The wrapped detector's released decision needs more than the
        # atmospheric path (radar / pressure indicators), so a supercell
        # sounding alone reads confidence 0.25, not tornado_likely.
        assert result["scores"][0] == 0.0
        assert result["scores"][1] == pytest.approx(0.25)
        assert result["is_anomaly"] is False

    def test_features_echo_the_accepted_measurements(self) -> None:
        features = TornadoConditionsAdapter().extract_features(np.array([SUPERCELL_SOUNDING]))
        confidence, likely, cape, helicity = features[0]
        assert (cape, helicity) == (3500.0, 450.0)
        assert likely == 0.0
        assert confidence == pytest.approx(0.25)


class TestHurricanePressure:
    def test_920_mb_is_a_detected_cyclone(self) -> None:
        result = HurricanePressureAdapter().detect(
            np.array([[1013.0, 1013.0, 26.0], [920.0, 1010.0, 30.0]])
        )
        assert result["scores"][0] == 0.0
        assert result["scores"][1] == pytest.approx(0.7)
        assert result["is_anomaly"] is True

    def test_wind_feature_is_the_atkinson_holliday_value(self) -> None:
        features = HurricanePressureAdapter().extract_features(np.array([[920.0, 1010.0, 30.0]]))
        wind_kt = features[0][2]
        assert wind_kt == pytest.approx(6.7 * (1010.0 - 920.0) ** 0.644, rel=1e-6)


class TestFloodPrecipitation:
    def test_heavy_rain_outranks_dry(self) -> None:
        result = FloodPrecipitationAdapter().detect(
            np.array([[0.0, 0.0, 0.1, 0.0], [1.5, 4.0, 6.0, 3.0]])
        )
        assert result["scores"][0] == 0.0
        assert result["scores"][1] > 0.0

    def test_feature_echoes_the_24h_depth(self) -> None:
        features = FloodPrecipitationAdapter().extract_features(np.array([[1.5, 4.0, 6.0, 3.0]]))
        assert features[0][2] == 6.0


class TestFinancialStress:
    def test_crash_and_banking_stress_is_an_imminent_systemic_crisis(self) -> None:
        calm = [0.2, 15.0, 1.0, 80.0, 1.0, 1.5]
        crisis = [-9.0, 75.0, 4.0, 600.0, 9.0, 0.4]
        result = FinancialStressAdapter().detect(np.array([calm, crisis]))
        assert result["scores"][0] == 0.0
        assert result["scores"][1] == pytest.approx(0.8)
        assert result["is_anomaly"] is True
        features = FinancialStressAdapter().extract_features(np.array([crisis]))
        _, crisis_imminent, market_crash, banking_stress = features[0]
        assert (crisis_imminent, market_crash, banking_stress) == (1.0, 1.0, 1.0)


class TestSolarWindGeomag:
    def test_score_is_the_noaa_g_level_over_five(self) -> None:
        result = SolarWindGeomagAdapter().detect(np.array([[380.0, 1.0], [800.0, -25.0]]))
        assert result["scores"][0] == 0.0  # G0
        assert result["scores"][1] * 5 == int(result["scores"][1] * 5) > 0
        assert result["is_anomaly"] is True  # G >= 1 is NOAA's storm

    def test_features_carry_the_predicted_kp(self) -> None:
        features = SolarWindGeomagAdapter().extract_features(np.array([[800.0, -25.0]]))
        kp, g_level, confidence = features[0]
        assert 0.0 <= kp <= 9.0 and g_level >= 1 and 0.0 <= confidence <= 1.0


class TestXrayFlare:
    def test_score_is_the_noaa_r_level_over_five(self) -> None:
        result = XrayFlareAdapter().detect(np.array([[5e-7], [5e-4]]))
        assert result["scores"][0] == 0.0  # B5: below R1
        assert result["scores"][1] == pytest.approx(3 / 5)  # X5 is R3
        assert result["is_anomaly"] is True


class TestGICMinuteSeries:
    @staticmethod
    def _quiet() -> np.ndarray:
        return np.column_stack([20000 + np.linspace(0, 5, 60), 1000 + np.linspace(0, 3, 60)])

    def test_quiet_series_is_low_tier(self) -> None:
        result = GICMinuteSeriesAdapter().detect(self._quiet())
        assert result["anomaly_score"] == 0.0
        assert result["is_anomaly"] is False

    def test_500_nt_per_min_is_the_severe_tier(self) -> None:
        storm = self._quiet()
        storm[30:, 0] += np.cumsum(np.full(30, 500.0))
        adapter = GICMinuteSeriesAdapter()
        result = adapter.detect(storm)
        assert result["anomaly_score"] == 1.0  # severe / 3 tiers
        assert result["is_anomaly"] is True
        features = adapter.extract_features(storm)
        assert features.shape == (60, 3)
        assert features[0][0] == pytest.approx(500.0, rel=1e-2)  # peak dB/dt
        assert np.all(features[:, 2] == 3.0)  # tier repeats over the window

    def test_cadence_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="cadence_s"):
            GICMinuteSeriesAdapter(config={"cadence_s": 0.0})


class TestMutationClosedGaps:
    """Boundary and shape cases the first mutation run proved untested.

    A 45-mutant sweep of hazard_adapter.py against this file measured a
    66.7% kill rate; each test here kills real survivors (decision
    boundaries mutated from >= to >, the market/banking column split, 1-D
    input handling, the cadence guard) and one caught a genuine crash:
    float(None) on the sustained metric for windows under 10 minutes.
    """

    def test_one_dimensional_row_is_one_sample(self) -> None:
        adapter = TornadoConditionsAdapter()
        flat = adapter.detect(np.array(SUPERCELL_SOUNDING))
        square = adapter.detect(np.array([SUPERCELL_SOUNDING]))
        assert flat["scores"].shape == (1,)
        np.testing.assert_array_equal(flat["scores"], square["scores"])

    def test_even_width_one_dimensional_row_is_still_one_sample(self) -> None:
        # Six columns divide evenly, so a wrong reshape would silently make
        # two bogus three-column rows instead of erroring; pin one sample.
        crisis = np.array([-9.0, 75.0, 4.0, 600.0, 9.0, 0.4])
        result = FinancialStressAdapter().detect(crisis)
        assert result["scores"].shape == (1,)
        assert result["scores"][0] == pytest.approx(0.8)

    def test_liquidity_shortage_alone_is_banking_stress_not_crisis(self) -> None:
        calm_market_dry_liquidity = [0.2, 15.0, 1.0, 80.0, 1.0, 0.05]
        adapter = FinancialStressAdapter()
        features = adapter.extract_features(np.array([calm_market_dry_liquidity]))
        confidence, crisis_imminent, market_crash, banking_stress = features[0]
        assert (crisis_imminent, market_crash, banking_stress) == (0.0, 0.0, 1.0)
        assert confidence == pytest.approx(0.8)

    def test_exactly_g1_is_a_storm(self) -> None:
        result = SolarWindGeomagAdapter().detect(np.array([[650.0, -15.0]]))
        assert result["scores"][0] == pytest.approx(1 / 5)  # exactly G1
        assert result["is_anomaly"] is True  # G1 IS a storm (NOAA), not G2+

    def test_m1_flux_is_exactly_r1_and_c9_is_not(self) -> None:
        result = XrayFlareAdapter().detect(np.array([[1e-5], [9e-6]]))
        assert result["scores"][0] == pytest.approx(1 / 5)  # M1.0 is R1
        assert result["scores"][1] == 0.0  # C9 is below R1
        assert result["is_anomaly"] is True

    def test_100_nt_per_min_is_exactly_the_moderate_tier(self) -> None:
        series = np.column_stack([20000 + np.zeros(60), 1000 + np.zeros(60)])
        series[30:, 0] += np.cumsum(np.full(30, 100.0))
        result = GICMinuteSeriesAdapter().detect(series)
        assert result["anomaly_score"] == pytest.approx(1 / 3)  # moderate
        assert result["is_anomaly"] is True  # >= 90 nT/min acts, not > moderate

    def test_second_cadence_is_averaged_and_short_sustained_is_nan(self) -> None:
        quiet_3_minutes = np.column_stack([20000 + np.zeros(180), 1000 + np.zeros(180)])
        adapter = GICMinuteSeriesAdapter(config={"cadence_s": 1.0})
        result = adapter.detect(quiet_3_minutes)
        assert result["anomaly_score"] == 0.0 and result["is_anomaly"] is False
        features = adapter.extract_features(quiet_3_minutes)
        # Regression: float(None) crashed here. The 10-minute rolling metric
        # is unmeasured on a 3-minute window: NaN, never a quiet zero.
        assert np.isfinite(features[0][0]) and np.isnan(features[0][1])


# ---------------------------------------------------------------------------
# The shared bases, driven through stubs: their contracts, pinned boundary by
# boundary (the mutation-testing lane runs this file against hazard_adapter.py).
# ---------------------------------------------------------------------------


def _assessment(**overrides: Any) -> Assessment:
    fields: dict[str, Any] = {
        "score": 0.5,
        "flag": False,
        "features": (1.0,),
        "result": SimpleNamespace(set_aside=None),
    }
    fields.update(overrides)
    return Assessment(**fields)


class _Rows(HazardRowAdapter):
    COLUMNS = ("a", "b", "c")
    FEATURE_NAMES = ("score",)

    def _analyze_row(self, row: np.ndarray[Any, Any]) -> tuple[float, bool, list[float]]:
        return 0.25, False, [float(np.nansum(row))]


class _Window(HazardWindowAdapter):
    COLUMNS = ("a", "b")
    OPTIONAL_COLUMNS = frozenset({"b"})
    FEATURE_NAMES = ("x",)
    assessment: ClassVar[Assessment]

    def _assess(self, unit: Any) -> Assessment:
        return type(self).assessment


class _Records(HazardRecordAdapter):
    FEATURE_NAMES = ("x",)
    REQUIRED_KEYS = frozenset({"id"})
    ALLOWED_KEYS: ClassVar[frozenset[str] | None] = frozenset({"id", "value"})

    def _assess(self, unit: Any) -> Assessment:
        return _assessment()


class _Text(HazardTextAdapter):
    FEATURE_NAMES = ("x",)

    def _assess(self, unit: Any) -> Assessment:
        return _assessment()


class TestRowBase:
    def test_a_one_dimensional_row_is_a_batch_of_one(self) -> None:
        assert _Rows().extract_features(np.array([1.0, 2.0, 3.0])).shape == (1, 1)

    def test_measured_leaves_out_exactly_the_nan_columns(self) -> None:
        measured = _Rows()._measured(np.array([1.0, np.nan, 3.0]))
        assert measured == {"a": 1.0, "c": 3.0}

    def test_measured_refuses_a_row_of_the_wrong_length(self) -> None:
        with pytest.raises(ValueError):
            _Rows()._measured(np.array([1.0, 2.0]))

    def test_the_set_aside_of_a_direct_row_call_is_recorded_under_row_zero(self) -> None:
        adapter = _Rows()
        adapter._note_set_aside(SimpleNamespace(set_aside=[{"where": "x"}]))
        assert adapter._row_set_aside == {0: [{"where": "x"}]}

    def test_a_row_inside_a_batch_records_its_own_index(self) -> None:
        adapter = _Rows()
        adapter._row_set_aside = {}
        adapter._row_index = 3
        adapter._note_set_aside(SimpleNamespace(set_aside=[{"where": "y"}]))
        assert adapter._row_set_aside == {3: [{"where": "y"}]}

    def test_a_result_with_nothing_set_aside_records_nothing(self) -> None:
        adapter = _Rows()
        adapter._note_set_aside(SimpleNamespace(set_aside=[]))
        assert not hasattr(adapter, "_row_set_aside")

    def test_a_partial_verdict_is_marked_partial(self) -> None:
        class Partial(_Rows):
            def _analyze_row(self, row: np.ndarray[Any, Any]) -> tuple[float, bool, list[float]]:
                self._note_set_aside(SimpleNamespace(set_aside=[{"where": "row"}]))
                return 0.25, False, [1.0]

        out = Partial().detect(np.ones((2, 3)))
        assert out["partial"] is True
        assert out["set_aside"] == {0: [{"where": "row"}], 1: [{"where": "row"}]}

    def test_a_whole_verdict_is_not_marked_partial(self) -> None:
        assert "partial" not in _Rows().detect(np.ones((2, 3)))

    def test_nan_marks_an_unmeasured_value_only_where_declared(self) -> None:
        class Optional(_Rows):
            OPTIONAL_COLUMNS = frozenset({"b"})

        assert Optional().detect(np.array([[1.0, np.nan, 3.0]]))["scores"].shape == (1,)
        with pytest.raises(InvalidObservationError, match="only in"):
            Optional().detect(np.array([[np.nan, 2.0, 3.0]]))
        with pytest.raises(InvalidObservationError):
            Optional().detect(np.array([[1.0, np.inf, 3.0]]))

    def test_fit_marks_the_adapter_fitted(self) -> None:
        assert _Rows().fit(np.ones((2, 3)))._is_fitted is True


class TestAssessment:
    def test_is_immutable(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            _assessment().score = 0.9  # type: ignore[misc]

    @pytest.mark.parametrize("score", [0.0, 0.05, 1.0])
    def test_scores_on_and_inside_the_unit_interval_are_served(self, score: float) -> None:
        _Window.assessment = _assessment(score=score)
        assert _Window().predict(np.ones((2, 2)))["anomaly_scores"][0] == score

    @pytest.mark.parametrize("score", [-0.05, 1.05, 2.0])
    def test_scores_outside_the_unit_interval_are_refused(self, score: float) -> None:
        _Window.assessment = _assessment(score=score)
        with pytest.raises(RuntimeError, match="malformed assessment"):
            _Window().predict(np.ones((2, 2)))

    @pytest.mark.parametrize("features", [(), (1.0, 2.0)])
    def test_a_feature_count_that_is_not_declared_is_refused(
        self, features: tuple[float, ...]
    ) -> None:
        _Window.assessment = _assessment(features=features)
        with pytest.raises(RuntimeError, match="malformed assessment"):
            _Window().detect(np.ones((2, 2)))

    def test_unit_adapters_define_nan_as_unmeasured(self) -> None:
        assert HazardUnitAdapter.handles_unmeasured is True
        assert HazardRowAdapter.handles_unmeasured is True

    def test_fit_marks_a_unit_adapter_fitted(self) -> None:
        _Window.assessment = _assessment()
        assert _Window().fit(np.ones((2, 2)))._is_fitted is True


class TestSetAsideOfAUnit:
    def test_the_assessments_own_entries_and_the_results_both_count(self) -> None:
        both = _assessment(
            set_aside=({"where": "own"},), result=SimpleNamespace(set_aside=[{"where": "res"}])
        )
        _Window.assessment = both
        out = _Window().detect(np.ones((2, 2)))
        assert out["partial"] is True
        assert out["set_aside"] == {0: [{"where": "own"}]}

    def test_the_result_carries_it_when_the_assessment_does_not(self) -> None:
        _Window.assessment = _assessment(result=SimpleNamespace(set_aside=[{"where": "res"}]))
        out = _Window().detect(np.ones((2, 2)))
        assert out["partial"] is True
        assert out["set_aside"] == {0: [{"where": "res"}]}

    def test_a_whole_unit_is_not_partial(self) -> None:
        _Window.assessment = _assessment()
        out = _Window().detect(np.ones((2, 2)))
        assert "partial" not in out
        assert out["metadata"]["units"] == 1


class TestWindowBase:
    def test_two_samples_are_enough_and_one_is_not(self) -> None:
        _Window.assessment = _assessment()
        _Window().detect(np.ones((2, 2)))
        with pytest.raises(InvalidObservationError, match=r"at least 2 samples; got 1$"):
            _Window().detect(np.ones((1, 2)))

    def test_an_empty_window_names_zero_samples(self) -> None:
        with pytest.raises(InvalidObservationError, match=r"got 0$"):
            _Window().detect(np.ones((0, 2)))

    @pytest.mark.parametrize(
        "shape", [(4,), (4, 1), (4, 3), (4, 2, 2)], ids=["1-D", "narrow", "wide", "3-D"]
    )
    def test_a_window_that_is_not_samples_by_declared_columns_is_refused(
        self, shape: tuple[int, ...]
    ) -> None:
        with pytest.raises(InputModalityError, match="expects one window"):
            _Window().detect(np.ones(shape))

    def test_text_is_not_a_numeric_window(self) -> None:
        with pytest.raises(InputModalityError, match="numeric window"):
            _Window().detect([["x", "y"], ["z", "w"]])

    def test_a_gap_is_allowed_only_in_an_optional_column(self) -> None:
        _Window.assessment = _assessment()
        _Window().detect(np.array([[1.0, np.nan], [2.0, 3.0]]))
        with pytest.raises(InvalidObservationError, match="non-finite"):
            _Window().detect(np.array([[np.nan, 1.0], [2.0, 3.0]]))

    def test_an_infinity_is_refused_even_where_a_gap_is_allowed(self) -> None:
        with pytest.raises(InvalidObservationError, match="non-finite"):
            _Window().detect(np.array([[1.0, np.inf], [2.0, 3.0]]))

    def test_column_returns_the_named_column(self) -> None:
        window = np.array([[1.0, 10.0], [2.0, 20.0]])
        assert _Window().column(window, "b").tolist() == [10.0, 20.0]


class TestRecordBase:
    def test_a_mapping_or_a_sequence_of_mappings_is_accepted(self) -> None:
        assert _Records().predict({"id": 1})["anomaly_scores"].shape == (1,)
        assert _Records().predict(({"id": 1}, {"id": 2}))["anomaly_scores"].shape == (2,)

    @pytest.mark.parametrize("data", [[], (), "text", b"bytes", 7, None])
    def test_anything_else_is_refused(self, data: Any) -> None:
        with pytest.raises(InputModalityError, match="expects a record"):
            _Records().predict(data)

    def test_a_non_mapping_record_is_refused_by_position(self) -> None:
        with pytest.raises(InputModalityError, match="record 1 is a int"):
            _Records().predict([{"id": 1}, 5])

    def test_missing_required_keys_are_named(self) -> None:
        with pytest.raises(InvalidObservationError, match=r"record 0 is missing \['id'\]"):
            _Records().predict({"value": 1})

    def test_unknown_keys_are_set_aside_by_name_and_the_verdict_is_partial(self) -> None:
        out = _Records().detect([{"id": 1, "vlaue": 2}])
        assert out["partial"] is True
        (entry,) = out["set_aside"][0]
        assert entry["where"] == "record[0].vlaue"
        assert "accepted: ['id', 'value']" in entry["reason"]

    def test_only_the_unknown_keys_are_dropped(self) -> None:
        seen: list[Any] = []

        class Spy(_Records):
            def _parse(self, record: Any) -> Any:
                seen.append(dict(record))
                return record

        Spy().predict({"id": 1, "value": 2, "extra": 3})
        assert seen == [{"id": 1, "value": 2}]

    def test_a_record_of_nothing_but_unknown_keys_has_nothing_usable(self) -> None:
        class Free(_Records):
            REQUIRED_KEYS = frozenset()

        with pytest.raises(InvalidObservationError, match="nothing usable"):
            Free().predict({"nope": 1})

    def test_any_key_is_accepted_where_none_are_declared(self) -> None:
        class Open(_Records):
            ALLOWED_KEYS = None

        assert "partial" not in Open().detect({"id": 1, "whatever": 2})


class TestTextBase:
    def test_a_string_or_a_sequence_of_strings_is_accepted(self) -> None:
        assert _Text().predict("abc")["anomaly_scores"].shape == (1,)
        assert _Text().predict(["a", "b"])["anomaly_scores"].shape == (2,)

    @pytest.mark.parametrize("data", [[], (), b"bytes", 7, None])
    def test_anything_else_is_refused(self, data: Any) -> None:
        with pytest.raises(InputModalityError, match="expects text"):
            _Text().predict(data)

    def test_a_non_string_payload_is_refused_by_position(self) -> None:
        with pytest.raises(InputModalityError, match="payload 1 is a int, not text"):
            _Text().predict(["a", 3])

    def test_the_limit_is_65536_characters_inclusive(self) -> None:
        assert HazardTextAdapter.MAX_CHARS == 65536
        assert "partial" not in _Text().detect("x" * 65536)

    def test_a_longer_payload_is_truncated_and_the_rest_is_named(self) -> None:
        graded: list[int] = []

        class Spy(_Text):
            def _assess(self, unit: Any) -> Assessment:
                graded.append(len(unit))
                return _assessment()

        out = Spy().detect("x" * 65537)
        assert graded == [65536]
        assert out["partial"] is True
        (entry,) = out["set_aside"][0]
        assert entry["where"] == "payload[0][65536:]"
        assert entry["reason"].startswith("1 of 65537 characters lie beyond")


class TestHelpers:
    def test_ordinal_score_is_the_position_over_the_top_rung(self) -> None:
        ladder = ("low", "mid", "high")
        assert [ordinal_score(x, ladder) for x in ladder] == [0.0, 0.5, 1.0]

    def test_a_level_off_the_ladder_is_refused(self) -> None:
        with pytest.raises(InvalidObservationError, match="not on the ladder"):
            ordinal_score("other", ("low", "high"))

    def test_epoch_seconds_are_utc(self) -> None:
        assert utc_datetime(0, "t") == datetime(1970, 1, 1, tzinfo=UTC)
        assert utc_datetime(np.float64(86400.0), "t") == datetime(1970, 1, 2, tzinfo=UTC)

    @pytest.mark.parametrize("value", [True, False])
    def test_a_boolean_is_not_epoch_seconds(self, value: bool) -> None:
        with pytest.raises(InvalidObservationError, match="expected a datetime"):
            utc_datetime(value, "t")

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_non_finite_epoch_seconds_are_refused(self, value: float) -> None:
        with pytest.raises(InvalidObservationError, match="non-finite epoch"):
            utc_datetime(value, "t")


class _Detector:
    """Records what an adapter hands the wrapped detector."""

    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[Any] = []

    def predict_financial_crisis(self, payload: Any) -> Any:
        self.calls.append(payload)
        return self.result

    def predict_solar_storm(self, payload: Any) -> Any:
        self.calls.append(payload)
        return self.result


class TestWiringOfTheConcreteAdapters:
    def test_the_financial_columns_split_three_market_three_banking(self) -> None:
        wrapped = _Detector(
            SimpleNamespace(
                confidence=0.5,
                crisis_imminent=False,
                market_crash_detected=False,
                banking_stress=0.1,
                set_aside=None,
            )
        )
        adapter = FinancialStressAdapter()
        adapter._detector = wrapped
        adapter.detect(np.array([[-1.0, 20.0, 1.1, 150.0, 2.0, 0.4]]))
        (payload,) = wrapped.calls
        assert payload["market_data"] == {
            "price_change_pct": -1.0,
            "vix": 20.0,
            "volume_vs_average": 1.1,
        }
        assert payload["banking_data"] == {
            "cds_spread_bps": 150.0,
            "default_rate_pct": 2.0,
            "liquidity_ratio": 0.4,
        }

    @pytest.mark.parametrize(
        ("kp", "level"),
        [(None, "minor"), (4.0, "gale"), (None, "gale")],
        ids=["kp", "level", "both"],
    )
    def test_a_solar_prediction_missing_either_half_is_refused(
        self, kp: float | None, level: str
    ) -> None:
        wrapped = _Detector(
            SimpleNamespace(kp_index=kp, geomagnetic_storm_level=level, confidence=0.5)
        )
        adapter = SolarWindGeomagAdapter()
        adapter._detector = wrapped
        row = [500.0] * len(SolarWindGeomagAdapter.COLUMNS)
        with pytest.raises(InvalidObservationError, match="no geomagnetic level"):
            adapter.detect(np.array([row]))

    def test_the_gic_row_cadence_defaults_to_a_minute(self) -> None:
        assert GICMinuteSeriesAdapter()._cadence_s == 60.0
        assert GICMinuteSeriesAdapter({"cadence_s": 1})._cadence_s == 1.0
