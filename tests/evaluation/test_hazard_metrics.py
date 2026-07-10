# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the hazard skill-score primitives.

Every formula is verified against a worked example from the literature
definition:

* Contingency scores against the classic Finley (1884) tornado-forecast
  table (a=28 hits, b=72 false alarms, c=23 misses, d=2680 correct
  negatives), the canonical worked example in Wilks (2011, table 8.3):
  POD 28/51, FAR 72/100, CSI 28/123, bias 100/51, HSS 0.3553.
* Haversine against exact geometry (1 degree of equatorial longitude =
  R * pi/180 = 111.19493 km; pole-to-equator = R * pi/2).
* Brier against the hand-computed mean of squared probability errors.
* G-scale bucketing against the published NOAA Space Weather Scale
  thresholds (G1 = Kp 5 ... G5 = Kp 9).
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.evaluation import hazard_metrics as hm

# Finley (1884) tornado forecasts: the classic verification dataset.
FINLEY = {"hits": 28, "false_alarms": 72, "misses": 23, "correct_negatives": 2680}


class TestContingencyScores:
    def test_contingency_table_from_binary_arrays(self) -> None:
        y_true = [1, 1, 1, 0, 0, 0, 0]
        y_pred = [1, 1, 0, 1, 0, 0, 0]
        assert hm.contingency_table(y_true, y_pred) == (2, 1, 1, 3)

    def test_pod_matches_finley(self) -> None:
        pod = hm.probability_of_detection(FINLEY["hits"], FINLEY["misses"])
        assert pod == pytest.approx(28 / 51)  # 0.549
        assert pod == pytest.approx(0.5490, abs=1e-4)

    def test_far_matches_finley(self) -> None:
        far = hm.false_alarm_ratio(FINLEY["hits"], FINLEY["false_alarms"])
        assert far == pytest.approx(72 / 100)  # 0.720

    def test_csi_matches_finley(self) -> None:
        csi = hm.critical_success_index(FINLEY["hits"], FINLEY["misses"], FINLEY["false_alarms"])
        assert csi == pytest.approx(28 / 123)  # 0.228
        assert csi == pytest.approx(0.2276, abs=1e-4)

    def test_frequency_bias_matches_finley(self) -> None:
        bias = hm.frequency_bias(FINLEY["hits"], FINLEY["misses"], FINLEY["false_alarms"])
        assert bias == pytest.approx(100 / 51)  # 1.96: Finley over-forecast

    def test_hss_matches_finley(self) -> None:
        # 2(ad - bc) / [(a+c)(c+d) + (a+b)(b+d)]
        #   = 2(28*2680 - 72*23) / (51*2703 + 100*2752) = 146768 / 413053
        hss = hm.heidke_skill_score(
            FINLEY["hits"],
            FINLEY["misses"],
            FINLEY["false_alarms"],
            FINLEY["correct_negatives"],
        )
        assert hss == pytest.approx(146768 / 413053)
        assert hss == pytest.approx(0.3553, abs=1e-4)  # literature value

    def test_perfect_forecast_scores(self) -> None:
        assert hm.probability_of_detection(10, 0) == 1.0
        assert hm.false_alarm_ratio(10, 0) == 0.0
        assert hm.critical_success_index(10, 0, 0) == 1.0
        assert hm.frequency_bias(10, 0, 0) == 1.0
        assert hm.heidke_skill_score(10, 0, 0, 10) == 1.0

    def test_pod_undefined_without_observed_events(self) -> None:
        with pytest.raises(ValueError, match="POD undefined"):
            hm.probability_of_detection(0, 0)

    def test_far_undefined_without_forecasts(self) -> None:
        with pytest.raises(ValueError, match="FAR undefined"):
            hm.false_alarm_ratio(0, 0)

    def test_csi_undefined_on_all_correct_negatives(self) -> None:
        with pytest.raises(ValueError, match="CSI undefined"):
            hm.critical_success_index(0, 0, 0)

    def test_hss_undefined_on_degenerate_table(self) -> None:
        # Only hits observed and forecast: chance denominator is zero.
        with pytest.raises(ValueError, match="HSS undefined"):
            hm.heidke_skill_score(5, 0, 0, 0)

    def test_negative_counts_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            hm.probability_of_detection(-1, 5)

    def test_non_integer_counts_rejected(self) -> None:
        with pytest.raises(ValueError, match="integer count"):
            hm.probability_of_detection(1.5, 5)  # type: ignore[arg-type]

    def test_contingency_table_rejects_non_binary(self) -> None:
        with pytest.raises(ValueError, match="binary"):
            hm.contingency_table([0, 1, 2], [0, 1, 1])

    def test_contingency_table_rejects_nan(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            hm.contingency_table([0.0, np.nan], [0.0, 1.0])

    def test_contingency_table_rejects_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            hm.contingency_table([0, 1], [0, 1, 1])


class TestLeadTimes:
    def test_lead_is_event_minus_first_alert(self) -> None:
        leads = hm.lead_times([100.0, 200.0], [80.0, 150.0])
        np.testing.assert_allclose(leads, [20.0, 50.0])

    def test_late_alert_gives_negative_lead(self) -> None:
        leads = hm.lead_times([100.0], [130.0])
        np.testing.assert_allclose(leads, [-30.0])

    def test_nan_alert_rejected(self) -> None:
        """A missed event must be handled via POD, never averaged as NaN."""
        with pytest.raises(ValueError, match="non-finite"):
            hm.lead_times([100.0], [np.nan])

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            hm.lead_times([], [])


class TestMagnitudeAndLocation:
    def test_magnitude_error_hand_example(self) -> None:
        mae, bias = hm.magnitude_error([5.0, 6.0], [5.5, 5.5])
        assert mae == pytest.approx(0.5)
        assert bias == pytest.approx(0.0)  # +0.5 and -0.5 cancel

    def test_magnitude_bias_sign_is_overestimation(self) -> None:
        _, bias = hm.magnitude_error([5.0, 5.0], [5.4, 5.2])
        assert bias == pytest.approx(0.3)

    def test_haversine_zero_distance(self) -> None:
        err = hm.location_error_km([35.0], [139.0], [35.0], [139.0])
        assert err[0] == pytest.approx(0.0, abs=1e-9)

    def test_haversine_one_degree_equatorial_longitude(self) -> None:
        # Exact: R * (pi / 180) = 6371.0088 * 0.0174533 = 111.19508 km.
        err = hm.location_error_km([0.0], [0.0], [0.0], [1.0])
        assert err[0] == pytest.approx(hm.EARTH_RADIUS_KM * np.pi / 180.0)
        assert err[0] == pytest.approx(111.19508, abs=1e-4)

    def test_haversine_pole_to_equator(self) -> None:
        # Exact quarter meridian: R * pi / 2.
        err = hm.location_error_km([0.0], [10.0], [90.0], [10.0])
        assert err[0] == pytest.approx(hm.EARTH_RADIUS_KM * np.pi / 2.0)

    def test_haversine_symmetric(self) -> None:
        ab = hm.location_error_km([48.8566], [2.3522], [51.5074], [-0.1278])
        ba = hm.location_error_km([51.5074], [-0.1278], [48.8566], [2.3522])
        assert ab[0] == pytest.approx(ba[0])
        assert 330.0 < ab[0] < 350.0  # Paris-London is ~343 km

    def test_haversine_rejects_bad_latitude(self) -> None:
        with pytest.raises(ValueError, match=r"\[-90, 90\]"):
            hm.location_error_km([95.0], [0.0], [0.0], [0.0])


class TestOrdinalScales:
    VOLCANO_LEVELS = ("normal", "advisory", "watch", "warning")

    def test_ordinal_accuracy_hand_example(self) -> None:
        true = ["normal", "watch", "warning"]
        pred = ["advisory", "watch", "normal"]
        exact, within1 = hm.ordinal_accuracy(true, pred, self.VOLCANO_LEVELS)
        assert exact == pytest.approx(1 / 3)  # only "watch" exact
        assert within1 == pytest.approx(2 / 3)  # warning->normal is 3 off

    def test_ordinal_accuracy_flare_classes(self) -> None:
        exact, within1 = hm.ordinal_accuracy(
            ["M", "X", "C"], ["M", "M", "A"], ("A", "B", "C", "M", "X")
        )
        assert exact == pytest.approx(1 / 3)
        assert within1 == pytest.approx(2 / 3)  # X->M adjacent; C->A is 2 off

    def test_ordinal_accuracy_rejects_unknown_label(self) -> None:
        with pytest.raises(ValueError, match="not in levels"):
            hm.ordinal_accuracy(["normal"], ["red"], self.VOLCANO_LEVELS)

    def test_ordinal_accuracy_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            hm.ordinal_accuracy([], [], self.VOLCANO_LEVELS)

    def test_vei_accuracy_hand_example(self) -> None:
        exact, within1 = hm.vei_accuracy([2, 4], [2, 6])
        assert exact == pytest.approx(0.5)
        assert within1 == pytest.approx(0.5)

    def test_vei_accuracy_within_one_credit(self) -> None:
        exact, within1 = hm.vei_accuracy([3, 3, 3], [3, 4, 1])
        assert exact == pytest.approx(1 / 3)
        assert within1 == pytest.approx(2 / 3)

    def test_vei_rejects_out_of_scale(self) -> None:
        with pytest.raises(ValueError, match=r"VEI scale \[0, 8\]"):
            hm.vei_accuracy([9], [3])

    def test_vei_rejects_fractional(self) -> None:
        with pytest.raises(ValueError, match="integral"):
            hm.vei_accuracy([2.5], [2.0])


class TestProbabilisticAndKp:
    def test_brier_hand_example(self) -> None:
        # ((1-1)^2 + (0-0)^2 + (0.5-1)^2) / 3 = 0.25 / 3.
        assert hm.brier_score([1.0, 0.0, 0.5], [1, 0, 1]) == pytest.approx(0.25 / 3)

    def test_brier_uninformative_constant_half(self) -> None:
        assert hm.brier_score([0.5, 0.5], [0, 1]) == pytest.approx(0.25)

    def test_brier_rejects_probability_outside_unit_interval(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            hm.brier_score([1.2], [1])

    def test_brier_rejects_non_binary_outcomes(self) -> None:
        with pytest.raises(ValueError, match="binary"):
            hm.brier_score([0.5], [0.5])

    def test_kp_mae_hand_example(self) -> None:
        assert hm.kp_mae([2.0, 6.0], [3.0, 4.0]) == pytest.approx(1.5)

    def test_kp_mae_rejects_out_of_scale(self) -> None:
        with pytest.raises(ValueError, match=r"Kp scale \[0, 9\]"):
            hm.kp_mae([2.0], [9.5])

    def test_g_scale_buckets_match_noaa_thresholds(self) -> None:
        # Published NOAA Space Weather Scale onsets.
        assert hm.g_scale_bucket(4.99) == "G0"
        assert hm.g_scale_bucket(5.0) == "G1"
        assert hm.g_scale_bucket(6.0) == "G2"
        assert hm.g_scale_bucket(7.0) == "G3"
        assert hm.g_scale_bucket(8.0) == "G4"
        assert hm.g_scale_bucket(9.0) == "G5"
        assert hm.g_scale_bucket(0.0) == "G0"

    def test_g_bucket_accuracy_hand_example(self) -> None:
        # Buckets: (G0, G1, G3) vs (G0, G1, G2) -> 2/3 match.
        acc = hm.g_bucket_accuracy([1.0, 5.33, 7.67], [0.5, 5.1, 6.9])
        assert acc == pytest.approx(2 / 3)

    def test_g_scale_bucket_rejects_nan(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            hm.g_scale_bucket(float("nan"))
