# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What ``EMPDetector`` serves, on sourced reference cases.

E1: the IEC 61000-2-9 benchmark (DOE HEMP memo 2021: 50 kV/m, 2.5 ns rise,
23 ns FWHM). E2: the DOE benchmark window (10-100 V/m, ~0.01 us-1 ms). E3:
Meta-R-321's blast wave (~4500 nT within 1-10 s). Storms: the Gannon storm at
Sitka (553.7 nT/min) and the March 1989 Hydro-Quebec level (~480 nT/min).
IEMI: IEC 61000-2-13 HPEM (> 100 V/m) and the Giri-Tesche band ratios.
These fail on the previous detector: the blast wave was rejected (it required
more than 60 s), storms never produced a grid impact above "low", E1 alone
read "non-nuclear", the grid was listed for dB/dt = 0, NaN read benign, an
untrained network decided "intentional EMI" from its random weights, and E2,
pulse duration and source location were declared but never computed.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from omni_mercury_engine.core.exceptions import InputModalityError, InvalidObservationError
from omni_mercury_engine.detectors.energy import emp_detector as emp
from omni_mercury_engine.detectors.energy.emp_detector import (
    E3PulseDetector,
    EMPDetector,
    EMPPredictionResult,
    EMPType,
    ThreatLevel,
)
from omni_mercury_engine.space.gic_detector import DBDT_RISK_TIERS_NT_PER_MIN, GICDetector

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "space_weather"

IEC_E1 = {"field_strength_vm": 50_000.0, "rise_time_ns": 2.47, "peak_frequency_mhz": 100.0}
BLAST_WAVE = {"db_dt_nt_s": 4512.0 / 5.0, "duration_seconds": 5.0}
GANNON_SITKA = {"db_dt_nt_s": 553.7 / 60.0, "duration_seconds": 600.0}
HYDRO_QUEBEC_1989 = {"db_dt_nt_s": 480.0 / 60.0, "duration_seconds": 600.0}


def _iec_e1_waveform(fs: float = 20e9, n: int = 20_000, onset_s: float = 5e-9) -> np.ndarray:
    """IEC 61000-2-9 / DOE 2021 E1: 50 kV/m x 1.3 (exp(-4e7 t) - exp(-6e8 t))."""
    t = np.arange(n) / fs - onset_s
    return np.where(t >= 0, 50e3 * 1.3 * (np.exp(-4e7 * t) - np.exp(-6e8 * t)), 0.0)


def _damped_sine(
    f0_hz: float, q: float, peak_vm: float, fs: float = 40e9, n: int = 8000
) -> np.ndarray:
    """IEC 61000-2-13 hypo/mesoband damped sinusoid starting 2 ns into the record."""
    t = np.arange(n) / fs - 2e-9
    tau = q / (np.pi * f0_hz)
    return np.where(t > 0, peak_vm * np.exp(-t / tau) * np.sin(2 * np.pi * f0_hz * t), 0.0)


@pytest.fixture()
def detector() -> EMPDetector:
    return EMPDetector()


def test_nothing_measured_is_refused(detector: EMPDetector) -> None:
    """An empty record used to grade "benign": an all-clear on no measurement."""
    with pytest.raises(InvalidObservationError, match="nothing measured"):
        detector.predict_emp({})
    with pytest.raises(InvalidObservationError, match="nothing measured"):
        detector.predict_emp({"signature_data": {"repetition_rate_hz": 100.0}})


def test_quiet_field_is_benign(detector: EMPDetector) -> None:
    result = detector.predict_emp({"sensor_data": {"field_strength_vm": 5.0}})
    assert (result.emp_detected, result.emp_type, result.threat_level) == (
        False,
        EMPType.NONE.value,
        ThreatLevel.BENIGN.value,
    )
    assert result.affected_infrastructure == []
    assert result.intentional_attack_probability is None


@pytest.mark.parametrize(
    "record",
    [
        {"sensor_data": {"field_strength": 50_000.0}},  # misspelt: read as 0 V/m
        {"magnetometer_data": {"dbdt_nt_s": 5000.0}},  # misspelt: read as 0 nT/s
        {"sensor_data": {"rise_time_ns": 2.0}},  # no field measured
        {"sensor_array": {"positions_m": [[0.0, 0.0, 0.0]]}},  # half an array
    ],
)
def test_misspelt_or_unmeasured_section_is_refused(
    detector: EMPDetector, record: dict[str, Any]
) -> None:
    with pytest.raises(InvalidObservationError, match=r"unknown keys|no measurement"):
        detector.predict_emp(record)


@pytest.mark.parametrize("section", ["sensor_data", "magnetometer_data"])
def test_non_mapping_section_is_refused(detector: EMPDetector, section: str) -> None:
    with pytest.raises(InvalidObservationError, match="must be a mapping"):
        detector.predict_emp({section: [1.0, 2.0]})


def test_non_mapping_record_is_refused(detector: EMPDetector) -> None:
    with pytest.raises(InputModalityError, match="mapping"):
        detector.predict_emp([1.0, 2.0])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("extra", "where"),
    [
        ({"magnetometer_data": {"db_dt_nt_s": float("nan")}}, "magnetometer_data.db_dt_nt_s"),
        ({"magnetometer_data": {"dbdt": 900.0}}, "magnetometer_data.dbdt"),
        ({"magnetometer_data": [1.0]}, "magnetometer_data"),
        ({"radar_data": {"x": 1.0}}, "radar_data"),
        (
            {"sensor_array": {"positions_m": [[0.0, 0.0]], "arrival_times_s": [0.0, 1.0]}},
            "sensor_array",
        ),
    ],
)
def test_measured_e1_is_graded_beside_an_unusable_section(
    detector: EMPDetector, extra: dict[str, Any], where: str
) -> None:
    """Regression: one unusable section refused a measured IEC E1 pulse."""
    result = detector.predict_emp({"sensor_data": IEC_E1, **extra})
    assert result.e1_component_detected is True
    assert result.partial is True
    assert any(entry["where"] == where for entry in result.set_aside), result.set_aside


def test_clean_record_is_not_partial(detector: EMPDetector) -> None:
    result = detector.predict_emp({"sensor_data": IEC_E1})
    assert result.partial is False and result.set_aside == []


def test_unmeasured_rise_time_and_frequency_are_not_fabricated(detector: EMPDetector) -> None:
    """1000 ns and 1 MHz used to be reported as measured when absent."""
    result = detector.predict_emp({"sensor_data": {"field_strength_vm": 50_000.0}})
    assert result.rise_time_ns is None
    assert result.frequency_mhz is None
    assert not result.e1_component_detected  # the E1 rule needs both


def test_iec_e1_waveform_is_measured_to_the_benchmark() -> None:
    m = emp.analyze_field_waveform(_iec_e1_waveform(), 20e9)
    assert m["peak_field_vm"] == pytest.approx(50e3, rel=1e-3)
    assert m["rise_time_ns"] == pytest.approx(2.5, abs=0.1)
    assert m["pulse_duration_ns"] == pytest.approx(23.0, abs=0.5)
    assert 1.0 < m["band_high_mhz"] < 1000.0  # E1 content within 1-1000 MHz


def test_iec_e1_record_is_hemp(detector: EMPDetector) -> None:
    result = detector.predict_emp(
        {"sensor_data": {"waveform_vm": _iec_e1_waveform(), "sample_rate_hz": 20e9}}
    )
    assert result.e1_component_detected is True
    assert result.emp_type == EMPType.HEMP.value
    assert result.pulse_duration_ns == pytest.approx(23.0, abs=0.5)


def test_iec_e1_summary_is_hemp(detector: EMPDetector) -> None:
    result = detector.predict_emp({"sensor_data": IEC_E1})
    assert result.e1_component_detected is True
    assert result.emp_type == EMPType.HEMP.value
    assert result.threat_level == ThreatLevel.THREAT.value
    assert result.rise_time_ns == pytest.approx(2.47)


def test_e2_is_reported_after_e1(detector: EMPDetector) -> None:
    e2 = {"e2_field_vm": 100.0, "e2_time_us": 10.0}
    after_e1 = detector.predict_emp({"sensor_data": {**IEC_E1, **e2}})
    assert after_e1.e2_component_detected is True
    assert "Long-line surge protection" in after_e1.affected_infrastructure
    # Alone, an intermediate-time field is lightning-like, not E2.
    alone = detector.predict_emp({"sensor_data": {"field_strength_vm": 100.0, **e2}})
    assert alone.e2_component_detected is False
    late = detector.predict_emp(
        {"sensor_data": {**IEC_E1, "e2_field_vm": 100.0, "e2_time_us": 5e3}}
    )
    assert late.e2_component_detected is False


def test_e2_field_without_its_time_is_set_aside_and_e1_still_graded(
    detector: EMPDetector,
) -> None:
    """An E2 field with no time cannot be placed in the E2 window; E1 still decides."""
    result = detector.predict_emp({"sensor_data": {**IEC_E1, "e2_field_vm": 100.0}})
    assert result.e1_component_detected is True
    assert result.e2_component_detected is False
    (entry,) = result.set_aside
    assert entry["where"] == "sensor_data.e2"
    assert "e2_time_us" in entry["reason"]


def test_meta_r_321_blast_wave_is_e3(detector: EMPDetector) -> None:
    result = detector.predict_emp({"magnetometer_data": BLAST_WAVE})
    assert result.e3_component_detected is True
    assert result.emp_type == EMPType.HEMP.value
    assert "Power transmission grid" in result.affected_infrastructure


def test_e1_with_e3_is_a_confirmed_nuclear_burst(detector: EMPDetector) -> None:
    result = detector.predict_emp({"sensor_data": IEC_E1, "magnetometer_data": BLAST_WAVE})
    assert result.emp_type == EMPType.NUCLEAR_EMP.value
    assert result.threat_level == ThreatLevel.CRITICAL.value


@pytest.mark.parametrize("storm", [GANNON_SITKA, HYDRO_QUEBEC_1989])
def test_severe_storm_is_a_critical_grid_disturbance_not_an_emp(
    detector: EMPDetector, storm: dict[str, float]
) -> None:
    result = detector.predict_emp({"magnetometer_data": storm})
    assert result.e3_component_detected is False
    assert result.emp_detected is False
    assert result.emp_type == EMPType.SOLAR_GIC.value
    assert result.grid_impact_assessment == "critical"
    assert "Transformers" in result.affected_infrastructure
    assert "Load shedding to reduce transformer stress" in result.protective_actions


@pytest.mark.parametrize(
    ("tier", "threat"),
    [
        ("severe", ThreatLevel.THREAT),  # grid impact "critical": load shedding
        ("high", ThreatLevel.SUSPICIOUS),
        ("moderate", ThreatLevel.ANOMALOUS),
    ],
)
def test_grid_impact_sets_a_threat_floor(
    detector: EMPDetector, tier: str, threat: ThreatLevel
) -> None:
    # Each GIC tier's lower bound as a 1-minute rate (480 nT/min is the 1989
    # Hydro-Quebec level), so the grid impact is that tier's.
    nt_per_min = {name: bound for bound, name in DBDT_RISK_TIERS_NT_PER_MIN}[tier]
    result = detector.predict_emp(
        {"magnetometer_data": {"db_dt_nt_s": nt_per_min / 60.0, "duration_seconds": 600.0}}
    )
    assert result.emp_type == EMPType.SOLAR_GIC.value
    assert result.threat_level == threat.value


def test_threat_is_the_more_severe_of_verdict_and_grid_floor(detector: EMPDetector) -> None:
    sensor = {"waveform_vm": _damped_sine(1.3e9, 10.0, 2000.0), "sample_rate_hz": 40e9}
    moderate = {"db_dt_nt_s": 90.0 / 60.0, "duration_seconds": 600.0}
    # Suspected IEMI (suspicious) is not lowered by a moderate storm's floor ...
    result = detector.predict_emp({"sensor_data": sensor, "magnetometer_data": moderate})
    assert result.threat_level == ThreatLevel.SUSPICIOUS.value
    # ... and is raised by a critical one's.
    result = detector.predict_emp({"sensor_data": sensor, "magnetometer_data": HYDRO_QUEBEC_1989})
    assert result.threat_level == ThreatLevel.THREAT.value


def _frd_one_second() -> tuple[list[datetime], np.ndarray, np.ndarray]:
    data = json.loads((FIXTURES / "usgs_geomag_frd_1s_2024_05_10.json").read_text())["data"]
    times = [datetime.fromisoformat(t.replace("Z", "+00:00")) for t in data["times"]]
    series = {v["id"]: np.asarray(v["values"], dtype=float) for v in data["values"]}
    return times, series["X"], series["Y"]


@pytest.mark.parametrize("cadence_s", [1.0, 300.0])
def test_rate_at_another_cadence_is_not_graded_on_minute_tiers(
    detector: EMPDetector, cadence_s: float
) -> None:
    times, bx, by = _frd_one_second()
    peak_1s = float(np.max(np.hypot(np.diff(bx), np.diff(by))))  # nT/s, 12.75
    assert GICDetector().assess(times, bx, by).risk_level == "low"
    result = detector.predict_emp(
        {
            "magnetometer_data": {
                "db_dt_nt_s": peak_1s,
                "duration_seconds": 1.0,
                "dbdt_cadence_s": cadence_s,
            }
        }
    )
    assert result.grid_impact_assessment == "unassessed"
    assert result.emp_type == EMPType.NONE.value
    assert result.threat_level == ThreatLevel.BENIGN.value
    assert "Load shedding to reduce transformer stress" not in result.protective_actions


def test_one_minute_rate_agrees_with_the_gic_detector(detector: EMPDetector) -> None:
    times, bx, by = _frd_one_second()
    gic = GICDetector().assess(times, bx, by)
    result = detector.predict_emp(
        {
            "magnetometer_data": {
                "db_dt_nt_s": gic.peak_dbdt_nt_per_min / 60.0,
                "duration_seconds": 60.0,
                "dbdt_cadence_s": 60.0,
            }
        }
    )
    assert result.grid_impact_assessment == gic.risk_level == "low"


def test_non_positive_dbdt_cadence_is_refused(detector: EMPDetector) -> None:
    with pytest.raises(InvalidObservationError, match="dbdt_cadence_s"):
        detector.predict_emp({"magnetometer_data": {"db_dt_nt_s": 1.0, "dbdt_cadence_s": 0.0}})


def test_released_result_fields_keep_their_positions() -> None:
    released = [
        "emp_detected",
        "confidence",
        "emp_type",
        "threat_level",
        "e1_component_detected",
        "e2_component_detected",
        "e3_component_detected",
        "field_strength_vm",
        "frequency_mhz",
        "pulse_duration_ns",
        "grid_impact_assessment",
        "affected_infrastructure",
        "source_localization",
        "intentional_attack_probability",
        "protective_actions",
        "recovery_actions",
    ]
    names = [f.name for f in dataclasses.fields(EMPPredictionResult)]
    assert names[: len(released)] == released
    positional = EMPPredictionResult(
        False, 0.0, "none", "benign", False, False, False, 1.0, 2.0, 3.0
    )
    assert positional.pulse_duration_ns == 3.0
    assert positional.rise_time_ns is None


def test_released_e3_key_is_kept_without_a_fabricated_value() -> None:
    result = E3PulseDetector().detect_e3_pulse(HYDRO_QUEBEC_1989)
    assert "gic_amplitude_a" in result
    assert result["gic_amplitude_a"] is None


@pytest.mark.parametrize("n_samples", [160, 240])
def test_record_that_stops_before_the_pulse_falls_has_no_rise_time(n_samples: int) -> None:
    # At 20 GS/s the benchmark peaks 9.84 ns into the record: 160 samples end
    # on the rise, 240 end past the peak but above 90% of it.
    cut = emp.analyze_field_waveform(_iec_e1_waveform(n=n_samples), 20e9)
    assert cut["rise_time_ns"] is None
    assert cut["peak_is_lower_bound"] is True
    whole = emp.analyze_field_waveform(_iec_e1_waveform(), 20e9)
    assert whole["peak_is_lower_bound"] is False
    assert cut["peak_field_vm"] <= whole["peak_field_vm"]


def test_e3_floor_sits_above_every_storm_estimate() -> None:
    # ~20 nT/s observed in March 1989; ~290 nT/s in simulated Carrington extremes.
    assert emp.E3_MIN_DBDT_NT_PER_S > 290.0


def test_quiet_magnetometer_does_not_list_the_grid(detector: EMPDetector) -> None:
    result = detector.predict_emp({"magnetometer_data": {"db_dt_nt_s": 0.0}})
    assert result.grid_impact_assessment == "none"
    assert result.affected_infrastructure == []


@pytest.mark.parametrize(
    ("record", "band_class"),
    [
        (_damped_sine(1.3e9, 10.0, 2000.0), "mesoband"),
        (500.0 * np.sin(2 * np.pi * 2.8e9 * np.arange(40_000) / 40e9), "hypoband"),
        (_iec_e1_waveform(), "hyperband"),
    ],
)
def test_band_class_follows_the_band_ratio(record: np.ndarray, band_class: str) -> None:
    fs = 20e9 if band_class == "hyperband" else 40e9
    assert emp.analyze_field_waveform(record, fs)["band_class"] == band_class


def test_band_ratio_boundaries() -> None:
    assert [emp.classify_band_ratio(br) for br in (1.0, 1.01, 1.02, 3.0, 3.1, 10.0, 10.5)] == [
        "hypoband",
        "hypoband",
        "mesoband",
        "mesoband",
        "sub_hyperband",
        "sub_hyperband",
        "hyperband",
    ]


def test_in_band_hpem_record_is_suspected_iemi(detector: EMPDetector) -> None:
    record = _damped_sine(1.3e9, 10.0, 2000.0)
    result = detector.predict_emp(
        {
            "sensor_data": {"waveform_vm": record, "sample_rate_hz": 40e9},
            "signature_data": {"repetition_rate_hz": 100.0},
        }
    )
    assert result.hpem_environment is True
    assert result.band_class == "mesoband"
    assert result.emp_type == EMPType.INTENTIONAL_EMI.value
    assert result.threat_level == ThreatLevel.SUSPICIOUS.value
    assert result.protective_actions[0].startswith("Check licensed high-power emitters")
    assert result.intentional_attack_probability is None  # untrained network


@pytest.mark.parametrize("peak_vm", [50.0, 100.0])
def test_field_at_or_below_hpem_level_is_not_iemi(detector: EMPDetector, peak_vm: float) -> None:
    record = _damped_sine(1.3e9, 10.0, peak_vm)
    result = detector.predict_emp({"sensor_data": {"waveform_vm": record, "sample_rate_hz": 40e9}})
    assert result.emp_type == EMPType.NONE.value
    assert result.threat_level == ThreatLevel.BENIGN.value


def test_lightning_band_field_is_not_iemi(detector: EMPDetector) -> None:
    # 10 GS/s: the Nyquist frequency reaches the top of the IEMI band, so the
    # band screen, not the sampling, decides.
    fs = 10e9
    assert fs / 2.0 >= emp.IEMI_BAND_MHZ[1] * 1e6
    # The IEC 60060-1 1.2/50 us lightning impulse (double-exponential fit,
    # tau = 68.22 us and 0.4074 us) at 5 kV/m: above the 100 V/m HPEM level,
    # below the 10 kV/m E1 floor; 2**21 samples hold it to 5% of its peak.
    t = np.arange(2**21) / fs
    shape = np.exp(-t / 68.22e-6) - np.exp(-t / 0.4074e-6)
    lightning = 5e3 * shape / shape.max()
    metrics = emp.analyze_field_waveform(lightning, fs)
    assert metrics["band_high_mhz"] < 20.0  # lightning-band content only
    result = detector.predict_emp({"sensor_data": {"waveform_vm": lightning, "sample_rate_hz": fs}})
    assert result.hpem_environment is True
    assert result.e1_component_detected is False
    assert result.emp_type == EMPType.NONE.value
    # The same field strength in the IEMI band, at the same rate, is flagged.
    in_band = _damped_sine(1.3e9, 10.0, 5e3, fs=fs)
    flagged = detector.predict_emp({"sensor_data": {"waveform_vm": in_band, "sample_rate_hz": fs}})
    assert flagged.emp_type == EMPType.INTENTIONAL_EMI.value


def test_untrained_network_is_not_consulted() -> None:
    data = {
        "sensor_data": {"field_strength_vm": 2000.0},
        "signature_data": {"repetition_rate_hz": 1e6},
    }
    verdicts = {
        (r.emp_type, r.threat_level, r.intentional_attack_probability)
        for r in (EMPDetector().predict_emp(data) for _ in range(10))
    }
    assert verdicts == {(EMPType.NONE.value, ThreatLevel.ANOMALOUS.value, None)}


def _training_set(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Simulated signatures: damped sinusoids (1) against E1-like pulses (0)."""
    rng = np.random.default_rng(seed)
    features, labels = [], []
    for _ in range(24):
        record = _damped_sine(rng.uniform(0.3e9, 4e9), rng.uniform(3, 30), rng.uniform(300, 5e3))
        features.append(emp.iemi_signature(record, 40e9, rng.uniform(10, 1e3)))
        labels.append(1)
        a, b = rng.uniform(1e7, 8e7), rng.uniform(3e8, 1e9)
        t = np.arange(8000) / 40e9
        pulse = rng.uniform(5e3, 5e4) * (np.exp(-a * t) - np.exp(-b * t))
        features.append(emp.iemi_signature(pulse, 40e9, 0.0))
        labels.append(0)
    return np.stack(features), np.asarray(labels)


def test_trained_network_is_loaded_and_consulted(tmp_path: Path) -> None:
    x, y = _training_set(seed=1)
    checkpoint = emp.train_iemi_network(x, y, epochs=150, seed=0)
    assert checkpoint["feature_spec"] == emp.IEMI_FEATURE_SPEC
    assert checkpoint["validation"]["roc_auc"] > 0.9
    path = tmp_path / "iemi.pt"
    torch.save(checkpoint, path)

    detector = EMPDetector()
    detector.load_neural_weights(str(path))
    record = _damped_sine(1.0e9, 8.0, 1500.0)
    result = detector.predict_emp(
        {
            "sensor_data": {"waveform_vm": record, "sample_rate_hz": 40e9},
            "signature_data": {"repetition_rate_hz": 100.0},
        }
    )
    assert result.intentional_attack_probability is not None
    assert (
        result.intentional_attack_probability
        > checkpoint["operating_point"]["attack_prob_threshold"]
    )
    assert result.threat_level == ThreatLevel.THREAT.value


def test_checkpoint_with_another_feature_spec_is_refused(tmp_path: Path) -> None:
    x, y = _training_set(seed=2)
    checkpoint = emp.train_iemi_network(x, y, epochs=5, seed=0)
    checkpoint["feature_spec"] = "repetition-rate-padded"
    path = tmp_path / "iemi.pt"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="feature spec"):
        EMPDetector().load_neural_weights(str(path))


def test_training_needs_both_classes() -> None:
    x, y = _training_set(seed=3)
    with pytest.raises(InvalidObservationError, match="each class"):
        emp.train_iemi_network(x[y == 1], y[y == 1])


def test_tdoa_locates_the_source(detector: EMPDetector) -> None:
    positions = np.array(
        [[0, 0, 0], [100, 0, 5], [0, 120, -3], [90, 110, 8], [50, -60, 20], [-40, 70, 2]],
        dtype=float,
    )
    source = np.array([230.0, -40.0, 15.0])
    times = 1e-3 + np.linalg.norm(positions - source, axis=1) / emp.SPEED_OF_LIGHT_M_S
    result = detector.predict_emp(
        {"sensor_array": {"positions_m": positions, "arrival_times_s": times}}
    )
    loc = result.source_localization
    assert loc is not None
    np.testing.assert_allclose([loc["x_m"], loc["y_m"], loc["z_m"]], source, atol=1e-6)
    assert loc["emission_time_s"] == pytest.approx(1e-3, abs=1e-12)

    jitter = np.random.default_rng(0).normal(0.0, 0.1e-9, times.size)  # 0.1 ns timing
    noisy = emp.localize_tdoa(positions, times + jitter)
    assert np.linalg.norm([noisy["x_m"] - 230.0, noisy["y_m"] + 40.0, noisy["z_m"] - 15.0]) < 1.0


def test_tdoa_refuses_too_few_or_collinear_sensors() -> None:
    times = np.array([0.0, 1e-7, 2e-7, 3e-7])
    with pytest.raises(InvalidObservationError, match="at least 5"):
        emp.localize_tdoa(np.zeros((4, 3)), times)
    line = np.column_stack([np.arange(5.0) * 10, np.zeros(5)])
    with pytest.raises(InvalidObservationError, match="degenerate"):
        emp.localize_tdoa(line, np.arange(5.0) * 1e-8)


@pytest.mark.parametrize(
    "data",
    [
        {"sensor_data": {"field_strength_vm": float("nan")}},
        {"magnetometer_data": {"db_dt_nt_s": float("inf")}},
        {"sensor_data": {"field_strength_vm": -5.0}},
        {"sensor_data": {"waveform_vm": [float("nan")] * 32, "sample_rate_hz": 1e9}},
        {"sensor_data": {"waveform_vm": [1.0] * 32}},
    ],
)
def test_non_physical_measurements_are_refused(detector: EMPDetector, data: dict[str, Any]) -> None:
    with pytest.raises(InvalidObservationError):
        detector.predict_emp(data)
