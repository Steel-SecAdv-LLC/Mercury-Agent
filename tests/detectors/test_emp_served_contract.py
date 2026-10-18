# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
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

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from omni_mercury_engine.core.exceptions import InvalidObservationError
from omni_mercury_engine.detectors.energy import emp_detector as emp
from omni_mercury_engine.detectors.energy.emp_detector import (
    EMPDetector,
    EMPType,
    ThreatLevel,
)

if TYPE_CHECKING:
    from pathlib import Path

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


def test_nothing_measured_is_benign(detector: EMPDetector) -> None:
    result = detector.predict_emp({})
    assert (result.emp_detected, result.emp_type, result.threat_level) == (
        False,
        EMPType.NONE.value,
        ThreatLevel.BENIGN.value,
    )
    assert result.affected_infrastructure == []
    assert result.intentional_attack_probability is None


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


def test_e2_field_without_its_time_is_refused(detector: EMPDetector) -> None:
    with pytest.raises(InvalidObservationError, match="e2_time_us"):
        detector.predict_emp({"sensor_data": {**IEC_E1, "e2_field_vm": 100.0}})


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
    fs = 50e6
    t = np.arange(10_000) / fs
    lightning = np.concatenate([np.zeros(10), 5e3 * (np.exp(-1.5e4 * t) - np.exp(-6e6 * t))])
    result = detector.predict_emp({"sensor_data": {"waveform_vm": lightning, "sample_rate_hz": fs}})
    assert result.emp_type == EMPType.NONE.value
    assert result.e1_component_detected is False


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
