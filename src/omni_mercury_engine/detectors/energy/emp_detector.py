# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Electromagnetic pulse (EMP), intentional EMI and geomagnetic-disturbance classification.

What it assesses, from the measurements a caller supplies:

* **E1** (HEMP early time) from a field sensor: peak field above 10 kV/m,
  10-90% rise below 10 ns, frequency content within 1-1000 MHz. The IEC
  61000-2-9 benchmark (DOE HEMP memo, January 2021) is 50 kV/m x 1.3
  (exp(-4e7 t) - exp(-6e8 t)): 2.5 ns rise, 23 ns FWHM.
* **E2** (HEMP intermediate time): 10-100 V/m from about 0.01 us to 1 ms
  after the burst (DOE 2021: E0 = 100 V/m, a = 1e3 /s, b = 6e8 /s). Alone it
  is indistinguishable from a lightning field, so it is reported only after
  a detected E1.
* **E3** (HEMP late time) from a magnetometer: |dB/dt| of at least 450 nT/s
  for at least 1 s. Meta-R-321's blast-wave example is ~4500 nT within
  1-10 s, i.e. 450-4500 nT/s, while the largest geomagnetic storm rates are
  ~20 nT/s observed (March 1989) and ~290 nT/s in simulated Carrington
  extremes; the floor separates the two.
* **Geomagnetic disturbance (GIC)** below the E3 floor: the grid impact
  grades ``db_dt_nt_s`` x 60 on the cited 1-minute dB/dt tiers of
  :class:`~omni_mercury_engine.space.gic_detector.GICDetector`, so it takes
  the rate to be a 1-minute one (from values a minute apart). Only then do
  the two detectors agree on a storm: dB/dt grows with sampling rate, and a
  sub-minute rate overstates the tiers (over 17:00-17:20Z of the FRD
  2024-05-10 1-second record the peak 1-second rate, 12.75 nT/s, would grade
  "critical", while GICDetector's 1-minute means of the same samples peak at
  53.3 nT/min, "low"). A caller that states another cadence in
  ``dbdt_cadence_s`` gets the grid impact "unassessed" rather than a grade
  on tiers it was not measured on.
* **Intentional EMI (IEMI)** after IEC 61000-2-13 and Giri & Tesche (IEEE
  Trans. EMC 46(3), 2004): a radiated high-power (HPEM) environment is a
  peak field above 100 V/m; its band class follows the band ratio
  br = fH/fL of the narrowest band holding 90% of the energy (hypoband
  br <= 1.01, mesoband <= 3, sub-hyperband <= 10, hyperband above; SUMMA
  System Design and Assessment Note 41, 2014). An HPEM field with content in
  200 MHz-5 GHz, where IEMI upsets or damages electronics (Giri, Circuit and
  Electromagnetic System Design Note 50, 2006), and no E1 is flagged:
  lightning's significant content ends near 20 MHz and E1 is classified
  above, but a licensed high-power emitter (radar, broadcast) can produce
  it, so the verdict is "suspicious" with that check as the first action.
  :class:`IntentionalEMIDetector` adds an attack probability only once
  trained weights are loaded (:meth:`EMPDetector.load_neural_weights`;
  :func:`train_iemi_network` trains it on labelled ``iemi-signature-v1``
  features). No public labelled IEMI corpus exists, so no weights ship;
  untrained, its output follows the random initialisation (given only a
  1 MHz repetition rate, "critical" in 52 of torch seeds 0-99) and is not
  consulted.

From a sampled field record (``waveform_vm`` with ``sample_rate_hz``) the
peak, rise time, FWHM pulse duration, centre frequency and 90% energy band
are measured (:func:`analyze_field_waveform`) rather than taken as given. A
sensor array's arrival times locate the source (:func:`localize_tdoa`).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch
from torch import nn

from omni_mercury_engine.detectors._hazard_input import (
    InvalidObservationError,
    finite_array,
    finite_scalar,
)
from omni_mercury_engine.security.safe_torch import safe_torch_load
from omni_mercury_engine.space.gic_detector import (
    DBDT_CADENCE_S,
    DBDT_CADENCE_TOLERANCE,
    classify_dbdt_risk,
)

#: E2 (HEMP intermediate-time) amplitude floor and window after the burst,
#: microseconds (DOE 2021 HEMP memo: 10-100 V/m for ~0.01 us to 1 ms).
E2_MIN_FIELD_VM = 10.0
E2_WINDOW_US = (0.01, 1000.0)

#: E3 (HEMP late-time) floor on |dB/dt| and its minimum duration (see the
#: module docstring: Meta-R-321 blast wave 450-4500 nT/s over 1-10 s, above
#: every geomagnetic-storm estimate).
E3_MIN_DBDT_NT_PER_S = 450.0
E3_MIN_DURATION_S = 1.0

#: IEC 61000-2-13 high-power condition: a peak field exceeding 100 V/m
#: (26.5 W/m^2 plane-wave power density).
HPEM_MIN_FIELD_VM = 100.0

#: Frequency range in which IEMI upsets or damages electronics, MHz (Giri,
#: CESDN 50).
IEMI_BAND_MHZ = (200.0, 5000.0)

#: Upper band-ratio bound of each class below hyperband (SDAN 41, Table 2.2).
BAND_RATIO_CLASSES = ((1.01, "hypoband"), (3.0, "mesoband"), (10.0, "sub_hyperband"))

#: Energy fraction of the band that defines fL and fH.
BAND_ENERGY_FRACTION = 0.90

#: Propagation speed for time-difference-of-arrival location, m/s.
SPEED_OF_LIGHT_M_S = 299_792_458.0

#: Feature contract of :class:`IntentionalEMIDetector`: the fraction of the
#: record's spectral energy in 60 log-spaced bins over 1 MHz-10 GHz, then
#: log10 of the peak field (V/m), of 1 + repetition rate (Hz), of the rise
#: time (ns) and of the FWHM (ns). An unmeasurable rise or FWHM (a pulse
#: cut by the record, or a continuous wave) takes the record duration.
IEMI_FEATURE_SPEC = "iemi-signature-v1"
IEMI_SPECTRAL_BINS = 60
IEMI_SPECTRUM_RANGE_HZ = (1e6, 1e10)
IEMI_FEATURE_DIM = IEMI_SPECTRAL_BINS + 4


class EMPType(Enum):
    """EMP classifications."""

    NONE = "none"
    HEMP = "high_altitude_emp"
    NUCLEAR_EMP = "nuclear_emp"
    NON_NUCLEAR_EMP = "non_nuclear_emp"
    LIGHTNING = "lightning_surge"
    SOLAR_GIC = "solar_geomagnetic_current"
    INTENTIONAL_EMI = "intentional_emi"


class ThreatLevel(Enum):
    """Threat severity levels."""

    BENIGN = "benign"
    ANOMALOUS = "anomalous"
    SUSPICIOUS = "suspicious"
    THREAT = "threat"
    CRITICAL = "critical"


@dataclass
class EMPPredictionResult:
    """EMP prediction results.

    The fields through ``recovery_actions`` keep their released order, so
    positional construction is unchanged; fields added since follow them.
    """

    emp_detected: bool
    confidence: float
    emp_type: str
    threat_level: str

    e1_component_detected: bool = False
    e2_component_detected: bool = False
    e3_component_detected: bool = False

    field_strength_vm: float | None = None
    frequency_mhz: float | None = None
    #: Full width at half maximum of |E|, ns.
    pulse_duration_ns: float | None = None

    #: ``none`` (dB/dt = 0), ``low`` / ``moderate`` / ``high`` / ``critical``
    #: (the 1-minute GIC tiers), or ``unassessed`` (a stated dB/dt cadence
    #: other than 1 minute; see the module docstring).
    grid_impact_assessment: str = "none"
    affected_infrastructure: list[str] = field(default_factory=list)

    #: From :func:`localize_tdoa` when a sensor array is supplied.
    source_localization: dict[str, float] | None = None
    #: The trained IEMI network's probability; None until weights are loaded.
    #: (The released 0.0 default, and the random-weight output it was
    #: replaced by, claimed a probability no classifier had computed; see
    #: DEPRECATION.md section 6.12.)
    intentional_attack_probability: float | None = None

    protective_actions: list[str] = field(default_factory=list)
    recovery_actions: list[str] = field(default_factory=list)

    rise_time_ns: float | None = None
    peak_dbdt_nt_per_s: float | None = None
    #: IEC 61000-2-13: peak field above 100 V/m.
    hpem_environment: bool = False
    band_ratio: float | None = None
    band_class: str | None = None


def classify_band_ratio(band_ratio: float) -> str:
    """IEC 61000-2-13 band class for a band ratio fH/fL (>= 1, or inf)."""
    for bound, name in BAND_RATIO_CLASSES:
        if band_ratio <= bound:
            return name
    return "hyperband"


def _one_sided_energy(record: np.ndarray[Any, Any], sample_rate_hz: float) -> tuple[Any, Any, Any]:
    """Frequencies (Hz), amplitude spectrum and one-sided energy of a real record."""
    amplitude = np.abs(np.fft.rfft(record))
    energy = amplitude**2
    # Bins other than DC (and Nyquist, for an even length) carry the matching
    # negative frequency as well.
    stop = energy.size - 1 if record.size % 2 == 0 else energy.size
    energy[1:stop] *= 2.0
    return np.fft.rfftfreq(record.size, d=1.0 / sample_rate_hz), amplitude, energy


def _crossing(magnitude: np.ndarray[Any, Any], lo: int, level: float) -> float:
    """Fractional index where ``magnitude`` crosses ``level`` between ``lo`` and ``lo + 1``."""
    a, b = float(magnitude[lo]), float(magnitude[lo + 1])
    return lo + (level - a) / (b - a) if b != a else float(lo)


def analyze_field_waveform(field_vm: Any, sample_rate_hz: Any) -> dict[str, Any]:
    """Measure a sampled electric-field record.

    Args:
        field_vm: Field samples, V/m, one record holding the whole pulse.
        sample_rate_hz: Sampling rate, Hz.

    Returns:
        ``peak_field_vm``; ``peak_is_lower_bound``, True when |E| does not
        fall back below 90% of that maximum before the record ends (the
        record stops before the pulse peaks, or on its crest); ``rise_time_ns``
        (10-90% of the peak of |E|), None when the record starts after the
        pulse onset or ``peak_is_lower_bound`` (its 90% level is unknown);
        ``pulse_duration_ns`` (FWHM of |E|), None when the record cuts the
        pulse; ``center_frequency_mhz`` (spectral peak);
        ``band_low_mhz``/``band_high_mhz`` (narrowest band holding 90% of the
        energy); ``band_ratio`` (inf when the band reaches DC) and
        ``band_class``; ``record_duration_ns``.

    Raises:
        InvalidObservationError: On a non-finite, too short or all-zero
            record, or a non-positive sampling rate.
    """
    record = finite_array(field_vm, "field waveform", unit="V/m", ndim=1, min_size=16)
    fs = finite_scalar(sample_rate_hz, "sample_rate_hz", unit="Hz", lo=0.0)
    if fs == 0.0:
        raise InvalidObservationError("sample_rate_hz must be positive")
    magnitude = np.abs(record)
    peak = float(magnitude.max())
    if peak == 0.0:
        raise InvalidObservationError("field waveform is identically zero")
    ns_per_sample = 1e9 / fs
    p = int(np.argmax(magnitude))
    # Unless |E| falls back below the 90% level after its maximum, the record
    # may end before the pulse peaks: the maximum under-reads the peak and
    # the 90% point of the rise is not known.
    peak_is_lower_bound = not bool(np.any(magnitude[p + 1 :] < 0.9 * peak))

    rise_time_ns: float | None = None
    below = np.flatnonzero(magnitude[:p] < 0.1 * peak)
    if below.size and not peak_is_lower_bound:
        i10 = int(below[-1])
        i90 = i10 + 1 + int(np.argmax(magnitude[i10 + 1 : p + 1] >= 0.9 * peak))
        t10 = _crossing(magnitude, i10, 0.1 * peak)
        t90 = _crossing(magnitude, i90 - 1, 0.9 * peak)
        rise_time_ns = (t90 - t10) * ns_per_sample

    pulse_duration_ns: float | None = None
    above = np.flatnonzero(magnitude >= 0.5 * peak)
    first, last = int(above[0]), int(above[-1])
    if first > 0 and last < record.size - 1:
        start = _crossing(magnitude, first - 1, 0.5 * peak)
        stop = _crossing(magnitude, last, 0.5 * peak)
        pulse_duration_ns = (stop - start) * ns_per_sample

    freqs, amplitude, energy = _one_sided_energy(record, fs)
    cumulative = np.concatenate(([0.0], np.cumsum(energy)))
    target = BAND_ENERGY_FRACTION * cumulative[-1]
    starts = np.arange(energy.size)
    ends = np.searchsorted(cumulative, cumulative[starts] + target, side="left") - 1
    valid = ends < energy.size
    widths = np.where(valid, freqs[np.minimum(ends, energy.size - 1)] - freqs[starts], np.inf)
    lo_bin = int(np.argmin(widths))
    f_low, f_high = float(freqs[lo_bin]), float(freqs[ends[lo_bin]])
    band_ratio = f_high / f_low if f_low > 0.0 else math.inf

    return {
        "peak_field_vm": peak,
        "peak_is_lower_bound": peak_is_lower_bound,
        "rise_time_ns": rise_time_ns,
        "pulse_duration_ns": pulse_duration_ns,
        "center_frequency_mhz": float(freqs[int(np.argmax(amplitude))]) / 1e6,
        "band_low_mhz": f_low / 1e6,
        "band_high_mhz": f_high / 1e6,
        "band_ratio": band_ratio,
        "band_class": classify_band_ratio(band_ratio),
        "record_duration_ns": record.size * ns_per_sample,
    }


def iemi_signature(field_vm: Any, sample_rate_hz: Any, repetition_rate_hz: float = 0.0) -> Any:
    """The ``iemi-signature-v1`` feature vector of a field record.

    Args:
        field_vm: Field samples, V/m.
        sample_rate_hz: Sampling rate, Hz.
        repetition_rate_hz: Pulse repetition rate, Hz (0 for a single pulse).

    Returns:
        ``float32`` array of :data:`IEMI_FEATURE_DIM` values (see
        :data:`IEMI_FEATURE_SPEC`).
    """
    metrics = analyze_field_waveform(field_vm, sample_rate_hz)
    prf = finite_scalar(repetition_rate_hz, "repetition_rate_hz", unit="Hz", lo=0.0)
    record = np.asarray(field_vm, dtype=np.float64)
    freqs, _amplitude, energy = _one_sided_energy(record, float(sample_rate_hz))
    edges = np.geomspace(*IEMI_SPECTRUM_RANGE_HZ, IEMI_SPECTRAL_BINS + 1)
    spectral, _ = np.histogram(freqs, bins=edges, weights=energy)
    duration = metrics["record_duration_ns"]
    rise = metrics["rise_time_ns"] or duration
    fwhm = metrics["pulse_duration_ns"] or duration
    scalars = [
        math.log10(metrics["peak_field_vm"]),
        math.log10(1.0 + prf),
        math.log10(max(rise, 1e-3)),
        math.log10(max(fwhm, 1e-3)),
    ]
    return np.concatenate((spectral / float(energy.sum()), scalars)).astype(np.float32)


def assess_iemi_environment(
    peak_field_vm: float,
    band_low_mhz: float,
    band_high_mhz: float,
    *,
    repetition_rate_hz: float = 0.0,
    e1_detected: bool = False,
) -> dict[str, Any]:
    """Physics IEMI screen after IEC 61000-2-13 (see the module docstring).

    Args:
        peak_field_vm: Peak field, V/m.
        band_low_mhz: Lower edge of the 90% energy band, MHz.
        band_high_mhz: Upper edge, MHz (equal to the lower edge for a
            single reported frequency).
        repetition_rate_hz: Pulse repetition rate, Hz.
        e1_detected: Whether the record is a HEMP E1 (then it is not IEMI).

    Returns:
        ``hpem`` (field above 100 V/m), ``in_iemi_band``, ``repetitive``,
        ``iemi_suspected`` (HPEM in the IEMI band, not E1) and
        ``indicator_fraction`` (of the three indicators).
    """
    hpem = peak_field_vm > HPEM_MIN_FIELD_VM
    in_band = band_high_mhz >= IEMI_BAND_MHZ[0] and band_low_mhz <= IEMI_BAND_MHZ[1]
    repetitive = repetition_rate_hz > 0.0
    return {
        "hpem": hpem,
        "in_iemi_band": in_band,
        "repetitive": repetitive,
        "iemi_suspected": hpem and in_band and not e1_detected,
        "indicator_fraction": (int(hpem) + int(in_band) + int(repetitive)) / 3.0,
    }


def localize_tdoa(
    sensor_positions_m: Any,
    arrival_times_s: Any,
    *,
    wave_speed_m_s: float = SPEED_OF_LIGHT_M_S,
    max_iterations: int = 20,
) -> dict[str, float]:
    """Locate a pulse source from its arrival times at a sensor array.

    A linear least-squares solution (each range equation minus the earliest
    sensor's) starts a Gauss-Newton fit of ``t_i = t0 + |s - p_i| / v``. The
    fit is well conditioned for a source within a few array apertures; a
    distant source (a HEMP burst hundreds of km up) is determined in
    direction, not range.

    Args:
        sensor_positions_m: ``(n, 2)`` or ``(n, 3)`` sensor coordinates, m.
        arrival_times_s: ``(n,)`` arrival times, s, on one clock.
        wave_speed_m_s: Propagation speed (light in air by default).
        max_iterations: Gauss-Newton iteration cap.

    Returns:
        ``x_m``, ``y_m`` (and ``z_m`` in 3-D), ``emission_time_s``,
        ``rms_residual_ns`` and ``n_sensors``.

    Raises:
        InvalidObservationError: On non-finite input, fewer than ``d + 2``
            sensors in ``d`` dimensions, or a degenerate array geometry.
    """
    positions = finite_array(sensor_positions_m, "sensor_positions_m", unit="m", ndim=2)
    n, dim = positions.shape
    if dim not in (2, 3):
        raise InvalidObservationError(f"sensor_positions_m must be (n, 2) or (n, 3), got {dim}-D")
    times = finite_array(arrival_times_s, "arrival_times_s", unit="s", ndim=1)
    if times.size != n:
        raise InvalidObservationError(f"{times.size} arrival times for {n} sensors")
    if n < dim + 2:
        raise InvalidObservationError(f"TDOA in {dim}-D needs at least {dim + 2} sensors, got {n}")
    speed = finite_scalar(wave_speed_m_s, "wave_speed_m_s", unit="m/s", lo=0.0)

    ref = int(np.argmin(times))
    origin = positions[ref]
    rel = positions - origin
    ranges = (times - times[ref]) * speed  # path differences, m
    others = np.delete(np.arange(n), ref)
    # |s - r_i| = R_i + D, with D = |s| the range to the reference sensor:
    # squaring and subtracting |s|^2 = D^2 leaves a system linear in (s, D).
    design = np.column_stack([-2.0 * rel[others], -2.0 * ranges[others]])
    rhs = ranges[others] ** 2 - np.sum(rel[others] ** 2, axis=1)
    if np.linalg.matrix_rank(design) < dim + 1:
        raise InvalidObservationError("sensor geometry is degenerate for TDOA location")
    solution, *_ = np.linalg.lstsq(design, rhs, rcond=None)
    source, offset = solution[:dim], float(solution[dim])

    for _ in range(max_iterations):
        delta_vec = source - rel
        dist = np.maximum(np.linalg.norm(delta_vec, axis=1), 1e-12)
        residual = ranges - (dist - offset)
        jacobian = np.column_stack([delta_vec / dist[:, None], -np.ones(n)])
        step, *_ = np.linalg.lstsq(jacobian, residual, rcond=None)
        source = source + step[:dim]
        offset += float(step[dim])
        if float(np.linalg.norm(step)) < 1e-9:
            break

    dist = np.linalg.norm(source - rel, axis=1)
    residual = ranges - (dist - offset)
    located = source + origin
    result = {"x_m": float(located[0]), "y_m": float(located[1])}
    if dim == 3:
        result["z_m"] = float(located[2])
    result["emission_time_s"] = float(times[ref] - offset / speed)
    result["rms_residual_ns"] = float(np.sqrt(np.mean(residual**2)) / speed * 1e9)
    result["n_sensors"] = float(n)
    return result


class E1PulseDetector:
    """E1 component detection (prompt gamma ray pulse).

    Characteristics: Very fast (nanoseconds), high frequency
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        self.logger = logging.getLogger(__name__)

    def detect_e1_pulse(self, sensor_data: dict[str, Any]) -> dict[str, Any]:
        """Detect E1 pulse component.

        Args:
            sensor_data: Either a sampled record (``waveform_vm``, V/m, with
                ``sample_rate_hz``), measured by :func:`analyze_field_waveform`,
                or the summary values ``field_strength_vm``, ``rise_time_ns``,
                ``peak_frequency_mhz`` and optionally ``pulse_width_ns`` (FWHM).

        Returns:
            E1 pulse detection results
        """
        if "waveform_vm" in sensor_data:
            metrics = analyze_field_waveform(
                sensor_data["waveform_vm"], sensor_data.get("sample_rate_hz")
            )
            field_strength_vm = metrics["peak_field_vm"]
            rise_time_ns = metrics["rise_time_ns"]
            frequency_mhz = metrics["center_frequency_mhz"]
            pulse_width_ns = metrics["pulse_duration_ns"]
            band_low, band_high = metrics["band_low_mhz"], metrics["band_high_mhz"]
            band_ratio, band_class = metrics["band_ratio"], metrics["band_class"]
            # The upper band edge stands for the frequency content: a
            # unipolar pulse such as E1 has its spectral peak at DC.
            rule_frequency_mhz = band_high
        else:
            field_strength_vm = finite_scalar(
                sensor_data.get("field_strength_vm", 0.0), "field_strength_vm", unit="V/m", lo=0.0
            )
            rise_time_ns = finite_scalar(
                sensor_data.get("rise_time_ns", 1000.0), "rise_time_ns", unit="ns", lo=0.0
            )
            frequency_mhz = finite_scalar(
                sensor_data.get("peak_frequency_mhz", 1.0), "peak_frequency_mhz", unit="MHz", lo=0.0
            )
            pulse_width_ns = (
                finite_scalar(sensor_data["pulse_width_ns"], "pulse_width_ns", unit="ns", lo=0.0)
                if "pulse_width_ns" in sensor_data
                else None
            )
            band_low = band_high = rule_frequency_mhz = frequency_mhz
            band_ratio, band_class = None, None

        e1_threshold_vm = 10000.0
        e1_rise_time_ns = 10.0
        e1_freq_range = (1.0, 1000.0)

        e1_detected = (
            field_strength_vm > e1_threshold_vm
            and rise_time_ns is not None
            and rise_time_ns < e1_rise_time_ns
            and e1_freq_range[0] < rule_frequency_mhz < e1_freq_range[1]
        )

        severity = "critical" if e1_detected else "low"

        return {
            "e1_detected": e1_detected,
            "field_strength_vm": float(field_strength_vm),
            "rise_time_ns": None if rise_time_ns is None else float(rise_time_ns),
            "frequency_mhz": float(frequency_mhz),
            "pulse_duration_ns": pulse_width_ns,
            "band_low_mhz": float(band_low),
            "band_high_mhz": float(band_high),
            "band_ratio": band_ratio,
            "band_class": band_class,
            "severity": severity,
        }


class E2PulseDetector:
    """E2 component detection (intermediate-time HEMP, lightning-like).

    Characteristics: 10-100 V/m from ~0.01 us to 1 ms after the burst.
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        self.logger = logging.getLogger(__name__)

    def detect_e2_pulse(self, sensor_data: dict[str, Any], *, e1_detected: bool) -> dict[str, Any]:
        """Detect the E2 component following a detected E1.

        Args:
            sensor_data: ``e2_field_vm`` (peak field, V/m) and ``e2_time_us``
                (its time after the E1 onset, us).
            e1_detected: Whether an E1 was detected on the same record.

        Returns:
            ``e2_detected`` and the measured ``e2_field_vm`` (None if absent).

        Raises:
            InvalidObservationError: If the field is given without its time.
        """
        if "e2_field_vm" not in sensor_data:
            return {"e2_detected": False, "e2_field_vm": None}
        e2_field = finite_scalar(sensor_data["e2_field_vm"], "e2_field_vm", unit="V/m", lo=0.0)
        if "e2_time_us" not in sensor_data:
            raise InvalidObservationError(
                "e2_field_vm needs e2_time_us (time after the E1 onset, us) to place it in "
                f"the E2 window {E2_WINDOW_US[0]:g}-{E2_WINDOW_US[1]:g} us"
            )
        e2_time = finite_scalar(sensor_data["e2_time_us"], "e2_time_us", unit="us", lo=0.0)
        in_window = E2_WINDOW_US[0] <= e2_time <= E2_WINDOW_US[1]
        return {
            "e2_detected": e1_detected and in_window and e2_field >= E2_MIN_FIELD_VM,
            "e2_field_vm": e2_field,
        }


class E3PulseDetector:
    """E3 component detection (magnetohydrodynamic EMP).

    Characteristics: Slow (seconds to minutes), low frequency, GIC induction
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        self.logger = logging.getLogger(__name__)

    def detect_e3_pulse(self, magnetometer_data: dict[str, Any]) -> dict[str, Any]:
        """Detect E3 pulse (GIC).

        Args:
            magnetometer_data: ``db_dt_nt_s`` (|dB/dt|, nT/s),
                ``duration_seconds`` and optionally ``dbdt_cadence_s``, the
                spacing (s) of the samples the rate was taken from. Without
                it the rate is taken to be a 1-minute rate (see the module
                docstring).

        Returns:
            ``e3_detected``; ``peak_dbdt_nt_per_s``; ``grid_impact``: ``none``
            for dB/dt = 0, else the 1-minute GIC tier of ``db_dt_nt_s`` x 60
            (``low`` / ``moderate`` / ``high`` / ``critical``), or
            ``unassessed`` when ``dbdt_cadence_s`` is not 1 minute; and
            ``gic_amplitude_a``, always None: a GIC in amperes follows from
            the geoelectric field through a model of the network, not from
            dB/dt alone, and the released 0.1 x |dB/dt| had neither a source
            nor a unit (kept as a key for released callers; DEPRECATION.md
            section 6.12).

        Raises:
            InvalidObservationError: On non-finite values, a negative
                duration, or a non-positive ``dbdt_cadence_s``.
        """
        db_dt_nt_s = abs(
            finite_scalar(magnetometer_data.get("db_dt_nt_s", 0.0), "db_dt_nt_s", unit="nT/s")
        )
        duration_seconds = finite_scalar(
            magnetometer_data.get("duration_seconds", 0.0), "duration_seconds", unit="s", lo=0.0
        )
        cadence_s: float | None = None
        if magnetometer_data.get("dbdt_cadence_s") is not None:
            cadence_s = finite_scalar(
                magnetometer_data["dbdt_cadence_s"], "dbdt_cadence_s", unit="s", lo=0.0
            )
            if cadence_s == 0.0:
                raise InvalidObservationError("dbdt_cadence_s must be positive")

        # The former rule (> 2400 nT/s for more than 60 s) matched neither E3
        # phase: the blast wave lasts 1-10 s, the heave is slower.
        e3_detected = db_dt_nt_s >= E3_MIN_DBDT_NT_PER_S and duration_seconds >= E3_MIN_DURATION_S

        # Grid impact on the GIC detector's cited 1-minute dB/dt tiers. A rate
        # stated at another cadence is not graded on them: a sub-minute rate
        # overstates them and a coarser one under-reads them (GICDetector
        # averages the first and refuses the second).
        if db_dt_nt_s == 0.0:
            grid_impact = "none"
        elif cadence_s is not None and (
            abs(cadence_s - DBDT_CADENCE_S) > DBDT_CADENCE_TOLERANCE * DBDT_CADENCE_S
        ):
            grid_impact = "unassessed"
        else:
            grid_impact = _GRID_IMPACT[classify_dbdt_risk(db_dt_nt_s * 60.0)]

        return {
            "e3_detected": e3_detected,
            "peak_dbdt_nt_per_s": float(db_dt_nt_s),
            "grid_impact": grid_impact,
            "gic_amplitude_a": None,
        }


#: GIC dB/dt tier -> grid impact wording.
_GRID_IMPACT = {"low": "low", "moderate": "moderate", "high": "high", "severe": "critical"}

#: The least threat level each grid impact warrants (see
#: :meth:`EMPDetector._assess_threat_level`).
_GRID_IMPACT_THREAT = {
    "moderate": ThreatLevel.ANOMALOUS,
    "high": ThreatLevel.SUSPICIOUS,
    "critical": ThreatLevel.THREAT,
}

#: Threat levels from least to most severe (the enum's definition order).
_THREAT_ORDER = tuple(ThreatLevel)


class IntentionalEMIDetector(nn.Module):
    """Intentional electromagnetic interference (IEMI) detection.

    Distinguishes attacks from natural/accidental sources. Consulted by
    :class:`EMPDetector` only after trained weights are loaded; the input is
    the ``iemi-signature-v1`` vector (:func:`iemi_signature`).
    """

    def __init__(self, input_dim: int = IEMI_FEATURE_DIM) -> None:
        """Initialize the instance."""
        super().__init__()

        phi = 1.618

        self.signature_analyzer = nn.Sequential(
            nn.Linear(input_dim, int(128 * phi)),
            nn.BatchNorm1d(int(128 * phi)),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(int(128 * phi), int(64 * phi)),
            nn.BatchNorm1d(int(64 * phi)),
            nn.ReLU(),
            nn.Linear(int(64 * phi), 64),
        )

        self.attack_classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, em_signature: torch.Tensor) -> torch.Tensor:
        """Classify intentional vs.

        natural EM events.
                Note: This model uses BatchNorm which requires batch_size > 1 during
                training. For inference with any batch size, call model.eval() first.
        """
        # Ensure eval mode for inference to handle batch_size=1 with BatchNorm
        was_training = self.training
        if em_signature.size(0) == 1 and self.training:
            self.eval()

        features = self.signature_analyzer(em_signature)
        attack_prob = self.attack_classifier(features)

        # Restore training mode if it was changed
        if was_training and not self.training:
            self.train()

        return attack_prob


def _roc_auc(scores: np.ndarray[Any, Any], labels: np.ndarray[Any, Any]) -> float:
    """Mann-Whitney ROC AUC (ties count one half)."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (pos.size * neg.size))


def train_iemi_network(
    signatures: Any,
    labels: Any,
    *,
    epochs: int = 200,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.25,
    seed: int = 0,
) -> dict[str, Any]:
    """Train :class:`IntentionalEMIDetector` on labelled signatures.

    A stratified, seeded split holds out ``validation_fraction`` of each
    class; the operating point is the validation threshold of maximal
    Youden J (sensitivity + specificity - 1), and the validation ROC AUC is
    recorded. The result is a checkpoint for ``torch.save`` and
    :meth:`EMPDetector.load_neural_weights`; a model trained on simulated
    waveforms has learned the simulator, which its provenance should say.

    Args:
        signatures: ``(n, IEMI_FEATURE_DIM)`` ``iemi-signature-v1`` vectors.
        labels: ``(n,)`` 1 for intentional EMI, 0 otherwise.
        epochs: Full-batch Adam epochs.
        learning_rate: Adam learning rate.
        validation_fraction: Held-out share of each class.
        seed: Seed of the split and the initialisation.

    Returns:
        ``{"iemi_model", "feature_spec", "input_dim", "operating_point",
        "validation"}``.

    Raises:
        InvalidObservationError: On malformed features or labels, or fewer
            than 10 examples of either class.
    """
    x = finite_array(signatures, "signatures", unit="iemi-signature-v1", ndim=2)
    if x.shape[1] != IEMI_FEATURE_DIM:
        raise InvalidObservationError(
            f"signatures have {x.shape[1]} features; {IEMI_FEATURE_SPEC} has {IEMI_FEATURE_DIM}"
        )
    y = np.asarray(labels)
    if y.shape != (x.shape[0],) or not np.isin(y, (0, 1)).all():
        raise InvalidObservationError("labels must be one 0/1 value per signature")
    y = y.astype(np.int64)
    counts = np.bincount(y, minlength=2)
    if counts.min() < 10:
        raise InvalidObservationError(f"need at least 10 examples of each class, got {counts}")

    rng = np.random.default_rng(seed)
    val_idx = np.concatenate(
        [
            rng.permutation(np.flatnonzero(y == c))[: max(1, int(validation_fraction * counts[c]))]
            for c in (0, 1)
        ]
    )
    train_mask = np.ones(y.size, dtype=bool)
    train_mask[val_idx] = False

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = IntentionalEMIDetector(input_dim=IEMI_FEATURE_DIM)
        optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
        x_train = torch.tensor(x[train_mask], dtype=torch.float32)
        y_train = torch.tensor(y[train_mask], dtype=torch.float32).unsqueeze(1)
        loss_fn = nn.BCELoss()
        model.train()
        for _ in range(epochs):
            optimiser.zero_grad()
            loss = loss_fn(model(x_train), y_train)
            loss.backward()
            optimiser.step()

    model.eval()
    with torch.no_grad():
        scores = model(torch.tensor(x[val_idx], dtype=torch.float32)).numpy().ravel()
    val_labels = y[val_idx]
    candidates = np.unique(scores)
    youden = [
        float(np.mean(scores[val_labels == 1] >= t) + np.mean(scores[val_labels == 0] < t) - 1.0)
        for t in candidates
    ]
    best = int(np.argmax(youden))
    # Midway to the next lower score: the same validation J, with margin
    # instead of a threshold on the lowest positive score itself.
    lower = float(candidates[best - 1]) if best > 0 else 0.0
    threshold = 0.5 * (lower + float(candidates[best]))
    threshold = min(max(threshold, 1e-6), 1.0 - 1e-6)

    return {
        "iemi_model": model.state_dict(),
        "feature_spec": IEMI_FEATURE_SPEC,
        "input_dim": IEMI_FEATURE_DIM,
        "operating_point": {"attack_prob_threshold": threshold},
        "validation": {
            "n_samples": int(val_idx.size),
            "roc_auc": _roc_auc(scores, val_labels),
            "youden_j": float(max(youden)),
        },
    }


class EMPDetector:
    """Comprehensive EMP and electromagnetic surge detection system.

    Classifies E1, E2 and E3 HEMP components, storm-level geomagnetic
    disturbance and intentional EMI from measured fields, and locates the
    source from array arrival times (see the module docstring).
    """

    def __init__(
        self,
        enable_e1_detection: bool = True,
        enable_e3_detection: bool = True,
        enable_attack_classification: bool = True,
    ):
        """Initialize the instance."""
        self.enable_e1 = enable_e1_detection
        self.enable_e3 = enable_e3_detection
        self.enable_attack = enable_attack_classification

        self.e1_detector = E1PulseDetector() if enable_e1_detection else None
        self.e2_detector = E2PulseDetector() if enable_e1_detection else None
        self.e3_detector = E3PulseDetector() if enable_e3_detection else None
        self.emi_detector = IntentionalEMIDetector() if enable_attack_classification else None
        self._iemi_trained = False
        self._iemi_threshold = 0.5

        self.logger = logging.getLogger(__name__)

    def load_neural_weights(self, checkpoint_path: str) -> None:
        """Load trained IEMI-network weights; until then the network is not consulted.

        Args:
            checkpoint_path: A checkpoint from :func:`train_iemi_network`
                (``iemi_model`` state dict, ``feature_spec``,
                ``operating_point.attack_prob_threshold``).

        Raises:
            ValueError: If attack classification is disabled, the feature
                spec differs, or the threshold is not a probability.
        """
        if self.emi_detector is None:
            raise ValueError("attack classification is disabled; there is no network to load")
        checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
        spec = checkpoint.get("feature_spec")
        if spec != IEMI_FEATURE_SPEC:
            raise ValueError(f"checkpoint feature spec {spec!r} is not {IEMI_FEATURE_SPEC!r}")
        tau = float(checkpoint["operating_point"]["attack_prob_threshold"])
        if not np.isfinite(tau) or not (0.0 < tau < 1.0):
            raise ValueError(f"checkpoint operating-point threshold {tau} is not a probability")
        self.emi_detector.load_state_dict(checkpoint["iemi_model"])
        self.emi_detector.eval()
        self._iemi_threshold = tau
        self._iemi_trained = True
        self.logger.info(
            "IEMI network weights loaded from %s (threshold %.3f)", checkpoint_path, tau
        )

    def predict_emp(self, emp_data: dict[str, Any]) -> EMPPredictionResult:
        """Comprehensive EMP prediction.

        Args:
            emp_data: Measurements, each optional:
                - sensor_data: a field record (``waveform_vm`` V/m with
                  ``sample_rate_hz``) or ``field_strength_vm`` (V/m),
                  ``rise_time_ns``, ``peak_frequency_mhz``,
                  ``pulse_width_ns``; and ``e2_field_vm`` with ``e2_time_us``
                - magnetometer_data: ``db_dt_nt_s`` (|dB/dt|, nT/s),
                  ``duration_seconds`` and optionally ``dbdt_cadence_s``
                  (see :meth:`E3PulseDetector.detect_e3_pulse`)
                - signature_data: ``repetition_rate_hz``; and
                  ``signature_features`` (``iemi-signature-v1``) for the
                  trained network, otherwise derived from the record
                - sensor_array: ``positions_m`` and ``arrival_times_s`` for
                  :func:`localize_tdoa`

        Returns:
            EMP prediction with threat assessment and protective actions.

        Raises:
            InvalidObservationError: On non-finite or negative measurements.
        """
        result = EMPPredictionResult(
            emp_detected=False,
            confidence=0.0,
            emp_type=EMPType.NONE.value,
            threat_level=ThreatLevel.BENIGN.value,
        )

        components_detected = 0
        e1_result: dict[str, Any] | None = None
        signature_data = emp_data.get("signature_data", {})
        repetition_rate = finite_scalar(
            signature_data.get("repetition_rate_hz", 0.0), "repetition_rate_hz", unit="Hz", lo=0.0
        )

        if self.enable_e1 and "sensor_data" in emp_data and self.e1_detector is not None:
            e1_result = self.e1_detector.detect_e1_pulse(emp_data["sensor_data"])
            result.e1_component_detected = e1_result["e1_detected"]
            result.field_strength_vm = e1_result["field_strength_vm"]
            result.rise_time_ns = e1_result["rise_time_ns"]
            result.frequency_mhz = e1_result["frequency_mhz"]
            result.pulse_duration_ns = e1_result["pulse_duration_ns"]
            result.band_ratio = e1_result["band_ratio"]
            result.band_class = e1_result["band_class"]

            if e1_result["e1_detected"]:
                components_detected += 1
                result.confidence = max(result.confidence, 0.9)
                result.emp_detected = True

            assert self.e2_detector is not None
            e2_result = self.e2_detector.detect_e2_pulse(
                emp_data["sensor_data"], e1_detected=result.e1_component_detected
            )
            result.e2_component_detected = e2_result["e2_detected"]
            if e2_result["e2_detected"]:
                components_detected += 1

        if self.enable_e3 and "magnetometer_data" in emp_data and self.e3_detector is not None:
            e3_result = self.e3_detector.detect_e3_pulse(emp_data["magnetometer_data"])
            result.e3_component_detected = e3_result["e3_detected"]
            result.grid_impact_assessment = e3_result["grid_impact"]
            result.peak_dbdt_nt_per_s = e3_result["peak_dbdt_nt_per_s"]

            if e3_result["e3_detected"]:
                components_detected += 1
                result.confidence = max(result.confidence, 0.8)
                result.emp_detected = True

        # E1 is the HEMP early-time pulse (the IEC 61000-2-9 waveform), so E1
        # alone is HEMP, not "non-nuclear"; E1 with E3 confirms the burst.
        if result.e1_component_detected and result.e3_component_detected:
            result.emp_type = EMPType.NUCLEAR_EMP.value
        elif result.e1_component_detected or result.e3_component_detected:
            result.emp_type = EMPType.HEMP.value
        elif result.grid_impact_assessment in ("moderate", "high", "critical"):
            # Storm-level dB/dt below the E3 floor is a geomagnetic
            # disturbance, not an EMP; it carries its grid impact.
            result.emp_type = EMPType.SOLAR_GIC.value

        iemi_suspected = False
        if e1_result is not None:
            iemi = assess_iemi_environment(
                e1_result["field_strength_vm"],
                e1_result["band_low_mhz"],
                e1_result["band_high_mhz"],
                repetition_rate_hz=repetition_rate,
                e1_detected=result.e1_component_detected,
            )
            result.hpem_environment = iemi["hpem"]
            iemi_suspected = iemi["iemi_suspected"]
            if iemi_suspected and not result.emp_detected:
                result.emp_type = EMPType.INTENTIONAL_EMI.value
                result.confidence = max(result.confidence, iemi["indicator_fraction"])

        attack_confirmed = False
        if self.enable_attack:
            attack_result = self._classify_intentional_attack(
                signature_data, emp_data.get("sensor_data"), repetition_rate
            )
            result.intentional_attack_probability = attack_result["attack_probability"]
            attack_confirmed = attack_result["attack_detected"] and not result.emp_detected
            if attack_confirmed:
                result.emp_type = EMPType.INTENTIONAL_EMI.value
                result.confidence = max(result.confidence, attack_result["attack_probability"])

        if "sensor_array" in emp_data:
            array = emp_data["sensor_array"]
            result.source_localization = localize_tdoa(
                array.get("positions_m"), array.get("arrival_times_s")
            )

        result.threat_level = self._assess_threat_level(
            result, components_detected, iemi_suspected=iemi_suspected, attack=attack_confirmed
        )

        result.affected_infrastructure = self._identify_affected_infrastructure(result)
        result.protective_actions = self._generate_protective_actions(result)
        result.recovery_actions = self._generate_recovery_actions(result)

        return result

    def _classify_intentional_attack(
        self,
        signature_data: dict[str, Any],
        sensor_data: dict[str, Any] | None = None,
        repetition_rate_hz: float = 0.0,
    ) -> dict[str, Any]:
        """Trained-network IEMI probability; None while the network is untrained."""
        if self.emi_detector is None or not self._iemi_trained:
            return {"attack_detected": False, "attack_probability": None}

        if "signature_features" in signature_data:
            features = finite_array(
                signature_data["signature_features"],
                "signature_features",
                unit=IEMI_FEATURE_SPEC,
                ndim=1,
            )
        elif sensor_data is not None and "waveform_vm" in sensor_data:
            features = iemi_signature(
                sensor_data["waveform_vm"], sensor_data.get("sample_rate_hz"), repetition_rate_hz
            )
        else:
            return {"attack_detected": False, "attack_probability": None}
        if features.size != IEMI_FEATURE_DIM:
            raise InvalidObservationError(
                f"signature_features has {features.size} values; "
                f"{IEMI_FEATURE_SPEC} has {IEMI_FEATURE_DIM}"
            )

        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        self.emi_detector.eval()
        with torch.no_grad():
            attack_prob = float(self.emi_detector(features_tensor)[0].item())

        return {
            "attack_detected": attack_prob >= self._iemi_threshold,
            "attack_probability": attack_prob,
        }

    def _assess_threat_level(
        self,
        result: EMPPredictionResult,
        components: int,
        *,
        iemi_suspected: bool = False,
        attack: bool = False,
    ) -> str:
        """Assess overall threat level from the measured components.

        The level is the more severe of the EMP/IEMI verdict (critical for a
        confirmed E1 + E3 burst, threat for a detected EMP or a confirmed
        attack, suspicious for suspected IEMI, anomalous for a field above
        1 kV/m) and the floor the grid impact sets on its own:

        * ``critical`` (grid relays and load shedding advised) -> ``threat``,
          the highest level short of ``critical``, which is reserved for a
          confirmed burst and adds the EMP actions (hardened backups,
          Faraday-cage protocols) that a storm does not call for;
        * ``high`` (the same grid actions, one tier lower) -> ``suspicious``;
        * ``moderate`` -> ``anomalous``;
        * ``low``, ``none`` and ``unassessed`` set no floor.
        """
        if result.emp_type == EMPType.NUCLEAR_EMP.value:
            level = ThreatLevel.CRITICAL
        elif result.emp_detected or attack:
            level = ThreatLevel.THREAT
        elif iemi_suspected:
            level = ThreatLevel.SUSPICIOUS
        elif result.field_strength_vm and result.field_strength_vm > 1000:
            level = ThreatLevel.ANOMALOUS
        else:
            level = ThreatLevel.BENIGN
        floor = _GRID_IMPACT_THREAT.get(result.grid_impact_assessment, ThreatLevel.BENIGN)
        return max(level, floor, key=_THREAT_ORDER.index).value

    def _identify_affected_infrastructure(self, result: EMPPredictionResult) -> list[str]:
        """Identify affected critical infrastructure."""
        infrastructure = []

        if result.e1_component_detected or result.emp_type == EMPType.INTENTIONAL_EMI.value:
            infrastructure.extend(
                [
                    "Electronics and semiconductors",
                    "Communication systems",
                    "Computer networks",
                ]
            )

        if result.e2_component_detected:
            # E2 couples to long lines like lightning, after E1 may have
            # disabled their surge protection.
            infrastructure.append("Long-line surge protection")

        # The grid is listed only for E3 or a grid impact of moderate or more
        # (it was listed for any magnetometer reading, dB/dt = 0 included).
        if result.e3_component_detected or result.grid_impact_assessment in (
            "moderate",
            "high",
            "critical",
        ):
            infrastructure.extend(["Power transmission grid", "Transformers"])

        if result.emp_type == "nuclear_emp":
            infrastructure.extend(
                [
                    "All electronic infrastructure",
                    "Transportation systems",
                    "Financial networks",
                ]
            )

        return list(set(infrastructure))

    def _generate_protective_actions(self, result: EMPPredictionResult) -> list[str]:
        """Generate protective actions."""
        actions = []

        if result.threat_level == "critical":
            actions.append("CRITICAL EMP THREAT: Activate hardened backup systems")
            actions.append("Disconnect non-essential electronics")
            actions.append("Implement Faraday cage protocols")

        if result.grid_impact_assessment in ["critical", "high"]:
            actions.append("Activate grid protective relays")
            actions.append("Load shedding to reduce transformer stress")

        if result.emp_type == EMPType.INTENTIONAL_EMI.value:
            actions.append(
                "Check licensed high-power emitters (radar, broadcast) near the sensor "
                "before attributing intent"
            )
            actions.append("Alert cybersecurity and physical security teams")
            actions.append("Trace source of electromagnetic interference")

        return actions

    def _generate_recovery_actions(self, result: EMPPredictionResult) -> list[str]:
        """Generate recovery actions."""
        recovery = []

        if result.emp_detected:
            recovery.append("Assess extent of equipment damage")
            recovery.append("Prioritize critical infrastructure restoration")

        if result.emp_type == "nuclear_emp":
            recovery.append("Implement national emergency response plan")
            recovery.append("Coordinate with FEMA and DOE")

        return recovery
