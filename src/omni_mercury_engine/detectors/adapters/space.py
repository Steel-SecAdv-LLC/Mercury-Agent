# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Engine and registry adapters for the space-weather detectors.

Each adapter serves one domain detector through ``fit`` / ``detect`` /
``extract_features`` / ``predict`` over that detector's true input -- a row
of named measurements, a time-ordered window, or a record -- and maps it
one-to-one onto the detector's own entry point.  Scores are the detector's
own or its cited NOAA scale over the scale's top; decisions are the
detector's own (or the scale's first actionable level, named per adapter).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

import numpy as np

from omni_mercury_engine.core.exceptions import InputModalityError, InvalidObservationError
from omni_mercury_engine.detectors._hazard_input import finite_scalar
from omni_mercury_engine.detectors.hazard_adapter import (
    Assessment,
    HazardRecordAdapter,
    HazardRowAdapter,
    HazardWindowAdapter,
    nan_if_none,
    ordinal_score,
    utc_datetime,
)

__all__ = [
    "CMEArrivalAdapter",
    "GICWindowAdapter",
    "SEPProtonWindowAdapter",
    "ScintillationAdapter",
    "SolarGICCascadeAdapter",
    "SolarStormSnapshotAdapter",
]

#: Physical bounds checked before the detector sees a value (the detector
#: itself accepts e.g. a negative solar-wind speed).
_SOLAR_WIND_BOUNDS: dict[str, tuple[str, float | None, float | None]] = {
    "solar_wind_speed_km_s": ("km/s", 1.0, 5000.0),
    "bz_imf_nt": ("nT", -500.0, 500.0),
    "by_imf_nt": ("nT", -500.0, 500.0),
    "imf_magnitude_nt": ("nT", 0.0, 500.0),
    "proton_density_p_cm3": ("p/cm^3", 0.0, 1000.0),
    "proton_temperature_k": ("K", 1.0, 1e8),
    "flow_pressure_npa": ("nPa", 0.0, 1000.0),
    "flux_long_wm2": ("W/m^2", 0.0, 1.0),
    "proton_flux_pfu": ("pfu", 0.0, 1e7),
}


class SolarStormSnapshotAdapter(HazardRowAdapter):
    """Space-weather storm snapshot: solar wind/IMF, X-ray flux and protons.

    Each row is one snapshot: ``solar_wind_speed_km_s`` and ``bz_imf_nt``
    are required; ``by_imf_nt``, ``imf_magnitude_nt``,
    ``proton_density_p_cm3``, ``proton_temperature_k``, ``flow_pressure_npa``
    (the geomagnetic model's further inputs), ``flux_long_wm2`` (GOES
    0.1-0.8 nm) and ``proton_flux_pfu`` (>= 10 MeV) may be NaN (unmeasured;
    the detector then runs without them -- the geomagnetic network fills
    its own training medians, it is never handed a substitute here).

    ``SolarStormDetector.predict_solar_storm`` grades the snapshot on NOAA's
    three scales: G from its own ``geomagnetic_storm_level``, R
    (``radio_blackout_level``) and S (``radiation_storm_level``).  The score
    is the highest of the three over 5; the decision is any scale at level
    1 or above.
    """

    COLUMNS = tuple(_SOLAR_WIND_BOUNDS)
    OPTIONAL_COLUMNS = frozenset(COLUMNS) - {"solar_wind_speed_km_s", "bz_imf_nt"}
    FEATURE_NAMES = ("kp_index", "g_level", "r_level", "s_level", "confidence")
    _GEOMAG = (
        "solar_wind_speed_km_s",
        "bz_imf_nt",
        "by_imf_nt",
        "imf_magnitude_nt",
        "proton_density_p_cm3",
        "proton_temperature_k",
        "flow_pressure_npa",
    )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize with ``BaseDetector`` config; the detector loads lazily."""
        super().__init__(config)
        self._detector: Any = None

    def _analyze_row(self, row: np.ndarray[Any, Any]) -> tuple[float, bool, list[float]]:
        from omni_mercury_engine.evaluation.hazard_metrics import G_SCALE_NAMES

        measured = self._measured(row)
        for name, value in measured.items():
            unit, lo, hi = _SOLAR_WIND_BOUNDS[name]
            finite_scalar(value, name, unit=unit, lo=lo, hi=hi)
        if self._detector is None:
            from omni_mercury_engine.space.solar_storm_detector import SolarStormDetector

            self._detector = SolarStormDetector()
        storm_data: dict[str, Any] = {
            "magnetosphere_data": {k: v for k, v in measured.items() if k in self._GEOMAG}
        }
        if "flux_long_wm2" in measured:
            storm_data["xray_data"] = {"flux_long_wm2": measured["flux_long_wm2"]}
        if "proton_flux_pfu" in measured:
            storm_data["proton_flux_pfu"] = measured["proton_flux_pfu"]
        result = self._detector.predict_solar_storm(storm_data)
        if result.geomagnetic_storm_level not in G_SCALE_NAMES:
            raise InvalidObservationError(
                f"solar storm prediction returned no G level ({result.geomagnetic_storm_level!r})"
            )
        g = G_SCALE_NAMES.index(result.geomagnetic_storm_level)
        r = int(result.radio_blackout_level)
        s = result.radiation_storm_level
        top = max(g, r, 0 if s is None else int(s))
        return (
            top / 5.0,
            top >= 1,
            [nan_if_none(result.kp_index), float(g), float(r), nan_if_none(s), result.confidence],
        )


class CMEArrivalAdapter(HazardRowAdapter):
    """Earth arrival of a coronal mass ejection from its cone kinematics.

    Each row is one CME at 21.5 solar radii: ``speed_km_s``,
    ``latitude_deg`` / ``longitude_deg`` (Stonyhurst; Earth at 0, 0),
    ``half_angle_deg`` and ``time_21_5_epoch_s`` (UTC epoch seconds).
    ``CMEArrivalDetector.predict`` (Gopalswamy ESA + Vrsnak DBM ensemble)
    decides ``earth_directed``; the score is its own directedness class on
    the ladder miss < unlikely < flank < head_on, so the decision is exactly
    score >= 2/3.  (Its ``confidence`` is model agreement scaled by
    geometry, not severity -- a fast head-on CME can score 0 -- so it is a
    feature, not the score.)
    """

    COLUMNS = ("speed_km_s", "latitude_deg", "longitude_deg", "half_angle_deg", "time_21_5_epoch_s")
    FEATURE_NAMES = (
        "angular_separation_deg",
        "most_probable_arrival_hours",
        "earliest_arrival_hours",
        "latest_arrival_hours",
        "arrival_speed_km_s",
        "model_spread_hours",
        "confidence",
    )
    _LADDER = ("miss", "unlikely", "flank", "head_on")

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize with ``BaseDetector`` config; the detector loads lazily."""
        super().__init__(config)
        self._detector: Any = None

    def _analyze_row(self, row: np.ndarray[Any, Any]) -> tuple[float, bool, list[float]]:
        from omni_mercury_engine.space.cme_arrival_detector import (
            CMEArrivalDetector,
            CMEKinematics,
        )

        if self._detector is None:
            self._detector = CMEArrivalDetector()
        speed, lat, lon, half, epoch = (float(v) for v in row)
        try:
            p = self._detector.predict(
                CMEKinematics(
                    speed_km_s=speed,
                    latitude_deg=lat,
                    longitude_deg=lon,
                    half_angle_deg=half,
                    time_21_5=utc_datetime(epoch, "time_21_5_epoch_s"),
                )
            )
        except ValueError as e:  # the detector's domain/validity refusals
            raise InvalidObservationError(str(e)) from e
        score = ordinal_score(p.directedness, self._LADDER)
        return (
            score,
            bool(p.earth_directed),
            [
                p.angular_separation_deg,
                p.most_probable_arrival_hours,
                p.earliest_arrival_hours,
                p.latest_arrival_hours,
                p.arrival_speed_km_s,
                p.model_spread_hours,
                p.confidence,
            ],
        )


class SEPProtonWindowAdapter(HazardWindowAdapter):
    """Solar energetic particle (radiation) storm from a proton-flux window.

    The window is ``[n, 3]`` over ``epoch_s`` (UTC epoch seconds, ascending),
    ``flux_ge10mev_pfu`` and ``flux_ge100mev_pfu`` (pfu, GOES integral
    channels); the >= 100 MeV column is either measured throughout or NaN
    throughout.  ``SEPStormDetector.assess`` decides the NOAA proton event --
    3 consecutive samples at or above 10 pfu (``onset_detected``) -- and the
    score is its NOAA S level (from the peak) over 5, so a lone spike scores
    S1 but is not an event.
    """

    COLUMNS = ("epoch_s", "flux_ge10mev_pfu", "flux_ge100mev_pfu")
    OPTIONAL_COLUMNS = frozenset({"flux_ge100mev_pfu"})
    MIN_SAMPLES = 1
    FEATURE_NAMES = (
        "peak_flux_10mev_pfu",
        "s_level",
        "onset_detected",
        "event_active",
        "peak_flux_100mev_pfu",
    )

    def _assess(self, unit: Any) -> Assessment:
        from omni_mercury_engine.space.sep_storm_detector import SEPStormDetector

        if self._detector is None:
            self._detector = SEPStormDetector()
        flux100 = self.column(unit, "flux_ge100mev_pfu")
        gaps = np.isnan(flux100)
        if gaps.any() and not gaps.all():
            raise InvalidObservationError(
                "flux_ge100mev_pfu is partly unmeasured; it must be measured on every sample "
                "or NaN on every sample"
            )
        times = [utc_datetime(t, "epoch_s") for t in self.column(unit, "epoch_s")]
        try:
            a = self._detector.assess(
                times,
                self.column(unit, "flux_ge10mev_pfu"),
                flux_ge100mev_pfu=None if gaps.all() else flux100,
            )
        except ValueError as e:
            raise InvalidObservationError(str(e)) from e
        s_level = int(a.s_scale.removeprefix("S"))
        return Assessment(
            score=s_level / 5.0,
            flag=bool(a.onset_detected),
            features=(
                a.peak_flux_10mev_pfu,
                float(s_level),
                float(a.onset_detected),
                float(a.event_active),
                nan_if_none(a.peak_flux_100mev_pfu),
            ),
            result=a,
        )


class GICWindowAdapter(HazardWindowAdapter):
    """Geomagnetically induced current risk from a magnetometer window.

    The window is ``[n, 3]`` over ``epoch_s`` (UTC epoch seconds, strictly
    ascending, real timestamps -- gaps and sub-minute data are handled by
    the detector), ``bx_nt`` and ``by_nt`` (nT); NaN marks a gap, and an
    all-NaN ``by_nt`` selects the detector's H-only mode.  ``config`` may set
    ``observatory`` and the detector's ``conductivity_model`` /
    ``sigma_s_per_m``.  ``GICDetector.assess`` grades the dB/dt tier (low,
    moderate >= 90, high >= 300, severe >= 480 nT/min; Pulkkinen et al.
    2013, Meta-R-321); the score is the tier over 3 and the decision is >=
    moderate.  Unlike ``GICMinuteSeriesAdapter`` this keeps the real sample
    times, gaps, the geoelectric-field estimate and per-component peaks, and
    serves the window's verdict once.
    """

    COLUMNS = ("epoch_s", "bx_nt", "by_nt")
    OPTIONAL_COLUMNS = frozenset({"bx_nt", "by_nt"})
    FEATURE_NAMES = (
        "peak_dbdt_nt_per_min",
        "sustained_dbdt_nt_per_min",
        "geoelectric_peak_v_per_km",
        "risk_tier",
        "n_gaps",
    )
    _TIERS = ("low", "moderate", "high", "severe")

    def _assess(self, unit: Any) -> Assessment:
        from omni_mercury_engine.space.gic_detector import GICDetector

        if self._detector is None:
            kwargs = {
                k: self._config[k]
                for k in ("conductivity_model", "sigma_s_per_m")
                if k in self._config
            }
            self._detector = GICDetector(**kwargs)
        by = self.column(unit, "by_nt")
        times = [utc_datetime(t, "epoch_s") for t in self.column(unit, "epoch_s")]
        try:
            a = self._detector.assess(
                times,
                self.column(unit, "bx_nt"),
                by_nt=None if np.isnan(by).all() else by,
                observatory=str(self._config.get("observatory", "")),
            )
        except ValueError as e:
            raise InvalidObservationError(str(e)) from e
        tier = self._TIERS.index(a.risk_level)
        return Assessment(
            score=tier / 3.0,
            flag=tier >= 1,
            features=(
                a.peak_dbdt_nt_per_min,
                nan_if_none(a.sustained_dbdt_nt_per_min),
                nan_if_none(a.geoelectric_peak_v_per_km),
                float(tier),
                float(a.n_gaps),
            ),
            result=a,
        )


class ScintillationAdapter(HazardRecordAdapter):
    """GNSS ionospheric scintillation measured from raw receiver samples.

    Each record holds ``intensity`` (signal power samples, linear units)
    and/or ``phase_rad`` (carrier phase, radians) with ``sample_rate_hz``
    (required with phase; conventionally a 60 s window at >= 50 Hz).
    ``IonosphericScintillationDetector.measure`` computes S4 and sigma-phi
    and classes the GNSS degradation weak / moderate (>= 0.3) / strong
    (> 0.6); the score is that class over 2 and the decision is >= moderate.
    """

    FEATURE_NAMES = ("s4", "sigma_phi_rad", "degradation_level")
    _LADDER = ("weak", "moderate", "strong")

    def _parse(self, record: Mapping[str, Any]) -> Any:
        if "intensity" not in record and "phase_rad" not in record:
            raise InvalidObservationError("a scintillation record needs intensity and/or phase_rad")
        parsed: dict[str, Any] = {}
        for key in ("intensity", "phase_rad"):
            if key in record:
                try:
                    parsed[key] = np.asarray(record[key], dtype=np.float64)
                except (TypeError, ValueError) as e:
                    raise InputModalityError(f"{key} must be a numeric sample array") from e
        if "sample_rate_hz" in record:
            parsed["sample_rate_hz"] = finite_scalar(
                record["sample_rate_hz"], "sample_rate_hz", unit="Hz", lo=1e-6
            )
        return parsed

    def _assess(self, unit: Any) -> Assessment:
        from omni_mercury_engine.space.ionospheric_scintillation_detector import (
            IonosphericScintillationDetector,
        )

        if self._detector is None:
            self._detector = IonosphericScintillationDetector()
        try:
            m = self._detector.measure(**unit)
        except ValueError as e:
            raise InvalidObservationError(str(e)) from e
        level = self._LADDER.index(m.gnss_degradation)
        return Assessment(
            score=ordinal_score(m.gnss_degradation, self._LADDER),
            flag=level >= 1,
            features=(nan_if_none(m.s4), nan_if_none(m.sigma_phi_rad), float(level)),
            result=m,
        )


class SolarGICCascadeAdapter(HazardRecordAdapter):
    """The solar-to-grid cascade evaluated at one time from its evidence.

    Each record is one evaluation: ``now`` (time), ``flares`` (DONKI FLR
    records), ``cme_analyses`` (DONKI CMEAnalysis records), ``kp_series``
    (``[[time, kp], ...]``), ``mag_times`` / ``mag_bx_nt`` / ``mag_by_nt``
    (magnetometer minute series; ``mag_by_nt`` optional) and
    ``observatory``.  Times are datetimes, ISO-8601 strings or epoch
    seconds.  ``SolarGICCascadeDetector.evaluate`` awards a stage by strict
    escalation (quiet < watch < warning < storm_in_progress); the score is
    the stage over 3 and the decision is any stage above quiet.
    """

    REQUIRED_KEYS = frozenset({"now"})
    FEATURE_NAMES = (
        "stage_level",
        "earth_directed_cmes",
        "arrival_window_open",
        "peak_dbdt_nt_per_min",
    )

    def _parse(self, record: Mapping[str, Any]) -> Any:
        from omni_mercury_engine.space.solar_gic_cascade import CascadeInputs

        def _list(key: str) -> list[Any]:
            value = record.get(key, [])
            if not isinstance(value, (list, tuple)):
                raise InputModalityError(f"{key} must be a list of records")
            return list(value)

        kp_series = []
        for i, pair in enumerate(_list("kp_series")):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise InputModalityError(f"kp_series[{i}] must be a [time, kp] pair")
            kp_series.append(
                (
                    utc_datetime(pair[0], f"kp_series[{i}] time"),
                    finite_scalar(pair[1], f"kp_series[{i}] kp", unit="Kp", lo=0.0, hi=9.0),
                )
            )
        mag_times = record.get("mag_times")
        bx, by = record.get("mag_bx_nt"), record.get("mag_by_nt")
        if (mag_times is None) != (bx is None):
            raise InvalidObservationError("mag_times and mag_bx_nt come together or not at all")
        return CascadeInputs(
            now=utc_datetime(record["now"], "now"),
            flares=_list("flares"),
            cme_analyses=_list("cme_analyses"),
            kp_series=kp_series,
            mag_times=(
                None
                if mag_times is None
                else [utc_datetime(t, f"mag_times[{i}]") for i, t in enumerate(mag_times)]
            ),
            mag_bx_nt=None if bx is None else np.asarray(bx, dtype=np.float64),
            mag_by_nt=None if by is None else np.asarray(by, dtype=np.float64),
            observatory=str(record.get("observatory", "")),
        )

    def _assess(self, unit: Any) -> Assessment:
        from omni_mercury_engine.space.solar_gic_cascade import (
            CascadeStage,
            SolarGICCascadeDetector,
        )

        if self._detector is None:
            self._detector = SolarGICCascadeDetector()
        try:
            a = self._detector.evaluate(unit)
        except ValueError as e:
            raise InvalidObservationError(str(e)) from e
        stages = tuple(CascadeStage)
        level = stages.index(a.stage)
        gic = a.gic_assessment
        return Assessment(
            score=level / (len(stages) - 1),
            flag=level >= 1,
            features=(
                float(level),
                float(len(a.earth_directed_cmes)),
                float(a.active_arrival_window is not None),
                float("nan") if gic is None else float(gic.peak_dbdt_nt_per_min),
            ),
            result=a,
        )
