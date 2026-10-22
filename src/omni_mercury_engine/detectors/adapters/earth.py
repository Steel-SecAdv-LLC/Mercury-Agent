# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Engine and registry adapters for the geophysical, hydro-meteorological and marine detectors.

Each adapter serves one domain detector through ``fit`` / ``detect`` /
``extract_features`` / ``predict`` over that detector's true input and maps
it one-to-one onto the detector's own entry point.  Scores are the
detector's own confidence or probability, or its cited ladder (USGS volcano
alert levels, EAWS danger levels, ...) over the ladder's top; decisions are
the detector's own, or the ladder's first actionable level where the
detector defines none (named per adapter).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from omni_mercury_engine.core.exceptions import InputModalityError, InvalidObservationError
from omni_mercury_engine.detectors.hazard_adapter import (
    Assessment,
    HazardRecordAdapter,
    HazardRowAdapter,
    nan_if_none,
    ordinal_score,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "AvalancheProfileAdapter",
    "EqTsunamiCascadeAdapter",
    "FireDebrisFlowAdapter",
    "LandslideAdapter",
    "MarineEcosystemAdapter",
    "RockfallAdapter",
    "SubsidenceAdapter",
    "SurgeFloodCascadeAdapter",
    "VolcanicAdapter",
    "WildfireSceneAdapter",
]


def _domain_refusal(error: ValueError) -> ValueError:
    """The detector's own refusal under the adapters' typed errors.

    A typed refusal (wrong modality, invalid observation) passes through
    unchanged; a plain ``ValueError`` is a domain refusal of the values.
    """
    if isinstance(error, (InputModalityError, InvalidObservationError)):
        return error
    return InvalidObservationError(str(error))


class VolcanicAdapter(HazardRecordAdapter):
    """Volcanic eruption forecast from a snapshot of monitoring streams.

    Each record is ``VolcanicEruptionDetector.predict_eruption``'s own input:
    ``seismic_sequence``, ``thermal_data``, ``gas_data``, ``insar_data``,
    ``schumann_elf``, ``precursor_rate`` and/or ``fused_features`` (the
    volcano-seismic-v1 vector, standardised as the checkpoint requires).
    Score and decision are the detector's own ``confidence`` and
    ``eruption_imminent``; its USGS alert level is a feature.
    """

    ALLOWED_KEYS = frozenset(
        {
            "seismic_sequence",
            "thermal_data",
            "gas_data",
            "insar_data",
            "schumann_elf",
            "precursor_rate",
            "fused_features",
        }
    )
    FEATURE_NAMES = (
        "confidence",
        "alert_level",
        "seismic_swarm_detected",
        "thermal_anomaly_detected",
        "gas_flux_anomaly",
        "deformation_detected",
        "time_to_eruption_hours",
    )
    _ALERTS = ("normal", "advisory", "watch", "warning")

    def _assess(self, unit: Any) -> Assessment:
        from omni_mercury_engine.detectors.geological.volcanic import VolcanicEruptionDetector

        if self._detector is None:
            self._detector = VolcanicEruptionDetector()
        try:
            r = self._detector.predict_eruption(unit)
        except ValueError as e:
            raise _domain_refusal(e) from e
        return Assessment(
            score=float(r.confidence),
            flag=bool(r.eruption_imminent),
            features=(
                float(r.confidence),
                float(self._ALERTS.index(r.alert_level)),
                float(r.seismic_swarm_detected),
                float(r.thermal_anomaly_detected),
                float(r.gas_flux_anomaly),
                float(r.deformation_detected),
                nan_if_none(r.time_to_eruption_hours),
            ),
            result=r,
        )


class LandslideAdapter(HazardRowAdapter):
    """Landslide likelihood from one site's monitoring row.

    Columns, grouped as ``LandslideDetector.predict_landslide`` reads them:
    rainfall ``intensity_mm_hr``, ``duration_hours``, ``antecedent_7day_mm``;
    seismic ``pga_g``, ``magnitude``, ``distance_km``; ``slope_angle_deg``;
    slope sensors ``soil_saturation_pct``, ``displacement_rate_mm_day``; and
    ``snowmelt_mm_day``.  NaN marks an unmeasured channel: it is left out of
    the record, never replaced (no rain is 0 mm/h, a measurement), and a row
    with nothing measured is refused by the detector.  The decision is the
    detector's own ``landslide_imminent`` (a trigger fired AND the
    slope-failure probability clears its bar); the score is that
    slope-failure probability, the model's own estimate.
    """

    _SECTIONS: ClassVar[dict[str, tuple[str, ...]]] = {
        "rainfall_data": ("intensity_mm_hr", "duration_hours", "antecedent_7day_mm"),
        "seismic_data": ("pga_g", "magnitude", "distance_km"),
        "slope_data": ("slope_angle_deg",),
        "sensor_data": ("soil_saturation_pct", "displacement_rate_mm_day"),
        "weather_data": ("snowmelt_mm_day",),
    }
    COLUMNS = tuple(key for keys in _SECTIONS.values() for key in keys)
    OPTIONAL_COLUMNS = frozenset(COLUMNS)
    FEATURE_NAMES = (
        "slope_failure_probability",
        "confidence",
        "rainfall_trigger",
        "seismic_trigger",
        "risk_level",
    )
    _RISK = ("low", "moderate", "high", "very_high", "extreme")

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize with ``BaseDetector`` config; the detector loads lazily."""
        super().__init__(config)
        self._detector: Any = None

    def _analyze_row(self, row: np.ndarray[Any, Any]) -> tuple[float, bool, list[float]]:
        from omni_mercury_engine.detectors.geological.landslide import LandslideDetector

        if self._detector is None:
            self._detector = LandslideDetector()
        m = self._measured(row)
        record: dict[str, Any] = {}
        for section, keys in self._SECTIONS.items():
            values = {k: m[k] for k in keys if k in m}
            if values:
                record[section] = values
        try:
            r = self._detector.predict_landslide(record)
        except ValueError as e:
            raise _domain_refusal(e) from e
        return (
            float(r.slope_failure_probability),
            bool(r.landslide_imminent),
            [
                float(r.slope_failure_probability),
                float(r.confidence),
                float(r.rainfall_trigger),
                float(r.seismic_trigger),
                float(self._RISK.index(r.risk_level)),
            ],
        )


class WildfireSceneAdapter(HazardRecordAdapter):
    """Active fire (or next-day ignition) from a thermal scene.

    Each record is one scene for ``WildfireDetector.predict_wildfire``:
    ``thermal_image`` (required: a brightness-temperature image in K, or a
    wildfire-firms-v1 ``(3, H, W)`` raster), ``pixel_size_km``, and optional
    ``weather_data`` / ``fire_data``.  A scene without a thermal image is
    refused: on weather alone the detector's decision is constant (never a
    fire), so grading it would be theatre.  Score and decision are the
    detector's own ``confidence`` and ``fire_detected``; ``assessment`` in
    the result names whether that is an active fire or a next-day forecast.
    """

    REQUIRED_KEYS = frozenset({"thermal_image"})
    ALLOWED_KEYS = frozenset(
        {
            "thermal_image",
            "pixel_size_km",
            "weather_data",
            "fire_data",
            "evacuation_zones",
            "containment_strategy",
        }
    )
    FEATURE_NAMES = (
        "confidence",
        "thermal_hotspots",
        "fire_perimeter_km2",
        "spread_rate_km_hr",
        "risk_level",
    )
    _RISK = ("low", "moderate", "high", "very_high", "extreme")

    def _parse(self, record: Mapping[str, Any]) -> Any:
        try:
            image = np.asarray(record["thermal_image"], dtype=np.float64)
        except (TypeError, ValueError) as e:
            raise InputModalityError("thermal_image must be a numeric raster") from e
        if image.ndim not in (2, 3):
            raise InputModalityError(
                f"thermal_image must be (H, W) or (C, H, W); got shape {image.shape}"
            )
        return {**record, "thermal_image": image}

    def _assess(self, unit: Any) -> Assessment:
        from omni_mercury_engine.detectors.geological.wildfire import WildfireDetector

        if self._detector is None:
            self._detector = WildfireDetector()
        try:
            r = self._detector.predict_wildfire(unit)
        except ValueError as e:
            raise _domain_refusal(e) from e
        return Assessment(
            score=float(r.confidence),
            flag=bool(r.fire_detected),
            features=(
                float(r.confidence),
                float(r.thermal_hotspots),
                nan_if_none(r.fire_perimeter_km2),
                nan_if_none(r.spread_rate_km_hr),
                float(self._RISK.index(r.risk_level)),
            ),
            result=r,
        )


class AvalancheProfileAdapter(HazardRecordAdapter):
    """Avalanche danger from a snow profile and 24 h of weather.

    Each record holds ``AvalancheDetector.predict_avalanche``'s inputs:
    ``layers`` (surface downward; each ``{thickness_m, density_kg_m3,
    temperature_c, grain_type}``), ``weak_layer_index``,
    ``measured_strength_pa``, ``ski_penetration_m``, ``new_snow_24h_cm``,
    ``wind_speed_10m_ms``, ``rain_mm_24h``, ``air_temperature_c`` and
    ``gradient_duration_days``.  The decision is the detector's own
    ``avalanche_likely`` (EAWS danger >= 3, considerable); the score is the
    EAWS level over the scale, (level - 1) / 4.
    """

    ALLOWED_KEYS = frozenset(
        {
            "layers",
            "weak_layer_index",
            "slope_angle_deg",
            "measured_strength_pa",
            "ski_penetration_m",
            "new_snow_24h_cm",
            "wind_speed_10m_ms",
            "rain_mm_24h",
            "air_temperature_c",
            "gradient_duration_days",
        }
    )
    FEATURE_NAMES = (
        "danger_level",
        "sk38",
        "effective_new_snow_24h_cm",
        "max_temperature_gradient_k_m",
        "confidence",
    )

    def _parse(self, record: Mapping[str, Any]) -> Any:
        from omni_mercury_engine.detectors.geological.avalanche_detector import SnowLayer

        kwargs = {k: v for k, v in record.items() if k != "layers"}
        if "layers" in record:
            layers = record["layers"]
            if not isinstance(layers, (list, tuple)):
                raise InputModalityError("layers must be a list of layer mappings")
            try:
                kwargs["layers"] = [
                    SnowLayer(**layer) if isinstance(layer, dict) else layer for layer in layers
                ]
            except (TypeError, ValueError) as e:
                raise InvalidObservationError(f"invalid snow layer: {e}") from e
            if not all(isinstance(layer, SnowLayer) for layer in kwargs["layers"]):
                raise InputModalityError("each layer must be a mapping of SnowLayer fields")
        return kwargs

    def _assess(self, unit: Any) -> Assessment:
        from omni_mercury_engine.detectors.geological.avalanche_detector import AvalancheDetector

        if self._detector is None:
            self._detector = AvalancheDetector()
        try:
            r = self._detector.predict_avalanche(**unit)
        except ValueError as e:
            raise _domain_refusal(e) from e
        return Assessment(
            score=(int(r.danger_level) - 1) / 4.0,
            flag=bool(r.avalanche_likely),
            features=(
                float(r.danger_level),
                float("nan") if r.sk38 is None else float(r.sk38.sk38),
                float(r.effective_new_snow_24h_cm),
                float(r.max_temperature_gradient_k_m),
                float(r.confidence),
            ),
            result=r,
        )


class RockfallAdapter(HazardRecordAdapter):
    """Rockfall hazard from the monitoring channels of one cliff section.

    Each record is ``RockfallDetector.predict_rockfall``'s own input: any of
    ``rock_temperature_c`` (+ ``precipitation_mm`` or ``surface_wet``),
    ``rain_mm``, ``displacement_mm`` + ``displacement_time``,
    ``microseismic_event_times`` + ``observation_window``, and
    ``sample_interval_hours``.  The detector defines no boolean; its hazard
    ladder is low < moderate < high < critical, so the score is the level
    over 3 and the decision is any level above low (a trigger fired).
    """

    ALLOWED_KEYS = frozenset(
        {
            "rock_temperature_c",
            "precipitation_mm",
            "surface_wet",
            "sample_interval_hours",
            "rain_mm",
            "displacement_mm",
            "displacement_time",
            "microseismic_event_times",
            "observation_window",
        }
    )
    FEATURE_NAMES = (
        "hazard_level",
        "confidence",
        "estimated_frequency_multiplier",
        "rain_mean_intensity_mm_h",
        "microseismic_rate_ratio",
    )
    _LADDER = ("low", "moderate", "high", "critical")

    def _assess(self, unit: Any) -> Assessment:
        from omni_mercury_engine.detectors.geological.rockfall_detector import RockfallDetector

        if self._detector is None:
            self._detector = RockfallDetector()
        try:
            r = self._detector.predict_rockfall(dict(unit))
        except ValueError as e:
            raise _domain_refusal(e) from e
        level = self._LADDER.index(r.hazard_level)
        return Assessment(
            score=ordinal_score(r.hazard_level, self._LADDER),
            flag=level >= 1,
            features=(
                float(level),
                float(r.confidence),
                float(r.estimated_frequency_multiplier),
                nan_if_none(r.rain_mean_intensity_mm_h),
                nan_if_none(r.microseismic_rate_ratio),
            ),
            result=r,
        )


class SubsidenceAdapter(HazardRecordAdapter):
    """Ground subsidence and sinkhole precursors from an InSAR scene.

    Each record is one scene for ``SubsidenceDetector.analyze``:
    ``displacements_mm`` (points x epochs, line-of-sight mm, NaN = missing
    epoch), ``epochs_years`` and optional ``coordinates_m`` / ``coherence``.
    Score and decision are the detector's own ``confidence`` and
    ``anomaly_detected``; the stricter ``sinkhole_precursor_detected`` and
    the severity class are features.
    """

    REQUIRED_KEYS = frozenset({"displacements_mm", "epochs_years"})
    ALLOWED_KEYS = frozenset({"displacements_mm", "epochs_years", "coordinates_m", "coherence"})
    FEATURE_NAMES = (
        "confidence",
        "severity",
        "max_subsidence_velocity_mm_yr",
        "n_accelerating_points",
        "sinkhole_precursor_detected",
    )
    _SEVERITY = ("stable", "slow", "moderate", "fast", "extreme")

    def _parse(self, record: Mapping[str, Any]) -> Any:
        parsed = {}
        for key, value in record.items():
            try:
                parsed[key] = np.asarray(value, dtype=np.float64)
            except (TypeError, ValueError) as e:
                raise InputModalityError(f"{key} must be numeric") from e
        for key in ("coordinates_m", "coherence"):
            if key in parsed and not np.all(np.isfinite(parsed[key])):
                # The detector drops NaN-coherence points and silently finds
                # no cluster from NaN coordinates: refuse instead.
                raise InvalidObservationError(f"{key} has non-finite values")
        return parsed

    def _assess(self, unit: Any) -> Assessment:
        from omni_mercury_engine.detectors.geological.subsidence_detector import (
            SubsidenceDetector,
        )

        if self._detector is None:
            self._detector = SubsidenceDetector()
        try:
            r = self._detector.analyze(**unit)
        except ValueError as e:
            raise _domain_refusal(e) from e
        return Assessment(
            score=float(r.confidence),
            flag=bool(r.anomaly_detected),
            features=(
                float(r.confidence),
                float(self._SEVERITY.index(r.severity)),
                float(r.max_subsidence_velocity_mm_yr),
                float(r.n_accelerating_points),
                float(r.sinkhole_precursor_detected),
            ),
            result=r,
        )


class EqTsunamiCascadeAdapter(HazardRecordAdapter):
    """The earthquake-to-tsunami cascade replayed over one event sequence.

    Each record is an evaluation: ``events`` (USGS GeoJSON features or flat
    ``{magnitude, depth_km, ...}`` dicts, in arrival order), optional
    ``offshore`` (applied to every event; True / False / None) and optional
    ``water_level_m`` + ``sampling_interval_s`` (a DART or coastal gauge
    series).  A fresh ``EqTsunamiCascadeDetector`` processes the events in
    order and, when the stage reaches TSUNAMI_WATCH, confirms against the
    water level (NOAA's DART algorithm).  A water level supplied without a
    watch-class event is refused: confirmation has no event to confirm.
    The score is the stage on quiet < evaluating < tsunami_watch <
    tsunami_threat over 3; the decision is any stage above quiet.
    """

    REQUIRED_KEYS = frozenset({"events"})
    ALLOWED_KEYS = frozenset({"events", "offshore", "water_level_m", "sampling_interval_s"})
    FEATURE_NAMES = ("stage_level", "magnitude", "depth_km", "max_residual_m")
    _STAGES = ("quiet", "evaluating", "tsunami_watch", "tsunami_threat")

    def _parse(self, record: Mapping[str, Any]) -> Any:
        events = record["events"]
        if not isinstance(events, (list, tuple)) or not events:
            raise InputModalityError("events must be a non-empty list of earthquake events")
        if not all(isinstance(e, dict) for e in events):
            raise InputModalityError("each event must be a mapping")
        if ("water_level_m" in record) != ("sampling_interval_s" in record):
            raise InvalidObservationError("water_level_m and sampling_interval_s come together")
        offshore = record.get("offshore")
        if offshore is not None and not isinstance(offshore, bool):
            raise InvalidObservationError("offshore must be True, False or None")
        return dict(record)

    def _assess(self, unit: Any) -> Assessment:
        from omni_mercury_engine.detectors.geological.eq_tsunami_cascade import (
            EqTsunamiCascadeDetector,
        )

        cascade = EqTsunamiCascadeDetector()
        try:
            for event in unit["events"]:
                state = cascade.process_event(event, offshore=unit.get("offshore"))
            if "water_level_m" in unit:
                if state.stage != "tsunami_watch":
                    raise InvalidObservationError(
                        "a water level was supplied but no event reached TSUNAMI_WATCH "
                        f"(stage {state.stage}); there is no watch to confirm"
                    )
                state = cascade.confirm_water_level(
                    np.asarray(unit["water_level_m"], dtype=np.float64),
                    float(unit["sampling_interval_s"]),
                    run_spectral_analysis=False,
                )
        except ValueError as e:
            raise _domain_refusal(e) from e
        screening = state.screening
        return Assessment(
            score=ordinal_score(state.stage, self._STAGES),
            flag=state.stage != "quiet",
            features=(
                float(self._STAGES.index(state.stage)),
                float("nan") if screening is None else float(screening.magnitude),
                float("nan") if screening is None else float(screening.depth_km),
                float("nan") if state.water_level is None else state.water_level.max_residual_m,
            ),
            result=state,
        )


class FireDebrisFlowAdapter(HazardRecordAdapter):
    """Post-wildfire debris-flow likelihood for a burned basin and a storm.

    Each record holds ``FireDebrisFlowCascadeDetector.assess``'s inputs:
    ``t_proportion``, ``dnbr_mean``, ``kf_factor`` (the Staley 2017 M1
    terrain, burn and soil terms), ``rain_mm`` (mm per step) and
    ``step_minutes``, plus ``burned_mh_km2`` and ``relief_m`` for the
    Gartner 2014 volume.  ``config`` may set ``primary_duration_min`` and
    ``cannon_region``.  Score and decision are the detector's own M1
    ``likelihood`` and ``threshold_exceeded`` (the USGS p = 0.5 rule).
    """

    REQUIRED_KEYS = frozenset({"t_proportion", "dnbr_mean", "kf_factor", "rain_mm", "step_minutes"})
    ALLOWED_KEYS = REQUIRED_KEYS | {"burned_mh_km2", "relief_m"}
    FEATURE_NAMES = (
        "likelihood",
        "threshold_intensity_mm_h",
        "cannon_exceeded",
        "volume_m3",
        "volume_class",
    )

    def _parse(self, record: Mapping[str, Any]) -> Any:
        try:
            rain = np.asarray(record["rain_mm"], dtype=np.float64)
        except (TypeError, ValueError) as e:
            raise InputModalityError("rain_mm must be a numeric series") from e
        return {**record, "rain_mm": rain}

    def _assess(self, unit: Any) -> Assessment:
        from omni_mercury_engine.detectors.geological.fire_debris_flow_cascade import (
            FireDebrisFlowCascadeDetector,
        )

        if self._detector is None:
            kwargs = {
                k: self._config[k]
                for k in ("primary_duration_min", "cannon_region")
                if k in self._config
            }
            self._detector = FireDebrisFlowCascadeDetector(**kwargs)
        try:
            r = self._detector.assess(**unit)
        except ValueError as e:
            raise _domain_refusal(e) from e
        return Assessment(
            score=float(r.likelihood),
            flag=bool(r.threshold_exceeded),
            features=(
                float(r.likelihood),
                float(r.threshold_intensity_mm_h),
                float("nan") if r.cannon_exceeded is None else float(r.cannon_exceeded),
                nan_if_none(r.volume_m3),
                nan_if_none(r.volume_class),
            ),
            result=r,
        )


class SurgeFloodCascadeAdapter(HazardRecordAdapter):
    """The hurricane-to-surge-to-compound-flood cascade from one evidence set.

    Each record carries the cascade's evidence: ``hurricane`` (the
    ``HurricaneDetector.predict_hurricane`` input -- the hurricane gate is
    fed that detector's real result, never a hand-built one),
    ``coops_observed`` + ``coops_predicted`` (NOAA CO-OPS water-level
    payloads, metres) with ``station_id``, and ``river_gauge`` (an NWPS gauge
    payload).  A fresh ``SurgeFloodCascade`` takes the evidence and awards a
    stage; the score is the stage (quiet 0 ... compound flood 3) over 3 and
    the decision is any stage above quiet.
    """

    ALLOWED_KEYS = frozenset(
        {"hurricane", "coops_observed", "coops_predicted", "station_id", "river_gauge"}
    )
    FEATURE_NAMES = (
        "stage_level",
        "hurricane_qualifies",
        "surge_qualifies",
        "river_qualifies",
        "max_residual_m",
    )

    _EVIDENCE = ("hurricane", "coops_observed", "river_gauge")

    def _parse(self, record: Mapping[str, Any]) -> Any:
        if ("coops_observed" in record) != ("coops_predicted" in record):
            raise InvalidObservationError("coops_observed and coops_predicted come together")
        if not any(key in record for key in self._EVIDENCE):
            # A station id alone graded QUIET: an all-clear on no evidence.
            raise InvalidObservationError(
                "a surge cascade record needs evidence: hurricane, coops_observed + "
                "coops_predicted, or river_gauge"
            )
        return dict(record)

    def _assess(self, unit: Any) -> Assessment:
        from omni_mercury_engine.detectors.geological.hurricane_detector import HurricaneDetector
        from omni_mercury_engine.detectors.meteorological.surge_flood_cascade import (
            SurgeFloodCascade,
        )

        cascade = SurgeFloodCascade()
        try:
            if "hurricane" in unit:
                if self._detector is None:
                    self._detector = HurricaneDetector()
                cascade.update_hurricane_evidence(
                    self._detector.predict_hurricane(unit["hurricane"])
                )
            if "coops_observed" in unit:
                cascade.update_surge_evidence(
                    unit["coops_observed"],
                    unit["coops_predicted"],
                    str(unit.get("station_id", "")),
                )
            if "river_gauge" in unit:
                cascade.update_river_evidence(unit["river_gauge"])
            a = cascade.evaluate()
        except ValueError as e:
            raise _domain_refusal(e) from e
        level = int(a.stage.value)
        return Assessment(
            score=level / 3.0,
            flag=level >= 1,
            features=(
                float(level),
                float(a.hurricane_qualifies),
                float(a.surge_qualifies),
                float(a.river_qualifies),
                float("nan") if a.surge is None else float(a.surge.max_residual_m),
            ),
            result=a,
        )


class MarineEcosystemAdapter(HazardRowAdapter):
    """Marine ecosystem threat from coral, chemistry and temperature indicators.

    Columns: ``sst_anomaly_c`` and ``degree_heating_weeks`` (coral thermal
    stress; measured together or NaN together), ``ph`` and
    ``temperature_anomaly_c`` (each NaN when unmeasured).  A row with no
    measured indicator is refused by the detector, never graded healthy.
    ``MarineBiodiversityDetector.predict_biodiversity_threat`` assesses the
    measured indicators; its decision is ``ecosystem_threatened`` (any
    assessed indicator fires) and the score is its health ladder healthy <
    stressed < critical over 2, so the decision is exactly score >= 1/2.
    """

    COLUMNS = ("sst_anomaly_c", "degree_heating_weeks", "ph", "temperature_anomaly_c")
    OPTIONAL_COLUMNS = frozenset(COLUMNS)
    FEATURE_NAMES = (
        "confidence",
        "coral_bleaching_detected",
        "ocean_acidification",
        "marine_heatwave",
    )
    _HEALTH = ("healthy", "stressed", "critical")

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize with ``BaseDetector`` config; the detector loads lazily."""
        super().__init__(config)
        self._detector: Any = None

    def _analyze_row(self, row: np.ndarray[Any, Any]) -> tuple[float, bool, list[float]]:
        from omni_mercury_engine.detectors.marine.biodiversity_detector import (
            MarineBiodiversityDetector,
        )

        m = self._measured(row)
        record: dict[str, Any] = {}
        coral = {k: m[k] for k in ("sst_anomaly_c", "degree_heating_weeks") if k in m}
        if coral:
            # Half a coral pair reaches the detector, which refuses it.
            record["coral_data"] = coral
        if "ph" in m:
            record["chemistry_data"] = {"ph": m["ph"]}
        if "temperature_anomaly_c" in m:
            record["temperature_data"] = {"anomaly_c": m["temperature_anomaly_c"]}
        if self._detector is None:
            self._detector = MarineBiodiversityDetector()
        try:
            r = self._detector.predict_biodiversity_threat(record)
        except ValueError as e:
            raise _domain_refusal(e) from e
        return (
            ordinal_score(r.health_status, self._HEALTH),
            bool(r.ecosystem_threatened),
            [
                float(r.confidence),
                float(r.coral_bleaching_detected),
                float(r.ocean_acidification),
                float(r.marine_heatwave),
            ],
        )
