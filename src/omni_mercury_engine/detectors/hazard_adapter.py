# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""BaseDetector adapters for the hazard detectors.

The hazard detectors (tornado, hurricane, flood, financial crisis, solar
storm, solar flare, GIC) expose domain APIs — a dict of named measurements in,
a typed result out — and therefore registered in ``DETECTOR_MANIFEST`` without
the engine contract (``fit`` / ``detect`` / ``extract_features`` / ``predict``),
so the registry's feature extraction silently skipped them. Each adapter here
is a :class:`~omni_mercury_engine.core.base.BaseDetector` whose input is a
batch of rows over *declared, named, physical columns* (``COLUMNS``, with
units in each adapter's docstring), mapped one-to-one onto the wrapped
detector's own API. Nothing is fabricated on the way through:

- a row is scored by the wrapped detector's own entry point, and the score is
  the detector's own confidence or its cited scale (NOAA G/5, NOAA R/5, the
  Pulkkinen/Meta-R-321 dB/dt tiers/3) — never an invented rescaling;
- ``is_anomaly`` is the detector's own decision (``tornado_likely``,
  ``crisis_imminent``, G >= 1, R >= 1, dB/dt >= 90 nT/min);
- a batch of the wrong width is refused, and a non-finite value raises just
  as it does at the wrapped detector's front door;
- ``fit`` validates the declared schema against reference data and performs
  no training: the wrapped detectors are physics- or checkpoint-backed, not
  fitted to the caller's batch.

Not adapted, deliberately: ``WildfireDetector`` measures thermal rasters —
on scalar weather rows alone its prediction is constant, so a row adapter
would emit a constant-zero detector; ``EMPDetector`` scores a sampled
waveform whose per-row attribution would be invented. They stay on their
domain APIs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar

import numpy as np

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.core.exceptions import InputModalityError, InvalidObservationError
from omni_mercury_engine.detectors.provenance import WrappedDecisionSource

__all__ = [
    "Assessment",
    "FinancialStressAdapter",
    "FloodPrecipitationAdapter",
    "GICMinuteSeriesAdapter",
    "HazardRecordAdapter",
    "HazardRowAdapter",
    "HazardTextAdapter",
    "HazardUnitAdapter",
    "HazardWindowAdapter",
    "HurricanePressureAdapter",
    "SolarWindGeomagAdapter",
    "TornadoConditionsAdapter",
    "XrayFlareAdapter",
    "nan_if_none",
    "ordinal_score",
    "utc_datetime",
]


class HazardRowAdapter(BaseDetector):
    """Engine contract over a named-column batch, one hazard analysis per row.

    Subclasses declare ``COLUMNS`` (the batch's column names, in order) and
    ``FEATURE_NAMES`` (the measured outputs served to fusion), and implement
    :meth:`_analyze_row`.
    """

    #: Adapters define NaN as "this column/field is unmeasured" by contract
    #: (OPTIONAL_COLUMNS / _measured): the shared base guard defers to that
    #: contract and only enforces the never-serve-a-NaN-score post-check.
    handles_unmeasured = True

    COLUMNS: ClassVar[tuple[str, ...]] = ()
    FEATURE_NAMES: ClassVar[tuple[str, ...]] = ()
    #: Columns a row may leave unmeasured, as NaN: the wrapped detector is
    #: then called without that measurement (never with a substitute).
    #: Infinities are refused in every column.
    OPTIONAL_COLUMNS: ClassVar[frozenset[str]] = frozenset()

    def _analyze_row(self, row: np.ndarray[Any, Any]) -> tuple[float, bool, list[float]]:
        """One row's (score in [0, 1], detector decision, measured features)."""
        raise NotImplementedError

    def _measured(self, row: np.ndarray[Any, Any]) -> dict[str, float]:
        """``{column: value}`` for the row's measured (non-NaN) columns."""
        return {c: float(v) for c, v in zip(self.COLUMNS, row, strict=True) if not np.isnan(v)}

    def _validated(self, data: Any) -> np.ndarray[Any, Any]:
        """``data`` as a finite 2-D batch of ``len(COLUMNS)`` columns."""
        batch = np.asarray(data, dtype=np.float64)
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)
        if batch.ndim != 2 or batch.shape[1] != len(self.COLUMNS):
            raise InvalidObservationError(
                f"{self._name} takes rows over the declared columns "
                f"{self.COLUMNS}; got shape {batch.shape}"
            )
        optional = np.array([c in self.OPTIONAL_COLUMNS for c in self.COLUMNS], dtype=bool)
        if np.isinf(batch).any() or np.isnan(batch[:, ~optional]).any():
            raise InvalidObservationError(
                f"{self._name}: non-finite values are not measurements; refusing to grade them"
                + (
                    f" (NaN marks an unmeasured value only in {sorted(self.OPTIONAL_COLUMNS)})"
                    if self.OPTIONAL_COLUMNS
                    else ""
                )
            )
        return batch

    def _analyze(
        self, data: Any
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        batch = self._validated(data)
        scores = np.empty(batch.shape[0])
        flags = np.empty(batch.shape[0], dtype=bool)
        features = np.empty((batch.shape[0], len(self.FEATURE_NAMES)))
        for i, row in enumerate(batch):
            score, flag, row_features = self._analyze_row(row)
            scores[i], flags[i], features[i] = score, flag, row_features
        return scores, flags, features

    def fit(self, data: Any) -> HazardRowAdapter:
        """Validate the declared schema on reference data; no training occurs."""
        self._validated(data)
        self._is_fitted = True
        return self

    def extract_features(self, data: Any) -> np.ndarray[Any, Any]:
        """Per-row measured outputs, columns named by ``FEATURE_NAMES``."""
        return self._analyze(data)[2]

    def predict(self, data: Any) -> dict[str, Any]:
        """Per-row scores under the registry's prediction contract."""
        scores, flags, _ = self._analyze(data)
        return {"anomaly_scores": scores, "flags": flags}

    def detect(self, data: Any) -> dict[str, Any]:
        """Batch decision: the worst row decides, each row's score is served."""
        scores, flags, _ = self._analyze(data)
        return {
            "anomaly_score": float(scores.max()),
            "is_anomaly": bool(flags.any()),
            "scores": scores,
            "flags": flags,
            "metadata": {"columns": self.COLUMNS, "rows": int(scores.size)},
        }


@dataclass(frozen=True)
class Assessment:
    """One assessed unit: a window, a record, an event sequence or a payload.

    Attributes:
        score: In [0, 1]: the wrapped detector's own confidence or
            probability, or its cited scale's level over the scale's top.
        flag: The wrapped detector's own decision (or the scale's first
            actionable level, named in each adapter's docstring).
        features: Measured outputs, in the adapter's ``FEATURE_NAMES`` order;
            NaN marks a quantity the detector reported as unmeasured.
        result: The wrapped detector's full result object, unchanged.
    """

    score: float
    flag: bool
    features: tuple[float, ...]
    result: Any


def ordinal_score(level: str, ladder: tuple[str, ...]) -> float:
    """``level``'s position on ``ladder`` (lowest first) over its top, in [0, 1]."""
    if level not in ladder:
        raise InvalidObservationError(f"level {level!r} is not on the ladder {ladder}")
    return ladder.index(level) / (len(ladder) - 1)


def nan_if_none(value: Any) -> float:
    """``value`` as a float; ``None`` (unmeasured) becomes NaN, never zero."""
    return float("nan") if value is None else float(value)


def utc_datetime(value: Any, name: str) -> datetime:
    """A timezone-aware UTC datetime from a datetime, ISO-8601 string or epoch seconds.

    Raises:
        InvalidObservationError: For naive datetimes, unparseable strings,
            non-finite numbers or other types.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise InvalidObservationError(f"{name}: naive datetime {value!r}; timezone required")
        return value.astimezone(UTC)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as e:
            raise InvalidObservationError(f"{name}: unparseable time {value!r}") from e
        return utc_datetime(parsed, name)
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        seconds = float(value)
        if not np.isfinite(seconds):
            raise InvalidObservationError(f"{name}: non-finite epoch seconds {value!r}")
        return datetime.fromtimestamp(seconds, tz=UTC)
    raise InvalidObservationError(
        f"{name}: expected a datetime, ISO-8601 string or epoch seconds, got {type(value).__name__}"
    )


class HazardUnitAdapter(BaseDetector):
    """Engine and registry contract over assessed units of a domain detector.

    A payload is split into units (:meth:`_units`, which also validates
    them) and every unit is assessed by the wrapped detector's own entry
    point (:meth:`_assess`).  Nothing is trained: ``fit`` validates the
    payload schema.  ``scores`` / ``flags`` carry one value per unit, so a
    window-level verdict is served once, not repeated over its samples.
    """

    #: Adapters define NaN as "this column/field is unmeasured" by contract
    #: (OPTIONAL_COLUMNS / _measured): the shared base guard defers to that
    #: contract and only enforces the never-serve-a-NaN-score post-check.
    handles_unmeasured = True

    FEATURE_NAMES: ClassVar[tuple[str, ...]] = ()

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize with ``BaseDetector`` config; the detector loads lazily."""
        super().__init__(config)
        self._detector: Any = None

    def _units(self, data: Any) -> list[Any]:
        """Validated units of ``data``."""
        raise NotImplementedError

    def _assess(self, unit: Any) -> Assessment:
        """The wrapped detector's assessment of one unit."""
        raise NotImplementedError

    def _assessments(self, data: Any) -> list[Assessment]:
        assessments = [self._assess(unit) for unit in self._units(data)]
        for a in assessments:
            if len(a.features) != len(self.FEATURE_NAMES) or not 0.0 <= a.score <= 1.0:
                raise RuntimeError(f"{self._name}: malformed assessment {a!r}")
        return assessments

    def fit(self, data: Any) -> HazardUnitAdapter:
        """Validate the payload schema on reference data; no training occurs."""
        self._units(data)
        self._is_fitted = True
        return self

    def extract_features(self, data: Any) -> np.ndarray[Any, Any]:
        """One row of measured outputs per unit, columns named by ``FEATURE_NAMES``."""
        return np.array([a.features for a in self._assessments(data)], dtype=np.float64)

    def predict(self, data: Any) -> dict[str, Any]:
        """Per-unit scores under the registry's prediction contract."""
        assessments = self._assessments(data)
        return {
            "anomaly_scores": np.array([a.score for a in assessments]),
            "flags": np.array([a.flag for a in assessments], dtype=bool),
        }

    def detect(self, data: Any) -> dict[str, Any]:
        """Batch decision: the worst unit decides; every unit's result is served."""
        assessments = self._assessments(data)
        scores = np.array([a.score for a in assessments])
        flags = np.array([a.flag for a in assessments], dtype=bool)
        return {
            "anomaly_score": float(scores.max()),
            "is_anomaly": bool(flags.any()),
            "scores": scores,
            "flags": flags,
            "metadata": {
                "units": len(assessments),
                "feature_names": self.FEATURE_NAMES,
                "results": [a.result for a in assessments],
            },
        }


class HazardWindowAdapter(HazardUnitAdapter):
    """One time-ordered window per payload: rows over declared ``COLUMNS``.

    A 2-D ``[n_samples, len(COLUMNS)]`` array is one window and one
    assessment.  ``OPTIONAL_COLUMNS`` may hold NaN (a gap or an unmeasured
    channel); every other value must be finite.
    """

    COLUMNS: ClassVar[tuple[str, ...]] = ()
    OPTIONAL_COLUMNS: ClassVar[frozenset[str]] = frozenset()
    MIN_SAMPLES: ClassVar[int] = 2

    def _units(self, data: Any) -> list[Any]:
        try:
            window = np.asarray(data, dtype=np.float64)
        except (TypeError, ValueError) as e:
            raise InputModalityError(
                f"{self._name} expects a numeric window over {self.COLUMNS}"
            ) from e
        if window.ndim != 2 or window.shape[1] != len(self.COLUMNS):
            raise InputModalityError(
                f"{self._name} expects one window [n_samples, {len(self.COLUMNS)}] over "
                f"the columns {self.COLUMNS}; got shape {window.shape}"
            )
        if window.shape[0] < self.MIN_SAMPLES:
            raise InvalidObservationError(
                f"{self._name} needs at least {self.MIN_SAMPLES} samples; got {window.shape[0]}"
            )
        optional = np.array([c in self.OPTIONAL_COLUMNS for c in self.COLUMNS], dtype=bool)
        if np.isinf(window).any() or np.isnan(window[:, ~optional]).any():
            raise InvalidObservationError(
                f"{self._name}: non-finite values outside the gap-capable columns "
                f"{sorted(self.OPTIONAL_COLUMNS)}; refusing to grade them"
            )
        return [window]

    def column(self, window: np.ndarray[Any, Any], name: str) -> np.ndarray[Any, Any]:
        """One named column of a validated window."""
        return np.asarray(window[:, self.COLUMNS.index(name)], dtype=np.float64)


class HazardRecordAdapter(HazardUnitAdapter):
    """Records: one mapping, or a sequence of mappings, each assessed alone.

    Subclasses validate a record in :meth:`_parse` (returning what
    :meth:`_assess` consumes) so ``fit`` refuses malformed records without
    running the detector.
    """

    #: Keys every record must carry.
    REQUIRED_KEYS: ClassVar[frozenset[str]] = frozenset()
    #: Keys a record may carry; ``None`` accepts any.  Declared where the
    #: wrapped detector ignores unknown keys, so a misspelt measurement is
    #: refused instead of silently going unassessed.
    ALLOWED_KEYS: ClassVar[frozenset[str] | None] = None

    def _parse(self, record: Mapping[str, Any]) -> Any:
        """Validate one record; default: the record itself."""
        return record

    def _units(self, data: Any) -> list[Any]:
        if isinstance(data, Mapping):
            records: list[Any] = [data]
        elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)) and data:
            records = list(data)
        else:
            raise InputModalityError(
                f"{self._name} expects a record (mapping) or a non-empty sequence of records; "
                f"got {type(data).__name__}"
            )
        parsed = []
        for i, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise InputModalityError(
                    f"{self._name}: record {i} is a {type(record).__name__}, not a mapping"
                )
            missing = sorted(self.REQUIRED_KEYS - set(record))
            if missing:
                raise InvalidObservationError(f"{self._name}: record {i} is missing {missing}")
            if self.ALLOWED_KEYS is not None:
                unknown = sorted(set(record) - self.ALLOWED_KEYS)
                if unknown:
                    raise InvalidObservationError(
                        f"{self._name}: record {i} has unknown keys {unknown}; "
                        f"accepted: {sorted(self.ALLOWED_KEYS)}"
                    )
            parsed.append(self._parse(record))
        return parsed


class HazardTextAdapter(HazardUnitAdapter):
    """Text payloads: one string, or a sequence of strings, each assessed alone."""

    #: Longest payload graded, in characters; longer ones are refused, never
    #: truncated (a truncated payload would be graded on text it does not hold).
    MAX_CHARS: ClassVar[int] = 65536

    def _units(self, data: Any) -> list[Any]:
        if isinstance(data, str):
            payloads = [data]
        elif isinstance(data, Sequence) and not isinstance(data, bytes) and data:
            payloads = list(data)
        else:
            raise InputModalityError(
                f"{self._name} expects text (str) or a non-empty sequence of strings; "
                f"got {type(data).__name__}"
            )
        for i, payload in enumerate(payloads):
            if not isinstance(payload, str):
                raise InputModalityError(
                    f"{self._name}: payload {i} is a {type(payload).__name__}, not text"
                )
            if len(payload) > self.MAX_CHARS:
                raise InvalidObservationError(
                    f"{self._name}: payload {i} has {len(payload)} characters, above the "
                    f"{self.MAX_CHARS}-character analysis limit; split it"
                )
        return payloads


class TornadoConditionsAdapter(HazardRowAdapter):
    """Tornado potential from sounding-derived convective parameters.

    Columns: ``cape_j_kg`` (J/kg), ``cin_j_kg`` (J/kg, <= 0), ``srh_m2_s2``
    (m²/s²), ``bulk_shear_kt`` (kt), ``lcl_m`` (m). Score and decision are
    ``TornadoDetector.predict_tornado``'s own confidence and
    ``tornado_likely`` on the atmospheric path.
    """

    decision_source = WrappedDecisionSource(
        "omni_mercury_engine.detectors.geological.tornado_detector:TornadoDetector"
    )

    COLUMNS = ("cape_j_kg", "cin_j_kg", "srh_m2_s2", "bulk_shear_kt", "lcl_m")
    FEATURE_NAMES = ("confidence", "tornado_likely", "cape_value", "helicity_value")

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize with ``BaseDetector`` config; the detector loads lazily."""
        super().__init__(config)
        self._detector: Any = None

    def _analyze_row(self, row: np.ndarray[Any, Any]) -> tuple[float, bool, list[float]]:
        if self._detector is None:
            from omni_mercury_engine.detectors.geological.tornado_detector import (
                TornadoDetector,
            )

            self._detector = TornadoDetector()
        result = self._detector.predict_tornado(
            {"atmospheric_data": dict(zip(self.COLUMNS, (float(v) for v in row)))}
        )
        return (
            float(result.confidence),
            bool(result.tornado_likely),
            [
                float(result.confidence),
                float(result.tornado_likely),
                float(result.cape_value),
                float(result.helicity_value),
            ],
        )


class HurricanePressureAdapter(HazardRowAdapter):
    """Cyclone intensity from central pressure and sea-surface temperature.

    Columns: ``central_pressure_mb`` (mb), ``environmental_pressure_mb`` (mb),
    ``sst_celsius`` (degC). Score and decision are
    ``HurricaneDetector.predict_hurricane``'s own confidence and
    ``cyclone_detected``; the wind feature is its Atkinson & Holliday (1977)
    pressure-wind value.
    """

    decision_source = WrappedDecisionSource(
        "omni_mercury_engine.detectors.geological.hurricane_detector:HurricaneDetector"
    )

    COLUMNS = ("central_pressure_mb", "environmental_pressure_mb", "sst_celsius")
    FEATURE_NAMES = ("confidence", "cyclone_detected", "max_wind_speed_kt", "min_pressure_mb")

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize with ``BaseDetector`` config; the detector loads lazily."""
        super().__init__(config)
        self._detector: Any = None

    def _analyze_row(self, row: np.ndarray[Any, Any]) -> tuple[float, bool, list[float]]:
        if self._detector is None:
            from omni_mercury_engine.detectors.geological.hurricane_detector import (
                HurricaneDetector,
            )

            self._detector = HurricaneDetector()
        central, environmental, sst = (float(v) for v in row)
        result = self._detector.predict_hurricane(
            {
                "pressure_data": {
                    "central_pressure_mb": central,
                    "environmental_pressure_mb": environmental,
                },
                "sst_data": {"sst_celsius": sst},
            }
        )
        return (
            float(result.confidence),
            bool(result.cyclone_detected),
            [
                float(result.confidence),
                float(result.cyclone_detected),
                float(result.max_wind_speed_kt),
                float(result.min_pressure_mb),
            ],
        )


class FloodPrecipitationAdapter(HazardRowAdapter):
    """Flood likelihood from measured and forecast precipitation depths.

    Columns: ``precipitation_1h_inches``, ``precipitation_6h_inches``,
    ``precipitation_24h_inches``, ``forecast_24h_inches`` (inches). Score and
    decision are ``FloodDetector.predict_flood``'s own confidence and
    ``flood_likely`` on the precipitation path.
    """

    decision_source = WrappedDecisionSource(
        "omni_mercury_engine.detectors.geological.flood_detector:FloodDetector"
    )

    COLUMNS = (
        "precipitation_1h_inches",
        "precipitation_6h_inches",
        "precipitation_24h_inches",
        "forecast_24h_inches",
    )
    FEATURE_NAMES = ("confidence", "flood_likely", "precipitation_24h_inches")

    #: The wrapped FloodDetector is physics-backed and lazily built; it learns
    #: nothing from the caller's batch, so the adapter's checkpoint carries
    #: only the trivial fitted flag and base attributes.
    _UNSAVED_ATTRS = BaseDetector._UNSAVED_ATTRS | {"_detector"}

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize with ``BaseDetector`` config; the detector loads lazily."""
        super().__init__(config)
        self._detector: Any = None

    def _analyze_row(self, row: np.ndarray[Any, Any]) -> tuple[float, bool, list[float]]:
        if self._detector is None:
            from omni_mercury_engine.detectors.geological.flood_detector import FloodDetector

            self._detector = FloodDetector()
        result = self._detector.predict_flood(
            {"precip_data": dict(zip(self.COLUMNS, (float(v) for v in row)))}
        )
        return (
            float(result.confidence),
            bool(result.flood_likely),
            [
                float(result.confidence),
                float(result.flood_likely),
                float(result.precipitation_24h_inches),
            ],
        )


class FinancialStressAdapter(HazardRowAdapter):
    """Financial crisis risk from measured market and banking metrics.

    Columns: ``price_change_pct`` (%), ``vix`` (index points),
    ``volume_vs_average`` (ratio), ``cds_spread_bps`` (bps),
    ``default_rate_pct`` (%), ``liquidity_ratio`` (ratio) — the wrapped
    detector's own input names. Score and decision are
    ``FinancialCrisisDetector.predict_financial_crisis``'s own confidence and
    ``crisis_imminent``. The untrained fraud network stays unconsulted, as in
    the wrapped detector.
    """

    decision_source = WrappedDecisionSource(
        "omni_mercury_engine.detectors.economic.financial_crisis_detector:FinancialCrisisDetector"
    )

    COLUMNS = (
        "price_change_pct",
        "vix",
        "volume_vs_average",
        "cds_spread_bps",
        "default_rate_pct",
        "liquidity_ratio",
    )
    FEATURE_NAMES = ("confidence", "crisis_imminent", "market_crash", "banking_stress")

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize with ``BaseDetector`` config; the detector loads lazily."""
        super().__init__(config)
        self._detector: Any = None

    def _analyze_row(self, row: np.ndarray[Any, Any]) -> tuple[float, bool, list[float]]:
        if self._detector is None:
            from omni_mercury_engine.detectors.economic.financial_crisis_detector import (
                FinancialCrisisDetector,
            )

            self._detector = FinancialCrisisDetector()
        values = [float(v) for v in row]
        result = self._detector.predict_financial_crisis(
            {
                "market_data": dict(zip(self.COLUMNS[:3], values[:3])),
                "banking_data": dict(zip(self.COLUMNS[3:], values[3:])),
            }
        )
        return (
            float(result.confidence),
            bool(result.crisis_imminent),
            [
                float(result.confidence),
                float(result.crisis_imminent),
                float(result.market_crash_detected),
                float(result.banking_stress),
            ],
        )


class SolarWindGeomagAdapter(HazardRowAdapter):
    """Geomagnetic storm level from solar-wind speed and IMF Bz.

    Columns: ``solar_wind_speed_km_s`` (km/s), ``bz_imf_nt`` (nT). The score
    is NOAA's G scale over 5 and the decision is G >= 1, where G is the
    detector's own ``geomagnetic_storm_level`` -- including its trained
    operating point, which raises a sub-G1 Kp estimate to G1 when the storm
    probability clears it.  Re-deriving G from ``kp_index`` alone (as before)
    disagreed with the detector: (600 km/s, Bz -10 nT) is "minor" to the
    detector but was served as G0, not anomalous.
    """

    decision_source = WrappedDecisionSource(
        "omni_mercury_engine.space.solar_storm_detector:SolarStormDetector"
    )

    COLUMNS = ("solar_wind_speed_km_s", "bz_imf_nt")
    FEATURE_NAMES = ("kp_index", "g_level", "confidence")

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize with ``BaseDetector`` config; the detector loads lazily."""
        super().__init__(config)
        self._detector: Any = None

    def _analyze_row(self, row: np.ndarray[Any, Any]) -> tuple[float, bool, list[float]]:
        if self._detector is None:
            from omni_mercury_engine.space.solar_storm_detector import SolarStormDetector

            self._detector = SolarStormDetector()
        from omni_mercury_engine.evaluation.hazard_metrics import G_SCALE_NAMES

        result = self._detector.predict_solar_storm(
            {"magnetosphere_data": dict(zip(self.COLUMNS, (float(v) for v in row)))}
        )
        if result.kp_index is None or result.geomagnetic_storm_level not in G_SCALE_NAMES:
            raise InvalidObservationError(
                "solar storm prediction returned no geomagnetic level for a measured row"
            )
        g_level = G_SCALE_NAMES.index(result.geomagnetic_storm_level)
        return (
            g_level / 5.0,
            g_level >= 1,
            [float(result.kp_index), float(g_level), float(result.confidence)],
        )


class XrayFlareAdapter(HazardRowAdapter):
    """Solar flare class from the GOES long-band X-ray flux.

    Column: ``flux_long_wm2`` (W/m², the 0.1-0.8 nm band NOAA defines the
    class on). The score is NOAA's own R scale over 5
    (``evaluation.hazard_metrics.noaa_r_scale``); the decision is R >= 1
    (an M1 flare, NOAA's Minor radio blackout).
    """

    decision_source = WrappedDecisionSource(
        "omni_mercury_engine.space.solar_storm_detector:SolarFlareDetector"
    )

    COLUMNS = ("flux_long_wm2",)
    FEATURE_NAMES = ("flare_intensity", "r_level", "flare_detected")

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize with ``BaseDetector`` config; the detector loads lazily."""
        super().__init__(config)
        self._detector: Any = None

    def _analyze_row(self, row: np.ndarray[Any, Any]) -> tuple[float, bool, list[float]]:
        if self._detector is None:
            from omni_mercury_engine.space.solar_storm_detector import SolarFlareDetector

            self._detector = SolarFlareDetector()
        from omni_mercury_engine.evaluation.hazard_metrics import noaa_r_scale

        flux = float(row[0])
        result = self._detector.detect_solar_flare({"flux_long_wm2": flux})
        r_level = noaa_r_scale(flux)
        return (
            r_level / 5.0,
            r_level >= 1,
            [float(result["flare_intensity"]), float(r_level), float(result["flare_detected"])],
        )


class GICMinuteSeriesAdapter(HazardRowAdapter):
    """GIC grid risk from a magnetometer minute series.

    The batch is one time series: each row is one consecutive 1-minute sample
    (``config={"cadence_s": ...}`` declares a different fixed cadence) over
    columns ``bx_nt``, ``by_nt`` (nT). The whole window is assessed by
    ``GICDetector.assess``; the score is the cited dB/dt tier over 3 (low 0,
    moderate 1/3, high 2/3, severe 1 — Pulkkinen et al. 2013; Oak Ridge
    Meta-R-321; Bolduc 2002) and the decision is >= moderate (90 nT/min).
    The assessment is window-level, so its score and measured features repeat
    on every row of the window.
    """

    decision_source = WrappedDecisionSource("omni_mercury_engine.space.gic_detector:GICDetector")

    COLUMNS = ("bx_nt", "by_nt")
    FEATURE_NAMES = ("peak_dbdt_nt_per_min", "sustained_dbdt_nt_per_min", "risk_tier")

    _TIERS = ("low", "moderate", "high", "severe")

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize; ``config["cadence_s"]`` declares the row cadence (60 s)."""
        super().__init__(config)
        self._detector: Any = None
        cadence = self._config.get("cadence_s", 60.0)
        if (
            isinstance(cadence, bool)
            or not isinstance(cadence, (int, float))
            or not np.isfinite(cadence)
            or cadence <= 0
        ):
            raise ValueError(f"cadence_s must be a positive number, got {cadence!r}")
        self._cadence_s = float(cadence)

    def _analyze(
        self, data: Any
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        batch = self._validated(data)
        if self._detector is None:
            from omni_mercury_engine.space.gic_detector import GICDetector

            self._detector = GICDetector()
        start = datetime(2000, 1, 1, tzinfo=UTC)
        times = [start + timedelta(seconds=self._cadence_s * i) for i in range(batch.shape[0])]
        assessment = self._detector.assess(times, batch[:, 0], by_nt=batch[:, 1])
        tier = float(self._TIERS.index(assessment.risk_level))
        score, flag = tier / 3.0, tier >= 1.0
        n = batch.shape[0]
        # The 10-minute rolling metric is unmeasured on a shorter window; the
        # detector reports None there, served as NaN, never as a quiet zero.
        sustained = assessment.sustained_dbdt_nt_per_min
        features = np.tile(
            [
                float(assessment.peak_dbdt_nt_per_min),
                float("nan") if sustained is None else float(sustained),
                tier,
            ],
            (n, 1),
        )
        return np.full(n, score), np.full(n, flag, dtype=bool), features

    def _analyze_row(self, row: np.ndarray[Any, Any]) -> tuple[float, bool, list[float]]:
        raise NotImplementedError("GIC risk is defined on a minute series, not one row")
