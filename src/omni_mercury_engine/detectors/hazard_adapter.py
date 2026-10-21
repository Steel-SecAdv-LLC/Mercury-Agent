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

from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar

import numpy as np

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.core.exceptions import InvalidObservationError

__all__ = [
    "FinancialStressAdapter",
    "FloodPrecipitationAdapter",
    "GICMinuteSeriesAdapter",
    "HazardRowAdapter",
    "HurricanePressureAdapter",
    "SolarWindGeomagAdapter",
    "TornadoConditionsAdapter",
    "XrayFlareAdapter",
]


class HazardRowAdapter(BaseDetector):
    """Engine contract over a named-column batch, one hazard analysis per row.

    Subclasses declare ``COLUMNS`` (the batch's column names, in order) and
    ``FEATURE_NAMES`` (the measured outputs served to fusion), and implement
    :meth:`_analyze_row`.
    """

    COLUMNS: ClassVar[tuple[str, ...]] = ()
    FEATURE_NAMES: ClassVar[tuple[str, ...]] = ()

    def _analyze_row(self, row: np.ndarray[Any, Any]) -> tuple[float, bool, list[float]]:
        """One row's (score in [0, 1], detector decision, measured features)."""
        raise NotImplementedError

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
        if not np.all(np.isfinite(batch)):
            raise InvalidObservationError(
                f"{self._name}: non-finite values are not measurements; " "refusing to grade them"
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
            "metadata": {"columns": self.COLUMNS, "rows": int(scores.size)},
        }


class TornadoConditionsAdapter(HazardRowAdapter):
    """Tornado potential from sounding-derived convective parameters.

    Columns: ``cape_j_kg`` (J/kg), ``cin_j_kg`` (J/kg, <= 0), ``srh_m2_s2``
    (m²/s²), ``bulk_shear_kt`` (kt), ``lcl_m`` (m). Score and decision are
    ``TornadoDetector.predict_tornado``'s own confidence and
    ``tornado_likely`` on the atmospheric path.
    """

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

    COLUMNS = (
        "precipitation_1h_inches",
        "precipitation_6h_inches",
        "precipitation_24h_inches",
        "forecast_24h_inches",
    )
    FEATURE_NAMES = ("confidence", "flood_likely", "precipitation_24h_inches")

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
    is NOAA's own scale — the predicted G level over 5 — and the decision is
    G >= 1, exactly the storm definition the training labels use
    (``evaluation.hazard_metrics.noaa_g_scale``).
    """

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
        from omni_mercury_engine.evaluation.hazard_metrics import noaa_g_scale

        result = self._detector.predict_solar_storm(
            {"magnetosphere_data": dict(zip(self.COLUMNS, (float(v) for v in row)))}
        )
        if result.kp_index is None:
            raise InvalidObservationError(
                "solar storm prediction returned no Kp for a measured row"
            )
        g_level = noaa_g_scale(float(result.kp_index))
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
