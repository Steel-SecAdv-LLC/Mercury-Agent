# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hurricane -> storm surge -> compound flood cascade.

Composes three REAL upstream signals into a staged state machine, with the
full evidence chain carried in the output:

1. **Hurricane evidence** - a
   :class:`~omni_mercury_engine.detectors.geological.hurricane_detector.HurricanePredictionResult`
   from the branch's hurricane detector physics core (pressure-deficit /
   Dvorak pressure-wind relationship; ``cyclone_detected``, ``category``,
   ``max_wind_speed_kt``, ``min_pressure_mb``).

2. **Surge evidence** - NOAA CO-OPS observed water levels versus the
   astronomical tide prediction for the same station and datum.  The
   CO-OPS ``datagetter`` API exposes both series as separate products
   (``product=water_level`` and ``product=predictions``); the storm-surge
   residual is their difference::

       surge(t) = observed_water_level(t) - predicted_tide(t)

   which is the standard non-tidal residual used in surge monitoring.
   Series are aligned on exact timestamps; the cascade fails loudly when
   the two series do not overlap instead of interpolating fabricated
   values.

3. **River flood evidence** - NOAA NWPS (National Water Prediction
   Service) gauge status: the observed stage versus the NWS flood
   categories (action/minor/moderate/major) published per gauge.

Stages (each requires its full upstream evidence chain):

- ``QUIET``            - no qualifying evidence.
- ``WATCH``            - hurricane detected at or above the configured
  minimum category.
- ``SURGE_OBSERVED``   - WATCH evidence AND observed surge residual at or
  above the threshold.
- ``COMPOUND_FLOOD``   - SURGE_OBSERVED evidence AND a river gauge at or
  above its NWS minor-flood category.

``evaluate()`` is a pure function of the currently-held evidence: stages
downgrade automatically when evidence is retracted or replaced by
non-qualifying data.  Every ``update_*`` call appends an
:class:`EvidenceRecord` to the chain with the raw values that drove it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from omni_mercury_engine.detectors.geological.hurricane_detector import (
        HurricanePredictionResult,
    )

logger = logging.getLogger(__name__)

#: Saffir-Simpson ordering used for the WATCH gate.
_CATEGORY_ORDER: tuple[str, ...] = (
    "no_cyclone",
    "tropical_depression",
    "tropical_storm",
    "category_1",
    "category_2",
    "category_3",
    "category_4",
    "category_5",
)

#: NWPS flood categories that qualify as river flooding.
_FLOODING_CATEGORIES: frozenset[str] = frozenset({"minor", "moderate", "major"})


class CascadeStage(Enum):
    """Stages of the hurricane -> surge -> compound-flood cascade."""

    QUIET = 0
    WATCH = 1
    SURGE_OBSERVED = 2
    COMPOUND_FLOOD = 3


@dataclass
class EvidenceRecord:
    """One piece of upstream evidence feeding the cascade.

    Attributes:
        kind: "hurricane" | "surge" | "river".
        source: Human-readable origin (detector / station / gauge id).
        timestamp: ISO timestamp of the evidence (observation or update).
        qualifies: Whether this record meets its stage gate on its own.
        summary: The raw values that drove the qualification decision.
    """

    kind: str
    source: str
    timestamp: str
    qualifies: bool
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class SurgeSeries:
    """Aligned observed / predicted / residual water-level series (metres).

    Attributes:
        timestamps: Aligned timestamp strings ("YYYY-MM-DD HH:MM", GMT).
        observed_m: Observed water levels.
        predicted_m: Predicted (astronomical tide) water levels.
        residual_m: observed - predicted, the storm-surge residual.
        max_residual_m: Maximum residual.
        time_of_max: Timestamp of the maximum residual.
    """

    timestamps: list[str]
    observed_m: np.ndarray
    predicted_m: np.ndarray
    residual_m: np.ndarray
    max_residual_m: float
    time_of_max: str


@dataclass
class CascadeAssessment:
    """Output of :meth:`SurgeFloodCascade.evaluate`.

    Attributes:
        stage: The awarded :class:`CascadeStage`.
        stage_name: Its string name.
        hurricane_qualifies: Hurricane gate state.
        surge_qualifies: Surge gate state.
        river_qualifies: River gate state.
        evidence_chain: Every evidence record held, in arrival order.
        surge: The aligned surge series, when surge evidence exists.
        details: Gate-by-gate explanation of the awarded stage.
    """

    stage: CascadeStage
    stage_name: str
    hurricane_qualifies: bool
    surge_qualifies: bool
    river_qualifies: bool
    evidence_chain: list[EvidenceRecord] = field(default_factory=list)
    surge: SurgeSeries | None = None
    details: list[str] = field(default_factory=list)


def _parse_coops_series(payload: Any, kind: str) -> dict[str, float]:
    """Parse a CO-OPS datagetter payload into a timestamp -> value map.

    Accepts the raw JSON dict of ``product=water_level`` (records under
    ``"data"``) or ``product=predictions`` (records under
    ``"predictions"``), or an already-flat list of ``{"t":..., "v":...}``
    records.

    Args:
        payload: CO-OPS JSON payload or record list.
        kind: "observed" or "predicted" (error messages only).

    Returns:
        Ordered mapping of timestamp string to float value.

    Raises:
        ValueError: On a CO-OPS error payload, unrecognized structure, or
            zero parseable records.
    """
    records: Any
    if isinstance(payload, dict):
        if "error" in payload:
            message = payload["error"].get("message", str(payload["error"]))
            raise ValueError(f"CO-OPS returned an error for the {kind} series: {message}")
        records = payload.get("data") or payload.get("predictions")
        if records is None:
            raise ValueError(
                f"unrecognized CO-OPS payload for the {kind} series: expected "
                "'data' (water_level) or 'predictions' (predictions) key"
            )
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError(f"unsupported CO-OPS payload type for {kind}: {type(payload)!r}")

    out: dict[str, float] = {}
    for rec in records:
        t = rec.get("t")
        v = rec.get("v")
        if not t or v in (None, ""):
            continue
        try:
            out[str(t)] = float(v)
        except (TypeError, ValueError):
            continue
    if not out:
        raise ValueError(f"CO-OPS {kind} series contained no parseable records")
    return out


class SurgeFloodCascade:
    """Staged hurricane -> surge -> compound-flood cascade detector."""

    def __init__(
        self,
        surge_threshold_m: float = 0.3,
        min_watch_category: str = "tropical_storm",
    ) -> None:
        """Initialize the cascade.

        Args:
            surge_threshold_m: Storm-surge residual (metres) required for
                the SURGE_OBSERVED gate.  The 0.3 m default is a
                conservative monitoring threshold - well above CO-OPS
                verified-water-level noise, well below damaging surge.
            min_watch_category: Minimum Saffir-Simpson category string
                (as produced by the hurricane detector) for the WATCH
                gate.

        Raises:
            ValueError: On a non-positive threshold or unknown category.
        """
        if surge_threshold_m <= 0.0:
            raise ValueError(f"surge_threshold_m must be > 0, got {surge_threshold_m}")
        if min_watch_category not in _CATEGORY_ORDER:
            raise ValueError(
                f"unknown category {min_watch_category!r}; expected one of {_CATEGORY_ORDER}"
            )
        self.surge_threshold_m = surge_threshold_m
        self.min_watch_category = min_watch_category
        self.logger = logging.getLogger(__name__)

        self._evidence: list[EvidenceRecord] = []
        self._hurricane_qualifies = False
        self._surge_qualifies = False
        self._river_qualifies = False
        self._surge_series: SurgeSeries | None = None

    # ------------------------------------------------------------------
    # Evidence updates
    # ------------------------------------------------------------------

    def update_hurricane_evidence(
        self,
        result: HurricanePredictionResult,
        timestamp: str | None = None,
    ) -> EvidenceRecord:
        """Register hurricane-detector output as WATCH evidence.

        Args:
            result: Output of ``HurricaneDetector.predict_hurricane`` run
                on real observations.
            timestamp: ISO timestamp of the underlying observations;
                defaults to now (UTC).

        Returns:
            The appended :class:`EvidenceRecord`.
        """
        min_rank = _CATEGORY_ORDER.index(self.min_watch_category)
        try:
            rank = _CATEGORY_ORDER.index(result.category)
        except ValueError as exc:
            raise ValueError(
                f"hurricane result carries unknown category {result.category!r}"
            ) from exc
        qualifies = bool(result.cyclone_detected) and rank >= min_rank
        record = EvidenceRecord(
            kind="hurricane",
            source="HurricaneDetector.predict_hurricane",
            timestamp=timestamp or datetime.now(UTC).isoformat(),
            qualifies=qualifies,
            summary={
                "cyclone_detected": bool(result.cyclone_detected),
                "category": result.category,
                "max_wind_speed_kt": float(result.max_wind_speed_kt),
                "min_pressure_mb": float(result.min_pressure_mb),
                "confidence": float(result.confidence),
                "min_watch_category": self.min_watch_category,
            },
        )
        self._evidence.append(record)
        self._hurricane_qualifies = qualifies
        return record

    def compute_surge_residual(
        self,
        observed_payload: Any,
        predicted_payload: Any,
    ) -> SurgeSeries:
        """Align observed and predicted CO-OPS series and compute the residual.

        Args:
            observed_payload: CO-OPS ``product=water_level`` JSON payload
                (or flat record list).
            predicted_payload: CO-OPS ``product=predictions`` JSON payload
                (or flat record list).

        Returns:
            The aligned :class:`SurgeSeries`.

        Raises:
            ValueError: If either series is empty/unparseable or the
                timestamps do not overlap (the residual would otherwise
                have to be interpolated from fabricated pairs).
        """
        observed = _parse_coops_series(observed_payload, "observed")
        predicted = _parse_coops_series(predicted_payload, "predicted")

        common = [t for t in observed if t in predicted]
        if not common:
            raise ValueError(
                "observed and predicted CO-OPS series share no timestamps; "
                "fetch both products for the same station, interval and "
                "time window"
            )
        obs = np.array([observed[t] for t in common], dtype=np.float64)
        pred = np.array([predicted[t] for t in common], dtype=np.float64)
        residual = obs - pred
        idx_max = int(np.argmax(residual))
        return SurgeSeries(
            timestamps=common,
            observed_m=obs,
            predicted_m=pred,
            residual_m=residual,
            max_residual_m=float(residual[idx_max]),
            time_of_max=common[idx_max],
        )

    def update_surge_evidence(
        self,
        observed_payload: Any,
        predicted_payload: Any,
        station_id: str = "",
    ) -> EvidenceRecord:
        """Register CO-OPS observed-vs-predicted water levels as surge evidence.

        Args:
            observed_payload: CO-OPS water_level payload.
            predicted_payload: CO-OPS predictions payload.
            station_id: Station identifier for the evidence chain.

        Returns:
            The appended :class:`EvidenceRecord`.

        Raises:
            ValueError: Propagated from :meth:`compute_surge_residual`.
        """
        series = self.compute_surge_residual(observed_payload, predicted_payload)
        qualifies = series.max_residual_m >= self.surge_threshold_m
        record = EvidenceRecord(
            kind="surge",
            source=f"NOAA CO-OPS station {station_id or 'unknown'}",
            timestamp=series.time_of_max,
            qualifies=qualifies,
            summary={
                "max_residual_m": series.max_residual_m,
                "surge_threshold_m": self.surge_threshold_m,
                "n_aligned_samples": len(series.timestamps),
                "window_start": series.timestamps[0],
                "window_end": series.timestamps[-1],
            },
        )
        self._evidence.append(record)
        self._surge_qualifies = qualifies
        self._surge_series = series
        return record

    def update_surge_evidence_from_datapoints(self, points: list[Any]) -> EvidenceRecord:
        """Register surge evidence from ``NOAACOOPSSource`` data points.

        Splits the fetched points by ``data["product"]`` ("water_level" vs
        "predictions") and reuses :meth:`update_surge_evidence`.

        Args:
            points: DataPoint list from
                ``NOAACOOPSSource(products=[WATER_LEVEL, PREDICTIONS]).fetch()``.

        Returns:
            The appended :class:`EvidenceRecord`.

        Raises:
            ValueError: If either product is absent from the points.
        """
        observed: list[dict[str, Any]] = []
        predicted: list[dict[str, Any]] = []
        station_id = ""
        for point in points:
            data = getattr(point, "data", None)
            if not isinstance(data, dict):
                continue
            station_id = str(data.get("station_id", station_id))
            timestamp = getattr(point, "timestamp", None)
            t_str = timestamp.strftime("%Y-%m-%d %H:%M") if timestamp is not None else ""
            rec = {"t": t_str, "v": data.get("value")}
            if data.get("product") == "water_level":
                observed.append(rec)
            elif data.get("product") == "predictions":
                predicted.append(rec)
        if not observed or not predicted:
            raise ValueError(
                "need both water_level and predictions products in the "
                f"CO-OPS points (got {len(observed)} observed / "
                f"{len(predicted)} predicted records)"
            )
        return self.update_surge_evidence(observed, predicted, station_id=station_id)

    def update_river_evidence(self, gauge_payload: dict[str, Any]) -> EvidenceRecord:
        """Register an NWPS gauge status as river-flood evidence.

        Args:
            gauge_payload: JSON dict from the NWPS v1 ``/gauges/{lid}``
                endpoint (fields: ``lid``, ``status.observed.primary``,
                ``status.observed.floodCategory``,
                ``flood.categories.minor.stage`` ...).

        Returns:
            The appended :class:`EvidenceRecord`.

        Raises:
            ValueError: If the payload lacks an observed stage value.
        """
        if not isinstance(gauge_payload, dict):
            raise ValueError(f"gauge_payload must be a dict, got {type(gauge_payload)!r}")
        lid = str(gauge_payload.get("lid", "unknown"))
        status = gauge_payload.get("status", {}) or {}
        observed = status.get("observed", {}) or {}
        primary = observed.get("primary")
        if primary is None or primary in ("", -999):
            raise ValueError(f"NWPS gauge {lid} has no observed stage value")
        stage = float(primary)
        flood_category = str(observed.get("floodCategory", "")).lower()
        categories = (gauge_payload.get("flood", {}) or {}).get("categories", {}) or {}
        minor_stage = categories.get("minor", {}).get("stage")

        in_flood = flood_category in _FLOODING_CATEGORIES
        if not in_flood and minor_stage is not None and float(minor_stage) > 0:
            in_flood = stage >= float(minor_stage)

        record = EvidenceRecord(
            kind="river",
            source=f"NOAA NWPS gauge {lid}",
            timestamp=str(observed.get("validTime", "")),
            qualifies=in_flood,
            summary={
                "lid": lid,
                "observed_stage": stage,
                "observed_unit": str(observed.get("primaryUnit", "")),
                "flood_category": flood_category,
                "minor_flood_stage": minor_stage,
            },
        )
        self._evidence.append(record)
        self._river_qualifies = in_flood
        return record

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def evaluate(self) -> CascadeAssessment:
        """Evaluate the staged state machine on the current evidence.

        Each stage requires the full upstream chain: surge evidence alone
        (no qualifying hurricane) does not leave QUIET, and river flooding
        alone does not reach COMPOUND_FLOOD.

        Returns:
            A :class:`CascadeAssessment` with the awarded stage and the
            complete evidence chain.
        """
        details: list[str] = []
        stage = CascadeStage.QUIET
        if self._hurricane_qualifies:
            stage = CascadeStage.WATCH
            details.append(f"WATCH: hurricane evidence at/above {self.min_watch_category}")
            if self._surge_qualifies:
                stage = CascadeStage.SURGE_OBSERVED
                assert self._surge_series is not None
                details.append(
                    "SURGE_OBSERVED: max residual "
                    f"{self._surge_series.max_residual_m:.2f} m >= "
                    f"{self.surge_threshold_m:.2f} m"
                )
                if self._river_qualifies:
                    stage = CascadeStage.COMPOUND_FLOOD
                    details.append("COMPOUND_FLOOD: river gauge at/above NWS minor flood")
                elif any(e.kind == "river" for e in self._evidence):
                    details.append("river evidence present but below flood stage")
            else:
                if any(e.kind == "surge" for e in self._evidence):
                    details.append("surge evidence present but below threshold")
                if self._river_qualifies:
                    details.append(
                        "river flooding observed but surge gate not met - "
                        "stage capped at WATCH (each stage requires its "
                        "upstream evidence)"
                    )
        elif self._surge_qualifies or self._river_qualifies:
            details.append(
                "surge/river evidence held without qualifying hurricane "
                "evidence - stage stays QUIET (upstream chain required)"
            )
        else:
            details.append("no qualifying evidence")

        assessment = CascadeAssessment(
            stage=stage,
            stage_name=stage.name,
            hurricane_qualifies=self._hurricane_qualifies,
            surge_qualifies=self._surge_qualifies,
            river_qualifies=self._river_qualifies,
            evidence_chain=list(self._evidence),
            surge=self._surge_series,
            details=details,
        )
        self.logger.info("Cascade stage: %s (%s)", stage.name, "; ".join(details))
        return assessment

    def reset(self) -> None:
        """Clear all held evidence and return to QUIET."""
        self._evidence.clear()
        self._hurricane_qualifies = False
        self._surge_qualifies = False
        self._river_qualifies = False
        self._surge_series = None
