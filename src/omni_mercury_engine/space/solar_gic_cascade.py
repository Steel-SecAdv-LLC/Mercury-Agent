# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Solar-to-grid (flare/CME -> Kp -> dB/dt) cascade escalation detector.

Composes real upstream signals into a staged escalation state machine for
grid operators, mirroring the physical causal chain of a geomagnetic storm:

1. **WATCH** — an energetic solar driver has been *observed*: an M/X-class
   flare (NASA DONKI FLR) and/or an Earth-directed CME
   (DONKI CMEAnalysis assessed by
   :class:`~omni_mercury_engine.space.cme_arrival_detector.CMEArrivalDetector`,
   Gopalswamy 2001 ESA + Vršnak 2013 DBM arrival windows).
2. **WARNING** — the predicted CME arrival window is *open* (now inside
   [earliest, latest]) AND the magnetosphere is responding: an observed
   planetary Kp at or above the watch threshold (default Kp >= 5, NOAA G1;
   NOAA SWPC Kp product or DONKI GST ``allKpIndex``).
3. **STORM_IN_PROGRESS** — ground truth: *measured* magnetometer dB/dt at
   or above the GIC-relevant tier (default >= 100 nT/min, the "moderate"
   operational tier of
   :class:`~omni_mercury_engine.space.gic_detector.GICDetector`).

Each stage requires its REAL upstream signal — the machine never escalates
on inference alone, and it never de-attributes: elevated Kp or dB/dt
without an observed solar driver is surfaced as an explicit
``unattributed_*`` note while the stage remains un-escalated (strict
escalation discipline). Only observations timestamped at or before ``now``
are considered; future samples in the inputs are ignored (real-time
semantics).

The output carries the full evidence chain: which flare / CME / Kp /
magnetometer datapoints triggered which stage, with values and timestamps.

Physics + composition only — no neural network.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np

from omni_mercury_engine.space.cme_arrival_detector import (
    CMEArrivalDetector,
    CMEArrivalPrediction,
)
from omni_mercury_engine.space.gic_detector import GICAssessment, GICDetector

logger = logging.getLogger(__name__)


class CascadeStage(Enum):
    """Escalation stages of the solar-to-grid cascade."""

    QUIET = "quiet"
    WATCH = "watch"
    WARNING = "warning"
    STORM_IN_PROGRESS = "storm_in_progress"


#: Flare classes that arm the WATCH stage (GOES letter classes).
_WATCH_FLARE_CLASSES: frozenset[str] = frozenset({"M", "X"})
#: Lookback horizon for solar drivers (flares), hours. CME relevance is
#: bounded by its own predicted arrival window instead.
FLARE_LOOKBACK_HOURS: float = 72.0
#: Maximum age of a Kp observation to count as "current", hours (Kp is a
#: 3-hourly index; 6 h tolerates one missed cadence).
KP_MAX_AGE_HOURS: float = 6.0


@dataclass
class CascadeInputs:
    """Real upstream observations for one cascade evaluation.

    Attributes:
        now: Evaluation time (timezone-aware). Observations after ``now``
            are ignored.
        flares: DONKI FLR records (``flrID``, ``classType``, ``beginTime``,
            ``sourceLocation``...).
        cme_analyses: DONKI CMEAnalysis records (kinematics at 21.5 Rs).
        kp_series: Observed planetary Kp as (timestamp, kp) pairs — from
            the SWPC Kp product or DONKI GST ``allKpIndex`` entries.
        mag_times: Magnetometer sample timestamps (minute cadence), or
            None when no ground magnetometer data is available.
        mag_bx_nt: Northward (X) component, nT; aligned with ``mag_times``.
        mag_by_nt: Eastward (Y) component, nT, or None for H-only input.
        observatory: Magnetometer station identifier for the evidence
            chain.
    """

    now: datetime
    flares: list[dict[str, Any]] = field(default_factory=list)
    cme_analyses: list[dict[str, Any]] = field(default_factory=list)
    kp_series: list[tuple[datetime, float]] = field(default_factory=list)
    mag_times: list[datetime] | None = None
    mag_bx_nt: np.ndarray[Any, Any] | None = None
    mag_by_nt: np.ndarray[Any, Any] | None = None
    observatory: str = ""


@dataclass
class StageEvidence:
    """Evidence record for one cascade stage.

    Attributes:
        stage: Stage name this evidence belongs to.
        satisfied: Whether the stage requirement was met by real data.
        reason: Human-readable explanation of the decision.
        datapoints: The concrete triggering observations (IDs, values,
            timestamps) — the audit trail for the escalation.
    """

    stage: str
    satisfied: bool
    reason: str
    datapoints: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CascadeAssessment:
    """Result of a cascade evaluation.

    Attributes:
        stage: Highest stage whose full upstream chain is satisfied.
        evidence_chain: Per-stage evidence, in escalation order.
        earth_directed_cmes: Arrival predictions for all Earth-directed
            CMEs observed at or before ``now``.
        active_arrival_window: (earliest, latest) union of the arrival
            windows open at ``now`` (None when no window is open).
        gic_assessment: The measured-dB/dt assessment (None without
            magnetometer data).
        notes: Data-quality and attribution notes (malformed records,
            unattributed signals).
    """

    stage: CascadeStage
    evidence_chain: list[StageEvidence]
    earth_directed_cmes: list[CMEArrivalPrediction]
    active_arrival_window: tuple[datetime, datetime] | None
    gic_assessment: GICAssessment | None
    notes: list[str] = field(default_factory=list)


def _parse_donki_time(value: str) -> datetime:
    """Parse DONKI timestamps like ``2024-05-08T05:36Z`` to aware UTC."""
    from datetime import UTC

    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


class SolarGICCascadeDetector:
    """Staged flare/CME -> Kp -> dB/dt escalation state machine.

    Composes :class:`CMEArrivalDetector` (arrival windows) and
    :class:`GICDetector` (measured dB/dt) with observed Kp into the
    QUIET / WATCH / WARNING / STORM_IN_PROGRESS ladder, emitting the full
    evidence chain for every decision.

    Example:
        >>> cascade = SolarGICCascadeDetector()
        >>> result = cascade.evaluate(inputs)
        >>> result.stage, [e.reason for e in result.evidence_chain]
    """

    def __init__(
        self,
        cme_detector: CMEArrivalDetector | None = None,
        gic_detector: GICDetector | None = None,
        kp_watch_threshold: float = 5.0,
        dbdt_storm_threshold_nt_per_min: float = 100.0,
    ) -> None:
        """Initialize the cascade detector.

        Args:
            cme_detector: Arrival-window model (default configuration of
                :class:`CMEArrivalDetector` when None).
            gic_detector: Ground-response model (default configuration of
                :class:`GICDetector` when None).
            kp_watch_threshold: Observed Kp required for WARNING
                (default 5.0 = NOAA G1).
            dbdt_storm_threshold_nt_per_min: Measured peak dB/dt required
                for STORM_IN_PROGRESS (default 100 nT/min, the moderate
                GIC tier).

        Raises:
            ValueError: On out-of-range thresholds.
        """
        if not 0.0 <= kp_watch_threshold <= 9.0:
            raise ValueError(f"kp_watch_threshold must be in [0, 9]; got {kp_watch_threshold!r}.")
        if dbdt_storm_threshold_nt_per_min <= 0.0:
            raise ValueError(
                "dbdt_storm_threshold_nt_per_min must be positive; got "
                f"{dbdt_storm_threshold_nt_per_min!r}."
            )
        self.cme_detector = cme_detector or CMEArrivalDetector()
        self.gic_detector = gic_detector or GICDetector()
        self.kp_watch_threshold = float(kp_watch_threshold)
        self.dbdt_storm_threshold = float(dbdt_storm_threshold_nt_per_min)
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, inputs: CascadeInputs) -> CascadeAssessment:
        """Evaluate the cascade stage from real observations.

        Args:
            inputs: Upstream observations (see :class:`CascadeInputs`).

        Returns:
            Stage + full evidence chain.

        Raises:
            ValueError: If ``now`` or any supplied timestamp is naive, or
                Kp values fall outside [0, 9].
        """
        now = inputs.now
        if now.tzinfo is None:
            raise ValueError("CascadeInputs.now must be timezone-aware.")
        for ts, kp in inputs.kp_series:
            if ts.tzinfo is None:
                raise ValueError(f"Kp timestamp {ts!r} is naive; must be timezone-aware.")
            if not 0.0 <= kp <= 9.0:
                raise ValueError(f"Kp value {kp!r} at {ts.isoformat()} outside [0, 9].")

        notes: list[str] = []

        # ---- Stage 1: WATCH (observed solar driver) -------------------
        flare_evidence = self._flare_evidence(inputs.flares, now, notes)
        cme_predictions, cme_evidence = self._cme_evidence(inputs.cme_analyses, now, notes)
        watch_datapoints = flare_evidence + cme_evidence
        watch_satisfied = bool(watch_datapoints)
        if watch_satisfied:
            watch_reason = (
                f"{len(flare_evidence)} M/X flare(s) in the last "
                f"{FLARE_LOOKBACK_HOURS:.0f} h and {len(cme_evidence)} "
                "Earth-directed CME(s) observed"
            )
        else:
            watch_reason = "no M/X flare or Earth-directed CME observed"

        # ---- Stage 2: WARNING (arrival window open + elevated Kp) -----
        open_windows = [
            p
            for p in cme_predictions
            if p.earth_directed and p.earliest_arrival <= now <= p.latest_arrival
        ]
        active_window: tuple[datetime, datetime] | None = None
        if open_windows:
            active_window = (
                min(p.earliest_arrival for p in open_windows),
                max(p.latest_arrival for p in open_windows),
            )
        kp_hit = self._current_kp(inputs.kp_series, now)
        kp_elevated = kp_hit is not None and kp_hit[1] >= self.kp_watch_threshold

        warning_datapoints: list[dict[str, Any]] = [
            {
                "type": "cme_arrival_window",
                "cme_id": p.cme_id,
                "earliest": p.earliest_arrival.isoformat(),
                "latest": p.latest_arrival.isoformat(),
                "most_probable": p.most_probable_arrival.isoformat(),
            }
            for p in open_windows
        ]
        if kp_hit is not None:
            warning_datapoints.append(
                {
                    "type": "kp_observation",
                    "time": kp_hit[0].isoformat(),
                    "kp": kp_hit[1],
                    "threshold": self.kp_watch_threshold,
                }
            )

        warning_satisfied = watch_satisfied and bool(open_windows) and kp_elevated
        if warning_satisfied:
            # warning_satisfied implies kp_elevated, and kp_elevated is only
            # True when kp_hit is present (see kp_elevated above).
            assert kp_hit is not None
            warning_reason = (
                f"{len(open_windows)} arrival window(s) open at now and observed "
                f"Kp {kp_hit[1]:.2f} >= {self.kp_watch_threshold:.1f} "
                "(G1 watch level)"
            )
        elif not watch_satisfied:
            warning_reason = "upstream WATCH not satisfied"
        elif not open_windows:
            warning_reason = "no predicted CME arrival window is open at now"
        elif kp_hit is None:
            warning_reason = f"no Kp observation within {KP_MAX_AGE_HOURS:.0f} h of now"
        else:
            warning_reason = (
                f"observed Kp {kp_hit[1]:.2f} below watch threshold "
                f"{self.kp_watch_threshold:.1f}"
            )
        if kp_elevated and not watch_satisfied:
            # kp_elevated is only True when kp_hit is present (see above).
            assert kp_hit is not None
            notes.append(
                f"unattributed_kp: observed Kp {kp_hit[1]:.2f} with no upstream "
                "solar driver in the inputs; not escalating without the causal chain."
            )

        # ---- Stage 3: STORM_IN_PROGRESS (measured dB/dt) --------------
        gic_assessment, storm_datapoints, dbdt_hit = self._gic_evidence(inputs, now, notes)
        storm_satisfied = warning_satisfied and dbdt_hit
        if storm_satisfied:
            assert gic_assessment is not None  # dbdt_hit implies assessment
            storm_reason = (
                f"measured peak dB/dt {gic_assessment.peak_dbdt_nt_per_min:.1f} "
                f"nT/min >= {self.dbdt_storm_threshold:.0f} nT/min at "
                f"{gic_assessment.observatory or 'magnetometer'}"
            )
        elif not warning_satisfied:
            storm_reason = "upstream WARNING not satisfied"
        elif gic_assessment is None:
            storm_reason = "no magnetometer data supplied at or before now"
        else:
            storm_reason = (
                f"measured peak dB/dt {gic_assessment.peak_dbdt_nt_per_min:.1f} "
                f"nT/min below the {self.dbdt_storm_threshold:.0f} nT/min storm tier"
            )
        if dbdt_hit and not warning_satisfied:
            assert gic_assessment is not None
            notes.append(
                f"unattributed_dbdt: measured {gic_assessment.peak_dbdt_nt_per_min:.1f} "
                "nT/min without a satisfied upstream chain; not escalating "
                "without the causal chain."
            )

        # ---- Stage resolution (strict escalation) ---------------------
        if storm_satisfied:
            stage = CascadeStage.STORM_IN_PROGRESS
        elif warning_satisfied:
            stage = CascadeStage.WARNING
        elif watch_satisfied:
            stage = CascadeStage.WATCH
        else:
            stage = CascadeStage.QUIET

        evidence_chain = [
            StageEvidence("watch", watch_satisfied, watch_reason, watch_datapoints),
            StageEvidence("warning", warning_satisfied, warning_reason, warning_datapoints),
            StageEvidence("storm_in_progress", storm_satisfied, storm_reason, storm_datapoints),
        ]

        assessment = CascadeAssessment(
            stage=stage,
            evidence_chain=evidence_chain,
            earth_directed_cmes=[p for p in cme_predictions if p.earth_directed],
            active_arrival_window=active_window,
            gic_assessment=gic_assessment,
            notes=notes,
        )
        self.logger.info(
            "Cascade @ %s: %s (watch=%s, warning=%s, storm=%s)",
            now.isoformat(),
            stage.value,
            watch_satisfied,
            warning_satisfied,
            storm_satisfied,
        )
        return assessment

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------

    def _flare_evidence(
        self, flares: list[dict[str, Any]], now: datetime, notes: list[str]
    ) -> list[dict[str, Any]]:
        """Collect M/X flares within the lookback window ending at now."""
        evidence: list[dict[str, Any]] = []
        horizon = now - timedelta(hours=FLARE_LOOKBACK_HOURS)
        for record in flares:
            class_type = record.get("classType")
            begin = record.get("beginTime")
            if not class_type or not begin:
                notes.append(
                    f"malformed_flare: record {record.get('flrID', '<no id>')!r} "
                    "missing classType/beginTime; excluded from evidence."
                )
                continue
            try:
                begin_time = _parse_donki_time(str(begin))
            except ValueError:
                notes.append(
                    f"malformed_flare: record {record.get('flrID', '<no id>')!r} "
                    f"has unparseable beginTime {begin!r}; excluded from evidence."
                )
                continue
            if str(class_type)[0].upper() not in _WATCH_FLARE_CLASSES:
                continue
            if not horizon <= begin_time <= now:
                continue
            evidence.append(
                {
                    "type": "flare",
                    "flare_id": record.get("flrID", ""),
                    "class_type": class_type,
                    "begin_time": begin_time.isoformat(),
                    "source_location": record.get("sourceLocation"),
                }
            )
        return evidence

    def _cme_evidence(
        self, cme_analyses: list[dict[str, Any]], now: datetime, notes: list[str]
    ) -> tuple[list[CMEArrivalPrediction], list[dict[str, Any]]]:
        """Predict arrivals for CMEs observed at or before now."""
        predictions: list[CMEArrivalPrediction] = []
        evidence: list[dict[str, Any]] = []
        for record in cme_analyses:
            try:
                time_21_5 = _parse_donki_time(str(record["time21_5"]))
            except (KeyError, ValueError):
                notes.append(
                    "malformed_cme: record "
                    f"{record.get('associatedCMEID', '<no id>')!r} has no "
                    "parseable time21_5; excluded from evidence."
                )
                continue
            if time_21_5 > now:
                continue  # not yet observed at evaluation time
            try:
                prediction = self.cme_detector.predict_from_donki(record)
            except ValueError as exc:
                notes.append(f"malformed_cme: {exc}")
                continue
            predictions.append(prediction)
            if prediction.earth_directed and prediction.latest_arrival >= now:
                evidence.append(
                    {
                        "type": "cme",
                        "cme_id": prediction.cme_id,
                        "directedness": prediction.directedness,
                        "earliest_arrival": prediction.earliest_arrival.isoformat(),
                        "latest_arrival": prediction.latest_arrival.isoformat(),
                        "confidence": prediction.confidence,
                    }
                )
        return predictions, evidence

    def _current_kp(
        self, kp_series: list[tuple[datetime, float]], now: datetime
    ) -> tuple[datetime, float] | None:
        """Most recent Kp observation within KP_MAX_AGE_HOURS of now."""
        recent = [
            (ts, kp)
            for ts, kp in kp_series
            if ts <= now and (now - ts) <= timedelta(hours=KP_MAX_AGE_HOURS)
        ]
        if not recent:
            return None
        return max(recent, key=lambda pair: pair[0])

    def _gic_evidence(
        self, inputs: CascadeInputs, now: datetime, notes: list[str]
    ) -> tuple[GICAssessment | None, list[dict[str, Any]], bool]:
        """Assess measured dB/dt from magnetometer samples up to now."""
        if inputs.mag_times is None or inputs.mag_bx_nt is None:
            return None, [], False
        bx = np.asarray(inputs.mag_bx_nt, dtype=np.float64)
        by = None if inputs.mag_by_nt is None else np.asarray(inputs.mag_by_nt, dtype=np.float64)
        keep = [i for i, ts in enumerate(inputs.mag_times) if ts <= now]
        if len(keep) < 2:
            notes.append(
                "magnetometer data supplied but fewer than 2 samples at or "
                "before now; measured-dB/dt stage cannot be evaluated."
            )
            return None, [], False
        times = [inputs.mag_times[i] for i in keep]
        assessment = self.gic_detector.assess(
            times,
            bx[keep],
            None if by is None else by[keep],
            observatory=inputs.observatory,
        )
        hit = assessment.peak_dbdt_nt_per_min >= self.dbdt_storm_threshold
        datapoints = [
            {
                "type": "dbdt_measurement",
                "observatory": assessment.observatory,
                "peak_dbdt_nt_per_min": assessment.peak_dbdt_nt_per_min,
                "peak_time": assessment.peak_dbdt_time.isoformat(),
                "risk_level": assessment.risk_level,
                "threshold_nt_per_min": self.dbdt_storm_threshold,
            }
        ]
        return assessment, datapoints, hit


__all__ = [
    "FLARE_LOOKBACK_HOURS",
    "KP_MAX_AGE_HOURS",
    "CascadeAssessment",
    "CascadeInputs",
    "CascadeStage",
    "SolarGICCascadeDetector",
    "StageEvidence",
]
