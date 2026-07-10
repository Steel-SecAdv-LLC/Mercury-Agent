# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Earthquake → Tsunami Cascade Detector — PTWC-style screening + staged state machine.

Composes real signals end to end:

1. **Event intake** from the USGS FDSN earthquake feed — either live
   :class:`~omni_mercury_engine.data_sources.earth_science.USGSEarthquakeSource`
   data points or injected/raw GeoJSON features (the standard USGS product
   shape).
2. **Tsunamigenic screening** per the published PTWC bulletin criteria
   (IOC-UNESCO, *Operational Users Guide for the Pacific Tsunami Warning and
   Mitigation System*, IOC Technical Series No. 87, §4.3.2, table of message
   types and criteria):

   ===================  ==========  ===================  ==========================
   Moment magnitude     Depth       Location             Product
   ===================  ==========  ===================  ==========================
   6.5 – 7.5            any         any                  Tsunami Information Bulletin
   >= 7.6               >= 100 km   any                  Tsunami Information Bulletin
   >= 7.6               < 100 km    inland               Tsunami Information Bulletin
   7.6 – 7.8            < 100 km    near shore/offshore  Regional Fixed Warning
   >= 7.9               < 100 km    near shore/offshore  Regional Expanding Warning
   ===================  ==========  ===================  ==========================

   The same guide states that "for tsunamigenesis to be possible the
   hypocenter must be within 100 km of the earth's surface and either under
   the sea or very near the sea".
3. **Staged state machine** ``QUIET → EVALUATING → TSUNAMI_WATCH →
   TSUNAMI_THREAT``, each transition requiring real evidence:

   - ``EVALUATING``: an event at or above the information-bulletin floor
     (M >= 6.5) arrived and is being screened.
   - ``TSUNAMI_WATCH``: the screening produced a warning-class product
     (M >= 7.6, depth < 100 km, not known-inland).
   - ``TSUNAMI_THREAT``: **only** on deterministic water-level confirmation —
     a DART-style residual on a supplied sea-level series exceeding the
     30 mm deep-ocean tsunami detection threshold used by the NOAA DART
     algorithm (Mofjeld, NOAA/PMEL tsunami detection algorithm; Meinig et
     al., 2005), or an explicit confirmed observation. The
     :class:`~omni_mercury_engine.detectors.geological.disaster_detectors.
     TsunamiDetector` spectral analysis (its deterministic resonance
     evidence) is attached to the evidence chain as *supplementary* signal
     when torch is available, but its neural confidence is **never** a
     transition criterion — the analyser network is untrained at
     construction (training is a separate opt-in path in
     ``disaster_detectors``) and random-weight output must not gate a
     THREAT.

Honesty notes on USGS event metadata:

- The basic USGS GeoJSON feed provides magnitude, hypocentre and a place
  string but **no focal mechanism**; a thrust-vs-strike-slip criterion is
  therefore *not* implemented rather than proxied from nothing. When a
  caller supplies mechanism knowledge (``rake_deg``), it is recorded in the
  evidence chain but the published PTWC table above (which is
  mechanism-free) still decides the stage.
- The feed's ``tsunami`` flag reflects *real-time* alert state and is
  routinely 0 for historical archive queries (it is 0 for the archived 2011
  Tōhoku event); it is recorded as evidence but never used as a criterion.
- Whether the epicentre is under sea is not encoded in the feed. The caller
  may pass ``offshore=True/False`` when known; when unknown, screening
  proceeds conservatively (a warning-class event is not suppressed by
  missing location knowledge — PTWC issues first bulletins on magnitude and
  depth) and the evidence chain marks the location criterion ``unverified``.

References:
    - IOC-UNESCO (2009). Operational Users Guide for the Pacific Tsunami
      Warning and Mitigation System (PTWS). IOC Technical Series No. 87.
      §3.5.1 (tsunamigenesis depth/sea criteria) and §4.3.2 (message types
      and criteria table).
    - Meinig, C., Stalin, S.E., Nakamura, A.I., Milburn, H.B. (2005).
      Real-time deep-ocean tsunami measuring, monitoring, and reporting
      system: the NOAA DART II description and disclosure. NOAA/PMEL.
      (30 mm deep-ocean detection threshold of the DART algorithm.)
    - Mofjeld, H.O. NOAA/PMEL tsunami detection algorithm (DART):
      cubic-polynomial tide prediction from the preceding observations,
      trigger on residual amplitude.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: PTWC information-bulletin magnitude floor (IOC TS No. 87 §4.3.2).
INFO_BULLETIN_MIN_MAGNITUDE = 6.5

#: PTWC regional fixed warning band lower bound (IOC TS No. 87 §4.3.2).
REGIONAL_WARNING_MIN_MAGNITUDE = 7.6

#: PTWC regional expanding warning/watch lower bound (IOC TS No. 87 §4.3.2).
EXPANDING_WARNING_MIN_MAGNITUDE = 7.9

#: Maximum tsunamigenic hypocentral depth, km (IOC TS No. 87 §3.5.1).
MAX_TSUNAMIGENIC_DEPTH_KM = 100.0

#: DART deep-ocean tsunami detection amplitude threshold, metres
#: (Meinig et al. 2005; Mofjeld detection algorithm).
DART_DETECTION_THRESHOLD_M = 0.03


class CascadeStage(Enum):
    """Cascade state machine stages."""

    QUIET = "quiet"
    EVALUATING = "evaluating"
    TSUNAMI_WATCH = "tsunami_watch"
    TSUNAMI_THREAT = "tsunami_threat"


class ScreeningProduct(Enum):
    """PTWC-style screening outcome (IOC TS No. 87 §4.3.2)."""

    NONE = "none"
    INFORMATION_BULLETIN = "information_bulletin"
    REGIONAL_FIXED_WARNING = "regional_fixed_warning"
    REGIONAL_EXPANDING_WARNING = "regional_expanding_warning"


@dataclass
class ScreeningResult:
    """Outcome of tsunamigenic screening for one earthquake event.

    Attributes:
        product: PTWC-style product class.
        magnitude: Event magnitude used.
        depth_km: Hypocentral depth used, km.
        offshore: Caller-supplied sea/land knowledge (None = unknown).
        criteria: Named criteria evaluated, for the evidence chain.
        event_id: USGS event id when available.
        place: USGS place string when available.
    """

    product: ScreeningProduct
    magnitude: float
    depth_km: float
    offshore: bool | None
    criteria: dict[str, Any] = field(default_factory=dict)
    event_id: str = ""
    place: str = ""


@dataclass
class WaterLevelConfirmation:
    """Deterministic DART-style water-level analysis record.

    Attributes:
        confirmed: Residual amplitude reached the detection threshold.
        max_residual_m: Largest |observed - predicted tide| in the
            evaluation window, metres.
        threshold_m: Threshold applied.
        n_samples: Series length.
        supplementary_spectral: Optional TsunamiDetector deterministic
            resonance evidence (None when torch unavailable or skipped).
    """

    confirmed: bool
    max_residual_m: float
    threshold_m: float
    n_samples: int
    supplementary_spectral: dict[str, Any] | None = None


@dataclass
class CascadeState:
    """Full cascade output after processing evidence.

    Attributes:
        stage: Current stage value.
        screening: Latest screening result (None in QUIET).
        water_level: Latest water-level confirmation (None before THREAT
            evaluation).
        evidence_chain: Ordered evidence records for every transition.
    """

    stage: str
    screening: ScreeningResult | None = None
    water_level: WaterLevelConfirmation | None = None
    evidence_chain: list[dict[str, Any]] = field(default_factory=list)


class EqTsunamiCascadeDetector:
    """Earthquake → tsunami cascade with PTWC-style staged escalation.

    The detector is deterministic and works untrained. Thresholds default to
    the published PTWC criteria and the DART detection amplitude; they are
    constructor arguments only so tests can exercise the gates, and the
    provenance of each default is documented at the module constants.

    Args:
        info_min_magnitude: Information-bulletin floor. Default 6.5.
        watch_min_magnitude: Regional warning floor. Default 7.6.
        expanding_min_magnitude: Expanding warning floor. Default 7.9.
        max_depth_km: Maximum tsunamigenic depth. Default 100 km.
        dart_threshold_m: Water-level residual threshold. Default 0.03 m.
    """

    def __init__(
        self,
        info_min_magnitude: float = INFO_BULLETIN_MIN_MAGNITUDE,
        watch_min_magnitude: float = REGIONAL_WARNING_MIN_MAGNITUDE,
        expanding_min_magnitude: float = EXPANDING_WARNING_MIN_MAGNITUDE,
        max_depth_km: float = MAX_TSUNAMIGENIC_DEPTH_KM,
        dart_threshold_m: float = DART_DETECTION_THRESHOLD_M,
    ) -> None:
        """Initialize the instance."""
        self.info_min_magnitude = info_min_magnitude
        self.watch_min_magnitude = watch_min_magnitude
        self.expanding_min_magnitude = expanding_min_magnitude
        self.max_depth_km = max_depth_km
        self.dart_threshold_m = dart_threshold_m
        self.logger = logging.getLogger(__name__)
        self._stage = CascadeStage.QUIET
        self._screening: ScreeningResult | None = None
        self._water_level: WaterLevelConfirmation | None = None
        self._evidence: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Event parsing
    # ------------------------------------------------------------------
    @staticmethod
    def parse_event(event: dict[str, Any]) -> dict[str, Any]:
        """Normalise a USGS event into the fields screening needs.

        Accepts either a raw USGS GeoJSON *feature* (``properties`` +
        ``geometry``) or a flat dict in the
        :class:`~omni_mercury_engine.data_sources.base.DataPoint` ``data``
        shape (``magnitude`` + ``depth_km``).

        Args:
            event: Raw feature or flat event dict.

        Returns:
            Dict with event_id, magnitude, depth_km, place, tsunami_flag,
            and rake_deg (None unless the caller supplied focal-mechanism
            knowledge — the basic USGS feed has none).

        Raises:
            ValueError: When magnitude or depth cannot be found — screening
                without them would be fabrication.
        """
        if "properties" in event and "geometry" in event:
            props = event.get("properties") or {}
            coords = (event.get("geometry") or {}).get("coordinates") or []
            magnitude = props.get("mag")
            depth = coords[2] if len(coords) > 2 else None
            place = str(props.get("place", ""))
            event_id = str(event.get("id", ""))
            tsunami_flag = bool(props.get("tsunami", 0))
            rake = props.get("rake_deg")
        else:
            magnitude = event.get("magnitude", event.get("mag"))
            depth = event.get("depth_km", event.get("depth"))
            place = str(event.get("place", ""))
            event_id = str(event.get("event_id", event.get("id", "")))
            tsunami_flag = bool(event.get("tsunami", False))
            rake = event.get("rake_deg")

        if magnitude is None or not np.isfinite(float(magnitude)):
            raise ValueError(f"event {event_id!r} has no finite magnitude; cannot screen")
        if depth is None or not np.isfinite(float(depth)):
            raise ValueError(f"event {event_id!r} has no finite depth; cannot screen")

        return {
            "event_id": event_id,
            "magnitude": float(magnitude),
            "depth_km": float(depth),
            "place": place,
            "tsunami_flag": tsunami_flag,
            "rake_deg": float(rake) if rake is not None else None,
        }

    # ------------------------------------------------------------------
    # Screening (IOC TS No. 87 §4.3.2)
    # ------------------------------------------------------------------
    def screen_event(self, event: dict[str, Any], offshore: bool | None = None) -> ScreeningResult:
        """Screen one earthquake event against the PTWC bulletin criteria.

        Args:
            event: USGS GeoJSON feature or flat event dict (see
                :meth:`parse_event`).
            offshore: True when the epicentre is known under sea / near
                coast, False when known inland, None when unknown. Unknown
                locations do not suppress a warning-class product (PTWC's
                first bulletins are magnitude/depth driven) but are recorded
                as ``location_criterion: "unverified"``.

        Returns:
            ScreeningResult with the product class and the full criteria
            record.
        """
        parsed = self.parse_event(event)
        magnitude = parsed["magnitude"]
        depth = parsed["depth_km"]

        shallow = depth < self.max_depth_km
        criteria: dict[str, Any] = {
            "magnitude": magnitude,
            "depth_km": depth,
            "depth_lt_100km": shallow,
            "location_criterion": (
                "offshore" if offshore else "inland" if offshore is False else "unverified"
            ),
            "usgs_tsunami_flag": parsed["tsunami_flag"],
            # Recorded for the evidence chain only; the published PTWC table
            # is mechanism-free, so rake never changes the product class.
            "focal_mechanism": (
                "unavailable_in_basic_feed"
                if parsed["rake_deg"] is None
                else {"rake_deg": parsed["rake_deg"], "source": "caller_supplied"}
            ),
            "citation": "IOC Technical Series No. 87 (PTWS Users Guide) sec. 4.3.2",
        }

        if magnitude < self.info_min_magnitude:
            product = ScreeningProduct.NONE
        elif magnitude < self.watch_min_magnitude or not shallow or offshore is False:
            product = ScreeningProduct.INFORMATION_BULLETIN
        elif magnitude < self.expanding_min_magnitude:
            product = ScreeningProduct.REGIONAL_FIXED_WARNING
        else:
            product = ScreeningProduct.REGIONAL_EXPANDING_WARNING

        return ScreeningResult(
            product=product,
            magnitude=magnitude,
            depth_km=depth,
            offshore=offshore,
            criteria=criteria,
            event_id=parsed["event_id"],
            place=parsed["place"],
        )

    def screen_catalog(
        self, geojson: dict[str, Any], offshore: bool | None = None
    ) -> list[ScreeningResult]:
        """Screen every feature of a USGS GeoJSON FeatureCollection.

        Args:
            geojson: Parsed USGS FeatureCollection (``features`` list).
            offshore: Shared location knowledge (usually None for catalogs).

        Returns:
            One ScreeningResult per feature, in catalog order.

        Raises:
            ValueError: If the collection has no ``features`` list.
        """
        features = geojson.get("features")
        if not isinstance(features, list) or not features:
            raise ValueError("GeoJSON has no non-empty 'features' list")
        return [self.screen_event(f, offshore=offshore) for f in features]

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------
    def process_event(self, event: dict[str, Any], offshore: bool | None = None) -> CascadeState:
        """Advance the state machine with a new earthquake event.

        QUIET/any → EVALUATING (event at/above the information floor), then
        immediately to TSUNAMI_WATCH when screening yields a warning-class
        product. THREAT is *never* reached here — it requires water-level
        confirmation via :meth:`confirm_water_level`.

        Args:
            event: USGS GeoJSON feature or flat event dict.
            offshore: Location knowledge (see :meth:`screen_event`).

        Returns:
            CascadeState snapshot after the transition.
        """
        screening = self.screen_event(event, offshore=offshore)
        self._screening = screening

        if screening.product is ScreeningProduct.NONE:
            self._record(
                "screening",
                stage_after=self._stage,
                detail={
                    "event_id": screening.event_id,
                    "product": screening.product.value,
                    **screening.criteria,
                },
            )
            return self.state()

        self._stage = CascadeStage.EVALUATING
        self._record(
            "event_intake",
            stage_after=self._stage,
            detail={
                "event_id": screening.event_id,
                "place": screening.place,
                "magnitude": screening.magnitude,
                "depth_km": screening.depth_km,
                "source": "USGS FDSN event feed",
            },
        )

        if screening.product in (
            ScreeningProduct.REGIONAL_FIXED_WARNING,
            ScreeningProduct.REGIONAL_EXPANDING_WARNING,
        ):
            self._stage = CascadeStage.TSUNAMI_WATCH
        self._record(
            "screening",
            stage_after=self._stage,
            detail={
                "event_id": screening.event_id,
                "product": screening.product.value,
                **screening.criteria,
            },
        )
        return self.state()

    def confirm_water_level(
        self,
        water_level_m: np.ndarray[Any, Any],
        sampling_interval_s: float,
        run_spectral_analysis: bool = True,
    ) -> CascadeState:
        """Evaluate a water-level series for tsunami confirmation.

        DART-style deterministic detection: a cubic polynomial (the tide
        model order of the DART/Mofjeld algorithm) is fitted to the leading
        75% of the series and extrapolated over the trailing 25% evaluation
        window; the maximum absolute residual there is compared against the
        detection threshold. Only this deterministic gate can raise
        TSUNAMI_THREAT.

        Args:
            water_level_m: Sea-level/pressure-derived water column series in
                metres (DART-style bottom-pressure or coastal gauge).
            sampling_interval_s: Sampling interval, seconds.
            run_spectral_analysis: Also attach the TsunamiDetector's
                deterministic resonance evidence when torch is available
                (supplementary only; never a transition criterion).

        Returns:
            CascadeState snapshot after evaluation.

        Raises:
            RuntimeError: When called before any watch-class screening —
                a THREAT without an earthquake in context would break the
                cascade's evidence chain.
            ValueError: On an unusable series (too short, non-finite) or
                non-positive sampling interval.
        """
        if self._stage is not CascadeStage.TSUNAMI_WATCH:
            raise RuntimeError(
                "confirm_water_level requires stage TSUNAMI_WATCH (got "
                f"{self._stage.value}); process a warning-class earthquake first."
            )
        series = np.asarray(water_level_m, dtype=float)
        if series.ndim != 1 or series.size < 40:
            raise ValueError(
                f"water-level series must be 1-D with >= 40 samples, got {series.shape}"
            )
        if not np.all(np.isfinite(series)):
            raise ValueError("water-level series contains non-finite values")
        if sampling_interval_s <= 0:
            raise ValueError(f"sampling_interval_s must be > 0, got {sampling_interval_s}")

        n = series.size
        split = int(n * 0.75)
        t = np.arange(n, dtype=float) * sampling_interval_s
        # Cubic tide prediction from the leading window (Mofjeld/DART order).
        coef = np.polyfit(t[:split], series[:split], 3)
        predicted = np.polyval(coef, t[split:])
        residual = series[split:] - predicted
        max_residual = float(np.max(np.abs(residual)))
        confirmed = max_residual >= self.dart_threshold_m

        supplementary: dict[str, Any] | None = None
        if run_spectral_analysis:
            supplementary = self._supplementary_spectral(series)

        self._water_level = WaterLevelConfirmation(
            confirmed=confirmed,
            max_residual_m=max_residual,
            threshold_m=self.dart_threshold_m,
            n_samples=n,
            supplementary_spectral=supplementary,
        )
        if confirmed:
            self._stage = CascadeStage.TSUNAMI_THREAT
        self._record(
            "water_level_confirmation",
            stage_after=self._stage,
            detail={
                "confirmed": confirmed,
                "max_residual_m": max_residual,
                "threshold_m": self.dart_threshold_m,
                "method": "cubic detide + residual amplitude (DART-style)",
                "citation": "Meinig et al. 2005; Mofjeld NOAA/PMEL detection algorithm",
                "supplementary_spectral": supplementary,
            },
        )
        return self.state()

    def confirm_observation(self, description: str, source: str) -> CascadeState:
        """Escalate on an explicit confirmed tsunami observation.

        For credible external confirmations (national agency report, tide
        gauge operator) that arrive as text rather than a series. The
        description and source are mandatory — an empty confirmation is
        refused.

        Args:
            description: What was observed (non-empty).
            source: Who reported it (non-empty).

        Returns:
            CascadeState snapshot.

        Raises:
            RuntimeError: When not in TSUNAMI_WATCH.
            ValueError: On empty description/source.
        """
        if self._stage is not CascadeStage.TSUNAMI_WATCH:
            raise RuntimeError(
                f"confirm_observation requires stage TSUNAMI_WATCH (got {self._stage.value})"
            )
        if not description.strip() or not source.strip():
            raise ValueError("confirmed observation needs a non-empty description and source")
        self._stage = CascadeStage.TSUNAMI_THREAT
        self._record(
            "confirmed_observation",
            stage_after=self._stage,
            detail={"description": description, "source": source},
        )
        return self.state()

    def reset(self) -> None:
        """Return to QUIET, clearing screening/water-level state.

        The evidence chain is preserved (append-only audit trail); a reset
        is recorded as evidence.
        """
        self._stage = CascadeStage.QUIET
        self._screening = None
        self._water_level = None
        self._record("reset", stage_after=self._stage, detail={})

    def state(self) -> CascadeState:
        """Snapshot the current cascade state.

        Returns:
            CascadeState with the (copied) evidence chain.
        """
        return CascadeState(
            stage=self._stage.value,
            screening=self._screening,
            water_level=self._water_level,
            evidence_chain=list(self._evidence),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _record(self, kind: str, stage_after: CascadeStage, detail: dict[str, Any]) -> None:
        """Append one evidence record."""
        self._evidence.append({"kind": kind, "stage_after": stage_after.value, "detail": detail})

    def _supplementary_spectral(self, series: np.ndarray[Any, Any]) -> dict[str, Any] | None:
        """Attach TsunamiDetector deterministic resonance evidence if possible.

        Only the deterministic FFT outputs (resonance score, dominant
        frequencies) are recorded; the neural confidence is explicitly
        excluded because the network ships untrained.

        Args:
            series: Water-level series.

        Returns:
            Evidence dict, or None when torch/TsunamiDetector is
            unavailable (recorded honestly by the caller as absent).
        """
        try:
            from omni_mercury_engine.detectors.geological.disaster_detectors import (
                TsunamiDetector,
            )

            result = TsunamiDetector().predict_tsunami(series.astype(np.float32))
        except ImportError:
            self.logger.info(
                "TsunamiDetector unavailable (torch not installed); skipping "
                "supplementary spectral evidence."
            )
            return None
        return {
            "resonance_score": float(result.resonance_score),
            "dominant_frequencies_hz": [float(f) for f in result.dominant_frequencies],
            "note": (
                "deterministic FFT resonance evidence only; neural confidence "
                "excluded (analyser network untrained at construction)"
            ),
        }
