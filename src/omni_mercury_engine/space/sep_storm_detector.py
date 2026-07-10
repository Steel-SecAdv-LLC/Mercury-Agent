# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Solar energetic particle (SEP) / radiation-storm detection from GOES proton flux.

Consumes NOAA SWPC integral proton flux (``>=10 MeV`` and optionally
``>=100 MeV``, in pfu = protons cm^-2 s^-1 sr^-1) from the
``json/goes/primary/integral-protons-7-day.json`` product
(:class:`omni_mercury_engine.data_sources.space_weather.NOAASWPCSource` with
``SWPCProduct.INTEGRAL_PROTONS``) and produces:

* **NOAA S-scale classification** — S1..S5 at 10 / 100 / 1,000 / 10,000 /
  100,000 pfu of >=10 MeV integral flux, per the NOAA Space Weather Scales
  ("Solar Radiation Storms", https://www.swpc.noaa.gov/noaa-scales-explanation).
* **Onset detection** — threshold crossing with persistence. NOAA defines a
  proton event start as the first of three consecutive 5-minute readings at
  or above 10 pfu (SWPC Solar Proton Events definition,
  https://www.swpc.noaa.gov/products/goes-proton-flux); the threshold and
  persistence count are parameterised with those defaults.
* **Peak-flux estimate** — maximum observed flux and its timestamp.
* **Well-connected-flare precursor heuristic** — given a DONKI flare record
  (class + source location), flags magnetically well-connected western
  flares. The Parker-spiral field line reaching Earth for a nominal
  ~400 km/s solar wind is rooted near W50-W60; SEP events observed at Earth
  are strongly organised by source longitude, with prompt, hard-spectrum
  events favoured for sources in roughly W20-W85 (Reames 1999, Space Sci.
  Rev. 90, 413; Cane & Lario 2006, Space Sci. Rev. 123, 45). This is a
  precursor *advisory only* — it never fabricates flux.

Physics-only module: no neural network. Fail-loud contract: empty or
non-finite flux series, mismatched lengths, and negative fluxes raise
``ValueError``.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import pairwise
from typing import Any

logger = logging.getLogger(__name__)

#: NOAA S-scale thresholds in pfu of >=10 MeV integral proton flux
#: (NOAA Space Weather Scales, "Solar Radiation Storms").
S_SCALE_THRESHOLDS_PFU: tuple[tuple[float, str], ...] = (
    (1e5, "S5"),
    (1e4, "S4"),
    (1e3, "S3"),
    (1e2, "S2"),
    (1e1, "S1"),
)

#: NOAA proton-event start threshold (pfu, >=10 MeV).
SEP_EVENT_THRESHOLD_PFU: float = 10.0
#: NOAA persistence requirement: 3 consecutive 5-minute readings.
SEP_EVENT_PERSISTENCE_SAMPLES: int = 3

#: SWPC additionally tracks >=100 MeV events at the 1 pfu level (relevant
#: to aviation dose and EVA constraints).
HUNDRED_MEV_EVENT_THRESHOLD_PFU: float = 1.0

#: Magnetically well-connected heliolongitude band (degrees west) for
#: prompt SEP arrival at Earth (Reames 1999; Cane & Lario 2006).
WELL_CONNECTED_WEST_LONGITUDE_RANGE: tuple[float, float] = (20.0, 85.0)

_SOURCE_LOCATION_RE = re.compile(r"^([NS])(\d{1,2})([EW])(\d{1,3})$")


def classify_s_scale(flux_ge10mev_pfu: float) -> str:
    """Classify >=10 MeV integral proton flux on the NOAA S-scale.

    Thresholds (NOAA Space Weather Scales): S1 at 10 pfu, S2 at 100, S3 at
    1,000, S4 at 10,000, S5 at 100,000. Below 10 pfu is "S0" (no storm).

    Args:
        flux_ge10mev_pfu: Integral >=10 MeV proton flux in pfu.

    Returns:
        One of "S0".."S5".

    Raises:
        ValueError: If the flux is negative or non-finite.
    """
    if not math.isfinite(flux_ge10mev_pfu) or flux_ge10mev_pfu < 0.0:
        raise ValueError(f"Proton flux must be finite and non-negative; got {flux_ge10mev_pfu!r}.")
    for threshold, scale in S_SCALE_THRESHOLDS_PFU:
        if flux_ge10mev_pfu >= threshold:
            return scale
    return "S0"


def parse_flare_longitude(source_location: str) -> float:
    """Parse a solar source location like ``S15W45`` to signed longitude.

    Args:
        source_location: DONKI-style location string (e.g. ``N25W60``).

    Returns:
        Longitude in degrees, west positive (matching the well-connected
        band convention used here).

    Raises:
        ValueError: If the location string cannot be parsed.
    """
    match = _SOURCE_LOCATION_RE.match(source_location.strip().upper())
    if not match:
        raise ValueError(
            f"Cannot parse solar source location {source_location!r} (want e.g. 'S15W45')."
        )
    _ns, _lat, ew, lon = match.groups()
    return float(lon) if ew == "W" else -float(lon)


def assess_flare_connectivity(
    class_type: str,
    source_location: str,
    min_class: str = "M",
) -> dict[str, Any]:
    """Assess whether a flare is a well-connected SEP precursor.

    Magnetic-connectivity physics: energetic protons stream along the
    interplanetary magnetic field; for the average ~400 km/s solar wind the
    Parker spiral connects Earth to ~W50-W60 on the Sun, so flares between
    roughly W20 and W85 deliver prompt, intense proton events at Earth,
    while eastern events arrive delayed and attenuated (Reames 1999, Space
    Sci. Rev. 90, 413; Cane & Lario 2006, Space Sci. Rev. 123, 45).

    Args:
        class_type: GOES flare class string (e.g. ``X2.1``, ``M5.4``).
        source_location: Solar source location (e.g. ``N25W60``).
        min_class: Minimum flare class letter to consider (default M).

    Returns:
        Dict with ``well_connected`` (bool), ``west_longitude_deg``,
        ``flare_class``, and ``reason``.

    Raises:
        ValueError: On unparseable class or location strings.
    """
    class_type = class_type.strip().upper()
    if not class_type or class_type[0] not in "ABCMX":
        raise ValueError(f"Cannot parse GOES flare class {class_type!r}.")
    class_order = "ABCMX"
    energetic = class_order.index(class_type[0]) >= class_order.index(min_class.upper())
    west_lon = parse_flare_longitude(source_location)

    lo, hi = WELL_CONNECTED_WEST_LONGITUDE_RANGE
    connected_geometry = lo <= west_lon <= hi
    well_connected = energetic and connected_geometry

    if not energetic:
        reason = f"flare class {class_type} below {min_class}-class precursor threshold"
    elif not connected_geometry:
        reason = (
            f"source at W{west_lon:.0f}" if west_lon >= 0 else f"source at E{-west_lon:.0f}"
        ) + f" outside the well-connected W{lo:.0f}-W{hi:.0f} band"
    else:
        reason = (
            f"{class_type} flare at W{west_lon:.0f} lies in the magnetically "
            f"well-connected W{lo:.0f}-W{hi:.0f} band (Parker-spiral footpoint "
            "~W50-W60 for 400 km/s wind)"
        )
    return {
        "well_connected": well_connected,
        "west_longitude_deg": west_lon,
        "flare_class": class_type,
        "reason": reason,
    }


@dataclass
class SEPStormAssessment:
    """Result of an SEP / radiation-storm assessment.

    Attributes:
        s_scale: NOAA S-scale at peak >=10 MeV flux ("S0".."S5").
        peak_flux_10mev_pfu: Maximum observed >=10 MeV flux, pfu.
        peak_time_10mev: Timestamp of the >=10 MeV peak.
        onset_detected: Whether a NOAA-style event onset occurred.
        onset_time: Time of the first sample of the persistence run that
            crossed the event threshold (None if no onset).
        event_active: Whether the latest sample is still at/above threshold.
        threshold_pfu: Event threshold used (default 10 pfu).
        persistence_samples: Consecutive samples required for onset.
        peak_flux_100mev_pfu: Maximum >=100 MeV flux if supplied, else None.
        hundred_mev_event: Whether >=100 MeV flux reached 1 pfu (None when
            no >=100 MeV series supplied).
        n_samples: Number of >=10 MeV samples assessed.
        precursor: Optional well-connected-flare advisory dict.
        precursor_advisory: Human-readable advisory (empty if none).
    """

    s_scale: str
    peak_flux_10mev_pfu: float
    peak_time_10mev: datetime
    onset_detected: bool
    onset_time: datetime | None
    event_active: bool
    threshold_pfu: float
    persistence_samples: int
    peak_flux_100mev_pfu: float | None = None
    hundred_mev_event: bool | None = None
    n_samples: int = 0
    precursor: dict[str, Any] | None = None
    precursor_advisory: str = ""
    notes: list[str] = field(default_factory=list)


class SEPStormDetector:
    """NOAA S-scale SEP storm detector over GOES integral proton flux.

    Physics-based (threshold + persistence per the SWPC proton-event
    definition); no neural network. Works directly on
    ``SWPCProduct.INTEGRAL_PROTONS`` rows or on plain time/flux arrays.

    Example:
        >>> detector = SEPStormDetector()
        >>> result = detector.assess(times, flux_ge10mev)
        >>> result.s_scale, result.onset_detected
    """

    def __init__(
        self,
        event_threshold_pfu: float = SEP_EVENT_THRESHOLD_PFU,
        persistence_samples: int = SEP_EVENT_PERSISTENCE_SAMPLES,
    ) -> None:
        """Initialize the detector.

        Args:
            event_threshold_pfu: >=10 MeV onset threshold, pfu (NOAA: 10).
            persistence_samples: Consecutive samples required at/above the
                threshold to declare onset (NOAA: 3 five-minute readings).

        Raises:
            ValueError: On non-positive threshold or persistence.
        """
        if not math.isfinite(event_threshold_pfu) or event_threshold_pfu <= 0:
            raise ValueError(f"event_threshold_pfu must be positive; got {event_threshold_pfu!r}.")
        if persistence_samples < 1:
            raise ValueError(f"persistence_samples must be >= 1; got {persistence_samples!r}.")
        self.event_threshold_pfu = float(event_threshold_pfu)
        self.persistence_samples = int(persistence_samples)
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Core assessment
    # ------------------------------------------------------------------

    def assess(
        self,
        times: list[datetime],
        flux_ge10mev_pfu: list[float],
        flux_ge100mev_pfu: list[float] | None = None,
        flare: dict[str, Any] | None = None,
    ) -> SEPStormAssessment:
        """Assess an integral proton-flux time series.

        Args:
            times: Sample timestamps (timezone-aware, ascending).
            flux_ge10mev_pfu: >=10 MeV integral flux per sample, pfu.
            flux_ge100mev_pfu: Optional >=100 MeV flux series. If provided
                it must align 1:1 with ``times``.
            flare: Optional DONKI flare record (``classType`` +
                ``sourceLocation``) for the well-connected precursor
                advisory.

        Returns:
            Full SEP storm assessment.

        Raises:
            ValueError: On empty series, length mismatch, unsorted or naive
                timestamps, or negative / non-finite flux values.
        """
        if not times or not flux_ge10mev_pfu:
            raise ValueError("Empty proton-flux series: refusing to assess (no data is not S0).")
        if len(times) != len(flux_ge10mev_pfu):
            raise ValueError(
                f"times ({len(times)}) and >=10 MeV flux ({len(flux_ge10mev_pfu)}) "
                "lengths differ."
            )
        if flux_ge100mev_pfu is not None and len(flux_ge100mev_pfu) != len(times):
            raise ValueError(
                f"times ({len(times)}) and >=100 MeV flux ({len(flux_ge100mev_pfu)}) "
                "lengths differ."
            )
        for ts in times:
            if ts.tzinfo is None:
                raise ValueError(f"Timestamp {ts!r} is naive; timestamps must be timezone-aware.")
        if any(t2 < t1 for t1, t2 in pairwise(times)):
            raise ValueError("Timestamps must be ascending.")
        for label, series in (
            (">=10 MeV", flux_ge10mev_pfu),
            (">=100 MeV", flux_ge100mev_pfu or []),
        ):
            for value in series:
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(
                        f"Non-finite or negative {label} flux value {value!r}; "
                        "refusing to assess corrupted data."
                    )

        peak_idx = max(range(len(flux_ge10mev_pfu)), key=flux_ge10mev_pfu.__getitem__)
        peak_flux = flux_ge10mev_pfu[peak_idx]
        peak_time = times[peak_idx]
        s_scale = classify_s_scale(peak_flux)

        onset_idx = self._find_onset_index(flux_ge10mev_pfu)
        onset_detected = onset_idx is not None
        onset_time = times[onset_idx] if onset_idx is not None else None
        event_active = flux_ge10mev_pfu[-1] >= self.event_threshold_pfu

        peak_100: float | None = None
        hundred_event: bool | None = None
        if flux_ge100mev_pfu is not None:
            peak_100 = max(flux_ge100mev_pfu)
            hundred_event = peak_100 >= HUNDRED_MEV_EVENT_THRESHOLD_PFU

        precursor: dict[str, Any] | None = None
        advisory = ""
        notes: list[str] = []
        if flare is not None:
            class_type = flare.get("classType")
            source_location = flare.get("sourceLocation")
            if not class_type or not source_location:
                raise ValueError(
                    "Flare record supplied without classType/sourceLocation; "
                    "cannot assess magnetic connectivity from incomplete data."
                )
            precursor = assess_flare_connectivity(str(class_type), str(source_location))
            if precursor["well_connected"] and not onset_detected:
                advisory = (
                    f"Well-connected {precursor['flare_class']} flare "
                    f"(W{precursor['west_longitude_deg']:.0f}): elevated SEP "
                    "likelihood in the next hours; monitoring advised. "
                    "Advisory only — no proton enhancement observed yet."
                )
            elif precursor["well_connected"]:
                advisory = (
                    f"Well-connected {precursor['flare_class']} flare consistent "
                    "with the observed proton onset."
                )

        assessment = SEPStormAssessment(
            s_scale=s_scale,
            peak_flux_10mev_pfu=peak_flux,
            peak_time_10mev=peak_time,
            onset_detected=onset_detected,
            onset_time=onset_time,
            event_active=event_active,
            threshold_pfu=self.event_threshold_pfu,
            persistence_samples=self.persistence_samples,
            peak_flux_100mev_pfu=peak_100,
            hundred_mev_event=hundred_event,
            n_samples=len(times),
            precursor=precursor,
            precursor_advisory=advisory,
            notes=notes,
        )
        self.logger.info(
            "SEP assessment: %s (peak %.3g pfu at %s; onset=%s, active=%s)",
            s_scale,
            peak_flux,
            peak_time.isoformat(),
            onset_detected,
            event_active,
        )
        return assessment

    def _find_onset_index(self, flux: list[float]) -> int | None:
        """First index of a persistence run at/above the event threshold."""
        run = 0
        for idx, value in enumerate(flux):
            if value >= self.event_threshold_pfu:
                run += 1
                if run >= self.persistence_samples:
                    return idx - self.persistence_samples + 1
            else:
                run = 0
        return None

    # ------------------------------------------------------------------
    # SWPC product ingestion
    # ------------------------------------------------------------------

    def assess_from_swpc(
        self,
        rows: list[dict[str, Any]],
        flare: dict[str, Any] | None = None,
    ) -> SEPStormAssessment:
        """Assess raw NOAA SWPC integral-proton product rows.

        Args:
            rows: Parsed JSON rows of the SWPC
                ``json/goes/primary/integral-protons-7-day.json`` product;
                each row carries ``time_tag``, ``flux`` and ``energy``
                (e.g. ``>=10 MeV``).
            flare: Optional DONKI flare record for the precursor advisory.

        Returns:
            Full SEP storm assessment.

        Raises:
            ValueError: If no ``>=10 MeV`` rows are present or rows are
                malformed.
        """
        by_energy: dict[str, list[tuple[datetime, float]]] = {">=10 MeV": [], ">=100 MeV": []}
        for row in rows:
            energy = row.get("energy")
            if energy not in by_energy:
                continue
            time_tag = row.get("time_tag")
            flux = row.get("flux")
            if time_tag is None or flux is None:
                raise ValueError(f"Malformed SWPC integral-proton row (missing time/flux): {row!r}")
            timestamp = datetime.fromisoformat(str(time_tag).replace("Z", "+00:00"))
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=UTC)
            by_energy[energy].append((timestamp, float(flux)))

        ten = sorted(by_energy[">=10 MeV"])
        if not ten:
            raise ValueError(
                "SWPC integral-proton payload contains no '>=10 MeV' rows; "
                "cannot classify a radiation storm without the S-scale channel."
            )
        times = [t for t, _ in ten]
        flux10 = [f for _, f in ten]

        hundred = sorted(by_energy[">=100 MeV"])
        flux100: list[float] | None = None
        if (
            hundred
            and len(hundred) == len(ten)
            and all(ht == tt for (ht, _), (tt, _) in zip(hundred, ten, strict=True))
        ):
            flux100 = [f for _, f in hundred]
        elif hundred:
            # Timestamps disagree between channels: assess >=100 MeV
            # separately rather than silently misaligning samples.
            flux100 = None

        assessment = self.assess(times, flux10, flux_ge100mev_pfu=flux100, flare=flare)
        if hundred and flux100 is None:
            peak_100 = max(f for _, f in hundred)
            assessment.peak_flux_100mev_pfu = peak_100
            assessment.hundred_mev_event = peak_100 >= HUNDRED_MEV_EVENT_THRESHOLD_PFU
            assessment.notes.append(
                ">=100 MeV channel timestamps did not align 1:1 with the "
                ">=10 MeV channel; peak assessed independently."
            )
        return assessment


__all__ = [
    "HUNDRED_MEV_EVENT_THRESHOLD_PFU",
    "SEP_EVENT_PERSISTENCE_SAMPLES",
    "SEP_EVENT_THRESHOLD_PFU",
    "S_SCALE_THRESHOLDS_PFU",
    "WELL_CONNECTED_WEST_LONGITUDE_RANGE",
    "SEPStormAssessment",
    "SEPStormDetector",
    "assess_flare_connectivity",
    "classify_s_scale",
    "parse_flare_longitude",
]
