# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Derecho detector: published widespread-damaging-wind criteria over report series.

Implements the derecho identification criteria of Johns & Hirt (1987):
"Derechos: Widespread Convectively Induced Windstorms", *Wea. Forecasting*,
2, 32-49, with the swath-dimension revision proposed by Corfidi et al.
(2016): "A Proposed Revision to the Definition of 'Derecho'", *Bull. Amer.
Meteor. Soc.*, 97, 935-949, evaluated over a **supplied** series of
severe-wind reports (time, latitude, longitude, gust, optional F/EF damage
rating):

1. **Swath length**: major-axis extent >= 650 km (~400 mi) -- Corfidi et
   al. (2016) length criterion (Johns & Hirt used 400 km).
2. **Swath width**: cross-axis extent >= 100 km (~60 mi) -- Corfidi et al.
   (2016) width criterion.
3. **Report continuity**: no more than 3 h elapse between successive wind
   reports (Johns & Hirt 1987, criterion 4).
4. **Intensity anchors**: at least 3 reports of F1-intensity wind damage or
   measured gusts >= 33 m/s, mutually separated by >= 64 km (Johns & Hirt
   1987, criterion 3).
5. **Chronological progression**: the reports must progress along the swath
   axis with time, "either as a singular swath or as a series of swaths"
   (Johns & Hirt 1987, criterion 5).  Operationalized here as the Pearson
   correlation between report time and along-axis great-circle projection
   being >= 0.6 (our documented numeric cut for the published qualitative
   requirement).

All geometry uses great-circle math on the IUGG sphere (haversine
distances, cross-track / along-track projections about the major axis).

**Progressive vs serial classification** (Johns & Hirt 1987 storm types):
progressive derechos are produced by a single bowing MCS sweeping a
relatively narrow swath; serial derechos by an extensive squall line
(multiple bow echoes / LEWPs) sweeping a broad swath.  Operationalized
geometrically: swaths with width/length >= 0.4 are labelled ``serial``,
narrower swaths ``progressive`` -- our documented numeric cut for the
published qualitative distinction.

**Bow-echo precursor heuristic** (real inputs only): when the MCS motion
vector and the mean cloud-layer (850-300 hPa) wind are supplied, the
propagation component (storm motion minus mean wind; Corfidi 2003:
"Cold Pools and MCS Propagation: Forecasting the Motion of
Downwind-Developing MCSs", *Wea. Forecasting*, 18, 997-1017) is examined:
forward-propagating MCSs -- storm motion faster than the mean wind with a
downwind-directed propagation component (angle < 45 deg) -- favor bow-echo
organization and progressive derechos.  Missing inputs raise; nothing is
inferred from unavailable data.

No neural network; the criteria evaluate deterministically from supplied
real reports.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from omni_mercury_engine.utils.geo import (
    EARTH_RADIUS_KM,
    haversine_km,
    pairwise_haversine_km,
)

logger = logging.getLogger(__name__)

__all__ = ["DerechoDetector", "DerechoResult", "SwathGeometry", "WindReport"]

# --- Published criteria constants ---------------------------------------------
_MIN_SWATH_LENGTH_KM = 650.0  # Corfidi et al. (2016); ~400 statute miles
_MIN_SWATH_WIDTH_KM = 100.0  # Corfidi et al. (2016); ~60 statute miles
_MAX_REPORT_GAP_S = 3.0 * 3600.0  # Johns & Hirt (1987) criterion 4
_SIG_GUST_MS = 33.0  # Johns & Hirt (1987): F1-equivalent gust
_SIG_SEPARATION_KM = 64.0  # Johns & Hirt (1987): 40 mi separation
_MIN_SIG_REPORTS = 3  # Johns & Hirt (1987) criterion 3
_PROGRESSION_MIN_CORR = 0.6  # own numeric cut for J&H criterion 5
_SERIAL_ASPECT_RATIO = 0.4  # own numeric cut for J&H storm types
_BOW_ECHO_MAX_ANGLE_DEG = 45.0  # forward-propagation alignment cut (Corfidi 2003)


@dataclass(frozen=True)
class WindReport:
    """One severe-wind report.

    Attributes:
        time_s: Report epoch time (seconds).
        lat: Latitude (decimal degrees).
        lon: Longitude (decimal degrees).
        gust_ms: Measured/estimated gust (m/s); ``None`` when only a damage
            rating is available.
        f_scale: Optional F/EF damage rating (0-5) for damage-based reports.
    """

    time_s: float
    lat: float
    lon: float
    gust_ms: float | None = None
    f_scale: int | None = None


@dataclass(frozen=True)
class SwathGeometry:
    """Great-circle geometry of the damage swath.

    Attributes:
        length_km: Major-axis great-circle extent.
        width_km: Cross-axis extent (max positive minus max negative
            cross-track distance about the major-axis great circle).
        axis_start: (lat, lon) of the major-axis start endpoint (earlier
            report of the axis pair).
        axis_end: (lat, lon) of the major-axis end endpoint.
        axis_bearing_deg: Initial great-circle bearing start -> end.
        duration_h: Time span first report -> last report (hours).
    """

    length_km: float
    width_km: float
    axis_start: tuple[float, float]
    axis_end: tuple[float, float]
    axis_bearing_deg: float
    duration_h: float


@dataclass
class DerechoResult:
    """Outcome of the derecho criteria evaluation.

    Attributes:
        is_derecho: True when all five published criteria are satisfied.
        classification: ``progressive`` / ``serial`` when a derecho, else
            ``none``.
        geometry: Swath geometry.
        criteria: Per-criterion booleans (``length``, ``width``,
            ``continuity``, ``intensity_anchors``, ``progression``).
        max_report_gap_h: Largest gap between successive reports (hours).
        n_reports: Total reports evaluated.
        n_significant: Reports with gust >= 33 m/s or F1+ damage.
        n_significant_separated: Size of the mutually >= 64 km separated
            significant-report subset found (greedy).
        progression_correlation: Pearson r between time and along-axis
            projection.
        notes: Derivation notes.
    """

    is_derecho: bool
    classification: str
    geometry: SwathGeometry
    criteria: dict[str, bool]
    max_report_gap_h: float
    n_reports: int
    n_significant: int
    n_significant_separated: int
    progression_correlation: float
    notes: list[str] = field(default_factory=list)


def _initial_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing from point 1 to point 2, degrees in [0, 360)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    return math.degrees(math.atan2(x, y)) % 360.0


def _cross_track_km(
    lat: float, lon: float, start: tuple[float, float], bearing_deg: float
) -> float:
    """Signed cross-track distance of a point from a great circle.

    The great circle passes through ``start`` with initial bearing
    ``bearing_deg`` (standard navigation formula:
    d_xt = asin(sin(d13/R) * sin(theta13 - theta12)) * R).
    """
    d13 = haversine_km(start[0], start[1], lat, lon) / EARTH_RADIUS_KM
    theta13 = math.radians(_initial_bearing_deg(start[0], start[1], lat, lon))
    theta12 = math.radians(bearing_deg)
    return math.asin(math.sin(d13) * math.sin(theta13 - theta12)) * EARTH_RADIUS_KM


def _along_track_km(
    lat: float, lon: float, start: tuple[float, float], bearing_deg: float
) -> float:
    """Along-track distance of a point projected onto a great circle.

    The great circle passes through ``start`` with bearing ``bearing_deg``
    (d_at = acos(cos(d13/R) / cos(d_xt/R)) * R, signed by the along-axis
    bearing agreement).
    """
    d13 = haversine_km(start[0], start[1], lat, lon) / EARTH_RADIUS_KM
    dxt = _cross_track_km(lat, lon, start, bearing_deg) / EARTH_RADIUS_KM
    cos_ratio = math.cos(d13) / math.cos(dxt) if math.cos(dxt) != 0 else 1.0
    cos_ratio = min(1.0, max(-1.0, cos_ratio))
    dat = math.acos(cos_ratio) * EARTH_RADIUS_KM
    # Sign: negative when the point projects behind the start point.
    theta13 = _initial_bearing_deg(start[0], start[1], lat, lon)
    delta = (theta13 - bearing_deg + 180.0) % 360.0 - 180.0
    return dat if abs(delta) <= 90.0 else -dat


class DerechoDetector:
    """Derecho identification per Johns & Hirt (1987) / Corfidi et al. (2016).

    Deterministic evaluation of the published criteria over a supplied
    severe-wind report series; see the module docstring for citations.

    Example:
        >>> detector = DerechoDetector()
        >>> result = detector.evaluate(reports)  # doctest: +SKIP
    """

    #: Fixed fusion feature dimension (see :meth:`extract_features`).
    FEATURE_DIM = 16

    def __init__(
        self,
        min_length_km: float = _MIN_SWATH_LENGTH_KM,
        min_width_km: float = _MIN_SWATH_WIDTH_KM,
        max_gap_s: float = _MAX_REPORT_GAP_S,
        progression_min_corr: float = _PROGRESSION_MIN_CORR,
    ) -> None:
        """Initialize the detector.

        Args:
            min_length_km: Swath-length criterion (default 650 km, Corfidi
                et al. 2016).
            min_width_km: Swath-width criterion (default 100 km, Corfidi
                et al. 2016).
            max_gap_s: Maximum gap between successive reports (default 3 h,
                Johns & Hirt 1987).
            progression_min_corr: Minimum time/along-axis Pearson r for the
                chronological-progression criterion (default 0.6, our
                documented operationalization).
        """
        if min_length_km <= 0 or min_width_km <= 0 or max_gap_s <= 0:
            raise ValueError("Criteria thresholds must be positive.")
        if not 0.0 < progression_min_corr <= 1.0:
            raise ValueError("progression_min_corr must be in (0, 1].")
        self.min_length_km = float(min_length_km)
        self.min_width_km = float(min_width_km)
        self.max_gap_s = float(max_gap_s)
        self.progression_min_corr = float(progression_min_corr)

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_reports(reports: Any) -> list[WindReport]:
        """Validate and normalize the report series (fail-loud).

        Args:
            reports: Sequence of :class:`WindReport` or dicts with keys
                ``time_s`` / ``lat`` / ``lon`` and optional ``gust_ms`` /
                ``f_scale``.

        Returns:
            Time-sorted list of :class:`WindReport`.

        Raises:
            ValueError: On empty input, missing keys, non-finite
                coordinates/times, or out-of-range lat/lon.
        """
        if reports is None:
            raise ValueError("reports is None; supply a severe-wind report series.")
        items = list(reports)
        if len(items) < 2:
            raise ValueError(
                f"Need >= 2 wind reports to evaluate a swath; got {len(items)}. "
                "The derecho criteria are undefined for a single report."
            )
        out: list[WindReport] = []
        for i, item in enumerate(items):
            if isinstance(item, WindReport):
                rec = item
            elif isinstance(item, dict):
                missing = [k for k in ("time_s", "lat", "lon") if k not in item]
                if missing:
                    raise ValueError(f"Report {i} missing required keys: {missing}.")
                rec = WindReport(
                    time_s=float(item["time_s"]),
                    lat=float(item["lat"]),
                    lon=float(item["lon"]),
                    gust_ms=None if item.get("gust_ms") is None else float(item["gust_ms"]),
                    f_scale=None if item.get("f_scale") is None else int(item["f_scale"]),
                )
            else:
                raise ValueError(f"Report {i} has unsupported type {type(item).__name__}.")
            if not (
                math.isfinite(rec.time_s) and math.isfinite(rec.lat) and math.isfinite(rec.lon)
            ):
                raise ValueError(f"Report {i} has non-finite time/lat/lon: {rec}.")
            if not (-90.0 <= rec.lat <= 90.0 and -180.0 <= rec.lon <= 180.0):
                raise ValueError(f"Report {i} lat/lon out of range: ({rec.lat}, {rec.lon}).")
            if rec.gust_ms is not None and (not math.isfinite(rec.gust_ms) or rec.gust_ms < 0):
                raise ValueError(f"Report {i} gust_ms={rec.gust_ms} invalid.")
            out.append(rec)
        return sorted(out, key=lambda r: r.time_s)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    @staticmethod
    def _major_axis(reports: list[WindReport]) -> tuple[int, int, float]:
        """Return (i, j, distance_km) of the most-separated report pair.

        ``i`` is the chronologically earlier endpoint.
        """
        lats = np.array([r.lat for r in reports])
        lons = np.array([r.lon for r in reports])
        dist = pairwise_haversine_km(lats, lons)
        flat_idx = int(np.argmax(dist))
        raw_i, raw_j = np.unravel_index(flat_idx, dist.shape)
        i, j = int(raw_i), int(raw_j)
        d = float(dist[i, j])
        return (i, j, d) if reports[i].time_s <= reports[j].time_s else (j, i, d)

    def compute_swath_geometry(self, reports: list[WindReport]) -> SwathGeometry:
        """Compute the swath's great-circle geometry.

        The major axis is the most-separated report pair (earlier report =
        start); width is the cross-track span (max positive minus max
        negative signed cross-track distance) about the axis great circle.

        Args:
            reports: Time-sorted validated reports.

        Returns:
            :class:`SwathGeometry`.
        """
        i, j, length_km = self._major_axis(reports)
        start = (reports[i].lat, reports[i].lon)
        end = (reports[j].lat, reports[j].lon)
        bearing = _initial_bearing_deg(start[0], start[1], end[0], end[1])

        cross = [_cross_track_km(r.lat, r.lon, start, bearing) for r in reports]
        width_km = float(max(cross) - min(cross))
        duration_h = (reports[-1].time_s - reports[0].time_s) / 3600.0
        return SwathGeometry(
            length_km=float(length_km),
            width_km=width_km,
            axis_start=start,
            axis_end=end,
            axis_bearing_deg=float(bearing),
            duration_h=float(duration_h),
        )

    # ------------------------------------------------------------------
    # Criteria
    # ------------------------------------------------------------------

    @staticmethod
    def _significant_reports(reports: list[WindReport]) -> list[WindReport]:
        """Reports meeting the J&H intensity anchor (gust >= 33 m/s or F1+)."""
        return [
            r
            for r in reports
            if (r.gust_ms is not None and r.gust_ms >= _SIG_GUST_MS)
            or (r.f_scale is not None and r.f_scale >= 1)
        ]

    @staticmethod
    def _max_separated_subset(reports: list[WindReport], min_km: float) -> int:
        """Greedy size of a mutually >= ``min_km``-separated subset (time order)."""
        chosen: list[WindReport] = []
        for r in reports:
            if all(haversine_km(r.lat, r.lon, c.lat, c.lon) >= min_km for c in chosen):
                chosen.append(r)
        return len(chosen)

    def evaluate(self, reports: Any) -> DerechoResult:
        """Evaluate the published derecho criteria over a wind-report series.

        Args:
            reports: Sequence of :class:`WindReport` or dicts (see
                :meth:`_coerce_reports`).

        Returns:
            :class:`DerechoResult` with swath geometry, per-criterion
            booleans, and progressive/serial classification.

        Raises:
            ValueError: On invalid input series (fail-loud; see
                :meth:`_coerce_reports`).
        """
        recs = self._coerce_reports(reports)
        geometry = self.compute_swath_geometry(recs)

        times = np.array([r.time_s for r in recs])
        gaps = np.diff(times)
        max_gap_s = float(np.max(gaps)) if gaps.size else 0.0

        sig = self._significant_reports(recs)
        n_sig_separated = self._max_separated_subset(sig, _SIG_SEPARATION_KM)

        along = np.array(
            [
                _along_track_km(r.lat, r.lon, geometry.axis_start, geometry.axis_bearing_deg)
                for r in recs
            ]
        )
        if np.std(along) > 0 and np.std(times) > 0:
            corr = float(np.corrcoef(times, along)[0, 1])
        else:
            corr = 0.0

        criteria = {
            "length": geometry.length_km >= self.min_length_km,
            "width": geometry.width_km >= self.min_width_km,
            "continuity": max_gap_s <= self.max_gap_s,
            "intensity_anchors": n_sig_separated >= _MIN_SIG_REPORTS,
            "progression": corr >= self.progression_min_corr,
        }
        is_derecho = all(criteria.values())

        if is_derecho:
            aspect = geometry.width_km / geometry.length_km if geometry.length_km > 0 else 0.0
            classification = "serial" if aspect >= _SERIAL_ASPECT_RATIO else "progressive"
        else:
            classification = "none"

        notes = [
            f"length={geometry.length_km:.0f} km (>= {self.min_length_km:.0f} required)",
            f"width={geometry.width_km:.0f} km (>= {self.min_width_km:.0f} required)",
            f"max gap={max_gap_s / 3600.0:.2f} h (<= {self.max_gap_s / 3600.0:.1f} required)",
            f"significant anchors separated >= {_SIG_SEPARATION_KM:.0f} km: "
            f"{n_sig_separated} (>= {_MIN_SIG_REPORTS} required)",
            f"progression r={corr:.3f} (>= {self.progression_min_corr:.2f} required)",
        ]
        result = DerechoResult(
            is_derecho=is_derecho,
            classification=classification,
            geometry=geometry,
            criteria=criteria,
            max_report_gap_h=max_gap_s / 3600.0,
            n_reports=len(recs),
            n_significant=len(sig),
            n_significant_separated=n_sig_separated,
            progression_correlation=corr,
            notes=notes,
        )
        logger.info("Derecho evaluation: %s -> %s", criteria, classification)
        return result

    # ------------------------------------------------------------------
    # Bow-echo precursor (real inputs only)
    # ------------------------------------------------------------------

    @staticmethod
    def bow_echo_precursor(
        mcs_motion_u_ms: float,
        mcs_motion_v_ms: float,
        mean_wind_u_ms: float,
        mean_wind_v_ms: float,
    ) -> dict[str, Any]:
        """Forward-propagating MCS signature from motion vs mean wind.

        Decomposes MCS motion into advection (mean cloud-layer wind) plus
        propagation (Corfidi 2003).  A forward-propagating (downwind-
        developing) MCS -- storm speed exceeding the mean wind speed with
        the propagation component directed within 45 deg of the mean wind
        -- favors bow-echo organization and progressive derechos.

        Args:
            mcs_motion_u_ms: MCS motion east component (m/s).
            mcs_motion_v_ms: MCS motion north component (m/s).
            mean_wind_u_ms: 850-300 hPa mean wind east component (m/s).
            mean_wind_v_ms: 850-300 hPa mean wind north component (m/s).

        Returns:
            Dict with ``propagation_speed_ms``, ``propagation_angle_deg``
            (angle between propagation component and mean wind),
            ``storm_speed_ms``, ``mean_wind_speed_ms``, and
            ``forward_propagating`` (bool).

        Raises:
            ValueError: On non-finite inputs or a calm mean wind (angle
                undefined).
        """
        vals = {
            "mcs_motion_u_ms": mcs_motion_u_ms,
            "mcs_motion_v_ms": mcs_motion_v_ms,
            "mean_wind_u_ms": mean_wind_u_ms,
            "mean_wind_v_ms": mean_wind_v_ms,
        }
        for name, v in vals.items():
            if v is None or not math.isfinite(float(v)):
                raise ValueError(f"bow_echo_precursor input '{name}'={v!r} must be finite.")
        cu, cv = float(mcs_motion_u_ms), float(mcs_motion_v_ms)
        wu, wv = float(mean_wind_u_ms), float(mean_wind_v_ms)

        wind_speed = math.hypot(wu, wv)
        if wind_speed < 1e-6:
            raise ValueError(
                "Mean cloud-layer wind is calm; the propagation angle is undefined "
                "and no bow-echo inference is possible from these inputs."
            )
        pu, pv = cu - wu, cv - wv
        prop_speed = math.hypot(pu, pv)
        storm_speed = math.hypot(cu, cv)

        if prop_speed < 1e-6:
            angle_deg = 0.0
        else:
            cos_angle = (pu * wu + pv * wv) / (prop_speed * wind_speed)
            angle_deg = math.degrees(math.acos(min(1.0, max(-1.0, cos_angle))))

        forward = storm_speed > wind_speed and angle_deg < _BOW_ECHO_MAX_ANGLE_DEG
        return {
            "propagation_speed_ms": prop_speed,
            "propagation_angle_deg": angle_deg,
            "storm_speed_ms": storm_speed,
            "mean_wind_speed_ms": wind_speed,
            "forward_propagating": bool(forward),
        }

    # ------------------------------------------------------------------
    # Fusion interface
    # ------------------------------------------------------------------

    def extract_features(self, data: Any) -> torch.Tensor:
        """Extract a fixed-width feature vector for the fusion registry.

        A list/tuple of report dicts (or a dict with a ``"reports"`` key)
        runs the real criteria evaluation; an anonymous numeric array yields
        documented robust summary statistics only.

        Args:
            data: Report series, dict with ``reports``, or numeric array.

        Returns:
            ``torch.Tensor`` of shape ``(FEATURE_DIM,)``.

        Raises:
            ValueError: Propagated from :meth:`evaluate` on invalid series.
        """
        report_input = None
        if isinstance(data, dict) and "reports" in data:
            report_input = data["reports"]
        elif isinstance(data, (list, tuple)) and data and isinstance(data[0], (dict, WindReport)):
            report_input = data

        if report_input is not None:
            res = self.evaluate(report_input)
            features = [
                1.0 if res.is_derecho else 0.0,
                res.geometry.length_km / 1000.0,
                res.geometry.width_km / 1000.0,
                res.geometry.duration_h / 24.0,
                res.max_report_gap_h,
                float(res.n_reports) / 100.0,
                float(res.n_significant) / 10.0,
                float(res.n_significant_separated) / 10.0,
                res.progression_correlation,
                1.0 if res.classification == "progressive" else 0.0,
                1.0 if res.classification == "serial" else 0.0,
            ]
        else:
            arr = np.asarray(
                data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else data,
                dtype=np.float64,
            ).ravel()
            if arr.size == 0 or not np.all(np.isfinite(arr)):
                raise ValueError("extract_features requires a non-empty, finite numeric array.")
            q25, q75 = np.percentile(arr, [25, 75])
            features = [
                float(np.mean(arr)),
                float(np.std(arr)),
                float(np.min(arr)),
                float(np.max(arr)),
                float(np.median(arr)),
                float(q75 - q25),
            ]

        features = features[: self.FEATURE_DIM]
        features.extend([0.0] * (self.FEATURE_DIM - len(features)))
        return torch.tensor(features, dtype=torch.float32)
