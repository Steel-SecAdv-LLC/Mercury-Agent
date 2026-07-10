# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Lightning Detector - flash-rate anomaly detection (lightning jump algorithm).

Implements the "2-sigma" lightning jump algorithm of Schultz, Petersen &
Carey (2009), "Preliminary development and evaluation of lightning jump
algorithms for the real-time detection of severe weather", *J. Appl.
Meteor. Climatol.* 48, 2543-2563 (see also Schultz et al. 2011, *Wea.
Forecasting* 26, 744-755):

1.  The total flash rate is averaged over consecutive 2-minute periods,
    FR(t) in flashes/min.
2.  The rate of change DFRDT(t) = (FR(t) - FR(t-1)) / dt is computed from
    consecutive 2-minute average rates (flashes/min^2).
3.  The standard deviation sigma of DFRDT over the preceding 12 minutes
    (the five DFRDT values ending at t-1) provides the jump threshold.
4.  A lightning jump is declared when DFRDT(t) >= 2 * sigma AND the current
    flash rate is at least 10 flashes/min (the activation threshold that
    suppresses noisy low-rate cells).

A detected jump is a severe-weather precursor: the cited studies report
mean lead times of roughly 20-23 minutes between the jump and the onset of
severe weather at the ground (Schultz et al. 2009 report ~23 min for the
2-sigma variant; the larger 2011 sample gives 20.65 min).  The detector
exposes that literature value as an *expected* lead-time estimate - it is a
climatological figure from the cited papers, not a per-storm computation.

Cell clustering uses simple spatial windowing: flashes are binned onto a
fixed lat/lon grid (default 0.15 deg, roughly 15 km at mid-latitudes) and
each occupied bin is reported as a cell with its centroid and flash count.
This is deliberately simple (no tracking, no merging across bins) and is
documented as such; it is adequate for associating flash bursts with NWS
warning polygons, not for storm-cell lifecycle analysis.

NWS alert cross-check: cells can be matched against active NWS convective
products (Severe Thunderstorm / Tornado Warnings fetched via
``NWSWeatherAlertsSource``) by great-circle distance to the alert polygon
centroid.

Pure statistics core: works untrained, no neural networks, fails loudly on
malformed input.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

#: Mean earth radius (km) for the haversine distance.
_EARTH_RADIUS_KM: float = 6371.0

#: Literature mean lead time (minutes) from lightning jump to severe
#: weather: ~23 min in Schultz et al. (2009), 20.65 min in Schultz et al.
#: (2011).  Exposed as a climatological expectation, not a per-storm value.
EXPECTED_LEAD_TIME_MIN: float = 20.65

#: NWS convective products relevant to the cross-check.
_CONVECTIVE_EVENTS: frozenset[str] = frozenset(
    {"severe thunderstorm warning", "tornado warning", "special weather statement"}
)


@dataclass
class LightningJump:
    """One detected lightning jump (2-sigma exceedance)."""

    bin_index: int
    time_s: float
    flash_rate_per_min: float
    dfrdt_per_min2: float
    sigma_per_min2: float
    expected_lead_time_min: float = EXPECTED_LEAD_TIME_MIN


@dataclass
class LightningJumpResult:
    """Result of lightning-jump analysis over a flash-time series.

    Attributes:
        jump_detected: True when at least one jump was found.
        jumps: All detected jumps.
        bin_starts_s: Start time (s) of each 2-minute bin.
        flash_rates_per_min: Average flash rate per bin (flashes/min).
        severe_weather_precursor: True when a jump was found - per the
            cited studies severe weather is expected within roughly the
            following 45 minutes.
    """

    jump_detected: bool
    jumps: list[LightningJump] = field(default_factory=list)
    bin_starts_s: np.ndarray | None = None
    flash_rates_per_min: np.ndarray | None = None
    severe_weather_precursor: bool = False
    warning_actions: list[str] = field(default_factory=list)


@dataclass
class LightningCell:
    """One spatial cell (grid bin) of flash activity."""

    lat_bin: int
    lon_bin: int
    centroid_lat: float
    centroid_lon: float
    flash_count: int


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in kilometres.

    Args:
        lat1: Latitude of the first point (deg).
        lon1: Longitude of the first point (deg).
        lat2: Latitude of the second point (deg).
        lon2: Longitude of the second point (deg).

    Returns:
        Distance in kilometres.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    return 2.0 * _EARTH_RADIUS_KM * math.asin(math.sqrt(a))


class LightningDetector:
    """Flash-rate anomaly detector implementing the 2-sigma lightning jump.

    Configuration mirrors Schultz et al. (2009): 2-minute averaging bins, a
    2-sigma DFRDT threshold over the preceding 12 minutes, and a
    10 flashes/min activation rate.
    """

    def __init__(
        self,
        bin_seconds: float = 120.0,
        sigma_multiplier: float = 2.0,
        sigma_history_bins: int = 6,
        activation_rate_per_min: float = 10.0,
        cell_size_deg: float = 0.15,
    ) -> None:
        """Initialize the detector.

        Args:
            bin_seconds: Flash-rate averaging period (120 s in the paper).
            sigma_multiplier: DFRDT threshold multiplier (2.0 = the
                "2-sigma" variant).
            sigma_history_bins: Number of trailing rate bins forming the
                sigma history (6 bins x 2 min = the paper's 12 minutes,
                yielding 5 DFRDT values).
            activation_rate_per_min: Minimum current flash rate for a jump
                to be considered (10 flashes/min in the paper).
            cell_size_deg: Grid size for simple spatial windowing.

        Raises:
            ValueError: On non-positive configuration values.
        """
        if bin_seconds <= 0.0:
            raise ValueError(f"bin_seconds must be > 0, got {bin_seconds}")
        if sigma_multiplier <= 0.0:
            raise ValueError(f"sigma_multiplier must be > 0, got {sigma_multiplier}")
        if sigma_history_bins < 3:
            raise ValueError(
                f"sigma_history_bins must be >= 3 (>= 2 DFRDT values), got {sigma_history_bins}"
            )
        if activation_rate_per_min < 0.0:
            raise ValueError(f"activation_rate_per_min must be >= 0, got {activation_rate_per_min}")
        if cell_size_deg <= 0.0:
            raise ValueError(f"cell_size_deg must be > 0, got {cell_size_deg}")
        self.bin_seconds = bin_seconds
        self.sigma_multiplier = sigma_multiplier
        self.sigma_history_bins = sigma_history_bins
        self.activation_rate_per_min = activation_rate_per_min
        self.cell_size_deg = cell_size_deg
        self.logger = logging.getLogger(__name__)

    def compute_flash_rates(self, flash_times_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Average the total flash rate over consecutive bins.

        Args:
            flash_times_s: 1-D array of flash timestamps in seconds
                (any epoch; only differences matter).  Need not be sorted.

        Returns:
            Tuple ``(bin_starts_s, rates_per_min)``.

        Raises:
            ValueError: If the series is empty, non-finite, or spans fewer
                than two averaging bins (no rate change is computable).
        """
        times = np.asarray(flash_times_s, dtype=np.float64)
        if times.ndim != 1 or times.size == 0:
            raise ValueError("flash_times_s must be a non-empty 1-D array")
        if not np.all(np.isfinite(times)):
            raise ValueError("flash_times_s contains non-finite values")
        times = np.sort(times)

        span = float(times[-1] - times[0])
        n_bins = max(int(span // self.bin_seconds) + 1, 1)
        if n_bins < 2:
            raise ValueError(
                f"flash series spans {span:.0f} s, less than two "
                f"{self.bin_seconds:.0f}-s bins; flash-rate trend undefined"
            )
        edges = times[0] + self.bin_seconds * np.arange(n_bins + 1)
        counts, _ = np.histogram(times, bins=edges)
        rates = counts / (self.bin_seconds / 60.0)
        return edges[:-1], rates.astype(np.float64)

    def detect_lightning_jumps(self, flash_times_s: np.ndarray) -> LightningJumpResult:
        """Run the 2-sigma lightning jump algorithm on a flash-time series.

        Args:
            flash_times_s: 1-D array of flash timestamps in seconds.

        Returns:
            A :class:`LightningJumpResult`.

        Raises:
            ValueError: Propagated from :meth:`compute_flash_rates`, or if
                the series is too short to form the sigma history (needs
                ``sigma_history_bins + 1`` bins).
        """
        bin_starts, rates = self.compute_flash_rates(flash_times_s)
        dt_min = self.bin_seconds / 60.0
        dfrdt = np.diff(rates) / dt_min  # flashes/min^2, aligned to bins 1..n-1

        min_bins = self.sigma_history_bins + 1
        if rates.size < min_bins:
            raise ValueError(
                f"need >= {min_bins} rate bins ({min_bins * self.bin_seconds:.0f} s "
                f"of data) to form the sigma history, got {rates.size}"
            )

        jumps: list[LightningJump] = []
        n_hist_dfrdt = self.sigma_history_bins - 1  # DFRDT values in history
        for k in range(n_hist_dfrdt, dfrdt.size):
            history = dfrdt[k - n_hist_dfrdt : k]
            sigma = float(np.std(history, ddof=1))
            if sigma <= 0.0:
                # A flat history gives a zero threshold; declaring a jump on
                # any positive change would be spurious.  Documented choice:
                # skip the evaluation instead of triggering.
                continue
            current_rate = float(rates[k + 1])
            if (
                float(dfrdt[k]) >= self.sigma_multiplier * sigma
                and current_rate >= self.activation_rate_per_min
            ):
                jumps.append(
                    LightningJump(
                        bin_index=k + 1,
                        time_s=float(bin_starts[k + 1]),
                        flash_rate_per_min=current_rate,
                        dfrdt_per_min2=float(dfrdt[k]),
                        sigma_per_min2=sigma,
                    )
                )

        detected = bool(jumps)
        result = LightningJumpResult(
            jump_detected=detected,
            jumps=jumps,
            bin_starts_s=bin_starts,
            flash_rates_per_min=rates,
            severe_weather_precursor=detected,
            warning_actions=self._generate_warnings(jumps),
        )
        self.logger.info("Lightning jump analysis: %d bin(s), %d jump(s)", rates.size, len(jumps))
        return result

    def cluster_cells(
        self,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
    ) -> list[LightningCell]:
        """Cluster flash locations by simple spatial windowing (grid binning).

        Flashes are assigned to fixed lat/lon bins of ``cell_size_deg``;
        each occupied bin becomes one cell with its flash centroid and
        count.  No merging or tracking is performed (documented
        limitation).

        Args:
            latitudes: Flash latitudes (deg), 1-D.
            longitudes: Flash longitudes (deg), 1-D, same length.

        Returns:
            Cells sorted by descending flash count.

        Raises:
            ValueError: On shape mismatch, non-finite values, or
                out-of-range coordinates.
        """
        lats = np.asarray(latitudes, dtype=np.float64)
        lons = np.asarray(longitudes, dtype=np.float64)
        if lats.ndim != 1 or lats.shape != lons.shape:
            raise ValueError(
                f"latitudes/longitudes must be 1-D of equal length, got "
                f"{lats.shape} / {lons.shape}"
            )
        if lats.size == 0:
            raise ValueError("no flash locations supplied")
        if not (np.all(np.isfinite(lats)) and np.all(np.isfinite(lons))):
            raise ValueError("flash coordinates contain non-finite values")
        if np.any(np.abs(lats) > 90.0) or np.any(np.abs(lons) > 180.0):
            raise ValueError("flash coordinates outside [-90, 90] / [-180, 180]")

        lat_bins = np.floor(lats / self.cell_size_deg).astype(np.int64)
        lon_bins = np.floor(lons / self.cell_size_deg).astype(np.int64)
        cells: dict[tuple[int, int], list[int]] = {}
        for idx, key in enumerate(zip(lat_bins.tolist(), lon_bins.tolist())):
            cells.setdefault(key, []).append(idx)

        out = [
            LightningCell(
                lat_bin=key[0],
                lon_bin=key[1],
                centroid_lat=float(np.mean(lats[members])),
                centroid_lon=float(np.mean(lons[members])),
                flash_count=len(members),
            )
            for key, members in cells.items()
        ]
        out.sort(key=lambda c: c.flash_count, reverse=True)
        return out

    @staticmethod
    def cross_check_alerts(
        cells: list[LightningCell],
        alerts: list[Any],
        radius_km: float = 50.0,
    ) -> dict[int, list[str]]:
        """Match cells to active NWS convective warnings by distance.

        Accepts the ``DataPoint`` objects produced by
        ``NWSWeatherAlertsSource`` (event under ``point.data["event"]``,
        polygon centroid under ``point.location``) or plain dicts with
        ``"event"`` and ``"location"`` keys.

        Args:
            cells: Cells from :meth:`cluster_cells`.
            alerts: Alert records.
            radius_km: Match radius (km) between cell centroid and alert
                polygon centroid.

        Returns:
            Mapping cell-list index -> list of matched alert event names
            (only convective products are considered).
        """
        matches: dict[int, list[str]] = {}
        for alert in alerts:
            payload = getattr(alert, "data", None)
            location = getattr(alert, "location", None)
            if payload is None and isinstance(alert, dict):
                payload = alert
                location = alert.get("location")
            if not isinstance(payload, dict) or location is None:
                continue
            event = str(payload.get("event", ""))
            if event.lower() not in _CONVECTIVE_EVENTS:
                continue
            alert_lat, alert_lon = float(location[0]), float(location[1])
            for cell_idx, cell in enumerate(cells):
                dist = haversine_km(cell.centroid_lat, cell.centroid_lon, alert_lat, alert_lon)
                if dist <= radius_km:
                    matches.setdefault(cell_idx, []).append(event)
        return matches

    @staticmethod
    def _generate_warnings(jumps: list[LightningJump]) -> list[str]:
        """Generate advisory strings."""
        if not jumps:
            return []
        return [
            (
                "LIGHTNING JUMP: rapid flash-rate increase detected - severe "
                f"weather possible within ~45 min (literature mean lead time "
                f"~{EXPECTED_LEAD_TIME_MIN:.0f} min; Schultz et al. 2009, 2011)"
            ),
            "Cross-check radar and active NWS warnings for the affected cell",
        ]

    def extract_features(self, data: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Extract a fixed 20-dim feature vector for ML fusion.

        Treats the input as a flash-rate-like series (first axis time) and
        emits rate statistics plus normalized max rate-of-change.

        Args:
            data: Input array or tensor.

        Returns:
            Feature tensor of shape (20,).
        """
        if isinstance(data, torch.Tensor):
            arr: np.ndarray = data.detach().cpu().numpy()
        else:
            arr = np.asarray(data, dtype=np.float64)
        flat = arr.astype(np.float64).flatten()
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            return torch.zeros(20, dtype=torch.float32)

        features: list[float] = [
            float(np.mean(flat)),
            float(np.std(flat)),
            float(np.min(flat)),
            float(np.max(flat)),
            float(np.median(flat)),
        ]
        if flat.size >= 2:
            diffs = np.diff(flat)
            sigma = float(np.std(diffs, ddof=1)) if diffs.size >= 2 else 0.0
            features.extend(
                [
                    float(np.max(diffs)),
                    float(np.max(diffs) / sigma) if sigma > 0.0 else 0.0,
                ]
            )
        while len(features) < 20:
            features.append(0.0)
        return torch.tensor(features[:20], dtype=torch.float32)
