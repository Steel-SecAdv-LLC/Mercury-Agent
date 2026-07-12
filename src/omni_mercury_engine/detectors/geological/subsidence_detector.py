# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Subsidence & Sinkhole-Precursor Detector — InSAR-style displacement-series analysis.

Consumes the standard InSAR persistent-scatterer / distributed-scatterer (PS/DS)
product shape: per-point line-of-sight (LOS) displacement time series in
millimetres over a common set of acquisition epochs, with optional temporal
coherence per point.  All methods are deterministic, literature-anchored
signal processing — there is no neural network in this detector and nothing
is ever inferred from unavailable data.

Methods implemented:

- **Robust linear velocity** per point via the Theil–Sen estimator (median of
  pairwise slopes), the standard robust velocity estimate for noisy InSAR
  series (used operationally by wide-area PSI services; see Crosetto et al.,
  2020).
- **Acceleration detection** per point via a nested-model comparison: an
  ordinary least-squares quadratic fit is tested against the linear fit with
  an F-test on the residual sums of squares *and* a BIC comparison.  A point
  is flagged "accelerating" only when both criteria agree and the quadratic
  term reinforces the Theil–Sen trend direction.  Precursory acceleration in
  satellite InSAR series is the established failure-forecast signal
  (Intrieri et al., 2018).
- **Spatial density clustering** of accelerating, subsiding points: a
  fixed-radius neighbour graph (default 100 m, documented below) whose
  connected components with at least ``cluster_min_points`` members form
  clusters.
- **Sinkhole-precursor screening**: a cluster is flagged as a sinkhole
  precursor when it is a *localized accelerating subsidence bowl* — median
  cluster velocity beyond the subsidence threshold, statistically significant
  acceleration (by construction of the cluster), spatial extent below
  ``sinkhole_max_extent_m``, and cluster motion clearly below the scene
  median.  Localized precursory subsidence of exactly this form was observed
  before Dead Sea sinkhole collapses (Nof et al., 2013).

Velocity severity classes (absolute LOS velocity) are anchored to published
observations: the ~2 mm/yr class boundary is the wide-area PSI noise floor
(Crosetto et al., 2020, EGMS precision ~1 mm/yr), and the upper classes span
the cm/yr-to-dm/yr range documented for compacting basins, up to the
~400 mm/yr extreme measured over Mexico City (Cigna & Tapete, 2021).

Transparent data note:
    Real InSAR feeds (EGMS, COMET-LiCS) are offline product downloads that
    require registration; there is no free real-time API. This detector
    therefore consumes their standard *product shape* (points × epochs LOS
    displacement + coherence) rather than pretending to fetch live data.
    It fails loudly on fewer than ``min_epochs`` epochs or on any all-NaN
    point series — it never fabricates a velocity.

References:
    - Intrieri, E., Raspini, F., Fumagalli, A., Lu, P., Del Conte, S.,
      Farina, P., Allievi, J., Ferretti, A., Casagli, N. (2018). The Maoxian
      landslide as seen from space: detecting precursors of failure with
      Sentinel-1 data. Landslides 15(1), 123-133.
    - Nof, R.N., Baer, G., Ziv, A., Raz, E., Atzori, S., Salvi, S. (2013).
      Sinkhole precursors along the Dead Sea, Israel, revealed by SAR
      interferometry. Geology 41(9), 1019-1022.
    - Cigna, F., Tapete, D. (2021). Present-day land subsidence rates,
      surface faulting hazard and risk in Mexico City with 2014-2020
      Sentinel-1 IW InSAR. Remote Sensing of Environment 253, 112161.
    - Crosetto, M., Solari, L., Mroz, M., et al. (2020). The evolution of
      wide-area DInSAR: from regional and national services to the European
      Ground Motion Service. Remote Sensing 12(12), 2043.
    - Kass, R.E., Raftery, A.E. (1995). Bayes factors. JASA 90(430), 773-795.
      (ΔBIC > 2 = positive evidence, used as the BIC margin here.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class SubsidenceSeverity(Enum):
    """Absolute-LOS-velocity severity classes (see module docstring)."""

    STABLE = "stable"  # |v| < 2 mm/yr (PSI noise floor, Crosetto et al. 2020)
    SLOW = "slow"  # 2-10 mm/yr
    MODERATE = "moderate"  # 10-50 mm/yr
    FAST = "fast"  # 50-150 mm/yr
    EXTREME = "extreme"  # > 150 mm/yr (Mexico City class, Cigna & Tapete 2021)


#: Class boundaries in mm/yr for :class:`SubsidenceSeverity` (documented above).
SEVERITY_BOUNDS_MM_YR: tuple[float, float, float, float] = (2.0, 10.0, 50.0, 150.0)


@dataclass
class PointKinematics:
    """Per-point kinematic estimates from one LOS displacement series.

    Attributes:
        index: Point index in the input array.
        velocity_mm_yr: Theil-Sen LOS velocity (negative = subsidence,
            following the InSAR LOS convention).
        velocity_ci_mm_yr: 95% confidence interval on the Theil-Sen slope.
        acceleration_mm_yr2: Twice the quadratic coefficient of the OLS
            quadratic fit (d2 displacement / dt2), in mm/yr^2.
        accelerating: True when the quadratic model beats the linear model on
            both the F-test and the BIC margin AND the quadratic term
            reinforces the Theil-Sen trend direction.
        f_pvalue: p-value of the nested-model F-test (quadratic vs linear).
        delta_bic: BIC(linear) - BIC(quadratic); > margin favours quadratic.
        n_valid_epochs: Number of non-NaN epochs used.
    """

    index: int
    velocity_mm_yr: float
    velocity_ci_mm_yr: tuple[float, float]
    acceleration_mm_yr2: float
    accelerating: bool
    f_pvalue: float
    delta_bic: float
    n_valid_epochs: int


@dataclass
class SinkholeCluster:
    """A spatially concentrated cluster of accelerating, subsiding points.

    Attributes:
        point_indices: Indices of the member points.
        centroid_m: (x, y) centroid of the member points, metres.
        extent_m: Maximum member distance from the centroid, metres.
        median_velocity_mm_yr: Median Theil-Sen velocity of members.
        median_acceleration_mm_yr2: Median acceleration of members.
        sinkhole_precursor: True when the cluster passes all
            sinkhole-precursor criteria (see module docstring).
        criteria: Named boolean criteria that were evaluated, for the
            evidence trail.
    """

    point_indices: list[int]
    centroid_m: tuple[float, float]
    extent_m: float
    median_velocity_mm_yr: float
    median_acceleration_mm_yr2: float
    sinkhole_precursor: bool
    criteria: dict[str, bool] = field(default_factory=dict)


@dataclass
class SubsidencePredictionResult:
    """Full result of a subsidence / sinkhole-precursor analysis.

    Attributes:
        anomaly_detected: True when any point exceeds the stable class or any
            cluster was found.
        severity: Worst :class:`SubsidenceSeverity` value over all points.
        confidence: Deterministic evidence score in [0, 1] (fraction of
            fired criteria; never from an untrained model).
        max_subsidence_velocity_mm_yr: Most negative Theil-Sen velocity.
        n_points: Number of points analysed.
        n_accelerating_points: Points flagged accelerating.
        point_kinematics: Per-point estimates.
        clusters: Density clusters of accelerating subsiding points
            (empty when coordinates were not supplied).
        sinkhole_precursor_detected: True when any cluster passes all
            sinkhole-precursor criteria.
        notes: Transparent notes about skipped stages (e.g. no coordinates).
    """

    anomaly_detected: bool
    severity: str
    confidence: float
    max_subsidence_velocity_mm_yr: float
    n_points: int
    n_accelerating_points: int
    point_kinematics: list[PointKinematics] = field(default_factory=list)
    clusters: list[SinkholeCluster] = field(default_factory=list)
    sinkhole_precursor_detected: bool = False
    notes: list[str] = field(default_factory=list)


class SubsidenceDetector:
    """Subsidence and sinkhole-precursor detector for InSAR-style series.

    Deterministic physics/statistics core; works untrained by construction.

    Args:
        min_epochs: Minimum number of acquisition epochs; the Theil-Sen and
            quadratic-vs-linear comparisons are meaningless below this
            (fail-loud). Default 8.
        accel_p_value: F-test significance level for the quadratic term.
            Default 0.01.
        bic_margin: Required BIC(linear) - BIC(quadratic) margin; 2.0 is the
            "positive evidence" bound of Kass & Raftery (1995).
        subsidence_velocity_mm_yr: Velocity threshold (magnitude, mm/yr) for
            a cluster to count as actively subsiding; default 10 mm/yr, the
            boundary of the cm/yr classes mapped by Cigna & Tapete (2021).
        cluster_radius_m: Neighbour radius for density clustering. Default
            100 m — sinkhole-precursor bowls observed at the Dead Sea span
            tens to a few hundreds of metres (Nof et al., 2013).
        cluster_min_points: Minimum cluster membership (DBSCAN-style minPts).
        sinkhole_max_extent_m: Maximum cluster extent (centroid-to-member)
            for the "localized bowl" criterion. Default 500 m.
        coherence_min: Points with temporal coherence below this are dropped
            before analysis (standard PS selection practice). Default 0.3.
    """

    def __init__(
        self,
        min_epochs: int = 8,
        accel_p_value: float = 0.01,
        bic_margin: float = 2.0,
        subsidence_velocity_mm_yr: float = 10.0,
        cluster_radius_m: float = 100.0,
        cluster_min_points: int = 4,
        sinkhole_max_extent_m: float = 500.0,
        coherence_min: float = 0.3,
    ) -> None:
        """Initialize the instance."""
        if min_epochs < 5:
            raise ValueError(
                f"min_epochs must be >= 5 for the quadratic-vs-linear F-test "
                f"(n - 3 > 0 with margin); got {min_epochs}"
            )
        self.min_epochs = min_epochs
        self.accel_p_value = accel_p_value
        self.bic_margin = bic_margin
        self.subsidence_velocity_mm_yr = subsidence_velocity_mm_yr
        self.cluster_radius_m = cluster_radius_m
        self.cluster_min_points = cluster_min_points
        self.sinkhole_max_extent_m = sinkhole_max_extent_m
        self.coherence_min = coherence_min
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(
        self,
        displacements_mm: np.ndarray[Any, Any],
        epochs_years: np.ndarray[Any, Any],
        coordinates_m: np.ndarray[Any, Any] | None = None,
        coherence: np.ndarray[Any, Any] | None = None,
    ) -> SubsidencePredictionResult:
        """Analyse a stack of LOS displacement series.

        Args:
            displacements_mm: LOS displacement in mm, shape ``(n_points,
                n_epochs)`` (a single 1-D series is accepted and treated as
                one point). NaN marks a missing acquisition for a point.
            epochs_years: Acquisition times in decimal years (or any strictly
                increasing time axis in year units), shape ``(n_epochs,)``.
            coordinates_m: Optional projected point coordinates in metres,
                shape ``(n_points, 2)``. Required for clustering / sinkhole
                screening; without it those stages are skipped with a note.
            coherence: Optional temporal coherence per point, shape
                ``(n_points,)``, in [0, 1].

        Returns:
            SubsidencePredictionResult with per-point kinematics, clusters,
            and the sinkhole-precursor flag.

        Raises:
            ValueError: On fewer than ``min_epochs`` epochs, on any all-NaN
                point series, on a point with fewer than ``min_epochs`` valid
                epochs, on non-increasing epochs, or on shape mismatches.
        """
        disp = np.asarray(displacements_mm, dtype=float)
        if disp.ndim == 1:
            disp = disp.reshape(1, -1)
        if disp.ndim != 2:
            raise ValueError(f"displacements_mm must be 1-D or 2-D; got ndim={disp.ndim}")

        t = np.asarray(epochs_years, dtype=float)
        if t.ndim != 1 or t.shape[0] != disp.shape[1]:
            raise ValueError(
                f"epochs_years shape {t.shape} does not match n_epochs={disp.shape[1]}"
            )
        if t.shape[0] < self.min_epochs:
            raise ValueError(
                f"InSAR series has {t.shape[0]} epochs; need >= {self.min_epochs}. "
                "Refusing to estimate velocity/acceleration from an under-sampled stack."
            )
        if np.any(~np.isfinite(t)) or np.any(np.diff(t) <= 0):
            raise ValueError("epochs_years must be finite and strictly increasing")

        n_points = disp.shape[0]
        all_nan = np.all(np.isnan(disp), axis=1)
        if np.any(all_nan):
            bad = np.flatnonzero(all_nan).tolist()
            raise ValueError(
                f"Point(s) {bad} contain only NaN displacements. Refusing to "
                "fabricate kinematics for empty series; drop them upstream."
            )

        notes: list[str] = []
        keep = np.ones(n_points, dtype=bool)
        if coherence is not None:
            coh = np.asarray(coherence, dtype=float)
            if coh.shape != (n_points,):
                raise ValueError(f"coherence shape {coh.shape} != (n_points,) = ({n_points},)")
            keep = coh >= self.coherence_min
            if not np.any(keep):
                raise ValueError(
                    f"All {n_points} points fall below coherence_min={self.coherence_min}; "
                    "no reliable scatterers to analyse."
                )
            n_dropped = int(np.sum(~keep))
            if n_dropped:
                notes.append(
                    f"{n_dropped} point(s) dropped below coherence_min={self.coherence_min}"
                )

        kinematics: list[PointKinematics] = []
        for i in np.flatnonzero(keep):
            kinematics.append(self._point_kinematics(int(i), disp[i], t))

        accel_subsiding = [k for k in kinematics if k.accelerating and k.velocity_mm_yr < 0.0]

        clusters: list[SinkholeCluster] = []
        if coordinates_m is not None:
            coords = np.asarray(coordinates_m, dtype=float)
            if coords.shape != (n_points, 2):
                raise ValueError(
                    f"coordinates_m shape {coords.shape} != (n_points, 2) = ({n_points}, 2)"
                )
            scene_median_v = float(np.median([k.velocity_mm_yr for k in kinematics]))
            clusters = self._cluster_accelerating_points(accel_subsiding, coords, scene_median_v)
        elif accel_subsiding:
            notes.append(
                "coordinates_m not supplied: spatial clustering and sinkhole-precursor "
                "screening skipped (acceleration flags are still per-point valid)"
            )

        velocities = np.array([k.velocity_mm_yr for k in kinematics])
        max_sub_v = float(np.min(velocities))
        worst_abs_v = float(np.max(np.abs(velocities)))
        severity = self._classify_severity(worst_abs_v)

        sinkhole = any(c.sinkhole_precursor for c in clusters)
        anomaly = severity != SubsidenceSeverity.STABLE.value or bool(clusters)
        confidence = self._evidence_confidence(
            severity, len(accel_subsiding), len(kinematics), clusters
        )

        return SubsidencePredictionResult(
            anomaly_detected=anomaly,
            severity=severity,
            confidence=confidence,
            max_subsidence_velocity_mm_yr=max_sub_v,
            n_points=len(kinematics),
            n_accelerating_points=sum(1 for k in kinematics if k.accelerating),
            point_kinematics=kinematics,
            clusters=clusters,
            sinkhole_precursor_detected=sinkhole,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Per-point kinematics
    # ------------------------------------------------------------------
    def _point_kinematics(
        self, index: int, series_mm: np.ndarray[Any, Any], t_years: np.ndarray[Any, Any]
    ) -> PointKinematics:
        """Estimate Theil-Sen velocity and test for acceleration at one point.

        Args:
            index: Point index (for the result record).
            series_mm: LOS displacement series (may contain NaN gaps).
            t_years: Epoch times in years.

        Returns:
            PointKinematics for the point.

        Raises:
            ValueError: If fewer than ``min_epochs`` valid epochs remain.
        """
        valid = np.isfinite(series_mm)
        n_valid = int(np.sum(valid))
        if n_valid < self.min_epochs:
            raise ValueError(
                f"Point {index} has {n_valid} valid epochs (< {self.min_epochs}). "
                "Refusing to estimate kinematics from an under-sampled series."
            )
        y = series_mm[valid]
        x = t_years[valid]

        ts = stats.theilslopes(y, x, alpha=0.95)
        velocity = float(ts.slope)
        ci = (float(ts.low_slope), float(ts.high_slope))

        # Nested-model comparison: linear (2 params) vs quadratic (3 params).
        xc = x - x.mean()  # centre time to condition the design matrix
        lin_coef = np.polyfit(xc, y, 1)
        quad_coef = np.polyfit(xc, y, 2)
        rss_lin = float(np.sum((y - np.polyval(lin_coef, xc)) ** 2))
        rss_quad = float(np.sum((y - np.polyval(quad_coef, xc)) ** 2))
        n = n_valid
        dof = n - 3
        eps = 1e-12
        f_stat = ((rss_lin - rss_quad) / 1.0) / max(rss_quad / dof, eps)
        f_pvalue = float(stats.f.sf(f_stat, 1, dof))
        # BIC with Gaussian errors: n*ln(RSS/n) + k*ln(n)
        bic_lin = n * np.log(max(rss_lin, eps) / n) + 2 * np.log(n)
        bic_quad = n * np.log(max(rss_quad, eps) / n) + 3 * np.log(n)
        delta_bic = float(bic_lin - bic_quad)

        accel = 2.0 * float(quad_coef[0])  # d2y/dt2 in mm/yr^2
        # Accelerating = both tests agree AND the quadratic term speeds up the
        # Theil-Sen trend (same sign), i.e. |velocity| is increasing.
        reinforces = accel * velocity > 0.0
        accelerating = f_pvalue < self.accel_p_value and delta_bic > self.bic_margin and reinforces

        return PointKinematics(
            index=index,
            velocity_mm_yr=velocity,
            velocity_ci_mm_yr=ci,
            acceleration_mm_yr2=accel,
            accelerating=bool(accelerating),
            f_pvalue=f_pvalue,
            delta_bic=delta_bic,
            n_valid_epochs=n_valid,
        )

    # ------------------------------------------------------------------
    # Spatial clustering + sinkhole screening
    # ------------------------------------------------------------------
    def _cluster_accelerating_points(
        self,
        accel_points: list[PointKinematics],
        coords: np.ndarray[Any, Any],
        scene_median_velocity: float,
    ) -> list[SinkholeCluster]:
        """Cluster accelerating subsiding points and screen for precursors.

        Fixed-radius connected components (radius ``cluster_radius_m``),
        keeping components with >= ``cluster_min_points`` members.

        Args:
            accel_points: Accelerating, subsiding points.
            coords: All point coordinates in metres, shape (n_points, 2).
            scene_median_velocity: Median velocity of all analysed points.

        Returns:
            List of clusters with sinkhole-precursor screening applied.
        """
        if len(accel_points) < self.cluster_min_points:
            return []

        idx = np.array([k.index for k in accel_points])
        pts = coords[idx]
        m = len(idx)

        # Union-find over the fixed-radius neighbour graph.
        parent = list(range(m))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        d2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=-1)
        r2 = self.cluster_radius_m**2
        for a in range(m):
            for b in range(a + 1, m):
                if d2[a, b] <= r2:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[rb] = ra

        groups: dict[int, list[int]] = {}
        for a in range(m):
            groups.setdefault(find(a), []).append(a)

        clusters: list[SinkholeCluster] = []
        by_index = {k.index: k for k in accel_points}
        for members in groups.values():
            if len(members) < self.cluster_min_points:
                continue
            member_idx = [int(idx[a]) for a in members]
            member_pts = pts[members]
            centroid = member_pts.mean(axis=0)
            extent = float(np.max(np.linalg.norm(member_pts - centroid, axis=1)))
            velocities = [by_index[i].velocity_mm_yr for i in member_idx]
            accels = [by_index[i].acceleration_mm_yr2 for i in member_idx]
            med_v = float(np.median(velocities))
            med_a = float(np.median(accels))

            criteria = {
                # Actively subsiding beyond the cm/yr class boundary.
                "velocity_below_threshold": med_v <= -self.subsidence_velocity_mm_yr,
                # Localized bowl, not a basin-wide signal (Nof et al. 2013).
                "localized_extent": extent <= self.sinkhole_max_extent_m,
                # Bowl is moving distinctly faster down than the scene.
                "below_scene_median": med_v
                < scene_median_velocity - self.subsidence_velocity_mm_yr / 2.0,
                # Members accelerate downward (negative LOS acceleration).
                "downward_acceleration": med_a < 0.0,
            }
            clusters.append(
                SinkholeCluster(
                    point_indices=member_idx,
                    centroid_m=(float(centroid[0]), float(centroid[1])),
                    extent_m=extent,
                    median_velocity_mm_yr=med_v,
                    median_acceleration_mm_yr2=med_a,
                    sinkhole_precursor=all(criteria.values()),
                    criteria=criteria,
                )
            )
        return clusters

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _classify_severity(abs_velocity_mm_yr: float) -> str:
        """Map |velocity| to a :class:`SubsidenceSeverity` class value."""
        b1, b2, b3, b4 = SEVERITY_BOUNDS_MM_YR
        if abs_velocity_mm_yr < b1:
            return SubsidenceSeverity.STABLE.value
        if abs_velocity_mm_yr < b2:
            return SubsidenceSeverity.SLOW.value
        if abs_velocity_mm_yr < b3:
            return SubsidenceSeverity.MODERATE.value
        if abs_velocity_mm_yr < b4:
            return SubsidenceSeverity.FAST.value
        return SubsidenceSeverity.EXTREME.value

    @staticmethod
    def _evidence_confidence(
        severity: str,
        n_accel_subsiding: int,
        n_points: int,
        clusters: list[SinkholeCluster],
    ) -> float:
        """Deterministic evidence score in [0, 1].

        Combines three independent lines of evidence with equal weight:
        velocity severity beyond the noise floor, the fraction of points
        with significant downward acceleration, and spatial concentration
        (any cluster; full weight when a cluster passes all sinkhole
        criteria).

        Args:
            severity: Severity class value.
            n_accel_subsiding: Count of accelerating subsiding points.
            n_points: Total analysed points.
            clusters: Density clusters found.

        Returns:
            Confidence in [0, 1].
        """
        sev_rank = {s.value: r for r, s in enumerate(SubsidenceSeverity)}
        sev_term = sev_rank[severity] / (len(SubsidenceSeverity) - 1)
        accel_term = min(1.0, n_accel_subsiding / max(1, n_points) * 4.0)
        if any(c.sinkhole_precursor for c in clusters):
            spatial_term = 1.0
        elif clusters:
            spatial_term = 0.5
        else:
            spatial_term = 0.0
        return float(round((sev_term + accel_term + spatial_term) / 3.0, 6))
