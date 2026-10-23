# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Maritime AIS trajectory datasets with expert-annotated abnormality labels.

``DTUAISLoader`` serves the labelled evaluation set of "AIS Trajectories
from Danish Waters for Abnormal Behavior Detection" (Olesen, Clemmensen &
Christensen, Technical University of Denmark): 521 real vessel
trajectories from 2021-12-13 in Danish waters (ROI 54.5-56N, 13-16E,
120 s resampling), of which 25 are manually labelled abnormal — colliding
vessels, Search-and-Rescue responders, law enforcement, and commercial
traffic forced off its normal course by the accident. Olesen et al. (J.
Mar. Sci. Eng. 11(11):2085, 2023, section 4.7 and Figure 12a) train on the
Bornholm December 2021 unlabelled data except the 13th and report ROC-AUC
on all 521 tracks: VRNN (the GeoTrackNet architecture) 0.87, their two-step
clustering 0.73 (0.79 with stationary tracks removed) and RVAE 0.72 —
making it the only public, expert-annotated real-AIS benchmark with
published baselines.  Kristoffer Vinther Olesen's DTU PhD thesis (2023,
Figure 6.1) adds RAE 0.86, Seq-2-Seq 0.84 and Seq-2-Seq with attention 0.83
on the same tracks, from models trained on 1 June to 30 November traffic.

Dataset DOI: 10.11583/DTU.21511815 (collection 10.11583/DTU.c.6287841).
License: CC BY 4.0 — attribution is carried in :attr:`DTUAISLoader.CITATION`.

Each trajectory is converted to a fixed vector of 22 kinematic/geometric
features (:data:`FEATURE_NAMES`); the benchmark harness consumes the
resulting ``(521, 22)`` matrix like any other tabular dataset.

Security: the upstream files are Python pickles. They are only ever
deserialised through a fail-closed restricted unpickler whose allowlist
covers plain numpy array reconstruction and nothing else — any other
global (the code-execution surface of pickle) raises instead of loading.
Download integrity is pinned by SHA-256 over the extracted members (the
figshare zip wrapper is generated per request, so the wrapper itself
cannot be pinned).
"""

from __future__ import annotations

import hashlib
import io
import logging
import pickle
import zipfile
from typing import TYPE_CHECKING, Any, BinaryIO

import numpy as np

from omni_mercury_engine.datasets.base import DatasetLoader, http_get_with_retry

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["FEATURE_NAMES", "DTUAISLoader", "restricted_pickle_load", "track_features"]

_EARTH_RADIUS_KM = 6371.0

#: Trajectory-level features, in column order.
FEATURE_NAMES: tuple[str, ...] = (
    "duration_h",
    "n_points",
    "speed_mean",
    "speed_std",
    "speed_max",
    "speed_p95",
    "speed_min",
    "frac_below_half_knot",
    "accel_abs_mean",
    "accel_abs_max",
    "course_resultant",
    "total_turn_deg",
    "turn_rate_p95",
    "frac_turn_gt30",
    "path_len_km",
    "net_disp_km",
    "straightness",
    "bbox_diag_km",
    "lat_mean",
    "lon_mean",
    "lat_std",
    "lon_std",
)

#: The only globals a DTU AIS pickle may import: plain numpy array
#: reconstruction. Everything else — any callable, any class — is refused.
_ALLOWED_PICKLE_GLOBALS: frozenset[tuple[str, str]] = frozenset(
    {
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "scalar"),
    }
)


class _RestrictedUnpickler(pickle.Unpickler):
    """Fail-closed unpickler for third-party dataset pickles."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in _ALLOWED_PICKLE_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refusing to unpickle forbidden global {module}.{name!s}")


def restricted_pickle_load(stream: BinaryIO) -> Any:
    """Deserialise one pickle record through the fail-closed allowlist."""
    return _RestrictedUnpickler(stream).load()


def _haversine_km(
    lat1: np.ndarray[Any, Any] | float,
    lon1: np.ndarray[Any, Any] | float,
    lat2: np.ndarray[Any, Any] | float,
    lon2: np.ndarray[Any, Any] | float,
) -> np.ndarray[Any, Any]:
    """Great-circle distance in kilometres."""
    lat1r, lon1r, lat2r, lon2r = (
        np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2)
    )
    a = (
        np.sin((lat2r - lat1r) / 2) ** 2
        + np.cos(lat1r) * np.cos(lat2r) * np.sin((lon2r - lon1r) / 2) ** 2
    )
    return np.asarray(2 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


def track_features(track: dict[str, Any]) -> np.ndarray[Any, Any]:
    """One trajectory record -> the 22-feature vector of :data:`FEATURE_NAMES`.

    Course over ground can be unavailable in real AIS (30 of the 521 DTU
    evaluation tracks carry NaN course, some for the whole track), so every
    course-derived feature is computed over the finite values only:
    ``course_resultant`` is the mean resultant of the finite courses (0.0 —
    no directional concentration — when none exist) and the turn features
    use consecutive finite pairs (0.0 when there are none). This keeps the
    missing-data policy here, explicit and tested, instead of leaking NaN
    for a downstream consumer to impute silently.

    Args:
        track: Record with ``lat``/``lon``/``speed``/``course``/``timestamp``
            sequences (knots, degrees, epoch seconds; 120 s resampling).

    Returns:
        ``(22,)`` float vector, all entries finite. Single-point tracks
        yield zero motion terms.

    Raises:
        ValueError: a non-finite feature survived (unexpected upstream data
            such as NaN positions) — fail loudly rather than emit NaN.
    """
    lat = np.asarray(track["lat"], dtype=float)
    lon = np.asarray(track["lon"], dtype=float)
    speed = np.asarray(track["speed"], dtype=float)
    course = np.asarray(track["course"], dtype=float)
    ts = np.asarray(track["timestamp"], dtype=float)
    n = len(lat)
    duration_h = (ts[-1] - ts[0]) / 3600.0 if n > 1 else 0.0

    dt = np.diff(ts)
    dt = np.where(dt > 0, dt, 120.0)
    accel = np.abs(np.diff(speed)) / dt if n > 1 else np.zeros(1)

    finite_course = course[np.isfinite(course)]
    if len(finite_course):
        rad = np.radians(finite_course)
        course_resultant = float(np.hypot(np.mean(np.sin(rad)), np.mean(np.cos(rad))))
    else:
        course_resultant = 0.0
    if n > 1:
        pair_mask = np.isfinite(course[:-1]) & np.isfinite(course[1:])
        turn = np.abs(course[1:][pair_mask] - course[:-1][pair_mask])
        turn = np.minimum(turn, 360.0 - turn)  # wrap to [0, 180]
    else:
        turn = np.zeros(0)
    if turn.size == 0:
        turn = np.zeros(1)

    seg_km = _haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:]) if n > 1 else np.zeros(1)
    path_len = float(seg_km.sum())
    net_disp = float(_haversine_km(lat[0], lon[0], lat[-1], lon[-1])) if n > 1 else 0.0
    straightness = net_disp / path_len if path_len > 1e-6 else 1.0
    bbox_diag = float(_haversine_km(lat.min(), lon.min(), lat.max(), lon.max()))

    features = np.array(
        [
            duration_h,
            float(n),
            float(speed.mean()),
            float(speed.std()),
            float(speed.max()),
            float(np.percentile(speed, 95)),
            float(speed.min()),
            float((speed < 0.5).mean()),
            float(accel.mean()),
            float(accel.max()),
            course_resultant,
            float(turn.sum()),
            float(np.percentile(turn, 95)),
            float((turn > 30.0).mean()),
            path_len,
            net_disp,
            straightness,
            bbox_diag,
            float(lat.mean()),
            float(lon.mean()),
            float(lat.std()),
            float(lon.std()),
        ]
    )
    if not np.all(np.isfinite(features)):
        bad = [FEATURE_NAMES[i] for i in np.flatnonzero(~np.isfinite(features))]
        raise ValueError(f"non-finite trajectory features {bad} for mmsi={track.get('mmsi')}")
    return features


class DTUAISLoader(DatasetLoader):
    """DTU Danish-waters labelled AIS abnormality evaluation set.

    521 trajectories / 25 expert-annotated abnormal, as 22 trajectory-level
    features. See the module docstring for provenance, published baselines
    and the pickle security posture.
    """

    DATASET_NAME = "dtu_ais_abnormal"
    LABEL_SOURCE = "expert_annotated"
    DATASET_URL = "https://doi.org/10.11583/DTU.21511815"
    LICENSE = "CC BY 4.0"
    CITATION = (
        "Olesen, K. V., Clemmensen, L. K. H., & Christensen, A. N. (2023). "
        "Labelled evaluation datasets of AIS Trajectories from Danish Waters "
        "for Abnormal Behavior Detection. Technical University of Denmark. "
        "https://doi.org/10.11583/DTU.21511815"
    )
    REQUIRES_CREDENTIALS = False

    #: figshare whole-article archive for the labelled set (port-443 route;
    #: the per-file endpoints redirect to object storage on a blocked port).
    ARCHIVE_URL = "https://ndownloader.figshare.com/articles/21511815/versions/1"

    #: The two members actually consumed (120 s resampling variant), with
    #: SHA-256 pinned over the member bytes — the zip wrapper is generated
    #: per request and cannot be pinned.
    INFO_MEMBER = (
        "datasetInfo_AIS_Custom_13122021_13122021_CarFisHigMilPasPleSaiTan_600_43200_120.pkl"
    )
    DATA_MEMBER = "data_AIS_Custom_13122021_13122021_CarFisHigMilPasPleSaiTan_600_43200_120.pkl"
    MEMBER_SHA256: dict[str, str] = {
        INFO_MEMBER: "cfeb4de5981ccb081a9023cbad91552d78f01ad3aa84c4942205bbe97342c9bb",
        DATA_MEMBER: "7241c1b8b9c71188ca0c8c9f85587ff97e1dc6933a616918908c35a20c8729f5",
    }

    def _member_path(self, member: str) -> Path:
        return self.data_path / member

    def _members_verified(self) -> bool:
        for member, expected in self.MEMBER_SHA256.items():
            path = self._member_path(member)
            if not path.is_file():
                return False
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected:
                logger.warning(
                    "%s: cached %s has digest %s != pinned %s",
                    self.DATASET_NAME,
                    member,
                    digest,
                    expected,
                )
                return False
        return True

    def download(self) -> bool:
        """Fetch and cache the labelled archive, verifying member digests.

        Returns:
            True once both consumed members are cached with pinned SHA-256.

        Raises:
            ValueError: a downloaded member does not match its pinned digest.
        """
        if self._members_verified():
            return True
        body = http_get_with_retry(self.ARCHIVE_URL, timeout=300)
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            for member, expected in self.MEMBER_SHA256.items():
                data = archive.read(member)
                digest = hashlib.sha256(data).hexdigest()
                if digest != expected:
                    raise ValueError(
                        f"{self.DATASET_NAME}: {member} digest {digest} does not "
                        f"match pinned {expected}; refusing to cache"
                    )
                self._member_path(member).write_bytes(data)
        return self._members_verified()

    def _iter_tracks(self) -> Iterator[dict[str, Any]]:
        """Yield trajectory records in label order (``indicies`` offsets)."""
        info = self._load_info()
        with open(self._member_path(self.DATA_MEMBER), "rb") as fh:
            for offset in info["indicies"]:
                fh.seek(int(offset))
                yield restricted_pickle_load(fh)

    def _load_info(self) -> dict[str, Any]:
        with open(self._member_path(self.INFO_MEMBER), "rb") as fh:
            info = restricted_pickle_load(fh)
        if not isinstance(info, dict) or "indicies" not in info or "outlierLabels" not in info:
            raise ValueError(f"{self.DATASET_NAME}: datasetInfo missing expected keys")
        return info

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Return ``(X, y)``: 22 features per trajectory, expert labels."""
        info = self._load_info()
        labels = np.asarray(info["outlierLabels"], dtype=int)
        features = np.vstack([track_features(track) for track in self._iter_tracks()])
        if len(features) != len(labels):
            raise ValueError(
                f"{self.DATASET_NAME}: {len(features)} trajectories vs {len(labels)} labels"
            )
        return features, labels

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Features are consumed as-is; scaling is the harness's concern."""
        return data
