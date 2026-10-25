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

``DTUAISTrainingSet`` serves the unlabelled training files of the same
collection (DOI 10.11583/DTU.21511842) that the published models were
trained on: June to November 2021 for the thesis, December 2021 without the
13th for the JMSE paper.

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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

import numpy as np

from omni_mercury_engine.datasets.base import (
    _DEFAULT_DATASET_UA,
    DatasetLoader,
    http_get_with_retry,
)
from omni_mercury_engine.datasets.zip_stream import extract_stored_members

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

logger = logging.getLogger(__name__)

__all__ = [
    "FEATURE_NAMES",
    "DTUAISLoader",
    "DTUAISTrainingSet",
    "restricted_pickle_load",
    "track_features",
]

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


@dataclass(frozen=True)
class _TrainingPeriod:
    """One unlabelled training file of the DTU AIS collection."""

    member: str
    #: SHA-256 over the member bytes, pinned from a copy whose MD5 equals
    #: figshare's published one (``md5``).
    sha256: str
    md5: str
    n_tracks: int


class DTUAISTrainingSet:
    """Unlabelled DTU AIS training trajectories for the published protocols.

    Companion files of :class:`DTUAISLoader`'s labelled set, same region and
    preprocessing (moving updates, eight ship types, 10 min to 12 h tracks,
    120 s resampling); no labels, so this is not a :class:`DatasetLoader`
    and never enters the labelled benchmark pool. Two periods:

    * ``"june_november"`` -- 1 June to 30 November 2021, 106,728 tracks:
      the training data of the models in Figure 6.1 of Olesen's DTU PhD
      thesis (2023, p. 48).
    * ``"december"`` -- December 2021 without the 13th (the file holds no
      point between 2021-12-13 00:00 and 2021-12-14 00:00 UTC), 12,591
      tracks: the training data of J. Mar. Sci. Eng. 11(11):2085 (2023),
      section 4.7, Figure 12a.

    figshare's per-file links redirect to DTU object storage
    (``s3q.ait.dtu.dk:9000``), which some networks cannot reach. The
    whole-article archive is served by figshare itself on port 443 but is
    6.5 GB, so it is streamed and closed once the period's member is written
    and matches its pinned SHA-256 (680 MB in for June-November, the first
    member; 1.1 GB for December, the fourth).
    """

    DATASET_URL = "https://doi.org/10.11583/DTU.21511842"
    LICENSE = "CC BY 4.0"
    CITATION = (
        "Olesen, K. V., Clemmensen, L. K. H., & Christensen, A. N. (2023). "
        "Unlabelled training datasets of AIS Trajectories from Danish Waters "
        "for Abnormal Behavior Detection. Technical University of Denmark. "
        "https://doi.org/10.11583/DTU.21511842"
    )
    ARCHIVE_URL = "https://ndownloader.figshare.com/articles/21511842/versions/1"
    PERIODS: dict[str, _TrainingPeriod] = {
        "june_november": _TrainingPeriod(
            member="data_AIS_Custom_01062021_30112021_CarFisHigMilPasPleSaiTan_600_43200_120.pkl",
            sha256="801b94417724928d4d872da876704d8b01607725c9df0b97f5bb1f69e31fc16b",
            md5="b7fa03df6c9e3d9abd830fbe72d83075",
            n_tracks=106_728,
        ),
        "december": _TrainingPeriod(
            member="data_AIS_Custom_01122021_31122021_CarFisHigMilPasPleSaiTan_600_43200_120.pkl",
            sha256="8daf08d1868512b163b16640553c9047e9dfd471bac43803874d688cec5a272f",
            md5="d7bd2d763d39473a8d824daeb67d3395",
            n_tracks=12_591,
        ),
    }

    def __init__(self, data_dir: str | Path, period: str = "june_november") -> None:
        """Cache under ``data_dir/dtu_ais_training``.

        Raises:
            ValueError: ``period`` is not one of :attr:`PERIODS`.
        """
        if period not in self.PERIODS:
            raise ValueError(f"unknown period {period!r}; one of {sorted(self.PERIODS)}")
        self._period_name = period
        self.period = self.PERIODS[period]
        self.data_path = Path(data_dir) / "dtu_ais_training"

    @property
    def member_path(self) -> Path:
        """Where the period's member is cached."""
        return self.data_path / self.period.member

    def is_cached(self) -> bool:
        """Whether the cached member exists and matches its pinned SHA-256."""
        if not self.member_path.is_file():
            return False
        digest = hashlib.sha256()
        with self.member_path.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest() == self.period.sha256

    def download(self, timeout: float = 300.0) -> None:
        """Stream the archive until the period's member is cached and verified.

        Args:
            timeout: Seconds to wait for each read of the stream.

        Raises:
            OfflineModeError: Network access is disabled.
            ZipStreamError: The member is absent, corrupt or not the pinned bytes.
            requests.HTTPError: figshare answered with an error status.
        """
        self.download_periods(self.data_path.parent, [self._period_name], timeout=timeout)

    @classmethod
    def download_periods(
        cls, data_dir: str | Path, periods: Iterable[str], timeout: float = 300.0
    ) -> None:
        """Cache several periods from one pass over the archive.

        The members lie in one stream, so fetching them together reads each
        byte once (December alone would stream past June-November anyway).

        Args:
            data_dir: As for the constructor.
            periods: Names from :attr:`PERIODS`.
            timeout: Seconds to wait for each read of the stream.

        Raises:
            ValueError: An unknown period.
            OfflineModeError: Network access is disabled.
            ZipStreamError: A member is absent, corrupt or not the pinned bytes.
            requests.HTTPError: figshare answered with an error status.
        """
        sets = [cls(data_dir, name) for name in periods]
        wanted = {t.period.member: t.period.sha256 for t in sets if not t.is_cached()}
        if not wanted:
            return
        from omni_mercury_engine.datasets.exceptions import OfflineModeError, offline_mode_active
        from omni_mercury_engine.security.safe_http import SafeHTTPClient

        if offline_mode_active():
            raise OfflineModeError(cls.ARCHIVE_URL)
        target = sets[0].data_path
        target.mkdir(parents=True, exist_ok=True)
        with SafeHTTPClient.get(
            cls.ARCHIVE_URL,
            headers={"User-Agent": _DEFAULT_DATASET_UA},
            timeout=timeout,
            stream=True,
        ) as response:
            response.raise_for_status()
            extract_stored_members(response.iter_content(1 << 20), wanted, target)

    def iter_tracks(self) -> Iterator[dict[str, Any]]:
        """Yield the trajectory records in file order (one pickle each)."""
        with self.member_path.open("rb") as fh:
            while True:
                try:
                    yield restricted_pickle_load(fh)
                except EOFError:
                    return

    def features(self) -> np.ndarray[Any, Any]:
        """The ``(n_tracks, 22)`` matrix of :data:`FEATURE_NAMES`.

        Raises:
            ValueError: The file does not hold the period's ``n_tracks``.
        """
        rows = [track_features(track) for track in self.iter_tracks()]
        if len(rows) != self.period.n_tracks:
            raise ValueError(
                f"{self.period.member}: {len(rows)} trajectories, "
                f"expected {self.period.n_tracks}"
            )
        return np.vstack(rows)
