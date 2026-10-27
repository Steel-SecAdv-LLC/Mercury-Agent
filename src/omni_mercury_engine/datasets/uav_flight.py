# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""ALFA UAV fault-flight loader - real actuator faults with injection-time ground truth.

The AIR Lab Failure and Anomaly (ALFA) dataset (Keipour, Mousaei & Scherer,
Carnegie Mellon University; CC0) records 47 autonomous fixed-wing (carbonZ)
flight sequences, most with a genuine actuator fault - engine power loss,
aileron, rudder or elevator lock - injected mid-flight by the safety pilot.
Each faulted sequence carries a ``failure_status-<surface>`` ROS topic that
publishes a nonzero code from the injection instant to the end of the
sequence, so the anomaly labels are event ground truth recorded at the
source (``LABEL_SOURCE = "ground_truth"``), not thresholds on features.

Per-topic MAVROS CSVs are resampled onto a fixed-rate grid by zero-order
hold with a staleness bound: a tick survives only when every required topic
has a sample at most ``max_staleness_s`` old, so sensor dropouts become
dropped ticks rather than stale fabricated rows. Features are the flight
physics a fault actually disturbs: commanded-versus-measured attitude and
airspeed tracking, navigation errors, body rates, accelerations, throttle
against climb, and the wind estimate.

Source: https://doi.org/10.1184/R1/12707963 (ALFA), tools and paper at
https://theairlab.org/alfa-dataset/.
"""

from __future__ import annotations

import csv
import hashlib
import io
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from omni_mercury_engine.datasets.base import (
    DatasetConfig,
    DatasetLoader,
    http_get_with_retry,
)
from omni_mercury_engine.datasets.exceptions import DataSourceUnavailableError
from omni_mercury_engine.security.input_validation import TrustedEndpoints

logger = logging.getLogger(__name__)

#: MAVROS topic suffix -> the CSV columns consumed from it.
_TOPIC_COLUMNS: dict[str, tuple[str, ...]] = {
    "mavros-nav_info-roll": ("field.commanded", "field.measured"),
    "mavros-nav_info-pitch": ("field.commanded", "field.measured"),
    "mavros-nav_info-yaw": ("field.commanded", "field.measured"),
    "mavros-nav_info-airspeed": ("field.commanded", "field.measured"),
    "mavros-nav_info-errors": ("field.alt_error", "field.aspd_error", "field.xtrack_error"),
    "mavros-imu-data": (
        "field.angular_velocity.x",
        "field.angular_velocity.y",
        "field.angular_velocity.z",
        "field.linear_acceleration.x",
        "field.linear_acceleration.y",
        "field.linear_acceleration.z",
    ),
    "mavros-vfr_hud": ("field.groundspeed", "field.throttle", "field.climb"),
    "mavros-global_position-rel_alt": ("field.data",),
    "mavros-wind_estimation": ("field.twist.linear.x", "field.twist.linear.y"),
    # Servo outputs: the actuation the autopilot actually commands. A locked
    # control surface shows up here as a frozen channel (variance collapse)
    # while its healthy partner keeps compensating -- the most direct
    # recorded signature of ALFA's aileron/rudder/elevator faults. Channels
    # 0, 6 and 7 are constant on the carbonZ and are not consumed.
    "mavros-rc-out": (
        "field.channels1",
        "field.channels2",
        "field.channels3",
        "field.channels4",
        "field.channels5",
    ),
}

_FAILURE_STATUS_MARKER = "-failure_status-"
_NS_PER_S = 1.0e9


def wrap_angle_deg(angle: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Wrap angles in degrees to ``(-180, 180]``."""
    return (np.asarray(angle, dtype=np.float64) - 180.0) % -360.0 + 180.0


def parse_topic_csv(
    text: str, columns: tuple[str, ...]
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Parse one MAVROS topic CSV into ``(times_s, values)``.

    Args:
        text: Raw CSV contents; first column is ``%time`` in nanoseconds.
        columns: Header names of the value columns to keep, in order.

    Returns:
        ``(times_s, values)`` where ``times_s`` is seconds (float64,
        ascending) and ``values`` has one column per requested name. Rows
        with a non-numeric cell in a requested column are dropped.

    Raises:
        ValueError: The header is missing ``%time`` or a requested column.
    """
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        raise ValueError("topic CSV is empty") from None
    if not header or header[0] != "%time":
        raise ValueError(f"topic CSV does not start with %time: {header[:3]!r}")
    try:
        indices = [header.index(name) for name in columns]
    except ValueError as exc:
        raise ValueError(f"topic CSV is missing a requested column: {exc}") from None
    times: list[float] = []
    rows: list[list[float]] = []
    for row in reader:
        if len(row) <= max(indices):
            continue
        try:
            stamp = int(row[0]) / _NS_PER_S
            values = [float(row[i]) for i in indices]
        except ValueError:
            continue
        if not all(np.isfinite(values)):
            continue
        times.append(stamp)
        rows.append(values)
    if not times:
        return np.empty(0, dtype=np.float64), np.empty((0, len(columns)), dtype=np.float64)
    order = np.argsort(np.asarray(times, dtype=np.float64), kind="stable")
    return (
        np.asarray(times, dtype=np.float64)[order],
        np.asarray(rows, dtype=np.float64)[order],
    )


def zoh_resample(
    times: np.ndarray[Any, Any],
    values: np.ndarray[Any, Any],
    grid: np.ndarray[Any, Any],
    max_staleness_s: float,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Zero-order-hold *values* onto *grid* with a staleness bound.

    Args:
        times: Ascending sample times in seconds.
        values: Samples, one row per time.
        grid: Target tick times in seconds.
        max_staleness_s: A grid tick is valid only when the held sample is
            at most this old.

    Returns:
        ``(resampled, valid)``: held rows per grid tick (undefined where
        invalid) and the boolean validity mask.
    """
    grid = np.asarray(grid, dtype=np.float64)
    if len(times) == 0:
        return (
            np.zeros((len(grid), values.shape[1] if values.ndim == 2 else 1)),
            np.zeros(len(grid), dtype=bool),
        )
    idx = np.searchsorted(times, grid, side="right") - 1
    valid = idx >= 0
    held = values[np.clip(idx, 0, len(times) - 1)]
    age = grid - times[np.clip(idx, 0, len(times) - 1)]
    valid &= age <= max_staleness_s
    return held, valid


@dataclass(frozen=True)
class ALFASequence:
    """One resampled ALFA flight sequence with ground-truth fault labels.

    Attributes:
        name: Sequence directory name (e.g.
            ``carbonZ_2018-07-18-15-53-31_2_engine_failure``).
        fault_surfaces: Faulted topics (``engines``, ``aileron``, ``rudder``,
            ``elevator``); empty for a no-failure flight.
        times_s: Tick times in seconds (absolute ROS epoch).
        features: ``(n_ticks, len(FEATURE_NAMES))`` matrix.
        labels: 1 from the recorded fault-injection instant onward, else 0.
    """

    name: str
    fault_surfaces: tuple[str, ...]
    times_s: np.ndarray[Any, Any]
    features: np.ndarray[Any, Any]
    labels: np.ndarray[Any, Any]


def build_sequence(
    name: str,
    topic_texts: dict[str, str],
    failure_status_texts: dict[str, str],
    sample_hz: float,
    max_staleness_s: float,
) -> ALFASequence | None:
    """Assemble one flight sequence from its per-topic CSV texts.

    Args:
        name: Sequence name (for diagnostics and the result record).
        topic_texts: Topic suffix -> raw CSV text, for every key in
            :data:`_TOPIC_COLUMNS`.
        failure_status_texts: Faulted surface name -> raw CSV text of its
            ``failure_status`` topic (empty for a no-failure flight).
        sample_hz: Grid rate in Hz.
        max_staleness_s: Zero-order-hold staleness bound in seconds.

    Returns:
        The resampled sequence, or None when a required topic is missing
        or no tick has every topic fresh.
    """
    parsed: dict[str, tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]] = {}
    for topic, columns in _TOPIC_COLUMNS.items():
        text = topic_texts.get(topic)
        if text is None:
            logger.warning("ALFA sequence %s: missing topic %s; skipped", name, topic)
            return None
        times, values = parse_topic_csv(text, columns)
        if len(times) == 0:
            logger.warning("ALFA sequence %s: topic %s has no rows; skipped", name, topic)
            return None
        parsed[topic] = (times, values)

    start = max(times[0] for times, _ in parsed.values())
    end = min(times[-1] for times, _ in parsed.values())
    if end <= start:
        logger.warning("ALFA sequence %s: topics do not overlap in time; skipped", name)
        return None
    grid = np.arange(start, end, 1.0 / sample_hz)

    held: dict[str, np.ndarray[Any, Any]] = {}
    valid = np.ones(len(grid), dtype=bool)
    for topic, (times, values) in parsed.items():
        resampled, topic_valid = zoh_resample(times, values, grid, max_staleness_s)
        held[topic] = resampled
        valid &= topic_valid
    if not valid.any():
        logger.warning("ALFA sequence %s: no tick has every topic fresh; skipped", name)
        return None

    roll = held["mavros-nav_info-roll"]
    pitch = held["mavros-nav_info-pitch"]
    yaw = held["mavros-nav_info-yaw"]
    airspeed = held["mavros-nav_info-airspeed"]
    errors = held["mavros-nav_info-errors"]
    imu = held["mavros-imu-data"]
    hud = held["mavros-vfr_hud"]
    rel_alt = held["mavros-global_position-rel_alt"]
    wind = held["mavros-wind_estimation"]
    rc_out = held["mavros-rc-out"]
    features = np.column_stack(
        [
            roll,  # commanded, measured
            pitch,  # commanded, measured
            wrap_angle_deg(yaw[:, 1] - yaw[:, 0]),  # yaw tracking error
            airspeed,  # commanded, measured
            errors,  # alt_error, aspd_error, xtrack_error
            imu,  # angular velocity xyz, linear acceleration xyz
            hud,  # groundspeed, throttle, climb
            rel_alt,
            wind,  # east, north
            rc_out,  # servo PWM channels 1-5
        ]
    )[valid]
    times_s = grid[valid]

    fault_start = np.inf
    surfaces: list[str] = []
    for surface, text in sorted(failure_status_texts.items()):
        reader = csv.reader(io.StringIO(text))
        next(reader, None)  # header
        stamps = [
            int(row[0]) / _NS_PER_S for row in reader if len(row) >= 2 and row[1].strip() != "0"
        ]
        if stamps:
            surfaces.append(surface)
            fault_start = min(fault_start, *stamps)
    labels = (times_s >= fault_start).astype(np.int64)
    return ALFASequence(
        name=name,
        fault_surfaces=tuple(surfaces),
        times_s=times_s,
        features=features,
        labels=labels,
    )


class ALFAUAVLoader(DatasetLoader):
    """ALFA fixed-wing UAV fault flights with injection-time ground truth.

    Note:
        Labels are genuine event ground truth (the onboard
        ``failure_status`` topics record the fault-injection instant), so
        this loader has **no synthetic fallback**: fabricating fault
        windows would forge ground truth. When neither the cache, a local
        copy, nor the pinned download is available it refuses loudly.
    """

    DATASET_NAME = "alfa_uav"
    LABEL_SOURCE = "ground_truth"
    DATASET_URL = "https://doi.org/10.1184/R1/12707963"
    LICENSE = "CC0 (Carnegie Mellon University, AIR Lab)"
    CITATION = """Keipour, A., Mousaei, M., & Scherer, S. (2021): ALFA: A Dataset for UAV
    Fault and Anomaly Detection. The International Journal of Robotics Research, 40(2-3).
    https://theairlab.org/alfa-dataset/"""
    REQUIRES_CREDENTIALS = False

    ARCHIVE_URL = TrustedEndpoints.ALFA_UAV_PROCESSED
    #: SHA-256 of the ``processed.zip`` figshare file (id 24095870), pinned
    #: from a verified download of the published article version.
    ARCHIVE_SHA256 = "02562756b9428aa24fbd27105f91687b38541ced8d5432766c14176573d0b4c9"

    FEATURE_NAMES = [
        "roll_cmd_deg",
        "roll_meas_deg",
        "pitch_cmd_deg",
        "pitch_meas_deg",
        "yaw_err_deg",
        "airspeed_cmd_ms",
        "airspeed_meas_ms",
        "alt_error_m",
        "aspd_error_ms",
        "xtrack_error_m",
        "ang_vel_x_rads",
        "ang_vel_y_rads",
        "ang_vel_z_rads",
        "lin_acc_x_ms2",
        "lin_acc_y_ms2",
        "lin_acc_z_ms2",
        "groundspeed_ms",
        "throttle",
        "climb_ms",
        "rel_alt_m",
        "wind_east_ms",
        "wind_north_ms",
        "rcout_ch1_pwm",
        "rcout_ch2_pwm",
        "rcout_ch3_pwm",
        "rcout_ch4_pwm",
        "rcout_ch5_pwm",
    ]

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize the instance."""
        super().__init__(config)
        self.sample_hz = float(config.preprocessing.get("sample_hz", 4.0))
        self.max_staleness_s = float(config.preprocessing.get("max_staleness_s", 1.5))
        #: Optional path to an already-extracted ``processed/`` root (the
        #: directory whose children are the ``carbonZ_*`` sequence dirs).
        self.local_dir = config.preprocessing.get("local_dir")
        self._is_real_data = False

    @property
    def is_real_data(self) -> bool:
        """Return True if real data was loaded."""
        return self._is_real_data

    def _real_cache_name(self) -> str:
        """Cache name carries grid parameters and feature width: both baked in."""
        return (
            f"alfa_uav_real_hz{self.sample_hz:g}_stale{self.max_staleness_s:g}"
            f"_f{len(self.FEATURE_NAMES)}.npz"
        )

    def download(self) -> bool:
        """Build the resampled cache from a local copy or the pinned archive.

        Returns:
            True once the cache exists.

        Raises:
            DataSourceUnavailableError: No local copy and the pinned
                download failed; there is deliberately no synthetic path.
            ValueError: The downloaded archive does not match the pinned
                SHA-256.
        """
        cache_file = self.data_path / self._real_cache_name()
        if cache_file.exists():
            self._is_real_data = True
            return True
        sequences = (
            self._sequences_from_directory(Path(self.local_dir))
            if self.local_dir
            else self._sequences_from_download()
        )
        if not sequences:
            raise DataSourceUnavailableError(
                loader_name="ALFAUAV",
                source_url=self.ARCHIVE_URL,
                reason="no ALFA flight sequences could be parsed",
            )
        self._write_cache(cache_file, sequences)
        self._is_real_data = True
        logger.info(
            "ALFA UAV loaded: %d sequences, %d ticks, %d fault-active (is_real_data=True)",
            len(sequences),
            sum(len(s.labels) for s in sequences),
            sum(int(s.labels.sum()) for s in sequences),
        )
        return True

    # ------------------------------------------------------------------
    # Acquisition
    # ------------------------------------------------------------------

    def _sequences_from_download(self) -> list[ALFASequence]:
        """Download the pinned archive and parse its CSV members."""
        TrustedEndpoints.validate_url(self.ARCHIVE_URL)
        body = http_get_with_retry(
            self.ARCHIVE_URL, timeout=900, timeout_per_attempt=600, retries=1
        )
        digest = hashlib.sha256(body).hexdigest()
        if digest != self.ARCHIVE_SHA256:
            raise ValueError(
                f"{self.DATASET_NAME}: archive digest {digest} does not match "
                f"pinned {self.ARCHIVE_SHA256}; refusing to parse"
            )
        texts_by_sequence: dict[str, dict[str, str]] = {}
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            for member in archive.namelist():
                if not member.endswith(".csv"):
                    continue
                parts = Path(member).parts
                if len(parts) < 2:
                    continue
                sequence = parts[-2]
                texts_by_sequence.setdefault(sequence, {})[Path(member).name] = archive.read(
                    member
                ).decode("utf-8", errors="replace")
        return self._build_sequences(texts_by_sequence)

    def _sequences_from_directory(self, root: Path) -> list[ALFASequence]:
        """Parse an already-extracted ``processed/`` tree."""
        texts_by_sequence: dict[str, dict[str, str]] = {}
        for seq_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            texts_by_sequence[seq_dir.name] = {
                f.name: f.read_text(encoding="utf-8", errors="replace")
                for f in seq_dir.glob("*.csv")
            }
        return self._build_sequences(texts_by_sequence)

    def _build_sequences(self, texts_by_sequence: dict[str, dict[str, str]]) -> list[ALFASequence]:
        """Resample every usable sequence; skip those without ground truth."""
        sequences: list[ALFASequence] = []
        for name in sorted(texts_by_sequence):
            files = texts_by_sequence[name]
            failure_texts = {
                fname.split(_FAILURE_STATUS_MARKER, 1)[1].removesuffix(".csv"): text
                for fname, text in files.items()
                if _FAILURE_STATUS_MARKER in fname
            }
            if not failure_texts and "no_failure" not in name:
                # e.g. the one "no_ground_truth" sequence: unlabelable.
                logger.info("ALFA sequence %s has no ground truth; skipped", name)
                continue
            topic_texts: dict[str, str] = {}
            for topic in _TOPIC_COLUMNS:
                fname = f"{name}-{topic}.csv"
                if fname in files:
                    topic_texts[topic] = files[fname]
            sequence = build_sequence(
                name, topic_texts, failure_texts, self.sample_hz, self.max_staleness_s
            )
            if sequence is not None:
                sequences.append(sequence)
        return sequences

    # ------------------------------------------------------------------
    # Cache layout: flat arrays plus per-sequence offsets
    # ------------------------------------------------------------------

    def _write_cache(self, cache_file: Path, sequences: list[ALFASequence]) -> None:
        offsets = np.cumsum([0] + [len(s.labels) for s in sequences])
        np.savez_compressed(
            cache_file,
            features=np.concatenate([s.features for s in sequences]),
            labels=np.concatenate([s.labels for s in sequences]),
            times_s=np.concatenate([s.times_s for s in sequences]),
            offsets=offsets,
            names=np.asarray([s.name for s in sequences]),
            fault_surfaces=np.asarray([",".join(s.fault_surfaces) for s in sequences]),
        )

    def load_sequences(self) -> list[ALFASequence]:
        """Return the per-sequence view (streaming evaluation needs order).

        Returns:
            One :class:`ALFASequence` per cached flight, in cache order.

        Raises:
            FileNotFoundError: The cache does not exist; call
                :meth:`download` first.
        """
        cache_file = self.data_path / self._real_cache_name()
        if not cache_file.exists():
            raise FileNotFoundError(
                f"No cached ALFA data under {self.data_path}; call download() first."
            )
        data = np.load(cache_file, allow_pickle=False)
        offsets = data["offsets"]
        self._is_real_data = True
        return [
            ALFASequence(
                name=str(data["names"][i]),
                fault_surfaces=tuple(s for s in str(data["fault_surfaces"][i]).split(",") if s),
                times_s=data["times_s"][offsets[i] : offsets[i + 1]],
                features=data["features"][offsets[i] : offsets[i + 1]],
                labels=data["labels"][offsets[i] : offsets[i + 1]],
            )
            for i in range(len(data["names"]))
        ]

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Load the cached matrix (ticks concatenated across sequences)."""
        cache_file = self.data_path / self._real_cache_name()
        if not cache_file.exists():
            raise FileNotFoundError(
                f"No cached ALFA data under {self.data_path}; call download() first."
            )
        data = np.load(cache_file, allow_pickle=False)
        self._is_real_data = True
        return data["features"], data["labels"]

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """No-op: stale or incomplete ticks were dropped at resampling."""
        return data


# Register at import so DatasetRegistry-driven callers can construct by name.
from omni_mercury_engine.datasets.base import DatasetRegistry

DatasetRegistry.register("alfa_uav", ALFAUAVLoader)
