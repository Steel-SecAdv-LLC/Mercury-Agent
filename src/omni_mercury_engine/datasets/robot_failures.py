# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""UCI Robot Execution Failures loader - real manipulator force/torque, real outcomes.

The classic UCI dataset 138 (Lopes & Camarinha-Matos, 1998; CC BY 4.0,
DOI 10.24432/C5M89N): 463 executions of a real robot manipulator, each
recorded as 15 consecutive force/torque samples (Fx, Fy, Fz, Tx, Ty, Tz)
taken at 315 ms intervals immediately after failure detection. Five
learning problems cover distinct task stages:

====  =========================================  =====================================
LP    Task stage                                 Classes
====  =========================================  =====================================
lp1   approach to grasp position                 normal, collision, fr_collision,
                                                 obstruction
lp2   transfer of a part                         normal, front/back/right/left_col
lp3   position of part after a transfer failure  ok, slightly_moved, moved, lost
lp4   approach to ungrasp position               normal, collision, obstruction
lp5   motion with part                           normal, bottom_collision,
                                                 bottom_obstruction,
                                                 collision_in_part, collision_in_tool
====  =========================================  =====================================

Labels are the **observed execution outcome** of each run - event ground
truth annotated by the operators, independent of the force/torque values
Mercury scores (``LABEL_SOURCE = "ground_truth"``). Anomaly = any
non-normal outcome (``normal``/``ok`` are the nominal classes). The
failure share is high (~72% over all five problems; per-problem 55-79%),
which this loader reports rather than hides: evaluate per problem with the
contamination in mind (``benchmarks/robot_failures_benchmark.py``).

A subtlety the raw files confirm (their numeric rows are byte-identical):
**lp3 relabels lp2's 47 transfer executions** by the part's resulting
position instead of the collision direction. Each labeling is genuine, but
pooling both would double-count those 47 runs, so the default
:attr:`DEFAULT_PROBLEMS` excludes lp3; pass
``preprocessing={"problems": RobotExecutionFailuresLoader.PROBLEMS}`` for
all five views.

Force/torque sensing is the contact modality of manipulators and humanoid
limbs, which makes this small corpus a genuine - if compact - robotics
fault benchmark rather than a toy generator.

Security note: the published archive also ships a compiled ``a.out`` and
helper sources; this loader reads **only** the ``lp*.data`` text members
and never writes or executes anything else from the archive.
"""

from __future__ import annotations

import hashlib
import io
import logging
import zipfile
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

#: Samples per execution and force/torque channels per sample.
_SAMPLES_PER_RUN = 15
_CHANNELS = 6

#: Learning problem -> its nominal (non-failure) class name.
_NOMINAL_CLASS: dict[str, str] = {
    "lp1": "normal",
    "lp2": "normal",
    "lp3": "ok",
    "lp4": "normal",
    "lp5": "normal",
}


def parse_lp_file(text: str, problem: str) -> tuple[np.ndarray[Any, Any], list[str]]:
    """Parse one ``lp*.data`` file into instance matrices and class names.

    Args:
        text: Raw file contents: blank-line-separated blocks of a class
            name followed by 15 tab-separated rows of 6 integers.
        problem: Problem key (``lp1``..``lp5``), for diagnostics.

    Returns:
        ``(features, class_names)``: features has one row per execution,
        the 15x6 force/torque samples flattened to 90 columns in time
        order; ``class_names`` is the per-row outcome label.

    Raises:
        ValueError: A block does not have 15 rows of 6 numeric values.
    """
    features: list[np.ndarray[Any, Any]] = []
    class_names: list[str] = []
    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    for block in blocks:
        lines = block.splitlines()
        if len(lines) != _SAMPLES_PER_RUN + 1:
            raise ValueError(
                f"{problem}: block starting {lines[0]!r} has {len(lines) - 1} "
                f"sample rows, expected {_SAMPLES_PER_RUN}"
            )
        class_names.append(lines[0].strip())
        rows = []
        for line in lines[1:]:
            parts = line.split()
            if len(parts) != _CHANNELS:
                raise ValueError(
                    f"{problem}: sample row has {len(parts)} values, expected {_CHANNELS}"
                )
            rows.append([float(value) for value in parts])
        features.append(np.asarray(rows, dtype=np.float64).ravel())
    return np.asarray(features, dtype=np.float64), class_names


class RobotExecutionFailuresLoader(DatasetLoader):
    """UCI Robot Execution Failures with observed-outcome ground truth.

    Note:
        The outcome annotation is genuine, so there is **no synthetic
        fallback**: fabricating execution outcomes would forge ground
        truth. Without cache or the pinned download this loader refuses
        loudly.
    """

    DATASET_NAME = "robot_failures"
    LABEL_SOURCE = "ground_truth"
    DATASET_URL = "https://archive.ics.uci.edu/dataset/138/robot+execution+failures"
    LICENSE = "CC BY 4.0"
    CITATION = """Lopes, L. S., & Camarinha-Matos, L. M. (1998). Robot Execution
    Failures [Dataset]. UCI Machine Learning Repository.
    https://doi.org/10.24432/C5M89N"""
    REQUIRES_CREDENTIALS = False

    ARCHIVE_URL = TrustedEndpoints.UCI_ROBOT_FAILURES
    #: SHA-256 of the UCI static archive, pinned from a verified download.
    ARCHIVE_SHA256 = "cdfbe0b260f4a21c5249fe0da4b1baf4041959663a9182253885ee1f41a4568e"

    PROBLEMS: tuple[str, ...] = ("lp1", "lp2", "lp3", "lp4", "lp5")
    #: Default pooled view: lp3 is excluded because it relabels lp2's 47
    #: transfer executions (identical force/torque rows), and pooling both
    #: would double-count those runs. Request it explicitly for the
    #: part-position labeling.
    DEFAULT_PROBLEMS: tuple[str, ...] = ("lp1", "lp2", "lp4", "lp5")

    #: 15 samples x (Fx, Fy, Fz, Tx, Ty, Tz), flattened in time order.
    FEATURE_NAMES = [
        f"{channel}_t{t:02d}"
        for t in range(_SAMPLES_PER_RUN)
        for channel in ("fx", "fy", "fz", "tx", "ty", "tz")
    ]

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize the instance."""
        super().__init__(config)
        problems = config.preprocessing.get("problems", self.DEFAULT_PROBLEMS)
        unknown = set(problems) - set(self.PROBLEMS)
        if unknown:
            raise ValueError(f"unknown problems {sorted(unknown)}; choose from {self.PROBLEMS}")
        self.problems: tuple[str, ...] = tuple(p for p in self.PROBLEMS if p in set(problems))
        self._is_real_data = False

    @property
    def is_real_data(self) -> bool:
        """Return True if real data was loaded."""
        return self._is_real_data

    def _real_cache_name(self) -> str:
        """Cache name carries the problem subset: labels are baked in."""
        return f"robot_failures_real_{'-'.join(self.problems)}.npz"

    def download(self) -> bool:
        """Fetch the pinned UCI archive and cache the parsed matrices.

        Returns:
            True once the cache exists.

        Raises:
            DataSourceUnavailableError: The pinned download failed; there
                is deliberately no synthetic path.
            ValueError: The archive does not match the pinned SHA-256 or a
                member is malformed.
        """
        cache_file = self.data_path / self._real_cache_name()
        if cache_file.exists():
            self._is_real_data = True
            return True
        TrustedEndpoints.validate_url(self.ARCHIVE_URL)
        try:
            body = http_get_with_retry(
                self.ARCHIVE_URL, timeout=120, timeout_per_attempt=60, retries=1
            )
        except Exception as exc:
            raise DataSourceUnavailableError(
                loader_name="RobotExecutionFailures",
                source_url=self.ARCHIVE_URL,
                reason=f"UCI archive fetch failed: {exc}",
            ) from exc
        digest = hashlib.sha256(body).hexdigest()
        if digest != self.ARCHIVE_SHA256:
            raise ValueError(
                f"{self.DATASET_NAME}: archive digest {digest} does not match "
                f"pinned {self.ARCHIVE_SHA256}; refusing to parse"
            )
        features_parts: list[np.ndarray[Any, Any]] = []
        labels_parts: list[np.ndarray[Any, Any]] = []
        problem_parts: list[np.ndarray[Any, Any]] = []
        class_parts: list[str] = []
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            members = {Path(member).name: member for member in archive.namelist()}
            for problem in self.problems:
                member = members.get(f"{problem}.data")
                if member is None:
                    raise ValueError(f"{self.DATASET_NAME}: archive lacks {problem}.data")
                # Text members only; the archive's a.out binary and helper
                # sources are never read.
                text = archive.read(member).decode("ascii", errors="strict")
                features, class_names = parse_lp_file(text, problem)
                nominal = _NOMINAL_CLASS[problem]
                labels = np.asarray(
                    [0 if name == nominal else 1 for name in class_names], dtype=np.int64
                )
                features_parts.append(features)
                labels_parts.append(labels)
                problem_parts.append(np.full(len(labels), problem))
                class_parts.extend(class_names)
        features_all = np.concatenate(features_parts)
        labels_all = np.concatenate(labels_parts)
        np.savez_compressed(
            cache_file,
            features=features_all,
            labels=labels_all,
            problems=np.concatenate(problem_parts),
            class_names=np.asarray(class_parts),
        )
        self._is_real_data = True
        logger.info(
            "Robot Execution Failures loaded: %d executions over %s, %d failures "
            "(%.0f%%) (is_real_data=True)",
            len(labels_all),
            "/".join(self.problems),
            int(labels_all.sum()),
            100.0 * float(labels_all.mean()),
        )
        return True

    def load_problem_arrays(
        self,
    ) -> dict[str, tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]]:
        """Return per-problem ``(features, labels, class_names)`` views.

        Returns:
            Mapping of problem key to its arrays, in :attr:`problems`
            order - the per-problem evaluation the class imbalance calls
            for.

        Raises:
            FileNotFoundError: The cache does not exist; call
                :meth:`download` first.
        """
        cache_file = self.data_path / self._real_cache_name()
        if not cache_file.exists():
            raise FileNotFoundError(
                f"No cached robot-failures data under {self.data_path}; call download() first."
            )
        data = np.load(cache_file, allow_pickle=False)
        problems = data["problems"]
        self._is_real_data = True
        return {
            problem: (
                data["features"][problems == problem],
                data["labels"][problems == problem],
                data["class_names"][problems == problem],
            )
            for problem in self.problems
        }

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Load the cached matrix (all requested problems concatenated)."""
        cache_file = self.data_path / self._real_cache_name()
        if not cache_file.exists():
            raise FileNotFoundError(
                f"No cached robot-failures data under {self.data_path}; call download() first."
            )
        data = np.load(cache_file, allow_pickle=False)
        self._is_real_data = True
        return data["features"], data["labels"]

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """No-op: the 90-column layout is the published representation."""
        return data


# Register at import so DatasetRegistry-driven callers can construct by name.
from omni_mercury_engine.datasets.base import DatasetRegistry

DatasetRegistry.register("robot_failures", RobotExecutionFailuresLoader)
