# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""UCI Robot Execution Failures loader: parsing, labels, and the lp2/lp3 trap.

The end-to-end tests feed the loader a pinned in-memory zip (its digest
patched to match) so the full download -> verify -> parse -> cache -> load
path runs without network access.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from omni_mercury_engine.datasets.base import DatasetConfig
from omni_mercury_engine.datasets.robot_failures import (
    RobotExecutionFailuresLoader,
    parse_lp_file,
)


def _block(class_name: str, fill: int) -> str:
    rows = "\n".join("\t" + "\t".join(str(fill + c) for c in range(6)) for _ in range(15))
    return f"{class_name}\n{rows}"


def _lp_text(*blocks: tuple[str, int]) -> str:
    return "\n\n".join(_block(name, fill) for name, fill in blocks) + "\n"


class TestParseLPFile:
    def test_parses_blocks_in_order(self) -> None:
        text = _lp_text(("normal", 1), ("collision", 50))
        features, names = parse_lp_file(text, "lp1")
        assert features.shape == (2, 90)
        assert names == ["normal", "collision"]
        assert features[0, 0] == 1.0 and features[1, 0] == 50.0

    def test_wrong_row_count_raises(self) -> None:
        text = "normal\n" + "\n".join("\t1\t2\t3\t4\t5\t6" for _ in range(14))
        with pytest.raises(ValueError, match="sample rows"):
            parse_lp_file(text, "lp1")

    def test_wrong_column_count_raises(self) -> None:
        text = "normal\n" + "\n".join("\t1\t2\t3\t4\t5" for _ in range(15))
        with pytest.raises(ValueError, match="values"):
            parse_lp_file(text, "lp1")


def _fixture_zip() -> bytes:
    """A miniature archive mirroring the real layout, lp2/lp3 trap included."""
    transfer_signals = [("x", 10), ("x", 11), ("x", 90)]  # shared rows
    files = {
        "lp1.data": _lp_text(("normal", 1), ("collision", 80), ("obstruction", 70)),
        # lp2 and lp3 share numeric rows, relabelled - as in the real data.
        "lp2.data": _lp_text(
            *[(n, f) for (_, f), n in zip(transfer_signals, ("normal", "normal", "front_col"))]
        ),
        "lp3.data": _lp_text(
            *[(n, f) for (_, f), n in zip(transfer_signals, ("ok", "ok", "moved"))]
        ),
        "lp4.data": _lp_text(("normal", 2), ("collision", 60)),
        "lp5.data": _lp_text(("normal", 3), ("collision_in_tool", 55)),
        "a.out": "\x7fELF-not-to-be-read",
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return buffer.getvalue()


def _loader(tmp_path: Any, **preprocessing: Any) -> RobotExecutionFailuresLoader:
    config = DatasetConfig(
        name="robot_failures",
        data_dir=str(tmp_path / "data"),
        cache_dir=str(tmp_path / "cache"),
        preprocessing=preprocessing,
    )
    return RobotExecutionFailuresLoader(config)


def _download_with_fixture(loader: RobotExecutionFailuresLoader, body: bytes) -> None:
    with (
        patch.object(
            RobotExecutionFailuresLoader,
            "ARCHIVE_SHA256",
            hashlib.sha256(body).hexdigest(),
        ),
        patch(
            "omni_mercury_engine.datasets.robot_failures.http_get_with_retry",
            return_value=body,
        ),
    ):
        assert loader.download()


class TestLoaderEndToEnd:
    def test_labels_map_nominal_to_zero_per_problem(self, tmp_path: Any) -> None:
        loader = _loader(tmp_path, problems=RobotExecutionFailuresLoader.PROBLEMS)
        _download_with_fixture(loader, _fixture_zip())
        arrays = loader.load_problem_arrays()
        assert arrays["lp1"][1].tolist() == [0, 1, 1]
        assert arrays["lp3"][1].tolist() == [0, 0, 1]  # "ok" is lp3's nominal
        assert loader.is_real_data

    def test_default_pooled_view_excludes_the_lp3_relabeling(self, tmp_path: Any) -> None:
        """lp3 duplicates lp2's signals; the default must not double-count."""
        loader = _loader(tmp_path)
        assert loader.problems == ("lp1", "lp2", "lp4", "lp5")
        _download_with_fixture(loader, _fixture_zip())
        features, labels = loader._load_raw()
        assert features.shape == (3 + 3 + 2 + 2, 90)
        assert labels.sum() == 2 + 1 + 1 + 1

    def test_digest_mismatch_refuses_to_parse(self, tmp_path: Any) -> None:
        loader = _loader(tmp_path)
        with (
            patch(
                "omni_mercury_engine.datasets.robot_failures.http_get_with_retry",
                return_value=_fixture_zip(),
            ),
            pytest.raises(ValueError, match="digest"),
        ):
            loader.download()

    def test_unknown_problem_rejected_at_construction(self, tmp_path: Any) -> None:
        with pytest.raises(ValueError, match="unknown problems"):
            _loader(tmp_path, problems=("lp1", "lp9"))

    def test_cache_name_carries_problem_subset(self, tmp_path: Any) -> None:
        assert "lp1-lp2-lp4-lp5" in _loader(tmp_path)._real_cache_name()
        full = _loader(tmp_path, problems=RobotExecutionFailuresLoader.PROBLEMS)
        assert "lp1-lp2-lp3-lp4-lp5" in full._real_cache_name()

    def test_feature_names_match_width(self, tmp_path: Any) -> None:
        loader = _loader(tmp_path)
        _download_with_fixture(loader, _fixture_zip())
        features, _ = loader._load_raw()
        assert features.shape[1] == len(RobotExecutionFailuresLoader.FEATURE_NAMES)
        assert np.isfinite(features).all()
