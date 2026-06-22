# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the Epilepsy (Bonn EEG, Andrzejak et al. 2001) loader.

The official UPF source is Cloudflare-gated, so the loader takes the official
data via a local path (``preprocessing['bonn_dir']``) and reconstructs the
canonical 11500 x 178 tabular form. These tests exercise that reconstruction,
the labelling (set S = seizure = anomaly), both archive layouts, the fail-loud
paths and the delegate wiring against a small synthetic fixture that keeps the
real 4097 -> 23 x 178 chunking (``FILES_PER_SET`` monkeypatched down for speed).
"""

from __future__ import annotations

import zipfile
from typing import TYPE_CHECKING

import numpy as np
import pytest

from omni_mercury_engine.datasets.adrepository import ADRepositoryLoader
from omni_mercury_engine.datasets.base import DatasetConfig
from omni_mercury_engine.datasets.exceptions import DataSourceUnavailableError
from omni_mercury_engine.datasets.timeseries import EpilepsyLoader

if TYPE_CHECKING:
    from pathlib import Path


def _write_set(
    base: Path, set_name: str, n_files: int, *, n_samples: int = 4097, layout: str = "zip"
) -> None:
    rng = np.random.default_rng(ord(set_name))
    files = {
        f"{set_name}{i:03d}.txt": "\n".join(str(int(v)) for v in rng.integers(-300, 300, n_samples))
        for i in range(1, n_files + 1)
    }
    if layout == "zip":
        with zipfile.ZipFile(base / f"{set_name}.zip", "w") as zf:
            for name, txt in files.items():
                zf.writestr(name, txt)
    else:
        directory = base / set_name
        directory.mkdir()
        for name, txt in files.items():
            (directory / name).write_text(txt)


def _make_bonn(tmp_path: Path, *, n_files: int = 2, layout: str = "zip") -> Path:
    base = tmp_path / "bonn"
    base.mkdir()
    for set_name in EpilepsyLoader.SETS:
        _write_set(base, set_name, n_files, layout=layout)
    return base


def _loader(tmp_path: Path, bonn: Path) -> EpilepsyLoader:
    cfg = DatasetConfig(
        name="epilepsy",
        data_dir=str(tmp_path),
        cache_dir=str(tmp_path / "c"),
        preprocessing={"bonn_dir": str(bonn)},
    )
    return EpilepsyLoader(cfg)


class TestReconstruction:
    """Canonical 11500x178-style reconstruction (scaled down via FILES_PER_SET)."""

    @pytest.mark.parametrize("layout", ["zip", "dir"])
    def test_shapes_and_seizure_labels(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, layout: str
    ) -> None:
        monkeypatch.setattr(EpilepsyLoader, "FILES_PER_SET", 2)
        bonn = _make_bonn(tmp_path, n_files=2, layout=layout)
        X, y = _loader(tmp_path, bonn)._load_raw()
        # 5 sets x 2 files x 23 segments = 230 rows, 178 features
        assert X.shape == (230, 178)
        # only set S (seizure) is anomalous: 2 files x 23 = 46 -> 0.20 ratio
        assert int(y.sum()) == 46
        assert abs(float(y.mean()) - 0.20) < 1e-9
        assert X.dtype == np.float64 and y.dtype == np.int64

    def test_segment_chunking_is_real_4097_to_23x178(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # A single 4097-sample recording must yield exactly 23 rows of 178.
        monkeypatch.setattr(EpilepsyLoader, "FILES_PER_SET", 1)
        bonn = _make_bonn(tmp_path, n_files=1)
        X, y = _loader(tmp_path, bonn)._load_raw()
        assert X.shape == (5 * 1 * 23, 178)  # 23 rows per set
        assert int(y.sum()) == 23  # set S


class TestFailLoud:
    """Never fabricate; fail loud naming the official source."""

    def test_download_without_bonn_dir_raises_naming_bonn(self, tmp_path: Path) -> None:
        cfg = DatasetConfig(name="epilepsy", data_dir=str(tmp_path))
        with pytest.raises(DataSourceUnavailableError) as exc:
            EpilepsyLoader(cfg).download()
        msg = str(exc.value)
        assert "Bonn" in msg and "bonn_dir" in msg

    def test_incomplete_set_fails_loud(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(EpilepsyLoader, "FILES_PER_SET", 3)
        bonn = _make_bonn(tmp_path, n_files=2)  # only 2 files/set, expects 3
        with pytest.raises(ValueError, match="expected 3 recordings"):
            _loader(tmp_path, bonn)._load_raw()

    def test_short_recording_fails_loud(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(EpilepsyLoader, "FILES_PER_SET", 1)
        base = tmp_path / "bonn"
        base.mkdir()
        for set_name in EpilepsyLoader.SETS:
            _write_set(base, set_name, 1, n_samples=1000)  # < 4094 needed
        with pytest.raises(ValueError, match="need >="):
            _loader(tmp_path, base)._load_raw()


class TestWiring:
    """Registry + delegate routing."""

    def test_epilepsy_is_delegated_not_failloud(self) -> None:
        assert "epilepsy" in ADRepositoryLoader._TIMESERIES_DELEGATES
        assert "epilepsy" not in ADRepositoryLoader._TIMESERIES_NO_LOADER
        assert ADRepositoryLoader._TIMESERIES_DELEGATES["epilepsy"][1] == "EpilepsyLoader"

    def test_delegate_loads_real_via_local_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(EpilepsyLoader, "FILES_PER_SET", 2)
        bonn = _make_bonn(tmp_path, n_files=2)
        cfg = DatasetConfig(
            name="epilepsy",
            data_dir=str(tmp_path),
            cache_dir=str(tmp_path / "c"),
            preprocessing={"bonn_dir": str(bonn)},
        )
        delegate = ADRepositoryLoader(cfg, dataset_name="epilepsy")
        X, y = delegate._load_raw()
        assert X.shape == (230, 178)
        assert delegate.is_real_data is True
        assert int(y.sum()) == 46
