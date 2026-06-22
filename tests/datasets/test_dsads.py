# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the DSADS (UCI 256) anomaly-detection loader.

The live archive is ~170 MB / 9120 segments, so these tests exercise the
parsing, featurisation, labelling, validation and delegate wiring against a
small synthetic archive with the *real* segment shape (125 x 45) and the
class constants monkeypatched down to a tiny structure. The full real-data
path (9120 x 405) is covered by the network-gated benchmark, not here.
"""

from __future__ import annotations

import zipfile
from typing import TYPE_CHECKING

import numpy as np
import pytest

from omni_mercury_engine.datasets.adrepository import ADRepositoryLoader
from omni_mercury_engine.datasets.base import DatasetConfig
from omni_mercury_engine.datasets.timeseries import DSADSLoader

if TYPE_CHECKING:
    from pathlib import Path


def _make_dsads_zip(
    path: Path, n_act: int, n_subj: int, n_seg: int, *, rows: int = 125, cols: int = 45
) -> None:
    """Write a synthetic DSADS archive mirroring ``data/aNN/pN/sNN.txt`` layout."""
    rng = np.random.default_rng(0)
    with zipfile.ZipFile(path, "w") as zf:
        for a in range(1, n_act + 1):
            for p in range(1, n_subj + 1):
                for s in range(1, n_seg + 1):
                    seg = rng.standard_normal((rows, cols))
                    txt = "\n".join(",".join(f"{v:.6f}" for v in row) for row in seg)
                    zf.writestr(f"data/a{a:02d}/p{p}/s{s:02d}.txt", txt)


def _tiny_loader(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **prep: object) -> DSADSLoader:
    """A DSADSLoader over a 2-activity x 1-subject x 2-segment synthetic archive."""
    monkeypatch.setattr(DSADSLoader, "N_ACTIVITIES", 2)
    monkeypatch.setattr(DSADSLoader, "N_SUBJECTS", 1)
    monkeypatch.setattr(DSADSLoader, "N_SEGMENTS", 2)
    (tmp_path / "dsads").mkdir(parents=True, exist_ok=True)
    _make_dsads_zip(tmp_path / "dsads" / "dsads_data.zip", 2, 1, 2)
    cfg = DatasetConfig(
        name="dsads",
        data_dir=str(tmp_path),
        cache_dir=str(tmp_path / "c"),
        preprocessing=prep or {"anomaly_activities": [2]},
    )
    return DSADSLoader(cfg)


class TestSegmentFeatures:
    """The 125 x 45 -> 405 statistical reduction is deterministic and correct."""

    def test_dimension_is_405(self) -> None:
        seg = np.random.default_rng(1).standard_normal((125, 45))
        feats = DSADSLoader._segment_features(seg)
        assert feats.shape == (405,)  # 9 stats x 45 channels

    def test_statistics_are_correct(self) -> None:
        # Column c is the constant c, so every per-channel stat is analytic.
        seg = np.tile(np.arange(45, dtype=np.float64), (125, 1))
        feats = DSADSLoader._segment_features(seg).reshape(9, 45)
        mean, std, mn, mx, med, q25, q75, ptp, rms = feats
        np.testing.assert_allclose(mean, np.arange(45))
        np.testing.assert_allclose(std, 0.0, atol=1e-9)
        np.testing.assert_allclose(mn, np.arange(45))
        np.testing.assert_allclose(mx, np.arange(45))
        np.testing.assert_allclose(med, np.arange(45))
        np.testing.assert_allclose(ptp, 0.0, atol=1e-9)
        np.testing.assert_allclose(rms, np.arange(45))  # constant column -> rms == value


class TestLoadRaw:
    """End-to-end parse of the synthetic archive."""

    def test_shapes_and_constructed_labels(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        loader = _tiny_loader(monkeypatch, tmp_path, anomaly_activities=[2])
        X, y = loader._load_raw()
        assert X.shape == (4, 405)  # 2 act x 1 subj x 2 seg, 405 features
        # sorted members: a01/s01, a01/s02, a02/s01, a02/s02 -> activity 2 anomalous
        assert y.tolist() == [0, 0, 1, 1]
        assert X.dtype == np.float64 and y.dtype == np.int64

    def test_anomaly_convention_is_configurable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        loader = _tiny_loader(monkeypatch, tmp_path, anomaly_activities=[1])
        _, y = loader._load_raw()
        assert y.tolist() == [1, 1, 0, 0]  # activity 1 now anomalous

    def test_incomplete_archive_fails_loud(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(DSADSLoader, "N_ACTIVITIES", 2)
        monkeypatch.setattr(DSADSLoader, "N_SUBJECTS", 1)
        monkeypatch.setattr(DSADSLoader, "N_SEGMENTS", 2)
        (tmp_path / "dsads").mkdir(parents=True)
        _make_dsads_zip(tmp_path / "dsads" / "dsads_data.zip", 2, 1, 1)  # only 2 of 4 segments
        cfg = DatasetConfig(
            name="dsads", data_dir=str(tmp_path), preprocessing={"anomaly_activities": [2]}
        )
        with pytest.raises(ValueError, match="layout unexpected"):
            DSADSLoader(cfg)._load_raw()


class TestValidationAndWiring:
    """Convention validation and the ADRepository delegate routing."""

    def test_invalid_anomaly_activity_rejected(self, tmp_path: Path) -> None:
        cfg = DatasetConfig(
            name="dsads", data_dir=str(tmp_path), preprocessing={"anomaly_activities": [99]}
        )
        with pytest.raises(ValueError, match="non-empty subset"):
            DSADSLoader(cfg)

    def test_empty_anomaly_activities_rejected(self, tmp_path: Path) -> None:
        cfg = DatasetConfig(
            name="dsads", data_dir=str(tmp_path), preprocessing={"anomaly_activities": []}
        )
        with pytest.raises(ValueError, match="non-empty subset"):
            DSADSLoader(cfg)

    def test_dsads_is_delegated_not_failloud(self) -> None:
        # dsads moved from the fail-loud set to a real dedicated loader.
        assert "dsads" in ADRepositoryLoader._TIMESERIES_DELEGATES
        assert "dsads" not in ADRepositoryLoader._TIMESERIES_NO_LOADER
        assert ADRepositoryLoader._TIMESERIES_DELEGATES["dsads"][1] == "DSADSLoader"

    def test_delegate_loads_real_via_cached_archive(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(DSADSLoader, "N_ACTIVITIES", 2)
        monkeypatch.setattr(DSADSLoader, "N_SUBJECTS", 1)
        monkeypatch.setattr(DSADSLoader, "N_SEGMENTS", 2)
        (tmp_path / "dsads").mkdir(parents=True)
        _make_dsads_zip(tmp_path / "dsads" / "dsads_data.zip", 2, 1, 2)
        cfg = DatasetConfig(
            name="dsads",
            data_dir=str(tmp_path),
            cache_dir=str(tmp_path / "c"),
            preprocessing={"anomaly_activities": [2]},
        )
        delegate = ADRepositoryLoader(cfg, dataset_name="dsads")
        X, y = delegate._load_raw()
        assert X.shape == (4, 405)
        assert delegate.is_real_data is True
