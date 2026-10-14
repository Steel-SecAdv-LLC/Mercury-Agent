# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""DTU AIS loader: restricted unpickling, feature contract, digest gate."""

from __future__ import annotations

import io
import pickle
import zipfile
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from omni_mercury_engine.datasets.base import DatasetConfig
from omni_mercury_engine.datasets.maritime import (
    FEATURE_NAMES,
    DTUAISLoader,
    restricted_pickle_load,
    track_features,
)


def _make_track(
    n: int, *, speed: float = 8.0, lat0: float = 55.0, lon0: float = 14.0
) -> dict[str, Any]:
    ts = 1_639_369_092.0 + 120.0 * np.arange(n)
    return {
        "mmsi": 209_183_000,
        "shiptype": 60,
        "track_length": n,
        "lat": (lat0 + 0.008 * np.arange(n)).tolist(),
        "lon": (lon0 + 0.004 * np.arange(n)).tolist(),
        "speed": np.full(n, speed).tolist(),
        "course": np.full(n, 332.0).tolist(),
        "timestamp": ts.tolist(),
    }


def _write_fixture(data_dir: Path, tracks: list[dict[str, Any]], labels: list[int]) -> None:
    """Write DTU-format sequential pickles + datasetInfo into ``data_dir``."""
    target = data_dir / DTUAISLoader.DATASET_NAME
    target.mkdir(parents=True, exist_ok=True)
    offsets: list[int] = []
    blob = io.BytesIO()
    for track in tracks:
        offsets.append(blob.tell())
        pickle.dump(track, blob, protocol=4)
    (target / DTUAISLoader.DATA_MEMBER).write_bytes(blob.getvalue())
    info = {"indicies": offsets, "outlierLabels": np.asarray(labels, dtype=np.int32)}
    (target / DTUAISLoader.INFO_MEMBER).write_bytes(pickle.dumps(info, protocol=4))


def test_load_raw_features_and_label_alignment(tmp_path: Path) -> None:
    tracks = [_make_track(30), _make_track(12, speed=0.2), _make_track(50)]
    _write_fixture(tmp_path, tracks, labels=[0, 1, 0])
    loader = DTUAISLoader(
        DatasetConfig(name="t", data_dir=str(tmp_path), cache_dir=str(tmp_path / "c"))
    )
    X, y = loader._load_raw()
    assert X.shape == (3, len(FEATURE_NAMES))
    assert y.tolist() == [0, 1, 0]
    # Loitering track: high stationary fraction, labelled row matches.
    frac_slow = X[1, FEATURE_NAMES.index("frac_below_half_knot")]
    assert frac_slow == 1.0


def test_track_features_are_physical() -> None:
    vec = track_features(_make_track(40))
    named = dict(zip(FEATURE_NAMES, vec.tolist(), strict=True))
    assert 0.0 <= named["straightness"] <= 1.0 + 1e-9
    assert named["n_points"] == 40.0
    assert named["path_len_km"] >= named["net_disp_km"] - 1e-9
    assert 0.0 <= named["course_resultant"] <= 1.0 + 1e-9
    single = track_features(_make_track(1))
    assert np.all(np.isfinite(single))


def test_malicious_pickle_is_refused() -> None:
    class _Evil:
        def __reduce__(self) -> tuple[Any, tuple[str]]:
            import os

            return (os.system, ("echo pwned",))

    payload = pickle.dumps(_Evil(), protocol=4)
    with pytest.raises(pickle.UnpicklingError, match="forbidden global"):
        restricted_pickle_load(io.BytesIO(payload))


def test_download_refuses_unpinned_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tampered archive members must never reach the cache."""
    blob = io.BytesIO()
    with zipfile.ZipFile(blob, "w") as archive:
        archive.writestr(DTUAISLoader.INFO_MEMBER, b"not the pinned bytes")
        archive.writestr(DTUAISLoader.DATA_MEMBER, b"also not the pinned bytes")

    monkeypatch.setattr(
        "omni_mercury_engine.datasets.maritime.http_get_with_retry",
        lambda url, **kwargs: blob.getvalue(),
    )
    loader = DTUAISLoader(
        DatasetConfig(name="t", data_dir=str(tmp_path), cache_dir=str(tmp_path / "c"))
    )
    with pytest.raises(ValueError, match=r"does not\s+match pinned"):
        loader.download()
    assert not (tmp_path / DTUAISLoader.DATASET_NAME / DTUAISLoader.DATA_MEMBER).exists()


def test_label_source_is_expert_annotated() -> None:
    from omni_mercury_engine.datasets.metadata import is_supervised_eval_safe

    assert DTUAISLoader.LABEL_SOURCE == "expert_annotated"
    assert is_supervised_eval_safe(DTUAISLoader.LABEL_SOURCE)
