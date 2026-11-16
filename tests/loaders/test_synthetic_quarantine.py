# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Synthetic frames are quarantined from de-leaked measurements.

The energy loader's synthetic historical series builds solar-wind speed as
``350 + 50 * kp``, density as ``10 - 0.5 * kp`` and flare classes from the
same Kp profile that builds the ``kp >= 7`` labels, so dropping the five
declared Kp columns still leaves a matrix that is a function of the label.
Such a frame carries a synthetic mark, and ``engineer_scored_features``
refuses it instead of reporting a disjoint matrix; ``engineer_features``
still returns the full matrix for display. Since the loader reads recorded
history (OMNI2, EIA-930), the synthetic series is reached only when that
archive is unreachable and the opt-in is set, and it is never cached.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from omni_mercury_engine.loaders.base import SYNTHETIC_ATTR, SyntheticDataNotDeleakable
from omni_mercury_engine.loaders.energy_loader import EnergyLoader
from omni_mercury_engine.loaders.pandemic_loader import PandemicLoader

if TYPE_CHECKING:
    from pathlib import Path

_GET_BYTES = "omni_mercury_engine.loaders.base.SafeHTTPClient.get_bytes"


def _unreachable(*_args: object, **_kwargs: object) -> bytes:
    raise ConnectionError("simulated: archive unreachable")


def _real_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 64
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-05-10", periods=n, freq="3h", tz="UTC"),
            "kp": rng.integers(0, 9, size=n).astype(float),
            "solar_wind_speed": rng.uniform(300, 800, size=n),
            "solar_wind_density": rng.uniform(1, 20, size=n),
            "xray_class": rng.integers(1, 6, size=n).astype(float),
        }
    )


def test_synthetic_energy_history_is_marked_and_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stand-in, now only reached when the recorded archive cannot be."""
    monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "1")
    monkeypatch.setattr(_GET_BYTES, _unreachable)
    loader = EnergyLoader(cache_dir=tmp_path, max_retries=0)
    frame = loader.fetch_historical("quebec_1989")
    assert frame.attrs.get(SYNTHETIC_ATTR) is True
    assert (frame["series_source"] == "synthetic").all()
    full = loader.engineer_features(frame)
    assert full.shape == (len(frame), len(EnergyLoader.FEATURE_COLUMNS))
    with pytest.raises(SyntheticDataNotDeleakable, match="synthetic"):
        loader.engineer_scored_features(frame)


def test_the_stand_in_is_never_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Was: cached, and then served to a process that had not opted in.

    The mark did survive the cache, but so did the frame: with
    ``MERCURY_ALLOW_SYNTHETIC=0`` the second loader still returned synthetic
    data because the cache answered. A stand-in is not stored; without the
    opt-in the second loader is refused.
    """
    from omni_mercury_engine.datasets.exceptions import DataSourceUnavailableError

    monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "1")
    monkeypatch.setattr(_GET_BYTES, _unreachable)
    EnergyLoader(cache_dir=tmp_path, max_retries=0).fetch_historical("quebec_1989")
    assert list(tmp_path.glob("*.json")) == []

    monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "0")
    with pytest.raises(DataSourceUnavailableError, match="quebec_1989"):
        EnergyLoader(cache_dir=tmp_path, max_retries=0).fetch_historical("quebec_1989")


def test_a_pre_provenance_cache_entry_is_not_trusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare column-dict payload (the old shape) has unknown provenance and is refetched."""
    monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "1")
    monkeypatch.setattr(_GET_BYTES, _unreachable)
    loader = EnergyLoader(cache_dir=tmp_path, max_retries=0)
    frame = _real_frame()
    # The old shape, as the previous writer produced it: a column dict with
    # timestamps already rendered as text (``_write_cache`` writes nothing for
    # a payload JSON cannot serialise, which would make this test vacuous).
    legacy = frame.assign(timestamp=frame["timestamp"].astype(str)).to_dict(orient="list")
    for key in ("energy_historical_quebec_1989", "energy_historical_v4_quebec_1989"):
        loader._write_cache(key, legacy)
        assert loader._read_cache(key) == legacy
        assert loader._read_frame_cache(key) is None
    frame = loader.fetch_historical("quebec_1989")
    assert loader.is_synthetic(frame)


def test_fetched_frames_still_get_the_deleaked_matrix(tmp_path: Path) -> None:
    loader = EnergyLoader(cache_dir=tmp_path)
    frame = _real_frame()
    assert not loader.is_synthetic(frame)
    scored = loader.engineer_scored_features(frame)
    assert scored.shape == (len(frame), len(EnergyLoader.scored_feature_columns()))


def test_pandemic_synthetic_series_is_marked(tmp_path: Path) -> None:
    loader = PandemicLoader(cache_dir=tmp_path)
    frame = loader.mark_synthetic(loader._synthetic_ebola_2014())
    assert loader.is_synthetic(frame)
    assert loader.is_synthetic(frame.sort_values(frame.columns[0]).reset_index(drop=True))
