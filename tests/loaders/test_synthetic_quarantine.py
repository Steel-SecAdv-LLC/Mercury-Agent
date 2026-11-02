# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Synthetic frames are quarantined from de-leaked measurements.

The energy loader's synthetic historical series builds solar-wind speed as
``350 + 50 * kp``, density as ``10 - 0.5 * kp`` and flare classes from the
same Kp profile that builds the ``kp >= 7`` labels, so dropping the five
declared Kp columns still leaves a matrix that is a function of the label.
Such a frame now carries a synthetic mark that survives the loader's cache,
and ``engineer_scored_features`` refuses it instead of reporting a disjoint
matrix; ``engineer_features`` still returns the full matrix for display.
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
    monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "1")
    monkeypatch.delenv("EIA_API_KEY", raising=False)
    loader = EnergyLoader(cache_dir=tmp_path)
    frame = loader.fetch_historical("quebec_1989")
    assert frame.attrs.get(SYNTHETIC_ATTR) is True
    full = loader.engineer_features(frame)
    assert full.shape == (len(frame), len(EnergyLoader.FEATURE_COLUMNS))
    with pytest.raises(SyntheticDataNotDeleakable, match="synthetic"):
        loader.engineer_scored_features(frame)


def test_the_mark_survives_the_loader_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second process reads the cached frame: its provenance must come back with it."""
    monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "1")
    monkeypatch.delenv("EIA_API_KEY", raising=False)
    EnergyLoader(cache_dir=tmp_path).fetch_historical("quebec_1989")
    monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "0")  # the cache, not a fresh synthesis, answers
    again = EnergyLoader(cache_dir=tmp_path)
    cached = again.fetch_historical("quebec_1989")
    assert again.is_synthetic(cached)
    with pytest.raises(SyntheticDataNotDeleakable):
        again.engineer_scored_features(cached)


def test_a_pre_provenance_cache_entry_is_not_trusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare column-dict payload (the old shape) has unknown provenance and is refetched."""
    monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "1")
    monkeypatch.delenv("EIA_API_KEY", raising=False)
    loader = EnergyLoader(cache_dir=tmp_path)
    frame = _real_frame()
    # The old shape, as the previous writer produced it: a column dict with
    # timestamps already rendered as text (``_write_cache`` writes nothing for
    # a payload JSON cannot serialise, which would make this test vacuous).
    legacy = frame.assign(timestamp=frame["timestamp"].astype(str)).to_dict(orient="list")
    loader._write_cache("energy_historical_quebec_1989", legacy)
    assert loader._read_cache("energy_historical_quebec_1989") == legacy
    assert loader._read_frame_cache("energy_historical_quebec_1989") is None
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
