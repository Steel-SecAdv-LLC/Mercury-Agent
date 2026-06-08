# Copyright (C) 2025 Steel Security Advisors LLC
"""Offline regression tests for ADBench dataset selection.

Guards two bugs where every ADBench dataset name silently collapsed to the
``fraud`` default:

1. ``ADBenchLoader.__init__`` only read ``preprocessing['dataset']`` and
   ignored ``config.name`` / the ``adbench-<name>`` registry key.
2. The base ``_check_data_exists`` is directory-level; since all ADBench
   datasets share one ``adbench/`` directory, a single cached file made
   ``load()`` skip the download for every other dataset.

All assertions are offline — dataset *resolution* and existence checks happen
without any network access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from omni_mercury_engine.datasets.adbench import ADBenchLoader
from omni_mercury_engine.datasets.base import DatasetConfig, DatasetRegistry

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("config_name", "expected_name", "expected_index"),
    [
        ("adbench-cardio", "cardio", 6),
        ("adbench-WBC", "WBC", 42),
        ("adbench_pima", "Pima", 29),
        ("cardio", "cardio", 6),
        ("30", "satellite", 30),
        ("adbench", "fraud", 13),  # bare alias keeps the historical default
        ("totally-unknown", "fraud", 13),  # unrecognised name falls back
    ],
)
def test_name_resolves_to_dataset(
    config_name: str, expected_name: str, expected_index: int, tmp_path: Path
) -> None:
    loader = ADBenchLoader(DatasetConfig(name=config_name, data_dir=str(tmp_path)))
    assert loader._dataset_name == expected_name
    assert loader._dataset_index == expected_index


def test_explicit_preprocessing_overrides_name(tmp_path: Path) -> None:
    loader = ADBenchLoader(
        DatasetConfig(
            name="adbench-cardio",
            data_dir=str(tmp_path),
            preprocessing={"dataset": "WBC"},
        )
    )
    assert loader._dataset_name == "WBC"


def test_registry_creates_distinct_loaders(tmp_path: Path) -> None:
    names = ["adbench-cardio", "adbench-wbc", "adbench-pima"]
    resolved: list[tuple[int, str, str]] = []
    for name in names:
        loader = DatasetRegistry.create(name, DatasetConfig(name=name, data_dir=str(tmp_path)))
        assert isinstance(loader, ADBenchLoader)
        resolved.append((loader._dataset_index, loader._dataset_name, loader.npz_filename))
    # No two registry names collapse to the same NPZ.
    assert len({r[2] for r in resolved}) == len(names)


def test_check_data_exists_is_file_specific(tmp_path: Path) -> None:
    """A cached file for one dataset must not mark another as present."""
    cardio = ADBenchLoader(DatasetConfig(name="adbench-cardio", data_dir=str(tmp_path)))
    wbc = ADBenchLoader(DatasetConfig(name="adbench-wbc", data_dir=str(tmp_path)))

    assert not cardio._check_data_exists()
    cardio.data_path.mkdir(parents=True, exist_ok=True)
    np.savez(cardio.data_path / cardio.npz_filename, X=np.zeros((2, 2)), y=np.zeros(2))

    assert cardio._check_data_exists()
    # WBC shares the directory but its own NPZ is absent.
    assert not wbc._check_data_exists()
