"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Tests that the runtime refuses every legacy pickle ingress.

Pickle is removed from Mercury Agent's data flow.  These tests assert
that:

* ``datasets.base.DatasetLoader._load_and_cache`` refuses a pickle-backed
  cache file with ``RuntimeError`` (no silent fall-through).
* ``datasets.adrepository.ADRepositoryLoader._load_from_file`` refuses
  pickle-backed ``.npz`` archives the same way (both at the top level
  and inside an extracted zip).
* ``integrations.stubs.cache.RedisCache`` has no pickle plumbing left at
  all — no ``_get_signing_key``, no ``_restricted_loads``, no
  ``serializer="pickle"`` knob.
* ``tools.migrate_pkl._do_migration`` refuses to load pickle unless the
  hardened-subprocess sentinel is set.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# datasets/base.py — legacy cache refusal
# ---------------------------------------------------------------------------


def test_load_and_cache_refuses_pickle_backed_cache(tmp_path: Path) -> None:
    from omni_mercury_engine.datasets.base import DatasetConfig, DatasetLoader, DatasetSplit

    class _Loader(DatasetLoader):
        DATASET_NAME = "pickle-refusal"

        def download(self) -> bool:  # pragma: no cover - never reached
            return True

        def _load_raw(self):  # pragma: no cover - never reached
            raise RuntimeError("should not reach raw load — cache exists")

        def preprocess(self, data):  # pragma: no cover
            return data

    cfg = DatasetConfig(
        name="pickle-refusal",
        data_dir=str(tmp_path / "data"),
        cache_dir=str(tmp_path / "cache"),
        download=False,
    )
    loader = _Loader(cfg)

    # Plant a "legacy" cache file: an .npz that requires allow_pickle=True to load.
    cache_key = cfg.get_cache_key()
    cache_file = loader.cache_path / f"{cache_key}.npz"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_file,
        train_features=np.array([{"a": 1}], dtype=object),
        train_labels=np.array([0]),
        val_features=np.array([0]),
        val_labels=np.array([0]),
        test_features=np.array([0]),
        test_labels=np.array([0]),
    )
    assert cache_file.exists()

    with pytest.raises(RuntimeError, match="legacy pickle-backed cache"):
        loader.load(DatasetSplit.ALL)


# ---------------------------------------------------------------------------
# datasets/adrepository.py — pickle-backed .npz refusal
# ---------------------------------------------------------------------------


def test_adrepository_refuses_pickle_npz(tmp_path: Path) -> None:
    """A .npz that needs allow_pickle=True raises RuntimeError, no synthetic fallback."""
    from omni_mercury_engine.datasets.adrepository import ADRepositoryLoader
    from omni_mercury_engine.datasets.base import DatasetConfig

    cfg = DatasetConfig(
        name="thyroid",
        data_dir=str(tmp_path / "d"),
        cache_dir=str(tmp_path / "c"),
        download=False,
    )
    loader = ADRepositoryLoader(cfg, dataset_name="thyroid")
    # Hostile .npz that requires allow_pickle on load.
    bad = tmp_path / "thyroid.npz"
    np.savez(bad, X=np.array([{"a": 1}], dtype=object), y=np.array([0]))
    with pytest.raises(RuntimeError, match="Refusing to load pickle-backed"):
        loader._load_from_file(bad)


# ---------------------------------------------------------------------------
# integrations/stubs/cache.py — pickle surface fully removed
# ---------------------------------------------------------------------------


def test_redis_cache_has_no_pickle_surface() -> None:
    from omni_mercury_engine.integrations.stubs.cache import RedisCache

    # No helpers left over from the removed pickle path.
    assert not hasattr(RedisCache, "_get_signing_key")
    assert not hasattr(RedisCache, "_restricted_loads")

    # The serializer is JSON, period — no kwarg, no env override.
    cache = RedisCache(fallback_to_stub=True)
    assert cache.SERIALIZER == "json"
    assert cache.serializer == "json"

    # Passing serializer="pickle" must be a constructor error.
    with pytest.raises(TypeError):
        RedisCache(serializer="pickle")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# tools/migrate_pkl.py — sentinel gate on pickle load
# ---------------------------------------------------------------------------


def test_do_migration_refuses_without_sentinel(tmp_path: Path) -> None:
    """Calling ``_do_migration`` outside the hardened subprocess raises."""
    from omni_mercury_engine.tools import migrate_pkl

    legacy = tmp_path / "legacy.pkl"
    payload = {
        "features": {"a": np.zeros((4, 4), dtype=np.float32)},
        "labels": np.array([0, 1, 0, 1], dtype=np.int64),
    }
    with legacy.open("wb") as f:
        pickle.dump(payload, f)

    args = type(
        "Args",
        (),
        {
            "input": str(legacy),
            "output": str(tmp_path / "out.npz"),
            "max_bytes": 256 * 1024 * 1024,
            "sign_key_hex": None,
        },
    )()

    # Ensure sentinel is NOT set: simulate a misuse where _do_migration is
    # called directly without going through main()'s hardening path.
    saved = os.environ.pop(migrate_pkl._HARDENED_SENTINEL, None)
    try:
        with pytest.raises(RuntimeError, match="hardened subprocess"):
            migrate_pkl._do_migration(args)
    finally:
        if saved is not None:
            os.environ[migrate_pkl._HARDENED_SENTINEL] = saved
