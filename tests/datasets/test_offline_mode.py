# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Offline (air-gapped) operation contract for the dataset layer.

Production deployments require both connectivity modes: online (fetch and
cache) and offline (serve the cache, refuse the network fail-closed). These
tests pin that contract at the single network chokepoint
(``base.http_get_with_retry``) and on a representative loader (ADBench).
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from omni_mercury_engine.datasets.adbench import ADBenchLoader
from omni_mercury_engine.datasets.base import DatasetConfig, http_get_with_retry
from omni_mercury_engine.datasets.exceptions import (
    MERCURY_OFFLINE_VAR,
    DataSourceUnavailableError,
    OfflineModeError,
    offline_mode_active,
)


def _prime_cardio_cache(data_dir: Path) -> Path:
    """Write a small, valid cardio NPZ exactly where the loader caches it."""
    cache_dir = data_dir / "adbench"
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cache_dir / "6_cardio.npz"
    rng = np.random.default_rng(0)
    buf = io.BytesIO()
    np.savez(buf, X=rng.normal(size=(24, 5)), y=np.array([0] * 20 + [1] * 4))
    npz_path.write_bytes(buf.getvalue())
    return npz_path


class TestOfflineFlag:
    @pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
    def test_truthy_values_activate(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv(MERCURY_OFFLINE_VAR, value)
        assert offline_mode_active() is True

    @pytest.mark.parametrize("value", ["", "0", "false", "off"])
    def test_falsy_values_do_not_activate(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv(MERCURY_OFFLINE_VAR, value)
        assert offline_mode_active() is False

    def test_unset_is_online(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(MERCURY_OFFLINE_VAR, raising=False)
        assert offline_mode_active() is False


class TestNetworkChokepoint:
    def test_offline_refuses_before_any_socket(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(MERCURY_OFFLINE_VAR, "1")
        with pytest.raises(OfflineModeError, match="MERCURY_OFFLINE"):
            http_get_with_retry("https://raw.githubusercontent.com/anything")

    def test_error_carries_url_and_remediation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(MERCURY_OFFLINE_VAR, "1")
        with pytest.raises(OfflineModeError) as excinfo:
            http_get_with_retry("https://raw.githubusercontent.com/some.npz")
        assert "some.npz" in str(excinfo.value)
        assert "prefetch_datasets" in str(excinfo.value)


class TestOfflineLoaderBehavior:
    def test_primed_cache_serves_fully_offline(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _prime_cardio_cache(tmp_path)
        monkeypatch.setenv(MERCURY_OFFLINE_VAR, "1")
        loader = ADBenchLoader(
            DatasetConfig(
                name="adbench",
                data_dir=str(tmp_path),
                cache_dir=str(tmp_path / "cache"),
                preprocessing={"dataset": "cardio"},
            )
        )
        assert loader.download() is True  # cache hit; no network involved
        X, y = loader.load()
        assert X.shape[0] == 24
        assert int(np.asarray(y).sum()) == 4

    def test_uncached_dataset_fails_closed_offline(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv(MERCURY_OFFLINE_VAR, "1")
        loader = ADBenchLoader(
            DatasetConfig(
                name="adbench",
                data_dir=str(tmp_path),
                cache_dir=str(tmp_path / "cache"),
                preprocessing={"dataset": "thyroid"},
            )
        )
        with pytest.raises(DataSourceUnavailableError) as excinfo:
            loader.download()
        assert "MERCURY_OFFLINE" in str(excinfo.value)


class TestEnvAwareDirectories:
    def test_data_dir_default_honors_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("MERCURY_DATA_DIR", str(tmp_path / "d"))
        monkeypatch.setenv("MERCURY_CACHE_DIR", str(tmp_path / "c"))
        config = DatasetConfig(name="adbench")
        assert config.data_dir == str(tmp_path / "d")
        assert config.cache_dir == str(tmp_path / "c")

    def test_defaults_unchanged_without_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MERCURY_DATA_DIR", raising=False)
        monkeypatch.delenv("MERCURY_CACHE_DIR", raising=False)
        config = DatasetConfig(name="adbench")
        assert config.data_dir == "./data"
        assert config.cache_dir == "./cache"

    def test_explicit_argument_beats_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("MERCURY_DATA_DIR", str(tmp_path / "env"))
        config = DatasetConfig(name="adbench", data_dir=str(tmp_path / "explicit"))
        assert config.data_dir == str(tmp_path / "explicit")
