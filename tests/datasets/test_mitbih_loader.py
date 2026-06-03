"""Tests for MIT-BIH loader network-bound failure handling."""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.datasets.base import DatasetConfig
from omni_mercury_engine.datasets.exceptions import DataSourceUnavailableError
from omni_mercury_engine.datasets.mitbih import MITBIHLoader


def _config(tmp_path: Any, **preprocessing: Any) -> DatasetConfig:
    return DatasetConfig(
        name="mitbih",
        data_dir=str(tmp_path / "data"),
        cache_dir=str(tmp_path / "cache"),
        max_samples=1,
        preprocessing=preprocessing,
    )


def test_wfdb_requests_receive_bounded_timeout(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WFDB's request calls are bounded even though WFDB omits timeouts."""
    observed_timeouts: list[float] = []

    def fake_request(_session: Any, _method: str, _url: str, **kwargs: Any) -> object:
        observed_timeouts.append(float(kwargs["timeout"]))
        return object()

    def fake_rdrecord(_record_id: str, *, pn_dir: str) -> object:
        import requests

        assert pn_dir == "mitdb"
        requests.Session().request("GET", "https://physionet.org/content/mitdb/")
        return types.SimpleNamespace(p_signal=np.ones((720, 2)))

    def fake_rdann(_record_id: str, _extension: str, *, pn_dir: str) -> object:
        assert pn_dir == "mitdb"
        return types.SimpleNamespace(sample=np.array([360]), symbol=["N"])

    fake_wfdb = types.SimpleNamespace(rdrecord=fake_rdrecord, rdann=fake_rdann)
    monkeypatch.setitem(sys.modules, "wfdb", fake_wfdb)
    monkeypatch.setattr("requests.Session.request", fake_request)

    loader = MITBIHLoader(_config(tmp_path, records=["100"], request_timeout=0.25))

    assert loader.download() is True
    assert observed_timeouts == [0.25]


def test_mitbih_download_stops_after_consecutive_record_failures(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dead PhysioNet path fails fast instead of iterating all 48 records."""
    attempts: list[str] = []

    def fake_rdrecord(record_id: str, *, pn_dir: str) -> object:
        assert pn_dir == "mitdb"
        attempts.append(record_id)
        raise TimeoutError("physionet timed out")

    fake_wfdb = types.SimpleNamespace(rdrecord=fake_rdrecord, rdann=lambda *_a, **_k: None)
    monkeypatch.setitem(sys.modules, "wfdb", fake_wfdb)

    loader = MITBIHLoader(
        _config(
            tmp_path,
            records=["100", "101", "102", "103"],
            max_record_failures=2,
            request_timeout=0.1,
        )
    )

    with pytest.raises(DataSourceUnavailableError, match="failure threshold"):
        loader.download()

    assert attempts == ["100", "101"]
