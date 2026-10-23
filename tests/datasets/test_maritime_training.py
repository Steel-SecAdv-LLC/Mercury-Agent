# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The DTU AIS unlabelled training set, streamed from figshare's archive.

The download is exercised offline against an archive laid out as figshare
streams it (stored members, sizes in the local header, signed data
descriptors); the real 680 MB / 1.1 GB stream is run by
``examples/mercury_quickstart.py --published-protocols`` in the
iso-hardening workflow.
"""

from __future__ import annotations

import hashlib
import pickle
import struct
import zlib
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from omni_mercury_engine.datasets.exceptions import OfflineModeError
from omni_mercury_engine.datasets.maritime import (
    FEATURE_NAMES,
    DTUAISTrainingSet,
    _TrainingPeriod,
)
from omni_mercury_engine.datasets.zip_stream import ZipStreamError
from omni_mercury_engine.security.safe_http import SafeHTTPClient

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def _track(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = 30
    return {
        "mmsi": seed,
        "shiptype": 70,
        "track_length": n,
        "lat": list(55.0 + np.cumsum(rng.normal(0, 1e-3, n))),
        "lon": list(14.0 + np.cumsum(rng.normal(0, 1e-3, n))),
        "speed": list(np.abs(rng.normal(10, 1, n))),
        "course": list(rng.uniform(0, 360, n)),
        "timestamp": list(1_622_505_600.0 + 120.0 * np.arange(n)),
    }


MEMBER = "data_test_period.pkl"
PAYLOAD = b"".join(pickle.dumps(_track(i), protocol=4) for i in range(3))


def _figshare_archive(members: dict[str, bytes]) -> bytes:
    out = bytearray()
    for name, data in members.items():
        crc, raw = zlib.crc32(data), name.encode()
        out += b"PK\x03\x04" + struct.pack(
            "<HHHHHIIIHH", 20, 0x0808, 0, 0, 0, crc, len(data), len(data), len(raw), 0
        )
        out += raw + data + b"PK\x07\x08" + struct.pack("<III", crc, len(data), len(data))
    return bytes(out)


class _Response:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.consumed = 0

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, size: int) -> Iterator[bytes]:
        for i in range(0, len(self.body), size):
            self.consumed += len(self.body[i : i + size])
            yield self.body[i : i + size]


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Serve a figshare-style archive and register a 3-track test period."""
    archive = _figshare_archive({MEMBER: PAYLOAD, "later_member.pkl": b"\x00" * 3_000_000})
    calls: dict[str, Any] = {"urls": [], "archive_size": len(archive)}

    def fake_get(url: str, **kwargs: Any) -> _Response:
        calls["urls"].append((url, kwargs))
        response = _Response(archive)
        calls["response"] = response
        return response

    monkeypatch.setattr(SafeHTTPClient, "get", fake_get)
    monkeypatch.delenv("MERCURY_OFFLINE", raising=False)
    period = _TrainingPeriod(
        member=MEMBER, sha256=hashlib.sha256(PAYLOAD).hexdigest(), md5="-", n_tracks=3
    )
    monkeypatch.setitem(DTUAISTrainingSet.PERIODS, "test", period)
    return calls


def test_download_streams_verifies_and_stops_at_the_member(
    served: dict[str, Any], tmp_path: Path
) -> None:
    training = DTUAISTrainingSet(tmp_path, "test")
    assert not training.is_cached()

    training.download()

    url, kwargs = served["urls"][0]
    assert url == DTUAISTrainingSet.ARCHIVE_URL
    assert kwargs["stream"] is True
    assert training.member_path.read_bytes() == PAYLOAD
    assert training.is_cached()
    # Stopped within one read of the member: the 3 MB member after it was not read.
    assert served["response"].consumed <= 1 << 20  # one read; the archive is ~3 MB
    assert served["archive_size"] > 2 * (1 << 20)

    features = training.features()
    assert features.shape == (3, len(FEATURE_NAMES))
    assert np.all(np.isfinite(features))

    training.download()  # cached: no second request
    assert len(served["urls"]) == 1


def test_bytes_other_than_the_pinned_ones_are_refused(
    served: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pinned = DTUAISTrainingSet.PERIODS["test"]
    monkeypatch.setitem(
        DTUAISTrainingSet.PERIODS,
        "test",
        _TrainingPeriod(member=pinned.member, sha256="0" * 64, md5="-", n_tracks=3),
    )
    training = DTUAISTrainingSet(tmp_path, "test")
    with pytest.raises(ZipStreamError, match="SHA-256"):
        training.download()
    assert not training.member_path.exists()


def test_a_short_file_is_refused(
    served: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pinned = DTUAISTrainingSet.PERIODS["test"]
    monkeypatch.setitem(
        DTUAISTrainingSet.PERIODS,
        "test",
        _TrainingPeriod(member=pinned.member, sha256=pinned.sha256, md5="-", n_tracks=4),
    )
    training = DTUAISTrainingSet(tmp_path, "test")
    training.download()
    with pytest.raises(ValueError, match="3 trajectories, expected 4"):
        training.features()


def test_a_forbidden_global_in_a_record_is_refused(tmp_path: Path) -> None:
    """The records go through the fail-closed unpickler, as the labelled set's."""
    training = DTUAISTrainingSet(tmp_path)
    training.data_path.mkdir(parents=True)
    training.member_path.write_bytes(pickle.dumps(_track(0)) + pickle.dumps(print))
    with pytest.raises(pickle.UnpicklingError, match="forbidden global"):
        list(training.iter_tracks())


def test_offline_mode_refuses_before_any_request(
    served: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MERCURY_OFFLINE", "1")
    with pytest.raises(OfflineModeError):
        DTUAISTrainingSet(tmp_path, "test").download()
    assert served["urls"] == []


def test_an_unknown_period_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown period"):
        DTUAISTrainingSet(tmp_path, "january")


def test_the_pinned_periods_are_the_published_files() -> None:
    """figshare file listing for article 21511842 (sizes and MD5s published there)."""
    periods = DTUAISTrainingSet.PERIODS
    assert periods["june_november"].md5 == "b7fa03df6c9e3d9abd830fbe72d83075"
    assert periods["june_november"].n_tracks == 106_728
    assert periods["december"].md5 == "d7bd2d763d39473a8d824daeb67d3395"
    assert periods["december"].n_tracks == 12_591


def test_two_periods_come_from_one_pass(
    served: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    later = b"\x00" * 3_000_000
    monkeypatch.setitem(
        DTUAISTrainingSet.PERIODS,
        "later",
        _TrainingPeriod(
            member="later_member.pkl", sha256=hashlib.sha256(later).hexdigest(), md5="-", n_tracks=0
        ),
    )

    DTUAISTrainingSet.download_periods(tmp_path, ["test", "later"])

    assert len(served["urls"]) == 1
    assert DTUAISTrainingSet(tmp_path, "test").is_cached()
    assert DTUAISTrainingSet(tmp_path, "later").is_cached()
    DTUAISTrainingSet.download_periods(tmp_path, ["test", "later"])  # all cached
    assert len(served["urls"]) == 1
