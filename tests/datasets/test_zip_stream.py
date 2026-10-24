# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Extracting zip members from a stream, as figshare's archives arrive."""

from __future__ import annotations

import hashlib
import io
import struct
import zipfile
import zlib
from typing import TYPE_CHECKING

import pytest

from omni_mercury_engine.datasets.zip_stream import ZipStreamError, extract_stored_members

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from _typeshed import ReadableBuffer

MEMBERS = {
    "data_a.pkl": bytes(range(256)) * 4096,  # 1 MiB: spans read blocks
    "skip_me.bin": b"\x00" * 300_000,
    "info_a.pkl": b"offsets",
}


class _Unseekable(io.RawIOBase):
    """A write-only sink: zipfile then writes data descriptors after members."""

    def __init__(self) -> None:
        self.data = bytearray()

    def writable(self) -> bool:
        return True

    def write(self, b: ReadableBuffer, /) -> int:
        view = memoryview(b)
        self.data += view
        return view.nbytes


def _archive(
    *, seekable: bool, force_zip64: bool = False, method: int = zipfile.ZIP_STORED
) -> bytes:
    sink: io.BytesIO | _Unseekable = io.BytesIO() if seekable else _Unseekable()
    with zipfile.ZipFile(sink, "w", compression=method) as archive:
        for name, data in MEMBERS.items():
            with archive.open(name, "w", force_zip64=force_zip64) as member:
                member.write(data)
    return bytes(sink.getvalue() if isinstance(sink, io.BytesIO) else sink.data)


def _figshare_style(*, zip64: bool) -> bytes:
    """Members as figshare streams them: the descriptor flag set, yet the
    sizes and CRC already in the local header, then a signed descriptor."""
    out = bytearray()
    for name, data in MEMBERS.items():
        crc, raw = zlib.crc32(data), name.encode()
        size = 0xFFFFFFFF if zip64 else len(data)
        extra = struct.pack("<HHQQ", 1, 16, len(data), len(data)) if zip64 else b""
        out += b"PK\x03\x04" + struct.pack(
            "<HHHHHIIIHH", 45, 0x0808, 0, 0, 0, crc, size, size, len(raw), len(extra)
        )
        out += raw + extra + data + b"PK\x07\x08" + struct.pack("<I", crc)
        out += struct.pack("<QQ", len(data), len(data)) if zip64 else struct.pack("<II", size, size)
    return bytes(out)


def _chunks(data: bytes, size: int = 65_537) -> Iterator[bytes]:
    for i in range(0, len(data), size):
        yield data[i : i + size]


def _sha(name: str) -> str:
    return hashlib.sha256(MEMBERS[name]).hexdigest()


@pytest.mark.parametrize(
    "kind", ["sizes-in-header", "zip64", "figshare-descriptor", "figshare-zip64-descriptor"]
)
def test_wanted_members_are_written_and_verified(tmp_path: Path, kind: str) -> None:
    archive = {
        "sizes-in-header": lambda: _archive(seekable=True),
        "zip64": lambda: _archive(seekable=True, force_zip64=True),
        "figshare-descriptor": lambda: _figshare_style(zip64=False),
        "figshare-zip64-descriptor": lambda: _figshare_style(zip64=True),
    }[kind]()
    wanted = {"data_a.pkl": _sha("data_a.pkl"), "info_a.pkl": _sha("info_a.pkl")}

    seen = extract_stored_members(_chunks(archive), wanted, tmp_path)

    assert seen == list(MEMBERS)
    for name in wanted:
        assert (tmp_path / name).read_bytes() == MEMBERS[name]
    assert not (tmp_path / "skip_me.bin").exists()
    assert not list(tmp_path.glob("*.part"))


def test_reading_stops_once_every_wanted_member_is_written(tmp_path: Path) -> None:
    archive = _figshare_style(zip64=False)
    consumed = 0

    def counting() -> Iterator[bytes]:
        nonlocal consumed
        for chunk in _chunks(archive):
            consumed += len(chunk)
            yield chunk

    seen = extract_stored_members(counting(), {"data_a.pkl": _sha("data_a.pkl")}, tmp_path)

    assert seen == ["data_a.pkl"]
    assert consumed < len(MEMBERS["data_a.pkl"]) + 2 * 65_537  # not the rest of the archive


def test_a_corrupt_byte_fails_the_crc_and_leaves_nothing(tmp_path: Path) -> None:
    archive = bytearray(_archive(seekable=True))
    archive[archive.index(MEMBERS["data_a.pkl"][:64]) + 1000] ^= 0xFF

    with pytest.raises(ZipStreamError, match="CRC-32"):
        extract_stored_members(
            _chunks(bytes(archive)), {"data_a.pkl": _sha("data_a.pkl")}, tmp_path
        )
    assert not list(tmp_path.iterdir())


def test_a_digest_mismatch_is_refused_and_removed(tmp_path: Path) -> None:
    with pytest.raises(ZipStreamError, match="SHA-256"):
        extract_stored_members(_chunks(_archive(seekable=True)), {"info_a.pkl": "0" * 64}, tmp_path)
    assert not list(tmp_path.iterdir())


def test_a_size_known_only_after_the_data_is_refused(tmp_path: Path) -> None:
    """zipfile writing to an unseekable sink puts 0 in the header: where such
    a stored member ends cannot be known while streaming."""
    with pytest.raises(ZipStreamError, match="deferred to a data descriptor"):
        extract_stored_members(
            _chunks(_archive(seekable=False)), {"data_a.pkl": _sha("data_a.pkl")}, tmp_path
        )
    assert not list(tmp_path.iterdir())


def test_a_compressed_member_is_refused(tmp_path: Path) -> None:
    archive = _archive(seekable=True, method=zipfile.ZIP_DEFLATED)
    with pytest.raises(ZipStreamError, match="only stored"):
        extract_stored_members(_chunks(archive), {"info_a.pkl": _sha("info_a.pkl")}, tmp_path)


def test_a_stream_that_ends_early_is_an_error(tmp_path: Path) -> None:
    archive = _archive(seekable=True)
    with pytest.raises(ZipStreamError, match="stream ended"):
        extract_stored_members(
            _chunks(archive[:500_000]), {"data_a.pkl": _sha("data_a.pkl")}, tmp_path
        )
    assert not (tmp_path / "data_a.pkl").exists()


def test_a_wanted_member_absent_from_the_archive_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ZipStreamError, match="members still wanted"):
        extract_stored_members(
            _chunks(_archive(seekable=True)), {"missing.pkl": "0" * 64}, tmp_path
        )
