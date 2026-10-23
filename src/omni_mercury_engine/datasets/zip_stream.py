# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Extract members from a zip archive as it streams in.

figshare serves a whole-article archive as one zip generated per request:
range requests are ignored, so the central directory at its end cannot be
read first, and an archive can be several gigabytes. Its members are
stored (method 0) with their sizes in the local headers, so they can be
taken in order as the bytes arrive, and the stream closed as soon as every
wanted member is written.
"""

from __future__ import annotations

import hashlib
import struct
import zlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

__all__ = ["ZipStreamError", "extract_stored_members"]

_LOCAL_HEADER = b"PK\x03\x04"
_DESCRIPTOR = b"PK\x07\x08"
_ZIP64_EXTRA = 0x0001
_HAS_DESCRIPTOR = 0x0008
_SIZE_IN_ZIP64 = 0xFFFFFFFF
_BLOCK = 1 << 20


class ZipStreamError(ValueError):
    """The stream is not a zip this reader handles, or a member fails a check."""


class _Reader:
    """Exact-length reads over an iterable of byte chunks."""

    def __init__(self, chunks: Iterable[bytes]) -> None:
        self._chunks = iter(chunks)
        self._buf = bytearray()
        self.offset = 0

    def read(self, n: int) -> bytes:
        while len(self._buf) < n:
            chunk = next(self._chunks, None)
            if chunk is None:
                raise ZipStreamError(
                    f"stream ended at byte {self.offset + len(self._buf):,}, " f"{n:,} bytes wanted"
                )
            self._buf += chunk
        data = bytes(self._buf[:n])
        del self._buf[:n]
        self.offset += n
        return data


def _zip64_sizes(extra: bytes, size: int, csize: int) -> tuple[int, int, bool]:
    """Sizes from the zip64 extra field, where the header defers to it."""
    i = 0
    while i + 4 <= len(extra):
        tag, length = struct.unpack("<HH", extra[i : i + 4])
        if tag == _ZIP64_EXTRA:
            values = list(struct.unpack(f"<{length // 8}Q", extra[i + 4 : i + 4 + length]))
            if size == _SIZE_IN_ZIP64 and values:
                size = values.pop(0)
            if csize == _SIZE_IN_ZIP64 and values:
                csize = values.pop(0)
            return size, csize, True
        i += 4 + length
    return size, csize, False


def extract_stored_members(
    chunks: Iterable[bytes], wanted: dict[str, str], dest: Path
) -> list[str]:
    """Write each wanted member to ``dest``, verified, from a zip byte stream.

    Every member is CRC-32 checked as it passes, wanted or not, so a
    misaligned or corrupt stream fails instead of yielding wrong bytes.

    Args:
        chunks: The archive's bytes in order (e.g. ``response.iter_content()``).
        wanted: Member name -> expected SHA-256 hex digest.
        dest: Directory the members are written into, under their names.

    Returns:
        The names of the members read, in archive order. Reading stops once
        every wanted member is written.

    Raises:
        ZipStreamError: On a record that is not a local file header, a
            compressed member, a CRC-32 or SHA-256 mismatch (the member is
            removed), or a stream that ends before every wanted member.
    """
    reader = _Reader(chunks)
    remaining = dict(wanted)
    seen: list[str] = []
    while remaining:
        signature = reader.read(4)
        if signature != _LOCAL_HEADER:
            raise ZipStreamError(
                f"byte {reader.offset - 4:,}: expected a local file header, got "
                f"{signature!r}; members still wanted: {sorted(remaining)}"
            )
        _version, flags, method, _time, _date, crc, csize, size, name_len, extra_len = (
            struct.unpack("<HHHHHIIIHH", reader.read(26))
        )
        name = reader.read(name_len).decode("utf-8")
        size, csize, zip64 = _zip64_sizes(reader.read(extra_len), size, csize)
        if method != 0 or csize != size:
            raise ZipStreamError(f"{name}: compression method {method}; only stored is read")
        if flags & _HAS_DESCRIPTOR and size == 0 and crc == 0 and not zip64:
            raise ZipStreamError(f"{name}: size deferred to a data descriptor; unreadable")
        seen.append(name)
        expected_sha = remaining.get(name)
        part = dest / f"{name}.part" if expected_sha is not None else None
        sha = hashlib.sha256()
        running_crc = 0
        left = size
        sink = part.open("wb") if part is not None else None
        try:
            while left:
                block = reader.read(min(_BLOCK, left))
                left -= len(block)
                running_crc = zlib.crc32(block, running_crc)
                if sink is not None:
                    sha.update(block)
                    sink.write(block)
        finally:
            if sink is not None:
                sink.close()
        if flags & _HAS_DESCRIPTOR:
            head = reader.read(4)
            if head == _DESCRIPTOR:
                head = reader.read(4)
            crc = struct.unpack("<I", head)[0]
            reader.read(16 if zip64 else 8)
        if running_crc != crc:
            if part is not None:
                part.unlink(missing_ok=True)
            raise ZipStreamError(f"{name}: CRC-32 {running_crc:08x}, header says {crc:08x}")
        if part is not None:
            digest = sha.hexdigest()
            if digest != expected_sha:
                part.unlink(missing_ok=True)
                raise ZipStreamError(f"{name}: SHA-256 {digest}, pinned {expected_sha}")
            part.replace(dest / name)
            del remaining[name]
    return seen
