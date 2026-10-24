# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Read-only ASPRS LAS 1.2-1.4 point-cloud reader.

Implements the public header block and point data records of the ASPRS "LAS
Specification 1.4 - R15" (ASPRS, 2019), which also defines the layout shared
with LAS 1.2/1.3: file signature ``LASF``, version, header size, offset to
point data, point data record format and length, legacy and 64-bit point
counts, and the X/Y/Z scale factors and offsets.  Point record formats 0-3
and 6-7 are decoded (scaled float64 x/y/z, intensity, classification, GPS
time and RGB where the format carries them).

Honest limits: compressed LAZ files are refused (bit 7 of the point data
format byte), variable length records are counted but not decoded, and extra
bytes beyond a format's standard fields are skipped via the declared record
length.  Malformed input raises ``ValueError`` with position information.
The reader never executes or writes anything.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["read_las_points"]

#: Core field layout per point data record format (name, numpy dtype, offset).
_FMT_FIELDS: dict[int, list[tuple[str, str, int]]] = {
    0: [
        ("X", "<i4", 0),
        ("Y", "<i4", 4),
        ("Z", "<i4", 8),
        ("intensity", "<u2", 12),
        ("return_byte", "u1", 14),
        ("raw_classification", "u1", 15),
        ("scan_angle", "i1", 16),
        ("user_data", "u1", 17),
        ("point_source_id", "<u2", 18),
    ],
}
_FMT_FIELDS[1] = _FMT_FIELDS[0] + [("gps_time", "<f8", 20)]
_FMT_FIELDS[2] = _FMT_FIELDS[0] + [("red", "<u2", 20), ("green", "<u2", 22), ("blue", "<u2", 24)]
_FMT_FIELDS[3] = _FMT_FIELDS[1] + [("red", "<u2", 28), ("green", "<u2", 30), ("blue", "<u2", 32)]
_FMT_FIELDS[6] = [
    ("X", "<i4", 0),
    ("Y", "<i4", 4),
    ("Z", "<i4", 8),
    ("intensity", "<u2", 12),
    ("return_byte", "u1", 14),
    ("flags_byte", "u1", 15),
    ("classification", "u1", 16),
    ("user_data", "u1", 17),
    ("scan_angle", "<i2", 18),
    ("point_source_id", "<u2", 20),
    ("gps_time", "<f8", 22),
]
_FMT_FIELDS[7] = _FMT_FIELDS[6] + [("red", "<u2", 30), ("green", "<u2", 32), ("blue", "<u2", 34)]

#: Minimum point record length per supported format.
_FMT_SIZE = {0: 20, 1: 28, 2: 26, 3: 34, 6: 30, 7: 36}


def read_las_points(path: str | Path, max_points: int | None = None) -> dict[str, Any]:
    """Read an ASPRS LAS 1.2-1.4 file's header and point records, read-only.

    Decodes the public header block (point data record format, legacy and
    1.4 64-bit point counts, scale factors and offsets) and the point records
    for formats 0-3 and 6-7, per the ASPRS LAS Specification 1.4 - R15.
    X/Y/Z are returned scaled to float64 (``raw * scale + offset``); GPS time
    and RGB are included where the format carries them.

    Args:
        path: Filesystem path of the ``.las`` file.
        max_points: Optional cap on the number of points decoded; ``None``
            reads every point the header declares.

    Returns:
        Dict with header fields (``version`` tuple, ``header_size``,
        ``point_format``, ``point_record_length``, ``point_count``,
        ``scales``, ``offsets``, ``n_vlrs``), ``points_read`` (after the
        ``max_points`` cap) and numpy arrays ``x``, ``y``, ``z`` (float64),
        ``intensity`` (uint16), ``classification`` (uint8; for formats 0-3
        the 5-bit class with the flag bits also exposed as
        ``classification_flags``), plus ``gps_time`` (float64) and ``red`` /
        ``green`` / ``blue`` (uint16) where the point format has them.

    Raises:
        ValueError: On a negative ``max_points``, a wrong file signature, an unsupported version or point
            format, a compressed (LAZ) point format bit, a record length
            below the format's standard size, or a file shorter than the
            declared point data -- each naming the offending offset or count.
    """
    if max_points is not None and max_points < 0:
        # ``np.frombuffer`` reads a count of -1 as "everything", so a
        # negative cap must not reach it.
        raise ValueError(f"max_points must be >= 0 or None, got {max_points}")
    data = Path(path).read_bytes()
    if len(data) < 227:
        raise ValueError(
            f"LAS offset 0: file is {len(data)} bytes, shorter than the smallest "
            "public header block (227 bytes)"
        )
    if data[0:4] != b"LASF":
        raise ValueError(f"LAS offset 0: bad file signature {data[0:4]!r}, expected b'LASF'")
    version = (data[24], data[25])
    if version[0] != 1 or not 2 <= version[1] <= 4:
        raise ValueError(
            f"LAS offset 24: version {version[0]}.{version[1]} unsupported "
            "(this reader handles 1.2-1.4)"
        )
    header_size = int.from_bytes(data[94:96], "little")
    if header_size > len(data):
        raise ValueError(
            f"LAS offset 94: header size {header_size} exceeds the file length {len(data)}"
        )
    offset_to_points = int.from_bytes(data[96:100], "little")
    n_vlrs = int.from_bytes(data[100:104], "little")
    format_byte = data[104]
    if format_byte & 0x80:
        raise ValueError(
            "LAS offset 104: point data format byte has the compression bit set "
            "(LAZ); this reader handles uncompressed LAS only"
        )
    point_format = format_byte & 0x3F
    if point_format not in _FMT_FIELDS:
        raise ValueError(
            f"LAS offset 104: point data record format {point_format} unsupported "
            f"(supported: {sorted(_FMT_FIELDS)})"
        )
    point_record_length = int.from_bytes(data[105:107], "little")
    base_size = _FMT_SIZE[point_format]
    if point_record_length < base_size:
        raise ValueError(
            f"LAS offset 105: point record length {point_record_length} below the "
            f"{base_size}-byte standard size of format {point_format}"
        )
    legacy_count = int.from_bytes(data[107:111], "little")
    scales = struct.unpack("<3d", data[131:155])
    offsets = struct.unpack("<3d", data[155:179])
    point_count = legacy_count
    if version[1] >= 4:
        if header_size < 375 or len(data) < 255:
            raise ValueError(
                f"LAS offset 94: LAS 1.4 header must be at least 375 bytes, got {header_size}"
            )
        count_64 = int.from_bytes(data[247:255], "little")
        if count_64:
            point_count = count_64
    if offset_to_points < header_size:
        raise ValueError(
            f"LAS offset 96: offset to point data {offset_to_points} lies inside "
            f"the {header_size}-byte header"
        )

    n_read = point_count if max_points is None else min(point_count, max_points)
    needed = offset_to_points + point_count * point_record_length
    if needed > len(data):
        raise ValueError(
            f"LAS offset {offset_to_points}: header declares {point_count} points of "
            f"{point_record_length} bytes ({needed} bytes total) but the file holds "
            f"only {len(data)} bytes"
        )
    fields = _FMT_FIELDS[point_format]
    dtype = np.dtype(
        {
            "names": [name for name, _, _ in fields],
            "formats": [fmt for _, fmt, _ in fields],
            "offsets": [off for _, _, off in fields],
            "itemsize": point_record_length,
        }
    )
    records = np.frombuffer(data, dtype=dtype, count=n_read, offset=offset_to_points)

    result: dict[str, Any] = {
        "version": version,
        "header_size": header_size,
        "offset_to_points": offset_to_points,
        "n_vlrs": n_vlrs,
        "point_format": point_format,
        "point_record_length": point_record_length,
        "point_count": point_count,
        "points_read": n_read,
        "scales": tuple(scales),
        "offsets": tuple(offsets),
        "x": records["X"].astype(np.float64) * scales[0] + offsets[0],
        "y": records["Y"].astype(np.float64) * scales[1] + offsets[1],
        "z": records["Z"].astype(np.float64) * scales[2] + offsets[2],
        "intensity": records["intensity"].copy(),
    }
    if point_format <= 3:
        raw_class = records["raw_classification"]
        result["classification"] = (raw_class & 0x1F).astype(np.uint8)
        result["classification_flags"] = (raw_class >> 5).astype(np.uint8)
    else:
        result["classification"] = records["classification"].copy()
        result["classification_flags"] = (records["flags_byte"] & 0x0F).astype(np.uint8)
    # A structured dtype always has field names; ``or ()`` narrows the stub's Optional.
    field_names = dtype.names or ()
    if "gps_time" in field_names:
        result["gps_time"] = records["gps_time"].copy()
    if "red" in field_names:
        result["red"] = records["red"].copy()
        result["green"] = records["green"].copy()
        result["blue"] = records["blue"].copy()
    return result
