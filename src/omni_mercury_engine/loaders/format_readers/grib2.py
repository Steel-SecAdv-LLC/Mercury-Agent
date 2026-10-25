# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Read-only GRIB2 section-structure reader (WMO FM 92 GRIB Edition 2).

Parses the indicator section (the ``GRIB`` magic, discipline, edition 2 and
the 8-byte total message length), the per-message sequence of sections 1-7
(each with its 4-byte length and 1-byte section number), the identification
section fields (originating centre/sub-centre, reference date-time), and the
grid/product/data-representation template numbers, per the WMO Manual on
Codes, FM 92 GRIB Edition 2 regulations (sections 92.6.1-92.6.4).

Honest limits: **full decoding of packed data templates is OUT of scope** --
this reader reports template numbers and exposes the grid-definition and
product-definition sections' raw bytes, but never unpacks section 7 data.
Multi-message files are supported (messages are read back to back until end
of file).  Malformed input raises ``ValueError`` with the byte offset.  The
reader never executes or writes anything.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["read_grib2_sections"]

_INDICATOR_LEN = 16  # "GRIB" + 2 reserved + discipline + edition + 8-byte length
_END_MARKER = b"7777"


def _be(buffer: bytes, start: int, size: int) -> int:
    """Big-endian unsigned integer of ``size`` bytes at ``start``."""
    return int.from_bytes(buffer[start : start + size], "big")


def _parse_identification(body: bytes, offset: int) -> dict[str, Any]:
    """Decode section 1 (identification) fields per FM 92 GRIB Edition 2."""
    if len(body) < 21:
        raise ValueError(
            f"GRIB2 offset {offset}: identification section is {len(body)} bytes, "
            "shorter than the 21-byte fixed part"
        )
    return {
        "centre": _be(body, 5, 2),
        "subcentre": _be(body, 7, 2),
        "master_tables_version": body[9],
        "local_tables_version": body[10],
        "significance_of_reference_time": body[11],
        "year": _be(body, 12, 2),
        "month": body[14],
        "day": body[15],
        "hour": body[16],
        "minute": body[17],
        "second": body[18],
    }


def _parse_message(data: bytes, pos: int) -> tuple[dict[str, Any], int]:
    """Parse one GRIB2 message at byte ``pos``; return ``(message, next_pos)``."""
    if len(data) - pos < _INDICATOR_LEN:
        raise ValueError(
            f"GRIB2 offset {pos}: only {len(data) - pos} bytes remain, "
            f"indicator section needs {_INDICATOR_LEN}"
        )
    magic = data[pos : pos + 4]
    if magic != b"GRIB":
        raise ValueError(f"GRIB2 offset {pos}: expected b'GRIB' magic, found {magic!r}")
    discipline = data[pos + 6]
    edition = data[pos + 7]
    if edition != 2:
        raise ValueError(f"GRIB2 offset {pos + 7}: edition {edition}, this reader handles only 2")
    total_length = _be(data, pos + 8, 8)
    if total_length < _INDICATOR_LEN + len(_END_MARKER):
        raise ValueError(f"GRIB2 offset {pos + 8}: implausible total length {total_length}")
    if pos + total_length > len(data):
        raise ValueError(
            f"GRIB2 offset {pos}: message declares {total_length} bytes but only "
            f"{len(data) - pos} remain (truncated file)"
        )
    end = data[pos + total_length - 4 : pos + total_length]
    if end != _END_MARKER:
        raise ValueError(
            f"GRIB2 offset {pos + total_length - 4}: message does not end with "
            f"b'7777', found {end!r}"
        )

    message: dict[str, Any] = {
        "offset": pos,
        "total_length": total_length,
        "discipline": discipline,
        "edition": edition,
        "sections": [],
        "identification": None,
        "grid_definitions": [],
        "product_definitions": [],
    }
    sp = pos + _INDICATOR_LEN
    body_end = pos + total_length - len(_END_MARKER)
    while sp < body_end:
        if body_end - sp < 5:
            raise ValueError(
                f"GRIB2 offset {sp}: {body_end - sp} bytes remain in the message, "
                "too few for a section header (4-byte length + section number)"
            )
        section_length = _be(data, sp, 4)
        section_number = data[sp + 4]
        if not 1 <= section_number <= 7:
            raise ValueError(f"GRIB2 offset {sp + 4}: invalid section number {section_number}")
        if section_length < 5 or sp + section_length > body_end:
            raise ValueError(
                f"GRIB2 offset {sp}: section {section_number} declares "
                f"{section_length} bytes, overrunning the message end at {body_end}"
            )
        body = data[sp : sp + section_length]
        record: dict[str, Any] = {
            "number": section_number,
            "length": section_length,
            "offset": sp,
        }
        if section_number == 1:
            message["identification"] = _parse_identification(body, sp)
        elif section_number == 3:
            if section_length < 14:
                raise ValueError(
                    f"GRIB2 offset {sp}: grid definition section is {section_length} "
                    "bytes, shorter than the 14-byte fixed part"
                )
            record["grid_definition_template_number"] = _be(body, 12, 2)
            message["grid_definitions"].append(
                {"template_number": _be(body, 12, 2), "raw": bytes(body)}
            )
        elif section_number == 4:
            if section_length < 9:
                raise ValueError(
                    f"GRIB2 offset {sp}: product definition section is {section_length} "
                    "bytes, shorter than the 9-byte fixed part"
                )
            record["product_definition_template_number"] = _be(body, 7, 2)
            message["product_definitions"].append(
                {"template_number": _be(body, 7, 2), "raw": bytes(body)}
            )
        elif section_number == 5:
            if section_length < 11:
                raise ValueError(
                    f"GRIB2 offset {sp}: data representation section is {section_length} "
                    "bytes, shorter than the 11-byte fixed part"
                )
            record["data_representation_template_number"] = _be(body, 9, 2)
        message["sections"].append(record)
        sp += section_length
    if sp != body_end:
        raise ValueError(
            f"GRIB2 offset {sp}: sections overrun the end-section boundary at {body_end}"
        )
    if message["identification"] is None:
        raise ValueError(f"GRIB2 offset {pos}: message carries no identification section (1)")
    return message, pos + total_length


def read_grib2_sections(path: str | Path) -> dict[str, Any]:
    """Read the section structure of a GRIB2 file (WMO FM 92 GRIB Edition 2).

    Parses every message's indicator section, the (number, length, offset) of
    each section 1-7, the identification section fields (originating centre,
    sub-centre, reference date), and the grid / product / data-representation
    template numbers.  Full decoding of packed data templates (sections 5-7
    payloads) is out of scope; the grid-definition and product-definition
    sections' raw bytes are exposed instead.

    Args:
        path: Filesystem path of the GRIB2 file.

    Returns:
        Dict with key ``messages``: a list, one dict per message, each with
        ``offset``, ``total_length``, ``discipline``, ``edition``,
        ``sections`` (list of ``{"number", "length", "offset"}`` plus the
        template number for sections 3/4/5), ``identification`` (centre,
        subcentre, table versions, reference date-time fields),
        ``grid_definitions`` and ``product_definitions`` (each a list of
        ``{"template_number", "raw"}`` with the section's raw bytes).

    Raises:
        ValueError: On wrong magic, wrong edition, truncated messages,
            invalid section numbers or lengths, or a missing ``7777`` end
            marker -- always naming the byte offset.
    """
    data = Path(path).read_bytes()
    if not data:
        raise ValueError("GRIB2 offset 0: file is empty")
    messages: list[dict[str, Any]] = []
    pos = 0
    while pos < len(data):
        message, pos = _parse_message(data, pos)
        messages.append(message)
    return {"messages": messages}
