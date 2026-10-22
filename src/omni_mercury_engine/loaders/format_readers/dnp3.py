# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Read-only DNP3 (IEEE 1815-2012) data-link-layer frame parser.

Parses DNP3 data-link frames per IEEE Std 1815-2012 ("IEEE Standard for
Electric Power Systems Communications -- Distributed Network Protocol
(DNP3)"): the 0x05 0x64 start bytes, the length octet, the control octet
(DIR, PRM, FCB, FCV/DFC and the link function code), destination and source
addresses, the header CRC, and the per-16-byte-block body CRCs.  The DNP3
CRC-16 uses polynomial 0x3D65 (reflected 0xA6BC), is transmitted ones'
complemented, low byte first; every CRC is verified and a mismatch is a loud
``ValueError`` naming the offset.  The transport-layer segment header (FIR,
FIN, 6-bit sequence) is decoded when a frame carries user data.

Honest limits: **application-layer object parsing is OUT of scope** -- the
transport segment's application bytes are returned raw, never interpreted.
The parser is read-only: it never executes or writes anything.
"""

from __future__ import annotations

from typing import Any

__all__ = ["parse_dnp3_frames"]

_START = b"\x05\x64"
_BLOCK = 16

_PRIMARY_FUNCTIONS = {
    0: "RESET_LINK_STATES",
    2: "TEST_LINK_STATES",
    3: "CONFIRMED_USER_DATA",
    4: "UNCONFIRMED_USER_DATA",
    9: "REQUEST_LINK_STATUS",
}

_SECONDARY_FUNCTIONS = {
    0: "ACK",
    1: "NACK",
    11: "LINK_STATUS",
    15: "NOT_SUPPORTED",
}


def crc_dnp3(data: bytes) -> int:
    """DNP3 CRC-16 per IEEE 1815-2012: poly 0x3D65 (reflected 0xA6BC), complemented.

    Args:
        data: The bytes the CRC covers (a link header's first 8 bytes, or one
            body block of up to 16 bytes).

    Returns:
        The 16-bit CRC value as transmitted (ones' complement applied; sent
        low byte first on the wire).
    """
    crc = 0x0000
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA6BC if crc & 1 else crc >> 1
    return (~crc) & 0xFFFF


def _check_crc(data: bytes, start: int, length: int, what: str) -> None:
    """Verify the CRC following ``data[start:start+length]`` or raise loudly."""
    declared = int.from_bytes(data[start + length : start + length + 2], "little")
    computed = crc_dnp3(data[start : start + length])
    if declared != computed:
        raise ValueError(
            f"DNP3 offset {start + length}: {what} CRC mismatch (frame carries "
            f"0x{declared:04X}, computed 0x{computed:04X})"
        )


def parse_dnp3_frames(data: bytes) -> dict[str, Any]:
    """Parse DNP3 data-link frames from raw bytes, read-only.

    Per IEEE Std 1815-2012 clause 9 (data link layer): each frame starts with
    0x05 0x64, a length octet (counting control, destination, source and user
    octets, excluding CRCs), the control octet, 16-bit little-endian
    destination and source addresses, a header CRC, and the user data in
    blocks of up to 16 octets, each followed by its own CRC.  All CRCs are
    verified (DNP3 CRC-16, polynomial 0x3D65, transmitted complemented).  The
    transport-layer segment header (FIR/FIN/sequence, clause 8) is decoded
    when user data is present; application-layer object parsing is out of
    scope and the application bytes are returned raw.

    Args:
        data: The raw frame bytes (one or more back-to-back frames).

    Returns:
        Dict with ``frames``: a list of per-frame dicts, each with
        ``offset``, ``length`` (the length octet), ``control`` (raw octet),
        ``dir``, ``prm``, ``fcb``, ``fcv_dfc``, ``function_code``,
        ``function`` (name per the PRM direction), ``destination``,
        ``source``, and -- when the frame carries user data -- ``transport``
        (``{"fir", "fin", "sequence"}``) and ``application_bytes`` (raw,
        undecoded) or, for a frame without a transport segment, ``user_data``.

    Raises:
        ValueError: On wrong start bytes, an implausible length octet, a
            truncated frame, or any header/body CRC mismatch -- always naming
            the byte offset.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValueError(f"data must be bytes, got {type(data).__name__}")
    data = bytes(data)
    if not data:
        raise ValueError("DNP3 offset 0: input is empty")

    frames: list[dict[str, Any]] = []
    offset = 0
    while offset < len(data):
        if len(data) - offset < 10:
            raise ValueError(
                f"DNP3 offset {offset}: {len(data) - offset} byte(s) remain, "
                "shorter than the 10-byte link header"
            )
        if data[offset : offset + 2] != _START:
            raise ValueError(
                f"DNP3 offset {offset}: bad start bytes "
                f"{data[offset : offset + 2]!r}, expected 0x05 0x64"
            )
        length = data[offset + 2]
        if length < 5:
            raise ValueError(
                f"DNP3 offset {offset + 2}: length octet {length} below the "
                "minimum of 5 (control + destination + source)"
            )
        control = data[offset + 3]
        prm = bool(control & 0x40)
        function_code = control & 0x0F
        destination = int.from_bytes(data[offset + 4 : offset + 6], "little")
        source = int.from_bytes(data[offset + 6 : offset + 8], "little")
        _check_crc(data, offset, 8, "link header")

        user_len = length - 5
        n_blocks = (user_len + _BLOCK - 1) // _BLOCK
        frame_size = 10 + user_len + 2 * n_blocks
        if offset + frame_size > len(data):
            raise ValueError(
                f"DNP3 offset {offset}: frame needs {frame_size} bytes "
                f"({user_len} user octets in {n_blocks} CRC block(s)) but only "
                f"{len(data) - offset} remain"
            )
        user = bytearray()
        block_start = offset + 10
        remaining = user_len
        while remaining > 0:
            block_len = min(_BLOCK, remaining)
            _check_crc(data, block_start, block_len, "body block")
            user.extend(data[block_start : block_start + block_len])
            block_start += block_len + 2
            remaining -= block_len

        names = _PRIMARY_FUNCTIONS if prm else _SECONDARY_FUNCTIONS
        frame: dict[str, Any] = {
            "offset": offset,
            "length": length,
            "control": control,
            "dir": bool(control & 0x80),
            "prm": prm,
            "fcb": bool(control & 0x20),
            "fcv_dfc": bool(control & 0x10),
            "function_code": function_code,
            "function": names.get(function_code, f"RESERVED ({function_code})"),
            "destination": destination,
            "source": source,
        }
        if user:
            if prm and function_code in (3, 4):
                transport_header = user[0]
                frame["transport"] = {
                    "fin": bool(transport_header & 0x80),
                    "fir": bool(transport_header & 0x40),
                    "sequence": transport_header & 0x3F,
                }
                frame["application_bytes"] = bytes(user[1:])
            else:
                frame["user_data"] = bytes(user)
        frames.append(frame)
        offset += frame_size
    return {"frames": frames}
