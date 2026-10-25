# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Read-only Modbus frame parser (Modbus Application Protocol V1.1b3).

Parses Modbus ADUs per the "MODBUS Application Protocol Specification
V1.1b3" (Modbus Organization, 2012) and the "MODBUS over Serial Line
Specification and Implementation Guide V1.02" for RTU framing: the TCP MBAP
header (transaction identifier, protocol identifier 0, length, unit
identifier) followed by the PDU, or an RTU frame (address + PDU + CRC-16,
polynomial 0xA001 reflected, initial value 0xFFFF, transmitted low byte
first) whose CRC is verified -- a mismatch is a loud ``ValueError``.

PDUs of the public function codes 1-6, 15 and 16 are decoded as requests or
responses; exception responses (function code | 0x80) are decoded with their
exception-code names.  Because a lone PDU does not state its direction, the
request/response role is inferred from the PDU layout (documented per
function code below); a PDU that fits neither layout is refused.  Function
codes outside the supported set are returned with role ``"unsupported"`` and
their raw data -- reported, never silently dropped.  The parser is read-only:
it never executes or writes anything.
"""

from __future__ import annotations

from typing import Any

__all__ = ["parse_modbus_frames"]

_EXCEPTION_NAMES = {
    0x01: "ILLEGAL FUNCTION",
    0x02: "ILLEGAL DATA ADDRESS",
    0x03: "ILLEGAL DATA VALUE",
    0x04: "SERVER DEVICE FAILURE",
    0x05: "ACKNOWLEDGE",
    0x06: "SERVER DEVICE BUSY",
    0x08: "MEMORY PARITY ERROR",
    0x0A: "GATEWAY PATH UNAVAILABLE",
    0x0B: "GATEWAY TARGET DEVICE FAILED TO RESPOND",
}

_FUNCTION_NAMES = {
    1: "Read Coils",
    2: "Read Discrete Inputs",
    3: "Read Holding Registers",
    4: "Read Input Registers",
    5: "Write Single Coil",
    6: "Write Single Register",
    15: "Write Multiple Coils",
    16: "Write Multiple Registers",
}


def crc16_modbus(data: bytes) -> int:
    """CRC-16 per the Modbus serial-line spec: reflected poly 0xA001, init 0xFFFF.

    Args:
        data: The bytes covered by the CRC (address + PDU).

    Returns:
        The 16-bit CRC value (transmitted low byte first on the wire).
    """
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc


def _registers(payload: bytes, offset: int) -> list[int]:
    """Big-endian 16-bit register values from ``payload``."""
    if len(payload) % 2:
        raise ValueError(f"Modbus offset {offset}: odd register payload of {len(payload)} bytes")
    return [int.from_bytes(payload[i : i + 2], "big") for i in range(0, len(payload), 2)]


def _decode_pdu(function_code: int, payload: bytes, offset: int) -> dict[str, Any]:
    """Decode one PDU's function-specific fields.

    Args:
        function_code: The PDU function code byte.
        payload: The PDU bytes after the function code.
        offset: Absolute byte offset of the PDU, for error messages.

    Returns:
        Dict with ``function_code``, ``function`` (name), ``role`` and the
        decoded fields (or ``exception_code``/``exception`` for an exception
        response, or raw ``data`` for an unsupported public code).

    Raises:
        ValueError: When the payload fits neither the request nor the
            response layout of its function code.
    """
    if function_code & 0x80:
        if len(payload) != 1:
            raise ValueError(
                f"Modbus offset {offset}: exception response carries {len(payload)} "
                "data byte(s), expected exactly 1 exception code"
            )
        code = payload[0]
        return {
            "function_code": function_code & 0x7F,
            "function": _FUNCTION_NAMES.get(function_code & 0x7F, "Unknown"),
            "role": "exception",
            "exception_code": code,
            "exception": _EXCEPTION_NAMES.get(code, f"UNKNOWN ({code})"),
        }
    decoded: dict[str, Any] = {
        "function_code": function_code,
        "function": _FUNCTION_NAMES.get(function_code, "Unknown"),
    }
    if function_code in (1, 2, 3, 4):
        # Request: starting address (2) + quantity (2).  Response: byte count
        # (1) + data.  A 4-byte payload whose first byte equals 3 fits both;
        # the request reading is preferred (documented heuristic).
        if len(payload) == 4:
            decoded["role"] = "request"
            decoded["start_address"] = int.from_bytes(payload[0:2], "big")
            decoded["quantity"] = int.from_bytes(payload[2:4], "big")
        elif len(payload) >= 2 and payload[0] == len(payload) - 1:
            decoded["role"] = "response"
            decoded["byte_count"] = payload[0]
            decoded["data"] = payload[1:]
            if function_code in (3, 4):
                decoded["registers"] = _registers(payload[1:], offset + 2)
        else:
            raise ValueError(
                f"Modbus offset {offset}: function {function_code} PDU of "
                f"{len(payload)} byte(s) fits neither the request (4 bytes) nor "
                "the response (byte count + data) layout"
            )
    elif function_code in (5, 6):
        # Request and response are identical: address (2) + value (2).
        if len(payload) != 4:
            raise ValueError(
                f"Modbus offset {offset}: function {function_code} PDU must be "
                f"4 bytes (address + value), got {len(payload)}"
            )
        decoded["role"] = "request_or_response"
        decoded["address"] = int.from_bytes(payload[0:2], "big")
        decoded["value"] = int.from_bytes(payload[2:4], "big")
    elif function_code in (15, 16):
        # Request: address (2) + quantity (2) + byte count (1) + data.
        # Response: address (2) + quantity (2).
        if len(payload) == 4:
            decoded["role"] = "response"
            decoded["start_address"] = int.from_bytes(payload[0:2], "big")
            decoded["quantity"] = int.from_bytes(payload[2:4], "big")
        elif len(payload) >= 5 and payload[4] == len(payload) - 5:
            decoded["role"] = "request"
            decoded["start_address"] = int.from_bytes(payload[0:2], "big")
            decoded["quantity"] = int.from_bytes(payload[2:4], "big")
            decoded["byte_count"] = payload[4]
            decoded["data"] = payload[5:]
            if function_code == 16:
                decoded["registers"] = _registers(payload[5:], offset + 6)
        else:
            raise ValueError(
                f"Modbus offset {offset}: function {function_code} PDU of "
                f"{len(payload)} byte(s) fits neither the request (byte count + "
                "data) nor the response (4 bytes) layout"
            )
    else:
        # A public code outside the supported set: reported raw, never dropped.
        decoded["role"] = "unsupported"
        decoded["data"] = payload
    return decoded


def _parse_tcp(data: bytes) -> list[dict[str, Any]]:
    """Parse back-to-back Modbus TCP ADUs (MBAP header + PDU)."""
    frames: list[dict[str, Any]] = []
    offset = 0
    while offset < len(data):
        if len(data) - offset < 8:
            raise ValueError(
                f"Modbus offset {offset}: {len(data) - offset} byte(s) remain, "
                "shorter than the 7-byte MBAP header plus function code"
            )
        transaction_id = int.from_bytes(data[offset : offset + 2], "big")
        protocol_id = int.from_bytes(data[offset + 2 : offset + 4], "big")
        if protocol_id != 0:
            raise ValueError(
                f"Modbus offset {offset + 2}: MBAP protocol identifier is "
                f"{protocol_id}, the Modbus protocol requires 0"
            )
        length = int.from_bytes(data[offset + 4 : offset + 6], "big")
        if length < 2:
            raise ValueError(
                f"Modbus offset {offset + 4}: MBAP length {length} cannot cover "
                "the unit identifier and a function code"
            )
        if offset + 6 + length > len(data):
            raise ValueError(
                f"Modbus offset {offset}: MBAP declares {length} bytes but only "
                f"{len(data) - offset - 6} remain (truncated frame)"
            )
        unit_id = data[offset + 6]
        function_code = data[offset + 7]
        payload = data[offset + 8 : offset + 6 + length]
        frame = {
            "framing": "tcp",
            "offset": offset,
            "transaction_id": transaction_id,
            "protocol_id": protocol_id,
            "length": length,
            "unit_id": unit_id,
        }
        frame.update(_decode_pdu(function_code, payload, offset + 7))
        frames.append(frame)
        offset += 6 + length
    return frames


def _parse_rtu(data: bytes) -> list[dict[str, Any]]:
    """Parse one Modbus RTU frame (RTU framing is by line silence, not length)."""
    if len(data) < 4:
        raise ValueError(
            f"Modbus offset 0: RTU frame of {len(data)} byte(s), minimum is 4 "
            "(address + function code + CRC)"
        )
    declared_crc = int.from_bytes(data[-2:], "little")
    computed_crc = crc16_modbus(data[:-2])
    if declared_crc != computed_crc:
        raise ValueError(
            f"Modbus offset {len(data) - 2}: CRC-16 mismatch (frame carries "
            f"0x{declared_crc:04X}, computed 0x{computed_crc:04X})"
        )
    frame = {
        "framing": "rtu",
        "offset": 0,
        "address": data[0],
    }
    frame.update(_decode_pdu(data[1], data[2:-2], 1))
    return [frame]


def parse_modbus_frames(data: bytes, framing: str = "rtu") -> dict[str, Any]:
    """Parse Modbus frames from raw bytes, read-only.

    Per the MODBUS Application Protocol Specification V1.1b3: with
    ``framing="tcp"``, ``data`` holds one or more back-to-back MBAP-framed
    ADUs (transaction identifier, protocol identifier 0, length, unit
    identifier, PDU); with ``framing="rtu"``, ``data`` holds exactly one
    serial frame (address + PDU + CRC-16, polynomial 0xA001, verified) --
    RTU frame boundaries exist only as line silence, so a byte dump cannot be
    split into multiple RTU frames without timing information.

    Args:
        data: The raw frame bytes.
        framing: ``"rtu"`` or ``"tcp"``.

    Returns:
        Dict with ``frames``: a list of per-frame dicts carrying the framing
        fields (MBAP header fields for TCP; address for RTU) and the decoded
        PDU (``function_code``, ``function``, ``role``, request/response
        fields, or ``exception_code``/``exception`` for an exception
        response).

    Raises:
        ValueError: On an unknown framing name, a non-zero MBAP protocol
            identifier, truncated frames, an RTU CRC mismatch, or a PDU that
            fits none of its function code's layouts -- always naming the
            byte offset.
    """
    if framing not in ("rtu", "tcp"):
        raise ValueError(f"framing must be 'rtu' or 'tcp', got {framing!r}")
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValueError(f"data must be bytes, got {type(data).__name__}")
    data = bytes(data)
    if not data:
        raise ValueError("Modbus offset 0: input is empty")
    frames = _parse_tcp(data) if framing == "tcp" else _parse_rtu(data)
    return {"frames": frames}
