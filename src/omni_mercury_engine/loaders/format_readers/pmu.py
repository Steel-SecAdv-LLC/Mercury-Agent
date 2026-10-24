# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Read-only IEEE C37.118.2-2011 synchrophasor frame reader.

Parses the frame structure of IEEE Std C37.118.2-2011 ("IEEE Standard for
Synchrophasor Data Transfer for Power Systems"): the SYNC word (0xAA lead
byte plus frame type/version), FRAMESIZE, IDCODE, SOC and FRACSEC; for
configuration frames CFG-1/CFG-2 (frame types 2/3) and CFG-3 (frame type 5)
the station and channel layout (names, FORMAT word, phasor/analog/digital
counts, nominal frequency, data rate); and for data frames (type 0) the
per-PMU measurements, with phasors returned as numpy complex values honoring
the FORMAT word (rectangular/polar, 16-bit integer/32-bit float).

A data frame needs a configuration: the reader applies the configuration
frame with the same IDCODE seen earlier in the stream, and refuses a data
frame whose IDCODE has no configuration yet.  Every frame's CRC-CCITT check
word (polynomial X^16+X^12+X^5+1, initial value 0xFFFF, no final mask, per
the standard) is verified -- a bad checksum is a loud ``ValueError``.

Honest limits: integer phasor/analog values are returned unscaled (the
configuration's conversion-factor words are exposed so a caller can apply
them); command frame payloads beyond the CMD word and header-frame text are
returned raw.  The reader never executes or writes anything.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["read_pmu_frames"]

_FRAME_TYPES = {
    0: "data",
    1: "header",
    2: "cfg1",
    3: "cfg2",
    4: "command",
    5: "cfg3",
}

#: Minimum frame size: SYNC(2) + FRAMESIZE(2) + IDCODE(2) + SOC(4) + FRACSEC(4) + CHK(2).
_MIN_FRAME = 16


def crc_ccitt(data: bytes) -> int:
    """CRC-CCITT per IEEE C37.118.2-2011: poly 0x1021, init 0xFFFF, no final mask.

    Args:
        data: The bytes covered by the check word (the whole frame except CHK).

    Returns:
        The 16-bit CRC value.
    """
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


class _Cursor:
    """Bounded reader over a frame payload, failing loudly with positions."""

    def __init__(self, payload: bytes, base_offset: int) -> None:
        """Wrap ``payload``; ``base_offset`` is its absolute offset for errors."""
        self.payload = payload
        self.base = base_offset
        self.pos = 0

    def take(self, n: int, what: str) -> bytes:
        """Consume ``n`` bytes or raise ``ValueError`` naming the field."""
        if self.pos + n > len(self.payload):
            raise ValueError(
                f"PMU frame offset {self.base + self.pos}: truncated while reading "
                f"{what} ({n} bytes needed, {len(self.payload) - self.pos} remain)"
            )
        chunk = self.payload[self.pos : self.pos + n]
        self.pos += n
        return chunk

    def u16(self, what: str) -> int:
        """Consume a big-endian unsigned 16-bit field."""
        return int.from_bytes(self.take(2, what), "big")

    def i16(self, what: str) -> int:
        """Consume a big-endian signed 16-bit field."""
        return int.from_bytes(self.take(2, what), "big", signed=True)

    def u32(self, what: str) -> int:
        """Consume a big-endian unsigned 32-bit field."""
        return int.from_bytes(self.take(4, what), "big")

    def f32(self, what: str) -> float:
        """Consume a big-endian IEEE-754 single-precision field."""
        return float(struct.unpack(">f", self.take(4, what))[0])


def _decode_format(word: int) -> dict[str, Any]:
    """Decode the FORMAT word (C37.118.2-2011 Table 9) into named flags."""
    return {
        "word": word,
        "coordinates": "polar" if word & 0x1 else "rectangular",
        "phasor_type": "float" if word & 0x2 else "int",
        "analog_type": "float" if word & 0x4 else "int",
        "freq_type": "float" if word & 0x8 else "int",
    }


def _parse_cfg12(cur: _Cursor) -> dict[str, Any]:
    """Parse a CFG-1/CFG-2 frame body (identical layouts, Table 8)."""
    time_base = cur.u32("TIME_BASE")
    num_pmu = cur.u16("NUM_PMU")
    pmus: list[dict[str, Any]] = []
    for index in range(num_pmu):
        station = cur.take(16, f"PMU {index} STN").decode("ascii", errors="replace").rstrip()
        idcode = cur.u16(f"PMU {index} IDCODE")
        fmt = _decode_format(cur.u16(f"PMU {index} FORMAT"))
        phnmr = cur.u16(f"PMU {index} PHNMR")
        annmr = cur.u16(f"PMU {index} ANNMR")
        dgnmr = cur.u16(f"PMU {index} DGNMR")
        n_names = phnmr + annmr + 16 * dgnmr
        names = [
            cur.take(16, f"PMU {index} CHNAM {k}").decode("ascii", errors="replace").rstrip()
            for k in range(n_names)
        ]
        phunit = [cur.u32(f"PMU {index} PHUNIT {k}") for k in range(phnmr)]
        anunit = [cur.u32(f"PMU {index} ANUNIT {k}") for k in range(annmr)]
        digunit = [cur.u32(f"PMU {index} DIGUNIT {k}") for k in range(dgnmr)]
        fnom = cur.u16(f"PMU {index} FNOM")
        cfgcnt = cur.u16(f"PMU {index} CFGCNT")
        pmus.append(
            {
                "station": station,
                "idcode": idcode,
                "format": fmt,
                "phasor_count": phnmr,
                "analog_count": annmr,
                "digital_count": dgnmr,
                "phasor_names": names[:phnmr],
                "analog_names": names[phnmr : phnmr + annmr],
                "digital_names": names[phnmr + annmr :],
                "phasor_units": phunit,
                "analog_units": anunit,
                "digital_units": digunit,
                "nominal_freq_hz": 50.0 if fnom & 0x1 else 60.0,
                "config_count": cfgcnt,
            }
        )
    data_rate = cur.i16("DATA_RATE")
    return {"time_base": time_base & 0xFFFFFF, "pmus": pmus, "data_rate": data_rate}


def _take_name(cur: _Cursor, what: str) -> str:
    """Consume one CFG-3 variable-length name (1 length byte + characters)."""
    length = cur.take(1, f"{what} length")[0]
    return cur.take(length, what).decode("utf-8", errors="replace")


def _parse_cfg3(cur: _Cursor) -> dict[str, Any]:
    """Parse a CFG-3 frame body (C37.118.2-2011 Table 10, variable-length names)."""
    cont_idx = cur.u16("CONT_IDX")
    time_base = cur.u32("TIME_BASE")
    num_pmu = cur.u16("NUM_PMU")
    pmus: list[dict[str, Any]] = []
    for index in range(num_pmu):
        station = _take_name(cur, f"PMU {index} STN")
        idcode = cur.u16(f"PMU {index} IDCODE")
        g_pmu_id = cur.take(16, f"PMU {index} G_PMU_ID")
        fmt = _decode_format(cur.u16(f"PMU {index} FORMAT"))
        phnmr = cur.u16(f"PMU {index} PHNMR")
        annmr = cur.u16(f"PMU {index} ANNMR")
        dgnmr = cur.u16(f"PMU {index} DGNMR")
        n_names = phnmr + annmr + 16 * dgnmr
        names = [_take_name(cur, f"PMU {index} CHNAM {k}") for k in range(n_names)]
        phscale = [cur.take(12, f"PMU {index} PHSCALE {k}") for k in range(phnmr)]
        anscale = [cur.take(8, f"PMU {index} ANSCALE {k}") for k in range(annmr)]
        digunit = [cur.u32(f"PMU {index} DIGUNIT {k}") for k in range(dgnmr)]
        lat = cur.f32(f"PMU {index} PMU_LAT")
        lon = cur.f32(f"PMU {index} PMU_LON")
        elev = cur.f32(f"PMU {index} PMU_ELEV")
        svc_class = cur.take(1, f"PMU {index} SVC_CLASS").decode("ascii", errors="replace")
        window = cur.u32(f"PMU {index} WINDOW")
        grp_dly = cur.u32(f"PMU {index} GRP_DLY")
        fnom = cur.u16(f"PMU {index} FNOM")
        cfgcnt = cur.u16(f"PMU {index} CFGCNT")
        pmus.append(
            {
                "station": station,
                "idcode": idcode,
                "global_pmu_id": g_pmu_id,
                "format": fmt,
                "phasor_count": phnmr,
                "analog_count": annmr,
                "digital_count": dgnmr,
                "phasor_names": names[:phnmr],
                "analog_names": names[phnmr : phnmr + annmr],
                "digital_names": names[phnmr + annmr :],
                "phasor_scales_raw": phscale,
                "analog_scales_raw": anscale,
                "digital_units": digunit,
                "latitude": lat,
                "longitude": lon,
                "elevation": elev,
                "service_class": svc_class,
                "window": window,
                "group_delay": grp_dly,
                "nominal_freq_hz": 50.0 if fnom & 0x1 else 60.0,
                "config_count": cfgcnt,
            }
        )
    data_rate = cur.i16("DATA_RATE")
    return {
        "continuation_index": cont_idx,
        "time_base": time_base & 0xFFFFFF,
        "pmus": pmus,
        "data_rate": data_rate,
    }


def _read_phasor(cur: _Cursor, fmt: dict[str, Any], what: str) -> complex:
    """One phasor as a complex value, honoring the FORMAT word.

    Integer rectangular: two signed 16-bit values (real, imaginary), unscaled.
    Integer polar: unsigned 16-bit magnitude and signed 16-bit angle in units
    of 1e-4 radian.  Float: two IEEE-754 singles (rectangular real/imaginary,
    or polar magnitude/angle-in-radians), per C37.118.2-2011 Table 5.
    """
    if fmt["phasor_type"] == "float":
        first = cur.f32(f"{what} first word")
        second = cur.f32(f"{what} second word")
        if fmt["coordinates"] == "polar":
            return complex(first * np.cos(second), first * np.sin(second))
        return complex(first, second)
    if fmt["coordinates"] == "polar":
        magnitude = int.from_bytes(cur.take(2, f"{what} magnitude"), "big")
        angle = int.from_bytes(cur.take(2, f"{what} angle"), "big", signed=True) * 1e-4
        return complex(magnitude * np.cos(angle), magnitude * np.sin(angle))
    real = int.from_bytes(cur.take(2, f"{what} real"), "big", signed=True)
    imag = int.from_bytes(cur.take(2, f"{what} imaginary"), "big", signed=True)
    return complex(real, imag)


def _parse_data(cur: _Cursor, config: dict[str, Any]) -> dict[str, Any]:
    """Parse a data frame body against ``config`` (C37.118.2-2011 Table 5)."""
    measurements: list[dict[str, Any]] = []
    for pmu in config["pmus"]:
        fmt = pmu["format"]
        stat = cur.u16(f"{pmu['station']} STAT")
        phasors = np.array(
            [
                _read_phasor(cur, fmt, f"{pmu['station']} phasor {k}")
                for k in range(pmu["phasor_count"])
            ],
            dtype=np.complex128,
        )
        if fmt["freq_type"] == "float":
            freq = cur.f32(f"{pmu['station']} FREQ")
            dfreq = cur.f32(f"{pmu['station']} DFREQ")
        else:
            freq = cur.i16(f"{pmu['station']} FREQ") * 1e-3  # mHz deviation
            dfreq = cur.i16(f"{pmu['station']} DFREQ") * 1e-2  # 0.01 Hz/s
        if fmt["analog_type"] == "float":
            analogs = [cur.f32(f"{pmu['station']} ANALOG {k}") for k in range(pmu["analog_count"])]
        else:
            analogs = [
                float(cur.i16(f"{pmu['station']} ANALOG {k}")) for k in range(pmu["analog_count"])
            ]
        digitals = [cur.u16(f"{pmu['station']} DIGITAL {k}") for k in range(pmu["digital_count"])]
        measurements.append(
            {
                "station": pmu["station"],
                "stat": stat,
                "phasors": phasors,
                "freq_deviation_hz": freq,
                "rocof_hz_per_s": dfreq,
                "analogs": np.array(analogs, dtype=np.float64),
                "digitals": digitals,
            }
        )
    return {"pmus": measurements}


def read_pmu_frames(path_or_bytes: str | Path | bytes | bytearray | memoryview) -> dict[str, Any]:
    """Read IEEE C37.118.2-2011 synchrophasor frames from a file or byte string.

    Parses every frame's SYNC, frame type, FRAMESIZE, IDCODE, SOC and FRACSEC
    and verifies its CRC-CCITT check word.  Configuration frames (CFG-1/CFG-2,
    frame types 2/3, and CFG-3, frame type 5) yield the station/channel
    layout; data frames (type 0) are decoded against the configuration frame
    with the same IDCODE seen earlier in the stream, with phasors returned as
    numpy complex values honoring the FORMAT word (rectangular/polar,
    int/float).  Header frames return their text; command frames their CMD
    word.

    Args:
        path_or_bytes: Filesystem path of a capture, or the raw frame bytes.

    Returns:
        Dict with ``frames`` (list of per-frame dicts, each with ``offset``,
        ``frame_type`` (name), ``version``, ``frame_size``, ``idcode``,
        ``soc``, ``fracsec`` (raw 24-bit fraction), ``time_quality``, plus the
        type-specific payload: ``config`` for configuration frames, ``pmus``
        for data frames, ``text`` for header frames, ``command`` for command
        frames) and ``configs`` (dict mapping IDCODE to its latest parsed
        configuration).

    Raises:
        ValueError: On a bad SYNC byte, unknown frame type, truncated frame,
            CRC mismatch (always naming offset, expected and computed values),
            a data frame whose IDCODE has no configuration yet, or trailing
            payload bytes a frame's declared layout does not account for.
    """
    if isinstance(path_or_bytes, (bytes, bytearray, memoryview)):
        data = bytes(path_or_bytes)
    else:
        data = Path(path_or_bytes).read_bytes()
    if not data:
        raise ValueError("PMU frame offset 0: input is empty")

    frames: list[dict[str, Any]] = []
    configs: dict[int, dict[str, Any]] = {}
    offset = 0
    while offset < len(data):
        if len(data) - offset < _MIN_FRAME:
            raise ValueError(
                f"PMU frame offset {offset}: {len(data) - offset} bytes remain, "
                f"shorter than the {_MIN_FRAME}-byte minimum frame"
            )
        if data[offset] != 0xAA:
            raise ValueError(
                f"PMU frame offset {offset}: SYNC lead byte is "
                f"0x{data[offset]:02X}, expected 0xAA"
            )
        type_code = (data[offset + 1] >> 4) & 0x7
        version = data[offset + 1] & 0xF
        frame_type = _FRAME_TYPES.get(type_code)
        if frame_type is None:
            raise ValueError(f"PMU frame offset {offset + 1}: unknown frame type {type_code}")
        frame_size = int.from_bytes(data[offset + 2 : offset + 4], "big")
        if frame_size < _MIN_FRAME:
            raise ValueError(
                f"PMU frame offset {offset + 2}: FRAMESIZE {frame_size} below the "
                f"{_MIN_FRAME}-byte minimum"
            )
        if offset + frame_size > len(data):
            raise ValueError(
                f"PMU frame offset {offset}: FRAMESIZE {frame_size} overruns the "
                f"input ({len(data) - offset} bytes remain)"
            )
        declared_crc = int.from_bytes(data[offset + frame_size - 2 : offset + frame_size], "big")
        computed_crc = crc_ccitt(data[offset : offset + frame_size - 2])
        if declared_crc != computed_crc:
            raise ValueError(
                f"PMU frame offset {offset}: CRC-CCITT mismatch "
                f"(frame carries 0x{declared_crc:04X}, computed 0x{computed_crc:04X})"
            )
        idcode = int.from_bytes(data[offset + 4 : offset + 6], "big")
        soc = int.from_bytes(data[offset + 6 : offset + 10], "big")
        fracsec_word = int.from_bytes(data[offset + 10 : offset + 14], "big")
        frame: dict[str, Any] = {
            "offset": offset,
            "frame_type": frame_type,
            "version": version,
            "frame_size": frame_size,
            "idcode": idcode,
            "soc": soc,
            "time_quality": (fracsec_word >> 24) & 0xFF,
            "fracsec": fracsec_word & 0xFFFFFF,
        }
        cur = _Cursor(data[offset + 14 : offset + frame_size - 2], offset + 14)
        if frame_type in ("cfg1", "cfg2"):
            config = _parse_cfg12(cur)
            frame["config"] = config
            configs[idcode] = config
        elif frame_type == "cfg3":
            config = _parse_cfg3(cur)
            frame["config"] = config
            configs[idcode] = config
        elif frame_type == "data":
            active = configs.get(idcode)
            if active is None:
                raise ValueError(
                    f"PMU frame offset {offset}: data frame for IDCODE {idcode} "
                    "arrived before any configuration frame for that IDCODE"
                )
            frame.update(_parse_data(cur, active))
        elif frame_type == "header":
            frame["text"] = cur.take(len(cur.payload), "header text").decode(
                "ascii", errors="replace"
            )
        else:  # command
            frame["command"] = cur.u16("CMD")
            frame["extended_data"] = cur.take(len(cur.payload) - cur.pos, "EXTFRAME")
        if frame_type != "header" and cur.pos != len(cur.payload):
            raise ValueError(
                f"PMU frame offset {offset + 14 + cur.pos}: {len(cur.payload) - cur.pos} "
                f"unaccounted payload byte(s) after the declared {frame_type} layout"
            )
        frames.append(frame)
        offset += frame_size
    return {"frames": frames, "configs": configs}
