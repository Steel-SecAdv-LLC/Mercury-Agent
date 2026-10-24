# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Format readers verified against byte-exact, hand-constructed samples.

Every sample below is built in the test, field by field, per the published
specification (RINEX 3.05 / 2.11, WMO FM 92 GRIB Edition 2, IEEE
C37.118.2-2011, ASPRS LAS 1.4 R15, Modbus V1.1b3, IEEE 1815-2012).  The
frame CRCs are computed with independent in-test implementations (table
driven, anchored to the published check values for "123456789") so a reader
bug cannot hide behind its own checksum code.  Malformed variants (wrong
magic, bad CRC, truncation) must raise ``ValueError``.  No network access,
no downloaded fixtures.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import numpy as np
import pytest

from omni_mercury_engine.loaders.format_readers import (
    parse_dnp3_frames,
    parse_modbus_frames,
    read_grib2_sections,
    read_las_points,
    read_pmu_frames,
    read_rinex_obs,
)

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Independent CRC implementations (table-driven, unlike the readers' bitwise
# loops), anchored to the published check values for b"123456789".
# ---------------------------------------------------------------------------


def _make_table_msb(poly: int) -> list[int]:
    table = []
    for i in range(256):
        crc = i << 8
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
        table.append(crc)
    return table


def _make_table_lsb(poly_reflected: int) -> list[int]:
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = (crc >> 1) ^ poly_reflected if crc & 1 else crc >> 1
        table.append(crc)
    return table


_CCITT_TABLE = _make_table_msb(0x1021)
_MODBUS_TABLE = _make_table_lsb(0xA001)
_DNP_TABLE = _make_table_lsb(0xA6BC)


def crc_ccitt_ref(data: bytes) -> int:
    """CRC-CCITT (FALSE): init 0xFFFF, poly 0x1021 (check value 0x29B1)."""
    crc = 0xFFFF
    for byte in data:
        crc = ((crc << 8) & 0xFFFF) ^ _CCITT_TABLE[((crc >> 8) ^ byte) & 0xFF]
    return crc


def crc_modbus_ref(data: bytes) -> int:
    """CRC-16/MODBUS: init 0xFFFF, reflected poly 0xA001 (check value 0x4B37)."""
    crc = 0xFFFF
    for byte in data:
        crc = (crc >> 8) ^ _MODBUS_TABLE[(crc ^ byte) & 0xFF]
    return crc


def crc_dnp_ref(data: bytes) -> int:
    """CRC-16/DNP: init 0, reflected poly 0xA6BC, complemented (check 0xEA82)."""
    crc = 0
    for byte in data:
        crc = (crc >> 8) ^ _DNP_TABLE[(crc ^ byte) & 0xFF]
    return (~crc) & 0xFFFF


def test_reference_crcs_match_published_check_values() -> None:
    assert crc_ccitt_ref(b"123456789") == 0x29B1
    assert crc_modbus_ref(b"123456789") == 0x4B37
    assert crc_dnp_ref(b"123456789") == 0xEA82


# ---------------------------------------------------------------------------
# RINEX observation files
# ---------------------------------------------------------------------------


def _rinex_line(content: str, label: str) -> str:
    return content.ljust(60) + label


def _rinex3_text() -> str:
    lines = [
        _rinex_line(f"{'3.05':>9}{'':11}OBSERVATION DATA    M", "RINEX VERSION / TYPE"),
        _rinex_line("TEST MARKER", "MARKER NAME"),
        _rinex_line("SOME PROGRAM        SOMEONE", "PGM / RUN BY / DATE"),
        _rinex_line("G    4 C1C L1C D1C S1C", "SYS / # / OBS TYPES"),
        _rinex_line("E    2 C1X L1X", "SYS / # / OBS TYPES"),
        _rinex_line("", "END OF HEADER"),
        "> 2022 01 02 03 04  5.0000000  0  2",
        "G01" + f"{23619095.45:14.3f}" + "  " + f"{124127654.321:14.3f}" + "  ",
        "E11" + f"{25101233.5:14.3f}" + "  ",
    ]
    return "\n".join(lines) + "\n"


def _rinex2_text() -> str:
    lines = [
        _rinex_line(f"{'2.11':>9}{'':11}OBSERVATION DATA    G (GPS)", "RINEX VERSION / TYPE"),
        _rinex_line("OLD MARKER", "MARKER NAME"),
        _rinex_line(f"{2:6d}    C1    L1", "# / TYPES OF OBSERV"),
        _rinex_line("", "END OF HEADER"),
        f" 22  1  1  0  0{0.0:11.7f}  0  1G01",
        f"{23619095.45:14.3f}" + "  " + f"{124127654.321:14.3f}" + "  ",
    ]
    return "\n".join(lines) + "\n"


class TestRinexObs:
    def test_rinex3_header_and_epochs(self, tmp_path: Path) -> None:
        path = tmp_path / "obs.rnx"
        path.write_text(_rinex3_text())
        result = read_rinex_obs(path)
        assert result["version"] == 3.05
        assert result["file_type"] == "O"
        assert result["marker"] == "TEST MARKER"
        assert result["obs_types"]["G"] == ["C1C", "L1C", "D1C", "S1C"]
        assert result["obs_types"]["E"] == ["C1X", "L1X"]
        # Unknown header label preserved, never dropped.
        assert any(r["label"] == "PGM / RUN BY / DATE" for r in result["raw_header"])
        assert len(result["epochs"]) == 1
        epoch = result["epochs"][0]
        assert epoch["time"] == (2022, 1, 2, 3, 4, 5.0)
        assert epoch["flag"] == 0
        g01 = epoch["satellites"]["G01"]
        assert g01["C1C"] == pytest.approx(23619095.45)
        assert g01["L1C"] == pytest.approx(124127654.321)
        assert g01["D1C"] is None  # blank field
        assert g01["S1C"] is None
        e11 = epoch["satellites"]["E11"]
        assert e11["C1X"] == pytest.approx(25101233.5)
        assert e11["L1X"] is None

    def test_rinex211_types_and_epochs(self, tmp_path: Path) -> None:
        path = tmp_path / "obs.94o"
        path.write_text(_rinex2_text())
        result = read_rinex_obs(path)
        assert result["version"] == 2.11
        assert result["obs_types"]["*"] == ["C1", "L1"]
        epoch = result["epochs"][0]
        assert epoch["time"] == (2022, 1, 1, 0, 0, 0.0)
        sat = epoch["satellites"]["G01"]
        assert sat["C1"] == pytest.approx(23619095.45)
        assert sat["L1"] == pytest.approx(124127654.321)

    def test_non_observation_file_is_refused(self, tmp_path: Path) -> None:
        bad = _rinex_line(f"{'3.05':>9}{'':11}NAVIGATION DATA", "RINEX VERSION / TYPE")
        path = tmp_path / "nav.rnx"
        path.write_text(bad + "\n")
        with pytest.raises(ValueError, match=r"line 1.*not an observation file"):
            read_rinex_obs(path)

    def test_missing_end_of_header_is_refused(self, tmp_path: Path) -> None:
        text = _rinex3_text().replace(_rinex_line("", "END OF HEADER") + "\n", "")
        path = tmp_path / "trunc.rnx"
        path.write_text(text)
        with pytest.raises(ValueError, match="END OF HEADER"):
            read_rinex_obs(path)

    def test_truncated_epoch_is_refused(self, tmp_path: Path) -> None:
        # Epoch declares 2 satellites but only 1 line follows.
        text = _rinex3_text().rsplit("\n", 2)[0] + "\n"
        path = tmp_path / "trunc2.rnx"
        path.write_text(text)
        with pytest.raises(ValueError, match="declares 2 satellites"):
            read_rinex_obs(path)

    def test_bad_version_field_is_refused(self, tmp_path: Path) -> None:
        bad = _rinex_line(f"{'x.yz':>9}{'':11}OBSERVATION DATA", "RINEX VERSION / TYPE")
        path = tmp_path / "bad.rnx"
        path.write_text(bad + "\n" + _rinex_line("", "END OF HEADER") + "\n")
        with pytest.raises(ValueError, match=r"line 1.*version"):
            read_rinex_obs(path)


# ---------------------------------------------------------------------------
# GRIB2
# ---------------------------------------------------------------------------


def _grib2_section(number: int, body: bytes) -> bytes:
    length = 5 + len(body)
    return length.to_bytes(4, "big") + bytes([number]) + body


def _grib2_message(centre: int = 7, grid_template: int = 0, product_template: int = 0) -> bytes:
    ident_body = (
        centre.to_bytes(2, "big")  # originating centre
        + (0).to_bytes(2, "big")  # subcentre
        + bytes([2, 1, 1])  # master tables, local tables, significance
        + (2022).to_bytes(2, "big")
        + bytes([1, 2, 3, 4, 5])  # month day hour minute second
        + bytes([0, 1])  # production status (octet 20), type of data (octet 21)
    )
    sec1 = _grib2_section(1, ident_body)
    grid_body = (
        bytes([0])  # source of grid definition
        + (65160).to_bytes(4, "big")  # number of data points
        + bytes([0, 0])  # optional-list octets, interpretation
        + grid_template.to_bytes(2, "big")
    )
    sec3 = _grib2_section(3, grid_body)
    product_body = (0).to_bytes(2, "big") + product_template.to_bytes(2, "big")
    sec4 = _grib2_section(4, product_body)
    repr_body = (65160).to_bytes(4, "big") + (0).to_bytes(2, "big")
    sec5 = _grib2_section(5, repr_body)
    sec6 = _grib2_section(6, bytes([255]))  # bitmap: none
    sec7 = _grib2_section(7, b"")
    body = sec1 + sec3 + sec4 + sec5 + sec6 + sec7
    total = 16 + len(body) + 4
    indicator = b"GRIB" + b"\x00\x00" + bytes([0, 2]) + total.to_bytes(8, "big")
    return indicator + body + b"7777"


class TestGrib2Sections:
    def test_single_message_sections_and_identification(self, tmp_path: Path) -> None:
        path = tmp_path / "one.grib2"
        path.write_bytes(_grib2_message(centre=7, grid_template=30, product_template=8))
        result = read_grib2_sections(path)
        assert len(result["messages"]) == 1
        msg = result["messages"][0]
        assert msg["edition"] == 2
        assert msg["discipline"] == 0
        assert [s["number"] for s in msg["sections"]] == [1, 3, 4, 5, 6, 7]
        ident = msg["identification"]
        assert ident["centre"] == 7
        assert (ident["year"], ident["month"], ident["day"]) == (2022, 1, 2)
        assert (ident["hour"], ident["minute"], ident["second"]) == (3, 4, 5)
        assert msg["grid_definitions"][0]["template_number"] == 30
        assert msg["product_definitions"][0]["template_number"] == 8
        # Raw section bytes exposed, undecoded.
        assert msg["grid_definitions"][0]["raw"][4] == 3
        sec5 = next(s for s in msg["sections"] if s["number"] == 5)
        assert sec5["data_representation_template_number"] == 0

    def test_multi_message_file(self, tmp_path: Path) -> None:
        path = tmp_path / "two.grib2"
        path.write_bytes(_grib2_message(centre=7) + _grib2_message(centre=98))
        result = read_grib2_sections(path)
        assert len(result["messages"]) == 2
        assert result["messages"][0]["identification"]["centre"] == 7
        assert result["messages"][1]["identification"]["centre"] == 98
        assert result["messages"][1]["offset"] == result["messages"][0]["total_length"]

    def test_wrong_magic_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.grib2"
        path.write_bytes(b"GRIX" + _grib2_message()[4:])
        with pytest.raises(ValueError, match=r"offset 0.*GRIB"):
            read_grib2_sections(path)

    def test_wrong_edition_is_refused(self, tmp_path: Path) -> None:
        message = bytearray(_grib2_message())
        message[7] = 1
        path = tmp_path / "ed1.grib2"
        path.write_bytes(bytes(message))
        with pytest.raises(ValueError, match="edition 1"):
            read_grib2_sections(path)

    def test_truncated_message_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "trunc.grib2"
        path.write_bytes(_grib2_message()[:-10])
        with pytest.raises(ValueError, match="truncated"):
            read_grib2_sections(path)

    def test_missing_end_marker_is_refused(self, tmp_path: Path) -> None:
        message = bytearray(_grib2_message())
        message[-4:] = b"8888"
        path = tmp_path / "noend.grib2"
        path.write_bytes(bytes(message))
        with pytest.raises(ValueError, match="7777"):
            read_grib2_sections(path)


# ---------------------------------------------------------------------------
# IEEE C37.118.2 synchrophasor frames
# ---------------------------------------------------------------------------

_PMU_IDCODE = 7734


def _pmu_frame(frame_type: int, idcode: int, body: bytes, soc: int = 1_460_000_000) -> bytes:
    size = 16 + len(body)
    head = (
        bytes([0xAA, (frame_type << 4) | 0x01])
        + size.to_bytes(2, "big")
        + idcode.to_bytes(2, "big")
        + soc.to_bytes(4, "big")
        + (0).to_bytes(4, "big")
    )
    return head + body + crc_ccitt_ref(head + body).to_bytes(2, "big")


def _pmu_cfg2(format_word: int) -> bytes:
    names = ["VA", "VB", "ANALOG1"] + [f"BREAKER {k} STATUS" for k in range(16)]
    body = (1_000_000).to_bytes(4, "big") + (1).to_bytes(2, "big")  # TIME_BASE, NUM_PMU
    body += b"STATION A".ljust(16)
    body += _PMU_IDCODE.to_bytes(2, "big")
    body += format_word.to_bytes(2, "big")
    body += (2).to_bytes(2, "big") + (1).to_bytes(2, "big") + (1).to_bytes(2, "big")
    for name in names:
        body += name.encode("ascii")[:16].ljust(16)
    body += (1).to_bytes(4, "big") * 2  # PHUNIT x2
    body += (0).to_bytes(4, "big")  # ANUNIT
    body += (0xFFFF).to_bytes(4, "big")  # DIGUNIT
    body += (0).to_bytes(2, "big") + (1).to_bytes(2, "big")  # FNOM (60 Hz), CFGCNT
    body += (30).to_bytes(2, "big")  # DATA_RATE
    return _pmu_frame(3, _PMU_IDCODE, body)


def _pmu_data_float_polar() -> tuple[bytes, float, float]:
    magnitude, angle = 14635.0, -2.0943951023931953
    body = (0).to_bytes(2, "big")  # STAT
    body += struct.pack(">ff", magnitude, 0.0)
    body += struct.pack(">ff", magnitude, angle)
    body += struct.pack(">ff", 0.0, 0.0)  # FREQ, DFREQ
    body += struct.pack(">f", 100.0)  # ANALOG
    body += (0x3C12).to_bytes(2, "big")  # DIGITAL
    return _pmu_frame(0, _PMU_IDCODE, body), magnitude, angle


class TestPmuFrames:
    def test_cfg2_station_and_channel_layout(self) -> None:
        result = read_pmu_frames(_pmu_cfg2(0x0F))
        frame = result["frames"][0]
        assert frame["frame_type"] == "cfg2"
        assert frame["idcode"] == _PMU_IDCODE
        config = frame["config"]
        assert config["time_base"] == 1_000_000
        assert config["data_rate"] == 30
        pmu = config["pmus"][0]
        assert pmu["station"] == "STATION A"
        assert pmu["format"] == {
            "word": 0x0F,
            "coordinates": "polar",
            "phasor_type": "float",
            "analog_type": "float",
            "freq_type": "float",
        }
        assert pmu["phasor_names"] == ["VA", "VB"]
        assert pmu["analog_names"] == ["ANALOG1"]
        assert len(pmu["digital_names"]) == 16
        assert pmu["nominal_freq_hz"] == 60.0

    def test_data_frame_polar_float_phasors_are_complex(self) -> None:
        data_frame, magnitude, angle = _pmu_data_float_polar()
        result = read_pmu_frames(_pmu_cfg2(0x0F) + data_frame)
        frame = result["frames"][1]
        assert frame["frame_type"] == "data"
        measurement = frame["pmus"][0]
        phasors = measurement["phasors"]
        assert phasors.dtype == np.complex128
        assert phasors[0] == pytest.approx(complex(magnitude, 0.0), rel=1e-6)
        angle32 = struct.unpack(">f", struct.pack(">f", angle))[0]
        expected = magnitude * np.exp(1j * angle32)
        assert phasors[1] == pytest.approx(expected, rel=1e-6)
        assert measurement["analogs"][0] == pytest.approx(100.0)
        assert measurement["digitals"] == [0x3C12]

    def test_data_frame_rectangular_int_phasors(self) -> None:
        body = (0).to_bytes(2, "big")
        body += struct.pack(">hh", 100, -200) + struct.pack(">hh", -32000, 123)
        body += struct.pack(">hh", 10, -5)  # FREQ (mHz), DFREQ (0.01 Hz/s)
        body += struct.pack(">h", -42)  # ANALOG int
        body += (0).to_bytes(2, "big")
        data_frame = _pmu_frame(0, _PMU_IDCODE, body)
        result = read_pmu_frames(_pmu_cfg2(0x00) + data_frame)
        measurement = result["frames"][1]["pmus"][0]
        assert measurement["phasors"][0] == complex(100, -200)
        assert measurement["phasors"][1] == complex(-32000, 123)
        assert measurement["freq_deviation_hz"] == pytest.approx(0.010)
        assert measurement["rocof_hz_per_s"] == pytest.approx(-0.05)
        assert measurement["analogs"][0] == -42.0

    def test_bad_crc_is_refused_loudly(self) -> None:
        frame = bytearray(_pmu_cfg2(0x0F))
        frame[20] ^= 0xFF
        with pytest.raises(ValueError, match="CRC-CCITT mismatch"):
            read_pmu_frames(bytes(frame))

    def test_data_frame_without_config_is_refused(self) -> None:
        data_frame, _, _ = _pmu_data_float_polar()
        with pytest.raises(ValueError, match="before any configuration"):
            read_pmu_frames(data_frame)

    def test_bad_sync_byte_is_refused(self) -> None:
        frame = bytearray(_pmu_cfg2(0x0F))
        frame[0] = 0xAB
        with pytest.raises(ValueError, match=r"offset 0.*0xAA"):
            read_pmu_frames(bytes(frame))

    def test_truncated_frame_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"FRAMESIZE.*overruns"):
            read_pmu_frames(_pmu_cfg2(0x0F)[:-8])


# ---------------------------------------------------------------------------
# ASPRS LAS point clouds
# ---------------------------------------------------------------------------


def _las_header(
    version: tuple[int, int],
    point_format: int,
    record_length: int,
    count: int,
    header_size: int,
) -> bytearray:
    header = bytearray(header_size)
    header[0:4] = b"LASF"
    header[24] = version[0]
    header[25] = version[1]
    header[94:96] = header_size.to_bytes(2, "little")
    header[96:100] = header_size.to_bytes(4, "little")  # points start right after
    header[100:104] = (0).to_bytes(4, "little")  # no VLRs
    header[104] = point_format
    header[105:107] = record_length.to_bytes(2, "little")
    if version[1] < 4:
        header[107:111] = count.to_bytes(4, "little")
    else:
        header[247:255] = count.to_bytes(8, "little")
    header[131:155] = struct.pack("<3d", 0.01, 0.01, 0.001)
    header[155:179] = struct.pack("<3d", 1000.0, 2000.0, 100.0)
    return header


class TestLasPoints:
    def test_las12_format3_points(self, tmp_path: Path) -> None:
        # classification byte 0x45: class 5 (high vegetation) + synthetic bit.
        points = struct.pack(
            "<3lHBBbBHd3H", 150, -230, 4200, 777, 0x11, 0x45, -5, 0, 42, 123.5, 1, 2, 3
        ) + struct.pack("<3lHBBbBHd3H", -100, 50, -75, 0, 0x11, 0x02, 10, 0, 42, 124.5, 9, 8, 7)
        path = tmp_path / "pts.las"
        path.write_bytes(bytes(_las_header((1, 2), 3, 34, 2, 227)) + points)
        result = read_las_points(path)
        assert result["version"] == (1, 2)
        assert result["point_format"] == 3
        assert result["point_count"] == 2
        assert result["points_read"] == 2
        np.testing.assert_allclose(result["x"], [150 * 0.01 + 1000.0, -100 * 0.01 + 1000.0])
        np.testing.assert_allclose(result["y"], [-230 * 0.01 + 2000.0, 50 * 0.01 + 2000.0])
        np.testing.assert_allclose(result["z"], [4200 * 0.001 + 100.0, -75 * 0.001 + 100.0])
        assert result["intensity"].tolist() == [777, 0]
        assert result["classification"].tolist() == [5, 2]
        assert result["classification_flags"].tolist() == [0x45 >> 5, 0]
        np.testing.assert_allclose(result["gps_time"], [123.5, 124.5])
        assert result["red"].tolist() == [1, 9]
        assert result["blue"].tolist() == [3, 7]

    def test_las14_format6_points_and_max_points(self, tmp_path: Path) -> None:
        record = struct.pack("<3lHBBBBhHd", 10, 20, 30, 5, 0x22, 0x01, 17, 0, -1500, 3, 200.25)
        path = tmp_path / "pts14.las"
        path.write_bytes(bytes(_las_header((1, 4), 6, 30, 3, 375)) + record * 3)
        result = read_las_points(path, max_points=2)
        assert result["version"] == (1, 4)
        assert result["point_format"] == 6
        assert result["point_count"] == 3
        assert result["points_read"] == 2
        assert result["x"].shape == (2,)
        assert result["classification"].tolist() == [17, 17]
        np.testing.assert_allclose(result["gps_time"], [200.25, 200.25])

    def test_bad_signature_is_refused(self, tmp_path: Path) -> None:
        header = _las_header((1, 2), 0, 20, 0, 227)
        header[0:4] = b"XASF"
        path = tmp_path / "bad.las"
        path.write_bytes(bytes(header))
        with pytest.raises(ValueError, match="signature"):
            read_las_points(path)

    def test_truncated_point_data_is_refused(self, tmp_path: Path) -> None:
        record = struct.pack("<3lHBBbBH", 1, 2, 3, 4, 0, 0, 0, 0, 0)
        path = tmp_path / "short.las"
        path.write_bytes(bytes(_las_header((1, 2), 0, 20, 5, 227)) + record)
        with pytest.raises(ValueError, match="declares 5 points"):
            read_las_points(path)

    def test_compressed_laz_bit_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "laz.las"
        path.write_bytes(bytes(_las_header((1, 2), 0x80 | 3, 34, 0, 227)))
        with pytest.raises(ValueError, match="compression bit"):
            read_las_points(path)

    def test_unsupported_point_format_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "fmt5.las"
        path.write_bytes(bytes(_las_header((1, 3), 5, 63, 0, 235)))
        with pytest.raises(ValueError, match="format 5 unsupported"):
            read_las_points(path)


# ---------------------------------------------------------------------------
# Modbus
# ---------------------------------------------------------------------------


def _rtu(address: int, pdu: bytes) -> bytes:
    frame = bytes([address]) + pdu
    return frame + crc_modbus_ref(frame).to_bytes(2, "little")


def _tcp(transaction: int, unit: int, pdu: bytes) -> bytes:
    return (
        transaction.to_bytes(2, "big")
        + (0).to_bytes(2, "big")
        + (len(pdu) + 1).to_bytes(2, "big")
        + bytes([unit])
        + pdu
    )


class TestModbusFrames:
    def test_rtu_read_holding_registers_request(self) -> None:
        # The spec's canonical example: slave 0x11, FC 3, start 0x006B, qty 3.
        frame = _rtu(0x11, bytes.fromhex("03006B0003"))
        assert frame[-2:] == bytes.fromhex("7687")  # published example CRC
        result = parse_modbus_frames(frame, framing="rtu")
        decoded = result["frames"][0]
        assert decoded["address"] == 0x11
        assert decoded["function_code"] == 3
        assert decoded["function"] == "Read Holding Registers"
        assert decoded["role"] == "request"
        assert decoded["start_address"] == 0x6B
        assert decoded["quantity"] == 3

    def test_rtu_bad_crc_is_refused(self) -> None:
        frame = bytearray(_rtu(0x11, bytes.fromhex("03006B0003")))
        frame[-1] ^= 0x01
        with pytest.raises(ValueError, match="CRC-16 mismatch"):
            parse_modbus_frames(bytes(frame), framing="rtu")

    def test_tcp_response_registers_decoded(self) -> None:
        pdu = bytes.fromhex("0306022B00000064")  # FC3 response: 3 registers
        result = parse_modbus_frames(_tcp(1, 0xFF, pdu), framing="tcp")
        decoded = result["frames"][0]
        assert decoded["transaction_id"] == 1
        assert decoded["unit_id"] == 0xFF
        assert decoded["role"] == "response"
        assert decoded["byte_count"] == 6
        assert decoded["registers"] == [0x022B, 0x0000, 0x0064]

    def test_tcp_multiple_adus(self) -> None:
        stream = _tcp(1, 1, bytes.fromhex("0100130025")) + _tcp(2, 1, bytes.fromhex("0500ACFF00"))
        result = parse_modbus_frames(stream, framing="tcp")
        assert len(result["frames"]) == 2
        assert result["frames"][0]["function"] == "Read Coils"
        assert result["frames"][0]["role"] == "request"
        assert result["frames"][1]["function"] == "Write Single Coil"
        assert result["frames"][1]["role"] == "request_or_response"
        assert result["frames"][1]["address"] == 0xAC
        assert result["frames"][1]["value"] == 0xFF00

    def test_write_multiple_registers_both_roles(self) -> None:
        request = bytes.fromhex("10000100020400 0A0102".replace(" ", ""))
        decoded = parse_modbus_frames(_tcp(5, 1, request), framing="tcp")["frames"][0]
        assert decoded["role"] == "request"
        assert decoded["start_address"] == 1
        assert decoded["quantity"] == 2
        assert decoded["registers"] == [0x000A, 0x0102]
        response = bytes.fromhex("1000010002")
        decoded = parse_modbus_frames(_tcp(5, 1, response), framing="tcp")["frames"][0]
        assert decoded["role"] == "response"
        assert decoded["quantity"] == 2

    def test_exception_response_names_the_code(self) -> None:
        decoded = parse_modbus_frames(_tcp(9, 1, bytes.fromhex("8302")), framing="tcp")["frames"][0]
        assert decoded["role"] == "exception"
        assert decoded["function_code"] == 3
        assert decoded["exception_code"] == 2
        assert decoded["exception"] == "ILLEGAL DATA ADDRESS"

    def test_nonzero_protocol_id_is_refused(self) -> None:
        frame = bytearray(_tcp(1, 1, bytes.fromhex("0100130025")))
        frame[3] = 7
        with pytest.raises(ValueError, match="protocol identifier"):
            parse_modbus_frames(bytes(frame), framing="tcp")

    def test_truncated_tcp_frame_is_refused(self) -> None:
        with pytest.raises(ValueError, match="truncated"):
            parse_modbus_frames(_tcp(1, 1, bytes.fromhex("0100130025"))[:-2], framing="tcp")

    def test_unknown_framing_is_refused(self) -> None:
        with pytest.raises(ValueError, match="framing"):
            parse_modbus_frames(b"\x01\x03", framing="ascii")


# ---------------------------------------------------------------------------
# DNP3
# ---------------------------------------------------------------------------


def _dnp3_frame(control: int, dest: int, source: int, user: bytes) -> bytes:
    header = bytes([0x05, 0x64, 5 + len(user), control])
    header += dest.to_bytes(2, "little") + source.to_bytes(2, "little")
    frame = header + crc_dnp_ref(header).to_bytes(2, "little")
    for start in range(0, len(user), 16):
        block = user[start : start + 16]
        frame += block + crc_dnp_ref(block).to_bytes(2, "little")
    return frame


class TestDnp3Frames:
    def test_user_data_frame_with_transport_header(self) -> None:
        # DIR=1, PRM=1, function 4 (unconfirmed user data); FIR|FIN, seq 1.
        app = bytes.fromhex("C001 3C02 06".replace(" ", ""))
        user = bytes([0xC1]) + app
        result = parse_dnp3_frames(_dnp3_frame(0xC4, 1, 10, user))
        frame = result["frames"][0]
        assert frame["dir"] is True
        assert frame["prm"] is True
        assert frame["function_code"] == 4
        assert frame["function"] == "UNCONFIRMED_USER_DATA"
        assert frame["destination"] == 1
        assert frame["source"] == 10
        assert frame["transport"] == {"fin": True, "fir": True, "sequence": 1}
        assert frame["application_bytes"] == app

    def test_multi_block_body_crcs(self) -> None:
        user = bytes([0xC5]) + bytes(range(30))  # 31 user octets -> 2 blocks
        result = parse_dnp3_frames(_dnp3_frame(0xC4, 100, 200, user))
        frame = result["frames"][0]
        assert frame["length"] == 5 + 31
        assert frame["application_bytes"] == bytes(range(30))
        assert frame["transport"]["sequence"] == 5

    def test_link_only_ack_frame(self) -> None:
        result = parse_dnp3_frames(_dnp3_frame(0x00, 3, 4, b""))
        frame = result["frames"][0]
        assert frame["prm"] is False
        assert frame["function"] == "ACK"
        assert "transport" not in frame
        assert "user_data" not in frame

    def test_two_frames_back_to_back(self) -> None:
        stream = _dnp3_frame(0xC4, 1, 10, bytes([0xC1, 0x01])) + _dnp3_frame(0x00, 10, 1, b"")
        result = parse_dnp3_frames(stream)
        assert len(result["frames"]) == 2
        assert result["frames"][1]["function"] == "ACK"

    def test_bad_start_bytes_are_refused(self) -> None:
        frame = bytearray(_dnp3_frame(0xC4, 1, 10, bytes([0xC1, 0x01])))
        frame[1] = 0x65
        with pytest.raises(ValueError, match=r"offset 0.*start bytes"):
            parse_dnp3_frames(bytes(frame))

    def test_bad_header_crc_is_refused(self) -> None:
        frame = bytearray(_dnp3_frame(0xC4, 1, 10, bytes([0xC1, 0x01])))
        frame[4] ^= 0x01  # corrupt destination; header CRC no longer matches
        with pytest.raises(ValueError, match="link header CRC mismatch"):
            parse_dnp3_frames(bytes(frame))

    def test_bad_body_crc_is_refused(self) -> None:
        frame = bytearray(_dnp3_frame(0xC4, 1, 10, bytes([0xC1, 0x01, 0x02])))
        frame[11] ^= 0x01  # corrupt a user octet; its block CRC no longer matches
        with pytest.raises(ValueError, match="body block CRC mismatch"):
            parse_dnp3_frames(bytes(frame))

    def test_truncated_frame_is_refused(self) -> None:
        with pytest.raises(ValueError, match="remain"):
            parse_dnp3_frames(_dnp3_frame(0xC4, 1, 10, bytes([0xC1, 0x01]))[:-3])
