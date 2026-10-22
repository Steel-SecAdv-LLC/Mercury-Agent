# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Format readers checked against recorded third-party files, not our own bytes.

The synthetic suite (``test_format_readers.py``) builds every sample in the
test, so a misreading of a specification shared by reader and test would
pass there.  Here each reader parses files produced by other software and
is checked against evidence that does not come from the reader:

* RINEX -- receiver files from georinex's test data, checked against the
  file's own fixed columns.
* LAS -- PDAL's ``simple.las`` (TerraScan), checked against the bounds and
  point count its producer wrote into the public header.
* GRIB2 -- a NOAA GFS 1-degree subset, checked against the request that
  produced it (2 m TMP and MSL PRMSL over 40-50N, 250-260E).
* Modbus/TCP and DNP3 -- Zeek's recorded traces, checked against Zeek's own
  decoded baselines.
* IEEE C37.118 -- Wireshark wiki captures (network-gated, pinned by SHA-256:
  the captures carry no stated licence, so they are not vendored), checked
  against physics (three balanced phases 120 degrees apart, 50 frames/s).

Provenance and licences: ``tests/fixtures/format_readers/README.md``.
"""

from __future__ import annotations

import hashlib
import re
import struct
import urllib.request
from collections import Counter
from pathlib import Path

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

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "format_readers"

FlowKey = tuple[str, str, int, str, int]


def _pcap_flows(data: bytes) -> dict[FlowKey, bytes]:
    """Per-direction TCP payload streams (sequence-ordered) and UDP datagrams joined.

    A minimal classic-pcap (little-endian, Ethernet) walker, stdlib only;
    retransmitted TCP segments (same sequence number) are kept once.
    """
    if struct.unpack("<I", data[:4])[0] != 0xA1B2C3D4 or struct.unpack("<I", data[20:24])[0] != 1:
        raise ValueError("expected a little-endian classic pcap with Ethernet link type")
    segments: dict[FlowKey, dict[int, bytes]] = {}
    pos = 24
    while pos + 16 <= len(data):
        incl = struct.unpack("<I", data[pos + 8 : pos + 12])[0]
        packet = data[pos + 16 : pos + 16 + incl]
        pos += 16 + incl
        ethertype, off = struct.unpack("!H", packet[12:14])[0], 14
        if ethertype == 0x8100:
            ethertype, off = struct.unpack("!H", packet[16:18])[0], 18
        if ethertype != 0x0800:
            continue
        ihl = (packet[off] & 0x0F) * 4
        total = struct.unpack("!H", packet[off + 2 : off + 4])[0]
        proto = packet[off + 9]
        src = ".".join(str(b) for b in packet[off + 12 : off + 16])
        dst = ".".join(str(b) for b in packet[off + 16 : off + 20])
        l4 = packet[off + ihl : off + total]
        if proto == 6:
            sport, dport, seq = struct.unpack("!HHI", l4[:8])
            payload = l4[(l4[12] >> 4) * 4 :]
            if payload:
                segments.setdefault(("tcp", src, sport, dst, dport), {}).setdefault(seq, payload)
        elif proto == 17:
            sport, dport = struct.unpack("!HH", l4[:4])
            flow = segments.setdefault(("udp", src, sport, dst, dport), {})
            flow[len(flow)] = l4[8:]
    return {key: b"".join(seg[k] for k in sorted(seg)) for key, seg in segments.items()}


class TestRinexRecorded:
    def test_rinex2_epoch_matches_the_files_fixed_columns(self) -> None:
        result = read_rinex_obs(FIXTURES / "demo.10o")
        assert result["version"] == 2.11
        assert result["marker"] == "MRKR"
        assert result["obs_types"]["*"] == ["L1", "L2", "P1", "P2", "C1", "S1", "S2"]
        assert len(result["epochs"]) == 2
        first = result["epochs"][0]
        assert first["time"] == (2010, 3, 5, 0, 0, 0.0)
        # 14 satellites, the last two (G15, S24) on the continuation line.
        assert len(first["satellites"]) == 14
        assert {"G15", "S24"} <= set(first["satellites"])
        g13 = first["satellites"]["G13"]
        assert g13["L1"] == pytest.approx(121367582.205)
        assert g13["P2"] == pytest.approx(23095481.949)
        assert g13["S2"] == pytest.approx(40.0)
        # R19 records L1, C1 and S1 only: the blank fields are unmeasured.
        r19 = first["satellites"]["R19"]
        assert r19["L2"] is None and r19["P1"] is None and r19["S2"] is None
        assert r19["C1"] == pytest.approx(23095483.463)

    def test_rinex3_per_system_types(self) -> None:
        result = read_rinex_obs(FIXTURES / "minimal3.10o")
        assert result["version"] == 3.01
        assert result["obs_types"]["R"] == ["L1C", "C1C", "S1C"]
        epoch = result["epochs"][0]
        assert epoch["time"] == (2010, 3, 5, 0, 0, 30.0)
        assert epoch["satellites"]["G13"]["L1C"] == pytest.approx(130321269.801)
        assert set(epoch["satellites"]["R19"]) == {"L1C", "C1C", "S1C"}


class TestLasRecorded:
    def test_points_reproduce_the_producers_header_bounds(self) -> None:
        raw = (FIXTURES / "simple.las").read_bytes()
        # Public header block, LAS 1.2: legacy point count at 107, then the
        # max/min X, Y, Z doubles at 179..227 as written by the producer.
        header_count = struct.unpack("<I", raw[107:111])[0]
        max_x, min_x, max_y, min_y, max_z, min_z = struct.unpack("<6d", raw[179:227])

        result = read_las_points(FIXTURES / "simple.las")
        assert result["version"] == (1, 2)
        assert result["point_format"] == 3
        assert result["point_count"] == header_count == result["points_read"] == 1065
        half_step = 0.5 * max(result["scales"])
        for values, lo, hi in (
            (result["x"], min_x, max_x),
            (result["y"], min_y, max_y),
            (result["z"], min_z, max_z),
        ):
            assert float(values.min()) == pytest.approx(lo, abs=half_step)
            assert float(values.max()) == pytest.approx(hi, abs=half_step)
        # Global encoding bit 0 clear: GPS time is GPS week seconds (LAS 1.2).
        assert struct.unpack("<H", raw[6:8])[0] & 1 == 0
        assert np.all((result["gps_time"] >= 0.0) & (result["gps_time"] < 604800.0))

    def test_max_points_caps_and_refuses_a_negative_cap(self) -> None:
        assert read_las_points(FIXTURES / "simple.las", max_points=10)["x"].shape == (10,)
        with pytest.raises(ValueError, match="max_points"):
            read_las_points(FIXTURES / "simple.las", max_points=-1)


class TestGrib2Recorded:
    def test_messages_match_the_request_that_produced_them(self) -> None:
        path = FIXTURES / "gfs_20261021_t00z_f000_subset.grib2"
        result = read_grib2_sections(path)
        messages = result["messages"]
        assert len(messages) == 2
        assert sum(m["total_length"] for m in messages) == path.stat().st_size
        products = set()
        for message in messages:
            assert message["edition"] == 2 and message["discipline"] == 0
            ident = message["identification"]
            assert ident["centre"] == 7  # NCEP
            assert (ident["year"], ident["month"], ident["day"], ident["hour"]) == (2026, 10, 21, 0)
            grid = message["grid_definitions"][0]
            assert grid["template_number"] == 0  # regular latitude/longitude
            g = grid["raw"]
            ni, nj = int.from_bytes(g[30:34], "big"), int.from_bytes(g[34:38], "big")
            la1, lo1 = int.from_bytes(g[46:50], "big"), int.from_bytes(g[50:54], "big")
            la2, lo2 = int.from_bytes(g[55:59], "big"), int.from_bytes(g[59:63], "big")
            assert (ni, nj) == (11, 11)
            assert (la1, lo1, la2, lo2) == (40_000_000, 250_000_000, 50_000_000, 260_000_000)
            p = message["product_definitions"][0]["raw"]
            products.add((p[9], p[10], p[22], int.from_bytes(p[24:28], "big")))
        # (category, number, first fixed surface, its value): PRMSL at mean
        # sea level, TMP at 2 m above ground (WMO Code Tables 4.2 / 4.5).
        assert products == {(3, 1, 101, 0), (0, 0, 103, 2)}


def _zeek_modbus_messages(baseline: Path) -> Counter[tuple[int, bool, int, int, int, int]]:
    """Zeek's decoded Modbus messages: (orig_port, is_orig, tid, uid, fc, len) counts."""
    pattern = re.compile(
        r"^(\d+) modbus_message, \[orig_h=\S+ orig_p=(\d+)/tcp.*\], "
        r"\[tid=(\d+), pid=0, uid=(\d+), function_code=(\d+), len=(\d+)\], ([TF])$"
    )
    counts: Counter[tuple[int, bool, int, int, int, int]] = Counter()
    for line in baseline.read_text().splitlines():
        match = pattern.match(line)
        if match:
            n, port, tid, uid, fc, length, orig = match.groups()
            counts[(int(port), orig == "T", int(tid), int(uid), int(fc), int(length))] += int(n)
    return counts


class TestIndustrialProtocolsRecorded:
    def test_modbus_tcp_frames_equal_zeeks_decode(self) -> None:
        expected = _zeek_modbus_messages(FIXTURES / "zeek_modbus_coil_parsing_small.baseline")
        assert sum(expected.values()) == 32
        ours: Counter[tuple[int, bool, int, int, int, int]] = Counter()
        flows = _pcap_flows((FIXTURES / "modbusSmall.pcap").read_bytes())
        for (proto, _src, sport, _dst, dport), stream in flows.items():
            if proto != "tcp" or 502 not in (sport, dport):
                continue  # the trace also carries DNS
            is_orig = dport == 502
            for frame in parse_modbus_frames(stream, framing="tcp")["frames"]:
                key = (
                    sport if is_orig else dport,
                    is_orig,
                    frame["transaction_id"],
                    frame["unit_id"],
                    frame["function_code"],
                    frame["length"],
                )
                ours[key] += 1
        assert ours == expected

    def test_dnp3_link_and_application_headers_equal_zeeks_decode(self) -> None:
        flows = _pcap_flows((FIXTURES / "dnp3_read.pcap").read_bytes())
        decoded = {}
        for (_proto, _src, _sport, _dst, dport), stream in flows.items():
            frames = parse_dnp3_frames(stream)["frames"]
            decoded[dport == 20000] = [
                (f["length"], f["control"], f["destination"], f["source"])
                + tuple(f["application_bytes"][:2])
                for f in frames
            ]
        # Zeek baseline scripts.base.protocols.dnp3.dnp3_read: header blocks
        # (len, ctrl, dest, src) and application (control, function code).
        assert decoded[True] == [(20, 196, 2, 3, 200, 1)]
        assert decoded[False] == [(116, 68, 3, 2, 200, 129)]


#: Wireshark wiki IEEE C37.118 captures, pinned by content.
_C37_CAPTURES = {
    "C37.118_1PMU_TCP.pcap": "a42fa6644dda7c4d68e4ea4079ecf431d44e1356c489828013fe498853ee979a",
    "C37.118_2PMUsInSync_TCP.pcap": (
        "fdc68c55e2325173dbcbb991f3709ed90a129dac20527623af4cac948a372f1e"
    ),
}
_C37_BASE = "https://wiki.wireshark.org/uploads/__moin_import__/attachments/IEEE-C37.118/"


@pytest.mark.network
@pytest.mark.parametrize("name", sorted(_C37_CAPTURES))
def test_pmu_frames_from_recorded_captures_obey_power_system_physics(name: str) -> None:
    with urllib.request.urlopen(_C37_BASE + name, timeout=60) as response:  # noqa: S310
        capture = response.read()
    assert hashlib.sha256(capture).hexdigest() == _C37_CAPTURES[name], "capture changed upstream"
    streams = [
        stream
        for (proto, _src, sport, _dst, _dport), stream in _pcap_flows(capture).items()
        if proto == "tcp" and sport == 4712
    ]
    assert streams
    for stream in streams:
        result = read_pmu_frames(stream)  # every frame's CRC-CCITT is verified
        (config,) = result["configs"].values()
        assert config["pmus"][0]["nominal_freq_hz"] == 50.0
        data = [f for f in result["frames"] if f["frame_type"] == "data"]
        assert len(data) > 200
        times = np.array([f["soc"] + f["fracsec"] / config["time_base"] for f in data])
        assert np.allclose(np.diff(times), 1.0 / 50.0, atol=1e-5)  # 50 frames/s
        # Captured 2008-08-01 (UTC), per the SOC timestamps.
        assert 1217548800 <= int(times[0]) < 1217635200
        for frame in data:
            for pmu in frame["pmus"]:
                phasors = np.asarray(pmu["phasors"])
                three_phase = phasors[-3:]  # VA, VB, VC
                magnitude = np.abs(three_phase)
                assert np.ptp(magnitude) / magnitude.mean() < 0.01  # balanced
                spacing = np.angle(three_phase[1:] / three_phase[:-1], deg=True)
                assert np.allclose(np.abs(spacing), 120.0, atol=1.0)
