# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Read-only parsers for domain sensor/telemetry file and wire formats.

Each reader parses one published format into plain dicts and numpy arrays,
refuses malformed input loudly (``ValueError`` with position information),
and never executes or writes anything.  Stdlib + numpy only.

Readers and the specifications they implement:

* :func:`read_rinex_obs` -- RINEX observation files, 2.11 and 3.x (IGS/RTCM
  RINEX 3.05 specification).
* :func:`read_grib2_sections` -- GRIB2 section structure (WMO FM 92 GRIB
  Edition 2).
* :func:`read_pmu_frames` -- IEEE C37.118.2-2011 synchrophasor frames.
* :func:`read_las_points` -- ASPRS LAS 1.2-1.4 point clouds (LAS 1.4 R15).
* :func:`parse_modbus_frames` -- Modbus Application Protocol V1.1b3 (RTU and
  TCP framing).
* :func:`parse_dnp3_frames` -- DNP3 / IEEE 1815-2012 data-link frames.
"""

from __future__ import annotations

from omni_mercury_engine.loaders.format_readers.dnp3 import parse_dnp3_frames
from omni_mercury_engine.loaders.format_readers.grib2 import read_grib2_sections
from omni_mercury_engine.loaders.format_readers.las import read_las_points
from omni_mercury_engine.loaders.format_readers.modbus import parse_modbus_frames
from omni_mercury_engine.loaders.format_readers.pmu import read_pmu_frames
from omni_mercury_engine.loaders.format_readers.rinex import read_rinex_obs

__all__ = [
    "parse_dnp3_frames",
    "parse_modbus_frames",
    "read_grib2_sections",
    "read_las_points",
    "read_pmu_frames",
    "read_rinex_obs",
]
