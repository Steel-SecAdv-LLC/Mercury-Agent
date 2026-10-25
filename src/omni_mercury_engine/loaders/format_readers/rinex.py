# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Read-only RINEX observation file reader (versions 2.11 and 3.x).

Implements the common subset of the RINEX observation format per the IGS/RTCM
specification "RINEX: The Receiver Independent Exchange Format, Version 3.05"
(IGS/RTCM, 2020) and the RINEX 2.11 definition (Gurtner & Estey, "RINEX: The
Receiver Independent Exchange Format Version 2.11", IGS, 2007): header records
are label-dispatched on columns 61-80; observation types come from
``SYS / # / OBS TYPES`` (3.x, per constellation) or ``# / TYPES OF OBSERV``
(2.11, one list for all satellites); epoch records carry per-satellite
observable values (14-character fields; LLI/SSI flags are not decoded).

Honest limits: special-event epoch blocks (epoch flag > 1) are skipped, not
interpreted; unknown header record labels are preserved verbatim in the
result's ``raw_header`` list, never dropped silently.  Malformed input raises
``ValueError`` naming the offending line number.  The reader never executes or
writes anything.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

__all__ = ["read_rinex_obs"]

#: Header labels parsed into named fields; everything else goes to raw_header.
_PARSED_LABELS = frozenset(
    {
        "RINEX VERSION / TYPE",
        "MARKER NAME",
        "SYS / # / OBS TYPES",
        "# / TYPES OF OBSERV",
        "END OF HEADER",
    }
)


def _fail(lineno: int, message: str) -> None:
    """Raise ``ValueError`` carrying the 1-based line number."""
    raise ValueError(f"RINEX line {lineno}: {message}")


def _parse_float(field: str, lineno: int, what: str) -> float:
    """Parse ``field`` as a float, failing loudly with position info."""
    try:
        return float(field)
    except ValueError:
        _fail(lineno, f"cannot parse {what} from {field.strip()!r}")
        raise  # unreachable; keeps type-checkers satisfied


def _parse_int(field: str, lineno: int, what: str) -> int:
    """Parse ``field`` as an int, failing loudly with position info."""
    try:
        return int(field.strip())
    except ValueError:
        _fail(lineno, f"cannot parse {what} from {field.strip()!r}")
        raise


def _parse_header(
    lines: list[str],
) -> tuple[dict[str, Any], int]:
    """Parse the header block; return ``(header, index_of_first_body_line)``."""
    header: dict[str, Any] = {
        "version": None,
        "file_type": None,
        "marker": None,
        "obs_types": {},
        "raw_header": [],
    }
    pending_sys: str | None = None
    declared_counts: dict[str, int] = {}
    index = 0
    ended = False
    for index, line in enumerate(lines):
        lineno = index + 1
        label = line[60:80].strip()
        if index == 0:
            if label != "RINEX VERSION / TYPE":
                _fail(lineno, f"first record must be 'RINEX VERSION / TYPE', got {label!r}")
            header["version"] = _parse_float(line[:9], lineno, "RINEX version")
            file_type = line[20:21]
            if file_type != "O":
                _fail(lineno, f"not an observation file (type {file_type!r}, expected 'O')")
            header["file_type"] = file_type
            continue
        if label == "END OF HEADER":
            ended = True
            break
        if label == "MARKER NAME":
            header["marker"] = line[:60].strip()
        elif label == "SYS / # / OBS TYPES":
            system = line[0:1]
            if system.strip():
                pending_sys = system
                declared_counts[system] = _parse_int(line[3:6], lineno, "observation-type count")
                header["obs_types"][pending_sys] = []
            elif pending_sys is None:
                _fail(lineno, "continuation of 'SYS / # / OBS TYPES' without a system line")
            types = line[6:60].split()
            assert pending_sys is not None
            header["obs_types"][pending_sys].extend(types)
            if len(header["obs_types"][pending_sys]) > declared_counts[pending_sys]:
                _fail(
                    lineno,
                    f"system {pending_sys!r} declares {declared_counts[pending_sys]} "
                    f"observation types but lists {len(header['obs_types'][pending_sys])}",
                )
        elif label == "# / TYPES OF OBSERV":
            if line[:6].strip():
                pending_sys = "*"
                declared_counts["*"] = _parse_int(line[:6], lineno, "observation-type count")
                header["obs_types"]["*"] = []
            elif "*" not in header["obs_types"]:
                _fail(lineno, "continuation of '# / TYPES OF OBSERV' without a count line")
            header["obs_types"]["*"].extend(line[6:60].split())
            if len(header["obs_types"]["*"]) > declared_counts["*"]:
                _fail(
                    lineno,
                    f"header declares {declared_counts['*']} observation types but lists "
                    f"{len(header['obs_types']['*'])}",
                )
        else:
            header["raw_header"].append({"label": label, "content": line[:60].rstrip()})
    if not ended:
        _fail(len(lines), "header never terminated by 'END OF HEADER'")
    for system, types in header["obs_types"].items():
        declared = declared_counts[system]
        if len(types) != declared:
            _fail(index + 1, f"system {system!r}: {declared} types declared, {len(types)} listed")
    if not header["obs_types"]:
        _fail(index + 1, "header lists no observation types")
    return header, index + 1


def _obs_value(line: str, position: int) -> float | None:
    """One F14.3 observable from ``line`` at 16-char slot ``position`` (None if blank)."""
    field = line[3 + 16 * position : 3 + 16 * position + 14]
    if not field.strip():
        return None
    try:
        return float(field)
    except ValueError:
        return math.nan


def _epochs_v3(
    lines: list[str], start: int, obs_types: dict[str, list[str]]
) -> list[dict[str, Any]]:
    """Parse RINEX 3.x epoch records (``>``-prefixed) and satellite lines."""
    epochs: list[dict[str, Any]] = []
    i = start
    while i < len(lines):
        line = lines[i]
        lineno = i + 1
        if not line.strip():
            i += 1
            continue
        if not line.startswith(">"):
            _fail(lineno, f"expected an epoch record starting with '>', got {line[:20]!r}")
        parts = line[1:].split()
        if len(parts) < 8:
            _fail(lineno, "epoch record carries fewer than 8 fields")
        time = (
            _parse_int(parts[0], lineno, "year"),
            _parse_int(parts[1], lineno, "month"),
            _parse_int(parts[2], lineno, "day"),
            _parse_int(parts[3], lineno, "hour"),
            _parse_int(parts[4], lineno, "minute"),
            _parse_float(parts[5], lineno, "second"),
        )
        flag = _parse_int(parts[6], lineno, "epoch flag")
        count = _parse_int(parts[7], lineno, "satellite/record count")
        i += 1
        if flag > 1:
            # Special event: `count` records follow; preserved nowhere, skipped
            # honestly (documented out of the common subset).
            i += count
            continue
        satellites: dict[str, dict[str, float | None]] = {}
        for _ in range(count):
            if i >= len(lines):
                _fail(len(lines), f"epoch at line {lineno} declares {count} satellites; file ends")
            sat_line = lines[i]
            sat = sat_line[:3].replace(" ", "0")
            system = sat[0]
            types = obs_types.get(system)
            if types is None:
                _fail(i + 1, f"satellite {sat!r} uses system {system!r} with no declared types")
            assert types is not None
            satellites[sat] = {t: _obs_value(sat_line, k) for k, t in enumerate(types)}
            i += 1
        epochs.append({"time": time, "flag": flag, "satellites": satellites})
    return epochs


def _v2_year(two_digit: int) -> int:
    """Expand a 2-digit RINEX 2 year (80-99 -> 1900s, 00-79 -> 2000s)."""
    return 1900 + two_digit if two_digit >= 80 else 2000 + two_digit


def _epochs_v2(lines: list[str], start: int, types: list[str]) -> list[dict[str, Any]]:
    """Parse RINEX 2.11 epoch records, satellite lists and observation lines."""
    epochs: list[dict[str, Any]] = []
    lines_per_sat = max(1, -(-len(types) // 5))  # ceil(ntypes / 5)
    i = start
    while i < len(lines):
        line = lines[i]
        lineno = i + 1
        if not line.strip():
            i += 1
            continue
        try:
            time = (
                _v2_year(int(line[1:3])),
                int(line[4:6]),
                int(line[7:9]),
                int(line[10:12]),
                int(line[13:15]),
                float(line[15:26]),
            )
            flag = int(line[26:29])
            count = int(line[29:32])
        except ValueError:
            _fail(lineno, f"cannot parse epoch record {line[:32]!r}")
            raise
        i += 1
        if flag > 1:
            # Special event: `count` header-style records follow; skipped.
            i += count
            continue
        sats: list[str] = []
        sat_field = line[32:68]
        while True:
            for k in range(12):
                token = sat_field[3 * k : 3 * k + 3]
                if token.strip():
                    system = token[0] if token[0] != " " else "G"
                    sats.append(system + token[1:].replace(" ", "0"))
            if len(sats) >= count:
                break
            if i >= len(lines):
                _fail(len(lines), f"epoch at line {lineno} declares {count} satellites; file ends")
            sat_field = lines[i][32:68]
            i += 1
        if len(sats) != count:
            _fail(lineno, f"epoch declares {count} satellites but lists {len(sats)}")
        satellites: dict[str, dict[str, float | None]] = {}
        for sat in sats:
            values: list[float | None] = []
            for block in range(lines_per_sat):
                if i >= len(lines):
                    _fail(len(lines), f"observations for {sat!r} truncated at end of file")
                obs_line = lines[i]
                i += 1
                remaining = len(types) - 5 * block
                for k in range(min(5, remaining)):
                    field = obs_line[16 * k : 16 * k + 14]
                    values.append(float(field) if field.strip() else None)
            satellites[sat] = dict(zip(types, values))
        epochs.append({"time": time, "flag": flag, "satellites": satellites})
    return epochs


def read_rinex_obs(path: str | Path) -> dict[str, Any]:
    """Read a RINEX observation file (version 2.11 or 3.x), read-only.

    Parses header records (version, file type, marker name, observation types
    per constellation for 3.x / the single ``# / TYPES OF OBSERV`` list for
    2.11) and epoch records with per-satellite observable values, per the
    IGS/RTCM RINEX 3.05 specification and the RINEX 2.11 definition (see
    module docstring).  Unknown header record labels are preserved in the
    returned ``raw_header`` list, never dropped silently.

    Args:
        path: Filesystem path of the RINEX observation file.

    Returns:
        Dict with keys ``version`` (float), ``file_type`` (``"O"``),
        ``marker`` (str or None), ``obs_types`` (dict mapping constellation
        letter to its observation-type list; the key ``"*"`` holds the 2.11
        all-systems list), ``raw_header`` (list of ``{"label", "content"}``
        dicts for unparsed header records), and ``epochs`` (list of
        ``{"time": (y, m, d, h, min, sec), "flag": int, "satellites":
        {sat_id: {obs_type: value-or-None}}}``; a blank observable field is
        ``None``).

    Raises:
        ValueError: On any malformed record, naming the offending line number
            (wrong first record, non-observation file type, inconsistent
            type counts, truncated epoch blocks, unparsable fields).
    """
    text = Path(path).read_text(encoding="latin-1")
    lines = text.splitlines()
    if not lines:
        raise ValueError("RINEX line 1: file is empty")
    header, body_start = _parse_header(lines)
    version = float(header["version"])
    if version >= 3.0:
        if "*" in header["obs_types"]:
            _fail(body_start, "version 3 file uses the 2.11 '# / TYPES OF OBSERV' record")
        epochs = _epochs_v3(lines, body_start, header["obs_types"])
    else:
        if "*" not in header["obs_types"]:
            _fail(body_start, "version 2 file declares no '# / TYPES OF OBSERV' record")
        epochs = _epochs_v2(lines, body_start, header["obs_types"]["*"])
    header["epochs"] = epochs
    return header
