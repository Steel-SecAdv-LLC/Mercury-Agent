# Copyright (C) 2025 Steel Security Advisors LLC
"""(at your option) any later version."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Any

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.time_source_probe/v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.time_source_probe",
        description="Probe NTP/chrony/PTP time source and report kernel offset.",
    )
    parser.add_argument(
        "--max-offset-ms",
        type=float,
        default=100.0,
        help="Maximum permitted absolute kernel offset (ms).",
    )
    return parser


def _run(cmd: list[str]) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
        return proc.returncode, proc.stdout, proc.stderr
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 127, "", ""


def _chronyc() -> dict[str, Any] | None:
    if not shutil.which("chronyc"):
        return None
    rc, out, _ = _run(["chronyc", "-c", "tracking"])
    if rc != 0 or not out.strip():
        return None
    fields = out.strip().split(",")
    # chronyc -c tracking columns:
    # 0: ref, 1: stratum, 2: ref_time, 3: system_time, 4: last_offset, ...
    try:
        return {
            "tool": "chronyc",
            "ref": fields[0],
            "stratum": int(fields[1]),
            "last_offset_ms": float(fields[4]) * 1000.0,
        }
    except (ValueError, IndexError):
        return {"tool": "chronyc", "raw": out.strip()}


def _ntpq() -> dict[str, Any] | None:
    if not shutil.which("ntpq"):
        return None
    rc, out, _ = _run(["ntpq", "-c", "rv 0 offset"])
    if rc != 0 or "offset" not in out:
        return None
    for tok in out.replace(",", " ").split():
        if tok.startswith("offset="):
            try:
                return {"tool": "ntpq", "last_offset_ms": float(tok.split("=", 1)[1])}
            except ValueError:
                return {"tool": "ntpq", "raw": out.strip()}
    return None


def _timedatectl() -> dict[str, Any] | None:
    if not shutil.which("timedatectl"):
        return None
    rc, out, _ = _run(["timedatectl", "show"])
    if rc != 0:
        return None
    parsed: dict[str, Any] = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            parsed[k.strip()] = v.strip()
    return {"tool": "timedatectl", "values": parsed}


def _ptp_devices() -> list[dict[str, Any]]:
    root = Path("/sys/class/ptp")
    if not root.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for ptp in sorted(root.iterdir()):
        info: dict[str, Any] = {"name": ptp.name}
        for attr in ("clock_name", "max_adjustment", "n_alarms"):
            f = ptp / attr
            if f.exists():
                try:
                    info[attr] = f.read_text().strip()
                except OSError:
                    pass
        out.append(info)
    return out


def _collect(args: argparse.Namespace) -> Certificate:
    sources: list[dict[str, Any]] = []
    for fn in (_chronyc, _ntpq, _timedatectl):
        r = fn()
        if r is not None:
            sources.append(r)
    ptp = _ptp_devices()

    offsets = [s["last_offset_ms"] for s in sources if "last_offset_ms" in s]
    max_offset = max((abs(o) for o in offsets), default=None)
    body: dict[str, Any] = {
        "sources": sources,
        "ptp_devices": ptp,
        "max_offset_ms": max_offset,
        "offset_threshold_ms": float(args.max_offset_ms),
    }
    warnings: list[str] = []
    status = "ok"
    if not sources:
        warnings.append("no NTP / chrony / timedatectl source available")
        status = "warn"
    elif max_offset is not None and max_offset > float(args.max_offset_ms):
        warnings.append(f"clock offset {max_offset:.1f}ms > threshold {args.max_offset_ms}ms")
        status = "fail"
    return Certificate(
        tool="time_source_probe",
        schema=_SCHEMA,
        status=status,
        body=body,
        warnings=warnings,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
