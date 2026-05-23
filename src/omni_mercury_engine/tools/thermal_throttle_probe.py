"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

------------------------------------------------------------------------

Operator tool: thermal-throttle probe.

Samples thermal zones during a benchmark window and flags any
throttled period so the latency numbers from
:mod:`run_hardware_benchmark` stay comparable across runs.

Strategy:

1. Try :mod:`psutil.sensors_temperatures` first — the cross-platform
   path.
2. Fall back to the handwritten ``/sys/class/hwmon`` walker so the
   tool runs on Mercury's container image without ``psutil``
   installed.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.thermal_throttle_probe/v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.thermal_throttle_probe",
        description="Sample thermal zones during a window and flag throttled periods.",
    )
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument(
        "--throttle-threshold-c",
        type=float,
        default=85.0,
        help="Per-sensor temperature ≥ this is treated as throttled.",
    )
    return parser


def _psutil_temps() -> dict[str, Any] | None:
    try:
        # ``psutil`` is covered by ``pyproject.toml`` mypy overrides;
        # no ``type: ignore`` required.
        import psutil
    except ImportError:
        return None
    try:
        out = psutil.sensors_temperatures(fahrenheit=False)
    except AttributeError:
        return None
    if not out:
        return None
    return {
        name: [
            {
                "label": getattr(entry, "label", "") or "",
                "current_c": float(entry.current),
                "high_c": float(entry.high) if entry.high is not None else None,
                "critical_c": float(entry.critical) if entry.critical is not None else None,
            }
            for entry in entries
        ]
        for name, entries in out.items()
    }


def _hwmon_temps() -> dict[str, Any]:
    """Handwritten fallback over ``/sys/class/hwmon`` for Linux containers without psutil."""
    root = Path("/sys/class/hwmon")
    if not root.is_dir():
        return {}
    out: dict[str, Any] = {}
    for hwmon in sorted(root.iterdir()):
        name_path = hwmon / "name"
        name = name_path.read_text().strip() if name_path.exists() else hwmon.name
        entries: list[dict[str, Any]] = []
        for input_path in sorted(hwmon.glob("temp*_input")):
            try:
                raw = int(input_path.read_text().strip())
            except (OSError, ValueError):
                continue
            label_path = input_path.with_name(input_path.name.replace("_input", "_label"))
            label = label_path.read_text().strip() if label_path.exists() else input_path.name
            entries.append({"label": label, "current_c": raw / 1000.0})
        if entries:
            out[name] = entries
    return out


def _collect(args: argparse.Namespace) -> Certificate:
    samples: list[dict[str, Any]] = []
    deadline = time.monotonic() + float(args.duration)
    throttled = False
    sources_used: set[str] = set()
    while time.monotonic() < deadline:
        temps = _psutil_temps()
        if temps is None:
            temps = _hwmon_temps()
            if temps:
                sources_used.add("/sys/class/hwmon")
        else:
            sources_used.add("psutil")
        sample = {"t": time.monotonic(), "temps": temps}
        # Throttle check on this sample.
        for entries in temps.values():
            for e in entries:
                if e.get("current_c", 0.0) >= float(args.throttle_threshold_c):
                    throttled = True
        samples.append(sample)
        time.sleep(float(args.interval))

    body: dict[str, Any] = {
        "duration_s": float(args.duration),
        "samples": samples,
        "throttle_threshold_c": float(args.throttle_threshold_c),
        "throttled_observed": throttled,
        "sources": sorted(sources_used),
    }
    return Certificate(
        tool="thermal_throttle_probe",
        schema=_SCHEMA,
        status="warn" if throttled else "ok",
        body=body,
        warnings=(
            [f"thermal throttle observed (>= {args.throttle_threshold_c}C)"] if throttled else []
        ),
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
