# Copyright (C) 2025 Steel Security Advisors LLC
"""(at your option) any later version."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import platform
from pathlib import Path
from typing import Any

from omni_mercury_engine.tools._base import (
    Certificate,
    mercury_env,
    require_real_component,
    run_tool,
)

_SCHEMA = "mercury.tools.hwrng_audit/v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.hwrng_audit",
        description="Probe /dev/hwrng and kernel entropy sources.",
    )
    parser.add_argument("--sample-bytes", type=int, default=4096)
    return parser


def _shannon(b: bytes) -> float:
    if not b:
        return 0.0
    counts: dict[int, int] = {}
    for v in b:
        counts[v] = counts.get(v, 0) + 1
    return -sum((c / len(b)) * math.log2(c / len(b)) for c in counts.values())


def _probe_hwrng(n: int) -> dict[str, Any]:
    p = Path("/dev/hwrng")
    if not p.exists():
        return {"available": False, "reason": "/dev/hwrng not present"}
    try:
        with p.open("rb") as fh:
            buf = fh.read(n)
    except PermissionError:
        return {"available": False, "reason": "/dev/hwrng permission denied"}
    return {
        "available": True,
        "bytes_read": len(buf),
        "entropy_bits_per_byte": _shannon(buf),
        "sha256": hashlib.sha256(buf).hexdigest(),
    }


def _probe_urandom(n: int) -> dict[str, Any]:
    buf = os.urandom(n)
    return {
        "available": True,
        "bytes_read": len(buf),
        "entropy_bits_per_byte": _shannon(buf),
        "sha256": hashlib.sha256(buf).hexdigest(),
    }


def _probe_rng_source() -> dict[str, Any]:
    p = Path("/sys/class/misc/hw_random/rng_current")
    if not p.exists():
        return {"available": False, "reason": "rng_current not present"}
    try:
        current = p.read_text().strip()
    except OSError as exc:
        return {"available": False, "reason": str(exc)}
    return {"available": True, "rng_current": current}


def _collect(args: argparse.Namespace) -> Certificate:
    n = int(args.sample_bytes)
    body: dict[str, Any] = {
        "platform": platform.platform(),
        "hwrng": _probe_hwrng(n),
        "rng_current": _probe_rng_source(),
        "urandom": _probe_urandom(n),
    }
    hwrng_available = bool(body["hwrng"].get("available"))
    require_real_component("/dev/hwrng", hwrng_available)
    status = "ok"
    warnings: list[str] = []
    if not hwrng_available:
        if mercury_env() == "production":
            status = "fail"
        else:
            status = "warn"
        warnings.append(f"/dev/hwrng unavailable: {body['hwrng'].get('reason', 'unknown')}")
    # An entropy below ~7.5 bits/byte over 4 KiB is suspicious — flag.
    h_ent = body["hwrng"].get("entropy_bits_per_byte")
    if isinstance(h_ent, (int, float)) and h_ent < 7.5:
        warnings.append(f"hwrng entropy {h_ent:.2f} bits/byte < 7.5")
        if status == "ok":
            status = "warn"
    return Certificate(
        tool="hwrng_audit",
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
