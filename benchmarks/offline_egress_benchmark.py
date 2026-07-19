#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Measure the offline / air-gapped egress path (latency + throughput).

Air-gap is functionally proven (``tests/security/test_offline_egress_gate.py``:
zero external sockets under ``MERCURY_OFFLINE``), but the path's *performance* was
never measured. This harness quantifies it, entirely on-box — it opens no
external socket, which is the whole point:

1. **Refusal cost** — with ``MERCURY_OFFLINE=1``, how long ``validate_url``
   takes to refuse an external URL (pure CPU, pre-DNS, no socket).
2. **Loopback-permit cost** — the gate overhead for an allowed loopback URL.
3. **Loopback round-trip** — real latency + throughput of
   ``SafeHTTPClient.get_bytes(loopback_only=True)`` against a local
   ``http.server`` bound to 127.0.0.1 (the on-box Ollama/sidecar transport).

Stdlib-only (``statistics``, not numpy), mirroring
``benchmarks/crypto_backend_benchmark.py``. If the loopback server cannot bind,
the round-trip section reports ``available: false`` rather than fabricating
numbers.

Run::

    python -m benchmarks.offline_egress_benchmark
    python -m benchmarks.offline_egress_benchmark --iters 500 --kb 128
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

_REPO = Path(__file__).resolve().parent.parent


def _git_commit() -> str:
    """Return the short HEAD commit, or "unknown"."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def _time_call(fn: Callable[[], Any], iters: int, warmup: int = 5) -> dict[str, float]:
    """Time ``fn`` over ``iters`` iterations after ``warmup`` untimed calls."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return {
        "median_ms": statistics.median(samples) * 1e3,
        "mean_ms": statistics.fmean(samples) * 1e3,
        "min_ms": min(samples) * 1e3,
        "p95_ms": (
            statistics.quantiles(samples, n=20)[18] * 1e3
            if len(samples) >= 20
            else max(samples) * 1e3
        ),
    }


def _make_handler(payload: bytes) -> type[BaseHTTPRequestHandler]:
    """Build a loopback handler that returns a fixed payload and logs nothing."""

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args: Any) -> None:  # silence per-request logging
            return

    return _Handler


def _bench_gate(iters: int) -> dict[str, Any]:
    """Measure validate_url refusal + loopback-permit overhead under offline mode."""
    from omni_mercury_engine.datasets.exceptions import OfflineModeError
    from omni_mercury_engine.security.safe_http import SafeHTTPClient

    def refuse_external() -> None:
        try:
            SafeHTTPClient.validate_url("https://example.com/data")
        except OfflineModeError:
            return
        raise AssertionError("offline gate did not refuse an external URL")

    def permit_loopback() -> None:
        SafeHTTPClient.validate_url(
            "http://127.0.0.1:9/x",
            allow_http=True,
            user_configured=True,
            loopback_only=True,
        )

    return {
        "offline_refusal_external": _time_call(refuse_external, iters),
        "offline_permit_loopback_validate": _time_call(permit_loopback, iters),
    }


def _bench_loopback_roundtrip(iters: int, payload: bytes) -> dict[str, Any]:
    """Measure real loopback GET latency + throughput (or 'not measured')."""
    from omni_mercury_engine.security.safe_http import SafeHTTPClient

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(payload))
    except OSError as exc:  # pragma: no cover - environment dependent
        return {"available": False, "reason": f"could not bind loopback server: {exc}"}

    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}/payload"

    try:

        def fetch() -> None:
            body = SafeHTTPClient.get_bytes(
                url,
                allow_http=True,
                user_configured=True,
                loopback_only=True,
                timeout=10.0,
            )
            if len(body) != len(payload):
                raise AssertionError("loopback payload length mismatch")

        timing = _time_call(fetch, iters)
    finally:
        server.shutdown()
        server.server_close()

    median_s = timing["median_ms"] / 1e3
    reqs_per_sec = (1.0 / median_s) if median_s > 0 else float("inf")
    mb = len(payload) / (1024 * 1024)
    mb_per_sec = (mb / median_s) if median_s > 0 else float("inf")
    return {
        "available": True,
        "payload_bytes": len(payload),
        "latency": timing,
        "requests_per_sec": round(reqs_per_sec, 1),
        "throughput_mb_per_sec": round(mb_per_sec, 2),
    }


def run(iters: int, payload_kb: int) -> dict[str, Any]:
    """Run all offline-egress measurements under MERCURY_OFFLINE."""
    prior = os.environ.get("MERCURY_OFFLINE")
    os.environ["MERCURY_OFFLINE"] = "1"
    payload = os.urandom(max(1, payload_kb) * 1024)
    try:
        gate = _bench_gate(iters)
        roundtrip = _bench_loopback_roundtrip(iters, payload)
    finally:
        if prior is None:
            os.environ.pop("MERCURY_OFFLINE", None)
        else:
            os.environ["MERCURY_OFFLINE"] = prior

    return {
        "schema": "offline_egress_benchmark/v1",
        "provenance": {
            "commit": _git_commit(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
            "note": (
                "Offline egress path measured entirely on-box (MERCURY_OFFLINE=1). "
                "No external socket is opened — refusal timings are pure CPU, the "
                "round-trip is loopback-only (127.0.0.1)."
            ),
        },
        "iters": iters,
        "gate_overhead": gate,
        "loopback_roundtrip": roundtrip,
    }


def _summarise(result: dict[str, Any]) -> str:
    """Return a compact human-readable summary."""
    g = result["gate_overhead"]
    rt = result["loopback_roundtrip"]
    lines = [
        f"offline egress benchmark (commit {result['provenance']['commit']}, "
        f"{result['iters']} iters)",
        f"  gate refuse external  : {g['offline_refusal_external']['median_ms']:.4f} ms median "
        "(pure CPU, pre-DNS, no socket)",
        f"  gate permit loopback  : {g['offline_permit_loopback_validate']['median_ms']:.4f} ms median",
    ]
    if rt.get("available"):
        lines.append(
            f"  loopback round-trip   : {rt['latency']['median_ms']:.3f} ms median, "
            f"{rt['requests_per_sec']:.0f} req/s, {rt['throughput_mb_per_sec']:.1f} MB/s "
            f"({rt['payload_bytes'] // 1024} KiB payload)"
        )
    else:
        lines.append(f"  loopback round-trip   : not measured ({rt.get('reason')})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=300, help="Timed iterations.")
    parser.add_argument("--kb", type=int, default=64, help="Loopback payload size (KiB).")
    parser.add_argument("--out", default="artifacts/offline_egress_benchmark.json")
    args = parser.parse_args(argv)

    result = run(args.iters, args.kb)
    print(_summarise(result))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
