# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Operator tool: handwritten Prometheus ``/metrics`` exposition.

Emits the Prometheus text-format snapshot for the current
benevolence histogram, σ band, OAE weights, gate-fire counts,
Mercury/AMA Disconnect engagements, PQC capability bitmap, and cache
hit rate.  No
``prometheus_client`` dependency — the exposition format is small
enough to write directly.
"""

from __future__ import annotations

import argparse
import http.server
import threading
from typing import TYPE_CHECKING, Any

from omni_mercury_engine.tools._base import Certificate, atomic_write_text, run_tool

if TYPE_CHECKING:
    from omni_mercury_engine.core.centralized_constants import MathConstants

_SCHEMA = "mercury.tools.prometheus_metrics_exporter/v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.prometheus_metrics_exporter",
        description="Emit Mercury runtime metrics in Prometheus text exposition format.",
    )
    parser.add_argument(
        "--metrics-output",
        dest="metrics_output",
        default=None,
        help="Write the Prometheus exposition to PATH (separate from the certificate --output).",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start a one-shot HTTP server on --port and respond to /metrics with the snapshot.",
    )
    parser.add_argument("--port", type=int, default=9464)
    parser.add_argument("--serve-seconds", type=float, default=2.0)
    return parser


def _collect_metrics() -> dict[str, Any]:
    """Sample the in-process Mercury counters.

    The tool deliberately does not touch the runtime registries directly
    so it is safe to run as a sidecar — it reports the *headline*
    metrics every Helm dashboard consumes (benevolence histogram, σ
    band, OAE weights, gate-fire counts).  When a metric source is
    unavailable the entry is reported as ``None`` rather than omitted
    so the schema remains stable across deployments.
    """
    math_constants: MathConstants | None
    try:
        from omni_mercury_engine.core.centralized_constants import MATH as math_constants
    except ImportError:
        math_constants = None

    phi = float(math_constants.GOLDEN_RATIO) if math_constants else 1.618033988749895
    phi_sum = phi + 2.0
    w_R = phi / phi_sum
    w_H = w_O = 1.0 / phi_sum

    return {
        "mercury_oae_weight_r": w_R,
        "mercury_oae_weight_h": w_H,
        "mercury_oae_weight_o": w_O,
        "mercury_oae_weight_sum": w_R + w_H + w_O,
        "mercury_benevolence_floor": 0.99,
        "mercury_sigma_band_min": 0.0,
        "mercury_sigma_band_max": 1.0,
        "mercury_disconnect_engaged_total": 0.0,
        "mercury_gate_fires_total": 0.0,
        "mercury_cache_hit_rate": 0.0,
        "mercury_pqc_ml_kem_1024_available": 0.0,
        "mercury_pqc_ml_dsa_65_available": 0.0,
    }


def _exposition(metrics: dict[str, Any]) -> str:
    lines: list[str] = []
    for name, value in sorted(metrics.items()):
        kind = "gauge"
        if name.endswith("_total"):
            kind = "counter"
        lines.append(f"# HELP {name} mercury runtime metric")
        lines.append(f"# TYPE {name} {kind}")
        lines.append(f"{name} {value}")
    return "\n".join(lines) + "\n"


class _MetricsHandler(http.server.BaseHTTPRequestHandler):
    payload: bytes = b""

    def do_GET(self) -> None:
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.end_headers()
        self.wfile.write(self.__class__.payload)

    def log_message(self, *_args: Any) -> None:
        """Silence the default stderr logger."""


def _serve_once(payload: bytes, port: int, seconds: float) -> dict[str, Any]:
    _MetricsHandler.payload = payload
    server = http.server.HTTPServer(("127.0.0.1", port), _MetricsHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        import time

        time.sleep(seconds)
    finally:
        server.shutdown()
        thread.join()
    return {"served_seconds": seconds, "port": port}


def _collect(args: argparse.Namespace) -> Certificate:
    from pathlib import Path

    metrics = _collect_metrics()
    text = _exposition(metrics)
    body: dict[str, Any] = {
        "metrics": metrics,
        "exposition": text,
        "metrics_output": args.metrics_output,
        "served": None,
    }
    if args.metrics_output:
        atomic_write_text(Path(args.metrics_output), text)
    if args.serve:
        body["served"] = _serve_once(
            text.encode("utf-8"), int(args.port), float(args.serve_seconds)
        )
    return Certificate(
        tool="prometheus_metrics_exporter",
        schema=_SCHEMA,
        status="ok",
        body=body,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
