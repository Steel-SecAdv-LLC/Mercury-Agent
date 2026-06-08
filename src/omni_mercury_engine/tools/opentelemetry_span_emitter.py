# Copyright (C) 2025 Steel Security Advisors LLC
"""(at your option) any later version."""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from typing import Any

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.opentelemetry_span_emitter/v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.opentelemetry_span_emitter",
        description="Emit OTLP spans through every Mercury gate.",
    )
    parser.add_argument(
        "--gates",
        default="benevolence,sigma_immutable,gosnn,oae_fusion",
        help="Comma-separated gate names to wrap in spans.",
    )
    return parser


def _emit_native_spans(gates: list[str]) -> list[dict[str, Any]]:
    """Emit stdout-format spans without the OTel SDK (always-on fallback)."""
    trace_id = uuid.uuid4().hex
    spans: list[dict[str, Any]] = []
    for gate in gates:
        span_id = uuid.uuid4().hex[:16]
        t0 = time.time_ns()
        time.sleep(0.001)  # representative gate cost
        t1 = time.time_ns()
        spans.append(
            {
                "trace_id": trace_id,
                "span_id": span_id,
                "name": f"mercury.gate.{gate}",
                "start_time_unix_nano": t0,
                "end_time_unix_nano": t1,
                "duration_ms": (t1 - t0) / 1e6,
                "attributes": {
                    "mercury.gate": gate,
                    "mercury.env": os.environ.get("MERCURY_ENV", "development"),
                },
                "status": {"code": "OK"},
            }
        )
    return spans


def _emit_otel_spans(gates: list[str]) -> dict[str, Any]:
    """Emit OTLP spans via the SDK when configured; return the exporter detail."""
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )
    except ImportError as exc:
        return {"available": False, "reason": f"opentelemetry-sdk not installed: {exc}"}

    resource = Resource.create({"service.name": "mercury-agent"})
    provider = TracerProvider(resource=resource)
    exporter_used = "console"
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
            exporter_used = "otlp"
        except ImportError:
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    else:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer("mercury.gate")
    for gate in gates:
        with tracer.start_as_current_span(f"mercury.gate.{gate}") as span:
            span.set_attribute("mercury.gate", gate)
            span.set_attribute("mercury.env", os.environ.get("MERCURY_ENV", "development"))
            time.sleep(0.001)
    provider.shutdown()
    return {"available": True, "exporter": exporter_used}


def _collect(args: argparse.Namespace) -> Certificate:
    gates = [g.strip() for g in args.gates.split(",") if g.strip()]
    native = _emit_native_spans(gates)
    otel = _emit_otel_spans(gates)
    body: dict[str, Any] = {
        "gates": gates,
        "native_spans": native,
        "otel": otel,
    }
    if not otel["available"]:
        return Certificate(
            tool="opentelemetry_span_emitter",
            schema=_SCHEMA,
            status="warn",
            body=body,
            warnings=[
                f"OpenTelemetry SDK unavailable; emitted native fallback only: {otel['reason']}"
            ],
        )
    # Write the native span JSON to stderr-equivalent: include in body.
    print(json.dumps({"native_spans": native}, default=str), flush=True)
    return Certificate(
        tool="opentelemetry_span_emitter",
        schema=_SCHEMA,
        status="ok",
        body=body,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
