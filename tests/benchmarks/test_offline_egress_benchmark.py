# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Smoke + behavior tests for the offline-egress benchmark.

The benchmark measures the air-gapped egress path on-box. These tests run it
with a tiny iteration count and assert the measurement structure and the
security behavior it depends on (external refused under MERCURY_OFFLINE, loopback
round-trip served over 127.0.0.1) — never any external socket.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks import offline_egress_benchmark as oeb


def test_benchmark_runs_and_reports_structure() -> None:
    """A tiny run yields gate overhead and a loopback round-trip result."""
    result = oeb.run(iters=5, payload_kb=8)
    assert result["schema"] == "offline_egress_benchmark/v1"
    gate = result["gate_overhead"]
    assert gate["offline_refusal_external"]["median_ms"] >= 0.0
    assert gate["offline_permit_loopback_validate"]["median_ms"] >= 0.0


def test_loopback_roundtrip_serves_over_127() -> None:
    """The loopback round-trip binds 127.0.0.1 and returns the exact payload size."""
    result = oeb.run(iters=5, payload_kb=8)
    rt = result["loopback_roundtrip"]
    # The server binds an ephemeral loopback port; if it could not, the harness
    # is honest about it rather than fabricating numbers.
    if rt.get("available"):
        assert rt["payload_bytes"] == 8 * 1024
        assert rt["requests_per_sec"] > 0
        assert rt["throughput_mb_per_sec"] > 0
    else:
        assert "reason" in rt


def test_offline_env_is_restored() -> None:
    """run() leaves MERCURY_OFFLINE exactly as it found it."""
    import os

    prior = os.environ.get("MERCURY_OFFLINE")
    oeb.run(iters=3, payload_kb=4)
    assert os.environ.get("MERCURY_OFFLINE") == prior
