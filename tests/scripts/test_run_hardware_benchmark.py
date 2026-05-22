"""Smoke tests for ``scripts/run_hardware_benchmark.py``.

These tests verify exit-code semantics and the JSON report schema.
They do *not* assert specific timing numbers — those depend on the
host and are validated by the harness itself when ``--min-ops-per-sec``
is provided.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_hardware_benchmark.py"
CANONICAL = ROOT / "configs" / "lyapunov_canonical.yaml"


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_missing_config_exits_2(tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    res = _run(["--config", str(tmp_path / "nope.yaml"), "--out", str(out)])
    assert res.returncode == 2, res.stderr
    assert not out.exists()


def test_canonical_run_exit_0_and_schema(tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    res = _run(
        [
            "--config",
            str(CANONICAL),
            "--iters",
            "30",
            "--warmup",
            "5",
            "--out",
            str(out),
        ]
    )
    assert res.returncode == 0, res.stderr
    report = json.loads(out.read_text())

    # Required top-level keys.
    assert set(report) >= {"config", "environment", "validation", "timing"}

    # Validation must have certified the canonical config.
    assert report["validation"]["ok"] is True
    assert report["validation"]["mode"] == "quadratic"
    assert report["validation"]["computed_lambda"] >= report["validation"]["claimed_lambda"]

    # Timing schema invariants.
    t = report["timing"]
    assert t["iters"] == 30
    assert t["warmup"] == 5
    assert t["samples"] == 25
    for key in ("mean_s", "p50_s", "p95_s", "p99_s", "max_s", "total_s", "ops_per_sec"):
        assert isinstance(t[key], float)
        assert t[key] > 0
    # Percentile ordering must hold (sanity check on the helper).
    assert t["p50_s"] <= t["p95_s"] <= t["p99_s"] <= t["max_s"]
    # Reported throughput is ``samples / total_s``.  Because ``mean_s``
    # is the arithmetic mean of the same samples, this quantity equals
    # ``1 / mean_s`` up to floating-point round-off; we pin the
    # stronger identity (``ops_per_sec == samples / total_s`` exactly)
    # because the harness emits both ``total_s`` and ``samples``
    # explicitly to make this re-derivation independent of the mean.
    assert abs(t["ops_per_sec"] - (t["samples"] / t["total_s"])) < 1e-9
    # Sanity: the two formulas agree, demonstrating the invariant
    # holds for the documented arithmetic-mean estimator.
    assert abs(t["ops_per_sec"] - (1.0 / t["mean_s"])) < 1e-6

    # Environment fingerprint must capture the version-sensitive bits.
    env = report["environment"]
    for key in ("python", "numpy", "platform", "cpu_count"):
        assert env[key], f"missing environment field {key!r}"
    # ``cpu_governor`` may be None on non-Linux runners; the key must
    # still be present so consumers can detect missing data explicitly.
    assert "cpu_governor" in env


def test_throughput_floor_triggers_regression_exit(tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    res = _run(
        [
            "--config",
            str(CANONICAL),
            "--iters",
            "20",
            "--warmup",
            "2",
            "--out",
            str(out),
            # An unreachable throughput floor — the harness must report code 4.
            "--min-ops-per-sec",
            "1e18",
        ]
    )
    assert res.returncode == 4, res.stderr
    # The report should still have been written before the regression check.
    assert out.exists()
    report = json.loads(out.read_text())
    assert report["validation"]["ok"] is True


def test_warmup_validation() -> None:
    # warmup >= iters is illegal and must surface as a non-zero exit.
    res = _run(["--iters", "5", "--warmup", "5"])
    assert res.returncode != 0
    assert "warmup" in (res.stderr + res.stdout).lower()


@pytest.mark.parametrize("bad_iters", ["0", "-1"])
def test_non_positive_iters_rejected(bad_iters: str) -> None:
    res = _run(["--iters", bad_iters, "--warmup", "0"])
    assert res.returncode != 0
