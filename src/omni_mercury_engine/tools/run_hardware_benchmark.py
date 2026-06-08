# Copyright (C) 2025 Steel Security Advisors LLC
"""(at your option) any later version."""

from __future__ import annotations

import argparse
import os
import platform
import time
from typing import Any

import numpy as np

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.run_hardware_benchmark/v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.run_hardware_benchmark",
        description="Hardware capability + deterministic micro-benchmark certificate.",
    )
    parser.add_argument(
        "--matmul-dim",
        type=int,
        default=512,
        help="Square matrix dimension for the matmul micro-bench (default 512).",
    )
    parser.add_argument(
        "--fft-n",
        type=int,
        default=1 << 18,
        help="FFT input length, power-of-two (default 262144).",
    )
    parser.add_argument(
        "--repeats", type=int, default=5, help="Repeat count per micro-bench (default 5)."
    )
    return parser


def _cpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count_logical": os.cpu_count(),
    }
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    info["model_name"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    info["mem_total_kb"] = int(line.split()[1])
                    break
    except OSError:
        pass
    return info


def _gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"available": False}
    try:
        import torch

        if torch.cuda.is_available():
            info["available"] = True
            info["count"] = torch.cuda.device_count()
            info["devices"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        info["torch"] = "not installed"
    return info


def _matmul_bench(dim: int, repeats: int) -> dict[str, float]:
    rng = np.random.default_rng(0)
    A = rng.standard_normal((dim, dim)).astype(np.float64)
    B = rng.standard_normal((dim, dim)).astype(np.float64)
    # Warm-up
    np.dot(A, B)
    lat: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        np.dot(A, B)
        lat.append(time.perf_counter() - t0)
    arr = np.asarray(lat)
    return {
        "dim": float(dim),
        "median_seconds": float(np.median(arr)),
        "min_seconds": float(arr.min()),
        "gflops_median": float(2.0 * dim**3 / np.median(arr) / 1e9),
    }


def _fft_bench(n: int, repeats: int) -> dict[str, float]:
    rng = np.random.default_rng(1)
    x = rng.standard_normal(n).astype(np.float64)
    np.fft.fft(x)  # warmup
    lat: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        np.fft.fft(x)
        lat.append(time.perf_counter() - t0)
    arr = np.asarray(lat)
    return {
        "n": float(n),
        "median_seconds": float(np.median(arr)),
        "min_seconds": float(arr.min()),
    }


def _collect(args: argparse.Namespace) -> Certificate:
    body: dict[str, Any] = {
        "cpu": _cpu_info(),
        "gpu": _gpu_info(),
        "numpy_version": np.__version__,
        "matmul_benchmark": _matmul_bench(args.matmul_dim, args.repeats),
        "fft_benchmark": _fft_bench(args.fft_n, args.repeats),
    }
    return Certificate(
        tool="run_hardware_benchmark",
        schema=_SCHEMA,
        status="ok",
        body=body,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
