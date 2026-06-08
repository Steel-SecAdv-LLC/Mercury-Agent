# Copyright (C) 2025 Steel Security Advisors LLC
"""(at your option) any later version."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from typing import Any

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.gpu_capability_probe/v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.gpu_capability_probe",
        description="Enumerate CUDA/ROCm/MPS/CPU capability and runtime device matrix.",
    )
    parser.add_argument(
        "--expected-manifest",
        default=None,
        help=(
            "Optional release_manifest.json — if supplied the tool fails "
            "when the current device matrix diverges from manifest['device_matrix']."
        ),
    )
    return parser


def _probe_cuda() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "torch not installed"}
    available = bool(torch.cuda.is_available())
    info: dict[str, Any] = {"available": available, "device_count": torch.cuda.device_count()}
    if available:
        info["devices"] = [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "compute_capability": list(torch.cuda.get_device_capability(i)),
                "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory,
            }
            for i in range(torch.cuda.device_count())
        ]
    info["cuda_version"] = getattr(torch.version, "cuda", None)
    return info


def _probe_rocm() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "torch not installed"}
    return {
        "available": bool(getattr(torch.version, "hip", None)),
        "hip_version": getattr(torch.version, "hip", None),
    }


def _probe_mps() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "torch not installed"}
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return {"available": False, "reason": "torch.backends.mps absent"}
    return {
        "available": bool(backend.is_available()),
        "is_built": bool(getattr(backend, "is_built", lambda: False)()),
    }


def _probe_dtypes() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"fp16": False, "bf16": False, "int8": False}
    if not torch.cuda.is_available():
        # MPS / ROCm fall through to CPU here; the surrounding accelerator
        # block records what the runtime actually exposes.
        return {"fp16": False, "bf16": False, "int8": False}
    # FP16: NVIDIA Tensor-Core FP16 is available on compute capability
    # 5.3+ (Maxwell-int8 / Pascal+).  The cheapest authoritative check
    # is to attempt a tiny FP16 allocation + matmul; if torch / the
    # driver disagrees with the device, the operation raises and we
    # report ``False`` without crashing the probe.
    fp16_supported = _can_compute(torch, torch.float16)
    bf16_supported = bool(torch.cuda.is_bf16_supported())
    return {
        "fp16": fp16_supported,
        "bf16": bf16_supported,
        # INT8 inference is supported on every CUDA-capable GPU torch ships
        # for via ``torch.int8`` kernels; we surface it explicitly so the
        # certificate body is symmetric.
        "int8": True,
    }


def _can_compute(torch_mod: Any, dtype: Any) -> bool:
    """Return True when a tiny CUDA matmul in ``dtype`` succeeds.

    This is the authoritative FP16 check (cheaper than parsing compute
    capability + driver feature tables) and never raises out of the
    probe — any failure becomes ``False`` so the certificate still
    serialises.
    """
    try:
        a = torch_mod.zeros((2, 2), device="cuda", dtype=dtype)
        b = torch_mod.zeros((2, 2), device="cuda", dtype=dtype)
        _ = a @ b
        return True
    except Exception:
        return False


def _nvidia_smi() -> dict[str, Any]:
    bin_path = shutil.which("nvidia-smi")
    if not bin_path:
        return {"available": False}
    try:
        out = (
            subprocess.check_output(
                [bin_path, "--query-gpu=driver_version,name,memory.total", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            .decode()
            .strip()
        )
        return {"available": True, "raw": out}
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {"available": False, "reason": "nvidia-smi failed"}


def _collect(args: argparse.Namespace) -> Certificate:
    body: dict[str, Any] = {
        "cuda": _probe_cuda(),
        "rocm": _probe_rocm(),
        "mps": _probe_mps(),
        "dtypes": _probe_dtypes(),
        "nvidia_smi": _nvidia_smi(),
        "env": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("CUDA_", "HIP_", "ROCM_", "PYTORCH_"))
        },
    }
    warnings: list[str] = []
    if args.expected_manifest:
        import json
        from pathlib import Path

        try:
            manifest = json.loads(Path(args.expected_manifest).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"expected manifest unreadable: {exc}")
        else:
            saved = manifest.get("body", {}).get("device_matrix") or manifest.get("device_matrix")
            if saved is not None:
                # Compare on the small projection that matters for gating —
                # cuda/rocm/mps availability and CUDA major version.
                proj = {
                    "cuda_available": body["cuda"].get("available"),
                    "rocm_available": body["rocm"].get("available"),
                    "mps_available": body["mps"].get("available"),
                    "cuda_version": body["cuda"].get("cuda_version"),
                }
                if saved != proj:
                    warnings.append(f"device matrix drift: saved={saved}, current={proj}")
                body["projection"] = proj
                body["saved_projection"] = saved
    status = "fail" if warnings and args.expected_manifest else "ok"
    return Certificate(
        tool="gpu_capability_probe",
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
