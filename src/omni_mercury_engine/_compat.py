# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Centralized optional-dependency availability checks.

Import flags from this module instead of scattering ``try/except ImportError``
blocks throughout the codebase.  Every flag uses ``importlib.util.find_spec``
which probes for the package **without importing it**, keeping startup fast and
side-effect-free.

Usage::

    from omni_mercury_engine._compat import HAS_TORCH, HAS_TORCHVISION

    if HAS_TORCH:
        import torch
"""

from __future__ import annotations

from importlib.util import find_spec

# ---------------------------------------------------------------------------
# Core ML stack (mercury-agent[ml])
# ---------------------------------------------------------------------------
HAS_TORCH: bool = find_spec("torch") is not None
HAS_TORCHVISION: bool = find_spec("torchvision") is not None
HAS_PYTORCH_LIGHTNING: bool = find_spec("pytorch_lightning") is not None
# NOTE: scikit-learn is a *competitor*, not a Mercury dependency, and is
# confined to ``benchmarks/`` (head-to-head baselines only) — so there is
# deliberately no ``HAS_SKLEARN`` probe here. The repo-wide guard
# ``tests/test_no_sklearn_in_src.py`` fails loudly if sklearn reappears outside
# ``benchmarks/`` — including via a dynamic ``find_spec`` / importorskip probe.
HAS_TIMM: bool = find_spec("timm") is not None
HAS_CV2: bool = find_spec("cv2") is not None

# ---------------------------------------------------------------------------
# Visual / VLM extras (mercury-agent[visual], mercury-agent[vlm])
# ---------------------------------------------------------------------------
HAS_TRANSFORMERS: bool = find_spec("transformers") is not None
HAS_ACCELERATE: bool = find_spec("accelerate") is not None
HAS_PIL: bool = find_spec("PIL") is not None

# ---------------------------------------------------------------------------
# Foundation model extras (mercury-agent[foundation])
# ---------------------------------------------------------------------------
HAS_STUMPY: bool = find_spec("stumpy") is not None
HAS_NIXTLA: bool = find_spec("nixtla") is not None

# ---------------------------------------------------------------------------
# SOTA model extras (mercury-agent[sota])
# ---------------------------------------------------------------------------
HAS_EINOPS: bool = find_spec("einops") is not None

# ---------------------------------------------------------------------------
# Face recognition (mercury-agent[face])
# ---------------------------------------------------------------------------
HAS_DEEPFACE: bool = find_spec("deepface") is not None

# ---------------------------------------------------------------------------
# API server (mercury-agent[api])
# ---------------------------------------------------------------------------
HAS_FASTAPI: bool = find_spec("fastapi") is not None
HAS_HTTPX: bool = find_spec("httpx") is not None
HAS_UVICORN: bool = find_spec("uvicorn") is not None

# ---------------------------------------------------------------------------
# LLM integration (mercury-agent[llm])
# ---------------------------------------------------------------------------
HAS_OPENAI: bool = find_spec("openai") is not None
HAS_ANTHROPIC: bool = find_spec("anthropic") is not None

# ---------------------------------------------------------------------------
# Fairness (mercury-agent[fairness])
# ---------------------------------------------------------------------------
HAS_FAIRLEARN: bool = find_spec("fairlearn") is not None

# ---------------------------------------------------------------------------
# Streaming (mercury-agent[streaming])
# ---------------------------------------------------------------------------
HAS_REDIS: bool = find_spec("redis") is not None
HAS_AIOKAFKA: bool = find_spec("aiokafka") is not None

# ---------------------------------------------------------------------------
# Post-Quantum Cryptography
# ---------------------------------------------------------------------------
HAS_AMA_CRYPTOGRAPHY: bool = find_spec("ama_cryptography") is not None
HAS_AVA_GUARDIAN: bool = HAS_AMA_CRYPTOGRAPHY or find_spec("ava_guardian") is not None

# ---------------------------------------------------------------------------
# Optimization / benchmarking
# ---------------------------------------------------------------------------
HAS_JOBLIB: bool = find_spec("joblib") is not None
HAS_PSUTIL: bool = find_spec("psutil") is not None
HAS_HYPOTHESIS: bool = find_spec("hypothesis") is not None
HAS_MATPLOTLIB: bool = find_spec("matplotlib") is not None

# ---------------------------------------------------------------------------
# Math / science
# ---------------------------------------------------------------------------
HAS_SYMPY: bool = find_spec("sympy") is not None
HAS_MPMATH: bool = find_spec("mpmath") is not None

# ---------------------------------------------------------------------------
# Convenience groupings for common checks
# ---------------------------------------------------------------------------
HAS_ML_STACK: bool = HAS_TORCH
"""
True when PyTorch is installed (minimum for ML detectors).

Mercury uses its own native ML primitives.
"""

HAS_VISUAL_STACK: bool = HAS_TORCH and HAS_TORCHVISION and HAS_TIMM
"""True when the full visual anomaly detection stack is available."""

HAS_VLM_STACK: bool = HAS_TORCH and HAS_TRANSFORMERS and HAS_ACCELERATE
"""True when Vision-Language Model detectors can run."""


def preload_triton_before_tensorflow() -> None:
    """Bind triton's native LLVM symbols before TensorFlow can load its own.

    TensorFlow and triton (torch's compiler backend) each bundle an LLVM;
    importing triton *after* TensorFlow hard-segfaults the process during
    ``libtriton`` initialisation (mismatched LLVM symbol resolution) —
    reproduced with ``import tensorflow; import triton`` under tensorflow
    2.21 + triton 3.7.1, while the reverse order is safe.  The hazard is
    real in any ``[all]``-style install on Python <= 3.13: deepface pulls
    TensorFlow into the process, and any later torchvision-backed detector
    import reaches ``torchvision.ops`` -> ``torch._dynamo`` -> its triton
    probe -> crash.  Call this immediately before any import that can pull
    TensorFlow (the deepface guards in ``models/biometric*.py``) so
    detector-registry discovery order can never produce the fatal
    TF-then-triton sequence.
    """
    if not HAS_TORCH:
        # No torch => torch._dynamo will never probe triton in this
        # process; the ordering hazard cannot arise.
        return
    try:
        import triton  # noqa: F401
    except Exception:  # nosec B110 - triton absent or unloadable here is fine:
        # torch._dynamo's own probe would fail the same way and proceed
        # without it, so there is no later crash to prevent.
        pass
