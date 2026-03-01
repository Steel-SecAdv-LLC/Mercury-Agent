"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

from __future__ import annotations

"""
Centralized optional-dependency availability checks.

Import flags from this module instead of scattering ``try/except ImportError``
blocks throughout the codebase.  Every flag uses ``importlib.util.find_spec``
which probes for the package **without importing it**, keeping startup fast and
side-effect-free.

Usage::

    from omni_mercury_engine._compat import HAS_TORCH

    if HAS_TORCH:
        import torch
"""

from importlib.util import find_spec

# ---------------------------------------------------------------------------
# Core ML stack (mercury-agent[ml])
# ---------------------------------------------------------------------------
HAS_TORCH: bool = find_spec("torch") is not None
HAS_TORCHVISION: bool = find_spec("torchvision") is not None
HAS_PYTORCH_LIGHTNING: bool = find_spec("pytorch_lightning") is not None
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
HAS_AVA_GUARDIAN: bool = find_spec("ava_guardian") is not None
HAS_LIBOQS: bool = find_spec("oqs") is not None

# ---------------------------------------------------------------------------
# Optimization / benchmarking
# ---------------------------------------------------------------------------
HAS_JOBLIB: bool = find_spec("joblib") is not None
HAS_PSUTIL: bool = find_spec("psutil") is not None
HAS_HYPOTHESIS: bool = find_spec("hypothesis") is not None
HAS_MATPLOTLIB: bool = find_spec("matplotlib") is not None

# ---------------------------------------------------------------------------
# Reinforcement learning
# ---------------------------------------------------------------------------
HAS_STABLE_BASELINES: bool = find_spec("stable_baselines3") is not None

# ---------------------------------------------------------------------------
# Math / science
# ---------------------------------------------------------------------------
HAS_SYMPY: bool = find_spec("sympy") is not None
HAS_MPMATH: bool = find_spec("mpmath") is not None

# ---------------------------------------------------------------------------
# Convenience groupings for common checks
# ---------------------------------------------------------------------------
HAS_ML_STACK: bool = HAS_TORCH
"""True when PyTorch is installed (minimum for ML detectors). Mercury-native utils replace scikit-learn."""

HAS_VISUAL_STACK: bool = HAS_TORCH and HAS_TORCHVISION and HAS_TIMM
"""True when the full visual anomaly detection stack is available."""

HAS_VLM_STACK: bool = HAS_TORCH and HAS_TRANSFORMERS and HAS_ACCELERATE
"""True when Vision-Language Model detectors can run."""
