# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Which pinned HuggingFace models are present in the local cache.

Shared by the ``MERCURY_REQUIRES_HF_MODELS`` session gate (``conftest.py``)
and the real-model tier's skip condition, so the two can never disagree.
"""

from __future__ import annotations

from pathlib import Path


def missing_pinned_models() -> list[str]:
    """``model@revision`` for every pinned default whose weights are not cached."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return ["huggingface_hub (not installed)"]
    import sys

    scripts = Path(__file__).resolve().parent.parent / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        from prefetch_pinned_models import pinned_models
    finally:
        sys.path.remove(str(scripts))

    missing = []
    for model_id, revision in sorted(pinned_models().items()):
        config = try_to_load_from_cache(model_id, "config.json", revision=revision)
        weights = [
            try_to_load_from_cache(model_id, name, revision=revision)
            for name in ("model.safetensors", "pytorch_model.bin")
        ]
        if not (isinstance(config, str) and any(isinstance(w, str) for w in weights)):
            missing.append(f"{model_id}@{revision[:12]}")
    return missing
