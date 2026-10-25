# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Which pinned HuggingFace models are present in the local cache.

Shared by the ``MERCURY_REQUIRES_HF_MODELS`` session gate (``conftest.py``)
and the real-model tier's skip condition, so the two can never disagree.
"""

from __future__ import annotations


def missing_pinned_models() -> list[str]:
    """``model@revision`` for every pinned default whose weights are not cached."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return ["huggingface_hub (not installed)"]
    from omni_mercury_engine.detectors.vlm.lvlm_backends import pinned_default_models

    missing = []
    for model_id, revision in sorted(pinned_default_models().items()):
        config = try_to_load_from_cache(model_id, "config.json", revision=revision)
        weights = [
            try_to_load_from_cache(model_id, name, revision=revision)
            for name in ("model.safetensors", "pytorch_model.bin")
        ]
        if not (isinstance(config, str) and any(isinstance(w, str) for w in weights)):
            missing.append(f"{model_id}@{revision[:12]}")
    return missing
