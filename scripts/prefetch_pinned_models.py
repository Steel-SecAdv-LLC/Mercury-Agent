#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Download the pinned HuggingFace models the shipped VLM defaults load.

The list is read from the backends' own ``DEFAULT_REVISIONS`` (the
SafeHFLoader pins), so this can never fetch a revision the code would not
load.  Populates the local HuggingFace cache; afterwards the real-model test
tier (``tests/detectors/test_vlm_real_models.py``) runs offline, and
``MERCURY_REQUIRES_HF_MODELS=1`` can declare it a must-run tier.

Usage::

    python scripts/prefetch_pinned_models.py
"""

from __future__ import annotations

import sys


def pinned_models() -> dict[str, str]:
    """``{model_id: revision}`` for every backend default with a shipped pin."""
    from omni_mercury_engine.detectors.vlm.lvlm_backends import (
        BLIPCaptionBackend,
        BLIPVQABackend,
        HFTextReasoningBackend,
    )

    pins: dict[str, str] = {}
    for backend in (BLIPCaptionBackend, BLIPVQABackend, HFTextReasoningBackend):
        pins.update(backend.DEFAULT_REVISIONS)
    return pins


def main() -> int:
    """Fetch every pinned model's weights and configuration files."""
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    for model_id, revision in sorted(pinned_models().items()):
        files = set(api.list_repo_files(model_id, revision=revision))
        # One weight format: safetensors where the revision has it, else the
        # PyTorch pickle (loaded by transformers under its safe-load rules).
        weights = "*.safetensors" if any(f.endswith(".safetensors") for f in files) else "*.bin"
        path = snapshot_download(
            model_id,
            revision=revision,
            allow_patterns=["*.json", "*.txt", "*.model", weights],
        )
        print(f"{model_id}@{revision} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
