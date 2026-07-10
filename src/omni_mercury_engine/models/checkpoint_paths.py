# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Locate and load shipped model checkpoints.

Shipped checkpoints live in ``omni_mercury_engine/models/checkpoints/`` (they
are packaged via ``[tool.setuptools.package-data]``). Every shipped file is
accompanied by a ``<name>.provenance.json`` sidecar recording the real data it
was trained on and the held-out learned-vs-physics evaluation that justified
shipping it; loaders surface that provenance so a model never runs silently.

This module is a stdlib+torch leaf so both detectors and the training
pipeline can import it without cycles.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def checkpoints_dir() -> Path:
    """Directory holding shipped checkpoints inside the installed package."""
    return Path(__file__).resolve().parent / "checkpoints"


def shipped_checkpoint_path(name: str) -> Path:
    """Path of a shipped checkpoint by basename (no extension).

    The file may not exist; callers that require it should use
    :func:`load_shipped_checkpoint`, which fails loudly.
    """
    return checkpoints_dir() / f"{name}.pt"


def load_shipped_checkpoint(name: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Load a shipped checkpoint payload plus its provenance sidecar.

    Args:
        name: Checkpoint basename without extension (e.g.
            ``"solar_storm_geomag"``).

    Returns:
        Tuple of (checkpoint payload, provenance dict or None when the
        sidecar is absent).

    Raises:
        FileNotFoundError: No checkpoint of that name has been shipped. The
            message names the pipeline command that produces one.
        RuntimeError: The checkpoint file exists but does not match its
            provenance sidecar's pinned sha256, or cannot be deserialized
            (corruption or tampering) -- never silently ignored.
    """
    path = shipped_checkpoint_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"no shipped checkpoint at {path}. Train and ship one with: "
            f"python scripts/train_hazard_checkpoints.py --hook <hook> "
            "--fetch --train --evaluate --ship (the merit gate must pass)."
        )

    provenance: dict[str, Any] | None = None
    sidecar = path.with_name(f"{name}.provenance.json")
    if sidecar.exists():
        try:
            provenance = json.loads(sidecar.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"provenance sidecar {sidecar} is unreadable: {exc}. A shipped "
                "checkpoint without valid provenance must not load silently."
            ) from exc
        # Integrity first: verify the artifact against the sidecar's pinned
        # digest BEFORE deserializing anything from it.
        pinned_sha = provenance.get("checkpoint_sha256")
        if pinned_sha:
            actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_sha != pinned_sha:
                raise RuntimeError(
                    f"shipped checkpoint {path} does not match its provenance "
                    f"sidecar (sha256 {actual_sha} != pinned {pinned_sha}). The "
                    "artifact was modified after shipping or the sidecar is "
                    "stale; refusing to load unverifiable weights. Re-ship "
                    "both from the training pipeline."
                )

    try:
        # Hard-pinned to weights_only=True (house convention): shipped
        # checkpoints are state dicts + primitive containers only, so the
        # safe-load mode admits them and arbitrary-code pickles are refused.
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"shipped checkpoint {path} is unreadable/corrupt: {exc}. Refusing to "
            "run with broken weights; re-ship from the training pipeline."
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"shipped checkpoint {path} has unexpected payload type "
            f"{type(payload).__name__}; expected a dict of state dicts."
        )

    if provenance is not None:
        evaluation = provenance.get("evaluation", {})
        logger.info(
            "checkpoint %s provenance: trained seed=%s, test_years=%s, %s learned=%s vs "
            "physics=%s (learned_beats_physics=%s)",
            name,
            provenance.get("seed"),
            evaluation.get("test_years"),
            evaluation.get("primary_metric"),
            (evaluation.get("learned") or {}).get(evaluation.get("primary_metric", "")),
            (evaluation.get("physics") or {}).get(evaluation.get("primary_metric", "")),
            evaluation.get("learned_beats_physics"),
        )
    else:
        logger.warning(
            "checkpoint %s has no provenance sidecar (%s missing); it predates the "
            "provenance requirement or was copied in by hand",
            name,
            sidecar,
        )
    return payload, provenance
