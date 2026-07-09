#!/usr/bin/env python
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
r"""Train, evaluate, and ship hazard-detector checkpoints on real data.

Drives the :mod:`omni_mercury_engine.ml.hazard_training` pipeline per
``load_neural_weights()`` hook. Every stage is deterministic (seeded),
cached on disk with sha256 provenance, and gated: ``ship`` refuses unless
the trained model beats the detector's physics fallback on the held-out
test years (the merit gate — see ``docs/HAZARD_CHECKPOINT_TRAINING.md``).

Usage::

    # Full run for the geomagnetic-storm hook (fetch -> ... -> ship):
    PYTHONPATH=src python scripts/train_hazard_checkpoints.py \
        --hook solar_storm --stage all

    # Audit table for every hook (what trains where, and why not here):
    python scripts/train_hazard_checkpoints.py --audit
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from omni_mercury_engine.ml.hazard_training.common import (
    HazardDataUnavailableError,
    MeritGateError,
    PipelineContext,
)
from omni_mercury_engine.ml.hazard_training.registry import (
    HOOK_REGISTRY,
    get_hook,
    run_stage,
)

logger = logging.getLogger("train_hazard_checkpoints")

_ALL_STAGES = ("fetch", "build", "train", "evaluate", "ship")


def _print_audit() -> None:
    """Print the per-hook audit table (category, data requirement)."""
    width = max(len(name) for name in HOOK_REGISTRY)
    print(f"{'hook':<{width}}  cat  detector / data requirement")
    print("-" * 100)
    for name, entry in sorted(HOOK_REGISTRY.items()):
        print(f"{name:<{width}}  ({entry.category})  {entry.detector}")
        print(f"{'':<{width}}       arch: {entry.architecture}")
        print(f"{'':<{width}}       data: {entry.data_requirement}")
        print()


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hook", help="Hook registry key (see --audit)")
    parser.add_argument(
        "--stage",
        default="all",
        choices=[*_ALL_STAGES, "all"],
        help="Pipeline stage to run (default: all, in order)",
    )
    parser.add_argument("--audit", action="store_true", help="Print the 11-hook audit table")
    parser.add_argument("--data-dir", type=Path, default=None, help="Cache/workspace override")
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument(
        "--ship-dir",
        type=Path,
        default=None,
        help="Override the shipped-checkpoints directory (tests)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.audit:
        _print_audit()
        return 0
    if not args.hook:
        parser.error("--hook is required unless --audit is given")

    entry = get_hook(args.hook)
    ctx_kwargs: dict[str, object] = {"seed": args.seed, "max_epochs": args.max_epochs}
    if args.data_dir is not None:
        ctx_kwargs["data_dir"] = args.data_dir
    if args.ship_dir is not None:
        ctx_kwargs["ship_dir"] = args.ship_dir
    ctx = PipelineContext(**ctx_kwargs)  # type: ignore[arg-type]

    stages = _ALL_STAGES if args.stage == "all" else (args.stage,)
    try:
        for stage in stages:
            logger.info("=== %s: %s ===", entry.name, stage)
            result = run_stage(entry.name, stage, ctx)
            if stage == "evaluate":
                print(json.dumps(result.to_json(), indent=2))
            elif stage == "ship":
                ckpt, sidecar = result
                print(f"shipped: {ckpt}\nprovenance: {sidecar}")
    except HazardDataUnavailableError as exc:
        logger.error("%s", exc)
        return 2
    except MeritGateError as exc:
        logger.error("MERIT GATE REFUSED SHIP: %s", exc)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
