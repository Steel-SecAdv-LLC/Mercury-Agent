#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Harvest the real intact σ_Immutable config-integrity baseline.

The σ_Immutable gate is a **config-integrity / tamper check**: on a healthy
system the 127 operational governance scalars are bit-constant, and the gate's
job is to detect when that intact configuration is corrupted.  Historically the
gate was trained on a *synthetic* corpus (``U[0,2]`` random vectors labelled by
a threshold rule) that never occurs in production — so the shipped network
passed the real production vector only by generalisation, and a more
label-faithful retrain would have refused every call (a latent DoS).

This script closes that gap by harvesting the **actual** intact operational
vector the running engine produces, across multiple domains, and persisting it
as ``sigma_immutable_baseline.json``.  ``sigma_immutable_corpus.generate_corpus``
then builds its positives from this real baseline (plus intact-preserving
variations) and its negatives from real tamper mutations of it — so the
retrained network passes the real configuration *by construction* and learns to
agree with the deterministic critical-ethical floor on real anchor collapse.

The harvest is a **build-time** step (like training); the runtime gate never
imports the engine.  Re-run whenever the operational scalar layout changes.

Usage::

    python scripts/harvest_sigma_baseline.py [--domains default medical ...]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_SECURITY_DIR = Path(__file__).resolve().parent.parent / "src" / "omni_mercury_engine" / "security"
BASELINE_PATH = _SECURITY_DIR / "sigma_immutable_baseline.json"

BASELINE_SCHEMA = "sigma_immutable_baseline/v1"


def harvest(domains: list[str]) -> dict[str, Any]:
    """Harvest the intact operational vector and verify cross-domain constancy."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from omni_mercury_engine.core.global_omni_scalar_network import (
        get_global_scalar_network,
    )
    from omni_mercury_engine.security.sigma_immutable_gate import (
        SIGMA_ETHICAL_BAND_END,
        SIGMA_IMMUTABLE_DIM,
        SIGMA_USED_BAND_END,
        get_sigma_immutable_gate,
    )

    gate = get_sigma_immutable_gate()

    reference: dict[str, float] | None = None
    reference_names: list[str] = []
    per_domain_scores: dict[str, float] = {}

    for domain in domains:
        net = get_global_scalar_network(domain=domain)
        collected = net._collect_all_scalars()
        names = list(collected.keys())
        vals = np.asarray(list(collected.values()), dtype=np.float64)

        padded = np.zeros(SIGMA_IMMUTABLE_DIM, dtype=np.float64)
        padded[: len(vals)] = vals
        score = float(gate.evaluate(padded).score)
        per_domain_scores[domain] = round(score, 10)

        if reference is None:
            reference = dict(collected)
            reference_names = names
        else:
            # The governance scalars are config properties, not per-domain — the
            # intact vector must be identical across domains, or the "config
            # integrity" premise is broken.
            if names != reference_names:
                raise SystemExit(
                    f"scalar NAMES differ between domains ({domain} vs reference) — "
                    "the operational layout is not domain-stable; aborting harvest."
                )
            drift = max(abs(float(collected[n]) - float(reference[n])) for n in reference_names)
            if drift > 0.0:
                raise SystemExit(
                    f"scalar VALUES drift by {drift} between domains ({domain}) — "
                    "the intact config is not constant; aborting harvest."
                )

    assert reference is not None
    vals = np.asarray([reference[n] for n in reference_names], dtype=np.float64)
    anchors = get_global_scalar_network().critical_ethical_anchors()

    return {
        "schema": BASELINE_SCHEMA,
        "provenance": (
            "Harvested intact operational scalar vector from a healthy engine "
            "(GlobalOmniScalarNetwork._collect_all_scalars). This is the "
            "config-integrity reference the σ_Immutable gate is trained to "
            "recognise; deviations are tampers. Regenerate via "
            "scripts/harvest_sigma_baseline.py when the operational layout changes."
        ),
        "domains_verified": list(domains),
        "domains_constant": len(set(per_domain_scores.values())) == 1,
        "reference_sigma_score": per_domain_scores[domains[0]],
        "reference_sigma_score_note": (
            "Point-in-time snapshot: the score assigned by the σ_Immutable "
            "gate whose weights were shipped when this harvest ran. "
            "Retraining the gate from this baseline "
            "(scripts/train_sigma_immutable.py) changes the shipped gate's "
            "live constant (security/sigma_calibration.py::"
            "SIGMA_FROZEN_CONSTANT) without touching this file, so this "
            "field lags the live score until the next harvest. Corpus "
            "generation and training consume names/values_hex/anchor_names "
            "only; this score is informational."
        ),
        "ethical_band_end": int(SIGMA_ETHICAL_BAND_END),
        "used_band_end": int(SIGMA_USED_BAND_END),
        "input_dim": int(SIGMA_IMMUTABLE_DIM),
        "n_operational": len(vals),
        "anchor_names": list(anchors.keys()),
        "names": reference_names,
        # float.hex → bit-exact round-trip, matching the corpus serialisation.
        "values_hex": [float(v).hex() for v in vals.tolist()],
    }


def main() -> int:
    """Harvest the intact baseline and persist it, or exit non-zero on drift."""
    parser = argparse.ArgumentParser(description="Harvest the σ_Immutable intact baseline")
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["default", "medical", "infrastructure", "humanitarian"],
        help="Domains to harvest and cross-check for constancy.",
    )
    args = parser.parse_args()

    baseline = harvest(args.domains)
    if not baseline["domains_constant"]:
        raise SystemExit(
            f"σ score is not constant across domains: {baseline['reference_sigma_score']} "
            "— the config-integrity premise is violated; aborting."
        )

    BASELINE_PATH.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
    logger.info(
        "σ_Immutable baseline harvested: %d operational scalars, %d anchors, "
        "σ=%s (constant across %s) → %s",
        baseline["n_operational"],
        len(baseline["anchor_names"]),
        baseline["reference_sigma_score"],
        ",".join(baseline["domains_verified"]),
        BASELINE_PATH,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
