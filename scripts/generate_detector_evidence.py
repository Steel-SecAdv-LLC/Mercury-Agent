#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generate the Detector Evidence Ledger - machine-readable, tiers computed.

One card per manifest detector, assembled ONLY from committed sources:

* the declared ontology (``core/detector_ontology.py``, itself gated so
  every validation claim cites a committed evidence path);
* the measured evidence seed (``artifacts/detector_evidence/
  2026-10-27_ledger_seed.json``: the full-manifest sweep, its rev-2
  harness corrections, and the fitted-reference re-measurements);
* the manifest's own structural truth (category, modality, seams).

Maturity tiers are COMPUTED from those inputs by the fixed rules below -
never hand-assigned, or the tier column becomes the next vanity badge:

* T0  measured broken on its declared modality (constant output, or
      below-chance with no shipped fix);
* T1  runs; ungraded, graded at ~chance, or protocol-mismatched;
* T2  verified: real-data mean AUROC in [0.60, 0.75), or driven on real
      domain inputs by committed tests without a recorded benchmark metric;
* T3  benchmark-proven: real-data mean AUROC >= 0.75 on committed fixtures;
* T4  domain-proven: a committed domain artifact records detection
      performance on real labeled domain data (e.g. ALFA);
* T5  production-proven: reserved; nothing may claim it until live
      deployment evidence exists.

Usage:
    python scripts/generate_detector_evidence.py            # write ledger
    python scripts/generate_detector_evidence.py --check    # verify in sync
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from omni_mercury_engine.core.detector_ontology import ONTOLOGY
from omni_mercury_engine.core.detector_registry import DETECTOR_MANIFEST

SEED_PATH = REPO_ROOT / "artifacts" / "detector_evidence" / "2026-10-27_ledger_seed.json"
LEDGER_PATH = REPO_ROOT / "artifacts" / "detector_evidence" / "ledger.json"

#: Detectors whose generic grade is a protocol mismatch, not a verdict
#: (point-wise AUROC under-credits sequential change/tail statistics).
PROTOCOL_MISMATCH = frozenset({"bocpd", "sprt", "page_hinkley", "spot_evt", "hawkes"})

#: Measured-broken on the declared modality, with no shipped fix yet:
#: routes to the ontology's input-contract correction, stated as T0.
MEASURED_BROKEN = frozenset({"affective", "chemistry", "parapsychology"})


def _metrics_for(name: str, seed: dict[str, Any]) -> dict[str, Any]:
    """Collect this detector's measured numbers from the committed seed."""
    out: dict[str, Any] = {}
    fixed = seed.get("rev2_corrections", {}).get("fixed_via_fitted_reference", {})
    harness = seed.get("rev2_corrections", {}).get("harness_errors", {})
    if name in fixed and isinstance(fixed[name], dict):
        out["adbench_mean_auroc"] = fixed[name]["adbench_mean_auroc"]
        out["burst_auroc"] = fixed[name].get("burst_hi")
        out["note"] = "fitted-reference retrofit, re-measured 2026-10-27"
        return out
    if name in harness:
        out["adbench_mean_auroc"] = harness[name]["adbench_mean_auroc"]
        out["note"] = "rev-2 harness correction (never broken)"
        return out
    for row in seed.get("sweep", {}).get("results", []):
        if row.get("name") != name:
            continue
        scored = [v["auroc"] for v in row.get("data", {}).values() if v.get("status") == "scored"]
        per_ds = {
            ds: (round(v["auroc"], 4) if v.get("status") == "scored" else v.get("status"))
            for ds, v in row.get("data", {}).items()
        }
        out["datasets"] = per_ds
        if scored:
            key = "adbench_mean_auroc" if any("thyroid" in d for d in per_ds) else "nab_mean_auroc"
            out[key] = round(float(np.mean(scored)), 4)
            out["n_datasets_scored"] = len(scored)
        burst = seed.get("burst_probe", {})
        if name in burst:
            out["burst_auroc"] = burst[name]
        return out
    burst = seed.get("burst_probe", {})
    if name in burst:
        out["burst_auroc"] = burst[name]
    return out


def _tier(name: str, metrics: dict[str, Any], ontology_row: Any) -> tuple[str, str]:
    """Apply the fixed tier rules; returns (tier, basis)."""
    artifact_domain_claims = [
        c
        for c in ontology_row.validated
        if c.evidence.startswith("artifacts/")
        and c.domain not in {"generic_series", "generic_tabular"}
    ]
    if artifact_domain_claims:
        c = artifact_domain_claims[0]
        return "T4", f"domain artifact: {c.evidence} ({c.domain})"
    if name in MEASURED_BROKEN:
        return "T0", "constant output on declared modality (measured); input contract fix pending"
    mean = metrics.get("adbench_mean_auroc") or metrics.get("nab_mean_auroc")
    if mean is not None:
        n_scored = metrics.get("n_datasets_scored")
        if n_scored == 1:
            return "T2", f"single-dataset grade ({mean}); tier capped pending full grading"
        if mean >= 0.75:
            return "T3", f"real-data mean AUROC {mean}"
        if mean >= 0.60:
            return "T2", f"real-data mean AUROC {mean}"
        if name in PROTOCOL_MISMATCH:
            return "T1", f"graded {mean} under a mismatched point-wise protocol"
        return "T1", f"~chance on real data ({mean})"
    test_claims = [c for c in ontology_row.validated if c.evidence.startswith("tests/")]
    if test_claims:
        return "T2", f"driven on real inputs by {test_claims[0].evidence}; no benchmark metric yet"
    burst = metrics.get("burst_auroc")
    if burst is not None and burst >= 0.95:
        return "T1", f"window-level burst separation {burst}; no per-point grade"
    if burst is not None and burst <= 0.65:
        return "T0", f"fails the planted-burst floor ({burst})"
    return "T1", "runs; no committed grade on labeled data"


def build_ledger() -> dict[str, Any]:
    """Assemble the full ledger from committed sources only."""
    seed = json.loads(SEED_PATH.read_text())
    cards: dict[str, Any] = {}
    for entry in DETECTOR_MANIFEST:
        row = ONTOLOGY[entry.name]
        metrics = _metrics_for(entry.name, seed)
        tier, basis = _tier(entry.name, metrics, row)
        cards[entry.name] = {
            "category": entry.category.name,
            "modality": entry.modality,
            "seams": sorted(entry.seams),
            "fit_required": entry.fit_required,
            "external": entry.external,
            "detects": list(row.detects),
            "cannot_detect": list(row.cannot_detect),
            "validated": [asdict(c) for c in row.validated],
            "not_validated_note": row.not_validated_note,
            "metrics": metrics,
            "tier": tier,
            "tier_basis": basis,
        }
    tiers = sorted(c["tier"] for c in cards.values())
    return {
        "schema": "detector_evidence_ledger/v1",
        "generated_from": {
            "seed": SEED_PATH.relative_to(REPO_ROOT).as_posix(),
            "ontology": "src/omni_mercury_engine/core/detector_ontology.py",
            "manifest": "src/omni_mercury_engine/core/detector_registry.py",
        },
        "tier_rules": "computed by scripts/generate_detector_evidence.py; never hand-assigned",
        "tier_counts": {t: tiers.count(t) for t in sorted(set(tiers))},
        "detectors": cards,
    }


def _serialise(ledger: dict[str, Any]) -> str:
    return json.dumps(ledger, indent=1, sort_keys=True, allow_nan=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify the committed ledger")
    args = parser.parse_args(argv)
    text = _serialise(build_ledger())
    if args.check:
        if not LEDGER_PATH.exists() or LEDGER_PATH.read_text() != text:
            print(
                "detector evidence ledger is OUT OF SYNC; regenerate with "
                "python scripts/generate_detector_evidence.py",
                file=sys.stderr,
            )
            return 1
        print("detector evidence ledger is in sync.")
        return 0
    LEDGER_PATH.write_text(text)
    counts = json.loads(text)["tier_counts"]
    print(f"wrote {LEDGER_PATH.relative_to(REPO_ROOT)} | tiers: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
