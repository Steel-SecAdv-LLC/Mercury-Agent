# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Probe-local calibration and ranking on live contributors, measured.

Phase 2 item: "probe/detector-local calibration and ranking only on live
contributors."  The shipped ensemble already carries the pieces — per-probe
normalizer calibration at fit, decorrelation multipliers that down-weight
redundant probes, a domain-affinity rank order — but the rank order is
computed over all 21 probes whether or not the decorrelator has demoted them,
and no measurement existed of (a) assigning ranks among *live* contributors
only, or (b) replacing the fit-scale normalizer with a probe-local ECDF
calibration.  This harness measures both against the shipped default on the
recorded corpora (four NAB files, five ADBench tables x three seeds).

Candidates, all through the shipped ``PhiWeightedFusion`` mathematics:

* ``uniform`` — the shipped default (validated identical to ``detect()``).
* ``phi_affinity_all`` — the released alternative: PHI^-rank over the full
  affinity order.
* ``phi_affinity_live`` — ranking only on live contributors: probes whose
  decorrelation multiplier is at least ``LIVE_FLOOR`` take the top affinity
  ranks; demoted probes follow, in affinity order, after every live one.
* ``uniform_live`` — uniform fusion of the live contributors only (demoted
  probes dropped entirely).
* ``local_rank`` — probe-local calibration: each probe's held-out scores are
  mapped through that probe's own fit-time empirical CDF before the uniform
  fuse, so every probe speaks in the same calibrated units.

Pre-registered adoption bar (the same discipline as the depth and
weight-entropy decisions): a candidate is adopted only if its overall mean
AUC beats the shipped default by more than the 0.003 flat band, neither
corpus mean falls, and no dataset regresses by more than 0.002.  Anything
less is recorded and refused.

Writes ``artifacts/math_arrest_live_ranking.json``.
``tests/detectors/test_live_contributor_ranking.py`` pins the verdict.

Run from the repository root::

    python research/math_arrest/harness_live_contributor_ranking.py
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from omni_mercury_engine.detectors.math_arrest.base_probe import ProbeResult
from omni_mercury_engine.detectors.math_arrest.domain_affinity import get_affinity_order
from omni_mercury_engine.detectors.math_arrest.fusion import (
    CorrelationAwareDecorrelator,
    PhiWeightedFusion,
)
from research.math_arrest.harness_probe_redundancy import (
    FittedDataset,
    dataset_auc,
    fuse_subset,
    load_datasets,
)

ARTIFACT_PATH = _ROOT / "artifacts" / "math_arrest_live_ranking.json"

#: A probe is a *live contributor* when the decorrelator kept its weight at
#: or above this floor (demoted cluster members get 0.5x per cluster step).
LIVE_FLOOR = 0.75

#: Adoption bar: overall mean AUC must beat the shipped default by more than
#: this (the flat band the depth sweep measured), with no corpus mean falling
#: and no dataset regressing by more than REGRESSION_FLOOR.
ADOPTION_BAND = 0.003
REGRESSION_FLOOR = 0.002


def _calibrated_decorrelator(dataset: FittedDataset) -> CorrelationAwareDecorrelator:
    decorrelator = CorrelationAwareDecorrelator()
    decorrelator.calibrate(dataset.fit_matrix, dataset.probe_names, dataset.fit_qualities)
    return decorrelator


def _results(dataset: FittedDataset, matrix: npt.NDArray[np.float64]) -> list[ProbeResult]:
    return [
        ProbeResult(
            probe_name=name,
            deviation_scores=matrix[:, idx],
            confidence=dataset.fit_qualities[name],
            trajectory_fit_quality=dataset.fit_qualities[name],
            anomaly_geometry="",
        )
        for idx, name in enumerate(dataset.probe_names)
    ]


def _live_names(dataset: FittedDataset) -> list[str]:
    multipliers = _calibrated_decorrelator(dataset).weight_multipliers
    return [n for n in dataset.probe_names if multipliers.get(n, 1.0) >= LIVE_FLOOR]


def fuse_phi(dataset: FittedDataset, *, live_first: bool) -> npt.NDArray[np.float64]:
    """PHI^-rank fusion; ``live_first`` reranks live contributors ahead."""
    decorrelator = _calibrated_decorrelator(dataset)
    order = get_affinity_order("default", dataset.probe_names)
    if live_first:
        multipliers = decorrelator.weight_multipliers
        live = [i for i in order if multipliers.get(dataset.probe_names[i], 1.0) >= LIVE_FLOOR]
        demoted = [i for i in order if i not in set(live)]
        order = live + demoted
    fusion = PhiWeightedFusion(n_probes=len(dataset.probe_names), weighting="phi_affinity")
    return fusion.fuse(
        _results(dataset, dataset.held_matrix), affinity_order=order, decorrelator=decorrelator
    )


def fuse_local_rank(dataset: FittedDataset) -> npt.NDArray[np.float64]:
    """Uniform fuse of probe-locally ECDF-calibrated scores."""
    n_fit = dataset.fit_matrix.shape[0]
    calibrated = np.empty_like(dataset.held_matrix)
    for j in range(dataset.held_matrix.shape[1]):
        fit_sorted = np.sort(dataset.fit_matrix[:, j])
        # Right-continuous ECDF of the probe's own fit-time scores: the
        # probe-local calibration, consuming nothing but the probe itself.
        calibrated[:, j] = np.searchsorted(
            fit_sorted, dataset.held_matrix[:, j], side="right"
        ) / float(n_fit)
    decorrelator = _calibrated_decorrelator(dataset)
    fusion = PhiWeightedFusion(n_probes=len(dataset.probe_names), weighting="uniform")
    return fusion.fuse(
        _results(dataset, calibrated),
        affinity_order=get_affinity_order("default", dataset.probe_names),
        decorrelator=decorrelator,
    )


def evaluate(datasets: list[FittedDataset]) -> dict[str, dict[str, float]]:
    """Per-dataset held-out AUROC for every candidate."""
    out: dict[str, dict[str, float]] = {
        "uniform": {},
        "phi_affinity_all": {},
        "phi_affinity_live": {},
        "uniform_live": {},
        "local_rank": {},
    }
    for dataset in datasets:
        shipped = fuse_subset(dataset, dataset.probe_names)
        reference = dataset_auc(dataset, dataset.shipped_fused)
        reconstructed = dataset_auc(dataset, shipped)
        if abs(reference - reconstructed) > 1e-9:
            raise AssertionError(
                f"{dataset.name}: reconstruction drifted from detect() "
                f"({reconstructed} vs {reference})"
            )
        out["uniform"][dataset.name] = reconstructed
        out["phi_affinity_all"][dataset.name] = dataset_auc(
            dataset, fuse_phi(dataset, live_first=False)
        )
        out["phi_affinity_live"][dataset.name] = dataset_auc(
            dataset, fuse_phi(dataset, live_first=True)
        )
        live = _live_names(dataset)
        out["uniform_live"][dataset.name] = (
            dataset_auc(dataset, fuse_subset(dataset, live)) if live else float("nan")
        )
        out["local_rank"][dataset.name] = dataset_auc(dataset, fuse_local_rank(dataset))
    return out


def verdicts(per_dataset: dict[str, dict[str, float]]) -> dict[str, dict[str, object]]:
    """Apply the pre-registered bar to every candidate against ``uniform``."""
    shipped = per_dataset["uniform"]
    results: dict[str, dict[str, object]] = {}
    for candidate, aucs in per_dataset.items():
        if candidate == "uniform":
            continue
        deltas = {name: aucs[name] - shipped[name] for name in shipped}
        mean_delta = float(np.mean(list(deltas.values())))
        worst_name, worst_delta = min(deltas.items(), key=lambda item: item[1])
        adopted = bool(mean_delta > ADOPTION_BAND and worst_delta >= -REGRESSION_FLOOR)
        results[candidate] = {
            "mean_auc": float(np.mean(list(aucs.values()))),
            "mean_delta_vs_shipped": mean_delta,
            "worst_dataset": worst_name,
            "worst_delta": float(worst_delta),
            "adopted": adopted,
        }
    return results


def main() -> int:
    datasets = load_datasets()
    per_dataset = evaluate(datasets)
    decision = verdicts(per_dataset)
    shipped_mean = float(np.mean(list(per_dataset["uniform"].values())))
    artifact = {
        "schema": "mercury.research.math_arrest_live_ranking/v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "protocol": {
            "live_floor": LIVE_FLOOR,
            "adoption_band": ADOPTION_BAND,
            "regression_floor": REGRESSION_FLOOR,
            "n_evaluations": len(datasets),
        },
        "shipped_uniform_mean_auc": shipped_mean,
        "per_dataset": per_dataset,
        "verdicts": decision,
    }
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"shipped uniform mean AUC {shipped_mean:.4f} over {len(datasets)} evaluations")
    for name, verdict in decision.items():
        print(
            f"{name}: mean {verdict['mean_auc']:.4f} "
            f"(delta {verdict['mean_delta_vs_shipped']:+.4f}, "
            f"worst {verdict['worst_dataset']} {verdict['worst_delta']:+.4f}) "
            f"-> {'ADOPT' if verdict['adopted'] else 'refused'}"
        )
    print(f"written: {ARTIFACT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
