# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Leave-one-domain-out validation of the probe affinity maps.

``DOMAIN_AFFINITY_21`` reorders the 21 probes per domain and takes effect
under the ``phi_affinity`` weighting, whose φ^-rank prior puts 76% of the
fused weight on the first three ranks. The maps were hand-written; this
harness is the first out-of-domain measurement of whether *any* affinity
ordering — released or trained on Mercury's own measurements — beats the
global weight vector the ensemble actually ships (the uniform mean).

Protocol, on the 19 recorded evaluations (4 NAB files, 5 ADBench tables x 3
seeds; the ADBench seeds of one table always travel together):

* **LODO-trained map**: hold out one dataset; rank the 21 probes by their
  mean per-probe held-out AUROC measured on the *other* datasets (Mercury's
  own instrument, no external learner); apply that ordering as the
  ``phi_affinity`` prior on the held-out dataset.
* **Released maps**: every ``DOMAIN_AFFINITY_21`` ordering, applied to every
  dataset under ``phi_affinity``.
* **Global reference**: the shipped uniform mean (the weight vector the
  default fusion already uses).

Retention rule (pinned by ``tests/detectors/test_affinity_lodo.py``): an
affinity map is retained for default use only if it improves the system's own
mean AUROC over the global uniform vector. Results go to
``artifacts/math_arrest_affinity_lodo.json``.

Run from the repository root::

    python research/math_arrest/harness_affinity_lodo.py
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from omni_mercury_engine.detectors.math_arrest.base_probe import ProbeResult
from omni_mercury_engine.detectors.math_arrest.domain_affinity import DOMAIN_AFFINITY_21
from omni_mercury_engine.detectors.math_arrest.fusion import (
    CorrelationAwareDecorrelator,
    PhiWeightedFusion,
)
from research.math_arrest.harness_probe_redundancy import (
    FittedDataset,
    _git_commit,
    dataset_auc,
    load_datasets,
)
from tests.recorded_data import auroc

ARTIFACT_PATH = _ROOT / "artifacts" / "math_arrest_affinity_lodo.json"


def fuse_with_order(dataset: FittedDataset, ordering: list[str]) -> npt.NDArray[np.float64]:
    """The shipped ``phi_affinity`` fusion under an explicit probe ordering.

    Mirrors ``AnomalyMathArrest.detect`` with ``weighting="phi_affinity"``:
    results reordered by the affinity ranking, φ^-rank weights modulated by
    probe confidence, decorrelator multipliers applied, renormalised.
    """
    names = dataset.probe_names
    decorrelator = CorrelationAwareDecorrelator()
    decorrelator.calibrate(dataset.fit_matrix, names, {n: dataset.fit_qualities[n] for n in names})
    results = [
        ProbeResult(
            probe_name=name,
            deviation_scores=dataset.held_matrix[:, i],
            confidence=dataset.fit_qualities[name],
            trajectory_fit_quality=dataset.fit_qualities[name],
            anomaly_geometry="",
        )
        for i, name in enumerate(names)
    ]
    rank_of = {name: i for i, name in enumerate(names)}
    affinity_order = [rank_of[n] for n in ordering if n in rank_of]
    fusion = PhiWeightedFusion(n_probes=len(names), weighting="phi_affinity")
    return fusion.fuse(results, affinity_order=affinity_order, decorrelator=decorrelator)


def per_probe_auc(dataset: FittedDataset) -> dict[str, float]:
    positive = dataset.positive[: dataset.held_matrix.shape[0]]
    out: dict[str, float] = {}
    for i, name in enumerate(dataset.probe_names):
        column = dataset.held_matrix[:, i]
        if 0 < positive.sum() < len(positive):
            out[name] = auroc(column[positive], column[~positive])
        else:
            out[name] = 0.5
    return out


def _table_key(dataset: FittedDataset) -> str:
    """ADBench seeds of one table share a leakage group; NAB files stand alone."""
    return dataset.name.split("/")[0] if dataset.corpus == "adbench" else dataset.name


def main() -> int:
    datasets = load_datasets()
    names = datasets[0].probe_names
    probe_aucs = {d.name: per_probe_auc(d) for d in datasets}

    # The global reference: the shipped uniform mean.
    from research.math_arrest.harness_probe_redundancy import fuse_subset

    uniform_auc = {d.name: dataset_auc(d, fuse_subset(d, names)) for d in datasets}

    # LODO-trained ordering per held-out group.
    lodo_auc: dict[str, float] = {}
    lodo_orderings: dict[str, list[str]] = {}
    groups = sorted({_table_key(d) for d in datasets})
    for group in groups:
        training = [d for d in datasets if _table_key(d) != group]
        held = [d for d in datasets if _table_key(d) == group]
        mean_auc = {n: float(np.mean([probe_aucs[d.name][n] for d in training])) for n in names}
        ordering = sorted(names, key=lambda n: -mean_auc[n])
        lodo_orderings[group] = ordering
        for d in held:
            lodo_auc[d.name] = dataset_auc(d, fuse_with_order(d, ordering))

    # Released maps, applied everywhere.
    released_auc: dict[str, dict[str, float]] = {}
    for domain, ordering in DOMAIN_AFFINITY_21.items():
        released_auc[domain] = {
            d.name: dataset_auc(d, fuse_with_order(d, list(ordering))) for d in datasets
        }

    def overall(aucs: dict[str, float]) -> float:
        per_group: dict[str, list[float]] = {}
        for d in datasets:
            per_group.setdefault(_table_key(d), []).append(aucs[d.name])
        return float(np.mean([float(np.mean(v)) for v in per_group.values()]))

    summary: dict[str, Any] = {
        "uniform_global": {"mean_auc": round(overall(uniform_auc), 4)},
        "lodo_trained_phi": {
            "mean_auc": round(overall(lodo_auc), 4),
            "improves_on_global": overall(lodo_auc) > overall(uniform_auc),
        },
    }
    for domain in sorted(released_auc):
        mean_value = overall(released_auc[domain])
        summary[f"released_{domain}"] = {
            "mean_auc": round(mean_value, 4),
            "improves_on_global": mean_value > overall(uniform_auc),
        }
    retained = [k for k, v in summary.items() if v.get("improves_on_global")]

    artifact = {
        "schema": "mercury.research.math_arrest_affinity_lodo/v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "commit": _git_commit(),
        "groups": groups,
        "per_dataset": {
            "uniform_global": {k: round(v, 4) for k, v in uniform_auc.items()},
            "lodo_trained_phi": {k: round(v, 4) for k, v in lodo_auc.items()},
            **{
                f"released_{dom}": {k: round(v, 4) for k, v in aucs.items()}
                for dom, aucs in released_auc.items()
            },
        },
        "lodo_orderings": lodo_orderings,
        "summary": summary,
        "retained_maps": retained,
    }
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")

    print(f"{'configuration':26s} {'mean AUC':>9s}  improves on global?")
    for key, cell in summary.items():
        print(f"{key:26s} {cell['mean_auc']:9.4f}  {cell.get('improves_on_global', '-')}")
    print(f"\nretained maps: {retained or 'none'}")
    print(f"wrote {ARTIFACT_PATH.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
