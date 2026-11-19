# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Leave-one-domain-out verdict on the affinity maps, pinned.

``research/math_arrest/harness_affinity_lodo.py`` measured every released
``DOMAIN_AFFINITY_21`` ordering and a LODO-trained ordering (probes ranked by
their mean per-probe AUROC on the other recorded datasets) under the
``phi_affinity`` weighting, against the global uniform mean the ensemble
ships. Measured: uniform 0.7088; the LODO-trained map 0.6946 (the best any
affinity ordering reaches — it beats every hand-written map); released maps
0.5868–0.6564. The retention rule — keep a map only if it improves the
system's own mean AUROC over the global vector — therefore retains none,
and this module holds that verdict against the recorded corpora.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = [
    pytest.mark.xdist_group("affinity_lodo"),
    pytest.mark.timeout(900),
]

from omni_mercury_engine.detectors.math_arrest.domain_affinity import DOMAIN_AFFINITY_21
from research.math_arrest.harness_affinity_lodo import (
    _table_key,
    fuse_with_order,
    per_probe_auc,
)
from research.math_arrest.harness_probe_redundancy import (
    dataset_auc,
    fuse_subset,
    load_datasets,
)

ARTIFACT = Path(__file__).resolve().parents[2] / "artifacts" / "math_arrest_affinity_lodo.json"

#: Floors under the measured gaps (uniform − candidate): LODO-trained 0.0142,
#: worst released map 0.1220. Each leaves room for BLAS-level drift only.
MIN_UNIFORM_OVER_LODO = 0.005
MIN_UNIFORM_OVER_RELEASED = 0.03


@pytest.fixture(scope="module")
def measured() -> dict[str, float]:
    """Mean AUROC of uniform, the LODO-trained map and each released map."""
    datasets = load_datasets()
    names = datasets[0].probe_names
    probe_aucs = {d.name: per_probe_auc(d) for d in datasets}
    groups = sorted({_table_key(d) for d in datasets})

    def overall(aucs: dict[str, float]) -> float:
        per_group: dict[str, list[float]] = {}
        for d in datasets:
            per_group.setdefault(_table_key(d), []).append(aucs[d.name])
        return float(np.mean([float(np.mean(v)) for v in per_group.values()]))

    out = {
        "uniform_global": overall({d.name: dataset_auc(d, fuse_subset(d, names)) for d in datasets})
    }

    lodo: dict[str, float] = {}
    for group in groups:
        training = [d for d in datasets if _table_key(d) != group]
        mean_auc = {n: float(np.mean([probe_aucs[d.name][n] for d in training])) for n in names}
        ordering = sorted(names, key=lambda n: -mean_auc[n])
        for d in datasets:
            if _table_key(d) == group:
                lodo[d.name] = dataset_auc(d, fuse_with_order(d, ordering))
    out["lodo_trained_phi"] = overall(lodo)

    for domain, ordering in DOMAIN_AFFINITY_21.items():
        out[f"released_{domain}"] = overall(
            {d.name: dataset_auc(d, fuse_with_order(d, list(ordering))) for d in datasets}
        )
    return out


def test_no_affinity_map_beats_the_global_uniform_vector(measured: dict[str, float]) -> None:
    """The retention rule retains nothing: uniform leads every ordering."""
    uniform = measured["uniform_global"]
    lodo_gap = uniform - measured["lodo_trained_phi"]
    assert lodo_gap >= MIN_UNIFORM_OVER_LODO, (
        f"the LODO-trained affinity map is within {lodo_gap:.4f} of the global "
        "uniform vector; if it has genuinely overtaken it, retain it and "
        "re-pin this verdict"
    )
    for domain in DOMAIN_AFFINITY_21:
        gap = uniform - measured[f"released_{domain}"]
        assert gap >= MIN_UNIFORM_OVER_RELEASED, (
            f"released map {domain!r} is within {gap:.4f} of uniform; "
            "re-run the LODO harness and re-pin if this is real"
        )


def test_the_trained_ordering_beats_every_released_map(measured: dict[str, float]) -> None:
    """Mercury's own measured ranking outranks every hand-written ordering.

    This is the constructive half of the finding: the φ^-rank prior loses to
    uniform however it is ordered, but *if* an ordering is wanted, measuring
    it beats writing it by hand on every released map.
    """
    trained = measured["lodo_trained_phi"]
    for domain in DOMAIN_AFFINITY_21:
        assert (
            trained > measured[f"released_{domain}"]
        ), f"hand-written map {domain!r} outranks the measured LODO ordering"


def test_the_artifact_records_the_retention_verdict() -> None:
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["schema"] == "mercury.research.math_arrest_affinity_lodo/v1"
    assert artifact["retained_maps"] == []
    summary = artifact["summary"]
    assert set(summary) >= {"uniform_global", "lodo_trained_phi"}
    for key, cell in summary.items():
        if key != "uniform_global":
            assert cell["improves_on_global"] is False, key
    # One LODO ordering per held-out group, each a permutation of all 21 probes.
    for group, ordering in artifact["lodo_orderings"].items():
        assert len(set(ordering)) == 21, group
