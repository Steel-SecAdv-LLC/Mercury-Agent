# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The fusion-mode ablation's verdicts, pinned against the artifact and live.

``research/ensembles/harness_fusion_mode_ablation.py`` measured Mercury's
elaborate fusion mechanisms against its own simplest compositions on the five
recorded ADBench tables (three seeds). The verdicts this module holds:

* The trained multi-head-attention fusion earns its complexity: mean AUROC
  0.9535 against 0.9369 for the best untrained composition (the 0.7/0.3
  blend) and 0.9067 for the R/K/IG detector's adaptive weights.
* The co-trained symbolic satisfaction term is within noise on these fully
  labelled subsets: mean AUROC moves at most 0.0024 across
  ``symbolic_weight`` in {0, 0.05, 0.1, 0.25, adaptive, 0.5} — the measured
  answer to "enabled, disabled and fixed intermediate weights".
* The adaptive R/K/IG weights add little over their own simple baselines
  (+0.003 over the plain component average), and two-stream composition
  (fibring φ, 0.7/0.3) beats the plain average of the same streams but every
  mix degrades the stronger stream when the weaker one is far behind.

The live tier re-measures one (table, seed) end to end so the pins cannot
drift from what the code produces.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

pytestmark = [
    pytest.mark.xdist_group("fusion_mode_ablation"),
    pytest.mark.timeout(900),
]

ARTIFACT = Path(__file__).resolve().parents[2] / "artifacts" / "fusion_mode_ablation.json"

TRAINED = (
    "trained_neural_only",
    "trained_symbolic_adaptive",
    "trained_symbolic_0.05",
    "trained_symbolic_0.1",
    "trained_symbolic_0.25",
    "trained_symbolic_0.5",
)
UNTRAINED = (
    "mercury_adaptive",
    "mercury_average",
    "mercury_fixed_403030",
    "mercury_best_component",
    "arrest_uniform",
    "arrest_phi_prior",
    "arrest_best_probe",
    "fibring_phi",
    "blend_070_030",
    "stream_average",
)
#: The symbolic term's measured effect band on these subsets (max - min mean
#: AUROC across the trained family). Measured 0.0024; the pin allows drift to
#: 0.015 before calling the finding broken in either direction.
MAX_SYMBOLIC_SPREAD = 0.015
#: The trained fusion's measured lead over the best untrained composition
#: (0.9535 - 0.9369 = 0.0166); floor leaves room for numerical drift only.
MIN_TRAINED_LEAD = 0.005


def _artifact() -> dict[str, Any]:
    result: dict[str, Any] = json.loads(ARTIFACT.read_text())
    return result


def test_artifact_covers_every_configuration() -> None:
    artifact = _artifact()
    assert artifact["schema"] == "mercury.research.fusion_mode_ablation/v1"
    results = artifact["results"]
    assert set(results) == set(TRAINED) | set(UNTRAINED)
    tables = set(artifact["config"]["tables"])
    for config, cell in results.items():
        assert set(cell["per_table_auc"]) == tables, config
        assert 0.0 <= cell["mean_auc"] <= 1.0, config
        assert 1.0 <= cell["mean_rank"] <= len(results), config


def test_the_trained_fusion_leads_every_simple_composition() -> None:
    """The attention fusion's complexity is bought with measured AUROC."""
    results = _artifact()["results"]
    trained_best = max(results[c]["mean_auc"] for c in TRAINED)
    untrained_best = max(results[c]["mean_auc"] for c in UNTRAINED)
    assert trained_best - untrained_best >= MIN_TRAINED_LEAD, (
        f"trained fusion {trained_best:.4f} no longer leads the best simple "
        f"composition {untrained_best:.4f} by {MIN_TRAINED_LEAD}"
    )


def test_the_symbolic_term_stays_within_its_measured_noise_band() -> None:
    """Enabled, disabled or fixed λ — the term moves mean AUROC marginally.

    If this fails high, the symbolic term has started mattering on fully
    labelled data (re-measure and update the finding); if a single arm
    collapses, the co-training broke that arm.
    """
    results = _artifact()["results"]
    values = [results[c]["mean_auc"] for c in TRAINED]
    assert max(values) - min(values) <= MAX_SYMBOLIC_SPREAD, (
        f"trained-family spread {max(values) - min(values):.4f} exceeds " f"{MAX_SYMBOLIC_SPREAD}"
    )


def test_two_stream_composition_beats_its_own_plain_average() -> None:
    """Fibring φ and the 0.7/0.3 blend outrank the average of the same streams."""
    results = _artifact()["results"]
    average = results["stream_average"]["mean_auc"]
    assert results["blend_070_030"]["mean_auc"] > average
    assert results["fibring_phi"]["mean_auc"] > average


def test_live_remeasure_matches_the_artifact_on_one_split() -> None:
    """One (table, seed) end to end: the artifact numbers are reproducible."""
    from research.ensembles.harness_fusion_mode_ablation import (
        _arrest_scores,
        _trained_probs,
    )
    from tests.recorded_data import adbench, auroc, stratified_split

    artifact = _artifact()
    features, labels = adbench("breastw")
    train, test = stratified_split(labels, 0)
    mean = features[train].mean(axis=0)
    spread = features[train].std(axis=0)
    spread[spread < 1e-8] = 1.0
    x_train = (features[train] - mean) / spread
    x_test = (features[test] - mean) / spread
    held = labels[test] == 1

    neural = _trained_probs(x_train, labels[train], x_test, 0.0, 0)
    neural_auc = auroc(neural[held], neural[~held])
    arrest = _arrest_scores(x_train, labels[train], x_test)
    arrest_auc = auroc(arrest["arrest_uniform"][held], arrest["arrest_uniform"][~held])

    pinned_neural = artifact["results"]["trained_neural_only"]["per_table_auc"]["breastw"]
    pinned_arrest = artifact["results"]["arrest_uniform"]["per_table_auc"]["breastw"]
    # The artifact cell averages three seeds; one seed sits near it.
    assert neural_auc == pytest.approx(pinned_neural, abs=0.05)
    assert arrest_auc == pytest.approx(pinned_arrest, abs=0.05)
    assert neural_auc > arrest_auc, "the trained fusion lost to the plain probe mean live"
