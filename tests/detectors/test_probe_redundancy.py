# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The measured probe-redundancy reduction, pinned against the recorded corpora.

``research/math_arrest/harness_probe_redundancy.py`` measured the fit-time
correlation structure of the 21 probes on every recorded dataset and swept
cluster-removal thresholds through the shipped fusion. At mean |r| >= 0.80 two
clusters exist — {additive, harmonic_oscillator, r3_recursion_resonance,
quantum_annealing, modified_zscore} and {catalan_optimized,
exponential_decay} — and keeping only each cluster's strongest member measures
better than the full ensemble on both corpora. That selection ships as the
``"decorrelated"`` preset; this module pins it:

* the preset matches the persisted selection in
  ``artifacts/math_arrest_probe_correlation.json``;
* the reduced ensemble's mean AUROC reproduces the measurement on both
  corpora and does not degrade against the full ensemble measured in the
  same run;
* its mean rank over the 19 evaluations is no worse than the full ensemble's.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = [
    pytest.mark.xdist_group("probe_redundancy"),
    pytest.mark.timeout(900),
]

from omni_mercury_engine.detectors.math_arrest.arrest import (
    _PROBE_REGISTRY,
    PROBE_PRESETS,
    AnomalyMathArrest,
)
from tests.recorded_data import ADBENCH_NAMES, adbench, auroc, nab_corpus, stratified_split

ARTIFACT = Path(__file__).resolve().parents[2] / "artifacts" / "math_arrest_probe_correlation.json"
SEEDS = (0, 1, 2)
TOLERANCE = 0.03

#: Measured mean AUROC of the ``"decorrelated"`` preset (NAB, ADBench), from
#: the harness run recorded in the artifact. The full ensemble measured
#: (0.7494, 0.6763) in the same run.
MEASURED_REDUCED = (0.7584, 0.6859)

#: The reduction may not cost more than this per corpus against the full
#: ensemble measured in the same session (it measured as a *gain* on both).
MAX_DEGRADATION = 0.005

#: The five probes the measured selection removes (short names).
REMOVED = frozenset(
    {
        "exponential_decay",
        "harmonic_oscillator",
        "modified_zscore",
        "quantum_annealing",
        "r3_recursion_resonance",
    }
)


def _class_to_short() -> dict[str, str]:
    """Class name -> the short name each probe reports, via the real path."""
    rng = np.random.default_rng(0)
    series = np.sin(np.linspace(0.0, 8.0, 80)) + 0.05 * rng.standard_normal(80)
    mapping: dict[str, str] = {}
    for class_name, probe_cls in _PROBE_REGISTRY.items():
        probe = probe_cls()
        probe.fit_trajectory(series)
        mapping[class_name] = probe.deviation_score(series).probe_name
    return mapping


@pytest.fixture(scope="module")
def evaluations() -> dict[str, dict[str, float]]:
    """AUROC of the full and reduced ensembles on all 19 recorded evaluations."""
    out: dict[str, dict[str, float]] = {"full": {}, "reduced": {}}

    def run(train: np.ndarray, held: np.ndarray, positive: np.ndarray, name: str) -> None:
        for config, probes in (("full", None), ("reduced", "decorrelated")):
            detector = AnomalyMathArrest(probes=probes)
            detector.fit(train)
            fused = np.asarray(detector.detect(held), dtype=float).ravel()
            out[config][name] = auroc(fused[positive], fused[~positive])

    for series in nab_corpus():
        split = series.n_probation
        run(series.values[:split], series.values[split:], series.positive[split:], series.name)
    for table in ADBENCH_NAMES:
        features, labels = adbench(table)
        for seed in SEEDS:
            train, test = stratified_split(labels, seed)
            mean = features[train].mean(axis=0)
            spread = features[train].std(axis=0)
            spread[spread < 1e-8] = 1.0
            run(
                (features[train] - mean) / spread,
                (features[test] - mean) / spread,
                labels[test] == 1,
                f"{table}/seed{seed}",
            )
    return out


def _corpus_means(aucs: dict[str, float]) -> tuple[float, float]:
    nab = [v for k, v in aucs.items() if not k.split("/")[-1].startswith("seed")]
    tables: dict[str, list[float]] = {}
    for k, v in aucs.items():
        if k.split("/")[-1].startswith("seed"):
            tables.setdefault(k.split("/")[0], []).append(v)
    adb = [float(np.mean(v)) for v in tables.values()]
    return float(np.mean(nab)), float(np.mean(adb))


def test_the_preset_matches_the_persisted_selection() -> None:
    """``"decorrelated"`` is exactly the artifact's measured kept set."""
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["schema"] == "mercury.research.math_arrest_probe_correlation/v1"
    assert len(artifact["probe_names"]) == 21
    selected = artifact["selected"]
    assert selected is not None, "the harness found no non-degrading reduction"
    assert set(selected["removed"]) == REMOVED
    mapping = _class_to_short()
    preset_short = {mapping[name] for name in PROBE_PRESETS["decorrelated"]}
    assert preset_short == set(selected["kept"])
    assert len(PROBE_PRESETS["decorrelated"]) == 21 - len(REMOVED)


def test_the_artifact_holds_a_correlation_matrix_per_dataset() -> None:
    artifact = json.loads(ARTIFACT.read_text())
    datasets = artifact["datasets"]
    assert len(datasets) == 19  # 4 NAB + 5 ADBench tables x 3 seeds
    for name, record in datasets.items():
        matrix = np.asarray(record["correlation"], dtype=float)
        assert matrix.shape == (21, 21), name
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-9)
        np.testing.assert_allclose(np.diag(matrix), 1.0)
        assert set(record["per_probe_auc"]) == set(artifact["probe_names"])


def test_the_reduced_ensemble_reproduces_the_measurement(
    evaluations: dict[str, dict[str, float]],
) -> None:
    nab, adb = _corpus_means(evaluations["reduced"])
    assert nab == pytest.approx(MEASURED_REDUCED[0], abs=TOLERANCE)
    assert adb == pytest.approx(MEASURED_REDUCED[1], abs=TOLERANCE)


def test_the_reduction_does_not_degrade_mean_auc(
    evaluations: dict[str, dict[str, float]],
) -> None:
    full_nab, full_adb = _corpus_means(evaluations["full"])
    reduced_nab, reduced_adb = _corpus_means(evaluations["reduced"])
    assert reduced_nab >= full_nab - MAX_DEGRADATION, (
        f"reduced NAB mean AUROC {reduced_nab:.4f} degrades the full "
        f"ensemble's {full_nab:.4f} beyond {MAX_DEGRADATION}"
    )
    assert reduced_adb >= full_adb - MAX_DEGRADATION, (
        f"reduced ADBench mean AUROC {reduced_adb:.4f} degrades the full "
        f"ensemble's {full_adb:.4f} beyond {MAX_DEGRADATION}"
    )


def test_the_reduction_does_not_degrade_mean_rank(
    evaluations: dict[str, dict[str, float]],
) -> None:
    """Average rank on ties, the competitive-benchmark convention."""
    names = sorted(evaluations["full"])
    totals = {"full": 0.0, "reduced": 0.0}
    for name in names:
        full_value = evaluations["full"][name]
        reduced_value = evaluations["reduced"][name]
        if full_value == reduced_value:
            totals["full"] += 1.5
            totals["reduced"] += 1.5
        elif reduced_value > full_value:
            totals["reduced"] += 1.0
            totals["full"] += 2.0
        else:
            totals["reduced"] += 2.0
            totals["full"] += 1.0
    mean_full = totals["full"] / len(names)
    mean_reduced = totals["reduced"] / len(names)
    assert (
        mean_reduced <= mean_full
    ), f"reduced mean rank {mean_reduced:.3f} is worse than full {mean_full:.3f}"
