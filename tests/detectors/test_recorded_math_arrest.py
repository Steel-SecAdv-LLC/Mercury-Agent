# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The 21-probe math-arrest ensemble on real, labelled data, measured and pinned.

``AnomalyMathArrest`` is a 21-equation instrument (Hankel trajectories, chaos,
harmonic/wave, robust location) exported from the package root and gated in CI
for its probe count, but until this suite it was never measured on real labels —
so there was no evidence it carried anomaly signal at all, which the
repository's anti-theater rule requires before a mechanism is believed.

Two corpora, both recorded and byte-verified in ``tests/fixtures``:

* the four NAB files (fitted on each file's probationary prefix, scored on the
  rest, AUROC against NAB's own windows) — the instrument's declared modality,
  since every probe is a *series* equation;
* five ADBench tables (stratified 70/30, three seeds, train-standardised) — the
  modality the class docstring claims by calling itself an IsolationForest
  replacement. ``BaseEquationProbe._to_1d`` reduces a matrix to its per-row
  mean, so matrix input is scored on that row summary.

What the measurement found, and what these tests hold:

1. The released ``phi_affinity`` weighting put 76% of the fused score on the
   three top-ranked probes of a hand-written affinity order, which measure
   0.531 / 0.524 / 0.378 on NAB, while the two strongest probes receive ~2%
   between them. The 21-probe fusion therefore scored *below its own best
   member*. The default is now ``uniform`` (the plain mean): +0.13 AUC on both
   corpora. ``phi_affinity`` is retained and still reproduces its released
   numbers exactly.
2. ``SVDProjectionProbe`` recomputed its SVD basis from the data it was
   scoring, so the fitted rank-1 subspace was never used and residuals were
   compared against fit-time statistics from a different basis. It now projects
   onto the fitted basis, directionally.
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.detectors.math_arrest.arrest import AnomalyMathArrest
from omni_mercury_engine.detectors.math_arrest.fusion import WEIGHTINGS, PhiWeightedFusion
from tests.recorded_data import ADBENCH_NAMES, adbench, auroc, nab_corpus, stratified_split

SEEDS = (0, 1, 2)
TOLERANCE = 0.03

#: weighting -> (mean NAB AUROC, mean ADBench AUROC) of the 21-probe "all" preset.
MEASURED: dict[str, tuple[float, float]] = {
    "uniform": (0.7498, 0.6846),
    "confidence": (0.7194, 0.6771),
    "phi_affinity": (0.6183, 0.5558),
}

#: Per-probe mean NAB AUROC, the evidence the weighting finding rests on. Only
#: the entries the finding names are pinned; the rest are free to move.
MEASURED_PROBE_NAB: dict[str, float] = {
    # The two strongest probes, which phi_affinity ranked 7th and lower.
    "WavePropagationProbe": 0.697,
    "SVDProjectionProbe": 0.669,
    # The three phi_affinity ranked first, second and third (76% of the weight).
    "VarianceAdaptedProbe": 0.531,
    "MomentumProbe": 0.524,
    "AdditiveProbe": 0.378,
}


def _nab_auroc(**kwargs: object) -> float:
    scores = []
    for series in nab_corpus():
        values, split = series.values, series.n_probation
        positive = series.positive[split:]
        detector = AnomalyMathArrest(**kwargs)  # type: ignore[arg-type]
        detector.fit(values[:split])
        fused = np.asarray(detector.detect(values[split:]), dtype=float).ravel()
        scores.append(auroc(fused[positive], fused[~positive]))
    return float(np.mean(scores))


def _adbench_auroc(**kwargs: object) -> float:
    per_table = []
    for table in ADBENCH_NAMES:
        features, labels = adbench(table)
        seed_values = []
        for seed in SEEDS:
            train, test = stratified_split(labels, seed)
            mean = features[train].mean(axis=0)
            spread = features[train].std(axis=0)
            spread[spread < 1e-8] = 1.0
            detector = AnomalyMathArrest(**kwargs)  # type: ignore[arg-type]
            detector.fit((features[train] - mean) / spread)
            fused = np.asarray(
                detector.detect((features[test] - mean) / spread), dtype=float
            ).ravel()
            held = labels[test]
            seed_values.append(auroc(fused[held == 1], fused[held == 0]))
        per_table.append(float(np.mean(seed_values)))
    return float(np.mean(per_table))


@pytest.fixture(scope="module")
def measured() -> dict[str, tuple[float, float]]:
    return {w: (_nab_auroc(weighting=w), _adbench_auroc(weighting=w)) for w in MEASURED}


@pytest.mark.parametrize("weighting", sorted(MEASURED))
def test_weighting_matches_the_measurement(
    weighting: str, measured: dict[str, tuple[float, float]]
) -> None:
    """Each weighting reproduces its recorded AUROC on both corpora."""
    pinned_nab, pinned_adbench = MEASURED[weighting]
    got_nab, got_adbench = measured[weighting]
    assert got_nab == pytest.approx(pinned_nab, abs=TOLERANCE), (
        f"{weighting}: NAB AUROC {got_nab:.4f} vs measured {pinned_nab:.4f}; "
        "record the change if it is real"
    )
    assert got_adbench == pytest.approx(pinned_adbench, abs=TOLERANCE), (
        f"{weighting}: ADBench AUROC {got_adbench:.4f} vs measured {pinned_adbench:.4f}; "
        "record the change if it is real"
    )


def test_the_default_weighting_is_the_measured_best(
    measured: dict[str, tuple[float, float]],
) -> None:
    """``uniform`` is the default and beats the released prior on both corpora.

    Not a tie-break: +0.13 AUC on each, far outside the 0.03 tolerance.
    """
    assert AnomalyMathArrest()._fusion.weighting == "uniform"
    assert PhiWeightedFusion().weighting == "uniform"
    for corpus in (0, 1):
        assert measured["uniform"][corpus] > measured["phi_affinity"][corpus] + 0.08
        assert measured["uniform"][corpus] >= measured["confidence"][corpus]


def test_the_released_prior_is_still_reachable_and_unchanged(
    measured: dict[str, tuple[float, float]],
) -> None:
    """``phi_affinity`` reproduces the released behaviour for anyone pinned to it."""
    assert "phi_affinity" in WEIGHTINGS
    assert measured["phi_affinity"][0] == pytest.approx(0.6183, abs=TOLERANCE)


def test_an_unknown_weighting_is_refused() -> None:
    with pytest.raises(ValueError, match="Unknown weighting"):
        PhiWeightedFusion(weighting="golden")
    with pytest.raises(ValueError, match="Unknown weighting"):
        AnomalyMathArrest(weighting="golden")


@pytest.fixture(scope="module")
def probe_nab() -> dict[str, float]:
    out = {}
    for probe in MEASURED_PROBE_NAB:
        out[probe] = _nab_auroc(probes=[probe])
    return out


@pytest.mark.parametrize("probe", sorted(MEASURED_PROBE_NAB))
def test_per_probe_nab_auroc_matches_the_measurement(
    probe: str, probe_nab: dict[str, float]
) -> None:
    assert probe_nab[probe] == pytest.approx(MEASURED_PROBE_NAB[probe], abs=TOLERANCE)


def test_the_affinity_prior_is_inverted_against_measured_signal(
    probe_nab: dict[str, float],
) -> None:
    """The evidence for the default change: the prior's top ranks are its weakest probes.

    ``phi_affinity`` ranks ``variance_adapted``, ``momentum`` and ``additive``
    first (76% of the fused weight); every one of them measures below the two
    probes the prior buries at rank 7 and lower.
    """
    from omni_mercury_engine.detectors.math_arrest.domain_affinity import DOMAIN_AFFINITY_21

    order = DOMAIN_AFFINITY_21["default"]
    assert order[:3] == ["variance_adapted", "momentum", "additive"]
    weights = PhiWeightedFusion(n_probes=21).base_weights
    assert float(weights[:3].sum()) > 0.75

    strongest = min(probe_nab["WavePropagationProbe"], probe_nab["SVDProjectionProbe"])
    for weak in ("VarianceAdaptedProbe", "MomentumProbe", "AdditiveProbe"):
        assert probe_nab[weak] < strongest


def test_svd_probe_scores_against_the_fitted_basis() -> None:
    """The fitted rank-1 subspace is stored and used, not re-derived per batch.

    Scoring used to recompute the SVD of the evaluation data, so the learned
    basis was unused and the anomalies being scored helped define the subspace
    they were measured against.
    """
    from omni_mercury_engine.detectors.math_arrest.probes.svd_projection import (
        SVDProjectionProbe,
    )

    rng = np.random.default_rng(0)
    train = np.sin(np.linspace(0.0, 12.0, 400)) + 0.01 * rng.standard_normal(400)
    probe = SVDProjectionProbe()
    probe.fit_trajectory(train)
    assert probe._basis is not None
    basis = probe._basis.copy()

    # Scoring a different batch must not move the basis.
    other = np.sin(np.linspace(3.0, 15.0, 300)) + 0.01 * rng.standard_normal(300)
    probe.deviation_score(other)
    np.testing.assert_allclose(probe._basis, basis)

    # The residual is the distance to that basis, and it is directional: a
    # window fitting the subspace better than training is not deviant.
    scores = np.asarray(probe.deviation_score(train).deviation_scores, dtype=float)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)
    spiked = train.copy()
    spiked[200:205] += 8.0
    spiked_scores = np.asarray(probe.deviation_score(spiked).deviation_scores, dtype=float)
    assert spiked_scores[200:215].max() > float(np.quantile(scores, 0.99))
