# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The 21-probe math-arrest ensemble on real, labelled data, measured and pinned.

Two recorded corpora: the four NAB files (fitted on each file's probationary
prefix) and five ADBench tables (stratified 70/30, three seeds). Every probe is
a series equation and ``BaseEquationProbe._to_1d`` reduces a matrix to its
per-row mean, so tabular input is scored on that row summary.

Pins two measured defects, both fixed:

1. ``phi_affinity`` put 76% of the fused weight on three probes measuring
   0.531 / 0.524 / 0.378 on NAB while the two strongest got ~2% between them,
   so the fusion scored below its own best member. Default is now ``uniform``
   (+0.13 AUC on both corpora); ``phi_affinity`` still reproduces exactly.
2. ``SVDProjectionProbe`` re-derived its basis from the data being scored, so
   the fitted subspace was unused. It now projects onto the fitted basis.
"""

from __future__ import annotations

import numpy as np
import pytest

#: The lane runs ``-n 4 --dist=loadgroup --timeout=300`` with
#: ``func_only: False``, so module-scoped fixture setup counts toward a
#: test's budget, and these measurement fixtures are genuinely expensive.
#:
#: Two separate problems, so two separate controls. Ungrouped, the fixtures
#: were recomputed on *every* worker receiving one of this module's tests
#: (measured 4 x ~115s): waste, fixed by the ``xdist_group`` marker, which
#: works only because of the shim in ``tests/conftest.py``
#: (``tests/scripts/test_xdist_group.py`` measures that it does). Grouping
#: does not make the work fast, though -- it fits 21 probes over 4 NAB files
#: and 5 ADBench tables x 3 seeds -- so the budget is a second, independent
#: control. It is not a workaround for the first: dropping it after grouping
#: landed turned Core Tests (3.12) red at >300s on a CI runner while 3.13
#: passed, which is runner variance, not duplication. Measured here at ~40s
#: after the single-fit rework below; the budget covers the tail.
pytestmark = [
    pytest.mark.xdist_group("recorded_math_arrest"),
    pytest.mark.timeout(900),
]

from omni_mercury_engine.detectors.math_arrest.arrest import AnomalyMathArrest
from omni_mercury_engine.detectors.math_arrest.fusion import WEIGHTINGS, PhiWeightedFusion
from tests.recorded_data import ADBENCH_NAMES, adbench, auroc, nab_corpus, stratified_split

SEEDS = (0, 1, 2)
TOLERANCE = 0.03

#: weighting -> (mean NAB AUROC, mean ADBench AUROC) of the 21-probe "all" preset.
#:
#: Re-measured 2026-11-05, when the shared normalizer stopped rescaling every
#: call by the scored array's own 99th percentile and began carrying the
#: fit-time scale (``BaseEquationProbe._normalize_scores``). The previous
#: figures were ``uniform`` (0.7581, 0.6826), ``confidence`` (0.7277, 0.6686)
#: and ``phi_affinity`` (0.6519, 0.5423).
#:
#: The movement is small on ``uniform`` and ``confidence`` and *upward* on
#: ``phi_affinity``, which narrows the gap the default rests on from 0.1062 to
#: 0.0752 on NAB. It stays decisive on both corpora, and the reason for the
#: repair is not a ranking figure at all: the released normalizer left the
#: alarm rate a function of the caller's batch size (0.052 whole-series to
#: 0.461 at 32-point batches, against a cut promising alpha = 0.05).
MEASURED: dict[str, tuple[float, float]] = {
    "uniform": (0.7494, 0.6763),
    "confidence": (0.7201, 0.6578),
    "phi_affinity": (0.6742, 0.5364),
}

#: Per-probe mean NAB AUROC, the evidence the weighting finding rests on. Only
#: the entries the finding names are pinned; the rest are free to move.
MEASURED_PROBE_NAB: dict[str, float] = {
    # The two strongest probes, which phi_affinity ranked 7th and lower.
    "WavePropagationProbe": 0.697,
    "SVDProjectionProbe": 0.669,
    # The three phi_affinity ranked first, second and third (76% of the weight).
    # Only the first two are below the strongest probes; AdditiveProbe measured
    # 0.378 until its fit-time intercept was found not to transfer to a scored
    # window (see the probe's deviation_score) and now measures 0.685.
    "VarianceAdaptedProbe": 0.531,
    "MomentumProbe": 0.524,
    "AdditiveProbe": 0.685,
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


def _all_weightings_nab() -> dict[str, float]:
    """Mean NAB AUROC for every weighting, fitting each series once.

    ``weighting`` only configures ``AnomalyMathArrest._fusion``, which is read
    at detect time; ``fit`` never consults it. So one fit serves all three,
    and swapping the fusion afterwards reproduces the per-weighting numbers
    bit-identically -- verified to 1e-12 against fitting separately.
    """
    scores: dict[str, list[float]] = {w: [] for w in MEASURED}
    for series in nab_corpus():
        split = series.n_probation
        positive = series.positive[split:]
        detector = AnomalyMathArrest()
        detector.fit(series.values[:split])
        for weighting in MEASURED:
            detector._fusion = PhiWeightedFusion(
                n_probes=len(detector._probes), weighting=weighting
            )
            fused = np.asarray(detector.detect(series.values[split:]), dtype=float).ravel()
            scores[weighting].append(auroc(fused[positive], fused[~positive]))
    return {w: float(np.mean(v)) for w, v in scores.items()}


def _all_weightings_adbench() -> dict[str, float]:
    """Mean ADBench AUROC for every weighting, fitting each split once.

    Same single-fit argument as :func:`_all_weightings_nab`. This is where the
    cost is -- 5 tables x 3 seeds x 3 weightings was 45 fits where 15 do --
    and the rework measured 79.4s to 30.9s with identical numbers.
    """
    per_table: dict[str, list[float]] = {w: [] for w in MEASURED}
    for table in ADBENCH_NAMES:
        features, labels = adbench(table)
        per_seed: dict[str, list[float]] = {w: [] for w in MEASURED}
        for seed in SEEDS:
            train, test = stratified_split(labels, seed)
            mean = features[train].mean(axis=0)
            spread = features[train].std(axis=0)
            spread[spread < 1e-8] = 1.0
            detector = AnomalyMathArrest()
            detector.fit((features[train] - mean) / spread)
            held = labels[test]
            for weighting in MEASURED:
                detector._fusion = PhiWeightedFusion(
                    n_probes=len(detector._probes), weighting=weighting
                )
                fused = np.asarray(
                    detector.detect((features[test] - mean) / spread), dtype=float
                ).ravel()
                per_seed[weighting].append(auroc(fused[held == 1], fused[held == 0]))
        for weighting in MEASURED:
            per_table[weighting].append(float(np.mean(per_seed[weighting])))
    return {w: float(np.mean(v)) for w, v in per_table.items()}


@pytest.fixture(scope="module")
def measured() -> dict[str, tuple[float, float]]:
    nab = _all_weightings_nab()
    adb = _all_weightings_adbench()
    return {w: (nab[w], adb[w]) for w in MEASURED}


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

    Not a tie-break: +0.0751 NAB and +0.1399 ADBench, both far outside the
    0.03 pin tolerance.

    This assertion used to be one global ``+ 0.08`` applied to both corpora.
    That value is **unsatisfiable on the true measurement**: the NAB gap is
    0.0751, so the floor sat above the quantity it was gating, and it only
    ever passed because the released normalizer's per-call p99 inflated
    ``uniform``. Replacing the defect with the fit-time scale on 2026-11-05
    moved ``uniform`` down 0.009 and ``phi_affinity`` up 0.022 on NAB,
    narrowing the real gap from 0.1062 to 0.0751.

    One global floor is also the wrong shape, because it is pinned to
    whichever corpus is tightest and leaves the other unguarded: at 0.05 the
    ADBench side carried 0.0899 of unused slack and would have accepted a
    0.09 collapse in silence. The floors below are therefore **per corpus**
    and set just under each measured gap, which the fixture supports because
    both corpora are recorded and the sweep is deterministic -- two
    consecutive runs agree to every printed digit (0.075126 / 0.139926).
    Each floor keeps roughly 0.005 for cross-platform BLAS variation and
    nothing more, so a real convergence of the two weightings fails here
    instead of being absorbed.
    """
    assert AnomalyMathArrest()._fusion.weighting == "uniform"
    assert PhiWeightedFusion().weighting == "uniform"
    for corpus, name in ((0, "NAB"), (1, "ADBench")):
        phi_gap = measured["uniform"][corpus] - measured["phi_affinity"][corpus]
        assert phi_gap >= MIN_UNIFORM_OVER_PHI[corpus], (
            f"{name}: uniform leads phi_affinity by {phi_gap:.4f}, under the "
            f"{MIN_UNIFORM_OVER_PHI[corpus]:.3f} floor"
        )
        conf_gap = measured["uniform"][corpus] - measured["confidence"][corpus]
        assert conf_gap >= MIN_UNIFORM_OVER_CONFIDENCE[corpus], (
            f"{name}: uniform leads confidence by {conf_gap:.4f}, under the "
            f"{MIN_UNIFORM_OVER_CONFIDENCE[corpus]:.3f} floor"
        )


def test_the_released_prior_is_still_reachable_and_unchanged(
    measured: dict[str, tuple[float, float]],
) -> None:
    """``phi_affinity`` reproduces the released behaviour for anyone pinned to it."""
    assert "phi_affinity" in WEIGHTINGS
    assert measured["phi_affinity"][0] == pytest.approx(0.6519, abs=TOLERANCE)


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
    # Ranks 1 and 2 only. Rank 3 (additive) measured 0.378 when this finding was
    # first recorded and supported it; repairing that probe took it to 0.685,
    # above both strong probes, so it no longer does. The inversion still holds
    # on the two ranks carrying the most weight.
    for weak in ("VarianceAdaptedProbe", "MomentumProbe"):
        assert probe_nab[weak] < strongest
    assert probe_nab["AdditiveProbe"] > strongest, (
        "additive is no longer a weak probe; if it regresses below the strong "
        "ones again, check that its level is still re-anchored per window"
    )


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


#: NAB file -> effective independent dimensions the decorrelator measures for
#: the 21-probe ensemble, fitted on that file's probationary prefix. A probe
#: count is not an evidence count: the ensemble is named for 21 equations but
#: measures 11.64-16.00 effective dimensions on real streams, and the
#: decorrelator down-weights the duplicates (one pair reaches r=0.9976).
MEASURED_EFFECTIVE_DIMS: dict[str, tuple[float, int]] = {
    "realAWSCloudwatch/iio_us-east-1_i-a2eb1cd9_NetworkIn.csv": (12.90, 22),
    "realAdExchange/exchange-3_cpc_results.csv": (11.64, 23),
    "realKnownCause/rogue_agent_key_hold.csv": (14.52, 19),
    "realTraffic/speed_7578.csv": (16.00, 13),
}


@pytest.fixture(scope="module")
def effective_dims() -> dict[str, tuple[float, int]]:
    out: dict[str, tuple[float, int]] = {}
    for series in nab_corpus():
        detector = AnomalyMathArrest()
        detector.fit(series.values[: series.n_probation])
        decorrelator = detector._decorrelator
        out[series.name] = (
            float(decorrelator.effective_probe_count),
            len(decorrelator.redundant_pairs),
        )
    return out


def test_effective_dimensionality_is_measured_not_assumed(
    effective_dims: dict[str, tuple[float, int]],
) -> None:
    """The 21-probe ensemble's real width, pinned per recorded file.

    Guards the claim in both directions: the count never silently reaches 21
    (which would mean the decorrelator stopped finding the duplicates that are
    demonstrably there), and never collapses below the measured floor.
    """
    for name, (pinned_dims, pinned_pairs) in MEASURED_EFFECTIVE_DIMS.items():
        got_dims, got_pairs = effective_dims[name]
        assert got_dims == pytest.approx(pinned_dims, abs=0.6), (
            f"{name}: effective dimensions {got_dims:.2f} vs measured "
            f"{pinned_dims:.2f}; record the change if it is real"
        )
        assert (
            abs(got_pairs - pinned_pairs) <= 3
        ), f"{name}: {got_pairs} redundant pairs vs measured {pinned_pairs}"
        assert got_dims < 21.0, f"{name}: 21 probes are not 21 independent dimensions"


def test_the_decorrelator_applies_under_every_weighting() -> None:
    """Changing the default weighting must not bypass the decorrelation.

    ``uniform`` builds its weight vector from scratch rather than from the
    ``phi_affinity`` prior, so this pins that the multipliers are still
    applied afterwards: a down-weighted duplicate stays down-weighted.
    """
    from omni_mercury_engine.detectors.math_arrest.base_probe import ProbeResult
    from omni_mercury_engine.detectors.math_arrest.fusion import (
        CorrelationAwareDecorrelator,
        PhiWeightedFusion,
    )

    # Two probes: "solo" reads 1.0 everywhere, "dupe" reads 0.0 everywhere.
    results = [
        ProbeResult(
            probe_name="solo",
            deviation_scores=np.ones(8),
            confidence=1.0,
            trajectory_fit_quality=1.0,
            anomaly_geometry="point",
        ),
        ProbeResult(
            probe_name="dupe",
            deviation_scores=np.zeros(8),
            confidence=1.0,
            trajectory_fit_quality=1.0,
            anomaly_geometry="point",
        ),
    ]
    decorrelator = CorrelationAwareDecorrelator()
    decorrelator._weight_multipliers = {"solo": 1.0, "dupe": 0.25}
    decorrelator._is_calibrated = True

    for weighting in WEIGHTINGS:
        fusion = PhiWeightedFusion(n_probes=2, weighting=weighting)
        plain = float(fusion.fuse(results)[0])
        decorrelated = float(fusion.fuse(results, decorrelator=decorrelator)[0])
        assert decorrelated > plain, (
            f"{weighting}: the decorrelator did not shift the fused score, so "
            "its multipliers are being ignored in this mode"
        )
    # And the arithmetic is exactly the renormalised weighting, under uniform.
    uniform = PhiWeightedFusion(n_probes=2, weighting="uniform")
    assert float(uniform.fuse(results, decorrelator=decorrelator)[0]) == pytest.approx(1.0 / 1.25)


def test_the_detector_seam_serves_the_instrument() -> None:
    """The ``BaseDetector`` adapter the engine, registry and console reach.

    Before this adapter the instrument was importable but unreachable: its
    ``detect`` returned a bare array, satisfying no detector contract
    (``docs/DORMANCY_LEDGER.md`` 2b).
    """
    from omni_mercury_engine.detectors import MathArrestDetector

    rng = np.random.default_rng(7)
    train = np.sin(np.linspace(0.0, 40.0, 400)) + 0.05 * rng.standard_normal(400)
    detector = MathArrestDetector().fit(train)

    # The alarm is the fit-time conformal cut, not a bare 0.5 on a
    # batch-relative score.
    assert 0.0 < detector.threshold < 1.0
    assert detector.threshold != pytest.approx(0.5)

    result = detector.detect(train)
    assert result["scores"].shape == (400,)
    assert len(result["probe_names"]) == 21
    assert set(result["probe_scores"]) == set(result["probe_names"])
    # Honest ensemble width, not the probe count.
    assert 0.0 < result["effective_probe_count"] < 21.0
    assert result["redundant_probe_pairs"] > 0
    # Conformal evidence is present because a fit reference exists.
    assert result["p_values"].shape == (400,)
    assert 0.0 <= result["fdr_level"] <= 1.0

    # Features are the interpretable per-probe breakdown.
    assert detector.extract_features(train).shape[1] == 21

    # A clear spike is caught.
    spiked = train.copy()
    spiked[200:206] += 9.0
    assert bool(detector.detect(spiked)["is_anomaly"][200:215].any())


#: Caller batch sizes the alarm rate must not depend on, and the spread it may
#: not exceed across them. Measured on 7dd10e16, before the normalizer carried
#: the fit-time scale, the same sweep ran 0.0520 -> 0.4612 on
#: ``realAWSCloudwatch`` -- a spread of 0.41 against a cut promising 0.05.
#: Per-corpus floors on ``uniform``'s lead over the other two weightings,
#: indexed (NAB, ADBench) to match ``MEASURED``. Set just under the measured
#: gaps -- uniform over phi_affinity 0.075126 / 0.139926, uniform over
#: confidence 0.029283 / 0.018507 -- leaving only enough room for
#: cross-platform floating-point variation. ``uniform`` over ``confidence``
#: was previously a bare ``>=``, which would have accepted the entire 0.029
#: NAB lead eroding to zero without failing.
MIN_UNIFORM_OVER_PHI = (0.070, 0.134)
MIN_UNIFORM_OVER_CONFIDENCE = (0.024, 0.013)

BATCH_SIZES = (None, 256, 128, 64, 32)
MAX_ALARM_RATE_SPREAD = 0.02


@pytest.mark.parametrize("file_index", (0, 1))
def test_the_alarm_rate_does_not_depend_on_the_callers_batch_size(file_index: int) -> None:
    """The conformal promise is about the data, not about how it is chunked.

    ``_normalize_scores`` divided by the 99th percentile of the array being
    scored, so a quiet batch's deviations were stretched to fill ``[0, 1]``
    exactly as a violent batch's were. Within one call that is a monotone
    transform and changes no ranking, which is why every AUROC measurement in
    this module was blind to it; across calls it meant a score had no fixed
    meaning, and the alarm rate -- which compares an absolute score against a
    fixed cut -- tracked the caller's chunking instead of the data.

    Measured before the repair, on ``realAWSCloudwatch``: 0.0520 scoring the
    held-out series whole, then 0.1769 / 0.2176 / 0.3037 / 0.4612 at 256 / 128
    / 64 / 32-point batches. A streaming caller received a 46% false-alarm rate
    from a detector promising 5%.

    This is the pin for the property, not for the numbers: the rate must be
    stable across chunkings and must honour alpha. It is the regression guard
    the repair needs, because nothing else in the suite can see the defect.
    """
    from omni_mercury_engine.detectors.math_arrest.detector import MathArrestDetector

    series = nab_corpus()[file_index]
    split = series.n_probation
    detector = MathArrestDetector()
    detector.fit(series.values[:split])
    held_out = series.values[split:]

    rates = []
    for batch in BATCH_SIZES:
        size = len(held_out) if batch is None else batch
        flags = [
            np.asarray(detector.detect(held_out[i : i + size])["is_anomaly"], dtype=bool)
            for i in range(0, len(held_out), size)
            if len(held_out[i : i + size]) >= 8
        ]
        rates.append(float(np.concatenate(flags).mean()))

    spread = max(rates) - min(rates)
    assert spread <= MAX_ALARM_RATE_SPREAD, (
        f"{series.name}: alarm rate spans {spread:.4f} across batch sizes "
        f"{BATCH_SIZES} ({[round(r, 4) for r in rates]}); the conformal cut "
        "must bound the rate regardless of how the caller chunks its input"
    )
    assert max(rates) <= MathArrestDetector.CALIBRATION_ALPHA, (
        f"{series.name}: worst batched alarm rate {max(rates):.4f} exceeds the "
        f"promised alpha {MathArrestDetector.CALIBRATION_ALPHA}"
    )


#: Probes whose scores for a window do not depend on what else was in the call.
#:
#: Measured 2026-11-05, after the normalizer began carrying the fit-time scale:
#: 7 of the 21 are now exactly batch-invariant. The other 14 are not, and that
#: is a different thing from the defect the normalizer carried.
#:
#: The normalizer's batch dependence was **spurious**: dividing by the scored
#: array's own 99th percentile is a pure rescaling with no modelling content,
#: and it applied to all 21. What remains is **intrinsic** to what the
#: remaining probes measure -- a Lyapunov exponent, a harmonic fit, a fractal
#: dimension or a re-anchored trend over a 64-point window is genuinely a
#: different quantity from the same statistic over a 1000-point series, and
#: ``AdditiveProbe``'s per-window re-anchoring is a documented, deliberate
#: repair (see its ``deviation_score``). Measured residuals, alone versus
#: in-context on a 64-point window: lyapunov_chaos 0.965, harmonic_oscillator
#: 0.911, additive 0.900, fractal_self_similarity 0.884, variance_adapted
#: 0.339, helix_multiplicative 0.226, then nine probes below 0.08.
#:
#: So the detector is *not* claimed to be score-invariant across batchings.
#: What is claimed, and pinned above, is that its **alarm rate** is bounded and
#: stable across them, which is what the conformal cut promises. This set is
#: the regression guard for the part the repair did make exact.
BATCH_INVARIANT_PROBES = frozenset(
    {
        "zeta_harmonic",
        "r3_recursion_resonance",
        "quantum_superposition",
        "quantum_annealing",
        "modified_zscore",
        "iqr_robust",
        "ethical_constrained",
    }
)


def test_the_batch_invariant_probes_stay_exactly_batch_invariant() -> None:
    """The part of the repair that is exact, pinned as exact.

    Before the normalizer carried the fit-time scale, **no** probe was
    batch-invariant: every one was rescaled by the 99th percentile of whatever
    array it was handed, so the same window scored alone and in context
    differed by up to 0.2288 on the fused score, and the installed cut then
    flagged 28.1% of that window's points against 1.6% in context.

    These seven are now bit-identical either way. If one leaves this set, a
    window-local statistic has been introduced into a probe that did not have
    one, which is the change worth catching.
    """
    series = nab_corpus()[0]
    split = series.n_probation
    arrest = AnomalyMathArrest(weighting="uniform")
    arrest.fit(series.values[:split])
    held_out = series.values[split:]

    width, index = 64, 3
    start = index * width
    in_context = arrest.per_probe_scores(held_out)
    alone = arrest.per_probe_scores(held_out[start : start + width])

    measured_invariant = set()
    for name, context_column in in_context.items():
        if name not in alone:
            continue
        context_slice = np.asarray(context_column, dtype=float)[start : start + width]
        alone_slice = np.asarray(alone[name], dtype=float)[:width]
        count = min(len(context_slice), len(alone_slice))
        if count == 0:
            continue
        if np.abs(context_slice[:count] - alone_slice[:count]).max() <= 1e-9:
            measured_invariant.add(name)

    assert measured_invariant == set(BATCH_INVARIANT_PROBES), (
        "the set of batch-invariant probes changed: gained "
        f"{sorted(measured_invariant - BATCH_INVARIANT_PROBES)}, lost "
        f"{sorted(set(BATCH_INVARIANT_PROBES) - measured_invariant)}. A probe "
        "leaving the set has acquired a window-local statistic; one joining it "
        "has lost one. Either is a real change -- record it."
    )
