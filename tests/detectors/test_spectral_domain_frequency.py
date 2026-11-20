# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Contract + behavioural tests for ``SpectralDomainFrequency``.

This detector ships registered in ``DETECTOR_MANIFEST`` (BASE) but previously
had no dedicated test. These exercise the real fit -> extract_features -> detect
path on synthetic time-domain signals, plus the BaseDetector contract, so a
shipped detector is not relying on incidental coverage.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.detectors.spectral_domain_frequency import SpectralDomainFrequency


def _signals(
    n: int = 12, t: int = 512, fs: float = 1000.0, freq: float = 50.0, seed: int = 0
) -> np.ndarray[Any, Any]:
    """``(n, t)`` time-domain sinusoids at ``freq`` Hz with light noise."""
    rng = np.random.RandomState(seed)
    tt = np.arange(t) / fs
    base = np.sin(2.0 * np.pi * freq * tt)
    return np.stack([base + rng.normal(0.0, 0.1, t) for _ in range(n)]).astype(np.float64)


def test_is_basedetector_and_instantiable_no_args() -> None:
    """cls() with no args must work — DETECTOR_MANIFEST auto-discovery relies on it."""
    det = SpectralDomainFrequency()
    assert isinstance(det, BaseDetector)


def test_fit_then_extract_features_is_per_sample() -> None:
    X = _signals()
    det = SpectralDomainFrequency().fit(X)
    assert det.is_fitted()
    feats = det.extract_features(X)
    # Per-sample leading dim is the contract _extract_fusion_features requires.
    assert feats.shape[0] == X.shape[0]
    assert feats.ndim == 2


def test_detect_returns_contract_keys() -> None:
    X = _signals()
    out = SpectralDomainFrequency().fit(X).detect(X)
    assert "anomaly_score" in out
    assert "is_anomaly" in out
    assert np.isfinite(float(out["anomaly_score"]))


def test_single_signal_1d_input_is_accepted() -> None:
    X = _signals()
    det = SpectralDomainFrequency().fit(X)
    feats = det.extract_features(_signals(n=1)[0])  # 1-D (T,)
    assert feats.shape[0] == 1


def test_band_scores_respond_to_the_observed_power() -> None:
    """The per-band z-score used to mix a dimensionless ratio with absolute power.

    ``z = (corrected_ratio - ref_mean) / ref_std`` ran whenever the noise
    colour had been estimated, which is the usual case.  ``corrected_ratio`` is
    observed power over reference power over the band's expected share of a
    beta-slope spectrum -- dimensionless -- while ``ref_mean`` and ``ref_std``
    are absolute band powers.  Whenever the band powers were far from 1 the
    subtraction was dominated by ``-ref_mean / ref_std``, ``|z| / 3`` clipped
    to 1.0 for every input, and every resolvable band scored exactly 0.6 --
    including a signal drawn from the same distribution as the reference.
    0.6 is ``0.6 * 1.0 + 0.4 * 0.0``: a saturated z-score and no change point.
    """
    det = SpectralDomainFrequency().fit(_signals())
    assert det._noise_beta != 0.0, "the broken branch was the one gated on an estimated beta"

    # A signal from the reference distribution must not saturate every band.
    same_bands = det.detect(_signals(n=1, seed=1)[0])["influence_vector"].band_scores
    scoring = {k: v for k, v in same_bands.items() if v > 0.0}
    assert scoring, "some band must resolve at this sample rate"
    assert not all(v == pytest.approx(0.6) for v in scoring.values())

    # Three amplitudes, three different verdicts: the score tracks the power.
    scores = [
        det.detect(_signals(n=1, seed=1)[0] * gain)["anomaly_score"] for gain in (0.01, 1.0, 100.0)
    ]
    assert len(set(scores)) == 3
    # A hundredfold amplitude is a genuine saturation, and still bounded.
    assert scores[2] == max(scores) <= 1.0


def test_the_noise_colour_correction_stays_a_reported_diagnostic() -> None:
    """It cannot enter the z-score: a per-band constant cancels in a standardisation."""
    det = SpectralDomainFrequency().fit(_signals())
    results = det.detect(_signals(n=1, seed=2)[0])["band_results"]

    assert results
    for band in results:
        assert np.isfinite(band.corrected_power_ratio)
        if det._noise_beta == 0.0:
            assert band.corrected_power_ratio == band.power_ratio


def test_phase_coupling_responds_to_coupling_known_by_construction() -> None:
    """The replacement for the term that returned exactly 1.0 for every input.

    Squared bicoherence asks whether the phase at ``f1 + f2`` tracks the sum
    of the phases at ``f1`` and ``f2`` across segments. Its direction is
    checked here against signals whose coupling is known: broadband Gaussian
    noise has none, a periodic impulse train phase-locks all its harmonics,
    and an impulse train convolved with a decaying resonance is the textbook
    bearing-fault model.
    """
    det = SpectralDomainFrequency().fit(_signals())
    rng = np.random.default_rng(0)
    size = 16384
    impulses = np.zeros(size)
    impulses[::97] = 1.0
    resonance = np.exp(-np.arange(60) / 12) * np.sin(2 * np.pi * 0.3 * np.arange(60))

    noise_coupling = det._compute_phase_coherence(rng.normal(size=size))
    impulse_coupling = det._compute_phase_coherence(impulses + 0.05 * rng.normal(size=size))
    bearing_coupling = det._compute_phase_coherence(
        np.convolve(impulses, resonance, "same") + 0.05 * rng.normal(size=size)
    )

    assert noise_coupling < 0.02
    assert impulse_coupling > 0.2
    assert bearing_coupling > 0.1
    assert impulse_coupling > bearing_coupling > noise_coupling


def test_phase_coupling_abstains_when_there_are_too_few_segments() -> None:
    """The uncoupled expectation is 1/K, so too few segments is a noise floor.

    Abstaining at 0.0 is neutral in the influence multiplier, which reads the
    value as "how much coupling", not "how much coherence".
    """
    det = SpectralDomainFrequency().fit(_signals())

    assert det._compute_phase_coherence(np.random.default_rng(0).normal(size=100)) == (
        SpectralDomainFrequency.PHASE_COUPLING_NOT_MEASURED
    )
    assert SpectralDomainFrequency.PHASE_COUPLING_NOT_MEASURED == 0.0


def test_detect_emits_no_numerical_warning() -> None:
    """Twelve ``invalid value encountered in divide`` per call came from 0/0 coherence.

    Under ``-W error`` they were exceptions, and ``PhysicsIntegratedDetector``
    caught them and dropped the Oracle without the caller ever seeing why.
    """
    det = SpectralDomainFrequency().fit(_signals())

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = det.detect(_signals(n=1, seed=4)[0])

    assert np.isfinite(float(result["anomaly_score"]))


def _parseval_signal(kind: str) -> np.ndarray:
    rng = np.random.default_rng(1)
    t = np.arange(1000.0)
    return {
        "white": rng.standard_normal(1000),
        "sine": np.sin(2.0 * np.pi * 7.0 * t / 256.0),
        "large_mean": 70.0 + rng.standard_normal(1000),
        "random_walk": np.cumsum(rng.standard_normal(1000)),
        "alternating": np.where(t % 2 == 0, 1.0, -1.0),
        "odd_length": 5.0 + rng.standard_normal(999),
        "shorter_than_window": rng.standard_normal(101),
    }[kind]


@pytest.mark.parametrize(
    "kind",
    [
        "white",
        "sine",
        "large_mean",
        "random_walk",
        "alternating",
        "odd_length",
        "shorter_than_window",
    ],
)
def test_parseval_identity_is_exact_for_any_signal(kind: str) -> None:
    """The windowed DFT's energy equals the windowed frames' energy, exactly.

    The check used to compare ``|X|/N`` squared sums against raw time energy
    with no correction (ratio ~0.19 for every input, so it could never pass). A
    flat one-sided x2 then double-counted the DC bin (ratio up to 1.7 on a
    signal with a mean), and comparing against the WHOLE signal's mean square
    read 1.3 on a random walk. Per frame, with DC once, interior bins twice and
    the Nyquist bin restored, Parseval is an identity for every one of these.
    """
    signal = _parseval_signal(kind)
    oracle = SpectralDomainFrequency({"domain": "medical", "sample_rate": 256.0})
    freq_matrix, _ = oracle._compute_frequency_matrix(signal)
    ratio = oracle._parseval_energy_ratio(signal, freq_matrix)
    assert abs(ratio - 1.0) < 1e-9, f"{kind}: Parseval energy ratio {ratio!r} is not 1.0"
    assert oracle._validate_parseval_energy(signal, freq_matrix) is True


@pytest.mark.parametrize("factor", [2.0, 0.5, 0.19**0.5])
def test_parseval_check_catches_a_mis_scaled_transform(factor: float) -> None:
    """A wrongly scaled magnitude matrix fails the self-check (and only that does)."""
    signal = _parseval_signal("white")
    oracle = SpectralDomainFrequency({"domain": "medical", "sample_rate": 256.0})
    freq_matrix, _ = oracle._compute_frequency_matrix(signal)
    assert oracle._validate_parseval_energy(signal, freq_matrix * factor) is False


def test_parseval_check_rejects_a_matrix_from_another_signal() -> None:
    """A frame-count mismatch is reported as a failure, not silently passed."""
    oracle = SpectralDomainFrequency({"domain": "medical", "sample_rate": 256.0})
    long_matrix, _ = oracle._compute_frequency_matrix(_parseval_signal("white"))
    assert (
        oracle._validate_parseval_energy(_parseval_signal("shorter_than_window"), long_matrix)
        is False
    )


def test_frequency_matrix_covers_the_newest_samples() -> None:
    """Framing must reach the signal tail, not drop the newest win_len/2-1.

    With ``while start + win_len <= n`` and a half-window hop, a spike in the
    final samples fell outside every frame and never reached the spectrum.
    """
    oracle = SpectralDomainFrequency({"domain": "medical", "sample_rate": 64.0, "inner_window": 16})
    # n=23, win_len=16, hop=8: the old stride emits a single frame [0:16], so
    # indices 16..22 are dropped. Put all the energy there (interior to the
    # added tail frame [7:23], away from the Hann-zeroed endpoints).
    signal = np.zeros(23, dtype=np.float64)
    signal[16:22] = [5.0, -5.0, 5.0, -5.0, 5.0, -5.0]
    freq_matrix, _ = oracle._compute_frequency_matrix(signal)
    assert (
        float(np.sum(np.abs(freq_matrix))) > 0.01
    ), "the newest samples did not reach the spectrum; the tail frame is missing"


def test_fisher_aggregate_p_survives_underflow() -> None:
    """``chi2.sf`` keeps a tiny positive p where ``1 - chi2.cdf`` underflows to 0."""
    from scipy.stats import chi2

    p_values = [1e-4] * 8
    log_p_sum = -2.0 * float(np.sum(np.log(np.clip(p_values, 1e-300, 1.0))))
    dof = 2 * len(p_values)
    # The old inline expression underflows to exactly zero...
    assert (1.0 - chi2.cdf(log_p_sum, dof)) == 0.0
    # ...the survival function keeps the tiny positive tail.
    agg = SpectralDomainFrequency._fisher_aggregate_p(p_values)
    assert 0.0 < agg < 1e-12, f"aggregate p should be tiny-positive, got {agg}"
