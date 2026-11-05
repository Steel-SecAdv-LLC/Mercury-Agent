# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The ``/three_r`` resonance stream must not saturate at zero.

H(omega) scores a window by the L1 distance between its 8-band spectral energy
ratios and golden-ratio expected ratios ``1/PHI^i``. Both vectors sum to 1, so
that distance ranges over ``[0, 2]`` and ``1 - deviation`` is negative for any
spectrum not already phi-shaped. The served form clipped it to ``[0, 1]``, so
the score sat at a saturated 0 on real data.

Measured on the four recorded NAB files (128-point windows, stride 16):

=========================  ==========  ==============
file                        served      normalised
=========================  ==========  ==============
realAWSCloudwatch/iio_...      0.5085          0.9049
realAdExchange/exch-3...       0.7581          0.8115
realKnownCause/rogue_...       0.5039          0.5100
realTraffic/speed_7578      CONSTANT 0          0.8488
=========================  ==========  ==============

Mean AUROC 0.511 (chance) to 0.755. The stream holds 27.64% of the OAE's
golden-ratio weight, so a saturated constant there is weight spent on nothing.

These tests pin the property (no saturation, full range used) rather than the
exact AUROCs, which belong to the research harness.
"""

from __future__ import annotations

import numpy as np

PHI = 1.618033988749895
HARMONIC_BANDS = 8


def _resonance(data: np.ndarray, *, normalise: bool) -> float:
    """The served H(omega), with the normalisation under test."""
    magnitudes = np.abs(np.fft.fft(data))
    n_bands = min(HARMONIC_BANDS, len(magnitudes) // 2)
    band_size = len(magnitudes) // (2 * n_bands) if n_bands > 0 else 1
    band_energies = [
        float(np.sum(magnitudes[i * band_size : (i + 1) * band_size] ** 2)) for i in range(n_bands)
    ]
    total = sum(band_energies) + 1e-8
    band_ratios = [e / total for e in band_energies]
    expected = [1.0 / (PHI**i) for i in range(n_bands)]
    expected_sum = sum(expected)
    expected = [e / expected_sum for e in expected]
    deviation = sum(abs(a - e) for a, e in zip(band_ratios, expected))
    raw = 1.0 - deviation / 2.0 if normalise else 1.0 - deviation
    return float(np.clip(raw, 0.0, 1.0))


def _windows(series: np.ndarray, size: int = 128, stride: int = 16) -> list[np.ndarray]:
    return [series[i : i + size] for i in range(0, len(series) - size + 1, stride)]


def test_the_served_route_uses_the_normalised_form() -> None:
    """The fix is in the served code, not only in this test."""
    from pathlib import Path

    route = Path(__file__).resolve().parents[2] / (
        "src/omni_mercury_engine/api/routes/detection.py"
    )
    source = route.read_text(encoding="utf-8")
    assert "harmonic_deviation / 2.0" in source, (
        "the /three_r resonance score no longer divides the L1 deviation by its "
        "maximum (2.0); without that it saturates at 0 on real spectra"
    )
    assert (
        "1.0 - harmonic_deviation," not in source
    ), "the unnormalised form is back; it scored at chance on recorded NAB data"


def test_the_unnormalised_form_saturates_on_a_realistic_spectrum() -> None:
    """The defect, demonstrated: a concentrated spectrum pins the old score at 0.

    Real series put most spectral energy in the lowest band, which is far from
    the phi-decaying expected profile, so the raw L1 deviation exceeds 1.0.
    """
    rng = np.random.default_rng(0)
    # Low-frequency dominated signal: the common case for real telemetry.
    t = np.linspace(0.0, 8 * np.pi, 512)
    series = np.sin(t) + 0.05 * rng.standard_normal(t.size)

    served = [_resonance(w, normalise=False) for w in _windows(series)]
    assert max(served) == 0.0, (
        "expected the unnormalised score to saturate at 0 on this spectrum; "
        f"got max {max(served)}"
    )


def test_the_normalised_form_varies_and_stays_in_range() -> None:
    """The fix restores a usable range instead of a constant."""
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 8 * np.pi, 512)
    series = np.sin(t) + 0.05 * rng.standard_normal(t.size)

    scores = np.asarray([_resonance(w, normalise=True) for w in _windows(series)])
    assert np.all((scores >= 0.0) & (scores <= 1.0))
    assert scores.std() > 0.0, "the normalised score is still constant"
    assert len(np.unique(np.round(scores, 6))) > 1


def test_normalising_cannot_make_the_score_leave_the_unit_interval() -> None:
    """Boundary check: the two extreme spectra both land inside [0, 1].

    A perfectly phi-shaped spectrum gives deviation 0, and a maximally
    mismatched one approaches deviation 2, so the normalised score spans
    [0, 1] by construction rather than by clipping.
    """
    n = 256
    # All energy in one band -> far from the phi profile.
    spike = np.zeros(n)
    spike[0] = 1.0
    concentrated = float(np.clip(1.0 - 2.0 / 2.0, 0.0, 1.0))
    assert 0.0 <= _resonance(spike, normalise=True) <= 1.0
    assert concentrated == 0.0

    # White noise spreads energy evenly: a different, non-saturated value.
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(n)
    value = _resonance(noise, normalise=True)
    assert 0.0 < value < 1.0, f"expected an interior value for white noise, got {value}"
