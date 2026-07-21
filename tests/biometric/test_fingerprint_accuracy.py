# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Labeled genuine-vs-impostor accuracy floor for fingerprint matching.

The existing fingerprint tests pin unit behaviour (one genuine pair, one
impostor pair, hand-built minutiae). None of them assert an *accuracy* — a
labeled multi-identity genuine/impostor sweep with a floor on the accept
rates. Without that, a silent regression in ``FingerprintMatcher`` (a broken
``_compute_score`` normaliser, an off-by-one in ``_count_matched_minutiae``,
a tolerance drift) could halve real-world matching accuracy while every unit
test stayed green.

This module builds a small deterministic labeled corpus at the minutiae layer
— several distinct identity templates, each probed by partial, translated,
jittered captures (genuine) and by the other identities' templates
(impostor) — runs the real matcher over every pair, and asserts floors on the
True-Accept Rate (TAR), the False-Accept Rate (FAR), and the genuine/impostor
score separation.

Measured baseline (2026-07-21, deterministic seeds, match_threshold=40.0):
TAR = 1.00 (min genuine score 88.9), FAR = 0.00 (max impostor score 12.5),
separation ≈ 84. The floors below sit under those measurements with margin
for environment variance while still failing on any real accuracy loss.
"""

from __future__ import annotations

import numpy as np

from omni_mercury_engine.biometric.fingerprint_recognition import (
    FingerprintFeatures,
    FingerprintMatcher,
    Minutia,
    MinutiaeType,
)

N_IDENTITIES = 5
PROBES_PER_IDENTITY = 3
_MINUTIAE_PER_TEMPLATE = 18
_SPAN = 250.0


def _features_from(minutiae: list[Minutia]) -> FingerprintFeatures:
    return FingerprintFeatures(
        minutiae=minutiae,
        singularities=[],
        orientation_field=np.zeros((4, 4)),
        ridge_frequency=np.zeros((4, 4)),
        quality_map=np.ones((4, 4)),
    )


def _template(seed: int) -> list[Minutia]:
    """A distinct, well-separated identity template."""
    rng = np.random.default_rng(seed)
    return [
        Minutia(
            x=float(rng.uniform(20.0, _SPAN)),
            y=float(rng.uniform(20.0, _SPAN)),
            orientation=float(rng.uniform(0.0, 2 * np.pi)),
            type=MinutiaeType.RIDGE_ENDING,
            quality=1.0,
        )
        for _ in range(_MINUTIAE_PER_TEMPLATE)
    ]


def _genuine_probe(
    base: list[Minutia],
    seed: int,
    *,
    drop: int = 2,
    jitter: float = 4.0,
    ang_jitter: float = 0.1,
    tx: float = 12.0,
    ty: float = -8.0,
) -> list[Minutia]:
    """A realistic genuine capture: partial, translated, and jittered.

    Models a real re-capture of the same finger — a rigid translation plus a
    few dropped minutiae (partial contact) and sub-tolerance position/angle
    noise. All perturbations stay within the matcher's tolerances so a genuine
    probe *should* match.
    """
    rng = np.random.default_rng(seed)
    kept = list(base)
    rng.shuffle(kept)
    kept = kept[: len(base) - drop]
    return [
        Minutia(
            x=m.x + tx + float(rng.uniform(-jitter, jitter)),
            y=m.y + ty + float(rng.uniform(-jitter, jitter)),
            orientation=(m.orientation + float(rng.uniform(-ang_jitter, ang_jitter)))
            % (2 * np.pi),
            type=m.type,
            quality=1.0,
        )
        for m in kept
    ]


def _run_sweep() -> dict[str, float]:
    matcher = FingerprintMatcher()
    templates = {i: _template(1000 + i) for i in range(N_IDENTITIES)}
    gallery = {i: _features_from(templates[i]) for i in range(N_IDENTITIES)}

    genuine_scores: list[float] = []
    impostor_scores: list[float] = []
    genuine_accept = genuine_total = 0
    impostor_accept = impostor_total = 0

    for i in range(N_IDENTITIES):
        for p in range(PROBES_PER_IDENTITY):
            probe = _features_from(
                _genuine_probe(templates[i], seed=5000 + i * 10 + p)
            )
            result = matcher.match(probe, gallery[i])
            genuine_scores.append(result.match_score)
            genuine_total += 1
            genuine_accept += int(result.is_match)
            for j in range(N_IDENTITIES):
                if j == i:
                    continue
                impostor = matcher.match(probe, gallery[j])
                impostor_scores.append(impostor.match_score)
                impostor_total += 1
                impostor_accept += int(impostor.is_match)

    return {
        "tar": genuine_accept / genuine_total,
        "far": impostor_accept / impostor_total,
        "genuine_min": float(np.min(genuine_scores)),
        "impostor_max": float(np.max(impostor_scores)),
        "genuine_mean": float(np.mean(genuine_scores)),
        "impostor_mean": float(np.mean(impostor_scores)),
        "genuine_total": float(genuine_total),
        "impostor_total": float(impostor_total),
    }


class TestFingerprintMatchingAccuracy:
    """Functional accuracy floor — catches silent matching regressions."""

    def test_true_accept_rate_floor(self) -> None:
        m = _run_sweep()
        assert m["genuine_total"] == N_IDENTITIES * PROBES_PER_IDENTITY
        # Measured 1.00; floor at 0.90 fails if even ~2 genuine probes stop
        # matching (a real accuracy regression).
        assert m["tar"] >= 0.90, f"TAR {m['tar']:.3f} below floor 0.90"

    def test_false_accept_rate_ceiling(self) -> None:
        m = _run_sweep()
        # Measured 0.00; a security-relevant ceiling — impostors must not be
        # accepted. Kept at 0.05 for environment variance.
        assert m["far"] <= 0.05, f"FAR {m['far']:.3f} above ceiling 0.05"

    def test_genuine_impostor_separation(self) -> None:
        m = _run_sweep()
        threshold = FingerprintMatcher()._threshold
        # Every genuine pair clears the threshold; no impostor does — the clean
        # separation the measured baseline shows (88.9 vs 12.5).
        assert m["genuine_min"] >= threshold, (
            f"weakest genuine score {m['genuine_min']:.1f} fell below the match "
            f"threshold {threshold}"
        )
        assert m["impostor_max"] < threshold, (
            f"strongest impostor score {m['impostor_max']:.1f} reached the match "
            f"threshold {threshold}"
        )
        # Mean-score margin: a coarse regression signal independent of the
        # threshold. Measured ≈ 84.
        assert (m["genuine_mean"] - m["impostor_mean"]) >= 40.0
