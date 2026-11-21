# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Adaptive 3R recursion depth: the rule, the measurement, the verdict.

``research/omni_equation/harness_adaptive_depth.py`` makes the recursion
depth a function of each window's own spectral entropy (the Resonance
stream's quantity) and measures it against every fixed depth on the recorded
NAB windows. Measured: adaptive (mean depth 3.86) outranks the served fixed
depth 5 (mean rank 3.25 vs 4.88, +0.0017 mean AUROC) at 41% lower R-stream
latency — but fixed depth 3 outranks both, and every difference sits inside
the 0.003 flat band the global sensitivity sweep measured, so the adoption
bar (clear the noise band, not merely edge the served value inside it)
refuses it. λ needs no sweep at all: the fused score is exactly flat in it
(``artifacts/oae_sensitivity.json``), so no λ adaptation can improve a
detection metric by construction.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from research.omni_equation.harness_adaptive_depth import adaptive_depth, spectral_entropy

ARTIFACT = Path(__file__).resolve().parents[2] / "artifacts" / "oae_adaptive_depth.json"


def test_spectral_entropy_separates_structure_from_noise() -> None:
    t = np.linspace(0.0, 16 * np.pi, 512)
    pure_tone = np.sin(t)
    rng = np.random.default_rng(0)
    broadband = rng.standard_normal(512)
    assert spectral_entropy(pure_tone) < 0.4
    assert spectral_entropy(broadband) > 0.8
    assert spectral_entropy(np.ones(128)) == 1.0  # degenerate: no spectrum


def test_the_depth_rule_spans_its_range_and_only_its_range() -> None:
    t = np.linspace(0.0, 16 * np.pi, 512)
    rng = np.random.default_rng(1)
    assert adaptive_depth(np.sin(t)) >= 5  # structured recurses deeper
    assert adaptive_depth(rng.standard_normal(512)) == 3  # broadband stays shallow
    for seed in range(5):
        window = np.random.default_rng(seed).standard_normal(256)
        assert 3 <= adaptive_depth(window) <= 7


def test_the_artifact_records_the_measured_refusal() -> None:
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["schema"] == "mercury.research.oae_adaptive_depth/v1"
    results = artifact["results"]
    # The constructive half: adaptive genuinely outranks the served depth and
    # is cheaper, so the rule works as designed...
    assert results["adaptive"]["mean_rank"] < results["depth_5"]["mean_rank"]
    assert results["adaptive"]["latency_s"] < results["depth_5"]["latency_s"]
    # ...and the honest half: the margin sits inside the measured noise band,
    # so it is not adopted. If adaptive ever clears the band, re-run and
    # re-pin — the adoption decision changes with evidence, not here.
    margin = results["adaptive"]["mean_auc"] - results["depth_5"]["mean_auc"]
    assert margin <= artifact["noise_margin"]
    assert artifact["adopted"] is False
    assert "flat in lambda" in artifact["lambda_verdict"]


def test_the_served_default_depth_is_unchanged() -> None:
    from research.omni_equation.harness_oae_weights import RECURSION_DEPTH

    assert RECURSION_DEPTH == 5
    artifact = json.loads(ARTIFACT.read_text())
    usage = artifact["adaptive_depth_usage"]
    assert 3.0 <= usage["mean_depth"] <= 7.0
    assert set(map(int, usage["depth_histogram"])) <= set(range(3, 8))


def test_the_adoption_bar_is_the_measured_noise_band() -> None:
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["noise_margin"] == pytest.approx(0.005)
