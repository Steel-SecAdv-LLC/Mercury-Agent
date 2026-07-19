# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the read-only σ_Immutable calibration + sweep harness."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from omni_mercury_engine.security import sigma_calibration as sc


def test_temperature_scale_matches_sigmoid() -> None:
    """temperature_scale is exactly sigmoid(logit / T)."""
    logits = np.array([-2.0, 0.0, 2.0, 9.45])
    out = sc.temperature_scale(logits, 2.0)
    expected = 1.0 / (1.0 + np.exp(-logits / 2.0))
    assert np.allclose(out, expected)


def test_frozen_constant_invariant_holds() -> None:
    """T=1 baseline score equals the frozen operational constant, both paths."""
    from omni_mercury_engine.security.sigma_immutable_corpus import load_baseline

    gate = sc.load_frozen_gate()
    inv = sc.baseline_constant_check(gate, load_baseline())
    assert inv["invariant_holds"] is True
    # The operational constant must be exact; the numpy-vs-torch sigmoid path
    # reproduces it only to float precision (~1e-8).
    assert inv["operational_score"] == pytest.approx(sc.SIGMA_FROZEN_CONSTANT, abs=1e-9)
    assert inv["logit_path_score_t1"] == pytest.approx(inv["operational_score"], abs=1e-6)


def test_measurement_is_read_only() -> None:
    """Running the harness does not change the operational gate score."""
    from omni_mercury_engine.security.sigma_immutable_corpus import load_baseline

    gate = sc.load_frozen_gate()
    baseline = load_baseline()
    padded = np.zeros(gate.input_dim, dtype=np.float64)
    padded[: len(baseline.values)] = baseline.values
    before = gate.evaluate(padded)[1]
    _ = sc.build_report(seed=999, n_positive=80, n_negative=80)
    after = gate.evaluate(padded)[1]
    assert before == after == pytest.approx(sc.SIGMA_FROZEN_CONSTANT, abs=1e-9)


def test_gate_discriminates_intact_from_tampered() -> None:
    """The frozen gate's score separates intact (1) from tampered (0)."""
    from omni_mercury_engine.security.sigma_immutable_corpus import (
        build_integrity_samples,
        load_baseline,
    )

    gate = sc.load_frozen_gate()
    x, y = build_integrity_samples(load_baseline(), seed=999, n_positive=200, n_negative=200)
    logits = sc.gate_logits(gate, x)
    point = sc.measure_at(logits, y, temperature=1.0, threshold=0.93)
    assert point.auroc > 0.6
    assert point.n == 400
    assert 0.0 <= point.ece <= 1.0


def test_temperature_sweep_finds_low_ece_point() -> None:
    """The temperature sweep exposes a T whose ECE is <= the operational ECE."""
    from omni_mercury_engine.security.sigma_immutable_corpus import (
        build_integrity_samples,
        load_baseline,
    )

    gate = sc.load_frozen_gate()
    x, y = build_integrity_samples(load_baseline(), seed=999, n_positive=200, n_negative=200)
    logits = sc.gate_logits(gate, x)
    op = sc.measure_at(logits, y, temperature=1.0, threshold=0.93)
    points, best_t = sc.temperature_sweep(
        logits, y, temperatures=[0.5, 1.0, 2.0, 3.0, 5.0], threshold=0.93
    )
    best_ece = min(p.ece for p in points)
    assert best_ece <= op.ece + 1e-9
    assert best_t > 0.0


def test_build_report_structure() -> None:
    """The report carries the invariant, operational point, and both sweeps."""
    report = sc.build_report(seed=999, n_positive=120, n_negative=120)
    assert report["frozen_constant_invariant"]["invariant_holds"] is True
    assert "operational_point" in report
    assert report["temperature_sweep"]["points"]
    assert report["threshold_sweep_t1"]["points"]
    assert "recommended_advisory" in report
    # The advisory block must not claim to change the operational threshold.
    assert report["operational_threshold"] == 0.93
