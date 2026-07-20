# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the metric-based clinical signal gate."""

from __future__ import annotations

import numpy as np

from omni_mercury_engine.medical.clinical_metrics import evaluate_clinical_scores
from omni_mercury_engine.medical.clinical_signal_gate import (
    ClinicalSignalGate,
    SignalCriteria,
)


def _signal_report(n: int = 1200, seed: int = 0):  # type: ignore[no-untyped-def]
    """A score with genuine, well-calibrated signal."""
    rng = np.random.RandomState(seed)
    q = rng.beta(2, 2, size=n)
    y = (rng.uniform(size=n) < q).astype(int)
    return evaluate_clinical_scores(y, q, seed=seed)


def _noise_report(n: int = 1200, seed: int = 1):  # type: ignore[no-untyped-def]
    """A score independent of the outcome (an untrained-net stand-in)."""
    rng = np.random.RandomState(seed)
    y = (rng.uniform(size=n) < 0.4).astype(int)
    s = rng.uniform(size=n)
    return evaluate_clinical_scores(y, s, seed=seed)


def test_gate_proves_real_signal() -> None:
    """A calibrated, discriminating score clears the gate."""
    gate = ClinicalSignalGate()
    verdict = gate.evaluate(_signal_report())
    assert verdict.proven is True
    assert verdict.failures == []


def test_gate_rejects_noise() -> None:
    """A no-signal score is refused, and the AUROC criterion is among the failures."""
    gate = ClinicalSignalGate()
    verdict = gate.evaluate(_noise_report())
    assert verdict.proven is False
    assert any("auroc" in f for f in verdict.failures)


def test_gate_rejects_small_cohort() -> None:
    """Even a clean score is refused when the cohort is too small to trust."""
    gate = ClinicalSignalGate(SignalCriteria(min_n=500))
    verdict = gate.evaluate(_signal_report(n=60))
    assert verdict.proven is False
    assert any("min_n" in f for f in verdict.failures)


def test_emergency_sensitivity_floor_enforced() -> None:
    """An emergency criterion with a high sensitivity floor can veto a score."""
    report = _signal_report()
    strict = SignalCriteria(min_sensitivity=0.999)
    verdict = ClinicalSignalGate(strict).evaluate(report)
    assert verdict.proven is False
    assert any("sensitivity" in f for f in verdict.failures)


def test_verdict_is_json_serialisable() -> None:
    """The verdict serialises with its criteria and metrics attached."""
    verdict = ClinicalSignalGate().evaluate(_signal_report())
    d = verdict.to_dict()
    assert set(d) >= {"proven", "reasons", "failures", "criteria", "metrics"}
    assert isinstance(d["reasons"], list)
