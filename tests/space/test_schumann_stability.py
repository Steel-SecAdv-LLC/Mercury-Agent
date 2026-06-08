# Copyright (C) 2025 Steel Security Advisors LLC
"""WS-C: Schumann training-stability regression + root-cause diagnosis tests.

PR #262 recorded the Schumann sub-net as "seed-unstable" (per-seed AUC
``[0.97, 1.0, 0.23]``). This round root-caused that symptom to a **full-batch
optimisation artifact** and fixed it with mini-batch SGD (the default in
``schumann_eval.run_seed``). These tests guard:

1. the logit exposure (``confidence_logits``) leaves inference byte-identical;
2. the stable recipe (mini-batch) does NOT collapse on separable synthetic data;
3. the diagnostic's attribution logic blames the regime, not the objective.

They are deterministic and offline (no NOAA). They do **not** assert the
quarantine is lifted -- it stands on the data blocker (see pre-registration).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from benchmarks.schumann_diagnostic import build_synthetic_dataset, diagnose
from benchmarks.schumann_eval import run_seed
from omni_mercury_engine.space.schumann_resonance import SchumannHarmonicAnalyzer


def test_confidence_logits_matches_forward_confidence() -> None:
    """sigmoid(confidence_logits(x)) == forward(x)[1] exactly -> inference is
    unchanged by exposing the pre-sigmoid logit for stable training."""
    torch.manual_seed(0)
    model = SchumannHarmonicAnalyzer(spectrum_size=512).eval()
    x = torch.randn(7, 1, 512)
    with torch.no_grad():
        _logits, confidence = model(x)
        logit = model.confidence_logits(x)
    assert torch.allclose(torch.sigmoid(logit), confidence, atol=1e-6)


def test_confidence_head_param_names_unchanged() -> None:
    """The refactor must not rename params (checkpoints stay loadable)."""
    model = SchumannHarmonicAnalyzer(spectrum_size=512)
    names = dict(model.named_parameters())
    assert "confidence_head.0.weight" in names
    assert "confidence_head.0.bias" in names


def test_stable_recipe_does_not_collapse() -> None:
    """The fix: mini-batch SGD trains every seed to a non-degenerate solution on
    separable synthetic signal (no AUC < 0.5 sign-inversion)."""
    specs, labels = build_synthetic_dataset(n=300, pos_frac=0.10, seed=0)
    cut = int(len(labels) * 0.7)
    tr, te = np.arange(cut), np.arange(cut, len(labels))
    aucs = [
        run_seed(specs, labels, tr, te, epochs=20, seed=s, regime="minibatch", objective="logits")
        for s in (0, 1, 2)
    ]
    assert min(aucs) > 0.5, f"stable recipe collapsed: {aucs}"
    assert float(np.mean(aucs)) > 0.8, f"stable recipe underperformed: {aucs}"


def test_diagnosis_attributes_collapse_to_regime() -> None:
    """The attribution logic: full-batch collapses, mini-batch does not ->
    root cause is the optimisation regime, not the objective."""
    configs = [
        {"regime": "full_batch", "objective": "sigmoid", "collapse_rate": 0.17},
        {"regime": "full_batch", "objective": "logits", "collapse_rate": 0.17},
        {"regime": "minibatch", "objective": "sigmoid", "collapse_rate": 0.0},
        {"regime": "minibatch", "objective": "logits", "collapse_rate": 0.0},
    ]
    d = diagnose(configs)
    assert d["regime_is_cause"] is True
    assert "REGIME" in d["root_cause"].upper()
    assert "minibatch" in d["fix"].lower()


def test_diagnosis_quarantine_note_is_honest() -> None:
    """The diagnosis must state the quarantine still stands on the data blocker."""
    d = diagnose(
        [
            {"regime": "full_batch", "objective": "sigmoid", "collapse_rate": 0.17},
            {"regime": "minibatch", "objective": "logits", "collapse_rate": 0.0},
        ]
    )
    note = d["quarantine_note"].lower()
    assert "quarantine still stands" in note
    assert "elf" in note or "data" in note
