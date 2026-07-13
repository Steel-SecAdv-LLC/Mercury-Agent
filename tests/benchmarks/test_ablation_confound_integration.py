# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""The confound guard must be wired into BOTH ablation verdicts (WS-B).

Regression for PR #262's spurious +0.48: a KEEP built on a collapsed
(inverted-ranking) baseline arm must be downgraded to a forced QUARANTINE in the
domain-encoder ablation and the neuro-symbolic ablation alike. Offline (no
training): synthetic result dicts in each harness's serialized shape.
"""

from __future__ import annotations

from benchmarks.domain_encoder_ablation import derive_verdict as de_verdict
from benchmarks.neurosymbolic_ablation import derive_verdict as ns_verdict


def test_domain_encoder_confounded_keep_is_downgraded() -> None:
    confounded = [
        {
            "dataset": "imbalanced",
            "fractions": [
                {
                    "fraction": 0.25,
                    "delta_auc": 0.48,
                    "seeds_encoder_wins": 3,
                    "n_seeds": 3,
                    "baseline_seed_aucs": [0.05, 0.03, 0.04],  # inverted baseline
                    "encoder_seed_aucs": [0.53, 0.50, 0.49],
                }
            ],
        }
    ]
    v = de_verdict(confounded)
    assert v["raw_cleared_bar"] is True
    assert v["confound"]["confounded"] is True
    assert v["cleared_bar"] is False
    assert "FORCED QUARANTINE" in v["verdict"]


def test_domain_encoder_clean_subthreshold_is_honest_quarantine() -> None:
    clean = [
        {
            "dataset": "Pima",
            "fractions": [
                {
                    "fraction": 0.25,
                    "delta_auc": 0.004,
                    "seeds_encoder_wins": 1,
                    "n_seeds": 3,
                    "baseline_seed_aucs": [0.58, 0.60, 0.59],
                    "encoder_seed_aucs": [0.60, 0.61, 0.58],
                }
            ],
        }
    ]
    v = de_verdict(clean)
    assert v["confound"]["confounded"] is False
    assert v["cleared_bar"] is False
    assert "does not clear" in v["verdict"]


def _ns_fr(frac, neural_aucs, symbolic_aucs, delta, wins):
    return {
        "fraction": frac,
        "delta_auc_mean": delta,
        "delta_fpr_mean": 0.0,
        "seeds_auc_better": wins,
        "n_seeds": len(neural_aucs),
        "neural": {"aucs": neural_aucs},
        "symbolic": {"aucs": symbolic_aucs},
    }


def test_neurosymbolic_confounded_keep_is_downgraded() -> None:
    results = [
        {
            "dataset": "imbalanced",
            "fractions": [_ns_fr(1.0, [0.04, 0.03, 0.05], [0.55, 0.52, 0.5], 0.5, 3)],
        }
    ]
    v = ns_verdict(results)
    assert v["raw_passed"] is True
    assert v["confound"]["confounded"] is True
    assert v["passed"] is False
    assert "FORCED QUARANTINE" in v["verdict"]


def test_neurosymbolic_clean_is_not_downgraded() -> None:
    results = [
        {
            "dataset": "breastw",
            "fractions": [_ns_fr(1.0, [0.80, 0.81, 0.79], [0.80, 0.80, 0.80], 0.0, 1)],
        }
    ]
    v = ns_verdict(results)
    assert v["confound"]["confounded"] is False
    assert v["passed"] is False  # transparent quarantine (no improvement), not confound
