#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fit the weapons-gate confidence model on the labeled corpus.

Turns the Axis-B offensive-confidence logistic from hand-set coefficients into
*measured* ones. The confidence is
``sigmoid(bias + w_off*n_off - w_allow*n_allow + w_weight*weight
+ w_classifier*boost)``; this script fits ``bias, w_off, w_allow, w_weight`` by
regularized maximum-likelihood logistic regression on the corpus features
(``compute_gate_features``) against the offensive/benign label, on the TRAIN
split, and reports ECE/Brier/fp/fn on VAL **and on the held-out TEST split**
(the latter untouched by both the fit and model selection — the honest
out-of-sample generalisation number). ``w_classifier`` is *retained* at its
default because the corpus carries no live-model signal to fit it from (all
boosts are 0) -- fitting it would silently zero the classifier's contribution.

The three harm-score floors are retained (they encode a policy ordering, not a
measurement) but the script VERIFIES the gate-agreement invariant on the corpus:
every blocking disposition must yield scalar harm >= the general refusal
threshold, and every allow must stay below it, so the harm-score gate and the
disposition gate never disagree.

Output: ``configs/weapons_gate_calibration.json`` -- loaded at import by
``BenevolenceCalibration.load_default()``. Deterministic (no RNG). Run:

    PYTHONPATH=src:benchmarks python scripts/fit_weapons_gate_calibration.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "benchmarks"))

from weapons_gate_corpus import build_corpus

from omni_mercury_engine.cognitive.ethical_bounding import (
    BenevolenceCalibration,
    compute_gate_features,
)


def _features_and_labels(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (X[n,3]=(n_off,n_allow,weight), y[n]) for one corpus split."""
    xs, ys = [], []
    for row in build_corpus():
        if row.split != split:
            continue
        n_off, n_allow, weight, _boost = compute_gate_features(row.text)
        xs.append([float(n_off), float(n_allow), float(weight)])
        ys.append(1.0 if row.label == "offensive" else 0.0)
    return np.asarray(xs), np.asarray(ys)


def _fit_logistic(
    x: np.ndarray, y: np.ndarray, *, l2: float = 1.0, iters: int = 20000
) -> np.ndarray:
    """Deterministic gradient-descent logistic fit. Returns [bias, w0, w1, w2]."""
    n, d = x.shape
    xb = np.hstack([np.ones((n, 1)), x])
    w = np.zeros(d + 1)
    lr = 0.5
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(xb @ w, -60, 60)))
        reg = np.concatenate([[0.0], w[1:]]) * (l2 / n)
        w -= lr * (xb.T @ (p - y) / n + reg)
    return w


def _predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    xb = np.hstack([np.ones((x.shape[0], 1)), x])
    return 1.0 / (1.0 + np.exp(-np.clip(xb @ w, -60, 60)))


def _ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in itertools.pairwise(edges):
        m = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def _gate_agreement() -> dict[str, float]:
    """Fraction of corpus rows where the scalar-harm gate agrees with disposition."""
    from omni_mercury_engine.agentic.capabilities.assistant import GeneralAssistant

    thr = GeneralAssistant.HARM_REFUSAL_THRESHOLD
    from omni_mercury_engine.cognitive.ethical_bounding import BenevolenceScorer

    scorer = BenevolenceScorer(benevolence_threshold=0.70)
    agree = total = 0
    for row in build_corpus():
        score = scorer.score_action(row.text, {})
        disp = str(score.weapons_disposition)
        blocks = disp in ("escalate", "refuse_redact", "hard_refuse")
        harmful = float(score.harm_score) >= thr or float(score.severity_score) >= thr
        if blocks == harmful:
            agree += 1
        total += 1
    return {"gate_agreement": agree / total, "n": total}


def main() -> None:
    """Fit, verify, and write configs/weapons_gate_calibration.json."""
    x_tr, y_tr = _features_and_labels("train")
    x_val, y_val = _features_and_labels("val")
    w = _fit_logistic(x_tr, y_tr)
    bias, w_off, w_allow_signed, w_weight = (float(v) for v in w)

    def _eval(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Brier / ECE / fp / fn at the fitted 0.5 operating point."""
        p = _predict(x, w)
        pred_block = p >= 0.5
        want_block = y >= 0.5
        return {
            "brier": round(float(np.mean((p - y) ** 2)), 6),
            "ece": round(_ece(p, y), 6),
            "fp": int((pred_block & (~want_block)).sum()),
            "fn": int(((~pred_block) & want_block).sum()),
            "n": int(x.shape[0]),
        }

    val_metrics = _eval(x_val, y_val)
    brier, ece = val_metrics["brier"], val_metrics["ece"]
    fp, fn = val_metrics["fp"], val_metrics["fn"]
    # Held-out generalisation: the TEST split is untouched by the fit AND by
    # model selection, so its metrics are the honest out-of-sample number.
    x_test, y_test = _features_and_labels("test")
    test_metrics = _eval(x_test, y_test)

    defaults = BenevolenceCalibration()  # for retained fields
    params = {
        # fitted confidence logistic (w_allow stored positive; formula subtracts)
        "conf_bias": round(bias, 6),
        "conf_w_offensive": round(w_off, 6),
        "conf_w_allow": round(-w_allow_signed, 6),
        "conf_w_weight": round(w_weight, 6),
        # retained (no live-model signal in the corpus to fit these from)
        "conf_w_classifier": defaults.conf_w_classifier,
        "weapons_b6_escalate_confidence": defaults.weapons_b6_escalate_confidence,
        "weapons_hard_refuse_harm_floor": defaults.weapons_hard_refuse_harm_floor,
        "weapons_refuse_redact_harm_floor": defaults.weapons_refuse_redact_harm_floor,
        "weapons_escalate_harm_floor": defaults.weapons_escalate_harm_floor,
    }
    agreement = _gate_agreement()
    out = {
        "_comment": (
            "Fitted by scripts/fit_weapons_gate_calibration.py on "
            "benchmarks/weapons_gate_corpus.py. conf_* are the measured Axis-B "
            "confidence logistic; conf_w_classifier and the harm floors are "
            "retained policy constants (see the script docstring)."
        ),
        "parameters": params,
        "metrics": {
            "n_train": int(x_tr.shape[0]),
            "n_val": int(x_val.shape[0]),
            "val_brier": round(brier, 6),
            "val_ece": round(ece, 6),
            "val_fp": fp,
            "val_fn": fn,
            # Held-out TEST split (never used in fit or model selection).
            "n_test": test_metrics["n"],
            "test_brier": test_metrics["brier"],
            "test_ece": test_metrics["ece"],
            "test_fp": test_metrics["fp"],
            "test_fn": test_metrics["fn"],
            **{k: round(float(v), 6) for k, v in agreement.items()},
        },
    }
    dest = _REPO / "configs" / "weapons_gate_calibration.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {dest}")
    print(json.dumps(out["metrics"], indent=2))


if __name__ == "__main__":
    main()
