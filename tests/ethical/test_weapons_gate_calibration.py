# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Merit gate for the weapons-gate confidence calibration's held-out generalisation.

``scripts/fit_weapons_gate_calibration.py`` fits the Axis-B offensive-confidence
logistic on the TRAIN split of the real 362-case corpus and records its
generalisation on the untouched TEST split (never used in the fit or in model
selection) into ``configs/weapons_gate_calibration.json``.

This gate pins two things so the shipped calibration cannot silently rot:

1. **Freshness** — the config's recorded held-out TEST metrics match a live
   recomputation of the fitted logistic on the TEST split.
2. **Generalisation budget** — the held-out TEST split has zero false
   positives / false negatives at the 0.5 operating point and the calibration
   error stays small, so the real-corpus fit genuinely generalises rather than
   memorising the training split.

Complements ``test_weapons_gate_eval.py`` (which gates the *disposition-level*
fp/fn rate); this gate covers the *confidence logistic's* calibration.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "benchmarks"))

from weapons_gate_corpus import build_corpus

from omni_mercury_engine.cognitive.ethical_bounding import compute_gate_features

_CONFIG = _REPO / "configs" / "weapons_gate_calibration.json"


def _test_split_features() -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for row in build_corpus():
        if row.split != "test":
            continue
        n_off, n_allow, weight, _boost = compute_gate_features(row.text)
        xs.append([float(n_off), float(n_allow), float(weight)])
        ys.append(1.0 if row.label == "offensive" else 0.0)
    return np.asarray(xs), np.asarray(ys)


def _live_test_metrics() -> dict[str, float]:
    """Recompute held-out TEST metrics from the shipped fitted logistic."""
    cfg = json.loads(_CONFIG.read_text())
    p = cfg["parameters"]
    x, y = _test_split_features()
    # confidence = sigmoid(bias + w_off*n_off - w_allow*n_allow + w_weight*weight);
    # conf_w_allow is stored positive and subtracted (see the fit script).
    z = (
        p["conf_bias"]
        + p["conf_w_offensive"] * x[:, 0]
        - p["conf_w_allow"] * x[:, 1]
        + p["conf_w_weight"] * x[:, 2]
    )
    pred = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
    block = pred >= 0.5
    want = y >= 0.5
    return {
        "n": int(x.shape[0]),
        "fp": int((block & (~want)).sum()),
        "fn": int(((~block) & want).sum()),
        "brier": float(np.mean((pred - y) ** 2)),
    }


def test_recorded_test_metrics_are_fresh() -> None:
    """The config's held-out TEST metrics must match a live recomputation."""
    cfg = json.loads(_CONFIG.read_text())
    m = cfg["metrics"]
    live = _live_test_metrics()
    assert m["n_test"] == live["n"], "TEST split size drifted from the config"
    assert m["test_fp"] == live["fp"], f"recorded test_fp {m['test_fp']} != live {live['fp']}"
    assert m["test_fn"] == live["fn"], f"recorded test_fn {m['test_fn']} != live {live['fn']}"
    assert abs(m["test_brier"] - live["brier"]) < 1e-6, "recorded test_brier is stale"


def test_held_out_generalisation_budget() -> None:
    """The real-corpus fit must generalise: zero fp/fn and low calibration error."""
    live = _live_test_metrics()
    assert live["n"] >= 60, "TEST split unexpectedly small"
    assert live["fp"] == 0, f"held-out false positives regressed: {live['fp']}"
    assert live["fn"] == 0, f"held-out false negatives regressed: {live['fn']}"
    assert live["brier"] < 0.05, f"held-out Brier {live['brier']:.4f} regressed past 0.05"
