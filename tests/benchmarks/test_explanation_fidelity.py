# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the explanation-fidelity harness (``benchmarks.explanation_fidelity``): the synthetic-data generator and that the dependency-free ``IntegratedGradientsExplainer`` + ``FaithfulnessEvaluator`` (revived from the dormant ``explainability.py``) run and produce a well-formed, finite faithfulness score. The full recovery/faithfulness verdict is the benchmark's job; this keeps the harness honest and importable without ``shap``/``lime``."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from benchmarks.explanation_fidelity import (
    N_FEATURES,
    N_INFORMATIVE,
    _make_data,
    _train_predict_fn,
)


def test_make_data_shapes_and_informative_set() -> None:
    x_tr, y_tr, x_te, y_te, informative = _make_data(0)
    assert x_tr.shape[1] == N_FEATURES
    assert informative == set(range(N_INFORMATIVE))
    assert set(np.unique(y_tr)) <= {0, 1}


def test_explainer_and_evaluator_run() -> None:
    from omni_mercury_engine.cognitive.explainability import (
        FaithfulnessEvaluator,
        IntegratedGradientsExplainer,
    )

    x_tr, y_tr, x_te, _, _ = _make_data(0)
    predict_fn = _train_predict_fn(x_tr, y_tr, 0)
    expl = IntegratedGradientsExplainer(n_steps=16).explain(
        predict_fn, x_te[0], [f"f{i}" for i in range(N_FEATURES)]
    )
    assert len(expl.feature_importances) == N_FEATURES
    scores = FaithfulnessEvaluator().evaluate(predict_fn, x_te[0], expl)
    assert "comprehensiveness" in scores
    assert np.isfinite(scores["comprehensiveness"])


def test_faithfulness_non_regression_comprehensiveness_and_recovery() -> None:
    """IG is faithful above random and recovers the informative features.

    This is the committed faithfulness *non-regression* gate (WS5): it pins the
    two contracts the serve-path explainer relies on, so a future change that
    quietly breaks attribution quality fails CI rather than landing silently.

    - Comprehensiveness: removing IG's top-ranked features moves the prediction
      *more* than removing random features (``comp_ig > comp_random + 0.01``).
    - Recovery@k: IG's top-k attributions land on the genuinely-informative
      features well above chance (``recovery > 2 * chance``).

    Mirrors the harness verdict (``benchmarks/explanation_fidelity.py``); two
    seeds keep the gate fast while still averaging out per-seed noise.
    """
    from benchmarks.explanation_fidelity import run

    seeds = (0, 1)
    results = [run(s) for s in seeds]
    mean_comp_ig = float(np.mean([r["comprehensiveness_ig"] for r in results]))
    mean_comp_random = float(np.mean([r["comprehensiveness_random"] for r in results]))
    mean_recovery = float(np.mean([r["recovery_at_k"] for r in results]))
    chance = results[0]["chance_recovery"]

    assert mean_comp_ig > mean_comp_random + 0.01, (
        f"IG comprehensiveness {mean_comp_ig:.3f} not above random "
        f"{mean_comp_random:.3f} + 0.01"
    )
    assert (
        mean_recovery > 2 * chance
    ), f"IG recovery@k {mean_recovery:.3f} not above 2x chance {2 * chance:.3f}"
