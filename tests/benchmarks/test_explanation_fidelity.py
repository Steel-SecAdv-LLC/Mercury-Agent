"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Tests for the explanation-fidelity harness (``benchmarks.explanation_fidelity``):
the synthetic-data generator and that the dependency-free
``IntegratedGradientsExplainer`` + ``FaithfulnessEvaluator`` (revived from the
dormant ``explainability.py``) run and produce a well-formed, finite faithfulness
score. The full recovery/faithfulness verdict is the benchmark's job; this keeps
the harness honest and importable without ``shap``/``lime``.
"""

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
