# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for path-dependent Tree SHAP and the global interaction matrix.

``TreeShapExplainer._tree_shap_single`` previously returned all-zeros (a stub);
these pin the correct behaviour: exact SHAP additivity, zero attribution for
features the tree never splits on, and a defined covariance-based interaction
matrix. Every case uses a hand-built tree (numpy arrays only) so the suite has
no external ML dependency and runs on a base install.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from omni_mercury_engine.explainability.shap import (
    GlobalExplanation,
    ShapExplanation,
    TreeShapExplainer,
)


def _fake_tree_model() -> SimpleNamespace:
    """A 3-feature binary tree; feature 1 is never used (must get 0 SHAP).

    Structure (decision-tree ``tree_`` array layout; value = class counts [neg, pos];
    leaves marked by ``feature == -2``):
        node0: split f0 <= 0 ? node1 : node2
        node1: leaf, proba_pos = 0.2
        node2: split f2 <= 0 ? node3 : node4
        node3: leaf, proba_pos = 0.6
        node4: leaf, proba_pos = 0.9
    """
    tree_ = SimpleNamespace(
        node_count=5,
        feature=np.array([0, -2, 2, -2, -2]),
        threshold=np.array([0.0, -2.0, 0.0, -2.0, -2.0]),
        children_left=np.array([1, -1, 3, -1, -1]),
        children_right=np.array([2, -1, 4, -1, -1]),
        value=np.array(
            [
                [[50.0, 50.0]],
                [[8.0, 2.0]],
                [[20.0, 40.0]],
                [[4.0, 6.0]],
                [[1.0, 9.0]],
            ]
        ),
        weighted_n_node_samples=np.array([100.0, 40.0, 60.0, 30.0, 30.0]),
    )
    # base ShapExplainer needs a predict/decision_function to construct.
    return SimpleNamespace(tree_=tree_, predict=lambda X: np.zeros(len(X)))


def _proba_pos(x: np.ndarray) -> float:
    """The tree's own positive-class probability for one instance."""
    if x[0] <= 0.0:
        return 0.2
    return 0.6 if x[2] <= 0.0 else 0.9


def test_tree_shap_additivity_and_sparsity() -> None:
    model = _fake_tree_model()
    explainer = TreeShapExplainer(model, feature_names=["f0", "f1", "f2"], seed=0)

    instances = np.array(
        [
            [-1.0, 5.0, 0.5],  # -> node1, 0.2
            [1.0, -3.0, -0.5],  # -> node3, 0.6
            [1.0, 9.0, 0.5],  # -> node4, 0.9
        ]
    )
    explanations = explainer.explain(instances)
    assert isinstance(explanations, list)

    for x, exp in zip(instances, explanations):
        # Exact SHAP additivity: base + sum(shap) == tree output.
        total = exp.base_value + float(np.sum(exp.shap_values))
        assert total == pytest.approx(_proba_pos(x), abs=1e-9)
        assert exp.prediction == pytest.approx(_proba_pos(x), abs=1e-9)
        # Feature 1 is never split on -> exactly zero attribution.
        assert exp.shap_values[1] == pytest.approx(0.0, abs=1e-12)
        # Feature 0 is always on the path -> it must carry signal somewhere.
    # Across the varied instances feature 0 is not uniformly zero.
    assert any(abs(e.shap_values[0]) > 1e-6 for e in explanations)


def test_tree_shap_single_instance_returns_scalar_explanation() -> None:
    model = _fake_tree_model()
    explainer = TreeShapExplainer(model, seed=0)
    exp = explainer.explain(np.array([1.0, 0.0, 0.5]))
    assert isinstance(exp, ShapExplanation)
    assert exp.base_value + float(np.sum(exp.shap_values)) == pytest.approx(0.9, abs=1e-9)


def test_explain_rejects_instance_missing_features() -> None:
    """A too-short instance raises a clear ValueError, not an opaque IndexError.

    The fake tree splits on feature index 2, so a 2-element instance cannot
    address it; the explainer must say so instead of crashing deep in recursion.
    """
    explainer = TreeShapExplainer(_fake_tree_model(), seed=0)
    with pytest.raises(ValueError, match=r"feature index|features"):
        explainer.explain(np.array([1.0, 0.0]))  # model needs 3 features


def test_global_interaction_matrix_is_defined() -> None:
    model = _fake_tree_model()
    explainer = TreeShapExplainer(model, seed=0)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 3))
    global_exp = explainer.explain_global(X)
    interactions = global_exp.get_interaction_values()
    assert interactions is not None
    assert interactions.shape == (3, 3)
    # Symmetric covariance matrix.
    assert np.allclose(interactions, interactions.T)


def test_interaction_matrix_none_with_too_few_instances() -> None:
    single = GlobalExplanation(
        shap_values=np.array([[0.1, 0.2, 0.3]]),
        base_value=0.0,
        feature_names=None,
        data=np.zeros((1, 3)),
    )
    assert single.get_interaction_values() is None


def test_deep_tree_does_not_overflow_the_stack() -> None:
    """A tree deeper than Python's recursion limit must still explain (no crash).

    The conditional-expectation value function is iterative, so a legitimately
    deep tree (sklearn defaults to max_depth=None) does not raise RecursionError.
    """
    depth = 2000
    n = 2 * depth + 1
    feature = np.full(n, -2)
    threshold = np.full(n, -2.0)
    left = np.full(n, -1)
    right = np.full(n, -1)
    cov = np.ones(n)
    value = np.zeros((n, 1, 1))
    node, nxt = 0, 1
    for d in range(depth):
        feature[node] = 0
        threshold[node] = 0.0
        lc, rc = nxt, nxt + 1
        nxt += 2
        left[node] = lc
        right[node] = rc
        value[lc] = [[float(d)]]  # left leaf
        cov[node] = float(depth - d + 1)
        node = rc  # descend right
    value[node] = [[999.0]]
    tree_ = SimpleNamespace(
        node_count=n,
        feature=feature,
        threshold=threshold,
        children_left=left,
        children_right=right,
        value=value,
        weighted_n_node_samples=cov,
    )
    model = SimpleNamespace(tree_=tree_, predict=lambda X: np.zeros(len(X)))
    explainer = TreeShapExplainer(model, seed=0)
    exp = explainer.explain(np.array([1.0]))  # x[0]=1 > 0 -> always right -> 999
    assert isinstance(exp, ShapExplanation)  # single instance -> scalar explanation
    total = exp.base_value + float(np.sum(exp.shap_values))
    assert total == pytest.approx(999.0, abs=1e-6)


def test_leaf_value_distinguishes_regression_from_classification_by_shape() -> None:
    """(n_outputs, 1) multi-output regression and (1, n_classes) classification
    must be reduced differently despite raveling to the same size."""
    explainer = TreeShapExplainer(_fake_tree_model(), seed=0)
    # Multi-output regression leaf (2 outputs, last axis == 1) -> first output.
    reg = explainer._leaf_value({"value": np.array([[[7.0], [8.0]]])}, 0)
    assert reg == pytest.approx(7.0)
    # Binary classification leaf (1 output, 2 classes) -> P(class 1).
    clf = explainer._leaf_value({"value": np.array([[[8.0, 2.0]]])}, 0)
    assert clf == pytest.approx(0.2)


def test_tree_shap_multi_tree_ensemble_additivity() -> None:
    """Ensemble decomposition (estimators_) averages per-tree SHAP and stays additive."""
    forest = SimpleNamespace(
        estimators_=[_fake_tree_model(), _fake_tree_model()],
    )
    explainer = TreeShapExplainer(
        SimpleNamespace(estimators_=forest.estimators_, predict=lambda X: np.zeros(len(X))),
        seed=0,
    )
    x = np.array([1.0, 0.0, -0.5])  # both trees -> node3, proba 0.6
    exp = explainer.explain(x)
    assert isinstance(exp, ShapExplanation)
    assert exp.base_value + float(np.sum(exp.shap_values)) == pytest.approx(0.6, abs=1e-9)
