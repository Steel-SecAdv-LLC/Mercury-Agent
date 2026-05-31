"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Integration tests for the opt-in differentiable domain encoder in ``fit_fusion``
(WS-B / Target 2). Verify the *contract*, deterministically and without network:

* ``domain_encoder=False`` (and the default) is the unchanged neural path -- no
  domain-encoder state, no injected feature group, and the served scores match
  the default path to within the fusion path's own floating-point
  non-determinism floor (~1e-15; true bit-equality is precluded by that
  pre-existing noise, not by WS-B -- two identical default fits differ at the
  same scale).
* ``domain_encoder=True`` builds + jointly trains a ``DomainEncoderStack``,
  injects its ``differentiable_domain`` feature into the trained groups, and
  serves finite scores.

Whether the encoder *improves* held-out detection is the empirical question
settled by ``benchmarks/domain_encoder_ablation.py`` on real labels --
deliberately NOT asserted here, where a synthetic pass would be meaningless.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

pytestmark = pytest.mark.xdist_group("fusion_domain_encoder")


def _separable_fixture(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    n_normal, n_anom, dim = 320, 40, 12
    normal = rng.normal(0.0, 1.0, (n_normal, dim))
    anomaly = rng.normal(3.0, 1.0, (n_anom, dim))
    X = np.vstack([normal, anomaly]).astype(np.float32)
    y = np.concatenate([np.zeros(n_normal), np.ones(n_anom)]).astype(np.int64)
    order = rng.permutation(len(X))
    return X[order], y[order]


def _engine() -> Any:
    from omni_mercury_engine.engine import OmniMercuryEngine

    return OmniMercuryEngine(mode="fusion", device="cpu")


def _fit_and_score(
    domain_encoder: Any, *, pass_flag: bool
) -> tuple[Any, dict[str, Any], np.ndarray]:
    torch.manual_seed(0)
    np.random.seed(0)
    X, y = _separable_fixture()
    n_train = int(len(X) * 0.7)
    kw: dict[str, Any] = {"epochs": 8, "batch_size": 32, "validation_split": 0.25}
    if pass_flag:
        kw["domain_encoder"] = domain_encoder
    engine = _engine()
    metrics = engine.fit_fusion(X[:n_train], y[:n_train], **kw)
    scores = engine.score_fusion(X[n_train:])
    return engine, metrics, scores


class TestNeuralPathUnchanged:
    """domain_encoder=False must not alter the neural training contract."""

    def test_default_has_no_domain_state(self) -> None:
        engine, _, _ = _fit_and_score(None, pass_flag=False)
        assert engine._domain_encoder is None
        assert "differentiable_domain" not in (engine._fusion_feature_groups or [])

    def test_false_flag_has_no_domain_state(self) -> None:
        engine, _, _ = _fit_and_score(False, pass_flag=True)
        assert engine._domain_encoder is None
        assert "differentiable_domain" not in (engine._fusion_feature_groups or [])

    def test_off_path_scores_match_default_within_noise(self) -> None:
        """Off-path serves identically to default, up to the ~1e-15 baseline floor."""
        _, _, s_default = _fit_and_score(None, pass_flag=False)
        _, _, s_off = _fit_and_score(False, pass_flag=True)
        assert s_default.shape == s_off.shape
        # Tolerance is far below any real effect but far above the baseline's own
        # run-to-run float noise (~1e-15), so this proves domain_encoder=False is a no-op.
        assert np.allclose(s_default, s_off, atol=1e-6)


class TestDomainEncoderActive:
    """domain_encoder=True co-trains and wires the encoder into the fusion path."""

    def test_true_flag_builds_and_wires_encoder(self) -> None:
        engine, _, scores = _fit_and_score(True, pass_flag=True)
        from omni_mercury_engine.ml.domain_encoders import DomainEncoderStack

        assert isinstance(engine._domain_encoder, DomainEncoderStack)
        assert engine._domain_scaler is not None
        assert "differentiable_domain" in (engine._fusion_feature_groups or [])
        n_test = 360 - int(360 * 0.7)  # matches _fit_and_score's split exactly
        assert scores.shape == (n_test,)
        assert np.isfinite(scores).all()

    def test_encoder_feature_present_at_inference(self) -> None:
        engine, _, _ = _fit_and_score(True, pass_flag=True)
        X, _ = _separable_fixture()
        feats = engine._extract_fusion_features(X[:16], fit_detectors=False)
        assert "differentiable_domain" in feats
        assert feats["differentiable_domain"].shape[0] == 16
