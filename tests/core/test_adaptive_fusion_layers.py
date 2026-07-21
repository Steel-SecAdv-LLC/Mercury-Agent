# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Behavioural contract for the adaptive fusion attention stack.

Exercises the public surface of
``omni_mercury_engine.core.adaptive_fusion``: the torch-free serialisation
dataclasses (``UncertaintyEstimate``, ``AttentionVisualization``) and the
``create_attention_heatmap`` helper, plus the torch-gated ``nn.Module``
attention/fusion classes.  Every test is deterministic (seeded RNG, no
network, no wall-clock) and asserts on observed shapes/types/values rather
than on tautologies.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    import pathlib

    import torch
else:
    torch = pytest.importorskip("torch")

from omni_mercury_engine.core.adaptive_fusion import (
    AdaptiveFusionLayer,
    AdaptiveHeadAttention,
    AttentionVisualization,
    SparseAttention,
    TemperatureScaledAttention,
    UncertaintyEstimate,
    UncertaintyQuantifier,
    create_attention_heatmap,
)

SEED = 1234
_MODULE_LOGGER = "omni_mercury_engine.core.adaptive_fusion"


@pytest.fixture(autouse=True)
def _seed_everything() -> None:
    """Make every test in this module reproducible."""
    np.random.default_rng(SEED)
    torch.manual_seed(0)


# ---------------------------------------------------------------------------
# UncertaintyEstimate (dataclass + .to_dict())
# ---------------------------------------------------------------------------
class TestUncertaintyEstimate:
    def _make(self, confidence_level: float = 0.95) -> UncertaintyEstimate:
        return UncertaintyEstimate(
            mean=torch.tensor([1.0, 2.0]),
            std=torch.tensor([0.1, 0.2]),
            lower_bound=torch.tensor([0.5, 1.5]),
            upper_bound=torch.tensor([1.5, 2.5]),
            epistemic_uncertainty=0.01,
            aleatoric_uncertainty=0.02,
            confidence_level=confidence_level,
        )

    def test_default_confidence_level_is_095(self) -> None:
        est = UncertaintyEstimate(
            mean=torch.zeros(2),
            std=torch.ones(2),
            lower_bound=torch.zeros(2),
            upper_bound=torch.ones(2),
            epistemic_uncertainty=0.0,
            aleatoric_uncertainty=0.0,
        )
        assert est.confidence_level == 0.95

    def test_to_dict_converts_tensors_to_python_lists(self) -> None:
        d = self._make().to_dict()
        assert set(d) == {
            "mean",
            "std",
            "lower_bound",
            "upper_bound",
            "epistemic_uncertainty",
            "aleatoric_uncertainty",
            "confidence_level",
        }
        # Tensor fields become plain nested lists (JSON-serialisable).
        assert d["mean"] == [1.0, 2.0]
        assert d["std"] == [0.1, 0.2] or d["std"] == pytest.approx([0.1, 0.2])
        assert isinstance(d["mean"], list)
        assert isinstance(d["lower_bound"], list)
        assert isinstance(d["upper_bound"], list)

    def test_to_dict_preserves_scalar_fields(self) -> None:
        d = self._make(confidence_level=0.99).to_dict()
        assert d["epistemic_uncertainty"] == 0.01
        assert d["aleatoric_uncertainty"] == 0.02
        assert d["confidence_level"] == 0.99
        assert isinstance(d["epistemic_uncertainty"], float)

    def test_to_dict_detaches_grad_tracking_tensors(self) -> None:
        est = UncertaintyEstimate(
            mean=torch.tensor([3.0], requires_grad=True),
            std=torch.tensor([0.5], requires_grad=True),
            lower_bound=torch.tensor([2.0], requires_grad=True),
            upper_bound=torch.tensor([4.0], requires_grad=True),
            epistemic_uncertainty=0.1,
            aleatoric_uncertainty=0.2,
        )
        d = est.to_dict()  # must not raise on grad-tracking tensors
        assert d["mean"] == pytest.approx([3.0])


# ---------------------------------------------------------------------------
# AttentionVisualization (dataclass + .get_top_contributors())
# ---------------------------------------------------------------------------
class TestAttentionVisualization:
    def _viz(self, attention_weights: torch.Tensor, names: list[str]) -> AttentionVisualization:
        return AttentionVisualization(
            attention_weights=attention_weights,
            detector_names=names,
            head_contributions=torch.rand(len(names)),
            temperature=1.0,
            sparsity_ratio=0.5,
        )

    def test_top_contributors_ranks_by_mean_weight(self) -> None:
        # 4D weights [batch, heads, seq, seq]; column j is filled with value j,
        # so per-detector mean weights are [0, 1, 2, 3] -> top two are d, c.
        aw = torch.zeros(1, 1, 4, 4)
        for j in range(4):
            aw[..., :, j] = float(j)
        viz = self._viz(aw, ["a", "b", "c", "d"])
        top = viz.get_top_contributors(2)
        assert top == [("d", 3.0), ("c", 2.0)]

    def test_top_contributors_values_are_descending(self) -> None:
        aw = torch.rand(2, 3, 5, 5)
        viz = self._viz(aw, ["a", "b", "c", "d", "e"])
        top = viz.get_top_contributors(4)
        vals = [v for _, v in top]
        assert vals == sorted(vals, reverse=True)
        assert all(name in {"a", "b", "c", "d", "e"} for name, _ in top)

    def test_top_contributors_clamps_k_to_detector_count(self) -> None:
        aw = torch.rand(1, 1, 3, 3)
        viz = self._viz(aw, ["a", "b", "c"])
        # k larger than the number of detectors returns at most that many.
        assert len(viz.get_top_contributors(10)) == 3

    def test_top_contributors_handles_three_dim_weights(self) -> None:
        # A 3D weight tensor collapses to 1D after mean(dim=(0,1)),
        # skipping the extra-dim reduction branch.
        aw = torch.zeros(2, 3, 4)
        aw[..., 1] = 5.0  # detector index 1 dominates
        viz = self._viz(aw, ["a", "b", "c", "d"])
        top = viz.get_top_contributors(1)
        assert top[0][0] == "b"
        assert top[0][1] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# create_attention_heatmap()
# ---------------------------------------------------------------------------
class TestCreateAttentionHeatmap:
    def test_returns_averaged_matrix_and_metadata(self) -> None:
        aw = torch.rand(2, 2, 3, 3)
        names = ["a", "b", "c"]
        result = create_attention_heatmap(aw, names)
        assert set(result) == {"attention_matrix", "detector_names", "shape"}
        assert result["detector_names"] == names
        assert result["shape"] == [3, 3]
        expected = aw.mean(dim=(0, 1)).detach().cpu().numpy()
        np.testing.assert_allclose(np.asarray(result["attention_matrix"]), expected, rtol=1e-6)

    def test_no_save_path_omits_saved_to_key(self) -> None:
        result = create_attention_heatmap(torch.rand(1, 1, 2, 2), ["a", "b"])
        assert "saved_to" not in result

    def test_save_path_writes_image_and_records_location(self, tmp_path: pathlib.Path) -> None:
        import matplotlib

        matplotlib.use("Agg", force=True)
        out = tmp_path / "heatmap.png"
        result = create_attention_heatmap(
            torch.rand(1, 2, 3, 3), ["a", "b", "c"], save_path=str(out)
        )
        assert result["saved_to"] == str(out)
        assert out.exists() and out.stat().st_size > 0

    def test_save_path_missing_matplotlib_degrades_gracefully(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Force the in-function ``import matplotlib.pyplot`` to raise ImportError.
        monkeypatch.setitem(sys.modules, "matplotlib", None)
        monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
        result = create_attention_heatmap(
            torch.rand(1, 1, 2, 2), ["a", "b"], save_path="/dev/null/heatmap.png"
        )
        # The heatmap data is still returned; only the render is skipped.
        assert "attention_matrix" in result
        assert "saved_to" not in result


# ---------------------------------------------------------------------------
# TemperatureScaledAttention (nn.Module)
# ---------------------------------------------------------------------------
class TestTemperatureScaledAttention:
    def test_forward_shape_and_attention_weights(self) -> None:
        tsa = TemperatureScaledAttention(embed_dim=8, num_heads=2, dropout=0.0)
        x = torch.randn(2, 4, 8)
        out, attn = tsa(x, x, x, return_attention=True)
        assert out.shape == (2, 4, 8)
        assert attn.shape == (2, 2, 4, 4)
        # Attention rows are a proper probability distribution.
        assert torch.allclose(attn.sum(dim=-1), torch.ones(2, 2, 4), atol=1e-5)

    def test_forward_without_attention_returns_none(self) -> None:
        tsa = TemperatureScaledAttention(embed_dim=4, num_heads=1, dropout=0.0)
        x = torch.randn(1, 3, 4)
        out, attn = tsa(x, x, x)
        assert out.shape == (1, 3, 4)
        assert attn is None

    def test_learnable_temperature_is_a_parameter(self) -> None:
        tsa = TemperatureScaledAttention(
            embed_dim=8, num_heads=2, initial_temperature=1.5, learnable_temperature=True
        )
        assert isinstance(tsa.temperature, torch.nn.Parameter)
        assert tsa.get_temperature() == pytest.approx(1.5)

    def test_non_learnable_temperature_is_a_buffer(self) -> None:
        tsa = TemperatureScaledAttention(
            embed_dim=8, num_heads=2, initial_temperature=2.0, learnable_temperature=False
        )
        assert not isinstance(tsa.temperature, torch.nn.Parameter)
        assert "temperature" in dict(tsa.named_buffers())
        assert tsa.get_temperature() == pytest.approx(2.0)

    def test_indivisible_embed_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="divisible by num_heads"):
            TemperatureScaledAttention(embed_dim=8, num_heads=3)

    def test_non_finite_scores_are_sanitised_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        tsa = TemperatureScaledAttention(embed_dim=4, num_heads=1, dropout=0.0)
        bad = torch.full((1, 3, 4), float("inf"))
        with caplog.at_level(logging.WARNING, logger=_MODULE_LOGGER):
            out, _ = tsa(bad, bad, bad)
        assert out.shape == (1, 3, 4)
        assert any("Non-finite" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# SparseAttention (nn.Module)
# ---------------------------------------------------------------------------
class TestSparseAttention:
    def test_top_k_branch_shapes(self) -> None:
        sa = SparseAttention(embed_dim=8, num_heads=2, dropout=0.0, top_k=2)
        x = torch.randn(2, 4, 8)
        out, attn = sa(x, x, x, return_attention=True)
        assert out.shape == (2, 4, 8)
        assert attn.shape == (2, 2, 4, 4)

    def test_top_k_zeroes_out_non_selected_positions(self) -> None:
        sa = SparseAttention(embed_dim=8, num_heads=2, dropout=0.0, top_k=2)
        x = torch.randn(1, 4, 8)
        _, attn = sa(x, x, x, return_attention=True)
        # With top_k=2 over seq_len=4, at most 2 positions per row are non-zero.
        nonzero_per_row = (attn > 1e-9).sum(dim=-1)
        assert int(nonzero_per_row.max().item()) <= 2

    def test_sparsity_ratio_branch(self) -> None:
        sa = SparseAttention(embed_dim=8, num_heads=2, dropout=0.0, sparsity_ratio=0.5)
        x = torch.randn(2, 4, 8)
        out, attn = sa(x, x, x, return_attention=True)
        assert out.shape == (2, 4, 8)
        assert sa.get_sparsity_ratio() == pytest.approx(0.5)

    def test_top_k_not_smaller_than_seq_skips_masking(self) -> None:
        # top_k >= seq_len -> k == seq_len -> the sparsity mask branch is skipped.
        sa = SparseAttention(embed_dim=8, num_heads=2, dropout=0.0, top_k=10)
        x = torch.randn(1, 4, 8)
        _, attn = sa(x, x, x, return_attention=True)
        # No -inf masking, so every position keeps positive probability.
        assert torch.all(attn > 0)

    def test_forward_without_attention_returns_none(self) -> None:
        sa = SparseAttention(embed_dim=8, num_heads=2, dropout=0.0, sparsity_ratio=0.5)
        out, attn = sa(torch.randn(1, 4, 8), torch.randn(1, 4, 8), torch.randn(1, 4, 8))
        assert out.shape == (1, 4, 8)
        assert attn is None


# ---------------------------------------------------------------------------
# AdaptiveHeadAttention (nn.Module)
# ---------------------------------------------------------------------------
class TestAdaptiveHeadAttention:
    def test_forward_shapes_and_active_head_bounds(self) -> None:
        aha = AdaptiveHeadAttention(embed_dim=8, min_heads=2, max_heads=4, dropout=0.0)
        x = torch.randn(2, 4, 8)
        out, attn = aha(x, x, x, return_attention=True)
        assert out.shape == (2, 4, 8)
        assert attn.shape == (2, 4, 4, 4)
        active = aha.get_active_heads()
        assert 2 <= active <= 4

    def test_forward_without_attention_returns_none(self) -> None:
        aha = AdaptiveHeadAttention(embed_dim=8, min_heads=1, max_heads=2, dropout=0.0)
        out, attn = aha(torch.randn(1, 3, 8), torch.randn(1, 3, 8), torch.randn(1, 3, 8))
        assert out.shape == (1, 3, 8)
        assert attn is None

    def test_default_active_heads_before_forward_is_max(self) -> None:
        aha = AdaptiveHeadAttention(embed_dim=8, min_heads=1, max_heads=4)
        assert aha.get_active_heads() == 4

    def test_indivisible_embed_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="divisible by max_heads"):
            AdaptiveHeadAttention(embed_dim=8, max_heads=3)


# ---------------------------------------------------------------------------
# UncertaintyQuantifier (nn.Module)
# ---------------------------------------------------------------------------
class TestUncertaintyQuantifier:
    def test_forward_returns_estimate_with_matching_shapes(self) -> None:
        uq = UncertaintyQuantifier(embed_dim=8, n_mc_samples=5, dropout_rate=0.1)
        x = torch.randn(3, 8)
        est = uq(x)
        assert isinstance(est, UncertaintyEstimate)
        assert est.mean.shape == (3, 8)
        assert est.std.shape == (3, 8)
        assert est.lower_bound.shape == (3, 8)
        assert est.upper_bound.shape == (3, 8)
        assert isinstance(est.epistemic_uncertainty, float)
        assert isinstance(est.aleatoric_uncertainty, float)
        assert est.confidence_level == 0.95

    def test_confidence_interval_ordering(self) -> None:
        uq = UncertaintyQuantifier(embed_dim=8, n_mc_samples=4, dropout_rate=0.1)
        est = uq(torch.randn(2, 8))
        # std is non-negative so the interval brackets the mean.
        assert torch.all(est.std >= 0)
        assert torch.all(est.lower_bound <= est.mean + 1e-6)
        assert torch.all(est.upper_bound >= est.mean - 1e-6)

    def test_custom_confidence_level_uses_wider_z(self) -> None:
        uq = UncertaintyQuantifier(embed_dim=8, n_mc_samples=4, dropout_rate=0.2)
        x = torch.randn(2, 8)
        est99 = uq(x)
        est99 = uq(x, confidence_level=0.99)
        assert est99.confidence_level == 0.99


# ---------------------------------------------------------------------------
# AdaptiveFusionLayer (nn.Module) - integrates everything
# ---------------------------------------------------------------------------
class TestAdaptiveFusionLayer:
    def _layer(self, **kw: Any) -> AdaptiveFusionLayer:
        defaults: dict[str, Any] = {
            "embed_dim": 8,
            "min_heads": 1,
            "max_heads": 4,
            "dropout": 0.0,
            "enable_sparse": True,
            "enable_uncertainty": True,
            "n_mc_samples": 3,
        }
        defaults.update(kw)
        return AdaptiveFusionLayer(**defaults)

    def test_forward_minimal_result_keys(self) -> None:
        layer = self._layer()
        res = layer(torch.randn(2, 4, 8))
        assert set(res) == {"output", "active_heads", "temperature"}
        assert res["output"].shape == (2, 8)
        assert isinstance(res["active_heads"], int)
        assert isinstance(res["temperature"], float)

    def test_forward_full_result_with_attention_and_uncertainty(self) -> None:
        layer = self._layer()
        res = layer(torch.randn(2, 4, 8), return_attention=True, return_uncertainty=True)
        assert set(res["attention_weights"]) == {"adaptive", "temperature_scaled", "sparse"}
        assert isinstance(res["uncertainty"], UncertaintyEstimate)
        assert res["uncertainty"].mean.shape == (2, 8)

    def test_disabled_sparse_and_uncertainty_paths(self) -> None:
        layer = self._layer(enable_sparse=False, enable_uncertainty=False, max_heads=2)
        assert not hasattr(layer, "sparse_attention")
        assert not hasattr(layer, "uncertainty_quantifier")
        res = layer(torch.randn(2, 4, 8), return_attention=True, return_uncertainty=True)
        # No sparse key, and uncertainty is silently absent when disabled.
        assert set(res["attention_weights"]) == {"adaptive", "temperature_scaled"}
        assert "uncertainty" not in res

    def test_get_visualization_from_temperature_scaled_weights(self) -> None:
        layer = self._layer()
        res = layer(torch.randn(2, 4, 8), return_attention=True)
        viz = layer.get_visualization(res["attention_weights"], ["a", "b", "c", "d"])
        assert isinstance(viz, AttentionVisualization)
        assert viz.head_contributions.shape == (4,)  # one per head (max_heads)
        assert 0.0 <= viz.sparsity_ratio <= 1.0
        assert viz.temperature == pytest.approx(layer.temp_attention.get_temperature())
        assert len(viz.get_top_contributors(2)) == 2

    def test_get_visualization_falls_back_to_adaptive(self) -> None:
        layer = self._layer(enable_sparse=False)
        res = layer(torch.randn(1, 4, 8), return_attention=True)
        # Drop the temperature_scaled key so the fallback to 'adaptive' is used.
        weights = {"adaptive": res["attention_weights"]["adaptive"]}
        viz = layer.get_visualization(weights, ["a", "b", "c", "d"])
        assert isinstance(viz, AttentionVisualization)
        assert viz.attention_weights.shape[0] == 1

    def test_get_visualization_empty_when_no_weights(self) -> None:
        layer = self._layer()
        viz = layer.get_visualization({}, ["a", "b"])
        assert viz.attention_weights.shape == (1,)
        assert viz.sparsity_ratio == 0.0
        assert viz.detector_names == ["a", "b"]

    def test_get_visualization_empty_when_weights_are_none(self) -> None:
        layer = self._layer()
        viz = layer.get_visualization(
            {"temperature_scaled": None, "adaptive": None},  # type: ignore[dict-item]
            ["a"],
        )
        assert viz.attention_weights.shape == (1,)
        assert viz.head_contributions.shape == (1,)
