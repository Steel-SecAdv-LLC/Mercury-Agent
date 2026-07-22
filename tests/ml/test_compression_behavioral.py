# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Behavioral tests for :mod:`omni_mercury_engine.ml.compression`.

These tests exercise the real, observed behavior of the model-compression
primitives (quantized/pruned linear layers, the student model, knowledge
distillation, and the unified :class:`ModelCompressor`) together with the
module-level factory / analysis helpers.

All randomness is seeded (``torch.manual_seed(0)`` plus
``numpy.random.default_rng``) so the suite is deterministic. No network,
sleeps, or wall-clock dependence: :func:`benchmark_inference` is asserted
only on its structural contract (keys/types/positivity), never on timing
magnitudes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

import torch
from torch import nn

from omni_mercury_engine.ml.compression import (
    CompressionConfig,
    CompressionMethod,
    KnowledgeDistiller,
    ModelCompressor,
    PrunedLinear,
    QuantizedLinear,
    StudentModel,
    benchmark_inference,
    create_compressed_model,
    estimate_model_size,
)

SEED = 0


@pytest.fixture(autouse=True)
def _seed_everything() -> None:
    """Seed torch and numpy for every test in this module."""
    torch.manual_seed(SEED)
    np.random.default_rng(SEED)


def _make_sequential(bias: bool = True) -> nn.Sequential:
    """A small flat Linear-ReLU-Linear model (flat module names)."""
    return nn.Sequential(
        nn.Linear(8, 16, bias=bias),
        nn.ReLU(),
        nn.Linear(16, 1, bias=bias),
    )


# ---------------------------------------------------------------------------
# CompressionMethod / CompressionConfig
# ---------------------------------------------------------------------------


class TestCompressionMethodAndConfig:
    """Enum values and dataclass defaults/overrides."""

    def test_enum_members_and_values(self) -> None:
        assert CompressionMethod.NONE.value == "none"
        assert CompressionMethod.QUANTIZATION.value == "quantization"
        assert CompressionMethod.PRUNING.value == "pruning"
        assert CompressionMethod.DISTILLATION.value == "distillation"
        assert CompressionMethod.LAYER_FUSION.value == "layer_fusion"
        assert CompressionMethod.COMBINED.value == "combined"

    def test_enum_lookup_by_value(self) -> None:
        assert CompressionMethod("pruning") is CompressionMethod.PRUNING

    def test_enum_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            CompressionMethod("does-not-exist")

    def test_config_defaults(self) -> None:
        config = CompressionConfig()
        assert config.method is CompressionMethod.NONE
        assert config.quantization_bits == 8
        assert config.pruning_ratio == pytest.approx(0.3)
        assert config.distillation_temperature == pytest.approx(3.0)
        assert config.distillation_alpha == pytest.approx(0.5)
        assert config.enable_dynamic_quantization is True
        assert config.calibration_samples == 100
        assert config.extra_params == {}

    def test_config_overrides_and_independent_extra_params(self) -> None:
        config = CompressionConfig(
            method=CompressionMethod.PRUNING,
            quantization_bits=4,
            pruning_ratio=0.7,
            extra_params={"k": "v"},
        )
        assert config.method is CompressionMethod.PRUNING
        assert config.quantization_bits == 4
        assert config.pruning_ratio == pytest.approx(0.7)
        assert config.extra_params == {"k": "v"}
        # default_factory gives each instance its own dict
        assert CompressionConfig().extra_params is not config.extra_params


# ---------------------------------------------------------------------------
# QuantizedLinear
# ---------------------------------------------------------------------------


class TestQuantizedLinear:
    """INT8 quantized linear layer."""

    def test_init_registers_buffers_and_bias(self) -> None:
        layer = QuantizedLinear(4, 3, bias=True, bits=8)
        assert layer.in_features == 4
        assert layer.out_features == 3
        assert layer.bits == 8
        assert layer._is_quantized is False
        assert layer.quantized_weight.shape == (3, 4)
        assert layer.quantized_weight.dtype == torch.int8
        assert layer.weight_scale.shape == (1,)
        assert layer.weight_zero_point.shape == (1,)
        assert isinstance(layer.bias, nn.Parameter)
        assert layer.bias.shape == (3,)

    def test_init_without_bias(self) -> None:
        layer = QuantizedLinear(4, 3, bias=False)
        assert layer.bias is None

    def test_forward_before_quantization_uses_zero_weight(self) -> None:
        # Not yet quantized: weight is the all-zero int8 buffer, so the
        # output is just the (zero-initialised) bias broadcast over the batch.
        layer = QuantizedLinear(4, 3, bias=True)
        out = layer(torch.randn(2, 4))
        assert out.shape == (2, 3)
        assert torch.allclose(out, torch.zeros(2, 3))

    def test_quantize_weights_sets_state(self) -> None:
        layer = QuantizedLinear(4, 3, bits=8)
        weight = torch.randn(3, 4)
        layer.quantize_weights(weight)
        assert layer._is_quantized is True
        # int8 range for 8 bits is [-128, 127]
        assert int(layer.quantized_weight.min()) >= -128  # type: ignore[operator]
        assert int(layer.quantized_weight.max()) <= 127  # type: ignore[operator]
        assert float(layer.weight_scale) > 0.0  # type: ignore[arg-type]

    def test_forward_after_quantization_shape(self) -> None:
        layer = QuantizedLinear(4, 3, bias=True)
        layer.quantize_weights(torch.randn(3, 4))
        out = layer(torch.randn(5, 4))
        assert out.shape == (5, 3)

    def test_dequantized_weight_approximates_original(self) -> None:
        # Round-trip quantize/dequantize should be close to the original.
        layer = QuantizedLinear(6, 4, bias=False, bits=8)
        weight = torch.randn(4, 6)
        layer.quantize_weights(weight)
        qweight = layer.quantized_weight.float()
        recon = (qweight - layer.weight_zero_point) * layer.weight_scale  # type: ignore[operator]
        assert torch.allclose(recon, weight, atol=0.05)

    def test_bits_parameter_controls_range(self) -> None:
        layer = QuantizedLinear(4, 3, bits=4)
        assert layer.bits == 4
        layer.quantize_weights(torch.randn(3, 4))
        # 4-bit signed range is [-8, 7]
        assert int(layer.quantized_weight.min()) >= -8  # type: ignore[operator]
        assert int(layer.quantized_weight.max()) <= 7  # type: ignore[operator]


# ---------------------------------------------------------------------------
# PrunedLinear
# ---------------------------------------------------------------------------


class TestPrunedLinear:
    """Magnitude-pruned linear layer."""

    def test_init_shapes_and_mask(self) -> None:
        layer = PrunedLinear(4, 3, bias=True)
        assert layer.weight.shape == (3, 4)
        assert layer.mask.shape == (3, 4)
        assert torch.all(layer.mask == 1.0)  # type: ignore[arg-type]
        assert layer.bias.shape == (3,)

    def test_init_without_bias(self) -> None:
        layer = PrunedLinear(4, 3, bias=False)
        assert layer.bias is None

    def test_prune_zero_ratio_removes_nothing(self) -> None:
        layer = PrunedLinear(4, 3)
        pruned = layer.prune(0.0)
        assert pruned == 0
        assert torch.all(layer.mask == 1.0)  # type: ignore[arg-type]

    def test_prune_half_ratio(self) -> None:
        layer = PrunedLinear(4, 3)  # 12 weights
        pruned = layer.prune(0.5)
        assert pruned == 6
        # roughly half the mask entries are zeroed
        assert float(layer.mask.sum()) == pytest.approx(6.0)  # type: ignore[operator]

    def test_prune_full_ratio_keeps_at_least_max(self) -> None:
        # At ratio 1.0 the threshold equals the max magnitude, and >= keeps
        # that single largest weight, so exactly numel-1 are pruned.
        layer = PrunedLinear(4, 3)  # 12 weights
        pruned = layer.prune(1.0)
        assert pruned == 11

    def test_forward_applies_mask(self) -> None:
        layer = PrunedLinear(4, 3, bias=False)
        layer.prune(0.5)
        x = torch.randn(2, 4)
        out = layer(x)
        assert out.shape == (2, 3)
        # forward must equal masked-weight linear
        masked = layer.weight * layer.mask  # type: ignore[operator]
        expected = torch.nn.functional.linear(x, masked, None)
        assert torch.allclose(out, expected)

    def test_forward_pruned_weights_have_no_effect(self) -> None:
        layer = PrunedLinear(4, 3, bias=False)
        layer.prune(0.5)
        # Zeroing weights already masked out must not change the output.
        x = torch.randn(2, 4)
        before = layer(x)
        with torch.no_grad():
            layer.weight[layer.mask == 0] += 100.0
        after = layer(x)
        assert torch.allclose(before, after)


# ---------------------------------------------------------------------------
# StudentModel
# ---------------------------------------------------------------------------


class TestStudentModel:
    """Compact student network."""

    def test_default_two_layer_architecture(self) -> None:
        model = StudentModel(input_dim=8)
        assert model.input_dim == 8
        assert model.hidden_dim == 64
        assert model.output_dim == 1
        # 2 layers => Linear, ReLU, Dropout, Linear
        assert len(model.network) == 4
        out = model(torch.randn(3, 8))
        assert out.shape == (3, 1)

    def test_single_layer_has_no_activation(self) -> None:
        model = StudentModel(input_dim=8, hidden_dim=4, output_dim=2, num_layers=1)
        assert len(model.network) == 1
        out = model(torch.randn(3, 8))
        assert out.shape == (3, 2)

    def test_three_layer_stacks_blocks(self) -> None:
        model = StudentModel(input_dim=8, hidden_dim=5, output_dim=2, num_layers=3)
        # 2 hidden blocks (Linear+ReLU+Dropout) + final Linear = 7 modules
        assert len(model.network) == 7
        out = model(torch.randn(4, 8))
        assert out.shape == (4, 2)

    def test_forward_deterministic_in_eval(self) -> None:
        model = StudentModel(input_dim=8, hidden_dim=4)
        model.eval()
        x = torch.randn(2, 8)
        assert torch.allclose(model(x), model(x))


# ---------------------------------------------------------------------------
# KnowledgeDistiller
# ---------------------------------------------------------------------------


class TestKnowledgeDistiller:
    """Knowledge-distillation trainer."""

    def _pair(self, output_dim: int = 3) -> tuple[StudentModel, StudentModel]:
        teacher = StudentModel(input_dim=8, hidden_dim=8, output_dim=output_dim)
        student = StudentModel(input_dim=8, hidden_dim=4, output_dim=output_dim)
        return teacher, student

    def test_init_freezes_teacher(self) -> None:
        teacher, student = self._pair()
        KnowledgeDistiller(teacher, student, temperature=3.0, alpha=0.5)
        assert all(not p.requires_grad for p in teacher.parameters())
        assert not teacher.training  # eval mode

    def test_distillation_loss_without_targets_is_scalar(self) -> None:
        teacher, student = self._pair()
        kd = KnowledgeDistiller(teacher, student)
        x = torch.randn(5, 8)
        loss = kd.distillation_loss(student(x), teacher(x))
        assert loss.dim() == 0
        assert float(loss) >= 0.0

    def test_distillation_loss_bce_branch_for_1d_targets(self) -> None:
        teacher, student = self._pair(output_dim=1)
        kd = KnowledgeDistiller(teacher, student)
        x = torch.randn(5, 8)
        targets = torch.randint(0, 2, (5,)).float()
        loss = kd.distillation_loss(student(x), teacher(x), targets)
        assert loss.dim() == 0
        assert float(loss) > 0.0

    def test_distillation_loss_mse_branch_for_2d_targets(self) -> None:
        teacher, student = self._pair(output_dim=3)
        kd = KnowledgeDistiller(teacher, student)
        x = torch.randn(5, 8)
        targets = torch.randn(5, 3)
        loss = kd.distillation_loss(student(x), teacher(x), targets)
        assert loss.dim() == 0
        assert float(loss) > 0.0

    def test_train_step_without_optimizer_returns_float(self) -> None:
        teacher, student = self._pair()
        kd = KnowledgeDistiller(teacher, student)
        loss = kd.train_step(torch.randn(4, 8))
        assert isinstance(loss, float)

    def test_train_step_with_optimizer_updates_student(self) -> None:
        teacher, student = self._pair(output_dim=3)
        kd = KnowledgeDistiller(teacher, student)
        optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
        before = [p.detach().clone() for p in student.parameters()]
        x = torch.randn(6, 8)
        targets = torch.randn(6, 3)
        loss = kd.train_step(x, targets, optimizer)
        assert isinstance(loss, float)
        after = list(student.parameters())
        # At least one parameter should have moved after a gradient step.
        assert any(not torch.allclose(b, a) for b, a in zip(before, after))

    def test_train_step_leaves_teacher_frozen(self) -> None:
        teacher, student = self._pair(output_dim=3)
        kd = KnowledgeDistiller(teacher, student)
        optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
        teacher_before = [p.detach().clone() for p in teacher.parameters()]
        kd.train_step(torch.randn(4, 8), torch.randn(4, 3), optimizer)
        for b, p in zip(teacher_before, teacher.parameters()):
            assert torch.allclose(b, p)


# ---------------------------------------------------------------------------
# ModelCompressor
# ---------------------------------------------------------------------------


class TestModelCompressor:
    """Unified compression entry point."""

    def test_default_config_is_none_method(self) -> None:
        compressor = ModelCompressor()
        assert compressor.config.method is CompressionMethod.NONE
        assert compressor.get_compression_stats() == {}

    def test_none_returns_same_object(self) -> None:
        compressor = ModelCompressor(CompressionConfig(method=CompressionMethod.NONE))
        model = _make_sequential()
        assert compressor.compress(model) is model

    def test_dynamic_quantization(self) -> None:
        compressor = ModelCompressor(
            CompressionConfig(
                method=CompressionMethod.QUANTIZATION,
                enable_dynamic_quantization=True,
            )
        )
        compressed = compressor.compress(_make_sequential())
        stats = compressor.get_compression_stats()
        assert stats["quantization"] == {"method": "dynamic", "dtype": "qint8"}
        assert compressed(torch.randn(2, 8)).shape == (2, 1)

    def test_static_quantization_replaces_linears(self) -> None:
        compressor = ModelCompressor(
            CompressionConfig(
                method=CompressionMethod.QUANTIZATION,
                enable_dynamic_quantization=False,
                quantization_bits=8,
            )
        )
        compressed = compressor.compress(_make_sequential())
        types = [type(m).__name__ for m in compressed.modules()]
        assert types.count("QuantizedLinear") == 2
        assert compressor.get_compression_stats()["quantization"] == {
            "method": "static",
            "bits": 8,
        }
        assert compressed(torch.randn(2, 8)).shape == (2, 1)

    def test_static_quantization_without_bias(self) -> None:
        compressor = ModelCompressor(
            CompressionConfig(
                method=CompressionMethod.QUANTIZATION,
                enable_dynamic_quantization=False,
            )
        )
        compressed = compressor.compress(_make_sequential(bias=False))
        quant_layers = [m for m in compressed.modules() if isinstance(m, QuantizedLinear)]
        assert len(quant_layers) == 2
        assert all(layer.bias is None for layer in quant_layers)
        assert compressed(torch.randn(2, 8)).shape == (2, 1)

    def test_pruning_replaces_linears_and_records_stats(self) -> None:
        compressor = ModelCompressor(
            CompressionConfig(method=CompressionMethod.PRUNING, pruning_ratio=0.5)
        )
        compressed = compressor.compress(_make_sequential())
        types = [type(m).__name__ for m in compressed.modules()]
        assert types.count("PrunedLinear") == 2
        stats = compressor.get_compression_stats()["pruning"]
        assert stats["ratio"] == pytest.approx(0.5)
        assert stats["total_params"] == 8 * 16 + 16 * 1
        assert 0.0 < stats["actual_sparsity"] <= 1.0
        assert compressed(torch.randn(2, 8)).shape == (2, 1)

    def test_pruning_without_bias(self) -> None:
        compressor = ModelCompressor(
            CompressionConfig(method=CompressionMethod.PRUNING, pruning_ratio=0.3)
        )
        compressed = compressor.compress(nn.Sequential(nn.Linear(6, 4, bias=False)))
        pruned = [m for m in compressed.modules() if isinstance(m, PrunedLinear)]
        assert len(pruned) == 1
        assert pruned[0].bias is None

    def test_pruning_model_without_linear_layers(self) -> None:
        # total_params stays 0 -> actual_sparsity divide-by-zero guard.
        compressor = ModelCompressor(CompressionConfig(method=CompressionMethod.PRUNING))
        compressor.compress(nn.Sequential(nn.ReLU()))
        stats = compressor.get_compression_stats()["pruning"]
        assert stats["total_params"] == 0
        assert stats["actual_sparsity"] == 0

    def test_pruning_on_nested_module_names(self) -> None:
        # StudentModel exposes dotted module paths (network.0, network.3),
        # exercising the nested-getattr branch of _replace_module.
        compressor = ModelCompressor(
            CompressionConfig(method=CompressionMethod.PRUNING, pruning_ratio=0.4)
        )
        model = StudentModel(input_dim=8, hidden_dim=6, output_dim=2)
        compressed = compressor.compress(model)
        types = [type(m).__name__ for m in compressed.modules()]
        assert types.count("PrunedLinear") == 2
        assert compressed(torch.randn(3, 8)).shape == (3, 2)

    def test_layer_fusion_counts_linear_relu_pairs(self) -> None:
        compressor = ModelCompressor(CompressionConfig(method=CompressionMethod.LAYER_FUSION))
        compressed = compressor.compress(_make_sequential())
        stats = compressor.get_compression_stats()["layer_fusion"]
        assert stats["fused_pairs"] == 1
        # fusion is bookkeeping-only; the model still runs.
        assert compressed(torch.randn(2, 8)).shape == (2, 1)

    def test_layer_fusion_ignores_linear_not_followed_by_relu(self) -> None:
        # Two back-to-back Linears: no Linear->ReLU pair to fuse.
        compressor = ModelCompressor(CompressionConfig(method=CompressionMethod.LAYER_FUSION))
        compressor.compress(nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 2)))
        assert compressor.get_compression_stats()["layer_fusion"]["fused_pairs"] == 0

    def test_deep_copy_falls_back_to_deepcopy_on_bad_init_args(self) -> None:
        # A model that records _init_args that do not match its constructor
        # makes the state_dict fast path raise TypeError; _deep_copy_model
        # must catch it and fall back to copy.deepcopy.
        class BadArgs(nn.Module):
            def __init__(self, dim: int) -> None:
                super().__init__()
                # Deliberately wrong: too many positional args for __init__.
                self._init_args = (dim, "unexpected", "extra")
                self._init_kwargs: dict[str, Any] = {}
                self.fc = nn.Linear(dim, dim)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out: torch.Tensor = self.fc(x)
                return out

        compressor = ModelCompressor(
            CompressionConfig(method=CompressionMethod.PRUNING, pruning_ratio=0.5)
        )
        model = BadArgs(4)
        compressed = compressor.compress(model)
        assert compressed is not model
        assert isinstance(compressed.fc, PrunedLinear)
        assert compressed(torch.randn(2, 4)).shape == (2, 4)

    def test_combined_applies_pruning_and_quantization(self) -> None:
        compressor = ModelCompressor(
            CompressionConfig(
                method=CompressionMethod.COMBINED,
                enable_dynamic_quantization=True,
                pruning_ratio=0.5,
            )
        )
        compressed = compressor.compress(_make_sequential())
        stats = compressor.get_compression_stats()
        assert "pruning" in stats
        assert "quantization" in stats
        assert compressed(torch.randn(2, 8)).shape == (2, 1)

    def test_get_compression_stats_returns_copy(self) -> None:
        compressor = ModelCompressor(CompressionConfig(method=CompressionMethod.PRUNING))
        compressor.compress(_make_sequential())
        stats = compressor.get_compression_stats()
        stats["injected"] = True
        assert "injected" not in compressor.get_compression_stats()

    def test_deep_copy_does_not_mutate_original(self) -> None:
        compressor = ModelCompressor(
            CompressionConfig(method=CompressionMethod.PRUNING, pruning_ratio=0.5)
        )
        model = _make_sequential()
        original_first_layer = model[0]
        compressor.compress(model)
        # The source model's layers must remain plain nn.Linear instances.
        assert model[0] is original_first_layer
        assert isinstance(model[0], nn.Linear)

    def test_deep_copy_uses_init_args_fast_path(self) -> None:
        # A model that records its construction args takes the state_dict
        # fast path in _deep_copy_model rather than copy.deepcopy.
        class Recorded(nn.Module):
            def __init__(self, dim: int) -> None:
                super().__init__()
                self._init_args = (dim,)
                self._init_kwargs: dict[str, Any] = {}
                self.fc = nn.Linear(dim, dim)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out: torch.Tensor = self.fc(x)
                return out

        compressor = ModelCompressor(
            CompressionConfig(method=CompressionMethod.PRUNING, pruning_ratio=0.5)
        )
        model = Recorded(4)
        compressed = compressor.compress(model)
        assert compressed is not model
        assert isinstance(compressed.fc, PrunedLinear)
        assert compressed(torch.randn(2, 4)).shape == (2, 4)


# ---------------------------------------------------------------------------
# ModelCompressor.estimate_speedup
# ---------------------------------------------------------------------------


class TestEstimateSpeedup:
    """Speedup multiplier bookkeeping."""

    def test_no_stats_is_unity(self) -> None:
        assert ModelCompressor().estimate_speedup() == pytest.approx(1.0)

    def test_dynamic_quantization_speedup(self) -> None:
        compressor = ModelCompressor(
            CompressionConfig(
                method=CompressionMethod.QUANTIZATION,
                enable_dynamic_quantization=True,
            )
        )
        compressor.compress(_make_sequential())
        assert compressor.estimate_speedup() == pytest.approx(2.0)

    def test_static_quantization_speedup(self) -> None:
        compressor = ModelCompressor(
            CompressionConfig(
                method=CompressionMethod.QUANTIZATION,
                enable_dynamic_quantization=False,
            )
        )
        compressor.compress(_make_sequential())
        assert compressor.estimate_speedup() == pytest.approx(2.5)

    def test_pruning_speedup_scales_with_sparsity(self) -> None:
        compressor = ModelCompressor(
            CompressionConfig(method=CompressionMethod.PRUNING, pruning_ratio=0.5)
        )
        compressor.compress(_make_sequential())
        sparsity = compressor.get_compression_stats()["pruning"]["actual_sparsity"]
        expected = 1.0 / (1.0 - sparsity * 0.5)
        assert compressor.estimate_speedup() == pytest.approx(expected)

    def test_layer_fusion_speedup(self) -> None:
        compressor = ModelCompressor(CompressionConfig(method=CompressionMethod.LAYER_FUSION))
        compressor.compress(_make_sequential())
        fused = compressor.get_compression_stats()["layer_fusion"]["fused_pairs"]
        assert compressor.estimate_speedup() == pytest.approx(1.0 + fused * 0.05)

    def test_combined_speedup_multiplies_factors(self) -> None:
        compressor = ModelCompressor(
            CompressionConfig(
                method=CompressionMethod.COMBINED,
                enable_dynamic_quantization=True,
                pruning_ratio=0.5,
            )
        )
        compressor.compress(_make_sequential())
        stats = compressor.get_compression_stats()
        sparsity = stats["pruning"]["actual_sparsity"]
        expected = 2.0 * (1.0 / (1.0 - sparsity * 0.5))
        assert compressor.estimate_speedup() == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


class TestCreateCompressedModel:
    """Factory function."""

    def test_none_method_returns_same_model(self) -> None:
        model = _make_sequential()
        assert create_compressed_model(model, method="none") is model

    def test_pruning_via_factory(self) -> None:
        model = _make_sequential()
        compressed = create_compressed_model(model, method="pruning", pruning_ratio=0.5)
        types = [type(m).__name__ for m in compressed.modules()]
        assert types.count("PrunedLinear") == 2

    def test_quantization_via_factory(self) -> None:
        compressed = create_compressed_model(_make_sequential(), method="quantization")
        assert compressed(torch.randn(2, 8)).shape == (2, 1)

    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValueError):
            create_compressed_model(_make_sequential(), method="nonsense")


class TestEstimateModelSize:
    """Parameter / byte accounting."""

    def test_size_matches_manual_count(self) -> None:
        model = _make_sequential()
        info = estimate_model_size(model)
        expected_params = sum(p.numel() for p in model.parameters())
        assert info["total_params"] == expected_params
        assert info["trainable_params"] == expected_params
        assert info["size_bytes"] == expected_params * 4  # float32
        assert info["size_mb"] == pytest.approx(info["size_bytes"] / (1024 * 1024))

    def test_frozen_params_not_counted_as_trainable(self) -> None:
        model = nn.Linear(4, 2)
        for param in model.parameters():
            param.requires_grad = False
        info = estimate_model_size(model)
        assert info["total_params"] == 10
        assert info["trainable_params"] == 0

    def test_buffers_included_in_size(self) -> None:
        # QuantizedLinear carries int8/float buffers that add to size_bytes
        # even though it has few trainable parameters.
        layer = QuantizedLinear(4, 3, bias=True)
        info = estimate_model_size(layer)
        # Only the bias (3 floats) is a trainable parameter.
        assert info["total_params"] == 3
        assert info["size_bytes"] > info["total_params"] * 4


class TestBenchmarkInference:
    """Structural contract of the benchmark helper (no timing assertions)."""

    def test_returns_expected_keys_and_types(self) -> None:
        model = _make_sequential()
        result = benchmark_inference(model, input_shape=(2, 8), num_iterations=3, warmup=1)
        assert set(result) == {
            "mean_ms",
            "std_ms",
            "min_ms",
            "max_ms",
            "throughput_per_sec",
        }
        assert all(isinstance(v, float) for v in result.values())

    def test_timing_values_are_non_negative(self) -> None:
        result = benchmark_inference(
            _make_sequential(), input_shape=(1, 8), num_iterations=2, warmup=0
        )
        assert result["mean_ms"] >= 0.0
        assert result["min_ms"] >= 0.0
        assert result["max_ms"] >= result["min_ms"]
        assert result["std_ms"] >= 0.0
        assert result["throughput_per_sec"] > 0.0
