# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for Knowledge Distillation modules.

Tests Dual-Student Knowledge Distillation for anomaly detection.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Any

import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

HAS_TORCHVISION = importlib.util.find_spec("torchvision") is not None

pytestmark = [pytest.mark.slow, pytest.mark.visual]


def _make_stub_teacher(base: torch.Tensor) -> Any:
    """Build a deterministic teacher stub that repeats ``base`` per batch.

    The stub counts forward calls so tests can pin how many batches the
    training loop visits.
    """

    class _StubTeacher(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0
            self.seen: list[torch.Tensor] = []

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            self.calls += 1
            self.seen.append(x.detach().cpu())
            return {"feat": base.repeat(x.shape[0], 1, 1, 1)}

    return _StubTeacher()


def _make_const_shift(delta: float) -> Any:
    """Build a module that adds a constant to its input."""

    class _ConstShift(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + delta

    return _ConstShift()


def _make_const_attention(hidden_dim: int, value: float) -> Any:
    """Build an attention stub whose gate is ``value`` everywhere."""

    class _ConstAttention(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.full((x.shape[0], hidden_dim), value)

    return _ConstAttention()


def _make_zeros_block() -> Any:
    """Build a module that maps any input to zeros of the same shape."""

    class _ZerosBlock(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(x)

    return _ZerosBlock()


@pytest.mark.skipif(not HAS_TORCH or not HAS_TORCHVISION, reason="torch/torchvision not installed")
class TestDualStudentConfig:
    """Tests for DualStudentConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        from omni_mercury_engine.ml.distillation import DualStudentConfig

        config = DualStudentConfig()
        assert config.backbone == "resnet18"
        assert config.hidden_dim == 256
        assert config.learning_rate == 1e-4
        assert config.num_epochs == 100
        assert config.temperature == 4.0
        assert config.alpha == 0.5
        assert config.batch_size == 32

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        from omni_mercury_engine.ml.distillation import DualStudentConfig

        config = DualStudentConfig(
            backbone="resnet34",
            hidden_dim=128,
            learning_rate=1e-3,
            num_epochs=50,
        )
        assert config.backbone == "resnet34"
        assert config.hidden_dim == 128
        assert config.learning_rate == 1e-3


@pytest.mark.skipif(not HAS_TORCH or not HAS_TORCHVISION, reason="torch/torchvision not installed")
class TestEncoderDecoderStudent:
    """Tests for Encoder-Decoder student network."""

    def test_initialization(self) -> None:
        """Test Encoder-Decoder student initialization."""
        from omni_mercury_engine.ml.distillation.dual_student import EncoderDecoderStudent

        student = EncoderDecoderStudent(in_channels=256, hidden_dim=128)
        assert student is not None

    def test_forward_pass(self) -> None:
        """Test Encoder-Decoder forward pass."""
        from omni_mercury_engine.ml.distillation.dual_student import EncoderDecoderStudent

        student = EncoderDecoderStudent(in_channels=256, hidden_dim=128)

        # Create input [B, C, H, W]
        x = torch.randn(2, 256, 14, 14)
        output = student(x)

        assert output.shape == x.shape


@pytest.mark.skipif(not HAS_TORCH or not HAS_TORCHVISION, reason="torch/torchvision not installed")
class TestEncoderEncoderStudent:
    """Tests for Encoder-Encoder student network."""

    def test_initialization(self) -> None:
        """Test Encoder-Encoder student initialization."""
        from omni_mercury_engine.ml.distillation.dual_student import EncoderEncoderStudent

        student = EncoderEncoderStudent(in_channels=256, hidden_dim=128)
        assert student is not None

    def test_forward_pass(self) -> None:
        """Test Encoder-Encoder forward pass."""
        from omni_mercury_engine.ml.distillation.dual_student import EncoderEncoderStudent

        student = EncoderEncoderStudent(in_channels=256, hidden_dim=128)

        # Create input [B, C, H, W]
        x = torch.randn(2, 256, 14, 14)
        output = student(x)

        assert output.shape == x.shape

    def test_attention_mechanism(self) -> None:
        """Test attention mechanism in Encoder-Encoder student."""
        from omni_mercury_engine.ml.distillation.dual_student import EncoderEncoderStudent

        student = EncoderEncoderStudent(in_channels=256, hidden_dim=128)

        # Verify attention layers exist
        assert hasattr(student, "attention")


@pytest.mark.skipif(not HAS_TORCH or not HAS_TORCHVISION, reason="torch/torchvision not installed")
class TestDualStudentDistillation:
    """Tests for Dual-Student Distillation main class."""

    def test_initialization_default(self) -> None:
        """Test initialization with default config."""
        from omni_mercury_engine.ml.distillation import DualStudentDistillation

        distiller = DualStudentDistillation()
        assert distiller is not None
        assert distiller.config.backbone == "resnet18"

    def test_initialization_with_dict_config(self) -> None:
        """Test initialization with dictionary config."""
        from omni_mercury_engine.ml.distillation import DualStudentDistillation

        config = {
            "backbone": "resnet18",
            "hidden_dim": 128,
            "num_epochs": 5,
        }
        distiller = DualStudentDistillation(config=config)
        assert distiller.config.hidden_dim == 128

    def test_initialization_with_dataclass_config(self) -> None:
        """Test initialization with dataclass config."""
        from omni_mercury_engine.ml.distillation import DualStudentConfig, DualStudentDistillation

        config = DualStudentConfig(hidden_dim=512)
        distiller = DualStudentDistillation(config=config)
        assert distiller.config.hidden_dim == 512

    def test_not_fitted_error(self, sample_image: Any) -> None:
        """Test error when calling detect before fit."""
        from omni_mercury_engine.ml.distillation import DualStudentDistillation

        distiller = DualStudentDistillation()

        with pytest.raises(RuntimeError, match="Must call fit"):
            distiller.detect(sample_image)

    @pytest.mark.slow
    def test_fit_and_detect(self, sample_image_batch: Any, sample_image: Any) -> None:
        """Test fitting and detection workflow."""
        from omni_mercury_engine.ml.distillation import DualStudentConfig, DualStudentDistillation

        # Use minimal config for fast testing
        config = DualStudentConfig(
            num_epochs=2,
            batch_size=2,
        )
        distiller = DualStudentDistillation(config=config)

        # Fit on normal images
        assert distiller.feature_dim == 0
        distiller.fit(sample_image_batch)
        assert distiller._is_fitted
        # resnet18 layer2 (128) + layer3 (256) channels
        assert distiller.feature_dim == 384
        # The teacher stays frozen throughout training.
        assert distiller.teacher is not None
        assert all(not p.requires_grad for p in distiller.teacher.parameters())

        # Detect anomalies
        result = distiller.detect(sample_image)

        assert "scores" in result
        assert "anomaly_maps" in result
        assert "is_anomaly" in result
        assert "student1_maps" in result
        assert "student2_maps" in result

    def test_distillation_loss(self) -> None:
        """Test distillation loss computation."""
        from omni_mercury_engine.ml.distillation import DualStudentDistillation

        distiller = DualStudentDistillation()

        teacher_feat = torch.randn(2, 256, 14, 14)
        student_feat = torch.randn(2, 256, 14, 14)

        loss = distiller._distillation_loss(teacher_feat, student_feat)

        assert loss.ndim == 0  # Scalar
        assert loss >= 0

    def test_aggregate_features(self) -> None:
        """Test feature aggregation from multiple layers."""
        from omni_mercury_engine.ml.distillation import DualStudentDistillation

        distiller = DualStudentDistillation()

        features = {
            "layer2": torch.randn(2, 128, 28, 28),
            "layer3": torch.randn(2, 256, 14, 14),
        }

        aggregated = distiller._aggregate_features(features)

        # Should concatenate and resize to smallest spatial size
        assert aggregated.shape[0] == 2
        assert aggregated.shape[2] == 14
        assert aggregated.shape[3] == 14


@pytest.mark.skipif(not HAS_TORCH or not HAS_TORCHVISION, reason="torch/torchvision not installed")
class TestDistillationModule:
    """Tests for distillation module imports and structure."""

    def test_module_imports(self) -> None:
        """Test that all exports are available."""
        from omni_mercury_engine.ml.distillation import DualStudentConfig, DualStudentDistillation

        assert DualStudentDistillation is not None
        assert DualStudentConfig is not None

    def test_module_all_exports(self) -> None:
        """Test __all__ exports."""
        from omni_mercury_engine.ml import distillation

        assert "DualStudentDistillation" in distillation.__all__
        assert "DualStudentConfig" in distillation.__all__


@pytest.mark.skipif(not HAS_TORCH or not HAS_TORCHVISION, reason="torch/torchvision not installed")
class TestMutationClosedGaps:
    """Behavioral pins for gaps found by mutation testing.

    Each test here kills at least one mutant that survived the
    2026-10-20 mutation run (45% kill rate on dual_student.py).
    """

    def test_students_default_hidden_dim(self) -> None:
        """Both student architectures default to a 256-channel hidden dim."""
        from omni_mercury_engine.ml.distillation.dual_student import (
            EncoderDecoderStudent,
            EncoderEncoderStudent,
        )

        assert EncoderDecoderStudent(in_channels=4).encoder[0].out_channels == 256
        assert EncoderEncoderStudent(in_channels=4).enc1[0].out_channels == 256

    def test_attention_gates_multiplicatively(self) -> None:
        """Zero attention fully suppresses features: only the output bias remains."""
        from omni_mercury_engine.ml.distillation.dual_student import EncoderEncoderStudent

        torch.manual_seed(0)
        student = EncoderEncoderStudent(in_channels=3, hidden_dim=8)
        student.eval()
        x = torch.randn(2, 3, 5, 5)

        with torch.no_grad():
            enc1 = student.enc1(x)
            enc2 = student.enc2(enc1) + enc1
            # The encoded features must be non-zero, or a closed gate
            # would be indistinguishable from an additive one.
            assert enc2.abs().sum() > 0

        student.attention = _make_const_attention(8, 0.0)
        with torch.no_grad():
            out = student(x)
            expected = student.output(torch.zeros(2, 8, 5, 5))

        torch.testing.assert_close(out, expected)

    def test_second_encoder_is_residual(self) -> None:
        """A zero second block passes the first encoding through unchanged."""
        from omni_mercury_engine.ml.distillation.dual_student import EncoderEncoderStudent

        torch.manual_seed(0)
        student = EncoderEncoderStudent(in_channels=3, hidden_dim=8)
        student.eval()
        x = torch.randn(2, 3, 5, 5)

        with torch.no_grad():
            enc1 = student.enc1(x)
            # The pass-through must be non-zero, or a sign flip on the
            # residual would be invisible.
            assert enc1.abs().sum() > 0

        student.enc2 = _make_zeros_block()
        student.attention = _make_const_attention(8, 1.0)
        with torch.no_grad():
            out = student(x)
            expected = student.output(enc1)

        torch.testing.assert_close(out, expected)

    def test_aggregate_features_nonsquare_minimums(self) -> None:
        """Aggregation resizes to (min height, min width), not (min width, min width)."""
        from omni_mercury_engine.ml.distillation import DualStudentDistillation

        distiller = DualStudentDistillation(config={"device": "cpu"})
        a = torch.randn(1, 2, 4, 6)
        b = torch.randn(1, 3, 8, 6)

        agg = distiller._aggregate_features({"a": a, "b": b})

        assert agg.shape == (1, 5, 4, 6)
        # The feature already at the minimum size passes through untouched.
        assert torch.equal(agg[:, :2], a)

    def test_aggregate_features_single_axis_mismatch(self) -> None:
        """A feature off in height alone is still resized before concatenation."""
        from omni_mercury_engine.ml.distillation import DualStudentDistillation

        distiller = DualStudentDistillation(config={"device": "cpu"})
        a = torch.randn(1, 2, 4, 4)
        b = torch.randn(1, 3, 6, 4)

        agg = distiller._aggregate_features({"a": a, "b": b})

        assert agg.shape == (1, 5, 4, 4)
        assert torch.equal(agg[:, :2], a)

    def test_aggregate_features_width_only_mismatch(self) -> None:
        """A feature off in width alone is still resized before concatenation."""
        from omni_mercury_engine.ml.distillation import DualStudentDistillation

        distiller = DualStudentDistillation(config={"device": "cpu"})
        a = torch.randn(1, 2, 4, 6)
        b = torch.randn(1, 3, 4, 9)

        agg = distiller._aggregate_features({"a": a, "b": b})

        assert agg.shape == (1, 5, 4, 6)
        assert torch.equal(agg[:, :2], a)

    def test_aggregate_features_bilinear_values(self) -> None:
        """Downsizing uses bilinear interpolation without corner alignment."""
        from omni_mercury_engine.ml.distillation import DualStudentDistillation

        distiller = DualStudentDistillation(config={"device": "cpu"})
        a = torch.zeros(1, 1, 2, 2)
        b = torch.arange(16.0).view(1, 1, 4, 4)

        agg = distiller._aggregate_features({"a": a, "b": b})

        # align_corners=False averages each 2x2 block; align_corners=True
        # would instead sample the corners (0, 3, 12, 15).
        expected = torch.tensor([[[2.5, 4.5], [10.5, 12.5]]])
        torch.testing.assert_close(agg[:, 1], expected)

    def test_distillation_loss_exact_values(self) -> None:
        """The loss is 1 minus the mean L2-normalized channel cosine similarity."""
        from omni_mercury_engine.ml.distillation import DualStudentDistillation

        distiller = DualStudentDistillation(config={"device": "cpu"})
        # Channel vectors (3, 4) and (4, 3): cosine = 0.96 exactly.
        teacher = torch.tensor([[[[3.0]], [[4.0]]]])
        student = torch.tensor([[[[4.0]], [[3.0]]]])

        loss = distiller._distillation_loss(teacher, student)
        torch.testing.assert_close(loss, torch.tensor(1.0 - 0.96))

        identical = distiller._distillation_loss(teacher, teacher)
        assert identical.item() == pytest.approx(0.0, abs=1e-6)

        orth_teacher = torch.tensor([[[[1.0]], [[0.0]]]])
        orth_student = torch.tensor([[[[0.0]], [[1.0]]]])
        orthogonal = distiller._distillation_loss(orth_teacher, orth_student)
        assert orthogonal.item() == pytest.approx(1.0, abs=1e-6)

    def test_fit_drops_partial_batches(self) -> None:
        """Training visits only full batches: 9 samples at batch size 4 is 2 steps."""
        from omni_mercury_engine.ml.distillation import DualStudentDistillation
        from omni_mercury_engine.ml.distillation.dual_student import (
            EncoderDecoderStudent,
            EncoderEncoderStudent,
        )

        torch.manual_seed(0)
        base = torch.arange(1.0, 13.0).view(1, 3, 2, 2)
        teacher = _make_stub_teacher(base)
        distiller = DualStudentDistillation(
            config={"device": "cpu", "num_epochs": 1, "batch_size": 4}
        )
        distiller.teacher = teacher
        distiller.student1 = EncoderDecoderStudent(3, 8)
        distiller.student2 = EncoderEncoderStudent(3, 8)

        distiller.fit(torch.randn(9, 3, 2, 2))

        assert teacher.calls == 2
        assert distiller._is_fitted

    def test_fit_trains_when_dataset_smaller_than_batch(self) -> None:
        """A dataset smaller than batch_size still trains instead of no-opping."""
        from omni_mercury_engine.ml.distillation import DualStudentDistillation
        from omni_mercury_engine.ml.distillation.dual_student import (
            EncoderDecoderStudent,
            EncoderEncoderStudent,
        )

        torch.manual_seed(0)
        base = torch.arange(1.0, 13.0).view(1, 3, 2, 2)
        teacher = _make_stub_teacher(base)
        distiller = DualStudentDistillation(
            config={"device": "cpu", "num_epochs": 1, "batch_size": 32, "learning_rate": 1e-2}
        )
        distiller.teacher = teacher
        distiller.student1 = EncoderDecoderStudent(3, 8)
        distiller.student2 = EncoderEncoderStudent(3, 8)
        before: list[torch.Tensor] = [p.clone() for p in distiller.student1.parameters()]

        distiller.fit(torch.randn(4, 3, 2, 2))

        assert teacher.calls == 1
        after = list(distiller.student1.parameters())
        assert any(not torch.equal(b, a) for b, a in zip(before, after, strict=True))

    def test_fit_shuffles_training_batches(self) -> None:
        """Training visits every sample once per epoch, in shuffled order."""
        from omni_mercury_engine.ml.distillation import DualStudentDistillation
        from omni_mercury_engine.ml.distillation.dual_student import (
            EncoderDecoderStudent,
            EncoderEncoderStudent,
        )

        torch.manual_seed(0)
        base = torch.arange(1.0, 13.0).view(1, 3, 2, 2)
        teacher = _make_stub_teacher(base)
        distiller = DualStudentDistillation(
            config={"device": "cpu", "num_epochs": 1, "batch_size": 4}
        )
        distiller.teacher = teacher
        distiller.student1 = EncoderDecoderStudent(3, 8)
        distiller.student2 = EncoderEncoderStudent(3, 8)
        # Sample i carries the constant value i, so the teacher stub's
        # recording reveals the exact visit order.
        data = torch.arange(8.0).view(8, 1, 1, 1).expand(8, 3, 2, 2).contiguous()

        distiller.fit(data)

        visited = torch.cat(teacher.seen)[:, 0, 0, 0].tolist()
        assert sorted(visited) == list(range(8))
        assert visited != list(range(8))

    def test_fit_rejects_empty_data(self) -> None:
        """An empty dataset raises instead of marking the model fitted untrained."""
        from omni_mercury_engine.ml.distillation import DualStudentDistillation
        from omni_mercury_engine.ml.distillation.dual_student import (
            EncoderDecoderStudent,
            EncoderEncoderStudent,
        )

        base = torch.arange(1.0, 13.0).view(1, 3, 2, 2)
        distiller = DualStudentDistillation(config={"device": "cpu", "num_epochs": 1})
        distiller.teacher = _make_stub_teacher(base)
        distiller.student1 = EncoderDecoderStudent(3, 8)
        distiller.student2 = EncoderEncoderStudent(3, 8)

        with pytest.raises(ValueError, match="at least one sample"):
            distiller.fit(torch.empty(0, 3, 2, 2))
        assert not distiller._is_fitted

    def test_epoch_logging_reports_exact_mean(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The epoch log fires at epoch 10 and reports the exact per-epoch mean loss."""
        from omni_mercury_engine.ml.distillation import DualStudentDistillation
        from omni_mercury_engine.ml.distillation.dual_student import (
            EncoderDecoderStudent,
            EncoderEncoderStudent,
        )

        torch.manual_seed(0)
        base = torch.arange(1.0, 13.0).view(1, 3, 2, 2)
        distiller = DualStudentDistillation(
            config={"device": "cpu", "num_epochs": 10, "batch_size": 4}
        )
        distiller.teacher = _make_stub_teacher(base)
        distiller.student1 = EncoderDecoderStudent(3, 8)
        distiller.student2 = EncoderEncoderStudent(3, 8)

        def _half_loss(teacher_feat: torch.Tensor, student_feat: torch.Tensor) -> torch.Tensor:
            return student_feat.sum() * 0.0 + 0.5

        monkeypatch.setattr(distiller, "_distillation_loss", _half_loss)

        with caplog.at_level(
            logging.INFO, logger="omni_mercury_engine.ml.distillation.dual_student"
        ):
            distiller.fit(torch.randn(4, 3, 2, 2))

        epoch_logs = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Epoch ")]
        # One batch per epoch contributing the alpha-weighted total
        # 0.5 * 0.5 + 0.5 * 0.5 = 0.5, so the per-epoch mean is exactly 0.5
        # (any denominator other than the true batch count shifts it) — and
        # only epoch 10 logs.
        assert epoch_logs == ["Epoch 10/10, Loss: 0.500000"]

    def test_detect_pins_discrepancy_values(self) -> None:
        """detect() returns per-student mean squared teacher discrepancies."""
        from omni_mercury_engine.ml.distillation import DualStudentDistillation

        base = torch.arange(1.0, 13.0).view(1, 3, 2, 2)
        distiller = DualStudentDistillation(config={"device": "cpu"})
        distiller.teacher = _make_stub_teacher(base)
        # Student 1 misses the teacher by -2 everywhere, student 2 by +1.
        distiller.student1 = _make_const_shift(-2.0)
        distiller.student2 = _make_const_shift(1.0)
        distiller._is_fitted = True

        result = distiller.detect(torch.zeros(2, 3, 2, 2))

        assert result["student1_maps"].shape == (2, 2, 2)
        assert result["student1_maps"] == pytest.approx(4.0)
        assert result["student2_maps"] == pytest.approx(1.0)
        assert result["anomaly_maps"] == pytest.approx(4.0)
        assert result["scores"] == pytest.approx([4.0, 4.0])
        # Scores equal the mean, and the anomaly cut is strictly greater-than.
        assert not result["is_anomaly"].any()


@pytest.mark.skipif(not HAS_TORCH or not HAS_TORCHVISION, reason="torch/torchvision not installed")
class TestAlphaWiring:
    """DualStudentConfig.alpha tilts training between the two students.

    ``total_loss = alpha * loss1 + (1 - alpha) * loss2``: 1.0 trains only the
    patch-level encoder-decoder student, 0.0 only the semantic
    encoder-encoder student. These fail on the unweighted ``loss1 + loss2``.
    """

    @staticmethod
    def _distiller(alpha: float) -> Any:
        from omni_mercury_engine.ml.distillation import DualStudentDistillation
        from omni_mercury_engine.ml.distillation.dual_student import (
            EncoderDecoderStudent,
            EncoderEncoderStudent,
        )

        torch.manual_seed(0)
        base = torch.arange(1.0, 13.0).view(1, 3, 2, 2)
        distiller = DualStudentDistillation(
            config={
                "device": "cpu",
                "num_epochs": 3,
                "batch_size": 8,
                "learning_rate": 1e-2,
                "alpha": alpha,
            }
        )
        distiller.teacher = _make_stub_teacher(base)
        distiller.student1 = EncoderDecoderStudent(3, 8)
        distiller.student2 = EncoderEncoderStudent(3, 8)
        return distiller

    @staticmethod
    def _params(module: Any) -> list[torch.Tensor]:
        return [p.detach().clone() for p in module.parameters()]

    @staticmethod
    def _changed(before: list[torch.Tensor], module: Any) -> bool:
        return any(not torch.equal(b, a) for b, a in zip(before, module.parameters(), strict=True))

    def test_alpha_one_trains_only_the_patch_student(self) -> None:
        distiller = self._distiller(alpha=1.0)
        s1_before = self._params(distiller.student1)
        s2_before = self._params(distiller.student2)
        distiller.fit(torch.randn(8, 3, 2, 2))
        assert self._changed(s1_before, distiller.student1)
        # loss2's coefficient is 0: the semantic student gets no gradient.
        assert not self._changed(s2_before, distiller.student2)

    def test_alpha_zero_trains_only_the_semantic_student_and_descends(self) -> None:
        """alpha=0 moves only student2, and MINIMIZES the semantic loss.

        One fit carries both assertions (the mutation gate replays the
        suite hundreds of times with four concurrent workers, where each
        training run inflates to seconds under contention): the
        parameter-motion check kills a swapped coefficient, the descent
        check kills a sign-flip on either loss term.
        """
        distiller = self._distiller(alpha=0.0)
        data = torch.randn(8, 3, 2, 2)
        s1_before = self._params(distiller.student1)
        distiller.student2.eval()  # compare like-for-like batch-norm statistics
        with torch.no_grad():
            teacher_agg = distiller._aggregate_features(distiller.teacher(data))
            loss_before = float(
                distiller._distillation_loss(teacher_agg, distiller.student2(teacher_agg))
            )
        distiller.fit(data)
        assert not self._changed(s1_before, distiller.student1)
        distiller.student2.eval()
        with torch.no_grad():
            loss_after = float(
                distiller._distillation_loss(teacher_agg, distiller.student2(teacher_agg))
            )
        assert loss_after < loss_before

    def test_default_alpha_trains_both_students(self) -> None:
        distiller = self._distiller(alpha=0.5)
        s1_before = self._params(distiller.student1)
        s2_before = self._params(distiller.student2)
        distiller.fit(torch.randn(8, 3, 2, 2))
        assert self._changed(s1_before, distiller.student1)
        assert self._changed(s2_before, distiller.student2)

    @pytest.mark.parametrize("bad", [-0.1, 1.05, 1.5, float("nan"), float("inf")])
    def test_alpha_outside_the_convex_range_is_refused(self, bad: float) -> None:
        from omni_mercury_engine.ml.distillation.dual_student import DualStudentConfig

        with pytest.raises(ValueError, match="alpha"):
            DualStudentConfig(alpha=bad)

    def test_alpha_bounds_are_valid(self) -> None:
        from omni_mercury_engine.ml.distillation.dual_student import DualStudentConfig

        assert DualStudentConfig(alpha=0.0).alpha == 0.0
        assert DualStudentConfig(alpha=1.0).alpha == 1.0

    def test_temperature_is_reserved_and_unwired(self) -> None:
        """Cosine-feature distillation has no logits: T changes nothing.

        Two trainings identical except for ``temperature`` produce
        byte-identical students (maintainer decision: the knob is kept for
        config compatibility and documented as reserved, never wired).
        """
        results = []
        for temperature in (4.0, 0.5):
            distiller = self._distiller(alpha=0.5)
            distiller.config.temperature = temperature
            torch.manual_seed(123)  # fix the loader shuffle order
            distiller.fit(torch.randn(8, 3, 2, 2))
            results.append(
                [p.detach().clone() for p in distiller.student1.parameters()]
                + [p.detach().clone() for p in distiller.student2.parameters()]
            )
        assert all(torch.equal(a, b) for a, b in zip(results[0], results[1], strict=True))
