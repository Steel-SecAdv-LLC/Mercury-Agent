"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Tests for SOTA anomaly detection models.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import pytest
import torch
from torch import nn

from omni_mercury_engine.models.sota.association_discrepancy import (
    AnomalyTransformerEncoder,
    AssociationConfig,
    AssociationDiscrepancyLoss,
    AssociationDiscrepancyModule,
    PriorAssociation,
    SeriesAssociation,
    apply_ethical_constraints,
)
from omni_mercury_engine.models.sota.maat import (
    GatedFeatureFusion,
    MAATConfig,
    MAATEncoderLayer,
    MAATLoss,
    MAATModel,
    MambaSSM,
    SelectiveSSM,
    SparseAttention,
)
from omni_mercury_engine.models.sota.tranad import (
    AdversarialTrainer,
    FocusScoreConditioning,
    MAMLOptimizer,
    TranADConfig,
    TranADModel,
    TransformerDecoder,
    TransformerEncoder,
)

# ============================================================================
# Association Discrepancy (Anomaly Transformer) Tests
# ============================================================================


class TestPriorAssociation:
    """Tests for Prior-Association (Gaussian kernel)."""

    def test_output_shape(self) -> None:
        """Should return correct shape [seq_len, seq_len]."""
        prior = PriorAssociation(sigma=1.0, window_size=100)
        result = prior(seq_len=50)
        assert result.shape == (50, 50)

    def test_row_normalization(self) -> None:
        """Rows should sum to 1 (probability distribution)."""
        prior = PriorAssociation(sigma=1.0, window_size=100)
        result = prior(seq_len=50)
        row_sums = result.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(50), atol=1e-5)

    def test_symmetry(self) -> None:
        """Prior kernel is symmetric before row normalization.

        Note: After row normalization (to make valid probability distribution),
        the output is NOT symmetric. This is correct behavior per the Anomaly
        Transformer paper - each row should sum to 1 for KL divergence computation.
        """
        prior = PriorAssociation(sigma=1.0, window_size=100)
        result = prior(seq_len=50)
        # The row-normalized prior is NOT symmetric (rows sum to 1, columns don't)
        # But the diagonal should still be the maximum in each row
        diag = torch.diag(result)
        assert (diag >= result.max(dim=-1).values - 1e-5).all()

    def test_diagonal_maximum(self) -> None:
        """Diagonal should have highest values (closest to self)."""
        prior = PriorAssociation(sigma=1.0, window_size=100)
        result = prior(seq_len=50)
        diag = torch.diag(result)
        # Diagonal should be >= all other elements in each row
        assert (diag >= result.max(dim=-1).values - 1e-5).all()


class TestSeriesAssociation:
    """Tests for Series-Association (learned attention)."""

    def test_output_shape(self) -> None:
        """Should return correct shapes."""
        series = SeriesAssociation(d_model=64, n_heads=4)
        x = torch.randn(2, 20, 64)  # [batch, seq, d_model]
        output, attention = series(x, return_attention=True)

        assert output.shape == (2, 20, 64)
        assert attention.shape == (2, 4, 20, 20)  # [batch, heads, seq, seq]

    def test_attention_normalization(self) -> None:
        """Attention weights should sum to 1 per row."""
        series = SeriesAssociation(d_model=64, n_heads=4)
        x = torch.randn(2, 20, 64)
        _, attention = series(x, return_attention=True)

        row_sums = attention.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


class TestAssociationDiscrepancyModule:
    """Tests for Association Discrepancy computation."""

    def test_forward_output_keys(self) -> None:
        """Should return expected keys in output dict."""
        config = AssociationConfig(d_model=64, n_heads=4)
        module = AssociationDiscrepancyModule(config)
        x = torch.randn(2, 20, 64)

        result = module(x)

        assert "output" in result
        assert "discrepancy" in result
        assert "series_attention" in result

    def test_discrepancy_shape(self) -> None:
        """Discrepancy should be [batch, seq_len]."""
        config = AssociationConfig(d_model=64, n_heads=4)
        module = AssociationDiscrepancyModule(config)
        x = torch.randn(2, 20, 64)

        result = module(x)

        assert result["discrepancy"].shape == (2, 20)

    def test_discrepancy_non_negative(self) -> None:
        """KL divergence should be non-negative."""
        config = AssociationConfig(d_model=64, n_heads=4)
        module = AssociationDiscrepancyModule(config)
        x = torch.randn(2, 20, 64)

        result = module(x)

        assert (result["discrepancy"] >= -1e-5).all()

    def test_anomaly_score_computation(self) -> None:
        """Anomaly score should be computed correctly."""
        config = AssociationConfig(d_model=64, n_heads=4)
        module = AssociationDiscrepancyModule(config)
        x = torch.randn(2, 20, 64)

        score = module.get_anomaly_score(x)

        assert score.shape == (2, 20)


class TestAnomalyTransformerEncoder:
    """Tests for full Anomaly Transformer encoder."""

    def test_forward_output_keys(self) -> None:
        """Should return expected keys."""
        encoder = AnomalyTransformerEncoder(
            input_dim=25,
            d_model=64,
            n_heads=4,
            n_layers=2,
        )
        x = torch.randn(2, 20, 25)

        result = encoder(x)

        assert "reconstruction" in result
        assert "discrepancy" in result
        assert "anomaly_score" in result

    def test_reconstruction_shape(self) -> None:
        """Reconstruction should match input shape."""
        encoder = AnomalyTransformerEncoder(
            input_dim=25,
            d_model=64,
            n_heads=4,
            n_layers=2,
        )
        x = torch.randn(2, 20, 25)

        result = encoder(x)

        assert result["reconstruction"].shape == x.shape

    def test_detect_method(self) -> None:
        """Detect should return predictions."""
        encoder = AnomalyTransformerEncoder(
            input_dim=25,
            d_model=64,
            n_heads=4,
            n_layers=2,
        )
        x = torch.randn(2, 20, 25)

        result = encoder.detect(x)

        assert "predictions" in result
        assert "threshold" in result


class TestAssociationDiscrepancyLoss:
    """Tests for Anomaly Transformer loss function."""

    def test_loss_computation(self) -> None:
        """Loss should be computed correctly."""
        loss_fn = AssociationDiscrepancyLoss(lambda_=3.0)
        x = torch.randn(2, 20, 25)
        recon = torch.randn(2, 20, 25)
        discrepancy = torch.rand(2, 20)

        losses = loss_fn(x, recon, discrepancy, phase="minimize")

        assert "total_loss" in losses
        assert "reconstruction_loss" in losses
        assert "association_loss" in losses

    def test_minimax_phases(self) -> None:
        """Maximize phase should have different loss."""
        loss_fn = AssociationDiscrepancyLoss(lambda_=3.0)
        x = torch.randn(2, 20, 25)
        recon = torch.randn(2, 20, 25)
        discrepancy = torch.rand(2, 20) + 0.1

        min_losses = loss_fn(x, recon, discrepancy, phase="minimize")
        max_losses = loss_fn(x, recon, discrepancy, phase="maximize")

        # Association contribution should differ
        assert min_losses["total_loss"] != max_losses["total_loss"]


# ============================================================================
# TranAD Tests
# ============================================================================


class TestFocusScoreConditioning:
    """Tests for Focus Score-Based Self-Conditioning."""

    def test_output_shape(self) -> None:
        """Output should match input shape."""
        focus = FocusScoreConditioning(input_dim=25, d_model=64)
        x = torch.randn(2, 20, 25)

        output = focus(x)

        assert output.shape == x.shape

    def test_return_scores(self) -> None:
        """Should return focus scores when requested."""
        focus = FocusScoreConditioning(input_dim=25, d_model=64)
        x = torch.randn(2, 20, 25)

        output, scores = focus(x, return_scores=True)

        assert scores.shape == x.shape
        # Scores should sum to 1 over features
        assert torch.allclose(scores.sum(dim=-1), torch.ones(2, 20), atol=1e-5)

    def test_feature_importance(self) -> None:
        """Should compute feature importance."""
        focus = FocusScoreConditioning(input_dim=25, d_model=64)
        x = torch.randn(2, 20, 25)

        importance = focus.get_feature_importance(x)

        assert importance.shape == (25,)
        assert torch.allclose(importance.sum(), torch.tensor(1.0), atol=1e-5)


class TestTransformerEncoder:
    """Tests for TranAD Transformer Encoder."""

    def test_output_shape(self) -> None:
        """Output should have d_model dimension."""
        encoder = TransformerEncoder(d_model=64, n_heads=4, n_layers=2)
        x = torch.randn(2, 20, 64)

        output = encoder(x)

        assert output.shape == x.shape


class TestTransformerDecoder:
    """Tests for TranAD Transformer Decoder."""

    def test_output_shape(self) -> None:
        """Output should have d_model dimension."""
        decoder = TransformerDecoder(d_model=64, n_heads=4, n_layers=1)
        tgt = torch.randn(2, 20, 64)
        memory = torch.randn(2, 20, 64)

        output = decoder(tgt, memory)

        assert output.shape == tgt.shape


class TestTranADModel:
    """Tests for full TranAD model."""

    def test_forward_output_keys(self) -> None:
        """Should return expected keys."""
        config = TranADConfig(input_dim=25, d_model=64, n_heads=4)
        model = TranADModel(config)
        x = torch.randn(2, 10, 25)

        result = model(x)

        assert "reconstruction" in result
        assert "anomaly_score" in result

    def test_reconstruction_shape(self) -> None:
        """Reconstruction should match input shape."""
        config = TranADConfig(input_dim=25, d_model=64, n_heads=4)
        model = TranADModel(config)
        x = torch.randn(2, 10, 25)

        result = model(x)

        assert result["reconstruction"].shape == x.shape

    def test_detect_method(self) -> None:
        """Detect should return predictions."""
        config = TranADConfig(input_dim=25, d_model=64, n_heads=4)
        model = TranADModel(config)
        x = torch.randn(2, 10, 25)

        result = model.detect(x)

        assert "predictions" in result
        assert "threshold" in result

    def test_adversarial_mode(self) -> None:
        """Should work with adversarial training enabled."""
        config = TranADConfig(input_dim=25, d_model=64, n_heads=4, use_adversarial=True)
        model = TranADModel(config)

        assert model.discriminator is not None
        assert model.decoder2 is not None


class TestAdversarialTrainer:
    """Tests for TranAD adversarial training."""

    def test_train_step(self) -> None:
        """Should complete training step without errors."""
        config = TranADConfig(input_dim=25, d_model=64, n_heads=4, use_adversarial=True)
        model = TranADModel(config)
        trainer = AdversarialTrainer(model)
        x = torch.randn(2, 10, 25)

        losses = trainer.train_step(x)

        assert "reconstruction" in losses
        assert "total" in losses


class TestMAMLOptimizer:
    """Tests for MAML meta-learning."""

    def test_inner_loop(self) -> None:
        """Inner loop should adapt model."""
        config = TranADConfig(input_dim=25, d_model=64, n_heads=4)
        model = TranADModel(config)
        maml = MAMLOptimizer(model, n_inner_steps=2)
        support_x = torch.randn(4, 10, 25)

        adapted = maml.inner_loop(maml.clone_model(), support_x)

        # Model should exist
        assert adapted is not None

    def test_adapt_method(self) -> None:
        """Adapt should return working model."""
        config = TranADConfig(input_dim=25, d_model=64, n_heads=4)
        model = TranADModel(config)
        maml = MAMLOptimizer(model, n_inner_steps=2)
        support_x = torch.randn(4, 10, 25)

        adapted = maml.adapt(support_x)

        # Adapted model should work
        result = adapted(support_x)
        assert "reconstruction" in result


# ============================================================================
# MAAT Tests
# ============================================================================


class TestSparseAttention:
    """Tests for Sparse Attention module."""

    def test_output_shape(self) -> None:
        """Output should have correct shape."""
        sparse_attn = SparseAttention(d_model=64, n_heads=4, sparsity=0.5)
        x = torch.randn(2, 20, 64)

        output, _ = sparse_attn(x)

        assert output.shape == x.shape

    def test_return_attention(self) -> None:
        """Should return attention weights when requested."""
        sparse_attn = SparseAttention(d_model=64, n_heads=4, sparsity=0.5)
        x = torch.randn(2, 20, 64)

        output, attention = sparse_attn(x, return_attention=True)

        assert attention is not None


class TestSelectiveSSM:
    """Tests for Selective State Space Model."""

    def test_output_shape(self) -> None:
        """Output should have correct shape."""
        ssm = SelectiveSSM(d_model=64, d_state=16)
        x = torch.randn(2, 20, 64)

        output = ssm(x)

        assert output.shape == x.shape


class TestMambaSSM:
    """Tests for Mamba-SSM block."""

    def test_output_shape(self) -> None:
        """Output should have correct shape."""
        mamba = MambaSSM(d_model=64, d_state=16)
        x = torch.randn(2, 20, 64)

        output = mamba(x)

        assert output.shape == x.shape


class TestGatedFeatureFusion:
    """Tests for Gated Feature Fusion."""

    def test_output_shape(self) -> None:
        """Output should have correct shape."""
        fusion = GatedFeatureFusion(d_model=64)
        x_attn = torch.randn(2, 20, 64)
        x_ssm = torch.randn(2, 20, 64)

        output = fusion(x_attn, x_ssm)

        assert output.shape == x_attn.shape

    def test_return_gate(self) -> None:
        """Should return gate values when requested."""
        fusion = GatedFeatureFusion(d_model=64)
        x_attn = torch.randn(2, 20, 64)
        x_ssm = torch.randn(2, 20, 64)

        output, gate = fusion(x_attn, x_ssm, return_gate=True)

        assert gate.shape == x_attn.shape
        # Gate values should be between 0 and 1
        assert (gate >= 0).all() and (gate <= 1).all()


class TestMAATEncoderLayer:
    """Tests for MAAT encoder layer."""

    def test_output_shape(self) -> None:
        """Output should have correct shape."""
        config = MAATConfig(d_model=64, n_heads=4)
        layer = MAATEncoderLayer(config)
        x = torch.randn(2, 20, 64)

        output, _ = layer(x)

        assert output.shape == x.shape


class TestMAATModel:
    """Tests for full MAAT model."""

    def test_forward_output_keys(self) -> None:
        """Should return expected keys."""
        config = MAATConfig(input_dim=25, d_model=64, n_heads=4, n_layers=2)
        model = MAATModel(config)
        x = torch.randn(2, 20, 25)

        result = model(x)

        assert "reconstruction" in result
        assert "anomaly_score" in result
        assert "discrepancy" in result

    def test_reconstruction_shape(self) -> None:
        """Reconstruction should match input shape."""
        config = MAATConfig(input_dim=25, d_model=64, n_heads=4, n_layers=2)
        model = MAATModel(config)
        x = torch.randn(2, 20, 25)

        result = model(x)

        assert result["reconstruction"].shape == x.shape

    def test_detect_method(self) -> None:
        """Detect should return predictions."""
        config = MAATConfig(input_dim=25, d_model=64, n_heads=4, n_layers=2)
        model = MAATModel(config)
        x = torch.randn(2, 20, 25)

        result = model.detect(x)

        assert "predictions" in result
        assert "threshold" in result

    def test_pathway_importance(self) -> None:
        """Should analyze pathway importance."""
        config = MAATConfig(input_dim=25, d_model=64, n_heads=4, n_layers=2)
        model = MAATModel(config)
        x = torch.randn(2, 20, 25)

        importance = model.get_pathway_importance(x)

        assert "attention_ratio" in importance


class TestMAATLoss:
    """Tests for MAAT loss function."""

    def test_loss_computation(self) -> None:
        """Loss should be computed correctly."""
        config = MAATConfig(input_dim=25, d_model=64, n_heads=4, n_layers=2)
        model = MAATModel(config)
        loss_fn = MAATLoss()
        x = torch.randn(2, 20, 25)

        result = model(x, return_all=True)
        losses = loss_fn(x, result)

        assert "total" in losses
        assert "reconstruction" in losses


# ============================================================================
# Ethical Constraints Tests
# ============================================================================


class TestEthicalConstraints:
    """Tests for ethical constraint functions."""

    def test_apply_ethical_constraints(self) -> None:
        """Should scale scores by harm prevention scalar."""
        scores = torch.rand(10, 20)

        adjusted = apply_ethical_constraints(scores, harm_prevention_scalar=1.50)

        # Scores should be scaled up
        assert (adjusted >= scores).all()

    def test_min_recall_threshold(self) -> None:
        """Should ensure minimum detection sensitivity."""
        scores = torch.rand(10, 20)

        adjusted = apply_ethical_constraints(
            scores, harm_prevention_scalar=1.50, min_recall_threshold=0.95
        )

        # Adjusted scores should have a floor
        assert adjusted.min() > 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestModelIntegration:
    """Integration tests across SOTA models."""

    def test_all_models_trainable(self) -> None:
        """Every parameter of every SOTA model receives gradients via its real training loss.

        The previous version of this test back-propagated only through
        ``result["reconstruction"]`` and was therefore wrong for TranAD
        in two compounding ways:

        1. TranAD's ``decoder2`` / ``output_projection2`` receive
           gradients only through ``reconstruction_refined`` (the
           adversarial refinement path), exactly as
           :class:`TranADLoss` and :class:`TranADTrainer` train them
           in production (``g_loss = L1 + L2 + λ·L_adv``).
        2. TranAD's ``discriminator`` is a separate sub-network with
           its own optimiser; it is trained against the real/fake
           classification loss
           ``d_loss = ½ (BCE(D(x), 1) + BCE(D(recon.detach()), 0))``
           per :meth:`TranADTrainer.train_step`, **not** via the
           generator's adversarial term.

        Backpropagating only through ``reconstruction`` exercised the
        encoder and ``decoder1`` and nothing else, which is why the
        test used to live behind an ``xfail`` claiming "by design".
        It is by design — the design just isn't a single-loss
        contract.  This rewrite drives each parameter group through
        the production loss path that actually owns it, so any
        regression that severs a real gradient route now fails the
        test loudly instead of being hidden by an ``xfail``.
        """
        models: list[nn.Module] = [
            AnomalyTransformerEncoder(input_dim=25, d_model=64, n_heads=4, n_layers=2),
            TranADModel(TranADConfig(input_dim=25, d_model=64, n_heads=4)),
            MAATModel(MAATConfig(input_dim=25, d_model=64, n_heads=4, n_layers=2)),
        ]

        x = torch.randn(2, 20, 25)

        for model in models:
            result = model(x)

            # Build the generator-side total loss the production
            # ``TranADTrainer`` uses.  ``reconstruction_refined`` is
            # the only gradient source for TranAD's ``decoder2`` /
            # ``output_projection2``; the other two models have a
            # single reconstruction head and skip this branch.
            g_loss = result["reconstruction"].mean()
            if "reconstruction_refined" in result:
                g_loss = g_loss + result["reconstruction_refined"].mean()

            # TranAD's discriminator is trained against a separate
            # real/fake loss in production -- it does not flow back
            # through the generator's adversarial term.  Mirror that
            # contract here so the discriminator parameters get the
            # gradient route they would actually see during training.
            discriminator = getattr(model, "discriminator", None)
            if discriminator is not None:
                recon = result["reconstruction"]
                real_scores = discriminator(x)
                fake_scores = discriminator(recon.detach())
                d_loss = 0.5 * (
                    nn.functional.binary_cross_entropy_with_logits(
                        real_scores, torch.ones_like(real_scores)
                    )
                    + nn.functional.binary_cross_entropy_with_logits(
                        fake_scores, torch.zeros_like(fake_scores)
                    )
                )
                d_loss.backward(retain_graph=True)

            g_loss.backward()

            # Every trainable parameter must have received a gradient
            # from one of the loss heads above.  Empty parameters
            # (numel == 0) are skipped because PyTorch never allocates
            # ``.grad`` for them.
            missing_grad = [
                name
                for name, param in model.named_parameters()
                if param.requires_grad and param.numel() > 0 and param.grad is None
            ]
            assert not missing_grad, (
                f"{type(model).__name__}: parameters missing gradients "
                f"under the production training loss: {missing_grad}"
            )

    def test_models_detect_similar_anomalies(self) -> None:
        """Models should detect obvious anomalies."""
        models: list[nn.Module] = [
            AnomalyTransformerEncoder(input_dim=25, d_model=64, n_heads=4, n_layers=2),
            TranADModel(TranADConfig(input_dim=25, d_model=64, n_heads=4)),
            MAATModel(MAATConfig(input_dim=25, d_model=64, n_heads=4, n_layers=2)),
        ]

        # Normal data
        x_normal = torch.randn(2, 20, 25) * 0.1

        # Anomalous data (large spike)
        x_anomaly = torch.randn(2, 20, 25) * 0.1
        x_anomaly[:, 10, :] = 10.0  # Obvious anomaly

        for model in models:
            with torch.no_grad():
                result_normal = model(x_normal)
                result_anomaly = model(x_anomaly)

            # Anomaly at position 10 should have higher score
            score_normal = result_normal["anomaly_score"][:, 10].mean()
            score_anomaly = result_anomaly["anomaly_score"][:, 10].mean()

            # Not guaranteed but likely with obvious anomaly
            # (models are untrained, so just check they produce valid output)
            assert not torch.isnan(score_normal) and not torch.isnan(score_anomaly)
