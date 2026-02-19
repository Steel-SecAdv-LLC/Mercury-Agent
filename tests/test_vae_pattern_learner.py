"""
Tests for VAE Pattern Learner module.

Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC
"""

import pytest  # noqa: E402
pytest.importorskip("torch")

import numpy as np
import torch

from omni_mercury_engine.ml.vae_pattern_learner import VAE, VAEPatternLearner


class TestVAE:
    """Tests for VAE class."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        vae = VAE(input_dim=10)
        assert vae.fc_mu.out_features == 32  # default latent_dim
        assert vae.fc_logvar.out_features == 32

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        vae = VAE(input_dim=20, latent_dim=64, hidden_dims=[256, 128])
        assert vae.fc_mu.out_features == 64
        assert vae.fc_logvar.out_features == 64

    def test_encode(self):
        """Test encoding input to latent distribution parameters."""
        vae = VAE(input_dim=10, latent_dim=16, hidden_dims=[32, 16])
        vae.eval()
        x = torch.randn(8, 10)  # batch=8, input_dim=10
        with torch.no_grad():
            mu, logvar = vae.encode(x)
        assert mu.shape == (8, 16)
        assert logvar.shape == (8, 16)

    def test_reparameterize(self):
        """Test reparameterization trick."""
        vae = VAE(input_dim=10, latent_dim=16)
        mu = torch.zeros(8, 16)
        logvar = torch.zeros(8, 16)
        z = vae.reparameterize(mu, logvar)
        assert z.shape == (8, 16)

    def test_reparameterize_deterministic_with_zero_variance(self):
        """Test reparameterization with zero variance."""
        vae = VAE(input_dim=10, latent_dim=16)
        mu = torch.ones(8, 16) * 5.0
        logvar = torch.ones(8, 16) * -100  # Very small variance
        z = vae.reparameterize(mu, logvar)
        assert z.shape == (8, 16)
        # With very small variance, z should be close to mu
        assert torch.allclose(z, mu, atol=0.1)

    def test_decode(self):
        """Test decoding latent vector to reconstruction."""
        vae = VAE(input_dim=10, latent_dim=16, hidden_dims=[32, 16])
        vae.eval()
        z = torch.randn(8, 16)
        with torch.no_grad():
            recon = vae.decode(z)
        assert recon.shape == (8, 10)

    def test_forward(self):
        """Test forward pass through VAE."""
        vae = VAE(input_dim=10, latent_dim=16, hidden_dims=[32, 16])
        vae.eval()
        x = torch.randn(8, 10)
        with torch.no_grad():
            recon, mu, logvar = vae.forward(x)
        assert recon.shape == (8, 10)
        assert mu.shape == (8, 16)
        assert logvar.shape == (8, 16)

    def test_compute_loss(self):
        """Test VAE loss computation."""
        vae = VAE(input_dim=10, latent_dim=16, hidden_dims=[32, 16])
        vae.eval()
        x = torch.randn(8, 10)
        with torch.no_grad():
            recon, mu, logvar = vae.forward(x)
            losses = vae.compute_loss(x, recon, mu, logvar)

        assert "total_loss" in losses
        assert "recon_loss" in losses
        assert "kl_loss" in losses
        assert losses["total_loss"] >= 0
        assert losses["recon_loss"] >= 0
        assert losses["kl_loss"] >= 0

    def test_compute_loss_with_beta(self):
        """Test VAE loss with different beta values."""
        vae = VAE(input_dim=10, latent_dim=16, hidden_dims=[32, 16])
        vae.eval()
        x = torch.randn(8, 10)
        with torch.no_grad():
            recon, mu, logvar = vae.forward(x)
            losses_beta1 = vae.compute_loss(x, recon, mu, logvar, beta=1.0)
            losses_beta10 = vae.compute_loss(x, recon, mu, logvar, beta=10.0)

        # Higher beta should give higher total loss (if KL > 0)
        if losses_beta1["kl_loss"] > 0:
            assert losses_beta10["total_loss"] >= losses_beta1["total_loss"]

    def test_anomaly_score(self):
        """Test anomaly score computation."""
        vae = VAE(input_dim=10, latent_dim=16, hidden_dims=[32, 16])
        vae.eval()
        x = torch.randn(8, 10)
        scores = vae.anomaly_score(x)
        assert scores.shape == (8,)
        assert torch.all(scores >= 0)

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        vae = VAE(input_dim=10, latent_dim=16, hidden_dims=[32, 16])
        vae.train()
        x = torch.randn(8, 10, requires_grad=True)
        recon, mu, logvar = vae.forward(x)
        losses = vae.compute_loss(x, recon, mu, logvar)
        losses["total_loss"].backward()
        assert x.grad is not None

    def test_different_batch_sizes(self):
        """Test model with different batch sizes."""
        vae = VAE(input_dim=10, latent_dim=16, hidden_dims=[32, 16])
        vae.eval()
        for batch_size in [2, 4, 8, 16, 32]:
            x = torch.randn(batch_size, 10)
            with torch.no_grad():
                recon, mu, logvar = vae.forward(x)
            assert recon.shape == (batch_size, 10)
            assert mu.shape == (batch_size, 16)
            assert logvar.shape == (batch_size, 16)


class TestVAEPatternLearner:
    """Tests for VAEPatternLearner class."""

    def test_init(self):
        """Test initialization."""
        learner = VAEPatternLearner(input_dim=10, latent_dim=16)
        assert learner.vae is not None
        assert learner.optimizer is not None
        assert learner.threshold is None

    def test_fit(self):
        """Test fitting on training data."""
        learner = VAEPatternLearner(input_dim=10, latent_dim=16)
        X_train = torch.randn(100, 10)
        result = learner.fit(X_train, epochs=2, batch_size=16)
        assert result is learner  # Returns self
        assert learner.threshold is not None

    def test_fit_with_beta(self):
        """Test fitting with different beta values."""
        learner = VAEPatternLearner(input_dim=10, latent_dim=16)
        X_train = torch.randn(100, 10)
        learner.fit(X_train, epochs=2, batch_size=16, beta=0.5)
        assert learner.threshold is not None

    def test_predict(self):
        """Test prediction on test data."""
        learner = VAEPatternLearner(input_dim=10, latent_dim=16)
        X_train = torch.randn(100, 10)
        learner.fit(X_train, epochs=2, batch_size=16)

        X_test = torch.randn(20, 10)
        result = learner.predict(X_test)

        assert "anomaly_scores" in result
        assert "is_anomaly" in result
        assert "threshold" in result
        assert result["anomaly_scores"].shape == (20,)
        assert result["is_anomaly"].shape == (20,)

    def test_predict_before_fit(self):
        """Test prediction before fitting (threshold is None)."""
        learner = VAEPatternLearner(input_dim=10, latent_dim=16)
        X_test = torch.randn(20, 10)
        result = learner.predict(X_test)

        assert result["threshold"] is None
        # Should use default threshold of 0.5
        assert "is_anomaly" in result

    def test_anomaly_detection(self):
        """Test that anomalies are detected."""
        learner = VAEPatternLearner(input_dim=10, latent_dim=16)

        # Train on normal data (centered around 0)
        X_train = torch.randn(200, 10) * 0.5
        learner.fit(X_train, epochs=5, batch_size=32)

        # Test on normal data
        X_normal = torch.randn(50, 10) * 0.5
        result_normal = learner.predict(X_normal)

        # Test on anomalous data (far from training distribution)
        X_anomaly = torch.randn(50, 10) * 5.0 + 10.0
        result_anomaly = learner.predict(X_anomaly)

        # Anomalous data should have higher scores on average
        assert np.mean(result_anomaly["anomaly_scores"]) > np.mean(result_normal["anomaly_scores"])

    def test_fit_returns_self(self):
        """Test that fit returns self for method chaining."""
        learner = VAEPatternLearner(input_dim=10, latent_dim=16)
        X_train = torch.randn(100, 10)
        result = learner.fit(X_train, epochs=1, batch_size=16)
        assert result is learner


class TestVAEEdgeCases:
    """Edge case tests for VAE."""

    def test_single_hidden_layer(self):
        """Test with single hidden layer."""
        vae = VAE(input_dim=10, latent_dim=8, hidden_dims=[32])
        vae.eval()
        x = torch.randn(8, 10)
        with torch.no_grad():
            recon, mu, logvar = vae.forward(x)
        assert recon.shape == (8, 10)

    def test_many_hidden_layers(self):
        """Test with many hidden layers."""
        vae = VAE(input_dim=10, latent_dim=8, hidden_dims=[128, 64, 32, 16])
        vae.eval()
        x = torch.randn(8, 10)
        with torch.no_grad():
            recon, mu, logvar = vae.forward(x)
        assert recon.shape == (8, 10)

    def test_large_latent_dim(self):
        """Test with large latent dimension."""
        vae = VAE(input_dim=10, latent_dim=128, hidden_dims=[64, 32])
        vae.eval()
        x = torch.randn(8, 10)
        with torch.no_grad():
            recon, mu, logvar = vae.forward(x)
        assert mu.shape == (8, 128)

    def test_small_input_dim(self):
        """Test with small input dimension."""
        vae = VAE(input_dim=2, latent_dim=4, hidden_dims=[8, 4])
        vae.eval()
        x = torch.randn(8, 2)
        with torch.no_grad():
            recon, mu, logvar = vae.forward(x)
        assert recon.shape == (8, 2)

    def test_large_input_dim(self):
        """Test with large input dimension."""
        vae = VAE(input_dim=1000, latent_dim=32, hidden_dims=[256, 128])
        vae.eval()
        x = torch.randn(4, 1000)
        with torch.no_grad():
            recon, mu, logvar = vae.forward(x)
        assert recon.shape == (4, 1000)
