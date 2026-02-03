"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""

from __future__ import annotations


"""
Tests for Cortical-Laminated Neural Network Architecture

Tests cover:
1. Cortical layer structure (6 layers)
2. Sparse coding functionality
3. Lateral inhibition effects
4. Hebbian learning updates
5. Thalamocortical gating
6. Brain stain analyzers (Golgi, Nissl, Weigert)
7. CorticalLoss function components
"""

import torch
from numpy.testing import assert_allclose

from omni_mercury_engine.ml.cortical_network import (
    CorticalColumn,
    CorticalConfig,
    CorticalLaminatedNetwork,
    CorticalLoss,
    GolgiAnalyzer,
    HebbianLearningRule,
    LateralInhibition,
    NisslAnalyzer,
    SparseCoding,
    SpikeTimingDependentPlasticity,
    ThalamocorticalGate,
    WeigertAnalyzer,
)


class TestSparseCoding:
    """Test sparse coding module."""

    def test_sparsity_level_maintained(self):
        """Output should have approximately target sparsity."""
        sparse_coder = SparseCoding(sparsity=0.1)

        # Input with all positive values
        x = torch.rand(10, 100)  # batch=10, features=100

        output = sparse_coder(x)

        # Count non-zero elements
        non_zero_ratio = (output.abs() > 1e-6).float().mean(dim=1)

        # Should be approximately 10% non-zero (sparsity=0.1)
        assert_allclose(non_zero_ratio.mean().item(), 0.1, atol=0.05)

    def test_sparse_output_shape(self):
        """Output shape should match input shape."""
        sparse_coder = SparseCoding(sparsity=0.2)
        x = torch.randn(5, 50)

        output = sparse_coder(x)

        assert output.shape == x.shape

    def test_sparse_preserves_top_k(self):
        """Top-k values should be preserved."""
        sparse_coder = SparseCoding(sparsity=0.2, temperature=0.1)

        # Create input with known ordering
        x = torch.arange(10).float().unsqueeze(0)  # [1, 10]

        output = sparse_coder(x)

        # With sparsity=0.2, should keep top 2 values (indices 8, 9)
        non_zero_indices = torch.where(output[0] > 0)[0]
        assert 9 in non_zero_indices  # Highest value preserved


class TestLateralInhibition:
    """Test lateral inhibition module."""

    def test_lateral_inhibition_shape(self):
        """Output shape should match input shape."""
        lateral = LateralInhibition(features=64, strength=0.5)
        x = torch.randn(8, 64)

        output = lateral(x)

        assert output.shape == x.shape

    def test_inhibition_reduces_spread(self):
        """Lateral inhibition should sharpen activations."""
        lateral = LateralInhibition(features=32, strength=0.8)

        # Input with broad activation
        x = torch.ones(1, 32) * 0.5
        x[0, 15:17] = 1.0  # Peak in middle

        output = lateral(x)

        # Peak should be relatively preserved, surroundings reduced
        peak_values = output[0, 15:17].mean().item()
        surround_values = torch.cat([output[0, :15], output[0, 17:]]).mean().item()

        # Peak should be higher than surroundings
        assert peak_values > surround_values

    def test_strength_zero_is_identity(self):
        """With strength=0, output should match input."""
        lateral = LateralInhibition(features=20, strength=0.0)
        x = torch.randn(4, 20)

        output = lateral(x)

        assert_allclose(output.numpy(), x.numpy(), rtol=1e-5)


class TestHebbianLearning:
    """Test Hebbian learning rule module."""

    def test_hebbian_weight_shape(self):
        """Weight matrix should have correct shape."""
        hebbian = HebbianLearningRule(input_dim=64, output_dim=32)

        assert hebbian.weight.shape == (32, 64)

    def test_hebbian_forward(self):
        """Forward pass should compute linear transformation."""
        hebbian = HebbianLearningRule(input_dim=10, output_dim=5)
        x = torch.randn(8, 10)

        output = hebbian(x)

        assert output.shape == (8, 5)

    def test_hebbian_update_direction(self):
        """Hebbian update should strengthen correlated connections."""
        hebbian = HebbianLearningRule(
            input_dim=4, output_dim=2, learning_rate=0.1, weight_decay=0.0
        )

        # Correlated pre and post-synaptic activity
        x = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # Only first input active
        y = torch.tensor([[1.0, 0.0]])  # Only first output active

        # Get current weight
        initial_weight_00 = hebbian.weight[0, 0].item()

        # Apply Hebbian update
        delta = hebbian.hebbian_update(x, y)

        # Weight connecting active pre to active post should increase
        assert delta[0, 0].item() > 0

    def test_hebbian_update_inplace(self):
        """In-place Hebbian update should modify weights."""
        hebbian = HebbianLearningRule(input_dim=8, output_dim=4, learning_rate=0.1)

        x = torch.randn(16, 8)
        y = hebbian(x)

        initial_weights = hebbian.weight.clone()

        hebbian.apply_hebbian_update(x, y)

        # Weights should have changed
        assert not torch.allclose(hebbian.weight, initial_weights)


class TestThalamocorticalGate:
    """Test thalamocortical gating mechanism."""

    def test_gate_output_shape(self):
        """Output should have correct shape."""
        gate = ThalamocorticalGate(input_dim=32, hidden_dim=64)
        sensory = torch.randn(8, 32)

        output = gate(sensory)

        assert output.shape == (8, 64)

    def test_gate_with_feedback(self):
        """Should incorporate cortical feedback."""
        gate = ThalamocorticalGate(input_dim=32, hidden_dim=64, feedback_dim=64)

        sensory = torch.randn(8, 32)
        feedback = torch.randn(8, 64)

        output_no_feedback = gate(sensory)
        output_with_feedback = gate(sensory, feedback)

        # Outputs should differ when feedback is provided
        assert not torch.allclose(output_no_feedback, output_with_feedback)

    def test_gate_modulates_input(self):
        """Gate should modulate sensory input based on feedback."""
        gate = ThalamocorticalGate(input_dim=16, hidden_dim=32, feedback_dim=32)

        # Same sensory input
        sensory = torch.ones(1, 16)

        # Different feedbacks should produce different outputs
        feedback_1 = torch.zeros(1, 32)
        feedback_2 = torch.ones(1, 32)

        output_1 = gate(sensory, feedback_1)
        output_2 = gate(sensory, feedback_2)

        assert not torch.allclose(output_1, output_2)


class TestCorticalColumn:
    """Test single cortical column."""

    def test_column_forward_shape(self):
        """Output should have correct shape."""
        config = CorticalConfig(input_dim=64, hidden_dim=128, output_dim=64)
        column = CorticalColumn(config)

        x = torch.randn(8, 64)
        output = column(x)

        assert output.shape == (8, 64)

    def test_column_returns_layer_activations(self):
        """Should return intermediate layer activations."""
        config = CorticalConfig(input_dim=32, hidden_dim=64, output_dim=32)
        column = CorticalColumn(config)

        x = torch.randn(4, 32)
        output, activations = column(x, return_layer_activations=True)

        # Should have activations for all layers
        assert "layer_i" in activations
        assert "layer_ii_iii" in activations
        assert "layer_iv" in activations
        assert "layer_v" in activations
        assert "layer_vi" in activations

    def test_column_feedback_modulation(self):
        """Feedback should modulate processing."""
        config = CorticalConfig(
            input_dim=32, hidden_dim=64, output_dim=32, feedback_strength=0.5
        )
        column = CorticalColumn(config)

        x = torch.randn(4, 32)

        output_no_fb = column(x, feedback=None)

        # Get Layer VI dimensions for feedback
        _, acts = column(x, return_layer_activations=True)
        feedback = torch.randn(4, acts["layer_vi"].shape[1])

        output_with_fb = column(x, feedback=feedback)

        # Should differ
        assert not torch.allclose(output_no_fb, output_with_fb, atol=1e-5)


class TestCorticalLaminatedNetwork:
    """Test full cortical network."""

    def test_network_forward(self):
        """Full network forward pass should work."""
        config = CorticalConfig(input_dim=64, hidden_dim=128, output_dim=64)
        network = CorticalLaminatedNetwork(config, num_columns=3)

        x = torch.randn(8, 64)
        output = network(x)

        assert output.shape == (8, 64)

    def test_network_with_activations(self):
        """Should return all activations when requested."""
        config = CorticalConfig(input_dim=32, hidden_dim=64, output_dim=32)
        network = CorticalLaminatedNetwork(config, num_columns=2)

        x = torch.randn(4, 32)
        result = network(x, return_all_activations=True)

        assert "output" in result
        assert "activations" in result
        assert "column_0" in result["activations"]
        assert "column_1" in result["activations"]

    def test_network_hebbian_during_training(self):
        """Hebbian learning should be applied during training."""
        config = CorticalConfig(input_dim=32, hidden_dim=64, output_dim=32)
        network = CorticalLaminatedNetwork(config, num_columns=2, use_hebbian=True)

        network.train()

        x = torch.randn(8, 32)
        initial_weights = network.hebbian.weight.clone()

        # Forward pass triggers Hebbian update in training mode
        _ = network(x)

        # Weights should have changed
        assert not torch.allclose(network.hebbian.weight, initial_weights)

    def test_network_no_hebbian_during_eval(self):
        """Hebbian learning should not be applied during eval."""
        config = CorticalConfig(input_dim=32, hidden_dim=64, output_dim=32)
        network = CorticalLaminatedNetwork(config, num_columns=2, use_hebbian=True)

        network.eval()

        x = torch.randn(8, 32)
        initial_weights = network.hebbian.weight.clone()

        # Forward pass should not update Hebbian weights in eval mode
        _ = network(x)

        assert torch.allclose(network.hebbian.weight, initial_weights)


class TestGolgiAnalyzer:
    """Test Golgi stain-inspired analysis."""

    def test_analyze_connectivity(self):
        """Should analyze network connectivity patterns."""
        # Create simple network
        model = torch.nn.Sequential(
            torch.nn.Linear(32, 64), torch.nn.ReLU(), torch.nn.Linear(64, 32)
        )

        analyzer = GolgiAnalyzer(model)
        results = analyzer.analyze_connectivity()

        assert "layer_connectivity" in results
        assert "receptive_field_sizes" in results
        assert "pathway_strengths" in results

        # Should have results for the linear layers
        assert len(results["layer_connectivity"]) > 0

    def test_visualize_dendrite_tree(self):
        """Should return weight matrix for visualization."""
        model = torch.nn.Sequential(torch.nn.Linear(16, 32))

        analyzer = GolgiAnalyzer(model)
        weight_matrix = analyzer.visualize_dendrite_tree("0")

        assert weight_matrix.shape == (32, 16)


class TestNisslAnalyzer:
    """Test Nissl stain-inspired analysis."""

    def test_capture_activations(self):
        """Should capture layer activations."""
        model = torch.nn.Sequential(
            torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 8)
        )

        analyzer = NisslAnalyzer(model)
        analyzer.register_hooks()

        # Run forward pass
        x = torch.randn(4, 16)
        _ = model(x)

        # Should have captured activations
        assert len(analyzer.activation_history) > 0

        analyzer.remove_hooks()

    def test_analyze_activations(self):
        """Should analyze activation patterns."""
        model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU())

        analyzer = NisslAnalyzer(model)
        analyzer.register_hooks()

        # Run multiple forward passes
        for _ in range(10):
            x = torch.randn(8, 16)
            _ = model(x)

        results = analyzer.analyze_activations()

        for _name, metrics in results.items():
            assert "sparsity" in metrics
            assert "mean" in metrics
            assert "std" in metrics

        analyzer.remove_hooks()


class TestWeigertAnalyzer:
    """Test Weigert stain-inspired analysis."""

    def test_analyze_connections(self):
        """Should analyze connection strength patterns."""
        model = torch.nn.Sequential(
            torch.nn.Linear(32, 64), torch.nn.ReLU(), torch.nn.Linear(64, 16)
        )

        analyzer = WeigertAnalyzer(model)
        results = analyzer.analyze_connections()

        assert "layer_metrics" in results
        assert "bottleneck_layers" in results

        # Should have metrics for linear layers
        for _name, metrics in results["layer_metrics"].items():
            assert "strong_connection_ratio" in metrics
            assert "weak_connection_ratio" in metrics
            assert "weight_entropy" in metrics


class TestCorticalLoss:
    """Test cortical loss function."""

    def test_loss_components(self):
        """Should compute all loss components."""
        loss_fn = CorticalLoss(
            task_weight=1.0, sparsity_weight=0.1, hebbian_weight=0.01, target_sparsity=0.1
        )

        predictions = torch.randn(8, 10)
        targets = torch.randint(0, 10, (8,))

        activations = {"layer_1": torch.randn(8, 64), "layer_2": torch.randn(8, 32)}

        loss_dict = loss_fn(predictions, targets, activations, task_type="classification")

        assert "total" in loss_dict
        assert "task" in loss_dict
        assert "sparsity" in loss_dict
        assert "hebbian" in loss_dict

    def test_loss_regression_mode(self):
        """Should work in regression mode."""
        loss_fn = CorticalLoss()

        predictions = torch.randn(8, 1)
        targets = torch.randn(8)

        loss_dict = loss_fn(predictions, targets, task_type="regression")

        assert loss_dict["total"].item() > 0

    def test_loss_without_activations(self):
        """Should work without activations."""
        loss_fn = CorticalLoss()

        predictions = torch.randn(8, 5)
        targets = torch.randint(0, 5, (8,))

        loss_dict = loss_fn(predictions, targets, activations=None)

        # Should compute task loss, others should be zero
        assert loss_dict["task"].item() > 0


class TestSTDP:
    """Test spike-timing dependent plasticity."""

    def test_stdp_potentiation(self):
        """Pre before post should cause potentiation."""
        stdp = SpikeTimingDependentPlasticity(a_plus=0.1, a_minus=0.1)

        # Pre-synaptic spike at t=0, post-synaptic at t=10
        pre_times = torch.tensor([[0.0]])
        post_times = torch.tensor([[10.0]])

        delta_w = stdp(pre_times, post_times)

        # Should be positive (potentiation)
        assert delta_w[0, 0].item() > 0

    def test_stdp_depression(self):
        """Post before pre should cause depression."""
        stdp = SpikeTimingDependentPlasticity(a_plus=0.1, a_minus=0.1)

        # Post-synaptic spike at t=0, pre-synaptic at t=10
        pre_times = torch.tensor([[10.0]])
        post_times = torch.tensor([[0.0]])

        delta_w = stdp(pre_times, post_times)

        # Should be negative (depression)
        assert delta_w[0, 0].item() < 0

    def test_stdp_shape(self):
        """Output should have correct shape."""
        stdp = SpikeTimingDependentPlasticity()

        pre_times = torch.randn(8, 10)  # batch=8, pre_neurons=10
        post_times = torch.randn(8, 5)  # batch=8, post_neurons=5

        delta_w = stdp(pre_times, post_times)

        # Should be [pre_neurons, post_neurons]
        assert delta_w.shape == (10, 5)
