"""
Mercury Agent ♱
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


"""Tests for Quantum Kernel Machines integration."""

import numpy as np
import pytest

from omni_mercury_engine.core.quantum_kernels import QuantumKernelMachine


class TestQuantumKernelMachine:
    """Test QuantumKernelMachine class."""

    def test_machine_initialization(self):
        """Test machine initialization."""
        machine = QuantumKernelMachine()
        assert machine.kernel_type == "quantum_inspired"
        assert machine.num_qubits == 4
        assert machine.entanglement_depth == 2
        assert machine.gamma == 1.0
        assert machine.training_data is None
        assert machine.anomaly_threshold is None

    def test_machine_custom_config(self):
        """Test machine with custom configuration."""
        config = {
            "kernel_type": "rbf",
            "num_qubits": 6,
            "entanglement_depth": 3,
            "gamma": 0.5,
        }
        machine = QuantumKernelMachine(config)
        assert machine.kernel_type == "rbf"
        assert machine.num_qubits == 6
        assert machine.entanglement_depth == 3
        assert machine.gamma == 0.5

    def test_quantum_inspired_kernel_same_input(self):
        """Test quantum-inspired kernel with same input."""
        machine = QuantumKernelMachine()
        x = np.random.randn(10)

        kernel_value = machine.quantum_inspired_kernel(x, x)
        assert isinstance(kernel_value, float)
        assert kernel_value > 0.0

    def test_quantum_inspired_kernel_different_inputs(self):
        """Test quantum-inspired kernel with different inputs."""
        machine = QuantumKernelMachine()
        x1 = np.random.randn(10)
        x2 = np.random.randn(10)

        kernel_value = machine.quantum_inspired_kernel(x1, x2)
        assert isinstance(kernel_value, float)

    def test_quantum_feature_map(self):
        """Test quantum feature map transformation."""
        machine = QuantumKernelMachine()
        x = np.random.randn(10)

        phi = machine._quantum_feature_map(x, depth=2)
        assert isinstance(phi, np.ndarray)
        assert len(phi) == 2**machine.num_qubits

    def test_quantum_feature_map_depth_zero(self):
        """Test quantum feature map with zero depth."""
        machine = QuantumKernelMachine()
        x = np.random.randn(10)

        phi = machine._quantum_feature_map(x, depth=0)
        assert len(phi) == 2**machine.num_qubits

    def test_quantum_feature_map_depth_one(self):
        """Test quantum feature map with depth one."""
        machine = QuantumKernelMachine()
        x = np.random.randn(10)

        phi = machine._quantum_feature_map(x, depth=1)
        assert len(phi) == 2**machine.num_qubits

    def test_rbf_kernel_same_input(self):
        """Test RBF kernel with same input."""
        machine = QuantumKernelMachine()
        x = np.random.randn(10)

        kernel_value = machine.rbf_kernel(x, x)
        assert kernel_value == 1.0

    def test_rbf_kernel_different_inputs(self):
        """Test RBF kernel with different inputs."""
        machine = QuantumKernelMachine()
        x1 = np.random.randn(10)
        x2 = np.random.randn(10)

        kernel_value = machine.rbf_kernel(x1, x2)
        assert 0.0 <= kernel_value <= 1.0

    def test_rbf_kernel_gamma_effect(self):
        """Test RBF kernel with different gamma values."""
        machine1 = QuantumKernelMachine({"gamma": 0.1})
        machine2 = QuantumKernelMachine({"gamma": 10.0})

        x1 = np.zeros(10)
        x2 = np.ones(10)

        k1 = machine1.rbf_kernel(x1, x2)
        k2 = machine2.rbf_kernel(x1, x2)

        assert k1 != k2

    def test_compute_kernel_matrix_shape(self):
        """Test kernel matrix computation shape."""
        machine = QuantumKernelMachine()
        X = np.random.randn(20, 10)

        K = machine.compute_kernel_matrix(X)
        assert K.shape == (20, 20)

    def test_compute_kernel_matrix_symmetric(self):
        """Test kernel matrix is symmetric."""
        machine = QuantumKernelMachine()
        X = np.random.randn(10, 5)

        K = machine.compute_kernel_matrix(X)
        np.testing.assert_array_almost_equal(K, K.T)

    def test_compute_kernel_matrix_diagonal_ones(self):
        """Test kernel matrix diagonal for RBF kernel."""
        machine = QuantumKernelMachine({"kernel_type": "rbf"})
        X = np.random.randn(10, 5)

        K = machine.compute_kernel_matrix(X)
        diagonal = np.diag(K)
        np.testing.assert_array_almost_equal(diagonal, np.ones(10))

    def test_fit_training_data(self):
        """Test fitting training data."""
        machine = QuantumKernelMachine()
        training_data = np.random.randn(50, 10)

        machine.fit(training_data)

        assert machine.training_data is not None
        assert machine.anomaly_threshold is not None

    def test_fit_sets_threshold(self):
        """Test that fit sets anomaly threshold."""
        machine = QuantumKernelMachine()
        training_data = np.random.randn(50, 10)

        assert machine.anomaly_threshold is None
        machine.fit(training_data)
        assert machine.anomaly_threshold is not None

    def test_predict_without_fit_raises_error(self):
        """Test that predict raises error if not fitted."""
        machine = QuantumKernelMachine()
        test_data = np.random.randn(10, 5)

        with pytest.raises(ValueError, match="Must fit model first"):
            machine.predict(test_data)

    def test_predict_basic(self):
        """Test basic prediction."""
        machine = QuantumKernelMachine()
        training_data = np.random.randn(50, 10)
        machine.fit(training_data)

        test_data = np.random.randn(10, 10)
        results = machine.predict(test_data)

        assert "anomaly_scores" in results
        assert "predictions" in results
        assert "threshold" in results
        assert "kernel_type" in results

    def test_predict_scores_shape(self):
        """Test prediction scores shape."""
        machine = QuantumKernelMachine()
        training_data = np.random.randn(50, 10)
        machine.fit(training_data)

        test_data = np.random.randn(15, 10)
        results = machine.predict(test_data)

        assert results["anomaly_scores"].shape == (15,)

    def test_predict_predictions_shape(self):
        """Test predictions shape."""
        machine = QuantumKernelMachine()
        training_data = np.random.randn(50, 10)
        machine.fit(training_data)

        test_data = np.random.randn(15, 10)
        results = machine.predict(test_data)

        assert results["predictions"].shape == (15,)

    def test_predict_predictions_boolean(self):
        """Test predictions are boolean."""
        machine = QuantumKernelMachine()
        training_data = np.random.randn(50, 10)
        machine.fit(training_data)

        test_data = np.random.randn(10, 10)
        results = machine.predict(test_data)

        assert results["predictions"].dtype == bool

    def test_predict_kernel_type_label(self):
        """Test kernel type label in results."""
        machine = QuantumKernelMachine({"kernel_type": "rbf"})
        training_data = np.random.randn(50, 10)
        machine.fit(training_data)

        test_data = np.random.randn(10, 10)
        results = machine.predict(test_data)

        assert results["kernel_type"] == "rbf"

    def test_kernel_type_quantum_inspired(self):
        """Test quantum-inspired kernel type."""
        config = {"kernel_type": "quantum_inspired"}
        machine = QuantumKernelMachine(config)
        assert machine.kernel_type == "quantum_inspired"

    def test_kernel_type_rbf(self):
        """Test RBF kernel type."""
        config = {"kernel_type": "rbf"}
        machine = QuantumKernelMachine(config)
        assert machine.kernel_type == "rbf"

    def test_num_qubits_config(self):
        """Test num_qubits configuration."""
        config = {"num_qubits": 8}
        machine = QuantumKernelMachine(config)
        assert machine.num_qubits == 8

    def test_entanglement_depth_config(self):
        """Test entanglement_depth configuration."""
        config = {"entanglement_depth": 4}
        machine = QuantumKernelMachine(config)
        assert machine.entanglement_depth == 4

    def test_gamma_config(self):
        """Test gamma configuration."""
        config = {"gamma": 0.25}
        machine = QuantumKernelMachine(config)
        assert machine.gamma == 0.25
