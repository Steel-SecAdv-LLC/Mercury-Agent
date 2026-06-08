# Copyright (C) 2025 Steel Security Advisors LLC
"""Tests for Quantum Computing Engine module."""

import numpy as np

from omni_mercury_engine.models.quantum_engine import (
    AnnealingResult,
    GroverSearchResult,
    QKDResult,
    QuantumCircuit,
    QuantumEngine,
    QuantumGate,
    QuantumState,
)


class TestQuantumState:
    """Tests for QuantumState class."""

    def test_init(self) -> None:
        """Test initialization."""
        amplitudes = np.array([1.0, 0.0], dtype=complex)
        state = QuantumState(amplitudes=amplitudes, num_qubits=1)
        assert state.num_qubits == 1
        assert len(state.amplitudes) == 2

    def test_normalize(self) -> None:
        """Test normalization."""
        amplitudes = np.array([3.0, 4.0], dtype=complex)
        state = QuantumState(amplitudes=amplitudes, num_qubits=1)
        state.normalize()
        norm = np.sqrt(np.sum(np.abs(state.amplitudes) ** 2))
        assert np.isclose(norm, 1.0)

    def test_normalize_zero_state(self) -> None:
        """Test normalization of zero state."""
        amplitudes = np.array([0.0, 0.0], dtype=complex)
        state = QuantumState(amplitudes=amplitudes, num_qubits=1)
        state.normalize()
        assert np.all(state.amplitudes == 0)

    def test_measure(self) -> None:
        """Test measurement."""
        amplitudes = np.array([1.0, 0.0], dtype=complex)
        state = QuantumState(amplitudes=amplitudes, num_qubits=1)
        result = state.measure()
        assert result == 0

    def test_measure_superposition(self) -> None:
        """Test measurement of superposition state."""
        amplitudes = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        state = QuantumState(amplitudes=amplitudes, num_qubits=1)
        results = [state.measure() for _ in range(100)]
        assert 0 in results
        assert 1 in results

    def test_get_probabilities(self) -> None:
        """Test probability calculation."""
        amplitudes = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        state = QuantumState(amplitudes=amplitudes, num_qubits=1)
        probs = state.get_probabilities()
        assert np.allclose(probs, [0.5, 0.5])


class TestQuantumGate:
    """Tests for QuantumGate class."""

    def test_hadamard(self) -> None:
        """Test Hadamard gate."""
        h = QuantumGate.hadamard()
        assert h.shape == (2, 2)
        assert np.allclose(h @ h, np.eye(2))

    def test_pauli_x(self) -> None:
        """Test Pauli-X gate."""
        x = QuantumGate.pauli_x()
        assert x.shape == (2, 2)
        assert np.allclose(x @ x, np.eye(2))

    def test_pauli_y(self) -> None:
        """Test Pauli-Y gate."""
        y = QuantumGate.pauli_y()
        assert y.shape == (2, 2)
        assert np.allclose(y @ y, np.eye(2))

    def test_pauli_z(self) -> None:
        """Test Pauli-Z gate."""
        z = QuantumGate.pauli_z()
        assert z.shape == (2, 2)
        assert np.allclose(z @ z, np.eye(2))

    def test_phase(self) -> None:
        """Test phase gate."""
        p = QuantumGate.phase(np.pi / 4)
        assert p.shape == (2, 2)
        assert np.isclose(p[0, 0], 1.0)
        assert np.isclose(np.abs(p[1, 1]), 1.0)

    def test_cnot(self) -> None:
        """Test CNOT gate."""
        cnot = QuantumGate.cnot()
        assert cnot.shape == (4, 4)

    def test_toffoli(self) -> None:
        """Test Toffoli gate."""
        toffoli = QuantumGate.toffoli()
        assert toffoli.shape == (8, 8)

    def test_swap(self) -> None:
        """Test SWAP gate."""
        swap = QuantumGate.swap()
        assert swap.shape == (4, 4)
        assert np.allclose(swap @ swap, np.eye(4))

    def test_t_gate(self) -> None:
        """Test T gate."""
        t = QuantumGate.t_gate()
        assert t.shape == (2, 2)
        assert np.isclose(t[0, 0], 1.0)

    def test_s_gate(self) -> None:
        """Test S gate."""
        s = QuantumGate.s_gate()
        assert s.shape == (2, 2)
        assert np.isclose(s[0, 0], 1.0)
        assert np.isclose(s[1, 1], 1j)


class TestQuantumCircuit:
    """Tests for QuantumCircuit class."""

    def test_init(self) -> None:
        """Test initialization."""
        circuit = QuantumCircuit(num_qubits=3)
        assert circuit.num_qubits == 3
        assert circuit.num_states == 8
        assert circuit.state.amplitudes[0] == 1.0

    def test_apply_hadamard(self) -> None:
        """Test applying Hadamard gate."""
        circuit = QuantumCircuit(num_qubits=1)
        circuit.apply_gate(QuantumGate.hadamard(), [0])
        probs = circuit.state.get_probabilities()
        assert np.allclose(probs, [0.5, 0.5])

    def test_apply_pauli_x(self) -> None:
        """Test applying Pauli-X gate."""
        circuit = QuantumCircuit(num_qubits=1)
        circuit.apply_gate(QuantumGate.pauli_x(), [0])
        assert np.isclose(circuit.state.amplitudes[1], 1.0)

    def test_measure(self) -> None:
        """Test measurement."""
        circuit = QuantumCircuit(num_qubits=1)
        result = circuit.measure()
        assert result == 0

    def test_get_state_vector(self) -> None:
        """Test getting state vector."""
        circuit = QuantumCircuit(num_qubits=2)
        state = circuit.get_state_vector()
        assert len(state) == 4
        assert state[0] == 1.0

    def test_reset(self) -> None:
        """Test circuit reset."""
        circuit = QuantumCircuit(num_qubits=2)
        circuit.apply_gate(QuantumGate.hadamard(), [0])
        circuit.reset()
        assert circuit.state.amplitudes[0] == 1.0
        assert len(circuit.gates) == 0

    def test_multi_qubit_circuit(self) -> None:
        """Test multi-qubit circuit."""
        circuit = QuantumCircuit(num_qubits=3)
        circuit.apply_gate(QuantumGate.hadamard(), [0])
        circuit.apply_gate(QuantumGate.hadamard(), [1])
        circuit.apply_gate(QuantumGate.hadamard(), [2])
        probs = circuit.state.get_probabilities()
        assert np.allclose(probs, np.ones(8) / 8)


class TestGroverSearchResult:
    """Tests for GroverSearchResult dataclass."""

    def test_init(self) -> None:
        """Test initialization."""
        result = GroverSearchResult(
            found=True,
            result=5,
            target=5,
            success_probability=0.95,
            iterations=3,
            classical_queries=8,
            quantum_queries=3,
            speedup=2.67,
        )
        assert result.found
        assert result.result == 5
        assert result.target == 5
        assert result.success_probability == 0.95


class TestQKDResult:
    """Tests for QKDResult dataclass."""

    def test_init(self) -> None:
        """Test initialization."""
        result = QKDResult(
            key="101010",
            key_length=6,
            error_rate=0.05,
            security_level=0.95,
            eavesdropping_detected=False,
        )
        assert result.key == "101010"
        assert result.key_length == 6
        assert result.protocol == "BB84"


class TestAnnealingResult:
    """Tests for AnnealingResult dataclass."""

    def test_init(self) -> None:
        """Test initialization."""
        result = AnnealingResult(
            best_state=[1, 0, 1, 0],
            best_cost=-10.0,
            confidence=0.9,
            iterations=100,
        )
        assert result.best_state == [1, 0, 1, 0]
        assert result.best_cost == -10.0


class TestQuantumEngine:
    """Tests for QuantumEngine class."""

    def test_init(self) -> None:
        """Test initialization."""
        engine = QuantumEngine()
        assert engine.golden_ratio == 0.618
        assert engine.quantum_factor == 1.2
        assert "omni_quantum_coherence" in engine.omni_scalars

    def test_grover_search_basic(self) -> None:
        """Test basic Grover search."""
        engine = QuantumEngine()
        result = engine.grover_search(database_size=8, target_item=3)
        assert isinstance(result, GroverSearchResult)
        assert result.target == 3
        assert result.iterations > 0
        assert result.speedup > 0

    def test_grover_search_power_of_two(self) -> None:
        """Test Grover search with power of 2 database size."""
        engine = QuantumEngine()
        result = engine.grover_search(database_size=16, target_item=7)
        assert result.target == 7
        assert result.classical_queries == 8
        assert result.quantum_queries > 0

    def test_grover_search_non_power_of_two(self) -> None:
        """Test Grover search with non-power of 2 database size."""
        engine = QuantumEngine()
        result = engine.grover_search(database_size=10, target_item=5)
        assert result.target == 5

    def test_grover_search_small_database(self) -> None:
        """Test Grover search with small database."""
        engine = QuantumEngine()
        result = engine.grover_search(database_size=2, target_item=1)
        assert result.target == 1

    def test_generate_entangled_pair(self) -> None:
        """Test entangled pair generation."""
        engine = QuantumEngine()
        state1, state2 = engine.generate_entangled_pair()
        assert isinstance(state1, QuantumState)
        assert isinstance(state2, QuantumState)
        assert state1.num_qubits == 1
        assert state2.num_qubits == 1

    def test_quantum_key_distribution(self) -> None:
        """Test QKD protocol."""
        engine = QuantumEngine()
        result = engine.quantum_key_distribution(key_length=64)
        assert isinstance(result, QKDResult)
        assert result.protocol == "BB84"
        assert result.error_rate >= 0
        assert result.security_level >= 0

    def test_quantum_key_distribution_short_key(self) -> None:
        """Test QKD with short key."""
        engine = QuantumEngine()
        result = engine.quantum_key_distribution(key_length=16)
        assert isinstance(result, QKDResult)

    def test_quantum_random_number(self) -> None:
        """Test quantum random number generation."""
        engine = QuantumEngine()
        random_bits = engine.quantum_random_number(num_bits=32)
        assert len(random_bits) == 32
        assert all(c in "01" for c in random_bits)

    def test_quantum_random_number_different_lengths(self) -> None:
        """Test quantum RNG with different lengths."""
        engine = QuantumEngine()
        for num_bits in [8, 16, 64, 128]:
            random_bits = engine.quantum_random_number(num_bits=num_bits)
            assert len(random_bits) == num_bits

    def test_simulate_quantum_annealing(self) -> None:
        """Test quantum annealing simulation."""
        engine = QuantumEngine()

        def cost_function(state):
            return -sum(state)

        result = engine.simulate_quantum_annealing(
            cost_function=cost_function, num_vars=4, num_iterations=10
        )
        assert isinstance(result, AnnealingResult)
        assert len(result.best_state) == 4
        assert result.iterations == 10

    def test_calculate_quantum_fidelity(self) -> None:
        """Test quantum fidelity calculation."""
        engine = QuantumEngine()
        state1 = np.array([1.0, 0.0], dtype=complex)
        state2 = np.array([1.0, 0.0], dtype=complex)
        fidelity = engine.calculate_quantum_fidelity(state1, state2)
        assert np.isclose(fidelity, 1.0)

    def test_calculate_quantum_fidelity_orthogonal(self) -> None:
        """Test fidelity of orthogonal states."""
        engine = QuantumEngine()
        state1 = np.array([1.0, 0.0], dtype=complex)
        state2 = np.array([0.0, 1.0], dtype=complex)
        fidelity = engine.calculate_quantum_fidelity(state1, state2)
        assert np.isclose(fidelity, 0.0)

    def test_quantum_phase_estimation(self) -> None:
        """Test quantum phase estimation."""
        engine = QuantumEngine()
        unitary = QuantumGate.pauli_z()
        eigenvector = np.array([0.0, 1.0], dtype=complex)  # |1> is eigenvector of Z
        result = engine.quantum_phase_estimation(unitary, eigenvector, precision=3)
        assert "estimated_phase" in result
        assert "precision_bits" in result

    def test_extract_quantum_features(self) -> None:
        """Test quantum feature extraction."""
        engine = QuantumEngine()
        data = np.random.randn(10, 5)
        features = engine.extract_quantum_features(data)
        # Returns numpy array with shape (n_samples, 5) containing
        # [coherence, entropy, phase_variance, norm, omni_quantum_coherence]
        assert isinstance(features, np.ndarray)
        assert features.shape == (10, 5)


class TestQuantumEngineEdgeCases:
    """Edge case tests for QuantumEngine."""

    def test_grover_search_target_zero(self) -> None:
        """Test Grover search with target 0."""
        engine = QuantumEngine()
        result = engine.grover_search(database_size=8, target_item=0)
        assert result.target == 0

    def test_grover_search_large_database(self) -> None:
        """Test Grover search with larger database."""
        engine = QuantumEngine()
        result = engine.grover_search(database_size=64, target_item=42)
        assert result.target == 42
        assert result.speedup > 1

    def test_qkd_very_short_key(self) -> None:
        """Test QKD with very short key."""
        engine = QuantumEngine()
        result = engine.quantum_key_distribution(key_length=4)
        assert isinstance(result, QKDResult)

    def test_quantum_random_single_bit(self) -> None:
        """Test quantum RNG with single bit."""
        engine = QuantumEngine()
        random_bit = engine.quantum_random_number(num_bits=1)
        assert len(random_bit) == 1
        assert random_bit in ["0", "1"]
