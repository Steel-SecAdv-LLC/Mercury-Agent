# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Deterministic unit tests for the quantum circuit building module.

Covers the NumPy-only simulation fallback that ships when Qiskit is not
installed: :class:`SimulatedQuantumCircuit`, :class:`QuantumCircuitBuilder`,
:class:`AnomalyEncodingCircuit`, :class:`VariationalCircuit`,
:class:`QuantumFeatureMap`, and :class:`ErrorMitigationCircuit`, together
with the :class:`EncodingType` / :class:`VariationalAnsatz` enums and the
:class:`CircuitMetadata` dataclass.

All randomness is seeded through the per-instance ``Generator`` exposed via
the ``seed=`` constructor argument, so every assertion is reproducible and
no network, sleeps, or wall-clock reads are involved.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from omni_mercury_engine.quantum_computing import circuits as circuits_mod
from omni_mercury_engine.quantum_computing.circuits import (
    AnomalyEncodingCircuit,
    CircuitMetadata,
    EncodingType,
    ErrorMitigationCircuit,
    QuantumCircuitBuilder,
    QuantumFeatureMap,
    SimulatedQuantumCircuit,
    VariationalAnsatz,
    VariationalCircuit,
)

SEED = 20240521


def _gate_names(circuit: SimulatedQuantumCircuit) -> list[str]:
    """Return the ordered list of recorded gate names."""
    return [gate[0] for gate in circuit._gates]


def _gate_counts(circuit: SimulatedQuantumCircuit) -> dict[str, int]:
    """Return a name -> count mapping of recorded gates."""
    return dict(Counter(_gate_names(circuit)))


class TestModuleEnvironment:
    """The test environment must exercise the NumPy simulation fallback."""

    def test_qiskit_is_unavailable(self) -> None:
        # These tests are written against the pure-NumPy fallback; if Qiskit
        # were importable the builder would return real QuantumCircuit objects
        # and the ``_gates`` inspection below would not apply.
        assert circuits_mod.QISKIT_AVAILABLE is False
        assert circuits_mod.QuantumCircuit is None  # type: ignore[attr-defined]


class TestEnums:
    """Enum membership and identity."""

    def test_encoding_type_members(self) -> None:
        names = {member.name for member in EncodingType}
        assert names == {"AMPLITUDE", "ANGLE", "BASIS", "IQP", "ZZ_FEATURE_MAP"}

    def test_variational_ansatz_members(self) -> None:
        names = {member.name for member in VariationalAnsatz}
        assert names == {
            "REAL_AMPLITUDES",
            "EFFICIENT_SU2",
            "TWO_LOCAL",
            "HARDWARE_EFFICIENT",
        }

    def test_enum_values_are_distinct(self) -> None:
        values = [member.value for member in EncodingType]
        assert len(values) == len(set(values))


class TestCircuitMetadata:
    """The metadata dataclass defaults and field storage."""

    def test_defaults(self) -> None:
        meta = CircuitMetadata(num_qubits=3, depth=5, num_parameters=7, num_gates=9)
        assert meta.num_qubits == 3
        assert meta.depth == 5
        assert meta.num_parameters == 7
        assert meta.num_gates == 9
        assert meta.gate_counts == {}
        assert meta.connectivity == "linear"

    def test_explicit_fields(self) -> None:
        meta = CircuitMetadata(
            num_qubits=2,
            depth=1,
            num_parameters=0,
            num_gates=2,
            gate_counts={"h": 2},
            connectivity="full",
        )
        assert meta.gate_counts == {"h": 2}
        assert meta.connectivity == "full"

    def test_gate_counts_default_is_not_shared(self) -> None:
        # ``field(default_factory=dict)`` must give each instance its own dict.
        first = CircuitMetadata(num_qubits=1, depth=0, num_parameters=0, num_gates=0)
        second = CircuitMetadata(num_qubits=1, depth=0, num_parameters=0, num_gates=0)
        first.gate_counts["h"] = 1
        assert second.gate_counts == {}


class TestSimulatedQuantumCircuitConstruction:
    """Construction and simple accessors of the simulated circuit."""

    def test_default_construction(self) -> None:
        circuit = SimulatedQuantumCircuit(3)
        assert circuit.num_qubits == 3
        assert circuit.num_clbits == 0
        assert circuit._gates == []
        assert circuit.depth() == 0
        assert circuit.num_parameters() == 0
        assert isinstance(circuit._rng, np.random.Generator)

    def test_construction_with_clbits_and_seed(self) -> None:
        circuit = SimulatedQuantumCircuit(2, num_clbits=2, seed=SEED)
        assert circuit.num_qubits == 2
        assert circuit.num_clbits == 2


class TestSimulatedQuantumCircuitGates:
    """Every gate method records the expected tuple and chains fluently."""

    def test_single_qubit_gates_are_recorded(self) -> None:
        circuit = SimulatedQuantumCircuit(1)
        returned = circuit.h(0)
        assert returned is circuit  # fluent interface returns self
        circuit.x(0).y(0).z(0)
        assert _gate_names(circuit) == ["h", "x", "y", "z"]
        assert all(gate[1] == [0] for gate in circuit._gates)
        assert all(gate[2] == [] for gate in circuit._gates)

    def test_rotation_gates_store_angles(self) -> None:
        circuit = SimulatedQuantumCircuit(1)
        circuit.rx(0.1, 0).ry(0.2, 0).rz(0.3, 0)
        assert _gate_names(circuit) == ["rx", "ry", "rz"]
        assert [gate[2][0] for gate in circuit._gates] == [0.1, 0.2, 0.3]

    def test_two_qubit_gates(self) -> None:
        circuit = SimulatedQuantumCircuit(2)
        circuit.cx(0, 1).cz(1, 0)
        assert circuit._gates[0] == ("cx", [0, 1], [])
        assert circuit._gates[1] == ("cz", [1, 0], [])

    def test_barrier_is_noop_but_chains(self) -> None:
        circuit = SimulatedQuantumCircuit(2)
        assert circuit.barrier(0, 1) is circuit
        assert circuit._gates == []  # barrier records nothing

    def test_measure_all_records_all_qubits(self) -> None:
        circuit = SimulatedQuantumCircuit(3)
        circuit.measure_all()
        assert circuit._gates[-1] == ("measure_all", [0, 1, 2], [])

    def test_measure_single_qubit(self) -> None:
        circuit = SimulatedQuantumCircuit(2)
        circuit.measure(1, 0)
        assert circuit._gates[-1] == ("measure", [1, 0], [])

    def test_depth_counts_gates(self) -> None:
        circuit = SimulatedQuantumCircuit(2)
        circuit.h(0).cx(0, 1).barrier(0, 1).measure_all()
        # barrier does not append; the other three do.
        assert circuit.depth() == 3


class TestSimulatedQuantumCircuitSimulation:
    """Statevector simulation and measurement sampling."""

    def test_empty_circuit_measures_ground_state(self) -> None:
        circuit = SimulatedQuantumCircuit(3, seed=SEED)
        counts = circuit.simulate(shots=256)
        assert counts == {"000": 256}

    def test_bell_state_only_correlated_outcomes(self) -> None:
        circuit = SimulatedQuantumCircuit(2, seed=SEED)
        circuit.h(0).cx(0, 1)
        counts = circuit.simulate(shots=1000)
        assert set(counts) <= {"00", "11"}
        assert sum(counts.values()) == 1000
        # Both correlated outcomes should appear for a maximally entangled pair.
        assert counts.get("00", 0) > 0
        assert counts.get("11", 0) > 0

    def test_x_gate_flips_to_excited_state(self) -> None:
        circuit = SimulatedQuantumCircuit(1, seed=SEED)
        circuit.x(0)
        counts = circuit.simulate(shots=64)
        assert counts == {"1": 64}

    def test_simulation_is_seed_deterministic(self) -> None:
        first = SimulatedQuantumCircuit(3, seed=SEED)
        first.h(0).h(1).h(2)
        second = SimulatedQuantumCircuit(3, seed=SEED)
        second.h(0).h(1).h(2)
        assert first.simulate(shots=500) == second.simulate(shots=500)

    def test_measurement_gates_are_skipped_during_simulation(self) -> None:
        # measure / measure_all must not alter the statevector.
        circuit = SimulatedQuantumCircuit(1, seed=SEED)
        circuit.x(0).measure(0, 0).measure_all()
        counts = circuit.simulate(shots=32)
        assert counts == {"1": 32}

    def test_all_gate_kinds_route_through_apply_gate(self) -> None:
        # Exercise every branch of _apply_gate / _apply_single_qubit_gate /
        # _apply_cnot / _apply_cz in a single normalized simulation.
        circuit = SimulatedQuantumCircuit(2, seed=SEED)
        circuit.x(0).y(1).z(0).h(0)
        circuit.rx(0.5, 0).ry(0.5, 1).rz(0.5, 0)
        circuit.cx(0, 1).cz(0, 1)
        counts = circuit.simulate(shots=400)
        assert sum(counts.values()) == 400
        assert all(len(bitstring) == 2 for bitstring in counts)

    def test_unknown_gate_is_a_noop_during_simulation(self) -> None:
        # _apply_gate returns the state unchanged for an unrecognized gate,
        # so an injected bogus gate must not perturb the ground state.
        circuit = SimulatedQuantumCircuit(1, seed=SEED)
        circuit._gates.append(("not_a_real_gate", [0], []))
        counts = circuit.simulate(shots=16)
        assert counts == {"0": 16}

    def test_probabilities_are_normalized(self) -> None:
        # A single Hadamard yields an even split within sampling noise.
        circuit = SimulatedQuantumCircuit(1, seed=SEED)
        circuit.h(0)
        counts = circuit.simulate(shots=2000)
        assert sum(counts.values()) == 2000
        assert set(counts) == {"0", "1"}
        # Roughly balanced (loose bound to stay deterministic and non-flaky).
        assert 700 < counts["0"] < 1300


class TestQuantumCircuitBuilder:
    """Factory + metadata behaviour of the builder."""

    def test_defaults_to_simulation_when_qiskit_absent(self) -> None:
        builder = QuantumCircuitBuilder()
        assert builder._use_qiskit is False

    def test_use_qiskit_flag_still_false_without_qiskit(self) -> None:
        builder = QuantumCircuitBuilder(use_qiskit=True)
        assert builder._use_qiskit is False

    def test_create_circuit_returns_simulated(self) -> None:
        builder = QuantumCircuitBuilder()
        circuit = builder.create_circuit(4)
        assert isinstance(circuit, SimulatedQuantumCircuit)
        assert circuit.num_qubits == 4
        assert circuit.num_clbits == 0

    def test_create_circuit_with_classical_bits(self) -> None:
        builder = QuantumCircuitBuilder()
        circuit = builder.create_circuit(3, num_clbits=3)
        assert isinstance(circuit, SimulatedQuantumCircuit)
        assert circuit.num_clbits == 3

    def test_get_metadata_reports_simulated_circuit(self) -> None:
        builder = QuantumCircuitBuilder()
        circuit = builder.create_circuit(2)
        circuit.h(0).cx(0, 1).ry(0.3, 1)
        meta = builder.get_metadata(circuit)
        assert isinstance(meta, CircuitMetadata)
        assert meta.num_qubits == 2
        assert meta.depth == 3
        assert meta.num_gates == 3
        # SimulatedQuantumCircuit never registers named parameters.
        assert meta.num_parameters == 0


class TestAnomalyEncodingCircuit:
    """All five encoding strategies plus their edge cases."""

    def test_default_encoding_type_is_angle(self) -> None:
        encoder = AnomalyEncodingCircuit(3)
        assert encoder._encoding_type is EncodingType.ANGLE
        assert encoder._reps == 1

    def test_angle_encoding_structure(self) -> None:
        encoder = AnomalyEncodingCircuit(3, EncodingType.ANGLE, reps=1)
        circuit = encoder.encode(np.array([0.1, 0.2, 0.3]))
        counts = _gate_counts(circuit)
        # reps=1 -> 3 Hadamards, then RY+RZ per feature.
        assert counts["h"] == 3
        assert counts["ry"] == 3
        assert counts["rz"] == 3
        assert "cx" not in counts  # entangling layer only runs between reps

    def test_angle_encoding_multi_rep_adds_entanglers(self) -> None:
        encoder = AnomalyEncodingCircuit(3, EncodingType.ANGLE, reps=2)
        circuit = encoder.encode(np.array([0.1, 0.2, 0.3]))
        counts = _gate_counts(circuit)
        assert counts["h"] == 6  # 3 per rep
        # A single entangling layer of (num_qubits - 1) CX between the reps.
        assert counts["cx"] == 2

    def test_angle_encoding_empty_data(self) -> None:
        encoder = AnomalyEncodingCircuit(3, EncodingType.ANGLE, reps=1)
        circuit = encoder.encode(np.array([]))
        # Only the Hadamard layer survives when there are no features.
        assert _gate_names(circuit) == ["h", "h", "h"]

    def test_angle_encoding_truncates_extra_features(self) -> None:
        encoder = AnomalyEncodingCircuit(2, EncodingType.ANGLE, reps=1)
        circuit = encoder.encode(np.array([0.1, 0.2, 0.3, 0.4]))
        counts = _gate_counts(circuit)
        # Only the first two features (num_qubits=2) are encoded.
        assert counts["ry"] == 2
        assert counts["rz"] == 2

    def test_basis_encoding_sets_bits_above_threshold(self) -> None:
        encoder = AnomalyEncodingCircuit(3, EncodingType.BASIS)
        circuit = encoder.encode(np.array([0.9, 0.1, 0.7]))
        # Values > 0.5 flip their qubit with an X gate.
        assert _gate_names(circuit) == ["x", "x"]
        assert [gate[1][0] for gate in circuit._gates] == [0, 2]

    def test_basis_encoding_all_below_threshold(self) -> None:
        encoder = AnomalyEncodingCircuit(3, EncodingType.BASIS)
        circuit = encoder.encode(np.array([0.1, 0.2, 0.3]))
        assert circuit._gates == []

    def test_amplitude_encoding_produces_ry_per_qubit(self) -> None:
        encoder = AnomalyEncodingCircuit(2, EncodingType.AMPLITUDE)
        circuit = encoder.encode(np.array([1.0, 2.0, 3.0, 4.0]))
        assert _gate_names(circuit) == ["ry", "ry"]

    def test_amplitude_encoding_truncates_oversized_data(self) -> None:
        # len(data) > 2**num_qubits triggers the truncation branch.
        encoder = AnomalyEncodingCircuit(2, EncodingType.AMPLITUDE)
        circuit = encoder.encode(np.arange(1.0, 7.0))  # length 6 > 4
        assert _gate_names(circuit) == ["ry", "ry"]

    def test_amplitude_encoding_zero_vector_yields_zero_angles(self) -> None:
        # a + b <= 1e-10 branch: all rotation angles collapse to 0.
        encoder = AnomalyEncodingCircuit(2, EncodingType.AMPLITUDE)
        circuit = encoder.encode(np.zeros(4))
        assert _gate_names(circuit) == ["ry", "ry"]
        assert all(gate[2][0] == 0.0 for gate in circuit._gates)

    def test_iqp_encoding_builds_entangled_structure(self) -> None:
        encoder = AnomalyEncodingCircuit(3, EncodingType.IQP, reps=1)
        circuit = encoder.encode(np.array([0.2, 0.4, 0.6]))
        counts = _gate_counts(circuit)
        assert counts["h"] == 3
        # Per feature RZ plus the extra RZ inside each CX-RZ-CX sandwich.
        assert counts["rz"] == 3 + (3 - 1)
        assert counts["cx"] == 2 * (3 - 1)

    def test_zz_feature_map_encoding_uses_all_pairs(self) -> None:
        encoder = AnomalyEncodingCircuit(3, EncodingType.ZZ_FEATURE_MAP, reps=1)
        circuit = encoder.encode(np.array([0.2, 0.4, 0.6]))
        counts = _gate_counts(circuit)
        assert counts["h"] == 3
        num_pairs = 3  # C(3, 2)
        assert counts["cx"] == 2 * num_pairs
        assert counts["rz"] == 3 + num_pairs

    def test_encode_dispatches_on_encoding_type(self) -> None:
        # Each configured encoding type must reach a distinct code path and
        # return a circuit on the configured number of qubits.
        data = np.array([0.3, 0.6, 0.9])
        for enc_type in EncodingType:
            encoder = AnomalyEncodingCircuit(3, enc_type, reps=1)
            circuit = encoder.encode(data)
            assert isinstance(circuit, SimulatedQuantumCircuit)
            assert circuit.num_qubits == 3


class TestVariationalCircuit:
    """Parameter counting and per-ansatz circuit construction."""

    def test_num_parameters_real_amplitudes(self) -> None:
        circuit = VariationalCircuit(4, VariationalAnsatz.REAL_AMPLITUDES, reps=2)
        assert circuit.num_parameters == 4 * (2 + 1)

    def test_num_parameters_efficient_su2(self) -> None:
        circuit = VariationalCircuit(4, VariationalAnsatz.EFFICIENT_SU2, reps=2)
        assert circuit.num_parameters == 3 * 4 * (2 + 1)

    def test_num_parameters_two_local(self) -> None:
        circuit = VariationalCircuit(4, VariationalAnsatz.TWO_LOCAL, reps=2)
        assert circuit.num_parameters == 2 * 4 * (2 + 1)

    def test_num_parameters_hardware_efficient(self) -> None:
        circuit = VariationalCircuit(4, VariationalAnsatz.HARDWARE_EFFICIENT, reps=2)
        assert circuit.num_parameters == 2 * 4 * (2 + 1)

    def test_build_random_parameters_are_seed_deterministic(self) -> None:
        first = VariationalCircuit(3, reps=2, seed=SEED).build()
        second = VariationalCircuit(3, reps=2, seed=SEED).build()
        assert first._gates == second._gates

    def test_build_random_parameters_count_matches(self) -> None:
        circuit = VariationalCircuit(3, VariationalAnsatz.REAL_AMPLITUDES, reps=2, seed=SEED)
        built = circuit.build()
        assert len(circuit._parameters) == circuit.num_parameters
        assert isinstance(built, SimulatedQuantumCircuit)

    def test_build_with_explicit_parameters_are_stored_and_used(self) -> None:
        circuit = VariationalCircuit(2, VariationalAnsatz.REAL_AMPLITUDES, reps=1, seed=SEED)
        params = np.array([0.1, 0.2, 0.3, 0.4])
        built = circuit.build(params)
        assert [float(p) for p in circuit._parameters] == [0.1, 0.2, 0.3, 0.4]
        ry_angles = [gate[2][0] for gate in built._gates if gate[0] == "ry"]
        assert ry_angles == [0.1, 0.2, 0.3, 0.4]

    def test_real_amplitudes_gate_composition(self) -> None:
        circuit = VariationalCircuit(4, VariationalAnsatz.REAL_AMPLITUDES, reps=2, seed=SEED)
        counts = _gate_counts(circuit.build())
        assert counts["ry"] == 4 * (2 + 1)
        assert counts["cx"] == 2 * (4 - 1)
        assert "rz" not in counts

    def test_efficient_su2_gate_composition(self) -> None:
        circuit = VariationalCircuit(4, VariationalAnsatz.EFFICIENT_SU2, reps=2, seed=SEED)
        counts = _gate_counts(circuit.build())
        assert counts["rz"] == 2 * 4 * (2 + 1)
        assert counts["ry"] == 4 * (2 + 1)
        assert counts["cx"] == 2 * (4 - 1)

    def test_two_local_uses_cz_entanglers(self) -> None:
        circuit = VariationalCircuit(4, VariationalAnsatz.TWO_LOCAL, reps=2, seed=SEED)
        counts = _gate_counts(circuit.build())
        assert counts["ry"] == 4 * (2 + 1)
        assert counts["rz"] == 4 * (2 + 1)
        assert counts["cz"] == 2 * (4 - 1)
        assert "cx" not in counts

    def test_hardware_efficient_brickwork_entanglers(self) -> None:
        circuit = VariationalCircuit(4, VariationalAnsatz.HARDWARE_EFFICIENT, reps=2, seed=SEED)
        counts = _gate_counts(circuit.build())
        assert counts["ry"] == 4 * (2 + 1)
        assert counts["rz"] == 4 * (2 + 1)
        # even layer: qubits (0,1),(2,3) -> 2 CX; odd layer: (1,2) -> 1 CX.
        assert counts["cx"] == 2 * (2 + 1)


class TestQuantumFeatureMap:
    """Feature mapping and quantum kernel estimation."""

    def test_map_returns_zz_feature_map_circuit(self) -> None:
        feature_map = QuantumFeatureMap(2, reps=1)
        circuit = feature_map.map(np.array([0.3, 0.6]))
        assert isinstance(circuit, SimulatedQuantumCircuit)
        assert circuit.num_qubits == 2
        # ZZ feature map begins with a Hadamard layer.
        assert _gate_names(circuit)[:2] == ["h", "h"]

    def test_kernel_of_identical_points_is_one(self) -> None:
        # phi(x) followed by phi(x)^-1 collapses to |0...0>, giving K == 1.
        feature_map = QuantumFeatureMap(2, reps=1)
        point = np.array([0.3, 0.6])
        kernel = feature_map.compute_kernel(point, point, shots=1024)
        assert kernel == pytest.approx(1.0)

    def test_kernel_of_different_points_is_a_valid_probability(self) -> None:
        feature_map = QuantumFeatureMap(2, reps=1)
        kernel = feature_map.compute_kernel(
            np.array([0.3, 0.6]),
            np.array([0.9, 0.1]),
            shots=1024,
        )
        assert isinstance(kernel, float)
        assert 0.0 <= kernel <= 1.0

    def test_create_kernel_circuit_appends_measurement(self) -> None:
        feature_map = QuantumFeatureMap(2, reps=1)
        c1 = feature_map.map(np.array([0.3, 0.6]))
        c2 = feature_map.map(np.array([0.4, 0.5]))
        combined = feature_map._create_kernel_circuit(c1, c2)
        assert isinstance(combined, SimulatedQuantumCircuit)
        assert combined._gates[-1][0] == "measure_all"

    def test_invert_gate_is_self_inverse(self) -> None:
        feature_map = QuantumFeatureMap(2, reps=1)
        for gate in ("h", "cx", "rz", "x"):
            assert feature_map._invert_gate(gate) == gate

    def test_apply_gate_maps_every_supported_gate(self) -> None:
        feature_map = QuantumFeatureMap(2, reps=1)
        target = SimulatedQuantumCircuit(2)
        feature_map._apply_gate(target, "h", [0], [])
        feature_map._apply_gate(target, "x", [1], [])
        feature_map._apply_gate(target, "rx", [0], [0.1])
        feature_map._apply_gate(target, "ry", [0], [0.2])
        feature_map._apply_gate(target, "rz", [1], [0.3])
        feature_map._apply_gate(target, "cx", [0, 1], [])
        feature_map._apply_gate(target, "cz", [0, 1], [])
        assert _gate_names(target) == ["h", "x", "rx", "ry", "rz", "cx", "cz"]

    def test_apply_gate_ignores_unsupported_gate(self) -> None:
        # Unsupported gate names are silently skipped (defensive no-op).
        feature_map = QuantumFeatureMap(2, reps=1)
        target = SimulatedQuantumCircuit(2)
        feature_map._apply_gate(target, "unsupported", [0], [])
        assert target._gates == []

    def test_simulate_qiskit_circuit_import_error_fallback(self) -> None:
        # qiskit_aer is not installed, so the ImportError branch must return a
        # stub distribution peaked on the all-zeros bitstring.
        feature_map = QuantumFeatureMap(3, reps=1)
        counts = feature_map._simulate_qiskit_circuit(object(), shots=100)
        assert counts == {"000": 50}


class TestErrorMitigationCircuit:
    """Zero-noise extrapolation and expectation-value utilities."""

    def test_default_configuration(self) -> None:
        mitigator = ErrorMitigationCircuit()
        assert mitigator._method == "zne"
        assert mitigator._noise_factors == [1.0, 2.0, 3.0]

    def test_custom_noise_factors(self) -> None:
        mitigator = ErrorMitigationCircuit(method="raw", noise_factors=[1.0, 2.0])
        assert mitigator._method == "raw"
        assert mitigator._noise_factors == [1.0, 2.0]

    def test_compute_expectation_even_parity_is_positive(self) -> None:
        mitigator = ErrorMitigationCircuit()
        # All even-parity outcomes -> expectation +1.
        assert mitigator._compute_expectation({"00": 500, "11": 500}) == pytest.approx(1.0)

    def test_compute_expectation_odd_parity_is_negative(self) -> None:
        mitigator = ErrorMitigationCircuit()
        assert mitigator._compute_expectation({"01": 1000}) == pytest.approx(-1.0)

    def test_compute_expectation_mixed_parity(self) -> None:
        mitigator = ErrorMitigationCircuit()
        # 750 even (+1), 250 odd (-1) -> (750 - 250) / 1000.
        value = mitigator._compute_expectation({"00": 750, "01": 250})
        assert value == pytest.approx(0.5)

    def test_scale_noise_factor_one_returns_same_object(self) -> None:
        mitigator = ErrorMitigationCircuit()
        circuit = SimulatedQuantumCircuit(2, seed=SEED)
        circuit.h(0).cx(0, 1)
        assert mitigator._scale_noise(circuit, 1.0) is circuit

    def test_scale_noise_repeats_gates_for_simulated_circuit(self) -> None:
        mitigator = ErrorMitigationCircuit()
        circuit = SimulatedQuantumCircuit(2, seed=SEED)
        circuit.h(0).cx(0, 1)
        scaled = mitigator._scale_noise(circuit, 3.0)
        assert isinstance(scaled, SimulatedQuantumCircuit)
        # Each of the two gates is repeated ``int(factor)`` == 3 times.
        assert len(scaled._gates) == 2 * 3
        assert scaled.num_qubits == 2

    def test_scale_noise_non_simulated_circuit_returns_unchanged(self) -> None:
        mitigator = ErrorMitigationCircuit()
        sentinel = object()
        # Non SimulatedQuantumCircuit inputs are returned untouched.
        assert mitigator._scale_noise(sentinel, 2.0) is sentinel

    def test_mitigate_zne_recovers_bell_expectation(self) -> None:
        # Bell state outcomes ('00' / '11') are all even parity, so the
        # per-noise-level expectation is 1.0 and ZNE extrapolates to 1.0.
        mitigator = ErrorMitigationCircuit()
        circuit = SimulatedQuantumCircuit(2, seed=SEED)
        circuit.h(0).cx(0, 1)

        def executor(circ: SimulatedQuantumCircuit) -> dict[str, int]:
            return circ.simulate(shots=1024)

        value = mitigator.mitigate(circuit, executor)
        assert value == pytest.approx(1.0, abs=1e-6)

    def test_mitigate_non_zne_uses_raw_expectation(self) -> None:
        mitigator = ErrorMitigationCircuit(method="raw")
        circuit = SimulatedQuantumCircuit(1, seed=SEED)
        circuit.x(0)  # deterministic |1> -> odd parity -> expectation -1.

        def executor(circ: SimulatedQuantumCircuit) -> dict[str, int]:
            return circ.simulate(shots=256)

        value = mitigator.mitigate(circuit, executor)
        assert value == pytest.approx(-1.0)

    def test_mitigate_zne_calls_executor_once_per_noise_factor(self) -> None:
        mitigator = ErrorMitigationCircuit(noise_factors=[1.0, 2.0, 3.0])
        circuit = SimulatedQuantumCircuit(1, seed=SEED)
        circuit.h(0)
        calls: list[int] = []

        def executor(circ: SimulatedQuantumCircuit) -> dict[str, int]:
            calls.append(len(circ._gates))
            return {"0": 128, "1": 128}

        result = mitigator.mitigate(circuit, executor)
        assert len(calls) == 3
        assert isinstance(result, float)
