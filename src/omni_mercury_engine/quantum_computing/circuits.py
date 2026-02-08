"""
Quantum Circuit Building for Mercury Agent.

Provides quantum circuit construction for anomaly detection, including
data encoding, variational circuits, and error mitigation.

References:
- Havlicek et al. (2019): Supervised learning with quantum-enhanced feature spaces
- Cerezo et al. (2021): Variational quantum algorithms
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt


if TYPE_CHECKING:
    from collections.abc import Callable


logger = logging.getLogger(__name__)


QISKIT_AVAILABLE = False
try:
    from qiskit import (  # noqa: F401
        ClassicalRegister as _ClassicalRegister,
        QuantumCircuit,
        QuantumRegister as _QuantumRegister,
    )
    from qiskit.circuit import (  # noqa: F401
        Parameter as _Parameter,
        ParameterVector as _ParameterVector,
    )

    QISKIT_AVAILABLE = True
except ImportError:
    logger.debug("Qiskit not available, using simulation fallback")
    QuantumCircuit = None


class EncodingType(Enum):
    """Types of data encoding for quantum circuits."""

    AMPLITUDE = auto()
    ANGLE = auto()
    BASIS = auto()
    IQP = auto()
    ZZ_FEATURE_MAP = auto()


class VariationalAnsatz(Enum):
    """Types of variational ansatz."""

    REAL_AMPLITUDES = auto()
    EFFICIENT_SU2 = auto()
    TWO_LOCAL = auto()
    HARDWARE_EFFICIENT = auto()


@dataclass
class CircuitMetadata:
    """Metadata about a quantum circuit."""

    num_qubits: int
    depth: int
    num_parameters: int
    num_gates: int
    gate_counts: dict[str, int] = field(default_factory=dict)
    connectivity: str = "linear"


class SimulatedQuantumCircuit:
    """
    Simulated quantum circuit for when Qiskit is not available.

    Provides a compatible interface for circuit construction and simulation.
    """

    def __init__(self, num_qubits: int, num_clbits: int = 0) -> None:
        """Initialize simulated circuit."""
        self.num_qubits = num_qubits
        self.num_clbits = num_clbits
        self._gates: list[tuple[str, list[int], list[float]]] = []
        self._parameters: list[str] = []
        self._state: npt.NDArray[Any] | None = None

    def h(self, qubit: int) -> SimulatedQuantumCircuit:
        """Hadamard gate."""
        self._gates.append(("h", [qubit], []))
        return self

    def x(self, qubit: int) -> SimulatedQuantumCircuit:
        """Pauli-X gate."""
        self._gates.append(("x", [qubit], []))
        return self

    def y(self, qubit: int) -> SimulatedQuantumCircuit:
        """Pauli-Y gate."""
        self._gates.append(("y", [qubit], []))
        return self

    def z(self, qubit: int) -> SimulatedQuantumCircuit:
        """Pauli-Z gate."""
        self._gates.append(("z", [qubit], []))
        return self

    def rx(self, theta: float, qubit: int) -> SimulatedQuantumCircuit:
        """Rotation around X axis."""
        self._gates.append(("rx", [qubit], [theta]))
        return self

    def ry(self, theta: float, qubit: int) -> SimulatedQuantumCircuit:
        """Rotation around Y axis."""
        self._gates.append(("ry", [qubit], [theta]))
        return self

    def rz(self, theta: float, qubit: int) -> SimulatedQuantumCircuit:
        """Rotation around Z axis."""
        self._gates.append(("rz", [qubit], [theta]))
        return self

    def cx(self, control: int, target: int) -> SimulatedQuantumCircuit:
        """CNOT gate."""
        self._gates.append(("cx", [control, target], []))
        return self

    def cz(self, control: int, target: int) -> SimulatedQuantumCircuit:
        """Controlled-Z gate."""
        self._gates.append(("cz", [control, target], []))
        return self

    def barrier(self, *qubits: int) -> SimulatedQuantumCircuit:
        """Barrier (no-op in simulation)."""
        return self

    def measure_all(self) -> SimulatedQuantumCircuit:
        """Add measurement to all qubits."""
        self._gates.append(("measure_all", list(range(self.num_qubits)), []))
        return self

    def measure(self, qubit: int, clbit: int) -> SimulatedQuantumCircuit:
        """Measure single qubit."""
        self._gates.append(("measure", [qubit, clbit], []))
        return self

    def depth(self) -> int:
        """Estimate circuit depth."""
        return len(self._gates)

    def num_parameters(self) -> int:
        """Count parameters."""
        return len(self._parameters)

    def simulate(self, shots: int = 1024) -> dict[str, int]:
        """
        Simulate the circuit and return measurement counts.

        Uses statevector simulation with NumPy.
        """
        n = self.num_qubits
        state = np.zeros(2**n, dtype=complex)
        state[0] = 1.0

        for gate_name, qubits, params in self._gates:
            if gate_name == "measure_all" or gate_name == "measure":
                continue

            state = self._apply_gate(state, gate_name, qubits, params)

        probabilities = np.abs(state) ** 2
        probabilities = probabilities / np.sum(probabilities)

        outcomes = np.random.choice(2**n, size=shots, p=probabilities)
        counts: dict[str, int] = {}
        for outcome in outcomes:
            bitstring = format(outcome, f"0{n}b")
            counts[bitstring] = counts.get(bitstring, 0) + 1

        return counts

    def _apply_gate(
        self,
        state: npt.NDArray[Any],
        gate_name: str,
        qubits: list[int],
        params: list[float],
    ) -> npt.NDArray[Any]:
        """Apply a gate to the state vector."""
        n = self.num_qubits

        if gate_name == "h":
            matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
            return self._apply_single_qubit_gate(state, matrix, qubits[0], n)

        elif gate_name == "x":
            matrix = np.array([[0, 1], [1, 0]])
            return self._apply_single_qubit_gate(state, matrix, qubits[0], n)

        elif gate_name == "y":
            matrix = np.array([[0, -1j], [1j, 0]])
            return self._apply_single_qubit_gate(state, matrix, qubits[0], n)

        elif gate_name == "z":
            matrix = np.array([[1, 0], [0, -1]])
            return self._apply_single_qubit_gate(state, matrix, qubits[0], n)

        elif gate_name == "rx":
            theta = params[0]
            c, s = np.cos(theta / 2), np.sin(theta / 2)
            matrix = np.array([[c, -1j * s], [-1j * s, c]])
            return self._apply_single_qubit_gate(state, matrix, qubits[0], n)

        elif gate_name == "ry":
            theta = params[0]
            c, s = np.cos(theta / 2), np.sin(theta / 2)
            matrix = np.array([[c, -s], [s, c]])
            return self._apply_single_qubit_gate(state, matrix, qubits[0], n)

        elif gate_name == "rz":
            theta = params[0]
            matrix = np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]])
            return self._apply_single_qubit_gate(state, matrix, qubits[0], n)

        elif gate_name == "cx":
            return self._apply_cnot(state, qubits[0], qubits[1], n)

        elif gate_name == "cz":
            return self._apply_cz(state, qubits[0], qubits[1], n)

        return state

    def _apply_single_qubit_gate(
        self,
        state: npt.NDArray[Any],
        matrix: npt.NDArray[Any],
        qubit: int,
        n_qubits: int,
    ) -> npt.NDArray[Any]:
        """Apply single-qubit gate using tensor product."""
        new_state = np.zeros_like(state)
        for i in range(len(state)):
            bit = (i >> qubit) & 1
            for new_bit in range(2):
                j = (i & ~(1 << qubit)) | (new_bit << qubit)
                new_state[j] += matrix[new_bit, bit] * state[i]
        return new_state

    def _apply_cnot(
        self,
        state: npt.NDArray[Any],
        control: int,
        target: int,
        n_qubits: int,
    ) -> npt.NDArray[Any]:
        """Apply CNOT gate."""
        new_state = state.copy()
        for i in range(len(state)):
            if (i >> control) & 1:
                j = i ^ (1 << target)
                new_state[i], new_state[j] = state[j], state[i]
        return new_state

    def _apply_cz(
        self,
        state: npt.NDArray[Any],
        control: int,
        target: int,
        n_qubits: int,
    ) -> npt.NDArray[Any]:
        """Apply CZ gate."""
        new_state = state.copy()
        for i in range(len(state)):
            if ((i >> control) & 1) and ((i >> target) & 1):
                new_state[i] = -state[i]
        return new_state


class QuantumCircuitBuilder:
    """
    Build quantum circuits for anomaly detection.

    Provides factory methods for creating various types of quantum circuits
    including encoding circuits, variational circuits, and feature maps.
    """

    def __init__(self, use_qiskit: bool = True) -> None:
        """Initialize the circuit builder."""
        self._use_qiskit = use_qiskit and QISKIT_AVAILABLE

    def create_circuit(
        self,
        num_qubits: int,
        num_clbits: int = 0,
    ) -> Any:
        """
        Create a new quantum circuit.

        Args:
            num_qubits: Number of qubits
            num_clbits: Number of classical bits

        Returns:
            QuantumCircuit (Qiskit) or SimulatedQuantumCircuit
        """
        if self._use_qiskit:
            if num_clbits > 0:
                return QuantumCircuit(num_qubits, num_clbits)
            return QuantumCircuit(num_qubits)
        else:
            return SimulatedQuantumCircuit(num_qubits, num_clbits)

    def get_metadata(self, circuit: Any) -> CircuitMetadata:
        """Get metadata about a circuit."""
        if self._use_qiskit:
            gate_counts = dict(circuit.count_ops())
            return CircuitMetadata(
                num_qubits=circuit.num_qubits,
                depth=circuit.depth(),
                num_parameters=circuit.num_parameters,
                num_gates=sum(gate_counts.values()),
                gate_counts=gate_counts,
            )
        else:
            return CircuitMetadata(
                num_qubits=circuit.num_qubits,
                depth=circuit.depth(),
                num_parameters=circuit.num_parameters(),
                num_gates=len(circuit._gates),
            )


class AnomalyEncodingCircuit:
    """
    Encode classical data into quantum states for anomaly detection.

    Supports multiple encoding strategies optimized for different data types.
    """

    def __init__(
        self,
        num_qubits: int,
        encoding_type: EncodingType = EncodingType.ANGLE,
        reps: int = 1,
    ) -> None:
        """Initialize the encoding circuit."""
        self._num_qubits = num_qubits
        self._encoding_type = encoding_type
        self._reps = reps
        self._builder = QuantumCircuitBuilder()

    def encode(self, data: npt.NDArray[Any]) -> Any:
        """
        Encode classical data into a quantum circuit.

        Args:
            data: Classical data to encode (normalized to [0, 2*pi] for angles)

        Returns:
            Quantum circuit with encoded data
        """
        if self._encoding_type == EncodingType.AMPLITUDE:
            return self._amplitude_encoding(data)
        elif self._encoding_type == EncodingType.ANGLE:
            return self._angle_encoding(data)
        elif self._encoding_type == EncodingType.BASIS:
            return self._basis_encoding(data)
        elif self._encoding_type == EncodingType.IQP:
            return self._iqp_encoding(data)
        elif self._encoding_type == EncodingType.ZZ_FEATURE_MAP:
            return self._zz_feature_map_encoding(data)
        else:
            return self._angle_encoding(data)

    def _amplitude_encoding(self, data: npt.NDArray[Any]) -> Any:
        """
        Amplitude encoding - encodes data in state amplitudes.

        Requires len(data) <= 2^n and normalized data.
        """
        circuit = self._builder.create_circuit(self._num_qubits)

        data_normalized = data / (np.linalg.norm(data) + 1e-10)

        if len(data_normalized) > 2**self._num_qubits:
            data_normalized = data_normalized[: 2**self._num_qubits]

        padded = np.zeros(2**self._num_qubits)
        padded[: len(data_normalized)] = data_normalized

        angles = self._compute_amplitude_angles(padded)

        idx = 0
        for qubit in range(self._num_qubits):
            if idx < len(angles):
                circuit.ry(angles[idx], qubit)
                idx += 1

        return circuit

    def _compute_amplitude_angles(self, amplitudes: npt.NDArray[Any]) -> list[float]:
        """Compute rotation angles for amplitude encoding."""
        angles = []
        n = len(amplitudes)

        for level in range(int(np.log2(n))):
            step = 2 ** (level + 1)
            for i in range(0, n, step):
                a = np.sum(amplitudes[i : i + step // 2] ** 2)
                b = np.sum(amplitudes[i + step // 2 : i + step] ** 2)

                if a + b > 1e-10:
                    angle = 2 * np.arccos(np.sqrt(a / (a + b)))
                else:
                    angle = 0.0
                angles.append(angle)

        return angles

    def _angle_encoding(self, data: npt.NDArray[Any]) -> Any:
        """
        Angle encoding - encodes each feature as a rotation angle.

        One qubit per feature, uses RY and RZ rotations.
        """
        circuit = self._builder.create_circuit(self._num_qubits)

        for rep in range(self._reps):
            for i in range(self._num_qubits):
                circuit.h(i)

            for i, val in enumerate(data[: self._num_qubits]):
                angle = float(val) * np.pi
                circuit.ry(angle, i)
                circuit.rz(angle, i)

            if rep < self._reps - 1:
                for i in range(self._num_qubits - 1):
                    circuit.cx(i, i + 1)

        return circuit

    def _basis_encoding(self, data: npt.NDArray[Any]) -> Any:
        """
        Basis encoding - encodes binary data in computational basis.

        Each qubit represents one bit of data.
        """
        circuit = self._builder.create_circuit(self._num_qubits)

        binary = (data > 0.5).astype(int)

        for i, bit in enumerate(binary[: self._num_qubits]):
            if bit:
                circuit.x(i)

        return circuit

    def _iqp_encoding(self, data: npt.NDArray[Any]) -> Any:
        """
        IQP (Instantaneous Quantum Polynomial) encoding.

        Creates entanglement structure with diagonal gates.
        """
        circuit = self._builder.create_circuit(self._num_qubits)

        for rep in range(self._reps):
            for i in range(self._num_qubits):
                circuit.h(i)

            for i, val in enumerate(data[: self._num_qubits]):
                circuit.rz(float(val) * np.pi, i)

            for i in range(self._num_qubits - 1):
                val_i = data[i] if i < len(data) else 0
                val_j = data[i + 1] if i + 1 < len(data) else 0
                circuit.cx(i, i + 1)
                circuit.rz(float(val_i * val_j) * np.pi, i + 1)
                circuit.cx(i, i + 1)

        return circuit

    def _zz_feature_map_encoding(self, data: npt.NDArray[Any]) -> Any:
        """
        ZZ Feature Map encoding for quantum kernels.

        Creates entanglement via ZZ interactions between features.
        """
        circuit = self._builder.create_circuit(self._num_qubits)

        for rep in range(self._reps):
            for i in range(self._num_qubits):
                circuit.h(i)

            for i, val in enumerate(data[: self._num_qubits]):
                circuit.rz(2 * float(val), i)

            for i in range(self._num_qubits - 1):
                for j in range(i + 1, self._num_qubits):
                    val_i = data[i] if i < len(data) else 0
                    val_j = data[j] if j < len(data) else 0

                    circuit.cx(i, j)
                    circuit.rz(2 * float(val_i * val_j), j)
                    circuit.cx(i, j)

        return circuit


class VariationalCircuit:
    """
    Variational quantum circuit for trainable quantum models.

    Implements various ansatz architectures for VQE and QAOA.
    """

    def __init__(
        self,
        num_qubits: int,
        ansatz: VariationalAnsatz = VariationalAnsatz.REAL_AMPLITUDES,
        reps: int = 2,
        entanglement: str = "linear",
    ) -> None:
        """Initialize the variational circuit."""
        self._num_qubits = num_qubits
        self._ansatz = ansatz
        self._reps = reps
        self._entanglement = entanglement
        self._builder = QuantumCircuitBuilder()
        self._parameters: list[float] = []

    @property
    def num_parameters(self) -> int:
        """Get number of trainable parameters."""
        if self._ansatz == VariationalAnsatz.REAL_AMPLITUDES:
            return self._num_qubits * (self._reps + 1)
        elif self._ansatz == VariationalAnsatz.EFFICIENT_SU2:
            return 3 * self._num_qubits * (self._reps + 1)
        elif self._ansatz == VariationalAnsatz.TWO_LOCAL:
            return 2 * self._num_qubits * (self._reps + 1)
        else:
            return 2 * self._num_qubits * (self._reps + 1)

    def build(self, parameters: npt.NDArray[Any] | None = None) -> Any:
        """
        Build the variational circuit with given parameters.

        Args:
            parameters: Parameter values (random if None)

        Returns:
            Quantum circuit with variational structure
        """
        if parameters is None:
            parameters = np.random.uniform(0, 2 * np.pi, self.num_parameters)

        self._parameters = list(parameters)

        if self._ansatz == VariationalAnsatz.REAL_AMPLITUDES:
            return self._real_amplitudes(parameters)
        elif self._ansatz == VariationalAnsatz.EFFICIENT_SU2:
            return self._efficient_su2(parameters)
        elif self._ansatz == VariationalAnsatz.TWO_LOCAL:
            return self._two_local(parameters)
        else:
            return self._hardware_efficient(parameters)

    def _real_amplitudes(self, parameters: npt.NDArray[Any]) -> Any:
        """Real amplitudes ansatz with RY rotations."""
        circuit = self._builder.create_circuit(self._num_qubits)
        param_idx = 0

        for rep in range(self._reps + 1):
            for i in range(self._num_qubits):
                circuit.ry(parameters[param_idx], i)
                param_idx += 1

            if rep < self._reps:
                for i in range(self._num_qubits - 1):
                    circuit.cx(i, i + 1)

        return circuit

    def _efficient_su2(self, parameters: npt.NDArray[Any]) -> Any:
        """Efficient SU2 ansatz with full single-qubit rotations."""
        circuit = self._builder.create_circuit(self._num_qubits)
        param_idx = 0

        for rep in range(self._reps + 1):
            for i in range(self._num_qubits):
                circuit.rz(parameters[param_idx], i)
                param_idx += 1
                circuit.ry(parameters[param_idx], i)
                param_idx += 1
                circuit.rz(parameters[param_idx], i)
                param_idx += 1

            if rep < self._reps:
                for i in range(self._num_qubits - 1):
                    circuit.cx(i, i + 1)

        return circuit

    def _two_local(self, parameters: npt.NDArray[Any]) -> Any:
        """Two-local ansatz with RY and RZ rotations."""
        circuit = self._builder.create_circuit(self._num_qubits)
        param_idx = 0

        for rep in range(self._reps + 1):
            for i in range(self._num_qubits):
                circuit.ry(parameters[param_idx], i)
                param_idx += 1
                circuit.rz(parameters[param_idx], i)
                param_idx += 1

            if rep < self._reps:
                for i in range(self._num_qubits - 1):
                    circuit.cz(i, i + 1)

        return circuit

    def _hardware_efficient(self, parameters: npt.NDArray[Any]) -> Any:
        """Hardware-efficient ansatz optimized for NISQ devices."""
        circuit = self._builder.create_circuit(self._num_qubits)
        param_idx = 0

        for rep in range(self._reps + 1):
            for i in range(self._num_qubits):
                circuit.ry(parameters[param_idx], i)
                param_idx += 1
                circuit.rz(parameters[param_idx], i)
                param_idx += 1

            if rep < self._reps:
                for i in range(0, self._num_qubits - 1, 2):
                    circuit.cx(i, i + 1)
                for i in range(1, self._num_qubits - 1, 2):
                    circuit.cx(i, i + 1)

        return circuit


class QuantumFeatureMap:
    """
    Quantum feature map for kernel-based learning.

    Maps classical data to quantum Hilbert space for kernel computation.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 2,
        entanglement: str = "full",
    ) -> None:
        """Initialize the feature map."""
        self._num_qubits = num_qubits
        self._reps = reps
        self._entanglement = entanglement
        self._encoder = AnomalyEncodingCircuit(
            num_qubits,
            EncodingType.ZZ_FEATURE_MAP,
            reps,
        )

    def map(self, data: npt.NDArray[Any]) -> Any:
        """
        Map classical data to quantum feature space.

        Args:
            data: Classical feature vector

        Returns:
            Quantum circuit representing the feature map
        """
        return self._encoder.encode(data)

    def compute_kernel(
        self,
        x1: npt.NDArray[Any],
        x2: npt.NDArray[Any],
        shots: int = 1024,
    ) -> float:
        """
        Compute quantum kernel between two data points.

        Uses fidelity estimation: K(x1, x2) = |<phi(x1)|phi(x2)>|^2

        Args:
            x1: First data point
            x2: Second data point
            shots: Number of measurement shots

        Returns:
            Kernel value between 0 and 1
        """
        circuit1 = self.map(x1)
        circuit2 = self.map(x2)

        combined = self._create_kernel_circuit(circuit1, circuit2)

        if isinstance(combined, SimulatedQuantumCircuit):
            counts = combined.simulate(shots)
        else:
            counts = self._simulate_qiskit_circuit(combined, shots)

        zero_string = "0" * self._num_qubits
        zero_count = counts.get(zero_string, 0)
        kernel_value = zero_count / shots

        return kernel_value

    def _create_kernel_circuit(self, circuit1: Any, circuit2: Any) -> Any:
        """Create circuit for kernel estimation."""
        builder = QuantumCircuitBuilder()
        combined = builder.create_circuit(self._num_qubits)

        if isinstance(circuit1, SimulatedQuantumCircuit):
            for gate, qubits, params in circuit1._gates:
                self._apply_gate(combined, gate, qubits, params)

            for gate, qubits, params in reversed(circuit2._gates):
                gate_inv = self._invert_gate(gate)
                params_inv = [-p for p in params] if params else []
                self._apply_gate(combined, gate_inv, qubits, params_inv)
        else:
            combined = circuit1.compose(circuit2.inverse())

        combined.measure_all()
        return combined

    def _apply_gate(
        self,
        circuit: SimulatedQuantumCircuit,
        gate: str,
        qubits: list[int],
        params: list[float],
    ) -> None:
        """Apply a gate to a simulated circuit."""
        if gate == "h":
            circuit.h(qubits[0])
        elif gate == "x":
            circuit.x(qubits[0])
        elif gate == "rx":
            circuit.rx(params[0], qubits[0])
        elif gate == "ry":
            circuit.ry(params[0], qubits[0])
        elif gate == "rz":
            circuit.rz(params[0], qubits[0])
        elif gate == "cx":
            circuit.cx(qubits[0], qubits[1])
        elif gate == "cz":
            circuit.cz(qubits[0], qubits[1])

    def _invert_gate(self, gate: str) -> str:
        """Get inverse of a gate (for most gates, it's the same)."""
        return gate

    def _simulate_qiskit_circuit(self, circuit: Any, shots: int) -> dict[str, int]:
        """Simulate a Qiskit circuit."""
        try:
            from qiskit_aer import AerSimulator

            simulator = AerSimulator()
            result = simulator.run(circuit, shots=shots).result()
            return dict(result.get_counts())
        except ImportError:
            return {"0" * self._num_qubits: shots // 2}


class ErrorMitigationCircuit:
    """
    Error mitigation techniques for NISQ devices.

    Implements Zero-Noise Extrapolation (ZNE) and other mitigation strategies.
    """

    def __init__(
        self,
        method: str = "zne",
        noise_factors: list[float] | None = None,
    ) -> None:
        """Initialize error mitigation."""
        self._method = method
        self._noise_factors = noise_factors or [1.0, 2.0, 3.0]

    def mitigate(
        self,
        circuit: Any,
        executor: Callable[[Any], dict[str, int]],
        observable: str = "expectation",
    ) -> float:
        """
        Execute circuit with error mitigation.

        Args:
            circuit: Quantum circuit to execute
            executor: Function to execute circuit and return counts
            observable: Type of value to estimate

        Returns:
            Error-mitigated expectation value
        """
        if self._method == "zne":
            return self._zero_noise_extrapolation(circuit, executor)
        else:
            counts = executor(circuit)
            return self._compute_expectation(counts)

    def _zero_noise_extrapolation(
        self,
        circuit: Any,
        executor: Callable[[Any], dict[str, int]],
    ) -> float:
        """
        Zero-Noise Extrapolation.

        Runs circuit at multiple noise levels and extrapolates to zero noise.
        """
        expectations = []

        for factor in self._noise_factors:
            scaled_circuit = self._scale_noise(circuit, factor)
            counts = executor(scaled_circuit)
            exp_val = self._compute_expectation(counts)
            expectations.append(exp_val)

        coeffs = np.polyfit(self._noise_factors, expectations, deg=len(self._noise_factors) - 1)
        mitigated = np.polyval(coeffs, 0.0)

        return float(mitigated)

    def _scale_noise(self, circuit: Any, factor: float) -> Any:
        """Scale circuit noise by repeating gates."""
        if factor == 1.0:
            return circuit

        QuantumCircuitBuilder()

        if isinstance(circuit, SimulatedQuantumCircuit):
            scaled = SimulatedQuantumCircuit(circuit.num_qubits, circuit.num_clbits)

            for gate, qubits, params in circuit._gates:
                repetitions = int(factor)
                for _ in range(repetitions):
                    scaled._gates.append((gate, qubits, params))

            return scaled
        else:
            return circuit

    def _compute_expectation(self, counts: dict[str, int]) -> float:
        """Compute expectation value from measurement counts."""
        total = sum(counts.values())
        expectation = 0.0

        for bitstring, count in counts.items():
            parity = sum(int(b) for b in bitstring) % 2
            sign = 1 if parity == 0 else -1
            expectation += sign * count / total

        return expectation
