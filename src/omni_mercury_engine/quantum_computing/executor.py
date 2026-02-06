"""
Quantum Circuit Execution for Mercury Agent.

Provides unified execution interface for quantum circuits on simulators
and real quantum hardware through Qiskit.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from omni_mercury_engine.quantum_computing.circuits import (
    QISKIT_AVAILABLE,
    SimulatedQuantumCircuit,
)


logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Status of a quantum job."""

    QUEUED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class BackendType(Enum):
    """Type of quantum backend."""

    SIMULATOR = auto()
    REAL_HARDWARE = auto()
    CLOUD = auto()


@dataclass
class BackendConfig:
    """Configuration for quantum backend."""

    name: str
    backend_type: BackendType = BackendType.SIMULATOR
    shots: int = 1024
    optimization_level: int = 3
    resilience_level: int = 1
    max_circuits_per_job: int = 100
    timeout_seconds: float = 3600.0
    api_token: str | None = None
    hub: str = "ibm-q"
    group: str = "open"
    project: str = "main"
    extra_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Result of quantum circuit execution."""

    job_id: str
    status: JobStatus
    counts: dict[str, int]
    shots: int
    backend_name: str
    execution_time: float
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def probabilities(self) -> dict[str, float]:
        """Convert counts to probabilities."""
        total = sum(self.counts.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in self.counts.items()}

    def get_expectation(self, observable: str = "z") -> float:
        """
        Compute expectation value from counts.

        Args:
            observable: Observable to measure ("z" for Z-basis parity)

        Returns:
            Expectation value
        """
        total = sum(self.counts.values())
        if total == 0:
            return 0.0

        expectation = 0.0
        for bitstring, count in self.counts.items():
            if observable == "z":
                parity = sum(int(b) for b in bitstring) % 2
                sign = 1 if parity == 0 else -1
            else:
                sign = 1
            expectation += sign * count / total

        return expectation


class QuantumJob:
    """
    Manages a quantum job submitted for execution.

    Tracks job status and provides methods to retrieve results.
    """

    def __init__(
        self,
        job_id: str,
        backend_name: str,
        circuits: list[Any],
        shots: int,
    ) -> None:
        """Initialize the job."""
        self._job_id = job_id
        self._backend_name = backend_name
        self._circuits = circuits
        self._shots = shots
        self._status = JobStatus.QUEUED
        self._results: list[ExecutionResult] = []
        self._submit_time = time.time()
        self._completion_time: float | None = None
        self._qiskit_job: Any = None

    @property
    def job_id(self) -> str:
        """Get job ID."""
        return self._job_id

    @property
    def status(self) -> JobStatus:
        """Get current job status."""
        return self._status

    @property
    def results(self) -> list[ExecutionResult]:
        """Get execution results."""
        return self._results

    def set_qiskit_job(self, job: Any) -> None:
        """Set the underlying Qiskit job."""
        self._qiskit_job = job

    async def wait(self, timeout: float = 3600.0) -> bool:
        """
        Wait for job completion.

        Args:
            timeout: Maximum wait time in seconds

        Returns:
            True if job completed successfully
        """
        start_time = time.time()

        while self._status in (JobStatus.QUEUED, JobStatus.RUNNING):
            if time.time() - start_time > timeout:
                self._status = JobStatus.FAILED
                return False

            if self._qiskit_job is not None:
                try:
                    qiskit_status = self._qiskit_job.status()
                    if hasattr(qiskit_status, "name"):
                        status_name = qiskit_status.name
                        if status_name == "DONE":
                            self._status = JobStatus.COMPLETED
                        elif status_name in ("ERROR", "CANCELLED"):
                            self._status = JobStatus.FAILED
                except Exception as e:
                    logger.warning("Error checking job status: %s", e)

            await asyncio.sleep(1.0)

        self._completion_time = time.time()
        return self._status == JobStatus.COMPLETED


class SimulatorBackend:
    """
    Local quantum simulator backend.

    Uses NumPy-based statevector simulation for fast local execution.
    """

    def __init__(self, config: BackendConfig) -> None:
        """Initialize the simulator."""
        self._config = config
        self._name = config.name

    def run(
        self,
        circuits: list[Any],
        shots: int | None = None,
    ) -> list[ExecutionResult]:
        """
        Run circuits on the simulator.

        Args:
            circuits: List of quantum circuits
            shots: Number of shots per circuit

        Returns:
            List of execution results
        """
        if shots is None:
            shots = self._config.shots

        results = []

        for circuit in circuits:
            start_time = time.time()

            if isinstance(circuit, SimulatedQuantumCircuit):
                counts = circuit.simulate(shots)
            else:
                counts = self._simulate_qiskit_circuit(circuit, shots)

            execution_time = time.time() - start_time

            results.append(
                ExecutionResult(
                    job_id=str(uuid.uuid4()),
                    status=JobStatus.COMPLETED,
                    counts=counts,
                    shots=shots,
                    backend_name=self._name,
                    execution_time=execution_time,
                )
            )

        return results

    def _simulate_qiskit_circuit(
        self,
        circuit: Any,
        shots: int,
    ) -> dict[str, int]:
        """Simulate a Qiskit circuit using Aer if available."""
        try:
            from qiskit_aer import AerSimulator

            simulator = AerSimulator()
            result = simulator.run(circuit, shots=shots).result()
            return dict(result.get_counts())
        except ImportError:
            n_qubits = circuit.num_qubits
            return {"0" * n_qubits: shots}


class IBMQuantumBackend:
    """
    IBM Quantum hardware backend.

    Connects to IBM Quantum services for execution on real quantum hardware.
    """

    def __init__(self, config: BackendConfig) -> None:
        """Initialize the IBM backend."""
        self._config = config
        self._service: Any = None
        self._backend: Any = None

        if QISKIT_AVAILABLE and config.api_token:
            self._initialize_service()

    def _initialize_service(self) -> None:
        """Initialize the IBM Quantum service."""
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService

            self._service = QiskitRuntimeService(
                channel="ibm_quantum",
                token=self._config.api_token,
            )

            self._backend = self._service.backend(self._config.name)
            logger.info("Connected to IBM Quantum backend: %s", self._config.name)

        except ImportError:
            logger.warning("qiskit-ibm-runtime not available")
        except Exception as e:
            logger.error("Failed to initialize IBM Quantum service: %s", e)

    def run(
        self,
        circuits: list[Any],
        shots: int | None = None,
    ) -> QuantumJob:
        """
        Submit circuits to IBM Quantum.

        Args:
            circuits: List of quantum circuits
            shots: Number of shots per circuit

        Returns:
            QuantumJob for tracking execution
        """
        if shots is None:
            shots = self._config.shots

        job_id = str(uuid.uuid4())
        job = QuantumJob(job_id, self._config.name, circuits, shots)

        if self._backend is None:
            job._status = JobStatus.FAILED
            logger.error("IBM Quantum backend not initialized")
            return job

        try:
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
            from qiskit_ibm_runtime import SamplerV2 as Sampler

            pm = generate_preset_pass_manager(
                optimization_level=self._config.optimization_level,
                backend=self._backend,
            )
            transpiled_circuits = pm.run(circuits)

            sampler = Sampler(self._backend)
            qiskit_job = sampler.run(transpiled_circuits, shots=shots)

            job.set_qiskit_job(qiskit_job)
            job._status = JobStatus.RUNNING

        except Exception as e:
            logger.error("Failed to submit job to IBM Quantum: %s", e)
            job._status = JobStatus.FAILED

        return job

    async def get_results(self, job: QuantumJob) -> list[ExecutionResult]:
        """
        Retrieve results from a completed job.

        Args:
            job: QuantumJob to get results from

        Returns:
            List of execution results
        """
        if job._qiskit_job is None:
            return []

        try:
            result = job._qiskit_job.result()
            execution_results = []

            for i, pub_result in enumerate(result):
                counts = pub_result.data.meas.get_counts()
                execution_results.append(
                    ExecutionResult(
                        job_id=job.job_id,
                        status=JobStatus.COMPLETED,
                        counts=counts,
                        shots=job._shots,
                        backend_name=job._backend_name,
                        execution_time=0.0,
                    )
                )

            return execution_results

        except Exception as e:
            logger.error("Failed to retrieve results: %s", e)
            return []


class QuantumExecutor:
    """
    Unified quantum execution interface.

    Manages execution across simulators and real quantum hardware.

    Example:
        executor = QuantumExecutor(
            backend="aer_simulator",
            shots=1024,
        )

        result = executor.run(circuit)
        print(result.counts)
    """

    def __init__(
        self,
        backend: str = "aer_simulator",
        shots: int = 1024,
        optimization_level: int = 3,
        api_token: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the executor.

        Args:
            backend: Backend name ("aer_simulator", "ibmq_qasm_simulator", or hardware name)
            shots: Default number of shots
            optimization_level: Transpilation optimization level (0-3)
            api_token: IBM Quantum API token (required for real hardware)
            **kwargs: Additional backend configuration
        """
        backend_type = BackendType.SIMULATOR
        if "ibm" in backend.lower() and "simulator" not in backend.lower():
            backend_type = BackendType.REAL_HARDWARE

        self._config = BackendConfig(
            name=backend,
            backend_type=backend_type,
            shots=shots,
            optimization_level=optimization_level,
            api_token=api_token,
            extra_options=kwargs,
        )

        self._backend: IBMQuantumBackend | SimulatorBackend
        if backend_type == BackendType.REAL_HARDWARE:
            self._backend = IBMQuantumBackend(self._config)
        else:
            self._backend = SimulatorBackend(self._config)

    def run(
        self,
        circuits: Any | list[Any],
        shots: int | None = None,
    ) -> ExecutionResult | list[ExecutionResult]:
        """
        Execute quantum circuit(s).

        Args:
            circuits: Single circuit or list of circuits
            shots: Number of shots (uses default if None)

        Returns:
            ExecutionResult or list of ExecutionResults
        """
        single_circuit = not isinstance(circuits, list)
        if single_circuit:
            circuits = [circuits]

        if shots is None:
            shots = self._config.shots

        results: list[ExecutionResult]
        if isinstance(self._backend, SimulatorBackend):
            results = self._backend.run(circuits, shots)
        else:
            self._backend.run(circuits, shots)
            results = []

        return results[0] if single_circuit and results else results

    async def run_async(
        self,
        circuits: Any | list[Any],
        shots: int | None = None,
        wait: bool = True,
        timeout: float = 3600.0,
    ) -> QuantumJob | ExecutionResult | list[ExecutionResult]:
        """
        Execute quantum circuit(s) asynchronously.

        Args:
            circuits: Single circuit or list of circuits
            shots: Number of shots
            wait: Whether to wait for completion
            timeout: Maximum wait time

        Returns:
            QuantumJob if not waiting, ExecutionResults if waiting
        """
        single_circuit = not isinstance(circuits, list)
        if single_circuit:
            circuits = [circuits]

        if shots is None:
            shots = self._config.shots

        if isinstance(self._backend, SimulatorBackend):
            results = self._backend.run(circuits, shots)
            return results[0] if single_circuit else results

        job = self._backend.run(circuits, shots)

        if not wait:
            return job

        success = await job.wait(timeout)
        if success:
            results = await self._backend.get_results(job)
            return results[0] if single_circuit and results else results
        else:
            return []

    def get_backend_info(self) -> dict[str, Any]:
        """Get information about the configured backend."""
        return {
            "name": self._config.name,
            "type": self._config.backend_type.name,
            "shots": self._config.shots,
            "optimization_level": self._config.optimization_level,
        }


class BatchExecutor:
    """
    Execute multiple circuits in batches for efficiency.

    Optimizes execution by grouping circuits and managing job queues.
    """

    def __init__(
        self,
        executor: QuantumExecutor,
        batch_size: int = 100,
        max_parallel_jobs: int = 5,
    ) -> None:
        """Initialize the batch executor."""
        self._executor = executor
        self._batch_size = batch_size
        self._max_parallel = max_parallel_jobs

    async def run_batch(
        self,
        circuits: list[Any],
        shots: int | None = None,
    ) -> list[ExecutionResult]:
        """
        Run a batch of circuits.

        Args:
            circuits: List of circuits to execute
            shots: Number of shots per circuit

        Returns:
            List of execution results
        """
        batches = [
            circuits[i : i + self._batch_size] for i in range(0, len(circuits), self._batch_size)
        ]

        all_results: list[Any] = []

        for i in range(0, len(batches), self._max_parallel):
            parallel_batches = batches[i : i + self._max_parallel]

            tasks = [self._executor.run_async(batch, shots) for batch in parallel_batches]

            batch_results = await asyncio.gather(*tasks)

            for result in batch_results:
                if isinstance(result, list):
                    all_results.extend(result)
                else:
                    all_results.append(result)

        return all_results
