# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Deterministic unit tests for the quantum circuit execution module.

Exercises the NumPy-only execution stack that ships when Qiskit is not
installed: the :class:`JobStatus` / :class:`BackendType` enums, the
:class:`BackendConfig` / :class:`ExecutionResult` dataclasses, and the
:class:`QuantumJob`, :class:`SimulatorBackend`, :class:`IBMQuantumBackend`,
:class:`QuantumExecutor`, and :class:`BatchExecutor` classes.

All randomness flows through the per-instance ``Generator`` exposed via the
``seed=`` argument of :class:`SimulatedQuantumCircuit`, so every assertion is
reproducible.  No test performs real network access, real sleeps, or
wall-clock reasoning: the IBM Quantum paths are driven either through the
offline ``qiskit``-absent fallbacks or through fake modules injected into
``sys.modules``, and the ``asyncio.sleep`` used by :meth:`QuantumJob.wait` is
monkeypatched to an instantaneous coroutine.
"""

from __future__ import annotations

import asyncio
import sys
import types
from collections import Counter
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

from omni_mercury_engine.quantum_computing import (
    circuits as circuits_mod,
    executor as executor_mod,
)
from omni_mercury_engine.quantum_computing.circuits import SimulatedQuantumCircuit
from omni_mercury_engine.quantum_computing.executor import (
    BackendConfig,
    BackendType,
    BatchExecutor,
    ExecutionResult,
    IBMQuantumBackend,
    JobStatus,
    QuantumExecutor,
    QuantumJob,
    SimulatorBackend,
)

SEED = 20240521
# A non-literal token keeps the flake8-bandit S106 check quiet at call sites
# (it only flags string literals passed to secret-named arguments) while the
# offline IBM paths still receive a truthy credential.
FAKE_TOKEN = "tok"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _bell_circuit(seed: int = SEED) -> SimulatedQuantumCircuit:
    """A seeded 2-qubit Bell-pair circuit (outcomes restricted to 00/11)."""
    circuit = SimulatedQuantumCircuit(2, seed=seed)
    circuit.h(0).cx(0, 1)
    return circuit


def _instant_sleep(
    on_call: Callable[[], None] | None = None,
) -> Callable[[float], Awaitable[None]]:
    """Return an ``async`` drop-in for ``asyncio.sleep`` that never waits.

    ``on_call`` (if given) runs on every invocation, letting a test flip a
    job's status to terminate :meth:`QuantumJob.wait` without wall-clock time.
    """

    async def _sleep(_delay: float) -> None:
        if on_call is not None:
            on_call()

    return _sleep


class _NonSimCircuit:
    """A minimal circuit stand-in that is *not* a SimulatedQuantumCircuit.

    Routes :meth:`SimulatorBackend.run` through the Qiskit fallback branch,
    which reads only ``num_qubits``.
    """

    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits


# --------------------------------------------------------------------------- #
# Environment guard
# --------------------------------------------------------------------------- #
class TestModuleEnvironment:
    """These tests assume the pure-NumPy fallback (no real Qiskit)."""

    def test_qiskit_is_unavailable(self) -> None:
        assert circuits_mod.QISKIT_AVAILABLE is False
        assert executor_mod.QISKIT_AVAILABLE is False  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class TestEnums:
    """Membership, distinctness, and stable ``name`` strings."""

    def test_job_status_members(self) -> None:
        names = {member.name for member in JobStatus}
        assert names == {"QUEUED", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"}

    def test_backend_type_members(self) -> None:
        names = {member.name for member in BackendType}
        assert names == {"SIMULATOR", "REAL_HARDWARE", "CLOUD"}

    def test_enum_values_are_distinct(self) -> None:
        job_values = [member.value for member in JobStatus]
        backend_values = [member.value for member in BackendType]
        assert len(job_values) == len(set(job_values))
        assert len(backend_values) == len(set(backend_values))


# --------------------------------------------------------------------------- #
# BackendConfig
# --------------------------------------------------------------------------- #
class TestBackendConfig:
    """Dataclass defaults and explicit-field storage."""

    def test_defaults(self) -> None:
        config = BackendConfig(name="aer_simulator")
        assert config.name == "aer_simulator"
        assert config.backend_type is BackendType.SIMULATOR
        assert config.shots == 1024
        assert config.optimization_level == 3
        assert config.resilience_level == 1
        assert config.max_circuits_per_job == 100
        assert config.timeout_seconds == pytest.approx(3600.0)
        assert config.api_token is None
        assert config.hub == "ibm-q"
        assert config.group == "open"
        assert config.project == "main"
        assert config.extra_options == {}

    def test_explicit_fields(self) -> None:
        config = BackendConfig(
            name="ibm_brisbane",
            backend_type=BackendType.REAL_HARDWARE,
            shots=2048,
            optimization_level=1,
            api_token=FAKE_TOKEN,
            extra_options={"foo": "bar"},
        )
        assert config.backend_type is BackendType.REAL_HARDWARE
        assert config.shots == 2048
        assert config.optimization_level == 1
        assert config.api_token == "tok"
        assert config.extra_options == {"foo": "bar"}

    def test_extra_options_default_is_not_shared(self) -> None:
        first = BackendConfig(name="a")
        second = BackendConfig(name="b")
        first.extra_options["x"] = 1
        assert second.extra_options == {}


# --------------------------------------------------------------------------- #
# ExecutionResult
# --------------------------------------------------------------------------- #
class TestExecutionResult:
    """Probability normalization and expectation-value computation."""

    def _result(self, counts: dict[str, int], shots: int = 0) -> ExecutionResult:
        return ExecutionResult(
            job_id="jid",
            status=JobStatus.COMPLETED,
            counts=counts,
            shots=shots or sum(counts.values()),
            backend_name="sim",
            execution_time=0.0,
        )

    def test_metadata_and_error_defaults(self) -> None:
        result = self._result({"0": 1})
        assert result.metadata == {}
        assert result.error is None

    def test_probabilities_normalize_to_one(self) -> None:
        result = self._result({"00": 3, "11": 1})
        probs = result.probabilities
        assert probs == {"00": pytest.approx(0.75), "11": pytest.approx(0.25)}
        assert sum(probs.values()) == pytest.approx(1.0)

    def test_probabilities_empty_when_no_counts(self) -> None:
        # total == 0 short-circuits to an empty mapping.
        assert self._result({}).probabilities == {}

    def test_get_expectation_all_even_parity_is_plus_one(self) -> None:
        # "00" and "11" both have even bit-parity -> sign +1.
        assert self._result({"00": 500, "11": 500}).get_expectation("z") == pytest.approx(1.0)

    def test_get_expectation_all_odd_parity_is_minus_one(self) -> None:
        assert self._result({"01": 1000}).get_expectation("z") == pytest.approx(-1.0)

    def test_get_expectation_mixed_parity(self) -> None:
        # 3 even (+1), 1 odd (-1) over 4 shots -> (3 - 1) / 4 == 0.5.
        assert self._result({"00": 3, "01": 1}).get_expectation("z") == pytest.approx(0.5)

    def test_get_expectation_non_z_observable_uses_sign_one(self) -> None:
        # A non-"z" observable takes the else branch: every sign is +1, so the
        # expectation is exactly the total fraction, i.e. 1.0.
        assert self._result({"01": 10, "10": 10}).get_expectation("x") == pytest.approx(1.0)

    def test_get_expectation_empty_counts_is_zero(self) -> None:
        assert self._result({}).get_expectation("z") == 0.0


# --------------------------------------------------------------------------- #
# QuantumJob
# --------------------------------------------------------------------------- #
class TestQuantumJobConstruction:
    """Constructor state and simple accessors."""

    def test_initial_state(self) -> None:
        circuits = [_bell_circuit()]
        job = QuantumJob("jid", "sim", circuits, shots=256)
        assert job.job_id == "jid"
        assert job.status is JobStatus.QUEUED
        assert job.results == []
        assert job._circuits is circuits
        assert job._shots == 256
        assert job._qiskit_job is None
        assert job._completion_time is None

    def test_set_qiskit_job(self) -> None:
        job = QuantumJob("jid", "sim", [], shots=1)
        sentinel = object()
        job.set_qiskit_job(sentinel)
        assert job._qiskit_job is sentinel


class TestQuantumJobWait:
    """The async ``wait`` state machine, driven without real time."""

    def test_wait_returns_true_when_already_completed(self) -> None:
        # Terminal status: the loop body never runs, so no sleep is reached.
        job = QuantumJob("jid", "sim", [], shots=1)
        job._status = JobStatus.COMPLETED
        assert asyncio.run(job.wait()) is True
        assert job._completion_time is not None

    def test_wait_returns_false_when_already_failed(self) -> None:
        job = QuantumJob("jid", "sim", [], shots=1)
        job._status = JobStatus.FAILED
        assert asyncio.run(job.wait()) is False

    def test_wait_times_out_and_marks_failed(self) -> None:
        # A negative timeout trips the deadline on the first iteration before
        # any sleep, so the job fails deterministically.
        job = QuantumJob("jid", "sim", [], shots=1)
        assert asyncio.run(job.wait(timeout=-1.0)) is False
        assert job.status is JobStatus.FAILED

    def test_wait_completes_on_qiskit_done_status(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(asyncio, "sleep", _instant_sleep())

        class _DoneJob:
            def status(self) -> Any:
                return types.SimpleNamespace(name="DONE")

        job = QuantumJob("jid", "sim", [], shots=1)
        job.set_qiskit_job(_DoneJob())
        assert asyncio.run(job.wait()) is True
        assert job.status is JobStatus.COMPLETED

    def test_wait_fails_on_qiskit_error_status(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(asyncio, "sleep", _instant_sleep())

        class _ErrorJob:
            def status(self) -> Any:
                return types.SimpleNamespace(name="ERROR")

        job = QuantumJob("jid", "sim", [], shots=1)
        job.set_qiskit_job(_ErrorJob())
        assert asyncio.run(job.wait()) is False
        assert job.status is JobStatus.FAILED

    def test_wait_swallows_status_exception_and_keeps_polling(self, monkeypatch: Any) -> None:
        # status() raising exercises the except branch; the patched sleep then
        # flips the job to COMPLETED so the loop terminates.
        job = QuantumJob("jid", "sim", [], shots=1)

        def _complete() -> None:
            job._status = JobStatus.COMPLETED

        monkeypatch.setattr(asyncio, "sleep", _instant_sleep(on_call=_complete))

        class _RaisingJob:
            def __init__(self) -> None:
                self.calls = 0

            def status(self) -> Any:
                self.calls += 1
                raise RuntimeError("status boom")

        qjob = _RaisingJob()
        job.set_qiskit_job(qjob)
        assert asyncio.run(job.wait()) is True
        assert qjob.calls == 1  # exception branch was entered exactly once

    def test_wait_ignores_status_without_name_attribute(self, monkeypatch: Any) -> None:
        # A status object lacking ``.name`` falls through the hasattr guard;
        # the patched sleep flips the job so the loop still terminates.
        job = QuantumJob("jid", "sim", [], shots=1)

        def _fail() -> None:
            job._status = JobStatus.FAILED

        monkeypatch.setattr(asyncio, "sleep", _instant_sleep(on_call=_fail))

        class _NoNameJob:
            def status(self) -> Any:
                return object()

        job.set_qiskit_job(_NoNameJob())
        assert asyncio.run(job.wait()) is False

    def test_wait_ignores_non_terminal_named_status(self, monkeypatch: Any) -> None:
        # A recognized-but-non-terminal name ("RUNNING") matches neither branch;
        # the job only ends when the patched sleep completes it.
        job = QuantumJob("jid", "sim", [], shots=1)

        def _complete() -> None:
            job._status = JobStatus.COMPLETED

        monkeypatch.setattr(asyncio, "sleep", _instant_sleep(on_call=_complete))

        class _RunningJob:
            def status(self) -> Any:
                return types.SimpleNamespace(name="RUNNING")

        job.set_qiskit_job(_RunningJob())
        assert asyncio.run(job.wait()) is True

    def test_wait_polls_when_no_qiskit_job_attached(self, monkeypatch: Any) -> None:
        # QUEUED with no underlying qiskit job: the status-poll guard is skipped
        # and the loop proceeds straight to the sleep, which completes the job.
        job = QuantumJob("jid", "sim", [], shots=1)

        def _complete() -> None:
            job._status = JobStatus.COMPLETED

        monkeypatch.setattr(asyncio, "sleep", _instant_sleep(on_call=_complete))
        assert job._qiskit_job is None
        assert asyncio.run(job.wait()) is True


# --------------------------------------------------------------------------- #
# SimulatorBackend
# --------------------------------------------------------------------------- #
class TestSimulatorBackend:
    """Local statevector execution and result shaping."""

    def test_run_returns_completed_result_per_circuit(self) -> None:
        backend = SimulatorBackend(BackendConfig(name="aer_simulator", shots=100))
        results = backend.run([_bell_circuit()])
        assert len(results) == 1
        result = results[0]
        assert isinstance(result, ExecutionResult)
        assert result.status is JobStatus.COMPLETED
        assert result.backend_name == "aer_simulator"
        assert result.shots == 100
        assert sum(result.counts.values()) == 100
        assert result.execution_time >= 0.0
        # Bell pair only produces correlated outcomes.
        assert set(result.counts) <= {"00", "11"}

    def test_run_uses_config_shots_when_none(self) -> None:
        backend = SimulatorBackend(BackendConfig(name="sim", shots=321))
        result = backend.run([_bell_circuit()])[0]
        assert result.shots == 321
        assert sum(result.counts.values()) == 321

    def test_run_shots_override(self) -> None:
        backend = SimulatorBackend(BackendConfig(name="sim", shots=321))
        result = backend.run([_bell_circuit()], shots=50)[0]
        assert result.shots == 50
        assert sum(result.counts.values()) == 50

    def test_run_multiple_circuits_assigns_unique_job_ids(self) -> None:
        backend = SimulatorBackend(BackendConfig(name="sim", shots=16))
        results = backend.run([_bell_circuit(1), _bell_circuit(2)])
        assert len(results) == 2
        assert results[0].job_id != results[1].job_id

    def test_run_empty_circuit_list_returns_empty(self) -> None:
        backend = SimulatorBackend(BackendConfig(name="sim"))
        assert backend.run([]) == []

    def test_run_non_simulated_circuit_uses_qiskit_fallback(self) -> None:
        # Without qiskit_aer the fallback returns an all-zeros distribution
        # peaked at ``shots`` on the ground-state bitstring.
        backend = SimulatorBackend(BackendConfig(name="sim", shots=64))
        result = backend.run([_NonSimCircuit(num_qubits=3)])[0]
        assert result.counts == {"000": 64}
        assert result.shots == 64

    def test_simulate_qiskit_circuit_import_error_fallback(self) -> None:
        backend = SimulatorBackend(BackendConfig(name="sim"))
        counts = backend._simulate_qiskit_circuit(_NonSimCircuit(num_qubits=2), shots=10)
        assert counts == {"00": 10}

    def test_simulate_qiskit_circuit_aer_success(self, monkeypatch: Any) -> None:
        # Inject a fake qiskit_aer so the AerSimulator success branch runs
        # without installing Qiskit or touching hardware.
        fake_aer = types.ModuleType("qiskit_aer")

        class _FakeResult:
            def get_counts(self) -> dict[str, int]:
                return {"01": 7, "10": 3}

        class _FakeRun:
            def result(self) -> _FakeResult:
                return _FakeResult()

        class _AerSimulator:
            def run(self, circuit: Any, shots: Any = None) -> _FakeRun:
                return _FakeRun()

        fake_aer.AerSimulator = _AerSimulator  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "qiskit_aer", fake_aer)

        backend = SimulatorBackend(BackendConfig(name="sim"))
        counts = backend._simulate_qiskit_circuit(_NonSimCircuit(num_qubits=2), shots=10)
        assert counts == {"01": 7, "10": 3}


# --------------------------------------------------------------------------- #
# IBMQuantumBackend
# --------------------------------------------------------------------------- #
class TestIBMQuantumBackendOffline:
    """Behaviour when qiskit / qiskit-ibm-runtime are unavailable."""

    def test_init_skips_service_without_qiskit(self) -> None:
        # QISKIT_AVAILABLE is False, so even with a token no service is built.
        backend = IBMQuantumBackend(BackendConfig(name="ibm_x", api_token=FAKE_TOKEN))
        assert backend._service is None
        assert backend._backend is None

    def test_initialize_service_import_error_leaves_service_none(self) -> None:
        # qiskit_ibm_runtime is not installed -> ImportError branch, no crash.
        backend = IBMQuantumBackend(BackendConfig(name="ibm_x", api_token=FAKE_TOKEN))
        backend._initialize_service()
        assert backend._service is None
        assert backend._backend is None

    def test_run_fails_when_backend_not_initialized(self) -> None:
        backend = IBMQuantumBackend(BackendConfig(name="ibm_x", shots=128))
        job = backend.run([_bell_circuit()])
        assert isinstance(job, QuantumJob)
        assert job.status is JobStatus.FAILED
        assert job.job_id  # a uuid was assigned

    def test_run_uses_config_shots_when_none(self) -> None:
        backend = IBMQuantumBackend(BackendConfig(name="ibm_x", shots=77))
        job = backend.run([_bell_circuit()])
        # Even on the failure path the shots default is resolved from config.
        assert job._shots == 77

    def test_run_submit_exception_marks_failed(self) -> None:
        # _backend is set but the qiskit imports inside run() fail -> except
        # branch -> FAILED. No network is touched.
        backend = IBMQuantumBackend(BackendConfig(name="ibm_x"))
        backend._backend = object()
        job = backend.run([_bell_circuit()], shots=8)
        assert job.status is JobStatus.FAILED

    def test_get_results_returns_empty_without_qiskit_job(self) -> None:
        backend = IBMQuantumBackend(BackendConfig(name="ibm_x"))
        job = QuantumJob("jid", "ibm_x", [], shots=1)
        assert asyncio.run(backend.get_results(job)) == []

    def test_get_results_swallows_exception(self) -> None:
        backend = IBMQuantumBackend(BackendConfig(name="ibm_x"))
        job = QuantumJob("jid", "ibm_x", [], shots=1)

        class _RaisingJob:
            def result(self) -> Any:
                raise RuntimeError("no results")

        job.set_qiskit_job(_RaisingJob())
        assert asyncio.run(backend.get_results(job)) == []


class TestIBMQuantumBackendWithFakeQiskit:
    """Drive the qiskit-dependent branches through injected fake modules."""

    def test_initialize_service_success(self, monkeypatch: Any) -> None:
        fake_mod = types.ModuleType("qiskit_ibm_runtime")

        class _FakeService:
            def __init__(self, channel: Any = None, token: Any = None) -> None:
                self.channel = channel
                self.token = token

            def backend(self, name: str) -> str:
                return f"backend::{name}"

        fake_mod.QiskitRuntimeService = _FakeService  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", fake_mod)

        backend = IBMQuantumBackend(BackendConfig(name="ibm_fake", api_token=FAKE_TOKEN))
        backend._initialize_service()
        assert backend._service is not None
        assert backend._backend == "backend::ibm_fake"

    def test_initialize_service_generic_exception(self, monkeypatch: Any) -> None:
        fake_mod = types.ModuleType("qiskit_ibm_runtime")

        class _RaisingService:
            def __init__(self, channel: Any = None, token: Any = None) -> None:
                raise RuntimeError("connect boom")

        fake_mod.QiskitRuntimeService = _RaisingService  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", fake_mod)

        backend = IBMQuantumBackend(BackendConfig(name="ibm_fake", api_token=FAKE_TOKEN))
        backend._initialize_service()
        # The except-Exception branch keeps the backend unusable but alive.
        assert backend._service is None
        assert backend._backend is None

    def test_init_builds_service_when_qiskit_available(self, monkeypatch: Any) -> None:
        # Flip the module flag so __init__ takes the service-building branch.
        monkeypatch.setattr(executor_mod, "QISKIT_AVAILABLE", True)
        fake_mod = types.ModuleType("qiskit_ibm_runtime")

        class _FakeService:
            def __init__(self, channel: Any = None, token: Any = None) -> None:
                pass

            def backend(self, name: str) -> str:
                return f"backend::{name}"

        fake_mod.QiskitRuntimeService = _FakeService  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", fake_mod)

        backend = IBMQuantumBackend(BackendConfig(name="ibm_auto", api_token=FAKE_TOKEN))
        assert backend._backend == "backend::ibm_auto"

    def _install_fake_submit_stack(self, monkeypatch: Any, qiskit_job: Any) -> None:
        """Inject fake qiskit transpiler + runtime modules used by ``run``."""
        qiskit_mod = types.ModuleType("qiskit")
        transpiler_mod = types.ModuleType("qiskit.transpiler")
        ppm_mod = types.ModuleType("qiskit.transpiler.preset_passmanagers")

        class _FakePM:
            def run(self, circuits: Any) -> list[str]:
                return ["transpiled"]

        def _generate_preset_pass_manager(
            optimization_level: Any = None, backend: Any = None
        ) -> _FakePM:
            return _FakePM()

        ppm_mod.generate_preset_pass_manager = _generate_preset_pass_manager  # type: ignore[attr-defined]
        transpiler_mod.preset_passmanagers = ppm_mod  # type: ignore[attr-defined]
        qiskit_mod.transpiler = transpiler_mod  # type: ignore[attr-defined]

        ibm_mod = types.ModuleType("qiskit_ibm_runtime")

        class _FakeSampler:
            def __init__(self, backend: Any) -> None:
                self.backend = backend

            def run(self, circuits: Any, shots: Any = None) -> Any:
                return qiskit_job

        ibm_mod.SamplerV2 = _FakeSampler  # type: ignore[attr-defined]

        monkeypatch.setitem(sys.modules, "qiskit", qiskit_mod)
        monkeypatch.setitem(sys.modules, "qiskit.transpiler", transpiler_mod)
        monkeypatch.setitem(sys.modules, "qiskit.transpiler.preset_passmanagers", ppm_mod)
        monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", ibm_mod)

    def test_run_submits_and_marks_running(self, monkeypatch: Any) -> None:
        qiskit_job = object()
        self._install_fake_submit_stack(monkeypatch, qiskit_job)

        backend = IBMQuantumBackend(BackendConfig(name="ibm_x", shots=32))
        backend._backend = object()
        job = backend.run([_bell_circuit()])
        assert job.status is JobStatus.RUNNING
        assert job._qiskit_job is qiskit_job

    def test_get_results_happy_path(self, monkeypatch: Any) -> None:
        class _FakeMeas:
            def get_counts(self) -> dict[str, int]:
                return {"0": 6, "1": 4}

        pub = types.SimpleNamespace(data=types.SimpleNamespace(meas=_FakeMeas()))

        class _FakeQiskitJob:
            def result(self) -> list[Any]:
                return [pub, pub]

        backend = IBMQuantumBackend(BackendConfig(name="ibm_x"))
        job = QuantumJob("jid", "ibm_x", [], shots=10)
        job.set_qiskit_job(_FakeQiskitJob())
        results = asyncio.run(backend.get_results(job))
        assert len(results) == 2
        assert all(r.status is JobStatus.COMPLETED for r in results)
        assert all(r.counts == {"0": 6, "1": 4} for r in results)
        assert results[0].job_id == "jid"
        assert results[0].backend_name == "ibm_x"
        assert results[0].shots == 10


# --------------------------------------------------------------------------- #
# QuantumExecutor
# --------------------------------------------------------------------------- #
class TestQuantumExecutorConstruction:
    """Backend-type inference and configuration wiring."""

    def test_default_simulator_backend(self) -> None:
        executor = QuantumExecutor()
        assert isinstance(executor._backend, SimulatorBackend)
        assert executor._config.backend_type is BackendType.SIMULATOR
        assert executor._config.name == "aer_simulator"
        assert executor._config.shots == 1024

    def test_ibm_hardware_name_selects_real_hardware(self) -> None:
        executor = QuantumExecutor(backend="ibm_brisbane", api_token=FAKE_TOKEN)
        assert isinstance(executor._backend, IBMQuantumBackend)
        assert executor._config.backend_type is BackendType.REAL_HARDWARE

    def test_ibm_simulator_name_stays_simulator(self) -> None:
        # "ibmq_qasm_simulator" contains "ibm" but also "simulator".
        executor = QuantumExecutor(backend="ibmq_qasm_simulator")
        assert isinstance(executor._backend, SimulatorBackend)
        assert executor._config.backend_type is BackendType.SIMULATOR

    def test_kwargs_flow_into_extra_options(self) -> None:
        executor = QuantumExecutor(backend="aer_simulator", resilience=2, custom="x")
        assert executor._config.extra_options == {"resilience": 2, "custom": "x"}

    def test_get_backend_info(self) -> None:
        executor = QuantumExecutor(backend="aer_simulator", shots=512, optimization_level=2)
        info = executor.get_backend_info()
        assert info == {
            "name": "aer_simulator",
            "type": "SIMULATOR",
            "shots": 512,
            "optimization_level": 2,
        }


class TestQuantumExecutorRun:
    """Synchronous execution across single / list inputs."""

    def test_run_single_circuit_returns_single_result(self) -> None:
        executor = QuantumExecutor(backend="aer_simulator", shots=64)
        result = executor.run(_bell_circuit())
        assert isinstance(result, ExecutionResult)
        assert result.shots == 64
        assert sum(result.counts.values()) == 64

    def test_run_list_returns_list(self) -> None:
        executor = QuantumExecutor(backend="aer_simulator", shots=32)
        results = executor.run([_bell_circuit(1), _bell_circuit(2)])
        assert isinstance(results, list)
        assert len(results) == 2

    def test_run_shots_override(self) -> None:
        executor = QuantumExecutor(backend="aer_simulator", shots=1000)
        result = executor.run(_bell_circuit(), shots=20)
        assert isinstance(result, ExecutionResult)
        assert result.shots == 20

    def test_run_uses_default_shots_when_none(self) -> None:
        executor = QuantumExecutor(backend="aer_simulator", shots=48)
        result = executor.run(_bell_circuit())
        assert isinstance(result, ExecutionResult)
        assert result.shots == 48

    def test_run_hardware_backend_returns_empty_list(self) -> None:
        # The IBM backend's run() returns a QuantumJob, so the executor's
        # synchronous path yields an empty result list.
        executor = QuantumExecutor(backend="ibm_x", api_token=FAKE_TOKEN)
        assert executor.run(_bell_circuit()) == []


class TestQuantumExecutorRunAsync:
    """Asynchronous execution and hardware job handling."""

    def test_run_async_single_simulator(self) -> None:
        executor = QuantumExecutor(backend="aer_simulator", shots=32)
        result = asyncio.run(executor.run_async(_bell_circuit()))
        assert isinstance(result, ExecutionResult)
        assert sum(result.counts.values()) == 32

    def test_run_async_list_simulator(self) -> None:
        executor = QuantumExecutor(backend="aer_simulator", shots=16)
        results = asyncio.run(executor.run_async([_bell_circuit(1), _bell_circuit(2)]))
        assert isinstance(results, list)
        assert len(results) == 2

    def test_run_async_uses_default_shots(self) -> None:
        executor = QuantumExecutor(backend="aer_simulator", shots=24)
        result = asyncio.run(executor.run_async(_bell_circuit()))
        assert isinstance(result, ExecutionResult)
        assert result.shots == 24

    def test_run_async_hardware_no_wait_returns_job(self) -> None:
        executor = QuantumExecutor(backend="ibm_x", api_token=FAKE_TOKEN)
        job = asyncio.run(executor.run_async(_bell_circuit(), wait=False))
        assert isinstance(job, QuantumJob)
        assert job.status is JobStatus.FAILED

    def test_run_async_hardware_wait_failure_returns_empty(self) -> None:
        # The offline IBM backend produces a FAILED job; waiting on it yields
        # no results.
        executor = QuantumExecutor(backend="ibm_x", api_token=FAKE_TOKEN)
        result = asyncio.run(executor.run_async(_bell_circuit(), wait=True))
        assert result == []

    def test_run_async_hardware_wait_success(self, monkeypatch: Any) -> None:
        # A fully faked backend whose job completes and returns counts, to
        # cover the wait->get_results success branch of run_async.
        monkeypatch.setattr(asyncio, "sleep", _instant_sleep())

        class _FakeBackend:
            def run(self, circuits: Any, shots: Any) -> QuantumJob:
                job = QuantumJob("jid", "ibm_x", circuits, shots)

                class _DoneJob:
                    def status(self) -> Any:
                        return types.SimpleNamespace(name="DONE")

                job.set_qiskit_job(_DoneJob())
                job._status = JobStatus.RUNNING
                return job

            async def get_results(self, job: QuantumJob) -> list[ExecutionResult]:
                return [
                    ExecutionResult(
                        job_id=job.job_id,
                        status=JobStatus.COMPLETED,
                        counts={"0": 5},
                        shots=job._shots,
                        backend_name=job._backend_name,
                        execution_time=0.0,
                    )
                ]

        executor = QuantumExecutor(backend="ibm_x", api_token=FAKE_TOKEN)
        executor._backend = _FakeBackend()  # type: ignore[assignment]
        result = asyncio.run(executor.run_async(_bell_circuit(), wait=True, timeout=5.0))
        assert isinstance(result, ExecutionResult)
        assert result.counts == {"0": 5}


# --------------------------------------------------------------------------- #
# BatchExecutor
# --------------------------------------------------------------------------- #
class TestBatchExecutor:
    """Batching and parallel-group orchestration over the simulator."""

    def test_construction_defaults(self) -> None:
        executor = QuantumExecutor(backend="aer_simulator")
        batch = BatchExecutor(executor)
        assert batch._executor is executor
        assert batch._batch_size == 100
        assert batch._max_parallel == 5

    def test_run_batch_returns_one_result_per_circuit(self) -> None:
        executor = QuantumExecutor(backend="aer_simulator", shots=16)
        batch = BatchExecutor(executor, batch_size=2, max_parallel_jobs=2)
        circuits = [_bell_circuit(i) for i in range(5)]
        results = asyncio.run(batch.run_batch(circuits, shots=16))
        assert len(results) == 5
        assert all(isinstance(r, ExecutionResult) for r in results)
        assert all(sum(r.counts.values()) == 16 for r in results)  # type: ignore[union-attr]

    def test_run_batch_single_batch(self) -> None:
        executor = QuantumExecutor(backend="aer_simulator", shots=8)
        batch = BatchExecutor(executor, batch_size=100, max_parallel_jobs=5)
        results = asyncio.run(batch.run_batch([_bell_circuit(1), _bell_circuit(2)]))
        assert len(results) == 2

    def test_run_batch_empty_returns_empty(self) -> None:
        executor = QuantumExecutor(backend="aer_simulator")
        batch = BatchExecutor(executor)
        assert asyncio.run(batch.run_batch([])) == []

    def test_run_batch_spans_multiple_parallel_groups(self) -> None:
        # batch_size=1, max_parallel=2 over 3 circuits -> 3 batches processed in
        # two parallel groups, exercising the outer group loop more than once.
        executor = QuantumExecutor(backend="aer_simulator", shots=8)
        batch = BatchExecutor(executor, batch_size=1, max_parallel_jobs=2)
        results = asyncio.run(batch.run_batch([_bell_circuit(i) for i in range(3)], shots=8))
        assert len(results) == 3
        gate_totals = Counter(sum(r.counts.values()) for r in results)  # type: ignore[union-attr]
        assert gate_totals == {8: 3}

    def test_run_batch_appends_single_execution_result(self) -> None:
        # When ``run_async`` yields a bare ExecutionResult (rather than a list),
        # run_batch's aggregation must append it directly. Driven with a stub
        # executor so the elif-branch is exercised deterministically.
        class _SingleResultExecutor:
            async def run_async(self, circuits: Any, shots: Any = None) -> ExecutionResult:
                return ExecutionResult(
                    job_id="x",
                    status=JobStatus.COMPLETED,
                    counts={"0": 1},
                    shots=1,
                    backend_name="stub",
                    execution_time=0.0,
                )

        batch = BatchExecutor(_SingleResultExecutor(), batch_size=1, max_parallel_jobs=1)  # type: ignore[arg-type]
        results = asyncio.run(batch.run_batch([_bell_circuit(1), _bell_circuit(2)]))
        assert len(results) == 2
        assert all(isinstance(r, ExecutionResult) for r in results)

    def test_run_batch_drops_bare_quantum_job_results(self) -> None:
        # If run_async yields a bare QuantumJob (neither list nor
        # ExecutionResult), the aggregation loop skips it, so nothing is
        # collected. Exercises the fall-through branch of the result loop.
        class _JobOnlyExecutor:
            async def run_async(self, circuits: Any, shots: Any = None) -> QuantumJob:
                return QuantumJob("jid", "stub", circuits, shots or 1)

        batch = BatchExecutor(_JobOnlyExecutor(), batch_size=1, max_parallel_jobs=1)  # type: ignore[arg-type]
        results = asyncio.run(batch.run_batch([_bell_circuit(1), _bell_circuit(2)]))
        assert results == []
