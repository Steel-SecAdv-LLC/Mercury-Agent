"""
Quantum Computing Module for Mercury Agent.

Provides production-ready quantum computing integration with Qiskit,
including quantum circuit building, execution, and hybrid optimization.

Key Components:
- QuantumCircuitBuilder: Build quantum circuits for anomaly detection
- QuantumExecutor: Execute circuits on simulators or real quantum hardware
- HybridOptimizer: Quantum-classical hybrid optimization
- QuantumAnomalyDetector: Quantum-enhanced anomaly detection

Note: When Qiskit is not available, the module provides simulation fallbacks.
"""

from omni_mercury_engine.quantum_computing.circuits import (
    QuantumCircuitBuilder,
    AnomalyEncodingCircuit,
    VariationalCircuit,
    QuantumFeatureMap,
    ErrorMitigationCircuit,
)
from omni_mercury_engine.quantum_computing.executor import (
    QuantumExecutor,
    ExecutionResult,
    BackendConfig,
    JobStatus,
)
from omni_mercury_engine.quantum_computing.hybrid import (
    HybridOptimizer,
    OptimizationResult,
    QuantumKernel,
    VQEAnomalyDetector,
    QAOAAnomalyDetector,
)
from omni_mercury_engine.quantum_computing.detector import (
    QuantumAnomalyDetector,
    QuantumDetectionResult,
    QuantumResourceEstimate,
)

__all__ = [
    # Circuit building
    "QuantumCircuitBuilder",
    "AnomalyEncodingCircuit",
    "VariationalCircuit",
    "QuantumFeatureMap",
    "ErrorMitigationCircuit",
    # Execution
    "QuantumExecutor",
    "ExecutionResult",
    "BackendConfig",
    "JobStatus",
    # Hybrid optimization
    "HybridOptimizer",
    "OptimizationResult",
    "QuantumKernel",
    "VQEAnomalyDetector",
    "QAOAAnomalyDetector",
    # Detection
    "QuantumAnomalyDetector",
    "QuantumDetectionResult",
    "QuantumResourceEstimate",
]
