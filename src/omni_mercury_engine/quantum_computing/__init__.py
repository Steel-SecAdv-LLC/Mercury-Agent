# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Quantum Computing Module for Mercury Agent.

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
    AnomalyEncodingCircuit,
    ErrorMitigationCircuit,
    QuantumCircuitBuilder,
    QuantumFeatureMap,
    VariationalCircuit,
)
from omni_mercury_engine.quantum_computing.detector import (
    QuantumAnomalyDetector,
    QuantumDetectionResult,
    QuantumResourceEstimate,
)
from omni_mercury_engine.quantum_computing.executor import (
    BackendConfig,
    ExecutionResult,
    JobStatus,
    QuantumExecutor,
)
from omni_mercury_engine.quantum_computing.hybrid import (
    HybridOptimizer,
    OptimizationResult,
    QAOAAnomalyDetector,
    QuantumKernel,
    VQEAnomalyDetector,
)

__all__ = [
    "AnomalyEncodingCircuit",
    "BackendConfig",
    "ErrorMitigationCircuit",
    "ExecutionResult",
    # Hybrid optimization
    "HybridOptimizer",
    "JobStatus",
    "OptimizationResult",
    "QAOAAnomalyDetector",
    # Detection
    "QuantumAnomalyDetector",
    # Circuit building
    "QuantumCircuitBuilder",
    "QuantumDetectionResult",
    # Execution
    "QuantumExecutor",
    "QuantumFeatureMap",
    "QuantumKernel",
    "QuantumResourceEstimate",
    "VQEAnomalyDetector",
    "VariationalCircuit",
]
