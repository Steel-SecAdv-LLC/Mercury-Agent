"""
OMNI ♱ AVA (O♱A)
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
Model modules for OMNI ♱ AVA anomaly detection.

Uses lazy imports to avoid circular dependency issues during package initialization.
"""

__all__ = [
    "AdvancedBiometricEngine",
    "AdversarialTrainer",
    "AffectiveAnomalyModel",
    "AgeProgressionEngine",
    "AnomalyTransformerEncoder",
    "AssociationDiscrepancyModule",
    "AstrophysicalAnomalyModel",
    "BiometricAnomalyModel",
    "ChemistryAnomalyDetector",
    "ConsciousnessPreservationModel",
    "FocusScoreConditioning",
    "GatedFeatureFusion",
    "MAATModel",
    "MAMLOptimizer",
    "MambaSSM",
    "MultiverseOmniEngine",
    "NeuralCognitiveModel",
    "NeurosymbolicEngine",
    "ParapsychologyDetector",
    "PriorAssociation",
    "QuantumAgeVariant",
    "QuantumAnomalyModel",
    "QuantumCircuit",
    "QuantumEngine",
    "QuantumGate",
    "SeriesAssociation",
    "SimulationModule",
    "SparseAttention",
    "TranADModel",
]

_LAZY_IMPORTS = {
    # Core Models
    "QuantumAnomalyModel": "omni_anomaly_engine.models.quantum",
    "AstrophysicalAnomalyModel": "omni_anomaly_engine.models.astrophysical",
    "BiometricAnomalyModel": "omni_anomaly_engine.models.biometric",
    "AffectiveAnomalyModel": "omni_anomaly_engine.models.affective",
    "NeuralCognitiveModel": "omni_anomaly_engine.models.neural",
    "ConsciousnessPreservationModel": "omni_anomaly_engine.models.consciousness",
    "MultiverseOmniEngine": "omni_anomaly_engine.models.multiverse",
    "NeurosymbolicEngine": "omni_anomaly_engine.models.neurosymbolic",
    "SimulationModule": "omni_anomaly_engine.models.simulation",
    "ChemistryAnomalyDetector": "omni_anomaly_engine.models.chemistry",
    "ParapsychologyDetector": "omni_anomaly_engine.models.parapsychology",
    "AdvancedBiometricEngine": "omni_anomaly_engine.models.biometric_advanced",
    "AgeProgressionEngine": "omni_anomaly_engine.models.biometric_advanced",
    "QuantumAgeVariant": "omni_anomaly_engine.models.biometric_advanced",
    "QuantumEngine": "omni_anomaly_engine.models.quantum_engine",
    "QuantumCircuit": "omni_anomaly_engine.models.quantum_engine",
    "QuantumGate": "omni_anomaly_engine.models.quantum_engine",
    # SOTA - Association Discrepancy (Anomaly Transformer, ICLR 2022)
    "AssociationDiscrepancyModule": "omni_anomaly_engine.models.sota.association_discrepancy",
    "AnomalyTransformerEncoder": "omni_anomaly_engine.models.sota.association_discrepancy",
    "PriorAssociation": "omni_anomaly_engine.models.sota.association_discrepancy",
    "SeriesAssociation": "omni_anomaly_engine.models.sota.association_discrepancy",
    # SOTA - TranAD (VLDB 2022)
    "TranADModel": "omni_anomaly_engine.models.sota.tranad",
    "FocusScoreConditioning": "omni_anomaly_engine.models.sota.tranad",
    "AdversarialTrainer": "omni_anomaly_engine.models.sota.tranad",
    "MAMLOptimizer": "omni_anomaly_engine.models.sota.tranad",
    # SOTA - MAAT (arXiv 2025)
    "MAATModel": "omni_anomaly_engine.models.sota.maat",
    "SparseAttention": "omni_anomaly_engine.models.sota.maat",
    "MambaSSM": "omni_anomaly_engine.models.sota.maat",
    "GatedFeatureFusion": "omni_anomaly_engine.models.sota.maat",
}


def __getattr__(name: str) -> type:
    """Lazy import models on first access."""
    if name in _LAZY_IMPORTS:
        import importlib

        module_path = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        return getattr(module, name)  # type: ignore[no-any-return]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
