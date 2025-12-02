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

"""
Model modules for OMNI ♱ AVA anomaly detection.

Uses lazy imports to avoid circular dependency issues during package initialization.
"""

__all__ = [
    "QuantumAnomalyModel",
    "AstrophysicalAnomalyModel",
    "BiometricAnomalyModel",
    "AffectiveAnomalyModel",
    "NeuralCognitiveModel",
    "ConsciousnessPreservationModel",
    "MultiverseOmniEngine",
    "NeurosymbolicEngine",
    "SimulationModule",
    "ChemistryAnomalyDetector",
    "ParapsychologyDetector",
]

_LAZY_IMPORTS = {
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
}


def __getattr__(name):
    """Lazy import models on first access."""
    if name in _LAZY_IMPORTS:
        import importlib

        module_path = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
