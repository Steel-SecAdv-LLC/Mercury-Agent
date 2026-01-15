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


"""
Model modules for Mercury Agent ♱ anomaly detection.

Uses lazy imports to avoid circular dependency issues during package initialization.
"""

from typing import TYPE_CHECKING


# Type-only imports for static analysis (CodeQL, mypy, etc.)
# These are not imported at runtime to support lazy loading
if TYPE_CHECKING:
    from omni_mercury_engine.models.affective import AffectiveAnomalyModel as AffectiveAnomalyModel
    from omni_mercury_engine.models.astrophysical import (
        AstrophysicalAnomalyModel as AstrophysicalAnomalyModel,
    )
    from omni_mercury_engine.models.biometric import BiometricAnomalyModel as BiometricAnomalyModel
    from omni_mercury_engine.models.biometric_advanced import (
        AdvancedBiometricEngine as AdvancedBiometricEngine,
        AgeProgressionEngine as AgeProgressionEngine,
        QuantumAgeVariant as QuantumAgeVariant,
    )
    from omni_mercury_engine.models.chemistry import (
        ChemistryAnomalyDetector as ChemistryAnomalyDetector,
    )
    from omni_mercury_engine.models.consciousness import (
        ConsciousnessPreservationModel as ConsciousnessPreservationModel,
    )
    from omni_mercury_engine.models.multiverse import MultiverseOmniEngine as MultiverseOmniEngine
    from omni_mercury_engine.models.neural import NeuralCognitiveModel as NeuralCognitiveModel
    from omni_mercury_engine.models.neurosymbolic import NeurosymbolicEngine as NeurosymbolicEngine
    from omni_mercury_engine.models.parapsychology import (
        ParapsychologyDetector as ParapsychologyDetector,
    )
    from omni_mercury_engine.models.quantum import QuantumAnomalyModel as QuantumAnomalyModel
    from omni_mercury_engine.models.quantum_engine import (
        QuantumCircuit as QuantumCircuit,
        QuantumEngine as QuantumEngine,
        QuantumGate as QuantumGate,
    )
    from omni_mercury_engine.models.simulation import SimulationModule as SimulationModule
    from omni_mercury_engine.models.sota.association_discrepancy import (
        AnomalyTransformerEncoder as AnomalyTransformerEncoder,
        AssociationDiscrepancyModule as AssociationDiscrepancyModule,
        PriorAssociation as PriorAssociation,
        SeriesAssociation as SeriesAssociation,
    )
    from omni_mercury_engine.models.sota.maat import (
        GatedFeatureFusion as GatedFeatureFusion,
        MAATModel as MAATModel,
        MambaSSM as MambaSSM,
        SparseAttention as SparseAttention,
    )
    from omni_mercury_engine.models.sota.tranad import (
        AdversarialTrainer as AdversarialTrainer,
        FocusScoreConditioning as FocusScoreConditioning,
        MAMLOptimizer as MAMLOptimizer,
        TranADModel as TranADModel,
    )

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
    "QuantumAnomalyModel": "omni_mercury_engine.models.quantum",
    "AstrophysicalAnomalyModel": "omni_mercury_engine.models.astrophysical",
    "BiometricAnomalyModel": "omni_mercury_engine.models.biometric",
    "AffectiveAnomalyModel": "omni_mercury_engine.models.affective",
    "NeuralCognitiveModel": "omni_mercury_engine.models.neural",
    "ConsciousnessPreservationModel": "omni_mercury_engine.models.consciousness",
    "MultiverseOmniEngine": "omni_mercury_engine.models.multiverse",
    "NeurosymbolicEngine": "omni_mercury_engine.models.neurosymbolic",
    "SimulationModule": "omni_mercury_engine.models.simulation",
    "ChemistryAnomalyDetector": "omni_mercury_engine.models.chemistry",
    "ParapsychologyDetector": "omni_mercury_engine.models.parapsychology",
    "AdvancedBiometricEngine": "omni_mercury_engine.models.biometric_advanced",
    "AgeProgressionEngine": "omni_mercury_engine.models.biometric_advanced",
    "QuantumAgeVariant": "omni_mercury_engine.models.biometric_advanced",
    "QuantumEngine": "omni_mercury_engine.models.quantum_engine",
    "QuantumCircuit": "omni_mercury_engine.models.quantum_engine",
    "QuantumGate": "omni_mercury_engine.models.quantum_engine",
    # SOTA - Association Discrepancy (Anomaly Transformer, ICLR 2022)
    "AssociationDiscrepancyModule": "omni_mercury_engine.models.sota.association_discrepancy",
    "AnomalyTransformerEncoder": "omni_mercury_engine.models.sota.association_discrepancy",
    "PriorAssociation": "omni_mercury_engine.models.sota.association_discrepancy",
    "SeriesAssociation": "omni_mercury_engine.models.sota.association_discrepancy",
    # SOTA - TranAD (VLDB 2022)
    "TranADModel": "omni_mercury_engine.models.sota.tranad",
    "FocusScoreConditioning": "omni_mercury_engine.models.sota.tranad",
    "AdversarialTrainer": "omni_mercury_engine.models.sota.tranad",
    "MAMLOptimizer": "omni_mercury_engine.models.sota.tranad",
    # SOTA - MAAT (arXiv 2025)
    "MAATModel": "omni_mercury_engine.models.sota.maat",
    "SparseAttention": "omni_mercury_engine.models.sota.maat",
    "MambaSSM": "omni_mercury_engine.models.sota.maat",
    "GatedFeatureFusion": "omni_mercury_engine.models.sota.maat",
}


def __getattr__(name: str) -> type:
    """Lazy import models on first access."""
    if name in _LAZY_IMPORTS:
        import importlib

        module_path = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        return getattr(module, name)  # type: ignore[no-any-return]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
