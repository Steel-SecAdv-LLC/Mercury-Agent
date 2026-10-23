# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What every public ``*Detector`` class is, when it is not a manifest entry.

``DETECTOR_MANIFEST`` lists the detectors reachable through the numeric seams
(and the adapter entries that serve the domain detectors behind them). The
source also holds many more classes named ``*Detector``: the parts a served
detector is built from, abstract bases, a deprecated shim, and domain
detectors with their own API that are reached by importing them. Left
unrecorded, the next one added is invisible: nobody can tell whether it is
meant to be a manifest entry, a part of something, or forgotten.

This table accounts for each of them, and
``tests/core/test_detector_roles.py`` re-derives every claim from the source
and fails when a ``*Detector`` class is in neither the manifest (as an entry
or an adapter's ``wraps`` target) nor here, or when a claim is false:

* ``abstract`` -- a Protocol, ABC or base with abstract methods; never
  instantiated on its own.
* ``component`` -- constructed by every class in ``parts``, which owns or
  drives it; it is not a seam of its own.
* ``variant`` -- a subclass of the class in ``parts`` (a manifest entry or
  another detector), reached by importing it.
* ``deprecated`` -- its module warns on import and names its replacement.
* ``library`` -- a detector with its own domain API (not a ``BaseDetector``,
  no numeric seam), reached by importing it; each has at least one test.

``scripts/generate_detector_catalog.py`` renders this table into
``docs/DETECTORS.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = ["DETECTOR_ROLES", "DetectorRole", "RoleKind"]

RoleKind = Literal["abstract", "component", "variant", "deprecated", "library"]


@dataclass(frozen=True)
class DetectorRole:
    """One detector class's role outside the manifest.

    Attributes:
        kind: See the module docstring.
        parts: ``module:Class`` of the classes the claim refers to: every
            constructor of a ``component``, the base of a ``variant``;
            empty otherwise.
    """

    kind: RoleKind
    parts: tuple[str, ...] = ()


#: ``module:Class`` -> role, for every public ``*Detector`` class that is not a
#: manifest entry or an adapter's ``wraps`` target.
DETECTOR_ROLES: dict[str, DetectorRole] = {
    # -- abstract (6) ----------------------------------------
    "omni_mercury_engine.core.base:BaseDetector": DetectorRole("abstract"),
    "omni_mercury_engine.core.stacking_fusion:BaseDetector": DetectorRole("abstract"),
    "omni_mercury_engine.core.rigorous_benchmark:AnomalyDetector": DetectorRole("abstract", ()),
    "omni_mercury_engine.detectors.visual.base_visual:BaseVisualDetector": DetectorRole(
        "abstract", ()
    ),
    "omni_mercury_engine.detectors.vlm.base_vlm:BaseVLMDetector": DetectorRole("abstract", ()),
    "omni_mercury_engine.ml.ensemble_coordinator:Detector": DetectorRole("abstract", ()),
    # -- component (53) ----------------------------------------
    "omni_mercury_engine.biometric.fingerprint_recognition:FingerprintLivenessDetector": DetectorRole(
        "component",
        (
            "omni_mercury_engine.biometric:BiometricAnomalyDetector",
            "omni_mercury_engine.biometric.fingerprint_recognition:FingerprintRecognizer",
        ),
    ),
    "omni_mercury_engine.biometric.iris_recognition:IrisLivenessDetector": DetectorRole(
        "component",
        (
            "omni_mercury_engine.biometric:BiometricAnomalyDetector",
            "omni_mercury_engine.biometric.iris_recognition:IrisRecognizer",
        ),
    ),
    "omni_mercury_engine.biometric.voice_recognition:VoiceActivityDetector": DetectorRole(
        "component", ("omni_mercury_engine.biometric.voice_recognition:VoiceRecognizer",)
    ),
    "omni_mercury_engine.biometric.voice_recognition:VoiceLivenessDetector": DetectorRole(
        "component",
        (
            "omni_mercury_engine.biometric:BiometricAnomalyDetector",
            "omni_mercury_engine.biometric.voice_recognition:VoiceRecognizer",
        ),
    ),
    "omni_mercury_engine.cognitive.anomaly_detection:IntegratedAnomalyDetector": DetectorRole(
        "component", ("omni_mercury_engine.cognitive.orchestrator:CognitiveOrchestrator",)
    ),
    "omni_mercury_engine.cognitive.neural_memory_layer:PatternDetector": DetectorRole(
        "component", ("omni_mercury_engine.cognitive.neural_memory_layer:NeuralMemoryLayer",)
    ),
    "omni_mercury_engine.cognitive.predictive_coding:PredictiveCodingDetector": DetectorRole(
        "component", ("omni_mercury_engine.cognitive.predictive_coding:MercuryPredictiveCoding",)
    ),
    "omni_mercury_engine.core.base_domains:BaseDomainDetector": DetectorRole(
        "component", ("omni_mercury_engine.core.gosnn_integration:GOSNNIntegration",)
    ),
    "omni_mercury_engine.detectors.adaptive_detector:CovarianceAwareDetector": DetectorRole(
        "component", ("omni_mercury_engine.detectors.adaptive_detector:AdaptiveAnomalyDetector",)
    ),
    "omni_mercury_engine.detectors.adaptive_detector:TemporalPatternDetector": DetectorRole(
        "component", ("omni_mercury_engine.detectors.adaptive_detector:AdaptiveAnomalyDetector",)
    ),
    "omni_mercury_engine.detectors.conformal_martingale:ConformalChangeDetector": DetectorRole(
        "component",
        ("omni_mercury_engine.medical.pandemic.bio_threats.pathogen_detector:PathogenDetector",),
    ),
    "omni_mercury_engine.detectors.drift:ChiSquaredDriftDetector": DetectorRole(
        "component", ("omni_mercury_engine.detectors.drift:EnsembleDriftDetector",)
    ),
    "omni_mercury_engine.detectors.drift:CovarianceDriftDetector": DetectorRole(
        "component", ("omni_mercury_engine.detectors.drift:EnsembleDriftDetector",)
    ),
    "omni_mercury_engine.detectors.drift:EnsembleDriftDetector": DetectorRole(
        "component", ("omni_mercury_engine.engine:OmniMercuryEngine",)
    ),
    "omni_mercury_engine.detectors.drift:KolmogorovSmirnovDriftDetector": DetectorRole(
        "component", ("omni_mercury_engine.detectors.drift:EnsembleDriftDetector",)
    ),
    "omni_mercury_engine.detectors.drift:OnlineDriftDetector": DetectorRole(
        "component", ("omni_mercury_engine.streaming.engine_stream:EngineStreamSession",)
    ),
    "omni_mercury_engine.detectors.drift:PopulationStabilityIndexDetector": DetectorRole(
        "component", ("omni_mercury_engine.detectors.drift:EnsembleDriftDetector",)
    ),
    "omni_mercury_engine.detectors.drift:PrincipalAxisDriftDetector": DetectorRole(
        "component", ("omni_mercury_engine.detectors.drift:EnsembleDriftDetector",)
    ),
    "omni_mercury_engine.detectors.economic.financial_crisis_detector:BankingStressDetector": DetectorRole(
        "component",
        (
            "omni_mercury_engine.detectors.economic.financial_crisis_detector:FinancialCrisisDetector",
        ),
    ),
    "omni_mercury_engine.detectors.economic.financial_crisis_detector:FraudDetector": DetectorRole(
        "component",
        (
            "omni_mercury_engine.detectors.economic.financial_crisis_detector:FinancialCrisisDetector",
        ),
    ),
    "omni_mercury_engine.detectors.economic.financial_crisis_detector:MarketCrashDetector": DetectorRole(
        "component",
        (
            "omni_mercury_engine.detectors.economic.financial_crisis_detector:FinancialCrisisDetector",
        ),
    ),
    "omni_mercury_engine.detectors.energy.emp_detector:E1PulseDetector": DetectorRole(
        "component", ("omni_mercury_engine.detectors.energy.emp_detector:EMPDetector",)
    ),
    "omni_mercury_engine.detectors.energy.emp_detector:E2PulseDetector": DetectorRole(
        "component", ("omni_mercury_engine.detectors.energy.emp_detector:EMPDetector",)
    ),
    "omni_mercury_engine.detectors.energy.emp_detector:E3PulseDetector": DetectorRole(
        "component", ("omni_mercury_engine.detectors.energy.emp_detector:EMPDetector",)
    ),
    "omni_mercury_engine.detectors.energy.emp_detector:IntentionalEMIDetector": DetectorRole(
        "component", ("omni_mercury_engine.detectors.energy.emp_detector:EMPDetector",)
    ),
    "omni_mercury_engine.detectors.geological.volcanic:InSARDeformationDetector": DetectorRole(
        "component", ("omni_mercury_engine.detectors.geological.volcanic:VolcanicEruptionDetector",)
    ),
    "omni_mercury_engine.detectors.geological.volcanic:SeismicSwarmDetector": DetectorRole(
        "component", ("omni_mercury_engine.detectors.geological.volcanic:VolcanicEruptionDetector",)
    ),
    "omni_mercury_engine.detectors.geological.volcanic:ThermalHotspotDetector": DetectorRole(
        "component", ("omni_mercury_engine.detectors.geological.volcanic:VolcanicEruptionDetector",)
    ),
    "omni_mercury_engine.detectors.geological.wildfire:FireIgnitionDetector": DetectorRole(
        "component", ("omni_mercury_engine.detectors.geological.wildfire:WildfireDetector",)
    ),
    "omni_mercury_engine.detectors.lstm_ae:AnomalyDetector": DetectorRole(
        "component", ("omni_mercury_engine.detectors.lstm_ae:LSTMAutoencoderDetector",)
    ),
    "omni_mercury_engine.detectors.marine.biodiversity_detector:CoralBleachingDetector": DetectorRole(
        "component",
        ("omni_mercury_engine.detectors.marine.biodiversity_detector:MarineBiodiversityDetector",),
    ),
    "omni_mercury_engine.detectors.statistical_extended:CUSUMDetector": DetectorRole(
        "component",
        ("omni_mercury_engine.detectors.statistical_extended:ExtendedStatisticalDetector",),
    ),
    "omni_mercury_engine.detectors.statistical_extended:DBSCANDetector": DetectorRole(
        "component",
        ("omni_mercury_engine.detectors.statistical_extended:ExtendedStatisticalDetector",),
    ),
    "omni_mercury_engine.detectors.statistical_extended:GESDTest": DetectorRole(
        "component",
        ("omni_mercury_engine.detectors.statistical_extended:ExtendedStatisticalDetector",),
    ),
    "omni_mercury_engine.detectors.statistical_extended:GrubbsTest": DetectorRole(
        "component",
        ("omni_mercury_engine.detectors.statistical_extended:ExtendedStatisticalDetector",),
    ),
    "omni_mercury_engine.detectors.statistical_extended:LOFDetector": DetectorRole(
        "component",
        ("omni_mercury_engine.detectors.statistical_extended:ExtendedStatisticalDetector",),
    ),
    "omni_mercury_engine.detectors.statistical_extended:MADDetector": DetectorRole(
        "component",
        ("omni_mercury_engine.detectors.statistical_extended:ExtendedStatisticalDetector",),
    ),
    "omni_mercury_engine.detectors.statistical_extended:MCDDetector": DetectorRole(
        "component",
        ("omni_mercury_engine.detectors.statistical_extended:ExtendedStatisticalDetector",),
    ),
    "omni_mercury_engine.distributed.cluster:DistributedAnomalyDetector": DetectorRole(
        "component", ("omni_mercury_engine.distributed.cluster:DistributedMercuryCluster",)
    ),
    "omni_mercury_engine.federated_learning.server:FederatedAnomalyDetector": DetectorRole(
        "component",
        ("omni_mercury_engine.federated_learning.cisa_coordinator:CISAFederatedCoordinator",),
    ),
    "omni_mercury_engine.infrastructure.healthcare_emergency:HealthcareEmergencyDetector": DetectorRole(
        "component",
        ("omni_mercury_engine.medical.medical_cure_predictor:TemporalVitalSignsDetector",),
    ),
    "omni_mercury_engine.medical.critical_care.neurocritical_care:StrokeDetector": DetectorRole(
        "component",
        (
            "omni_mercury_engine.medical.critical_care.neurocritical_care:NeurocriticalCarePredictor",
        ),
    ),
    "omni_mercury_engine.medical.medical_cure_predictor:MedicalImagingAnomalyDetector": DetectorRole(
        "component", ("omni_mercury_engine.medical.medical_cure_predictor:MedicalCurePredictor",)
    ),
    "omni_mercury_engine.medical.medical_cure_predictor:TemporalVitalSignsDetector": DetectorRole(
        "component", ("omni_mercury_engine.medical.medical_cure_predictor:MedicalCurePredictor",)
    ),
    "omni_mercury_engine.ml.alarm_policy:StreamingAlarmDetector": DetectorRole(
        "component", ("omni_mercury_engine.infrastructure.streaming:StreamingAnomalyPipeline",)
    ),
    "omni_mercury_engine.models.foundation.llm_adapter:ZeroShotAnomalyDetector": DetectorRole(
        "component",
        (
            "omni_mercury_engine.engine:OmniMercuryEngine",
            "omni_mercury_engine.models.foundation.llm_adapter:TextLogAnomalyDetector",
        ),
    ),
    "omni_mercury_engine.models.foundation.matrix_profile:MatrixProfileDetector": DetectorRole(
        "component", ("omni_mercury_engine.models.foundation.ensemble:FoundationEnsemble",)
    ),
    "omni_mercury_engine.quantum_computing.hybrid:QAOAAnomalyDetector": DetectorRole(
        "component", ("omni_mercury_engine.quantum_computing.detector:QuantumAnomalyDetector",)
    ),
    "omni_mercury_engine.quantum_computing.hybrid:VQEAnomalyDetector": DetectorRole(
        "component", ("omni_mercury_engine.quantum_computing.detector:QuantumAnomalyDetector",)
    ),
    "omni_mercury_engine.security.cybint_subprocessor:C2InfrastructureDetector": DetectorRole(
        "component", ("omni_mercury_engine.security.cybint_subprocessor:CYBINTSubProcessor",)
    ),
    "omni_mercury_engine.security.tempest_detection:VideoEmanationDetector": DetectorRole(
        "component", ("omni_mercury_engine.security.tempest_detection:TEMPESTDetector",)
    ),
    "omni_mercury_engine.security.traffic_analysis:CovertChannelDetector": DetectorRole(
        "component", ("omni_mercury_engine.security.traffic_analysis:TrafficAnalysisEngine",)
    ),
    "omni_mercury_engine.space.disaster_precursor_detector:IonosphericDisturbanceDetector": DetectorRole(
        "component",
        ("omni_mercury_engine.space.disaster_precursor_detector:DisasterPrecursorDetector",),
    ),
    # -- variant (1) ----------------------------------------
    "omni_mercury_engine.security.realtime_threat_detection:AdaptiveThreatDetector": DetectorRole(
        "variant",
        ("omni_mercury_engine.security.realtime_threat_detection:RealTimeThreatDetector",),
    ),
    # -- deprecated (1) ----------------------------------------
    "omni_mercury_engine.detectors.advanced.gwo_ensemble:GWOEnsembleDetector": DetectorRole(
        "deprecated", ()
    ),
    # -- library (29) ----------------------------------------
    "omni_mercury_engine.biometric:BiometricAnomalyDetector": DetectorRole("library", ()),
    "omni_mercury_engine.compliance.osha_anomaly:OSHAComplianceDetector": DetectorRole(
        "library", ()
    ),
    "omni_mercury_engine.core.conformal_prediction:ConformalAnomalyDetector": DetectorRole(
        "library", ()
    ),
    "omni_mercury_engine.detectors.advanced.adversarial_ae:AdversarialAutoencoderDetector": DetectorRole(
        "library", ()
    ),
    "omni_mercury_engine.detectors.advanced.contrastive_detector:ContrastiveLearningDetector": DetectorRole(
        "library", ()
    ),
    "omni_mercury_engine.detectors.advanced.copod_detector:COPODDetector": DetectorRole(
        "library", ()
    ),
    "omni_mercury_engine.detectors.advanced.multi_scale_transformer:MultiScaleTransformerDetector": DetectorRole(
        "library", ()
    ),
    "omni_mercury_engine.detectors.vlm.blip_vlm:BLIPVLMDetector": DetectorRole("library", ()),
    "omni_mercury_engine.detectors.vlm.statistical_vlm:StatisticalVLMDetector": DetectorRole(
        "library", ()
    ),
    "omni_mercury_engine.emergent.emergent_life_detector:EmergentLifeDetector": DetectorRole(
        "library", ()
    ),
    "omni_mercury_engine.infrastructure.chemical_nuclear:ChemicalNuclearDetector": DetectorRole(
        "library", ()
    ),
    "omni_mercury_engine.infrastructure.communications_it:CommunicationsITDetector": DetectorRole(
        "library", ()
    ),
    "omni_mercury_engine.infrastructure.energy_dams:EnergyDamsDetector": DetectorRole(
        "library", ()
    ),
    "omni_mercury_engine.infrastructure.humanitarian.agrifood_security:AgriFoodSecurityDetector": DetectorRole(
        "library", ()
    ),
    "omni_mercury_engine.infrastructure.humanitarian.climate_resilience:ClimateResilienceDetector": DetectorRole(
        "library", ()
    ),
    "omni_mercury_engine.infrastructure.humanitarian.economic_resilience:EconomicResilienceDetector": DetectorRole(
        "library", ()
    ),
    "omni_mercury_engine.infrastructure.humanitarian.education_equity:EducationEquityDetector": DetectorRole(
        "library", ()
    ),
    "omni_mercury_engine.infrastructure.humanitarian.neuroscience:NeuroscienceDetector": DetectorRole(
        "library", ()
    ),
    "omni_mercury_engine.medical.endocrinology_detector:EndocrinologyDetector": DetectorRole(
        "library", ()
    ),
    "omni_mercury_engine.medical.pandemic.bio_threats.pathogen_detector:PathogenDetector": DetectorRole(
        "library", ()
    ),
    "omni_mercury_engine.medical.pandemic.pandemic_detector:PandemicDetector": DetectorRole(
        "library", ()
    ),
    "omni_mercury_engine.ml.bias_detection:BiasDetector": DetectorRole("library", ()),
    "omni_mercury_engine.models.foundation.llm_adapter:TextLogAnomalyDetector": DetectorRole(
        "library", ()
    ),
    "omni_mercury_engine.quantum_computing.detector:QuantumAnomalyDetector": DetectorRole(
        "library", ()
    ),
    "omni_mercury_engine.security.anti_terrorism.pattern_recognition:TerrorismPatternDetector": DetectorRole(
        "library", ()
    ),
    "omni_mercury_engine.security.tempest_detection:TEMPESTDetector": DetectorRole("library", ()),
    "omni_mercury_engine.space.disaster_precursor_detector:DisasterPrecursorDetector": DetectorRole(
        "library", ()
    ),
    "omni_mercury_engine.space.interstellar_objects:InterstellarObjectDetector": DetectorRole(
        "library", ()
    ),
    "omni_mercury_engine.streaming.streaming_detector:StreamingDetector": DetectorRole(
        "library", ()
    ),
}
