# Step 4 — File-level match of every Omni .py file

Total Omni .py files inspected: **194**

## Classification totals
- SUPERSEDED: **111**
- PARTIAL_SUPERSEDED_REVIEW_NEEDED: **13**
- WEAK_MATCH_REVIEW_NEEDED: **12**
- NO_MERCURY_EQUIVALENT_REVIEW_NEEDED: **58**

## `<root>/`

| omni file | symbols | LOC | classification | best match (rule, overlap) |
|---|---|---|---|---|
| `src/omni_anomaly_engine/__init__.py` | ∅ | 28 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/__init__.py` (filename, 0/0) |
| `src/omni_anomaly_engine/cli.py` | biometric, detect, explain, main, security… | 126 | SUPERSEDED | `src/omni_mercury_engine/cli.py` (filename, 6/6) |
| `src/omni_anomaly_engine/cli_enhanced.py` | biometric, detect, main, run_chemistry, run_demo… | 661 | WEAK_MATCH_REVIEW_NEEDED | `src/omni_mercury_engine/cli.py` (symbol_overlap, 4/10) |
| `src/omni_anomaly_engine/engine.py` | OmniAnomalyEngine | 367 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/engine.py` (filename, 0/1) |
| `src/omni_anomaly_engine/mercury_a_agent.py` | AgentMemory, AgentState, DomainType, MercuryAgent, MercuryPlanner… | 977 | PARTIAL_SUPERSEDED_REVIEW_NEEDED | `src/omni_mercury_engine/agentic/mercury_a_agent.py` (prior_map, 6/8) |
| `src/omni_anomaly_engine/mercury_a_crews.py` | AgentRole, BaseCrew, CrewCoordinator, CrewTask, EmergentCrew… | 646 | WEAK_MATCH_REVIEW_NEEDED | `src/omni_mercury_engine/cognitive/multi_agent_coordination.py` (prior_map, 1/10) |
| `src/omni_anomaly_engine/mercury_a_learning.py` | AdaptiveLearner, AnomalyDetectionEnv, MercuryLearner, RewardConfig | 477 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/ml/ppo_trainer.py` (prior_map, 0/4) |
| `src/omni_anomaly_engine/truth_decipher.py` | TruthDecipherFramework, TruthDecipherResult | 348 | SUPERSEDED | `src/omni_mercury_engine/truth_decipher.py` (filename, 2/2) |

## `agentic/`

| omni file | symbols | LOC | classification | best match (rule, overlap) |
|---|---|---|---|---|
| `src/omni_anomaly_engine/agentic/__init__.py` | ∅ | 5 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/__init__.py` (filename, 0/0) |
| `src/omni_anomaly_engine/agentic/agentic_autonomy.py` | AgentAction, AgentState, AgenticAutonomy | 314 | SUPERSEDED | `src/omni_mercury_engine/agentic/agentic_autonomy.py` (filename, 3/3) |

## `agents/`

| omni file | symbols | LOC | classification | best match (rule, overlap) |
|---|---|---|---|---|
| `src/omni_anomaly_engine/agents/__init__.py` | ∅ | 27 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/__init__.py` (filename, 0/0) |
| `src/omni_anomaly_engine/agents/mercury_a.py` | MercuryA, MercuryArtifactGenerator, MercuryConfig, MercuryKnowledgeBase, MercuryMode… | 741 | WEAK_MATCH_REVIEW_NEEDED | `src/omni_mercury_engine/narrative/interface.py` (symbol_overlap, 1/8) |

## `api/`

| omni file | symbols | LOC | classification | best match (rule, overlap) |
|---|---|---|---|---|
| `src/omni_anomaly_engine/api/__init__.py` | ∅ | 5 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/__init__.py` (filename, 0/0) |
| `src/omni_anomaly_engine/api/server.py` | HealthResponse, MultivariateRequest, UnivariateRequest, detect_multivariate, detect_univariate… | 115 | SUPERSEDED | `src/omni_mercury_engine/api/server.py` (filename, 6/6) |

## `comparison/`

| omni file | symbols | LOC | classification | best match (rule, overlap) |
|---|---|---|---|---|
| `src/omni_anomaly_engine/comparison/__init__.py` | ∅ | 13 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/__init__.py` (filename, 0/0) |
| `src/omni_anomaly_engine/comparison/pyod_integration.py` | CombinationMethod, PyODAlgorithm, PyODComparison | 318 | SUPERSEDED | `src/omni_mercury_engine/comparison/pyod_integration.py` (filename, 3/3) |

## `core/`

| omni file | symbols | LOC | classification | best match (rule, overlap) |
|---|---|---|---|---|
| `src/omni_anomaly_engine/core/__init__.py` | ∅ | 0 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/__init__.py` (filename, 0/0) |
| `src/omni_anomaly_engine/core/ai_ethics.py` | EthicalAutonomyGovernor, EthicalPrinciple, EthicsConfig, EthicsResult, evaluate_refactoring_ethics | 401 | SUPERSEDED | `src/omni_mercury_engine/core/ai_ethics.py` (filename, 5/5) |
| `src/omni_anomaly_engine/core/base.py` | BaseDetector, BaseEncoder, BaseModel | 64 | PARTIAL_SUPERSEDED_REVIEW_NEEDED | `src/omni_mercury_engine/core/base.py` (filename, 2/3) |
| `src/omni_anomaly_engine/core/chaos_evolutionary.py` | ChaosEvolutionOptimizer, ChaoticMap | 281 | SUPERSEDED | `src/omni_mercury_engine/core/chaos_evolutionary.py` (filename, 2/2) |
| `src/omni_anomaly_engine/core/config.py` | DetectorConfig, DeviceType, EngineConfig, FusionConfig, FusionMode… | 101 | SUPERSEDED | `src/omni_mercury_engine/core/config.py` (filename, 6/6) |
| `src/omni_anomaly_engine/core/ethical_config.py` | EngineConfig, EthicalScalars | 371 | SUPERSEDED | `src/omni_mercury_engine/core/ethical_config.py` (filename, 2/2) |
| `src/omni_anomaly_engine/core/ethical_governor.py` | BiasMetrics, EthicalAutonomyGovernor, EthicalDecision, SigmaDirective | 427 | SUPERSEDED | `src/omni_mercury_engine/core/ethical_governor.py` (filename, 4/4) |
| `src/omni_anomaly_engine/core/ethical_risk_matrix.py` | AnomalyOracle, ComplianceRegime, ComplianceRule, EthicalRiskMatrix, GDPRCompliance… | 598 | SUPERSEDED | `src/omni_mercury_engine/core/ethical_risk_matrix.py` (filename, 9/9) |
| `src/omni_anomaly_engine/core/exceptions.py` | ConfigException, DataException, DetectorException, FusionException, ModelException… | 31 | SUPERSEDED | `src/omni_mercury_engine/core/exceptions.py` (filename, 7/7) |
| `src/omni_anomaly_engine/core/extended_anomaly_engine.py` | EngineConfig, EvolutionEngine, EvolutionStrategy, IntegrationEngine, OmniAXAEngine… | 306 | PARTIAL_SUPERSEDED_REVIEW_NEEDED | `src/omni_mercury_engine/core/extended_anomaly_engine.py` (filename, 5/6) |
| `src/omni_anomaly_engine/core/federated_learning.py` | FederatedAnomalyDetector | 113 | SUPERSEDED | `src/omni_mercury_engine/federated_learning/server.py` (symbol_overlap, 1/1) |
| `src/omni_anomaly_engine/core/fusion.py` | AttentionFusion, EarlyFusionEncoder, HybridFusionLayer, OmniAvaEngine | 1033 | WEAK_MATCH_REVIEW_NEEDED | `src/omni_mercury_engine/detectors/fusion/multimodal_fusion.py` (symbol_overlap, 1/4) |
| `src/omni_anomaly_engine/core/info_geometry.py` | InformationGeometryDetector | 131 | SUPERSEDED | `src/omni_mercury_engine/core/info_geometry.py` (filename, 1/1) |
| `src/omni_anomaly_engine/core/multivariate_timeseries.py` | ChaosMultivariateFusion, FractionalDifferentiator, MultivariateTSDetector | 391 | PARTIAL_SUPERSEDED_REVIEW_NEEDED | `src/omni_mercury_engine/core/multivariate_timeseries.py` (filename, 2/3) |
| `src/omni_anomaly_engine/core/neurosymbolic_engine.py` | NeurosymbolicConfig, NeurosymbolicEngine, ReadinessLevel, TrainingMetrics, TrainingPhase | 347 | SUPERSEDED | `src/omni_mercury_engine/core/code_analysis.py` (symbol_overlap, 5/5) |
| `src/omni_anomaly_engine/core/novel_class_discovery.py` | MultiElementBinarization, NovelClassDiscovery | 233 | SUPERSEDED | `src/omni_mercury_engine/core/novel_class_discovery.py` (filename, 2/2) |
| `src/omni_anomaly_engine/core/quantum_kernels.py` | QuantumKernelMachine | 177 | SUPERSEDED | `src/omni_mercury_engine/core/quantum_kernels.py` (filename, 1/1) |
| `src/omni_anomaly_engine/core/regenerative.py` | FeedbackLoop, PermaculturePrinciple, RegenerativeArchitecture | 301 | SUPERSEDED | `src/omni_mercury_engine/core/regenerative.py` (filename, 3/3) |
| `src/omni_anomaly_engine/core/self_healing.py` | AnomalySignature, CRISPRInspiredSelfHealing | 214 | PARTIAL_SUPERSEDED_REVIEW_NEEDED | `src/omni_mercury_engine/resilience/self_healing.py` (filename, 1/2) |
| `src/omni_anomaly_engine/core/symbolic_reasoning.py` | SymbolicReasoningEngine, SymbolicRule | 214 | SUPERSEDED | `src/omni_mercury_engine/core/symbolic_reasoning.py` (filename, 2/2) |
| `src/omni_anomaly_engine/core/three_r_mechanism.py` | AnomalyDetectionMethod, EvolutionStrategy, IssueSeverity, IssueType, RecursionEngine… | 1458 | PARTIAL_SUPERSEDED_REVIEW_NEEDED | `src/omni_mercury_engine/core/three_r/types.py` (symbol_overlap, 5/10) |

## `data_sources/`

| omni file | symbols | LOC | classification | best match (rule, overlap) |
|---|---|---|---|---|
| `src/omni_anomaly_engine/data_sources/__init__.py` | ∅ | 21 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/__init__.py` (filename, 0/0) |
| `src/omni_anomaly_engine/data_sources/realtime_apis.py` | DataSourceConfig, NASAFIRMSWildfireAPI, NOAASpaceWeatherAPI, RealtimeDataAggregator, USGSEarthquakeAPI | 504 | WEAK_MATCH_REVIEW_NEEDED | `src/omni_mercury_engine/data_sources/base.py` (symbol_overlap, 1/5) |

## `detectors/`

| omni file | symbols | LOC | classification | best match (rule, overlap) |
|---|---|---|---|---|
| `src/omni_anomaly_engine/detectors/__init__.py` | ∅ | 19 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/__init__.py` (filename, 0/0) |
| `src/omni_anomaly_engine/detectors/dimensional.py` | DimensionalAnalyzer, NeuralProjection | 276 | PARTIAL_SUPERSEDED_REVIEW_NEEDED | `src/omni_mercury_engine/detectors/dimensional.py` (filename, 1/2) |
| `src/omni_anomaly_engine/detectors/directive.py` | SigmaDirectiveDetector | 409 | SUPERSEDED | `src/omni_mercury_engine/detectors/directive.py` (filename, 1/1) |
| `src/omni_anomaly_engine/detectors/graph_based.py` | GraphAnomalyDetector | 152 | SUPERSEDED | `src/omni_mercury_engine/detectors/graph_based.py` (filename, 1/1) |
| `src/omni_anomaly_engine/detectors/marine/__init__.py` | ∅ | 5 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/__init__.py` (filename, 0/0) |
| `src/omni_anomaly_engine/detectors/spatial.py` | SpatialAnomalyDetector | 124 | SUPERSEDED | `src/omni_mercury_engine/detectors/spatial.py` (filename, 1/1) |
| `src/omni_anomaly_engine/detectors/statistical.py` | StatisticalAnomalyDetector | 142 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/detectors/statistical.py` (filename, 0/1) |
| `src/omni_anomaly_engine/detectors/temporal.py` | TemporalAnomalyDetector | 141 | SUPERSEDED | `src/omni_mercury_engine/detectors/temporal.py` (filename, 1/1) |

## `domains/`

| omni file | symbols | LOC | classification | best match (rule, overlap) |
|---|---|---|---|---|
| `src/omni_anomaly_engine/domains/__init__.py` | ∅ | 51 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/__init__.py` (filename, 0/0) |
| `src/omni_anomaly_engine/domains/ciad/__init__.py` | ∅ | 0 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/__init__.py` (filename, 0/0) |
| `src/omni_anomaly_engine/domains/ciad/compliance/__init__.py` | ∅ | 0 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/__init__.py` (filename, 0/0) |
| `src/omni_anomaly_engine/domains/ciad/compliance/iot_connector.py` | IoTAnomalyResult, IoTConnector, IoTDevice, IoTMode, MQTTQoS… | 530 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | — |
| `src/omni_anomaly_engine/domains/ciad/compliance/nist_csf_integrator.py` | ImplementationTier, NISTAssessment, NISTCSFIntegrator, NISTCategory, NISTFunction… | 578 | SUPERSEDED | `src/omni_mercury_engine/compliance/nist_csf_integrator.py` (filename, 7/7) |
| `src/omni_anomaly_engine/domains/ciad/compliance/osha_compliance_anomaly.py` | ComplianceLevel, HazardCategory, OSHAComplianceDetector, OSHAHazard, OSHASector… | 666 | SUPERSEDED | `src/omni_mercury_engine/compliance/osha_anomaly.py` (symbol_overlap, 7/7) |
| `src/omni_anomaly_engine/domains/ciad/compliance/tlp_handler.py` | TLPClassification, TLPColor, TLPHandler, get_tlp_handler | 313 | SUPERSEDED | `src/omni_mercury_engine/compliance/tlp_handler.py` (filename, 4/4) |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/__init__.py` | ∅ | 0 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/__init__.py` (filename, 0/0) |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/biometric.py` | BiometricAnomalyModel, FourierAnalyzer, HarmonicDecomposer | 271 | SUPERSEDED | `src/omni_mercury_engine/models/biometric.py` (filename, 3/3) |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/core.py` | OverwatchNexus, OverwatchNexusResult | 384 | SUPERSEDED | `src/omni_mercury_engine/security/counterintelligence.py` (symbol_overlap, 2/2) |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/crisis_monitor.py` | CrisisAlert, CrisisMonitor | 215 | SUPERSEDED | `src/omni_mercury_engine/infrastructure/humanitarian/crisis_monitoring/crisis_monitor.py` (filename, 2/2) |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/cyber/__init__.py` | ∅ | 6 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/__init__.py` (filename, 0/0) |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/cyber/cross_border_intel.py` | CrossBorderIntelligence | 115 | SUPERSEDED | `src/omni_mercury_engine/infrastructure/cyber/cross_border_intel.py` (filename, 1/1) |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/cyber/cyber_fortress.py` | CyberFortress, EncryptedTrafficAnomalyDetector, FortressResult, MultiverseZeroDaySimulator, ResonanceHashIntegrityChecker | 453 | PARTIAL_SUPERSEDED_REVIEW_NEEDED | `src/omni_mercury_engine/security/cyber_fortress.py` (filename, 4/5) |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/cyber/quantum_risk_cyber.py` | CryptoSystem, PostQuantumMigrationPlanner, QuantumRiskCyber, QuantumThreat, ThreatLevel | 876 | SUPERSEDED | `src/omni_mercury_engine/security/quantum_risk_cyber.py` (filename, 5/5) |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/cyber/space_infrastructure.py` | SpaceInfrastructureMonitor | 179 | SUPERSEDED | `src/omni_mercury_engine/infrastructure/cyber/space_infrastructure.py` (filename, 1/1) |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/cybint_subprocessor.py` | APTGroup, APTPatternRecognizer, C2InfrastructureDetector, CYBINTAnalysisResult, CYBINTSubProcessor… | 628 | SUPERSEDED | `src/omni_mercury_engine/security/cybint_subprocessor.py` (filename, 10/10) |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/encryption.py` | QuantumResistantEncryption, SecureDataHandler | 372 | SUPERSEDED | `src/omni_mercury_engine/security/encryption.py` (filename, 2/2) |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/hive_firewall.py` | HiveFirewall, HiveNode, ThreatBlocking | 446 | SUPERSEDED | `src/omni_mercury_engine/security/hive_firewall.py` (filename, 3/3) |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/int_sources.py` | COMINTAnalysisResult, COMINTProcessor, CYBINTAnalysisResult, CYBINTProcessor, CryptanalysisProcessor… | 1241 | SUPERSEDED | `src/omni_mercury_engine/security/int_sources.py` (filename, 28/28) |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/intelligence_fusion.py` | AllSourceFusionNetwork, IntelligenceDiscipline, IntelligenceFusionEngine, IntelligenceFusionResult, ThreatLevel… | 729 | SUPERSEDED | `src/omni_mercury_engine/security/intelligence_fusion.py` (filename, 6/6) |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/overwatch_nexus/__init__.py` | ∅ | 24 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/__init__.py` (filename, 0/0) |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/overwatch_nexus/anti_terrorism/__init__.py` | ∅ | 10 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/__init__.py` (filename, 0/0) |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/overwatch_nexus/bio_threats/__init__.py` | ∅ | 15 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/__init__.py` (filename, 0/0) |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/overwatch_nexus/humanitarian_ci/__init__.py` | ∅ | 13 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/__init__.py` (filename, 0/0) |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/overwatch_nexus/intel_types/__init__.py` | ∅ | 3 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/__init__.py` (filename, 0/0) |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/overwatch_nexus/pandemic_forecasting/__init__.py` | ∅ | 15 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/__init__.py` (filename, 0/0) |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/pattern_recognition.py` | TerrorismPatternDetector, TerrorismThreatResult | 128 | SUPERSEDED | `src/omni_mercury_engine/security/anti_terrorism/pattern_recognition.py` (filename, 2/2) |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/rate_limiting.py` | RateLimiter | 36 | SUPERSEDED | `src/omni_mercury_engine/security/rate_limiting.py` (filename, 1/1) |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/realtime_threat_detection.py` | AdaptiveThreatDetector, RealTimeThreatDetector, ThreatSignature | 280 | SUPERSEDED | `src/omni_mercury_engine/security/realtime_threat_detection.py` (filename, 3/3) |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/tempest_detection.py` | EMSECCountermeasureGenerator, EmanationType, RFSpectrumAnalyzer, SideChannelVulnerabilityAssessor, TEMPESTAnalysisResult… | 521 | SUPERSEDED | `src/omni_mercury_engine/security/tempest_detection.py` (filename, 8/8) |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/threat_detection.py` | BanishmentAction, ThreatDetector | 249 | SUPERSEDED | `src/omni_mercury_engine/security/threat_detection.py` (filename, 2/2) |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/traffic_analysis.py` | CommunicationGraphAnalyzer, CovertChannelDetector, EncryptedTrafficFingerprinter, NetworkFlowAnalyzer, TrafficAnalysisEngine… | 583 | SUPERSEDED | `src/omni_mercury_engine/security/traffic_analysis.py` (filename, 7/7) |
| `src/omni_anomaly_engine/domains/ciad/economic/__init__.py` | ∅ | 5 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/__init__.py` (filename, 0/0) |
| `src/omni_anomaly_engine/domains/ciad/economic/anomaly_detector_financial_crisis.py` | BankingStressDetector, CrisisSeverity, CrisisType, FinancialCrisisDetector, FinancialCrisisPredictionResult… | 478 | SUPERSEDED | `src/omni_mercury_engine/detectors/economic/financial_crisis_detector.py` (symbol_overlap, 8/8) |
| `src/omni_anomaly_engine/domains/ciad/economic/world_bank_sectors.py` | WorldBankSectorsMonitor | 256 | SUPERSEDED | `src/omni_mercury_engine/infrastructure/economic/world_bank_sectors.py` (filename, 1/1) |
| `src/omni_anomaly_engine/domains/ciad/government/__init__.py` | ∅ | 5 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/__init__.py` (filename, 0/0) |
| `src/omni_anomaly_engine/domains/ciad/government/communications_it.py` | CommunicationsITDetector | 192 | SUPERSEDED | `src/omni_mercury_engine/infrastructure/communications_it.py` (filename, 1/1) |
| `src/omni_anomaly_engine/domains/ciad/government/ncf_monitor.py` | NCFMonitor | 268 | SUPERSEDED | `src/omni_mercury_engine/infrastructure/resilience/ncf_monitor.py` (filename, 1/1) |
| `src/omni_anomaly_engine/domains/ehead/__init__.py` | ∅ | 0 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/__init__.py` (filename, 0/0) |
| `src/omni_anomaly_engine/domains/ehead/earth_sciences/__init__.py` | ∅ | 0 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/__init__.py` (filename, 0/0) |
| `src/omni_anomaly_engine/domains/ehead/earth_sciences/energy/__init__.py` | ∅ | 5 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/__init__.py` (filename, 0/0) |
| `src/omni_anomaly_engine/domains/ehead/earth_sciences/energy/anomaly_detector_emp.py` | E1PulseDetector, E3PulseDetector, EMPDetector, EMPPredictionResult, EMPType… | 417 | SUPERSEDED | `src/omni_mercury_engine/detectors/energy/emp_detector.py` (symbol_overlap, 7/7) |
| `src/omni_anomaly_engine/domains/ehead/earth_sciences/energy/energy_dams.py` | DamType, EnergyDamsDetector, EnergySubsector | 299 | SUPERSEDED | `src/omni_mercury_engine/infrastructure/energy_dams.py` (filename, 3/3) |
| `src/omni_anomaly_engine/domains/ehead/earth_sciences/energy/energy_optimization.py` | EnergyOptimization, EnergyProfile, EnergySource | 255 | SUPERSEDED | `src/omni_mercury_engine/energy/energy_optimization.py` (filename, 3/3) |
| `src/omni_anomaly_engine/domains/ehead/earth_sciences/geology/__init__.py` | ∅ | 5 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/__init__.py` (filename, 0/0) |
| `src/omni_anomaly_engine/domains/ehead/earth_sciences/geology/anomaly_detector_landslide.py` | LandslideDetector, LandslidePredictionResult, LandslideRiskLevel, LandslideType, RainfallTriggerModel… | 428 | SUPERSEDED | `src/omni_mercury_engine/detectors/geological/landslide.py` (symbol_overlap, 7/7) |
| `src/omni_anomaly_engine/domains/ehead/earth_sciences/geology/anomaly_detector_volcanic.py` | EruptionForecastModel, EruptionType, GasEmissionAnalyzer, InSARDeformationDetector, SeismicSwarmDetector… | 607 | SUPERSEDED | `src/omni_mercury_engine/detectors/geological/volcanic.py` (symbol_overlap, 9/9) |
| `src/omni_anomaly_engine/domains/ehead/earth_sciences/geology/anomaly_detector_wildfire.py` | FireIgnitionDetector, FireRiskLevel, FireSpreadModel, WildfireDetector, WildfirePredictionResult | 295 | SUPERSEDED | `src/omni_mercury_engine/detectors/geological/wildfire.py` (symbol_overlap, 5/5) |
| `src/omni_anomaly_engine/domains/ehead/earth_sciences/marine/__init__.py` | ∅ | 1 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/__init__.py` (filename, 0/0) |
| `src/omni_anomaly_engine/domains/ehead/earth_sciences/marine/anomaly_detector_biodiversity.py` | BiodiversityPredictionResult, CoralBleachingDetector, EcosystemHealth, MarineBiodiversityDetector | 166 | SUPERSEDED | `src/omni_mercury_engine/detectors/marine/biodiversity_detector.py` (symbol_overlap, 4/4) |
| `src/omni_anomaly_engine/domains/ehead/earth_sciences/marine/anomaly_detector_oceanography.py` | DepthLevel, OceanographyPatterns, WavePattern | 327 | SUPERSEDED | `src/omni_mercury_engine/ocean/oceanography_patterns.py` (symbol_overlap, 3/3) |
| `src/omni_anomaly_engine/domains/ehead/human_events/__init__.py` | ∅ | 6 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/__init__.py` (filename, 0/0) |
| `src/omni_anomaly_engine/domains/ehead/human_events/agrifood_security.py` | AgriFoodSecurityDetector, FoodSecurityThreat | 194 | SUPERSEDED | `src/omni_mercury_engine/infrastructure/humanitarian/agrifood_security.py` (filename, 2/2) |
| `src/omni_anomaly_engine/domains/ehead/human_events/climate_resilience.py` | ClimateEvent, ClimateResilienceDetector | 188 | SUPERSEDED | `src/omni_mercury_engine/infrastructure/humanitarian/climate_resilience.py` (filename, 2/2) |
| `src/omni_anomaly_engine/domains/ehead/human_events/economic_resilience.py` | EconomicResilienceDetector, EconomicThreat | 187 | SUPERSEDED | `src/omni_mercury_engine/infrastructure/humanitarian/economic_resilience.py` (filename, 2/2) |
| `src/omni_anomaly_engine/domains/ehead/human_events/education_equity.py` | EducationEquityDetector, EducationThreat | 182 | SUPERSEDED | `src/omni_mercury_engine/infrastructure/humanitarian/education_equity.py` (filename, 2/2) |
| `src/omni_anomaly_engine/domains/ehead/human_events/essential_workers.py` | EssentialWorkersMonitor | 292 | SUPERSEDED | `src/omni_mercury_engine/infrastructure/humanitarian/essential_workers.py` (filename, 1/1) |
| `src/omni_anomaly_engine/domains/ehead/human_events/government_facilities.py` | GovernmentFacilitiesMonitor | 334 | SUPERSEDED | `src/omni_mercury_engine/infrastructure/humanitarian/government_facilities.py` (filename, 1/1) |
| `src/omni_anomaly_engine/domains/ehead/human_events/neuroscience.py` | NeuralThreat, NeuroscienceDetector | 185 | SUPERSEDED | `src/omni_mercury_engine/infrastructure/humanitarian/neuroscience.py` (filename, 2/2) |
| `src/omni_anomaly_engine/domains/ehead/medical/__init__.py` | ∅ | 48 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/__init__.py` (filename, 0/0) |
| `src/omni_anomaly_engine/domains/ehead/medical/abms_disciplines.py` | ABMSBoard, ABMSDisciplineDetector, MedicalAnomalyResult, MultiSpecialtyNeuralNet, create_omni_medical_scalars | 759 | PARTIAL_SUPERSEDED_REVIEW_NEEDED | `src/omni_mercury_engine/medical/abms_disciplines.py` (filename, 4/5) |
| `src/omni_anomaly_engine/domains/ehead/medical/anesthesiology_predictor.py` | AnesthesiaPredictionResult, AnesthesiaRisk, AnesthesiaType, AnesthesiologyPredictor, HemodynamicMonitor… | 541 | SUPERSEDED | `src/omni_mercury_engine/medical/anesthesiology_predictor.py` (filename, 7/7) |
| `src/omni_anomaly_engine/domains/ehead/medical/cardiac_imaging.py` | CardiacImagingResult, CoronaryAngiogramAnalyzer, DICOMProcessor, ECGStreamProcessor, EchocardiogramAnalyzer… | 1031 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | — |
| `src/omni_anomaly_engine/domains/ehead/medical/cardiology_predictor.py` | AdvancedCardiacInterventions, ArrhythmiaType, CardiacBiomarkerAnalyzer, CardiologyPredictionResult, CardiologyPredictor… | 861 | PARTIAL_SUPERSEDED_REVIEW_NEEDED | `src/omni_mercury_engine/medical/cardiology/cardiology_predictor.py` (filename, 6/7) |
| `src/omni_anomaly_engine/domains/ehead/medical/endocrinology_detector.py` | CGMAnalyzer, EndocrinologyDetector, EndocrinologyPredictionResult, GLP1TherapyMonitor, GlycemicState… | 521 | SUPERSEDED | `src/omni_mercury_engine/medical/endocrinology_detector.py` (filename, 8/8) |
| `src/omni_anomaly_engine/domains/ehead/medical/healthcare_emergency.py` | EmergencyType, HealthcareEmergencyDetector, PatientStatus | 295 | SUPERSEDED | `src/omni_mercury_engine/infrastructure/healthcare_emergency.py` (filename, 3/3) |
| `src/omni_anomaly_engine/domains/ehead/medical/medical_cure_predictor.py` | CRISPRGeneEditingAnalyzer, MedicalCurePredictor, MedicalImagingAnomalyDetector, MedicalPredictionResult, PolygenicRiskScoreCalculator… | 734 | PARTIAL_SUPERSEDED_REVIEW_NEEDED | `src/omni_mercury_engine/medical/medical_cure_predictor.py` (filename, 6/9) |
| `src/omni_anomaly_engine/domains/ehead/medical/neurocritical_care.py` | ICPMonitor, NIHSSCalculator, NeurocriticalCarePredictor, NeurocriticalPredictionResult, SeizurePredictor… | 593 | SUPERSEDED | `src/omni_mercury_engine/medical/critical_care/neurocritical_care.py` (filename, 8/8) |
| `src/omni_anomaly_engine/domains/ehead/medical/neurology_detector.py` | BCIMonitoringSystem, BCIState, EEGAnalyzer, FocusedUltrasoundMonitor, NeurologicalCondition… | 562 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | — |
| `src/omni_anomaly_engine/domains/ehead/medical/pandemic/__init__.py` | ∅ | 0 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/__init__.py` (filename, 0/0) |
| `src/omni_anomaly_engine/domains/ehead/medical/pandemic/epidemic_model.py` | EpidemicForecaster, PandemicForecast | 361 | SUPERSEDED | `src/omni_mercury_engine/medical/pandemic/forecasting/epidemic_model.py` (filename, 2/2) |
| `src/omni_anomaly_engine/domains/ehead/medical/pandemic/pandemic_detector.py` | CaseSurgeDetector, MutationTracker, OutbreakSeverity, PandemicDetector, PandemicPredictionResult… | 404 | SUPERSEDED | `src/omni_mercury_engine/medical/pandemic/pandemic_detector.py` (filename, 7/7) |
| `src/omni_anomaly_engine/domains/ehead/medical/pandemic/pathogen_detector.py` | BioThreatResult, PathogenDetector | 319 | SUPERSEDED | `src/omni_mercury_engine/medical/pandemic/bio_threats/pathogen_detector.py` (filename, 2/2) |
| `src/omni_anomaly_engine/domains/ehead/medical/pathology_analyzer.py` | CoverslippingWorkstation, CryostatSimulator, EmbeddingStationSimulator, GrossingStationAnalyzer, LabStorageMonitor… | 1058 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | — |
| `src/omni_anomaly_engine/domains/ehead/medical/psychiatry_detector.py` | CrisisInterventionSystem, CrisisLevel, DBSMonitoringSystem, MoodDetectionLSTM, MoodState… | 609 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | — |
| `src/omni_anomaly_engine/domains/ehead/medical/sepsis_detector.py` | QuickSOFACalculator, SOFACalculator, SepsisDetector, SepsisPredictionResult, SepsisProgressionPredictor… | 520 | SUPERSEDED | `src/omni_mercury_engine/medical/critical_care/sepsis_detector.py` (filename, 6/6) |
| `src/omni_anomaly_engine/domains/stermad/__init__.py` | ∅ | 0 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/__init__.py` (filename, 0/0) |
| `src/omni_anomaly_engine/domains/stermad/chemistry/__init__.py` | ∅ | 1 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/__init__.py` (filename, 0/0) |
| `src/omni_anomaly_engine/domains/stermad/chemistry/anomaly_detector_periodic_table.py` | PeriodicProperty, PeriodicTableAnomalyDetector, PeriodicTableAnomalyResult, create_omni_periodic_table_scalars | 633 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | — |
| `src/omni_anomaly_engine/domains/stermad/chemistry/chemical_nuclear.py` | CISASector, ChemicalNuclearDetector | 217 | SUPERSEDED | `src/omni_mercury_engine/infrastructure/chemical_nuclear.py` (filename, 2/2) |
| `src/omni_anomaly_engine/domains/stermad/chemistry/chemistry_model.py` | ChemicalAnomalyResult, ChemistryAnomalyDetector, ElementGroup, PeriodicTableEncoder, create_omni_chemistry_scalars | 962 | SUPERSEDED | `src/omni_mercury_engine/models/chemistry.py` (symbol_overlap, 5/5) |
| `src/omni_anomaly_engine/domains/stermad/chemistry/isotope_predictor.py` | IsotopePredictionResult, IsotopePredictor, IsotopeRatioAnalyzer, IsotopeType, NuclearForensicsAnalyzer… | 486 | SUPERSEDED | `src/omni_mercury_engine/models/isotope_predictor.py` (filename, 7/7) |
| `src/omni_anomaly_engine/domains/stermad/engineering_robotics/__init__.py` | ∅ | 0 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/__init__.py` (filename, 0/0) |
| `src/omni_anomaly_engine/domains/stermad/engineering_robotics/anomaly_detector_robotics.py` | GWOAutoencoder, GraphAttentionLayer, GreyWolfOptimizer, RoboticsAnomalyDetector, RoboticsAnomalyResult… | 880 | WEAK_MATCH_REVIEW_NEEDED | `src/omni_mercury_engine/detectors/advanced/gwo_ensemble.py` (symbol_overlap, 1/8) |
| `src/omni_anomaly_engine/domains/stermad/engineering_robotics/drone_anomaly_detector.py` | DroneAnomalyDetector, DroneFault, DroneState, FaultType, MissionPhase… | 632 | SUPERSEDED | `src/omni_mercury_engine/detectors/drone/detector.py` (symbol_overlap, 6/6) |
| `src/omni_anomaly_engine/domains/stermad/mathematics/__init__.py` | ∅ | 0 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/__init__.py` (filename, 0/0) |
| `src/omni_anomaly_engine/domains/stermad/mathematics/math_unsolved.py` | ProblemCategory, UnsolvedMathProblems, UnsolvedProblemResult, create_omni_mathematics_scalars | 751 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | — |
| `src/omni_anomaly_engine/domains/stermad/scientific/__init__.py` | ∅ | 0 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/__init__.py` (filename, 0/0) |
| `src/omni_anomaly_engine/domains/stermad/scientific/emerging_tech_monitor.py` | EmergingTechMonitor | 315 | SUPERSEDED | `src/omni_mercury_engine/infrastructure/scientific/emerging_tech_monitor.py` (filename, 1/1) |
| `src/omni_anomaly_engine/domains/stermad/space/__init__.py` | ∅ | 17 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/__init__.py` (filename, 0/0) |
| `src/omni_anomaly_engine/domains/stermad/space/astrophysical.py` | AstrophysicalAnomalyModel | 123 | SUPERSEDED | `src/omni_mercury_engine/models/astrophysical.py` (filename, 1/1) |
| `src/omni_anomaly_engine/domains/stermad/space/disaster_precursor_detector.py` | DisasterPrecursorDetector, DisasterPrecursorResult, EarthquakePrecursorAnalyzer, GeomageticCorrelator, IonosphericDisturbanceDetector… | 534 | SUPERSEDED | `src/omni_mercury_engine/space/disaster_precursor_detector.py` (filename, 6/6) |
| `src/omni_anomaly_engine/domains/stermad/space/interstellar_objects.py` | ISOAnomalyType, InterstellarObjectAnalyzer, InterstellarObjectDetector, InterstellarObjectResult, NaturalExplanationConfidence… | 645 | SUPERSEDED | `src/omni_mercury_engine/space/interstellar_objects.py` (filename, 6/6) |
| `src/omni_anomaly_engine/domains/stermad/space/schumann_resonance.py` | SchumannAnomalyResult, SchumannHarmonicAnalyzer, SchumannResonanceDetector, create_omni_resonance_scalars | 594 | SUPERSEDED | `src/omni_mercury_engine/space/schumann_resonance.py` (filename, 4/4) |
| `src/omni_anomaly_engine/domains/stermad/space/solar_storm_detector.py` | CMETracker, GeomagneticStormPredictor, GeostormScale, SolarFlareClass, SolarFlareDetector… | 463 | SUPERSEDED | `src/omni_mercury_engine/space/solar_storm_detector.py` (filename, 7/7) |
| `src/omni_anomaly_engine/domains/stermad/space/space_exploration_analyzer.py` | SpaceExplorationAnalyzer | 491 | SUPERSEDED | `src/omni_mercury_engine/space/space_exploration_analyzer.py` (filename, 1/1) |
| `src/omni_anomaly_engine/domains/stermad/space/space_inspired.py` | RedundancyConfig, SpaceInspiredResilience, SystemState | 226 | SUPERSEDED | `src/omni_mercury_engine/space/space_inspired.py` (filename, 3/3) |

## `emergent/`

| omni file | symbols | LOC | classification | best match (rule, overlap) |
|---|---|---|---|---|
| `src/omni_anomaly_engine/emergent/__init__.py` | ∅ | 1 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/__init__.py` (filename, 0/0) |
| `src/omni_anomaly_engine/emergent/emergent_life_detector.py` | BioSignalPatternRecognizer, EmergentLifeDetector, LifeDetectionResult, MultiverseContactProtocolExplorer, SETICosmicSignalAnalyzer | 524 | SUPERSEDED | `src/omni_mercury_engine/emergent/emergent_life_detector.py` (filename, 5/5) |

## `federated/`

| omni file | symbols | LOC | classification | best match (rule, overlap) |
|---|---|---|---|---|
| `src/omni_anomaly_engine/federated/__init__.py` | ∅ | 19 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/__init__.py` (filename, 0/0) |
| `src/omni_anomaly_engine/federated/federated_detector.py` | CISAFederatedCoordinator, FederatedAnomalyDetector, FederatedStrategy, PrivacyLevel | 327 | WEAK_MATCH_REVIEW_NEEDED | `src/omni_mercury_engine/core/types.py` (symbol_overlap, 1/4) |
| `src/omni_anomaly_engine/federated/federated_robust.py` | ClientModel, FederatedAnomalyDetection, GlobalModel | 392 | SUPERSEDED | `src/omni_mercury_engine/federated_learning/federated_robust.py` (filename, 3/3) |

## `gui/`

| omni file | symbols | LOC | classification | best match (rule, overlap) |
|---|---|---|---|---|
| `src/omni_anomaly_engine/gui/__init__.py` | ∅ | 5 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/__init__.py` (filename, 0/0) |
| `src/omni_anomaly_engine/gui/live_monitoring_dashboard.py` | initialize_session_state, main, render_earthquake_monitoring, render_schumann_monitoring, render_solar_storm_monitoring… | 418 | WEAK_MATCH_REVIEW_NEEDED | `src/omni_mercury_engine/cli.py` (symbol_overlap, 1/7) |
| `src/omni_anomaly_engine/gui/streamlit_dashboard.py` | cardiology_interface, chemistry_analysis_page, cybint_interface, display_cardiology_results, display_cybint_results… | 536 | WEAK_MATCH_REVIEW_NEEDED | `src/omni_mercury_engine/cli.py` (symbol_overlap, 1/19) |

## `infrastructure/`

| omni file | symbols | LOC | classification | best match (rule, overlap) |
|---|---|---|---|---|
| `src/omni_anomaly_engine/infrastructure/__init__.py` | InfrastructureCoordinator | 292 | SUPERSEDED | `src/omni_mercury_engine/infrastructure/__init__.py` (filename, 1/1) |

## `ml/`

| omni file | symbols | LOC | classification | best match (rule, overlap) |
|---|---|---|---|---|
| `src/omni_anomaly_engine/ml/__init__.py` | ∅ | 39 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/__init__.py` (filename, 0/0) |
| `src/omni_anomaly_engine/ml/attention.py` | CrossModalAttention, MultiHeadDetectorAttention, SpatialAttention, TemporalAttention | 133 | SUPERSEDED | `src/omni_mercury_engine/ml/attention.py` (filename, 4/4) |
| `src/omni_anomaly_engine/ml/encoders.py` | AffectiveEncoder, AstrophysicalEncoder, BiometricEncoder, QuantumEncoder, StatisticalEncoder… | 245 | SUPERSEDED | `src/omni_mercury_engine/ml/encoders.py` (filename, 6/6) |
| `src/omni_anomaly_engine/ml/fusion_network.py` | OmniFusionModel, STEMDisciplineRouter | 471 | SUPERSEDED | `src/omni_mercury_engine/ml/fusion_network.py` (filename, 2/2) |
| `src/omni_anomaly_engine/ml/gwo_optimizer.py` | GreyWolfOptimizer | 164 | SUPERSEDED | `src/omni_mercury_engine/ml/gwo_optimizer.py` (filename, 1/1) |
| `src/omni_anomaly_engine/ml/harmonic_encoder.py` | FourierHarmonicAnalyzer, HarmonicEncoder, QuantumHarmonicOscillator, SphericalHarmonicDecomposer | 342 | PARTIAL_SUPERSEDED_REVIEW_NEEDED | `src/omni_mercury_engine/ml/harmonic_encoder.py` (filename, 3/4) |
| `src/omni_anomaly_engine/ml/hatcn_ad.py` | HATCN_AD, HierarchicalAttention, TemporalBlock | 170 | SUPERSEDED | `src/omni_mercury_engine/ml/hatcn_ad.py` (filename, 3/3) |
| `src/omni_anomaly_engine/ml/inference.py` | FusionInference | 193 | SUPERSEDED | `src/omni_mercury_engine/ml/inference.py` (filename, 1/1) |
| `src/omni_anomaly_engine/ml/layers.py` | EquilibriumPropagationLayer, LowPowerAnomalyDetector, compute_neuromorphic_efficiency | 373 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | — |
| `src/omni_anomaly_engine/ml/multimodal_fusion.py` | CrossModalAttention, MultimodalFusionNetwork | 108 | SUPERSEDED | `src/omni_mercury_engine/ml/multimodal_fusion.py` (filename, 2/2) |
| `src/omni_anomaly_engine/ml/optimizers.py` | AuxiliaryMaxVariance, DifferenceTargetPropagation, SyntheticGradientModule, SyntheticGradientPredictor, estimate_convergence_rate | 405 | SUPERSEDED | `src/omni_mercury_engine/ml/advanced_optimizers.py` (symbol_overlap, 5/5) |
| `src/omni_anomaly_engine/ml/regularizers.py` | DemographicParityLoss, HSICRegularizer, compute_fairness_metrics | 240 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | — |
| `src/omni_anomaly_engine/ml/training.py` | AnomalyDataset, AvaExponentialDecayOptimizer, AvaHarmonicOptimizer, AvaMomentumOptimizer, AvaOptimizer… | 391 | WEAK_MATCH_REVIEW_NEEDED | `src/omni_mercury_engine/ml/training.py` (filename, 2/7) |
| `src/omni_anomaly_engine/ml/vae_pattern_learner.py` | VAE, VAEPatternLearner | 177 | SUPERSEDED | `src/omni_mercury_engine/ml/vae_pattern_learner.py` (filename, 2/2) |

## `models/`

| omni file | symbols | LOC | classification | best match (rule, overlap) |
|---|---|---|---|---|
| `src/omni_anomaly_engine/models/__init__.py` | ∅ | 46 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/__init__.py` (filename, 0/0) |
| `src/omni_anomaly_engine/models/affective.py` | AffectiveAnomalyModel | 41 | SUPERSEDED | `src/omni_mercury_engine/models/affective.py` (filename, 1/1) |
| `src/omni_anomaly_engine/models/consciousness.py` | ConsciousnessPreservationModel | 140 | SUPERSEDED | `src/omni_mercury_engine/models/consciousness.py` (filename, 1/1) |
| `src/omni_anomaly_engine/models/multiverse.py` | MultiverseOmniEngine, Universe, UniverseState | 341 | SUPERSEDED | `src/omni_mercury_engine/models/multiverse.py` (filename, 3/3) |
| `src/omni_anomaly_engine/models/neural.py` | NeuralCognitiveModel | 161 | SUPERSEDED | `src/omni_mercury_engine/models/neural.py` (filename, 1/1) |
| `src/omni_anomaly_engine/models/neurosymbolic.py` | LogicTensorNetwork, NeurosymbolicEngine, SymbolicRule | 276 | SUPERSEDED | `src/omni_mercury_engine/models/neurosymbolic.py` (filename, 3/3) |
| `src/omni_anomaly_engine/models/parapsychology.py` | ConsciousnessFieldAnalyzer, ParapsychologyDetector, ParapsychologyResult, PsiPhenomenon, create_omni_psi_scalars | 608 | SUPERSEDED | `src/omni_mercury_engine/models/parapsychology.py` (filename, 5/5) |
| `src/omni_anomaly_engine/models/quantum.py` | QuantumAnomalyModel | 123 | SUPERSEDED | `src/omni_mercury_engine/models/quantum.py` (filename, 1/1) |
| `src/omni_anomaly_engine/models/simulation.py` | SimulationModule | 617 | SUPERSEDED | `src/omni_mercury_engine/models/simulation.py` (filename, 1/1) |

## `resilience/`

| omni file | symbols | LOC | classification | best match (rule, overlap) |
|---|---|---|---|---|
| `src/omni_anomaly_engine/resilience/__init__.py` | ∅ | 17 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/__init__.py` (filename, 0/0) |
| `src/omni_anomaly_engine/resilience/circuit_breaker.py` | CircuitBreaker, CircuitState | 79 | PARTIAL_SUPERSEDED_REVIEW_NEEDED | `src/omni_mercury_engine/core/types.py` (symbol_overlap, 1/2) |
| `src/omni_anomaly_engine/resilience/health_monitoring.py` | HealthMetrics, HealthMonitor | 72 | SUPERSEDED | `src/omni_mercury_engine/resilience/health_monitoring.py` (filename, 2/2) |
| `src/omni_anomaly_engine/resilience/retry.py` | RetryPolicy | 50 | SUPERSEDED | `src/omni_mercury_engine/resilience/retry.py` (filename, 1/1) |
| `src/omni_anomaly_engine/resilience/self_healing.py` | SelfHealingEngine | 91 | SUPERSEDED | `src/omni_mercury_engine/resilience/self_healing.py` (filename, 1/1) |

## `scaling/`

| omni file | symbols | LOC | classification | best match (rule, overlap) |
|---|---|---|---|---|
| `src/omni_anomaly_engine/scaling/__init__.py` | ∅ | 5 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/__init__.py` (filename, 0/0) |
| `src/omni_anomaly_engine/scaling/bain_ai_scaling.py` | BainAIScaling, ComputeResource | 271 | SUPERSEDED | `src/omni_mercury_engine/scaling/bain_ai_scaling.py` (filename, 2/2) |

## `utils/`

| omni file | symbols | LOC | classification | best match (rule, overlap) |
|---|---|---|---|---|
| `src/omni_anomaly_engine/utils/__init__.py` | compress_information, compute_complexity, compute_time_dilation, decompress_information, detect_singularity… | 308 | SUPERSEDED | `src/omni_mercury_engine/utils/__init__.py` (filename, 7/7) |
| `src/omni_anomaly_engine/utils/ancient_math.py` | is_vedic_enabled, robust_sqrt, robust_sqrt_vec, set_vedic_optimization, vedic_multiply… | 180 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | — |
| `src/omni_anomaly_engine/utils/comm.py` | AsyncMessageQueue, Message, MessagePriority, SimplePubSub | 221 | SUPERSEDED | `src/omni_mercury_engine/utils/comm.py` (filename, 4/4) |
| `src/omni_anomaly_engine/utils/logging_config.py` | EthicalAuditFilter, StructuredFormatter, descriptive_output, get_global_logger, get_logger… | 326 | WEAK_MATCH_REVIEW_NEEDED | `src/omni_mercury_engine/utils/logging.py` (symbol_overlap, 2/9) |
| `src/omni_anomaly_engine/utils/profiling.py` | PerformanceBenchmark, benchmark_function, get_profiling_logger, is_profiling_enabled, profile_complete… | 411 | SUPERSEDED | `src/omni_mercury_engine/utils/profiling.py` (filename, 10/10) |
| `src/omni_anomaly_engine/utils/report_generator.py` | EmailReportSender, PDFReportGenerator, PlainEnglishReportGenerator, ReportConfig, ReportManager | 441 | SUPERSEDED | `src/omni_mercury_engine/utils/report_generator.py` (filename, 5/5) |
| `src/omni_anomaly_engine/utils/rng.py` | DeterministicRNG, get_global_rng, reset_global_rng, set_global_seed | 260 | SUPERSEDED | `src/omni_mercury_engine/utils/rng.py` (filename, 4/4) |
| `src/omni_anomaly_engine/utils/validation.py` | ValidationResult, check_normality, detect_outliers, validate_data_reliability, validate_energy_data… | 535 | WEAK_MATCH_REVIEW_NEEDED | `src/omni_mercury_engine/security/input_validation.py` (symbol_overlap, 1/11) |

## `visualization/`

| omni file | symbols | LOC | classification | best match (rule, overlap) |
|---|---|---|---|---|
| `src/omni_anomaly_engine/visualization/__init__.py` | ∅ | 15 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | `src/omni_mercury_engine/__init__.py` (filename, 0/0) |
| `src/omni_anomaly_engine/visualization/live_visualizer.py` | LiveVisualizer, StreamingData, VisualizationConfig, VisualizationType | 902 | NO_MERCURY_EQUIVALENT_REVIEW_NEEDED | — |
