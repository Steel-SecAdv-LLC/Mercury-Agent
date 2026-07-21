<!--
Copyright (C) 2025 Steel Security Advisors LLC
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Mercury Agent — Capability Inventory

> Generated from source by `scripts/generate_capability_inventory.py` (`ast` walk of `src/omni_mercury_engine`, no runtime). Every row is a class that exists in the tree — this is the auditable answer to "what can Mercury do", not a hand-curated list. Re-run to refresh.

- **Total top-level classes:** 2,735
- **Capability-bearing classes:** 1,817 (excludes config/result/enum/error support types)
- **Subsystems (top-level packages):** 47
- **Refined via base-class analysis:** 79 classes categorized from their ancestor chain (e.g. `nn.Module` subclasses whose own name carries no suffix)
- **Unresolved (`Other`):** 1,031 — no name suffix and no informative ancestor (predominantly `object`-only classes, which base-class analysis cannot refine).

## Capability classes by category

| Category | Count |
|---|---|
| Other capability classes | 1031 |
| Support types (config / result / enum / error) | 918 |
| Detection | 171 |
| Neural models & layers | 157 |
| Engines & orchestration | 103 |
| Data sources & loaders | 102 |
| Training & optimization | 62 |
| Adapters & backends | 61 |
| Analysis & scoring | 56 |
| Monitoring | 24 |
| Prediction & forecasting | 21 |
| Ethics & governance | 12 |
| Solvers & scorers | 9 |
| Biometric & recognition | 8 |

## Classes by subsystem

### `(top-level)/` — 10 classes (7 capability)

**Adapters & backends**

- `MercuryMCPServer` (`mcp_server`) — Expose Mercury's capabilities to any MCP client over stdio.

**Engines & orchestration**

- `OmniMercuryEngine` (`engine`) — Unified neuro-symbolic detection engine with hybrid fusion.

**Monitoring**

- `MemoryMonitor` (`engine`) — Monitor and manage memory usage during detection.

**Other capability classes**

- `FeatureCache` (`engine`) — Thread-safe LRU cache for computed features.
- `ToolSpec` (`mcp_server`) — One MCP tool: its advertised schema and its handler.
- `TruthDecipherFramework` (`truth_decipher`) — Unified orchestrator for anomaly discovery, identification, ethical evaluation, and resolution.
- `_FallbackVoice` (`cli`) — Fallback voice for when narrative module unavailable.

<details><summary>Support types (3)</summary>

`MercuryProductionConfigError`, `ToolError`, `TruthDecipherResult`

</details>

### `agentic/` — 85 classes (56 capability)

**Engines & orchestration**

- `ComplianceAutomationAgent` (`agentic.subagents.specializations.compliance`) — Automated compliance monitoring and enforcement.
- `ComplianceSubAgent` (`agentic.subagents.specializations.compliance`) — Privacy/biometric compliance evaluation (BIPA/CCPA/CPRA).
- `CoordinatorSubAgent` (`agentic.subagents.coordinator`) — A subagent that *operates* its real engine subsystem(s), transparently.
- `DetectionSubAgent` (`agentic.subagents.specializations.detection`) — Delegated multi-agent anomaly detection over a real detector ensemble.
- `DetectorAgent` (`agentic.orchestration`) — A coordination agent backed by one of the engine's real detectors.
- `EthicalGuardrailSystem` (`agentic.subagents.specializations.guardrail`) — Rule-based guardrail rejecting prohibited or harmful actions.
- `EthicsEnforcementSubAgent` (`agentic.subagents.specializations.ethics`) — Delegated AI-ethics enforcement: IEEE EAD, EU AI Act, and fairness.
- `GeneralistSubAgent` (`agentic.subagents.specializations.generalist`) — A subagent with the main agent's full, unspecialized capability.
- `GuardrailSubAgent` (`agentic.subagents.specializations.guardrail`) — Screens actions and inputs for prohibited operations and manipulation.
- `MercuryAgent` (`agentic.mercury_a_agent`) — Mercury A.
- `MultiAgentOrchestrator` (`agentic.orchestration`) — Planner/critic/executor multi-agent orchestration for Mercury.
- `SubAgent` (`agentic.subagents.base`) — A named, Omni-Code-anchored subagent with the full main-agent toolkit.

**Ethics & governance**

- `AIEthicsEnforcer` (`agentic.subagents.specializations.ethics`) — AI ethics enforcement engine.
- `AutonomyGovernor` (`agentic.subagents.governor`) — Enforces the capability ceiling, ethical floor, corrigibility, tripwire.

**Monitoring**

- `SessionActionabilityTracker` (`agentic.capabilities.aggregate_gate`) — Session-scoped accretion tracker for the general-capability loop.

**Neural models & layers**

- `ManipulationResistanceLayer` (`agentic.subagents.specializations.guardrail`) — Local manipulation-pattern analysis over user input.

**Other capability classes**

- `AgentAction` (`agentic.agentic_autonomy`) — Represents an action taken by the agent.
- `AgentMemory` (`agentic.mercury_a_agent`) — Comprehensive memory system for Mercury Agent.
- `AgenticAutonomy` (`agentic.agentic_autonomy`) — Autonomous agent framework for anomaly detection.
- `BiasAssessment` (`agentic.subagents.specializations.ethics`) — Bias assessment results for one fairness metric.
- `CapabilityCeiling` (`agentic.subagents.governor`) — Hard limits the governor enforces on the fleet.
- `ComplianceRule` (`agentic.subagents.specializations.compliance`) — A single regulatory compliance rule.
- `ComplianceViolation` (`agentic.subagents.specializations.compliance`) — A detected breach of a :class:`ComplianceRule`.
- `ContextStats` (`agentic.bayesian_calibrator`) — Statistics for a single context (domain, goal_type).
- `ContractViolation` (`agentic.capabilities.contract`) — A capability contract is *misconfigured* (a decoration-time programmer error).
- `Document` (`agentic.capabilities.document_generator`) — A rendered document plus the structured content it was built from.
- `DocumentGenerator` (`agentic.capabilities.document_generator`) — Render structured content to Markdown / HTML / plain text.
- `EthicalViolation` (`agentic.subagents.specializations.ethics`) — Represents a single ethical violation.
- `Experience` (`agentic.agentic_autonomy`) — Experience tuple for reinforcement learning replay buffer.
- `ExtractiveSynthesizer` (`agentic.capabilities.text_synthesis`) — Rank-and-extract sentences/keywords from text (deterministic, no LLM).
- `GateVerdict` (`agentic.capabilities.assistant`) — Outcome of the unified harm gate for one general-capability action.
- `GeneralAssistant` (`agentic.capabilities.assistant`) — General-purpose research + document assistant for Mercury.
- `GovernorTripped` (`agentic.subagents.governor`) — Raised when the governor refuses a dispatch (fail-closed).
- `MemoryEntry` (`agentic.mercury_a_agent`) — Entry in agent memory system.
- `MercuryPlanner` (`agentic.mercury_a_agent`) — Goal decomposition and task orchestration with domain-specific planning.
- `MercuryReasoner` (`agentic.mercury_a_agent`) — Chain-of-thought reasoning engine with correlation graph building.
- `PlanTrace` (`agentic.orchestration`) — Record of the planner-driven episode execution.
- `PolicyMetrics` (`agentic.agentic_autonomy`) — Metrics tracking policy performance over time.
- `ReasoningStep` (`agentic.mercury_a_agent`) — A step in the reasoning chain.
- `RegulatoryIntelligence` (`agentic.subagents.specializations.compliance`) — Track regulatory changes and updates.
- `RosterEntry` (`agentic.subagents.roster`) — One pantheon member: identity → real subsystems → one Omni-Code anchor.
- `Section` (`agentic.capabilities.document_generator`) — One document section: a heading, body text, and optional bullet points.
- `SubAgentCapability` (`agentic.subagents.base`) — Routing descriptor for a subagent.
- `SubAgentFleet` (`agentic.subagents.fleet`) — Engine-mediated fleet of internal subagents under autonomy governance.
- `SubAgentRegistry` (`agentic.subagents.registry`) — The pantheon catalogue with deterministic capability routing.
- `SubAgentTask` (`agentic.subagents.base`) — A unit of work the fleet hands to a subagent.
- `Task` (`agentic.mercury_a_agent`) — Represents a task for the agent to execute.
- `WebResearcher` (`agentic.capabilities.web_research`) — Fetch, extract, and search the open web with the standard library only.
- `_DdgResultParser` (`agentic.capabilities.web_research`) — Collect ``(class, href, anchor-text)`` triples from <a> tags.
- `_DirectiveServingCache` (`agentic._incremental_serving`) — Incremental form of ``SigmaDirectiveDetector.detect`` for the last row.
- `_InternalAccess` (`agentic.subagents.base`) — Opaque construction token for the engine-mediated subagent boundary.
- `_ServingCache` (`agentic._incremental_serving`) — Base class: serve one row against the fixed reference, or refuse.
- `_SpatialServingCache` (`agentic._incremental_serving`) — Incremental form of ``SpatialAnomalyDetector.detect`` for the last row.
- `_TemporalServingCache` (`agentic._incremental_serving`) — Incremental form of ``TemporalAnomalyDetector.detect`` for the last row.
- `_TextExtractor` (`agentic.capabilities.web_research`) — Collect visible text from HTML, skipping script/style/etc.

**Training & optimization**

- `BayesianConfidenceCalibrator` (`agentic.bayesian_calibrator`) — Bayesian confidence calibrator using Beta-Bernoulli model.

<details><summary>Support types (29)</summary>

`AgentMode`, `AgentState`, `AggregateResult`, `CalibrationConfig`, `ComplianceFramework`, `ComplianceStatus`, `CoordinationBatch`, `DataCategory`, `DomainType`, `EpisodeResult`, `EthicalPrinciple`, `FairnessMetric`, `FetchResult`, `FleetResult`, `GovernorState`, `Invariant`, `LearningConfig`, `OrchestrationError`, `PlanResult`, `ReflectionRecord`, `ResearchReport`, `RiskLevel`, `SearchResult`, `SubAgentAccessError`, `SubAgentExecutionError`, `SubAgentResult`, `SupportsRefusal`, `TaskPriority`, `_EthicsRequest`

</details>

### `alerting/` — 8 classes (1 capability)

**Other capability classes**

- `CAPAlertGenerator` (`alerting.cap_generator`) — Generate CAP 1.2 XML alerts from Mercury anomaly detections.

<details><summary>Support types (7)</summary>

`CAPCategory`, `CAPCertainty`, `CAPMsgType`, `CAPScope`, `CAPSeverity`, `CAPStatus`, `CAPUrgency`

</details>

### `api/` — 81 classes (25 capability)

**Engines & orchestration**

- `AuthKeyManager` (`api.auth`) — AMA Key Management integration for Mercury's auth layer.

**Neural models & layers**

- `Model` (`api.routes.models`) — Model registry entry.
- `VoiceResponseModel` (`api.voice`) — Response model for voice speak endpoint.

**Other capability classes**

- `APIKey` (`api.auth`) — API key information.
- `APIKeyAuth` (`api.auth`) — API Key authentication dependency.
- `APIKeyStore` (`api.auth`) — In-memory API key store.
- `AnomalyPoint` (`api.server`) — Individual anomaly point information.
- `AuthProvider` (`api.auth`) — Abstract base class for authentication providers.
- `BatchJob` (`api.routes.batch`) — Batch job tracking record.
- `BatchJobStore` (`api.routes.batch`) — In-memory batch job store with TTL-based cleanup.
- `ComponentHealth` (`api.health`) — Health status of a single component.
- `CorrelationIDMiddleware` (`api.server`) — Middleware for request correlation ID tracking.
- `DataStore` (`api.routes.export`) — In-memory data store for demonstration.
- `ExportJob` (`api.routes.export`) — Export job record.
- `HealthCheck` (`api.health`) — Health check definition.
- `HealthChecker` (`api.health`) — Health check manager.
- `JWTAuth` (`api.auth`) — JWT Bearer token authentication dependency.
- `ModelMetrics` (`api.routes.models`) — Model performance metrics.
- `ModelRegistry` (`api.routes.models`) — Model registry with versioning and lifecycle management.
- `ModelVersion` (`api.routes.models`) — Model version record.
- `PIIMaskingFilter` (`api.server`) — Filter to mask PII data in log messages for security compliance.
- `RateLimitMiddleware` (`api.server`) — Token bucket rate limiting middleware.
- `RequestRateLimiter` (`api.auth`) — Request-aware rate limiter wrapper.
- `User` (`api.auth`) — Authenticated user information.
- `_FallbackVoice` (`api.voice`) — Fallback voice implementation when narrative module unavailable.

<details><summary>Support types (56)</summary>

`AuditLogRecord`, `AuthConfig`, `AuthMethod`, `AuthenticationError`, `AuthorizationError`, `BatchDetectRequest`, `BatchDetectionMethod`, `BatchJobResponse`, `BatchJobSubmitResponse`, `BatchResultsResponse`, `ComponentStatus`, `DeploymentRequest`, `DetailedHealthResponse`, `DetectionMethod`, `DetectionNarrationRequest`, `DetectionRecord`, `ErrorResponse`, `ExportFormat`, `ExportJobResponse`, `ExportRequest`, `ExportStatus`, `ExportSummaryResponse`, `ExportType`, `FlagshipDetectRequest`, `FusionRequest`, `FusionResponse`, `GreetingResponse`, `HazardVisualizeRequest`, `HealthResponse`, `HealthStatus`, `JobStatus`, `LivenessResponse`, `MetricsUpdateRequest`, `ModelCreateRequest`, `ModelFramework`, `ModelResponse`, `ModelStatus`, `ModelType`, `ModelVersionRequest`, `ModelVersionResponse`, `MultivariateRequest`, `MultivariateResponse`, `NarrationResponse`, `NeurosymbolicRequest`, `NeurosymbolicResponse`, `Permission`, `ReadinessResponse`, `RootCauseRequest`, `SeverityLevel`, `SpeakRequest`, `StatusResponse`, `ThreeRRequest`, `ThreeRResponse`, `TierDetectRequest`, `UnivariateRequest`, `UnivariateResponse`

</details>

### `automl/` — 26 classes (21 capability)

**Analysis & scoring**

- `SimpleClassifier` (`automl.optimizer`) — Simple classifier for default AutoML usage.

**Neural models & layers**

- `SimpleAnomalyModel` (`automl.optimizer`) — Simple anomaly detection model for default AutoML usage.

**Other capability classes**

- `CategoricalParameter` (`automl.search_space`) — Categorical parameter.
- `ConditionalParameter` (`automl.search_space`) — Parameter conditional on another parameter's value.
- `GaussianProcessSampler` (`automl.optimizer`) — Gaussian Process-based Bayesian optimization sampler.
- `HyperParameter` (`automl.search_space`) — Base class for hyperparameters.
- `HyperbandBracket` (`automl.schedulers`) — A single bracket in Hyperband.
- `IntUniformParameter` (`automl.search_space`) — Integer uniform distribution parameter.
- `LogUniformParameter` (`automl.search_space`) — Log-uniform distribution parameter.
- `MercuryAutoML` (`automl.optimizer`) — High-level AutoML interface for Mercury Agent.
- `RandomSampler` (`automl.optimizer`) — Random sampling from the search space.
- `Sampler` (`automl.optimizer`) — Base class for hyperparameter samplers.
- `SearchSpace` (`automl.search_space`) — Complete hyperparameter search space.
- `SimpleRegressor` (`automl.optimizer`) — Simple regressor for default AutoML usage.
- `TPESampler` (`automl.optimizer`) — Tree-structured Parzen Estimator (TPE) sampler.
- `UniformParameter` (`automl.search_space`) — Uniform distribution parameter.

**Training & optimization**

- `ASHAScheduler` (`automl.schedulers`) — Asynchronous Successive Halving Algorithm (ASHA).
- `BayesianOptimizer` (`automl.optimizer`) — Bayesian Optimization for hyperparameter tuning.
- `HyperbandScheduler` (`automl.schedulers`) — Hyperband scheduler for efficient hyperparameter optimization.
- `MedianStoppingScheduler` (`automl.schedulers`) — Median stopping rule scheduler.
- `TrialScheduler` (`automl.schedulers`) — Base class for trial schedulers.

<details><summary>Support types (5)</summary>

`OptimizationResult`, `SchedulerDecision`, `TrialInfo`, `TrialResult`, `TrialStatus`

</details>

### `biometric/` — 42 classes (30 capability)

**Biometric & recognition**

- `FingerprintMatcher` (`biometric.fingerprint_recognition`) — Match fingerprints using minutiae comparison.
- `FingerprintRecognizer` (`biometric.fingerprint_recognition`) — Complete fingerprint recognition system.
- `IrisMatcher` (`biometric.iris_recognition`) — Match iris codes using Hamming distance.
- `IrisRecognizer` (`biometric.iris_recognition`) — Complete iris recognition system.
- `VoiceMatcher` (`biometric.voice_recognition`) — Match voice samples using embedding similarity.
- `VoiceRecognizer` (`biometric.voice_recognition`) — Complete voice recognition system.

**Detection**

- `BiometricAnomalyDetector` (`biometric.__init__`) — Multi-modal biometric anomaly detection.
- `FingerprintLivenessDetector` (`biometric.fingerprint_recognition`) — Detect fingerprint presentation attacks.
- `IrisLivenessDetector` (`biometric.iris_recognition`) — Detect presentation attacks on iris recognition systems.
- `VoiceActivityDetector` (`biometric.voice_recognition`) — Detect voice activity in audio signal.
- `VoiceLivenessDetector` (`biometric.voice_recognition`) — Detect voice presentation attacks.

**Neural models & layers**

- `IrisEncoder` (`biometric.iris_recognition`) — Encode normalized iris to binary IrisCode.

**Other capability classes**

- `AudioPreprocessor` (`biometric.voice_recognition`) — Preprocess audio for voice recognition.
- `BiometricEnrollment` (`biometric.__init__`) — Enrolled biometric data for an identity.
- `EnergyExtractor` (`biometric.voice_recognition`) — Extract energy contour from audio frames.
- `FingerprintFeatures` (`biometric.fingerprint_recognition`) — Extracted fingerprint features.
- `GaborEnhancer` (`biometric.fingerprint_recognition`) — Enhance fingerprint using Gabor filters.
- `GaborFilter` (`biometric.iris_recognition`) — 2D Gabor filter bank for iris texture analysis.
- `IrisFeatures` (`biometric.iris_recognition`) — Extracted iris features.
- `IrisNormalizer` (`biometric.iris_recognition`) — Rubber sheet normalization (Daugman's method).
- `IrisSegmenter` (`biometric.iris_recognition`) — Iris segmentation using integro-differential operator.
- `MFCCExtractor` (`biometric.voice_recognition`) — Extract Mel-Frequency Cepstral Coefficients.
- `Minutia` (`biometric.fingerprint_recognition`) — A single fingerprint minutia point.
- `MinutiaeExtractor` (`biometric.fingerprint_recognition`) — Extract minutiae from enhanced fingerprint image.
- `PitchExtractor` (`biometric.voice_recognition`) — Extract pitch (fundamental frequency) contour.
- `Singularity` (`biometric.fingerprint_recognition`) — A fingerprint singularity point (core or delta).
- `SpeakerEmbedding` (`biometric.voice_recognition`) — Generate speaker embeddings from acoustic features.
- `VoiceFeatures` (`biometric.voice_recognition`) — Extracted voice features.

**Prediction & forecasting**

- `OrientationFieldEstimator` (`biometric.fingerprint_recognition`) — Estimate local ridge orientation field.
- `RidgeFrequencyEstimator` (`biometric.fingerprint_recognition`) — Estimate local ridge frequency.

<details><summary>Support types (12)</summary>

`BiometricAnomalyResult`, `BiometricModality`, `BiometricVerificationResult`, `FingerprintLivenessResult`, `FingerprintMatchResult`, `FusionStrategy`, `IrisMatchResult`, `LivenessResult`, `MinutiaeType`, `SingularityType`, `VoiceLivenessResult`, `VoiceMatchResult`

</details>

### `cognitive/` — 276 classes (181 capability)

**Analysis & scoring**

- `EquityCalculator` (`cognitive.ethical_bounding`) — Calculates equity metrics using Gini-like coefficients.
- `ReachabilityAnalyzer` (`cognitive.formal_verification`) — Analyzer for reachability properties.

**Data sources & loaders**

- `ExternalDataSource` (`cognitive.anomaly_detection_enhanced`) — Abstract base class for external data sources.
- `NOAAWeatherSource` (`cognitive.anomaly_detection_enhanced`) — Real NOAA Weather API client for production use.
- `SimulatedEnvironmentalSource` (`cognitive.anomaly_detection_enhanced`) — Simulated environmental data source (NOAA-style) for development/testing.
- `SimulatedGeologicalSource` (`cognitive.anomaly_detection_enhanced`) — Simulated geological data source (USGS-style) for development/testing.
- `USGSEarthquakeSource` (`cognitive.anomaly_detection_enhanced`) — Real USGS Earthquake API client for production use.

**Detection**

- `EnhancedAnomalyDetector` (`cognitive.anomaly_detection_enhanced`) — Enhanced Anomaly Detector with memory graph and external integration.
- `PatternDetector` (`cognitive.neural_memory_layer`) — Detect patterns from clustered memory embeddings.
- `PredictiveCodingDetector` (`cognitive.predictive_coding`) — Anomaly detector based on predictive coding principles.

**Engines & orchestration**

- `ActiveInferenceAgent` (`cognitive.predictive_coding`) — Active inference agent for anomaly detection.
- `AgentCoordinator` (`cognitive.multi_agent_coordination`) — Coordinator for multi-agent system.
- `CausalDiscoveryEngine` (`cognitive.causal_discovery`) — Production Causal Discovery and Inference Engine.
- `ChainOfHindsightEngine` (`cognitive.chain_of_hindsight`) — Main Chain of Hindsight learning engine.
- `ChainOfThoughtEngine` (`cognitive.chain_of_thought`) — Chain-of-Thought Reasoning Engine.
- `CognitiveOrchestrator` (`cognitive.orchestrator`) — Unified Cognitive Layer for Mercury-Agent.
- `CuriosityEngine` (`cognitive.cognitive_evolution_engine`) — Score observations by how far they are from the seen distribution.
- `DetectionAgent` (`cognitive.multi_agent_coordination`) — Abstract base class for detection agents.
- `DifferentiableLogicEngine` (`cognitive.differentiable_logic`) — Main engine for differentiable logic programming.
- `ExplainabilityEngine` (`cognitive.explainability`) — Unified explainability engine combining multiple explanation methods.
- `FormalVerificationEngine` (`cognitive.formal_verification`) — Main formal verification engine for Mercury Agent.
- `IPBEngine` (`cognitive.ipb_engine`) — Intelligence Preparation of the Battlefield Engine.
- `IndicatorDevelopmentSystem` (`cognitive.indicator_system`) — Indicator Development and Warning System.
- `MultiAgentDetectionSystem` (`cognitive.multi_agent_coordination`) — Complete multi-agent anomaly detection system.
- `NeurosymbolicFusionEngine` (`cognitive.neurosymbolic_fusion`) — Neuro-Symbolic Fusion Engine - Main interface for hybrid anomaly detection.
- `OODAAgent` (`cognitive.autonomous_agent`) — Autonomous agent implementing the OODA loop.
- `PlasticityEngine` (`cognitive.plasticity_engine`) — Production Neural Plasticity Engine.
- `ReasoningChain` (`cognitive.multi_hop_reasoner`) — A complete reasoning chain from premises to conclusion.
- `ReflexionEngine` (`cognitive.reflexion`) — Main Reflexion Engine.
- `SimpleDetectionAgent` (`cognitive.multi_agent_coordination`) — Simple detection agent implementation.
- `ThoughtChain` (`cognitive.chain_of_thought`) — A complete chain of thoughts.

**Ethics & governance**

- `SafetyBound` (`cognitive.formal_verification`) — Safety bounds for a variable.

**Neural models & layers**

- `GenerativeModel` (`cognitive.predictive_coding`) — A generative model at one level of hierarchy.
- `NeuralMemoryLayer` (`cognitive.neural_memory_layer`) — Neural Memory Layer - Main interface for memory-based pattern detection.
- `SymbolicLogicLayer` (`cognitive.symbolic_logic_layer`) — Symbolic Logic Layer - Main interface for symbolic reasoning.

**Other capability classes**

- `AdaptiveConformalInference` (`cognitive.uncertainty`) — Adaptive Conformal Inference for online uncertainty quantification.
- `AgentCapability` (`cognitive.multi_agent_coordination`) — Description of agent capabilities.
- `AlignmentAudit` (`cognitive.ethical_bounding`) — Audit record for alignment verification.
- `AnomalyChainOfHindsight` (`cognitive.chain_of_hindsight`) — Chain of Hindsight specialized for anomaly detection.
- `AnomalyChainOfThought` (`cognitive.chain_of_thought`) — Specialized Chain-of-Thought for anomaly detection.
- `AnomalyHierarchicalPlanner` (`cognitive.hierarchical_planning`) — Hierarchical planner specialized for anomaly detection.
- `AnomalyPrediction` (`cognitive.neural_memory_layer`) — Prediction of potential anomaly from patterns.
- `AnomalyReflexion` (`cognitive.reflexion`) — Reflexion framework specialized for anomaly detection.
- `AttentionMechanism` (`cognitive.neurosymbolic_fusion`) — Attention mechanism for neural-symbolic fusion.
- `BCMParameters` (`cognitive.plasticity_engine`) — Parameters for BCM (Bienenstock-Cooper-Munro) theory.
- `BaseExplainer` (`cognitive.explainability`) — Abstract base class for explainers.
- `BattlefieldAssessment` (`cognitive.ipb_engine`) — Complete IPB assessment result.
- `BenefitMaximizer` (`cognitive.ethical_bounding`) — Evaluates and maximizes potential benefits from actions.
- `BenevolenceCalibration` (`cognitive.ethical_bounding`) — Calibration knobs for the benevolence scorer.
- `Case` (`cognitive.case_based_reasoning`) — A case in the case base.
- `CaseBasedReasoner` (`cognitive.case_based_reasoning`) — Case-Based Reasoning Engine.
- `CausalEdge` (`cognitive.causal_discovery`) — An edge in the causal graph.
- `CausalEffect` (`cognitive.causal_discovery`) — Result of causal effect estimation.
- `CausalGraph` (`cognitive.causal_discovery`) — A causal graph structure (CPDAG or DAG).
- `Coalition` (`cognitive.multi_agent_coordination`) — A coalition of cooperating agents.
- `CompetitiveLearning` (`cognitive.plasticity_engine`) — Competitive learning with lateral inhibition.
- `ConsensusProtocol` (`cognitive.multi_agent_coordination`) — Protocol for reaching consensus among agents.
- `Constraint` (`cognitive.formal_verification`) — A constraint for verification.
- `CounterfactualExplainer` (`cognitive.explainability`) — Generate counterfactual explanations.
- `CreditAssignment` (`cognitive.chain_of_hindsight`) — Temporal credit assignment for sequential decisions.
- `Decision` (`cognitive.autonomous_agent`) — Decision made by the agent.
- `Decision` (`cognitive.reflexion`) — A decision made by the agent.
- `DetectedPattern` (`cognitive.neural_memory_layer`) — A detected pattern from memory analysis.
- `DifferentiableRule` (`cognitive.differentiable_logic`) — Differentiable logic rule with learnable parameters.
- `DifferentiableTNorm` (`cognitive.differentiable_logic`) — Abstract base for differentiable t-norms (fuzzy logic operators).
- `EligibilityTrace` (`cognitive.plasticity_engine`) — Eligibility trace for temporal credit assignment.
- `EmpathyAssessment` (`cognitive.ethical_bounding`) — Assessment of human-centric impact.
- `EmpathyModule` (`cognitive.ethical_bounding`) — Empathy module for human-centric decision making.
- `EnvironmentDefinition` (`cognitive.ipb_engine`) — Definition of the operational environment (Phase 1).
- `EnvironmentEffect` (`cognitive.ipb_engine`) — Environmental effect on operations (Phase 2).
- `EscalationBroker` (`cognitive.escalation`) — Route ESCALATE verdicts to a human reviewer under a bounded-autonomy cap.
- `EscalationDecision` (`cognitive.escalation`) — Outcome of an escalation review.
- `EthicalScore` (`cognitive.ethical_bounding`) — Comprehensive ethical evaluation score.
- `Experience` (`cognitive.reflexion`) — An experience combining decision and outcome.
- `ExperienceMemory` (`cognitive.reflexion`) — Memory store for past experiences.
- `ExplainableDecision` (`cognitive.symbolic_logic_layer`) — An explainable decision with full audit trail.
- `Explanation` (`cognitive.explainability`) — Explanation for a prediction.
- `ExternalDataIntegrator` (`cognitive.anomaly_detection_enhanced`) — Integrates external data sources for real-time anomaly detection.
- `ExternalDataPoint` (`cognitive.anomaly_detection_enhanced`) — Data point from external source.
- `FaithfulnessEvaluator` (`cognitive.explainability`) — Evaluate explanation faithfulness using various metrics.
- `FallbackGraph` (`cognitive.symbolic_logic_layer`) — Fallback graph implementation when NetworkX is not available.
- `FeatureImportance` (`cognitive.explainability`) — Feature importance score.
- `FeedbackProcessor` (`cognitive.chain_of_hindsight`) — Processes feedback to generate learning signals.
- `FormalProperty` (`cognitive.formal_verification`) — A formal property to verify.
- `FreeEnergy` (`cognitive.predictive_coding`) — Free energy computation result.
- `GNNMessagePassing` (`cognitive.knowledge_graph`) — Graph Neural Network message passing for representation learning.
- `GatedFusion` (`cognitive.neurosymbolic_fusion`) — Gated fusion mechanism for combining neural and symbolic outputs.
- `Goal` (`cognitive.hierarchical_planning`) — A goal in the hierarchical plan.
- `GoalDecomposer` (`cognitive.hierarchical_planning`) — Decomposes high-level goals into subgoals.
- `GodelTNorm` (`cognitive.differentiable_logic`) — Godel t-norm (minimum/maximum).
- `GrangerCausalityTest` (`cognitive.causal_discovery`) — Granger causality test using VAR models.
- `HarmReducer` (`cognitive.ethical_bounding`) — Evaluates and minimizes potential harm from actions.
- `HeuristicEvaluation` (`cognitive.reflexion`) — Evaluation using heuristic function.
- `HeuristicEvaluator` (`cognitive.reflexion`) — Evaluates decisions using heuristic functions.
- `HierarchicalPlan` (`cognitive.hierarchical_planning`) — A complete hierarchical plan.
- `HierarchicalPlanner` (`cognitive.hierarchical_planning`) — Main hierarchical planning engine.
- `HierarchicalPredictiveCoder` (`cognitive.predictive_coding`) — Hierarchical predictive coding network.
- `HierarchicalValueFunction` (`cognitive.hierarchical_planning`) — Value function decomposition for hierarchical planning.
- `HindsightRelabeler` (`cognitive.chain_of_hindsight`) — Relabels historical sequences with hindsight knowledge.
- `HindsightRelabeling` (`cognitive.chain_of_hindsight`) — Result of hindsight relabeling.
- `HistoricalSequence` (`cognitive.chain_of_hindsight`) — A complete historical sequence with feedback.
- `HybridAnomalyScore` (`cognitive.neurosymbolic_fusion`) — Hybrid anomaly score combining neural and symbolic components.
- `Indicator` (`cognitive.indicator_system`) — An intelligence indicator.
- `InferenceRule` (`cognitive.multi_hop_reasoner`) — A rule for inference.
- `IntegratedGradientsExplainer` (`cognitive.explainability`) — Integrated Gradients explainer for differentiable models.
- `IntelligenceRequirement` (`cognitive.indicator_system`) — A priority intelligence requirement (PIR).
- `IntervalBoundPropagator` (`cognitive.formal_verification`) — Interval bound propagation for neural network verification.
- `InvariantCondition` (`cognitive.formal_verification`) — An invariant condition that must always hold.
- `KMeansClusterer` (`cognitive.neural_memory_layer`) — K-means clustering for memory embeddings.
- `KnowledgeEdge` (`cognitive.knowledge_graph`) — An edge in the knowledge graph.
- `KnowledgeGraph` (`cognitive.knowledge_graph`) — Production Knowledge Graph for neuro-symbolic reasoning.
- `KnowledgeNode` (`cognitive.knowledge_graph`) — A node in the knowledge graph.
- `LIMEExplainer` (`cognitive.explainability`) — LIME-based explainer for anomaly detection models.
- `LearningSignal` (`cognitive.chain_of_hindsight`) — A learning signal derived from hindsight.
- `LinguisticFeedback` (`cognitive.reflexion`) — Verbal feedback for self-improvement.
- `LogicGraph` (`cognitive.symbolic_logic_layer`) — Logic Graph for symbolic reasoning.
- `LogicGraphEdge` (`cognitive.symbolic_logic_layer`) — An edge in the logic graph.
- `LogicGraphNode` (`cognitive.symbolic_logic_layer`) — A node in the logic graph.
- `LogicalAtom` (`cognitive.differentiable_logic`) — Logical atom (grounded predicate).
- `LukasiewiczTNorm` (`cognitive.differentiable_logic`) — Lukasiewicz t-norm for probabilistic semantics.
- `MCDropoutWrapper` (`cognitive.uncertainty`) — Monte Carlo Dropout wrapper for PyTorch models.
- `MemoryEmbedding` (`cognitive.neural_memory_layer`) — Embedded memory entry with vector representation.
- `MemoryKnowledgeGraph` (`cognitive.anomaly_detection_enhanced`) — Knowledge graph built from accumulated memories.
- `MemoryVectorizer` (`cognitive.neural_memory_layer`) — Vectorize memory entries into dense embeddings.
- `MercuryPredictiveCoding` (`cognitive.predictive_coding`) — Predictive coding integration for Mercury Agent.
- `MultiHopReasoner` (`cognitive.multi_hop_reasoner`) — Multi-Hop Reasoning Engine.
- `Observation` (`cognitive.autonomous_agent`) — Data observed by the agent.
- `Ontology` (`cognitive.knowledge_graph`) — Ontology management for anomaly detection knowledge graph.
- `OntologyClass` (`cognitive.knowledge_graph`) — Ontology class definition with typed predicates.
- `OntologyProperty` (`cognitive.knowledge_graph`) — Ontology property definition with domain/range constraints.
- `Option` (`cognitive.hierarchical_planning`) — A temporally extended action (option).
- `OptionLibrary` (`cognitive.hierarchical_planning`) — Library of reusable options (skills).
- `Orientation` (`cognitive.autonomous_agent`) — Agent's understanding of the situation.
- `Outcome` (`cognitive.reflexion`) — The outcome of a decision.
- `PartialCorrelationTest` (`cognitive.causal_discovery`) — Fisher's Z-transformed partial correlation test for conditional independence.
- `PlanNode` (`cognitive.hierarchical_planning`) — A node in the hierarchical plan tree.
- `PlasticConnection` (`cognitive.plasticity_engine`) — A plastic (adaptable) connection between knowledge elements.
- `Predicate` (`cognitive.differentiable_logic`) — Typed predicate with embedding support.
- `Prediction` (`cognitive.predictive_coding`) — A prediction generated by the model.
- `ProductTNorm` (`cognitive.differentiable_logic`) — Product t-norm for smooth gradients.
- `Proposition` (`cognitive.multi_hop_reasoner`) — A logical proposition.
- `RandomWalkEmbedding` (`cognitive.knowledge_graph`) — Learn node embeddings via random walks (DeepWalk/Node2Vec inspired).
- `ReasoningStep` (`cognitive.multi_hop_reasoner`) — A single step in a reasoning chain.
- `Reflection` (`cognitive.autonomous_agent`) — Agent's reflection on outcomes.
- `Reflection` (`cognitive.reflexion`) — A self-reflection on an experience.
- `SHAPExplainer` (`cognitive.explainability`) — SHAP-based explainer for anomaly detection models.
- `STDPParameters` (`cognitive.plasticity_engine`) — Parameters for Spike-Timing Dependent Plasticity.
- `SelfMaintenance` (`cognitive.autonomous_agent`) — Self-maintenance and diagnostic routines.
- `SequenceStep` (`cognitive.chain_of_hindsight`) — A single step in a historical sequence.
- `SubProblem` (`cognitive.chain_of_thought`) — A decomposed sub-problem for least-to-most reasoning.
- `Subgoal` (`cognitive.hierarchical_planning`) — A subgoal in the tactical layer.
- `SymbolicReasoner` (`cognitive.symbolic_logic_layer`) — Symbolic Reasoner for explainable decision making.
- `SymbolicRule` (`cognitive.symbolic_logic_layer`) — A symbolic rule in the logic graph.
- `TemperatureScaler` (`cognitive.uncertainty`) — Temperature scaling for neural network calibration.
- `Thought` (`cognitive.chain_of_thought`) — A single thought in a reasoning chain.
- `ThoughtGenerator` (`cognitive.chain_of_thought`) — Generates individual thoughts for reasoning chains.
- `ThreatCOA` (`cognitive.ipb_engine`) — Threat Course of Action (Phase 4).
- `ThreatCapability` (`cognitive.ipb_engine`) — Threat capability assessment (Phase 3).
- `ThresholdRule` (`cognitive.symbolic_logic_layer`) — A threshold-based rule for numeric comparisons.
- `UncertaintyEstimate` (`cognitive.uncertainty`) — Complete uncertainty estimate for a prediction.
- `UncertaintyQuantifier` (`cognitive.uncertainty`) — Production Uncertainty Quantification Engine.
- `UserSyncInterface` (`cognitive.autonomous_agent`) — Bidirectional interface for user synchronization.
- `ValueExtraction` (`cognitive.anomaly_detection_enhanced`) — Extracted value/opportunity from anomaly.
- `ValueExtractor` (`cognitive.anomaly_detection_enhanced`) — Extract value/opportunities from detected anomalies.
- `ValuePreservation` (`cognitive.ethical_bounding`) — Value preservation analysis.
- `ValuePreserver` (`cognitive.ethical_bounding`) — Value preservation module for maintaining positive outcomes.
- `Warning` (`cognitive.indicator_system`) — A warning generated from an indicator.
- `WeaponsRiskAssessment` (`cognitive.ethical_bounding`) — Result of the two-axis weapons/mass-casualty uplift assessment.
- `_GateEvidence` (`cognitive.ethical_bounding`) — Routed Axis-A/Axis-B evidence for a query (pre-disposition).

**Prediction & forecasting**

- `AnomalyPredictor` (`cognitive.neural_memory_layer`) — Predict future anomalies from detected patterns.
- `BayesianPredictor` (`cognitive.anomaly_detection_enhanced`) — Bayesian predictor for anomaly forecasting.
- `HeteroscedasticEstimator` (`cognitive.uncertainty`) — Estimates input-dependent (heteroscedastic) aleatoric uncertainty.
- `HiddenMarkovPredictor` (`cognitive.anomaly_detection_enhanced`) — Hidden Markov Model for sequence-based anomaly prediction.
- `LinkPredictor` (`cognitive.knowledge_graph`) — Predict missing or future links using learned node embeddings.
- `PrecisionEstimator` (`cognitive.predictive_coding`) — Estimates precision (inverse variance) of predictions.
- `PropensityScoreEstimator` (`cognitive.causal_discovery`) — Propensity score estimation for causal inference.

**Solvers & scorers**

- `AnomalyVerifier` (`cognitive.formal_verification`) — Formal verifier specialized for anomaly detection.
- `BenevolenceScorer` (`cognitive.ethical_bounding`) — Main benevolence scoring engine — the HARD decision-boundary gate.
- `CachedBenevolenceScorer` (`cognitive.benevolence_cache`) — Thread-safe LRU wrapper around :meth:`BenevolenceScorer.enforce`.
- `ConstraintSolver` (`cognitive.formal_verification`) — Solver for constraint satisfaction problems.
- `SafetyVerifier` (`cognitive.formal_verification`) — Verifier for safety properties.

<details><summary>Support types (95)</summary>

`AbstractionLevel`, `ActionResult`, `ActionRisk`, `AdaptationEvent`, `AdaptationResult`, `AdaptationType`, `AgentRole`, `AgentState`, `AgentStatus`, `AnomalyCategory`, `ApprovalRequest`, `ApprovalStatus`, `BeliefState`, `BenefitCategory`, `CalibrationResult`, `CaseOutcome`, `CausalRelationType`, `CognitiveAnalysisResult`, `ConfidenceLevel`, `ConfidenceLevel`, `ConsensusMethod`, `ConsensusResult`, `ConsistencyResult`, `ConstraintType`, `CoordinationStrategy`, `CounterfactualResult`, `CourseOfAction`, `DataType`, `DecisionType`, `DetectionResult`, `DiagnosticResult`, `EdgeType`, `EnvironmentDomain`, `EscalationRecord`, `EthicalConstraintViolationError`, `EthicalPrinciple`, `ExplanationType`, `ExplanationType`, `ExplorationResult`, `ExternalSourceCategory`, `FaithfulnessMetric`, `FeedbackQuality`, `FeedbackType`, `FusionResult`, `FusionStrategy`, `GoalStatus`, `HarmCategory`, `HazardDomain`, `ImprovementPriority`, `IndicatorStatus`, `IndicatorType`, `InferenceResult`, `InferenceStatus`, `InterventionType`, `LogicalConnective`, `MemoryType`, `Message`, `MessageType`, `NodeType`, `OntologyClassType`, `OperationalIntent`, `OutcomeType`, `PatternType`, `PlanExecutionState`, `PlannerType`, `PlasticityMode`, `PlasticityRule`, `PolicyUpdate`, `PredicateType`, `PredictionError`, `PredictionType`, `PredictionType`, `PredictiveResult`, `ProcessingLevel`, `PropertyType`, `PropertyType`, `ReasoningStrategy`, `ReasoningType`, `RefinementResult`, `ReflectionType`, `RelabelingStrategy`, `RetrievalResult`, `RuleType`, `SequenceType`, `SimilarityMetric`, `TemporalOperator`, `ThoughtType`, `ThreatCategory`, `TraversalResult`, `UncertaintyType`, `UpdateMode`, `VerificationReport`, `VerificationResult`, `WarningLevel`, `WeaponsDisposition`

</details>

### `comparison/` — 3 classes (1 capability)

**Other capability classes**

- `PyODComparison` (`comparison.pyod_integration`) — Run Mercury-vs-PyOD comparisons on a shared train/test split.

<details><summary>Support types (2)</summary>

`CombinationMethod`, `PyODAlgorithm`

</details>

### `compliance/` — 22 classes (13 capability)

**Adapters & backends**

- `ECFRClient` (`compliance.osha_anomaly`) — Read-only client for the public Code of Federal Regulations API.

**Detection**

- `OSHAComplianceDetector` (`compliance.osha_anomaly`) — OSHA compliance anomaly detector for industry-specific safety monitoring.

**Other capability classes**

- `NISTAssessment` (`compliance.nist_csf_integrator`) — Result of a single :class:`NISTFunction` assessment.
- `NISTCSFIntegrator` (`compliance.nist_csf_integrator`) — NIST CSF 2.0 integrator for risk management and reporting.
- `NISTCSFReferenceFetcher` (`compliance.nist_csf_integrator`) — Fetch and cache the live NIST CSF 2.0 reference catalogue.
- `NISTCategory` (`compliance.nist_csf_integrator`) — A CSF 2.0 category under a :class:`NISTFunction`.
- `NISTProfile` (`compliance.nist_csf_integrator`) — NIST CSF profile for current-vs-target gap analysis.
- `NISTSubcategory` (`compliance.nist_csf_integrator`) — A CSF 2.0 subcategory under a :class:`NISTCategory`.
- `OSHAHazard` (`compliance.osha_anomaly`) — Detected OSHA hazard.
- `OSHAStandard` (`compliance.osha_anomaly`) — Reference to an OSHA standard citation.
- `OSHATrainingRecommendation` (`compliance.osha_anomaly`) — OSHA training program recommendation.
- `TLPClassification` (`compliance.tlp_handler`) — Result of a TLP classification decision.
- `TLPHandler` (`compliance.tlp_handler`) — Automated TLP classification for Mercury Agent outputs.

<details><summary>Support types (9)</summary>

`ComplianceLevel`, `ECFRClientError`, `HazardCategory`, `ImplementationTier`, `NISTCSFReferenceError`, `NISTFunction`, `OSHASector`, `TLPColor`, `TLPValidationError`

</details>

### `core/` — 362 classes (230 capability)

**Adapters & backends**

- `CacheBackend` (`core.feature_pipeline`) — Protocol for cache backends.

**Analysis & scoring**

- `BinaryConformalClassifier` (`core.conformal_prediction`) — Class-conditional (Mondrian) split-conformal classifier for anomaly detection.
- `LyapunovStabilityAnalyzer` (`core.enhanced_model_domains`) — Lyapunov stability analysis for consciousness and state-based models.
- `MetricsCalculator` (`core.domain_metrics`) — Unified metrics calculator for all domains.
- `ScalarImportanceAnalyzer` (`core.gosnn_optimizer`) — SHAP-inspired importance analysis for GOSNN scalars.
- `ThresholdConfidenceIntervalCalculator` (`core.score_calibration`) — Bootstrap-based confidence interval calculator for thresholds.

**Detection**

- `AdaptiveAnomalyDetector` (`core.adaptive_detector`) — Main adaptive detector that combines all improvements.
- `AnomalyDetector` (`core.rigorous_benchmark`) — Protocol for anomaly detectors to benchmark.
- `BaseDetector` (`core.base`) — Abstract base class for all anomaly detectors.
- `BaseDetector` (`core.stacking_fusion`) — Protocol for base detectors in ensemble.
- `ConformalAnomalyDetector` (`core.conformal_prediction`) — Wrapper for anomaly detectors with conformal prediction.
- `CovarianceAwareDetector` (`core.adaptive_detector`) — Solves the batadal problem.
- `EnhancedBaseDetector` (`core.enhanced_base_domains`) — Enhanced base detector with adaptive thresholds and domain metrics.
- `InformationGeometryDetector` (`core.info_geometry`) — Information geometry-based OOD detector.
- `MultivariateTSDetector` (`core.multivariate_timeseries`) — Multivariate time-series anomaly detector using LTG architecture.
- `TemporalPatternDetector` (`core.adaptive_detector`) — Solves the smd problem.
- `TopologicalAnomalyDetector` (`core.topological_analysis`) — TDA-based anomaly detector using persistent homology features.
- `_MercuryLocalDensityDetector` (`core.adaptive_detector`) — KDTree-based local density anomaly detector (LOF-style, no sklearn).
- `_MercuryRandomProjectionDetector` (`core.adaptive_detector`) — Isolation-style anomaly detector using random projections (no trees/sklearn).

**Engines & orchestration**

- `AdaptiveDomainThresholdManager` (`core.adaptive_domain_thresholding`) — Adaptive per-domain thresholding manager.
- `ConfigurationManager` (`core.config`) — Hierarchical configuration management system.
- `CrossDomainTransferManager` (`core.gosnn_3r_integration`) — Manages cross-domain transfer learning between GOSNN-3R integrations.
- `DoubleHelixEvolutionEngine` (`core.fusion`) — Double-Helix Evolution Engine for state evolution and anomaly detection.
- `EvolutionEngine` (`core.extended_anomaly_engine`) — Evolution engine.
- `FeaturePipeline` (`core.feature_pipeline`) — Complete feature extraction pipeline combining all components.
- `FeatureVersionManager` (`core.feature_pipeline`) — Feature versioning with schema validation.
- `IntegrationEngine` (`core.extended_anomaly_engine`) — Integration engine.
- `Learnable3REngine` (`core.three_r.learnable_fusion`) — High-level engine for learnable 3R fusion.
- `MercuryEquationEngine` (`core.double_helix_engine`) — Double-Helix Evolution Engine implementing 18+ Ava Equation variants.
- `NeuroSymbolicHub` (`core.neurosymbolic_hub`) — Enhanced Neuro-Symbolic Hub for unified anomaly detection.
- `NeurosymbolicEngine` (`core.code_analysis`) — Neurosymbolic integration for code refactoring.
- `NoOpContextManager` (`core.metrics`) — No-op context manager for timing.
- `RecursionEngine` (`core.three_r.engines`) — Implements recursive self-referential processing for hierarchical feature extraction and multi-.
- `RefactoringEngine` (`core.three_r_mechanism`) — Implements dynamic code optimization through AST manipulation for continuous performance.
- `ResonanceEngine` (`core.three_r.engines`) — Implements frequency-domain signal amplification using Fourier analysis for pattern enhancement.
- `ScoreCalibrationManager` (`core.score_calibration`) — Unified calibration manager for anomaly detection scores.
- `SecurityEngine` (`core.extended_anomaly_engine`) — Security engine.
- `SymbolicReasoningEngine` (`core.symbolic_reasoning`) — Symbolic reasoning engine for explainable AI.
- `ThresholdCalibrationPipeline` (`core.calibration_pipeline`) — Threshold auto-calibration pipeline with full provenance tracking.
- `_LegacyRecursionEngine` (`core.three_r_mechanism`) — Implements recursive self-referential processing for hierarchical feature extraction and.
- `_LegacyResonanceEngine` (`core.three_r_mechanism`) — Implements frequency-domain signal amplification using Fourier analysis for pattern.

**Ethics & governance**

- `EthicalAutonomyGovernor` (`core.ai_ethics`) — Oversees AI operations and scores actions on ethical principles.
- `EthicalAutonomyGovernor` (`core.ethical_governor`) — Comprehensive ethical governance system.
- `EthicalGate` (`core.global_omni_scalar_network`) — Trained neural network gate for ethical compliance verification.
- `LyapunovRuntimeEnforcer` (`core.system_coherence`) — Runtime guard enforcing Lyapunov stability V_dot <= -lambda * V.
- `PreExecutionBlockingGate` (`core.ai_ethics`) — Pre-execution blocking gate for safety-critical operations.

**Monitoring**

- `GOSNNPerformanceMonitor` (`core.gosnn_integration`) — Performance monitor for GOSNN operations.

**Neural models & layers**

- `BaseModel` (`core.base`) — Abstract base class for all models.
- `EnhancedAffectiveModel` (`core.enhanced_model_domains`) — Enhanced affective computing model with entropy-based analysis.
- `EnhancedBiometricModel` (`core.enhanced_model_domains`) — Enhanced biometric model with fairness-aware scoring.
- `EnhancedQuantumModel` (`core.enhanced_model_domains`) — Enhanced quantum-inspired anomaly detection with rigorous calculations.
- `FusionDetectionHead` (`core.attention_fusion_stack`) — Detection head over the fused state: standardise -> MLP -> logit.
- `GlobalOmniScalarNetwork` (`core.global_omni_scalar_network`) — Global Omni-Scalar Network (GOSNN) - Central Intelligence Fusion Hub.
- `NeuralEncoder` (`core.neurosymbolic_hub`) — Neural encoder for neuro-symbolic fusion.
- `RefactoringTransformer` (`core.three_r_mechanism`) — AST transformer that applies real refactoring transformations.
- `TrainableFusionStack` (`core.attention_fusion_stack`) — The production fusion stack plus detection head, trainable end-to-end.

**Other capability classes**

- `APIConstants` (`core.centralized_constants`) — API and validation constants.
- `AdaptiveConformalInference` (`core.conformal_prediction`) — Adaptive Conformal Inference for streaming/online settings.
- `AdaptiveNoiseFilter` (`core.signal_processing`) — Adaptive noise filtering for enhanced anomaly detection.
- `AnomalyDetectionConstants` (`core.centralized_constants`) — Anomaly detection thresholds and parameters.
- `AnomalyOracle` (`core.ethical_risk_matrix`) — Anomaly oracle for risk forecasting via pattern-based simulations.
- `AttentionProvider` (`core.gosnn_optimizer`) — Interface for supplying real attention tensors to the optimizer.
- `AttentionVisualization` (`core.adaptive_fusion`) — Attention weights visualization data.
- `BanachRecursion` (`core.three_r.fusion`) — Convergence-bounded recursive computation via Banach contraction mapping.
- `BaseDomainExtractor` (`core.domain_feature_extractors`) — Abstract base class for domain-specific feature extractors.
- `BayesianModelAveraging` (`core.stacking_fusion`) — Bayesian Model Averaging (BMA) for detector fusion.
- `BayesianWeights` (`core.stacking_fusion`) — Bayesian model weights with uncertainty.
- `BenchmarkMetrics` (`core.realworld_benchmark`) — Metrics from a benchmark run.
- `BenevolenceDomainProfile` (`core.centralized_constants`) — Soft sigmoid benevolence weighting parameters for a specific domain.
- `BenevolenceGateConstants` (`core.centralized_constants`) — Domain-specific sigmoid benevolence gate profiles.
- `BenevolenceLoss` (`core.benevolence_optimization`) — Computes benevolence loss for optimization.
- `BetaCalibration` (`core.calibration`) — Beta calibration (Kull et al. 2017) fit by a composite proper objective.
- `BiasMetrics` (`core.ethical_governor`) — Bias audit metrics.
- `BinaryPredictionSet` (`core.conformal_prediction`) — Conformal label prediction sets over ``{0 = normal, 1 = anomaly}``.
- `CalibratedConfidence` (`core.confidence`) — Route a raw score to a calibrated probability, fit with a CV accept-gate.
- `CalibrationConstants` (`core.centralized_constants`) — Score calibration constants.
- `CalibrationDiagnostics` (`core.score_calibration`) — Diagnostic information about score distribution and calibration.
- `CalibrationEnsemble` (`core.adaptive_domain_thresholding`) — Ensemble of calibration methods for robust probability estimation.
- `CalibrationEnsemble` (`core.calibration`) — Ensemble of calibration methods with automatic selection.
- `ChaosMultivariateFusion` (`core.multivariate_timeseries`) — Fusion of chaos-evolutionary optimization with multivariate TS detection.
- `ChaoticMap` (`core.chaos_evolutionary`) — Chaotic map generators for CGO algorithm.
- `CodeIssue` (`core.three_r.types`) — Represents a detected code issue.
- `CognitiveComplexityVisitor` (`core.three_r_mechanism`) — AST visitor that calculates cognitive complexity following SonarQube rules.
- `ComplianceRule` (`core.ethical_risk_matrix`) — Compliance rule definition.
- `ComponentFactory` (`core.di`) — Factory for creating components with proper lifecycle management.
- `ComprehensiveMetrics` (`core.domain_metrics`) — Complete metrics suite for anomaly detection evaluation.
- `ConfidenceConstants` (`core.centralized_constants`) — Confidence classification thresholds.
- `ConformalCalibrationBridge` (`core.conformal_prediction`) — Bridge between conformal prediction and the calibration pipeline.
- `ConformalPredictionSet` (`core.conformal_prediction`) — Result of conformal prediction.
- `ConstantRegistry` (`core.centralized_constants`) — Registry for accessing and overriding constants.
- `DIDeprecationWarning` (`core.di`) — Deprecation warning for the ``core.di`` module.
- `DatasetFingerprint` (`core.calibration_pipeline`) — SHA-256 fingerprint of a dataset's summary statistics.
- `DatasetSpecificEnsemble` (`core.adaptive_detector`) — Ensemble detector that uses dataset-specific strategies.
- `DecisionCurve` (`core.decision_curve`) — A decision curve: net benefit of the model vs the treat-all/none envelopes.
- `DetectorManifestEntry` (`core.detector_registry`) — Declarative entry describing a discoverable detector.
- `DetectorMetrics` (`core.base`) — Metrics for detector performance tracking.
- `DetectorRegistry` (`core.detector_registry`) — Central registry for all anomaly detectors and models.
- `DomainAdaptiveOAEWeights` (`core.three_r.fusion`) — Domain-adaptive weight profiles for the OAE equation.
- `DomainFeatureExtractorFactory` (`core.domain_feature_extractors`) — Factory for creating domain-specific feature extractors.
- `DomainHarmonicConstants` (`core.centralized_constants`) — Domain-specific fundamental frequencies for harmonic analysis.
- `DomainMetrics` (`core.enhanced_base_domains`) — Comprehensive metrics for a detection domain.
- `DomainSpecificMetrics` (`core.domain_metrics`) — Domain-specific metrics for specialized detectors.
- `EthicalConstants` (`core.centralized_constants`) — Ethical governance thresholds and constants.
- `EthicalDecision` (`core.ethical_governor`) — Record of ethical decision with validation.
- `EthicalRiskMatrix` (`core.ethical_risk_matrix`) — Comprehensive ethical risk matrix with compliance and forecasting.
- `EthicalScalars` (`core.ethical_config`) — Weighted ethical-priority configuration for engine decision-making.
- `EthicallyConstrainedFusion` (`core.stacking_fusion`) — Fusion with ethical constraints integrated from GOSNN.
- `EventBasedMetrics` (`core.enhanced_base_domains`) — Event-based metrics for time-series anomaly detection.
- `ExplainableOutput` (`core.neurosymbolic_hub`) — Explainable output from neuro-symbolic hub.
- `FairnessMetrics` (`core.enhanced_model_domains`) — Fairness metrics for bias detection.
- `FeatureFlag` (`core.config`) — Feature flag for A/B testing and gradual rollouts.
- `FeatureImputer` (`core.feature_pipeline`) — Feature imputation for failed detectors using historical patterns.
- `FeatureSchema` (`core.feature_pipeline`) — Schema definition for feature validation.
- `FeatureSelector` (`core.feature_pipeline`) — Feature selection using mutual information and importance scoring.
- `FeatureStandardizer` (`core.feature_pipeline`) — Feature standardization pipeline with multiple scaling strategies.
- `FeatureStore` (`core.feature_pipeline`) — Feature store with caching support.
- `FeedbackLoop` (`core.regenerative`) — Represents a feedback loop in the system (closed-loop design).
- `FibringComposer` (`core.fibring_fusion`) — Stateful composer producing per-sample (neural, symbolic) fusion weights.
- `FibringWeights` (`core.fibring_fusion`) — Composed neural/symbolic weights and the diagnostics that produced them.
- `FinancialDomainConstants` (`core.centralized_constants`) — Financial domain-specific constants.
- `FinancialFeatureExtractor` (`core.domain_feature_extractors`) — Financial domain feature extractor.
- `FisherInformationMatrix` (`core.info_geometry`) — Compute and store the Fisher Information Matrix (FIM).
- `FisherRaoAdaptiveThreshold` (`core.info_geometry`) — Derive and adapt anomaly-detection thresholds from the Fisher metric.
- `FusionConstants` (`core.centralized_constants`) — Fusion and weighting constants.
- `FusionInterface` (`core.base`) — Interface specification for fusion-compatible components.
- `GDPRCompliance` (`core.ethical_risk_matrix`) — GDPR compliance framework with comprehensive validation.
- `GDPRComplianceViolation` (`core.ethical_risk_matrix`) — Detailed GDPR compliance violation with contextual metadata.
- `GDPRLegalBasis` (`core.ethical_risk_matrix`) — Legal basis for data processing under GDPR Article 6.
- `GOSNN3RIntegration` (`core.gosnn_3r_integration`) — Bidirectional integration between GOSNN and 3R mechanism.
- `GOSNNIntegration` (`core.gosnn_integration`) — Integration layer for GOSNN-based multi-domain anomaly detection.
- `HIPAACompliance` (`core.ethical_risk_matrix`) — HIPAA compliance hooks for US healthcare data.
- `InMemoryCache` (`core.feature_pipeline`) — Simple in-memory cache implementation.
- `InfoGeometryCertificate` (`core.governed_fusion`) — Mahalanobis certified-radius/witness for the info-geometry component.
- `InfrastructureConstants` (`core.centralized_constants`) — Infrastructure domain-specific constants.
- `InfrastructureFeatureExtractor` (`core.domain_feature_extractors`) — Infrastructure domain feature extractor.
- `IsotonicCalibration` (`core.calibration`) — Isotonic regression calibration (non-parametric).
- `KnowledgeGraph` (`core.neurosymbolic_hub`) — NetworkX-based knowledge graph for symbolic reasoning.
- `LearnableGOSNN` (`core.learnable_gosnn`) — Learnable Global Omni-Scalar Network.
- `LyapunovConstants` (`core.centralized_constants`) — Lyapunov stability framework constants.
- `LyapunovViolation` (`core.system_coherence`) — Record of a Lyapunov stability violation.
- `Manifold` (`core.riemannian_optimization`) — Abstract base class for a Riemannian manifold.
- `MathConstants` (`core.centralized_constants`) — Mathematical constants used throughout the system.
- `MedicalDomainConstants` (`core.centralized_constants`) — Medical domain-specific constants.
- `MedicalFeatureExtractor` (`core.domain_feature_extractors`) — Medical domain feature extractor.
- `MultiElementBinarization` (`core.novel_class_discovery`) — Multi-Element Binarization (MEBin) for anomaly region processing.
- `MultiHeadAttentionFusion` (`core.global_omni_scalar_network`) — Multi-head attention mechanism for 37D quantum fusion.
- `MultiHeadAttentionProvider` (`core.gosnn_optimizer`) — Concrete :class:`AttentionProvider` backed by real multi-head attention.
- `MultiObjectiveLoss` (`core.benevolence_optimization`) — Multi-objective loss combining detection, benevolence, and fairness.
- `MultiStageFilter` (`core.signal_processing`) — Multi-stage filtering pipeline for comprehensive noise reduction.
- `NaturalGradient` (`core.info_geometry`) — Compute the natural gradient: ``g_nat = F^{-1} g_euclid``.
- `NeuralNetConstants` (`core.centralized_constants`) — Neural network architecture constants.
- `NeurosymbolicEngineDeprecationWarning` (`core.neurosymbolic_engine`) — Custom deprecation warning for neurosymbolic_engine module.
- `NoOpMetric` (`core.metrics`) — No-op metric implementation when prometheus_client is not available.
- `NovelClassDiscovery` (`core.novel_class_discovery`) — Novel anomaly class discovery system.
- `OmniAvaEquation` (`core.three_r.fusion`) — Omni-Ava Equation (OAE) for unified precision scoring in 3R mechanism.
- `OmniMercury` (`core.extended_anomaly_engine`) — Omni mercury.
- `OperatingPoint` (`core.decision_curve`) — The single reconciled operating point + the conformal coverage diagnostic.
- `ParallelDetectorExecutor` (`core.enhanced_base_domains`) — Parallel execution of multiple detectors for efficiency.
- `ParetoFront` (`core.benevolence_optimization`) — Collection of Pareto-optimal solutions.
- `ParetoSolution` (`core.benevolence_optimization`) — A single solution on the Pareto front.
- `PerformanceMetric` (`core.gosnn_integration`) — Single performance measurement.
- `PersistenceDiagram` (`core.topological_analysis`) — Container for persistence diagram data.
- `PipelineStage` (`core.system_coherence`) — A single stage in the detection pipeline signal flow.
- `PlattScaling` (`core.calibration`) — Platt Scaling calibration using logistic regression.
- `QuantumKernelMachine` (`core.quantum_kernels`) — Quantum-inspired kernel machine for anomaly detection.
- `QuantumMetrics` (`core.enhanced_model_domains`) — Metrics for quantum-inspired anomaly detection.
- `RealWorldBenchmarkRunner` (`core.realworld_benchmark`) — Benchmark runner for real-world datasets.
- `RecursionConvergenceConstants` (`core.centralized_constants`) — Convergence bounds for recursive computations.
- `RedisCache` (`core.feature_pipeline`) — Redis cache backend implementation.
- `RegenerativeArchitecture` (`core.regenerative`) — Implements regenerative design principles for net-positive AI systems.
- `RiemannianAdam` (`core.riemannian_optimization`) — Adam optimiser adapted for Riemannian manifolds.
- `RiemannianGradientDescent` (`core.riemannian_optimization`) — Riemannian gradient descent with Armijo backtracking line search.
- `RigorousBenchmarkHarness` (`core.rigorous_benchmark`) — Rigorous benchmark harness for anomaly detection evaluation.
- `RiskScore` (`core.ethical_risk_matrix`) — Risk assessment with likelihood and impact.
- `RuntimeEquationProfile` (`core.equation_profiles`) — Runtime profile metadata and scoring weights.
- `SPDManifold` (`core.riemannian_optimization`) — Manifold of Symmetric Positive Definite (SPD) matrices.
- `ScalarImportance` (`core.gosnn_optimizer`) — Importance metrics for a scalar.
- `ScalarRegistration` (`core.global_omni_scalar_network`) — Registration record for component scalars.
- `ScoreDiagnostics` (`core.score_calibration`) — Tools for analyzing and diagnosing score distributions.
- `SelfHealingDeprecationWarning` (`core.self_healing`) — Deprecation warning for the ``core.self_healing`` compatibility shim.
- `ServiceContainer` (`core.di`) — Dependency injection container with lifecycle management.
- `ServiceDescriptor` (`core.di`) — Describes a registered service.
- `ServiceScope` (`core.di`) — Scoped service resolution context.
- `SigmaDirective` (`core.ethical_governor`) — Σ Directive: Supreme ethical overrides for critical situations.
- `SignalFlowGraph` (`core.system_coherence`) — Describes how signals propagate through the Mercury detection pipeline.
- `SimplexManifold` (`core.riemannian_optimization`) — The probability simplex Delta_n = {x in R^n : x_i >= 0, sum x_i = 1}.
- `SlidingWindowConstants` (`core.centralized_constants`) — Sliding window normalization constants.
- `SlidingWindowNormalizer` (`core.gosnn_3r_integration`) — Sliding window normalization for time-series inputs.
- `SpatialAutocorrelation` (`core.enhanced_base_domains`) — Spatial autocorrelation metrics for graph and spatial domains.
- `StabilityMetrics` (`core.enhanced_model_domains`) — Lyapunov stability metrics for consciousness/state analysis.
- `StackingFusion` (`core.stacking_fusion`) — Stacking (Stacked Generalization) for detector fusion.
- `StatisticalManifold` (`core.info_geometry`) — A point on the manifold of probability distributions.
- `StrictIsotonicCalibration` (`core.calibration`) — Isotonic calibration with a strictly-increasing tie-break (PR #275, X1).
- `SymbolicRule` (`core.neurosymbolic_hub`) — Enhanced symbolic rule with provenance tracking.
- `SymbolicRule` (`core.symbolic_reasoning`) — Represents a symbolic reasoning rule.
- `SyntheticDataGenerator` (`core.realworld_benchmark`) — Generate synthetic benchmark data mimicking real-world datasets.
- `TTLCache` (`core.gosnn_integration`) — Thread-safe LRU cache with TTL for caching detection results.
- `TemperatureScaling` (`core.calibration`) — Temperature Scaling calibration (single-parameter).
- `TemporalTrajectory` (`core.learnable_gosnn`) — Temporal trajectory of scalar values.
- `ThreeRMechanism` (`core.three_r_mechanism`) — Unified Recursion-Resonance-Refactoring mechanism for adaptive.
- `ThresholdConfidenceInterval` (`core.score_calibration`) — Confidence interval for a calibrated threshold.
- `ThresholdDefaults` (`core.config`) — Original threshold values preserved for reference.
- `TrainingMetrics` (`core.code_analysis`) — Metrics for training progress.
- `TriadicPhiWeighting` (`core.global_omni_scalar_network`) — Triadic phi-weighting layer for harmonic synergy in attention fusion.
- `USLawPolling` (`core.ethical_risk_matrix`) — Dynamic US-only law compliance polling.
- `UncertaintyEstimate` (`core.adaptive_fusion`) — Uncertainty quantification result with confidence intervals.
- `VietorisRipsFiltration` (`core.topological_analysis`) — Build a Vietoris-Rips simplicial complex and compute persistent homology.
- `_IdentityCalibration` (`core.calibration`) — No-op calibrator (exact-reducing fallback for the accept-gate).
- `_LegacyOmniAvaEquation` (`core.three_r_mechanism`) — Omni-Ava Equation (OAE) for unified precision scoring in 3R mechanism.
- `_UnionFind` (`core.topological_analysis`) — Weighted union-find with path compression.

**Prediction & forecasting**

- `CrossConformalPredictor` (`core.conformal_prediction`) — Cross-Conformal Prediction (K-fold aggregated).
- `MondrianConformalPredictor` (`core.conformal_prediction`) — Mondrian Conformal Prediction for label-conditional coverage.
- `SplitConformalPredictor` (`core.conformal_prediction`) — Split (Inductive) Conformal Prediction.

**Solvers & scorers**

- `NormalizationVerifier` (`core.system_coherence`) — Verifies that score ranges are compatible at every stage boundary.

**Training & optimization**

- `AdaptiveThresholdCalibrator` (`core.adaptive_detector`) — Solves the covtype F1=0 problem.
- `AdaptiveThresholdOptimizer` (`core.enhanced_base_domains`) — Adaptive threshold optimization using multiple methods.
- `AttentionOptimizer` (`core.gosnn_optimizer`) — Optimizer for 32-head triadic φ-weighting attention.
- `AutoThresholdOptimizer` (`core.score_calibration`) — Automatic threshold optimization using multiple strategies.
- `ChaosEvolutionOptimizer` (`core.chaos_evolutionary`) — Chaos-Evolutionary Optimizer using CGO algorithm.
- `ConstrainedParameterOptimizer` (`core.riemannian_optimization`) — High-level API for Riemannian-constrained parameter optimisation.
- `DomainEnsembleWeightOptimizer` (`core.adaptive_domain_thresholding`) — Domain-specific ensemble weighting optimizer.
- `EthicalGateOptimizer` (`core.gosnn_optimizer`) — Optimized ethical gating with hard σ_Immutable constraint.
- `GOSNNOptimizer` (`core.gosnn_optimizer`) — Main optimizer for GOSNN hub.
- `IsotonicCalibrator` (`core.adaptive_domain_thresholding`) — Isotonic regression for probability calibration.
- `LabelSmoothingCalibrator` (`core.score_calibration`) — Label smoothing for improved calibration in anomaly detection.
- `OAEWeightOptimizer` (`core.three_r.fusion`) — Optimizer for OAE weights using gradient-based methods.
- `ParetoOptimizer` (`core.benevolence_optimization`) — Pareto optimization for multi-objective benevolence optimization.
- `PlattScalingCalibrator` (`core.adaptive_domain_thresholding`) — Platt scaling for probability calibration.
- `VennAbersCalibrator` (`core.conformal_prediction`) — Inductive Venn-Abers predictor (Vovk, Petej & Fedorova 2015).
- `_LegacyOAEWeightOptimizer` (`core.three_r_mechanism`) — Optimizer for OAE weights using scipy.optimize.

<details><summary>Support types (132)</summary>

`AdaptiveThresholdResult`, `AnomalyDetectionMethod`, `AnomalyFusionResult`, `AnomalyType`, `BenchmarkResult`, `BenchmarkResult`, `BlockedActionCategory`, `BlockingGateResult`, `CalibrationMethod`, `CalibrationResult`, `CalibrationResult`, `CalibrationStrategy`, `CircuitState`, `CircularDependencyError`, `CoherenceReport`, `ComplianceRegime`, `ConfidenceLevel`, `ConfidenceReport`, `ConfigException`, `ConfigurableProtocol`, `ConfigurationError`, `ConformalMisconfigurationError`, `CoverageResult`, `DataCharacteristics`, `DataException`, `DataQualityConfig`, `DatasetInfo`, `DatasetProfile`, `DetectionResult`, `DetectorCategory`, `DetectorConfig`, `DetectorConfig`, `DetectorException`, `DetectorInfo`, `DetectorProtocol`, `DetectorProtocol`, `DetectorProtocol`, `DetectorResult`, `DetectorStatus`, `DeviceType`, `DeviceType`, `Domain`, `DomainCalibrationResult`, `DomainConfig`, `DomainFeatureConfig`, `DomainFeatureResult`, `DomainThresholdConfig`, `DomainType`, `DomainType`, `DomainType`, `DriftResult`, `EfficiencyConfig`, `EncoderProtocol`, `EngineConfig`, `EngineConfig`, `EngineConfig`, `EnhancementResult`, `EthicalConfig`, `EthicalConfig`, `EthicalPrinciple`, `EthicalPrinciple`, `EthicsConfig`, `EthicsResult`, `EvolutionConfig`, `EvolutionMode`, `EvolutionState`, `EvolutionStrategy`, `EvolutionStrategy`, `FeatureExtractionResult`, `FeatureExtractionResult`, `FeedbackDirection`, `FilterConfig`, `FilterType`, `FusionConfig`, `FusionException`, `FusionMode`, `FusionMode`, `FusionMode`, `FusionResult`, `FusionStrategy`, `FusionWeightConfig`, `FusionWeightConfig`, `GDPRComplianceResult`, `HandoffResult`, `IntegrationResult`, `IntegrationState`, `IssueSeverity`, `IssueType`, `Learnable3RConfig`, `Learnable3RResult`, `Lifecycle`, `MercuryEngineConfig`, `MercuryEngineConfig`, `MetricResult`, `ModelConfig`, `ModelException`, `ModelProtocol`, `ModelResult`, `NeurosymbolicConfig`, `ObjectiveResult`, `OmniAnomalyException`, `OptimizationResult`, `OptimizationResult`, `OracleActivation`, `PermaculturePrinciple`, `PrivacyLevel`, `ReadinessLevel`, `RefactoringConfig`, `RefactoringResult`, `RiskLevel`, `ScalarCategory`, `ScalarGroup`, `ScalarState`, `ScalingStrategy`, `ScoringFunction`, `SecurityException`, `ServiceNotFoundError`, `SlidingWindowConfig`, `TermType`, `ThreatLevel`, `ThreeRConfig`, `ThreeRConfig`, `ThresholdConfig`, `ThresholdRecord`, `ThresholdResult`, `ThresholdStatus`, `TrainingPhase`, `_LegacyAnomalyDetectionMethod`, `_LegacyEvolutionStrategy`, `_LegacyIssueSeverity`, `_LegacyIssueType`, `_LegacyRefactoringConfig`

</details>

### `data_sources/` — 53 classes (26 capability)

**Data sources & loaders**

- `BGSELFStationSource` (`data_sources.geomagnetic`) — British Geological Survey ELF Station data source.
- `EPAAirNowSource` (`data_sources.earth_science`) — EPA AirNow API data source.
- `GCPDataSource` (`data_sources.consciousness`) — Global Consciousness Project (GCP/EGG Network) data source.
- `GCPDotSource` (`data_sources.consciousness`) — GCPDot visualization/analysis data source.
- `HeartMathGCMSSource` (`data_sources.geomagnetic`) — HeartMath Global Coherence Monitoring System data source.
- `INTERMAGNETSource` (`data_sources.geomagnetic`) — INTERMAGNET (International Real-time Magnetic Observatory Network) data source.
- `JPLFireballSource` (`data_sources.jpl_ssd`) — NASA/JPL CNEOS Fireball API data source.
- `JPLSentrySource` (`data_sources.jpl_ssd`) — NASA/JPL Sentry impact-risk monitoring data source.
- `NASADONKISource` (`data_sources.space_weather`) — NASA DONKI (Space Weather Database) data source.
- `NASAEONETSource` (`data_sources.space_weather`) — NASA EONET (Earth Observatory Natural Event Tracker) data source.
- `NASANeoWsSource` (`data_sources.space_weather`) — NASA NeoWs (Near Earth Object Web Service) data source.
- `NOAACOOPSSource` (`data_sources.earth_science`) — NOAA CO-OPS (Tides & Currents) data source.
- `NOAANWPSSource` (`data_sources.earth_science`) — NOAA National Water Prediction Service data source (river gauges).
- `NOAASWPCSource` (`data_sources.space_weather`) — NOAA Space Weather Prediction Center data source.
- `NWSWeatherAlertsSource` (`data_sources.earth_science`) — National Weather Service Weather Alerts API data source.
- `SolarSystemOpenDataSource` (`data_sources.space_weather`) — Solar System OpenData (Le Système Solaire) data source.
- `SuperMAGSource` (`data_sources.geomagnetic`) — SuperMAG ground magnetometer network data source.
- `USGSEarthquakeSource` (`data_sources.earth_science`) — USGS Earthquake Hazards Program data source.
- `USGSGeomagnetismSource` (`data_sources.geomagnetic`) — USGS Geomagnetism Web Service data source.
- `USGSVolcanoSource` (`data_sources.earth_science`) — USGS Volcano Hazards Program data source (real HANS public API).

**Engines & orchestration**

- `DataSourceManager` (`data_sources.base`) — Unified manager for multiple data sources.

**Other capability classes**

- `DataPoint` (`data_sources.base`) — Standardized data container for all data sources.
- `DataSourceBase` (`data_sources.base`) — Abstract base class for all data sources.
- `GCPEgg` (`data_sources.consciousness`) — Representation of a GCP EGG (Random Number Generator) node.
- `LiveFetch` (`data_sources.live_ingestion`) — Result of a provenance-checked live fetch.
- `SentryImpactRisk` (`data_sources.jpl_ssd`) — NASA Sentry impact monitoring data.

<details><summary>Support types (27)</summary>

`AQICategory`, `AlertLevel`, `COOPSProduct`, `CacheConfig`, `CircuitBreakerConfig`, `CloseApproachEvent`, `DONKIConfig`, `DONKIEventType`, `DataSourceConfig`, `DataSourceError`, `DataSourceType`, `EONETCategory`, `FetchResult`, `FireballEvent`, `GCPAnalysisType`, `GCPDotColor`, `HeartMathSite`, `LiveDataError`, `MagneticElement`, `NWSAlertSeverity`, `RateLimitConfig`, `SWPCProduct`, `SimulatedDataError`, `SourceUnreachableError`, `USGSGeomagConfig`, `USGSObservatory`, `VolcanoAlertLevel`

</details>

### `datasets/` — 64 classes (52 capability)

**Data sources & loaders**

- `ADBenchLoader` (`datasets.adbench`) — ADBench tabular anomaly detection dataset loader.
- `ADRepositoryLoader` (`datasets.adrepository`) — Loader for ADRepository real-world anomaly detection datasets.
- `BATADALLoader` (`datasets.industrial`) — BATADAL (Battle of Attack Detection Algorithms) Dataset Loader.
- `CICIDSLoader` (`datasets.security`) — CICIDS 2017 Network Intrusion Detection Dataset Loader.
- `CWRUBearingLoader` (`datasets.ucr_archive`) — Alias for MBA loader (CWRU Bearing Data).
- `CopernicusERA5Loader` (`datasets.climate`) — Copernicus Climate Data Store ERA5 Reanalysis Loader.
- `CopernicusSeaLevelLoader` (`datasets.climate`) — Copernicus Climate Data Store Sea Level Loader.
- `DSADSLoader` (`datasets.timeseries`) — Daily and Sports Activities Dataset (DSADS) — UCI 256.
- `DatasetLoader` (`datasets.base`) — Abstract base class for dataset loaders.
- `EPAAirQualityLoader` (`datasets.epa_air`) — EPA Air Quality System daily PM2.5 loader.
- `EpilepsyLoader` (`datasets.timeseries`) — Bonn single-electrode EEG (Andrzejak et al. 2001) — Epileptic Seizure set.
- `FEMADisasterLoader` (`datasets.disaster`) — OpenFEMA Disaster Declarations Data Loader.
- `FEMAHazardMitigationLoader` (`datasets.disaster`) — OpenFEMA Hazard Mitigation Grant Program Data Loader.
- `MBALoader` (`datasets.ucr_archive`) — Machine Bearing Anomaly (MBA) Dataset Loader.
- `MIMICLoader` (`datasets.medical`) — MIMIC-III/IV Clinical Database Loader.
- `MITBIHLoader` (`datasets.mitbih`) — MIT-BIH Arrhythmia Database loader.
- `MSDSLoader` (`datasets.ucr_archive`) — Multi-Source Data Stream (MSDS) Dataset Loader.
- `NABLoader` (`datasets.timeseries`) — Numenta Anomaly Benchmark (NAB) Dataset Loader.
- `NASAExoplanetLoader` (`datasets.space`) — NASA Exoplanet Archive Data Loader.
- `NOAABuoyLoader` (`datasets.ocean`) — NOAA National Data Buoy Center (NDBC) Real-Time Data Loader.
- `NOAAERDDAPLoader` (`datasets.noaa_erddap`) — NOAA ERDDAP oceanographic data loader.
- `NOAAGSODLoader` (`datasets.noaa_gsod`) — NOAA Global Summary of the Day (GSOD) weather data loader.
- `NOAAStormEventsLoader` (`datasets.noaa_storm`) — NOAA Storm Events Database loader.
- `NOAAWeatherLoader` (`datasets.environmental`) — Weather/Climate Data Loader using Open-Meteo API.
- `NSLKDDLoader` (`datasets.security`) — NSL-KDD Network Intrusion Detection Dataset Loader.
- `PhysioNetLoader` (`datasets.medical`) — Generic PhysioNet dataset loader for vital sign data.
- `SETILoader` (`datasets.space`) — SETI Signal Dataset Loader.
- `SMAPMSLLoader` (`datasets.timeseries`) — NASA SMAP and MSL Spacecraft Telemetry Dataset Loader.
- `SMDLoader` (`datasets.timeseries`) — Server Machine Dataset (SMD) Loader.
- `SWaTLoader` (`datasets.industrial`) — Secure Water Treatment (SWaT) Dataset Loader.
- `SimonsCMAPLoader` (`datasets.climate`) — Simons Collaborative Marine Atlas Project (CMAP) Data Loader.
- `SolarDynamicsLoader` (`datasets.space`) — NOAA Space Weather Prediction Center Data Loader.
- `ThreatIntelLoader` (`datasets.security`) — MITRE ATT&CK Threat Intelligence Loader.
- `UCRLoader` (`datasets.ucr_archive`) — UCR Time Series Archive Loader.
- `USGSEarthquakeLoader` (`datasets.environmental`) — USGS Earthquake Catalog Data Loader.
- `USGSGeochemistryLoader` (`datasets.environmental`) — USGS Geochemistry Data Loader for Environmental Contamination Detection.
- `WADILoader` (`datasets.industrial`) — Water Distribution (WADI) Dataset Loader.
- `WildfireDataLoader` (`datasets.environmental`) — NASA FIRMS Active Fire Data Loader.
- `WorldOceanDatabaseLoader` (`datasets.climate`) — NCEI World Ocean Database (WOD) Loader.

**Other capability classes**

- `BaseImageDataset` (`datasets.benchmarks.base_dataset`) — Abstract base class for image anomaly detection datasets.
- `BaseVideoDataset` (`datasets.benchmarks.base_dataset`) — Abstract base class for video anomaly detection datasets.
- `BenchmarkComparison` (`datasets.benchmarks.suite`) — Comparison between multiple benchmark results.
- `CardiologyDataset` (`datasets.medical`) — Specialized loader for cardiology data (ECG + vitals).
- `DatasetRegistry` (`datasets.base`) — Registry of available dataset loaders.
- `LoaderDataset` (`datasets.metadata`) — Standardized dataset returned by every loader.
- `MVTecADDataset` (`datasets.benchmarks.mvtec`) — MVTec Anomaly Detection Dataset.
- `ProvenanceFinding` (`datasets.label_provenance`) — One label-provenance gate finding (a leak or inconsistency).
- `RealWorldBenchmarkSuite` (`datasets.benchmarks.suite`) — Comprehensive benchmark suite for real-world datasets.
- `SepsisDataset` (`datasets.medical`) — Specialized loader for Sepsis prediction using MIMIC data.
- `ShanghaiTechDataset` (`datasets.benchmarks.shanghai_tech`) — Shanghai Tech Campus Video Anomaly Detection Dataset.
- `UCFCrimeDataset` (`datasets.benchmarks.ucf_crime`) — UCF-Crime Video Anomaly Detection Dataset.
- `_DynamicSyntheticFlag` (`datasets.exceptions`) — Bool-like flag that reads MERCURY_ALLOW_SYNTHETIC from env dynamically.

<details><summary>Support types (12)</summary>

`BaseDatasetConfig`, `BenchmarkResult`, `DataSourceUnavailableError`, `DatasetConfig`, `DatasetMetadata`, `DatasetMetadata`, `DatasetSplit`, `LoaderDatasetMetadata`, `MVTecADConfig`, `OfflineModeError`, `ShanghaiTechConfig`, `UCFCrimeConfig`

</details>

### `decision/` — 11 classes (8 capability)

**Other capability classes**

- `DecisionAbstentionResponder` (`decision.decider`) — Decide-or-abstain over a detection certificate, then recommend a response.
- `DecisionLedger` (`decision.ledger`) — An append-only, optionally-bounded, thread-safe trail of decision records.
- `DecisionLoop` (`decision.loop`) — Decide -> deter -> verify over detection results, with an audit ledger.
- `DecisionPolicy` (`decision.policy`) — Thresholds governing when the loop decides vs. abstains.
- `Evidence` (`decision.evidence`) — A typed, normalised snapshot of one detection result.
- `ResponsePlan` (`decision.response`) — A bounded, non-destructive response recommendation.
- `ResponsePolicy` (`decision.response`) — Map a disposition + severity onto a :class:`ResponsePlan`.
- `_Verdict` (`decision.decider`) — Mutable scratch verdict the rule stages refine before it is frozen.

<details><summary>Support types (3)</summary>

`DecisionRecord`, `Disposition`, `ResponseAction`

</details>

### `detectors/` — 379 classes (261 capability)

**Adapters & backends**

- `BLIPCaptionBackend` (`detectors.vlm.lvlm_backends`) — BLIP image-captioning backend (Salesforce, BSD-3-Clause).
- `BLIPVQABackend` (`detectors.vlm.lvlm_backends`) — BLIP visual-question-answering backend (Salesforce, BSD-3-Clause).
- `DynamicThresholdAdapter` (`detectors.enhanced_statistical`) — Dynamic threshold adaptation for streaming anomaly detection.
- `LLaVABackend` (`detectors.vlm.lvlm_backends`) — LLaVA backend for vision-language tasks.
- `LVLMBackend` (`detectors.vlm.lvlm_backends`) — Abstract base class for LVLM backends.
- `MiniCPMVBackend` (`detectors.vlm.lvlm_backends`) — MiniCPM-V backend - efficient vision-language model.
- `MockLVLMBackend` (`detectors.vlm.lvlm_backends`) — Mock LVLM backend — hard-fails at construction.
- `Qwen2VLBackend` (`detectors.vlm.lvlm_backends`) — Qwen2-VL backend for vision-language tasks.

**Analysis & scoring**

- `AtmosphericInstabilityAnalyzer` (`detectors.geological.tornado_detector`) — Atmospheric instability analysis for tornado potential.
- `DimensionalAnalyzer` (`detectors.dimensional`) — Multi-dimensional analysis and projection for anomaly detection.
- `DopplerRadarAnalyzer` (`detectors.geological.tornado_detector`) — Doppler radar velocity pattern analyzer for mesocyclone detection.
- `GasEmissionAnalyzer` (`detectors.geological.volcanic`) — Volcanic gas emission anomaly detection.
- `PrecipitationAnalyzer` (`detectors.geological.flood_detector`) — Precipitation accumulation analysis for flood potential.
- `RecursionMultiScaleAnalyzer` (`detectors.geological.landslide`) — 3R Recursion mechanism for multi-scale landslide analysis.
- `ResonanceFrequencyAnalyzer` (`detectors.geological.wildfire`) — 3R Resonance mechanism for smoke pattern frequency analysis.
- `ResonancePatternAnalyzer` (`detectors.geological.tornado_detector`) — FFT-based resonance pattern analyzer for tornado signatures.
- `SVMRFEnsembleClassifier` (`detectors.geological.landslide`) — Ensemble classifier combining SVM and Random Forest for landslide detection.
- `SeaSurfaceTemperatureAnalyzer` (`detectors.geological.hurricane_detector`) — Sea surface temperature analysis for cyclone development potential.
- `SystemicRiskAnalyzer` (`detectors.economic.financial_crisis_detector`) — Systemic risk assessment via network contagion modeling.
- `WindPatternAnalyzer` (`detectors.geological.hurricane_detector`) — Wind pattern analyzer for cyclone structure detection.

**Detection**

- `AccelerationDynamicsDetector` (`detectors.acceleration_dynamics`) — Physics-based acceleration dynamics anomaly detector.
- `AdvancedPhysicsIntegratedDetector` (`detectors.advanced_physics_integration`) — Unified detector integrating all advanced physics-based modules.
- `AdversarialAutoencoderDetector` (`detectors.advanced.adversarial_ae`) — Adversarial Autoencoder Detector for Industrial Control Systems.
- `AnyAnomalyDetector` (`detectors.vlm.anyanomaly`) — AnyAnomaly zero-shot customizable anomaly detector.
- `AtmosphericRiverDetector` (`detectors.meteorological.atmospheric_river_detector`) — Atmospheric-river detector: IVT physics core + Ralph et al. AR scale.
- `AvalancheDetector` (`detectors.geological.avalanche_detector`) — Dry-slab avalanche hazard detector built on real snow-stability physics.
- `BLIPVLMDetector` (`detectors.vlm.blip_vlm`) — BLIP-based zero-shot anomaly detector.
- `BOCPDDetector` (`detectors.bocpd`) — Online change-point detector via Bayesian run-length inference.
- `BankingStressDetector` (`detectors.economic.financial_crisis_detector`) — Banking sector stress detection via credit metrics.
- `BaseVLMDetector` (`detectors.vlm.base_vlm`) — Abstract base class for VLM-based anomaly detectors.
- `BaseVisualDetector` (`detectors.visual.base_visual`) — Abstract base class for visual anomaly detectors.
- `CFlowDetector` (`detectors.visual.cflow`) — CFlow anomaly detector using conditional normalizing flows.
- `COPODDetector` (`detectors.advanced.copod_detector`) — COPOD: Copula-Based Outlier Detection.
- `CUSUMDetector` (`detectors.enhanced_statistical`) — Cumulative Sum (CUSUM) control chart for sequential anomaly detection.
- `ContrastiveLearningDetector` (`detectors.advanced.contrastive_detector`) — Contrastive Learning Detector for Anomaly Detection.
- `CoralBleachingDetector` (`detectors.marine.biodiversity_detector`) — Coral bleaching detection from temperature and stress indicators.
- `DBSCANDetector` (`detectors.enhanced_statistical`) — DBSCAN-based anomaly detector.
- `DeepLogSequenceDetector` (`detectors.deeplog_sequence`) — Back-off n-gram next-key surprisal detector for log/event sequences.
- `DeepSVDDDetector` (`detectors.deep_svdd`) — One-class SVDD hypersphere detector on a random-feature embedding.
- `DerechoDetector` (`detectors.meteorological.derecho_detector`) — Derecho identification per Johns & Hirt (1987) / Corfidi et al. (2016).
- `DiffusionReconstructionDetector` (`detectors.diffusion_ad`) — DDPM reconstruction-error anomaly detector over 1-D sliding windows.
- `DigitalTwinResidualDetector` (`detectors.digital_twin`) — Observed-vs-simulated divergence detector backed by an AR digital twin.
- `DroneAnomalyDetector` (`detectors.drone.detector`) — Drone fault detector combining rule-based and ML-based approaches.
- `DroughtDetector` (`detectors.meteorological.drought_detector`) — Multi-scale drought detector built on SPI/SPEI physics cores.
- `DustStormDetector` (`detectors.meteorological.dust_storm_detector`) — Dust-storm detector built on WMO / Fecan / Shao-Lu formulations.
- `E1PulseDetector` (`detectors.energy.emp_detector`) — E1 component detection (prompt gamma ray pulse).
- `E3PulseDetector` (`detectors.energy.emp_detector`) — E3 component detection (magnetohydrodynamic EMP).
- `EMPDetector` (`detectors.energy.emp_detector`) — Comprehensive EMP and electromagnetic surge detection system.
- `EarthquakeDetector` (`detectors.geological.disaster_detectors`) — Earthquake detector using P/S-wave spectrogram analysis.
- `EchoStateDetector` (`detectors.echo_state`) — Echo-State-Network one-step-ahead predictive residual detector.
- `EnergyBasedDetector` (`detectors.energy_based`) — Gaussian-family quadratic energy-based-model detector.
- `EnhancedStatisticalDetector` (`detectors.enhanced_statistical`) — Unified enhanced statistical anomaly detector.
- `EqTsunamiCascadeDetector` (`detectors.geological.eq_tsunami_cascade`) — Earthquake → tsunami cascade with PTWC-style staged escalation.
- `FinancialCrisisDetector` (`detectors.economic.financial_crisis_detector`) — Comprehensive financial crisis detection system.
- `FireDebrisFlowCascadeDetector` (`detectors.geological.fire_debris_flow_cascade`) — Staged wildfire → debris-flow cascade on the published USGS models.
- `FireIgnitionDetector` (`detectors.geological.wildfire`) — CNN scoring fire activity from satellite-derived thermal rasters.
- `FloodDetector` (`detectors.geological.flood_detector`) — Comprehensive flood detection system.
- `FraudDetector` (`detectors.economic.financial_crisis_detector`) — Algorithmic trading fraud and market manipulation detection.
- `FrequentPatternDetector` (`detectors.frequent_pattern`) — Apriori association-rule violation detector for binary transactions.
- `GWOEnsembleDetector` (`detectors.advanced.gwo_ensemble`) — GWO-Enhanced Ensemble Detector for Anomaly Detection.
- `GaussianProcessDetector` (`detectors.gaussian_process`) — Windowed RBF Gaussian-Process one-step-ahead residual detector.
- `GeoMovementAnomalyDetector` (`detectors.geo_movement`) — Movement-plausibility detector over (lat, lon, epoch-seconds) trajectories.
- `GraphAnomalyDetector` (`detectors.graph_based`) — Detect anomalies in graph-structured data.
- `HailDetector` (`detectors.meteorological.hail_detector`) — Hail / severe-convective detector built on the SPC SHIP formulation.
- `HawkesBurstDetector` (`detectors.hawkes`) — Self-exciting point-process detector for bursts in count streams.
- `HeatwaveDetector` (`detectors.meteorological.heatwave_detector`) — Percentile-climatology heatwave detector with EHF severity.
- `HurricaneDetector` (`detectors.geological.hurricane_detector`) — Comprehensive hurricane/cyclone/typhoon detection system.
- `IMMDetector` (`detectors.imm`) — Interacting-Multiple-Model switching state-space residual detector.
- `InSARDeformationDetector` (`detectors.geological.volcanic`) — InSAR ground deformation detection.
- `IntentionalEMIDetector` (`detectors.energy.emp_detector`) — Intentional electromagnetic interference (IEMI) detection.
- `KMeansDistanceDetector` (`detectors.kmeans_distance`) — Unsupervised detector emitting per-centroid distances as fusion features.
- `LAVADDetector` (`detectors.vlm.lavad`) — LAVAD training-free video anomaly detector.
- `LOFDetector` (`detectors.enhanced_statistical`) — Local Outlier Factor (LOF) detector.
- `LandslideDetector` (`detectors.geological.landslide`) — Comprehensive landslide and avalanche detection system.
- `LightningDetector` (`detectors.meteorological.lightning_detector`) — Flash-rate anomaly detector implementing the 2-sigma lightning jump.
- `MADDetector` (`detectors.enhanced_statistical`) — Median Absolute Deviation (MAD) based anomaly detector.
- `MCDDetector` (`detectors.enhanced_statistical`) — Minimum Covariance Determinant (MCD) based detector.
- `MarineBiodiversityDetector` (`detectors.marine.biodiversity_detector`) — Comprehensive marine biodiversity monitoring system.
- `MarketCrashDetector` (`detectors.economic.financial_crisis_detector`) — Stock market crash detection via volatility and momentum.
- `MercuryAnomalyDetector` (`detectors.statistical`) — Mercury's original anomaly detection ensemble.
- `MeteorDetector` (`detectors.geological.disaster_detectors`) — Meteor detector using optical/radar Bayesian filter with NASA/JPL integration.
- `MultiScaleTransformerDetector` (`detectors.advanced.multi_scale_transformer`) — Multi-Scale Transformer Detector for Time-Series Anomaly Detection.
- `PaDiMDetector` (`detectors.visual.padim`) — PaDiM anomaly detector.
- `ParticleFilterDetector` (`detectors.particle_filter`) — Bootstrap particle-filter detector scoring predictive innovations.
- `PatchCoreDetector` (`detectors.visual.patchcore`) — PatchCore anomaly detector.
- `ReverseDistillationDetector` (`detectors.visual.reverse_distillation`) — Reverse Distillation anomaly detector.
- `RockfallDetector` (`detectors.geological.rockfall_detector`) — Rockfall trigger and precursor detector.
- `RootCauseGraphDetector` (`detectors.rca`) — Random-walk root-cause localiser over a causal / service graph.
- `SPOTDetector` (`detectors.spot_evt`) — Streaming EVT detector with a data-driven Peaks-Over-Threshold threshold.
- `SRCNNDetector` (`detectors.srcnn`) — Spectral-Residual + CNN discriminator anomaly detector (Ren et al., 2019).
- `STFPMDetector` (`detectors.visual.stfpm`) — STFPM anomaly detector using teacher-student distillation.
- `SeismicSwarmDetector` (`detectors.geological.volcanic`) — Volcano-tectonic (VT) earthquake swarm detection.
- `SigmaDirectiveDetector` (`detectors.directive`) — Sigma Directive protocols for anomaly detection.
- `SpatialAnomalyDetector` (`detectors.spatial`) — Spatial anomaly detection for geographic data using:.
- `SpectralResidualDetector` (`detectors.spectral_residual`) — Streaming saliency detector built on the Spectral-Residual transform.
- `SpectralVibrationDetector` (`detectors.spectral_vibration`) — Advanced spectral vibration anomaly detector.
- `SpikingNetworkDetector` (`detectors.spiking`) — Leaky integrate-and-fire population spike-rate novelty detector.
- `StatisticalVLMDetector` (`detectors.vlm.statistical_vlm`) — Offline, deterministic concrete VLM detector (no network, no weights).
- `SubsidenceDetector` (`detectors.geological.subsidence_detector`) — Subsidence and sinkhole-precursor detector for InSAR-style series.
- `SurvivalHazardDetector` (`detectors.survival`) — Kaplan-Meier + Cox proportional-hazards inter-event-time detector.
- `TemporalAnomalyDetector` (`detectors.temporal`) — Time series anomaly detection using:.
- `ThermalHotspotDetector` (`detectors.geological.volcanic`) — Thermal infrared (TIR) hotspot detection.
- `TornadoDetector` (`detectors.geological.tornado_detector`) — Comprehensive tornado detection system.
- `TsunamiDetector` (`detectors.geological.disaster_detectors`) — Tsunami detector using oceanic waveform FFT analysis.
- `UIUXAnomalyDetector` (`detectors.uiux_anomaly`) — Comprehensive UI/UX anomaly detector.
- `VolcanicEruptionDetector` (`detectors.geological.volcanic`) — Comprehensive volcanic eruption detection system.
- `WildfireDetector` (`detectors.geological.wildfire`) — Comprehensive wildfire detection and prediction system.
- `WinterStormDetector` (`detectors.meteorological.winter_storm_detector`) — Winter / ice-storm detector built on cited operational formulations.

**Monitoring**

- `PressureGradientMonitor` (`detectors.geological.tornado_detector`) — Atmospheric pressure gradient monitoring for tornado precursors.
- `PressureTracker` (`detectors.geological.hurricane_detector`) — Central pressure tracking for cyclone intensity monitoring.
- `RiverGaugeMonitor` (`detectors.geological.flood_detector`) — River gauge monitoring for flood stage detection.

**Neural models & layers**

- `AdaptiveFusion` (`detectors.fusion.multimodal_fusion`) — Adaptive fusion that selects strategy based on input characteristics.
- `AdversarialAutoencoder` (`detectors.advanced.adversarial_ae`) — Adversarial Autoencoder for Industrial Control Systems.
- `AffineCoupling` (`detectors.visual.cflow`) — Affine coupling layer for normalizing flow.
- `AssociationDiscrepancy` (`detectors.advanced.multi_scale_transformer`) — Association Discrepancy module from Anomaly Transformer.
- `AttentionFusion` (`detectors.fusion.multimodal_fusion`) — Attention-based fusion that learns cross-modal interactions.
- `BehaviorClassificationNetwork` (`detectors.uiux_anomaly`) — Network for classifying user behavior types.
- `CachedModel` (`detectors.vlm.lvlm_cache`) — Container for a cached LVLM backend.
- `ClickPatternNetwork` (`detectors.uiux_anomaly`) — Network for analyzing click patterns.
- `ConditionalNormalizingFlow` (`detectors.visual.cflow`) — Conditional normalizing flow for anomaly detection.
- `ContrastiveEncoder` (`detectors.advanced.contrastive_detector`) — Encoder network for contrastive learning.
- `ContrastiveModel` (`detectors.advanced.contrastive_detector`) — Full contrastive learning model.
- `CrossScaleAttention` (`detectors.advanced.multi_scale_transformer`) — Cross-scale attention for learning inter-scale dependencies.
- `DecisionConfidenceFusion` (`detectors.fusion.multimodal_fusion`) — Decision-level fusion with confidence weighting.
- `Decoder` (`detectors.advanced.adversarial_ae`) — Decoder network with multi-scale reconstruction.
- `Discriminator` (`detectors.advanced.adversarial_ae`) — Discriminator for adversarial regularization.
- `Encoder` (`detectors.advanced.adversarial_ae`) — Encoder network with sensor correlation modeling.
- `EruptionForecastModel` (`detectors.geological.volcanic`) — Multi-parameter eruption forecasting neural network.
- `FeatureConcatFusion` (`detectors.fusion.multimodal_fusion`) — Feature concatenation fusion with optional projection.
- `FeatureExtractor` (`detectors.visual.backbone`) — Multi-scale feature extractor from pre-trained backbones.
- `FeatureProjection` (`detectors.vlm.blip_vlm`) — Projects BLIP features to 128D for fusion pipeline.
- `FireSpreadModel` (`detectors.geological.wildfire`) — Fire spread rate and direction prediction.
- `InteractionSequenceEncoder` (`detectors.uiux_anomaly`) — Encoder for sequences of user interactions.
- `MLIPVibrationEncoder` (`detectors.spectral_vibration`) — Machine Learning Interatomic Potential inspired vibrational encoder.
- `MouseTrajectoryNetwork` (`detectors.uiux_anomaly`) — Neural network for analyzing mouse trajectory patterns.
- `MultiScaleDecoder` (`detectors.visual.reverse_distillation`) — Multi-scale feature decoder for reconstruction.
- `MultiScaleEncoder` (`detectors.advanced.multi_scale_transformer`) — Multi-scale encoder extracting patterns at different temporal resolutions.
- `MultiScaleTransformerModel` (`detectors.advanced.multi_scale_transformer`) — Multi-Scale Transformer for Time-Series Anomaly Detection.
- `OCEBottleneck` (`detectors.visual.reverse_distillation`) — One-Class Embedding Bottleneck.
- `PatchEmbedding` (`detectors.visual.backbone`) — Extract patch-level embeddings from feature maps.
- `PhononInteractionNetwork` (`detectors.spectral_vibration`) — Neural network modeling phonon-like interactions between frequency modes.
- `PositionalEncoding` (`detectors.advanced.multi_scale_transformer`) — Sinusoidal positional encoding with learnable scale.
- `PositionalEncoding2D` (`detectors.visual.cflow`) — 2D positional encoding for spatial conditioning.
- `ProjectionHead` (`detectors.advanced.contrastive_detector`) — Projection head for contrastive learning.
- `RainfallTriggerModel` (`detectors.geological.landslide`) — Rainfall-induced landslide trigger analysis.
- `ScoreWeightedFusion` (`detectors.fusion.multimodal_fusion`) — Score-level fusion with learned or fixed weights.
- `SeismicTriggerModel` (`detectors.geological.landslide`) — Earthquake-induced landslide trigger analysis.
- `SlopeStabilityModel` (`detectors.geological.landslide`) — Neural network for slope stability assessment.
- `SnowLayer` (`detectors.geological.avalanche_detector`) — One snowpack layer, ordered from the surface downward.
- `SoilSaturationModel` (`detectors.geological.flood_detector`) — Soil saturation modeling for runoff prediction.
- `SpectralCNN` (`detectors.spectral_vibration`) — Convolutional Neural Network for spectral pattern recognition.
- `SpectralGNN` (`detectors.spectral_vibration`) — Complete Graph Neural Network for spectral analysis.
- `SpectralGraphLayer` (`detectors.spectral_vibration`) — Graph Neural Network layer for spectral analysis.
- `StudentNetwork` (`detectors.visual.stfpm`) — Lightweight student network that learns to mimic teacher.
- `TemporalConvBlock` (`detectors.advanced.multi_scale_transformer`) — Temporal convolution block for local pattern extraction.
- `WildfireCNN` (`detectors.geological.wildfire`) — Enhanced CNN for wildfire detection from thermal/NDVI satellite inputs.
- `_Denoiser` (`detectors.diffusion_ad`) — MLP that predicts the noise added to a window, conditioned on timestep.
- `_SRCNN` (`detectors.srcnn`) — 1-D CNN mapping a saliency window to its centre-point anomaly logit.

**Other capability classes**

- `AREpisode` (`detectors.meteorological.atmospheric_river_detector`) — One contiguous period of AR conditions (IVT >= 250 kg m^-1 s^-1).
- `AdditiveProbe` (`detectors.math_arrest.probes.additive`) — Detect level shifts and trend breaks via a linear fit.
- `AnomalyMathArrest` (`detectors.math_arrest.arrest`) — 21-probe Anomaly Math Arrest.
- `AppearanceContextProvider` (`detectors.vlm.advanced_context_providers`) — Appearance context provider for color and texture analysis.
- `AppearanceFeatures` (`detectors.vlm.advanced_context_providers`) — Container for appearance-based features.
- `BandOverlap` (`detectors.cross_domain_frequency`) — A single overlapping frequency band between two domains.
- `BaseContextProvider` (`detectors.vlm.context_providers`) — Abstract base class for context providers.
- `BaseEquationProbe` (`detectors.math_arrest.base_probe`) — Abstract base class for all Anomaly Math Arrest equation probes.
- `BaseFusionModule` (`detectors.fusion.multimodal_fusion`) — Abstract base class for fusion modules.
- `BayesianMeteorFilter` (`detectors.geological.disaster_detectors`) — Bayesian filter for meteor detection combining optical and radar data.
- `BlizzardCheck` (`detectors.meteorological.winter_storm_detector`) — Result of the NWS blizzard-criteria evaluation.
- `BoltzmannCouplingProbe` (`detectors.math_arrest.probes.boltzmann_coupling`) — Detect coupling structure breaks via multi-lag autocorrelation energy.
- `CacheStatistics` (`detectors.vlm.lvlm_cache`) — Statistics for the model cache.
- `CascadeAssessment` (`detectors.meteorological.surge_flood_cascade`) — Output of :meth:`SurgeFloodCascade.evaluate`.
- `CatalanOptimizedProbe` (`detectors.math_arrest.probes.catalan`) — Detect autocorrelation breaks using a Catalan-constant AR(1) model.
- `ClickAnalysis` (`detectors.uiux_anomaly`) — Analysis results for click patterns.
- `CombinedContextProvider` (`detectors.vlm.context_providers`) — Combines multiple context providers for rich context.
- `CorrelationAwareDecorrelator` (`detectors.math_arrest.fusion`) — Detect redundant probe clusters and reduce their weight contributions.
- `CrossDomainCorrelation` (`detectors.cross_domain_frequency`) — Result of cross-domain frequency correlation analysis.
- `CrossDomainFrequencyCorrelator` (`detectors.cross_domain_frequency`) — Detect overlapping significant frequency bands across domains.
- `DimensionalWeights` (`detectors.dimensional`) — Configurable weights for dimensional score combination.
- `DirectiveWeights` (`detectors.directive`) — Configurable weights for Sigma Directive score combination.
- `DroneFault` (`detectors.drone.detector`) — Detected drone fault record.
- `DustEventClass` (`detectors.meteorological.dust_storm_detector`) — Canonical WMO SDS visibility-class labels.
- `EmissionPotential` (`detectors.meteorological.dust_storm_detector`) — Friction-velocity emission-potential result.
- `EnergyMinimizationProbe` (`detectors.math_arrest.probes.energy_minimization`) — Detect energy well escapes via quadratic energy landscape.
- `EnhancedCombinedContextProvider` (`detectors.vlm.advanced_context_providers`) — Enhanced combined context provider with all context types.
- `EthicalConstrainedProbe` (`detectors.math_arrest.probes.ethical`) — Detect boundary violations using percentile-based envelopes.
- `ExponentialDecayProbe` (`detectors.math_arrest.probes.exponential`) — Detect signal degradation using optimal-lambda EWMA residuals.
- `FractalSelfSimilarityProbe` (`detectors.math_arrest.probes.fractal_similarity`) — Detect scale-invariance loss via cross-scale correlation at phi ratio.
- `FrequencyContextProvider` (`detectors.vlm.advanced_context_providers`) — Frequency-domain context provider for periodic pattern detection.
- `FrequencyFeatures` (`detectors.vlm.advanced_context_providers`) — Container for frequency-domain features.
- `FrequencyInfluenceVector` (`detectors.spectral_domain_frequency`) — Output of the Oracle for a single observation.
- `GESDTest` (`detectors.enhanced_statistical`) — Generalized Extreme Studentized Deviate (GESD) test.
- `GrubbsTest` (`detectors.enhanced_statistical`) — Grubbs' Test for detecting outliers.
- `HaboobSignature` (`detectors.meteorological.dust_storm_detector`) — Haboob gust-front detection result.
- `HailAssessment` (`detectors.meteorological.hail_detector`) — Full hail-environment assessment.
- `HarmonicOscillatorProbe` (`detectors.math_arrest.probes.harmonic`) — Detect periodicity violations using a damped harmonic oscillator fit.
- `HazardDiagnostics` (`detectors.hazard_diagnostics`) — Intermediate arrays a hazard detector genuinely computed for one prediction.
- `HelixMultiplicativeProbe` (`detectors.math_arrest.probes.helix`) — Detect multiplicative shocks via log-ratio analysis.
- `IQRRobustProbe` (`detectors.math_arrest.probes.iqr_robust`) — Detect distribution tail anomalies using Tukey IQR fences.
- `KinematicFeatures` (`detectors.acceleration_dynamics`) — Extracted kinematic features from time-series.
- `LVLMBackendCache` (`detectors.vlm.lvlm_cache`) — Thread-safe singleton cache for LVLM backends.
- `LightningCell` (`detectors.meteorological.lightning_detector`) — One spatial cell (grid bin) of flash activity.
- `LightningJump` (`detectors.meteorological.lightning_detector`) — One detected lightning jump (2-sigma exceedance).
- `LyapunovChaosProbe` (`detectors.math_arrest.probes.lyapunov_chaos`) — Detect chaos onset via nearest-neighbor trajectory divergence.
- `ModalityInput` (`detectors.fusion.multimodal_fusion`) — Input from a single modality (VLM or Visual detector).
- `ModifiedZScoreProbe` (`detectors.math_arrest.probes.modified_zscore`) — Detect robust location anomalies using MAD-based modified Z-scores.
- `MomentumProbe` (`detectors.math_arrest.probes.momentum`) — Detect sudden acceleration via second-order finite differences.
- `MovementAssessment` (`detectors.geo_movement`) — Result of a single movement-plausibility evaluation.
- `NDVIProcessor` (`detectors.geological.wildfire`) — NDVI (Normalized Difference Vegetation Index) processor for fuel load estimation.
- `NavigationAnalysis` (`detectors.uiux_anomaly`) — Analysis results for navigation patterns.
- `PhaseSpaceFeatures` (`detectors.acceleration_dynamics`) — Phase space analysis features.
- `PhiWeightedFusion` (`detectors.math_arrest.fusion`) — Phi-weighted score fusion with confidence modulation and decorrelation.
- `PhysicsGOSNNScalars` (`detectors.advanced_physics_integration`) — GOSNN scalar network for physics-based anomaly detection.
- `PointAdjustmentEvaluator` (`detectors.advanced.point_adjustment`) — Evaluator with point-adjustment for time-series anomaly detection.
- `PointKinematics` (`detectors.geological.subsidence_detector`) — Per-point kinematic estimates from one LOS displacement series.
- `PositionContextProvider` (`detectors.vlm.context_providers`) — Position context provider for spatial awareness.
- `PositionalContextExtractor` (`detectors.vlm.context_providers`) — Alias for PositionContextProvider for test compatibility.
- `QuantumAnnealingProbe` (`detectors.math_arrest.probes.quantum_annealing`) — Detect thermodynamic outliers via Boltzmann negative log-likelihood.
- `QuantumSuperpositionProbe` (`detectors.math_arrest.probes.quantum_superposition`) — Detect interference pattern breaks via cosine fringe analysis.
- `R3RecursionResonanceProbe` (`detectors.math_arrest.probes.r3_recursion`) — Detect nonlinear saturation via three fused nonlinear transforms.
- `RainfallAnalysis` (`detectors.geological.fire_debris_flow_cascade`) — Peak rolling accumulations extracted from a rain series.
- `RecursiveFeatureExtractor` (`detectors.geological.tornado_detector`) — Recursive feature extraction for multi-scale tornado pattern detection.
- `ResonanceFrequencyAmplifier` (`detectors.geological.hurricane_detector`) — FFT-based resonance frequency amplifier for storm signal detection.
- `SVDProjectionProbe` (`detectors.math_arrest.probes.svd_projection`) — Detect dimensional collapse via rank-1 SVD Hankel reconstruction.
- `ScrollAnalysis` (`detectors.uiux_anomaly`) — Analysis results for scroll behavior.
- `SemanticContextProvider` (`detectors.vlm.advanced_context_providers`) — Semantic context provider for scene-level understanding.
- `SemanticFeatures` (`detectors.vlm.advanced_context_providers`) — Container for extracted semantic features.
- `SessionAnalysis` (`detectors.uiux_anomaly`) — Complete session analysis results.
- `ShipComponents` (`detectors.meteorological.hail_detector`) — SHIP value with the post-clamp component terms that produced it.
- `SinkholeCluster` (`detectors.geological.subsidence_detector`) — A spatially concentrated cluster of accelerating, subsiding points.
- `SpectralDomainFrequency` (`detectors.spectral_domain_frequency`) — Full-power neuro-symbolic spectral-domain anomaly detection Oracle.
- `SpectralFeatures` (`detectors.spectral_vibration`) — Extracted spectral features from analysis.
- `StreamingScoreEnsemble` (`detectors.detection_tier`) — Calibrated stacking / BMA ensemble over tier detectors' per-point scores.
- `SurgeFloodCascade` (`detectors.meteorological.surge_flood_cascade`) — Staged hurricane -> surge -> compound-flood cascade detector.
- `SurgeSeries` (`detectors.meteorological.surge_flood_cascade`) — Aligned observed / predicted / residual water-level series (metres).
- `SwathGeometry` (`detectors.meteorological.derecho_detector`) — Great-circle geometry of the damage swath.
- `TemporalContextExtractor` (`detectors.vlm.context_providers`) — Alias for TemporalContextProvider for test compatibility.
- `TemporalContextProvider` (`detectors.vlm.context_providers`) — Temporal context provider for action understanding.
- `TemporalLagFeatureExtractor` (`detectors.geological.landslide`) — Extract temporal lag features for landslide prediction.
- `TimeSeriesAugmenter` (`detectors.advanced.contrastive_detector`) — Augmentation strategies for time-series data.
- `TopologyHomologyProbe` (`detectors.math_arrest.probes.topology_homology`) — Detect symmetry breaks via central finite differences.
- `UserInteraction` (`detectors.uiux_anomaly`) — Single user interaction event.
- `VarianceAdaptedProbe` (`detectors.math_arrest.probes.variance_adapted`) — Detect volatility anomalies by comparing rolling variance to training.
- `VibrationDiagnostic` (`detectors.spectral_vibration`) — Diagnostic result for vibration analysis.
- `VolcanicStateHMM` (`detectors.geological.volcanic`) — Hidden Markov Model for volcanic activity state transitions.
- `WaterLevelConfirmation` (`detectors.geological.eq_tsunami_cascade`) — Deterministic DART-style water-level analysis record.
- `WavePropagationProbe` (`detectors.math_arrest.probes.wave_propagation`) — Detect wave equation violations via smoothed discrete Laplacian.
- `ZetaHarmonicProbe` (`detectors.math_arrest.probes.zeta_harmonic`) — Detect phase coherence anomalies via sin/cos phase transform.
- `_NativeLOF` (`detectors.spatial`) — Local Outlier Factor via scipy KDTree (no sklearn dependency).
- `_NativePCA` (`detectors.dimensional`) — Minimal PCA via truncated SVD (no sklearn dependency).

**Solvers & scorers**

- `TierStreamingScorer` (`detectors.detection_tier`) — Adapt a tier detector to the streaming pipeline's ``dict -> dict`` callable.

**Training & optimization**

- `FloodPredictionOptimizer` (`detectors.geological.flood_detector`) — Dynamic model optimization engine for flood prediction.
- `GreyWolfOptimizer` (`detectors.advanced.gwo_ensemble`) — Grey Wolf Optimizer for weight optimization.
- `MultiModalFusionOptimizer` (`detectors.fusion.multimodal_fusion`) — High-level optimizer for multi-modal anomaly detection fusion.
- `MultiScaleFeatureAggregator` (`detectors.visual.backbone`) — Aggregate multi-scale features into a unified representation.
- `RefactoringAdaptiveOptimizer` (`detectors.geological.volcanic`) — 3R Refactoring mechanism for adaptive volcanic model optimization.
- `_EcdfCalibrator` (`detectors.detection_tier`) — Rank / empirical-CDF calibrator: map a score to ``P(reference <= score)``.
- `_IdentityCalibrator` (`detectors.detection_tier`) — No-op calibrator (``calibration='none'``): clip into ``[0, 1]`` only.
- `_IsotonicCalibrator` (`detectors.detection_tier`) — Isotonic-regression calibrator: monotone score->P(anomaly) from labels.
- `_PlattCalibrator` (`detectors.detection_tier`) — Platt-scaling calibrator: logistic score->P(anomaly) from labels.
- `_ScoreCalibrator` (`detectors.detection_tier`) — Per-detector monotone map from a raw score column into a calibrated ``[0, 1]``.

<details><summary>Support types (118)</summary>

`ARAssessmentResult`, `AccelerationAnomalyResult`, `AccelerationDynamicsConfig`, `AdvancedPhysicsConfig`, `AdversarialAEConfig`, `AnomalyCategory`, `AnomalyResult`, `AnyAnomalyConfig`, `AvalancheDangerLevel`, `AvalanchePredictionResult`, `BLIPConfig`, `BackboneType`, `BiodiversityPredictionResult`, `CFlowConfig`, `COPODConfig`, `CascadeStage`, `CascadeStage`, `CascadeState`, `ContextInfo`, `ContextType`, `ContrastiveConfig`, `CrisisSeverity`, `CrisisType`, `CycloneType`, `DerechoResult`, `DetectionConfig`, `DetectorProtocol`, `DomainBandInfo`, `DroneState`, `DroughtAssessmentResult`, `DroughtCategory`, `DynamicThresholdState`, `EMPPredictionResult`, `EMPType`, `EarthquakeMagnitude`, `EarthquakePredictionResult`, `EcosystemHealth`, `EnergyState`, `EruptionType`, `EvidenceRecord`, `FaultType`, `FinancialCrisisPredictionResult`, `FireDebrisFlowResult`, `FireRiskLevel`, `FloodPredictionResult`, `FloodSeverity`, `FloodType`, `FreezeThawResult`, `FrequencyBandResult`, `FrequencyWeighting`, `FusionResult`, `FusionStrategy`, `GWOEnsembleConfig`, `HeatRiskCategory`, `HeatwaveAssessmentResult`, `HeatwaveEvent`, `HeatwaveSeverity`, `HurricanePredictionResult`, `IVTResult`, `IceAccretionResult`, `IntegratedPhysicsResult`, `InteractionType`, `InverseVelocityResult`, `LAVADConfig`, `LVLMType`, `LandslidePredictionResult`, `LandslideRiskLevel`, `LandslideType`, `LightningJumpResult`, `MeteorPredictionResult`, `MeteorThreatLevel`, `MissionPhase`, `ModelState`, `MotionState`, `MultiScaleTransformerConfig`, `NaNPolicy`, `NewSnowLoadingAssessment`, `NonFinitePolicyError`, `PaDiMConfig`, `PatchCoreConfig`, `PhysicsDetectorType`, `PrecipType`, `ProbeResult`, `ReverseDistillationConfig`, `RockfallHazardLevel`, `RockfallPredictionResult`, `SK38Result`, `STFPMConfig`, `SaffirSimpsonCategory`, `ScreeningProduct`, `ScreeningResult`, `SegmentInfo`, `SolarFlareClass`, `SpectralAnalysisMode`, `SpectralDomainFrequencyConfig`, `SpectralVibrationConfig`, `StatisticalMethod`, `SubsidencePredictionResult`, `SubsidenceSeverity`, `TemperatureGradientAssessment`, `ThreatLevel`, `TornadoIntensity`, `TornadoPredictionResult`, `TornadoThreatLevel`, `TsunamiPredictionResult`, `TsunamiSeverity`, `UIUXAnomalyResult`, `UIUXConfig`, `UserBehaviorClass`, `VLMConfig`, `VQAResult`, `VibrationSignatureType`, `VisualDetectorConfig`, `VolcanicActivityLevel`, `VolcanicPredictionResult`, `WildfirePredictionResult`, `WindReport`, `_ThreadLocalState`

</details>

### `distributed/` — 25 classes (17 capability)

**Detection**

- `DistributedAnomalyDetector` (`distributed.cluster`) — Distributed anomaly detection across a Mercury Agent cluster.

**Other capability classes**

- `ClusterConfiguration` (`distributed.raft_consensus`) — Cluster configuration for Raft.
- `ClusterHealth` (`distributed.cluster`) — Overall cluster health status.
- `DataPartitioner` (`distributed.cluster`) — Partitions data across cluster nodes.
- `DistributedMercuryCluster` (`distributed.cluster`) — High-level interface for distributed Mercury Agent operations.
- `DistributedTask` (`distributed.cluster`) — A task to be executed across the cluster.
- `InMemoryTransport` (`distributed.raft_consensus`) — In-memory transport for testing and single-process clusters.
- `LogEntry` (`distributed.raft_consensus`) — A single entry in the Raft log.
- `MessageTransport` (`distributed.raft_consensus`) — Abstract message transport for Raft communication.
- `NodeHealth` (`distributed.cluster`) — Health status of a cluster node.
- `RaftCluster` (`distributed.raft_consensus`) — Manages a cluster of Raft nodes.
- `RaftLog` (`distributed.raft_consensus`) — Persistent log storage for Raft consensus.
- `RaftNode` (`distributed.raft_consensus`) — A single node in a Raft cluster.
- `StateMachine` (`distributed.raft_consensus`) — State machine that applies committed log entries.
- `TCPMessageTransport` (`distributed.tcp_transport`) — Native pure-stdlib TCP transport for Raft consensus.

**Training & optimization**

- `ResultAggregator` (`distributed.cluster`) — Reassembles results from distributed tasks.
- `WorkStealingScheduler` (`distributed.cluster`) — Work-stealing scheduler for load balancing.

<details><summary>Support types (8)</summary>

`AppendEntriesRequest`, `AppendEntriesResponse`, `NodeState`, `PartitionStrategy`, `RequestVoteRequest`, `RequestVoteResponse`, `TaskResult`, `TaskStatus`

</details>

### `emergent/` — 5 classes (4 capability)

**Analysis & scoring**

- `SETICosmicSignalAnalyzer` (`emergent.emergent_life_detector`) — SETI-like cosmic signal anomaly detection using resonance analysis.

**Biometric & recognition**

- `BioSignalPatternRecognizer` (`emergent.emergent_life_detector`) — Bio-signal pattern recognition for detecting life indicators.

**Detection**

- `EmergentLifeDetector` (`emergent.emergent_life_detector`) — Unified emergent life detector integrating SETI, biosignatures, and contact protocol.

**Other capability classes**

- `MultiverseContactProtocolExplorer` (`emergent.emergent_life_detector`) — Multiverse-based exploration of contact protocols.

<details><summary>Support types (1)</summary>

`LifeDetectionResult`

</details>

### `energy/` — 3 classes (2 capability)

**Other capability classes**

- `EnergyOptimization` (`energy.energy_optimization`) — Energy optimization for anomaly detection operations.
- `EnergyProfile` (`energy.energy_optimization`) — Energy consumption profile for an operation.

<details><summary>Support types (1)</summary>

`EnergySource`

</details>

### `ethical/` — 26 classes (16 capability)

**Engines & orchestration**

- `AthenaWisdomEngine` (`ethical.ethical_constraint_engine`) — Athena Wisdom Engine - Greek Strategic Intelligence.
- `ImmutableWisdomEngine` (`ethical.ethical_constraint_engine`) — Immutable Wisdom Engine - Unified Ethical AI Framework.
- `IndivisibleEngine` (`ethical.ethical_alignment_engine`) — Indivisible Engine - Weighted Ethical Principle Verification.
- `MaatBalanceEngine` (`ethical.ethical_constraint_engine`) — Ma'at Balance Engine - Egyptian Archetypal Ethical Verification.
- `PercipienceEngine` (`ethical.ethical_alignment_engine`) — Percipience Engine - Unified Ethical AI Framework.
- `StrategicEngine` (`ethical.ethical_alignment_engine`) — Strategic Engine - Decision Quality Assessment.
- `TwelveFoldVerificationSystem` (`ethical.ethical_alignment_engine`) — Twelve-Fold Verification System - Multi-Dimensional Validation.
- `TwelveFoldVerificationSystem` (`ethical.ethical_constraint_engine`) — Twelve-Fold Verification System - Multi-Dimensional Validation.

**Other capability classes**

- `ArchetypalAnalysis` (`ethical.ethical_alignment_engine`) — Alignment pattern analysis result.
- `GeometricPatternAnalysis` (`ethical.ethical_constraint_engine`) — Geometric pattern analysis result.
- `GeometricPatternProcessor` (`ethical.ethical_alignment_engine`) — Geometric Pattern Processor - Mathematical Pattern Analysis.
- `GeometryAnalysis` (`ethical.ethical_alignment_engine`) — Immutable geometry analysis result.
- `GeometryAnalysis` (`ethical.ethical_constraint_engine`) — Immutable geometry analysis result.
- `ImmutableGeometryProcessor` (`ethical.ethical_constraint_engine`) — Immutable Geometry Processor - Mathematical Pattern Analysis.
- `WisdomQuotient` (`ethical.ethical_alignment_engine`) — Wisdom quotient computation result.
- `WisdomQuotient` (`ethical.ethical_constraint_engine`) — Wisdom quotient computation result.

<details><summary>Support types (10)</summary>

`AlignmentArchetype`, `BalanceResult`, `BalanceResult`, `EthicalPrinciple`, `EthicalPrinciple`, `GeometricPattern`, `TwelveFoldResult`, `TwelveFoldResult`, `VerificationDimension`, `VerificationDimension`

</details>

### `evaluation/` — 9 classes (5 capability)

**Other capability classes**

- `AnomalyMetrics` (`evaluation.metrics`) — Container for anomaly detection evaluation metrics.
- `BaselineComparison` (`evaluation.baselines`) — Container for baseline comparison results.
- `BenchmarkDiagnostics` (`evaluation.benchmark_diagnostics`) — Main diagnostic tool for benchmarking.
- `MetricDiscrepancy` (`evaluation.benchmark_diagnostics`) — Identifies discrepancy between ranking and binary metrics.
- `PreregisteredCoincidenceTest` (`evaluation.event_coincidence`) — A pre-registered coincidence protocol. Fix every field before analysis.

<details><summary>Support types (4)</summary>

`CoincidenceReport`, `CoincidenceResult`, `ConfoundReport`, `DiagnosticResult`

</details>

### `explainability/` — 34 classes (24 capability)

**Other capability classes**

- `AnomalyExplanation` (`explainability.explainer`) — Comprehensive explanation for an anomaly detection result.
- `ChangedFeature` (`explainability.detection_counterfactuals`) — One feature the counterfactual changed.
- `Counterfactual` (`explainability.counterfactuals`) — A single counterfactual explanation.
- `CounterfactualGenerator` (`explainability.counterfactuals`) — Base class for counterfactual generators.
- `CounterfactualSet` (`explainability.counterfactuals`) — Set of diverse counterfactual explanations.
- `DetectionCounterfactual` (`explainability.detection_counterfactuals`) — A validated, minimized counterfactual for one detection decision.
- `DiCECounterfactual` (`explainability.counterfactuals`) — DiCE: Diverse Counterfactual Explanations.
- `ExactShapExplainer` (`explainability.shap`) — Exact Shapley value computation.
- `FeatureConstraint` (`explainability.counterfactuals`) — Constraint on a feature for counterfactual generation.
- `GDPRExplainer` (`explainability.gdpr_compliance`) — GDPR Article 22 compliant explainer.
- `GeneticCounterfactual` (`explainability.counterfactuals`) — Genetic-algorithm counterfactual search (CounterfactualMethod.GENETIC).
- `GlobalAnomalyExplanation` (`explainability.explainer`) — Global explanation for anomaly detection across a dataset.
- `GlobalExplanation` (`explainability.shap`) — Global SHAP explanation across multiple instances.
- `GrowingSpheresCounterfactual` (`explainability.counterfactuals`) — Growing Spheres counterfactual generation.
- `KernelShapExplainer` (`explainability.shap`) — Kernel SHAP explainer.
- `LinearShapExplainer` (`explainability.shap`) — SHAP explainer for linear models.
- `MercuryExplainer` (`explainability.explainer`) — Unified explainability interface for Mercury Agent.
- `PrototypeCounterfactual` (`explainability.counterfactuals`) — Prototype-based counterfactual generation.
- `SamplingShapExplainer` (`explainability.shap`) — Sampling-based SHAP explainer.
- `ShapExplainer` (`explainability.shap`) — Base class for SHAP explainers.
- `ShapExplanation` (`explainability.shap`) — SHAP explanation for a single instance.
- `TreeShapExplainer` (`explainability.shap`) — Tree SHAP explainer for tree-based models.
- `WachterCounterfactual` (`explainability.counterfactuals`) — Wachter et al.
- `_CountingScoreFn` (`explainability.detection_counterfactuals`) — Wrap a raw detector score function with validation and an eval counter.

<details><summary>Support types (10)</summary>

`ComplianceAuditRecord`, `CounterfactualMethod`, `DataSubjectInfo`, `DecisionCategory`, `DecisionInfo`, `DistanceMetric`, `ExplainerType`, `ExplanationLevel`, `ExplanationReport`, `NonFiniteScoreError`

</details>

### `federated_learning/` — 48 classes (28 capability)

**Adapters & backends**

- `FederatedClient` (`federated_learning.client`) — Federated Learning Client.
- `FederatedServer` (`federated_learning.server`) — Federated Learning Server.
- `GOSNNCouplingClient` (`federated_learning.gosnn_coupling`) — Client-side coupling: receives global state, publishes local updates.
- `GOSNNCouplingServer` (`federated_learning.gosnn_coupling`) — Server-side aggregator for bidirectional GOSNN scalar coupling.

**Detection**

- `FederatedAnomalyDetector` (`federated_learning.server`) — Federated Anomaly Detection system.

**Engines & orchestration**

- `CISAFederatedCoordinator` (`federated_learning.cisa_coordinator`) — Coordinates federated learning across CISA critical infrastructure sectors.
- `ClientManager` (`federated_learning.client`) — Manager for multiple federated clients.
- `PrivacyEngine` (`federated_learning.privacy`) — High-level privacy engine for federated learning.

**Neural models & layers**

- `ClientModel` (`federated_learning.federated_robust`) — Client model in federated learning.
- `GlobalModel` (`federated_learning.federated_robust`) — Global aggregated model.

**Other capability classes**

- `ClientHealth` (`federated_learning.client`) — Health metrics for a federated client with fault tolerance tracking.
- `DifferentialPrivacyMechanism` (`federated_learning.privacy`) — Base class for differential privacy mechanisms.
- `FederatedAnomalyDetection` (`federated_learning.federated_robust`) — Federated learning framework for distributed anomaly detection.
- `GaussianMechanism` (`federated_learning.privacy`) — Gaussian mechanism for (epsilon, delta)-differential privacy.
- `GradientClipper` (`federated_learning.privacy`) — Gradient clipping for differential privacy in deep learning.
- `LaplaceMechanism` (`federated_learning.privacy`) — Laplace mechanism for epsilon-differential privacy.
- `LocalDifferentialPrivacy` (`federated_learning.privacy`) — Local Differential Privacy for client-side privatization.
- `PrivacyAccountant` (`federated_learning.privacy`) — Privacy accountant for tracking cumulative privacy loss.
- `PrivacyBudget` (`federated_learning.privacy`) — Privacy budget tracker using (epsilon, delta) accounting.
- `SecureAggregatorWrapper` (`federated_learning.server`) — Secure aggregation with differential privacy.

**Training & optimization**

- `Aggregator` (`federated_learning.server`) — Base class for federated aggregators.
- `FedAdamAggregator` (`federated_learning.server`) — FedAdam aggregator with adaptive learning rates.
- `FedAvgAggregator` (`federated_learning.server`) — Federated Averaging (FedAvg) aggregator.
- `FedProxTrainer` (`federated_learning.client`) — FedProx trainer with proximal regularization.
- `LocalTrainer` (`federated_learning.client`) — Base class for local model trainers.
- `SGDTrainer` (`federated_learning.client`) — Stochastic Gradient Descent trainer.
- `ScaffoldAggregator` (`federated_learning.server`) — SCAFFOLD aggregator with variance reduction.
- `SecureAggregator` (`federated_learning.privacy`) — Secure aggregation for federated learning.

<details><summary>Support types (20)</summary>

`AggregationStrategy`, `ClientConfig`, `ClientConnectionStatus`, `ClientState`, `ClientStatus`, `CrossSectorResult`, `FederationConfig`, `GOSNNCouplingError`, `GOSNNGlobalState`, `GOSNNUpdate`, `LocalUpdate`, `PrivacyMechanism`, `PrivacyReport`, `RoundResult`, `SectorConfig`, `SectorPrivacyLevel`, `SectorType`, `ServerConfig`, `ServerStatus`, `TrainingResult`

</details>

### `federation/` — 4 classes (4 capability)

**Other capability classes**

- `DifferentialPrivacy` (`federation.privacy`) — Apply differential privacy noise to fitted statistics.
- `FederatedNode` (`federation.node`) — A federated Mercury node that trains locally and exports statistics.
- `FittedStatistics` (`federation.statistics`) — Container for a fitted MercuryAnomalyDetector's complete state.

**Training & optimization**

- `FederatedAggregator` (`federation.aggregator`) — Aggregates FittedStatistics from multiple federated nodes.

### `governance/` — 15 classes (10 capability)

**Other capability classes**

- `EuAiActTag` (`governance.eu_ai_act`) — A declarative EU AI Act tier tag -- a classification, never a registered scalar.
- `FailClosedSelfImprovementGovernance` (`governance.self_improvement`) — Default policy: withhold every autonomous self-improvement change.
- `FamilyVet` (`governance.contract`) — The recorded per-family signal vet: formula, signal, verdict, and one-line rationale.
- `GovernanceLedgerEntry` (`governance.contract`) — Provenance record for one adjudicated governance scalar.
- `GovernanceRegistry` (`governance.contract`) — Registers only *GROUNDED*, *metric-only* governance scalars into the GOSNN.
- `GovernanceReview` (`governance.self_improvement`) — Outcome of a governance review over a single proposal.
- `GovernanceScalar` (`governance.contract`) — A single descriptive governance measurement, or a transparent abstention.
- `MeasurementGovernance` (`governance.self_improvement`) — Explicit measurement policy: authorise the change to measure its effect.
- `ProposedRecalibration` (`governance.self_improvement`) — A drift-/performance-triggered recalibration awaiting governance.
- `ProposedThresholdChange` (`governance.self_improvement`) — A Reflexion-proposed operating-threshold change awaiting governance.

<details><summary>Support types (5)</summary>

`EuAiActTier`, `GovernanceOutcome`, `RecalibrationGovernance`, `SignalClass`, `ThresholdGovernance`

</details>

### `gui/` — 7 classes (4 capability)

**Other capability classes**

- `AnomalyDataPoint` (`gui.visualization_dashboard`) — Single data point for visualization.
- `AnomalyVisualizer` (`gui.visualization_dashboard`) — Core visualizer for anomaly detection results.
- `DashboardBuilder` (`gui.visualization_dashboard`) — Builder class for creating comprehensive anomaly detection dashboards.
- `HazardDiagnosticsVisualizer` (`gui.visualization_dashboard`) — Interactive Plotly panels for hazard detector diagnostics payloads.

<details><summary>Support types (3)</summary>

`ChartConfig`, `ChartType`, `DashboardConfig`

</details>

### `harmonics/` — 14 classes (13 capability)

**Analysis & scoring**

- `AdvancedHarmonicAnalyzer` (`harmonics.analyzer`) — High-level interface for spherical harmonic analysis.

**Other capability classes**

- `AssociatedLegendre` (`harmonics.transform`) — Compute associated Legendre polynomials with numerical stability.
- `Bispectrum` (`harmonics.features`) — Bispectrum (third-order statistics) of SH coefficients.
- `FastSHTransform` (`harmonics.transform`) — Optimized spherical harmonic transform using FFT.
- `HarmonicCoefficients` (`harmonics.transform`) — Spherical harmonic coefficients.
- `HarmonicDatabase` (`harmonics.analyzer`) — Database of reference harmonic signatures.
- `HarmonicFeatureExtractor` (`harmonics.features`) — Extract rotation-invariant features from spherical harmonic coefficients.
- `HarmonicSignature` (`harmonics.analyzer`) — Stored harmonic signature for reference.
- `HarmonicSimilarity` (`harmonics.features`) — Compute similarity between harmonic representations.
- `PowerSpectrum` (`harmonics.features`) — Power spectrum of spherical harmonic decomposition.
- `RotationInvariantDescriptor` (`harmonics.features`) — Collection of rotation-invariant shape descriptors.
- `SHBasis` (`harmonics.transform`) — Spherical harmonic basis functions.
- `SphericalHarmonicTransform` (`harmonics.transform`) — Fast spherical harmonic transform.

<details><summary>Support types (1)</summary>

`HarmonicAnomalyResult`

</details>

### `infrastructure/` — 59 classes (39 capability)

**Detection**

- `AgriFoodSecurityDetector` (`infrastructure.humanitarian.agrifood_security`) — Detect agricultural and food security anomalies.
- `ChemicalNuclearDetector` (`infrastructure.chemical_nuclear`) — Anomaly detection for CISA Chemical and Nuclear critical infrastructure.
- `ClimateResilienceDetector` (`infrastructure.humanitarian.climate_resilience`) — Detect climate anomalies for disaster prediction and resilience.
- `CommunicationsITDetector` (`infrastructure.communications_it`) — Anomaly detection for CISA Communications and Information Technology sectors.
- `EconomicResilienceDetector` (`infrastructure.humanitarian.economic_resilience`) — Detect economic anomalies and systemic risks.
- `EducationEquityDetector` (`infrastructure.humanitarian.education_equity`) — Detect educational equity anomalies and learning barriers.
- `EnergyDamsDetector` (`infrastructure.energy_dams`) — Anomaly detection for CISA Energy and Dams critical infrastructure.
- `HealthcareEmergencyDetector` (`infrastructure.healthcare_emergency`) — Anomaly detection for CISA Healthcare and Emergency Services sectors.
- `NeuroscienceDetector` (`infrastructure.humanitarian.neuroscience`) — Detect neural and cognitive pattern anomalies.

**Engines & orchestration**

- `InfrastructureCoordinator` (`infrastructure.__init__`) — Coordinator for infrastructure monitoring modules with flexible selection.
- `StreamingAnomalyPipeline` (`infrastructure.streaming`) — High-level pipeline for streaming anomaly detection.

**Monitoring**

- `CrisisMonitor` (`infrastructure.humanitarian.crisis_monitoring.crisis_monitor`) — Humanitarian Crisis Monitor (Survivor-First CI).
- `EmergingTechMonitor` (`infrastructure.scientific.emerging_tech_monitor`) — Emerging technology monitoring and anomaly detection.
- `EssentialWorkersMonitor` (`infrastructure.humanitarian.essential_workers`) — Essential critical infrastructure workers anomaly detector.
- `GovernmentFacilitiesMonitor` (`infrastructure.humanitarian.government_facilities`) — Government facilities and public administration anomaly detector.
- `NCFMonitor` (`infrastructure.resilience.ncf_monitor`) — National Critical Functions anomaly detector.
- `SpaceInfrastructureMonitor` (`infrastructure.cyber.space_infrastructure`) — Space infrastructure anomaly detector (EU Critical Entities unique sector).
- `WorldBankSectorsMonitor` (`infrastructure.economic.world_bank_sectors`) — World Bank economic sectors anomaly detector.

**Other capability classes**

- `AuditLogHandler` (`infrastructure.observability`) — Abstract base for audit log handlers.
- `AuditLogger` (`infrastructure.observability`) — Production audit logger with compliance support.
- `CircuitBreaker` (`infrastructure.streaming`) — Circuit breaker for streaming connections.
- `CrisisAlert` (`infrastructure.humanitarian.crisis_monitoring.crisis_monitor`) — Alert from crisis monitoring.
- `CrossBorderIntelligence` (`infrastructure.cyber.cross_border_intel`) — Cross-border threat intelligence correlation.
- `DistributedTracer` (`infrastructure.observability`) — Distributed tracing with OpenTelemetry support.
- `FileAuditHandler` (`infrastructure.observability`) — File-based audit log handler with rotation and proper resource management.
- `InMemoryAuditHandler` (`infrastructure.observability`) — In-memory audit log handler for development/testing.
- `InMemoryStreamBroker` (`infrastructure.streaming`) — In-memory message broker backing the ``memory`` streaming backend.
- `InMemoryStreamConsumer` (`infrastructure.streaming`) — In-memory consumer for testing and development.
- `InMemoryStreamProducer` (`infrastructure.streaming`) — In-memory producer for testing and development.
- `KafkaStreamConsumer` (`infrastructure.streaming`) — Kafka consumer with production-grade features.
- `KafkaStreamProducer` (`infrastructure.streaming`) — Kafka producer with production-grade features.
- `MetricPoint` (`infrastructure.observability`) — Metric data point.
- `MetricsCollector` (`infrastructure.observability`) — Metrics collector with Prometheus export support.
- `RedisStreamConsumer` (`infrastructure.streaming`) — Redis Streams consumer with consumer groups.
- `RedisStreamProducer` (`infrastructure.streaming`) — Redis Streams producer for low-latency streaming.
- `StreamConsumer` (`infrastructure.streaming`) — Abstract base class for stream consumers.
- `StreamConsumerFactory` (`infrastructure.streaming`) — Factory for creating stream consumers.
- `StreamProducer` (`infrastructure.streaming`) — Abstract base class for stream producers.
- `StreamProducerFactory` (`infrastructure.streaming`) — Factory for creating stream producers.

<details><summary>Support types (20)</summary>

`AuditAction`, `AuditEvent`, `AuditSeverity`, `CISASector`, `ClimateEvent`, `DamType`, `EconomicThreat`, `EducationThreat`, `EmergencyType`, `EnergySubsector`, `FoodSecurityThreat`, `NeuralThreat`, `PatientStatus`, `ResourceType`, `StreamConfig`, `StreamMessage`, `StreamingBackend`, `_CallBaseline`, `_ModuleInfo`, `_VitalSignRange`

</details>

### `integrations/` — 65 classes (35 capability)

**Adapters & backends**

- `HTTPClient` (`integrations.http.client`) — HTTP client with resilience patterns.
- `HTTPPlatformAdapter` (`integrations.cross_platform_hub`) — Generic HTTP-based platform adapter.
- `MercuryGuardianAdapter` (`integrations.mercury_amacrypto`) — Adapter integrating AMA Cryptography PQC with Mercury Agent.
- `OpenTelemetryAdapter` (`integrations.cross_platform_hub`) — OpenTelemetry collector adapter.
- `PlatformAdapter` (`integrations.cross_platform_hub`) — Abstract base class for platform adapters.
- `PrometheusAdapter` (`integrations.cross_platform_hub`) — Prometheus push gateway adapter.
- `USGSErosM2MClient` (`integrations.usgs_eros`) — Authenticated client for the USGS EROS M2M inventory API.

**Engines & orchestration**

- `CrossPlatformHub` (`integrations.cross_platform_hub`) — Central hub for cross-platform anomaly detection integration.
- `FallbackChain` (`integrations.routing.fallback`) — Chain of fallback handlers for graceful degradation.
- `RequestRouter` (`integrations.routing.router`) — Request router with pattern matching and middleware.

**Monitoring**

- `EWMATimingMonitor` (`integrations.mercury_amacrypto`) — Exponentially Weighted Moving Average timing monitor.

**Neural models & layers**

- `DataTransformer` (`integrations.cross_platform_hub`) — Transform data between different formats.

**Other capability classes**

- `AsyncDatabase` (`integrations.stubs.database`) — Production-ready async database client.
- `AsyncTransactionContext` (`integrations.stubs.database`) — Async database transaction context manager.
- `CacheEntry` (`integrations.stubs.cache`) — Cache entry with metadata.
- `CacheStub` (`integrations.stubs.cache`) — Stub implementation of cache service.
- `CryptoAnomaly` (`integrations.mercury_amacrypto`) — Detected cryptographic anomaly for GOSNN synapse.
- `DatabaseStub` (`integrations.stubs.database`) — Stub implementation of database connection.
- `FallbackHandler` (`integrations.routing.fallback`) — Individual fallback handler.
- `FallbackRegistry` (`integrations.routing.fallback`) — Registry of fallback chains.
- `FinancialService` (`integrations.stubs.financial`) — Production-ready financial data service with multiple API backends.
- `FinancialServiceStub` (`integrations.stubs.financial`) — Stub implementation of financial data service.
- `HTTPCircuitBreaker` (`integrations.http.client`) — Circuit breaker for HTTP requests.
- `HistoricalBar` (`integrations.stubs.financial`) — Historical price bar (OHLCV).
- `RedisCache` (`integrations.stubs.cache`) — Production-ready Redis cache client.
- `Route` (`integrations.routing.router`) — Route definition.
- `RouteMatch` (`integrations.routing.router`) — Result of route matching.
- `RouterGroup` (`integrations.routing.router`) — Group of routes with shared prefix and middleware.
- `SecurityPrice` (`integrations.stubs.financial`) — Security price data.
- `TimingStats` (`integrations.mercury_amacrypto`) — EWMA/MAD timing statistics for anomaly detection.
- `TransactionContext` (`integrations.stubs.database`) — Database transaction context manager.
- `WeatherForecast` (`integrations.stubs.weather`) — Weather forecast for a future time period.
- `WeatherService` (`integrations.stubs.weather`) — Production-ready weather data service with multiple API backends.
- `WeatherServiceStub` (`integrations.stubs.weather`) — Stub implementation of weather service.
- `_TimingAnomaly` (`integrations.mercury_amacrypto`) — Timing anomaly data for security reports.

<details><summary>Support types (30)</summary>

`AnomalyEvent`, `CacheBackend`, `CacheIntegrityError`, `CircuitOpenError`, `CryptoAnomalyType`, `DataFormat`, `DatabaseBackend`, `DatabaseError`, `FallbackError`, `FallbackReason`, `FallbackResult`, `FinancialAPIProvider`, `HTTPClientConfig`, `HTTPError`, `HTTPMethod`, `HTTPResponse`, `MarketData`, `MarketStatus`, `MethodNotAllowedError`, `PlatformConfig`, `PlatformType`, `ProtocolType`, `QueryResult`, `RouteMethod`, `RouteNotFoundError`, `TradingSignal`, `USGSErosError`, `WeatherAPIProvider`, `WeatherCondition`, `WeatherData`

</details>

### `intel/` — 39 classes (22 capability)

**Engines & orchestration**

- `ConfidenceCascadeRouter` (`intel.cascade`) — Routes items cheap-first, escalating to the heavy path on calibrated uncertainty.

**Other capability classes**

- `BoundaryDecision` (`intel.provenance`) — The output boundary's disposition of a candidate emission.
- `CascadeInstrumentation` (`intel.cascade`) — Accumulates per-path counts, compute cost, and latency for a run.
- `ClaimVerdict` (`intel.verifier_loop`) — An adjudicated claim with full provenance.
- `ConsistencyDecision` (`intel.self_consistency`) — A calibrated decision that consulted the self-consistency signal.
- `DurableLabeledQueue` (`intel.feedback_loop.queue`) — Append-only, deduped, fsync'd queue of :class:`LabeledExample`.
- `EmissionDecision` (`intel.verifier_loop`) — The verifier loop's disposition of a candidate emission.
- `LabeledExample` (`intel.feedback_loop.labeling`) — A human-verified labeled example destined for the feedback queue.
- `ModelEntry` (`intel.feedback_loop.rollback`) — A registered staged model pointer.
- `ModelRegistry` (`intel.feedback_loop.rollback`) — Two-pointer (active/previous) staged model registry with atomic writes.
- `Node` (`intel.propositional_claims`) — An AST node: a variable (``op='var'``) or an operator over children.
- `NonceLedger` (`intel.feedback_loop.trigger`) — A durable, append-only ledger of consumed retrain-trigger nonces.
- `Provenance` (`intel.provenance`) — The provenance record that travels with a value through the pipeline.
- `Provenanced` (`intel.provenance`) — A value paired with its :class:`Provenance` -- the typed companion.
- `RedTeamCandidate` (`intel.red_team`) — One mutated attack and how the gate dispositioned it.
- `RegressionVerdict` (`intel.feedback_loop.regression_gate`) — The gate's decision on a candidate vs the baseline.
- `RetrainTrigger` (`intel.feedback_loop.trigger`) — A signed authorization to run a gated retrain against a bound queue state.
- `RoutedOutcome` (`intel.cascade`) — The record of routing one item.
- `ValueMetric` (`intel.value_metrics`) — A stream's declared, measurable value with a baseline and a target.
- `VerifierLoop` (`intel.verifier_loop`) — Routes generative claims through oracles and gates emission on refutation.
- `_Parser` (`intel.propositional_claims`) — Recursive-descent parser over the token list.
- `_Tseitin` (`intel.propositional_claims`) — Tseitin transform of a normalized AST to CNF over propositional Literals.

<details><summary>Support types (17)</summary>

`AuditEvent`, `CandidateReport`, `CascadeConfig`, `ClaimStatus`, `Direction`, `ExampleSource`, `PathResult`, `PropositionalParseError`, `ProvenanceMode`, `ProvenanceOrigin`, `RedTeamConfig`, `RedTeamResult`, `RetrainResult`, `RollbackResult`, `RoutePath`, `SelfConsistencyResult`, `VerifierMode`

</details>

### `loaders/` — 22 classes (22 capability)

**Data sources & loaders**

- `BaseDomainLoader` (`loaders.base`) — Base class for all domain data loaders.
- `DroughtLoader` (`loaders.drought_loader`) — Loader for drought data from NOAA NCEI GSOM monthly summaries.
- `EarthquakeLoader` (`loaders.earthquake_loader`) — Loader for earthquake data from the USGS Earthquake Hazards Program.
- `EnergyLoader` (`loaders.energy_loader`) — Loader for EMP/energy grid data from NOAA SWPC and EIA.
- `FEMALoader` (`loaders.fema_loader`) — Domain loader for FEMA disaster declaration data from OpenFEMA.
- `FinancialLoader` (`loaders.financial_loader`) — Loader for financial crisis data from the FRED API.
- `FloodLoader` (`loaders.flood_loader`) — Loader for flood data from USGS Water Services and FEMA.
- `HailLoader` (`loaders.hail_loader`) — Loader for severe-hail data from the NOAA Storm Prediction Center.
- `HeatwaveLoader` (`loaders.heatwave_loader`) — Loader for heatwave data from NOAA NCEI GSOD daily summaries.
- `HurricaneLoader` (`loaders.hurricane_loader`) — Loader for hurricane/cyclone data from NOAA IBTrACS.
- `LandslideLoader` (`loaders.landslide_loader`) — Loader for landslide data from NASA COOLR (Global Landslide Catalog).
- `MarineLoader` (`loaders.marine_loader`) — Domain loader for marine biodiversity data from OBIS.
- `MeteorLoader` (`loaders.meteor_loader`) — Loader for fireball-archive and NEO close-approach data.
- `NetworkSecurityLoader` (`loaders.network_security_loader`) — Domain loader for network security intrusion detection datasets.
- `PandemicLoader` (`loaders.pandemic_loader`) — Loader for pandemic and outbreak data from OWID and WHO.
- `SepsisLoader` (`loaders.sepsis_loader`) — Loader for sepsis / critical care data from PhysioNet Challenge 2019.
- `SpaceWeatherLoader` (`loaders.space_weather_loader`) — Loader for geomagnetic-storm data (USGS magnetometer + DONKI Kp).
- `TornadoLoader` (`loaders.tornado_loader`) — Loader for tornado data from the NOAA Storm Prediction Center.
- `TsunamiLoader` (`loaders.tsunami_loader`) — Domain loader for NOAA NDBC DART tsunami buoy data.
- `VolcanicLoader` (`loaders.volcanic_loader`) — Loader for volcanic activity data from the USGS Volcano Hazards Program.
- `WildfireLoader` (`loaders.wildfire_loader`) — Loader for wildfire / active-fire data from NASA FIRMS.

**Other capability classes**

- `ProvenanceFinding` (`loaders.label_provenance`) — One label-provenance gate finding (a leak or inconsistency).

### `medical/` — 82 classes (52 capability)

**Adapters & backends**

- `_CoreCalibratorAdapter` (`medical.clinical_calibration`) — Adapt a :mod:`core.calibration` point calibrator to fit/transform.
- `_VennAbersAdapter` (`medical.clinical_calibration`) — Adapt :class:`VennAbersCalibrator` to the fit/transform surface.

**Analysis & scoring**

- `CGMAnalyzer` (`medical.endocrinology_detector`) — Bi-LSTM CGM analyser predicting glycemic state and trend.
- `CardiacBiomarkerAnalyzer` (`medical.cardiology.cardiology_predictor`) — Cardiac biomarker anomaly detection.
- `ECGRhythmAnalyzer` (`medical.cardiology.cardiology_predictor`) — 1D CNN + LSTM for ECG rhythm analysis.
- `FraminghamRiskCalculator` (`medical.cardiology.cardiology_predictor`) — Framingham Risk Score calculator for 10-year CVD risk.
- `NIHSSCalculator` (`medical.critical_care.neurocritical_care`) — NIH Stroke Scale (NIHSS) calculator.
- `QuickSOFACalculator` (`medical.critical_care.sepsis_detector`) — Quick SOFA (qSOFA) calculator for rapid sepsis screening.
- `SOFACalculator` (`medical.critical_care.sepsis_detector`) — Sequential Organ Failure Assessment (SOFA) score calculator.
- `TransmissionNetworkAnalyzer` (`medical.pandemic.pandemic_detector`) — Neural network for transmission network analysis.

**Data sources & loaders**

- `CGMDataSource` (`medical.data_sources`) — Abstract contract every CGM adapter must implement.
- `DexcomV3DataSource` (`medical.data_sources`) — CGM adapter for the Dexcom Developer API v3.
- `FHIRObservationVitalsSource` (`medical.data_sources`) — Vitals adapter for HL7 FHIR R4 ``Observation`` resources.
- `VitalsDataSource` (`medical.data_sources`) — Abstract contract every operating-room vitals adapter must implement.

**Detection**

- `ABMSDisciplineDetector` (`medical.abms_disciplines`) — ABMS Medical Disciplines Anomaly Detector.
- `CaseSurgeDetector` (`medical.pandemic.pandemic_detector`) — Epidemiological case surge detection.
- `EndocrinologyDetector` (`medical.endocrinology_detector`) — Integrated endocrine anomaly detector.
- `MedicalImagingAnomalyDetector` (`medical.medical_cure_predictor`) — Medical imaging anomaly detection using DeepFace-like approach.
- `PandemicDetector` (`medical.pandemic.pandemic_detector`) — Comprehensive pandemic and outbreak detection system.
- `PathogenDetector` (`medical.pandemic.bio_threats.pathogen_detector`) — QBM-Based Pathogen Detector (Medical Interdiction).
- `SepsisDetector` (`medical.critical_care.sepsis_detector`) — Comprehensive sepsis detection system integrating SOFA, qSOFA, and temporal progression.
- `StrokeDetector` (`medical.critical_care.neurocritical_care`) — Neural network for stroke detection and classification.
- `TemporalVitalSignsDetector` (`medical.medical_cure_predictor`) — Temporal vital signs anomaly detector using LSTM.

**Engines & orchestration**

- `MedicalCoordinator` (`medical.__init__`) — Coordinator for medical detection modules with flexible selection.
- `SmartInfusionController` (`medical.anesthesiology_predictor`) — PID closed-loop infusion controller for propofol/remifentanil.
- `TIVAMonitoringSystem` (`medical.anesthesiology_predictor`) — Bi-LSTM TIVA monitor predicting depth of anesthesia and risk profile.

**Ethics & governance**

- `ClinicalSignalGate` (`medical.clinical_signal_gate`) — Certify whether a clinical score has proven signal under set criteria.

**Monitoring**

- `GLP1TherapyMonitor` (`medical.endocrinology_detector`) — GLP-1 therapy monitor with FDA pancreatitis discontinuation rule.
- `HemodynamicMonitor` (`medical.anesthesiology_predictor`) — Hemodynamic monitor evaluating MAP / HR / SpO2 / EtCO2.
- `ICPMonitor` (`medical.critical_care.neurocritical_care`) — Intracranial Pressure (ICP) monitoring and prediction.
- `InhaledInsulinMonitor` (`medical.endocrinology_detector`) — Inhaled-insulin (Afrezza) monitor enforcing FEV1, dose-ceiling and technique guards.
- `MutationTracker` (`medical.pandemic.pandemic_detector`) — Viral mutation tracking via genomic surveillance.
- `SmartInsulinPenMonitor` (`medical.endocrinology_detector`) — Smart-pen monitor enforcing dose-stacking, bolus-ceiling and daily-total guards.

**Neural models & layers**

- `TemporalVitalSignsLSTM` (`medical.medical_cure_predictor`) — PyTorch LSTM for temporal vital signs anomaly detection.

**Other capability classes**

- `CalibrationComparison` (`medical.clinical_calibration`) — Before/after reliability for one calibration method on a clinical score.
- `ClinicalSafetyEnvelope` (`medical.safety`) — Safety metadata attached to every user/provider-facing medical result.
- `PandemicForecast` (`medical.pandemic.forecasting.epidemic_model`) — Result from pandemic forecasting.
- `ReferenceCohort` (`medical.reference_cohorts`) — A labelled reference cohort: aligned score + outcome arrays.
- `ReliabilityBin` (`medical.clinical_metrics`) — One bin of a reliability (calibration) curve.
- `SignalCriteria` (`medical.clinical_signal_gate`) — Thresholds a clinical score must clear to be certified as signal.
- `SignalVerdict` (`medical.clinical_signal_gate`) — Outcome of applying a :class:`SignalCriteria` to a metric report.
- `ThresholdOperatingPoint` (`medical.emergency_thresholds`) — Operating characteristics of one decision threshold.

**Prediction & forecasting**

- `AnesthesiologyPredictor` (`medical.anesthesiology_predictor`) — Integrated anesthesia risk predictor.
- `CardiologyPredictor` (`medical.cardiology.cardiology_predictor`) — Comprehensive cardiology prediction system integrating ECG analysis, biomarker detection, and.
- `EpidemicForecaster` (`medical.pandemic.forecasting.epidemic_model`) — SEIR-Based Pandemic Forecaster (Medical Interdiction).
- `MedicalCurePredictor` (`medical.medical_cure_predictor`) — Unified medical cure predictor integrating temporal analysis, imaging detection, and.
- `NeurocriticalCarePredictor` (`medical.critical_care.neurocritical_care`) — Comprehensive neurocritical care prediction system integrating stroke, seizure, ICP.
- `SeizurePredictor` (`medical.critical_care.neurocritical_care`) — LSTM-based seizure detection and prediction.
- `SepsisProgressionPredictor` (`medical.critical_care.sepsis_detector`) — Neural network for sepsis progression prediction.

**Training & optimization**

- `BayesianBinningCalibrator` (`medical.clinical_calibration`) — Beta-Binomial histogram calibrator (a Bayesian calibrator for scores).
- `TreatmentPathwayOptimizer` (`medical.medical_cure_predictor`) — Treatment pathway optimization using multiverse exploration.
- `_IdentityCalibrator` (`medical.clinical_calibration`) — Degraded fallback: pass scores through unchanged.

<details><summary>Support types (30)</summary>

`ABMSBoard`, `AnesthesiaPredictionResult`, `AnesthesiaRisk`, `AnesthesiaType`, `ArrhythmiaType`, `BioThreatResult`, `CGMReading`, `CalibratorProtocol`, `CardiologyPredictionResult`, `ClinicalMetricReport`, `ConfigurationError`, `DataSourceError`, `DexcomConfig`, `EmergencyThresholdReport`, `EndocrinologyPredictionResult`, `FHIRConfig`, `GlycemicState`, `InsulinDeliveryMethod`, `MedicalAnomalyResult`, `MedicalPredictionResult`, `NeurocriticalPredictionResult`, `OutbreakSeverity`, `PandemicPredictionResult`, `SeizureType`, `SepsisPredictionResult`, `SepsisStage`, `StrokeType`, `VariantConcern`, `VitalsReading`, `_ModuleInfo`

</details>

### `metrics/` — 4 classes (2 capability)

**Other capability classes**

- `AnomalyMetrics` (`metrics.anomaly_metrics`) — Unified anomaly detection metrics calculator.
- `BenchmarkEvaluator` (`metrics.benchmark_evaluator`) — Benchmark evaluation framework.

<details><summary>Support types (2)</summary>

`EvaluationResult`, `MetricResult`

</details>

### `ml/` — 303 classes (241 capability)

**Adapters & backends**

- `BaseDomainAdapter` (`ml.cross_domain_transfer`) — Base class for domain adaptation methods.
- `CORALAdapter` (`ml.cross_domain_transfer`) — Correlation Alignment (CORAL) domain adaptation.
- `DANNAdapter` (`ml.cross_domain_transfer`) — Domain-Adversarial Neural Network (DANN) adapter.
- `JDAAdapter` (`ml.cross_domain_transfer`) — Joint Distribution Adaptation (JDA) adapter.
- `MMDAdapter` (`ml.cross_domain_transfer`) — Maximum Mean Discrepancy (MMD) based domain adaptation.
- `MetaLearningAdapter` (`ml.__init__`) — Lazy-loaded MetaLearningAdapter wrapper.
- `MetaLearningAdapter` (`ml.meta_learning`) — Main Meta-Learning Adapter for Mercury Agent.
- `OptimalTransportAdapter` (`ml.cross_domain_transfer`) — Optimal Transport (OT) based domain adaptation.
- `SubspaceAlignmentAdapter` (`ml.cross_domain_transfer`) — Subspace Alignment for domain adaptation.
- `TCAAdapter` (`ml.cross_domain_transfer`) — Transfer Component Analysis (TCA) adapter.

**Analysis & scoring**

- `DegradationAnalyzer` (`ml.concept_drift_evaluation`) — Analyzes performance degradation patterns over time.
- `FourierHarmonicAnalyzer` (`ml.harmonic_encoder`) — Fourier harmonic analysis for frequency-domain pattern extraction.
- `GolgiAnalyzer` (`ml.cortical_network`) — Golgi stain-inspired analysis of network morphology.
- `GradientBoostingClassifier` (`ml.mercury_ml`) — Gradient Boosting using decision stumps.
- `NisslAnalyzer` (`ml.cortical_network`) — Nissl stain-inspired analysis of activation patterns.
- `PassiveAggressiveClassifier` (`ml.mercury_ml`) — Passive-Aggressive classifier.
- `RandomForestClassifier` (`ml.mercury_ml`) — Random Forest classifier.
- `SGDClassifier` (`ml.mercury_ml`) — SGD classifier for online learning.
- `WeigertAnalyzer` (`ml.cortical_network`) — Weigert stain-inspired analysis of connection strengths.

**Detection**

- `BiasDetector` (`ml.bias_detection`) — ML Bias Detection using Fairlearn metrics.
- `ChiSquaredDriftDetector` (`ml.drift`) — Chi-squared test based drift detector for categorical features.
- `Detector` (`ml.ensemble_coordinator`) — Protocol for anomaly detectors.
- `EnsembleDriftDetector` (`ml.drift`) — Ensemble drift detector combining multiple methods.
- `KolmogorovSmirnovDriftDetector` (`ml.drift`) — Kolmogorov-Smirnov test based drift detector.
- `OnlineDriftDetector` (`ml.drift`) — Online drift detection with adaptive windowing.
- `PopulationStabilityIndexDetector` (`ml.drift`) — Population Stability Index (PSI) based drift detector.

**Engines & orchestration**

- `CascadingPipeline` (`ml.ensemble_coordinator`) — Cascading detection pipeline for efficiency.
- `EnsembleCoordinator` (`ml.ensemble_coordinator`) — Advanced ensemble coordinator for hybrid anomaly detection.
- `InferenceEngine` (`ml.inference`) — General-purpose inference engine for PyTorch models.
- `MemoryManager` (`ml.optimization`) — Memory manager for preventing OOM errors.
- `OnlineLearningPipeline` (`ml.__init__`) — Lazy-loaded OnlineLearningPipeline wrapper.
- `OnlineLearningPipeline` (`ml.online_learning`) — Complete online learning pipeline with drift detection and adaptation.
- `STEMDisciplineRouter` (`ml.fusion_network`) — Routes data to appropriate engines based on STEM discipline.

**Ethics & governance**

- `ThalamocorticalGate` (`ml.cortical_network`) — Thalamocortical gating mechanism for attention.

**Neural models & layers**

- `AffectiveEncoder` (`ml.encoders`) — BiLSTM encoder for emotional sequences (handles both sequential and pre-extracted).
- `AstrophysicalEncoder` (`ml.encoders`) — Encodes astrophysical features (gravitational fields, event horizons).
- `BiometricEncoder` (`ml.encoders`) — Encoder for biometric features (handles both images and pre-extracted embeddings).
- `CorticalColumn` (`ml.cortical_network`) — Single cortical column implementing all 6 layers.
- `CorticalLaminatedNetwork` (`ml.cortical_network`) — Multi-column cortical network with thalamocortical input gating.
- `CorticalLoss` (`ml.cortical_network`) — Biologically-plausible loss combining multiple cortical constraints.
- `CrossModalAttention` (`ml.attention`) — Cross-modal attention between different modalities.
- `CrossModalAttention` (`ml.multimodal_fusion`) — Cross-attention between different modalities.
- `DetectorWeightLearner` (`ml.ensemble`) — Learns optimal weights for each detector based on input characteristics.
- `DomainEncoderStack` (`ml.domain_encoders`) — Joint differentiable domain encoder = spectral + kinematic + Fisher.
- `DualStudentDistillation` (`ml.distillation.dual_student`) — Dual-Student Knowledge Distillation for anomaly detection.
- `EncoderDecoderStudent` (`ml.distillation.dual_student`) — Encoder-Decoder student for patch-level anomaly detection.
- `EncoderEncoderStudent` (`ml.distillation.dual_student`) — Encoder-Encoder student for semantic anomaly detection.
- `EnsembleOmniFusionModel` (`ml.ensemble`) — Ensemble wrapper for OmniFusionModel with stacking and boosting.
- `FeatureEncoder` (`ml.meta_learning`) — Abstract base class for feature encoders.
- `FisherEntropyEncoder` (`ml.domain_encoders`) — Differentiable Fisher/info-geometry encoder.
- `FocalLoss` (`ml.fusion_network`) — Focal Loss for severe class imbalance in anomaly detection.
- `FusionNetwork` (`ml.fusion_network`) — Multi-modality fusion network for combining features from multiple input sources.
- `GatedFusion` (`ml.fusion_network`) — Gated fusion mechanism for combining two input tensors.
- `HATCN_AD` (`ml.hatcn_ad`) — Hierarchical Attention TCN for Anomaly Detection.
- `HebbianLearningRule` (`ml.cortical_network`) — Hebbian learning module implementing "neurons that fire together, wire together".
- `HierarchicalAttention` (`ml.hatcn_ad`) — Multi-scale hierarchical attention.
- `KinematicEncoder` (`ml.domain_encoders`) — Differentiable kinematic encoder via learnable finite-difference convs.
- `LabelEncoder` (`ml.mercury_ml`) — Encode target labels with value between 0 and n_classes-1.
- `LabelSmoothingLoss` (`ml.fusion_network`) — Label smoothing loss for improved calibration.
- `LateralInhibition` (`ml.cortical_network`) — Lateral inhibition module implementing competitive dynamics.
- `LightweightAutoencoder` (`ml.lightweight_primitives`) — Lightweight Autoencoder for unsupervised anomaly detection.
- `LyapunovAnomalyLoss` (`ml.training`) — Training loss with Lyapunov stability constraint for anomaly detection.
- `MLPEncoder` (`ml.meta_learning`) — Simple MLP-based feature encoder.
- `MetaLearner` (`ml.ensemble`) — Neural network meta-learner for stacking ensemble.
- `MultiHeadDetectorAttention` (`ml.attention`) — Multi-head attention over different detector outputs.
- `MultimodalFusion` (`ml.fusion_network`) — Multimodal fusion with named modalities and optional learned weights.
- `MultimodalFusionNetwork` (`ml.multimodal_fusion`) — Multimodal fusion with cross-attention for anomaly detection.
- `OmniFusionModel` (`ml.fusion_network`) — Unified fusion model integrating all detection engines through neural network.
- `PrunedLinear` (`ml.compression`) — Linear layer with weight pruning support.
- `QuantizedLinear` (`ml.compression`) — Quantized linear layer for INT8 inference.
- `QuantumEncoder` (`ml.encoders`) — Encodes quantum state vectors and observables.
- `SparseCoding` (`ml.cortical_network`) — Sparse coding module implementing k-winner-take-all activation.
- `SpatialAttention` (`ml.attention`) — Spatial attention for geographic anomalies.
- `SpectralEncoder` (`ml.domain_encoders`) — Differentiable spectral (resonance) encoder built on ``torch.fft``.
- `SpikeTimingDependentPlasticity` (`ml.cortical_network`) — STDP-inspired learning signal for temporal sequences.
- `StatisticalEncoder` (`ml.encoders`) — Encodes statistical features (z-scores, IQR, distributions).
- `StudentModel` (`ml.compression`) — Smaller student model for knowledge distillation.
- `SymbolicConstraintModule` (`ml.symbolic_constraint`) — Differentiable LTN constraint over detector consensus and fusion output.
- `TemporalAttention` (`ml.attention`) — Attention mechanism for time series anomalies.
- `TemporalBlock` (`ml.hatcn_ad`) — Dilated causal convolutional block.
- `TemporalEncoder` (`ml.encoders`) — LSTM-based encoder for time series features (handles both sequential and pre-extracted).
- `TemporalSequenceEncoder` (`ml.fusion_network`) — Temporal sequence encoder that preserves sequence dependencies.
- `ThreeRAnomalyTransformer` (`ml.three_r_attention`) — Full 3R Anomaly Transformer model for time-series anomaly detection.
- `ThreeRAttentionBlock` (`ml.three_r_attention`) — Complete 3R Attention Block mapping to existing engines.
- `VAE` (`ml.vae_pattern_learner`) — Variational Autoencoder for pattern learning.

**Other capability classes**

- `ActiveLearner` (`ml.__init__`) — Lazy-loaded ActiveLearner wrapper.
- `ActiveLearner` (`ml.active_learning`) — Active learning manager for iterative model improvement.
- `AnomalyDataset` (`ml.training`) — Dataset for anomaly detection training.
- `AnomalyExplainer` (`ml.__init__`) — Lazy-loaded AnomalyExplainer wrapper.
- `AnomalyExplainer` (`ml.explainability`) — Unified anomaly detection explainer.
- `AnomalyMetaLearner` (`ml.__init__`) — Lazy-loaded AnomalyMetaLearner wrapper.
- `AnomalyMetaLearner` (`ml.meta_learning`) — Meta-learner specialized for anomaly detection.
- `AuxiliaryMaxVariance` (`ml.advanced_optimizers`) — Auxiliary Maximum-Variance (AMAV) for multi-task learning.
- `BaseExplainer` (`ml.explainability`) — Base class for explainability methods.
- `BaseFewShotLearner` (`ml.few_shot_learning`) — Base class for few-shot learning methods.
- `BaseSampler` (`ml.active_learning`) — Base class for active learning samplers.
- `BasketDay` (`ml.hazard_training.consciousness_field`) — One parsed GCP day file (per-second per-egg 200-bit trial sums).
- `BatchInference` (`ml.inference`) — Batch inference processor for large datasets.
- `BiasmitigationProcessor` (`ml.fairness`) — Post-processing bias mitigation.
- `BlockCachedRangeReader` (`ml.hazard_training.seismic_wave`) — Read-only seekable file over a ``fetch_range(start, end)`` callable.
- `Catalog` (`ml.hazard_training.earthquake_precursor`) — Parsed earthquake catalog, time-sorted (days since 1980-01-01 UTC).
- `CatalogIndex` (`ml.hazard_training.earthquake_precursor`) — Per-cell and per-neighborhood views of a catalog for fast windowing.
- `ChannelStats` (`ml.rule_evolution`) — Per-channel training statistics anchoring predicate generation.
- `ConceptDriftEvaluator` (`ml.__init__`) — Lazy-loaded ConceptDriftEvaluator wrapper.
- `ConceptDriftEvaluator` (`ml.concept_drift_evaluation`) — Comprehensive concept drift evaluation framework.
- `CounterfactualExplainer` (`ml.explainability`) — Generates counterfactual explanations.
- `CounterfactualExplanation` (`ml.explainability`) — Counterfactual explanation showing what would change the prediction.
- `CrossDomainTransferLearner` (`ml.__init__`) — Lazy-loaded CrossDomainTransferLearner wrapper.
- `CrossDomainTransferLearner` (`ml.cross_domain_transfer`) — Unified cross-domain transfer learning for security anomaly detection.
- `DBSCAN` (`ml.mercury_ml`) — Density-Based Spatial Clustering of Applications with Noise.
- `DDPScaler` (`ml.optimization`) — Distributed Data Parallel scaler for multi-GPU training.
- `DailyGrids` (`ml.hazard_training.wildfire_ignition`) — Daily per-cell rasters of the CA detection census plus coverage.
- `DartArrival` (`ml.hazard_training.tsunami_waveform`) — One observed tsunami arrival at a DART bottom-pressure recorder.
- `DartYearGrid` (`ml.hazard_training.tsunami_waveform`) — One station-year of DART data on the uniform 15-minute grid.
- `DayCells` (`ml.hazard_training.wildfire_ignition`) — Per-cell aggregates for one day: unique active cells only.
- `DegradationAnalysis` (`ml.concept_drift_evaluation`) — Analysis of performance degradation over time.
- `DetectorEntry` (`ml.ensemble_coordinator`) — Entry for a detector in the ensemble.
- `DetectorMetrics` (`ml.ensemble_coordinator`) — Performance metrics for a detector.
- `DifferenceTargetPropagation` (`ml.advanced_optimizers`) — Difference Target Propagation (DTP) for biologically plausible learning.
- `DiversitySampler` (`ml.active_learning`) — Diversity-based sampling.
- `EarlyStopping` (`ml.training`) — Early stopping callback to prevent overfitting.
- `EarthquakeDataset` (`ml.hazard_training.earthquake_precursor`) — Feature/label matrices plus sampling weights and diagnostics targets.
- `Episode` (`ml.few_shot_learning`) — A single few-shot learning episode.
- `EpisodeGenerator` (`ml.few_shot_learning`) — Generates few-shot learning episodes from data.
- `EruptionOnset` (`ml.hazard_training.volcanic_eruption`) — A day-precision eruption onset used as a label anchor.
- `EvaluationOutcome` (`ml.hazard_training.common`) — Learned-vs-physics comparison on the held-out test years.
- `EvolvedRule` (`ml.rule_evolution`) — One evolved fuzzy rule: a conjunction of atoms implying a consequent.
- `EvolvedRuleSearch` (`ml.rule_evolution`) — Deterministic genetic search over rule genomes.
- `FairnessAuditor` (`ml.__init__`) — Lazy-loaded FairnessAuditor wrapper.
- `FairnessAuditor` (`ml.fairness`) — Fairness auditor for anomaly detection models.
- `FeatureAligner` (`ml.cross_domain_transfer`) — Aligns feature spaces between source and target domains.
- `FeatureImportance` (`ml.explainability`) — Feature importance result.
- `FewShotLearner` (`ml.__init__`) — Lazy-loaded FewShotLearner wrapper.
- `FewShotLearner` (`ml.few_shot_learning`) — Unified interface for few-shot learning experiments.
- `FitnessDataset` (`ml.rule_evolution`) — Train/validation detector-score matrices for one real labeled dataset.
- `FusionInference` (`ml.inference`) — Production inference wrapper for fusion model.
- `GaussianMixture` (`ml.mercury_ml`) — Gaussian Mixture Model via EM.
- `GenomeBounds` (`ml.rule_evolution`) — Complexity bounds every operator must respect.
- `GeomagDataset` (`ml.hazard_training.solar_storm`) — Feature/label matrices with per-sample year for temporal splitting.
- `GlobalExplanation` (`ml.explainability`) — Global explanation for model behavior.
- `GradientCache` (`ml.advanced_optimizers`) — LRU cache for synthetic gradient predictions.
- `HookEntry` (`ml.hazard_training.registry`) — One ``load_neural_weights()`` hook in the training registry.
- `HurricaneWindDataset` (`ml.hazard_training.hurricane_wind`) — Patch tensors + labels with per-sample year for temporal splitting.
- `HybridSampler` (`ml.active_learning`) — Hybrid sampling combining uncertainty and diversity.
- `IsotonicRegression` (`ml.mercury_ml`) — Isotonic regression via pool adjacent violators.
- `KFold` (`ml.mercury_ml`) — K-Fold cross-validator.
- `KMeans` (`ml.mercury_ml`) — K-Means clustering.
- `KnowledgeDistiller` (`ml.compression`) — Knowledge distillation trainer.
- `LabeledSample` (`ml.active_learning`) — A labeled sample from the oracle.
- `LandslideDataset` (`ml.hazard_training.landslide_stability`) — Assembled feature/label matrices plus per-sample physics inputs.
- `LightweightMLP` (`ml.lightweight_primitives`) — Lightweight Multi-Layer Perceptron using pure NumPy.
- `LocalExplanation` (`ml.explainability`) — Explanation for a single prediction.
- `LogisticRegression` (`ml.mercury_ml`) — Logistic Regression using L-BFGS.
- `MAML` (`ml.__init__`) — Lazy-loaded MAML wrapper.
- `MAML` (`ml.meta_learning`) — Model-Agnostic Meta-Learning implementation.
- `MAMLNumpy` (`ml.few_shot_learning`) — NumPy implementation of Model-Agnostic Meta-Learning (MAML).
- `MatchingNetworkNumpy` (`ml.few_shot_learning`) — NumPy implementation of Matching Networks.
- `MemoryEfficientCache` (`ml.__init__`) — Lazy-loaded MemoryEfficientCache wrapper.
- `MemoryEfficientCache` (`ml.optimization`) — LRU cache with memory limits.
- `MetaLearner` (`ml.ensemble_coordinator`) — Meta-learner for detector selection and combination.
- `MetaLearningAlgorithm` (`ml.__init__`) — Lazy-loaded MetaLearningAlgorithm enum wrapper.
- `ModelCompressor` (`ml.compression`) — Unified model compression interface.
- `ModelEnsemble` (`ml.inference`) — Ensemble of models with aggregation strategies.
- `NearestNeighbors` (`ml.mercury_ml`) — Unsupervised nearest neighbors using brute-force distance computation.
- `Omni2Hourly` (`ml.hazard_training.solar_storm`) — Parsed OMNI2 hourly records (NaN where the archive holds fill values).
- `OnlineLearner` (`ml.online_learning`) — Base class for online learning models.
- `OnlineLearningMetrics` (`ml.online_learning`) — Metrics for online learning performance.
- `PCA` (`ml.mercury_ml`) — Principal Component Analysis via truncated SVD.
- `ParallelExecutor` (`ml.__init__`) — Lazy-loaded ParallelExecutor wrapper.
- `ParallelExecutor` (`ml.optimization`) — Parallel executor for benchmark loops, backed by ``concurrent.futures``.
- `PassiveAggressiveOnlineLearner` (`ml.online_learning`) — Passive-Aggressive online learner.
- `PipelineContext` (`ml.hazard_training.common`) — Runtime options threaded through every pipeline stage.
- `PlannedDay` (`ml.hazard_training.volcanic_eruption`) — One planned (volcano, day) sample with its chosen station-channel.
- `Prototype` (`ml.meta_learning`) — A class prototype for prototypical networks.
- `PrototypicalNetworkNumpy` (`ml.few_shot_learning`) — NumPy implementation of Prototypical Networks.
- `PrototypicalNetworks` (`ml.__init__`) — Lazy-loaded PrototypicalNetworks wrapper.
- `PrototypicalNetworks` (`ml.meta_learning`) — Prototypical Networks for few-shot learning.
- `QuantumHarmonicOscillator` (`ml.harmonic_encoder`) — Quantum harmonic oscillator model for state evolution.
- `QueryByCommitteeSampler` (`ml.active_learning`) — Query by Committee (QBC) sampling.
- `RegWindowDataset` (`ml.hazard_training.consciousness_field`) — Paired null/fault window composites with full fault bookkeeping.
- `RelationNetworkNumpy` (`ml.few_shot_learning`) — NumPy implementation of Relation Networks.
- `RemoteZipReader` (`ml.hazard_training.schumann_harmonics`) — Random access into a ZIP archive through a byte-range fetcher.
- `Reptile` (`ml.__init__`) — Lazy-loaded Reptile wrapper.
- `Reptile` (`ml.meta_learning`) — Reptile meta-learning algorithm.
- `Rule` (`ml.symbolic_constraint`) — A single fuzzy first-order implication ``antecedent -> consequent``.
- `RuleFitnessEvaluator` (`ml.rule_evolution`) — Fitness = mean held-out validation F1 through the deployed scoring path.
- `RuleGenome` (`ml.rule_evolution`) — An individual of the search: a canonicalised set of evolved rules.
- `RuleGraph` (`ml.symbolic_constraint`) — An ordered, named collection of :class:`Rule` objects.
- `SGDOnlineLearner` (`ml.online_learning`) — Stochastic Gradient Descent based online learner.
- `SHAPExplainer` (`ml.explainability`) — SHAP-based explainability for anomaly detection.
- `SVC` (`ml.mercury_ml`) — SVC using kernel-based scoring.
- `SampleBuffer` (`ml.online_learning`) — Thread-safe buffer for streaming samples.
- `ScarcityWeightSchedule` (`ml.symbolic_constraint`) — Label-scarcity-adaptive schedule for the co-training weight ``lambda``.
- `SchumannDataset` (`ml.hazard_training.schumann_harmonics`) — Hour-level dataset: detector-parity spectra plus derived labels.
- `SeismicDataset` (`ml.hazard_training.seismic_wave`) — Training-side dataset: precomputed spectrograms for train and val.
- `SiameseNetworkNumpy` (`ml.few_shot_learning`) — NumPy implementation of Siamese Networks.
- `SpectrogramGroup` (`ml.hazard_training.seismic_wave`) — One same-shape batch group of spectrograms with aligned labels.
- `SphericalHarmonicDecomposer` (`ml.harmonic_encoder`) — Spherical harmonic decomposition for 3D surface analysis Provides rotation-invariant feature.
- `SplitPerformance` (`ml.concept_drift_evaluation`) — Performance metrics for a single temporal split.
- `StandardScaler` (`ml.mercury_ml`) — Standardize features by removing the mean and scaling to unit variance.
- `StratifiedKFold` (`ml.mercury_ml`) — Stratified K-Fold cross-validator preserving class proportions.
- `StratifiedShuffleSplit` (`ml.mercury_ml`) — Stratified shuffle-split cross-validator.
- `StreamingSample` (`ml.online_learning`) — A single sample from the data stream.
- `SyntheticGradientModule` (`ml.advanced_optimizers`) — Module wrapper for synthetic gradient training with bootstrap and blending.
- `Task` (`ml.meta_learning`) — A meta-learning task (episode).
- `TemporalSplit` (`ml.concept_drift_evaluation`) — A single temporal train/test split.
- `TemporalSplit` (`ml.hazard_training.common`) — A by-year train/validation/test split for time-series training.
- `TemporalSplitter` (`ml.concept_drift_evaluation`) — Generates temporal train/test splits preserving time ordering.
- `ThresholdAtom` (`ml.symbolic_constraint`) — One soft threshold test over a named score channel.
- `TornadoRadarDataset` (`ml.hazard_training.tornado_radar`) — Sector windows with labels and physics observables for training.
- `TsunamiDataset` (`ml.hazard_training.tsunami_waveform`) — Windowed DART dataset with labels and split metadata.
- `UncertaintySampler` (`ml.active_learning`) — Uncertainty-based sampling strategies.
- `VAEPatternLearner` (`ml.vae_pattern_learner`) — Wrapper for VAE-based unsupervised pattern learning.
- `VelSweep` (`ml.hazard_training.tornado_radar`) — Lowest-elevation radial-velocity sweep decoded from a Level-II volume.
- `VolcanicDataset` (`ml.hazard_training.volcanic_eruption`) — Assembled (volcano, day) samples ready for training/evaluation.
- `VolcanoLabels` (`ml.hazard_training.volcanic_eruption`) — Per-volcano label material derived from the GVP catalog.
- `VolcanoSpec` (`ml.hazard_training.volcanic_eruption`) — One named volcano in the training set.
- `VotingEnsemble` (`ml.ensemble`) — Simple voting ensemble for detector outputs.
- `WildfireDataset` (`ml.hazard_training.wildfire_ignition`) — Sampled patches with labels, per-sample year/month/kind, and counts.
- `ZipMember` (`ml.hazard_training.schumann_harmonics`) — One central-directory entry of a (possibly remote) ZIP archive.
- `_CellArrays` (`ml.hazard_training.earthquake_precursor`) — Time-sorted event arrays for one cell or one 3x3 neighborhood.
- `_DecisionStump` (`ml.mercury_ml`) — Simple decision tree (stump or shallow) for gradient boosting / RF.
- `_DownloadBudget` (`ml.hazard_training.tornado_radar`) — Loud accounting against the volume-download byte cap.
- `_HttpRangeFetcher` (`ml.hazard_training.seismic_wave`) — Thread-safe HTTP Range transport with mirror failover + byte counting.
- `_RangeFetcher` (`ml.hazard_training.schumann_harmonics`) — Thread-safe HTTP range fetcher with a hard total-byte budget.
- `_Sample` (`ml.hazard_training.landslide_stability`) — One (cell, date) candidate before feature extraction.
- `_SynopticPoints` (`ml.hazard_training.hurricane_wind`) — IBTrACS main-track synoptic points, parsed into flat arrays.

**Prediction & forecasting**

- `SyntheticGradientPredictor` (`ml.advanced_optimizers`) — Synthetic Gradient Predictor for decoupled neural network training.

**Solvers & scorers**

- `IsolationScorer` (`ml.lightweight_primitives`) — Lightweight isolation-based anomaly scorer.

**Training & optimization**

- `BayesianWeightOptimizer` (`ml.ensemble_coordinator`) — Bayesian weight optimization using Thompson Sampling.
- `FusionTrainer` (`ml.training`) — PyTorch Lightning trainer for fusion model.
- `GradientWeightOptimizer` (`ml.ensemble_coordinator`) — Gradient-based weight optimization using online learning.
- `GreyWolfOptimizer` (`ml.gwo_optimizer`) — GWO for feature selection and hyperparameter optimization.
- `LearningRateScheduler` (`ml.training`) — Wrapper for PyTorch learning rate schedulers.
- `MercuryExponentialDecayOptimizer` (`ml.training`) — Mercury optimizer with exponential decay.
- `MercuryHarmonicOptimizer` (`ml.training`) — Mercury optimizer with harmonic oscillator variant.
- `MercuryMomentumOptimizer` (`ml.training`) — Mercury optimizer with momentum variant.
- `MercuryOptimizer` (`ml.training`) — Base Mercury optimizer with state evolution dynamics.
- `ThreeRAnomalyTrainer` (`ml.training`) — PyTorch Lightning trainer for 3R Anomaly Transformer with Lyapunov stability.
- `Trainer` (`ml.training`) — General-purpose trainer for PyTorch models.
- `WeightOptimizer` (`ml.ensemble_coordinator`) — Abstract base class for weight optimization.

<details><summary>Support types (62)</summary>

`Activation`, `ActiveLearningState`, `AdaptationResult`, `AdaptationStrategy`, `AggregationMethod`, `BiasAuditConfig`, `BiasReport`, `CompressionConfig`, `CompressionMethod`, `ConceptDriftEvaluationResult`, `CorticalConfig`, `CorticalLayer`, `DegradationTrend`, `DetectorState`, `DomainAdaptationMethod`, `DomainData`, `DriftResult`, `DriftSeverity`, `DriftType`, `DualStudentConfig`, `EnsembleConfig`, `EnsembleMethod`, `EnsembleResult`, `EnsembleStrategy`, `EpisodeSamplingStrategy`, `EvolutionResult`, `ExplainabilityMethod`, `FairnessMetric`, `FairnessMetric`, `FairnessReport`, `FairnessResult`, `FewShotMethod`, `FewShotResult`, `FirmsYearData`, `FitnessReport`, `GenerationRecord`, `GlcEvent`, `HazardDataUnavailableError`, `HourRecord`, `LabelType`, `LayerParams`, `MLPConfig`, `MeritGateError`, `MetaLearningAlgorithm`, `MetaTrainingResult`, `MitigationStrategy`, `ModelProtocol`, `MutationConfig`, `NotFittedError`, `OptimizationConfig`, `OptimizationResult`, `QueryBatch`, `RetrainingEvent`, `RetrainingTrigger`, `RuleExplanation`, `SamplingStrategy`, `SecurityDataset`, `SymbolicExplanation`, `TemporalSplitStrategy`, `TrainingConfig`, `TransferResult`, `UpdateStrategy`

</details>

### `models/` — 141 classes (104 capability)

**Adapters & backends**

- `AnthropicCloudAdapter` (`models.foundation.ollama_adapter`) — Anthropic cloud adapter for Claude models.
- `BaseFoundationAdapter` (`models.foundation.base_foundation`) — Concrete adapter class for foundation models.
- `BaseLLMAdapter` (`models.foundation.llm_adapter`) — Abstract base class for LLM adapters.
- `ChronosAdapter` (`models.foundation.chronos_adapter`) — Chronos adapter for local time-series forecasting.
- `CohereCloudAdapter` (`models.foundation.ollama_adapter`) — Cohere Chat v2 cloud adapter (api.cohere.com).
- `CursorAdapter` (`models.foundation.ollama_adapter`) — Cursor cloud adapter (OpenAI-compatible).
- `DeepSeekAdapter` (`models.foundation.ollama_adapter`) — DeepSeek cloud adapter (api.deepseek.com, OpenAI-compatible).
- `GeminiCloudAdapter` (`models.foundation.ollama_adapter`) — Google Gemini cloud adapter (generativelanguage.googleapis.com).
- `HuggingFaceCloudAdapter` (`models.foundation.ollama_adapter`) — HuggingFace Inference API adapter.
- `HuggingFaceLLMAdapter` (`models.foundation.llm_adapter`) — HuggingFace Transformers LLM adapter for local models.
- `MockLLMAdapter` (`models.foundation.llm_adapter`) — Mock LLM adapter — hard-fails at construction.
- `OllamaLLMAdapter` (`models.foundation.ollama_adapter`) — Ollama LLM adapter for local model inference.
- `OpenAICloudAdapter` (`models.foundation.ollama_adapter`) — OpenAI cloud adapter for GPT models.
- `TemplateLLMAdapter` (`models.foundation.ollama_adapter`) — Template-based fallback adapter for offline operation.
- `TimeGPTAdapter` (`models.foundation.timegpt_adapter`) — TimeGPT adapter for time-series anomaly detection.
- `XAIGrokAdapter` (`models.foundation.ollama_adapter`) — xAI Grok cloud adapter (api.x.ai, OpenAI-compatible).
- `_OpenAICompatibleCloudAdapter` (`models.foundation.ollama_adapter`) — Shared base for OpenAI-compatible Chat Completions adapters.

**Analysis & scoring**

- `ConsciousnessFieldAnalyzer` (`models.parapsychology`) — Sequence model scoring REG windows for deviation-from-chance.
- `FourierAnalyzer` (`models.biometric`) — Fourier analysis for frequency-domain biometric features.
- `IsotopeRatioAnalyzer` (`models.isotope_predictor`) — Neural network for isotope ratio analysis.
- `NuclearForensicsAnalyzer` (`models.isotope_predictor`) — Nuclear forensics analysis for isotope attribution.
- `RadiologicalThreatAssessor` (`models.isotope_predictor`) — Radiological threat assessment.

**Detection**

- `AnomalyDetector` (`models.lstm_ae`) — Complete anomaly detection pipeline using LSTM-Autoencoder.
- `ChemistryAnomalyDetector` (`models.chemistry`) — Chemistry Discipline Anomaly Detector.
- `MatrixProfileDetector` (`models.foundation.matrix_profile`) — Matrix Profile-based anomaly detector using STUMPY.
- `ParapsychologyDetector` (`models.parapsychology`) — REG statistical-deviation detector (parapsychology-research tooling).
- `TextLogAnomalyDetector` (`models.foundation.llm_adapter`) — Specialized detector for text and log anomalies.
- `ZeroShotAnomalyDetector` (`models.foundation.llm_adapter`) — Zero-shot anomaly detector using LLM prompting.

**Engines & orchestration**

- `AdvancedBiometricEngine` (`models.biometric_advanced`) — Advanced biometric processing engine with neural-symbolic fusion.
- `AgeProgressionEngine` (`models.biometric_advanced`) — Age progression engine with quantum variant amplification.
- `EnhancedNeurosymbolicEngine` (`models.neurosymbolic_enhanced`) — Unified Enhanced Neurosymbolic Engine.
- `FallbackLLMChain` (`models.foundation.ollama_adapter`) — Graceful fallback chain for LLM operations.
- `MultiverseOmniEngine` (`models.multiverse`) — Multi-Hypothesis Optimization Engine - Parallel Solution Exploration.
- `NeurosymbolicEngine` (`models.neurosymbolic`) — Unified Neurosymbolic reasoning engine combining LTN with symbolic logic.
- `QuantumEngine` (`models.quantum_engine`) — Quantum computing engine with practical applications.

**Ethics & governance**

- `QuantumGate` (`models.quantum_engine`) — Collection of quantum gates for circuit operations.

**Neural models & layers**

- `AffectiveAnomalyModel` (`models.affective`) — Affective computing model for emotional state anomaly detection.
- `AnomalyTransformerEncoder` (`models.sota.association_discrepancy`) — Full Anomaly Transformer Encoder with Association Discrepancy.
- `AnomalyTransformerEncoderLayer` (`models.sota.association_discrepancy`) — Single encoder layer with Association Discrepancy attention.
- `AssociationDiscrepancyLoss` (`models.sota.association_discrepancy`) — Loss function for Anomaly Transformer with minimax strategy.
- `AssociationDiscrepancyModule` (`models.sota.association_discrepancy`) — Association Discrepancy computation module.
- `AstrophysicalAnomalyModel` (`models.astrophysical`) — Astrophysical anomaly detection using black hole physics and cosmic event modeling.
- `BaseFoundationModel` (`models.foundation.base_foundation`) — Abstract base class for foundation model adapters.
- `BiometricAnomalyModel` (`models.biometric`) — Biometric anomaly detection for facial recognition and analysis.
- `ConsciousnessPreservationModel` (`models.consciousness`) — Model for consciousness state preservation and anomaly detection.
- `Discriminator` (`models.sota.tranad`) — Discriminator for TranAD adversarial training.
- `EnhancedLogicTensorNetwork` (`models.neurosymbolic_enhanced`) — Enhanced Logic Tensor Network with multiple fuzzy semantics.
- `FocusScoreConditioning` (`models.sota.tranad`) — Focus Score-Based Self-Conditioning Module.
- `GatedFeatureFusion` (`models.sota.maat`) — Gated Feature Fusion for combining attention and SSM pathways.
- `LSTMAutoencoder` (`models.lstm_ae`) — LSTM-based Autoencoder for time-series anomaly detection.
- `LogicTensorNetwork` (`models.neurosymbolic`) — Neuro-symbolic inference head backed by the canonical co-trained module.
- `MAATEncoderLayer` (`models.sota.maat`) — Single MAAT Encoder Layer.
- `MAATLoss` (`models.sota.maat`) — Loss function for MAAT training.
- `MAATModel` (`models.sota.maat`) — MAAT: Mamba Adaptive Anomaly Transformer.
- `MambaSSM` (`models.sota.maat`) — Mamba-SSM Block for MAAT.
- `MetaCognitionLayer` (`models.neurosymbolic_enhanced`) — Meta-cognition layer for self-monitoring reasoning.
- `NeuralCognitiveModel` (`models.neural`) — Neural cognitive model for brain activity anomaly detection.
- `NoiseModel` (`models.quantum`) — Noise model for quantum decoherence simulation.
- `PeriodicTableEncoder` (`models.chemistry`) — Neural network encoder for periodic table relationships.
- `PositionalEncoding` (`models.sota.association_discrepancy`) — Sinusoidal positional encoding for sequence position information.
- `PositionalEncoding` (`models.sota.maat`) — Sinusoidal positional encoding.
- `PositionalEncoding` (`models.sota.tranad`) — Sinusoidal positional encoding.
- `PriorAssociation` (`models.sota.association_discrepancy`) — Prior-Association Distribution based on temporal proximity.
- `ProbabilisticLogicLayer` (`models.neurosymbolic_enhanced`) — Probabilistic logic for handling uncertainty.
- `QuantumAnomalyModel` (`models.quantum`) — Quantum-inspired anomaly detection using quantum state representations.
- `SelectiveSSM` (`models.sota.maat`) — Selective State Space Model (S6) approximation.
- `SeriesAssociation` (`models.sota.association_discrepancy`) — Series-Association via learned multi-head self-attention.
- `SparseAttention` (`models.sota.maat`) — Sparse Attention Module for efficient long-sequence processing.
- `SymbolicReasoningLayer` (`models.neurosymbolic`) — Symbolic reasoning layer for explainable AI.
- `TranADLoss` (`models.sota.tranad`) — Combined loss function for TranAD training.
- `TranADModel` (`models.sota.tranad`) — TranAD: Deep Transformer Networks for Anomaly Detection.
- `TransformerDecoder` (`models.sota.tranad`) — Transformer Decoder for TranAD reconstruction.
- `TransformerEncoder` (`models.sota.tranad`) — Transformer Encoder for TranAD.

**Other capability classes**

- `AnomalyPrompt` (`models.foundation.llm_adapter`) — Structured anomaly detection prompt.
- `BiometricFusion` (`models.biometric_advanced`) — Transformer-based neural-symbolic fusion for biometric matching.
- `CausalEdge` (`models.neurosymbolic_enhanced`) — A causal relationship between variables.
- `CausalReasoningModule` (`models.neurosymbolic_enhanced`) — Causal reasoning for anomaly detection.
- `CommonsenseRelation` (`models.neurosymbolic_enhanced`) — A commonsense knowledge relation.
- `FoundationEnsemble` (`models.foundation.ensemble`) — Ensemble of foundation models for robust anomaly detection.
- `FuzzyOperators` (`models.neurosymbolic_enhanced`) — Differentiable fuzzy logic operators.
- `GraphEdge` (`models.neurosymbolic_enhanced`) — Edge in a temporal knowledge graph.
- `GraphNode` (`models.neurosymbolic_enhanced`) — Node in a temporal knowledge graph.
- `HarmonicDecomposer` (`models.biometric`) — Simple harmonic decomposition using FFT for biometric feature analysis.
- `KnowledgeGraphBridge` (`models.neurosymbolic_enhanced`) — Bridge to external knowledge graphs for commonsense reasoning.
- `LLMModelRegistry` (`models.llm_registry`) — Instance-owned registry of selectable LLM model specs.
- `LLMModelSpec` (`models.llm_registry`) — Operator-declared facts about one selectable model.
- `LLMUsage` (`models.foundation.llm_usage`) — One provider-reported usage record for a single generation call.
- `ModelConfiguration` (`models.foundation.ollama_adapter`) — Configuration for model selection and swapping.
- `ModelProfile` (`models.foundation.ollama_adapter`) — Profile for a specific model's capabilities.
- `ProviderFacts` (`models.llm_registry`) — Code-grounded facts about one shipped provider adapter.
- `PsiPhenomenon` (`models.parapsychology`) — Types of psi phenomena.
- `QuantumAgeVariant` (`models.biometric_advanced`) — Quantum variant for age progression uncertainty modeling.
- `QuantumCircuit` (`models.quantum_engine`) — Quantum circuit simulator with state vector representation.
- `SOTARegistry` (`models.sota.registry`) — Registry for state-of-the-art anomaly detection models.
- `SimulationModule` (`models.simulation`) — Mathematical simulation for paradoxes, conjectures, and theoretical problems.
- `SymbolicRule` (`models.neurosymbolic`) — Represents a symbolic logical rule with explainability support.
- `TemporalGraphReasoner` (`models.neurosymbolic_enhanced`) — PyReason-inspired temporal graph reasoner.
- `TemporalRule` (`models.neurosymbolic_enhanced`) — Temporal logic rule with time constraints.
- `Universe` (`models.multiverse`) — Represents a parallel universe (solution pathway).
- `UsageLedger` (`models.foundation.llm_usage`) — Thread-safe accumulator of :class:`LLMUsage` records.
- `_Aggregate` (`models.foundation.llm_usage`) — Running counters for one ``(provider, model)`` key.

**Prediction & forecasting**

- `IsotopePredictor` (`models.isotope_predictor`) — Comprehensive isotope prediction and nuclear forensics system.

**Training & optimization**

- `AdversarialTrainer` (`models.sota.tranad`) — Adversarial Training for TranAD.
- `MAMLOptimizer` (`models.sota.tranad`) — Model-Agnostic Meta-Learning (MAML) for TranAD.

<details><summary>Support types (37)</summary>

`AgeProgressionResult`, `AnnealingResult`, `AssociationConfig`, `BiometricResult`, `ChemicalAnomalyResult`, `ChronosConfig`, `DecoherenceConfig`, `ElementGroup`, `EnsembleConfig`, `ErrorCorrectionCode`, `ForecastResult`, `FoundationModelConfig`, `FuzzySemantics`, `GroverSearchResult`, `IngestResult`, `IsotopePredictionResult`, `IsotopeType`, `LLMAnomalyResult`, `LLMConfig`, `LLMProvider`, `MAATConfig`, `MatchCategory`, `MatrixProfileConfig`, `ModelInfo`, `OllamaConfig`, `OllamaModel`, `ParapsychologyResult`, `QKDResult`, `QuantumState`, `ReasoningMode`, `ReasoningResult`, `ReasoningState`, `ThreatLevel`, `TimeGPTConfig`, `TranADConfig`, `UniverseState`, `_EgressKwargs`

</details>

### `narrative/` — 51 classes (27 capability)

**Data sources & loaders**

- `BaseExternalRetriever` (`narrative.external_retrieval`) — Abstract base class for external retrievers.
- `DatabaseRetriever` (`narrative.external_retrieval`) — Database retriever for querying local databases.
- `ExternalInformationRetriever` (`narrative.external_retrieval`) — Unified External Information Retrieval for Mercury Agent.
- `KnowledgeRetriever` (`narrative.retriever`) — Unified Knowledge Search for Mercury Agent.
- `WebSearchRetriever` (`narrative.external_retrieval`) — Web search retriever using DuckDuckGo or SearXNG.

**Engines & orchestration**

- `NarrativeEngine` (`narrative.engine`) — Truth-Dense Communication Synthesis Engine.
- `PersonalityEngine` (`narrative.personality`) — Shapes Communication Using Omni-Scalars.

**Monitoring**

- `ProactiveMonitor` (`narrative.proactive`) — Background Vigilance with Initiative Thresholds.

**Other capability classes**

- `AudioSegment` (`narrative.multimodal`) — Segment of interest in audio data.
- `CommunicationModifiers` (`narrative.personality`) — Modifiers for communication generation.
- `ConversationContext` (`narrative.interface`) — Context for a conversation session.
- `ConversationTurn` (`narrative.voice`) — A single turn in conversation.
- `InitiativeThreshold` (`narrative.proactive`) — Configuration for when to take initiative.
- `MemoryContext` (`narrative.memory_surface`) — Complete memory context for a detection.
- `MemorySurface` (`narrative.memory_surface`) — Surfaces Memory Context in Communications.
- `MercuryConversationInterface` (`narrative.interface`) — Unified Conversation Interface - Making Mercury "Alive".
- `MercuryVoice` (`narrative.voice`) — Mercury's Voice - True Conversational Interface.
- `MultiModalDetection` (`narrative.multimodal`) — Detection result from multi-modal analysis.
- `MultiModalNarration` (`narrative.multimodal`) — Narration of multi-modal detection.
- `MultiModalNarrator` (`narrative.multimodal`) — Narrator for multi-modal detection results.
- `PatternAccumulator` (`narrative.proactive`) — Tracks pattern accumulation for escalation.
- `PersonalityProfile` (`narrative.personality`) — Current personality configuration derived from scalars.
- `PredictionHistory` (`narrative.memory_surface`) — Historical prediction accuracy for a pattern type.
- `ReasoningChainNarrative` (`narrative.engine`) — Verbalized reasoning chain for transparency.
- `RegionOfInterest` (`narrative.multimodal`) — Region of interest in visual data.
- `ResultCache` (`narrative.external_retrieval`) — Cache for external retrieval results.
- `SearchContext` (`narrative.retriever`) — Context for a search operation.

<details><summary>Support types (24)</summary>

`AnomalyVisualType`, `AudioAnomalyType`, `CommunicationTone`, `ConfidenceLevel`, `ConversationType`, `ExternalResult`, `ExternalSearchConfig`, `ExternalSourceType`, `InitiativeEvent`, `InitiativeType`, `MemoryRelevance`, `MercuryResponse`, `ModalityType`, `NarrativeResult`, `NarrativeStyle`, `QueryIntent`, `RetrievalResult`, `RetrievalSource`, `SearchResponse`, `SimilarEvent`, `VerbosityLevel`, `VigilanceLevel`, `VoiceResponse`, `WebSearchProvider`

</details>

### `ocean/` — 3 classes (2 capability)

**Other capability classes**

- `OceanographyPatterns` (`ocean.oceanography_patterns`) — Pattern recognition system inspired by oceanographic methods.
- `WavePattern` (`ocean.oceanography_patterns`) — Represents a wave pattern in data.

<details><summary>Support types (1)</summary>

`DepthLevel`

</details>

### `quantum_computing/` — 27 classes (18 capability)

**Adapters & backends**

- `IBMQuantumBackend` (`quantum_computing.executor`) — IBM Quantum hardware backend.
- `SimulatorBackend` (`quantum_computing.executor`) — Local quantum simulator backend.

**Detection**

- `QAOAAnomalyDetector` (`quantum_computing.hybrid`) — Quantum Approximate Optimization Algorithm for anomaly detection.
- `QuantumAnomalyDetector` (`quantum_computing.detector`) — Quantum-enhanced anomaly detection.
- `VQEAnomalyDetector` (`quantum_computing.hybrid`) — Variational Quantum Eigensolver for anomaly detection.

**Other capability classes**

- `AnomalyEncodingCircuit` (`quantum_computing.circuits`) — Encode classical data into quantum states for anomaly detection.
- `BatchExecutor` (`quantum_computing.executor`) — Execute multiple circuits in batches for efficiency.
- `ErrorMitigationCircuit` (`quantum_computing.circuits`) — Error mitigation techniques for NISQ devices.
- `QuantumCircuitBuilder` (`quantum_computing.circuits`) — Build quantum circuits for anomaly detection.
- `QuantumExecutor` (`quantum_computing.executor`) — Unified quantum execution interface.
- `QuantumFeatureMap` (`quantum_computing.circuits`) — Quantum feature map for kernel-based learning.
- `QuantumJob` (`quantum_computing.executor`) — Manages a quantum job submitted for execution.
- `QuantumKernel` (`quantum_computing.hybrid`) — Quantum kernel for kernel-based machine learning.
- `QuantumResourceEstimate` (`quantum_computing.detector`) — Estimate of quantum resources required.
- `SimulatedQuantumCircuit` (`quantum_computing.circuits`) — Simulated quantum circuit for when Qiskit is not available.
- `VariationalCircuit` (`quantum_computing.circuits`) — Variational quantum circuit for trainable quantum models.

**Training & optimization**

- `ClassicalOptimizer` (`quantum_computing.hybrid`) — Classical optimizer for variational parameter updates.
- `HybridOptimizer` (`quantum_computing.hybrid`) — Hybrid quantum-classical optimizer.

<details><summary>Support types (9)</summary>

`BackendConfig`, `BackendType`, `CircuitMetadata`, `EncodingType`, `ExecutionResult`, `JobStatus`, `OptimizationResult`, `QuantumDetectionResult`, `VariationalAnsatz`

</details>

### `reasoning/` — 10 classes (8 capability)

**Adapters & backends**

- `LocalReasoningBackend` (`reasoning.backends`) — Offline-first reasoning over Mercury's local LLM chain.
- `MockReasoningBackend` (`reasoning.backends`) — Deterministic, offline, network-free backend for tests and CI.
- `ReasoningBackend` (`reasoning.backend`) — Abstract reasoning engine Mercury calls; never the front of the system.
- `RemoteReasoningBackend` (`reasoning.backends`) — Network-capable reasoning over an operator-declared frontier model.

**Engines & orchestration**

- `ReasoningRouter` (`reasoning.router`) — Select and delegate to a reasoning backend under an offline-first policy.

**Other capability classes**

- `Explanation` (`reasoning.schemas`) — A natural-language explanation of a Mercury finding.
- `Hypothesis` (`reasoning.schemas`) — A single proposed explanation for the cognitive engine to weigh.
- `ReasoningContext` (`reasoning.schemas`) — A request for reasoning, framed in Mercury's own terms.

<details><summary>Support types (2)</summary>

`ReasoningBackendUnavailableError`, `Report`

</details>

### `resilience/` — 10 classes (10 capability)

**Engines & orchestration**

- `AdaptiveDefenseSystem` (`resilience.self_healing`) — CRISPR-inspired adaptive defense system for anomaly pattern memory.
- `SelfHealingEngine` (`resilience.self_healing`) — Unified self-healing system for autonomous error recovery.

**Monitoring**

- `HealthMonitor` (`resilience.health_monitoring`) — Monitor health of components and agents.

**Other capability classes**

- `AnomalySignature` (`resilience.self_healing`) — Compact representation of anomaly pattern (analogous to CRISPR spacer).
- `DataLoaderCircuitBreaker` (`resilience.api_circuit_breakers`) — Circuit breaker optimized for data loader API calls.
- `DetectorCircuitBreaker` (`resilience.api_circuit_breakers`) — Circuit breaker optimized for detector invocations.
- `ExternalIntegrationCircuitBreaker` (`resilience.api_circuit_breakers`) — Circuit breaker optimized for external integration endpoints.
- `HealthMetrics` (`resilience.health_monitoring`) — Health metrics for a component.
- `RetryPolicy` (`resilience.retry`) — Retry policy with exponential backoff.
- `_CircuitBreakerWrapper` (`resilience.api_circuit_breakers`) — Type stub for wrapped functions with circuit_breaker attribute.

### `safeguards/` — 3 classes (2 capability)

**Analysis & scoring**

- `ResonanceAnalyzer` (`safeguards.nano_safeguards`) — FFT-based resonance analysis for frequency-domain micro-anomalies.

**Detection**

- `NanoSafeguardDetector` (`safeguards.nano_safeguards`) — Nano-Safeguard Detector for Micro-Anomaly Detection.

<details><summary>Support types (1)</summary>

`NanoSafeguardResult`

</details>

### `scaling/` — 14 classes (10 capability)

**Other capability classes**

- `AsyncProcessor` (`scaling.distributed_processor`) — Async processor for non-blocking anomaly detection.
- `BainAIScaling` (`scaling.bain_ai_scaling`) — AI scaling and compute optimization inspired by Bain 2025 report.
- `ChunkGenerator` (`scaling.distributed_processor`) — Generator for creating data chunks for distributed processing.
- `ComputeResource` (`scaling.bain_ai_scaling`) — Represents compute resources for AI operations.
- `DistributedProcessor` (`scaling.distributed_processor`) — Distributed processor for large-scale anomaly detection.
- `ProcessWorkerPool` (`scaling.distributed_processor`) — Process-based worker pool for CPU-bound tasks.
- `ProcessingStats` (`scaling.distributed_processor`) — Statistics for processing operation.
- `StreamProcessor` (`scaling.distributed_processor`) — Stream processor for real-time anomaly detection.
- `ThreadWorkerPool` (`scaling.distributed_processor`) — Thread-based worker pool for I/O-bound tasks.
- `WorkerPool` (`scaling.distributed_processor`) — Abstract base class for worker pools.

<details><summary>Support types (4)</summary>

`ChunkResult`, `LoadBalancer`, `ProcessingConfig`, `ProcessingStrategy`

</details>

### `security/` — 159 classes (92 capability)

**Adapters & backends**

- `InMemoryBackend` (`security.rate_limiting`) — Thread-safe in-memory rate limit backend.
- `RateLimitBackend` (`security.rate_limiting`) — Protocol for rate limit storage backends (e.g., Redis).
- `SafeHTTPClient` (`security.safe_http`) — Centralised outbound HTTP gate.
- `_PinnedDNSHTTPAdapter` (`security.safe_http`) — Lazy proxy for the real requests-based HTTPAdapter.

**Analysis & scoring**

- `CommunicationGraphAnalyzer` (`security.traffic_analysis`) — Graph neural network for communication pattern analysis.
- `MalwareTaxonomyClassifier` (`security.cybint_subprocessor`) — Malware family classification network.
- `NetworkFlowAnalyzer` (`security.traffic_analysis`) — Network flow statistical analysis.
- `PSYOPAnalyzer` (`security.psyop`) — Psychological Operations Analysis Engine.
- `RFSpectrumAnalyzer` (`security.tempest_detection`) — RF spectrum analysis for EM emanation detection.
- `SideChannelVulnerabilityAssessor` (`security.tempest_detection`) — Side-channel vulnerability assessment.
- `ZeroDayIndicatorAnalyzer` (`security.cybint_subprocessor`) — Zero-day exploitation indicator analysis.

**Biometric & recognition**

- `APTPatternRecognizer` (`security.cybint_subprocessor`) — Neural network for APT group attribution.

**Data sources & loaders**

- `SafeHFLoader` (`security.model_policy`) — The single from_pretrained / load_dataset entry point.

**Detection**

- `AdaptiveThreatDetector` (`security.realtime_threat_detection`) — Adaptive threat detector that updates based on new threats.
- `C2InfrastructureDetector` (`security.cybint_subprocessor`) — Command & Control infrastructure detection.
- `CovertChannelDetector` (`security.traffic_analysis`) — Covert channel detection in network traffic.
- `RealTimeThreatDetector` (`security.realtime_threat_detection`) — Real-time threat detection using Mercury-native ensemble anomaly detection.
- `TEMPESTDetector` (`security.tempest_detection`) — Comprehensive TEMPEST detection system integrating RF spectrum analysis, video emanation.
- `TerrorismPatternDetector` (`security.anti_terrorism.pattern_recognition`) — Terrorism Pattern Detector for CI.
- `ThreatDetector` (`security.threat_detection`) — Detect common security threats:.
- `VideoEmanationDetector` (`security.tempest_detection`) — Neural network for video display emanation detection.
- `_LocalDensityDetector` (`security.realtime_threat_detection`) — KDTree-based local density anomaly detector (LOF-style, no sklearn).
- `_RandomProjectionDetector` (`security.realtime_threat_detection`) — Isolation-style anomaly detector using random projections (no trees).
- `_RobustCovarianceDetector` (`security.realtime_threat_detection`) — Mahalanobis-distance detector with robust covariance (no sklearn).

**Engines & orchestration**

- `IntelligenceFusionEngine` (`security.intelligence_fusion`) — All-Source Intelligence Fusion Engine.
- `SecureHashChain` (`security.secure_audit_logging`) — Cryptographically secure hash chain for audit log integrity.
- `TrafficAnalysisEngine` (`security.traffic_analysis`) — Comprehensive traffic analysis engine integrating flow analysis, graph-based detection,.

**Ethics & governance**

- `SigmaImmutableGate` (`security.sigma_immutable_gate`) — Process-wide singleton enforcing σ_Immutable at every boundary.

**Monitoring**

- `RngHealthMonitor` (`security.rng_health`) — Advisory health monitor for raw RNG output streams.

**Neural models & layers**

- `AllSourceFusionNetwork` (`security.intelligence_fusion`) — Neural network for all-source intelligence fusion.

**Other capability classes**

- `Baseline` (`security.sigma_immutable_corpus`) — The harvested intact-config reference the corpus is built around.
- `COMINTProcessor` (`security.int_sources`) — Communications Intelligence (COMINT) Processor.
- `CYBINTProcessor` (`security.int_sources`) — Cyber Intelligence (CYBINT) Processor.
- `CYBINTSubProcessor` (`security.cybint_subprocessor`) — Comprehensive CYBINT sub-processor for detailed cyber threat analysis.
- `CorpusBundle` (`security.sigma_immutable_corpus`) — Materialised corpus + provenance metadata.
- `CryptanalysisProcessor` (`security.int_sources`) — Cryptanalysis Processor.
- `CryptoAuditTrail` (`security.pqc_backends`) — Cryptographic audit trail for PQC operations.
- `CryptoOperation` (`security.pqc_backends`) — Audit record for cryptographic operations.
- `CyberFortress` (`security.cyber_fortress`) — Unified Cyber Fortress for proactive threat elimination.
- `DilithiumKeyPair` (`security.pqc_backends`) — ML-DSA-65 (Dilithium) key pair.
- `ELINTProcessor` (`security.int_sources`) — Electronic Intelligence (ELINT) Processor.
- `EMSECCountermeasureGenerator` (`security.tempest_detection`) — Generate EMSEC countermeasures and mitigation strategies.
- `Ed25519Provider` (`security.crypto_api`) — Ed25519 classical signature provider.
- `EncapsulatedSecret` (`security.crypto_api`) — Key encapsulation result.
- `EncryptedTrafficFingerprinter` (`security.traffic_analysis`) — Encrypted traffic fingerprinting (JA3/JA4-style analysis).
- `FININTProcessor` (`security.int_sources`) — Financial Intelligence (FININT) Processor.
- `GEOINTProcessor` (`security.int_sources`) — Geospatial Intelligence (GEOINT) Processor.
- `HFModelPolicy` (`security.model_policy`) — Static policy gate for HuggingFace model / dataset loads.
- `HUMINTProcessor` (`security.int_sources`) — Human Intelligence (HUMINT) Processor.
- `HiveFirewall` (`security.hive_firewall`) — HCIS-inspired hive-structured firewall.
- `HiveNode` (`security.hive_firewall`) — Individual node in hive firewall.
- `HybridSignature` (`security.crypto_api`) — Hybrid signature combining classical and post-quantum.
- `HybridSignatureProvider` (`security.crypto_api`) — Hybrid signature provider combining classical and post-quantum.
- `IMINTProcessor` (`security.int_sources`) — Imagery Intelligence (IMINT) Processor.
- `InfluenceCampaignDetection` (`security.psyop`) — Detection results for influence campaign analysis.
- `InputValidator` (`security.input_validation`) — Production-grade input validator.
- `IntelligenceSourceRegistry` (`security.int_sources`) — Registry for all intelligence source processors.
- `KeyPair` (`security.crypto_api`) — Generic key pair container.
- `KyberEncapsulation` (`security.pqc_backends`) — Kyber key encapsulation result.
- `KyberKeyPair` (`security.pqc_backends`) — Kyber-1024 key pair.
- `KyberProvider` (`security.crypto_api`) — Kyber-1024 key encapsulation provider — delegates to AMA.
- `MASINTProcessor` (`security.int_sources`) — Measurement & Signature Intelligence (MASINT) Processor.
- `MLDSAProvider` (`security.crypto_api`) — ML-DSA-65 (Dilithium) post-quantum signature provider — delegates to AMA.
- `MercuryCrypto` (`security.crypto_api`) — Unified cryptographic interface for Mercury Agent.
- `MeteorologicalProcessor` (`security.int_sources`) — Meteorological Intelligence Processor.
- `NarrativeAnalysis` (`security.psyop`) — Analysis of a narrative or message.
- `OSINTProcessor` (`security.int_sources`) — Open Source Intelligence (OSINT) Processor.
- `OverwatchNexus` (`security.counterintelligence`) — Overwatch Nexus and Response Engine for Ethical Counterintelligence.
- `PIIMasker` (`security.secure_audit_logging`) — Masks Personally Identifiable Information in audit logs.
- `PQCProductionWarning` (`security.pqc_guards`) — Public exception type for PQC-availability warnings.
- `PostQuantumMigrationPlanner` (`security.quantum_risk_cyber`) — Post-quantum cryptography migration planning system.
- `QuantumResistantEncryption` (`security.encryption`) — Post-quantum hybrid encryption: ML-KEM-1024 + AES-256-GCM (via AMA).
- `QuantumRiskCyber` (`security.quantum_risk_cyber`) — Quantum cybersecurity risk management system.
- `QuantumThreat` (`security.quantum_risk_cyber`) — Represents a quantum cybersecurity threat.
- `RateLimiter` (`security.rate_limiting`) — Unified rate limiter with token bucket and sliding window algorithms.
- `ResonanceHashIntegrityChecker` (`security.cyber_fortress`) — Novel hash integrity checking using resonance amplification.
- `SIGINTProcessor` (`security.int_sources`) — Signals Intelligence (SIGINT) Processor.
- `SecureAnomalyChecker` (`security.constant_time`) — Timing-safe anomaly checking for security-critical applications.
- `SecureAuditLogger` (`security.secure_audit_logging`) — Production-grade secure audit logging system.
- `SecureDataHandler` (`security.encryption`) — Handle sensitive data securely with post-quantum encryption options.
- `SigmaCalibrationPoint` (`security.sigma_calibration`) — Discrimination + calibration of the σ gate at one (temperature, threshold).
- `SigmaImmutableEvaluation` (`security.sigma_immutable_gate`) — Outcome of a single σ_Immutable evaluation.
- `Signature` (`security.crypto_api`) — Digital signature with metadata.
- `SlhDsaKeyPair` (`security.pqc_backends`) — FIPS 205 SLH-DSA key pair (parameter-driven).
- `SphincsKeyPair` (`security.pqc_backends`) — SPHINCS+-256f key pair.
- `SphincsProvider` (`security.crypto_api`) — SPHINCS+ hash-based signature provider — delegates to AMA.
- `TargetAudienceProfile` (`security.psyop`) — Profile of a target audience for PSYOP analysis.
- `ThreatBlocking` (`security.hive_firewall`) — Threat blocking decision with reasoning.
- `ThreatSignature` (`security.realtime_threat_detection`) — Threat signature with metadata.
- `TrafficAnalysisProcessor` (`security.int_sources`) — Traffic Analysis Processor.
- `TrustedEndpoints` (`security.input_validation`) — Hardcoded trusted API endpoints for external data sources with secure URL opening.

**Solvers & scorers**

- `MultiverseZeroDaySimulator` (`security.cyber_fortress`) — Novel zero-day attack simulation using multiverse optimization.

<details><summary>Support types (67)</summary>

`APTGroup`, `AlgorithmType`, `AuditEvent`, `AuditEventCategory`, `AuditEventSeverity`, `BanishmentAction`, `COMINTAnalysisResult`, `CYBINTAnalysisResult`, `CYBINTAnalysisResult`, `CognitiveBias`, `CorpusVerificationError`, `CryptanalysisResult`, `CryptoBackend`, `CryptoPackageConfig`, `CryptoPackageResult`, `CryptoSystem`, `CyberKillChainStage`, `DecodeError`, `ELINTAnalysisResult`, `EmanationType`, `ExpiredSignatureError`, `FININTAnalysisResult`, `FortressResult`, `GEOINTAnalysisResult`, `HUMINTAnalysisResult`, `IMINTAnalysisResult`, `ImmatureSignatureError`, `InfluenceVector`, `InformationEnvironmentState`, `IntelligenceDiscipline`, `IntelligenceFusionResult`, `IntelligenceProcessor`, `InvalidAlgorithmError`, `InvalidSignatureError`, `InvalidTokenError`, `MASINTAnalysisResult`, `MalwareFamily`, `MeteorologicalIntelResult`, `MissingRequiredClaimError`, `NarrativeType`, `NativeJWTError`, `OSINTAnalysisResult`, `OverwatchNexusResult`, `PQCBackend`, `PSYOPCategory`, `RateLimitAlgorithm`, `RateLimitInfo`, `RngHealthReport`, `RngHealthVerdict`, `SIGINTAnalysisResult`, `SanitizationLevel`, `SecurityLevel`, `TEMPESTAnalysisResult`, `TEMPESTThreatLevel`, `TerrorismThreatResult`, `ThreatActorType`, `ThreatLevel`, `ThreatLevel`, `TrafficAnalysisResult`, `TrafficAnalysisResult`, `TrafficAnomalyType`, `UnsafeModelError`, `UnsafePayloadError`, `UnsafeSubprocessError`, `UnsafeURLError`, `ValidationError`, `ValidationResult`

</details>

### `space/` — 42 classes (30 capability)

**Analysis & scoring**

- `EarthquakePrecursorAnalyzer` (`space.disaster_precursor_detector`) — Regional seismicity-rate forecaster over catalog statistics.
- `InterstellarObjectAnalyzer` (`space.interstellar_objects`) — Neural network for interstellar object anomaly analysis.
- `SchumannHarmonicAnalyzer` (`space.schumann_resonance`) — Neural network for Schumann harmonic pattern analysis.
- `SpaceExplorationAnalyzer` (`space.space_exploration_analyzer`) — Hubble-inspired analyzer for cosmic anomalies and orbital threats.

**Detection**

- `CMEArrivalDetector` (`space.cme_arrival_detector`) — Predicts Earth arrival windows for CMEs from DONKI-style kinematics.
- `DisasterPrecursorDetector` (`space.disaster_precursor_detector`) — Comprehensive disaster precursor detection system.
- `GICDetector` (`space.gic_detector`) — dB/dt + plane-wave geoelectric GIC risk detector for grid operators.
- `InterstellarObjectDetector` (`space.interstellar_objects`) — Interstellar Object Anomaly Detector.
- `IonosphericDisturbanceDetector` (`space.disaster_precursor_detector`) — Detect ionospheric disturbances from Schumann data.
- `IonosphericScintillationDetector` (`space.ionospheric_scintillation_detector`) — S4 / sigma-phi measurement plus climatological GNSS-degradation risk.
- `SEPStormDetector` (`space.sep_storm_detector`) — NOAA S-scale SEP storm detector over GOES integral proton flux.
- `SchumannResonanceDetector` (`space.schumann_resonance`) — Schumann Resonance Anomaly Detector.
- `SolarFlareDetector` (`space.solar_storm_detector`) — Canonical solar-flare detector: GOES X-ray flux classification.
- `SolarGICCascadeDetector` (`space.solar_gic_cascade`) — Staged flare/CME -> Kp -> dB/dt escalation state machine.
- `SolarStormDetector` (`space.solar_storm_detector`) — Comprehensive solar and geomagnetic storm detection system.

**Monitoring**

- `CMETracker` (`space.solar_storm_detector`) — Coronal Mass Ejection tracking and arrival prediction.

**Other capability classes**

- `CMEArrivalPrediction` (`space.cme_arrival_detector`) — Arrival-time window and Earth-directedness assessment for one CME.
- `CMEKinematics` (`space.cme_arrival_detector`) — Kinematic inputs for a single CME arrival prediction.
- `CascadeAssessment` (`space.solar_gic_cascade`) — Result of a cascade evaluation.
- `CascadeInputs` (`space.solar_gic_cascade`) — Real upstream observations for one cascade evaluation.
- `EventWindow` (`space.schumann_labeling`) — A positive (anomaly-expected) interval coincident with a driver.
- `GICAssessment` (`space.gic_detector`) — GIC risk assessment for one observatory / magnetometer series.
- `GeomageticCorrelator` (`space.disaster_precursor_detector`) — Correlate Schumann anomalies with geomagnetic indices.
- `LabelCatalog` (`space.schumann_labeling`) — Weak-label catalog + full provenance.
- `SEPStormAssessment` (`space.sep_storm_detector`) — Result of an SEP / radiation-storm assessment.
- `ScintillationMeasurement` (`space.ionospheric_scintillation_detector`) — A real scintillation measurement from receiver samples.
- `ScintillationRisk` (`space.ionospheric_scintillation_detector`) — A climatological scintillation-occurrence risk (NOT a measurement).
- `SeismicCorrelator` (`space.disaster_precursor_detector`) — Correlate electromagnetic anomalies with seismic activity.
- `SpaceInspiredResilience` (`space.space_inspired`) — Resilience mechanisms inspired by space technology.
- `StageEvidence` (`space.solar_gic_cascade`) — Evidence record for one cascade stage.

<details><summary>Support types (12)</summary>

`CascadeStage`, `DisasterPrecursorResult`, `GeostormScale`, `ISOAnomalyType`, `InterstellarObjectResult`, `NaturalExplanationConfidence`, `RedundancyConfig`, `SchumannAnomalyResult`, `SolarFlareClass`, `SolarFlarePredictionResult`, `SolarStormPredictionResult`, `SystemState`

</details>

### `streaming/` — 1 classes (1 capability)

**Detection**

- `StreamingDetector` (`streaming.streaming_detector`) — Rolling-window anomaly detector for streaming data.

### `tools/` — 6 classes (4 capability)

**Other capability classes**

- `Certificate` (`tools._base`) — A signed-or-unsigned JSON evidence record produced by a tool.
- `DependencyMissing` (`tools._base`) — Raised by a tool's collector when a required runtime is absent.
- `_MetricsHandler` (`tools.prometheus_metrics_exporter`)
- `_Registry` (`tools.__init__`) — Lazy mapping of tool-name → ``main`` callable.

<details><summary>Support types (2)</summary>

`EnvelopeValidationError`, `_SentinelError`

</details>

### `utils/` — 44 classes (29 capability)

**Engines & orchestration**

- `ReportManager` (`utils.report_generator`) — Unified report management system.
- `ThreadSafeRNGManager` (`utils.rng`) — Thread-safe global RNG manager using thread-local storage.

**Other capability classes**

- `AsyncMessageQueue` (`utils.comm`) — Asynchronous message queue for inter-process communication Useful for distributed anomaly.
- `Bulkhead` (`utils.resilience`) — Bulkhead pattern for resource isolation.
- `CacheEntry` (`utils.feature_cache`) — Single cache entry with metadata.
- `CircuitBreaker` (`utils.resilience`) — Circuit breaker pattern implementation.
- `ColoredFormatter` (`utils.logging`) — Colored console formatter for development.
- `DeterministicRNG` (`utils.rng`) — Centralized random number generator for test determinism.
- `EmailReportSender` (`utils.report_generator`) — Email report sender (requires smtplib).
- `ExecutiveSummary` (`utils.report_generator`) — Executive summary for reports.
- `GracefulShutdown` (`utils.resilience`) — Handler for graceful application shutdown.
- `HealthChecker` (`utils.resilience`) — Health check manager for service health monitoring.
- `IncrementalFeatureComputer` (`utils.feature_cache`) — Incremental feature computation for efficient updates.
- `MathConstant` (`utils.constants`) — A mathematical constant with metadata.
- `MathematicalConstants` (`utils.constants`) — Centralized repository of mathematical constants.
- `MemoryEfficientFeatureCache` (`utils.feature_cache`) — Memory-efficient LRU cache for feature vectors.
- `OmniCode` (`utils.constants`) — An Omni-Code with helical parameters for ethical AI alignment.
- `OmniCodes` (`utils.constants`) — Seven foundational Omni-Codes governing Mercury Agent.
- `PDFReportGenerator` (`utils.report_generator`) — PDF report generation (requires reportlab).
- `PerformanceBenchmark` (`utils.profiling`) — Context manager for benchmarking an arbitrary code block.
- `PerformanceLogger` (`utils.logging`) — Logger for performance metrics.
- `PlainEnglishReportGenerator` (`utils.report_generator`) — Generate plain English reports from analysis results.
- `RNGContext` (`utils.rng`) — Hierarchical RNG context manager for scoped state isolation.
- `RNGRegistry` (`utils.rng`) — Registry pattern for named RNG generators.
- `ReportGenerator` (`utils.report_generator`) — General-purpose report generator with multiple format support.
- `ReportSection` (`utils.report_generator`) — Report section with optional subsections.
- `SimplePubSub` (`utils.comm`) — Simple publish-subscribe pattern for event-driven communication Useful for broadcasting.
- `StructuredFormatter` (`utils.logging`) — JSON formatter for structured logging.
- `TechnicalDetails` (`utils.report_generator`) — Technical details for reports.

<details><summary>Support types (15)</summary>

`AnomalyReport`, `BulkheadFullError`, `CacheConfig`, `CircuitBreakerConfig`, `CircuitBreakerOpenError`, `HealthStatus`, `LoggerMixin`, `Message`, `MessagePriority`, `Precision`, `QuantizationType`, `RNGState`, `ReportConfig`, `ReportFormat`, `ShutdownInProgressError`

</details>

### `validation/` — 22 classes (14 capability)

**Data sources & loaders**

- `DatasetLoader` (`validation.data_loaders`) — Abstract base class for dataset loaders.
- `MIMICLoader` (`validation.data_loaders`) — MIMIC-III Medical ICU Data Loader (IRB Placeholder Simulation).
- `NOAAHurricaneLoader` (`validation.data_loaders`) — NOAA National Hurricane Center Data Loader.
- `NOAAOceanLoader` (`validation.data_loaders`) — NOAA National Ocean Service Data Loader.
- `NOAASpaceWeatherLoader` (`validation.data_loaders`) — NOAA Space Weather Prediction Center Data Loader.
- `NSLKDDLoader` (`validation.data_loaders`) — NSL-KDD Network Intrusion Detection Dataset Loader.
- `USGSEarthquakeLoader` (`validation.data_loaders`) — USGS Earthquake Data Loader.

**Engines & orchestration**

- `ValidationPipeline` (`validation.pipeline`) — Comprehensive validation pipeline for anomaly detection models.

**Other capability classes**

- `ABTester` (`validation.pipeline`) — A/B Testing framework for model comparison.
- `APIRequestValidator` (`validation.api_validators`) — Unified API request validator.
- `DataArrayValidator` (`validation.api_validators`) — Validates numerical data arrays.
- `DataQualityChecker` (`validation.pipeline`) — Data quality validation checks.
- `InputSanitizer` (`validation.api_validators`) — Sanitizes input data to prevent injection attacks.
- `ParameterValidator` (`validation.api_validators`) — Validates API parameters.

<details><summary>Support types (8)</summary>

`ABTestResult`, `DatasetMetadata`, `QualityCheckResult`, `ValidationConfig`, `ValidationError`, `ValidationErrorType`, `ValidationResult`, `ValidationResult`

</details>

### `verifiers/` — 16 classes (14 capability)

**Other capability classes**

- `Dimension` (`verifiers.dimensional`) — An SI dimension as exact rational exponents over the seven base quantities.
- `GoldbachCertificate` (`verifiers.goldbach`) — A claimed Goldbach partition ``n = p + q`` awaiting adjudication.
- `LeanVerdict` (`verifiers.lean_theorem`) — Result of submitting a proof script to the Lean kernel.
- `LedgerEntry` (`verifiers.registry`) — A single adjudicated claim with full provenance.
- `Literal` (`verifiers.propositional`) — A propositional literal: a variable name with a polarity.
- `MysteryRegistry` (`verifiers.registry`) — Routes claims to oracles, records provenance, and grounds bounded GOSNN scalars.
- `ParadoxDefenseCertificate` (`verifiers.paradox`) — A named paradox with its naive (contradictory) and defended (consistent) theories.
- `PhysicsRelation` (`verifiers.physics`) — A claimed physical relation ``lhs = rhs`` to be checked for consistency.
- `TwinPrimeCertificate` (`verifiers.twin_primes`) — A claimed twin-prime pair ``(p, p + 2)`` awaiting adjudication.
- `Verdict` (`verifiers.collatz`) — Result of adjudicating a Collatz trajectory against the map.
- `Verdict` (`verifiers.goldbach`) — Result of adjudicating a certificate against the oracle.
- `Verdict` (`verifiers.paradox`) — Result of adjudicating a paradox defense.
- `Verdict` (`verifiers.physics`) — Result of adjudicating a physical relation.
- `Verdict` (`verifiers.twin_primes`) — Result of adjudicating a twin-prime certificate against the oracle.

<details><summary>Support types (2)</summary>

`Status`, `ThreeState`

</details>

