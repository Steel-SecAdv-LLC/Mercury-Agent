# Deep symbol match — Omni public-symbol coverage in Mercury

Files inspected: **148**

## Lowest Mercury coverage (potential extraction signal)

| omni file | symbols found / total | ratio | unmatched symbols |
|---|---|---|---|
| `src/omni_anomaly_engine/domains/ehead/medical/cardiac_imaging.py` | 0/12 | 0.00 | ImagingModality, CardiacImagingResult, ECGStreamProcessor, EchocardiogramAnalyzer, StressTestAnalyzer, WearableDeviceMonitor, MultiparameterMonitor, WFDBSyntheticECGGenerator… |
| `src/omni_anomaly_engine/domains/ehead/medical/pathology_analyzer.py` | 0/12 | 0.00 | PathologyStation, StainType, PathologyResult, MicrotomeSimulator, EmbeddingStationSimulator, GrossingStationAnalyzer, TissueProcessorLSTM, TissueProcessorAnalyzer… |
| `src/omni_anomaly_engine/domains/ciad/compliance/iot_connector.py` | 0/8 | 0.00 | IoTMode, SensorType, MQTTQoS, IoTDevice, SensorReading, IoTAnomalyResult, IoTConnector, get_iot_connector |
| `src/omni_anomaly_engine/domains/ehead/medical/neurology_detector.py` | 0/8 | 0.00 | NeurologicalCondition, BCIState, NeurologyPredictionResult, EEGAnalyzer, BCIMonitoringSystem, NeuromodulationMonitor, FocusedUltrasoundMonitor, NeurologyDetector |
| `src/omni_anomaly_engine/domains/ehead/medical/psychiatry_detector.py` | 0/8 | 0.00 | MoodState, CrisisLevel, PsychiatryPredictionResult, MoodDetectionLSTM, VocalPatternAnalyzer, DBSMonitoringSystem, CrisisInterventionSystem, PsychiatryDetector |
| `src/omni_anomaly_engine/utils/ancient_math.py` | 0/6 | 0.00 | robust_sqrt, vedic_reciprocal, vedic_multiply, robust_sqrt_vec, set_vedic_optimization, is_vedic_enabled |
| `src/omni_anomaly_engine/domains/stermad/chemistry/anomaly_detector_periodic_table.py` | 0/4 | 0.00 | PeriodicProperty, PeriodicTableAnomalyResult, PeriodicTableAnomalyDetector, create_omni_periodic_table_scalars |
| `src/omni_anomaly_engine/domains/stermad/mathematics/math_unsolved.py` | 0/4 | 0.00 | ProblemCategory, UnsolvedProblemResult, UnsolvedMathProblems, create_omni_mathematics_scalars |
| `src/omni_anomaly_engine/mercury_a_learning.py` | 0/4 | 0.00 | RewardConfig, AnomalyDetectionEnv, MercuryLearner, AdaptiveLearner |
| `src/omni_anomaly_engine/visualization/live_visualizer.py` | 0/4 | 0.00 | VisualizationType, VisualizationConfig, StreamingData, LiveVisualizer |
| `src/omni_anomaly_engine/ml/layers.py` | 0/3 | 0.00 | EquilibriumPropagationLayer, LowPowerAnomalyDetector, compute_neuromorphic_efficiency |
| `src/omni_anomaly_engine/ml/regularizers.py` | 0/3 | 0.00 | HSICRegularizer, DemographicParityLoss, compute_fairness_metrics |
| `src/omni_anomaly_engine/detectors/statistical.py` | 0/1 | 0.00 | StatisticalAnomalyDetector |
| `src/omni_anomaly_engine/engine.py` | 0/1 | 0.00 | OmniAnomalyEngine |
| `src/omni_anomaly_engine/gui/streamlit_dashboard.py` | 1/19 | 0.05 | medical_analysis_page, cardiology_interface, display_cardiology_results, sepsis_interface, display_sepsis_results, neurocritical_interface, general_medical_interface, security_analysis_page… |
| `src/omni_anomaly_engine/utils/validation.py` | 1/11 | 0.09 | validate_shape, validate_range, check_normality, detect_outliers, validate_data_reliability, validate_medical_data, validate_security_data, validate_energy_data… |
| `src/omni_anomaly_engine/mercury_a_crews.py` | 1/10 | 0.10 | CrewTask, BaseCrew, MedicalCrew, SecurityCrew, EnergyCrew, InfrastructureCrew, SpaceCrew, EmergentCrew… |
| `src/omni_anomaly_engine/agents/mercury_a.py` | 1/8 | 0.12 | MercuryMode, MercuryConfig, MercuryQuery, MercuryKnowledgeBase, MercuryReasoningEngine, MercuryArtifactGenerator, MercuryA |
| `src/omni_anomaly_engine/domains/stermad/engineering_robotics/anomaly_detector_robotics.py` | 1/8 | 0.12 | RoboticsAnomalyType, RoboticsAnomalyResult, GraphAttentionLayer, GWOAutoencoder, StudentTeacherAnomalyDetector, RoboticsAnomalyDetector, create_omni_robotics_scalars |
| `src/omni_anomaly_engine/gui/live_monitoring_dashboard.py` | 1/7 | 0.14 | initialize_session_state, sidebar_controls, render_earthquake_monitoring, render_wildfire_monitoring, render_solar_storm_monitoring, render_schumann_monitoring |
| `src/omni_anomaly_engine/data_sources/realtime_apis.py` | 1/5 | 0.20 | USGSEarthquakeAPI, NOAASpaceWeatherAPI, NASAFIRMSWildfireAPI, RealtimeDataAggregator |
| `src/omni_anomaly_engine/utils/logging_config.py` | 2/9 | 0.22 | EthicalAuditFilter, setup_logging, log_ethical_audit, log_data_stats, descriptive_output, init_global_logging, get_global_logger |
| `src/omni_anomaly_engine/core/fusion.py` | 1/4 | 0.25 | HybridFusionLayer, EarlyFusionEncoder, OmniAvaEngine |
| `src/omni_anomaly_engine/ml/training.py` | 2/7 | 0.29 | AvaOptimizer, AvaMomentumOptimizer, AvaExponentialDecayOptimizer, AvaHarmonicOptimizer, create_ava_optimizer |
| `src/omni_anomaly_engine/cli_enhanced.py` | 4/10 | 0.40 | run_medical, run_security, run_humanitarian, run_schumann, run_chemistry, run_demo |
| `src/omni_anomaly_engine/core/self_healing.py` | 1/2 | 0.50 | CRISPRInspiredSelfHealing |
| `src/omni_anomaly_engine/detectors/dimensional.py` | 1/2 | 0.50 | NeuralProjection |
| `src/omni_anomaly_engine/domains/ehead/medical/medical_cure_predictor.py` | 6/9 | 0.67 | CRISPRGeneEditingAnalyzer, PolygenicRiskScoreCalculator, RNATherapeuticsAnalyzer |
| `src/omni_anomaly_engine/core/base.py` | 2/3 | 0.67 | BaseEncoder |
| `src/omni_anomaly_engine/core/multivariate_timeseries.py` | 2/3 | 0.67 | FractionalDifferentiator |
| `src/omni_anomaly_engine/federated/federated_detector.py` | 3/4 | 0.75 | FederatedStrategy |
| `src/omni_anomaly_engine/ml/harmonic_encoder.py` | 3/4 | 0.75 | HarmonicEncoder |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/cyber/cyber_fortress.py` | 4/5 | 0.80 | EncryptedTrafficAnomalyDetector |
| `src/omni_anomaly_engine/domains/ehead/medical/abms_disciplines.py` | 4/5 | 0.80 | MultiSpecialtyNeuralNet |
| `src/omni_anomaly_engine/core/extended_anomaly_engine.py` | 5/6 | 0.83 | OmniAXAEngine |
| `src/omni_anomaly_engine/domains/ehead/medical/cardiology_predictor.py` | 6/7 | 0.86 | AdvancedCardiacInterventions |
| `src/omni_anomaly_engine/mercury_a_agent.py` | 7/8 | 0.88 | analyze_with_mercury |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/int_sources.py` | 28/28 | 1.00 |  |
| `src/omni_anomaly_engine/core/three_r_mechanism.py` | 10/10 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/cybint_subprocessor.py` | 10/10 | 1.00 |  |
| `src/omni_anomaly_engine/utils/profiling.py` | 10/10 | 1.00 |  |
| `src/omni_anomaly_engine/core/ethical_risk_matrix.py` | 9/9 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ehead/earth_sciences/geology/anomaly_detector_volcanic.py` | 9/9 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/tempest_detection.py` | 8/8 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ciad/economic/anomaly_detector_financial_crisis.py` | 8/8 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ehead/medical/endocrinology_detector.py` | 8/8 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ehead/medical/neurocritical_care.py` | 8/8 | 1.00 |  |
| `src/omni_anomaly_engine/core/exceptions.py` | 7/7 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ciad/compliance/nist_csf_integrator.py` | 7/7 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ciad/compliance/osha_compliance_anomaly.py` | 7/7 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/traffic_analysis.py` | 7/7 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ehead/earth_sciences/energy/anomaly_detector_emp.py` | 7/7 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ehead/earth_sciences/geology/anomaly_detector_landslide.py` | 7/7 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ehead/medical/anesthesiology_predictor.py` | 7/7 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ehead/medical/pandemic/pandemic_detector.py` | 7/7 | 1.00 |  |
| `src/omni_anomaly_engine/domains/stermad/chemistry/isotope_predictor.py` | 7/7 | 1.00 |  |
| `src/omni_anomaly_engine/domains/stermad/space/solar_storm_detector.py` | 7/7 | 1.00 |  |
| `src/omni_anomaly_engine/api/server.py` | 6/6 | 1.00 |  |
| `src/omni_anomaly_engine/cli.py` | 6/6 | 1.00 |  |
| `src/omni_anomaly_engine/core/config.py` | 6/6 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/intelligence_fusion.py` | 6/6 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ehead/medical/sepsis_detector.py` | 6/6 | 1.00 |  |
| `src/omni_anomaly_engine/domains/stermad/engineering_robotics/drone_anomaly_detector.py` | 6/6 | 1.00 |  |
| `src/omni_anomaly_engine/domains/stermad/space/disaster_precursor_detector.py` | 6/6 | 1.00 |  |
| `src/omni_anomaly_engine/domains/stermad/space/interstellar_objects.py` | 6/6 | 1.00 |  |
| `src/omni_anomaly_engine/ml/encoders.py` | 6/6 | 1.00 |  |
| `src/omni_anomaly_engine/core/ai_ethics.py` | 5/5 | 1.00 |  |
| `src/omni_anomaly_engine/core/neurosymbolic_engine.py` | 5/5 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/cyber/quantum_risk_cyber.py` | 5/5 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ehead/earth_sciences/geology/anomaly_detector_wildfire.py` | 5/5 | 1.00 |  |
| `src/omni_anomaly_engine/domains/stermad/chemistry/chemistry_model.py` | 5/5 | 1.00 |  |
| `src/omni_anomaly_engine/emergent/emergent_life_detector.py` | 5/5 | 1.00 |  |
| `src/omni_anomaly_engine/ml/optimizers.py` | 5/5 | 1.00 |  |
| `src/omni_anomaly_engine/models/parapsychology.py` | 5/5 | 1.00 |  |
| `src/omni_anomaly_engine/utils/report_generator.py` | 5/5 | 1.00 |  |
| `src/omni_anomaly_engine/core/ethical_governor.py` | 4/4 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ciad/compliance/tlp_handler.py` | 4/4 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ehead/earth_sciences/marine/anomaly_detector_biodiversity.py` | 4/4 | 1.00 |  |
| `src/omni_anomaly_engine/domains/stermad/space/schumann_resonance.py` | 4/4 | 1.00 |  |
| `src/omni_anomaly_engine/ml/attention.py` | 4/4 | 1.00 |  |
| `src/omni_anomaly_engine/utils/comm.py` | 4/4 | 1.00 |  |
| `src/omni_anomaly_engine/utils/rng.py` | 4/4 | 1.00 |  |
| `src/omni_anomaly_engine/agentic/agentic_autonomy.py` | 3/3 | 1.00 |  |
| `src/omni_anomaly_engine/comparison/pyod_integration.py` | 3/3 | 1.00 |  |
| `src/omni_anomaly_engine/core/regenerative.py` | 3/3 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/biometric.py` | 3/3 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/hive_firewall.py` | 3/3 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/realtime_threat_detection.py` | 3/3 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ehead/earth_sciences/energy/energy_dams.py` | 3/3 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ehead/earth_sciences/energy/energy_optimization.py` | 3/3 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ehead/earth_sciences/marine/anomaly_detector_oceanography.py` | 3/3 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ehead/medical/healthcare_emergency.py` | 3/3 | 1.00 |  |
| `src/omni_anomaly_engine/domains/stermad/space/space_inspired.py` | 3/3 | 1.00 |  |
| `src/omni_anomaly_engine/federated/federated_robust.py` | 3/3 | 1.00 |  |
| `src/omni_anomaly_engine/ml/hatcn_ad.py` | 3/3 | 1.00 |  |
| `src/omni_anomaly_engine/models/multiverse.py` | 3/3 | 1.00 |  |
| `src/omni_anomaly_engine/models/neurosymbolic.py` | 3/3 | 1.00 |  |
| `src/omni_anomaly_engine/core/chaos_evolutionary.py` | 2/2 | 1.00 |  |
| `src/omni_anomaly_engine/core/ethical_config.py` | 2/2 | 1.00 |  |
| `src/omni_anomaly_engine/core/novel_class_discovery.py` | 2/2 | 1.00 |  |
| `src/omni_anomaly_engine/core/symbolic_reasoning.py` | 2/2 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/core.py` | 2/2 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/crisis_monitor.py` | 2/2 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/encryption.py` | 2/2 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/pattern_recognition.py` | 2/2 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/threat_detection.py` | 2/2 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ehead/human_events/agrifood_security.py` | 2/2 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ehead/human_events/climate_resilience.py` | 2/2 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ehead/human_events/economic_resilience.py` | 2/2 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ehead/human_events/education_equity.py` | 2/2 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ehead/human_events/neuroscience.py` | 2/2 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ehead/medical/pandemic/epidemic_model.py` | 2/2 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ehead/medical/pandemic/pathogen_detector.py` | 2/2 | 1.00 |  |
| `src/omni_anomaly_engine/domains/stermad/chemistry/chemical_nuclear.py` | 2/2 | 1.00 |  |
| `src/omni_anomaly_engine/ml/fusion_network.py` | 2/2 | 1.00 |  |
| `src/omni_anomaly_engine/ml/multimodal_fusion.py` | 2/2 | 1.00 |  |
| `src/omni_anomaly_engine/ml/vae_pattern_learner.py` | 2/2 | 1.00 |  |
| `src/omni_anomaly_engine/resilience/circuit_breaker.py` | 2/2 | 1.00 |  |
| `src/omni_anomaly_engine/resilience/health_monitoring.py` | 2/2 | 1.00 |  |
| `src/omni_anomaly_engine/scaling/bain_ai_scaling.py` | 2/2 | 1.00 |  |
| `src/omni_anomaly_engine/truth_decipher.py` | 2/2 | 1.00 |  |
| `src/omni_anomaly_engine/core/federated_learning.py` | 1/1 | 1.00 |  |
| `src/omni_anomaly_engine/core/info_geometry.py` | 1/1 | 1.00 |  |
| `src/omni_anomaly_engine/core/quantum_kernels.py` | 1/1 | 1.00 |  |
| `src/omni_anomaly_engine/detectors/directive.py` | 1/1 | 1.00 |  |
| `src/omni_anomaly_engine/detectors/graph_based.py` | 1/1 | 1.00 |  |
| `src/omni_anomaly_engine/detectors/spatial.py` | 1/1 | 1.00 |  |
| `src/omni_anomaly_engine/detectors/temporal.py` | 1/1 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/cyber/cross_border_intel.py` | 1/1 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/cyber/space_infrastructure.py` | 1/1 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/rate_limiting.py` | 1/1 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ciad/economic/world_bank_sectors.py` | 1/1 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ciad/government/communications_it.py` | 1/1 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ciad/government/ncf_monitor.py` | 1/1 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ehead/human_events/essential_workers.py` | 1/1 | 1.00 |  |
| `src/omni_anomaly_engine/domains/ehead/human_events/government_facilities.py` | 1/1 | 1.00 |  |
| `src/omni_anomaly_engine/domains/stermad/scientific/emerging_tech_monitor.py` | 1/1 | 1.00 |  |
| `src/omni_anomaly_engine/domains/stermad/space/astrophysical.py` | 1/1 | 1.00 |  |
| `src/omni_anomaly_engine/domains/stermad/space/space_exploration_analyzer.py` | 1/1 | 1.00 |  |
| `src/omni_anomaly_engine/ml/gwo_optimizer.py` | 1/1 | 1.00 |  |
| `src/omni_anomaly_engine/ml/inference.py` | 1/1 | 1.00 |  |
| `src/omni_anomaly_engine/models/affective.py` | 1/1 | 1.00 |  |
| `src/omni_anomaly_engine/models/consciousness.py` | 1/1 | 1.00 |  |
| `src/omni_anomaly_engine/models/neural.py` | 1/1 | 1.00 |  |
| `src/omni_anomaly_engine/models/quantum.py` | 1/1 | 1.00 |  |
| `src/omni_anomaly_engine/models/simulation.py` | 1/1 | 1.00 |  |
| `src/omni_anomaly_engine/resilience/retry.py` | 1/1 | 1.00 |  |
| `src/omni_anomaly_engine/resilience/self_healing.py` | 1/1 | 1.00 |  |