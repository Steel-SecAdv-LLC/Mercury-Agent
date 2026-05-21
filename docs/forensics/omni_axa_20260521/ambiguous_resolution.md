# Step 4 — Ambiguous subpackage per-file resolution

Omni HEAD: `2a3c6dd9d7035e9fef39223ffb371af11cf0e0a3`  Mercury HEAD: `7af7837612008e86afe91d54a534e9a18b9e3804`

Classifications: SUPERSEDED · EXTRACT_AS_IS · EXTRACT_WITH_MERGE · PARTIAL_SUPERSEDED_REVIEW_NEEDED · WEAK_MATCH_REVIEW_NEEDED · NO_MERCURY_EQUIVALENT_REVIEW_NEEDED · OBSOLETE

## `agents/`

| omni file | classification | mercury counterpart | rationale |
|---|---|---|---|
| `src/omni_anomaly_engine/agents/__init__.py` | **SUPERSEDED** | `src/omni_mercury_engine/agentic/__init__.py` | Empty namespace package; Mercury's agentic/ package replaces it. |
| `src/omni_anomaly_engine/agents/mercury_a.py` | **SUPERSEDED** | `src/omni_mercury_engine/agentic/mercury_a_agent.py + narrative/{voice,interface,external_retrieval,engine}.py` | Omni's MercuryA agent (voice-activated, FAISS-backed RL learner, artifact generator) is split across Mercury's agentic/mercury_a_agent.py (core agent), narrative/voice.py (MercuryVoice replaces SpeechRecognition/whisper wrapper), narrative/interface.py (MercuryConversationInterface), narrative/external_retrieval.py (ExternalInformationRetriever replaces FAISS-only KB with web+DB+FAISS). The 8 Omni classes (MercuryMode, MercuryConfig, MercuryQuery, MercuryResponse, MercuryKnowledgeBase, MercuryReasoningEngine, MercuryArtifactGenerator, MercuryA) map to 8+ Mercury classes across these files. ArtifactGenerator (image/PDF/Excel/Markdown) appears intentionally dropped — Mercury exposes results via API rather than embedded artifacts. |

## `comparison/`

| omni file | classification | mercury counterpart | rationale |
|---|---|---|---|
| `src/omni_anomaly_engine/comparison/__init__.py` | **SUPERSEDED** | `src/omni_mercury_engine/comparison/__init__.py` | Mirror namespace file; functionally equivalent (Mercury's adds new license header). |
| `src/omni_anomaly_engine/comparison/pyod_integration.py` | **SUPERSEDED** | `src/omni_mercury_engine/comparison/pyod_integration.py` | Both define the same 3 public classes (PyODAlgorithm, CombinationMethod, PyODComparison). Mercury's version is 25 LOC shorter due to license-header rewording and renamed strings ('Omni-AXA-Engine' -> 'Mercury Agent'); functionally identical. |

## `visualization/`

| omni file | classification | mercury counterpart | rationale |
|---|---|---|---|
| `src/omni_anomaly_engine/visualization/__init__.py` | **OBSOLETE** | `(none)` | Mercury has no visualization/ subpackage. |
| `src/omni_anomaly_engine/visualization/live_visualizer.py` | **OBSOLETE** | `(none)` | 902 LOC Streamlit-based live dashboard (LiveVisualizer, StreamingData, VisualizationConfig, VisualizationType). Mercury intentionally excludes Streamlit (`streamlit`, `LiveVisualizer`, `StreamingData` have zero references across Mercury's src tree). Architectural decision: Mercury is headless/CLI/API-first. EXTRACTION REJECTED — out of Mercury's scope. |

## `domains/`

Inspected 71 files in `domains/`.

| omni file | classification | mercury counterpart | rationale |
|---|---|---|---|
| `src/omni_anomaly_engine/domains/ciad/compliance/iot_connector.py` | **NO_MERCURY_EQUIVALENT_REVIEW_NEEDED** | `Mercury integrations/cross_platform_hub.py (different design — no MQTT/edge stack)` | 530 LOC, 8 public classes/funcs (IoTMode, SensorType, MQTTQoS, IoTDevice, IoTAnomalyResult, IoTConnector). MQTT/edge-device integration not present in Mercury. |
| `src/omni_anomaly_engine/domains/ehead/medical/cardiac_imaging.py` | **NO_MERCURY_EQUIVALENT_REVIEW_NEEDED** | `Mercury medical/cardiology/cardiology_predictor.py (partial overlap, single-predictor)` | 1031 LOC, 12 public classes (DICOMProcessor, ECGStreamProcessor, EchocardiogramAnalyzer, StressTestAnalyzer, WearableDeviceMonitor, MultiparameterMonitor, WFDBSyntheticECGGenerator, etc.). Mercury covers cardiology *prediction* but NOT the imaging-acquisition stack. Real clinical capability gap. |
| `src/omni_anomaly_engine/domains/ehead/medical/neurology_detector.py` | **NO_MERCURY_EQUIVALENT_REVIEW_NEEDED** | `(none — Mercury has medical/critical_care/neurocritical_care.py with a different surface)` | 562 LOC, 8 public classes (EEGAnalyzer, BCIMonitoringSystem, NeuromodulationMonitor, FocusedUltrasoundMonitor, BCIState, NeurologyDetector). Mercury's neurocritical_care covers ICU neuro monitoring; EEG/BCI/Neuromod is a distinct capability not in Mercury. |
| `src/omni_anomaly_engine/domains/ehead/medical/pathology_analyzer.py` | **NO_MERCURY_EQUIVALENT_REVIEW_NEEDED** | `(none)` | 1058 LOC, 12 public classes (PathologyStation, StainType, MicrotomeSimulator, EmbeddingStationSimulator, GrossingStationAnalyzer, TissueProcessorLSTM, etc.). No Mercury counterpart. |
| `src/omni_anomaly_engine/domains/ehead/medical/psychiatry_detector.py` | **NO_MERCURY_EQUIVALENT_REVIEW_NEEDED** | `(none)` | 609 LOC, 8 public classes (MoodDetectionLSTM, VocalPatternAnalyzer, DBSMonitoringSystem, CrisisInterventionSystem, MoodState, CrisisLevel). No Mercury counterpart. |
| `src/omni_anomaly_engine/domains/stermad/mathematics/math_unsolved.py` | **NO_MERCURY_EQUIVALENT_REVIEW_NEEDED** | `(none — Mercury detectors/math_arrest/ has different scope: probe-based math-arrest detectors)` | 751 LOC, 4 public symbols (UnsolvedMathProblems, ProblemCategory, UnsolvedProblemResult, create_omni_mathematics_scalars). 35+ unsolved-problem computational verifications (Collatz, Goldbach, twin primes, Beatty sequences, etc.). NOT in Mercury. Out-of-scope for production anomaly detection? |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/cyber/cyber_fortress.py` | **PARTIAL_SUPERSEDED_REVIEW_NEEDED** | `see deep_match.md` | 4/5 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ehead/medical/abms_disciplines.py` | **PARTIAL_SUPERSEDED_REVIEW_NEEDED** | `see deep_match.md` | 4/5 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ehead/medical/cardiology_predictor.py` | **PARTIAL_SUPERSEDED_REVIEW_NEEDED** | `see deep_match.md` | 6/7 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ehead/medical/medical_cure_predictor.py` | **PARTIAL_SUPERSEDED_REVIEW_NEEDED** | `see deep_match.md` | 6/9 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/stermad/chemistry/anomaly_detector_periodic_table.py` | **PARTIAL_SUPERSEDED_REVIEW_NEEDED** | `src/omni_mercury_engine/models/chemistry.py` | Mercury HAS chemistry detection but with different classes: ChemistryAnomalyDetector + PeriodicTableEncoder (719 LOC). Omni's PeriodicTableAnomalyDetector + PeriodicProperty are a *different* implementation. Functional overlap is non-trivial — periodic-table encoding present in both, but anomaly scoring approach differs. |
| `src/omni_anomaly_engine/domains/stermad/engineering_robotics/anomaly_detector_robotics.py` | **PARTIAL_SUPERSEDED_REVIEW_NEEDED** | `src/omni_mercury_engine/detectors/advanced/gwo_ensemble.py` | Mercury has GreyWolfOptimizer (shared symbol) inside gwo_ensemble.py (a generic GWO ensemble detector). Omni's `RoboticsAnomalyDetector`, `GWOAutoencoder`, `StudentTeacherAnomalyDetector`, `GraphAttentionLayer` are robotics-specific and NOT in Mercury. Robotics-domain detection logic is unique to Omni. |
| `src/omni_anomaly_engine/domains/ciad/compliance/nist_csf_integrator.py` | **SUPERSEDED** | `see deep_match.md` | All 7 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ciad/compliance/osha_compliance_anomaly.py` | **SUPERSEDED** | `see deep_match.md` | All 7 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ciad/compliance/tlp_handler.py` | **SUPERSEDED** | `see deep_match.md` | All 4 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/biometric.py` | **SUPERSEDED** | `see deep_match.md` | All 3 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/core.py` | **SUPERSEDED** | `see deep_match.md` | All 2 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/crisis_monitor.py` | **SUPERSEDED** | `see deep_match.md` | All 2 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/cyber/cross_border_intel.py` | **SUPERSEDED** | `see deep_match.md` | All 1 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/cyber/quantum_risk_cyber.py` | **SUPERSEDED** | `see deep_match.md` | All 5 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/cyber/space_infrastructure.py` | **SUPERSEDED** | `see deep_match.md` | All 1 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/cybint_subprocessor.py` | **SUPERSEDED** | `see deep_match.md` | All 10 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/encryption.py` | **SUPERSEDED** | `see deep_match.md` | All 2 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/hive_firewall.py` | **SUPERSEDED** | `see deep_match.md` | All 3 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/int_sources.py` | **SUPERSEDED** | `see deep_match.md` | All 28 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/intelligence_fusion.py` | **SUPERSEDED** | `see deep_match.md` | All 6 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/pattern_recognition.py` | **SUPERSEDED** | `see deep_match.md` | All 2 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/rate_limiting.py` | **SUPERSEDED** | `see deep_match.md` | All 1 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/realtime_threat_detection.py` | **SUPERSEDED** | `see deep_match.md` | All 3 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/tempest_detection.py` | **SUPERSEDED** | `see deep_match.md` | All 8 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/threat_detection.py` | **SUPERSEDED** | `see deep_match.md` | All 2 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ciad/cyber_security/traffic_analysis.py` | **SUPERSEDED** | `see deep_match.md` | All 7 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ciad/economic/anomaly_detector_financial_crisis.py` | **SUPERSEDED** | `see deep_match.md` | All 8 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ciad/economic/world_bank_sectors.py` | **SUPERSEDED** | `see deep_match.md` | All 1 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ciad/government/communications_it.py` | **SUPERSEDED** | `see deep_match.md` | All 1 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ciad/government/ncf_monitor.py` | **SUPERSEDED** | `see deep_match.md` | All 1 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ehead/earth_sciences/energy/anomaly_detector_emp.py` | **SUPERSEDED** | `see deep_match.md` | All 7 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ehead/earth_sciences/energy/energy_dams.py` | **SUPERSEDED** | `see deep_match.md` | All 3 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ehead/earth_sciences/energy/energy_optimization.py` | **SUPERSEDED** | `see deep_match.md` | All 3 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ehead/earth_sciences/geology/anomaly_detector_landslide.py` | **SUPERSEDED** | `see deep_match.md` | All 7 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ehead/earth_sciences/geology/anomaly_detector_volcanic.py` | **SUPERSEDED** | `see deep_match.md` | All 9 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ehead/earth_sciences/geology/anomaly_detector_wildfire.py` | **SUPERSEDED** | `see deep_match.md` | All 5 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ehead/earth_sciences/marine/anomaly_detector_biodiversity.py` | **SUPERSEDED** | `see deep_match.md` | All 4 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ehead/earth_sciences/marine/anomaly_detector_oceanography.py` | **SUPERSEDED** | `see deep_match.md` | All 3 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ehead/human_events/agrifood_security.py` | **SUPERSEDED** | `see deep_match.md` | All 2 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ehead/human_events/climate_resilience.py` | **SUPERSEDED** | `see deep_match.md` | All 2 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ehead/human_events/economic_resilience.py` | **SUPERSEDED** | `see deep_match.md` | All 2 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ehead/human_events/education_equity.py` | **SUPERSEDED** | `see deep_match.md` | All 2 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ehead/human_events/essential_workers.py` | **SUPERSEDED** | `see deep_match.md` | All 1 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ehead/human_events/government_facilities.py` | **SUPERSEDED** | `see deep_match.md` | All 1 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ehead/human_events/neuroscience.py` | **SUPERSEDED** | `see deep_match.md` | All 2 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ehead/medical/anesthesiology_predictor.py` | **SUPERSEDED** | `see deep_match.md` | All 7 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ehead/medical/endocrinology_detector.py` | **SUPERSEDED** | `see deep_match.md` | All 8 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ehead/medical/healthcare_emergency.py` | **SUPERSEDED** | `see deep_match.md` | All 3 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ehead/medical/neurocritical_care.py` | **SUPERSEDED** | `see deep_match.md` | All 8 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ehead/medical/pandemic/epidemic_model.py` | **SUPERSEDED** | `see deep_match.md` | All 2 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ehead/medical/pandemic/pandemic_detector.py` | **SUPERSEDED** | `see deep_match.md` | All 7 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ehead/medical/pandemic/pathogen_detector.py` | **SUPERSEDED** | `see deep_match.md` | All 2 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/ehead/medical/sepsis_detector.py` | **SUPERSEDED** | `see deep_match.md` | All 6 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/stermad/chemistry/chemical_nuclear.py` | **SUPERSEDED** | `see deep_match.md` | All 2 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/stermad/chemistry/chemistry_model.py` | **SUPERSEDED** | `see deep_match.md` | All 5 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/stermad/chemistry/isotope_predictor.py` | **SUPERSEDED** | `see deep_match.md` | All 7 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/stermad/engineering_robotics/drone_anomaly_detector.py` | **SUPERSEDED** | `see deep_match.md` | All 6 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/stermad/scientific/emerging_tech_monitor.py` | **SUPERSEDED** | `see deep_match.md` | All 1 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/stermad/space/astrophysical.py` | **SUPERSEDED** | `see deep_match.md` | All 1 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/stermad/space/disaster_precursor_detector.py` | **SUPERSEDED** | `see deep_match.md` | All 6 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/stermad/space/interstellar_objects.py` | **SUPERSEDED** | `see deep_match.md` | All 6 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/stermad/space/schumann_resonance.py` | **SUPERSEDED** | `see deep_match.md` | All 4 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/stermad/space/solar_storm_detector.py` | **SUPERSEDED** | `see deep_match.md` | All 7 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/stermad/space/space_exploration_analyzer.py` | **SUPERSEDED** | `see deep_match.md` | All 1 public symbols defined in Mercury. |
| `src/omni_anomaly_engine/domains/stermad/space/space_inspired.py` | **SUPERSEDED** | `see deep_match.md` | All 3 public symbols defined in Mercury. |