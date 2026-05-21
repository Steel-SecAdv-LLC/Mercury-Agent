# Step 7 — EXTRACTION DECISION

Generated 2026-05-21T08:32:59Z UTC.  Omni HEAD `2a3c6dd9d703`  -> Mercury HEAD `7af783761200`.

## 1. Executive verdict

**Extract 0 files in the forensic audit PR.**

Of Omni-AXA-Engine's 194 .py files under `src/omni_anomaly_engine/`, my evidence shows:

- **111 files** are byte-or-symbol-equivalent and clearly superseded by Mercury (`file_match.md` → SUPERSEDED).
- **13 files** are partially superseded (Mercury has a redesigned, richer counterpart).
- **12 files** are weak matches (Mercury reproduces the *role* under different names).
- **58 files** are empty/near-empty `__init__.py` namespace stubs that simply duplicate Mercury's.
- **7 content-bearing files** have NO Mercury counterpart but show signs of *intentional architectural exclusion* (Streamlit dropped, robotics+chemistry redesigned, vedic/math-unsolved out of scope, etc.) — REJECTED with citations.
- **6 content-bearing files** have NO Mercury counterpart AND no clear evidence of intentional exclusion — these are the OPEN QUESTIONS for human review (see Section 4).

Per the task's '100% confidence' rule, the audit PR ports zero files. A follow-up extraction PR (separately, after the open questions are answered) may port up to 6 files.

## 2. Files to extract

| file_to_extract | source_path_omni | dest_path_mercury | reason | license_action | risk |
|---|---|---|---|---|---|
| _(none)_ | _(audit PR contains no source moves; see Open Questions in Section 4)_ |   |   |   |   |

## 3. Files explicitly REJECTED for extraction

| omni_path | proof_artifact (under /docs/forensics/<today>/) | rejection_reason |
|---|---|---|
| `src/omni_anomaly_engine/mercury_a_agent.py` | `symbol_diff/mercury_a_agent.md` | All 6 shared symbols superseded; AgentState + analyze_with_mercury reproduced in agentic/agentic_autonomy.py + create_mercury_agent factory. |
| `src/omni_anomaly_engine/mercury_a_crews.py` | `symbol_diff/mercury_a_crews.md` | Domain Crew classes intentionally redesigned into Mercury's per-domain subpackages. |
| `src/omni_anomaly_engine/mercury_a_learning.py` | `symbol_diff/mercury_a_learning.md` | Mercury's PPOTrainer is the production trainer; Omni's AnomalyDetectionEnv is example scaffold. |
| `src/omni_anomaly_engine/truth_decipher.py` | `symbol_diff/truth_decipher.md` | Both Omni public symbols present in Mercury, 5th Cognitive phase added. |
| `src/omni_anomaly_engine/engine.py` | `symbol_diff/engine.md` | 9.2x growth; OmniAnomalyEngine -> OmniMercuryEngine with full caching + monitoring. |
| `src/omni_anomaly_engine/cli.py` | `symbol_diff/cli__primary.md` | All 6 commands present; signatures identical. |
| `src/omni_anomaly_engine/cli_enhanced.py` | `symbol_diff/cli_enhanced__merged.md` | run_* commands replaced by Mercury's physics_*/voice/serve commands (intentional CLI redesign). |
| `src/omni_anomaly_engine/comparison/pyod_integration.py` | `ambiguous_resolution.md` | Identical 3 public classes (PyODAlgorithm, CombinationMethod, PyODComparison); Mercury version is the canonical. |
| `src/omni_anomaly_engine/detectors/statistical.py` | `Mercury's docstring at src/omni_mercury_engine/detectors/statistical.py:18-30` | Mercury explicitly replaces z-score + IQR + IsolationForest ensemble with physics-based ensemble. |
| `src/omni_anomaly_engine/visualization/live_visualizer.py` | `ambiguous_resolution.md (visualization/)` | Streamlit dashboards intentionally dropped — Mercury is headless. |
| `src/omni_anomaly_engine/gui/streamlit_dashboard.py` | `ambiguous_resolution.md (visualization/) + deep_match.md` | Streamlit dashboards intentionally dropped — Mercury is headless. |
| `src/omni_anomaly_engine/gui/live_monitoring_dashboard.py` | `ambiguous_resolution.md (visualization/) + deep_match.md` | Streamlit dashboards intentionally dropped — Mercury is headless. |
| `src/omni_anomaly_engine/agents/mercury_a.py` | `ambiguous_resolution.md (agents/)` | Functionality split across Mercury's agentic/ + narrative/ subpackages; MercuryArtifactGenerator intentionally dropped. |
| `src/omni_anomaly_engine/utils/ancient_math.py` | `deep_match.md` | Vedic / Babylonian math fallback utilities; Mercury intentionally excludes (no references); non-production research curiosity. |
| `src/omni_anomaly_engine/domains/stermad/mathematics/math_unsolved.py` | `deep_match.md` | 35+ unsolved-math-problem computational verifications; out of anomaly-detection scope; Mercury excludes. |
| `src/omni_anomaly_engine/domains/stermad/engineering_robotics/anomaly_detector_robotics.py` | `ambiguous_resolution.md (domains/)` | Mercury extracted only the GWO core (gwo_ensemble.py); robotics-domain wrappers intentionally not carried over. |
| `src/omni_anomaly_engine/domains/stermad/chemistry/anomaly_detector_periodic_table.py` | `ambiguous_resolution.md (domains/)` | Mercury has its own redesigned chemistry detector (models/chemistry.py); porting would create a duplicate. |
| `src/omni_anomaly_engine/ml/layers.py` | `deep_match.md` | Equilibrium Propagation + LowPowerAnomalyDetector — niche neuromorphic research not aligned with Mercury's production focus. |
| `src/omni_anomaly_engine/fix_flake8.py` | `prior_verification.md` | Repo hygiene script. |
| `src/omni_anomaly_engine/fix_unused_imports.py` | `prior_verification.md` | Repo hygiene script. |

## 4. Open questions requiring human review

**Q1.** Is the architectural exclusion of `src/omni_anomaly_engine/domains/ehead/medical/cardiac_imaging.py` from Mercury intentional, or should it be ported? Context: 1031 LOC, 12 cardiac-imaging classes (DICOM/ECG/Echo/Stress/Holter/Coronary/Multiparameter/Wearable) with no Mercury counterpart.

- _Blocker_: HUMAN_REVIEW: copyright holder must confirm whether this Mercury-side exclusion reflects an intentional product decision (= no port) or an unintended gap (= port). The prior triage table marked the parent subpackage 'superseded' which contradicts the file-level finding.

**Q2.** Is the architectural exclusion of `src/omni_anomaly_engine/domains/ehead/medical/pathology_analyzer.py` from Mercury intentional, or should it be ported? Context: 1058 LOC, 12 pathology-station classes (Microtome/Embedding/Grossing/TissueProcessor) with no Mercury counterpart.

- _Blocker_: HUMAN_REVIEW: copyright holder must confirm whether this Mercury-side exclusion reflects an intentional product decision (= no port) or an unintended gap (= port). The prior triage table marked the parent subpackage 'superseded' which contradicts the file-level finding.

**Q3.** Is the architectural exclusion of `src/omni_anomaly_engine/domains/ehead/medical/neurology_detector.py` from Mercury intentional, or should it be ported? Context: 562 LOC, 8 EEG/BCI/Neuromodulation classes with no Mercury counterpart.

- _Blocker_: HUMAN_REVIEW: copyright holder must confirm whether this Mercury-side exclusion reflects an intentional product decision (= no port) or an unintended gap (= port). The prior triage table marked the parent subpackage 'superseded' which contradicts the file-level finding.

**Q4.** Is the architectural exclusion of `src/omni_anomaly_engine/domains/ehead/medical/psychiatry_detector.py` from Mercury intentional, or should it be ported? Context: 609 LOC, 8 mood/crisis-intervention classes with no Mercury counterpart.

- _Blocker_: HUMAN_REVIEW: copyright holder must confirm whether this Mercury-side exclusion reflects an intentional product decision (= no port) or an unintended gap (= port). The prior triage table marked the parent subpackage 'superseded' which contradicts the file-level finding.

**Q5.** Is the architectural exclusion of `src/omni_anomaly_engine/domains/ciad/compliance/iot_connector.py` from Mercury intentional, or should it be ported? Context: 530 LOC, MQTT + edge-IoT integration with no Mercury counterpart.

- _Blocker_: HUMAN_REVIEW: copyright holder must confirm whether this Mercury-side exclusion reflects an intentional product decision (= no port) or an unintended gap (= port). The prior triage table marked the parent subpackage 'superseded' which contradicts the file-level finding.

**Q6.** Is the architectural exclusion of `src/omni_anomaly_engine/ml/regularizers.py` from Mercury intentional, or should it be ported? Context: 240 LOC fairness regularizers (HSIC, DemographicParity) — no Mercury counterpart; matches Mercury's ethical/ mission but is missing.

- _Blocker_: HUMAN_REVIEW: copyright holder must confirm whether this Mercury-side exclusion reflects an intentional product decision (= no port) or an unintended gap (= port). The prior triage table marked the parent subpackage 'superseded' which contradicts the file-level finding.

## 5. Repro footer

- Omni-AXA-Engine HEAD: `2a3c6dd9d7035e9fef39223ffb371af11cf0e0a3` (origin/main, default branch)
- Mercury-Agent HEAD:   `7af7837612008e86afe91d54a534e9a18b9e3804` (origin/main, default branch)
- Audit branch (both repos): `claude/forensic-extraction-audit-p0bB9`
- License gate: Omni MIT -> Mercury GPL-3.0+ (one-way compatible; any extracted file must preserve MIT header and gain a NOTICE entry in THIRD_PARTY_NOTICES.md citing the Omni SHA).
- Tool versions:
  - python Python 3.11.15
  - ruff ruff 0.15.8
  - mypy mypy 1.19.1 (compiled: yes)
  - bandit bandit 1.9.4
  - pip-audit pip-audit 2.10.0
- Generated: 2026-05-21T08:32:59Z
