<!--
Copyright (C) 2025 Steel Security Advisors LLC
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Capability‑vs‑Vision Matrix (code‑grounded)

Applies to Mercury Agent **v2.1.x**. Last updated: 2026-07-11.

**Method:** every row below is anchored to a file (and, where useful, a
symbol) in `src/omni_mercury_engine/`. Where a previously‑circulated claim
does not match the code, the row says so plainly — the project plans against
what exists, and corrects the record where it drifted.

The framing axis is **identify · interpret · depict · deter · detail** layered
over the longer capability vision (multi‑domain, neuro‑symbolic, multi‑model,
interpret‑with‑accuracy, multi‑agent, multi‑language, autonomous, depict,
deter, detail).

---

## 1. The matrix

| Vision axis | Status | Primary code evidence | Transparent note |
|---|---|---|---|
| **Multi‑domain** | **Confirmed** | `medical/`, `infrastructure/{cyber,economic,humanitarian,resilience,scientific}/`, `detectors/{geological,marine,energy,economic,drone}/`, `space/`, `ocean/`, `emergent/`; real loaders in `datasets/adbench.py` (47 datasets) and `datasets/adrepository.py` (11, in `ADREPOSITORY_DATASETS`) | ~11 primary domains (+5 infrastructure sub‑domains). Real dataset references exceed the 66/75 committed‑benchmark figure (ADBench alone is 47). |
| **Neuro‑symbolic** | **Confirmed** | LTN `SymbolicConstraintModule` (`ml/symbolic_constraint.py:616`), `consensus_rule_graph()` (`ml/symbolic_constraint.py:354`), GOSNN (`core/global_omni_scalar_network.py`), ethical gates (`cognitive/ethical_bounding.py` `MINIMUM_BENEVOLENCE_FLOOR = 0.70`; `security/sigma_immutable_gate.py` decision threshold `0.93`), conformal UQ (`core/conformal_prediction.py`) | Genuine co‑training: the label‑scarcity‑adaptive λ schedule enters the loss (`ml/symbolic_constraint.py`), not a post‑hoc blend. |
| **Multi‑model / multi‑type** | **Confirmed** | 88 `class *Detector` definitions under `detectors/` (CI‑gated count in the README Codebase Scale block; grown by the streaming/statistical/state‑space tier of PR #323 — `detectors/detection_tier.py`, `core/detector_registry.py` — and the space/hazard detectors of PR #333) plus the 21‑probe Math‑Arrest equation family (`detectors/math_arrest/probes/`); fusion stack `detectors/fusion/multimodal_fusion.py` with `AttentionFusion` (`nn.MultiheadAttention`, 128‑D, 4 heads), score/decision/adaptive fusion | The "30 engines" figure is conservative against the 88 measured detector classes. |
| **Interpret with accuracy** | **Strong** | `explainability/` (5 SHAP variants, 4 counterfactual methods, GDPR Art. 22); temperature calibration + ECE (`engine.py::_fit_fusion_temperature`); conformal certificate attached in `detect_with_fusion`; serve‑path integrated‑gradients attributions via `detect_with_fusion(explain=True)` | The calibrated/transparent‑confidence work (PR #278) and the serve‑path explanation wiring (2026‑06‑02) carry the *"with accuracy"* qualifier. |
| **Multi‑agent** | **Shipped + measured (2026‑06‑11)** | `MultiAgentOrchestrator` (`agentic/orchestration.py`): planner/critic/executor loop over the engine's five real detectors — `HierarchicalPlanner` sequences the pipeline stages as real options with TD value learning, `ConsensusProtocol` forms per‑sample consensus (explicit abstention below quorum), `AnomalyReflexion` adapts the operating threshold from real labeled feedback, `AnomalyChainOfThought` renders decision‑faithful traces; dual hard ethical gates at the decision boundary; engine wiring via `engine.py::enable_multi_agent_orchestration()` | Measured on real ADBench labels (`benchmarks/orchestration_validation.py`, 5 datasets × 3 seeds): consensus AUC 0.841 ≥ mean member 0.833 (best member 0.901 — **no claim of beating the trained fusion model**); reflexion +0.054 paired balanced accuracy; 129/129 planned episodes executed; 600/600 traces decision‑faithful. Single‑agent loops (`AgenticAutonomy`, `OODAAgent`) remain alongside. |
| **Multi‑language** | **Corrected** | Python (746 source files + 586 test modules, CI‑gated counts) **+ Rust** (`rust_crypto/` → `lib.rs`, `hashing.rs`, `encryption.rs`, `kdf.rs`, `random.rs`, via PyO3); PQC consumed from the external `ama_cryptography` backend (`integrations/mercury_amacrypto.py`) | The historical "Python + **C**" claim was incorrect: Mercury's own native code is **Rust**, and the PQC C library is an *external* dependency (AMA Cryptography), not Mercury‑owned C. |
| **Autonomous (identify→interpret→decide→deter)** | **Closed (PR #283)** | `decision/` package wired at the `detect_with_fusion` seam (`engine.py::enable_decision_layer()`); the prior loops stopped at a log/alert/store stub in `agentic/agentic_autonomy.py` | Before #283, abstention was an implicit threshold and the three‑state contract was not wired into detection. Now: an explicit, calibration‑grounded "don't‑know" gate plus a bounded response layer. |
| **Depict (visualize/explain)** | **Partial — first per‑event coupling shipped** | Infrastructure exists: `gui/visualization_dashboard.py` (Plotly), `narrative/` (`NarrativeEngine`, `ReasoningChainNarrative`, `MercuryResponse`), `explainability/`; per‑decision depiction via `DecisionRecord.explain()` with a `signals`/`reasons`/`caveats` provenance trail | A full depiction layer (per‑event visuals from the threaded certificates) remains an open pillar. |
| **Deter (response/countermeasure)** | **Introduced (PR #283)** | `decision/response.py` (`ResponsePolicy`, line 114) driving the bounded `ResponseAction` enum (`decision/states.py:68`); closes into existing channels via `decision/bridge.py` (CAP 1.2 alerts `alerting/cap_generator.py`; `AgentAction` `agentic/agentic_autonomy.py`) | Bounded, **non‑destructive, fail‑closed, human‑in‑the‑loop** by construction — recommend/escalate, never autonomously destroy. A test invariant enforces the no‑destructive‑verbs catalogue. |
| **Detail** | **Strengthened** | Certificates threaded through `detect_with_fusion` (`conformal`, `gosnn_metadata`, `symbolic_consistency`, `drift_detection`); audit trail in `decision/ledger.py` (`DecisionLedger`) + `decision/loop.py` (`DecisionLoop`) | `DecisionRecord` turns the certificates into an auditable, JSON‑safe record with the active policy attached; the append‑only, bounded, **O(1)‑summary**, thread‑safe, JSON‑persistable `DecisionLedger` makes a stream of decisions a queryable audit trail (the *verify* step). The `intel/` closed‑loop package (PR #320) formalizes that step with `intel/verifier_loop.py`, `intel/self_consistency.py`, and `intel/provenance.py`. |

---

## 2. How the autonomy loop closed

PR #278 (calibrated/transparent confidence) strengthened **interpret** and laid the
safety floor for autonomy. PR #283 (Decision/Abstention/Response) converted
that groundwork into a closed **identify → interpret → decide → deter** loop
with an explicit "don't‑know" gate, without touching detection accuracy.

**The `info_geometry_certificate` signal is deliberately a no‑op in the
decision gate**, on measured evidence: that certificate's own contract
certifies a *component's* boundary, not the fused/gated verdict, and its
adaptive operating point is computed from the batch — in the single‑sample
serve path it collapses to `price == threshold` (a point scored 4.1× over
threshold in a batch of 9 reduces to exactly 1.0× at batch size 1), carrying
no information that could soundly refine the fused decision. The gate
consumes the *authoritative* calibrated signal (the conformal set); the
empirical‑coverage test confirms the guarantee survives the projection (zero
gate‑vs‑certificate contradictions, ~94 % coverage at a 90 % target).

## 3. Corrections to previously-circulated claims

1. **Language stack is Python + Rust, not Python + C.** No Mercury‑owned C
   sources exist (`rust_crypto/` is Rust via PyO3); PQC is the external
   `ama_cryptography` backend, gated fail‑closed at import (`_pqc_gate.py`).
2. **Dataset/detector counts were understated.** Real dataset references and
   detector classes both exceed earlier circulated figures; the CI‑gated
   counts in the README Codebase Scale block are authoritative.
3. **Depiction is "partial", not "minimal".** Visualization, narrative and
   explainability infrastructure exist; the per‑decision coupling shipped
   with `DecisionRecord.explain()`.
4. **Autonomy loops existed but were not closed** until PR #283 wired an
   explicit abstention gate into detection and added the bounded response
   layer.

## 4. Build-out sequence toward the autonomous goal

In leverage order:

- **(A) Decision/abstention + response layer — shipped (PR #283).**
- **(B) Multi‑agent orchestration inside Mercury — shipped + measured
  (2026‑06‑11).** Planner/critic/executor over the existing detectors
  (`agentic/orchestration.py`), validated per‑module on real ADBench labels
  by `benchmarks/orchestration_validation.py`; revives DORMANCY_LEDGER
  rows 10–11 under the same ablation discipline as the rest of the repo.
- (C) Depiction layer (per‑event explanations/visuals from the certificates
  threaded through `detect()` and `DecisionRecord`) — open. The
  orchestrator's decision‑faithful chain‑of‑thought traces
  (`MultiAgentOrchestrator.explain()`, fidelity 600/600) are a first
  measured input to this pillar.
- (D) Decorrelated‑stream fusion — **executed and closed by #278**: the
  pre‑registered protocol ran end‑to‑end (SHIP rejected, runtime
  byte‑unchanged; `research/governed_fusion/FINDINGS.md`). Improving raw
  detection now requires a different lever.

Everything above is aligned to the project's **Omnidirectional, STEM
exploration/discovery + humanitarian‑impact, open‑source, verifiable‑only**
principles: each row cites code, each claim is checkable, and the decision
layer is non‑destructive and fail‑closed by construction.
