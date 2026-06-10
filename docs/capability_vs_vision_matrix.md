<!--
Copyright (C) 2025 Steel Security Advisors LLC
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Capability‑vs‑Vision Matrix (code‑grounded)

**Status:** verified against the source tree, not the repo index.
**Method:** every row below is anchored to a file (and, where useful, a line)
in `src/omni_mercury_engine/`. Where a previously‑circulated claim does not
match the code, the row says so plainly — *verifiable‑only* means we plan
against what exists, and we correct the record where it drifted.

The framing axis is **identify · interpret · depict · deter · detail** layered
over the longer capability vision (multi‑domain, neuro‑symbolic, multi‑model,
interpret‑with‑accuracy, multi‑agent, multi‑language, autonomous, depict,
deter, detail).

---

## 1. The matrix

| Vision axis | Status today | Primary code evidence | Honest note |
|---|---|---|---|
| **Multi‑domain** | **Confirmed** | `medical/`, `infrastructure/{cyber,economic,humanitarian,resilience,scientific}/`, `detectors/{geological,marine,energy,economic,space,drone}/`, `ocean/`, `emergent/`; real loaders in `datasets/adbench.py` (47 datasets) and `datasets/adrepository.py` (11, in `ADREPOSITORY_DATASETS`) | ~11 primary domains (+5 infrastructure sub‑domains). Real dataset **references exceed** the "65/75" index figure (ADBench alone is 47); the index *understated* this. |
| **Neuro‑symbolic** | **Confirmed** | LTN `SymbolicConstraintModule` (`ml/symbolic_constraint.py:374`), `consensus_rule_graph()` (`ml/symbolic_constraint.py:151`), GOSNN (`core/global_omni_scalar_network.py`), ethical gates (`cognitive/ethical_bounding.py` benevolence floor `0.70`; `security/sigma_immutable_gate.py` threshold `0.93`), conformal UQ (`core/conformal_prediction.py`) | Genuine co‑training (label‑scarcity‑adaptive λ schedule, `ml/symbolic_constraint.py:244`). |
| **Multi‑model / multi‑type** | **Confirmed (understated)** | 80+ detector classes under `detectors/`; fusion stack `detectors/fusion/multimodal_fusion.py` with `AttentionFusion` (`nn.MultiheadAttention(embed_dim=128, num_heads=4)`, line ~204), score/decision/adaptive fusion | The "22+ engines" figure is conservative; the Math‑Arrest family alone is ~22 probes. |
| **Interpret with accuracy** | **Strong — the recent win** | `explainability/` (5 SHAP variants, 4 counterfactual methods, GDPR Art. 22); temperature calibration + ECE (`engine.py` `_fit_fusion_temperature`); conformal certificate attached at `engine.py:~4058` | The calibrated/honest‑confidence work (PR #278 line) strengthens **interpret** and the *"with accuracy"* qualifier. |
| **Multi‑agent** | **Partial** | Single‑agent loops exist: `AgenticAutonomy` (`agentic/agentic_autonomy.py:88`), `OODAAgent` (`cognitive/autonomous_agent.py:593`) | No planner/critic/executor **multi‑agent orchestration** in Mercury yet. (A net‑new pillar.) |
| **Multi‑language** | **Corrected** | Python (~1.1k `.py`) **+ Rust** (`rust_crypto/` → `lib.rs`, `hashing.rs`, `encryption.rs`, `kdf.rs`, `random.rs`, via PyO3); PQC consumed from the external `ama_cryptography` backend (`integrations/mercury_amacrypto.py`) | The "Python + **C**" claim is **incorrect**: the native code is **Rust**, and PQC is an *external* dependency, not Mercury‑owned C. |
| **Autonomous (identify→interpret→deter)** | **Closed by this PR** | New `decision/` layer wired at the `detect_with_fusion` seam (`engine.py`, `enable_decision_layer()`); prior loops stopped at a log/alert/store stub (`agentic/agentic_autonomy.py:~707`) | Before: abstention was an *implicit* threshold and the three‑state contract was **not** wired into detection. Now: an explicit, calibration‑grounded "don't‑know" gate + bounded response. |
| **Depict (visualize/explain)** | **Partial (understated) → first per‑event coupling** | Infra exists: `gui/visualization_dashboard.py` (Plotly), `narrative/` (`NarrativeEngine`, `ReasoningChainNarrative`, `MercuryResponse`), `explainability/` | This PR adds the first *per‑decision* depiction: `DecisionRecord.explain()` + a `signals`/`reasons`/`caveats` provenance trail. Full depiction remains a pillar. |
| **Deter (response/countermeasure)** | **Introduced by this PR** | New `decision/response.py` (`ResponsePolicy`, bounded `ResponseAction`); closes the loop to existing channels via `decision/bridge.py` (CAP 1.2 alerts `alerting/cap_generator.py`; `AgentAction` `agentic/agentic_autonomy.py`) | Bounded, **non‑destructive, fail‑closed, human‑in‑the‑loop** by construction — recommend/escalate, never autonomously destroy. |
| **Detail** | **Strengthened** | Certificates threaded through `detect_with_fusion` (`conformal`, `gosnn_metadata`, `symbolic_consistency`, `drift_detection`); audit trail in `decision/ledger.py` (`DecisionLedger`) + `decision/loop.py` (`DecisionLoop`) | The `DecisionRecord` turns these into an auditable, JSON‑safe record with the active policy attached; an append‑only, bounded, **O(1)‑summary**, thread‑safe, JSON‑persistable `DecisionLedger` makes a stream of decisions a queryable audit trail (the *verify* step). |

---

## 2. What PR #278 moved — and what it did not

PR #278 (calibrated/honest confidence) **strengthens `interpret`** and the
*"with accuracy"* qualifier, and lays a safety floor for autonomy. It did
**not**, on its own, add multi‑agent orchestration, new languages, new
modalities/models, depiction, deterrence, or a closed‑loop. Those are net‑new
build‑outs, each PR‑scale.

This PR (**Pillar A: Decision/Abstention + Response**) is now rebased on top of
**merged #278** and converts that groundwork into autonomy: it turns the
calibrated certificate into a closed **identify → interpret → decide → deter**
loop with an explicit "don't‑know" gate. It strengthens **autonomous**,
introduces **deter**, and adds the first per‑event **depict** coupling, without
touching detection accuracy.

**Integrating #278's new runtime signal.** #278 added one new key to the
`detect_with_fusion` result — `result["info_geometry_certificate"]` (per‑detector
Mahalanobis price level‑sets). The gate evaluated it and **deliberately treats
it as an exact no‑op**, on measured evidence: the certificate's own contract
certifies a *component's* boundary, "NOT the fused/gated verdict", and its
adaptive operating point is computed from the batch — so in the single‑sample
serve path the gate runs in it collapses to `price == threshold` (a point scored
4.1× over threshold in a batch of 9 reduces to exactly 1.0× at batch size 1),
carrying no information that could soundly refine the fused decision. The gate
already consumes the *authoritative* calibrated signal (the conformal set); the
empirical‑coverage test proves that guarantee survives the projection (zero
gate‑vs‑certificate contradictions, ~94% coverage at a 90% target).

## 3. Honest corrections to the index

1. **Language stack is Python + Rust, not Python + C.** No C sources exist
   (`rust_crypto/` is Rust via PyO3); PQC is the external `ama_cryptography`
   backend, gated fail‑closed at import (`_pqc_gate.py`).
2. **Dataset/detector counts were understated.** Real dataset references and
   detector classes both exceed the index figures.
3. **Depiction is "partial", not "minimal".** Visualization, narrative and
   explainability infrastructure already exist; what was missing was a
   *per‑decision* coupling — which this PR begins.
4. **Autonomy loops existed but were not closed.** The gap was an explicit
   abstention gate wired into detection and a bounded response layer — exactly
   this PR's scope.

## 4. Recommended sequence (after this PR)

In honest leverage order toward the autonomous goal:

- **(A) Decision/abstention + response layer — _this PR._**
- (B) Multi‑agent orchestration inside Mercury (planner/critic/executor over
  the existing engines).
- (C) Depiction layer (per‑event explanations/visuals from the certificates
  now threaded through `detect()` and the new `DecisionRecord`).
- (D) Decorrelated‑stream fusion — **executed and closed by #278**: the
  pre‑registered protocol ran end‑to‑end (SHIP rejected, runtime byte‑unchanged;
  `research/governed_fusion/FINDINGS.md`). Pushing detection itself positive now
  needs a different lever, not this one.

Everything above is aligned to the project's **Omnidirectional, STEM
exploration/discovery + humanitarian‑impact, open‑source, verifiable‑only**
principles: each row cites code, each claim is checkable, and the new layer is
non‑destructive and fail‑closed by construction.
