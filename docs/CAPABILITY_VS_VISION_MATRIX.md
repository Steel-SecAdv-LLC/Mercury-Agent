# Capability vs. Vision Matrix (code-grounded)

**Status:** verified against the source tree at branch point, not the README index.
Every "Verified today" cell cites `file:line` in `src/omni_mercury_engine/`. Where a
capability is **absent**, that is stated plainly — absence is a finding, not a gap to
paper over. This document is the planning baseline for the Decision / Abstention /
Response layer (pillar **a**) and is intended to stay honest as the system grows.

Guiding principles for every row: **Omnidirectional**, STEM exploration/discovery +
humanitarian impact, open-source, and **verifiable-only** (claims trace to runnable
code, measured numbers, or an explicit "not yet").

---

## 1. The matrix

| Vision axis | Vision (target) | Verified today (code-grounded) | Honest verdict | Addressed by pillar **a**? |
|---|---|---|---|---|
| **multi-domain** | Many domains, mostly real data | **Yes.** 15 domain loaders (`loaders/*.py`); 24 dataset modules (`datasets/*.py`); ~47 real ADBench datasets + a handful more real, ~5 with label-leakage, ~10 unreachable. | Real, breadth understated on detectors, dataset realness slightly overstated. | No (operates across them; not its job) |
| **neuro-symbolic** | Co-trained neural + symbolic, ethical gates | **Yes.** `core/neurosymbolic_hub.py` (`NeuralEncoder` 413‑582, `KnowledgeGraph` 263‑410, fused predict 1282‑1332); `core/learnable_gosnn.py` `LearnableGOSNN` (384‑841); `core/symbolic_reasoning.py` (74‑249). | Real co-training + hard ethical gates. No explicit LTN loss term. | No |
| **multi-model / multi-type** | Many detector/model types fused | **Yes (partial→strong).** 48 registered detectors (`core/detector_registry.py:209‑586`), ~57 detector classes in tree; fusion in `core/fusion.py`, `ml/fusion_network.py`. | Real and extensive; README "22+" understates. | No |
| **interpret w/ accuracy** | Calibrated, honest confidence | **Yes, strengthened.** Temperature scaling (`engine.py:3906`); conformal sets (`core/conformal_prediction.py` `BinaryConformalClassifier`/`BinaryPredictionSet` 994‑1156); `engine.score_fusion_conformal` (2321), `detect_with_fusion` `conformal` sub-dict (4019‑4026). PR #278 adds Beta‑MCA + decision curve (**open, unmerged**). | The genuine recent win. Calibration is real; abstention info exists but **unused at the decision**. | **Yes — consumed** |
| **multi-agent** | Planner/critic/executor loop | **No (scaffolding).** Roles/messages/strategies are type definitions (`cognitive/multi_agent_coordination.py`); `cognitive/orchestrator.py` integrates *cognitive components*, not agents; planner/reflexion classes exist but no running loop. | Design reified as types; not an operating multi-agent system. | No (pillar **b**) |
| **multi-language** | Python + C/Rust for PQC | **Partial.** Python core; Rust *classical* crypto with PyO3 (`rust_crypto/src/lib.rs`); PQC is an **external** dependency via the `_pqc_gate.py` import gate, not native C/Rust here. | Honest but narrower than the badge implies. | No |
| **autonomous (identify→deter→act)** | Closed identify→decide→act loop | **No closed loop.** `agentic/agentic_autonomy.py` runs an OODA-shaped loop (159‑207) but actions are `["flag_anomaly","escalate","suppress","investigate","log"]` (119) and `_execute_action` (735‑744) only logs/alerts/stores. `cognitive/autonomous_agent.py act()` needs an externally supplied executor. | Autonomous *analysis*, not autonomous *action*. | **Yes — this is the build** |
| **depict (visualize/explain)** | Per-event explanation/visuals | **Exists, decoupled.** `explainability/explainer.py` (SHAP/counterfactual), `narrative/engine.py`, `gui/visualization_dashboard.py`. Not auto-wired into `detect_*` outputs. | Real components, not threaded through detect(). | Partial — decisions carry provenance certificates the depiction layer (pillar **c**) can render |
| **deter (response/countermeasure)** | Active, proportionate response | **No.** Searched `src/`: only alerting (`alerting/cap_generator.py` CAP XML), advisory recommendations (`safeguards/nano_safeguards.py:592‑619`, `narrative/engine.py:644‑670`). No actuated, closed-loop response. | Definitively absent. | **Yes — this is the build** |
| **detail (provenance/audit)** | Verifiable trail per decision | **Partial.** Three-state honesty contract + ledgers exist for governance (`governance/contract.py` `GovernanceLedgerEntry`/`GovernanceRegistry`) and verifiers (`verifiers/registry.py`), but no per-**decision** audit record. | Honest substrate, no decision-level ledger. | **Yes — adds a decision/response ledger** |

---

## 2. The honesty substrate already in the tree (what pillar **a** builds on)

The repo already carries a clean, cross-cutting **abstention vocabulary** — this is
the single most important reuse, not a net-new invention:

- **`ThreeState{GROUNDED, UNAVAILABLE, UNDECIDABLE}`** — `verifiers/three_state.py:49‑77`.
  - `GROUNDED` — a decision was reached and the value carries it.
  - `UNAVAILABLE` — decidable in principle, but not produced this run (missing input,
    exhausted budget). The honest "don't-know-**yet**".
  - `UNDECIDABLE` — no decision procedure in principle. Registers nothing, ever.
- **Builders** `grounded()/unavailable()/undecidable()` and the `GovernanceScalar`
  dataclass — `governance/contract.py:257‑352`; ledger + registry 355‑473.
- **Calibrated confidence** — temperature-scaled `anomaly_prob` (`engine.py:3906`) plus
  **conformal label sets** whose `set_size ∈ {0,1,2}` already names the three outcomes
  that matter (`core/conformal_prediction.py:994‑1024`):
  - singleton `{1}`/`{0}` → confident anomaly/normal,
  - `{0,1}` → genuine uncertainty (**abstain**),
  - `{}` → atypical point neither class explains (novel / out-of-distribution).
- **Dual hard ethical gate, fail-closed** — `engine._enforce_ethics_at_boundary`
  (`engine.py:3372‑3456`): `BenevolenceScorer` (floor 0.70, threshold 0.99) +
  `SigmaImmutableGate`, both raising `EthicalConstraintViolationError`
  (`cognitive/ethical_bounding.py`).

**The gap (precise):** nothing maps calibrated confidence + conformal set →
a *typed decision with a first-class "don't-know"*, and nothing turns a grounded
decision into a *proportionate, reversible, ethically-gated response*. The
`is_anomaly` field is always a bare `anomaly_prob > threshold` (`engine.py:3942`) —
two outcomes, never "abstain", never a response. The conformal `abstain` bit is
computed and then **discarded** at the decision boundary.

---

## 3. What PR #278 actually did (so we plan against reality)

PR #278 ("Phase 2 governed fusion substrate") is **open and unmerged** at this branch
point (`mergeable_state: blocked`). It strengthens **interpret + "with accuracy"** and
lays a safety floor; it does **not** add multi-agent, languages, modalities, depiction,
deterrence, or a closed loop. Concretely it contributes (on its own branch):

- Beta‑MCA calibration map (`calibration_map="mca"`, additive `calibrated_probabilities`),
  default-off / byte-exact.
- `core/decision_curve.py` — net-benefit `NB(t)`, Bayes operating point `t* = c/(c+b)`,
  `reconciled_operating_point` (read-only; changes no runtime verdict).
- Conformal recorded as a **conclusive negative** for raw F1 (−0.071 vs the threshold it
  displaces) but **kept as an opt-in coverage-floor diagnostic** — exactly the signal an
  abstention gate should consume.

**Consequence for pillar a:** build on `main`'s merged calibrated surface
(`anomaly_prob` + conformal sets), and make the decision layer **forward-compatible** so
that when #278 lands it transparently prefers `calibrated_probabilities` /
`reconciled_operating_point` if present. No dependency on unmerged work.

---

## 4. Net assessment and sequencing

Of **identify / interpret / depict / deter / detail**, the tree today is strong on
*identify* and (post-#278 groundwork) *interpret-with-accuracy*, partial on *detail*,
and **missing** *depict-wired*, *deter*, and the closed **autonomous** loop.

Recommended sequence (unchanged from the audit, now code-justified):

- **(a) Decision / abstention + response layer** — convert calibrated confidence into a
  closed identify→interpret→**decide(with don't-know)**→**deter** loop. *This PR.*
- **(b) Multi-agent orchestration inside Mercury** — planner/critic/executor over the
  existing engines.
- **(c) Depiction layer** — render the per-decision certificates this layer threads.
- **(d) Decorrelated-stream fusion** — the one logged-but-untried lever to push raw
  detection positive (needs the live-API suite + a net-new detector).

Pillar **a** is first because it has the highest leverage toward the autonomous goal and
the lowest new-dependency risk: every input it needs already exists and is tested.
