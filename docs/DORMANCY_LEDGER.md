# Dormancy & Salvage Ledger

This ledger accounts for the **dormant** code in Mercury Agent — modules that are
defined and exported in the public API but never run in any live inference,
training, or benchmark path. It exists because the repository's standing rule is
anti-theater: a module's *interface* (or its appearance in a README feature
list) is **not** evidence it carries signal. Only a paired measurement on real
held-out labels is.

**Policy for this ledger (operator-directed): _nothing is deleted._** Every
dormant module is retained as a reference implementation. The work here is to (1)
identify, on **real labels**, which dormant code carries genuine signal and
revive it under the same ablation discipline as the rest of the repo, and (2)
rank the remainder by honest, remaining salvage value so future revival is
prioritised, not guessed.

## 0. The lesson up front: interface ≠ signal

An automated salvage survey rated `predictive_coding.py` the **highest-value**
revival candidate because it exposes a clean `fit()/detect()` anomaly-scorer
interface. Measured on real ADBench labels it scores at **chance**
(mean AUC 0.536; cardio 0.477 — *below* chance). The same survey rated
`case_based_reasoning.py` a medium candidate; measured, it is also at chance
(0.572). The one module that actually carries signal — a k-means distance
scorer buried in `neural_memory_layer.py` — was *not* the survey's top pick.

The harness `benchmarks/dormant_module_revival.py` is the gate that produced
these numbers; `artifacts/dormant_module_revival.json` is the record. The
methodology mirrors `benchmarks/neurosymbolic_ablation.py`: real ADBench labels
only, stratified splits, multiple seeds, fail-closed if data is unavailable.

## 1. Measured revival — standalone anomaly AUC on real ADBench

Five datasets × three seeds. `lof_reference` is the live ensemble's own
distance/density detector (`detectors.spatial`), included so salvage is judged
against what already ships, not merely against chance.

| Candidate (orphaned source) | mean AUC | per-dataset (cardio/thyroid/breastw/WBC/Pima) | verdict |
|---|---|---|---|
| **`kmeans_distance`** (`neural_memory_layer.KMeansClusterer`) | **0.861** | 0.762 / 0.945 / 0.982 / 0.981 / 0.634 | **carries real signal → revived** |
| `predictive_coding` (`PredictiveCodingDetector`) | 0.536 | 0.477 / 0.528 / 0.528 / 0.608 / 0.537 | archive — no signal |
| `case_based_knn` (`CaseBasedReasoner`, supervised) | 0.572 | 0.689 / 0.633 / 0.534 / 0.500 / 0.505 | archive — no signal |
| `lof_reference` (live `detectors.spatial`) | 0.661 | 0.509 / 0.669 / 0.611 / 0.928 / 0.586 | (baseline) |

**Only `kmeans_distance` clears the standalone-signal bar (mean AUC ≥ 0.70)** —
and it beats the ensemble's own distance detector. It is therefore promoted to a
first-class detector, `detectors/kmeans_distance.py` (`KMeansDistanceDetector`),
which wraps the dormant `KMeansClusterer` and emits per-centroid distances as a
fusion feature group.

### Does it *add* to the fused ensemble?

Standalone signal is necessary but not sufficient — the ensemble already ships a
distance detector, so the revived detector earns a place in the **default** set
only if it improves the **fused** AUC. That is settled by the marginal ablation
`benchmarks/kmeans_ensemble_marginal.py` (`artifacts/kmeans_ensemble_marginal.json`),
which trains the fusion model with and without the detector from the same split.

> **Marginal verdict: HOLD — keep optional, do not add to the default ensemble.**
> 5 datasets × 3 seeds: mean fused ΔAUC **−0.0009** (cardio +0.0020 3/3, thyroid
> +0.0006, breastw −0.0014, WBC −0.0017, Pima −0.0040), seed agreement 0.53 —
> within the ±0.002 noise floor. The detector is strong *standalone* but
> **redundant with the live `spatial` detector** inside the fusion ensemble, so
> it does not move the fused score. Honest outcome: the dormant clusterer is
> genuinely revived as a tested, first-class detector and is available to opt in
> (`engine.detectors["kmeans_distance"] = KMeansDistanceDetector()`), but it is
> **not** enabled by default — adding a redundant detector to ship would be the
> bloat the anti-theater rule forbids.

## 2. Precedence ranking of the remaining dormant modules

For the modules that do **not** expose a per-sample anomaly score over tabular
features, there is no honest in-repo detection metric to revive them against
today. They are ranked by *remaining salvage value* — the plausibility and cost
of producing **some** honest measurable signal — so revival effort is spent in
order. None are deleted.

| Rank | Module | LOC | What it is | Honest measurable signal? | Salvage | Revival path |
|---|---|---|---|---|---|---|
| 1 | `symbolic_logic_layer.py` | 1127 | Forward-chaining rule reasoner (crisp) | **MEASURED ✓** — its `ThresholdRule` idea revived as a *differentiable* salience rule and ablated: consensus_salience +0.0022 vs consensus +0.0009 low-data ΔAUC, seed agreement 0.81 vs 0.63 — directionally better but +0.0013 sub-threshold | **done (KEEP consensus)** | Revived as the `consensus_salience` rule graph (`NEUROSYMBOLIC.md` §2.3); live, tested, selectable; the most promising symbolic follow-up, awaiting a larger-N confirmation. |
| 2 | `causal_discovery.py` | 1442 | Causal-graph discovery | **VALIDATED ✓ (non-AUC)** — not an anomaly scorer, but on its *own* metric (skeleton recovery vs a known SEM) it recovers structure well above chance: mean F1 **0.853** vs chance 0.286, degrading gracefully as samples thin (`benchmarks/causal_discovery_validation.py`) | **revived (causal tool)** | A genuinely working constraint-based causal-discovery engine; revived and measured as a causal tool, not an anomaly detector. |
| 3 | `explainability.py` | 1033 | IG / SHAP / LIME explainers + faithfulness evaluator | **VALIDATED ✓ (non-AUC, dep-free parts)** — IntegratedGradients recovers a model's informative features (recovery@3 **0.678** vs chance 0.300) and is ~2× more faithful than random (comprehensiveness **0.40 vs 0.22**); the `FaithfulnessEvaluator` works (`benchmarks/explanation_fidelity.py`). SHAP/LIME need the optional `shap`/`lime` libs (not installed) | **revived (IG + evaluator)** | Dep-free IntegratedGradients explainer + faithfulness evaluator revived & measured; SHAP/LIME remain dependency-gated. |
| 4 | `formal_verification.py` | 1591 | Constraint solvers + interval-bound-propagation verifier | **VALIDATED ✓ (non-AUC)** — its `IntervalBoundPropagator` is **100% sound** over 200 random ReLU nets / input boxes (the certificate always contains the densely-sampled true output range) with ~1.8× tightness, non-vacuous (`benchmarks/formal_verification_soundness.py`) | **revived (verifier)** | A genuinely sound interval-bound-propagation verifier; revived and measured as a certified-bounds tool. |
| 5 | `neurosymbolic_hub.py` + `gosnn_3r_integration.py` + `fibring_fusion.py` | 1602+906+273 | Alternative GOSNN/fibring fusion head | **Maybe** — fused AUC vs the live `OmniFusionModel` | LOW | Wire as an alternative fusion head → ablate; high effort, likely redundant with the trained `OmniFusionModel`. |
| 6 | `knowledge_graph.py` + `multi_hop_reasoner.py` | 2109+718 | Symbolic KB + multi-hop reasoning | **No (numeric)** — operate on symbolic facts, not feature vectors | LOW | Only via a rules/KB bridge to the symbolic constraint; no direct tabular signal. |
| 7 | `neural_memory_layer.py` (remainder) | 941 | Text/dict memory + pattern detection (the `KMeansClusterer` within is revived in §1) | **No (beyond the clusterer) ✓ checked** — `get_anomaly_score()` is *the same* k-means-distance path already revived; the memory/embedding path is hash-projection over dicts, not tabular | LOW | The salvageable part (`KMeansClusterer`) is revived; the rest is a text-memory system. |
| 8 | `predictive_coding.py` | 1296 | Predictive-coding / active-inference detector | **Measured — none** (0.536 AUC) | LOW | No revival path as a detector; retain as reference. |
| 9 | `case_based_reasoning.py` | 625 | Case-based retrieval reasoner | **Measured — none** (0.572 AUC) | LOW | No revival path as a detector; retain as reference. |
| 10 | `chain_of_thought.py` / `chain_of_hindsight.py` / `reflexion.py` | 1501/1548/1734 | LLM-style reasoning / self-reflection loops | **No** — generate text reasoning traces; no in-repo ground truth | LOW | Out of scope for anomaly detection; reference only. |
| 11 | `hierarchical_planning.py` / `multi_agent_coordination.py` / `plasticity_engine.py` | 1489/1293/953 | Planning / agent coordination / synaptic plasticity | **No** — control/meta machinery, not detectors | LOW | Out of scope for anomaly detection; reference only. |
| — | `differentiable_logic.py` | 988 | Scalar t-norms + embedding LTN modules | **Superseded** — its t-norm taxonomy (Gödel/Łukasiewicz) is revived as real tensor operators in the live `SymbolicConstraintModule` semantics (see `docs/NEUROSYMBOLIC.md` §2.1) | n/a | Concept revived in the measured path; file retained as reference. |

## 3. Correctly-quarantined (not dormant theater)

`space/schumann_resonance.py` and `models/parapsychology.py` run untrained
networks but **fall back** to deterministic physics / a neutral prior with a
one-time warning (per `docs/NEUROSYMBOLIC.md` §4). They emit no fabricated
signal and are correctly handled — no action.

## 4. Status & reproduce

```bash
# Which dormant modules carry standalone signal on real labels?
python -m benchmarks.dormant_module_revival \
    --datasets cardio thyroid breastw WBC Pima --seeds 0 1 2 \
    --out artifacts/dormant_module_revival.json

# Does the one that does (k-means distance) add to the fused ensemble?
python -m benchmarks.kmeans_ensemble_marginal \
    --datasets cardio thyroid breastw WBC Pima --seeds 0 1 2 \
    --out artifacts/kmeans_ensemble_marginal.json
```

Revival is data-driven and incremental: a dormant module is promoted only when a
pre-registered bar on real held-out labels is cleared, exactly as for the
neuro-symbolic constraint. Everything else is retained, ranked, and awaiting a
measurable signal — not deleted, and not asserted to work until it is shown to.

## 5. The revival frontier (honest boundary)

After measuring every dormant module that can produce a per-sample score on
tabular features, **the ADBench-AUC-measurable revival surface is effectively
exhausted**:

* **Standalone detectors:** `kmeans_distance` was the only real signal (revived;
  redundant in the ensemble, so kept opt-in). `predictive_coding` and
  `case_based_reasoning` measured at chance. `neural_memory_layer.get_anomaly_score`
  *is* the k-means path. Nothing else exposes a tabular anomaly score.
* **Constraint enrichment:** the crisp t-norm semantics (§2.2) and the salience
  rule graph (§2.3) are both measured; both are genuine, selectable, and
  currently sub-threshold (product / consensus retained).

The remaining dormant modules **do not speak the AUC metric** — `causal_discovery`
emits a causal graph, `explainability` emits feature attributions,
`formal_verification` emits satisfiability proofs, and the planning / reasoning /
coordination machinery operates on symbolic or text inputs. So each is revived
(or not) against the **right** measurement harness, never stretched onto a
detection metric it was never meant to clear:

* **Causal recovery — built ✓, VALIDATED.** `benchmarks/causal_discovery_validation.py`
  measures `causal_discovery` against a known synthetic linear-Gaussian SEM:
  skeleton **F1 0.853 vs chance 0.286**, high precision throughout with recall
  degrading gracefully as samples thin. A genuinely working causal tool — revived
  and measured *as a causal engine*, not an anomaly detector. This is the template
  for the rest.
* **Explanation fidelity — built ✓, VALIDATED.** `benchmarks/explanation_fidelity.py`
  trains a real model on data with a *known* informative-feature set and scores
  the dependency-free `IntegratedGradientsExplainer` + `FaithfulnessEvaluator`:
  recovery@3 **0.678 vs chance 0.300**, comprehensiveness **0.40 vs 0.22** random.
  The IG explainer and the faithfulness evaluator are revived and measured; the
  `shap`/`lime` explainers stay dependency-gated until those libs are installed.
* **Formal soundness — built ✓, VALIDATED.** `benchmarks/formal_verification_soundness.py`
  checks `formal_verification.py`'s `IntervalBoundPropagator` against densely-sampled
  ground truth over random ReLU networks: **100% sound** (the certificate always
  contains the true output range across 200 cases), ~1.8× tightness — a genuinely
  sound, non-vacuous certified-bounds verifier.

All three non-AUC frameworks land the same verdict that the AUC lens missed:
measured against the **right** metric, `causal_discovery`, `explainability`
(IntegratedGradients + faithfulness), and `formal_verification` (interval bound
propagation) are **genuinely working tools** — not theatre. The remaining
orphans (planning / reasoning / coordination / multi-agent machinery) operate on
symbolic or text inputs with no in-repo ground truth; they stay ranked and
retained until a fitting harness exists.

Building these harnesses — not fabricating a metric win — is how the rest of the
dormant subsystem earns its place. Until a module's harness exists and it clears
the bar, the module stays ranked and retained, not revived and not deleted.

## 6. Consolidation — the deliberate stopping point

The measurable revival frontier is now well covered: three modules revived on AUC
(adaptive co-training, the salience rule, the k-means detector) and three on their
own metrics (causal recovery, explanation fidelity, formal soundness). The
**reasoning / planning / coordination tier** — `reflexion`, `chain_of_thought`,
`chain_of_hindsight`, `hierarchical_planning`, `multi_agent_coordination`,
`plasticity_engine`, `knowledge_graph`, `multi_hop_reasoner` (rows 6, 10, 11) — is
explicitly marked **retained, no honest in-repo metric**. These operate on
symbolic or text inputs with no ground truth in this repository; the only way to
"measure" them would be to invent an arbitrary toy task, and a fabricated task is
the theatre this ledger exists to prevent. They are kept as reference
implementations, not deleted, and not asserted to work.

This is a deliberate stopping point, not an omission: every module with an honest,
non-contrived measurement has one; the rest wait — ranked and retained — for a
real task to measure them against, should one arise.
