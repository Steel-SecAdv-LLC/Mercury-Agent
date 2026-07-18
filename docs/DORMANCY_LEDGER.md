# Dormancy & Salvage Ledger

Applies to Mercury Agent **v2.1.x**. Last updated: 2026-07-14.

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
rank the remainder by auditable, remaining salvage value so future revival is
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
> it does not move the fused score. Transparent outcome: the dormant clusterer is
> genuinely revived as a tested, first-class detector and is available to opt in
> (`engine.detectors["kmeans_distance"] = KMeansDistanceDetector()`), but it is
> **not** enabled by default — adding a redundant detector to ship would be the
> bloat the anti-theater rule forbids.

## 2. Precedence ranking of the remaining dormant modules

For the modules that do **not** expose a per-sample anomaly score over tabular
features, there is no provenance-safe in-repo detection metric to revive them against
today. They are ranked by *remaining salvage value* — the plausibility and cost
of producing **some** independently measurable signal — so revival effort is spent in
order. None are deleted.

| Rank | Module | LOC | What it is | Independently measurable signal? | Salvage | Revival path |
|---|---|---|---|---|---|---|
| 1 | `symbolic_logic_layer.py` | ~1,100 | Forward-chaining rule reasoner (crisp) | **MEASURED ✓** — its `ThresholdRule` idea revived as a *differentiable* salience rule and ablated: consensus_salience +0.0022 vs consensus +0.0009 low-data ΔAUC, seed agreement 0.81 vs 0.63 — directionally better but +0.0013 sub-threshold | **done (KEEP consensus)** | Revived as the `consensus_salience` rule graph (`NEUROSYMBOLIC.md` §2.3); live, tested, selectable; the most promising symbolic follow-up, awaiting a larger-N confirmation. |
| 2 | `causal_discovery.py` | ~1,420 | Causal-graph discovery | **VALIDATED ✓ (non-AUC)** — not an anomaly scorer, but on its *own* metric (skeleton recovery vs a known SEM) it recovers structure well above chance: mean F1 **0.853** vs chance 0.286, degrading gracefully as samples thin (`benchmarks/causal_discovery_validation.py`) | **revived (causal tool)** | A genuinely working constraint-based causal-discovery engine; revived and measured as a causal tool, not an anomaly detector. |
| 3 | `explainability.py` | ~900 | IG / SHAP / LIME explainers + faithfulness evaluator | **VALIDATED ✓ (non-AUC, dep-free parts)** — IntegratedGradients recovers a model's informative features (recovery@3 **0.678** vs chance 0.300) and is ~2× more faithful than random (comprehensiveness **0.40 vs 0.22**); the `FaithfulnessEvaluator` works (`benchmarks/explanation_fidelity.py`). SHAP ships via the `[explainability]` extra; `lime` is excluded from the extras graph (its sole release cannot build on modern setuptools) and the LIME adapter falls back to the in-repo `_approximate_lime` linear surrogate | **revived (IG + evaluator)** | Dep-free IntegratedGradients explainer + faithfulness evaluator revived & measured; SHAP activates with the extra, LIME via manual install or the in-repo surrogate. |
| 4 | `formal_verification.py` | ~1,540 | Constraint solvers + interval-bound-propagation verifier | **VALIDATED ✓ (non-AUC)** — its `IntervalBoundPropagator` is **100% sound** over 200 random ReLU nets / input boxes (the certificate always contains the densely-sampled true output range) with ~1.8× tightness, non-vacuous (`benchmarks/formal_verification_soundness.py`) | **revived (verifier)** | A genuinely sound interval-bound-propagation verifier; revived and measured as a certified-bounds tool. |
| 5 | `neurosymbolic_hub.py` + `gosnn_3r_integration.py` + `fibring_fusion.py` | ~1,780+~880+~270 | Alternative GOSNN/fibring fusion head | **Maybe** — fused AUC vs the live `OmniFusionModel` | LOW | Wire as an alternative fusion head → ablate; high effort, likely redundant with the trained `OmniFusionModel`. |
| 6 | `knowledge_graph.py` + `multi_hop_reasoner.py` | ~2,080+~720 | Symbolic KB + multi-hop reasoning | **No (numeric)** — operate on symbolic facts, not feature vectors | LOW | Only via a rules/KB bridge to the symbolic constraint; no direct tabular signal. **Algorithmic correctness now behaviourally covered (2026-06-02):** `tests/cognitive/test_knowledge_graph_behavioral.py` asserts embedding recovery on a known two-cluster graph (intra > inter cosine over 5 seeds), GNN message passing, link-prediction recovery of a held-out intra-cluster edge, and transitive / symmetric inference — the reference methods compute what they claim even though the surface carries no anomaly-detection signal. |
| 7 | `neural_memory_layer.py` (remainder) | ~900 | Text/dict memory + pattern detection (the `KMeansClusterer` within is revived in §1) | **No (beyond the clusterer) ✓ checked** — `get_anomaly_score()` is *the same* k-means-distance path already revived; the memory/embedding path is hash-projection over dicts, not tabular | LOW | The salvageable part (`KMeansClusterer`) is revived; the rest is a text-memory system. |
| 8 | `predictive_coding.py` | ~1,250 | Predictive-coding / active-inference detector | **Measured — none** (0.536 AUC) | LOW | No revival path as a detector; retain as reference. |
| 9 | `case_based_reasoning.py` | ~610 | Case-based retrieval reasoner | **Measured — none** (0.572 AUC) | LOW | No revival path as a detector; retain as reference. **CBR cycle now behaviourally covered (2026-06-02):** `tests/cognitive/test_case_based_reasoning_behavioral.py` asserts retrieval ranking + counters, the REUSE-vs-REVISE branch in `solve` (incl. explicit `no_matching_cases` on an empty base), proportional `adapt` + adaptation history, and `learn_from_outcome` state updates. |
| 10 | `chain_of_thought.py` / `reflexion.py` | ~1,510/~1,820 | Reasoning traces / self-reflection loops | **MEASURED ✓ (non-AUC, 2026-06-11)** — wired into the live multi-agent orchestration (`agentic/orchestration.py`) and measured on real ADBench labels by `benchmarks/orchestration_validation.py`: chain-of-thought **trace fidelity 600/600** (every sampled decision's stated determination matches the issued decision; every quoted score is the real consensus score), reflexion **paired Δ balanced-accuracy +0.054** over a fixed operating point (15/15 dataset×seed runs acted, never harmed a well-calibrated point) | **revived (orchestration tier)** | Reflexion's original `fn > 2·fp` adaptation rule was itself measured **harmful** (Δ −0.071; WBC 0.98 → 0.50) and replaced by an evidence-grounded balanced-accuracy sweep with minimum-evidence and hysteresis guards — the harness caught a real defect before it shipped. CoT conclusions previously classified against hardcoded 0.7/0.4 bands regardless of the issuing boundary (fixed: traces classify at the decision threshold); the self-consistency strategy returned the vote *token* as its conclusion, stripping the human-readable determination (fixed). |
| 10b | `chain_of_hindsight.py` | ~1,500 | Hindsight relabeling / credit assignment | **No** — no in-repo harness yet | LOW | Retained as reference; candidate for the same orchestration-loop treatment (its `FeedbackProcessor` is the natural batch-level critic). |
| 11 | `hierarchical_planning.py` / `multi_agent_coordination.py` | ~1,480/~1,330 | Planning / agent coordination | **MEASURED ✓ (non-AUC, 2026-06-11)** — same harness: the planner drives every live detection episode via real options bound to the pipeline stages — **executability 129/129 episodes** with TD value learning on real stage rewards (initial-state value strictly increasing); coordination forms per-sample consensus over the five real engine detectors — **mean consensus AUC 0.841 ≥ mean member AUC 0.833** (best member 0.901; no claim of beating the trained fusion model is made), below-quorum cases **abstain explicitly** | **revived (orchestration tier)** | Both modules carried blocking defects while dormant: the planner could not select options at all (its option library returned dict projections that the planner type-checked away — every plan shipped empty, every action fell back to `default_action`), option eviction never fired, and below-quorum consensus returned a silent benign verdict (fail-open) while duck-typed dict votes were silently dropped. All fixed and pinned by `tests/cognitive/test_orchestration_behavioral.py`. |
| 11b | `plasticity_engine.py` | ~930 | Synaptic plasticity | **No** — control/meta machinery, no harness yet | LOW | Retained as reference. |
| — | `differentiable_logic.py` | ~990 | Scalar t-norms + embedding LTN modules | **Superseded** — its t-norm taxonomy (Gödel/Łukasiewicz) is revived as real tensor operators in the live `SymbolicConstraintModule` semantics (see `docs/NEUROSYMBOLIC.md` §2.1) | n/a | Concept revived in the measured path; file retained as reference. |

## 3. Correctly-quarantined (not dormant theater)

`space/schumann_resonance.py` and `models/parapsychology.py` run untrained
networks but **fall back** to deterministic physics / a neutral prior with a
one-time warning (per `docs/NEUROSYMBOLIC.md` §4). They emit no fabricated
signal and are correctly handled — no action.

`models/affective.py` joined this set on 2026-06-11: the stub previously
emitted **fresh RNG noise per call** as its features and anomaly scores
(64 columns of fabricated, nondeterministic signal in the fusion feature
set — the worst variant of interface ≠ signal). Generic (off-modality)
input now receives a deterministic neutral output (zero features, 0.5
scores) with a one-time warning; the serve-path determinism and
checkpoint-equivalence tests (`tests/test_fusion_checkpoint_roundtrip.py`)
keep it regression-checked. Since v2.1.x the model additionally has a real
**declared-modality** predict path: a dict carrying an emotion-probability
time series under the `"emotions"` key is analysed deterministically
(temporal aggregation + the documented entropy/negative-affect distress
heuristic from `core/enhanced_model_domains.py`), with malformed declared
input failing loud (`tests/models/test_affective_model.py`). The fusion
feature group stays constant zeros for every input — the shipped
`default_fusion.pt` was trained against exactly that group; a learned
extractor still requires a real labelled affect corpus.

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

# Does the revived planning/coordination/reflexion/chain-of-thought tier
# carry real signal on the engine's own task? (rows 10-11)
python -m benchmarks.orchestration_validation \
    --datasets cardio thyroid breastw WBC Pima --seeds 0 1 2 \
    --out artifacts/orchestration_validation.json
```

Revival is data-driven and incremental: a dormant module is promoted only when a
pre-registered bar on real held-out labels is cleared, exactly as for the
neuro-symbolic constraint. Everything else is retained, ranked, and awaiting a
measurable signal — not deleted, and not asserted to work until it is shown to.

## 5. The revival frontier (measurement boundary)

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
  `shap` explainer activates with the `[explainability]` extra; the LIME
  adapter uses the real `lime` library only if manually installed (it is
  excluded from the extras graph as unbuildable on modern setuptools) and
  otherwise runs its in-repo linear surrogate.
* **Formal soundness — built ✓, VALIDATED.** `benchmarks/formal_verification_soundness.py`
  checks `formal_verification.py`'s `IntervalBoundPropagator` against densely-sampled
  ground truth over random ReLU networks: **100% sound** (the certificate always
  contains the true output range across 200 cases), ~1.8× tightness — a genuinely
  sound, non-vacuous certified-bounds verifier.

All three non-AUC frameworks land the same verdict that the AUC lens missed:
measured against the **right** metric, `causal_discovery`, `explainability`
(IntegratedGradients + faithfulness), and `formal_verification` (interval bound
propagation) are **genuinely working tools** — not theatre.

* **Orchestration tier — built ✓, VALIDATED (2026-06-11).** The ledger held the
  planning / reasoning / coordination tier "retained until a fitting harness
  exists" — and refused to invent a toy task for it. The fitting task arrived
  as the engine's own: `agentic/orchestration.py` (`MultiAgentOrchestrator`)
  wires `hierarchical_planning` (planner), `multi_agent_coordination`
  (executor), `reflexion` (critic), and `chain_of_thought` (depictor) into one
  planner/critic/executor loop over the **live detector ensemble**, and
  `benchmarks/orchestration_validation.py` measures each module against its
  own pre-registered bar on real ADBench labels (5 datasets × 3 seeds):
  consensus preserves member signal (mean AUC **0.841 vs 0.833**, fail-closed
  abstention below quorum), the planner drives **129/129** episodes to goal
  completion with real TD value learning, reflexion's feedback loop improves
  the operating point by **+0.054** paired balanced accuracy (and its original
  adaptation rule, measured harmful at **−0.071**, was corrected to an
  evidence-grounded sweep before shipping), and every one of **600** sampled
  reasoning traces states exactly the decision the pipeline issued. The four
  modules are live in `MultiAgentOrchestrator` and reachable from the engine
  via `OmniMercuryEngine.enable_multi_agent_orchestration()`.

Building these harnesses — not fabricating a metric win — is how the rest of the
dormant subsystem earns its place. Until a module's harness exists and it clears
the bar, the module stays ranked and retained, not revived and not deleted.

## 6. Consolidation — the measurement boundary today

The measurable revival frontier now covers: three modules revived on AUC
(adaptive co-training, the salience rule, the k-means detector), three on their
own metrics (causal recovery, explanation fidelity, formal soundness), and four
on the orchestration harness (`hierarchical_planning`,
`multi_agent_coordination`, `reflexion`, `chain_of_thought` — rows 10/11,
2026-06-11). The remaining tier — `chain_of_hindsight`, `plasticity_engine`,
`knowledge_graph`, `multi_hop_reasoner` (rows 6, 10b, 11b) — stays explicitly
marked **retained, no provenance-safe in-repo metric yet**. The orchestration loop is
the natural future harness for `chain_of_hindsight` (batch-level credit
assignment over episode history); the others still lack a non-contrived task.
They are kept as reference implementations, not deleted, and not asserted to
work.

The stopping point remains principled, not an omission: every module with an
independent, non-contrived measurement has one; the rest wait — ranked and retained
— for a real task to measure them against, exactly as rows 10/11 did until the
orchestration task arose.

## 7. Standing fitness substrate — fusion-marginal ablation ledger

`benchmarks/dormant_module_revival.py` is the **one-off** revival harness:
operator-triggered, ADBench-corpus-only, results frozen in
`artifacts/dormant_module_revival.json`. Phase 1 of the governed recursive
self-improvement work introduces the **standing** complement to that
harness — `research/governed_fusion/measure_marginal_ablation.py`. It
measures the per-component leave-one-out lift of the default fusion stack
(`resonance`, `kinematic`, `info_geo`) on the *transparent fitness subset* of
the governed-fusion live suite — the audited externally-labelled events
only — and appends one record per CI run to
`research/governed_fusion/ablation_ledger.json`. The CI workflow
`.github/workflows/ablation-ledger.yml` runs it on every PR and nightly.

This is what closes the measurement-to-revival loop for live-API
domains: a future detector promoted via the `engine.py` registration
seam earns a ledger entry the first time it appears in the fusion stack;
a future component whose lift drifts to zero across consecutive ledger
entries is the candidate for retirement. The Phase 3 recurring
dormant-revival job (`.github/workflows/phase3-governance.yml`) runs the real
revival benchmark on schedule and routes revival candidates through the Phase 2
promotion gate.

See `docs/SELF_IMPROVEMENT_LOOP.md` for the full rollout narrative and
the scope boundaries between implemented Phases 1–3 and deferred Phases 4–8.

## 8. Transparently-labelled heuristics (interface claims tightened, not pruned)

These modules expose interfaces whose docstrings previously implied empirical or
algorithmic grounding they do not have. Per the anti-theater rule they are
**retained** but their claims are now scoped in-code so no caller mistakes a
heuristic for a measurement. None drives a safety-critical decision uncalibrated.

| Module | Claim before | Reality (now labelled) |
|---|---|---|
| `scaling/bain_ai_scaling.py` `estimate_power_consumption` | "typical power profiles from hyperscaler deployments" | Uncalibrated order-of-magnitude heuristic; coefficients are named, illustrative constants (`BASE_POWER_W`, …). `_op_hyperion` now returns `calibrated: False`. Relative comparison only. |
| `core/global_omni_scalar_network.py` (~82 ISO-25010/Halstead/DORA/SLSA/… scalars) | declared diagnostic scalars | Static placeholder weights, never computed (no collectors exist); filtered from every operational path. Kept for registration/reporting only. |
| `core/global_omni_scalar_network.py` `MultiHeadAttentionFusion` | "32-head attention" learned fusion | **Revived — serving after the constant-input defect was fixed (2026-07-16).** The degeneracy had a single root cause: detector/model anomaly scores reach the engine as `torch.Tensor` (`_normalize_scores` always returns a tensor), but the GOSNN base-scalar builder filtered on `isinstance(score, (np.ndarray, float, int))` and silently dropped every one, so the fusion's per-call base member was always empty and its input constant (the recorded **1 unique state list across 403 calls**). Coercing tensor scores to floats gives `fuse()` genuine per-call input; the write-only enhanced-scalars re-registration (which counted detector scores as critical ethical anchors and collapsed the σ_Immutable floor once populated) was removed. Re-run of `scripts/train_gosnn_fusion.py` on a real 3-dataset ADBench harvest now measures **450 unique state lists of 450** and the merit gate passes (`multitask`: held-out masked-member MSE 4.65e6 ≤ 0.95×reference 8.30e6, fused std 647 ≥ 0.20×member std 3082 — no collapse), so `gosnn_attention_fusion.pt` ships (`artifacts/gosnn_fusion.eval.json`, provenance sidecar). The fused output feeds observability only — `fusion_score`/`intelligence_contribution` now vary per call while `anomaly_prob` and the σ_Immutable verdict are unaffected. Pinned by `tests/core/test_attention_fusion_gate.py` (seam + artifact/package agreement) and `tests/core/test_gosnn_fusion_per_call_variation.py` (per-call variation + gate safety). **Consequential role evaluated and DEFERRED — measured no gain (2026-07-17).** A consequential channel now exists end-to-end (detection head over the fused state → `gosnn_metadata["detection"]` → the decision layer's abstention-only disagreement overlay, `DecisionPolicy.defer_on_gosnn_disagreement`), and `scripts/train_gosnn_fusion.py` was rebuilt to train fusion+head jointly on **labelled ADBench outcomes** (692/692-unique harvested production calls, ground-truth labels of the exact detected rows) behind a detection-metric merit gate. The gate **refused, and the refusal is decisive — dead-by-construction, not under-powered**. On a 9-dataset label-stratified harvest (1,469 labelled calls, 321 anomalies; held-out test pool 224 calls / 50 positives) with properly tuned training (validation AUC 0.862 — the optimizer is not the confound), the learned fused-state head measured held-out AUC **0.801**. The decisive baseline is the engine's OWN calibrated `anomaly_prob` — the verdict a disagreement overlay would second-guess — which separates the same held-out anomalies at **0.961**; the phi-reference-fusion head is 0.845 and the raw mean detector score 0.816. Bootstrap (2,000 dataset-stratified paired replicates) of learned − strongest-baseline is P5 **−0.201** / median −0.149: the head is decisively *worse* than the engine it would contradict, so on every disagreement the engine is the one more often right and demoting there removes net-correct verdicts (the threshold search independently returns no viable pair). A direct linear/MLP head on the raw detector-score member does better (0.886/0.904) — but still short of 0.961: the engine's OmniFusionModel already fuses those detector scores better than any re-weighting of them can, so a GOSNN-input consequential channel is **dead-by-construction**, not an architecture-tuning problem. Mechanistic root (measured 2026-07-17): of the 9 members the fusion receives on the detect path, only member 0 (the per-call detector scores) varies across calls (across-call std 2.47); the other 8 (static registry group scalars — ETHICAL/SECURITY/COSMIC/…) are bit-constant (std at machine epsilon). The "fusion" is therefore one signal averaged with eight constants, which is exactly why attention+mean-pool underperforms a probe on member 0. No per-call detection signal is recoverable from the 8 dead members, so detection-science investment belongs in the detector tier / OmniFusionModel, not GOSNN.

**GOSNN's resolved, correct role (audited 2026-07-17 — not "only losses").** GOSNN has two legitimate, functionally beneficial jobs on the detect path, and both were verified rather than assumed:
* **Consequential — the σ_Immutable ethical gate.** GOSNN aggregates the full 127-scalar operational vector into the hard ethical boundary (deterministic floor authoritative, learned score advisory). This is GOSNN's correct high-value use: a whole-system scalar aggregator feeding an ethics verdict. Signal-integrity hardening this pass: `detect_with_fusion` collected that 127-scalar vector **twice** per call (once for GOSNN's advisory gate inside `get_enhanced_scalars`, once for the authoritative σ gate) — two independent snapshots that a concurrent registration could make diverge. The enhancement now carries the snapshot it scored (`EnhancementResult.collected_scalars`) and the authoritative gate reuses it: one registry walk instead of two per call, and the advisory and authoritative gates provably evaluate the identical vector. σ score and verdict bit-identical (pinned by `tests/core/test_gosnn_scalar_snapshot_reuse.py`).
* **Observability — real signal, not theatre.** Measured on 120 real cardio detect calls: the surfaced `enhancement_fusion_score` (AUC-vs-label 0.878) and `intelligence_contribution` (0.836) genuinely track anomalousness and are only weakly correlated with `anomaly_prob` (−0.33 / −0.27), so — unlike the retired `harmonic_synergy` dead constant — they are kept. Honest caveat recorded: that signal is member-0-derived (detector scores), so it is informative but **not independent** of what the engine already fuses; it is a transparent observability lens, correctly never routed into a decision.
So GOSNN leaves this PR confined to exactly the two things it is measurably good at — ethical aggregation (consequential) and live observability — with the detection-model misuse refused on evidence, the double-collection waste cut, and nothing left surfacing as theatre. Gate hardening this pass: the engine's `anomaly_prob` is now a **mandatory merit baseline** (a disagreement head must out-separate the verdict it guards, not merely beat the phi/mean reference fusions), closing a real hole — the head clears phi/mean with bootstrap support and, against those strawmen alone, would have shipped a cross-check weaker than the engine. No detection head ships, the overlay stays structurally inert (`consequential.shipped=false` in `artifacts/gosnn_fusion.eval.json`, pinned by `test_consequential_verdict_matches_shipped_head`), and observability-only is the ratified, evidence-backed posture. Also in this pass: train/serve stacks unified in `core/attention_fusion_stack.py` (the serve-only harmonic-synergy output modulation — measured a constant ×1.0118 — is removed, so the shipped weights now serve under exactly the semantics their gate verdict measured), and the dead-constant `harmonic_synergy` metric (bit-identical 0.618034 on 15/15 production calls; FFT conjugate-symmetry pins its ratio at 1.0) is no longer surfaced in detection results. |
| `security/sigma_immutable_gate.py` + `core/global_omni_scalar_network.py` `EthicalGate` | learned σ_Immutable ethical gate | **Synthetic-trained; advisory only — the deterministic floor is the authoritative gate.** The network is trained by `scripts/train_sigma_immutable.py` on a synthetic corpus (generated scalar vectors labelled by a threshold rule over the ETHICAL group), not real ethical outcomes; measured alone it passed vectors with a critical ethical dimension zeroed (benevolence → 0). Production composes the deterministic critical-ethical floor (`SigmaImmutableGate.enforce_ethical_floor`) BEFORE the learned score at every boundary — a collapsed anchor is a categorical fail-closed refusal no learned score can override. Documented in both docstrings (2026-07-17). Real-data training requires a labelled ethical-outcome corpus that does not exist in-repo; deferred until one does. |
| `cognitive/hierarchical_planning.py` | Options/MAXQ/HAM/Feudal RL planner | Template-driven decomposition + greedy option selection; no search. `PlannerType` MAXQ/FEUDAL/HAM are reserved labels, not algorithms. |
| `cognitive/multi_hop_reasoner.py` `abduce` | "Bayesian-like P(H\|O)" | Jaccard token-overlap ranking (`0.3 + 0.7*overlap`); deduction `_match_premises` is substring containment, not unification. Lexical heuristic, not inference. |
| `cognitive/ethical_bounding.py` `assess_weapons_uplift` | two-axis weapons/mass-casualty gate | Axis A is (now obfuscation-normalized, multilingual) keyword routing; Axis B is regex intent matching **plus a reasoning-backed classifier wired by default on the text surface** (fail-open / offline-safe — it contributes only when a real local/cloud model serves). The deterministic core remains a lexical heuristic, **not** a trained model, and says so. What *is* now measured: the Axis-B confidence logistic is fit on a 362-case labeled corpus (`configs/weapons_gate_calibration.json`; `is_fitted`/`source` are transparent), and the false-positive/false-negative operating point is a measured, CI-failing metric (`tests/ethical/test_weapons_gate_eval.py`; currently 0% FP / 0% FN on held-out val+test) with property/fuzz coverage (`test_weapons_gate_properties.py`). Coverage limits + the now-implemented compensating controls (durable audit, provenance enforcement, HITL escalation) are stated in [`HARM_POLICY.md`](HARM_POLICY.md) §8. Active safety control, not dormant. |
