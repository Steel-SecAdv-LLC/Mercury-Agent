# Dormancy & Salvage Ledger

Applies to Mercury Agent **v3.0.x**. Last updated: 2026-11-04.

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

## 2b. Exported instruments that no serving path reaches (2026-11-04)

The §2 ranking covers the dormant *cognitive* modules. A reachability sweep of
the **ensembles, fusion heads and mathematical instruments** found two more
that are exported and tested but that no live path consumes, and one of them
was carrying a measurable defect:

| Instrument | LOC | Exported from | Reached by | Measured? |
|---|---|---|---|---|
| `detectors/math_arrest/` (`AnomalyMathArrest`, 21 probes) | ~1,900 | package root + `detectors/__init__` | **now wired ✓** — manifest entry `math_arrest` via a `BaseDetector` adapter | **now measured ✓** (below) |
| `detectors/fusion/multimodal_fusion.py` (`MultiModalFusionOptimizer`) | 827 | `detectors.fusion` | three test modules only | **no** — two inputs absent, see below |

### `MultiModalFusionOptimizer` — claims constrained, and the gap named exactly (2026-11-05)

This entry said "needs a real VLM+visual corpus", which is correct and too
vague to act on. The two missing inputs are missing for different reasons, and
the distinction is what says whether the measurement is deferred or blocked:

1. **No VLM backend exists in a default install.** The semantic arm needs a
   served vision-language model, and none of `transformers`, `open_clip` or
   `clip` is in the dependency graph. The arm cannot be exercised at all, so
   this is an absent input rather than an untuned one.
2. **No anomaly-labelled visual corpus is vendored.** `tests/fixtures/visual`
   holds four CC0 photographs recorded for determinism and seam checks; they
   carry no anomaly labels. `benchmarks/mvtec_visual_benchmark.py` names the
   corpus a real evaluation would use, and it is not vendored.

So the measurement is **blocked on inputs, not deferred by choice**, and the
module's docstring now says so in place of the "Key innovations" list it
carried. That list described adaptive weighting, uncertainty-aware fusion,
domain calibration and complementary routing as innovations; they are
implemented mechanisms with no evidence attached, and "innovation" is a claim
about outcomes. The mechanisms are retained and relabelled, per this ledger's
no-deletion rule.

When both inputs exist the bar is the one §2c applies to the other unreached
fusion heads: beat the plain mean of the same real per-modality scores first,
then out-separate the engine's own trained fusion (§8).

### `AnomalyMathArrest` — measured, and two real defects fixed

The instrument **was not in `DETECTOR_MANIFEST`**, so it was unreachable from
the engine, the registry, the served routes, the CLI and the console catalog: it
could only be used by importing it directly. It had never been measured on real
labels, so there was no evidence it carried signal — the exact "interface ≠
signal" gap this ledger exists to close. It is now measured on both recorded
corpora and pinned by `tests/detectors/test_recorded_math_arrest.py`.

**Defect 1 — the fusion prior was inverted against measured signal.** Scores
were combined with `weight[rank] = PHI^(-rank)` over a hand-written
affinity order. That decay is steep: the first three ranks carry 76% of the
weight and the last probe 2.5e-05, so the per-probe `confidence` term (a
multiplier in [0, 1]) cannot overcome a 70,000x prior spread. On the `default`
order those three ranks are `variance_adapted`, `momentum` and `additive`,
measuring **0.531 / 0.524 / 0.685** mean NAB AUROC — while the two strongest
probes, `wave_propagation` (**0.697**) and `svd_projection` (**0.669**), sit at
rank 7 and below and receive about 2% between them. The 21-probe fusion
therefore scored *below its own best member*. The default weighting is now
`uniform` (the plain mean — the dominant aggregation for outlier ensembles,
Aggarwal & Sathe 2017, the same result the detector tier's `consensus` combiner
cites); `phi_affinity` is retained and reproduces its released numbers exactly.

| weighting | NAB (4 files) | ADBench (5 tables x 3 seeds) |
|---|---|---|
| **`uniform`** (new default) | **0.7581** | **0.6826** |
| `confidence` (evidence only) | 0.7277 | 0.6686 |
| `phi_affinity` (released) | 0.6519 | 0.5423 |

**Defect 2 — `SVDProjectionProbe` never used its fitted basis.** `fit` computed
the SVD of the training Hankel matrix and stored residual statistics, but
`deviation_score` **recomputed the SVD on the evaluation data**, so the learned
rank-1 subspace was unused, the residuals were compared against fit-time
statistics belonging to a different basis, and the anomalies being scored
helped define the subspace they were measured against. Its `abs` also scored a
window fitting the subspace *better* than training (more collapsed, more
normal) as deviant. It now projects onto the stored basis, directionally, and
moved from the weakest probe to the second strongest (NAB 0.664 -> 0.669 with
the per-file spread tightening, 0.415 -> 0.440 on off-modality tabular input).

**Defect 3 — 21 probes are not 21 dimensions, and the correction was logged as
noise.** `CorrelationAwareDecorrelator` finds near-duplicate probes and
down-weights the non-dominant member of each correlated cluster to
`1/cluster_size`. It works, and it is needed: on the four recorded NAB files the
ensemble measures **11.64–16.00 effective dimensions out of 21**, with 13–23
redundant pairs per fit and one pair reaching **r = 0.9976**
(`catalan_optimized` ↔ `exponential_decay`; `additive` ↔ `modified_zscore` at
0.9864). Two corrections: the per-pair finding logged at **`warning`** severity,
emitting up to 23 `WARNING` records on every single fit for a condition this
class exists to handle and does handle — it is now `debug`, with one `info`
summary naming the effective dimensionality; and the measurement was computed but
never surfaced, so the served result now carries `effective_probe_count` and
`redundant_probe_pairs`, keeping the probe count from reading as an evidence
count. Pinned per file by `test_effective_dimensionality_is_measured_not_assumed`.
`test_the_decorrelator_applies_under_every_weighting` pins that the new
`uniform` default does not bypass the multipliers.

**Defect 4 — `AdditiveProbe`'s fit-time intercept did not transfer, and it
scored below chance.** The probe fits `x_hat = intercept + slope*t` with
`t = np.arange(len(x))`, an index local to whatever array is passed, so `t`
restarts at 0 on every scored window. The fit-time intercept belongs to the
*start* of the fit series; reusing it on a window drawn from later in that
series added a systematic offset of `slope x (true position)` that dominated
the residual. Measured mean NAB AUROC **0.378**, and **0.032** on
`realAWSCloudwatch/iio_us-east-1_i-a2eb1cd9_NetworkIn.csv` — not merely
uninformative but almost perfectly *anti*-correlated with the labels.

Carrying the true time index was measured too and is **worse** (0.368): it
extrapolates a linear fit far beyond its fitting range. The absolute level is
not recoverable at score time, so only the trend rate and residual scale are
carried and the level is re-anchored per window
(`x_hat = slope*t + median(x - slope*t)`). That measures **0.685**, and lifts
the 21-probe default from 0.7498 to **0.7581** on NAB. This is the same class
of defect as Defect 2 — fit-time coordinates not carried to score time — found
by the same method, measuring each probe standalone.

**Verdict: measured, repaired, wired, still not in the default set.** At 0.7581
on the NAB files the ensemble sits above every series detector pinned in
`tests/detectors/test_recorded_nab_series.py` except `lstm_autoencoder` (0.829),
so the reachability gap was worth closing, and it is closed: `detector.py` adds
the `BaseDetector` seam the ledger's own rule demands (the raw `detect` returned
a bare array, satisfying no detector contract), the manifest carries
`math_arrest`, and the entry earns its keep through the gates a new `series_1d`
entry is *forced* to satisfy — a measured stationary false-alarm pin
(**0.0615** against its promised α = 0.05), the NAB triple
(**0.7581 / 40.35 / 0.0213**), an evidence-tied ontology row, a knowledge
profile, an exact state round-trip and the registry/engine seam harvest.

A **default-set slot is still not granted**: that needs a marginal-gain ablation
against the live ensemble, which this entry does not claim. Reachable and
measured is not the same as promoted, and this repository only promotes on
measured marginal gain.

### The marginal-gain ablation, run (2026-11-05) — HOLD

The ablation the paragraph above deferred is now made, by
`benchmarks/math_arrest_ensemble_marginal.py`. It is the same shape as
`benchmarks/kmeans_ensemble_marginal.py`, which settled the same question for
the revived k-means detector, so the two decisions are comparable: each
(table, seed) trains the engine's fusion from the same split and initialisation
with and without the detector, differing only in that one feature group. The
pre-registered bar is the k-means one, unmoved: **add iff mean ΔAUC > +0.002
with a majority of seeds agreeing.**

Five recorded ADBench tables × **8 seeds** = 40 cells. Eight rather than the
k-means ablation's three because the first run came in at +0.0014 against a
+0.002 bar, which is too close to call on three seeds.

| table | base fused AUROC | + math_arrest | ΔAUC | seeds ≥ | headroom |
|---|--:|--:|--:|--:|--:|
| pima_real | 0.7861 | 0.7877 | +0.0016 | 4/8 | 0.2139 |
| cardio | 0.9950 | 0.9959 | +0.0008 | 5/8 | 0.0050 |
| breastw | 0.9906 | 0.9926 | **+0.0020** | **8/8** | 0.0094 |
| wbc | 0.9874 | 0.9888 | +0.0013 | 6/8 | 0.0126 |
| thyroid | 0.9984 | 0.9980 | −0.0004 | 3/8 | 0.0016 |
| **mean** | | | **+0.0011** | 0.65 | |

**Verdict: HOLD — reachable-but-not-default.** The mean is +0.0011 against a
+0.002 bar, so the bar is not cleared and promotion is refused.

**What the 40 cells support, stated more precisely than "no gain".** Over all
cells the mean is +0.00107 (sd 0.00460, se 0.00073) with 22 of 40 positive, and
a 20,000-replicate bootstrap puts the 95% interval at **[−0.00029, +0.00249]**:

* `P(mean > 0) = 0.937` — the gain is *probably real*;
* `P(mean > +0.002) = 0.098` — and *probably below the bar*.

So the honest reading is a **small positive effect that does not reach the
threshold**, not an absent one. `breastw` is the one table that reaches the bar
and it does so with 8/8 seeds agreeing at sd 0.0020, the most consistent cell
set in the run; one table at the bar out of five is not a promotion.

**The corpus limits what this test can detect, and that is not a hedge.** Four
of the five tables leave the trained `OmniFusionModel` with **≤ 0.013 of
headroom** (thyroid 0.0016, cardio 0.0050, breastw 0.0094, wbc 0.0126). A
marginal-gain bar of +0.002 against a 0.9984 baseline is asking a new feature
group to consume an eighth of the remaining error. **HOLD therefore means "not
demonstrated on this corpus", not "demonstrated absent"**, and the same caveat
applies to the k-means ablation that set this bar.

**A hypothesis raised and then withdrawn, because the seeds killed it.** At
three seeds the gain looked proportional to the headroom available —
`r(headroom, ΔAUC) = 0.854`, rising to **0.992** with the one high-headroom
table removed, which is the opposite of a leverage artefact and read as a real
structural finding. At eight seeds the same correlations are **0.346** and
**0.828**. With five points and per-table seed sd of 0.002–0.008, a high
correlation is cheap, and `pima_real`'s own mean moved from +0.0043 to +0.0016
once five more seeds were added. The direction survives weakly — the
lowest-headroom table is the only negative one — but it is **recorded as
unestablished**, not as a finding. This is the third small-sample pattern in
this cycle that did not survive more data (see §2e on the mean-shift `h`, and
the survivability note in `research/governed_fusion/FINDINGS.md`), which is the
argument for the eight seeds rather than three.

**The series side does not rescue the promotion case either.** The ablation
above is tabular because the engine's fusion is row-wise, which is the path a
default-set member actually joins; `math_arrest` is a `series_1d` entry, so it
is being measured off its declared modality and the feature group is a per-row
summary. That objection is real, and it was checked rather than used as a
defence: the group is **not** degenerate on tabular input (21 columns, median
542 distinct values over 548 rows, 0–2 constant columns per table), so the
ablation measures fusion rather than a dead group. And on the corpus where the
instrument *is* in-modality, its standing is mixed — among the 22 series
detectors pinned in `tests/detectors/test_recorded_nab_series.py` it ranks
**2nd of 22 by AUROC** (0.749, behind `lstm_autoencoder`'s 0.829) but **10th of
22 by the NAB standard score** (33.7, behind `digital_twin` 53.4,
`gaussian_process` 52.9, `survival` 48.4, `hawkes` 46.1, `temporal` 46.0 and
four others). By the metric §2e established as governing for this instrument —
the one that charges for a false alarm, where AUROC is blind — it is mid-pack,
and the normalizer repair recorded in `BaseEquationProbe._normalize_scores`
moved it there from 40.4. A strong ranker that is tenth on the streaming metric
is not an obvious promotion on the series route either.

**Disposition: shipped as reachable-but-not-default, with the measurement
behind it rather than a deferral.** It stays a manifest entry, available by
`engine.register_detector("math_arrest", MathArrestDetector())`, exactly as the
revived k-means detector is.

**Modality note (why the two corpora differ).** Every probe is a *series*
equation (Hankel trajectories, chaos, harmonic/wave), and
`BaseEquationProbe._to_1d` reduces a 2-D input to its **per-row mean**, so
matrix input is scored on that row summary and per-feature structure is never
seen. The class docstring nonetheless presents the instrument as an
IsolationForest replacement, which is a tabular role — hence both corpora are
measured and both numbers are published above.

## 2c. Full reachability sweep of the assistive mechanisms (2026-11-04)

§2b measured the two instruments found by a partial sweep. This is the
**complete** sweep: a transitive import closure from Mercury's real entry
points — `engine.py`, every `api/` module, `cli`, and the declared
`module_path` of all 114 manifest entries — against every module whose name or
role is an ensemble, fusion head, combiner, calibrator, meta-learner, drift or
uncertainty mechanism. 522 modules are reachable. The sweep was run three
times: the first two passes produced false negatives (dynamically-registered
operator tools, and classes referenced by their export name rather than their
module), both corrected below, which is why the dispositions here name the
*mechanism* that reaches each module rather than just a count.

**Reached and serving — not dormant.** `core/score_calibration.py` (Platt /
isotonic / Venn-Abers, served from `engine.py`), `core/calibration.py`,
`detectors/_calibration.py` (29 importers — the shared conformal layer),
`core/fusion.py`, `core/three_r/fusion.py`, `ml/fusion_network.py`,
`cognitive/uncertainty.py`, `detectors/drift.py`, `core/fibring_fusion.py`,
`core/governed_fusion.py`, `core/attention_fusion_stack.py`,
`core/calibration_pipeline.py`, `ml/meta_learning.py`,
`ml/concept_drift_evaluation.py`, `core/adaptive_fusion.py`,
`detectors/adaptive_detector.py` and `detectors/diffusion_ad.py` (both manifest
entries).

**Correctly unreached — the reason is the design, and it is stated in the
module.** These are not gaps:

| Module | LOC | Why unreached is correct |
|---|---|---|
| `core/stacking_fusion.py` | 802 | Explicitly deprecated, raises a `DeprecationWarning` at import, "retained for reference only". The production fusion is `OmniFusionModel`. |
| `medical/clinical_calibration.py` | 442 | A *measurement* layer: fits calibrators on a held-out clinical split and reports ECE / Brier / conformal coverage. It states it must not sit on the operational path. |
| `security/sigma_calibration.py` | 422 | Read-only ECE / threshold / temperature sweep over a *fresh* σ gate loading the same frozen weights. Explicitly "never trains, never writes weights", never touches the operational path. |
| `tools/{algorithm_name_drift_gate, workflow_version_drift_gate, sigma_immutable_drift_monitor, benevolence_calibration_report}` | 930 | Registered in `TOOL_REGISTRY` (65 tools), documented in `docs/TOOLS.md`, covered by the generic envelope/exit-code contract tests, and the two *gates* run as pre-commit hooks. The other two need operator-supplied inputs (a labelled probe set; a systemd timer's rolling state) that no CI lane has. **Reached — the first sweep pass missed them because the registry loads them dynamically.** |
| `detectors/advanced/gwo_ensemble.py` | 567 | Declared library-role with a dated measurement in its docstring; exported lazily to keep package import cheap, and tested as `GWOEnsembleDetector`. |
| `ml/drift.py`, `core/adaptive_detector.py` | 84 | Documented relocation shims that re-export the live implementation. |
| `federation/aggregator.py` | 420 | Belongs to the federated-learning subsystem, not the single-node detect path. |
| `ml/multimodal_fusion.py` | 113 | Self-declares "⚠️ SIMULATION-BASED: Trained on simulated multimodal data. Real-world validation required." |

### `core/adaptive_domain_thresholding.py` — measured, refused, one defect fixed

1,289 LOC promising "per-domain threshold calibration", Platt scaling, isotonic
regression, a "domain ensemble weighting optimizer", and "the strategic
recommendations for adaptive thresholding to improve domain competence across
Medical, Financial, and Infrastructure". No serving path reaches it, and it had
never been measured against the operating point Mercury actually ships.

Measured on the five recorded ADBench tables × 3 seeds, scoring the *same*
detector's scores under three thresholding policies, so the comparison isolates
the threshold and nothing else (`tests/core/test_adaptive_domain_thresholding_measured.py`):

| policy | mean F1 | mean FPR | FPR range | over the promised α = 0.05? |
|---|---|---|---|---|
| **conformal α = 0.05** (shipped) | 0.4059 | **0.0231** | 0.000 – 0.048 | **never, on any table or seed** |
| ADT label-free | 0.4549 | 0.0873 | 0.013 – 0.160 | 3 of 5 tables, up to 3.2× |
| ADT supervised | 0.3882 | **0.6335** | 0.055 – **0.993** | catastrophically |

**The supervised path is broken, not a tradeoff.** Given a labelled calibration
set it flags **98.7%** of normal rows on `pima_real`, 95.7% on `breastw` and
97.9% on `wbc` — and its mean F1 is *below* the shipped cut's while its
false-alarm rate is ~27× higher. Flagging almost every normal row is not an
operating point.

**The label-free path does not clear the bar either.** Its higher mean F1 is
not an improvement: it is bought by abandoning the false-alarm guarantee, and
the mean is carried by a single table (`breastw` 0.935 vs 0.253) while it
*loses* 0.36 F1 on `wbc` and 0.31 on `thyroid`. A policy whose per-table FPR
ranges over an order of magnitude is not comparable to one that holds α on
every table. On this ledger's pre-registered-bar rule, that is a refusal.

A methodological note, because the first comparison was wrong: an initial
"matched alarm budget" test returned a delta of exactly 0.0000 with zero
bootstrap spread. That was **circular** — on a fixed score vector any threshold
flagging *k* points flags *the top k*, so all policies at equal budget are
identical by construction. Thresholding cannot improve a ranking; the only
degree of freedom is the budget, and monotone recalibration (Platt and isotonic
both are) cannot change the ranking either. The table above is the non-circular
comparison: the candidate's own operating points placed against a sweep of the
shipped α.

**Defect fixed regardless of the refusal.** `PlattScalingCalibrator` evaluated
`1 / (1 + np.exp(A * s + B))` directly, which overflows `np.exp` on real score
distributions (reached on the recorded tables). Its comment claimed "clipping
for numerical stability", but the clip was applied to the *result* — after the
overflow had already happened inside `np.exp` — and `calibrate()` had no guard
at all. A second sigmoid used the branch-selecting `np.where` form, which
returns the right values (the overflowed branch is discarded) but evaluates
both arms eagerly, so it still warned and did twice the work. Both now use
`scipy.special.expit`. The fix is **value-preserving**: every number in the
table above is unchanged, and the fitted `A`/`B` are identical to six figures.

**Not promoted.** A module that does not reach its bar stays unreached; the
measurement is the deliverable, and it is pinned so the disposition cannot
drift back to resting on the docstring.

### Three of the four unreached heads, measured (2026-11-05)

The table below used to say "unreached, unmeasured" for four ensembles and
record a *hypothesis* for the coordinator — that the engine's own
`OmniFusionModel` already fuses detector scores better than any re-weighting of
them. Three of the four are now on a bar, by
`research/ensembles/harness_unreached_heads.py`.

**The bar, and what it is not.** The question asked is the first necessary one:
does the head beat **the plain mean of the same real detector scores**? For
unsupervised outlier ensembles the plain mean is the dominant aggregation
(Aggarwal & Sathe 2017) and it is what the detector tier's `consensus` combiner
and `math_arrest`'s fusion default already use on that authority, so a head that
cannot beat the mean of its own inputs cannot be worth wiring. Clearing it is
**not** promotion: §8 established that a fusion head must out-separate the
engine's trained `OmniFusionModel`, and that comparison is not made here.

Inputs are the five manifest detectors and the protocol of
`tests/detectors/test_recorded_adbench_tabular.py` — five recorded ADBench
tables, stratified 70/30, three seeds, features standardised on the training
rows — so the columns are the same real scores that module pins. Supervised
heads see training-row scores and labels only. 15 cells.

| head | mean AUROC | vs plain mean | cells won |
|---|--:|--:|--:|
| plain mean (the bar) | 0.8893 | — | 0/15 |
| **best single detector, chosen on train** | **0.9142** | **+0.0249** | **14/15** |
| `ensemble.VotingEnsemble` | 0.7577 | −0.1315 | 0/15 |
| `ensemble.MetaLearner` | 0.6298 | −0.2595 | 8/15 |
| `ensemble_coordinator.BayesianWeightOptimizer` | 0.8940 | +0.0047 | 11/15 |
| `ensemble_coordinator.GradientWeightOptimizer` | 0.8895 | +0.0002 | 7/15 |

**`VotingEnsemble` — refused, and it cannot rank at all.** −0.1315, losing every
one of 15 cells. The structural reason matters more than the number: it
publishes **no continuous fused score**. Its `confidence` is
`|weighted_avg - 0.5| * 2`, symmetric about the threshold, so a confidently
normal row and a confidently anomalous row receive the same value. That is a
defensible meaning for "confidence in a binary call" and it is not an anomaly
score; ranking by it inverts half the ordering (0.7566 against the plain mean's
0.9026 on cardio/seed 0). The figure above uses its binary `predictions`, which
is the most its contract supports.

**`MetaLearner` — refused, and the instability is the finding.** −0.2595 on the
mean, yet it wins 8 of 15 cells. It is not uniformly weak: it reaches 0.9515 on
cardio and 0.9891 on breastw, then collapses **below chance** on the two tables
with the fewest anomalies — 0.4709 / 0.5132 / 0.2222 on `wbc` and 0.1403 /
0.1624 / 0.1224 on `thyroid`. A head that scores 0.12 is not under-performing,
it is inverted, and a mean that hides a 0.12 behind a 0.99 is why the per-cell
column is published beside it. This is textbook overfitting of a supervised
second stage on a handful of positives, and it is a reason to distrust the
cells it wins, not to average them.

**The coordinator — not dead by construction, but dominated.** The ledger's
hypothesis was that re-weighting detector scores buys nothing. Measured, the
Bayesian optimizer buys a little: **+0.0047** with 11 of 15 cells, which clears
the ±0.002 noise floor this ledger uses for `kmeans_ensemble_marginal`. So the
hypothesis as stated is **not** confirmed. What refuses it instead is the
comparison it loses: simply **picking the best single detector by training
AUROC** scores **+0.0249** and wins 14 of 15 — five times the lift, from one
line, against 1,092 LOC of Thompson sampling, cascading pipelines and
meta-learning. The gradient optimizer measures +0.0002, inside the noise floor.
Neither is promoted. The honest verdict is "works, and is beaten by something
trivial", which is a different disposition from "does not work" and is recorded
as such.

**A finding about the baseline, not the heads.** That `best_single_on_train`
beats the plain mean by +0.0249 on 14 of 15 cells is itself worth recording: on
these five tables the five detectors are unequal enough that averaging them
dilutes the strongest. It is a statement about this detector set on this corpus,
not a recommendation — a label-free selection rule that good is also a
single-point-of-failure — and it is the reference any future fusion head on this
corpus should be made to beat.

**`models/foundation/ensemble.py` — not measured, and not refused.** It needs
the `[foundation]` extra to import its backends at all, so it could not be put
on this bar. "Unmeasured" here means the measurement was not attempted, which is
a weaker statement than the three above and is kept distinct from them.

**`core/three_r/learnable_fusion.py` — not measured on this bar, which is the
wrong bar for it.** It fuses the three OAE streams (R/H/O), not detector scores,
so it does not take this harness's input. Its own measurement is named in
`MATH_SPEC.md` §2.1.1 and is a caution: the served `PHI : 1 : 1` weighting of
those same streams loses to a plain mean by 0.0017, and weights *learned per
file* reach 0.8916 against 0.8301 — but the per-file optima are `(0, 1, 0)`,
`(0, 0.45, 0.55)` and `(0, 0, 1)`, so no fixed triple fits the four files.
Learning those weights is therefore both the plausible direction and the one
most exposed to overfitting four short series. The "state-of-the-art" wording is
removed from the module either way.

### Still unreached and still unmeasured — named, not assumed

| Module | LOC | Claims | Status |
|---|---|---|---|
| `core/three_r/learnable_fusion.py` | 1,123 | "State-of-the-art differentiable 3R fusion", backprop-trainable `w_R/w_H/w_O`, learned ethical gate, wavelet+attention | Unreached, 3 test modules, **no measurement on labelled data**. The "state-of-the-art" wording is not backed by a measurement in this repository. |
| `ml/ensemble_coordinator.py` | 1,092 | adaptive weight learning from performance feedback, cascading detection, meta-learning detector selection, uncertainty-aware fusion | Unreached, 1 test module, **unmeasured**. §7's fusion-marginal ledger and the GOSNN analysis both found the engine's own `OmniFusionModel` already fuses detector scores better than any re-weighting of them, so this is a likely dead-by-construction candidate — but that is a hypothesis until measured, and it is recorded as one. |
| `ml/ensemble.py` | 473 | stacking meta-learner, AdaBoost-style boosting, bagging | Unreached, **unmeasured**. Carried an uncited quantitative claim — "Research shows 30-40% accuracy improvement" — in both the module docstring and a class docstring. **Claim removed** this cycle: the averaging result it was gesturing at is real and cited where it is actually relied on (Aggarwal & Sathe 2017, in the detector tier's `consensus` combiner and `math_arrest`'s fusion default), but it is a statement about averaging dominance, not a percentage, and it does not transfer to these heads without measurement. |
| `models/foundation/ensemble.py` | 529 | combines predictions from multiple foundation models | Unreached, **unmeasured**; needs the `[foundation]` extra to run at all. |

## 2d. A gate that could not fail (2026-11-05)

`tools/oae_dimensionality_probe.py` advertised that it "verif[ies] the OAE
R/H/O branches preserve dimensionality through fusion at the tensor level".
What it did was draw three random Gaussian tensors, combine them inline as
`w_R * a + w_H * b + w_O * c`, and assert that the result kept its shape, that
the fused L2 norm was bounded by the largest input's, and that equal inputs
fused to themselves.

All three are convex-combination identities that hold for *any* non-negative
weights summing to 1. Measured by substituting the weight triple, it returned
`ok` for `(1, 0, 0)`, `(0, 0, 1)`, `(1/3, 1/3, 1/3)` and `(0.9, 0.05, 0.05)`
alike, and failed only on a negative weight — a case
`oae_weight_certifier` already covers. It never imported `OmniAvaEquation`, so
no change in the served equation could reach it. It measured no dimensionality
and certified nothing about Mercury: a gate incapable of failing, which is
worse than no gate, because the green certificate was read as evidence.

**Replaced, not deleted.** The tool now exercises the shipped
`OmniAvaEquation` and measures what it actually does against the weights it
publishes: each stream's influence on the fused score (sweeping one stream over
`[0, 1]` must move the fused score by exactly that stream's declared weight),
`eta ** Phi` decoupling across the permitted range, and convexity on the unit
cube and its diagonal. With `oae_weight_certifier` holding the declared table
to the published derivation, the two now form a chain of custody from
derivation to table to served behaviour.

It can fail, and that is pinned: `tests/tools/test_oae_dimensionality_probe.py`
injects a dead stream, drifted arithmetic, a re-coupled `eta` and a constant
fused score into the real equation and requires each to be caught. The
dead-stream case is not hypothetical — it is the `H(omega)` saturation recorded
in `MATH_SPEC.md` §2.1.6, which shipped and was found by hand. That class of
defect now fails a gate.

## 2e. Two hypotheses this pass measured and then abandoned (2026-11-05)

Both are recorded because the measurement is the deliverable either way, and
because each conclusion is the opposite of what the first evidence suggested.

**The probe decorrelator earns its place — on the metric that charges for a
false alarm.** `CorrelationAwareDecorrelator` down-weights members of a
correlated probe cluster. What that buys had never been measured. Isolated
through `PhiWeightedFusion.fuse` with everything else held fixed, *dropping*
the multipliers scores higher on ranking AUROC in all six weighting × corpus
cells and on 6 of the 9 individual datasets (uniform 0.7698 NAB / 0.7073
ADBench against 0.7581 / 0.6826; worst single harm −0.0476 on `wbc`,
−0.0874 on one NAB file). On that evidence the multipliers look like
complexity that costs accuracy, and dropping them also looked like the simpler
system.

The operational triple says otherwise, decisively:

| multipliers | AUROC | NAB standard score | alarm rate |
|---|--:|--:|--:|
| applied (shipped) | 0.758 | **40.4** | **0.021** |
| measured only | 0.770 | 19.6 | 0.034 |

AUROC is a pure ranking metric and is blind to false-alarm cost. The NAB
standard score charges for one and rewards earliness, and the alarm rate shows
the mechanism: suppressing a redundant cluster is what keeps the ensemble
quiet, at 0.021 against the 0.05 the conformal cut promises. Halving the
purpose-built streaming metric and adding 62% more false alarms to gain 0.012
of ranking is not a trade worth making. **The multipliers stay**, now for a
measured reason rather than an assumed one, and
`research/math_arrest/harness_decorrelation.py` reproduces both halves so the
next person to optimise this fusion on AUROC alone finds the counter-evidence
first. `tests/detectors/test_recorded_nab_series.py` pins the triple and is
what catches it.

**A real change-point statistic is wrong for this ensemble.** `AdditiveProbe`
was repaired this cycle but its three measured variants were all the same
"distance from a trend line" statistic with different anchors, so the obvious
question — would a proper change-point detector do better, and restore the
level-shift geometry the ensemble lost — stayed open. Measured on the same NAB
protocol, against the shipped probe's 0.6848 (reproduced to four decimals by
the harness, which validates it):

| statistic | mean NAB AUROC |
|---|--:|
| trend dispersion (shipped) | **0.6848** |
| two-sided CUSUM, k = 0.25 / 0.5 / 1.0 σ | 0.4244 / 0.4996 / 0.5456 |
| sliding mean shift, h = 16 / 48 | 0.6679 / 0.5498 |

CUSUM is at or below chance and strongly *anti*-correlated on one file
(0.0609–0.1084). The reason is structural, not a tuning failure: CUSUM
accumulates, so once a shift occurs the statistic stays elevated and flags the
entire remainder of the window, while NAB labels a bounded window around the
anomaly. A persistent statistic cannot score a bounded label. **The probe
stays a dispersion probe**, and the level-shift geometry stays unclaimed rather
than claimed by something that does not deliver it.

The sliding mean shift at h = 16 is the one lead worth naming: 0.6679
standalone, close to the shipped probe and nearly independent of it (max
\|r\| 0.319, mean 0.200), and adding it as a 22nd column lifted a plain
column mean by +0.0188 — gains on two files, −0.007 and −0.002 on the other
two. **Not promoted.** It is a change-point statistic, so ADBench cannot
corroborate it (tabular rows have no ordering), which leaves four NAB files as
the entire evidence base — below the bar this ledger applies to everything
else, and well below what adding a probe to a 21-probe published ensemble
should need. Recorded as a lead, not a finding.

### The lead, tested on two independent ordered corpora (2026-11-05)

ADBench cannot corroborate a change-point statistic, but this repository
records **two other ordered corpora with external labels**, and the objection
does not reach them: CHB-MIT scalp EEG (`tests/fixtures/eeg`, background against
clinician-annotated seizure) and CWRU bearing vibration (`tests/fixtures/cwru`,
healthy against machined 0.007-inch faults).
`research/math_arrest/harness_mean_shift.py` measures the same statistic on all
three. **The lead does not survive at the `h` it was named for.**

| corpus | labels | lift at h = 16 | lift at h = 48 |
|---|---|--:|--:|
| NAB (the original evidence) | point windows in one stream | +0.0289 | +0.0199 |
| CHB-MIT EEG | clinician annotation | **+0.0001** | +0.0161 |
| CWRU bearings | machined fault | +0.0171 | +0.1081 |

**`h = 16`, the value §2e named, measures no lift on EEG at all** (+0.0001),
and EEG is the independent corpus whose labelling geometry is closest to NAB's.
`h = 48` is positive on all three and much the stronger candidate — but `h = 48`
is the variant §2e measured as clearly *worse* (0.5498 against 0.6679), so the
choice of `h` that produced the lead points away from the `h` that generalizes.
That inversion is the result: the lead was an artifact of selecting `h` on the
same four files the lift was then measured on. **Still not promoted**, now for a
measured reason rather than a sample-size one, and promotion would need `h`
fixed in advance of the corpora plus the engine-level marginal gain.

**§2e's own figures could not be reproduced, and that is a defect in the
record.** The harness behind them was never committed. The new one reproduces
the *shipped* `additive` probe exactly (0.6848, before the normalizer repair
below moved it to 0.6990), which validates the protocol, split and AUROC — but
the mean-shift numbers differ (0.7369 against 0.6679 at h = 16), because the
statistic's definition is not recoverable from the ledger text. The definition
is now explicit in `SlidingMeanShiftProbe`, so the figures here are checkable in
a way §2e's were not.

**A caution the same run surfaced, flagged rather than claimed.** The absolute
AUROCs put the 21-probe ensemble at or below chance on bearing vibration:
`column_mean_21` measures 0.4913 on inner-race, 0.7750 on outer-race and
**0.0063** on ball — the last is near-total inversion, not weakness. That is
consistent with the modality note in §2b (every probe is a series equation
tuned on low-frequency telemetry; 12 kHz vibration is a different regime), but a
window-level reduction of a `series_1d` detector is **not** a use its contract
covers, and a figure of 0.0063 is as likely to indict the protocol as the
detector. It is recorded as a caution against off-modality use and explicitly
**not** as a measured property. The lift column above is unaffected either way:
it is a within-protocol difference, where the same fit, windows and reduction
serve both arms and only the extra column changes.

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
heuristic from `core/model_domains.py`), with malformed declared
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
| `core/global_omni_scalar_network.py` (~82 ISO-25010/Halstead/DORA/SLSA/… scalars) | declared diagnostic scalars | **36 of the 82 now carry REAL self-referential measurements; 46 stay documented placeholders.** `scripts/collect_sw_eng_metrics.py` computes real metrics on Mercury's own source tree — the Halstead suite (7), cyclomatic complexity (1) and Maintainability Index (3) via stdlib `ast`, the 10 Mercury-native supply-chain / repository-integrity checks from repo config, the 4 DORA metrics as VCS-history proxies from `git log`, the 4 NIST SSDF practice groups and 4 SLSA build-track measures from repo state, and the 3 computable NIST SAMATE assurance measures — into `core/sw_eng_metrics.json`, overlaid at GOSNN init by `_apply_measured_sw_eng_metrics` (existing keys only; values kept in their penalty/positive direction band). The remaining 46 (the 31 ISO/IEC 25010 quality characteristics, the 7 non-computable NIST SAMATE measures + 4 ISO/IEC 5055 assurance measures, and the essential/design/cognitive/npath complexity variants) stay static placeholders — computing them would be fabrication or needs external process/telemetry data. Merit-gated by `tests/test_sw_eng_metrics_collector.py`. **Mission-scope disposition (decided 2026-07-18, measured 127 = 82 metric-only + 45 operational SOFTWARE_ENGINEERING scalars):** code-quality telemetry is in scope only as *self-referential observability* — Mercury introspecting its own engineering health as part of the autonomous agent's self-maintenance mandate — and out of scope as a detection product (no external-codebase scoring service is planned). Consequences: (a) the 45 operational SW-eng scalars stay exactly where they are — they are part of the frozen σ_Immutable operational layout whose bit-stability *is* the config-integrity signal, and restructuring that layout is safety-forbidden (it collapsed the deterministic floor once); (b) the metric-only scalars stay filtered — the 36 measured ones and the 46 placeholders alike are kept out of the operational bands (verified: the σ score is unchanged by the overlay), and reviving any placeholder still requires a real collector plus the standard merit gate. |
| `core/global_omni_scalar_network.py` `MultiHeadAttentionFusion` | "32-head attention" learned fusion | **Revived — serving after the constant-input defect was fixed (2026-07-16).** The degeneracy had a single root cause: detector/model anomaly scores reach the engine as `torch.Tensor` (`_normalize_scores` always returns a tensor), but the GOSNN base-scalar builder filtered on `isinstance(score, (np.ndarray, float, int))` and silently dropped every one, so the fusion's per-call base member was always empty and its input constant (the recorded **1 unique state list across 403 calls**). Coercing tensor scores to floats gives `fuse()` genuine per-call input; the write-only enhanced-scalars re-registration (which counted detector scores as critical ethical anchors and collapsed the σ_Immutable floor once populated) was removed. Re-run of `scripts/train_gosnn_fusion.py` on a real 3-dataset ADBench harvest now measures **450 unique state lists of 450** and the merit gate passes (`multitask`: held-out masked-member MSE 4.65e6 ≤ 0.95×reference 8.30e6, fused std 647 ≥ 0.20×member std 3082 — no collapse), so `gosnn_attention_fusion.pt` ships (`artifacts/gosnn_fusion.eval.json`, provenance sidecar). The fused output feeds observability only — `fusion_score`/`intelligence_contribution` now vary per call while `anomaly_prob` and the σ_Immutable verdict are unaffected. Pinned by `tests/core/test_attention_fusion_gate.py` (seam + artifact/package agreement) and `tests/core/test_gosnn_fusion_per_call_variation.py` (per-call variation + gate safety). **Consequential role evaluated and DEFERRED — measured no gain (2026-07-17).** A consequential channel now exists end-to-end (detection head over the fused state → `gosnn_metadata["detection"]` → the decision layer's abstention-only disagreement overlay, `DecisionPolicy.defer_on_gosnn_disagreement`), and `scripts/train_gosnn_fusion.py` was rebuilt to train fusion+head jointly on **labelled ADBench outcomes** (692/692-unique harvested production calls, ground-truth labels of the exact detected rows) behind a detection-metric merit gate. The gate **refused, and the refusal is decisive — dead-by-construction, not under-powered**. On a 9-dataset label-stratified harvest (1,469 labelled calls, 321 anomalies; held-out test pool 224 calls / 50 positives) with properly tuned training (validation AUC 0.862 — the optimizer is not the confound), the learned fused-state head measured held-out AUC **0.801**. The decisive baseline is the engine's OWN calibrated `anomaly_prob` — the verdict a disagreement overlay would second-guess — which separates the same held-out anomalies at **0.961**; the phi-reference-fusion head is 0.845 and the raw mean detector score 0.816. Bootstrap (2,000 dataset-stratified paired replicates) of learned − strongest-baseline is P5 **−0.201** / median −0.149: the head is decisively *worse* than the engine it would contradict, so on every disagreement the engine is the one more often right and demoting there removes net-correct verdicts (the threshold search independently returns no viable pair). A direct linear/MLP head on the raw detector-score member does better (0.886/0.904) — but still short of 0.961: the engine's OmniFusionModel already fuses those detector scores better than any re-weighting of them can, so a GOSNN-input consequential channel is **dead-by-construction**, not an architecture-tuning problem. Mechanistic root (measured 2026-07-17): of the 9 members the fusion receives on the detect path, only member 0 (the per-call detector scores) varies across calls (across-call std 2.47); the other 8 (static registry group scalars — ETHICAL/SECURITY/COSMIC/…) are bit-constant (std at machine epsilon). The "fusion" is therefore one signal averaged with eight constants, which is exactly why attention+mean-pool underperforms a probe on member 0. No per-call detection signal is recoverable from the 8 dead members, so detection-science investment belongs in the detector tier / OmniFusionModel, not GOSNN.

**GOSNN's resolved, correct role (audited 2026-07-17 — not "only losses").** GOSNN has two legitimate, functionally beneficial jobs on the detect path, and both were verified rather than assumed:
* **Consequential — the σ_Immutable ethical gate.** GOSNN aggregates the full 127-scalar operational vector into the hard ethical boundary (deterministic floor authoritative, learned score advisory). This is GOSNN's correct high-value use: a whole-system scalar aggregator feeding an ethics verdict. Signal-integrity hardening this pass: `detect_with_fusion` collected that 127-scalar vector **twice** per call (once for GOSNN's advisory gate inside `get_enhanced_scalars`, once for the authoritative σ gate) — two independent snapshots that a concurrent registration could make diverge. The enhancement now carries the snapshot it scored (`EnhancementResult.collected_scalars`) and the authoritative gate reuses it: one registry walk instead of two per call, and the advisory and authoritative gates provably evaluate the identical vector. σ score and verdict bit-identical (pinned by `tests/core/test_gosnn_scalar_snapshot_reuse.py`).
* **Observability — real signal, not theatre.** Measured on 120 real cardio detect calls: the surfaced `enhancement_fusion_score` (AUC-vs-label 0.878) and `intelligence_contribution` (0.836) genuinely track anomalousness and are only weakly correlated with `anomaly_prob` (−0.33 / −0.27), so — unlike the retired `harmonic_synergy` dead constant — they are kept. Honest caveat recorded: that signal is member-0-derived (detector scores), so it is informative but **not independent** of what the engine already fuses; it is a transparent observability lens, correctly never routed into a decision.
So GOSNN leaves this PR confined to exactly the two things it is measurably good at — ethical aggregation (consequential) and live observability — with the detection-model misuse refused on evidence, the double-collection waste cut, and nothing left surfacing as theatre. Gate hardening this pass: the engine's `anomaly_prob` is now a **mandatory merit baseline** (a disagreement head must out-separate the verdict it guards, not merely beat the phi/mean reference fusions), closing a real hole — the head clears phi/mean with bootstrap support and, against those strawmen alone, would have shipped a cross-check weaker than the engine. No detection head ships, the overlay stays structurally inert (`consequential.shipped=false` in `artifacts/gosnn_fusion.eval.json`, pinned by `test_consequential_verdict_matches_shipped_head`), and observability-only is the ratified, evidence-backed posture. Also in this pass: train/serve stacks unified in `core/attention_fusion_stack.py` (the serve-only harmonic-synergy output modulation — measured a constant ×1.0118 — is removed, so the shipped weights now serve under exactly the semantics their gate verdict measured), and the dead-constant `harmonic_synergy` metric (bit-identical 0.618034 on 15/15 production calls; FFT conjugate-symmetry pins its ratio at 1.0) is no longer surfaced in detection results. |
| `security/sigma_immutable_gate.py` + `core/global_omni_scalar_network.py` `EthicalGate` | learned σ_Immutable ethical gate | **Harvested-corpus-trained; advisory only — the deterministic floor remains the authoritative gate.** The network is trained by `scripts/train_sigma_immutable.py` on a **real harvested config-integrity corpus** (not synthetic RNG): the positives are the actual intact operational vector harvested from a healthy engine (`scripts/harvest_sigma_baseline.py` → `sigma_immutable_baseline.json`) plus intact variations holding the 24 critical ethical *anchors* at-or-above threshold; the negatives are real tamper mutations (anchor collapse). Labelling now matches the deterministic floor's real 24-anchor contract, not the earlier all-27-ethical rule that mislabelled the real (narrative-low) config. Because the exact baseline is a training positive, the network passes the real production vector **by construction** — a `main()` DoS guard asserts `score(baseline) ≥ threshold` before shipping (measured 0.999922). Production still composes the deterministic critical-ethical floor (`SigmaImmutableGate.enforce_ethical_floor`) BEFORE the learned score at every boundary — a collapsed anchor is a categorical fail-closed refusal no learned score can override; the floor stays authoritative. The harvested retrain additionally closed the old synthetic net's false-assurance gap: measured alone the network now **refuses** a benevolence-zeroed vector (score 0.0) instead of passing it, so the advisory score agrees with the floor. **Scalar-dynamics audit:** the operational/metric-only split is clean — 82 placeholder SOFTWARE_ENGINEERING scalars are correctly filtered OUT of the gate (0 leak), 127 operational scalars feed it. All 127 are bit-constant across normal+anomalous inputs and four domains, and the surfaced `sigma_immutable_score` is the exact constant 0.9999216794967651 on every call. This is correct, not a defect: the σ gate is a **config-integrity / tamper check** (ethics is a system-configuration property, not a per-row property), reading "intact" constantly on normal operation and moving only if a critical ethical scalar is corrupted. Per-call, content-driven ethical enforcement is `BenevolenceScorer.enforce` (input-sensitive, runs before the GOSNN block); making the governance scalars per-call adaptive is exactly what collapsed the σ floor earlier and is forbidden. The constant score is a genuine integrity readout, not a live per-input measurement. |
| `cognitive/hierarchical_planning.py` | Options/MAXQ/HAM/Feudal RL planner | Template-driven decomposition + greedy option selection; no search. `PlannerType` MAXQ/FEUDAL/HAM are reserved labels, not algorithms. |
| `cognitive/multi_hop_reasoner.py` `abduce` | "Bayesian-like P(H\|O)" | Jaccard token-overlap ranking (`0.3 + 0.7*overlap`); deduction `_match_premises` is substring containment, not unification. Lexical heuristic, not inference. |
| `cognitive/ethical_bounding.py` `assess_weapons_uplift` | two-axis weapons/mass-casualty gate | Axis A is (now obfuscation-normalized, multilingual) keyword routing; Axis B is regex intent matching **plus a reasoning-backed classifier wired by default on the text surface** (fail-open / offline-safe — it contributes only when a real local/cloud model serves). The deterministic core remains a lexical heuristic, **not** a trained model, and says so. What *is* now measured: the Axis-B confidence logistic is fit on a 362-case labeled corpus (`configs/weapons_gate_calibration.json`; `is_fitted`/`source` are transparent), and the false-positive/false-negative operating point is a measured, CI-failing metric (`tests/ethical/test_weapons_gate_eval.py`; currently 0% FP / 0% FN on held-out val+test) with property/fuzz coverage (`test_weapons_gate_properties.py`). Coverage limits + the now-implemented compensating controls (durable audit, provenance enforcement, HITL escalation) are stated in [`HARM_POLICY.md`](HARM_POLICY.md) §8. Active safety control, not dormant. |
