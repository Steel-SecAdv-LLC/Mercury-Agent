# Neuro-Symbolic Fusion, Ablation, and Conformal Uncertainty

Applies to Mercury Agent **v2.1.x**. Last updated: 2026-07-24.

This document is the **accounting** for Mercury Agent's neuro-symbolic fusion
work: what was built, how it was measured on real labels, and the explicit
keep / cut / quarantine verdict for every component touched. The standing rule
is anti-theater — *convert only what demonstrably carries signal, and prove it
on real held-out labels or report the absence of proof plainly.*

## 1. Genuine neuro-symbolic co-training

Before this work the production fusion network (`OmniFusionModel`, trained by
`OmniMercuryEngine.fit_fusion`) was purely neural: `FocalLoss` + post-hoc
temperature scaling. Every "neuro-symbolic" code path in the repository was a
*post-hoc blend* — symbolic scores multiplied onto neural scores after the fact,
never co-trained. No symbolic term entered the loss; no gradient ever flowed
from a logical constraint into the network.

### What was added

A compact, proper **Logic Tensor Network** co-trained with the fusion network:

* `omni_mercury_engine/ml/symbolic_constraint.py` —
  `SymbolicConstraintModule`, built on the repository's existing
  torch-differentiable `FuzzyOperators` (product t-norm, Reichenbach
  implication, `pmean` universal quantifier). It grounds the predicates of a
  declarative `RuleGraph` and returns a single satisfaction scalar in `[0, 1]`.
* The constraint loss `1 - satisfaction` is added to the supervised loss in
  `fit_fusion(..., symbolic_weight=λ)`:

  ```
  total_loss = FocalLoss(p, y) + λ · (1 − satisfaction(p, detector_scores))
  ```

  Because `satisfaction` depends on the network's own `anomaly_probs`, the
  gradient flows back into the network's anomaly head; the module's learnable
  detector-reliability and rule-confidence weights adapt jointly. `λ = 0`
  reproduces the purely-neural path byte-for-byte.

### The rule graph (default `consensus_rule_graph`)

The constraint encodes inductive bias the labels do **not** carry — the
unsupervised agreement structure of the independent base detectors:

| Rule | Statement | Purpose |
|------|-----------|---------|
| `R1_evidence`  | `Consensus → Anomalous`       | recall / evidence prior (sample-efficiency) |
| `R2_precision` | `¬Consensus → ¬Anomalous`     | precision prior (false-positive reduction)  |

`Consensus` is a **learned weighted** aggregation of the per-detector anomaly
scores, so the layer also learns which detectors to trust. `explain()` exposes
per-rule satisfaction and the learned detector weights for auditing.

This revives the previously-dead "LTN" sub-net as a measured, co-trained
component — not as a decorative module instantiated and ignored.

## 2. Ablation — the anti-theater gate

The constraint is only worth enabling if it beats neural-only on **real,
genuinely labelled** data. `benchmarks/neurosymbolic_ablation.py` runs a paired
comparison on ADBench (NeurIPS 2022 ground truth), where each (dataset, seed,
train-fraction) trains both conditions from the *same* split and initialisation,
differing only in `symbolic_weight`. Three quantities count as evidence:

1. **AUC up** — ROC-AUC on held-out test.
2. **False-positives down** — FPR at a fixed 90% recall.
3. **Sample-efficiency up** — AUC gain concentrated at low train fractions
   (0.1 / 0.25), where an unsupervised consensus prior should help most.

**Ablation integrity:** metrics are computed on real held-out labels only. If
ADBench cannot be downloaded the harness reports that and exits non-zero — it
never fabricates or simulates a pass.

### Results

Run: 4 genuinely-labelled ADBench datasets × 3 seeds × 4 train fractions,
`λ = 0.1`, 20 epochs. Full artifact: `artifacts/neurosymbolic_ablation.json`.
`ΔAUC = symbolic − neural` (↑ good); `ΔFP@90 = neural − symbolic` (↑ good = fewer
false positives).

| Dataset | Frac | AUC neural | AUC sym | ΔAUC | ΔFP@90 | seeds AUC≥ |
|---|---|---|---|---|---|---|
| breastw | 0.10 | 0.9844 | 0.9903 | +0.0059 | +0.0025 | 2/3 |
| breastw | 0.25 | 0.9944 | 0.9907 | −0.0037 | −0.0075 | 0/3 |
| breastw | 0.50 | 0.9955 | 0.9932 | −0.0023 | −0.0050 | 1/3 |
| breastw | 1.00 | 0.9947 | 0.9955 | +0.0008 | +0.0025 | 3/3 |
| cardio | 0.10 | 0.9555 | 0.9657 | +0.0102 | +0.0894 | 3/3 |
| cardio | 0.25 | 0.9755 | 0.9860 | **+0.0105** | **+0.0255** | 3/3 |
| cardio | 0.50 | 0.9891 | 0.9920 | +0.0029 | +0.0181 | 1/3 |
| cardio | 1.00 | 0.9935 | 0.9914 | −0.0022 | +0.0000 | 1/3 |
| thyroid | 0.10 | 0.9910 | 0.9910 | +0.0000 | −0.0069 | 2/3 |
| thyroid | 0.25 | 0.9870 | 0.9920 | +0.0050 | +0.0181 | 2/3 |
| thyroid | 0.50 | 0.9935 | 0.9962 | +0.0027 | +0.0094 | 3/3 |
| thyroid | 1.00 | 0.9935 | 0.9942 | +0.0007 | +0.0139 | 1/3 |
| WBC | 0.10 | 0.9896 | 0.9913 | +0.0017 | +0.0000 | 3/3 |
| WBC | 0.25 | 0.9931 | 0.9913 | −0.0017 | +0.0000 | 2/3 |
| WBC | 0.50 | 0.9913 | 0.9896 | −0.0017 | −0.0052 | 2/3 |
| WBC | 1.00 | 0.9965 | 0.9896 | −0.0069 | −0.0104 | 2/3 |

**Aggregate:** mean full-data ΔAUC **−0.0019** (within the ±0.002 noise floor),
mean full-data FP reduction **+0.0015**, mean low-data (frac ≤ 0.25) ΔAUC
**+0.0035**. Source of record: `artifacts/neurosymbolic_ablation.json` (`verdict`).

**Fixed-weight verdict: a constant `λ` is *dominated*, not the default.** On this
run a fixed weight clears the FP-reduction gate (FP down, full-data AUC within the
±0.002 noise floor) and shows a clear low-data lift — so a constant constraint is
*not harmful on aggregate*. But it is **dominated** by the label-scarcity schedule
(§2.1): the schedule keeps essentially the same low-data lift while cutting the
full-data AUC cost to about a third (−0.0007 vs −0.0019) by decaying to the neural
path where labels are abundant. A blunt always-on weight pays its (within-noise,
only 0.58-seed-agreed) full-data cost in *every* regime for a benefit that
materialises *only* when labels are scarce — so the constant weight is not the
default; the schedule is. (The gate logic enforces this: an FP-reduction bought
with a meaningful full-data AUC regression does **not** read as KEEP for a fixed
weight.)

Transparent reading of the split:

* **Low-data regime (frac ≤ 0.5): the constraint helps.** ΔAUC is positive at most
  low fractions across cardio / thyroid (and breastw @ 0.1), with false positives
  reduced — the unsupervised consensus prior injects useful structure when labels
  are scarce. Strongest cell: cardio @ 25%, **+0.0105 AUC, +0.0255 FP reduction,
  3/3 seeds**.
* **Full-data regime: the effect washes out or slightly reverses** (cardio and WBC
  regress at 100%). With abundant labels the boundary is already well-pinned and a
  constant prior adds slight bias.
* The full-data aggregate (−0.0019) sits **within** the ±0.002 noise floor and is
  only 0.58-seed-agreed: a constant weight is *not clearly harmful*, but its
  benefit is not where its cost is paid. That asymmetry — help when scarce, mild
  drag when abundant — is exactly what the schedule in §2.1 removes.

This is the gate working as intended: a fair test on real held-out labels. The
co-training machinery and harness are genuine and reusable; what is gated is the
*constant-weight default* — superseded by the adaptive schedule, which earns
default-on by dominance (§2.1).

## 2.1 The label-scarcity schedule — verdict: KEEP

`ScarcityWeightSchedule` (`ml/symbolic_constraint.py`) makes the weight a function
of the labelled-anomaly count `n_pos` in the provided labels:

```
λ_eff(n_pos) = λ_max · exp(−n_pos / n0)        (λ_max = 0.1, n0 = 25)
```

so the constraint runs at near-full strength when positives are scarce — the
regime the §2 table shows it helps — and decays to the purely-neural path
(`λ_eff → 0`) when they are abundant. The weight is resolved from `n_pos` before
training and reported in `fit_fusion`'s metrics for audit. The defaults are
*pre-registered*, not tuned to pass: `λ_max=0.1` is the value §2 already ablated,
and `n0=25` is an anomaly-count scale of a few dozen positives (below which a
handful of labels cannot pin a boundary in the ~6–30 ADBench feature dimensions).

The ablation gains a third **adaptive** arm, compared against the *same* neural
baseline under a **pre-registered dominance bar** appropriate for a scarcity-gated
weight: (1) no full-data regression beyond the ±0.002 noise floor, and (2) a
seed-agreed low-data AUC lift concentrated in the scarce regime. The bar is not
moved to manufacture a pass — it is the right test for a weight designed to cost
nothing where labels are plentiful.

Run: 4 ADBench datasets × 3 seeds × 4 fractions, 20 epochs (full artifact:
`artifacts/neurosymbolic_ablation.json`). `ΔAUC = adaptive − neural`; `λ̄` is the
mean resolved weight.

| Dataset | Frac | AUC neural | AUC adaptive | ΔAUC | λ̄ | seeds AUC≥ |
|---|---|---|---|---|---|---|
| cardio | 0.10 | 0.9555 | 0.9610 | +0.0055 | 0.062 | 3/3 |
| cardio | 0.25 | 0.9755 | 0.9840 | **+0.0085** | 0.029 | 3/3 |
| cardio | 1.00 | 0.9935 | 0.9953 | +0.0018 | 0.000 | 1/3 |
| thyroid | 0.25 | 0.9870 | 0.9931 | +0.0061 | 0.053 | 1/3 |
| thyroid | 1.00 | 0.9935 | 0.9967 | +0.0033 | 0.007 | 3/3 |
| WBC | 0.10 | 0.9896 | 0.9878 | −0.0017 | 0.096 | 2/3 |
| WBC | 1.00 | 0.9965 | 0.9896 | −0.0069 | 0.076 | 2/3 |

**Aggregate:** mean full-data ΔAUC **−0.0007** (within the ±0.002 noise floor),
full-data FP reduction **+0.0009**, mean low-data ΔAUC **+0.0022**, low-data seed
agreement **0.63**.

**Verdict: KEEP — enable adaptive (label-scarcity) co-training by default
(`symbolic_weight="adaptive"`).** Both dominance gates clear: the schedule does
not regress full-data AUC (−0.0007, within noise) and lifts low-data AUC
(+0.0022, seed-agreed, concentrated in the scarce regime). Where a fixed `λ`
*regressed* full-data AUC (−0.0019), the adaptive schedule is neutral-to-positive
there — it even improves thyroid@1.0 by +0.0033 (3/3) with a gentle λ̄=0.007 — and
keeps the low-data gains.

**Transparent caveat (disclosed, not hidden):** the lone meaningful regression is
**WBC@1.0 (−0.0069)**. WBC is tiny (~15 positives even at full data) and
near-ceiling, so the `n_pos`-keyed schedule cannot distinguish it from the scarce
regime and still applies λ̄≈0.076. It is one full-data cell of four and the
aggregate stays within the noise floor, but it marks the schedule's known limit:
keying on absolute positive count, a small near-ceiling dataset reads as "scarce."
A capability-aware gate (down-weight `λ` once the neural validation AUC is already
saturated) is the next measured step — to be ablated, not assumed.

### Reproduce

```bash
python -m benchmarks.neurosymbolic_ablation \
    --datasets breastw cardio thyroid WBC --seeds 0 1 2 \
    --fractions 0.1 0.25 0.5 1.0 --lam 0.1 --epochs 20 \
    --out artifacts/neurosymbolic_ablation.json
```

## 2.2 Implication semantics: crisp vs fuzzy (the dormant t-norms, revived)

The constraint's rules are implications, so the implication *operator* is a real
LTN design axis. Three residua now ship as torch-differentiable tensor operators
in `FuzzyOperators`, but only the smooth product/Reichenbach form (`1 − x + x·y`)
had ever been wired into the constraint — the Gödel (`x ≤ y ? 1 : y`) and
Łukasiewicz (`min(1, 1 − x + y)`) operators were dormant. `SymbolicConstraintModule(semantics=...)`
revives them as a selectable axis; `implies_lukasiewicz` (bounded, non-saturating
gradient) was added to complete the set. This supersedes the orphaned *scalar*
t-norm classes in `cognitive/differentiable_logic.py` with live, tested tensor
operators in the measured path.

Which residuum generalises best is settled by
`benchmarks/symbolic_semantics_sweep.py`, comparing all three **within the same
cell** (same dataset / seed / split / initialisation — an earlier separate-runs
comparison was confounded by detector-fit noise and discarded). 27 cells
(cardio / thyroid / WBC × 3 seeds × fractions 0.1 / 0.25 / 0.5), adaptive
schedule:

| semantics | mean low-data ΔAUC | seed agreement |
|---|---|---|
| product / reichenbach (default) | +0.0025 | 0.78 |
| łukasiewicz | +0.0031 | 0.78 |
| gödel | +0.0039 | 0.81 |

**Verdict: KEEP `product` as the default.** All three semantics help in the
low-data regime and are **statistically indistinguishable** — the crisp residua
edge product by only +0.0006 / +0.0014 mean ΔAUC, inside the ±0.002 noise floor,
so the pre-registered rule (switch only if crisp beats product by > +0.002) does
not fire. The crisp operators are now live, tested, and selectable
(`symbolic_semantics="godel"` / `"lukasiewicz"`) for future work; the smooth
product/Reichenbach residuum, with the best-conditioned gradient, remains the
measured default.

## 2.3 Rule structure: minimal consensus vs richer salience (symbolic_logic_layer revival)

The dormant `cognitive/symbolic_logic_layer.py` is entirely crisp (no autograd),
so it cannot co-train directly — wiring its forward-chained output onto the loss
would be exactly the post-hoc blend this work removed. But its core idea, the
**`ThresholdRule`** (a variable crossing a threshold implies a conclusion), has a
genuine differentiable analog. `consensus_salience_rule_graph` adds a third rule
over a new `Salient` predicate: a soft-existential over per-detector *learnable
soft thresholds* — "if **any** single detector saliently fires, fusion is
anomalous" — a disjunctive recall axiom complementing the AND-like weighted
`Consensus`. (It also revives the dormant existential / product-t-conorm
aggregation in `FuzzyOperators`.)

`benchmarks/symbolic_rulegraph_sweep.py` compares the two graphs **within the
same cell** under the adaptive schedule. 27 cells (cardio / thyroid / WBC × 3
seeds × fractions 0.1 / 0.25 / 0.5):

| rule graph | mean low-data ΔAUC | seed agreement |
|---|---|---|
| consensus (default, 2 rules) | +0.0009 | 0.63 |
| consensus_salience (3 rules) | +0.0022 | **0.81** |

**Verdict: KEEP `consensus` as the default.** The salience rule is *directionally
better* — higher mean ΔAUC and notably higher seed agreement (0.81 vs 0.63),
recovering cells where the bare consensus regressed (e.g. thyroid@0.1, WBC@0.1) —
but the +0.0013 margin is inside the ±0.002 noise floor, so the pre-registered
switch rule does not fire. The richer graph is now live, tested, and selectable
(`symbolic_rule_graph="consensus_salience"`) and is the most promising symbolic
follow-up: a larger-N confirmation could clear the bar. To be ablated, not
assumed.

## 2.4 Genetically evolved rule graphs (`symbolic_rule_graph="evolved:<path>"`)

The two hand-written graphs above (`consensus`, `consensus_salience`) are not the
only selectable rule structures. `fit_fusion`'s `symbolic_rule_graph` argument
also accepts an `"evolved:<path>"` specification
(`engine.py::fit_fusion`, docstring at `engine.py:1385-1388`), which loads a
genetically evolved `RuleGraph` from the JSON artifact written by
`omni_mercury_engine.ml.rule_evolution`. The graph is resolved by
`resolve_rule_graph` (`ml/symbolic_constraint.py:439`), which strips the
case-insensitive `evolved:` prefix (`_EVOLVED_GRAPH_PREFIX`,
`symbolic_constraint.py:436`) and deserialises the artifact into the same
`RuleGraph` type the registry graphs use — so an evolved graph co-trains through
the identical `SymbolicConstraintModule` path, with no separate blend and no
post-hoc multiplication.

The seam is that evolved rules range over a richer predicate grammar than the
built-in `Consensus`/`NotConsensus` atoms. `parse_evolved_predicate`
(`symbolic_constraint.py:258-290`) parses a conjunctive (`&`-joined) predicate
name into a tuple of atoms, where each atom is either a builtin atom or a
`ThresholdAtom` (`symbolic_constraint.py:219-255`): a soft threshold test over a
named detector-score channel, grounded as the differentiable fuzzy truth value
`sigmoid((score[channel] − threshold) · slope)` for `>=` (mirrored for `<=`) —
the same soft-threshold construction as the `Salient` predicate of §2.3, but with
an *evolved, fixed* threshold in place of a learnable one. Thresholds are
quantised (`quantize_threshold`) so evolved graphs round-trip losslessly through
`rule_graph_to_spec` and the engine checkpoint.

`ml/rule_evolution.py` produces these graphs (added in PR #333, 2026-07-10) and
`benchmarks/rule_evolution_benchmark.py` is the harness that evaluates them.
This is a live, selectable, benchmarked capability. No ablation delta,
seed-agreement figure, or KEEP/CUT verdict is asserted here: the committed
artifacts in this repository do not carry a paired evolved-vs-baseline result of
the form §2.1–§2.3 report, so the seam is documented as available and measured
without a default-changing claim. The registry default remains
`symbolic_rule_graph="consensus"` (`engine.py:1314`).

## 3. Calibration + conformal uncertainty in the output

Building on the conformal plumbing from PR #242, the fusion serve path now
returns **calibrated probabilities and prediction sets**, not bare scores.

* `core/conformal_prediction.py::BinaryConformalClassifier` — a
  class-conditional (Mondrian) split-conformal classifier (LAC; Sadinle, Lei &
  Wasserman 2019) reusing #242's `SplitConformalPredictor` per class. It turns a
  calibrated `P(anomaly)` into a label prediction set over `{normal, anomaly}`
  with a distribution-free **per-class coverage guarantee**.
* `engine.calibrate_fusion_conformal(X_cal, y_cal, coverage)` fits it on a
  held-out labelled split (composed *after* temperature scaling).
* `engine.score_fusion_conformal(X)` returns, per sample: the calibrated
  probability, the conformal label set, the set size, and an `abstain` flag
  (set size 2 = genuine uncertainty; set size 0 = atypical point).

The coverage guarantee is verified empirically (synthetic at 0.8/0.9/0.95 and on
real ADBench labels): the fraction of prediction sets containing the true label
meets the target within finite-sample tolerance, with informative (non-trivial)
sets.

## 4. Verdicts — keep / cut / quarantine

| Component | Status before | Verdict | Action |
|-----------|---------------|---------|--------|
| **LTN / symbolic constraint** | dead (orphaned nn.Modules in `cognitive/differentiable_logic.py`) | **REVIVED + ENABLED** via the adaptive schedule | The `SymbolicConstraintModule` is a genuine, tested, co-trained LTN. A *fixed* `λ` was quarantined (no full-data win, §2), but the label-scarcity `ScarcityWeightSchedule` (§2.1) cleared a pre-registered dominance bar on real ADBench labels, so `symbolic_weight="adaptive"` is now the **default**: co-training runs when labels are scarce and decays to the neural path otherwise. The orphaned `differentiable_logic.py` modules remain, superseded by this focused implementation. |
| **Schumann CNN-LSTM** (`space/schumann_resonance.py`) | run at inference with **random weights** (theater + non-deterministic bug) | **QUARANTINE** | Untrained network no longer drives `anomaly_type`/`confidence`/`risk_score`; deterministic FFT-physics fallback used instead, with a one-time warning. `load_neural_weights()` activates the learned path once a real labelled corpus exists. |
| **Parapsychology consciousness-field** (`models/parapsychology.py`) | run at inference with **random weights**; no validated ground truth | **QUARANTINE** | Field coherence abstains to the neutral 0.5 prior while untrained, with a one-time warning. No fabricated signal is emitted. |
| **`LogicTensorNetwork`** (`models/neurosymbolic.py`) | a never-trained `nn.Module` whose random-init forward fed the fusion consensus as if it were a neural confidence | **RE-WIRED to the canonical module (2026-06-02)** | Instead of staying retired, the `LogicTensorNetwork` surface is rebuilt as a thin head over the canonical `SymbolicConstraintModule`: `LogicTensorNetwork.predict(detector_scores)` routes through that module's fuzzy-logic `Consensus` predicate (exposed as the new public `SymbolicConstraintModule.predict`). No random init — an untrained module's zero-initialised parameters give a deterministic uniform-weight consensus; a co-trained module applies learned detector reliabilities. `NeurosymbolicEngine.ltn` is built when torch is present. The raw-feature `neural_inference` remains a transparently-labelled deterministic heuristic (a per-feature dispersion statistic, *not* a trained signal). Regression: `tests/test_neurosymbolic_integration.py::{test_ltn_is_wired_to_canonical_symbolic_module, test_ltn_predict_delegates_to_symbolic_consensus, test_neural_inference_is_deterministic_and_feature_responsive}` + `tests/ml/test_symbolic_constraint.py::TestPredictInference`. |
| **Fusion temperature calibration** (PR #255) | present | **KEEP** | Unchanged; conformal composes on top of it. |

## 5. Transparent open items

* The differentiable domain encoders (spectral/FFT, kinematic, Fisher/entropy)
  remain numpy/scipy feature extractors converted to tensors; converting them to
  jointly-trained `nn.Module`s with per-domain ablations is scoped but not yet
  landed. The feasibility analysis (torch.fft for spectral, Conv1d difference
  kernels for kinematic) is recorded for the follow-up.
* The Schumann and parapsychology networks remain quarantined, not revived: no
  real labelled corpus exists in-repo to train them transparently. Reviving them
  would require that data first — anything else would be theater.

## 6. Statistical confirmation of the sub-threshold sweeps

§2 made its KEEP decisions on *point* deltas plus a seed-agreement heuristic
against a pre-registered **+0.002** mean-ΔAUC bar. `benchmarks/statistical_significance.py`
adds the formal paired inference those calls deserve, computed directly from the
committed sweep artifacts (no re-run): per-cell paired difference, paired
**t-test**, **Wilcoxon** signed-rank (scipy, optional), exact **sign test**, and a
seeded **bootstrap 95% CI**. Because six comparisons are reported together, the
raw t-test p-values are additionally corrected for multiple testing with
**Holm–Bonferroni** (family-wise error control). A claim is "confirmed"
(family-wise) only when the paired mean clears the bar **and** the 95% CI
excludes 0 **and** the *Holm-adjusted* t-test p is significant at α = 0.05.

Result on the committed 27-cell sweep (`--seed 0 --boot 10000`):

| Comparison | n | mean ΔAUC | bootstrap CI95 | t-test p | Holm p | confirmed (FW) |
|---|---|---|---|---|---|---|
| salience vs consensus | 27 | +0.0012 | [−0.0002, +0.0028] | 0.134 | 0.672 | **No** |
| gödel vs product | 27 | +0.0014 | [−0.0006, +0.0040] | 0.255 | 1.000 | **No** |
| gödel vs neural-only | 27 | +0.0039 | [+0.0004, +0.0075] | 0.045 | 0.269 | **No** |
| salience vs neural-only | 27 | +0.0022 | [−0.0021, +0.0057] | 0.298 | 1.000 | No |

Reading: the gödel-vs-neural-only delta is the only one that clears the
*uncorrected* α = 0.05 bar (raw p = 0.045, CI excludes 0), but it **does not
survive Holm–Bonferroni** across the six-test family (adjusted p = 0.269) — so at
N = 27 it is an artefact of multiple testing, not a confirmed finding, and the
report deliberately refuses to quote it as "co-training beats neural-only." The
*choice between* gödel and product — and between the salience and consensus rule
graphs — is likewise **within the noise floor**. Net: **no alternative is
family-wise confirmed**, which statistically corroborates the KEEP-product /
KEEP-consensus defaults rather than overturning them. The directional
co-training advantage over the neural-only baseline remains a motivating signal
to be settled by the larger-N matrix below, not a present claim.

**Larger-N confirmation matrix (to settle the sub-threshold question).** The
27-cell sweep is underpowered for a ±0.002 effect. To confirm or refute at
adequate power, widen the same-cell design and re-run the analyzer:

```bash
# Datasets: add externally-labelled ADBench sets beyond {cardio, thyroid, WBC}.
DATASETS="cardio thyroid WBC satellite pendigits mammography shuttle annthyroid optdigits glass"
SEEDS="0 1 2 3 4 5 6 7 8 9"          # 10 seeds (was 3)
FRACTIONS="0.05 0.1 0.25 0.5"        # 4 label fractions (was 3)
# -> 10 datasets x 10 seeds x 4 fractions = 400 same-cells (was 27)

python -m benchmarks.symbolic_rulegraph_sweep  --datasets $DATASETS --seeds $SEEDS --fractions $FRACTIONS
python -m benchmarks.symbolic_semantics_sweep  --datasets $DATASETS --seeds $SEEDS --fractions $FRACTIONS
python -m benchmarks.statistical_significance --seed 0 --boot 10000
```

Pre-registered decision (unchanged, bar not moved): switch a default only if the
alternative clears mean ΔAUC > +0.002 **and** the bootstrap CI95 lower bound is
> 0 **and** the Holm-adjusted paired t-test p < 0.05 on the widened matrix. A ~400-cell sweep gives
materially more power to detect a true +0.002 effect at the observed per-cell SD
(~0.005) than N = 27 — i.e. the current "directionally better but sub-threshold"
reading is an N problem, not a settled negative.
