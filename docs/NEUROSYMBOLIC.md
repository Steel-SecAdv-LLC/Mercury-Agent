# Neuro-Symbolic Fusion, Ablation, and Conformal Uncertainty

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
| breastw | 0.10 | 0.9844 | 0.9844 | +0.0000 | −0.0075 | 2/3 |
| breastw | 0.25 | 0.9942 | 0.9928 | −0.0015 | −0.0050 | 0/3 |
| breastw | 0.50 | 0.9955 | 0.9936 | −0.0019 | −0.0075 | 1/3 |
| breastw | 1.00 | 0.9934 | 0.9951 | +0.0018 | +0.0075 | 2/3 |
| cardio | 0.10 | 0.9615 | 0.9627 | +0.0013 | +0.0101 | 2/3 |
| cardio | 0.25 | 0.9738 | 0.9853 | **+0.0115** | **+0.0296** | 3/3 |
| cardio | 0.50 | 0.9897 | 0.9931 | +0.0034 | +0.0195 | 2/3 |
| cardio | 1.00 | 0.9946 | 0.9952 | +0.0006 | +0.0074 | 2/3 |
| thyroid | 0.10 | 0.9887 | 0.9915 | +0.0028 | +0.0000 | 2/3 |
| thyroid | 0.25 | 0.9871 | 0.9906 | +0.0035 | +0.0048 | 1/3 |
| thyroid | 0.50 | 0.9925 | 0.9961 | +0.0036 | +0.0205 | 3/3 |
| thyroid | 1.00 | 0.9974 | 0.9960 | −0.0014 | −0.0066 | 0/3 |
| WBC | 0.10 | 0.9896 | 0.9913 | +0.0017 | +0.0000 | 3/3 |
| WBC | 0.25 | 0.9896 | 0.9931 | +0.0035 | +0.0104 | 3/3 |
| WBC | 0.50 | 0.9896 | 0.9913 | +0.0017 | +0.0052 | 3/3 |
| WBC | 1.00 | 0.9965 | 0.9931 | −0.0035 | −0.0104 | 2/3 |

**Aggregate:** mean full-data ΔAUC **−0.0006**, mean full-data FP reduction
**−0.0005**, mean low-data (frac ≤ 0.25) ΔAUC **+0.0029**.

**Verdict: QUARANTINE — keep `symbolic_weight = 0` by default.** No gate cleared
its conservative threshold (AUC↑ > +0.002 full-data; FP↓ > 0 full-data;
sample-efficiency↑ > +0.005 low-data).

Honest reading of the split result:

* **Low-data regime (frac ≤ 0.5): the constraint helps.** On the three
  non-ceiling datasets (cardio, thyroid, WBC) ΔAUC is positive at every low
  fraction, often with 2–3/3 seeds agreeing, and false positives drop — the
  unsupervised consensus prior does inject useful structure when labels are
  scarce. The strongest cell is cardio @ 25%: +0.0115 AUC, +0.030 FP reduction,
  3/3 seeds.
* **Full-data regime: the effect washes out or slightly reverses** (thyroid and
  WBC regress at 100%). With enough labels the boundary is already well-pinned
  and the consensus prior adds slight bias.
* The low-data mean (+0.0029) is a real but **sub-threshold** signal: it does not
  clear the +0.005 bar set a priori. Per the anti-theater rule, the bar is **not**
  moved to manufacture a pass — the constraint stays off by default.

This is the gate working as intended: a fair test on real held-out labels that
does not clear the bar yields an honest quarantine, not a fabricated win. The
co-training machinery and harness are genuine and reusable; only the *default* is
gated off. A label-scarcity-targeted schedule (enable `λ>0` only when few labels
are available, suggested by the low-data cells) is a measured follow-up — to be
ablated, not assumed.

### Reproduce

```bash
python -m benchmarks.neurosymbolic_ablation \
    --datasets breastw cardio thyroid WBC --seeds 0 1 2 \
    --fractions 0.1 0.25 0.5 1.0 --lam 0.1 --epochs 20 \
    --out artifacts/neurosymbolic_ablation.json
```

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
| **LTN / symbolic constraint** | dead (orphaned nn.Modules in `cognitive/differentiable_logic.py`) | **REVIVED** as machinery; **constraint QUARANTINED** by the ablation | The `SymbolicConstraintModule` is a genuine, tested, co-trained LTN — but the default consensus constraint at `λ=0.1` did not clear the ablation bar, so `symbolic_weight=0` stays the default. The infrastructure is kept (real and reusable); the default is off. The orphaned `differentiable_logic.py` modules remain but are superseded by this focused, measured implementation. |
| **Schumann CNN-LSTM** (`space/schumann_resonance.py`) | run at inference with **random weights** (theater + non-deterministic bug) | **QUARANTINE** | Untrained network no longer drives `anomaly_type`/`confidence`/`risk_score`; deterministic FFT-physics fallback used instead, with a one-time warning. `load_neural_weights()` activates the learned path once a real labelled corpus exists. |
| **Parapsychology consciousness-field** (`models/parapsychology.py`) | run at inference with **random weights**; no validated ground truth | **QUARANTINE** | Field coherence abstains to the neutral 0.5 prior while untrained, with a one-time warning. No fabricated signal is emitted. |
| **Fusion temperature calibration** (PR #255) | present | **KEEP** | Unchanged; conformal composes on top of it. |

## 5. Honest open items

* The differentiable domain encoders (spectral/FFT, kinematic, Fisher/entropy)
  remain numpy/scipy feature extractors converted to tensors; converting them to
  jointly-trained `nn.Module`s with per-domain ablations is scoped but not yet
  landed. The feasibility analysis (torch.fft for spectral, Conv1d difference
  kernels for kinematic) is recorded for the follow-up.
* The Schumann and parapsychology networks remain quarantined, not revived: no
  real labelled corpus exists in-repo to train them honestly. Reviving them
  would require that data first — anything else would be theater.
