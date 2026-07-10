# Counterfactual Explanations & Genetic Rule Evolution

Applies to Mercury Agent v2.1.x.

## Counterfactuals on the detection path

A counterfactual answers: *what minimal change to this input would flip the
detector's decision?* Mercury computes **verified minimal counterfactuals**:

* **Flip verification**: every candidate is re-scored through the **actual
  detector decision function** (`explainability/detection_counterfactuals.py`
  wraps the real score function; flip = crossing the real flag threshold).
* **Minimality**: after a flip is found, a per-feature minimization pass
  walks each changed feature back toward the original; a change survives
  only if reverting it un-flips the decision, and the final counterfactual
  is re-scored once more (`minimal=True` only when this verification
  passes). Minimality is therefore *verified 1-sparsity-at-a-time
  irreducibility*, not an optimizer's claim.

### Methods

`wachter` | `dice` | `growing_spheres` | `prototype` | `genetic`
(`explainability/counterfactuals.py`). The gradient methods (Wachter, DiCE)
carry a **finite infeasibility barrier**: a candidate region where the
detector cannot score (non-finite output, or a fail-loud score wrapper
raising `NonFiniteScoreError`) repels the optimizer with a large finite
penalty instead of NaN-poisoning or aborting the search. Search failures are
logged with their exception type and recorded honestly — a failed search can
never be reported as a successful flip because validity is always re-scored.

`GeneticCounterfactual` is a seeded, derivative-free GA (tournament
selection, uniform crossover, range-scaled mutation) whose proximity term
uses **Gower distance when feature metadata is present** and L2 otherwise —
it works on piecewise-constant detection scores where gradients stall.

### Measured validation (committed)

`benchmarks/counterfactual_validation.py` explains the strongest
true-positive detections of a `MercuryAnomalyDetector` fitted on the real
ADBench **WBC** dataset; flip and minimality are re-scored through the real
`detect()` path. Committed results
(`benchmarks/counterfactual_validation_results.json`, seed 0):

| method | flip rate | verified minimality |
|---|---|---|
| wachter | 1.00 | 1.00 |
| prototype | 1.00 | 1.00 |
| genetic | 1.00 | 1.00 |
| dice | 0.00 | 0.00 |
| growing_spheres | 0.00 | 0.00 |

The dice/growing_spheres zeros are **honest structural results** on this
piecewise batch-scorer regime (their smooth-boundary correctness is locked
by unit tests in `tests/explainability/`); they are reported, not tuned away.

Reproduce: `PYTHONPATH=src:. python benchmarks/counterfactual_validation.py`
(provenance in the results file records the commit of the tree that produced
the run).

### Surfaces

* **CLI**: `mercury-agent tier-detect -i data.csv --counterfactual`
* **HTTP**: `POST /api/v1/detect/tier {"include_counterfactual": true}`;
  flagship GDPR-style report: `POST /api/v1/detect/flagship
  {"gdpr_report": true, "subject_id": ...}` (403 = ethical-gate refusal,
  fail-closed).
* **MCP**: `mercury_tier_detect` (counterfactual fields),
  `mercury_detect_fusion` (`gdpr_report`).

All defaults are **off**; candidate evaluation re-scores through the real
detector, so budgets are bounded.

## Genetic rule evolution

`ml/rule_evolution.py` evolves the neuro-symbolic rule graph with a
deterministic GA (seeded selection/mutation/crossover, elitism, patience).

* **No train/serve skew**: evolved predicates resolve into the existing
  `Rule`/`RuleGraph` representation and fitness is scored by the *same
  deployed* `SymbolicConstraintModule` scoring path used at serve time.
* **Fitness = mean held-out validation F1** across pre-registered real
  ADBench datasets (cardio, thyroid, WBC, Pima); thresholds are fit on the
  validation split only; **the test split is scored exactly once**, after
  the search finishes.
* The hand-written consensus graph is seeded into the initial population,
  so evolution can only be selected if it genuinely beats it.

### Measured result (committed, reproduced 2026-07-09)

`benchmarks/rule_evolution_results.json` (seed 0, pop 40, 30 generations):
evolved graph beats the consensus baseline on held-out test F1,
**mean 0.5439 vs 0.4119 (+0.1320)** — cardio +0.296, thyroid +0.227, WBC
tie, Pima +0.005; AUC higher on all four datasets. Reproduce:

```bash
PYTHONPATH=src:. python benchmarks/rule_evolution_benchmark.py
```

The champion ships as a schema-versioned artifact
(`benchmarks/evolved_rule_graph.json`) loadable through the same seam as the
hand-written graphs:

```python
engine.fit_fusion(..., symbolic_rule_graph="evolved:benchmarks/evolved_rule_graph.json")
```

`tests/ml/test_rule_evolution.py` locks operator mechanics, split
discipline, determinism, the committed champion artifact (it must load and
score through the deployed module), and the end-to-end GA on the committed
recorded-real Pima fixture (`tests/fixtures/adbench/pima_real.npz`) so the
real-data tests run in every offline CI lane.
