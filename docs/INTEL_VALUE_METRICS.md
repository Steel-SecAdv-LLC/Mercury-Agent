# Mercury Intelligence Layer — Value Metrics

Applies to Mercury Agent **v2.1.x**. Last updated: 2026-07-11.

Every intelligence-layer stream ships a **measured** value metric with a
**baseline** and a **target**. A stream that cannot state, and measure, the value
it delivers is theater. This document is the operator/reviewer view of that value
board: what each stream measures, what "good" is, and which command produces the
number.

- Source of truth: `omni_mercury_engine.intel.value_metrics.VALUE_METRICS`
  (in `src/omni_mercury_engine/intel/value_metrics.py`).
- Report benchmark: `benchmarks/intel_value_metrics_report.py` renders the whole
  board as `baseline -> measured (target)` per stream.
- The `ci/*` intel lanes and the unit tests **import** `VALUE_METRICS` rather
  than hard-coding thresholds, so a target is defined **once** and can never
  silently drift between the doc, the gate, and the test.

## 1. The value board

`VALUE_METRICS` is a `dict[str, ValueMetric]`, one entry per stream. Each
`ValueMetric` is a frozen dataclass with fields `stream`, `metric`, `unit`,
`direction` (`Direction.HIGHER_IS_BETTER` / `Direction.LOWER_IS_BETTER`),
`baseline`, `target`, `description`, and `aspirational` (`bool`, default
`False` — when set, the metric's `target` is treated as non-gating, as for
`adversarial_co_training`; see §5).

| Stream | Metric | Direction | Baseline | Target | How measured |
|---|---|---|---|---|---|
| `closed_feedback_loop` | `poisoned_candidate_block_rate` | higher-is-better | `0.0` | `1.0` | Fraction of poisoned retrain candidates the OOF/adversarial regression gate blocks; measured by `scripts/closed_loop_demo.py --poisoned` and `tests/intel/test_feedback_loop.py`. |
| `confidence_cascade` | `compute_saved_at_bounded_accuracy` | higher-is-better | `0.0` | `0.50` | Fraction of heavy-path calls avoided at accuracy loss within tolerance vs the all-heavy baseline; measured by `benchmarks/confidence_cascade_report.py`. |
| `self_consistency` | `disagreement_error_auroc` | higher-is-better | `0.50` | `0.70` | AUROC of the N-sample disagreement signal predicting prediction error on a held-out set; measured by `benchmarks/self_consistency_signal_report.py`. |
| `adversarial_co_training` | `gate_bypass_survival_rate` | lower-is-better | `0.34` | `0.0` | Surviving-bypass rate of the red-team harness against the gate; pinned no-weakening floor in `benchmarks/red_team_baseline.json`; measured by `benchmarks/red_team_harness.py --check`. |
| `verifier_in_loop` | `false_claim_block_rate` | higher-is-better | `0.0` | `1.0` | Fraction of oracle-refuted symbolic claims blocked in hard mode; measured by `tests/intel/test_verifier_loop.py`. |
| `provenance` | `boundary_provenance_enforcement_rate` | higher-is-better | `0.0` | `1.0` | Fraction of provenance-required emissions enforced (unprovenanced ones withheld) at the boundary; measured by `tests/intel/test_provenance.py`. |

## 2. Baseline vs. target

- **baseline** — the value *before* the stream exists, or the **no-weakening
  floor** the stream must never regress past. For the four `0.0` baselines it is
  "what happens with the stream absent" (a poisoned candidate lands, a false
  claim emits, an unprovenanced answer ships, nothing is saved). For
  `self_consistency` it is chance (AUROC `0.50`). For `adversarial_co_training`
  it is an empirically-pinned survival ceiling, not zero — see §5.
- **target** — the goal value the stream aims to reach. For `HIGHER_IS_BETTER`
  streams a larger measured value is the improvement; for `LOWER_IS_BETTER`
  (`adversarial_co_training`) a smaller one is.

`test_metrics_are_internally_consistent` (in `tests/intel/test_value_metrics.py`)
enforces that a target is an improvement over — or equal to — its baseline in the
declared direction, so a mislabeled row cannot land.

## 3. Adjudication: `meets_target` / `improves_on_baseline`

`ValueMetric` exposes two predicates over a measured value:

- `meets_target(measured)` — did the stream reach its goal?
  - `HIGHER_IS_BETTER`: `measured >= target`.
  - `LOWER_IS_BETTER`: `measured <= target`.
- `improves_on_baseline(measured)` — the **no-weakening check**: is the value at
  least as good as baseline?
  - `HIGHER_IS_BETTER`: `measured >= baseline`.
  - `LOWER_IS_BETTER`: `measured <= baseline` (sitting *at* the floor is not a
    weakening; rising above it is).

**`NaN` fails closed.** A metric that could not be computed is `NaN`, and both
predicates return `False` for it — a stream that failed to measure is never
treated as meeting its target or holding its floor.

```python
from omni_mercury_engine.intel.value_metrics import VALUE_METRICS

m = VALUE_METRICS["adversarial_co_training"]   # baseline 0.34, target 0.0, lower better
m.meets_target(0.0)            # True
m.meets_target(0.1)           # False
m.improves_on_baseline(0.34)  # True  — at the floor is not a weakening
m.improves_on_baseline(0.5)   # False — above the floor is a weakening
m.meets_target(float("nan"))  # False — fails closed
```

`get_value_metric(stream)` returns the `ValueMetric` for a stream and raises
`KeyError` for an unregistered one — a stream must declare a value metric
deliberately, never be measured after declaring none.

## 4. The report benchmark

`benchmarks/intel_value_metrics_report.py` collects the measured number for each
stream and renders the board as `baseline -> measured (target)`, using
`ValueMetric.summarize(measured)` for the per-stream verdict row
(`meets_target` / `improves_on_baseline` flags plus the raw values from
`as_dict`). One glance tells a reviewer which streams reach their target and which
are merely holding their floor.

```bash
PYTHONPATH=src python benchmarks/intel_value_metrics_report.py
```

Because the thresholds live only in `VALUE_METRICS`, the report, the CI lanes, and
the unit tests all read the same baseline/target. The stream tests import them
directly — e.g. `tests/intel/test_cascade.py`, `tests/intel/test_self_consistency.py`,
`tests/intel/test_verifier_loop.py`, and `tests/intel/test_provenance.py` each
assert their measured value against `VALUE_METRICS[<stream>].target`.

## 5. `adversarial_co_training`: aspirational target vs. enforced floor

This row is the exception to "target is the gate". Its **target is `0.0`** —
drive every surviving bypass to zero — but that is **aspirational**, reached by
triaging survivors into the corpus/pending set, not by a single benchmark run.

The **enforced CI floor is the pinned baseline `0.34`**. The dominant bypass class
is character obfuscation (spacing/punctuation) that defeats lexical matching; the
deterministic first-run survival rate (`0.333333`, see
`benchmarks/red_team_baseline.json`) is rounded up to the pinned no-weakening
ceiling `0.34`. The lane checks `improves_on_baseline`, not `meets_target`:

```bash
PYTHONPATH=src python benchmarks/red_team_harness.py --check    # no-weakening gate (exit 1)
PYTHONPATH=src python benchmarks/red_team_harness.py --update   # (re)pin the survival-rate baseline
```

`--check` fails (exit 1) when a gate change *raises* the survival rate above the
floor — i.e. weakens the gate. It reads the floor from the pinned
`red_team_baseline.json` (`survival_rate`, `0.333333`) and separately enforces
that this floor stays ≤ the declared
`VALUE_METRICS["adversarial_co_training"].baseline` (`0.34`, the ceiling), so a
higher floor can never be pinned without first re-declaring the value metric.
Triaged survivors feed the corpus to push the measured rate down toward the
aspirational `0.0` over time.

## 6. Measuring each stream

```bash
# closed_feedback_loop — poisoned candidate must be BLOCKED by the regression gate
PYTHONPATH=src python scripts/closed_loop_demo.py --poisoned
PYTHONPATH=src python -m pytest tests/intel/test_feedback_loop.py

# confidence_cascade — compute saved at bounded accuracy vs all-heavy
PYTHONPATH=src python benchmarks/confidence_cascade_report.py

# self_consistency — AUROC of disagreement predicting error on held-out
PYTHONPATH=src python benchmarks/self_consistency_signal_report.py

# adversarial_co_training — surviving-bypass rate against the gate (no-weakening)
PYTHONPATH=src python benchmarks/red_team_harness.py --check

# verifier_in_loop — oracle-refuted claims blocked in hard mode
PYTHONPATH=src python -m pytest tests/intel/test_verifier_loop.py

# provenance — provenance-required emissions enforced at the boundary
PYTHONPATH=src python -m pytest tests/intel/test_provenance.py

# whole board, rendered baseline -> measured (target)
PYTHONPATH=src python benchmarks/intel_value_metrics_report.py
```

## 7. Quick triage

| Symptom | Check |
|---|---|
| A stream reports `meets_target=False` but `improves_on_baseline=True` | The stream holds its floor but has not reached its goal — expected for `adversarial_co_training`; investigate for the others. |
| A metric row shows `NaN` / `meets_target=False` with no measured value | Measurement failed to compute; it fails closed. Fix the benchmark before trusting the board. |
| `red_team_harness.py --check` exits 1 | A gate change raised the bypass survival rate above the pinned floor (weakening) — do not merge; re-triage survivors or revert. |
| Threshold in doc/test/gate disagree | Something hard-coded a value instead of importing `VALUE_METRICS`; route it through the registry. |
| `get_value_metric(stream)` raises `KeyError` | The stream declared no value metric — add a `ValueMetric` to `VALUE_METRICS` before measuring it. |
