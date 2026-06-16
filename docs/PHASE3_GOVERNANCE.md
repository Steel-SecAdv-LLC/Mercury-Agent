# Phase 3 Governance: Reflexion, Drift Recalibration, Dormant Revival

Phase 3 closes the next governed self-improvement loop **at the point a change
would take effect**. Reflexion threshold adaptation, drift-/performance-triggered
recalibration, and dormant-module revival can no longer mutate Mercury's live
behaviour autonomously: each proposed change is reviewed by a governance policy
that decides — fail closed by default — whether it may be applied.

## The seam (engine-owned) and the policy (research-injected)

The interface lives in the engine so the production packages never depend on the
research tree:

* `src/omni_mercury_engine/governance/self_improvement.py` defines
  `ThresholdGovernance` and `RecalibrationGovernance` (the protocols the engine
  consults), the `ProposedThresholdChange` / `ProposedRecalibration` /
  `GovernanceReview` records, and two built-in policies:
  * `FailClosedSelfImprovementGovernance` — the default. Withholds every
    autonomous change. A live operating point or model changes only through an
    evidence-backed, human-approved promotion.
  * `MeasurementGovernance` — an explicit, named opt-in for held-out measurement
    harnesses (the orchestration and online-learning validation benchmarks)
    where applying the adaptation *is* the measurement.

The concrete gate-backed policies live in the research tier and are injected at
composition time (dependency: research → engine):

* `research/governed_fusion/phase3_governance_adapters.py` —
  `PromotionGateThresholdGovernance` and `PromotionGateRecalibrationGovernance`
  implement the seam by routing a live proposal through the Phase 2 promotion
  gate. They never authorise autonomous application: a gate `promote` is *queued
  for human approval*; everything else (including "no candidate evidence") is
  *withheld*.

## Contract

Every Phase 3 candidate must satisfy the Phase 2 gate in
`research/governed_fusion/promotion_gate.py`:

1. `optimization_bucket` remains fixed to `external_label`.
2. Held-out replay improves AUROC or F1 without AUROC/AUPRC/F1 regression.
3. σ_Immutable, benevolence, conformal coverage, and Lyapunov floors remain
   intact.
4. Capability regression evidence is present and passing.
5. Latest `status="ok"` marginal-ablation metrics are not regressed.
6. Promotion remains human-review gated.

## Reflexion executor wiring (live)

`MultiAgentOrchestrator.reflect()` reads
`AnomalyReflexion.get_threshold_recommendation()`. A `maintain` recommendation is
a no-op. An actionable `increase` / `decrease` recommendation becomes a
`ProposedThresholdChange` handed to the orchestrator's `threshold_governance`
policy:

* Default (`FailClosedSelfImprovementGovernance`): withheld — the live operating
  point does not move.
* Gate-backed without held-out-replay evidence: the gate returns `reject`;
  withheld.
* Gate-backed with passing evidence: the gate returns `promote`, but promotion
  is human-review gated, so the candidate is **queued**, not auto-applied.

The returned `ReflectionRecord` reports the effective outcome (`applied`) and the
governance disposition (`governance_outcome`, `governance_reasons`,
`governance_record`). Install the production policy when enabling the
orchestrator:

```python
from research.governed_fusion.phase3_governance_adapters import (
    PromotionGateThresholdGovernance,
)

orchestrator = engine.enable_multi_agent_orchestration(
    threshold_governance=PromotionGateThresholdGovernance(
        manifest=manifest, ledger=ledger
    ),
)
```

## Drift-triggered recalibration (live)

`OnlineLearningPipeline` routes its autonomous retrain triggers through an
optional `recalibration_governance` policy before `model.fit()`. Recalibration
is considered only after `is_drift=true` with `severity` of `high` or `critical`
(matching `route_drift_recalibration`); medium/low drift stays observable.
Performance-degradation retrains are also routed and, lacking a governed drift
surface, are withheld under a gate-backed policy. Used standalone (no policy
installed) the pipeline remains an autonomous online learner by construction;
`force_retrain()` is an operator action and is never gated.

## Dormant-module revival (offline) — measure → route

`route_dormant_revival_candidate()` consumes verdicts from
`benchmarks/dormant_module_revival.py`. A module below the pre-registered
standalone signal bar is archived. A module that clears the bar becomes
promotion-gate candidate evidence; revival still requires external-label replay
lift, safety floors, capability evidence, and ablation integrity.

The scheduled workflow `.github/workflows/phase3-governance.yml` now **closes the
loop**: it runs the real benchmark on schedule, then routes every measured
verdict through the gate (`--dormant-revival`) and publishes both the measurement
and the routing decisions. A candidate that carries signal but lacks held-out
promotion evidence is recorded as a fail-closed reject — the honest disposition.
Pull requests run the deterministic routing tests without network downloads.

## CLI

Composite Phase 3 reports — or a dormant-revival benchmark report directly — can
be evaluated with:

```bash
# Route a composite evidence report.
python research/governed_fusion/phase3_governance.py \
  --report artifacts/phase3_report.json \
  --check

# Route a dormant-module revival benchmark report (closes the measure->route loop).
python research/governed_fusion/phase3_governance.py \
  --dormant-revival artifacts/dormant_module_revival.json \
  --out artifacts/dormant_module_revival_routing.json
```

`--check` exits non-zero on `reject` or `rollback`. Without `--out`, the module
writes append-only decision records under `artifacts/phase3_governance/`.

## Verification

* Routing logic: `tests/research/test_phase3_governance.py`.
* Live wiring end-to-end against the real gate:
  `tests/research/test_phase3_live_wiring.py` — the autonomous threshold move and
  the autonomous drift retrain are both observed to be withheld by default,
  applied only under an explicit measurement stance, and queued (never
  auto-applied) when the gate-backed policy clears them.
