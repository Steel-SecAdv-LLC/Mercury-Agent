# Phase 3 Governance: Reflexion, Drift Recalibration, Dormant Revival

Phase 3 closes the next governed self-improvement loop without bypassing the
Phase 2 promotion gate. Reflexion, drift monitoring, and dormant-module revival
now produce candidate evidence; none of them directly changes runtime behavior.

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

## Reflexion executor wiring

`research/governed_fusion/phase3_governance.py` adds
`route_reflexion_executor()`, which reads
`AnomalyReflexion.get_threshold_recommendation()` and routes any threshold
change (`increase` / `decrease`) through the promotion gate.

`recommendation="maintain"` emits a `maintain` decision without touching the
gate because no candidate exists. A threshold change without candidate evidence
fails closed as `reject`.

## Drift-triggered recalibration

`route_drift_recalibration()` accepts drift result records such as those emitted
by `src/omni_mercury_engine/ml/drift.py`. Recalibration is considered only after
`is_drift=true` with `severity` of `high` or `critical`.

Medium/low drift remains observable but does not route an autonomous
recalibration candidate. High/critical drift without candidate replay evidence
rejects. High/critical drift with evidence is evaluated by the Phase 2 gate.

## Dormant-module revival

`route_dormant_revival_candidate()` consumes verdicts from
`benchmarks/dormant_module_revival.py`. A dormant module that does not clear the
pre-registered standalone signal bar remains archived. A module that clears the
signal bar becomes promotion-gate candidate evidence; revival still requires
external-label replay lift, safety floors, capability evidence, and ablation
integrity.

The scheduled workflow `.github/workflows/phase3-governance.yml` runs the real
dormant-revival benchmark weekly and uploads the report. Pull requests run the
deterministic Phase 3 routing tests without relying on network data downloads.

## CLI

Composite Phase 3 reports can be evaluated with:

```bash
python research/governed_fusion/phase3_governance.py \
  --report artifacts/phase3_report.json \
  --check
```

`--check` exits non-zero on `reject` or `rollback`. Without `--out`, the module
writes append-only decision records under `artifacts/phase3_governance/`.
