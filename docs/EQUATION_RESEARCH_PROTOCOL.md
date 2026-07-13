# Equation Research Protocol

Applies to Mercury Agent **v2.1.x**. Last updated: 2026-07-11.

This protocol implements a hard-governed path for strengthening Mercury's in-house equations without replacing them.

## What it enforces

1. **Control-model freeze**: OAE, Lyapunov, Banach, conformal, and σ_Immutable are treated as immutable reference surfaces.
2. **Three outcome tracks**: unknown discovery, humanitarian impact, and security hardening.
3. **Formal hypothesis matrix**: each component must clear explicit acceptance checks and fail conditions.
4. **Ablation/sweep discipline**: protocol outputs include command plans and compatibility with existing benchmark harnesses.
5. **First-class gates**: uncertainty, calibration, and boundary-stress checks are mandatory.
6. **Dual-lane consistency**: humanitarian-first and security-first lanes must remain consistent within configured tolerances.
7. **Promotion hard criteria**: empirical gain + ethical preservation + stability preservation + originality preservation.
8. **Traceable theorem artifacts**: promoted advances must include claim, assumptions, proof sketch, empirical evidence, risk limits, and deployment constraints.
9. **AI equation library**: known reference equations and Mercury in-house equations are separated so external math grounds rigor without replacing original work.

## Files

- Protocol config: `configs/equation_research_protocol.yaml`
- Runner: `scripts/run_equation_research_protocol.py`
- Runtime comparison: `scripts/compare_runtime_equation_profiles.py`
- Report output (default): `artifacts/equation_research_protocol_result.json`

## Run

```bash
python scripts/run_equation_research_protocol.py \
  --config configs/equation_research_protocol.yaml \
  --benchmark benchmarks/mercury_benchmark_results.json \
  --ablation artifacts/equation_protocol_ablation.json \
  --gates artifacts/equation_protocol_gates.json \
  --out artifacts/equation_research_protocol_result.json
```

To emit theorem artifacts for promoted advances:

```bash
python scripts/run_equation_research_protocol.py \
  --theorem-out artifacts/theorem_artifacts
```

To compare an opt-in runtime equation profile against the frozen original baseline:

```bash
python scripts/compare_runtime_equation_profiles.py \
  --baseline baseline_original_v1 \
  --candidate quiet_horizon_v1 \
  --out artifacts/runtime_equation_profile_comparison.json
```

## Expected ablation input schema

`artifacts/equation_protocol_ablation.json` is expected to contain:

- `component_deltas.<component>.delta_auc` / `delta_oracle_f1`
- `component_deltas.<component>.p_value`
- `hard_gates.benevolence_violations`
- `hard_gates.ethical_regressions`
- `hard_gates.sigma_immutable_violations`
- `hard_gates.gosnn_unavailable_violations`
- `advances[]` candidate records with:
  - `id`
  - `empirical_gain` (bool)
  - `ethical_gates_preserved` (bool)
  - `stability_preserved` (bool)
  - `originality_preserved` (bool)
  - theorem fields required by config

## Expected gate input schema

`artifacts/equation_protocol_gates.json` is expected to contain:

- `gates.uncertainty.coverage_ok` (bool)
- `gates.calibration.ece` (float)
- `gates.boundary_stress.pass` (bool)
- `lanes.humanitarian_first.mean_auc` / `mean_recall`
- `lanes.security_first.mean_auc` / `mean_recall`

## AI equation library

The protocol carries a dedicated AI equation section in
`configs/equation_research_protocol.yaml`:

- **Known reference equations** ground validation method: MAUT additive utility,
  split conformal quantiles, expected calibration error, CUSUM drift detection,
  and LTN fuzzy rule satisfaction.
- **In-house equations** remain the protected Mercury surfaces: OAE, universal
  generalized composite candidates, quiet-horizon runtime candidates,
  benevolence φ-index, and GOSNN hierarchical geometric scoring.

This separation is deliberate. External equations justify measurement,
calibration, stability, and neuro-symbolic evaluation discipline; they do not
replace Mercury's original equations. Promotion requires empirical lift,
originality preservation, hard ethical gates, σ_Immutable integrity, stability,
and rollback metadata.

## Decision justification

- Preserve original equations as immutable baselines and optimize only around
  explicit candidate profiles.
- Treat ethical, σ_Immutable, stability, and output-range checks as hard
  feasibility gates, not score components that can be averaged away.
- Evaluate anomaly detection across unknown-discovery, humanitarian, and
  security lanes, with calibration and uncertainty promoted to first-class
  criteria.
- Require theorem-to-test artifacts so a candidate is tied to a falsifiable
  claim, assumptions, proof sketch, empirical evidence, risk limits, and
  deployment constraints.

## Exit codes

- `0`: protocol passed and at least one promoted advance has complete theorem fields.
- `1`: protocol evaluation completed but one or more hard criteria failed.
- `2`: invalid input paths.
