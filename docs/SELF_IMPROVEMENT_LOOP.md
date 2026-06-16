# Governed Recursive Self-Improvement — Phases 1–2

> Status of this document: **Phase 2 in flight.** Phase 1 closed the
> measurement-provenance problem. Phase 2 adds the governed promotion gate
> that consumes that substrate, enforces safety/capability floors, and emits
> rollback-safe decision records. Phases 3–8 remain documented decisions, not
> unstated gaps.

## 0. What this loop is, in one paragraph

Recursive self-improvement is one loop: **measure → propose a change →
evaluate it safely → keep it only if it is measurably better → lock it so
it cannot regress.** Mercury contains the required control surfaces, but the
recursive proposal/execution arrows remain deliberately unwired until they can
operate behind measured gates. Phase 1 fixes the measurement substrate; Phase 2
wires the gated promotion path; Phase 3+ wire recursive proposals behind this
gate.

## 1. Phase 0 — verified ground truth (the map)

The Phase 1 audit verified the inventory the upstream prompt
asserted, against the live (post-#287) code. Where the prompt was right,
it's cited; where it was wrong, the reality is recorded here and the work
proceeds from reality.

### 1.1 The capabilities table, verified

| Capability | Existing seam (verified) | Notes |
|---|---|---|
| Measurement substrate | `research/governed_fusion/` (`suite.py`, `evaluate.py`, `metrics.py`, `measure_baseline.py`, `measure_conformal.py`, `measure_reliability_fusion.py`, `measure_survivability.py`, `measure_calibration*.py`, `measure_decorrelation.py`, `build_manifest.py`, `manifest.json`, `score_cache.py`) | Mature. The "fusion-marginal ablation" function the upstream prompt expected to find did **not** exist yet — Phase 1 adds it (`measure_marginal_ablation.py`). |
| Transparent-vs-leakage-flagged split | `README.md`, `docs/BENCHMARKS.md`, `src/omni_mercury_engine/datasets/label_provenance.py` (already gated by `tests/datasets/test_label_provenance_gate.py`). | Discipline existed only on the **`datasets/`** side. The **`loaders/`** side — which produces the governed-fusion live headline — had no provenance discipline at all. Phase 1 closes that gap. |
| Dead-loader tracking | `.github/workflows/dataset-reachability.yml`, v1.7.0 reachability harness. | Two-lane (offline / nightly) confirmed. |
| Fusion regression CI | `.github/workflows/fusion-regression.yml`, `benchmarks/fusion_regression_guard.py`. | Confirmed; runs on PR + weekly. |
| Dormancy / ablation (manual) | `benchmarks/dormant_module_revival.py`, `benchmarks/neurosymbolic_ablation.py`, `docs/DORMANCY_LEDGER.md`. | Confirmed manual passes. Phase 3 generalises to a recurring CI job — not in this PR. |
| Calibration / conformal | `src/omni_mercury_engine/core/calibration.py`, `core/score_calibration.py`, `core/conformal_prediction.py`, `research/governed_fusion/measure_conformal.py`. | Confirmed. Conformal coverage target 0.90 (measure_conformal.py L45). |
| Dual ethical gate (fail-closed) | `src/omni_mercury_engine/security/sigma_immutable_gate.py` (σ_Immutable, 256-D, 0.93/0.96), benevolence ≥ 0.99, `core/global_omni_scalar_network.py` (GOSNN), `benchmarks/run_ethics_audit.py`. | Confirmed fail-closed. Not exercised by Phase 1 (no promotion path enabled here). |
| Lyapunov pre-gate | `scripts/run_ablation.py`, `tools/lyapunov_validator.py`, `configs/ablation_3r_lyapunov.yaml`. | Confirmed. |
| Drift / online | `src/omni_mercury_engine/ml/drift.py`, `ml/concept_drift_evaluation.py`, `ml/online_learning.py`, `ml/ensemble_coordinator.py`. | Confirmed. |
| AutoML / search | `src/omni_mercury_engine/automl/optimizer.py`, `automl/search_space.py`, `automl/schedulers.py`, `ml/gwo_optimizer.py`, `detectors/advanced/gwo_ensemble.py`. | Confirmed. |
| Detector registry | `core/detector_registry.py`, `detectors/__init__.py`, `engine.py`. | **Prompt attribution corrected:** the enable/register path lives on `engine.py` (`OmniMercuryEngine.register_detector / enable_detector / available_detectors`), added in #287. `core/detector_registry.py` + `DETECTOR_MANIFEST` pre-existed. |
| Streaming | `infrastructure/streaming.py`, `streaming/streaming_detector.py` (aiokafka). | Confirmed. |
| Cognitive loop | `cognitive/reflexion.py`, `cognitive/cognitive_evolution_engine.py`, `cognitive/plasticity_engine.py`, `cognitive/neural_memory_layer.py`. | Confirmed present; **not wired** into measured fitness (Phase 3 work). |
| PQC / FIPS | AMA-Cryptography v3.2.0 hard-required-at-import, `.github/workflows/pqc-production-check.yml`, `integrations/mercury_amacrypto.py`. | Confirmed fail-closed; Phase 1 leaves it unchanged. |
| Hardware harness | `docs/HARDWARE_HARNESS.md`, `docs/drone/`, `docs/LIVE_DATA_VALIDATION.md`, `docs/DATASOURCES.md`. | Confirmed. Mercury today is **sensing-only**: no actuation path exists in code. The "human-in-the-loop for actuation" clause from the upstream prompt is therefore not a Phase 1 deliverable — it's a future Phase 8 contingency, not a gap. |

### 1.2 Transparent baseline metrics, locked

Reported as of this PR, computed from the committed
`research/governed_fusion/results/baseline_results.json`:

* **ADBench (47 datasets, externally labelled, comparable to published baselines):** Mean AUC **0.8180**, Mean F1 **0.5859**. *(`docs/BENCHMARKS.md`, `README.md` L182)*
* **Governed-fusion live suite (23 events, mixed regime — historical).** Macro-mean AUROC **0.8231**, F1 **0.2768**. This number is the headline before Phase 1's leakage-split: it blends 2 ground-truth-labelled events with 21 statistical-labelled ones.
* **Governed-fusion live suite, provenance split (Phase 1 result):**

| Bucket | n_events | mean AUROC | mean F1 | source |
|---|---:|---:|---:|---|
| `external_label` (FITNESS) | **2** | **0.7704** | **0.1863** | `network_security/{batadal, nsl_kdd}` |
| `self_label` (leakage-flagged) | 21 | 0.8282 | 0.2854 | every other live loader |
| `reconstructed` | 7 | n/a (reported separately by design) | | `tsunami/*, energy/*, pandemic/ebola_2014` |

The `external_label` row above is the transparent fitness signal Phase 2's
promotion gate will read from. It is **two events**, not 23 — and they're
both from the same domain (`network_security`). That is the real
rate-limiter on autonomous self-improvement today, and making it explicit
is Phase 1's deliverable.

> The numbers above are per-event macro means recomputed from the
> committed `baseline_results.json`. The external-label mean is **lower**
> than the mixed mean, which is at first counter-intuitive — the popular
> reading of "label leakage" assumes leaky labels inflate AUROC toward
> 1.0. Leaky labels can also produce *degenerate* splits (marine cells
> with 70% richness loss are sometimes harder, not easier, when the
> threshold is on a noisy ratio), so the headline can move either way.
> The right discipline is the same in both cases: don't optimise against
> labels that are a function of the scored signal.

### 1.3 The 10 unreachable datasets — reconciled, not "restored"

The upstream prompt asked to "restore the 10 non-loading datasets so the
eval is not blind". That oversells the recovery — several are
licence-gated or require registration, and redistributing fixtures has
exposure. The professional framing is **reconcile**: mirror what is
licence-compatible to mirror, mark the rest `cannot_score`. Phase 1 does
the audit and locks the policy; the actual mirror commits (where
appropriate) follow in a Phase-1.1 commit on the same branch or a
separate PR if licence review is needed.

| Dataset | Loader | Status | Policy |
|---|---|---|---|
| SMAP | `timeseries.SMAPMSLLoader` | OmniAnomaly mirror gone | `cannot_score` (archive-dependent). |
| MSL | `timeseries.SMAPMSLLoader` | same | `cannot_score`. |
| CICIDS-2017 | `security.CICIDSLoader` | All download sources failed | `cannot_score` (sources defunct). |
| MIT-BIH | `mitbih.MITBIHLoader` | Requires `wfdb` + PhysioBank | `cannot_score` (registration). |
| UCR | `ucr_archive.UCRLoader` | Archive access control | `cannot_score` (registration). |
| SWaT | `industrial.SWaTLoader` | Registration + ethics approval | `cannot_score` (registration). |
| WADI | `industrial.WADILoader` | Registration | `cannot_score` (registration). |
| USGS Geochemistry | `environmental.USGSGeochemistryLoader` | Public USGS mirror | Reachable in principle; v1.7.0 added the real download path. Stays on the nightly reachability watch-list. |
| NOAA StormEvents | `noaa_storm.NOAAStormEventsLoader` | Public, rate-limited | Watch-list. |
| NOAA ERDDAP | `noaa_erddap.NOAAERDDAPLoader` | Public, rate-limited | Watch-list. |
| FEMA HazardMitigation | `disaster.FEMAHazardMitigationLoader` | Public OpenFEMA | Watch-list. |

A `cannot_score` dataset is **never** given a fabricated label to fill the
gap — that would re-introduce the exact problem Phase 1 is solving. Every
`cannot_score` dataset is already registered in
`src/omni_mercury_engine/datasets/label_provenance.py` with its audited
justification.

## 2. Phase 1 — what this PR ships

### 2.1 Loader-side label-provenance discipline (mirror of `datasets/`)

* `src/omni_mercury_engine/loaders/base.py` — adds
  `LABEL_SOURCE: str = "ground_truth"` to `BaseDomainLoader` (a
  provenance-declaration default; loaders must override to declare
  their actual source).
* All 15 concrete loaders updated:
  * `ground_truth` (2): `network_security_loader.NetworkSecurityLoader`,
    `sepsis_loader.SepsisLoader`.
  * `statistical` (13): every other live-API loader. The justification on
    each `LABEL_SOURCE` block names the exact circular pattern (e.g.
    earthquake labels `magnitude >= mainshock_mag - 1.0` against
    feature[0] = `magnitude`).
* `src/omni_mercury_engine/loaders/label_provenance.py` — the canonical
  per-loader audit (`LABEL_PROVENANCE_REGISTRY`, `audit_label_provenance`,
  `scan_circular_label_construction`, `discover_loaders`,
  `ground_truth_loader_keys`). Mirrors the proven `datasets/` pattern.
  CLI: `python -m omni_mercury_engine.loaders.label_provenance --check`.
* `tests/loaders/test_label_provenance_gate.py` — 12 assertions: every
  concrete loader registered; declared `LABEL_SOURCE` matches audited
  value; AST circularity scanner catches `labels = (df[col] > c)` patterns;
  no false positives on the genuine loaders; the ground-truth set is
  exactly the Phase 1 finding.

### 2.2 Governed-fusion suite, wired to the audit

* `research/governed_fusion/label_provenance.py` — pivots from
  per-loader `LABEL_SOURCE` to per-event `(label_provenance,
  series_provenance, external_label)`. Single source of truth for the
  manifest builder, the aggregator, and the ablation ledger.
* `research/governed_fusion/manifest.json` — every entry now carries
  `label_provenance`, `series_provenance`, `external_label`; the
  top-level `provenance_summary` enumerates bucket counts. The committed
  manifest reflects the Phase 1 audit: 2 external-label, 21 self-label,
  7 reconstructed.
* `research/governed_fusion/build_manifest.py` — emits the new fields
  and the summary block.
* `research/governed_fusion/evaluate.py::aggregate` — adds a
  `per_provenance` breakdown alongside `per_event` / `per_domain` /
  `overall`. Phase 2's promotion gate reads
  `per_provenance["external_label"]` only.
* `research/governed_fusion/measure_baseline.py` — prints the
  per-provenance breakdown alongside the per-domain and overall blocks,
  with `external_label` tagged `(FITNESS)`.
* `tests/research/test_governed_fusion_label_provenance.py` — 7
  assertions: every manifest entry has the provenance fields; manifest
  matches loader registry (no drift); the `external_label` bucket is
  exactly `{network_security/batadal, network_security/nsl_kdd}`;
  reconstructed events are never `external_label`; summary counts match
  per-entry counts.

### 2.3 Standing fusion-marginal ablation ledger

* `research/governed_fusion/measure_marginal_ablation.py` — leave-one-
  component-out lift (`resonance`, `kinematic`, `info_geo`) on the
  external-label live subset only. Macro-mean ΔAUROC / ΔAUPRC / ΔF1 per
  component. Records timestamp, git SHA, components, the full vs
  ablated metrics, and the event keys actually scored. When the selected
  score cache (`$GF_CACHE_DIR` or `--cache-dir`) is absent on the runner,
  writes a `needs_cache` record (exit 0) so the ledger keeps a
  chronological reachability account without blocking the gate; `--check`
  flips that to exit 1 for the nightly job once Phase 2 wires a cache build.
* `research/governed_fusion/ablation_ledger.json` — seed ledger;
  `runs[]` grows append-only across CI runs.
* `tests/research/test_marginal_ablation_ledger.py` — 6 assertions:
  schema integrity of the committed ledger; informative components show
  positive lift; a noise component ranks below the informative ones;
  missing-cache path produces the `needs_cache` informational record
  without crashing; the output schema is locked.

### 2.4 CI gate

* `.github/workflows/ablation-ledger.yml` — runs on PR to `main` /
  `develop`, on dispatch, and nightly at 05:17 UTC:
  1. `python -m omni_mercury_engine.loaders.label_provenance --check`
  2. `pytest tests/research/test_governed_fusion_label_provenance.py -q`
  3. `pytest tests/research/test_marginal_ablation_ledger.py -q`
  4. `python research/governed_fusion/measure_marginal_ablation.py` (writes a record)
  5. Uploads `ablation_ledger.json` as a CI artifact for review.

### 2.5 Docs updated atomically

* `README.md` — adds the live-API loader provenance summary and points
  at this document.
* `docs/BENCHMARKS.md` — adds a Phase 1 provenance-split section (see below).
* `docs/DORMANCY_LEDGER.md` — flags the ablation ledger as the
  standing measurement substrate the future generalised dormant-revival
  job will write through.
* `CHANGELOG.md` — Phase 1 entry.

## 3. Self-validation (anti-doctoring)

Phase 1's CI-locked outputs match the numbers reported here. No gate,
threshold, ethics floor, or assertion was loosened to make CI pass:
* The σ_Immutable threshold (0.93/0.96), benevolence floor (0.99),
  Lyapunov certificate, conformal coverage target (0.90), and the
  ablation-floor gate are **unchanged**.
* The pre-existing `tests/datasets/test_label_provenance_gate.py` is
  **unchanged**; the loader-side mirror is additive.
* The pre-existing `fusion-regression.yml` floors are **unchanged**;
  the new `ablation-ledger.yml` is additive.

The two findings that *moved* and must be surfaced transparently:

1. The governed-fusion live headline's external-label subset is **2 events from
   one domain**, not 23 across seven. Every public Mercury claim of "live
   AUROC ~0.82 on real-world events" has been silently averaging two
   externally-labelled events with 21 leakage-contaminated ones. Phase 1
   stops doing that.
2. The two ground-truth events' external-label mean (AUROC 0.7704, F1 0.1863)
   is **below** the previously-reported mixed mean. Leakage does not
   only inflate; it can also degrade in either direction. The discipline
   is the same: don't grade self-improvement against it.

## 4. Phase 2 — governed promotion gate

Phase 2 ships `research/governed_fusion/promotion_gate.py` and
`tests/research/test_governed_promotion_gate.py`.

The gate:

* Reads only the `external_label` fitness bucket from the Phase 1 manifest.
  It rejects any candidate that attempts to optimise against `self_label`,
  `reconstructed`, or `overall` metrics.
* Uses `held_out_replay` as the CI evaluation surface. This is intentionally
  not called live shadow traffic because no production traffic exists in CI.
* Requires measurable AUROC or F1 lift with no AUROC/AUPRC/F1 regression on
  the external-label bucket.
* Enforces σ_Immutable, benevolence, conformal coverage, and Lyapunov floors
  without loosening any existing threshold.
* Requires the capability-regression suite to pass.
* Checks the latest `status="ok"` marginal-ablation ledger baseline when one
  exists.
* Writes append-only experiment records and emits `rollback` decisions for
  failed canaries. A `promote` result remains human-review-gated; the module
  does not perform unattended deployment.

See `docs/GOVERNED_PROMOTION_GATE.md` for the operator contract and CLI.

## 5. Explicit out-of-scope decisions (deferred, not omitted)

The upstream prompt asked for one PR covering Phases 1–8. The authorising
user capped this at **two PRs** and ordered Phase 1 first to prove the
fitness signal is provenance-safe before any automation rides on it. Phase
2 completes that approved two-PR path by adding the gate. Everything below
is a *decision* to defer, not a gap.

* **Phase 3 — Reflexion executor wiring, drift-triggered auto-
  calibration, recurring dormant-revival CI job.** Routed through the
  Phase 2 gate. Not in scope here.
* **Phase 4 — streaming + active-learning queue + versioned data.** Out
  of scope here.
* **Phase 5 — search mechanism (GWO/Optuna producing gated proposals).**
  NAS is explicitly excluded across both PRs: it is negative-ROI until
  the fitness signal is solid, and the fitness signal only became solid
  with Phase 1. That is a stated decision, not a gap.
* **Phase 6 — continual-learning memory + neuro-symbolic evolution.**
  Both must be either fully gated-and-real or fail-closed-and-explicitly-
  labelled; neither qualifies today. Stays disabled behind its existing
  flags with status explicitly documented.
* **Phase 7 — UBI 9/10 base + FIPS/PQC alignment.** The current main
  base is `python:3.13-slim-trixie` (post-#296). The UBI migration is a
  separate, contained PR — it does not belong on the fitness-substrate
  branch.
* **Phase 8 — hardware / signal-mesh adapters, GPU-on-drift orchestration,
  BOM.** Out of scope here. Mercury today is sensing-only (no actuation
  path in code), so the human-in-the-loop-for-actuation clause is a
  contingency for Phase 8, not a Phase 1 deliverable.

A documented decision is not a gap; an undocumented omission is.
