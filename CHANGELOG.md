# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Reproducibility note (applies to all 1.x release entries below):**
> Headline benchmark numbers vary by release cut; the current committed
> run is **65 reproducible datasets** (of 75 attempted), surfaced in the
> README "Latest Benchmark Results" block. 10 datasets currently
> fail to load due to unavailable external sources (SMAP, MSL,
> CICIDS-2017, MIT-BIH, UCR, SWaT, WADI, USGS Geochemistry,
> NOAA ERDDAP, FEMA HazardMitigation). As of v1.7.0
> the previously-flagged "FEMA Disaster — inverted scores" loader
> is no longer in the broken set; the label-polarity correction is
> documented under `[1.7.0]` below and locked by
> `tests/datasets/test_disaster.py::TestFEMAInvertedScoresCorrection`.
> The 11 watch-listed loaders now have a two-lane reachability harness
> (`tests/datasets/test_unreachable_loaders_{offline,network}.py`,
> plus the nightly `.github/workflows/dataset-reachability.yml`
> workflow) so an upstream provider outage surfaces as a failed
> nightly run rather than as a benchmark silently dropping a
> dataset.  See the README "Empirical Benchmark Results" section for
> the full reproducibility footnote and `docs/ROADMAP.md` for
> tracked fixes.

## [Unreleased]

### Statistical ensemble: fusion-margin sharpening, data-type gate correction, temporal robustness

A measurement-driven pass on `MercuryAnomalyDetector` (`detectors/statistical.py`)
that sharpens the unsupervised fusion weights, corrects a data-type
misclassification that mistreated tabular data as temporal, and adds opt-in
temporal robustness. All changes are label-free and validated on a committed,
re-runnable 18-set real-ADBench transductive harness
(`research/omni_equation/harness_adbench.py`; AUROC via Mann-Whitney). Net
transductive Mean AUROC **0.7397 → 0.7634 (+2.37 pts, 14 W / 2 tie / 2 L)** —
base `e118e1f` vs the hardened detector, full per-set results in
`research/omni_equation/adbench_results.json`; the two losses are within noise
(≤0.0003 AUROC). This supersedes an earlier ad-hoc `0.7804 → 0.8142` figure that
is **not reproducible**: its harness resolved several sets through the now-broken
ADRepository mirror, which silently substitutes synthetic data. Re-run:
`python research/omni_equation/harness_adbench.py`.

* **Sharpen the unsupervised fusion-weight margins (`_WEIGHT_MARGIN_POWER = 4.0`,
  new module constant).** The self-supervised separation AUCs that drive the
  unsupervised weighter saturate into a narrow 0.72-1.0 band on easy 3σ
  Gaussian-blob synthetic anomalies, so a *linear* AUC-separation margin leaves
  the consistently weakest component — kinematic, last on 18/18 ADBench sets —
  with 30-42% of the ensemble weight. Each surviving margin `(auc - 0.5)` is now
  raised to a power `P` before normalisation, in both
  `_compute_unsupervised_adaptive_weights()` and the supervised
  `_compute_adaptive_weights()`. `P=1` reproduces the prior behaviour exactly
  (regression-tested). On the committed harness the biggest wins land where
  kinematic was most over-weighted (wine, glass, PageBlocks, cardio — all wins
  vs base).

* **Fix the data-type gate so tabular data is not mislabeled temporal
  (`_detect_data_characteristics`).** The previous single-signal gate
  (autocorrelation **or** adjacent-row correlation > 0.3) classified 13/18
  shuffled-tabular ADBench sets as `TEMPORAL` on spurious lag-1 autocorrelation
  or class-ordering adjacency, handing KinematicScore the largest compatibility
  weight. The gate now requires *strong* evidence on either of two complementary
  signals — median lag-1 autocorrelation > `_TEMPORAL_AUTOCORR_THRESHOLD` (0.55)
  **or** median adjacent full-row coherence > `_TEMPORAL_ADJ_ROW_THRESHOLD`
  (0.75) — thresholds that sit in the empirical gap between the tabular cluster
  (autocorr ≤0.50, adjacency ≤0.67) and the temporal cluster (autocorr ≥0.60,
  adjacency ≥0.82, including spike-contaminated series whose per-feature
  autocorr is masked but whose adjacency stays ~0.99). Genuine temporal data
  (sine / AR(1) / random-walk fixtures, the existing
  `test_temporal_data_detected` case, and real SMD telemetry — autocorr 0.91)
  stays `TEMPORAL`; the existing `TABULAR → kinematic=0` hard-zero is untouched.

* **Restrict the temporal residual-frequency filter to genuinely temporal data
  (`detect()`).** The Phase-7 `_residual_frequency_filter` treats the score
  vector as an ordered time series (moving-average baseline + rFFT band-pass), so
  it is only meaningful when row order is meaningful. It was gated on a shape
  proxy (`n ≥ max(50, 10·d)`) that also fires on wide/long *tabular* sets, where
  it is rank-altering — measured to drag optdigits from 0.52 to **0.39 (below
  random)**. It is now gated on the fitted `_data_type == TEMPORAL`. This both
  removes that harm and makes detection **robust to test-row ordering**: on the
  inductive benchmark protocol the prior code scored 0.8828 on a class-grouped
  test set but only 0.8722 once the test rows are shuffled (it was exploiting the
  label ordering), whereas the corrected detector scores 0.8792 / 0.8800 — flat.

* **Opt-in inference-time multi-scale "time-dilation" TTA (DEFAULT-OFF).**
  New `multiscale_tta` / `multiscale_tta_pool` config flags. When enabled and the
  fitted data is `TEMPORAL`, `detect()` scores the series at
  `_MULTISCALE_TTA_SCALES = (0.8, 0.9, 1.0, 1.1, 1.25)` time-dilations
  (`np.interp` resample), maps each per-sample score vector back to the original
  length, and pools across scales (`mean`, the false-alarm-safe default, or
  `max`). DEFAULT-OFF is byte-identical to the prior output (Invariant I2). Pure
  inference, no fit change, K× scoring compute. Validated on real SMD
  machine-1-1 collective anomalies: mean-pool **+5.1 pts** clean (0.565 → 0.616)
  and **+5.8 pts** under a 0.85× rate change; on controlled collective anomalies
  max-pool wins 8/8 (+0.66 pts).

* **Sampling-jitter robustness characterization
  (`tests/detectors/test_sampling_jitter_robustness.py`, new).** Clock-skew /
  rate-drift on a temporal feed degrades AUROC −5 pts (eps 0.10) to −16 pts (eps
  0.20) — a documented limitation, not an invariance claim. The new test asserts
  a *tolerance band* (bounded AUROC loss, score-ranking correlation above a
  floor) so robustness regressions fail loudly, and pins the multi-scale TTA
  invariants (default-off byte-identity, temporal-only gating, valid pooled
  scores, collective-anomaly improvement).

### Data loaders: close the ADRepository silent-synthetic fallback, repoint the moved mirror

Removes the last silent synthetic-data fallback on the dataset layer — the exact
foot-gun that contaminated the superseded `0.7804 → 0.8142` headline. The
`ADRepositoryLoader` previously called `_create_synthetic_fallback()` directly in
three `except`/fallback sites with **no policy check**, so a moved mirror or an
unreachable host returned fabricated data that downstream code could not
distinguish from real signal. Every other dataset loader already routed synthetic
generation through `check_synthetic_allowed()`; ADRepository was the sole holdout
(confirmed by a full per-loader audit).

* **Fail loud by default (root-cause fix).** `_create_synthetic_fallback()` is now
  the single synthetic chokepoint and self-gates on `MERCURY_ALLOW_SYNTHETIC` via
  `check_synthetic_allowed()`. With the policy off (the default) it raises
  `DataSourceUnavailableError` with context; with `MERCURY_ALLOW_SYNTHETIC=1` it
  honours the opt-in and marks the result `is_real_data=False` / `synthetic=True`.
  No production path can fabricate data without the explicit deployment opt-in.
* **Repoint the moved mirror.** The ADRepository corpus moved
  `GuansongPang → mala-lab`; the old `github.com/.../raw/main/` form 301-redirects
  to `raw.githubusercontent.com`, which the SSRF-safe HTTP client refuses to
  follow — the failure that triggered the silent fallback. The loader now pins
  directly to the `raw.githubusercontent.com` mala-lab path (already on the
  trusted-domain allowlist, no redirect) and uses the real DevNet filenames
  (`annthyroid_21feat_normalised.csv`, `creditcardfraud_normalised.tar.xz`, …),
  with a single canonical source per dataset so a name always resolves to the
  same composition. `.tar.xz` archives are now supported alongside `.csv`/`.npz`,
  with path-traversal (tar-slip / zip-slip) guards on both extraction paths.
* **Harden the archive-extraction helpers (review follow-ups).** `_safe_extract_tar`
  now feature-detects `tarfile.data_filter` and falls back to an explicit member
  guard on the 3.11.0–3.11.3 patch releases that lack the `filter=` kwarg (where
  it would otherwise raise `TypeError` straight into the synthetic path);
  `_safe_extract_zip`'s docstring no longer implies the stdlib sanitises member
  paths (this helper *is* the sanitiser); and both archive paths clear any stale
  reuse directory before extracting, so `rglob('*.csv')` cannot pick up a leftover
  from a previously-extracted archive with a different layout. Covered by a new
  `TestArchiveExtractionSafety` (tar/zip-slip refusal, symlink refusal, the
  filter-absent fallback, stale-dir clearing).
* **Remove the dead ODDS path.** The ADRepository loader's legacy ODDS (Outlier
  Detection DataSets, Stony Brook) `.mat` codepath was removed — the host is
  unreachable (503) and no registered dataset name ever routed through it. Because
  nothing fetches from it, `odds.cs.stonybrook.edu` is deliberately **not** added
  to `TrustedEndpoints.TRUSTED_DOMAINS`, keeping the trusted-egress surface
  minimal; time-series sets without a dedicated loader fail loud naming their real
  upstream instead of routing to synthetic.
* **Tests.** `tests/datasets/test_adrepository.py` gains a `TestSyntheticPolicyGate`
  class (chokepoint raises when policy off; production load fails loud on an
  unreachable source; opt-in synthetic is labelled; time-series sets with no
  mirror fail loud) and a network-lane `TestADRepositoryRealMirror` that loads
  real thyroid (CSV) and backdoor (`.tar.xz`) from the repointed mirror with
  synthetic denied (real-or-raise). The transductive ADBench headline is
  unaffected — that harness uses the already-clean `ADBenchLoader`.

### Full inductive suite refreshed — real-data base-vs-current (75 attempted / 66 scored)

Re-ran the committed `benchmarks/mercury_benchmark.py` (inductive
normal-train/test-on-rest protocol) on the full 75-dataset registry, **real data
only** (`MERCURY_ALLOW_SYNTHETIC=0`) — base `e118e1f` vs the hardened current
tree. 9 datasets fail loud (unreachable upstream: SMAP, MSL, CICIDS-2017,
MIT-BIH, USGS_Geochemistry, FEMA_HazardMitigation, UCR, SWaT, WADI) and are
excluded, never synthetic-substituted; the ledger is identical in both runs. On
the deterministic ADBench subset the hardening is **0.8180 → 0.8251 (+0.0071,
18 W / 5 tie / 24 L)**; across all real datasets excluding the ADRepository
artifact, **+0.0029**. The naive all-sets number is a slight −0.0016 *because the
base commit's unfixed ADRepository loader silently returns synthetic data and the
detector scores a meaningless AUROC 1.0000 on it* — a live capture of the exact
foot-gun the loader fix closes (the hardened tree scores real thyroid 0.7086
there). The inductive deltas run smaller than the transductive +0.0237 because
the inductive protocol's class-grouped test ordering let the old temporal filter
exploit a row-order artifact the hardened detector correctly forgoes; both
protocols are net-positive once decontaminated. Full analysis + reproduction in
`docs/BENCHMARKS.md`; refreshed results in `benchmarks/mercury_benchmark_results.json`.

**De-leak follow-up (`632d3e5`).** The class-grouped test ordering is now closed
at the harness level — `X_test`/`y_test` are shuffled (fixed seed) after the cap,
so evaluation order carries no label information and no detector (base, current,
or future) can exploit it. The distortion is **material and bidirectional**: on
the 8-dataset deterministic regression guard (current detector, full
`MAX_SAMPLES`, grouped vs de-leaked) the **ensemble** AUC moved `pendigits`
0.761→0.874 (+0.113, grouping *deflated* it), `Pima` 0.741→0.705 (−0.036,
grouping *inflated* it), `glass` 0.745→0.768; guard mean 0.891→0.901. The
per-component **kinematic** AUC moves more still (max |Δ| **0.41** — `wine`
0.18→0.60; resonance / info-geometry are exactly order-invariant).
`benchmarks/anomaly_regression_baseline.json` is re-pinned on the de-leaked eval;
the full inductive base-vs-current table in `docs/BENCHMARKS.md` is superseded and
must be regenerated on the de-leaked harness.

### CI: gate the engine import (and full PQC contract) at the AMA build seam

The `build-ama-cryptography` composite action (used by ~21 engine-importing jobs
across 11 workflows) built AMA and verified only `dilithium_available`, then
assumed "AMA built" implied "`omni_mercury_engine` imports". Two gaps closed:

* **Full PQC contract.** The engine's import-time gate
  (`_pqc_gate._PQC_HARD_REQUIRED`) requires ML-DSA-65 **+** Kyber-1024 **+**
  SPHINCS+, but the action checked only ML-DSA — a Kyber/SPHINCS-incomplete build
  passed it and surfaced later as an opaque collection error. The backend check
  now asserts all three.
* **Engine-import gate.** A new step imports `omni_mercury_engine` immediately
  after the build, so an engine-side import regression or an AMA/engine version
  skew fails loudly at the seam rather than 20+ minutes deep in a test lane.
  Self-gating via `find_spec` keeps the action reusable for pure-AMA jobs. No new
  CI jobs, no DAG/latency change.

## [2.0.0] - 2026-06-17

### Docs↔code reconciliation + release-engineering pass (2026-06-17)

The pass that cut 2.0.0: published figures trued to the CI-gated source of truth,
plus the version and deployment-secret wiring that entries below already assumed.

* **Single source of truth for the distribution version
  (`omni_mercury_engine/_version.py`, new).** Resolves the version once from
  installed package metadata (`importlib.metadata`) with a source-tree fallback;
  `__version__`, `API_VERSION`, the `mercury --version` CLI, the OTLP
  `service.version` resource/exports, and the `crypto` / `models.sota` package
  versions all derive from it. Closes a live drift where `engine.py` reported
  `1.7.0` from stale installed metadata while the hard-coded literals said
  `2.0.0`; every surface now agrees and cannot desync on a release bump.
* **Helm chart now provisions every runtime secret the app reads.** The chart
  rendered only `JWT_SECRET_KEY`; `API_KEY_HASH_SALT` (required in production by
  `api/auth.py`) and `MERCURY_CACHE_SECRET` (the cache entry-signing key) were
  documented and read by the app but never placed in the Secret. Added both to
  `config.secrets` + `templates/secret.yaml` (auto-injected via the deployments'
  existing `envFrom: secretRef`); `JWT_SECRET_KEY` keeps its legacy `JWT_SECRET`
  values-key fallback. Verified with `helm lint` / `helm template`;
  `docs/DEPLOYMENT.md` install/upgrade commands corrected to the real
  `helm/mercury-agent/values.yaml` path and `config.secrets.*` keys.
* **Prometheus `/metrics` served at the conventional root path.** The exposition
  handler (`api/health.py::health_metrics`) is mounted at `/metrics` on the app
  — the target scraped by `monitoring/prometheus/prometheus.yml`, the k8s
  `prometheus.io/path` annotation, and the chart — instead of 404. Locked by
  `tests/test_api.py::TestAPI::test_metrics_endpoint`.
* **Documentation trued to CI-gated reality.** Engine/loader/test counts, dates,
  benchmark prose, the adaptive-ensemble detector docstring, and the OpenAPI
  version examples aligned to proven values; version-pinned tests assert against
  the resolved constants instead of frozen literals.
* **Integration test lane built out — `tests/integration/` now carries real
  cross-subsystem signal.** The CI `integration-tests` job built the full
  `.[api,dev]` stack and the native AMA backend only to run zero tests (the
  directory had never existed) and was timing out before reaching the no-op.
  Added end-to-end suites that exercise genuine component boundaries: the
  detection pipeline through the `OmniMercuryEngine` facade and its detector
  suite; the REST API request→engine→response path asserting detection
  *correctness* (not just shape) plus boundary validation; external-source
  ingestion (USGS loader → feature matrix → detector) with the network mocked;
  and post-quantum signing of a real detection artifact through Mercury's
  `MLDSAProvider` / `KyberProvider` on the live ML-DSA-65 / ML-KEM-1024 backend.
  No `skipif` escape hatch — a missing native backend fails loudly, matching the
  repo's PQC-gate convention.
* **CI `integration-tests` lane given the pip cache it was missing.** It was the
  single heaviest-install lane (`.[api,dev]` + API server stack) with no
  `actions/cache`, so it rebuilt the whole torch/dev tree from scratch every run
  and tipped past its 30-minute budget mid-setup. Added the cache (shared
  `-pip-` restore-key warms a cold first run from sibling lanes), bringing the
  lane in line with `core-tests` / `type-checking`. Root-cause fix, not a
  timeout bump.

### Phase 3: Governed self-improvement seam — live reflexion + drift arrows routed through the gate (2026-06-16)

Wires the live self-improvement arrows through a fail-closed governance seam at
the exact point a change would take effect, so neither the reflexion critic nor
the online-learning drift trigger can mutate Mercury's live behaviour
autonomously. (Supersedes the earlier routing-only Phase 3 surface, which
governed a copy of the decision while the live mutation still executed
ungoverned.)

* `src/omni_mercury_engine/governance/self_improvement.py` (new) — the
  engine-owned seam: `ThresholdGovernance` / `RecalibrationGovernance`
  protocols, the `ProposedThresholdChange` / `ProposedRecalibration` /
  `GovernanceReview` records, and two built-in policies —
  `FailClosedSelfImprovementGovernance` (default; withholds every autonomous
  change) and `MeasurementGovernance` (explicit, named opt-in for held-out
  measurement harnesses).
* `agentic/orchestration.py` — `MultiAgentOrchestrator.reflect()` now routes an
  actionable threshold recommendation through `threshold_governance` instead of
  applying it; `ReflectionRecord` gains the governance disposition. Default
  fail-closed; the live operating point moves only on an authorised review.
* `ml/online_learning.py` — `OnlineLearningPipeline` routes its
  high/critical-drift and performance-degradation retrain triggers through an
  optional `recalibration_governance` policy before `model.fit()`. Standalone it
  remains an autonomous online learner; in the governed composition it is
  fail-closed. Explicit `force_retrain()` is never gated.
* `engine.py` — `enable_multi_agent_orchestration(threshold_governance=…)`
  installs the policy on the production orchestrator.
* `research/governed_fusion/phase3_governance_adapters.py` (new) — the
  gate-backed policies (`PromotionGateThresholdGovernance`,
  `PromotionGateRecalibrationGovernance`) implement the engine seam by routing a
  proposal through the Phase 2 promotion gate (research → engine dependency). A
  gate `promote` is queued for human approval, never auto-applied.
* `research/governed_fusion/phase3_governance.py` — adds
  `dormant_revival_report_to_section` and a `--dormant-revival` CLI path so the
  recurring workflow routes every measured verdict through the gate instead of
  only uploading it (closes the measure → route loop). `maintain` remains a
  no-op; missing candidate evidence fails closed; `promote` remains human-review
  gated.
* `Phase3DecisionStore` — append-only JSON records plus `index.jsonl` for
  Reflexion, drift, and dormant-revival routing outcomes.
* `tests/research/test_phase3_governance.py` — routing logic, composite reports,
  append-only records, CLI `--check`, and the dormant report → routing loop.
* `tests/research/test_phase3_live_wiring.py` (new) — end-to-end proof against
  the real gate: the autonomous threshold move and the autonomous drift retrain
  are both observed to be withheld by default, applied only under an explicit
  measurement stance, and queued (never auto-applied) when the gate-backed
  policy clears them.
* `.github/workflows/phase3-governance.yml` — pull requests run deterministic
  Phase 3 routing tests; scheduled/manual runs execute the real dormant-module
  revival benchmark, route every verdict through the gate, and upload both the
  measurement and the routing decisions.
* `docs/PHASE3_GOVERNANCE.md`, `docs/SELF_IMPROVEMENT_LOOP.md`,
  `ARCHITECTURE.md`, `SECURITY.md`, and `README.md` — document the governed
  seam, the live wiring, and the fail-closed posture; Phases 4–8 remain
  explicitly deferred.

### Promotion-gate ingestion hardening — close fail-open paths on the decision boundary (2026-06-16)

Hardens the Phase 2 gate (`research/governed_fusion/promotion_gate.py`) and the
Phase 3 router CLI against malformed and non-finite evidence. Each fix closes a
fail-**open** path on the security-critical decision boundary; all are locked by
new tests in `tests/research/test_governed_promotion_gate.py`.

* `_as_float` now rejects `bool` (a Python `int` subclass) and non-finite
  values. Previously a JSON `true`/`false` in a safety field read as `1.0`/`0.0`
  and cleared a floor (`{"benevolence": true}` → promote), and an `Inf` metric
  satisfied the improvement delta while a `NaN` never compared below a regression
  floor.
* `_as_int` rejects `bool` and accepts an integral float (`2.0` → `2`) so a
  legitimate candidate whose metric code emits `2.0` is no longer rejected by a
  crash.
* A non-finite **ledger baseline** (e.g. a degenerate single-class measurement
  that produced `NaN`) is now a fail-closed reject reason, not a silently-skipped
  ablation check.
* Both CLI `main()` entry points convert validation errors into an auditable
  REJECT decision record plus a non-zero exit, instead of a bare traceback that
  leaves the store empty; decision records are written with `allow_nan=False`.

### Phase 2: Governed promotion gate — held-out replay, experiment store, rollback decisions (2026-06-16)

Adds the enforcement layer that consumes Phase 1's transparent fitness
substrate before any recursive self-improvement candidate can advance.

* `research/governed_fusion/promotion_gate.py` — deterministic promotion
  gate for candidate evidence records. The gate reads only the
  `external_label` bucket, requires AUROC or F1 improvement without
  AUROC/AUPRC/F1 regression, enforces σ_Immutable / benevolence /
  conformal / Lyapunov floors, requires capability-regression success,
  checks the latest `status="ok"` marginal-ablation baseline when present,
  and emits rollback decisions for failed canaries.
* `ExperimentStore` — append-only JSON decision records plus `index.jsonl`
  so promotion, rejection, and rollback outcomes remain reviewable.
* `tests/research/test_governed_promotion_gate.py` (8 assertions) —
  promotion eligibility, external-label-only optimization, broadened-bucket
  rejection, capability-evidence enforcement, ablation-regression rejection,
  canary rollback, manifest-count reconciliation, and append-only store
  behavior.
* `.github/workflows/ablation-ledger.yml` — extends the transparent fitness
  substrate CI lane with the governed promotion gate tests.
* `docs/GOVERNED_PROMOTION_GATE.md`, `docs/SELF_IMPROVEMENT_LOOP.md`, and
  `README.md` — document the Phase 2 operator contract, held-out replay
  terminology, human-review requirement, and rollback behavior.

### Phase 1: Transparent fitness substrate — leakage-split gate + fusion-marginal ablation ledger (2026-06-16)

Closes the **measurement** stage of the governed recursive self-improvement
loop. The autonomous self-improvement work that follows (Phase 2: governed
promotion gate; Phase 3: Reflexion / drift-triggered recalibration /
recurring dormant revival) optimises whatever the fitness signal measures —
and Phase 0 verification found the fitness signal was partly
provenance-contaminated. The `loaders/` side of the codebase had no
label-provenance discipline at all, even though
the governed-fusion suite (`research/governed_fusion/`) was already using
those loaders to compute its 23-event live headline. Of the 15 concrete
live-API loaders, **13** manufacture anomaly labels by thresholding a column
that is also a scored feature (label leakage / circularity); only **2**
(`network_security`, `sepsis`) produce labels independent of the scored
signal. Phase 1 surfaces that transparently and CI-enforces the discipline
so the autonomous loop reads only independently labelled live events.

* `src/omni_mercury_engine/loaders/base.py` — adds
  `LABEL_SOURCE: str = "ground_truth"` to `BaseDomainLoader`
  (provenance-declaration default; loaders override to declare their
  actual provenance). Mirrors `datasets/base.py`.
* Every concrete loader updated with an audited `LABEL_SOURCE` and a
  per-loader justification citing the exact circular pattern: earthquake
  (`magnitude >= mainshock_mag - 1.0` on feature[0]), hurricane (24h wind
  delta on the scored wind-delta feature), tornado (`mag >= 3` on
  feature[0] `ef_scale`), fema (DR + IA + PA flags scored as features),
  marine (richness-decline threshold on the scored richness-loss feature),
  pandemic (7d rolling > 2x 30d baseline on the same 7d feature; ebola_2014
  reconstructed), energy (`kp >= 7` on reconstructed kp), tsunami
  (reconstructed series), financial (VIX / yield-curve thresholds on scored
  features), flood (`gauge_height >= NWS flood-stage` on feature[0]),
  landslide (fatality/size on scored features), wildfire (FRP p90 on scored
  `frp`), volcanic (alert level on scored feature 1). The two
  ground-truth loaders (`network_security`, `sepsis`) are explicitly
  declared with their justifications.
* `src/omni_mercury_engine/loaders/label_provenance.py` — canonical
  per-loader registry (`LABEL_PROVENANCE_REGISTRY`), audit
  (`audit_label_provenance`), discovery (`discover_loaders`), AST
  circularity heuristic (`scan_circular_label_construction`), and the
  ground-truth subset helper (`ground_truth_loader_keys`). Module CLI:
  `python -m omni_mercury_engine.loaders.label_provenance --check`.
* `research/governed_fusion/label_provenance.py` — pivots from per-loader
  `LABEL_SOURCE` to per-event `(label_provenance, series_provenance,
  external_label)`. Single source of truth for the manifest, the
  aggregator, and the ablation ledger.
* `research/governed_fusion/manifest.json` — every entry now carries the
  three new audit fields, plus a top-level `provenance_summary` block.
  Today's headline: 2 external-label, 21 self-label, 7 reconstructed.
* `research/governed_fusion/build_manifest.py` — emits the new fields
  and summary; reads from the loader audit so manifest and loader
  registry can never silently drift.
* `research/governed_fusion/evaluate.py` — `aggregate()` adds a
  `per_provenance` breakdown alongside `per_event` / `per_domain` /
  `overall`. Phase 2's promotion gate reads `external_label` only.
* `research/governed_fusion/measure_baseline.py` — prints the
  per-provenance block alongside the per-domain block, with the
  external-label bucket tagged `(FITNESS)`.
* `research/governed_fusion/measure_marginal_ablation.py` — standing
  fusion-marginal ablation ledger: leave-one-component-out
  ΔAUROC / ΔAUPRC / ΔF1 on the external-label live subset only. Records
  timestamp, git SHA, components, event keys, and the selected cache
  path per run. When the selected score cache is absent (typical on a
  fresh CI runner) emits an informational `needs_cache` record (exit 0)
  so the ledger keeps a chronological reachability account.
* `research/governed_fusion/ablation_ledger.json` — seed ledger; the
  fitness function the autonomous loop will climb.
* `.github/workflows/ablation-ledger.yml` — runs on PR + nightly: loader
  audit, manifest integrity, ledger schema + math correctness, then the
  measurement itself. Uploads the ledger as a CI artifact.
* `tests/loaders/test_label_provenance_gate.py` (12 assertions) —
  mirrors the `datasets/` gate for the live-API path.
* `tests/research/test_governed_fusion_label_provenance.py` (7
  assertions) — per-event provenance integrity, manifest-vs-loader
  drift, the `external_label` bucket lock at exactly
  `{network_security/batadal, network_security/nsl_kdd}`.
* `tests/research/test_marginal_ablation_ledger.py` (6 assertions) —
  ledger schema, math correctness on synthetic events (informative
  components produce positive lift; a noise component ranks below),
  the missing-cache informational path, and `--cache-dir` propagation
  into the score-cache reader.
* `docs/SELF_IMPROVEMENT_LOOP.md` — the rollout narrative: the verified
  capabilities map, the transparent baseline, the 10-dataset reconciliation
  (mirror vs cannot-score), Phase 1 deliverables, and the explicit
  out-of-scope decisions for Phases 2–8 (each is a documented decision,
  not a gap). NAS is excluded; UBI 9/10 is a separate PR; HITL for
  actuation is documented as a Phase 8 contingency, not a Phase 1
  deliverable (Mercury is sensing-only today).
* `docs/BENCHMARKS.md` — adds the provenance-split block surfacing the
  Phase 1 finding: of the 23 live events, only 2 (network_security)
  produce labels independent of any scored feature; external-label mean AUROC
  **0.7704** / F1 **0.1863**.
* `docs/DORMANCY_LEDGER.md` — section 7 names the ablation ledger as
  the standing measurement substrate that Phase 3 revival routing reads.
* `README.md` — adds a paragraph on the live-API loader provenance
  audit and points at `docs/SELF_IMPROVEMENT_LOOP.md`.

No gate, threshold, ethics floor, or assertion was loosened to make CI
pass: σ_Immutable (0.93/0.96), benevolence floor (0.99), Lyapunov
certificate, conformal coverage target (0.90), the
`fusion-regression.yml` floors, and the existing `datasets/`-side
provenance gate are all unchanged. The new gates are additive.

### Geospatial kernel + movement-plausibility detector — one lat/lon geometry, a boreal undercount fixed, FINDΩYOU groundwork (2026-06-11)

Consolidates all latitude/longitude geometry behind a single numpy-only
kernel, fixes a measured correctness defect in the wildfire loader, and
adds the engine's first trajectory-level geographic detector.  The
detector core is a corrective rebuild of an externally audited geospatial
module (2026-06-10 audit); all four audited defects are reproduced as
regression tests against the corrected implementations.

* **New `omni_mercury_engine.utils.geo` — canonical geospatial kernel
  (numpy only).** `haversine_km`, `haversine_km_to_point`, and
  `pairwise_haversine_km` use the numerically robust `atan2` haversine
  form and the IUGG mean Earth radius (6371.0088 km).
  `neighbor_counts_within_km` (with an optional temporal co-occurrence
  window) and `dbscan_geo` (kilometre-denominated DBSCAN, Ester et al.
  1996) prune candidate pairs with an *exact* latitude band — on the
  sphere the central angle is at least |Δlat|, so the band can never drop
  a true neighbor — and never materialize the full n² distance matrix.
  `dbscan_geo` closes a real gap: the native `ml.mercury_ml.DBSCAN`
  delegates to `scipy.cdist`, which has no haversine metric, and no single
  euclidean-degrees eps is meaningful across latitudes (50 km spans 0.45°
  of longitude at the equator but 0.90° at 60°N).

* **Wildfire loader correctness fix (boreal undercount).**
  `WildfireLoader._compute_spatial_clustering` pre-filtered candidates
  with a flat `radius_km / 111 * 1.5` degree box before exact distance
  checks.  Above ~48° latitude one degree of longitude spans too few km
  for that buffer: at 62°N (boreal fire country, prime VIIRS territory)
  two detections 9.49 km apart — inside the 10 km radius — were dropped
  entirely, and a 400-point synthetic boreal swarm lost 9,886 true
  neighbor pairs (~20%).  The `cluster_count` feature is now exact at
  every latitude; `tests/loaders/test_geo_rewire.py` pins mid-latitude
  parity with a verbatim port of the legacy loop and demonstrates the
  boreal fix against brute-force ground truth.

* **Loader geometry rewired to the kernel — faster, behavior pinned.**
  Tornado temporal clustering (same-hour + 100 km co-occurrence): 18×
  faster at n=3,000 (4.7 s → 0.26 s).  Wildfire spatial clustering: 3.6×
  at n=3,000 and 9× at n=20,000 (12.7 s → 1.4 s).  Wildfire spread rate:
  2× (lookahead window vectorized; first-minimum tie semantics
  preserved).  Tornado centroid distance (`geo_anomaly` feature)
  vectorized.  Three private haversine copies collapse into the kernel.
  Standardizing the Earth radius (6371.0 → 6371.0088, +1.4e-6 relative)
  shifts a 10 km cutoff by ~1.4 cm; measured effect on dense synthetic
  swarms: 2–4 flipped neighbor counts per ~1.7M true pairs, versus VIIRS
  positional resolution of 375 m.  Deliberately untouched: the hurricane
  loader's translation speed (angular degrees by documented design) and
  the landslide loader's ArcGIS bounding box (correct cos-compensated
  server-side prefilter).

* **New `GeoMovementAnomalyDetector`
  (`detectors/geo_movement.py`) — movement plausibility for
  (lat, lon, time) trajectories.**  BaseDetector-compliant
  (`fit`/`detect`/`extract_features`, auto-calibration supported,
  registered in the detectors lazy-import table and `DETECTOR_MANIFEST`
  as `geo_movement`, BASE category, feature_dim 8), plus a casework
  `assess(history, current, now)` API for datetime-stamped sightings —
  groundwork for the FINDΩYOU missing-persons mission.  Three channels in
  [0, 1] (velocity vs. a 130 km/h feasibility ceiling, step-length z-score
  vs. *fitted* history, silence vs. expected reporting cadence) fuse by
  noisy-OR, so one saturated channel alone exceeds any threshold below
  1.0.  The audited module it rebuilds could not fire on a 450 km/h
  teleport: weighted-sum fusion (0.4/0.3/0.3 vs. a 0.72 threshold) capped
  any single channel at 0.4, and the jump statistic measured spread from
  the candidate point, suppressing its own signal.  Both failure modes
  are locked as tests (`test_impossible_jump_fires`,
  `test_single_saturated_channel_fires_alone`,
  `test_jump_scored_against_history_not_candidate`).

* **Tests: 36 new** across `tests/utils/test_geo.py` (ground-truth
  distances, brute-force neighbor parity, permutation invariance, DBSCAN
  transitivity/planted-cluster recovery), `tests/detectors/
  test_geo_movement.py` (audit regressions + interface contract), and
  `tests/loaders/test_geo_rewire.py` (parity with verbatim legacy
  implementations).  Affected existing suites pass unchanged (domain
  loaders, detector manifest, spatial detector).

### Engine integration — manifest detectors made reachable through `OmniMercuryEngine`; inference/training exception symmetry

Closes the wiring gap that left every `DETECTOR_MANIFEST` entry outside the
five built-in base detectors — notably the new `geo_movement`, and
`graph_based` — registered but unreachable through the engine: nothing
consumed the manifest, so such a detector never participated in the
detect → fuse → decide path it was nominally registered for.

* **Opt-in detector-registration seam on `OmniMercuryEngine`.**
  `register_detector(name, detector, *, replace=False)` adds a
  `BaseDetector` to the engine's active set; `enable_detector(name)` bridges
  the declarative manifest to a live instance
  (`engine.enable_detector("geo_movement")`); `available_detectors()` reports
  which manifest detectors are active on the engine. Purely additive — the
  default five-detector set (`statistical`, `temporal`, `spatial`,
  `dimensional`, `directive`) and the calibrated fusion path are byte-identical
  until a caller opts in. A detector registered after `fit_fusion` has trained
  is recorded but ignored at fusion inference — which is restricted to the
  trained feature groups (`_fusion_feature_groups`) — until a re-fit, and that
  case warns rather than silently misleading. Once enabled, the detector
  contributes its feature group through the single source of truth
  (`_extract_fusion_features`) on both the training and inference paths;
  verified end-to-end for `geo_movement` (an `(n, 8)` group on trajectory
  input, the five base groups still present).

* **Inference/training exception-handling symmetry (latent defect, fixed).**
  The inference feature extractors (`_extract_detector_features`,
  `_extract_model_features`) caught only six built-in exception types, while
  the training extractor (`_extract_fusion_features`) catches `Exception`. A
  detector or model that fail-louds with Mercury's own `OmniAnomalyException`
  (e.g. `geo_movement` raising `DetectorException` on non-trajectory data)
  was therefore skipped gracefully during training but would crash
  `detect_with_fusion` at inference. Both inference extractors now also catch
  `OmniAnomalyException`, honouring their documented graceful-degradation
  contract; the five built-ins never raise it, so the default path is
  unchanged. This asymmetry is what made a specialized, fail-loud detector
  unsafe to add — fixing it is the precondition for the seam above.

* **Tests: 15 new** (`tests/core/test_engine_detector_registration.py`):
  default-set lock, `register`/`enable`/`available` contracts, the
  post-training warning path, fusion-feature-group wiring for an enabled
  detector, and the inference graceful-skip regression lock (a fail-loud
  detector cannot crash `detect_with_fusion`).

* **Detector audit — orphaned first-class detector wired in, manifest hygiene.**
  A full audit of every `*Detector`/`*Analyzer` in `detectors/` (manifest
  membership, exports, engine reachability, callers, tests, `BaseDetector`
  conformance) found the framework largely sound — the apparent orphans are
  deprecated-on-purpose (`enhanced_statistical`), abstract bases
  (`BaseVLMDetector`, `BaseVisualDetector`), or wired subsystem components
  (the VLM / advanced families, the EMP pulse channels of `EMPDetector`). Two
  genuine gaps were closed:
  * **`KMeansDistanceDetector` is now a first-class registered detector.** The
    revived dormant clusterer (measured on ADBench, ROC-AUC ~0.8–0.98) existed
    as a plain duck-typed class in neither the manifest nor the exports — built
    as "a first-class detector" but reachable through nothing. It now subclasses
    `BaseDetector` (backward-compatible constructor; genuinely accepts numpy or
    torch input), is registered in `DETECTOR_MANIFEST` (`kmeans_distance`, BASE,
    feature_dim 9) and exported, so it is reachable via
    `engine.enable_detector("kmeans_distance")`. Opt-in, not a default base
    detector — the calibrated ensemble is unchanged.
  * **`SpectralDomainFrequency` now has a dedicated test.** A shipped manifest
    BASE detector that previously relied on incidental coverage; new
    `tests/detectors/test_spectral_domain_frequency.py` exercises the real
    fit → extract_features → detect path and the `BaseDetector` contract.

* **Manifest/engine drift lock (CI):
  `tests/detectors/test_detector_manifest_integrity.py`.** Pins the invariants
  that let the gap form: every manifest entry resolves to its class; every BASE
  entry is a `BaseDetector` exposing the engine's contract (the check that would
  have caught `kmeans_distance`); the engine's default detector set is a subset
  of the manifest, so the two catalogs cannot silently drift; and
  `geo_movement` / `graph_based` / `kmeans_distance` are reachable through the
  engine seam.

### Deployment image & review hardening

* **Deployment image on Debian trixie — accepted-CVE ledger cut 14 → 8 → 3
  (Critical 4 → 2 → 0); closes GOV-001.** The runtime base is
  `python:3.13-slim-trixie` (Debian 13.5). The `.trivyignore` ledger and the
  SECURITY.md table were rebuilt from a fresh trixie scan with the gate's own
  trivy 0.70.0 and cross-checked against the Debian Security Tracker: every
  retained entry suppresses a real, currently-present trixie CRITICAL/HIGH OS
  finding (verified 1:1 against the built-image scan — no inert entries). Six
  bookworm-era acceptances were **dropped, not carried inert** — CVE-2023-45853
  (zlib) and CVE-2025-7458 (SQLite) are resolved in trixie's newer packages;
  CVE-2025-59375 / CVE-2026-25210 / CVE-2026-45186 drop because `libexpat1` is
  not an installed dpkg package in the trixie image; CVE-2026-48959 (perl) is no
  longer a CRITICAL/HIGH finding (8 left). Then **`perl-base` was eliminated, not
  accepted**: the Dockerfile purges the Debian-essential `perl-base` and its
  `adduser` consumer right after `apt-get upgrade` (nothing in the runtime needs
  Perl — Mercury is Python/Rust/C and the user is made with `useradd`), removing
  5 more CVEs including **both former CRITICALs** (Archive::Tar / Perl overflow:
  CVE-2026-8376, CVE-2026-42496, CVE-2026-42497, CVE-2026-48962, CVE-2026-9538).
  Verified on the built image: the interpreter, `sqlite3`, `pip` and `useradd`
  all work without Perl. The 3 retained acceptances (`libsqlite3-0`, `ncurses`)
  are linked by CPython itself and have no trixie fix; the prior bookworm
  enumeration is superseded, and the earlier SECURITY.md "re-validation pending"
  caveat is removed — the posture is measured, not deferred.
* **Mesa GL stack removed from the runtime image — accepted-CVE ledger 15 → 14
  (Critical 5 → 4).** `libgl1-mesa-glx` was installed only as OpenCV's `libGL`
  dependency, but Mercury uses `opencv-python-headless` (whose `cv2` extension
  links no `libGL` — verified: zero GL linkage in the wheel's `.so`) and makes
  no `cv2` GUI calls, so it was dead weight carrying an unfixed **Critical** CVE
  (CVE-2026-40393). Dropping the package **eliminates** the CVE rather than
  accepting it: the Dockerfile no longer installs it, and the `.trivyignore`
  entry plus the SECURITY.md ledger row are removed. The ledger header now
  states plainly that the remaining entries are *irreducible* — CPython itself
  links `libsqlite3-0` / `libexpat1` / `zlib1g` / `ncurses` (the `sqlite3`,
  `pyexpat`, `zlib`, `readline`/`curses` stdlib modules), and `perl-base` is
  apt's own `adduser` dependency — rather than the previous "deferred slimmer
  image" framing. The blocking Trivy gate (`ignore-unfixed: false`) re-verifies
  the package stays gone on every build.
* **`register_detector` now warns when replacing a *trained* feature group.**
  Review surfaced that `replace=True` after `fit_fusion` left the fusion network
  consuming a different feature distribution for an existing group with no
  operator signal; it now warns and recommends a re-fit in that case too, not
  only when adding a new group. Locked by a new test.
* **Wildfire clustering docstring** corrected to describe the actual
  latitude-band pruning (`neighbor_counts_within_km`) instead of the
  inaccurate "vectorized in row blocks".

### LLM layer: free-weights-first & provider-neutral under Mercury's identity

Defaults, selection order, and naming only — no rewrite; OpenAI and every other
cloud adapter stay available, just never privileged.

* **Free/local is the baseline default.** `LLMModelRegistry.select()`
  (`models/llm_registry.py`) now orders **free/local ahead of paid cloud**:
  `local`/`builtin` providers are treated as genuinely cost-0, so they sort
  first, satisfy any budget, and win ties; an *undeclared cloud* price stays
  unknown and is excluded from budget queries (not assumed free). Replaces the
  prior "priced-first, unpriced-last" order. Pinned by
  `tests/models/test_llm_registry_local_first.py` and the updated
  `tests/models/test_llm_registry.py`.
* **`MERCURY_OFFLINE` is the master air-gap for the LLM layer too.**
  `FallbackLLMChain` (`models/foundation/ollama_adapter.py`) and the reasoning
  router now refuse to construct or call any cloud adapter when
  `MERCURY_OFFLINE` is set — reusing the dataset layer's
  `offline_mode_active()` so one switch governs both. Local + template only.
  Pinned by `tests/models/test_llm_offline_airgap.py` (zero cloud construction)
  and a reasoning-router offline test.
* **No vendor privileged by default.** `FallbackLLMChain` already tries local
  (Ollama) → optional cloud → template, with cloud off unless explicitly
  enabled and configured (confirmed, unchanged). `.env.example` now presents
  the reasoning backend as local/free by default and lists cloud keys as an
  unordered optional set (no OpenAI-first framing); `docs/OFFLINE_OPERATION.md`
  documents the now-*enforced* LLM air-gap and the free/local-first baseline.

### Reasoning backend layer — Mercury-owned, offline-first, ethics-gated co-AI interface (`reasoning/`)

* **`reasoning/` subpackage** (new): a pluggable reasoning layer Mercury *calls*
  as a subordinate dependency — never a wrapper Mercury sits behind.
  * `ReasoningBackend` (ABC): typed, provider-neutral surface — `explain()`,
    `propose_hypotheses()`, `synthesize_report()` (`reasoning/schemas.py`).
    Every operation passes Mercury's benevolence + σ_Immutable dual hard
    ethical gate (`enforce_dual_ethical_gate`) before any output is surfaced;
    the gate fails closed, so an integrated LLM cannot bypass Mercury's
    governance.
  * Implementations: `MockReasoningBackend` (deterministic, network-free, for
    CI), `LocalReasoningBackend` (offline-first / air-gap-safe over the local
    Ollama+template chain — free to run, no external call), and
    `RemoteReasoningBackend` (operator-declared frontier model; no hard-coded
    model name, credentials via env).
  * `ReasoningRouter`: offline-first routing — the local backend is the floor
    and default, the remote backend is reached only on explicit opt-in, and
    hard-offline mode provably never selects or calls a network backend.
  * Threads the PR #289 `UsageLedger` so token spend is accounted regardless of
    which backend serves a call.
  * **Wired into the engine (real consumer, not standalone substrate):**
    `OmniMercuryEngine.enable_reasoning()`, the lazy `reasoning_backend`
    property, and `explain_detection(result, domain=...)` let Mercury call the
    backend on its *own* `detect_with_fusion` certificates — Mercury owns the
    detection and the decision and invokes the subordinate backend only to
    render a governed, evidence-grounded explanation (offline-first; the dual
    ethical gate still fail-closes the call at the engine boundary).
    `reasoning_usage()` exposes the threaded ledger's provider-truthful totals
    (the offline template path books nothing — never fabricated tokens). The
    `LLMModelRegistry` is consumed too: `reasoning.backends.select_reasoning_model()`
    lets an operator-populated registry choose the local model (free/local-first)
    instead of a hard-coded default.
  * Tests: `tests/reasoning/test_reasoning_backend.py` (gate fail-closed,
    provenance stamping, hard-offline zero-network, ledger threading,
    registry-driven model selection) and `tests/core/test_engine_full.py`
    (engine-governed explanation, ethics fail-closed at the engine boundary,
    registry-driven local model).
* **Vendor-neutral defaults (identity):** `LLMConfig.model_name` default changed
  from `"gpt-4o"` to `""` (unset) — each adapter applies its own
  provider-appropriate default, so no vendor model id (nor the `"template"`
  sentinel) is ever sent across providers; the `llm_registry` docstring example
  now uses a local open-weights model instead of a hosted one. Mercury presents
  as itself by default, not as any external AI.

### Calibration — `StrictIsotonicCalibration` ported from PR #275 (X1 survivor) (2026-06-12)

* **`StrictIsotonicCalibration`** (`core/calibration.py`): isotonic calibration
  with a squeezed strictly-increasing tie-break, so the map preserves AUROC
  exactly while keeping isotonic's ECE (PR #275 measured vanilla isotonic
  losing AUROC on 5/6 ADBench datasets, e.g. thyroid 0.9788 -> 0.9535).
  Exposed as `calibrate_detector(..., method="strict_isotonic")`; pinned by
  `tests/test_calibration_brief.py`.
* Five factually wrong "requires scikit-learn" `ImportError` messages reworded
  to name the actual dependency (`omni_mercury_engine.ml.mercury_ml`, pure
  numpy/scipy) — PR #275's V12c finding.
* **Evidence suite** `benchmarks/calibration_brief/` (validation register
  V1–V11, explorations X1/X2/X8 with captured results) restored from the
  deleted PR #275 branch (`refs/pull/275/head`, commit `bc211a0`), with a
  disposition note reconciling it to what `main` has since shipped.
* Deliberately not ported: the brief's `BetaCalibration` (superseded by the
  accept-gated Beta landed in #278), the X12a eta-multiply lint (the eta^Phi
  term was settled as an opt-in decoupling, R6 default-off), and the
  MATH_SPEC phi_sum patch (superseded by the `Φ + 2` canonical derivation).

### Session bootstrap — no hosting-vendor files in the tree; Mercury-named script hardened (2026-06-12)

The repository carries no hosting-vendor configuration (the vendor-named
directory stays removed). The environment bootstrap (AMA Cryptography native
PQC build plus `[ml]`/tooling install) lives only at
`scripts/mercury_session_setup.sh`, now hardened with a
disposable-environment guard: it runs in hosted remote sessions (which mark
themselves at runtime) or when forced via
`MERCURY_SETUP_FORCE=1 bash scripts/mercury_session_setup.sh`, and refuses to
modify any other environment. Automatic session start-up, where desired, is
configured in the host environment itself, outside this repository, by
pointing the host's session-start hook at this script.

### Reproducibility & API consolidation — deterministic construction of the fusion feature path's random components; conformal serialization moved behind a first-class API (2026-06-11)

Consolidation follow-up to the checkpoint round-trip closure (ROADMAP row 16,
PR #286): salvages the source-level pieces of the superseded competing
implementation (PR #288) that remain advantageous on top of the shipped
fitted-state persistence design. No checkpoint format change; the shipped
`default_fusion.pt` is byte-unchanged and loads identically.

* **Deterministic construction (measured):** three fusion feature groups were
  still instance-dependent for *checkpoint-free* engines — two engines
  constructed under different ambient RNG states extracted different features
  (measured max|Δ| at the pre-fix commit: `dimensional` ≈ 3.01, `multiverse`
  ≈ 0.98, `temporal` ≈ 0.27). Root causes fixed at the source, each under a
  forked fixed-seed RNG that leaves the caller's global stream untouched:
  the temporal detector's untrained LSTM projector is now an architecture
  constant; the dimensional autoencoder's `fit()` init is deterministic, so
  fit is a pure function of its data (full-batch, shuffle-free loop); the
  default multiverse population is fixed-seed with index-only universe IDs
  (an explicitly injected `rng` still overrides). Measured after: **all 13
  feature groups bit-identical across instances** (max|Δ| = 0.0), pinned by
  `tests/test_deterministic_construction.py`. Checkpoints continue to
  persist fitted state, so every existing artifact reproduces exactly.
* **Conformal serialization API:** `BinaryConformalClassifier` gains
  `export_state()` / `from_state()` (checkpoint-safe primitives, layout
  identical to the `conformal_state` key fusion checkpoints already carry);
  `engine.save_model`/`load_model` now use them instead of poking private
  attributes. `from_state` coerces string-keyed thresholds, so a JSON
  round-trip of the mapping loads equally.
* **Stale-state drop on load:** `load_model` now *replaces* engine-local
  calibration state with the checkpoint's, including replaced-by-`None` — a
  temperature calibrator or conformal surface fitted against a previous
  fusion stack is dropped rather than silently misapplied to the loaded one
  (regression-pinned in `tests/test_fusion_checkpoint_roundtrip.py`).
* **Auto-fit audit record kept truthful across load** (defect surfaced by
  PR #288's review): restoring a detector's fitted state from a checkpoint
  now drops that detector from `_inference_auto_fit_detectors` — the
  first-batch leak it recorded no longer describes the engine, and a stale
  entry would also suppress the audit warning for any future post-load
  auto-fit. Legacy loads restore nothing, so a surviving contamination
  keeps its record (regression-pinned).
* **Tooling:** `scripts/train_default_fusion.py` self-check now asserts
  detectors-restored-as-fitted on load and save→load probability equivalence
  (verified here: max |dP| = 1.35e-07 on a scratch artifact); the fusion
  regression guard's "train rather than load" rationale updated to reflect
  the row-16 closure (gate re-run against the existing committed baseline:
  **PASS**, no re-pin needed).

### Performance & correctness — optimization/efficiency pass over the serving and checkpoint paths, with four pre-existing defect families fixed to completion (2026-06-11)

Profile-first discipline: every change below is driven by a fresh measurement
in this session, every fast path is a compiled form of pinned reference
semantics, and the candidates the profile did *not* confirm are recorded as
such instead of "optimized".

* **Serving purity (defect):** identical batches scored differently
  call-to-call — the directive detector's recursive-memory buffer and the
  neural cognitive model's hippocampal buffer leak streaming state across
  calls, so the first ``memory_depth`` rows of every batch depended on what
  the previous call left behind (measured: `coordinate()` on the same input
  twice returned different consensus scores; repeated `detect_with_fusion`
  drifted). Fixed at the *boundaries*, not in the components:
  `DetectorAgent.score_batch` and the engine's fusion feature extraction
  reset transient state per call (pure function of fitted state + batch);
  detectors/models keep their documented streaming semantics for direct
  callers. The orchestration-validation grid was re-run under the corrected
  semantics — **all four pre-registered bars still pass** (consensus AUC
  0.841 vs members 0.833; reflexion +0.054 paired balanced accuracy 15/15
  acted; planning 129/129; trace fidelity 600/600) and the artifact is
  regenerated.

  Disclosed consequence: purity exposed that **batch-calibrated conformal
  sets never matched the single-sample serving regime** — several detector
  features are batch-relative by design (recursive-memory deviation,
  batch-percentile stability, batch-max magnitude, sliding windows,
  within-batch min-max), so a row scored alone does not get its in-batch
  score; the empirical-coverage fixture's historical ~0.94 rode on the
  streaming buffers accidentally making serve calls resemble batches, and
  under purity it measured 0.77. The principled remedy ships:
  `calibrate_fusion_conformal(..., per_sample=True)` scores calibration
  rows in the exact serving regime (verified: per-row `score_fusion`
  equals the served `detect_with_fusion` probability bit-for-bit),
  restoring the guarantee by exchangeability-by-construction — measured
  **0.9405 empirical coverage at the 0.90 target** (abstention 5.95%,
  grounded accuracy 1.0, zero gate-vs-certificate contradictions). Batch
  consumers' default path is byte-unchanged.
* **Exact incremental single-sample serving (measured 7.5x).**
  `DetectorAgent.detect()` re-scored the full 512-row reference batch
  through every detector protocol per call (54 ms/sample across the five
  agents; directive 28 ms of it). New
  `agentic/_incremental_serving.py` serves the appended row *bit-identically*
  for the three profile-dominant detectors by caching row-independent terms
  and recomputing only what the row genuinely participates in: temporal
  9.4 -> 0.09 ms (105x), spatial 11.4 -> 0.29 ms (39x), directive
  27.7 -> 2.6 ms (11x; floor = the batch-global blend scalars, which include
  the new row by definition); total serve 52.6 -> 7.0 ms. Never an
  approximation: unsupported types/subclasses/configs, stale caches and
  non-finite inputs fall back to the verbatim full path, and a runtime
  guard refuses out-of-contract scores. Pinned bit-identical across seeds,
  reference sizes (1..512), duplicate-heavy ties and non-default configs by
  `tests/test_native_acceleration.py::TestIncrementalServingParity`.
* **GSIS numba lane (measured 2.9x at 1.1k rows, 5.5x at 4k).** The
  O(n^2 d) gravitational-stability term gets a parallel JIT kernel behind
  the existing optional `[performance]` extra that replicates numpy's
  pairwise-summation order *exactly* — bit-identical distances, percentiles,
  counts and scores (pinned across seeds/shapes/tie-heavy data, plus an
  explicit numba-vs-numpy cross-lane test). float64 rows up to 128 features
  only; other dtypes/widths and numba-less installs keep the byte-identical
  numpy path. The kernel is deliberately non-recursive: numpy's
  recursive-halving regime (d > 128) stays on the numpy lane (self-recursion
  inside a numba parallel region segfaulted under test).
* **Parallel agent scoring (measured 1.2x).** `coordinate()` now scores the
  five agents on a thread pool — admissible only because the purity fix
  makes scoring thread-independent: results are pinned bit-identical to
  serial (5-run repeat test), wall-clock 72.8 -> 62.2 ms on cardio-scale and
  285 -> 228 ms at 2.2k rows on the 4-core profiling box; the modest factor
  is honest (the detectors' internal numba/BLAS parallelism already uses the
  cores). Per-agent failure exclusion and quorum semantics unchanged.
* **Fusion checkpoint round-trip fidelity — ROADMAP row 16 CLOSED.**
  Measured root causes: a loaded engine auto-fit its base detectors on the
  first inference batch (the leak), torch-module domain models
  re-randomized per process, the multiverse population re-randomized per
  construction, and **two stub models emitted fresh RNG output per call**
  (`AffectiveAnomalyModel` fed 64 columns of pure noise into the fusion
  feature set at train and serve; `neural`'s buffer drifted) — same-engine
  repeat drift was ±0.08 before any save/load was attempted. Remediation:
  (1) `get_fitted_state()`/`set_fitted_state()` on all five base detectors
  (statistical incl. Oracle re-arm via the existing federation-stats path;
  dimensional incl. trained autoencoder weights; spatial rebuilds its
  KDTree from persisted points; temporal incl. its construction-random LSTM
  feature weights); (2) checkpoint now carries `model_state_dicts`
  (torch-module domain models), `model_fitted_state` (multiverse
  population) and `conformal_state` (calibrator thresholds); (3) the
  affective stub is quarantined to deterministic *neutral* output (zero
  features, 0.5 scores, one-time warning) per the repo's
  parapsychology/schumann precedent — it fabricates nothing; (4)
  `default_fusion.pt` regenerated with full state. **Measured: save->load
  max |dProb| = 5.9e-08 (bar < 1e-3; was ~0.76 historical / 0.427
  re-measured), same-engine repeat drift exactly 0.0, conformal sets
  identical.** Pinned by `tests/test_fusion_checkpoint_roundtrip.py`;
  legacy checkpoints still load with legacy behavior; fusion regression
  gate re-run green.
* **Spatial detector defects (found by the parity battery, fixed):**
  (a) `_NativeLOF.decision_function` raised `AxisError` whenever k == 1
  (single training sample, or `n_neighbors=1`) — scipy squeezes the
  neighbor axis for k=1; restored explicitly. (b) `detect()` crashed
  outright on NaN/Inf queries (`KDTree.query` refuses them) before its own
  downstream NaN guards could run — sanitized at the boundary exactly like
  the temporal detector; finite inputs are byte-unaffected. Regression
  tests in `tests/detectors/test_spatial_detector.py`.
* **Test-infra:** the suite's per-xdist-worker thread cap now covers
  numba's parallel kernels too (`NUMBA_NUM_THREADS=1`, mirroring the
  existing torch cap in `tests/conftest.py`) — without it each worker
  spawned a full physical-core pool for the new GSIS prange lane and the
  oversubscription starved borderline-heavy tests past the 300 s timeout.
* **Candidates re-profiled and *not* optimized (honest negative results):**
  (1) the inherited "fit_fusion 610 s / 20 epochs on cardio" figure does
  not reproduce — measured 6.8 s (labels), 10.1 s (semi-supervised), 9.1 s
  (domain encoder) for 20 epochs on this 4-core box, with feature
  extraction ~1.2 s of it; nothing profile-justified to attack. (2) GPU
  lane: `torch.cuda.is_available()` is False in this environment — no GPU
  work was attempted and no GPU numbers exist; the lane awaits the
  operator's GPU CI runners.

### DimensionalAnalyzer — dead spectral-divergence term given real, gate-cleared semantics (2026-06-11)

The DB term's spectral-divergence component — disclosed as identically zero
in the native-acceleration pass — now carries real semantics
(operator-approved): each row's feature-axis power spectrum is compared
against the fit-time mean training-row spectrum with the original
normalized-distance formula.

* **Pre-registered ablation gate, cleared decisively**
  (`benchmarks/db_spectral_ablation.py`, artifact
  `artifacts/db_spectral_ablation.json`; paired OFF/ON fits under identical
  global seeds, real ADBench, 5 datasets × 3 seeds): mean paired detector
  ΔAUC **+0.071** (bar +0.002), seed agreement **0.93** (14/15), the term
  alone moving from chance (**0.460** — confirming it was dead weight) to
  **0.811** AUC. Largest gains where spectral structure exists (thyroid
  +0.12…+0.20, cardio +0.08…+0.14); the single negative is −0.0013, inside
  the noise floor.
* **Default flipped to enabled** per the cleared gate;
  `{"db_spectral_divergence": False}` preserves the pre-2026-06-11 shipped
  scores exactly (pinned byte-equal by the legacy-oracle parity test).
* **Behavioral contract:** `tests/test_db_spectral_term.py` — the enabled
  term detects precisely what it claims (every spectrum-diverging row
  out-scores every conforming row on constructed spectral structure,
  multi-seed), fit-time baseline, and dimension-mismatch truncation.
* Orchestration-validation grid re-measured with the new default; all four
  pillar-B bars still pass (artifact regenerated).

### Operations — explicit online/offline (air-gapped) modes for the data layer (2026-06-11)

Production deployments require both connectivity modes; detection itself
was already fully local — the gap was the data layer.

* **`MERCURY_OFFLINE=1`** refuses every dataset-layer network fetch at the
  single HTTP chokepoint (`datasets/base.py::http_get_with_retry`) before
  any socket is opened, raising the new `OfflineModeError` with a
  remediation hint. Cached data keeps serving; uncached data fails closed —
  never stale, never synthetic. Read at call time (the `MERCURY_ENV`
  contract), so tests and operators toggle without restarts.
* **`MERCURY_DATA_DIR` / `MERCURY_CACHE_DIR`** env overrides for the
  dataset directories (defaults unchanged when unset), so air-gapped
  deployments pin a stable cache location.
* **`scripts/prefetch_datasets.py`** primes the cache while online
  (`--adbench cardio thyroid …` or `--adbench-all` for the 47-dataset
  catalog); exits non-zero on any failed fetch — a partial prime is
  reported, never silently accepted.
* **Docs:** `docs/OFFLINE_OPERATION.md` (mode matrix, env vars, priming
  runbook). **Tests:** `tests/datasets/test_offline_mode.py` pins flag
  parsing, refusal-before-socket, cached service under offline mode,
  fail-closed uncached errors, and the env-aware directory defaults.

### Performance — native acceleration pass on the live detection hot path, equivalence-pinned (2026-06-11)

A profile of the orchestrated detection path on a real-scale batch
(1,132 × 21) showed 91% of time inside two live detectors' per-sample
Python loops, not in the new orchestration layer. Every fast path below is
a *compiled form* of its reference semantics — outputs are pinned
bit-for-bit (or to 1e-12) by `tests/test_native_acceleration.py`, and the
full orchestration-validation grid reproduces its pre-optimization verdicts
(consensus 0.829 vs members 0.823; reflexion +0.078; planning 129/129;
fidelity 600/600).

* **`DimensionalAnalyzer._dimensional_code_breaking`:** one batched FFT
  over all rows replaces two FFT calls per sample. Disclosed in the same
  pass: the loop's spectral-divergence component is **identically zero**
  for every sample (a single row's per-column FFT has length 1, so the
  half-spectrum slice is empty and the divergence evaluates to
  `0/(0+1e-10)`), meaning 50% of the DB-score weight has always been a
  constant. Preserved as an explicit, documented zero — giving the term
  real semantics would change this live detector's shipped scores and
  therefore requires its own measured ablation gate first.
* **`SigmaDirectiveDetector._detect_micro_anomalies`:** a strided
  sliding-window view computes all window variances in one reduction
  (formerly ~24k one-window `np.var` calls per batch).
  **`_gravitational_stability_check`:** chunked pairwise broadcasting
  replaces the per-row loop (one full distance pass + one percentile call
  per sample); the O(n²·d) semantics are inherent and preserved.
* **Orchestrator consensus fast path:** the per-sample
  `ConsensusProtocol` derivation is vectorized for the default
  CONFIDENCE_WEIGHTED method; the protocol remains the semantic
  authority — a deterministic subsample of every batch is re-derived
  through the real protocol and any divergence fails the batch closed
  (`MultiAgentOrchestrator._spot_check_consensus`). Non-default methods
  keep the per-sample path.
* **Measured effect:** orchestrated `coordinate()` on the profiled
  workload 855 ms → 324 ms (**2.6×**); the wins land in the live
  detectors, so `detect_with_fusion` feature extraction and every other
  consumer of `dimensional`/`directive` benefit equally. Remaining cost is
  intrinsic O(n²) stability math and scipy KDTree — already native.
* **`[performance]` extra (new):** `numba` was referenced by the code and
  the mypy overrides but declared in no dependency group, so the designed
  `@jit` lanes (e.g. `detectors/spatial.py` distance scoring) could never
  be engaged by install — dormant in every environment including CI. Now
  installable via `pip install mercury-agent[performance]`; JIT-vs-numpy
  parity is pinned by `tests/test_native_acceleration.py` (skipped when
  numba is absent).
* **Reproducibility:** `benchmarks/orchestration_validation.py` now pins
  the global numpy/torch seeds per (dataset, seed) run — some live
  detectors carry stochastic components that follow the global seed
  (e.g. `DimensionalAnalyzer`'s autoencoder lane, instance-to-instance
  spread ≈0.56 on identical fits), so the grid was honest but not
  bit-reproducible run-to-run before this.
* **Environment/tooling:** the session-provisioning gap that made
  `pytest-asyncio`/`pytest-xdist`/`pytest-timeout` unavailable locally
  (one async test erroring with "async def functions are not natively
  supported", and the `asyncio_mode` config warning on every run) is
  closed by installing the already-declared dev extras — no code change;
  the failure predated this branch.

### Multi-agent orchestration (pillar B) — the dormant planning/coordination/reflexion/chain-of-thought tier revived as a measured planner/critic/executor loop over the live ensemble (2026-06-11)

Closes the capability-matrix "Multi-agent" gap ("no planner/critic/executor
multi-agent orchestration in Mercury yet") by wiring DORMANCY_LEDGER rows
10–11 (~6,000 LOC retained as "reference only") into one live execution path,
under the same ablation discipline as every other revival.

* **New subsystem: `agentic/orchestration.py` (`MultiAgentOrchestrator`).**
  `HierarchicalPlanner` plans and *drives* each detection episode — the real
  pipeline stages (score → consensus → decide) are registered as options
  whose initiation/termination predicates read the actual pipeline state,
  with TD value learning from real stage rewards; `AgentCoordinator` +
  `ConsensusProtocol` form per-sample consensus across `DetectorAgent`
  wrappers of the engine's five real detectors; `AnomalyReflexion` is the
  critic, turning real labeled outcomes into operating-threshold adaptation;
  `AnomalyChainOfThought` renders per-decision reasoning traces whose stated
  determination is contractually locked to the issued decision. Dual hard
  ethical gates (benevolence floor + σ_Immutable) run fail-closed at the
  orchestrator's decision boundary, exactly as on the engine boundary.
  Engine wiring: `OmniMercuryEngine.enable_multi_agent_orchestration()`.
* **Measured, not asserted:** `benchmarks/orchestration_validation.py`
  (artifact `artifacts/orchestration_validation.json`), real ADBench labels,
  5 datasets × 3 seeds, four pre-registered bars — all passed: consensus
  preserves member signal (mean AUC **0.827** vs mean member **0.821**; best
  member 0.903 reported as context — no claim of beating the trained fusion
  model); reflexion improves the operating point by **+0.079** paired
  balanced accuracy vs a fixed threshold on identical batches/agents
  (15/15 runs acted); plan executability **129/129** episodes with
  monotone initial-state value growth; trace fidelity **600/600** with
  exact numeric quotes.
* **Defects found and corrected (not deferred) in the dormant modules:**
  (1) `hierarchical_planning` could not select options at all — the option
  library returned dict projections that `plan()`/`select_action()`
  type-checked away, so every plan shipped empty and every action fell back
  to `default_action`; option eviction never fired (builtin prefix test
  matched every ID) and `Option.get_action` never consulted its own
  `"default"` policy. (2) `multi_agent_coordination` returned a silent
  benign verdict below quorum (fail-open — now an explicit `abstained`
  consensus flag), silently dropped duck-typed dict votes from consensus
  (now coerced and counted), and reported `individual_results` from a
  nonexistent attribute (always empty — now the real per-agent results).
  (3) `chain_of_thought` traces classified against hardcoded 0.7/0.4 bands
  regardless of the issuing pipeline's boundary, so a trace could state
  "POTENTIAL ANOMALY" while the verdict said anomaly (now: traces classify
  at the caller's `anomaly_threshold`); the self-consistency strategy
  returned the normalized vote *token* ("anomaly") as the chain conclusion,
  stripping the human-readable determination (now: winning path's full
  conclusion, token kept in metadata). (4) `reflexion`'s
  `execute_with_reflection` recomputed an identical decision every
  iteration (reflection never fed back — now critiques are answered with
  real evidence computed from the task data); its threshold-recommendation
  rule (`fn > 2·fp` → jump ≥ 0.1) was **measured harmful on real labels**
  (paired Δ −0.071; WBC balanced accuracy 0.98 → 0.50) and replaced with an
  evidence-grounded balanced-accuracy sweep over recorded outcomes with
  minimum-evidence and hysteresis guards (measured: +0.079, and
  well-calibrated points are left untouched).
* **Honesty semantics throughout:** below-quorum coordination abstains
  explicitly (never "benign"); agent fit/score failures surface and the
  orchestrator refuses below quorum; an unexecutable plan raises instead of
  bypassing the planner; an unfaithful trace raises instead of shipping.
* **Tests:** `tests/cognitive/test_orchestration_behavioral.py` — 45
  multi-seed behavioral tests pinning every defect fix and every
  orchestration contract (planner-driven sequencing, TD value growth,
  quorum abstention, reflexion adaptation with paired improvement, trace
  fidelity incl. post-adaptation issue-time fidelity, ethical-gate
  enforcement, engine wiring). Full `tests/cognitive/` suite: 577 passed.
* **Docs:** DORMANCY_LEDGER rows 10/11 updated to *revived (orchestration
  tier)* with the measured numbers; `chain_of_hindsight` and
  `plasticity_engine` split into rows 10b/11b (still retained — no honest
  harness yet); capability matrix "Multi-agent" row moved to
  **Shipped + measured** with the build-out item (B) closed.
### Added — multi-model substrate: provider-reported usage accounting + LLM model registry (2026-06-11)

First slice of the multi-agent/multi-model build-out
(`docs/capability_vs_vision_matrix.md` §4 item B): before orchestrating
across Mercury's ten LLM providers, calls must be *costed* and models must
be *selectable*. Both pieces are provider-truth-only — no client-side
token estimates, no hard-coded market facts.

* **Usage accounting**
  (`omni_mercury_engine.models.foundation.llm_usage`): `LLMUsage` (one
  immutable record per successful generation, counts exactly as the
  provider's response reported them) and `UsageLedger` (thread-safe;
  exact lifetime totals via running counters; bounded recent-history
  ring that never bends the totals). Calls whose provider reports no
  usage are recorded as **unmetered** (`reported=False`) so unmeasured
  spend is visible rather than silently absent.
* **Adapter wiring**: every shipped adapter's success path now parses
  its provider's usage block into `last_usage` and an optionally
  attached ledger (`BaseLLMAdapter.attach_usage_ledger`) — OpenAI +
  xAI/DeepSeek/Cursor (`usage.{prompt,completion,total}_tokens`),
  Anthropic (`usage.{input,output}_tokens`), Gemini (`usageMetadata`),
  Cohere v2 (`usage.tokens`), Ollama (`prompt_eval_count`/`eval_count`),
  HuggingFace Inference (no usage block → unmetered record). Failed
  calls book nothing — **every** adapter extracts the response content
  *before* recording usage, so a 200 carrying a usage block but
  unextractable content books nothing (uniform across all ten adapters,
  not only the OpenAI-style ones).
  `FallbackLLMChain(usage_ledger=...)` threads one ledger through every
  adapter in the chain and exposes `last_usage`.
* **LLM model registry** (`omni_mercury_engine.models.llm_registry`,
  importable without torch): `PROVIDER_CATALOG` — code-grounded facts
  about the ten shipped adapters (wire format, key env var, locality,
  explicit-base_url requirement, usage-reporting), drift-gated in CI
  against `IMPLEMENTED_LLM_PROVIDERS`; `LLMModelSpec` /
  `LLMModelRegistry` — operator-declared model facts with mandatory
  provenance (a spec carrying prices must carry `pricing_as_of`) and
  deterministic capability/context/budget selection (`select` /
  `select_one`; unpriced specs are excluded from budget queries rather
  than assumed free; unknown capabilities raise rather than silently
  matching nothing).
* **Locks:** `tests/models/test_llm_usage_ledger.py` (validation,
  aggregation, bounded-ring-vs-exact-totals, concurrent exactness),
  `tests/test_llm_usage_capture.py` (per-wire-format parsing against
  canned payloads, unmetered HF route, failed-call no-booking, chain
  threading), `tests/models/test_llm_registry.py` (spec/selection
  contracts + the provider-catalog drift gate).

### Security — owner-governed risk posture: enumerated CVE ledger, shared HD master seed, signed cache entries (2026-06-10)

Closes three standing gaps where risk was silently accepted or a promised
control was never wired, replacing each with an enforced mechanism.

* **Deployment-image CVE gate: blanket waiver retired.** The blocking
  Trivy gates (`ci.yml` Docker job, `security.yml` table scan) now run
  `ignore-unfixed: false` with the new repo-root `.trivyignore` — an
  enumerated, expiring (90-day `exp:`), per-CVE acceptance ledger. A new
  unfixed CRITICAL/HIGH finding, or an expired entry, now **fails the
  gate** instead of passing invisibly. Ledger as measured by the
  blocking gates' own built-image scan (trivy 0.70.0 via
  `trivy-action@v0.36.0`, 2026-06-10): 13 no-upstream-fix OS CVEs
  (5 Critical, 8 High) — nine first enumerated from the
  `python:3.13-slim-bookworm` base, of which seven (ncurses
  CVE-2025-69720 + six perl-base CVEs) had been present but undisclosed
  under the old blanket waiver, plus four surfaced only by the enforced
  built-image gate (three libexpat1 CVEs + mesa CVE-2026-40393 via the
  OpenCV `libgl1-mesa-glx` runtime dependency). Resolved entries
  removed from `SECURITY.md`: the fixed pip family (pip ≥ 26.1 floor)
  and five formerly-listed findings no longer present at gated
  severities.
* **`AMA_MASTER_SEED` — deterministic fleet-wide HD key derivation.**
  `api/auth.py::get_auth_key_manager()` now sources the AMA HD master
  seed from the `AMA_MASTER_SEED` env var (hex, ≥ 32 decoded bytes;
  malformed values raise instead of degrading to an ephemeral seed).
  With it set, HD-derived keys — including the production JWT signing
  key — are identical across workers, replicas, and restarts. Seedless
  production derivation still works but now logs an explicit warning
  naming the per-process-key hazard. A whitespace-only value is
  malformed (raises), never a silent unset: `bytes.fromhex` ignores
  ASCII whitespace, so a trailing newline on a valid seed is harmless
  while a garbage value still fails loudly. Locked by
  `tests/security/test_jwt_auth.py::TestAMAMasterSeed` (8 tests).
* **`MERCURY_CACHE_SECRET` wired — Redis cache entry signing.** The
  Helm-provisioned secret, previously read by no code path, is now
  consumed by `integrations/stubs/cache.py::RedisCache`: entries are
  wrapped in an HMAC-SHA256-signed envelope on `set` and verified on
  `get`; unsigned, tampered, or foreign-keyed entries raise the new
  `CacheIntegrityError` (loud, never a silent miss). Plain-JSON
  behaviour is byte-identical when the secret is unset. `_sign()` guards
  its signing-active contract with an explicit `CacheIntegrityError`
  instead of an `assert` (survives `python -O`); envelope detection
  requires the exact three-key shape `_serialize` writes, so user data
  that merely contains the marker key cannot be misclassified. Locked by
  `tests/integrations/test_cache_hmac.py::TestRedisCacheHMAC` (10 tests).
* **`.env.example` reconciled to the real env surface.** Dead toggles
  removed (`OMNI_ML_ENABLED`, `OMNI_QUANTUM_ENABLED`,
  `OMNI_EXPERIMENTAL_FEATURES`, `OMNI_CACHE_DIR`); `AMA_MASTER_SEED`,
  `MERCURY_ENV`, `API_KEY_HASH_SALT`, `MERCURY_CACHE_SECRET`, and
  `MERCURY_CORS_ORIGINS` documented; the stale "conditional PQC gate"
  comment and a nonexistent `validate_env` command corrected.

### Decision / Abstention / Response layer — closes identify→interpret→decide→deter with a calibration-grounded "don't-know" gate (2026-06-09)

Converts the calibrated detection certificate into a **closed autonomous loop**
with an explicit, principled abstention gate — the first time the engine‑wide
three‑state contract is wired into the *detection* path rather than only the
verifier/governance paths.

* **New `omni_mercury_engine.decision` package (pure Python, deterministic).**
  `DecisionAbstentionResponder.decide(result)` maps a `detect_with_fusion`
  result onto the existing `ThreeState` invariant
  (`verifiers/three_state.py`) and an operational `Disposition`:
  * the conformal certificate is **authoritative** — singleton `{1}`/`{0}` →
    `GROUNDED` (`ACT`/`CLEAR`) with the coverage level as the honest
    confidence; `{0,1}` → `UNAVAILABLE` (a *resolvable* don't‑know, `DEFER`);
    `{}` → `UNDECIDABLE` (an atypical point no class explains; fail‑closed
    `HOLD`);
  * with no certificate, a probability inside the threshold's indecision band
    is `UNAVAILABLE`, and a decision outside it is grounded but flagged
    `calibrated=False` (no coverage guarantee);
  * **demotion overlays** weaken a grounded verdict to `DEFER` on
    neuro‑symbolic disagreement (`symbolic_consistency.satisfaction` below a
    floor) or severe distribution drift — overlays only ever move *toward*
    abstention;
  * an explicit ethical‑gate refusal (`gosnn_metadata.ethical_gate_passed is
    False`) forces a fail‑closed `HOLD`, over any score.
* **Bounded, non‑destructive response layer (`decision/response.py`).** Every
  `ResponsePlan` is advisory/notifying — `MONITOR` / `ALERT` /
  `RECOMMEND_MITIGATION` / `ESCALATE_TO_HUMAN` / `REQUEST_INPUT` / `HOLD` —
  with severity‑banded urgency, human‑in‑the‑loop above a bar, and a
  fail‑closed hold that always requires a human. A test invariant asserts the
  catalogue contains no destructive verbs (Civilization‑First made concrete).
* **Auditable `DecisionRecord`.** A frozen, JSON‑safe record carrying the
  grounded label or honest abstention, the calibrated confidence, the bounded
  response, ordered `reasons`/`caveats`, and the full evidence + active policy
  as `signals` provenance; `explain()` renders a one‑paragraph operator
  account. The operational sibling of the governance layer's `GovernanceScalar`.
* **Closed into existing channels (`decision/bridge.py`).** `to_agent_action`
  adapts a record to the autonomy loop's existing `AgentAction` vocabulary;
  `to_cap_alert` emits a standards‑based CAP 1.2 alert (via the existing
  `alerting/cap_generator.py`) for any notifying decision — no new silo.
* **"Verify" step — audit ledger + closed loop (`decision/ledger.py`,
  `decision/loop.py`).** `DecisionLedger` is an append‑only, JSON‑serialisable,
  **bounded** (ring‑buffer) trail of every decision with a `summary()`
  (per‑state / per‑disposition / per‑response counts + abstention and
  calibrated rates). The `summary()` is now **O(1)** — aggregate counts are
  maintained incrementally (and decremented on ring‑buffer eviction) instead of
  re‑scanning the trail, byte‑identical output (measured: 14.6 µs vs 75.9 ms at
  10⁵ records, a ~5000× reduction). The ledger is **thread‑safe** (a lock guards
  every mutation/read; 8×1000 concurrent records lose no count) and
  **persistable** (`to_json`/`from_json`, `save`/`load`, with a new
  `DecisionRecord.from_dict`/`ResponsePlan.from_dict` inverse of `to_dict` so a
  serialised trail reloads). `DecisionLoop` runs decide → deter → **verify** over
  a stream, recording each pass and fanning it out to an optional `FeedbackSink`
  (the seam for outcomes to flow back to calibration / learning / a human
  queue) — keeping the responder's `decide()` a pure function.
* **Integrated onto merged #278.** The branch is rebased on the Phase‑2 governed
  fusion substrate. #278's one new `detect_with_fusion` key,
  `result["info_geometry_certificate"]`, was evaluated and is **deliberately a
  no‑op** in the gate (measured): it certifies a *component's* price level‑set,
  "NOT the fused/gated verdict", and is degenerate in the single‑sample serve
  path (its batch‑adaptive threshold collapses to `price == threshold`). The
  gate keeps consuming the authoritative conformal certificate; a new
  torch‑gated **empirical‑coverage test** proves that guarantee survives the
  projection — **zero gate‑vs‑certificate contradictions** and ~94 % coverage at
  a 90 % target on a held‑out split, with abstention lifting selective accuracy
  to ~100 % vs ~99 % raw. Defensive `.get()` reads; the layer is an exact no‑op
  on any absent or unmodelled signal.
* **Opt‑in engine wiring.** `OmniMercuryEngine.enable_decision_layer()` attaches
  a `result["decision"]` section to every `detect_with_fusion` result; an exact
  no‑op until enabled (mirrors `enable_drift_detection` / conformal). Pass
  `enable_decision_layer(ledger=DecisionLedger())` to record the audit trail
  (the serve path stays stateless otherwise). Most informative after
  `calibrate_fusion_conformal()`, which turns a thresholded guess into a
  coverage‑guaranteed decision.
* **Tests:** 123 pure‑Python tier tests (`tests/decision/`) + 11 torch‑gated
  tests (5 engine‑wiring in `tests/test_decision_layer_wiring.py` + 6
  empirical‑coverage in `tests/test_decision_coverage_empirical.py`) at **100%
  statement+branch coverage** of the package. Beyond the example‑based suites,
  **Hypothesis property tests** (`tests/decision/test_properties.py`) pin the
  gate's laws over the whole input space — fail‑closed dominance (an ethical
  block beats any score/certificate), monotonicity (more uncertainty never
  yields less abstention), and structural soundness + `to_dict`/`from_dict`/JSON
  round‑trips. The gate, fail‑closed invariants, determinism, the
  non‑destructive response contract, the O(1)/thread‑safe/persistable ledger,
  and the #278 no‑op contract are all pinned. Runnable demo:
  `examples/decision_abstention_response_demo.py`. A code‑grounded
  capability‑vs‑vision audit ships at `docs/capability_vs_vision_matrix.md`.

### Dependencies — corrected numpy floor to match the real support contract (2026-06-09)

The declared core dependency was `numpy>=1.24.0`, which was inaccurate on two
independent axes and is now corrected to `numpy>=2.4.0`:

* **Python-matrix reality.** The project is `requires-python>=3.11` and CI
  builds/tests on Python 3.11/3.12/3.13/3.14, but numpy `1.24` publishes no wheels
  for Python 3.12 (first added in `1.26`) or 3.13 (first added in `2.1`) — so
  the old `1.24.0` floor was not installable across the very matrix the
  project claims to support.
* **Strict-mypy type contract.** The codebase annotates ~300 sites with bare
  `np.ndarray`, which is only valid under `disallow_any_generics` once numpy's
  `ndarray` carries PEP-696 type-parameter defaults. Combined with the
  fancy-index shape typing in `core/conformal_prediction.py`, the strict gate
  emits 232 errors on numpy 2.2.6 and 5 on 2.3.5; **2.4.0 is the first release
  that type-checks cleanly** across all three mypy lanes. Bumping the floor
  (rather than rewriting ~300 annotations to `np.ndarray[Any, Any]`, which
  would erase type precision and break the 145 `isinstance`/constructor sites
  that must stay unsubscripted) keeps the contract honest without weakening it.

A new **"Run MyPy at the numpy floor"** step in the *Type Checking* CI job
pins `numpy==2.4.0` and re-runs all three mypy lanes, so the declared minimum
can never again silently drift from what actually type-checks. Updated in
`pyproject.toml`, `docs/INSTALLATION.md`, and `.github/workflows/ci.yml`.

### Research — executed the decorrelated-stream fusion protocol (Item D); SHIP rejected, runtime byte-unchanged (2026-06-09)

The last logged-but-untried fusion lever in `research/governed_fusion/FINDINGS.md`
— *decorrelation* (the hypothesis that fusion only dilutes because the three
committed streams are redundant, and a genuinely orthogonal net-new stream would
let a learned stacker beat best-single) — was run end-to-end under its
pre-registered, kill-criteria'd protocol rather than left as a promise. **No
runtime path changed; everything here is research-only and default-off.**

* **New `research/governed_fusion/measure_decorrelation.py`** executes all four
  protocol steps off the existing deterministic score cache: (1) redundancy
  diagnosis via mean pairwise Spearman `|ρ̄|`; (2) two net-new detectors — a
  learned multi-lag **autoregressive innovation** stream (the decorrelated
  candidate) and a **k-NN local-density** stream (pre-declared sensitivity check);
  (3) per-event 50/50 logistic stacking fit on calibration scores only; (4) a
  10,000-resample paired bootstrap against best-single with a deterministic seed.
* **Result (live headline, 20 stackable events): SHIP rejected.** Measured
  `|ρ̄| = 0.462` (moderate, **refuting** the pre-registered `≳ 0.6` prediction).
  The temporal stream is genuinely decorrelated (`|ρ| = 0.316`, driving the
  4-stream `|ρ̄|` down to `0.389`), yet stack-4 AUROC `0.8705` still **trails**
  best-single `0.8760` — paired `Δ = −0.0055`, 95% CI `[−0.050, +0.034]`, with a
  hurricane per-domain regression of `−0.125`. The reconstructed group is a clean
  KILL (`Δ = +0.0027`, CI upper `< +0.01`); the 27-event pooled power check stays
  negative (`Δ = −0.0034`). Decorrelation was necessary-by-hypothesis but
  **insufficient**: the lever is closed as a measured non-improvement.
* **No regression.** Both detectors stay in `research/`, tested and unused by
  runtime; the default `detect()` path and baseline ensemble are byte-unchanged.
  Committed artifact `research/governed_fusion/results/decorrelation_results.json`;
  offline guards in `tests/research/test_decorrelation_protocol.py` (7 tests, no
  network). FINDINGS.md and RUN.md record the verdict with exact numbers.

### Equations — golden-ratio fibring runtime profile + correlation-aware decorrelation (2026-06-02)

Improves the *math* of the runtime equation blend by harmonising it with
Mercury's canonical TIER-4.2 fusion (`core/fibring_fusion.py`) instead of an
arbitrary fixed split.

* **New `phi_fibring_v1` runtime profile** — blends the calibrated neural score
  with the frozen OAE R/H/O equation signal using the **golden-ratio split**
  `w_neural = φ/(1+φ) ≈ 0.618`, `w_equation = 1/(1+φ) ≈ 0.382` (the same
  `_phi_base()` ratio the production fibring fusion uses), rather than the
  source branch's ungrounded `0.70/0.30`.
* **Correlation-aware decorrelation** — when the equation signal merely echoes
  the neural score (`|Pearson r| ≥ REDUNDANCY_THRESHOLD = 0.85`, imported from
  `fibring_fusion` as the single source of truth), the lower-variance (less
  informative) stream's weight is shrunk by `1/(1+|r|)` and the pair is
  renormalised — so a redundant equation channel cannot double-count. The
  blend remains a convex combination of two `[0,1]` signals (output bounded).
  `score_runtime_equation_profile` now reports `neural_weight`,
  `equation_weight`, `correlation`, and `decorrelation_applied` in its metadata.
* `baseline_original_v1` / `quiet_horizon_v1` stay **frozen** (fixed `0.70/0.30`,
  no decorrelation), honouring the optimizer's freeze-and-add discipline.

Tests: `tests/core/test_equation_profiles.py` gains golden-ratio-split,
decorrelation-fires-on-redundant-signal, and distinct-from-baseline cases.
`black`/`ruff`/`flake8` clean; `mypy --strict` clean.

### Security — CodeQL/Trivy: real system-pip upgrade + scipy fetchers false-positive (2026-06-02)

* **pip CVEs (CVE-2025-8869 / CVE-2026-1703 / CVE-2026-6357)** — the runtime
  stage's `python -m pip install --upgrade pip>=26.1` was a no-op against the
  vulnerable pip Trivy flags: `ENV PATH` puts `/opt/venv/bin` first, so it
  upgraded the (already-patched) venv pip and left the base image's *system*
  pip at `/usr/local` untouched. Fixed to target `/usr/local/bin/python`
  explicitly and to drop the bundled `ensurepip` wheels that re-seed a stale
  `pip-*.dist-info`.
* **scipy "JWT token" false-positive** — `scipy/datasets/_fetchers.py` embeds
  the pooch dataset *registry* (SHA-256 hashes + URLs), which Trivy's secret
  scanner misclassifies as a JWT. Added to the existing Trivy `skip-files`
  allowlist (the repo's established mechanism) in both scan steps, since it is
  an upstream-dependency artifact, not a Mercury secret.

### Equations — universal equation optimizer + opt-in runtime equation profiles (2026-06-02)

Marshals the previously-unintegrated **universal equation optimizer** work
(developed on an earlier working branch and never opened as a PR) into a
single branch, adapted to the current engine surface and re-verified
end-to-end. The optimizer is reproducible and *safety-gated by
construction* — it does not weaken any existing gate, and when no candidate beats
the proven baseline under the hard constraints, the frozen baseline wins.

* **`tools/equation_optimizer.py`** — inventories the equation surfaces from
  `docs/MATH_SPEC.md`, freezes the original equations as an immutable baseline
  (`preserve_original_equations=True`), searches a constrained universal
  candidate family, and promotes a winner only when it clears *hard* gates
  (ethical-compliance invariant, output range `[0, 1]`, contraction `α ≤ 0.999`,
  Lyapunov `λ ≥ 1e-6`). Emits versioned artifacts + rollback / continuous-
  revalidation metadata. Registered in the `python -m tools` dispatcher
  (entry-point `_cli`).
* **`scripts/run_equation_research_protocol.py`** + `configs/equation_research_protocol.yaml`
  — a governed inventory → freeze → search → gate → decision-ledger runner.
* **`scripts/compare_runtime_equation_profiles.py`** — compares the frozen
  baseline against a candidate runtime profile on real data (AUC / ECE via
  `core.calibration.compute_ece`, neuro-symbolic satisfaction, latency, per-domain
  η), asserting the hard gates are preserved.
* **`omni_mercury_engine.core.equation_profiles`** + engine wiring — an **opt-in**
  runtime serve path. `OmniMercuryEngine(..., equation_profile=...)` (or a per-call
  override on `score_fusion` / `detect_with_fusion` / `detect_with_fusion_calibrated`)
  blends the calibrated neural probability with the golden-ratio R/H/O equation
  signal and surfaces the blend metadata under `result["equation_profile"]`.
  **`None` (default) is an exact no-op** — verified that the profile-less path
  returns the calibrated array unchanged — so existing serve/benchmark behaviour
  is byte-for-byte preserved. The per-domain η estimate (`_domain_eta`) is aligned
  to the canonical OAE `DOMAIN_THRESHOLDS` convention (medical 0.93, humanitarian
  0.95, infrastructure 0.995, default 0.96), correcting a stale source-branch table
  that carried invented values and keys (`security`/`humanitarian`) silently zeroed
  by `sanitize_domain`'s disjoint `EnvironmentDomain` vocabulary.

Tests: `tests/tools/test_equation_optimizer.py` (pipeline artifacts + CLI smoke),
`tests/core/test_equation_profiles.py` (bounded/distinct profiles, `None`
pass-through, unknown-profile rejection, channel→R/H/O mapping),
`tests/scripts/test_run_equation_research_protocol.py`,
`tests/scripts/test_compare_runtime_equation_profiles.py`. Regression-verified
against `tests/test_omni_mercury_engine.py`, `tests/test_fusion_explainability.py`,
`tests/test_fusion_symbolic_cotraining.py`, `tests/test_neurosymbolic_integration.py`,
and `tests/tools/test_dispatcher.py`. `black`/`ruff`/`flake8` clean; `mypy --strict`
clean on all changed source files (the source branch's `mypy` gaps in
`run_equation_research_protocol.py` were fixed in passing).

### Docs — removed dangling `make checkpoint` reference (2026-06-02)

`scripts/train_default_fusion.py`'s module docstring referenced a `make checkpoint`
target that does not exist in-tree (the `Makefile` lived only on the rejected
hidden_dim=128 `fusion-capacity-on-merits` branch). Corrected to point at the
script directly so the documented invocation is real.

### Neuro-symbolic — `LogicTensorNetwork` re-wired to the canonical co-trained module (2026-06-02)

Reverses PR #269's *retirement* of `LogicTensorNetwork` in favour of **wiring**
it (the mission's "do not retire — implement its predict to call the canonical
co-trained symbolic module"). The original class was a never-trained `nn.Module`
whose random-init noise fed fusion as if a neural confidence; rather than delete
the surface, it is rebuilt as a thin head over the canonical co-trained module:

* **`SymbolicConstraintModule.predict(detector_scores) -> (B,)`** — new public
  per-sample inference API exposing the module's fuzzy-logic `Consensus`
  predicate (the *same* component co-trained with real gradients inside
  `fit_fusion`). Fails closed on a shape/width mismatch.
* **`models/neurosymbolic.py::LogicTensorNetwork`** is re-introduced as an
  SCM-backed wrapper: `predict()` routes detector scores through
  `SymbolicConstraintModule.predict`. **No random init** — an untrained module's
  zero-initialised parameters yield a deterministic uniform-weight consensus; a
  co-trained module (e.g. restored from a fusion checkpoint via `module=`)
  applies its learned detector reliabilities. `NeurosymbolicEngine.ltn` is built
  when torch is importable (honestly reported by `get_statistics`:
  `ltn_available`, `ltn_backend="symbolic_constraint_module"`).
* The raw-feature `neural_inference` is **kept** as an honestly-labelled
  deterministic dispersion heuristic (it is a per-feature statistic, *not* a
  trained signal, and says so) — distinct from the trained-capable detector
  consensus now reachable via `ltn.predict`.

Tests: `tests/ml/test_symbolic_constraint.py::TestPredictInference` (6 tests:
per-sample shape/range, matches the consensus predicate, monotone in scores,
zero-detector trivial case, fail-closed shape checks) and
`tests/test_neurosymbolic_integration.py::{test_ltn_is_wired_to_canonical_symbolic_module,
test_ltn_predict_delegates_to_symbolic_consensus}` (LTN exists, is SCM-backed,
delegates exactly, deterministic, monotone). The prior `*_is_retired` assertions
are flipped to wired assertions. `black`/`ruff`/`flake8` clean; `mypy --strict`
clean on both changed source files.

### MLOps — fusion AUC/F1 + conformal-coverage CI regression gate (2026-06-02)

Finishes the WS5/WS6 "MLOps train→eval→register CI regression gates" item
deferred by PR #269. The fusion train→eval→register pipeline + AUC/ECE sweep
artifacts existed, but **no CI gate pinned the fusion AUC/F1 or the conformal
coverage floor**, so a fusion-stack regression could merge silently.

* **`benchmarks/fusion_regression_guard.py`** — deterministically trains the
  real fusion path on a fixed seeded synthetic corpus (`symbolic_weight=0.0`,
  `seed=20260526`), evaluates held-out AUC + F1, and measures empirical conformal
  coverage from a disjoint calibration split. `--update` re-pins the committed
  baseline (`benchmarks/fusion_capacity/fusion_gate_baseline.json`) **and** emits
  a timestamped metrics artifact under `artifacts/fusion/<ts>/metrics.json`;
  `--check` exits non-zero if AUC/F1 fall below `baseline − margin` or empirical
  coverage drops below `target − 0.05`. Mirrors the existing
  `anomaly_regression_guard.py` pattern.
* **`.github/workflows/fusion-regression.yml`** — runs the gate on every PR to
  `main`/`develop` (+ weekly + dispatch): builds AMA Cryptography (v3.2.0),
  installs `.[ml]`, runs the fast gate-logic unit tests, then the real
  `--check`.
* **Committed measurement.** First baseline (`seed=20260526`, 8 epochs):
  **AUC=1.0000, F1=0.9851, empirical coverage=0.9521** (target 0.90); the gate
  `--check` re-trains and passes against it.

**Why train, not load the shipped checkpoint.** A diagnostic found the
checkpoint round-trip drifts per-sample probabilities (max |Δ|≈0.76; AUC
survives at Δ≈0.002 but absolute probabilities — hence F1@0.5 / coverage — do
not), and the shipped `default_fusion.pt` underperforms in-distribution
(AUC≈0.70, base detectors auto-fit on inference because their state isn't
persisted). The gate therefore trains in-process for a bit-stable measurement
of *achievable* performance; the round-trip drift is a tracked follow-up (see
ROADMAP). Tests: `tests/benchmarks/test_fusion_regression_guard.py` (5 fast
gate-logic tests: floors, pass, AUC regression, coverage collapse, missing
baseline).

### Explainability — IntegratedGradients + faithfulness on the fusion serve path (2026-06-02)

Finishes the WS5 "explanations on the serve path" item deferred by PR #269: the
validated `IntegratedGradientsExplainer` + `FaithfulnessEvaluator`
(`cognitive/explainability.py`, DORMANCY_LEDGER #3) are now wired into the
detection path instead of living only in the benchmark harness.

* `OmniMercuryEngine.detect_with_fusion(..., explain=False)` gained an opt-in
  `explain` flag. When `True`, it attaches `result["explanation"]` — an IG
  attribution of **the same `score_fusion` calibrated probability the result
  reports** (so the explanation explains the served decision, not a proxy) plus
  its faithfulness scores (comprehensiveness / sufficiency / monotonicity). The
  attribution is computed by the new private helper `_explain_fusion_decision`.
* Opt-in (default `False`) because IG over the full fusion stack costs
  finite-difference forward passes per feature × interpolation step; steps are a
  module constant (`_EXPLAIN_IG_STEPS = 32`, matching the harness) so tests/CI
  can turn it down.

**Faithfulness non-regression gate (measurement).**
`tests/benchmarks/test_explanation_fidelity.py::test_faithfulness_non_regression_comprehensiveness_and_recovery`
pins, across seeds, that IG comprehensiveness exceeds a random-attribution
baseline by > 0.01 **and** recovery@k of the genuinely-informative features
exceeds 2× chance — the same contract as the harness verdict, so a future
attribution regression fails CI. Serve-path wiring is locked by
`tests/test_fusion_explainability.py` (explanation absent by default; present,
well-formed, JSON-serialisable, and matching the served probability when
requested). Measured this run: comprehensiveness(IG) ≈ 0.40 vs random ≈ 0.22,
recovery@k ≈ 0.68 (chance 0.30).

### Detectors — concrete offline `StatisticalVLMDetector` un-retires the VLM surface (2026-06-02)

Finishes the VLM half of ROADMAP **#3**: the public `detectors.vlm` path now
ships a *concrete, instantiable, network-free* detector instead of only an
abstract base. `detectors/vlm/statistical_vlm.py::StatisticalVLMDetector`
implements all five `BaseVLMDetector` contract methods for real, with no
`transformers` import and no model download:

* `extract_features` returns a deterministic `[N, 8]` per-frame salience
  statistics tensor (luma mean/std, H/V gradient energy, p10/p90, bright/dark
  fractions); `detect` maps those to a calibrated `[0, 1]` anomaly score and
  runs the genuine VQA control flow (`_create_prompt` → score →
  `_parse_response`).
* The base classes stay genuine `@abstractmethod` ABCs (the existing
  not-instantiable / `__abstractmethods__` contract tests are unchanged) — the
  surrogate is an *added* concrete subclass, not a weakening of the base.

**Surrogate, with a remediation plan (not a retirement).** The production VLM
backends (AnyAnomaly / LAVAD / BLIP) require `transformers` + a HuggingFace
download of a multi-billion-parameter LVLM, which is an external dependency not
available in every environment. Rather than retire the surface or leave an
un-instantiable ABC, the statistical detector preserves the public contract
offline; the module docstring documents the exact steps to graduate to a
trained LVLM (`pip install '...[vlm]'`, revision-pinned `BLIPVLMDetector` /
`get_lvlm_backend`). Coverage: `tests/test_vlm_detectors.py::TestStatisticalVLMDetector`
(11 tests — offline instantiation, full-contract shapes, salience ordering
[textured > flat], determinism, channel-last video, uint8 normalisation,
prompt/response round-trip).

### Security / σ_Immutable Wave C — narrative voice + federation dual-gate, GOSNN bidirectional coupling (2026-06-02)

Closes ROADMAP v1.7.x deferred items **#1** (σ_Immutable Wave C —
narrative voice + federation) and **#5** (GOSNN coupling wired into the
FL aggregator training loop).

**Shared σ_Immutable vector builder.** The canonical 256-D input-vector
builder — previously a copy at the engine boundary, a copy in
`CognitiveOrchestrator._build_sigma_immutable_vector`, and a copy in
`NeuroSymbolicHub._build_sigma_immutable_vector` — is promoted to a single
calibrated helper,
`security.sigma_immutable_gate.build_sigma_immutable_vector(benevolence_score,
severity, anomaly_prob)`, plus a shared `enforce_dual_ethical_gate(...)`
primitive that runs `BenevolenceScorer.enforce` →
`project_benevolence_to_sigma_band` → `SigmaImmutableGate.enforce`.  The
engine and orchestrator now delegate to the helper (the engine's
benevolence-only vector is reproduced byte-for-byte with
`severity == anomaly_prob == 0`); the hub uses the shared helper for its
base vector and keeps its one load-bearing difference (the richer
per-sample neural / symbolic / fused overlay).  One source of truth, no
drift between five boundaries.

**Three previously-ungated public surfaces now carry the dual hard gate**
(both gates fail closed; gates constructed eagerly in each `__init__`;
every caller-supplied domain hint routed through `sanitize_domain`):

* `narrative/voice.py::{speak, process_detection, alert}` — `alert` gains
  an optional `domain=` kwarg (defaulting through `sanitize_domain`);
  `process_detection` sources its σ_Immutable severity / anomaly signal
  from the detection being narrated.
* `federation/aggregator.py::{submit, aggregate}` — per-submission gate on
  `submit`, round-level gate on `aggregate`.
* `federated_learning/server.py::_execute_round` — round-level gate placed
  **outside** the per-client `try/except` so a benevolence- or
  σ_Immutable-violation fails the whole round closed instead of being
  swallowed as one client's error.

The synthetic-vector projection is calibrated against the *real* trained
gate (not asserted): legitimate voice / federation calls score ≈ 1.0 and
pass the 0.93 threshold, while sub-floor benevolence scores 0.0 and fails
— verified in-tree, see below.

**GOSNN coupling wired into the FL training loop.**
`FederatedServer._execute_round` now routes every round's weights through
`GOSNNCoupling{Server,Client}` (`publish → ingest → aggregate → receive`)
with SHA3-256 + shape + round integrity, closing the one-way
(server → client) gap.  Each client's absolute post-step weights
(`global + model_update`) are published with a digest, ingested under
shape / digest / round verification, aggregated, and broadcast back to
every contributing client (digest re-verified on `receive`).  Unit-LR
FedAvg flows through the coupling's own digested weighted mean
(mathematically identical to `global + Σ wᵢ·model_updateᵢ`); FedAdam /
SCAFFOLD / secure-aggregation / non-unit-LR FedAvg keep their
strategy-specific numerics and are installed + broadcast through a new
`GOSNNCouplingServer.install_global_state`, preserving the privacy-engine
and secure-aggregation branches and the `LocalUpdate` / `RoundResult`
field contracts.  A `GOSNNCouplingError` (digest / shape / round mismatch)
fails the round closed.

Coverage:

* `tests/ethical/test_hard_enforcement.py` — three new boundary classes
  (`TestNarrativeVoiceBoundary`, `TestFederatedAggregatorBoundary`,
  `TestFederatedServerRoundBoundary`), each pinning the four-way contract
  (legitimate pass on the real gate, `check="benevolence"`,
  `check="sigma_immutable"`, `check="gosnn_unavailable"`), plus a test that
  the FL round gate runs outside the per-client `try/except`.
* `tests/federated/test_no_silent_failure.py` —
  `test_federated_server_round_drives_gosnn_digested_fedavg_path` drives a
  live `FederatedServer` round end-to-end through the coupling and asserts
  the new global equals the sample-weighted FedAvg mean of the clients'
  absolute post-step weights; plus `install_global_state` round-trip /
  reshape-fails-closed coverage.

### Agentic — RL policy in the loop + real task execution, fail-closed (2026-06-02)

Truthing-up the autonomous-agent surfaces flagged in the v1.7 audit.

**`agentic/agentic_autonomy.py` — the RL policy now actually steers.**
`autonomous_detect` previously hardcoded `_decide_action → "flag_anomaly"`
and never consulted the Q-table, leaving `select_action_with_policy`,
experience replay, and reward shaping as dead machinery.  It now derives an
observation state, lets the **epsilon-greedy Q-policy** choose the action
type (explore vs. exploit), materialises that action, and learns from it.
The selection-time state features are carried on the `AgentAction` so the
TD update writes the exact Q-key the policy read (no off-by-one drift from
`action_history` mutating between selection and learning).  `_decide_action`
now produces action-type-specific parameters and rationale.

**`agentic/mercury_a_agent.py` — `_execute_task` does real work.**
The prior implementation was a no-op that always returned `completed` with
`output=f"Executed: {description}"`, forcing `success_rate` to 1.0.  It now:

* runs a **fail-closed ethical gate** first (scores the task via
  `BenevolenceScorer.enforce`, `sanitize_domain` on the domain hint), placed
  *before* the execution `try/except` so a harmful task halts the plan
  instead of being recorded as a benign result — mirroring the OODA
  reference (`cognitive/autonomous_agent.py`);
* dispatches a task that binds a registered tool (`task.metadata['tool']`)
  to that tool for real, with genuine success / failure (a raising tool →
  `status="failed"` with the error captured; the analysed batch is injected
  as `data=` for tools that accept it);
* marks a task with no bound tool as an honest `status="skipped"` — never a
  fabricated `completed`.  `_execute_plan` now measures `success_rate` over
  *executed* tasks, so a pure-reasoning plan reports `0.0`, not `1.0`.

Coverage (both files previously had **zero** behavioural tests for these
paths):

* `tests/test_agentic_autonomy.py::TestReinforcementLearningPolicy` — Q-table
  writes (`new_q == lr·reward`), reward-shaping contract, epsilon-greedy
  exploit-best-Q vs. explore-all-actions, experience-replay convergence
  logging, exploration decay, selection/learn Q-key consistency.
* `tests/test_mercury_a_agent.py` (new, 12 tests) — real tool dispatch,
  genuine tool failure, unregistered-tool failure, honest skip, fail-closed
  ethical block (and that the tool never runs on a blocked task),
  success-rate accounting, dependency gating, end-to-end `analyze`.

### Detectors — VLM / visual base contracts are now honest ABCs (2026-06-02)

Closes ROADMAP v1.7.x deferred items **#3** (VLM detector surface) and
**#4** (visual base detector).  Both bases previously raised
`NotImplementedError` from their contract methods — an ambiguous stub on a
public path.  They are now genuine `@abstractmethod` declarations:

* `detectors/vlm/base_vlm.py::BaseVLMDetector` — `_initialize_model`,
  `_create_prompt`, `_parse_response`, `detect`, `extract_features` are
  abstract; the class is explicitly **experimental** (the 2026-05 strategic
  decision keeps native detectors and does not ship BLIP/GPT adapters).
* `detectors/visual/base_visual.py::BaseVisualDetector` — `fit`, `detect`,
  `extract_features` are abstract; the native SOTA detectors (PatchCore,
  PaDiM, STFPM, ReverseDistillation, CFlow) remain the concrete
  implementations.

Direct instantiation of either base now raises `TypeError` (the honest
Python idiom) rather than constructing an object whose methods explode at
call time — no `NotImplementedError` remains on the public detector API.
Concrete subclasses are unaffected (all already implement every contract
method).  Coverage: `tests/test_vlm_detectors.py::TestBaseVLMDetector` and
`tests/test_visual_detectors.py::TestBaseVisualDetector` pin the abstract
contract (not-instantiable + `__abstractmethods__` set) and exercise the
concrete helpers (`_sample_frames`, `preprocess`, `postprocess`) via minimal
concrete subclasses.

### Neuro-symbolic — retire the untrained legacy LTN *random-init network*; co-training + conformal on the serve path (2026-06-02)

> **Partially superseded by the `LogicTensorNetwork` re-wire entry above.**
> This step removed the random-init *implementation*; the
> `LogicTensorNetwork` *surface* was subsequently **re-wired** to the canonical
> co-trained module — it is **not** left removed.

**Retired the never-trained random-init network.**
`models/neurosymbolic.py::LogicTensorNetwork` was a random-init `nn.Module`
whose forward pass returned meaningless noise that
`NeurosymbolicEngine.neural_inference` then fed into the fusion consensus *as
if it were a neural confidence* — the "untrained network as if live"
dishonesty the v1.7 audit flagged.  The random-init network is **removed** (the
`LogicTensorNetwork` surface itself is re-wired to the canonical module, not
retired — see the re-wire entry above); `neural_inference` now returns a
**deterministic statistical heuristic** (robust standardized-dispersion through
a logistic — bounded, reproducible, feature-responsive), and the module /
`neural_inference` / `get_statistics` docstrings carry a migration note to the
canonical *trained* neuro-symbolic surface,
`ml/symbolic_constraint.py::SymbolicConstraintModule` (co-trained in
`OmniMercuryEngine.fit_fusion`).  The module imports torch only lazily, inside
the re-wired LTN wrapper; the `NeurosymbolicEngine` reference surface stays
torch-free.  Regression: `tests/test_neurosymbolic_integration.py`
(deterministic + feature-responsive `neural_inference`, migration pointer; the
LTN-surface assertions were later flipped from *retired* to *wired* by the
re-wire above).

**Co-training + conformal verified together on the serve path.** New
`tests/test_fusion_symbolic_cotraining.py::TestCoTrainingConformalServePath`:

* `test_adaptive_cotraining_then_conformal_on_serve_path` — the production
  default (`symbolic_weight="adaptive"`) co-trains a `SymbolicConstraintModule`
  on a scarce-anomaly regime, then `calibrate_fusion_conformal` +
  `score_fusion_conformal` produce valid, bounded prediction sets and
  detection still separates the classes (ROC-AUC ≥ 0.85).
* `test_checkpoint_round_trip_no_symbolic_dimension_drift` — after
  co-training, `save_model` → `load_model` into a fresh engine reproduces
  `score_fusion` to within 1e-5: the co-trained symbolic constraint
  round-trips through the checkpoint (its `state_dict` + config are persisted
  and restored) yet stays inference-inert, so the symbolic channels never
  drift the persisted fusion dimensions consumed at scoring.

The adaptive-default and neural-only-escape (`symbolic_weight=0.0`) contracts
remain pinned by the existing `test_adaptive_is_the_default` /
`test_zero_weight_has_no_symbolic_state`.

### Cognitive — behavioural coverage for knowledge-graph + case-based reasoning (2026-06-02)

The largest cognitive file (`knowledge_graph.py`, 2109 LOC) and
`case_based_reasoning.py` had public methods that were import-clean-only (no
behavioural assertions).  Added behavioural tests pinning *algorithmic
correctness* (not anomaly-detection salvage — the DORMANCY_LEDGER measured
both at chance as detectors, a verdict left unchanged):

* `tests/cognitive/test_knowledge_graph_behavioral.py` (15 tests) —
  random-walk **embedding recovery** on a known two-cluster graph (intra >
  inter cosine over 5 seeds), `GNNMessagePassing.forward` shape + neighbour
  propagation, `LinkPredictor` / `KnowledgeGraph.predict_links` recovery of a
  held-out intra-cluster edge, and transitive-closure / symmetric-relation
  inference (including the empty result for non-transitive predicates).  Also
  documents that `RandomWalkEmbedding.fit` shuffles its `node_ids` argument
  in place.
* `tests/cognitive/test_case_based_reasoning_behavioral.py` (11 tests) —
  retrieval ranking + retrieval-count bookkeeping, domain filtering, the
  REUSE-vs-REVISE branch in `solve` (with the honest `no_matching_cases`
  result on an empty base), proportional `adapt` (numeric solution params
  whose key references the changed feature scale; unrelated params untouched)
  + adaptation history, and `learn_from_outcome` state updates.

DORMANCY_LEDGER rows #6 / #9 updated to note the new behavioural coverage.

### Core — concrete `AttentionProvider` wired to real multi-head attention (2026-06-02)

Closes ROADMAP deferred item **#7**.  `core/gosnn_optimizer.py::MultiHeadAttentionProvider`
is the first concrete `AttentionProvider`, backed by a real
`torch.nn.MultiheadAttention`: `observe(sequence)` runs a forward pass and
caches the **per-head** `(num_heads, seq_len, seq_len)` attention weights
(`average_attn_weights=False`); `get_attention()` returns them, or raises
`RuntimeError` before any forward ("model not yet run") so the optimizer
honestly skips the metric rather than scoring noise.  It defaults to 32 heads
to match `AttentionOptimizer`'s triadic φ-weighting, so wiring it into
`GOSNNOptimizer(attention_provider=…)` drives the real attention-overhead
metric — the removed deterministic-random placeholder is fully replaced.
Regression: `tests/core/test_attention_provider.py` (10 tests) — per-head
shape, softmax-normalised rows, fail-closed pre-forward, determinism, batched
input, and that a wired+observed provider drives the metric while an
unobserved one fails closed to a skip.

### Neuro-symbolic — label-scarcity-adaptive co-training default + dormant-module revival (PR #265, 2026-06-01)

`symbolic_weight="adaptive"` is the **default** for `OmniMercuryEngine.fit_fusion`.
A *fixed* `λ` was quarantined (no full-data win), but the label-scarcity
`ScarcityWeightSchedule` (`lambda_eff(n_pos) = lam_max·exp(-n_pos / n0)`,
defaults `lam_max=0.1, n0=25, floor=1e-3`) cleared a pre-registered dominance
bar on real ADBench labels, so the `SymbolicConstraintModule` co-trains when
positive labels are scarce and decays to the purely-neural path once they are
abundant (de-quarantining the genuine, co-trained LTN).  Companion
dormant-module revival survey (`benchmarks/dormant_module_revival.py`,
`docs/DORMANCY_LEDGER.md`) measures each archived module against real held-out
labels under the repo's fail-closed ablation discipline and records the
keep / cut / quarantine verdict per module.

### Neuro-symbolic — genuine co-training, conformal uncertainty, fail-closed AMA/PQC (PR #262, 2026-05-31)

* **Genuine neuro-symbolic co-training.** `ml/symbolic_constraint.py::SymbolicConstraintModule`
  — a differentiable fuzzy-logic constraint (product / Reichenbach /
  Łukasiewicz / Gödel semantics) over a consensus rule graph — is co-trained
  with the fusion network in `fit_fusion` when `symbolic_weight > 0`
  (`loss += λ · (1 − satisfaction)`), with gradients flowing to both the
  fusion model and the learnable rule confidences.  Pinned by
  `tests/ml/test_symbolic_constraint.py` and `tests/test_fusion_symbolic_cotraining.py`.
* **Conformal uncertainty on the serve path.** `BinaryConformalClassifier`
  with `calibrate_fusion_conformal` / `score_fusion_conformal` adds
  distribution-free prediction sets with a coverage guarantee on top of the
  temperature-calibrated fusion probability; a misconfigured conformal
  predictor raises `ConformalMisconfigurationError` instead of silently
  returning `confidence_intervals=None`.  Pinned by `tests/test_fusion_conformal.py`
  and `tests/federated/test_no_silent_failure.py`.
* **Fail-closed AMA / PQC.** AMA Cryptography native PQC is mandatory at
  package import regardless of `MERCURY_ENV`; Mercury fails closed (refuses to
  import) without the validated `ama-cryptography` backend.  Pinned by
  `tests/test_pqc_startup_gate.py` / `tests/security/test_pqc_gate_real_ama.py`.

### USGS Geochemistry — real NURE-HSSR downloader (2026-05-23)

`USGSGeochemistryLoader._download_from_usgs` was previously a literal
stub that returned `False` immediately and fell through to the
synthetic-distribution generator.  It now downloads and parses the
USGS NURE-HSSR (National Uranium Resource Evaluation Hydrogeochemical
and Stream Sediment Reconnaissance) bulk CSV from
`https://mrdata.usgs.gov/nure/sediment/nuresed-csv.zip` — a 39 MB
zipped CSV containing ~397K stream-sediment samples collected
across the continental US between 1973 and 1984, with per-sample
geochemistry for ~50 elements.  Public domain (US Government).

The new `_parse_nure_csv_zip` helper:

* materialises only the eleven columns the loader's `FEATURE_NAMES`
  schema exposes (lat/lon + EPA-screening metals + Fe/Ca/pH), so the
  235 MB expanded CSV is streamed without buffering the unused 134
  columns,
* stops once `max_samples` rows pass the configured region filter,
* applies the USGS-recommended below-detection-limit convention
  (NURE encodes "below threshold" as `-threshold`; the parser
  substitutes `threshold / 2` per Open-File Report 97-492),
* clamps invalid pH measurements (`<= 0` or `> 14`) to `0`,
* tags rows with the same EPA-screening-level anomaly labels the
  synthetic path used, so downstream detectors see a consistent
  feature/label contract whether real or synthetic data is in use.

Coverage:

* `tests/datasets/test_usgs_geochemistry.py` — 13 new unit tests
  exercising the happy path, below-detection handling, region
  filtering, max-samples cap, EPA-screening labels, pH clamping,
  schema-drift detection (missing required column raises),
  empty-zip rejection, cached-skip behaviour, and the two
  synthetic-fallback policy paths.  Tests are offline-only;
  the live network probe is the existing
  `tests/datasets/test_unreachable_loaders_network.py` USGS row
  which now exercises the real download() path end-to-end when
  `MERCURY_NETWORK_TESTS=1`.
* `src/omni_mercury_engine/security/input_validation.py` —
  `TrustedEndpoints.USGS_NURE_SEDIMENT_CSV` added.  The host
  `mrdata.usgs.gov` was already in `TRUSTED_DOMAINS`, so no SSRF
  allowlist surface change.

`USGSGeochemistry` remains in the reachability-harness watch list
(the harness exists to catch *upstream-provider* outages on top of
loader-code regressions; a working downloader does not retire the
provider-availability concern).  The benchmark "Empirical Benchmark
Results" headline still cites 64/75 because the next benchmark
refresh hasn't run yet — but `USGSGeochemistryLoader.is_real_data`
now returns `True` when the NURE CSV is reachable.

### Documentation / `docs/SECURITY.md` retired (2026-05-22)

`docs/SECURITY.md` was retired (commit 807b9c0) per owner directive
because the supply-chain posture it tracked is now covered by:

* `SECURITY.md` (the top-level public security policy, including the
  Two-Tier Dependency-CVE Coverage table that previously linked here);
* `.safety-policy.yml` (machine-readable v3 acceptance policy for the
  Safety CLI scanner — currently zero acceptances, OS-level only);
* the accepted-risk table in `SECURITY.md` plus the Trivy
  `ignore-unfixed` / `skip-files` gate configuration in
  `.github/workflows/{ci,security}.yml` (per-CVE acceptances for the
  deployment-image gate);
* dated security entries in this CHANGELOG (the per-PR rationale for
  remediations and risk acceptances).

Cross-references in `SECURITY.md`, `.safety-policy.yml`,
`.safety-policy-v2.yml`, `pyproject.toml`, and
`.github/workflows/ci.yml` were updated in PR #238 to point at the
surviving sources of truth.  No supply-chain posture changed; only
the location of the per-CVE documentation moved.

### ISO Hardening / Load Tests cold-start fix (2026-05-22)

The ISO Hardening Load Tests smoke job failed intermittently because
the first POST against `/api/v1/detect/univariate` paid a one-time
cold-start cost (Pydantic v2 model JIT compile + numpy SIMD dispatch
resolution + validator graph load) that pushed the smoke run's `p99`
HTTP duration above the production SLO threshold.  Real production
deployments would observe the same cold-start tail on the first
request after a worker spin-up.

The fix has three layers:

- `src/omni_mercury_engine/api/server.py` now wires a FastAPI
  `lifespan` async context manager that drives 3 in-process detection
  round-trips (univariate + multivariate + health) before uvicorn
  signals ready.  This means `/health` returns 200 only after
  Pydantic + numpy + the validator are warm, which is the correct
  posture for every deployment (k8s liveness probes also benefit
  because traffic is not routed until warmup completes).  Warmup
  failures propagate out of the lifespan hook and crash the worker
  so the orchestrator marks the deployment unhealthy — silently
  swallowing a warmup exception would let a broken detection path
  serve traffic to real callers, which is worse than a worker
  crashloop on a real regression.
- `.github/workflows/iso-hardening.yml` adds an explicit warmup loop
  (5× `/health` + 5× `/api/v1/detect/univariate`) between API
  readiness and k6 invocation, as defence-in-depth in case the
  lifespan hook is bypassed (e.g. a future env var or test override).
- `tests/load/k6_load_test.js` raises the `/health` `p(99)` ceiling
  from 50 ms to 150 ms.  Production health-check latency is observed
  at 5–20 ms; the 50 ms floor is achievable only on dedicated
  hardware and was the root cause of GHA-runner-jitter false
  positives.  Real regressions (p99 in the 200+ ms range) are still
  caught by the 150 ms threshold.

`tests/api/test_server_comprehensive.py::TestLifespanWarmup` (4 new
tests) pins the lifespan invariants: wiring, success path,
internal-failure propagation (the fail-fast contract above), and
TestClient context-manager exercise.

### Security / AMA-routed JWT HMAC signatures (HS256 + HS512) (2026-05-20)

Mercury's `native_jwt` signing primitive now routes `HS256` and
`HS512` through AMA Cryptography's ACVP-validated, constant-time,
zero-third-party-dep C HMAC backend when AMA Cryptography v3.2.0+ is
installed, falling back transparently to stdlib `hmac` over
`hashlib` otherwise.  This puts the JWT signing path on the same
crypto backend that already serves Mercury's PQC and HKDF stack
(matching AMA's INVARIANT-1 posture) and removes OpenSSL-backed
stdlib HMAC from the production auth path on AMA-enabled
deployments.

- **`pyproject.toml [pqc]`** pin bumped
  `ama-cryptography @ v3.1.0` → `@ v3.2.0`.  v3.2.0 exposes the
  Python bindings `native_hmac_sha256`, `native_hmac_sha256_2`
  (two-segment, concat-avoiding for JWT signing input), and
  `_HMAC_SHA256_NATIVE_AVAILABLE` over the ACVP-validated C symbol
  `ama_hmac_sha256` (150/150 vectors per
  `AMA-Cryptography/docs/compliance/ACVP_SELF_ATTESTATION.md`).  CI
  workflows `.github/workflows/ci.yml` and
  `.github/workflows/pqc-production-check.yml` updated to
  `AMA_REF: v3.2.0` in lockstep.
- **`src/omni_mercury_engine/security/ama_hmac.py`** (new, ~225
  lines): Mercury-side adapter that surfaces AMA's HMAC bindings
  with explicit availability flags (`HAS_AMA_HMAC_SHA256`,
  `HAS_AMA_HMAC_SHA512`), a public `available()` diagnostic helper
  for `/health` endpoints + audit logs, and a test-only
  `_reinitialize_for_tests` escape hatch.  Fail-closed semantics:
  the wrappers raise `RuntimeError` rather than silently falling
  back, so the routing decision is always explicit in
  `native_jwt._sign`.
- **`src/omni_mercury_engine/security/native_jwt.py`** refactored:
  `_sign()` now threads `(header_segment, payload_segment)`
  separately rather than the materialised concat, so the HS256 path
  can use AMA's `native_hmac_sha256_2(key, header || ".", payload)`
  fast path without ever copying the payload bytes in Python.  HS512
  routes through `ama_hmac_sha512` (one-segment; AMA does not yet
  ship a two-segment HMAC-SHA-512 variant).  HS384 stays on stdlib
  because AMA does not bind HMAC-SHA-384 in v3.2.0 (tracked in
  `docs/ROADMAP.md`).  A new public helper `get_signing_backend(alg)`
  returns `"ama"` or `"stdlib"` for diagnostic surfaces.
- **`tests/security/test_native_jwt_ama_routing.py`** (new, 17
  tests) locks four invariants:
  1. **RFC 4231 KAT.**  AMA's HMAC-SHA-256 and HMAC-SHA-512 output
     matches the canonical Test Case 1 and Test Case 7
     (oversized-key) vectors from RFC 4231 §4.2 / §4.7.
  2. **Stdlib byte-equivalence at the `_sign()` boundary.**  AMA-
     routed and stdlib-routed signatures are bit-identical for the
     same `(header, payload, key, alg)` triple (FIPS 198-1 /
     RFC 2104 invariant).
  3. **Fallback path.**  Monkeypatching
     `ama_hmac.HAS_AMA_HMAC_SHA256 = False` cleanly demotes the
     `_sign()` decision to stdlib and the JWT encode/decode round-
     trip still succeeds.
  4. **Cross-path interoperability.**  A token signed with AMA
     enabled verifies with AMA disabled, and vice versa — the
     routing decision is performance / hardening only, never a
     wire-format change.

  All 17 tests pass; the 92 existing native-JWT / auth contract
  tests in `tests/security/test_native_jwt.py`,
  `tests/security/test_jwt_auth.py`, and
  `tests/api/test_auth_comprehensive.py` remain green with no
  semantic change.

### Security / Permanent supply-chain remediation: native JWT + joblib removal (2026-05-20)

Three upstream-disputed advisories — `PYSEC-2024-277` /
`CVE-2024-34997` (joblib), `PYSEC-2026-97` / `CVE-2026-0846`
(nltk), and `PYSEC-2025-183` / `CVE-2025-45768` (pyjwt) — were
**permanently retired** from Mercury-Agent's audited supply chain
by removing the affected dependencies rather than ignoring the
advisories or accepting risk.

- **`pyjwt` removed.** Replaced by
  `src/omni_mercury_engine/security/native_jwt.py`, a pure-stdlib
  HS256 JWT module built on `hmac` + `hashlib` + `base64` + `json`
  with constant-time signature verification (via
  `security/constant_time.py`) and `alg: none` rejected by
  construction (HS256-only encoder; decoder whitelists algorithms
  before any HMAC work).  29 unit tests in
  `tests/security/test_native_jwt.py`; 14 contract tests in
  `tests/security/test_jwt_auth.py` and
  `tests/api/test_auth_comprehensive.py` adapted to the new module
  with no semantic change.  `api/auth.py` now imports
  `omni_mercury_engine.security.native_jwt as jwt`, so call sites
  and exception types (`InvalidTokenError`, `ExpiredSignatureError`,
  …) are preserved unchanged.

- **`joblib` removed.** `ParallelExecutor` in
  `src/omni_mercury_engine/ml/optimization.py` is rewritten on
  `concurrent.futures.{ProcessPoolExecutor, ThreadPoolExecutor}`.
  The legacy `enable_joblib` / `joblib_backend` config field names
  are preserved as compatibility aliases — `loky` /
  `multiprocessing` map to the process pool and `threading` to the
  thread pool — so downstream config files keep working unchanged.
  Locked by `tests/ml/test_new_modules.py::test_parallel_executor_no_joblib_import`.

- **`nltk` was never a Mercury dependency** — it appeared in the
  audit scope only because `safety` itself depends on it.  The CI
  audits (`ci.yml` and `security.yml`) now install Mercury into an
  isolated venv (`/tmp/mercury-audit-env`) and scan only that
  install, so auditor-internal transitives are excluded by
  construction.

- **`pyproject.toml`** drops `pyjwt>=2.12.0` from `[api]`,
  `joblib>=1.3.0` from `[optimization]` and `[benchmark]`, and the
  `jwt` / `joblib` mypy override entries from
  `[[tool.mypy.overrides]]`.

- **`docs/SECURITY.md`** (the supply-chain posture ledger) deletes
  the three IGNORE rows and replaces them with a "Permanent
  supply-chain remediations" section documenting each removal, the
  in-tree replacement, the commit, and the test that locks the
  remediation.

Verification on the isolated Mercury [api] install (42 packages,
Python 3.12, 2026-05-20):

```
safety check  → 0 vulnerabilities reported, 0 vulnerabilities ignored
pip-audit     → No known vulnerabilities found  (exit 0)
```

No `--ignore-vuln` / `--ignore` flags are wired into either
workflow.  The audit-ledger posture is now "zero risk acceptance":
findings are remediated by upgrade, isolation, or native re-
implementation — never by suppression.

### Security / Synthetic-data policy-gate bypass closure (2026-05-20)

The validation-pipeline loaders
(`omni_mercury_engine.validation.data_loaders`) exposed a
`use_synthetic: bool = False` argument on every concrete loader
(`NSLKDDLoader`, `USGSEarthquakeLoader`, `MIMICLoader`,
`NOAASpaceWeatherLoader`, `NOAAHurricaneLoader`, `NOAAOceanLoader`).
When set to `True`, the loader returned synthetic data unconditionally
— **bypassing the deployment-level `MERCURY_ALLOW_SYNTHETIC` policy**
enforced by
`omni_mercury_engine.datasets.exceptions.check_synthetic_allowed`.

This was a latent integrity hole: a benchmark, downstream notebook, or
operator script could request synthetic data via the keyword argument
and receive it even on a deployment that had explicitly forbidden
synthetic fallback (`MERCURY_ALLOW_SYNTHETIC=0` or unset).  For a
humanitarian crisis-response and missing-persons platform, silently
delivering simulated data when policy forbids it is far worse than
raising — operators can act on synthetic output without realising the
real source was never reached.

The fix is surgical and additive:

- Every `if use_synthetic:` branch in `validation/data_loaders.py`
  now calls `check_synthetic_allowed(loader_name, "Caller passed
  use_synthetic=True")` immediately before invoking
  `_generate_synthetic(...)`.  When the env var is not set, the call
  raises `DataSourceUnavailableError`; when it is set, the legacy
  contract (caller-flag honoured) holds.
- `MIMICLoader` (which is *only* available as a synthetic simulation
  because real MIMIC-III requires PhysioNet credentialing) was
  rewritten from a tangled `if not use_synthetic and not
  ALLOW_SYNTHETIC: raise` / `if not use_synthetic:
  check_synthetic_allowed` two-branch pattern to a single
  unconditional `check_synthetic_allowed(...)` call.  Both the
  explicit caller request (`use_synthetic=True`) and the implicit
  fallback (`use_synthetic=False`) now flow through the same gate,
  with the documentation pointing operators at the real PhysioNet
  download URL.
- `tests/validation/test_synthetic_policy_gate.py` is the regression
  lock: six tests prove the gate fires for each loader when policy is
  off, plus a seventh forward-compatibility test confirms the
  legacy contract (caller flag honoured under `MERCURY_ALLOW_SYNTHETIC=1`).

The closure is additive — no existing test or documented contract is
broken.  `tests/conftest.py` continues to set
`MERCURY_ALLOW_SYNTHETIC=1` by default for the test suite, so all 44
existing `tests/validation/test_validation_pipeline.py` tests still
pass without modification.  The fallback chain documented in
`docs/ROUTING_GUIDE.md` still works under the documented contract
(`MERCURY_ALLOW_SYNTHETIC=1`); without the env var set, the chain
fails closed at the synthetic step rather than silently degrading.

### Security / σ_Immutable Wave B Vector 2 + 4 closure (2026-05-20)

Closes two of the remaining σ_Immutable bypass vectors identified in the
v1.7 audit:

- **Vector 2 — engine boundary.** `engine._enforce_ethics_at_boundary`
  now routes the caller-supplied `domain` through a canonical
  `sanitize_domain()` helper instead of the prior bare
  `isinstance(domain, str)` check.  Hostile values such as
  `"damage_control"` or `"audit track"` are collapsed to the
  `"general"` sentinel before reaching the `BenevolenceScorer`, so a
  caller cannot inject harm/positive keywords into the action string
  to bias the scorer.
- **Vector 4 — neuro-symbolic hub.** `NeuroSymbolicHub.__init__` now
  sanitises the `domain` constructor argument before storing it on
  the audit surface (`self.domain` interpolated into σ_Immutable
  `details` payloads); the downstream feature-dispatch components
  (`FibringComposer`, `DomainFeatureExtractorFactory`,
  `GOSNN3RIntegration`) still see the raw caller value for legacy
  compatibility.  `NeuroSymbolicHub.predict` now pre-flights the
  σ_Immutable gate on `n_samples == 0` batches; previously an empty
  input would return `[]` immediately without ever firing the per-
  sample enforcement loop, giving callers a silent no-op bypass.
- **Canonical sanitiser** lifted into
  `omni_mercury_engine.cognitive.ethical_bounding.sanitize_domain`
  with a deferred `EnvironmentDomain` import to preserve
  `ethical_bounding`'s zero-cost import contract.
  `cognitive/orchestrator.py` now delegates to the same helper
  instead of carrying a local copy of the whitelist.
- **Regression**: `tests/ethical/test_hard_enforcement.py` grows
  `TestSanitizeDomainHelper` (4 cases) and
  `TestNeuroSymbolicHubEmptyBatchClosure` (2 cases) — the empty-batch
  closure is locked with a `monkeypatch` spy that asserts the gate
  fires with `details["empty_batch"] is True` and a sanitised
  `details["domain"] == "general"`.
- **Wave B Vector 3** (`CognitiveOrchestrator.analyze`) was already
  enforced in v1.6 and remains in place.  Vectors 5 (narrative voice)
  and 6 (federation aggregator + federated_learning server) are
  deferred to a Wave C follow-up: those subsystems do not currently
  carry σ_Immutable wiring, and adding it without breaking the
  existing calling contract requires a careful interface review that
  is out of scope for the v1.7.0 release cut.  The deferral is
  tracked in `docs/ROADMAP.md`.

### Security / CVE-2026-6357 regression guard (2026-05-20)

CVE-2026-6357 lets a malicious wheel hijack the install process on
`pip` versions earlier than 26.1.  Every Dockerfile stage, every CI
workflow that runs `pip install`, and every devcontainer / dev-tooling
script already floors `pip>=26.1` in the v1.7 baseline; this commit
adds a regression guard that makes the contract durable:

- `scripts/check_workflow_hardening.py` grows a
  `_check_pip_cve_2026_6357` step.  It walks every workflow YAML,
  groups lines by job (since runner site-packages is shared across
  all steps in a job), and fails the `Workflow Hardening` CI gate if
  any `pip install` (or `python -m pip install`) appears in a job
  without a prior `pip install --upgrade "pip>=26.1"`.  Documentation-
  emission lines that *write* the literal string `pip install` into
  a file (`echo`, `printf`, `>>` / `<<` redirection) are exempt and
  so are inline comments.
- `.github/workflows/format.yml` and `.github/workflows/network-tests.yml`
  were updated to floor pip explicitly (previously the `format` job
  installed Black without pinning pip first; `network-tests` issued a
  bare `python -m pip install --upgrade pip`).
- `tests/security/test_cve_2026_6357_regression.py` is the *gate on
  the gate*: it directly exercises the hardening checker plus the
  real workflow / Dockerfile inventory so a future drift — a new
  workflow that forgets the floor, a Dockerfile that re-introduces
  an unpinned `pip install`, or a regression in the checker itself —
  is caught by `pytest` long before it can land in a release branch.

## [1.7.0] - 2026-05-20

### Documentation refresh (2026-05-19)

Comprehensive documentation update covering the v1.7 development cycle.
Every doc surface dated 3+ months stale has been refreshed, and three
new docs were added for modules that had no operator-facing
documentation:

- **New: `docs/COMPLIANCE.md`** — first-party reference for the
  `omni_mercury_engine.compliance` package (NIST CSF 2.0 integrator,
  OSHA / eCFR detector with NWS Rothfusz heat-index regression, TLP
  2.0 handler with the full five-colour ladder including
  `AMBER+STRICT`).
- **New: `docs/PROFILING.md`** — operator reference for the
  `omni_mercury_engine.utils.profiling` toolkit (six decorators +
  `PerformanceBenchmark` + `benchmark_function`, gated by
  `set_profiling_enabled(...)`).
- **New: `docs/drone/SETUP.md`** — operator setup guide for the
  drone anomaly detector (referenced from the detector module
  docstring; `DroneState` contract, PX4 ULog / MAVLink ingest
  examples, three upstream-defect fixes recorded).
- **Updated: `ARCHITECTURE.md` (root)** — new sections covering the
  v1.7 governance framework modules, medical decision-support
  modules, drone detector, and profiling toolkit.
- **Updated: `DEPRECATION.md`** — new §6 "v1.7 Removals
  (security/correctness exceptions)" enumerating the four surfaces
  removed under the preservation policy override criteria
  (`SafeHTTPClient(..., allow_untrusted=True)`, silent
  `MockLLMAdapter` fallback, `strict_ethics=False`,
  `gosnn_metadata.fallback_mode=True`) and the one relocation
  (`tlp_handler` from `security/` to `compliance/`).
- **Updated: `SECURITY.md`** — v1.7 hard-gate boundary contract,
  `MERCURY_ENV` production-mode primitive, two-tier dependency-CVE
  coverage table, governance-framework cross-references.
- **Updated: `CONTRIBUTING.md`** (v2.5) — v1.7 do-not-restore
  items, medical / drone / compliance integration-ready
  contribution channels.
- **Updated: `docs/INSTALLATION.md`, `docs/DEPLOYMENT.md`** —
  `MERCURY_ENV` + `AMA_REQUIRE_REAL_PQC` production-mode primitives,
  `[compliance]` and `[pqc]` extras, v1.7 module installation notes.
- **Updated: `docs/API_REFERENCE.md`** — module-index table, quick-
  import blocks for compliance / medical / drone / profiling
  surfaces, decision-boundary contract banner.
- **Updated: `docs/index.md`** — `MERCURY_ENV` and compliance
  primitives in the "What you should know first" block; toctree
  entries for the new docs.
- **Updated: `docs/ROADMAP.md`** — capability-status table refreshed
  to 2026-05-19 with seven new "Functional" rows (NIST CSF 2.0
  integrator, TLP 2.0 handler, OSHA / eCFR detector, drone anomaly
  detector, endocrinology detector, anesthesiology predictor,
  performance profiling toolkit).
- **Updated: `docs/DATASOURCES.md`** — "Last verified" bumped to
  2026-05-19; v1.7 reachability harness and the 65/65 / 64/75 /
  51/55 benchmark trajectory reconciled.
- **Updated: `docs/SECURITY.md`** (renamed from
  `docs/PYTHON_DEP_CVE_AUDIT.md`) — audit date / next-review bumped
  (2026-05-19 → 2026-08-19); v1.7 dependency surface (`openpyxl` for
  the `compliance` extra, exact `v3.2.0`
  pin for the `pqc` extra) documented.
- **Updated: `docs/medical/SETUP.md`, `docs/BENCHMARKS.md`,
  `docs/ROUTING_GUIDE.md`, `docs/MATH_SPEC.md`,
  `docs/LIVE_DATA_VALIDATION.md`, `docs/ORACLE_NOISE_COLOR.md`,
  `docs/DOMAIN_PERFORMANCE.md`, `benchmarks/DATASETS.md`** —
  date-stamped to 2026-05-19 with v1.7 context where applicable.
- **Updated: `rust_crypto/README.md`** — clarified scope (classical
  crypto, **not** PQC); pointed PQC use cases at
  AMA Cryptography v3.2.0 and the `[pqc]` extra.
- **Updated: `CODE_OF_CONDUCT.md`** — added document-version
  metadata table and a Mercury-specific note tying the Code of
  Conduct to the dual hard ethical gates encoded in the software.

No source code or behaviour changed in this entry; the
documentation refresh is text-only.

### Omni-AXA → Mercury port, PR 1: infrastructure & stdlib-only modules

Three first-party modules ported from `Steel-SecAdv-LLC/Omni-AXA-Engine`
(GPL-3.0+) into Mercury Agent.  The runtime dependency surface added by
these ports is `numpy` (already required by Mercury Agent core),
`requests` (existing Mercury dependency, used by the NIST CSF live
fetcher), and `openpyxl` (new, gated behind the `compliance` extra in
`pyproject.toml` and only required when parsing the live NIST CSF
reference XLSX).  No new mandatory dependencies were introduced.  All
three ported modules are fully typed for `mypy --strict` and have no
overlap with existing Mercury code.

- **`omni_mercury_engine.utils.profiling`** (411 LOC source → 652 LOC port).
  Six public entry points: `@profile_func`, `@profile_memory`,
  `@profile_time`, `@profile_time_async`, `@profile_complete`,
  `PerformanceBenchmark` context manager, and `benchmark_function`.
  Mercury delta: added `@profile_time_async` for Mercury's asyncio
  paths and added an opt-in global enable flag exposed via
  `set_profiling_enabled(...)` / `is_profiling_enabled()` in
  `omni_mercury_engine.utils.profiling`.  Locked by
  `tests/test_profiling.py` (32 tests).

- **`omni_mercury_engine.compliance.nist_csf_integrator`**
  (578 LOC source → 1,349 LOC port).  Implements NIST CSF 2.0
  end-to-end: the six core functions (GOVERN, IDENTIFY, PROTECT,
  DETECT, RESPOND, RECOVER), 22 categories, and 106+ subcategories
  with implementation tier scoring (PARTIAL → ADAPTIVE),
  organisational profiles, gap analysis, supply-chain anomaly
  detection, continuous-monitoring deltas, and JSON-serialisable
  compliance reports.  Mercury delta: a new
  `NISTCSFReferenceFetcher` hits the live NIST CSF 2.0 Reference
  Tool at
  `https://csrc.nist.gov/extensions/nudp/services/json/csf/download?olirids=all`
  (XLSX, ~143 KB) with a 7-day on-disk cache under
  `$XDG_CACHE_HOME/mercury-agent/nist_csf` so callers see the
  authoritative subcategory tree rather than a hard-coded
  snapshot.  Locked by `tests/test_nist_csf_integrator.py`
  (29 unit + 2 `@pytest.mark.network` integration tests against
  csrc.nist.gov).

- **`omni_mercury_engine.compliance.tlp_handler`**
  (313 LOC source → 603 LOC port).  Implements FIRST.org / CISA
  TLP 2.0 classification with the full five-colour ladder
  (CLEAR / GREEN / AMBER / AMBER+STRICT / RED), single-anomaly and
  batch classification, per-colour statistics, watermark
  generation, and a JSON-serialisable export-metadata block.
  **Behavioural delta (known-issue fix):** the upstream module
  shipped only the four legacy TLP 1.0 colours; `AMBER+STRICT` has
  been added end-to-end (classification, reasoning, sharing
  guidelines, ethical considerations, watermark, export metadata)
  so Mercury is TLP-2.0 compliant out of the box.  Sharing
  guidelines are verbatim from FIRST.org TLP 2.0; bare
  `except:` clauses present in the upstream module were replaced
  with explicit `TLPValidationError` paths.  **Location delta:**
  the module lives in `omni_mercury_engine.compliance` alongside
  `nist_csf_integrator` and `osha_anomaly` rather than in
  `omni_mercury_engine.security`.  Mercury's `security/` package
  is reserved for implementation primitives (crypto, PQC, threat
  detection, audit logging); governance frameworks live in
  `compliance/`.  The upstream location was
  `domains/ciad/compliance/`, so this restores upstream intent.
  Internal callers (`utils.report_generator`,
  `tests.test_report_generator`, `tests.test_tlp_handler`) were
  updated to the new import path; a repository-wide
  `git grep` confirmed no other call sites, so no
  backwards-compatibility shim was added in `security/`.  Locked
  by `tests/test_tlp_handler.py` (45 tests covering every public
  surface including AMBER+STRICT escalation, watermark integrity,
  and export-metadata schema).

### `ReportGenerator` — first-class TLP 2.0 wiring

- **`ReportGenerator.apply_tlp_classification(...)`** is the canonical
  choke-point for tagging Mercury reports with a Traffic Light
  Protocol classification.  Callers either pass a pre-computed
  `TLPClassification` (preferred when the colour was decided at an
  earlier choke-point) or supply an `anomaly_score` and let the
  handler classify here.  The classification is rendered into
  every output format produced by `generate()`:
  - JSON output gains a top-level `"tlp"` block containing the
    canonical `tlp_label` / `tlp_color` / `tlp_rank` /
    `tlp_confidence` / `tlp_reasoning`, the FIRST.org
    sharing-guideline text, the ethical considerations list, and a
    watermark string.
  - HTML output renders a sanitised `<div class="tlp-banner">`
    above the report title with the watermark and sharing
    guidelines (HTML escaping is preserved end-to-end; sensitive
    output never bypasses the existing XSS protection).
  - Markdown output renders a `> **TLP:…**` blockquote at the
    top of the document.
  Default behaviour is unchanged when callers do not opt in,
  so existing report consumers are not affected.  Locked by
  `tests/test_report_generator.py::TestReportTLPIntegration`
  (10 tests including the no-classification baseline, the
  pre-computed path, the score-driven path, validation errors,
  per-format rendering checks, and the strict-sharing escalation
  path).

### Omni-AXA → Mercury port, PR 2: domain modules with external dependencies

Four domain modules ported from `Steel-SecAdv-LLC/Omni-AXA-Engine`
(GPL-3.0+) into Mercury Agent.  All known issues from the verdict table
are resolved in-PR; no follow-ups.  Compliance and drone modules are
wired to live public data sources (eCFR, PX4 / MAVLink).  The medical
modules ship **integration-ready, not pre-integrated**: real adapter
ABCs (`CGMDataSource`, `VitalsDataSource`), a reference Dexcom v3
OAuth2 adapter, and a reference HL7 FHIR R4 Observation adapter.
Mercury Agent never carries vendor medical credentials; the platform
refuses to start a misconfigured medical integration via
`ConfigurationError`.  See `docs/medical/SETUP.md` for the operator
runbook.

- **`omni_mercury_engine.compliance.osha_anomaly`**
  (666 LOC source → 1,003 LOC port).  Multi-sector OSHA compliance
  detector covering 12 hazard categories × 6 industry sectors with
  real CFR citations.  **Known-issue fix (heat-index regression):**
  the original implementation used the linear heuristic
  ``HI = T + 0.5 * RH``.  At ``T=95 °F, RH=70 %`` this returned
  ``130 °F``, materially over-reporting heat stress.  The port
  replaces this with the **National Weather Service Rothfusz
  regression**:
  ```
  HI = -42.379 + 2.04901523·T + 10.14333127·RH
       − 0.22475541·T·RH      − 0.00683783·T²
       − 0.05481717·RH²       + 0.00122874·T²·RH
       + 0.00085282·T·RH²     − 0.00000199·T²·RH²
  ```
  plus the two standard adjustments — low-humidity
  (``RH < 13 %`` and ``80 ≤ T ≤ 112 °F``) and
  low-temperature/high-humidity (``RH > 85 %`` and
  ``80 ≤ T ≤ 87 °F``).  Numeric NWS reference point at
  ``T=96 °F, RH=65 %`` now returns ``≈121 °F`` (verified against
  the published WPC table) instead of ``128 °F`` under the
  heuristic.  Citations may optionally be validated against the
  live **eCFR API** (`https://www.ecfr.gov/api/versioner/v1/`,
  60 req/min, no auth) via the new ``ECFRClient`` helper, which
  caches verifications in-process.  Locked by
  `tests/test_osha_anomaly.py` (28 tests covering every
  sector × hazard combination, the Rothfusz regression and both
  adjustments, eCFR parsing/caching/error paths, training
  recommendations, and the compliance report).

- **`omni_mercury_engine.anomaly.drone_detector`**
  (632 LOC source → 905 LOC port).  Multi-source drone anomaly
  detector combining rule-based RADD, an ML ensemble, and
  log-based DronLomaly.  **Three known-issue fixes in-PR:**
  1. **Missing DroneState fields.**  The upstream detection
     rules referenced ``altitude_rate``, ``horizontal_velocity``,
     ``vertical_velocity``, and ``distance_to_home`` but the
     ``DroneState`` dataclass did not have them, so every rule
     that gated on those fields silently no-op'd.  The port adds
     the four fields with explicit ``Optional[float]`` types and
     a ``__post_init__`` that derives them from
     ``velocity`` / ``position`` / ``home_position`` when not
     explicitly supplied.  Regression tests in
     ``tests/test_drone_detector.py`` exercise the previously
     silent rules end-to-end.
  2. **Mercury in-house anomaly ensemble.**  The hand-coded z-score
     "ensemble" was first replaced with three scikit-learn estimators
     (``IsolationForest``, ``EllipticEnvelope``, ``LocalOutlierFactor``)
     during the initial port.  That implementation is itself superseded
     in this PR by Mercury Agent's own first-class anomaly ensemble,
     :class:`~omni_mercury_engine.detectors.statistical.MercuryAnomalyDetector`,
     which combines three deterministic ``numpy``/``scipy`` scorers —
     **Resonance** (40 %; FFT-based harmonic spectral anomaly),
     **Kinematic** (30 %; physics-based jerk / curvature dynamics) and
     **InfoGeometry** (30 %; Fisher Information Matrix OOD detection).
     The previous default ensemble weights (``isolation_forest=0.40``,
     ``elliptic_envelope=0.25``, ``lof=0.35``) are replaced with
     ``resonance=0.40``, ``kinematic=0.30``, ``info_geometry=0.30``,
     matching the published MercuryAnomalyDetector ratio.  The
     ``DroneAnomalyDetector(ensemble_weights=...)`` keyword still
     accepts a custom override; callers must now key it on the three
     new component names.  scikit-learn is **removed from the drone
     detector's runtime dependency surface** — it survives only in the
     ``benchmark-comparison`` optional extra (per ``pyproject.toml``),
     where it is used to score Mercury against external baselines
     rather than to power Mercury itself.  The Mahalanobis fallback
     path and its ``_SKLEARN_AVAILABLE`` flag are removed; the
     in-house ensemble is deterministic after fit and has no optional
     dependency.  Locked by ``TestMercuryEnsemble`` in
     ``tests/test_drone_detector.py``, which asserts the absence of
     every previously-exported sklearn symbol on the module and
     verifies the three Mercury components are produced with scores
     in ``[0, 1]``.
  3. **Unvalidated 93.84 % recall claim removed.**  The
     upstream docstrings cited a 93.84 % recall number with no
     reproducible benchmark.  The claim is removed and a
     ``test_no_recall_claim_in_docstrings`` regression test
     pins the docstrings against re-introduction.
  Live data: integrates with PX4 / MAVLink telemetry via the
  detector's ``detect_faults(state, logs)`` entry point —
  callers translate their MAVLink / pyulog stream into the
  ``DroneState`` dataclass.  Locked by
  `tests/test_drone_detector.py` (27 tests covering all three
  known-issue fixes, every mission-phase rule that previously
  silent-no-op'd, the Mercury in-house ensemble component layout,
  degenerate-window robustness, log-based fault detection,
  flight-report aggregation, and per-fault recommendation lists).

  **Deviations from the original (drone_anomaly_detector.py):**
  - Ensemble vendor: the upstream module imports
    ``sklearn.ensemble.IsolationForest``,
    ``sklearn.covariance.EllipticEnvelope`` and
    ``sklearn.neighbors.LocalOutlierFactor`` directly and falls back
    to a hand-coded Mahalanobis scorer when sklearn is missing.  The
    port replaces both branches with Mercury Agent's in-house
    ``MercuryAnomalyDetector`` (Resonance / Kinematic / InfoGeometry).
    scikit-learn is no longer imported at runtime by this module; it
    remains available only via the ``benchmark-comparison`` optional
    extra for external baseline comparisons.
  - ``ensemble_weights`` keys: ``{"isolation_forest", "elliptic_envelope",
    "lof"}`` → ``{"resonance", "kinematic", "info_geometry"}``.  The
    weights themselves continue to be normalised to sum to ``1.0``.
  - ``random_state`` constructor argument is retained for API stability
    but is now a no-op; ``MercuryAnomalyDetector`` is deterministic
    after ``fit()`` and consumes no RNG seed.
  - ``_compute_sklearn_scores`` and ``_compute_fallback_scores`` are
    removed; the single ``_compute_ensemble_scores`` path now
    delegates to ``MercuryAnomalyDetector.fit`` /
    ``MercuryAnomalyDetector.detect`` and returns an empty dict on
    fit/detect failure rather than zero-filling components.

- **`omni_mercury_engine.medical.anesthesiology_predictor`**
  (541 LOC source → 741 LOC port).  Integrated anesthesiology
  prediction combining a Bi-LSTM TIVA monitor, a discrete-time
  PID infusion controller, and a hemodynamic monitor:
  - **TIVA Bi-LSTM** (``input_dim=8``, ``hidden_dim=64``,
    ``num_layers=2``, bidirectional, additive attention,
    164,066 parameters — matches the verified Omni-AXA
    parameter count).
  - **PID infusion controller** with the verified upstream
    gains ``kp=0.5 / ki=0.1 / kd=0.2``, target BIS 50, safe
    window ``[40, 60]``.  Decision support only; the controller
    refuses to run with ``dt ≤ 0`` and clamps outputs to
    documented propofol / remifentanil limits.
  - **Hemodynamic monitor** with the ASA-aligned ranges
    MAP 65–110 mmHg, HR 50–100 bpm, SpO₂ ≥ 92 %,
    EtCO₂ 30–45 mmHg.
  Integration: the predictor now requires a
  ``VitalsDataSource`` adapter at construction time when
  ``enable_hemodynamics`` is true (the default); without one the
  constructor raises ``ConfigurationError``.  A reference
  ``FHIRObservationVitalsSource`` ships in
  ``omni_mercury_engine.medical.data_sources`` and speaks HL7
  FHIR R4 ``Observation`` search with ``category=vital-signs`` —
  spec-compliant against Epic, Oracle/Cerner, MEDITECH, and the
  SMART-on-FHIR sandbox.  LOINC codes recognised: 8867-4 HR,
  8480-6 / 8462-4 SBP/DBP (MAP computed when absent), 8478-0 MAP
  direct, 2708-6 / 59408-5 SpO₂, 19911-5 EtCO₂.  Synthetic
  generators and the old ``VitalDBClient`` have been removed
  from production paths; integrators wire their own vendor
  adapter (Philips IntelliVue, GE CARESCAPE, Mindray, custom HL7
  v2 / FHIR endpoint) by subclassing ``VitalsDataSource``.
  Locked by `tests/test_anesthesiology_predictor.py` (rule-engine
  and integration tests against an in-process
  ``VitalsDataSource``) and `tests/test_medical_data_sources.py`
  (FHIR adapter end-to-end against sanitized fixtures).

- **`omni_mercury_engine.medical.endocrinology_detector`**
  (521 LOC source → 660 LOC port).  Integrated endocrine anomaly
  detection with CGM Bi-LSTM analysis and three FDA-aligned
  rules:
  - **CGM Bi-LSTM** (``input_dim=1``, ``hidden_dim=64``,
    ``num_layers=2``, bidirectional, additive attention) for
    glycemic-state classification (normal / hypo / hyper /
    severe-hypo / DKA) plus a scalar trend predictor.
  - **Afrezza FEV1 contraindication.**  Inhaled insulin is
    flagged inappropriate when ``FEV1 < 70 %``.  The threshold
    is strict-less-than, so ``FEV1 = 69.9 %`` is flagged and
    ``FEV1 = 70.0 %`` is permitted; the boundary case is locked
    by `TestInhaledInsulinMonitor::test_fev1_threshold`.
  - **GLP-1 pancreatitis discontinuation.**  Any occurrence of
    "pancreatitis" (case-insensitive) in the side-effect list
    sets ``continue_therapy=False`` and emits the FDA-aligned
    discontinuation recommendation.
  - **Dose-stacking guard.**  Rapid-acting insulin doses spaced
    less than ``min_dose_interval_hours = 2.0`` apart trip the
    smart-pen alert and require glucose verification before
    additional dosing.
  Integration: the detector now requires a ``CGMDataSource``
  adapter at construction time when ``enable_cgm`` is true (the
  default); without one the constructor raises
  ``ConfigurationError``.  A reference ``DexcomV3DataSource``
  ships in ``omni_mercury_engine.medical.data_sources`` and
  speaks the Dexcom Developer API v3 over OAuth2 refresh-token
  flow (``api.dexcom.com/v2/oauth2/token`` →
  ``/v3/users/self/egvs``).  Required environment variables:
  ``DEXCOM_CLIENT_ID``, ``DEXCOM_CLIENT_SECRET``,
  ``DEXCOM_REFRESH_TOKEN``, ``DEXCOM_REDIRECT_URI``;
  ``DEXCOM_BASE_URL`` defaults to production with sandbox
  override available.  Synthetic generators and the old
  ``TidepoolClient`` / ``DexcomCredentials`` helpers have been
  removed from production paths; integrators wire their own
  vendor adapter (Abbott LibreView, Medtronic CareLink, custom
  cloud bridge) by subclassing ``CGMDataSource``.  Locked by
  `tests/test_endocrinology_detector.py` (rule-engine and
  integration tests against an in-process ``CGMDataSource``)
  and `tests/test_medical_data_sources.py` (Dexcom v3 adapter
  end-to-end against sanitized fixtures, including OAuth token
  caching and HTTP error wrapping).

- **`omni_mercury_engine.medical.data_sources`** (new, 800 LOC).
  Common medical-data infrastructure:
  ``CGMDataSource`` / ``VitalsDataSource`` ABCs,
  ``CGMReading`` / ``VitalsReading`` dataclasses,
  ``ConfigurationError`` / ``DataSourceError`` typed exceptions,
  reference ``DexcomV3DataSource`` (OAuth2 refresh) and
  ``FHIRObservationVitalsSource`` (HL7 FHIR R4 Observation
  search), plus module-level
  ``parse_dexcom_egvs_payload`` / ``parse_fhir_observation_bundle``
  helpers so integrators can unit-test their own payloads
  against the same parsers Mercury uses internally.  See
  `docs/medical/SETUP.md` for the full operator runbook.

**Provenance.**  All four modules originate from
``Steel-SecAdv-LLC/Omni-AXA-Engine`` (private; GPL-3.0+).  The
upstream license matches Mercury's, and the user has full legal
standing to relicense across both repositories.

### Omni-AXA → Mercury port, PR 2 refinements — round 3 (Copilot review)

Round-3 refinement pass closing the twelve Copilot review alerts on
PR #224.  Every alert is addressed in-code; no `# noqa`, no
`# type: ignore`, no `pragma: no cover`, no broad except, and no
coverage-threshold lowering.  The changes harden SSRF / DNS-rebinding
posture on every outbound HTTP call made by the ported modules,
upgrade the DroneState input contract with explicit shape validation,
and reconcile the medical module top-level docstrings with the
deviations already documented under the per-module
*"Deviations from the original"* subsections below.

- **SSRF / DNS-rebinding gate on every medical and compliance HTTP
  call.**  `DexcomV3DataSource._refresh_access_token`,
  `DexcomV3DataSource.fetch_recent_readings`,
  `FHIRObservationVitalsSource.fetch_recent_vitals`, and
  `ECFRClient.verify_citation` all previously bypassed Mercury's
  central HTTP egress gate by going through
  `urllib.request.urlopen` directly.  The four methods now route
  through `omni_mercury_engine.security.safe_http.SafeHTTPClient`,
  picking up the scheme allowlist (HTTPS), private-network / IMDS
  block, DNS-rebinding pinning, and redirect refusal for free.
  `SafeHTTPClient.post_form` is a new helper added for the OAuth2
  token endpoint; it mirrors `post_json` but emits a
  form-urlencoded body and validates the response as JSON.
  `requests.HTTPError` raised by `raise_for_status()` is mapped to
  the adapter-specific `DataSourceError` / `ECFRClientError`
  semantics so the public contract is unchanged.
- **Hard URL allowlists for operator-supplied endpoints.**
  `DexcomConfig.__post_init__` now validates `base_url` against
  the two published Dexcom hosts (`https://api.dexcom.com`,
  `https://sandbox-api.dexcom.com`) and raises
  `ConfigurationError` on anything else.  `ECFRClient.__init__`
  similarly restricts `base_url` to
  `https://www.ecfr.gov` via the new
  `ECFRClient.ALLOWED_BASE_URLS` class constant.  Both fixes turn
  a hostile environment variable or operator typo into a hard,
  visible failure at construction time instead of a silent
  redirect of regulatory or PHI traffic.
- **HTTPS-by-default for FHIR PHI traffic.**  `FHIRConfig` rejects
  any non-HTTPS `base_url` unless the operator explicitly sets
  `allow_http=True` (or `FHIR_ALLOW_HTTP=1` in the environment).
  The opt-in is intended exclusively for documented local /
  development FHIR servers; vital-signs observations are PHI and
  must traverse TLS in production.  The flag is forwarded to
  `SafeHTTPClient.get_json`'s `allow_http` parameter and tested
  end-to-end in
  `tests/test_medical_data_sources.py::TestFHIRConfigHttpsPolicy`.
- **`DroneState.__post_init__` shape validation.**  The dataclass
  now rejects malformed `position` / `velocity` / `attitude`
  (3-vectors), `motor_speeds` (4-vector), and `home_position`
  (3-vector when supplied) with a clear `ValueError`.  RADD's
  invariant rules index those positions directly; a mis-shaped
  feed previously bubbled up as an obscure `IndexError` deep in
  the rule loop, which is the exact silent-failure class this
  port was commissioned to eliminate.  Pinned by
  `tests/test_drone_detector.py::TestDroneStateShapeValidation`
  (five cases covering each vector and the optional
  `home_position`).
- **Drone detector module-docstring correction.**  The previous
  module docstring referenced `omni_mercury_engine.anomaly.drone_telemetry`
  as the live-telemetry adapter module; that module does not
  exist in the Mercury tree.  The docstring now states the
  correct contract: the detector is transport-agnostic, callers
  populate `DroneState` from their ingest layer of choice
  (`pyulog.ULog`, `pymavlink`, custom feed), and an integration
  example lives in `docs/drone/SETUP.md`.
- **Medical detector top-level docstrings reconciled with documented
  deviations.**  Two of the medical modules previously claimed
  upstream architectural parity that the round-2 deviations section
  had already retracted:
  - `endocrinology_detector.py` no longer claims the neural
    architecture "matches the original verified implementation".
    The module docstring now references the `CGMAnalyzer`
    trend-head widening and the additive
    `GLP1TherapyMonitor` / `InhaledInsulinMonitor` rules under
    *"Deviations from the original"* explicitly.
  - `anesthesiology_predictor.py` similarly defers to the
    documented deviations for `HemodynamicMonitor`'s explicit
    SpO2 guard and `SmartInfusionController`'s test-introspection
    surface.  The PID controller gains and clinical vital ranges
    remain preserved verbatim, with their ASA / AARC citations
    intact.
- **Test plumbing.**  `tests/test_medical_data_sources.py` and
  `tests/test_osha_anomaly.py` now mock at the `SafeHTTPClient`
  public surface rather than at `urllib.request.urlopen`.  A
  reusable `_build_http_error` helper in the medical tests keeps
  the failure shape consistent with how `SafeHTTPClient` propagates
  `raise_for_status()` failures.  `tests/test_drone_detector.py`
  uses `pytest.importorskip("sklearn")` so the ensemble test
  module is gracefully skipped when the `benchmark-comparison`
  extra is not installed; the sklearn-unavailable fallback path is
  separately covered by
  `TestSklearnEnsemble.test_fallback_when_sklearn_unavailable`.

### Omni-AXA → Mercury port, PR 2 refinements (hard-guardrail review)

Round-2 refinement pass over the four ported domain modules driven by
the repo owner's explicit "no debt-for-debt trade" guardrails.  Every
removed rule from the upstream that the initial port had quietly
dropped is documented here under explicit "Deviations from the
original" subsections per module; every restored rule is pinned by a
named regression test; every cited threshold is re-anchored to a
module-level constant via the new clinical rule-pin harness.

#### Architectural reorganisation

- **`omni_mercury_engine.detectors.drone`** (new subpackage).  The
  ported drone detector now lives in
  `src/omni_mercury_engine/detectors/drone/detector.py` alongside
  Mercury's existing single-domain detector subpackages
  (`marine/`, `economic/`, `energy/`, `geological/`, …).  The
  `omni_mercury_engine.anomaly` package is **retained** (with a
  policy docstring) for future multi-modal anomaly detectors that
  fuse two or more `detectors/<domain>/` outputs into a single
  decision-support stream; no such detector ships in this PR.  All
  existing imports continue to work via the new
  `omni_mercury_engine.detectors.drone` package's `__init__.py`,
  which re-exports the public API
  (`DroneAnomalyDetector`, `DroneFault`, `DroneState`, `FaultType`,
  `MissionPhase`, `get_drone_detector`).  Test imports in
  `tests/test_drone_detector.py` were updated to the new path.

#### `omni_mercury_engine.medical.endocrinology_detector` — Deviations from the original

The initial port dropped three rule groups from the upstream
`endocrinology_detector.py` without CHANGELOG documentation.  All three
groups are restored in this pass with citation-pinned class constants
and regression tests:

- **`SmartInsulinPenMonitor` — large-bolus and daily-total guards (restored).**
  - `MAX_BOLUS_UNITS: Final[float] = 15.0` — fires
    `"Verify dose - risk of hypoglycemia"` and
    `"Consider splitting dose if meal is large"` when
    `dose_units > MAX_BOLUS_UNITS` **and**
    `insulin_type == "rapid_acting"`.  Citation: ADA Standards of
    Care; FDA insulin labeling — large rapid-acting boluses without
    sensitivity verification are a documented hypoglycemia risk.
  - `MAX_DAILY_INSULIN_UNITS: Final[float] = 50.0` — fires
    `"Review insulin sensitivity and dosing regimen"` when
    `daily_total_units > MAX_DAILY_INSULIN_UNITS`.  Citation: ADA
    Standards of Care — total daily insulin above ~50 U warrants a
    sensitivity / regimen review.
  - Both checks accept their inputs as **optional** parameters on
    `monitor_insulin_delivery()` so legacy callers that supplied
    only `recent_doses` / `adherence_rate` / `patient_glucose`
    continue to work unchanged.  Locked by four new tests in
    `TestSmartInsulinPenMonitor`
    (`test_smart_pen_large_bolus_alert_fires_above_15u_rapid_acting`,
    `test_smart_pen_large_bolus_does_not_fire_for_basal`,
    `test_smart_pen_daily_total_alert_fires_above_50u`,
    `test_smart_pen_no_alert_when_fields_omitted`).
- **`InhaledInsulinMonitor` — dose ceiling and technique guards (restored).**
  - `MAX_DOSE_UNITS: Final[int] = 12` — fires
    `"Consider subcutaneous insulin for large doses"` when
    `dose_units > MAX_DOSE_UNITS`.  Citation: FDA Afrezza label,
    Section 5 (Warnings and Precautions).
  - `MIN_TECHNIQUE_SCORE: Final[float] = 0.7` — fires
    `"Retrain on proper inhaler use"` and
    `"May result in suboptimal absorption"` when
    `inhalation_technique_score < MIN_TECHNIQUE_SCORE`.  Citation:
    AARC inhaler-technique guidance.
  - Inputs are optional on `monitor_inhaled_insulin()`; the FEV1
    contraindication still dominates the result when all three
    alerts fire concurrently.  Locked by three new tests in
    `TestInhaledInsulinMonitor`
    (`test_inhaled_dose_ceiling_alert_fires_above_12u`,
    `test_inhaled_technique_alert_fires_below_0_7`,
    `test_inhaled_contraindication_still_dominates`).
- **`GLP1TherapyMonitor` — duration-aware titration and GI handling (restored).**
  - `A1C_ESCALATION_WEEK: Final[int] = 12` /
    `A1C_INADEQUATE_DROP_PERCENT: Final[float] = -0.5` — when
    `a1c_change_percent > -0.5` **and** `duration_weeks >= 12`, the
    monitor recommends dose escalation.  Citation: ADA
    pharmacological guidance; FDA semaglutide / liraglutide
    labeling.
  - `WEIGHT_LOSS_REVIEW_WEEK: Final[int] = 16` /
    `WEIGHT_LOSS_TARGET_KG: Final[float] = 2.5` — when
    `abs(weight_loss_kg) < 2.5` **and** `duration_weeks >= 16`, the
    monitor recommends a diet / exercise review and dose escalation
    if tolerated.
  - GI side-effects: when `side_effects` contains `"nausea"` or
    `"vomiting"` (case-insensitive), the monitor emits
    `"Take with food, slower dose titration; consider antiemetics
    if severe"`.
  - `duration_weeks` is an **optional** parameter (default `0`).
    The pancreatitis discontinuation rule still dominates the
    `continue_therapy` flag when any of the new rules fire.  Locked
    by four new tests in `TestGLP1TherapyMonitor`
    (`test_glp1_dose_escalation_recommended_at_week_12_inadequate_a1c`,
    `test_glp1_no_escalation_before_week_12`,
    `test_glp1_gi_side_effects_trigger_titration_advice`,
    `test_glp1_pancreatitis_still_dominates`).
- **`CGMAnalyzer` — trend-head width (kept widened, docstring
  corrected).**  The upstream architecture has the trend head at
  `hidden_dim * 2 -> 32 -> 1`; Mercury's port widens it to
  `hidden_dim * 2 -> 64 -> 1` to match the glycemic classifier's
  hidden width.  Parameter count is approximately equal (~155K) but
  the resulting weights are **not interchangeable** with upstream
  checkpoints — any prior pretrained weights would need to be
  re-trained for this layout.  The class docstring has been updated
  to drop the previous "matches the verified Omni-AXA
  implementation" wording, which was directionally wrong.  Locked
  by the new
  `test_cgm_analyzer_parameter_count_is_approximately_155k` guard
  (asserts `145_000 <= count_cgm_parameters() <= 165_000`).
- **`SmartInsulinPenMonitor` — adherence scalar (intentionally
  simplified).**  Mercury accepts a single
  `adherence_rate ∈ [0, 1]` scalar rather than the upstream's
  `doses_taken / doses_prescribed` ratio because the
  `CGMDataSource` / vendor-adapter contract returns this as a
  pre-computed daily fraction; recomputing it inside the monitor
  would invite a divide-by-zero on partially-reported days.  The
  low-adherence alert still fires below `0.8`.

#### `omni_mercury_engine.medical.anesthesiology_predictor` — Deviations from the original

- **`HemodynamicMonitor.spo2_threshold` (kept explicit guard).**  The
  port keeps the explicit
  `intervention_needed = overall_risk > 0.6 or spo2 < spo2_threshold`
  short-circuit so that any sub-92 % SpO₂ reading triggers
  intervention even when the per-vital risk weights happen to
  average out below 0.6.  Citation: ASA standards for basic
  anesthetic monitoring.
- **`HemodynamicMonitor` bradycardia threshold (kept tightened at
  ≥ 0.5).**  Sub-50 bpm HR contributes `0.5 * (50 - hr) / 50`
  capped at `1.0`, so a reading of 45 bpm contributes 0.05 of risk
  rather than the upstream's 0.10 — the upstream value risked
  overcounting routine athletes-at-rest readings while the OR is
  otherwise stable.  Locked by
  `tests/test_anesthesiology_predictor.py::TestHemodynamicMonitor::test_bradycardia_contributes_to_risk`.
- **Synthetic vitals generator removed from production paths.**  The
  upstream module included a `VitalDBClient` synthetic-data
  generator that emitted fabricated MAP / HR / SpO₂ traces.  Mercury
  refuses to operate on synthetic vitals in production: the
  predictor raises `ConfigurationError` when
  `enable_hemodynamics=True` and no `VitalsDataSource` adapter is
  supplied.  Tests use sanitized FHIR-R4 `Observation` fixtures
  (`tests/fixtures/medical/fhir_observation_vitals.json`).

#### `omni_mercury_engine.compliance.osha_anomaly` — Deviations from the original

- **Heat-index regression direction (docstring correction).**  The
  initial port's module-level docstring claimed the simplified
  `T + 0.5*RH` heuristic "under-reported heat stress at high
  humidity."  The worked example actually shows **over**-reporting
  (the heuristic returns ~130 °F at T=95 °F / RH=70 % while the
  Rothfusz regression returns ~122 °F, an 8 °F over-report).  The
  docstring has been rewritten to capture both directions: high-RH
  over-reporting and low-RH under-reporting.  Three new NWS
  reference-point tests
  (`test_heat_index_known_values[80F/40%RH]`,
  `[95F/70%RH]`, `[100F/10%RH]`) pin the Steadman branch, the
  unadjusted Rothfusz branch, and the low-humidity adjustment
  branch respectively.
- **`OSHAComplianceAnomaly` legacy alias removed.**  The upstream
  module ended with
  `OSHAComplianceAnomaly = OSHAComplianceDetector`.  A repository-wide
  `git grep OSHAComplianceAnomaly` confirmed that the only external
  reference lives in the upstream Omni-AXA tree and Mercury's own
  `docs/ARCHITECTURE.md`; no Mercury runtime code, tests, or
  downstream callers depend on the alias.  The alias is **removed
  without replacement** in the Mercury port.  Importers that hit
  the missing name will get a clear `ImportError`; integrators
  should import `OSHAComplianceDetector` directly.
- **`ECFRClient` rate-limit handling (clarified, not enforced).**  The
  class docstring previously implied the client enforced the
  published 60 req/min/IP guidance; in fact it only caches.  The
  docstring has been rewritten to state explicitly that the client
  does **not** enforce the limit programmatically — operators
  running batch audits should cap concurrency at the call site
  (e.g. a thread / asyncio semaphore around `verify_citation()`).
  The in-process cache reduces duplicate lookups during a single
  audit run and is the primary mechanism by which Mercury stays
  under the published limit.

#### `omni_mercury_engine.detectors.drone.detector` — Deviations from the original

- **`_analyze_log_entry` keyword scoring (extended, not narrowed).**
  The upstream scored only mechanical-fault keywords
  (`critical`, `error`, `warning`, level guards, `timeout`,
  `connection lost`).  Mercury extends this with three
  Mercury-specific signals, weights tuned so operationally-noisy
  lines (routine "signal weak" advisories, expected
  `intrusion_detection` self-tests, transient thermal notes) stay
  below the `score > 0.75` fault gate while genuinely anomalous
  lines cross it:
  - `+0.55` for `attack | intrusion | unauthorized` (security).
  - `+0.40` for `overheat[ing]` (thermal).
  - `+0.35` for `signal lost` (telemetry loss).
  Pinned by
  `tests/test_drone_detector.py::TestDronLomalyLogs::test_log_keyword_scoring_does_not_overflag_routine_lines`,
  which feeds three benign-but-noisy lines and three genuinely
  anomalous lines and asserts each side of the threshold.

#### Testing

- **`tests/test_clinical_rule_pins.py`** (new, rule-vs-citation
  harness).  Pins every cited FDA / ADA / NWS / ASA / AARC
  threshold against the module-level constant the citation refers
  to via a parametrised pin table.  Regressions that change a
  comparison operator (e.g. `<` → `<=`) or threshold value
  (e.g. `70.0` → `70`) surface immediately with the citation URL
  alongside the failing assertion.  A second test prints the live
  pin table during `pytest -v` so the mapping is visible per run.
  Initial coverage: nine pins across
  `endocrinology_detector` (seven), `anesthesiology_predictor`
  (one), and `osha_anomaly` (one).
- **Weekly network-test cadence.**  `@pytest.mark.network`-marked
  tests now run weekly via
  `.github/workflows/network-tests.yml` (Mondays 13:00 UTC, plus
  `workflow_dispatch`).  Failures surface as a separate CI signal
  so external-endpoint schema drift (Dexcom v3, FHIR R4
  `Observation`, eCFR Title 29, NIST CSF Reference Tool) is caught
  within seven days even though those tests auto-skip on every
  per-PR run.

### FEMA Disaster loader — label-polarity correction (closes "known-broken" footnote item)

- **`FEMADisasterLoader._select_anomaly_polarity`** enforces the
  minority-as-anomaly convention used everywhere else in Mercury.
  Historical OpenFEMA records make "DR + multi-program" the
  *majority* class on most slices (major hurricanes / floods
  routinely activate IA, PA, and HM together), so handing those
  records label==1 inverted the anomaly detector and AUC drifted
  below 0.5.  The loader now logs a loud `INFO` line when the
  polarity flip kicks in and exposes the result via the new
  `loader.labels_inverted` property so benchmark reporters can
  surface the flip alongside their AUC numbers.  Behaviour is
  identical on the synthetic-fallback path so CI runs (which lack
  network access to the OpenFEMA API) exercise the same code
  path as production.  Locked by
  `tests/datasets/test_disaster.py::TestFEMAInvertedScoresCorrection`
  (5 regression tests covering majority-inversion, minority-no-op,
  empty mask, initial property state, and the real-data-shape
  processing pipeline).

### Dataset reachability harness — two lanes for the unreachable-11

- **`tests/datasets/test_unreachable_loaders_offline.py`** — runs
  in every CI lane.  Parametrised across all 11 historically-
  unreachable loaders (SMAP, MSL, CICIDS-2017, MIT-BIH, UCR, SWaT,
  WADI, USGS Geochemistry, NOAA StormEvents, NOAA ERDDAP, FEMA
  HazardMitigation), it asserts each loader (a) constructs against
  a valid `DatasetConfig`, (b) populates the metadata contract
  (`DATASET_NAME` / `DATASET_URL` / `LICENSE` / `CITATION`),
  (c) fails *loudly* (`DataSourceUnavailableError` /
  `ConnectionError` / `OSError`) when every HTTP surface is
  monkeypatched to simulate an upstream outage — never with a
  silent `False` return.  This is the regression contract that
  upgrades the loaders from "untested under outage" to "tested to
  fail loudly under outage".
- **`tests/datasets/test_unreachable_loaders_network.py`** —
  marked `@pytest.mark.network` and auto-skipped by
  `tests/conftest.py` unless `MERCURY_NETWORK_TESTS=1` is set.
  Calls the real `download()` against the upstream provider.  Run
  nightly via the new `.github/workflows/dataset-reachability.yml`
  workflow (04:17 UTC) with `MERCURY_ALLOW_SYNTHETIC=0` and
  `MERCURY_NETWORK_TESTS=1` so the synthetic fallback cannot mask an
  outage.
- **Coverage-drift gate.**  Both files include a
  `test_harness_covers_*_loaders` assertion that pins the matrix
  to exactly 11 entries.  Adding or removing a loader from the
  unreachable set fails the build unless `CHANGELOG.md`,
  `docs/DATASOURCES.md`, and both harness files are updated in
  the same commit.

### DATASOURCES.md — SafeHTTP DNS-fails-closed discoverability (§6 P1)

- **New `Operating the SafeHTTP gate` section in
  `docs/DATASOURCES.md`.**  Documents the intentional
  DNS-resolution-fails-closed behaviour of
  `SafeHTTPClient.validate_url(..., user_configured=True)` —
  previously memory-only knowledge that operators kept
  re-discovering when an internal mirror's hostname couldn't be
  resolved by the container's stub resolver.  Section names the
  supported remediations in preference order (fix the resolver / use
  an already-plumbed `SafeHTTPClient(..., allow_private=True)` call
  path / prefer `local_path` where a loader exposes it) and points at
  the regression test that locks the
  behaviour (`tests/loaders/test_base_loader.py:99`).  Cross-
  references `docs/MIGRATION-1.6-to-1.7.md` §1 so operators
  trying to re-enable the v1.6 `allow_untrusted=True` workaround
  see the migration path immediately.

### Production-mode primitive — `MERCURY_ENV` (new in v1.7.0)

- **New module `omni_mercury_engine._env`.**  Introduces a single
  canonical environment-mode flag, `MERCURY_ENV` (`development`
  default, `production`), and a shared fail-closed helper API
  (`get_mercury_env`, `is_production`, `require_real_component`,
  `MercuryProductionConfigError`).  Modules that historically had
  development-friendly stub fallbacks now have a single, uniform
  place to refuse to silently degrade when an operator opts into
  production mode.  The PQC import gate
  (`_pqc_gate._enforce_pqc_production_gate`) stays orthogonal — it
  has its own hard-required-build contract independent of the
  development/production distinction — so production deployments
  typically set **both** `MERCURY_ENV=production` and
  `AMA_REQUIRE_REAL_PQC=true`.  An unknown value such as
  `MERCURY_ENV=prod` raises `MercuryProductionConfigError` rather
  than silently falling through to development mode, so deployment
  typos are loud.  Locked by `tests/test_env.py`.

### Narrative voice — `MercuryVoice` LLM initialisation fixed (PR for #210 follow-up)

- **`narrative/voice.py:_init_llm` no longer crashes on
  `MercuryVoice(enable_llm=True)`.**  The pre-1.7.0 implementation
  unconditionally instantiated `MockLLMAdapter`, which started
  hard-failing at construction once the Phase 2 audit cure landed.
  The surrounding `except ImportError` did not catch the resulting
  `NotImplementedError`, so any caller that asked for LLM-enhanced
  narration got an unhandled exception.  v1.7.0 wires the real
  provider selection that was always intended via a new
  `llm_provider=` / `llm_model_name=` / `llm_revision=` /
  `llm_api_key=` / `llm_base_url=` parameter set on `MercuryVoice` and
  `create_mercury_voice`.  Behaviour matrix:
  - `enable_llm=False`: unchanged pure-template fast path.
  - `enable_llm=True, llm_provider="<supported>"`: delegates to
    `models.foundation.llm_adapter.create_llm_detector` and stores
    the underlying adapter on `self._llm_adapter`.
  - `enable_llm=True` with no provider, `MERCURY_ENV=production`:
    raises `MercuryProductionConfigError` with a remediation hint.
  - `enable_llm=True` with no provider, `MERCURY_ENV=development`:
    logs a WARNING and downgrades to template-only narration
    (`self._llm_adapter = None`).
  - `llm_provider="mock"`: always raises
    `MercuryProductionConfigError` — `MockLLMAdapter` hard-fails at
    construction by design and the rejection is now at the
    `MercuryVoice` call site rather than two frames away.
  - Unknown `llm_provider` value: raises a clean `ValueError`
    naming every supported provider, instead of routing through
    `create_llm_detector`'s legacy "unknown → mock" fallback and
    blowing up on `NotImplementedError` later.
  - HuggingFace requires an explicit `llm_model_name`; remote
    HuggingFace IDs also require `llm_revision=<40-char SHA>` so
    `SafeHFLoader` can enforce reproducible model loading.
  Locked by `tests/narrative/test_voice_llm.py`; full
  `tests/narrative/` suite (92 tests) is green.

### Migration guide — `docs/MIGRATION-1.6-to-1.7.md`

- Consolidated migration notes for v1.6.x → v1.7.0: PR #210's
  `allow_untrusted=True` removal, σ_Immutable hard-gate semantics
  (already shipped at every boundary surface; documentation
  reconciled with code reality), the new `MERCURY_ENV` primitive,
  and the explicit `llm_provider=` requirement on `MercuryVoice`.
  Each section names the regression test that locks the new
  behaviour.

### Documentation reconciliation

- **`docs/ROADMAP.md`.**  Capability-table row #2 (Biometric
  Modalities) and the §2 "Status" block updated to reflect the
  narrative-voice fix.  The "Ethics enforcement" cross-cutting row
  updated to reflect that σ_Immutable hard-gate promotion is
  complete at every boundary surface (the previous "deferred to a
  follow-up PR" wording was stale — `OmniMercuryEngine`,
  `CognitiveOrchestrator`, and `NeuroSymbolicHub` all raise
  `check="sigma_immutable"` / `check="gosnn_unavailable"` today,
  with regression coverage in `tests/ethical/test_hard_enforcement.py`).

### Security — HTTP escape hatch removed (PR #210)

- **`allow_untrusted=True` is removed from every `SafeHTTPClient` method**
  (`validate_url`, `_request`, `get`, `get_bytes`, `get_json`, `get_text`,
  `post_json`) and from the dataset/loader helpers that wrap it
  (`datasets.base.http_get_with_retry`, `loaders.base.BaseDomainLoader._fetch_url`).
  The kwarg had no production caller; it was a per-call bypass of the
  `TrustedEndpoints.TRUSTED_DOMAINS` allowlist that could be misused
  to pivot through an off-allowlist host while staying inside the
  hardened transport. PR #210 deletes it from the public API and
  asserts the removal via signature tests.
- **Migration path for operators who were using it:** call
  `SafeHTTPClient` directly with `user_configured=True` so the
  private-network / IMDS gate fires explicitly. For RFC1918
  destinations on a private VPC, additionally pass `allow_private=True`
  at the `SafeHTTPClient` call site; dataset loaders do not accept a
  generic `allow_private` preprocessing key unless a specific loader
  explicitly documents one. The IMDS / loopback / multicast /
  reserved / CGNAT ranges remain in the always-blocked set even then. See
  `tests/security/test_safe_http.py::TestMigrationFromAllowUntrusted`
  for the documented replacement.
- **Loader retry-exhaustion now chains the underlying exception.**
  `BaseDomainLoader._fetch_url` wraps a final failure as
  `ConnectionError(...) from last_exc` so the operator-facing
  traceback names the real cause (timeout, refused connection, 5xx)
  rather than burying it.
- **Test coverage added** for the migration path, the obsolete-kwarg
  removal at every public wrapper, the IPv6 always-blocked set
  (`::1`, `fe80::/10`, `ff00::/8`, `::`), the multi-IP failover path
  in `_request`, the `allow_redirects=False` transport contract, the
  `validate_url` DNS short-circuit for trusted https URLs, and the
  IP-literal short-circuit in `_resolve_ips`.

### Benchmark refresh (PR #210, via PR #203 source SHA `0f584529`)

- Mean ROC-AUC `0.8440` → `0.8464`; median ROC-AUC `0.9097` → `0.9100`;
  mean Oracle F1 `0.6383` → `0.6441`; datasets successful / total
  `64/64` → `65/65`. Computed on commit `ffafd17` (run timestamp
  `2026-05-14T22:14:04Z`). Regression gates unchanged: AUC ≥ 0.68 and
  F1 ≥ 0.50.


### Branch Reconciliation (v1.6.0 stack — PRs #188–#191)

- **PR #189** (Devin session `a7bea1074fbd420f9c9af8e6b3eea01f`): v1.6.0
  corrective sweep — PQC gate, FallbackChain ethical re-raise, RNG cure
  (138 sites across federated/climate/disaster/security), type-redef
  elimination (7 files from `copilot/refactor-34`), documentation refresh.
  Absorbs PR #188 (`claude/organize-project-directory-IIqcr`) in full.
- **PR #190** (Claude session `audit-pqc-fallback-chain-Po2fD`): tracked-debt
  sweep — remaining 17 type-redef suppressions eliminated across 8 files,
  62 unseeded `np.random` sites cured in cognitive/ and models/ modules,
  25-test RNG reproducibility regression suite added.
  `DifferentialPrivacy` dual-API (`rng=` / `seed=`) bridge.
- **PR #191** (Claude session `in-house-anomaly-datasets-vLn30`): dataset
  loader hardening — FEMA OData ISO-8601, NOAA Storm filename discovery,
  ERDDAP date offsets, EPA/GSOD year fallback, NSL-KDD/CICIDS/MITRE mirror
  failover. `http_get_with_retry` shared helper with SSRF default-deny.
  `_ConstReplacer` type-narrowing fix in `core/three_r_mechanism.py`.
- **PR #188** (Copilot session `organize-project-directory-IIqcr`): fully
  absorbed into PR #189; closed with attestation of commit SHAs
  (e198858, 12d2887, cc70fc9, 7e20c5b, c032c91, fd51186, 0553f1a).

### Wave B — σ_Immutable promoted to hard gate (deferred from PR #161)

- **σ_Immutable is now the second mandatory hard ethical gate** at every
  engine / hub / orchestrator decision boundary.  The Wave A landing
  reserved `check="sigma_immutable"` and `check="gosnn_unavailable"` in
  `EthicalConstraintViolationError`'s schema but did not raise them
  from any code path; Wave B flips the contract:
  - `OmniMercuryEngine._enforce_ethics_at_boundary` now runs both
    `BenevolenceScorer.enforce` *and* `SigmaImmutableGate.enforce`
    using a benevolence→σ-band projection helper
    (`security.sigma_immutable_gate.project_benevolence_to_sigma_band`).
  - `OmniMercuryEngine.detect_with_fusion` raises
    `EthicalConstraintViolationError(check="sigma_immutable")` when
    the trained σ_Immutable network scores below threshold for the
    full GOSNN scalar state, and
    `EthicalConstraintViolationError(check="gosnn_unavailable")` when
    GOSNN itself cannot run.  The previous
    "fall back to `gosnn_metadata.fallback_mode=True`" path is gone.
  - The `enable_gosnn` parameter on `detect_with_fusion` and
    `detect_with_fusion_calibrated` is renamed to the private
    `_enable_gosnn` so production callers can no longer skip the
    σ_Immutable gate.  Unit tests that need to bypass GOSNN must
    additionally set the auditable module-level flag
    `omni_mercury_engine.engine._GOSNN_TESTING_BYPASS`.
  - `NeuroSymbolicHub.predict` and `CognitiveOrchestrator.analyze`
    already wired σ_Immutable in Wave A; their σ-vector builders now
    source the layout from a single source of truth.
- **Single source of truth for the σ_Immutable layout.**
  `security.sigma_immutable_gate` exports
  `SIGMA_IMMUTABLE_DIM` (256), `SIGMA_ETHICAL_BAND_END` (27),
  `SIGMA_USED_BAND_END` (180) and the public alias `CORPUS_USED_DIM`.
  The corpus, the trainer, the orchestrator, the neurosymbolic hub,
  and the KAT regression test all import from this single location;
  the previous duplicate hard-coded ``180`` literals are gone.
- **σ_Immutable hard-gate regression suite.** The Wave A
  `tests/ethical/test_hard_enforcement.py::TestReservedChecksWaveB`
  block already pinned the post-Wave-B contract; they were
  `@pytest.mark.xfail(strict=True)` in Wave A and are positive tests
  here.  No additional xfail markers ship in Wave B.
- **Decision-boundary contract documented** in
  `src/omni_mercury_engine/ethical/__init__.py` and
  `ARCHITECTURE.md`: every public detect / analyze / predict surface
  must run BenevolenceScorer **and** σ_Immutable in order, or fail
  closed with the matching `check=…` value.

### Documentation

- **Project-wide documentation refresh (2026-05-05).** All 29 markdown
  files audited against current code state at v1.6.0; corrective
  edits landed in a single sweep on
  `claude/organize-project-directory-IIqcr`:
  - `SECURITY.md` — Supported-Versions table rewritten (was listing
    1.5.x as current with no v1.6.x entry); now declares 1.6.x as
    current, 1.5.x as previous (critical-CVE only), <1.5 EOL.
    `Last Updated` refreshed.
  - `README.md` — GOSNN section's "Ethical Gating" key-feature
    bullet rewritten to document the Wave B σ_Immutable hard-gate
    contract (was describing the old `≥0.93 with configurable
    fallback for medical domains` and a now-deleted `gosnn_metadata.fallback_mode=True`
    silent-fallback path). New decision-boundary contract callout
    added under the Status banner. Header date and footer
    `Last updated` refreshed.
  - `docs/MATH_SPEC.md` — new §2.1.5 "σ_Immutable Hard Gate (Wave B,
    PR #179)" added: boundary-contract piecewise definition, σ
    vector layout (`SIGMA_IMMUTABLE_DIM=256`,
    `SIGMA_ETHICAL_BAND_END=27`, `SIGMA_USED_BAND_END=180`),
    threshold provenance, test-only bypass note, composition with
    the sigmoid benevolence gate. Date refreshed.
  - `docs/index.md` — landing page rewritten to surface the dual-gate
    ethical contract, AMA Cryptography sole-backend hard-require,
    honest-benchmark framing (64/75), and pickle-removal up front.
  - `docs/ROUTING_GUIDE.md` — top-of-file callout that hard ethical
    gates run *inside* the prediction call and **must not** be
    masked by fallback handlers. "Fallback only applies to
    data-source / connectivity / latency failures" clarification
    added to the Overview.
  - `docs/BENCHMARKS.md`, `docs/DATASOURCES.md`,
    `docs/LIVE_DATA_VALIDATION.md` — reconciled the local 51/55
    figure with the canonical 64/75 reproducibility set as **two
    distinct measured baselines**: 64/75 = Mean AUC **0.8285** /
    Mean Oracle F1 **0.6370** (current README headline after the
    Oracle pipeline fix and dataset expansion); 51/55 = Mean AUC
    **0.8030** / Mean Oracle F1 **0.5886** (legacy CI
    regression-gate floor that the 64/75 run improved on).
  - `docs/INSTALLATION.md` — added "Post-Quantum Cryptography
    backend" section documenting the AMA Cryptography hard-require
    (`AMA_REQUIRE_REAL_PQC=true`); added C-toolchain / CMake
    requirements.
  - `docs/DEPLOYMENT.md` — Ethics-audit-failures-in-CI subsection
    rewritten: the gate is non-advisory; added remediation guidance
    for `check="sigma_immutable"` and `check="gosnn_unavailable"`
    failures.
  - `CONTRIBUTING.md` — "What NOT to Contribute" tightened to
    explicitly forbid weakening the Wave B dual-gate contract,
    reintroducing `pickle`, or adding a non-AMA Cryptography PQC
    backend. Document Version → 2.4, `Last Updated` refreshed,
    `Applies to` row added.
  - `ARCHITECTURE.md` — "System Scale" footer block re-derived from
    the current tree (42 subpackages verified, 276 test files /
    6,300+ tests, ~280,950 LOC); replaced unverifiable
    "85%+ coverage" with "measured per release".

### Code Quality

- **CI coverage floor raised from 10% to 15% / 35% (2026-05-05).**
  The previous 10% / 10% (`COVERAGE_THRESHOLD_CORE` /
  `COVERAGE_THRESHOLD_FULL`) floor in `.github/workflows/ci.yml`
  was scaffolding from the era when MyPy strict-mode churn was
  repeatedly breaking CI; that churn has been resolved (strict mode
  is on for `omni_mercury_engine.*` and the 569 surviving
  suppressions are `[unused-ignore]`-paired so they auto-clear as
  third-party stubs improve). An interim CI run on this branch
  measured the actual full-suite line coverage at **36.03%**, so
  the honest CI floor is set just below that — 35 — to defend the
  measured baseline without immediately failing CI for unrelated
  reasons. CORE is set to 15 since the core job runs a strictly
  smaller subset of tests against the full source tree. The 85
  `pyproject.toml [tool.coverage.report] fail_under` target stays
  in place as the strict aspirational nightly bar; a dedicated
  coverage push is planned to close the 36 → 85 gap. `.coveragerc`
  no longer sets `fail_under` at all (it used to, at the same
  value as the full-suite job, but that made every `pytest --cov`
  invocation across the repo silently inherit the same floor —
  including partial-suite jobs like `neuro-symbolic-tests` whose
  coverage shape is different); per-job CI floors are configured
  via `--cov-fail-under=${{ env.COVERAGE_THRESHOLD_* }}` flags in
  `.github/workflows/ci.yml` so they apply only to the jobs that
  explicitly opt in.
- **Differential-privacy noise no longer draws from global `np.random`.**
  `federation/privacy.py::DifferentialPrivacy` now owns a per-instance
  `np.random.Generator` (constructed with `default_rng(rng_or_None)`)
  and routes every noise draw — vector statistics, the symmetric
  precision-matrix perturbation, the log-determinant scalar — through
  it.  Callers can pass an explicit `rng` for audited / reproducible
  deployments; the legacy `np.random.normal(...)` global-state path
  is removed entirely.  This was a real DP-guarantee defect: a caller
  that called `np.random.seed(...)` elsewhere in the same process
  could de-randomise the privacy noise without touching this module.
- **Federated learning RNG plumbed through.** `federated_learning`:
  - `SGDTrainer` and `FedProxTrainer` now accept `seed: int | None`
    and shuffle minibatches via a per-instance `Generator`
    (`np.random.permutation` global-state calls removed).
  - `ClientManager` accepts `seed`; `random` and `weighted` selection
    strategies now use `self._rng.choice(...)` (`np.random.choice`
    global-state calls removed). Cross-organisational federated
    rounds can now be reconciled deterministically by passing the
    same seed to every participant.
  - `FederatedAnomalyDetector` accepts `seed` and draws the initial
    server weight vector via `self._rng.standard_normal(...)`
    (was `np.random.randn(...) * 0.01`).
- **`MercuryEquationEngine` (`core/double_helix_engine.py`) RNG
  plumbed through.**  The ethical-matrix initialisation, Hamiltonian
  symmetric matrix, Boltzmann-sampling noise, simulated-annealing
  exploration and Lyapunov-chaos perturbation all now draw from a
  per-instance `np.random.Generator` (constructor `seed: int | None`).
- **Tests:**
  - `tests/test_vlm_detectors.py::test_anyanomaly_detect_mock`
    deleted. `MockLVLMBackend` is intentionally a hard-fail by
    design (Phase 2 audit cure: silent mock degradation is not
    permitted in production), so any test that exercises it would
    have to fight that design. The dead `backend` field on
    `AnyAnomalyConfig` is also removed; backend selection has
    always been driven by the inherited `VLMConfig.model_type`
    field, which the factory in `lvlm_backends.get_lvlm_backend`
    consumes.
  - `tests/detectors/test_enhanced_geological_detectors.py` —
    `test_recursion_synapse_integration`,
    `test_resonance_synapse_integration`, and
    `test_refactoring_synapse_integration` previously skipped on
    `if not detector.enable_X` even though their fixtures explicitly
    construct the detector with that flag set, so the skip path was
    hiding an integration test that should have always been running.
    Replaced the `pytest.skip(...)` with an `assert` that surfaces
    the integration coverage instead of silently masking it.
- **Type-redefition suppressions eliminated (7 files).**
  Integrated from `copilot/refactor-34-unpaired-type-redefinitions`:
  function-stub fallbacks (`def Foo(*args): raise ImportError`) replaced
  with proper stub classes that preserve type identity, and optional-dep
  import guards restructured to `if TYPE_CHECKING: ... else: try: ...`
  so mypy sees the real types without runtime cost.  Files refactored:
  `detectors/acceleration_dynamics.py`,
  `detectors/dimensional.py`,
  `detectors/geological/disaster_detectors.py`,
  `integrations/mercury_amacrypto.py` (8 suppressions — the heaviest
  single file), `medical/abms_disciplines.py`,
  `ml/harmonic_encoder.py`, `safeguards/nano_safeguards.py`.
- **RNG cure extended to dataset generators.**
  Synthetic fallback generators in `datasets/disaster.py` (FEMA
  declarations, hazard mitigation) and `datasets/security.py` (NSL-KDD,
  CICIDS, MITRE ATT&CK) converted from `np.random.seed()` +
  `np.random.*()` global-state draws to per-method
  `np.random.default_rng(seed)` instances. CICIDS flow generators
  (`_generate_benign_flow`, `_generate_dos_flow`,
  `_generate_portscan_flow`, `_generate_attack_flow`) now accept an
  explicit `rng` parameter threaded from the caller.  All
  `max_samples` down-sampling sites also converted.
- **Tracked debt (partially resolved):** The original 35 unpaired
  `# type: ignore[no-redef]` suppressions are now reduced to
  ~28 across **8** remaining files:
  `core/adaptive_fusion.py`, `core/base.py`, `core/config.py`,
  `core/fusion.py`, `core/three_r_mechanism.py`,
  `security/crypto_api.py`, `security/cyber_fortress.py`,
  `utils/resilience.py`.  These should be refactored to a
  Protocol-or-inheritance pattern in a focused follow-up PR.
  ~250 unseeded `np.random` call sites remain in cognitive,
  model, and remaining dataset modules; tracked for a dedicated
  RNG-cure follow-up.

### Tooling

- **`benchmark.yml` push paths filter removed; auto-commit ordering
  reorganised; root-cause of broken auto-commit identified
  (2026-05-05).** Three findings:
  - The `on.push.paths` filter on the benchmark workflow restricted
    triggers to `benchmarks/**` or `src/omni_mercury_engine/**`
    only.  Recent merges to `main` (e.g. PR #187 — Dependabot
    consolidation, helm/Dockerfile/workflows-only) didn't touch
    those paths, so the workflow never ran on those pushes and the
    auto-commit step never had a chance to fire.  Filter removed:
    every push to `main` / `develop` now triggers the benchmark
    workflow regardless of which paths changed.
  - The "Update README" + "Commit and push" steps were moved to
    run immediately after the gating mercury benchmark step (and
    its artifact upload), before the supplementary empirical and
    seven-axis steps.  This is a structural cleanup: when the push
    issue below is resolved, mercury+README will commit in a
    focused single commit with the canonical
    `ci(benchmark): persist latest run` subject, and seven-axis
    will commit as a follow-up.
  - **Root cause of "Latest Benchmark Results" never updating
    on `main`:** the bot's `git push origin HEAD:main` is rejected
    by `main`'s branch protection ruleset (verified from run #660
    log).  The four blocking rules: "Changes must be made through
    a pull request", "22 of 22 required status checks", "Code
    scanning is waiting for results from CodeQL", and "Commits
    must have verified signatures" (the latter found 1 violation:
    the unsigned `github-actions[bot]` commit).  Every prior
    auto-commit attempt has been rejected for the same reasons,
    which is why `benchmarks/mercury_benchmark_results.json` has
    never appeared on `main` and the README "Latest Benchmark
    Results" block stays in `_pending first auto-commit_` state.
- **Benchmark auto-commit replaced with PR-based API flow
  (Option A).** New `scripts/persist_benchmark_to_pr.py` uses the
  GitHub Git Database API directly (`/git/blobs`, `/git/trees`,
  `/git/commits`, `/git/refs`) to commit on the
  `ci/benchmark-results` feature branch. The script itself does
  **not** supply a detached `signature` field in the
  `POST /git/commits` payload; rule 4 (*commits must have verified
  signatures*) is satisfied externally — GitHub auto-signs commits
  made via the Git Database API on behalf of `github-actions[bot]`
  with its web-flow signing key, but **only when the call is
  authenticated with the workflow's `GITHUB_TOKEN`** (or a GitHub
  App installation token that is allowed to act as the bot).  If
  the workflow is reconfigured to use a Personal Access Token via
  `BENCHMARK_BOT_TOKEN`, those commits will land **unverified**;
  the alternative is to extend the persister to compute and submit
  a detached PGP/SSH signature (out of scope for this branch).  The
  script then opens
  a PR into `main` (satisfies rule 1, *Changes must be made
  through a pull request*).  The 22 required status checks (rule
  2) run on the PR.  Both the mercury+README commit and the
  follow-up seven-axis commit re-use the same feature branch — the
  script force-updates the ref, so a single PR collects both
  changes rather than opening two PRs.
  - **Caveat — initial CI on the bot PR is pending:** GitHub
    deliberately does not trigger downstream workflows from events
    caused by the default `GITHUB_TOKEN`, so the 22 required
    checks land **pending** on the bot PR.  Unblock by re-running
    the CI workflow manually on the PR head, or by pushing an
    empty commit on the persistence branch as a maintainer.  Do
    **not** swap `GITHUB_TOKEN` for a Personal Access Token to
    break the loop-prevention: the persister does not submit a
    detached commit signature, and PAT-authenticated Git Database
    commits are not auto-signed by GitHub — they land `Unverified`
    and fail rule 4 of the protection ruleset.  The persister
    steps in `.github/workflows/benchmark.yml` therefore export
    `GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}` unconditionally.
    An earlier iteration of the workflow used a
    `secrets.BENCHMARK_BOT_TOKEN || secrets.GITHUB_TOKEN`
    expression to leave room for a future GitHub App installation
    token, but the expression cannot distinguish a PAT from an
    App token by secret name; the fallback was removed in this
    branch.  Wiring an installation token requires a separately
    named secret (e.g. `BENCHMARK_BOT_APP_TOKEN`) and is a
    focused-PR job.
  - **Auto-merge is not enabled by default.**  The persister
    script accepts `--enable-automerge --merge-method squash` for
    deployments that want the PR to merge as soon as required
    checks pass; the workflow does not pass that flag yet — that
    is an explicit maintainer policy decision.
- **`scripts/update_readme_benchmarks.py` reads canonical metadata**
  (`data["metadata"]["git_commit"]` / `data["metadata"]["timestamp"]`)
  before falling back to the older flat `data["commit"]` /
  `data["timestamp"]` keys, so the README block now renders the
  correct provenance for runs produced by `benchmarks/mercury_benchmark.py`.
  Regression: `tests/scripts/test_update_readme_benchmarks.py`.
- **Benchmark workflow self-trigger guard.**
  `.github/workflows/benchmark.yml` adds a defence-in-depth job-level
  ``if:`` that refuses to run when the head commit was written by the
  workflow's own auto-commit step (subject prefix
  ``ci(benchmark): persist latest run`` and the ``[skip benchmark]``
  marker).  GitHub's ``[skip ci]`` marker is still emitted as the
  primary skip mechanism.

### Anomaly Detection

- **21-probe Anomaly Math Arrest is the dominant path** (Phase 2 ITEM
  2). End-to-end audit confirms `AnomalyMathArrest` registers all 21
  mathematically-independent probes and discriminates injected
  anomalies across `earthquake` / `tsunami` / `pandemic` / `marine` /
  `geomagnetic` / `default` domain-affinity orderings. No live
  `IsolationForest` import or instantiation remains in `src/` — the
  only references are documentation strings explaining what the
  ensemble replaced. Regression suite:
  `tests/detectors/test_math_arrest_dominant_path.py` (11 tests).
  ROADMAP cross-cutting "21-probe Anomaly Math Arrest ensemble" row
  flips to Functional.

### GOSNN Placeholders

- **`gosnn_optimizer.optimize` no longer fabricates random attention
  tensors** (Phase 2 ITEM 3). The previous `rng.standard_normal((32,
  16, 16))` fallback is deleted. When no `AttentionProvider` is
  configured (or the configured one raises), the attention-overhead
  metric is *skipped* and that fact is surfaced in
  `OptimizationResult.recommendations`. Real tensors flow only via
  `AttentionProvider.get_attention`.
- **Conformal-prediction failures propagate** in
  `GOSNNIntegration.detect`. The blanket
  `except (ValueError, RuntimeError, AttributeError)` that left
  `confidence_intervals=None` is gone — callers now see the underlying
  exception. Regression: `tests/core/test_gosnn_placeholder_cures.py`.

### Distributed Processing

- **Native pure-stdlib TCP MessageTransport for Raft** (Phase 2 ITEM
  4). New `omni_mercury_engine.distributed.tcp_transport`:
  `asyncio.start_server` + length-prefixed binary frames + per-message
  Ed25519 signatures via Mercury's own AMA Cryptography surface. No
  third-party RPC framework, no protobuf, no msgpack, no zeromq — the
  wire format is Mercury's own. The five `NotImplementedError` sites
  in `raft_consensus.py` are gone; `RaftCluster(use_in_memory_transport=False)`
  now constructs real network nodes. Integration test
  `tests/distributed/test_tcp_transport.py::test_three_node_cluster_elects_and_re_elects`
  spins up 3 nodes on 3 TCP ports, elects a leader, kills it, and
  confirms re-election. ROADMAP "Distributed Processing" row flips to
  Functional.

### Cryptography

- **AMA Cryptography Known-Answer Tests + measured coverage** (Phase
  2 ITEM 5). New `tests/security/test_ama_kat.py` pins:
  - Ed25519 RFC 8032 §7.1 vectors bit-for-bit (always run).
  - ML-DSA-65 / Kyber-1024 / SPHINCS+ round-trips and ML-DSA
    deterministic-signing reproducibility (run when AMA's PQC
    backend is installed; skip — never silently pass — otherwise).
  The `pqc-production-check.yml` workflow runs the KATs on every PR
  and publishes a coverage XML for `crypto_api.py` + `pqc_backends.py`
  + `pqc_guards.py` as the `pqc-coverage` artifact. README PQC
  section now cites the in-repo evidence rather than external-audit
  framing.

### TODO / FIXME Discipline

- **Inline markers restored** for unresolved findings from the
  2026-03 in-tree audit (ITEM 6). High-impact cited lines now carry
  `# TODO(audit-2026-03, severity=critical|high|medium|low):` markers
  at the cited locations
  (`core/ai_ethics.py:139,141`, `core/ethical_governor.py:209-210`,
  `core/three_r/fusion.py:91`). Findings closed in the same cycle
  (GOSNN attention placeholder, GOSNN dead `_fusion`, conformal
  silent failure, ethics-decision-boundary advisory mode) are marked
  with citations to the regression suites in their respective source
  files. `CONTRIBUTING.md` codifies the rule going forward: every
  new `TODO` / `FIXME` MUST include a severity tag and a citing
  reference.

### Ethics

- **Hard ethics enforcement at the decision boundary** (Phase 2 audit
  cure, May 2026). Every top-level inference path now raises
  `EthicalViolation` (re-exported from
  `omni_mercury_engine.cognitive.ethical_bounding.EthicalConstraintViolationError`)
  on benevolence-threshold violation, replacing the prior
  logger.warning / `ethical_violations`-list / `strict_ethics`-flag
  advisory paths. Boundary surfaces:
  - `CognitiveOrchestrator.analyze` raises with `check="benevolence"`
    when the per-analysis benevolence score falls below the scorer's
    threshold. The `strict_ethics=False` constructor argument is
    deprecated and ignored — passing `False` emits a
    `DeprecationWarning` and the gate still fires.
  - `NeuroSymbolicHub.predict` raises with `check="benevolence"` for
    any sample whose computed benevolence is below
    `benevolence_threshold` (replaces the prior
    `result.ethical_compliant=False` advisory return).
  - `OmniMercuryEngine.detect_with_fusion` (and the `_calibrated`
    variant) raises with `check="benevolence"` via a per-engine
    `BenevolenceScorer.enforce` call against an action description
    rich in defensive-purpose keywords.  The boundary scorer is
    constructed eagerly at engine init so the first concurrent call
    cannot race the gate.  σ_Immutable is now trained (weights persisted at
    `src/omni_mercury_engine/security/sigma_immutable_weights.pt`)
    and `EthicalGate.evaluate` gates its torch path on
    `self._trained`.  At the engine boundary σ_Immutable remains
    *informational* in `result["gosnn_metadata"]` — the only hard
    enforcement gate is `BenevolenceScorer.enforce`.  Promotion of
    σ_Immutable to a second hard gate (and raising of
    `EthicalConstraintViolationError` with `check="sigma_immutable"`
    or `check="gosnn_unavailable"`) is deferred to a follow-up PR;
    those `check` field values are reserved in the exception schema
    but are not raised by any code path on the merge tip.  The
    previous "fall back to `ethical_gate_passed=True` if GOSNN
    errors" path is deleted.
  Decision-boundary contract documented in
  `src/omni_mercury_engine/ethical/__init__.py`. New regression
  suite at `tests/ethical/test_hard_enforcement.py` (13 tests, wired
  into the `Neuro-Symbolic Tests` CI job) makes a benevolence-threshold
  regression a build-time failure. ROADMAP cross-cutting "Ethics
  enforcement" row flips from Stubbed to Functional.
### Added (Wave A — post-PR-167 punch list, items 3, 4, 6, 7, 9)

- **Federated silent-failure gaps closed (item 9).** Two distinct gaps the
  2026-03 in-tree audit flagged on the federated/GOSNN path are now
  closed:
  1. `core/gosnn_integration.py::GOSNNIntegration.detect()` no longer
     swallows conformal failures into `confidence_intervals=None`. New
     exception `ConformalMisconfigurationError` is raised on any
     `(ValueError, RuntimeError, AttributeError)` from the conformal
     predictor when `use_conformal=True`. Callers who want no intervals
     must opt out explicitly with `use_conformal=False`.
  2. New module `federated_learning/gosnn_coupling.py` provides
     bidirectional GOSNN scalar coupling (`GOSNNCouplingServer` +
     `GOSNNCouplingClient` + `GOSNNUpdate` / `GOSNNGlobalState` payloads)
     with FedAvg-weighted aggregation and SHA3-256 digest checks on every
     leg (anchored to AMA Cryptography's
     ``CryptoPackageConfig.hash_algorithm`` standard), replacing the prior
     one-way (server → client) integration.
  Suite `tests/federated/test_no_silent_failure.py` (12 tests) pins
  conformal-misconfig raising for each flavour of upstream failure, the
  explicit-opt-out path returning `confidence_intervals=None`, full
  client → server → client round-trips, FedAvg correctness, multi-round
  state convergence, and rejection of shape / round / digest mismatches.
- **Seven-axis evaluation matrix.** New runner
  `benchmarks/seven_axis_runner.py` produces a deterministic JSON + markdown
  report across the seven NSAI evaluation axes (Generalization, Scalability,
  Data Efficiency, Reasoning, Robustness, Transferability, Interpretability)
  using only NumPy and the existing Mercury fusion primitives. The benchmark
  CI workflow (`.github/workflows/benchmark.yml`) runs the runner and uploads
  `benchmarks/seven_axis_results.json` as an artifact. The corresponding
  section in `docs/BENCHMARKS.md` is regenerated from the runner's output
  via `python -m benchmarks.seven_axis_runner --regenerate-docs` (delimited
  by `## Seven-Axis Evaluation Matrix` … `<!-- end seven-axis-section -->`,
  so it cannot be hand-edited without the next regeneration overwriting
  it). Suite `tests/benchmarks/test_seven_axis_runner.py` (9 tests) pins the
  axis names, score bounds, JSON round-trip, idempotent docs regeneration,
  in-place section replacement, and per-axis determinism (with a documented
  tolerance for the wall-clock-based Scalability axis).
- **Benevolence-decision cache (`CachedBenevolenceScorer`).** New
  `cognitive/benevolence_cache.py` provides a thread-safe LRU wrapper around
  `BenevolenceScorer.enforce`. Cache keys are prefixed with
  `ETHICAL.RULESET_VERSION` (new constant in `core/centralized_constants.py`)
  so bumping the ruleset atomically purges every stale entry. **Violations
  are never cached** — `EthicalConstraintViolationError` propagates without
  insertion, so positive cases always recompute. Suite
  `tests/ethical/test_benevolence_cache.py` (12 tests) pins ruleset-version
  invalidation, identical-input cache hits, never-cache-violations, LRU
  eviction at capacity, dict-order canonicalisation, and recovery-after-
  violation behaviour.
- **Cooperative convergence loop in `AdaptiveDomainThresholdManager`.** New
  `calibrate_iterative()` and `_cooperative_refine_threshold()` methods perform
  a 1-D EM-style mean-shift between the two soft score-mode centroids, with a
  bounded budget (default `max_iterations=4`, `epsilon=1e-3`). Returns full
  convergence diagnostics (`iterations`, `converged`, `threshold_path`).
  Regression suite `tests/core/test_adaptive_threshold_convergence.py` (8 tests)
  pins (a) convergence within the 4-iteration / ε=1e-3 budget on synthetic
  drift, (b) wall-clock cost ≤ 1.3× the one-shot path, (c) no oscillation on
  stationary input (≤1 sign-change in the threshold path), and (d) the loop
  is idempotent at its fixed point.
- **`FusionMode.FIBRING` is the named default top-level fusion mode.** Composes
  three already-present primitives — Phi-weighted base, correlation-aware
  decorrelation (sliding window), and per-domain affinity bias — in a single
  NSAI-taxonomy-faithful mode. New module `core/fibring_fusion.py` (`FibringComposer`,
  `FibringWeights`, `DOMAIN_AFFINITY_BIAS`). `NeuroSymbolicHub.__init__` and
  `create_neurosymbolic_hub` now default to `FusionMode.FIBRING` /
  `fusion_mode="fibring"`. `create_fusion_ensemble` defaults to `method="fibring"`,
  which returns an `EthicallyConstrainedFusion` with phi-weighted base. New
  regression suite `tests/core/test_fibring_default.py` (13 tests) pins the
  default routing, the composer's phi-baseline / decorrelation / affinity behaviour,
  and an ablation against BALANCED on a deterministic channel-symmetric synthetic
  workload (no AUROC or Brier regression). See `docs/ARCHITECTURE.md` §
  "Neuro-Symbolic Fusion (NSAI Taxonomy)".

### Security

- **CVE-2026-6357 remediated in Docker image (pip arbitrary code injection).**
  The `python:3.12-slim-bookworm` base image ships pip 25.0.1, which is
  vulnerable to arbitrary code execution via the self-update check logic
  when installing a malicious wheel package. Both Dockerfile builder and
  runtime stages now pin `pip>=26.1` (was `>=26.0`), eliminating
  CVE-2026-6357, CVE-2025-8869, and CVE-2026-1703 in a single version bump.
  The accepted-risk ledger in `SECURITY.md` records the CVE-2026-6357 entry.
- **Trivy CI scan hardened with `limit-severities-for-sarif: true`.**
  The `aquasecurity/trivy-action` SARIF format mode was dropping the severity
  filter, causing MEDIUM-severity pip CVEs to trigger the CRITICAL/HIGH
  blocking gate. Adding `limit-severities-for-sarif: true` ensures the SARIF
  report and exit-code respect the `severity: CRITICAL,HIGH` policy.
- **Pickle code path removed from training pipeline.** The legacy
  `.pkl` / `.pickle` branch in `OmniMercuryEngine.train_fusion_model`
  has been deleted. Pickle is structurally a code-execution format and
  the existing whitelist was both functionally broken under numpy 2.x
  (production whitelist used `numpy.core.multiarray` while numpy 2.x
  emits `numpy._core.multiarray`) and incomplete (rejected `bool_`,
  `uint8`, `float16`). Replacement: `omni_mercury_engine.security.safe_load`
  module exposing `safe_load_training_data`, optional HMAC-SHA-256
  provenance via `sign_npz` / `verify_npz_signature`, and a
  one-shot operator migration tool at
  `python -m omni_mercury_engine.tools.migrate_pkl` that runs in a
  separate hardened subprocess.
- **Zip-bomb defense in `safe_load`.** `_validate_zip_central_directory`
  inspects the `.npz` central directory before any decompression and
  rejects: corrupt zips, archives with more than 256 entries, per-entry
  uncompressed sizes above 1 GiB, cumulative uncompressed size above
  1 GiB, per-entry compression ratio above 1000:1 (zip-bomb signature),
  and entry names containing path-traversal components (`..`, leading
  `/`, drive letters, or backslashes — backslash is rejected outright
  because POSIX path parsing keeps `\\` as a literal character and a
  Windows extractor would interpret it as a directory separator).
- **`migrate_pkl` invocation.** The tool runs the conversion when
  invoked. There is no `--i-trust-this-file` / consent flag; it would
  have been theatre on top of real defenses (subprocess isolation,
  scrubbed env, pre-write object-dtype rejection). Optional
  `--sign-key-hex` and `--max-bytes` retained.
- **43 new tests** in `tests/security/test_safe_load.py` and
  `tests/security/test_migrate_pkl.py` pin: pickle path is gone;
  loader rejects wrong magic bytes, oversized files, object dtypes,
  pickle-disguised-as-`.npz`, zip-bombs (compression-ratio guard),
  too-many-entries, corrupt zips, path-traversal entries (forward
  slash, ``..``, backslash, drive-letter), tightened uncompressed-size
  caps; HMAC roundtrip and tamper detection work; migration tool
  relaunches itself in a hardened subprocess and refuses to overwrite
  existing outputs.

## [1.6.0] - 2026-05-01

### Security (Dependency CVE Remediation)

- **cryptography** `>=43.0.1` → `>=46.0.7`: CVE-2026-26007, CVE-2026-34073, CVE-2026-39892
- **pillow** `>=10.4.0` → `>=12.2.0`: CVE-2026-25990, CVE-2026-40192
- **requests** `>=2.32.0` → `>=2.33.0`: CVE-2026-25645
- **aiohttp** `>=3.9.0` → `>=3.13.4`: CVE-2026-34513 through CVE-2026-34525 (18 CVEs)
- **pytest** (dev) `>=7.4.0` → `>=9.0.3`: CVE-2025-71176
- **black** (dev) `>=24.0.0,<26.0.0` → `>=26.3.1,<27.0.0`: CVE-2026-32274

### AMA Cryptography Migration

- **AMA Cryptography Integration**: Migrated primary PQC backend from `ava-guardian` to
  `ama-cryptography`. The new `mercury_amacrypto.py` module is the canonical integration
  adapter, with `mercury_guardian.py` kept as a backward-compatibility re-export shim.
- **New `AMA_CRYPTOGRAPHY_AVAILABLE` flag**: Primary availability flag in
  `pqc_backends.py`, `pqc_guards.py`, `_compat.py`, and the integration adapter.
  `AVA_GUARDIAN_AVAILABLE` and `HAS_AVA_GUARDIAN` kept as aliases.
- **Updated GOSNN component name**: `ava_guardian_crypto` → `ama_cryptography_pqc` in
  the GOSNN synapse registration.
- **New `create_ama_cryptography_adapter()` factory**: Canonical factory function;
  `create_mercury_guardian_adapter()` kept as alias.
- **Env var updates**: `AMA_REQUIRE_REAL_PQC` and `AMA_REQUIRE_CONSTANT_TIME` are now
  the primary env vars; legacy `AVA_REQUIRE_*` names still accepted.
- **PQC Production Check CI**: Updated workflow triggers to also watch
  `mercury_amacrypto.py` and uses the new `AMA_CRYPTOGRAPHY_AVAILABLE` check.

### Added (Anomaly Math Arrest — 21-Probe Ensemble)

- **Anomaly Math Arrest**: 21-probe mathematically-independent equation ensemble
  replacing IsolationForest in the detection path. Each probe detects a different
  anomaly geometry using fundamentally different mathematical frameworks.
- **Probes 1-8**: Additive, HarmonicOscillator, Momentum, VarianceAdapted,
  EthicalConstrained, CatalanOptimized, ExponentialDecay, HelixMultiplicative.
- **Probes 9-21**: R3RecursionResonance, SVDProjection, LyapunovChaos,
  TopologyHomology, FractalSelfSimilarity, ZetaHarmonic, WavePropagation,
  QuantumSuperposition, EnergyMinimization, QuantumAnnealing,
  BoltzmannCoupling, IQRRobust, ModifiedZScore.
- **PhiWeightedFusion**: Golden-ratio-derived weight fusion with confidence
  modulation and domain affinity reordering.
- **CorrelationAwareDecorrelator**: Pairwise Pearson correlation audit with BFS
  connected component detection to prevent redundant probe clusters from
  over-weighting. Calibrated automatically during `fit()`.
- **Domain affinity maps**: 7 domains (earthquake, tsunami, pandemic, marine,
  geomagnetic, conflict, default) with per-domain probe ranking.
- **calibrate_decorrelator()**: Explicit correlation audit API.
- **get_correlation_report()**: Full transparency into redundant pairs, weight
  multipliers, and effective probe count.
- **83 new tests**: Core (34), Extended probes 9-21 (43), Fusion + decorrelator (6).

### Removed

- **IsolationForest**: Removed from the primary detection path. Mercury Agent now
  stands on its own transparent, auditable math via the Anomaly Math Arrest.

### Fixed (Test Suite Stabilization — 100+ Failures Resolved)

- **FastAPI dependency chain**: API auth module (`api/auth.py` line 45) imports
  FastAPI at module level, causing `ModuleNotFoundError` cascading through
  `api.server`, `test_correlation_id.py` (4 tests), `test_audit_improvements.py`
  (6 tests), and `test_jwt_auth.py` (12 tests). Added FastAPI to test
  dependencies.
- **Federation `to_detector()` missing Oracle init**: `FederatedAggregator.to_detector()`
  (aggregator.py lines 203-221) did not initialize `_oracle_detector` or
  `_oracle_metadata` attributes, causing `AttributeError` when reconstructed
  detector called `detect()` at `statistical.py` line 1886. Added initialization
  matching `from_statistics()` classmethod pattern.
- **Solar synthetic data zero labels**: `_create_synthetic_solar()` (space.py
  lines 635-642) generated `xray_short` via `np.random.exponential(1e-7)` but
  tested against threshold `1e-5`, producing all-zero labels. Lowered threshold
  to `5e-8` to match the exponential distribution's tail.
- **Oracle config type mismatch**: Fixed `SpectralDomainOracleConfig` type
  handling when passed through detector initialization pipeline.
- **Oracle influence pipeline**: Wired spectral influence multiplier end-to-end
  through `detect()` return path, enabling Oracle-augmented scoring across all
  75 benchmark datasets.

### Added (F1 Precision Directive — Phases 1-10)

- **Phase 1**: Renamed "Superintelligence Bootstrap" → "Cognitive Evolution Engine"
- **Phase 2**: Added pairwise Spearman inversion guard (ρ < -0.2) and unsupervised ensemble flip (median > 0.80)
- **Phase 3**: Created domain-adaptive weight presets (14 domains, 60/40 data-driven/prior blend)
- **Phase 4**: Implemented noise color estimation via log-log PSD regression (white/pink/brown detection)
- **Phase 5**: Added adaptive significance alpha based on window size and noise model confidence
- **Phase 6**: Applied asymmetric influence bias (1.5x amplification boost, 0.8x attenuation suppression)
- **Phase 7**: Added residual frequency filter via FFT bandpass (70/30 blend ratio)
- **Phase 8**: Upgraded to multi-strategy threshold selection (percentile, MAD, contamination-aware, linear sweep)
- **Phase 9**: Added DOMAIN_ANOMALY_SPECTRAL_HINTS and dynamic Oracle sensitivity based on initial severity
- **Phase 10**: Added 30+ tests for all F1 precision improvements, ORACLE_NOISE_COLOR.md documentation

### Changed (Benchmark Expansion)

- **Benchmark coverage**: Expanded from 51 to 75 total datasets (47 ADBench +
  28 domain loaders across 12 domains).
- **Mean AUC**: Improved from 0.8030 to **0.8379** after Oracle pipeline fix.
- **Median AUC**: Improved from 0.8852 to **0.9090**.
- **SpectralDomainOracle**: Auto-activated on 39 of 64 successful datasets
  for temporal/spectral domain augmentation.
- **README.md**: Updated all benchmark tables, added domain-level performance
  matrix, added quality improvements section, updated dataset counts and dates.

### Added (Mercury System Activation)

- **Oracle wired into MercuryAnomalyDetector**: SpectralDomainOracle now
  integrated into the primary detection pipeline (`statistical.py`). Oracle
  initialises during `fit()` for temporal data and applies spectral influence
  multiplier during `detect()`. Oracle metadata included in return dict.
- **20 new dataset loaders activated**: Environmental (USGS Earthquake, NOAA
  Weather, Wildfire, USGS Geochemistry), Ocean (NOAA Buoy), Climate (NOAA
  StormEvents, GSOD, ERDDAP), Air Quality (EPA), Disaster (FEMA x2), Space
  (NASA Exoplanet, SolarDynamics), Academic (UCR, CWRU Bearing, MSDS),
  Security (ThreatIntel), General (ADRepository), Industrial (SWaT, WADI).
  Total benchmark datasets: 75 (47 ADBench + 28 domain).
- **Domain-level benchmark summary**: Per-domain AUC/F1 aggregation,
  component performance analysis, Oracle activation tracking. Added to
  `mercury_benchmark_results.json` as `domain_summary`.
- **Benchmark CLI flags**: `--live-only` skips ADBench; `--domain <name>`
  filters by category.
- **Cross-domain frequency correlation module**: New
  `CrossDomainFrequencyCorrelator` detects overlapping significant frequency
  bands across concurrent Oracle instances. Outputs correlation only —
  always states "requires human assessment."
- **Oracle domain auto-selection**: `_infer_oracle_domain()` uses sample
  rate estimation, dominant FFT frequency, and feature count heuristics
  to select appropriate Oracle domain. User-specified domain overrides.
- **Federation Oracle serialization**: `get_oracle_statistics()` exports
  Oracle reference state; `from_statistics()` accepts `oracle_ref_stats`
  for federated Oracle round-trip.
- **Domain-specific cache TTL**: environmental=300s, security=60s,
  climate=3600s, default=600s. Added `get_domain_ttl()` to cache stub.
- **Structured per-dataset output**: `per_dataset_results.json` now
  includes `run_metadata` (run_id, timestamp, git_sha, branch,
  python_version), domain_summary, and expanded per-dataset diagnostics
  (adaptive_weights, weight_source, data_type, oracle_metadata).
- **Dataset catalog**: `benchmarks/DATASETS.md` documents all 75 active
  datasets with source, auth, samples, features, anomaly ratio, license.

### Changed (Mercury System Activation)

- **README Phase 7**: Renamed "Superintelligence Bootstrap" to "Cognitive
  Evolution Engine".
- **BaseDetector.extract_features()**: Signature updated to accept
  `dict[str, Any]` input and return `np.ndarray | torch.Tensor`.
  `_extract_combined_features()` normalises both return types.
- **validation/data_loaders.py**: Deprecated with module-level warning.
  Directs users to `datasets/` module. Will be removed in v2.0.
- **generate_docs_images.py**: Updated performance dashboard to show
  domain-level AUC/F1 bars, component AUC heatmap, and Oracle status.
  All data sourced from `mercury_benchmark_results.json`.

### Cherry-picked (from devin branch)

- 10 commits salvaged from `devin/1771750539-mercury-strategic-improvements`:
  strategic improvements, FrequencyDomainOracle implementation,
  SpectralDomainOracle rename, spectral flux/phase coherence/cepstral
  analysis, selective inference upgrade, and audit fixes.
- Fabricated README images (4 PNGs with invented performance metrics)
  discarded; original dark-themed images restored from master.

### Fixed
- **Oracle `extract_features()` type contract**: Now returns `torch.Tensor`
  per `BaseDetector` contract, eliminating runtime crash risk in 5 generic
  callers (detector_registry, gosnn_integration, metrics, gwo_ensemble).
  Removed all `# type: ignore` suppressions.
- **Selective Inference upgrade**: Replaced naive z-test with truncated
  normal conditioning (Lee et al., 2016). Guarantees Type I error control
  at declared α. SI p-values are 3-10× more conservative, eliminating
  false amplification from selection bias.

### Renamed
- **SpectralDomainOracle**: Renamed from `FrequencyDomainOracle` to reflect
  expanded capabilities (spectral flux, phase coherence, cepstral analysis).
  Backward-compatible aliases preserved (`FrequencyDomainOracle`,
  `FrequencyDomainOracleConfig`, `create_frequency_oracle`).

### Added
- **Spectral Flux**: Rate-of-spectral-change detection for slow-onset
  anomalies (frequency-domain analog of acceleration).
- **Phase Coherence**: Inter-band phase relationship monitoring via Welch's
  method cross-spectral estimation (leading indicator — degrades before
  amplitude changes).
- **Cepstral Coefficients**: Harmonic structure analysis via inverse FFT of
  log power spectrum (rotating machinery faults, quefrency-domain peaks).
- **Five-signal influence multiplier**: φ-weighted geometric mean now
  incorporates flux and coherence alongside score, entropy, and breadth.
- **Domain-aware Oracle auto-activation**: Oracle auto-enabled for
  infrastructure, security, medical; dampened for environmental, space;
  auto-disabled for financial, humanitarian; overridable via `oracle_mode`.
- **Per-dataset benchmark output**: `per_dataset_results.json` for
  individual dataset AUC/F1 verification (separate from aggregate results).

### Documentation Alignment Audit
- Repository documentation, metadata, and organizational state audited for
  accuracy against actual code
- README.md: Updated module count (455), line count (268,000+), test file
  count (227); removed phantom version reference (v1.4.1); removed reference
  to nonexistent benchmark results file; updated header and co-architects
- CHANGELOG.md: Removed orphaned [Unreleased] section whose content was
  already captured in versioned entries (v1.1.0, v1.2.0)
- docs/ROADMAP.md: Marked all 7 planned capabilities as implemented
- CONTRIBUTING.md: Fixed Python version requirement from 3.12 to 3.11+
- .gitignore: Added coverage for generated report files


### Renamed
- **OAE (Omni-Ava Equation)**: Renamed from "AVA Anomaly Fusion Equation (AAFE)"
  across 26 files. All old class names (`AnomalyFusionEquation`, `AAFEWeightOptimizer`,
  `DomainAdaptiveAAFEWeights`) and constants (`AAFE_WEIGHT_R/H/O`) remain functional
  as backward-compatible aliases per the Preservation Principle (see DEPRECATION.md).

### Calibration Pipeline Integration
- ThresholdCalibrationPipeline wired into MercuryAnomalyDetector.fit_with_labels()
  and detect() — adaptive strategy selection (best of Youden's J / F1-optimal)
- Per-component Mann-Whitney AUC measured during fit; components with inverted
  signal receive zero weight (replaces fixed 40/30/30 ensemble weighting)
- 10 real-world domains benchmarked with calibrated F1; results recorded in
  mercury_benchmark_results.json

### Version Reconciliation
- All version strings bumped from 1.5.1 → 1.6.0 across 28 files:
  pyproject.toml, __init__.py, cli.py, api/health.py, api/server.py,
  crypto/__init__.py, models/sota/__init__.py, .secrets.baseline,
  README.md, SECURITY.md, Dockerfile, data_sources/base.py,
  data_sources/earth_science.py, cognitive/anomaly_detection_enhanced.py,
  infrastructure/observability.py, integrations/cross_platform_hub.py,
  helm/mercury-agent/Chart.yaml, k8s/base/deployment.yaml,
  k8s/base/kustomization.yaml, k8s/overlays/distributed/streaming-workers.yaml,
  docs/MATH_SPEC.md, examples/physics_detectors_demo.py,
  benchmarks/generate_benchmark_visuals.py, benchmarks/generate_v1_2_visuals.py,
  benchmarks/live_dataset_benchmark.py, tests/test_cli_smoke.py, tests/test_api.py
- [Unreleased] section promoted to [1.6.0] — no unreleased content remains

---

## [1.5.1] - 2026-02-13

### Changed - Statistical Detector Ensemble Replacement

- **MercuryAnomalyDetector**: Replaced `z-score * 0.4 + IQR * 0.3 + IsolationForest * 0.3`
  ensemble with three Mercury-original mathematical frameworks:
  - **ResonanceScore (40%)**: FFT-based harmonic spectral anomaly detection
  - **KinematicScore (30%)**: Physics-based jerk/curvature dynamics via finite differences
  - **InfoGeometryScore (30%)**: Fisher Information Matrix OOD detection with Tikhonov
    regularization and Cholesky-decomposed precision matrix
- Performance: 49x faster fit (14.8 ms), 4.7x faster inference (13.3 ms) vs prior
  IsolationForest ensemble on 10,000 x 20 synthetic data
- Mean AUC 0.992 on synthetic ADBench-style benchmark (8 dataset patterns)
- 100% backward-compatible `detect()` return dict (all legacy keys preserved)

### Removed

- **sklearn dependency from core detectors**: `MercuryAnomalyDetector`,
  `DimensionalAnalyzer`, and `SpatialAnomalyDetector` no longer import sklearn.
  PCA replaced with numpy SVD; LOF replaced with scipy KDTree implementation.
- All remaining top-level sklearn imports in `src/` converted to lazy imports.
  sklearn is now a true optional dependency (`pip install mercury-agent[ml]`).
- Deleted stale benchmark PNG artifacts from `benchmarks/`

### Fixed

- Updated stale IsolationForest references across src/, docs/, benchmarks/, and
  issue templates to reflect the new ensemble architecture

---

## [1.5.0] - 2026-02-11

### Added - Mathematical Audit & Parameterization Overhaul

- **PHASE 1 — Equation Inventory** (`docs/equations_inventory.md`): Comprehensive catalog
  of 47 equations, formulas, constants, thresholds, and mathematical operations across
  the entire codebase with provenance tracking (EQ-001 through EQ-047)

- **PHASE 2 — Correctness Report** (`docs/correctness_report.md`): Mathematical
  correctness verification identifying 16 issues (1 critical, 3 high, 8 medium, 4 low)
  across numerical stability, edge cases, and documentation mismatches

- **PHASE 3 — Parameterization Overhaul**:
  - **Sigmoid Benevolence Gate**: Replaced hard threshold (≥ 0.99) with smooth sigmoid
    curve η(b) = 1/(1+exp(-k·(b-b₀))) with domain-specific profiles:
    Medical (b₀=0.93, k=30), Security (b₀=0.95, k=25), Environmental (b₀=0.90, k=20),
    Humanitarian (b₀=0.92, k=35), Infrastructure (b₀=0.94, k=25)
  - **Banach Contraction Recursion**: Added convergence-bounded recursive computation
    with α constrained via sigmoid (α_max=0.95), error bounds, and runtime contraction
    monitoring with halt on violation. Provenance: Banach fixed-point theorem (1922)
  - **Domain-Adaptive Harmonics**: Replaced universal Schumann resonance (7.83 Hz) with
    domain-specific fundamental frequencies: Medical (HRV bands), Infrastructure (power
    grid), Space (solar cycle), with adaptive peak detection for unknown domains
  - **Hierarchical Omni-Scalar Aggregation**: Added 3-level hierarchical aggregation
    (category grouping → weighted mean → geometric mean) for 180+ omni-scalars organized
    into safety, fairness, transparency, accountability, and beneficence categories
  - **Configurable OAE Exponent**: Made ethical scaling exponent configurable (default Φ)
    to support empirical optimization via parameter sweep
  - **NaN Guards**: Added NaN propagation prevention to OAE fusion equation
  - **Parameter Sweep Infrastructure** (`benchmarks/parameter_sweep.py`): Bayesian
    optimization via Optuna TPE over full parameter space with composite objective
    (F1 + calibration error + stability)

- **PHASE 3A — Parameter Sweep Execution**: Ran 1,000 Optuna TPE trials with
  composite objective (F1 + ECE + stability). Best composite: 0.816, Best F1: 0.895.
  53 Pareto-optimal configurations identified. Results saved to
  `benchmarks/parameter_sweep_results.json`

- **PHASE 3B — Phi Exponent Validation**: Statistically validated golden ratio exponent
  (Phi=1.618). Mean F1 near Phi: 0.9045 vs 0.8944 elsewhere (p < 0.001, t=8.05).
  Phi confirmed as near-optimal with best trial at 1.742

- **PHASE 3B — Domain-Adaptive OAE Weights** (`core/three_r/fusion.py`):
  `DomainAdaptiveOAEWeights` class that learns per-domain weight profiles from empirical
  data when cross-domain variance exceeds 10%. Falls back to golden-ratio defaults

- **PHASE 4A — Conformal Prediction Enhancement** (`core/conformal_prediction.py`):
  - `MondrianConformalPredictor`: Label-conditional coverage guarantees per group
    (Vovk et al. 2005 Chapter 8). Ensures balanced coverage across subpopulations
  - `ConformalCalibrationBridge`: Integrates split, adaptive, and Mondrian conformal
    prediction into the calibration pipeline

- **PHASE 4B — Topological Data Analysis** (`core/topological_analysis.py`):
  - `VietorisRipsFiltration`: Vietoris-Rips simplicial complex from point clouds
  - 0D and 1D persistent homology via union-find algorithm
  - `PersistenceDiagram`: Birth/death pairs, Betti numbers, persistence entropy
  - `TopologicalAnomalyDetector`: TDA-based anomaly detection with topological
    feature extraction (Betti numbers, entropy, landscape norms)
  - Wasserstein and bottleneck distances for persistence diagram comparison
  - Reference: Edelsbrunner & Harer (2010) "Computational Topology"

- **PHASE 4C — Fisher Information Metric** (`core/info_geometry.py`):
  - `FisherInformationMatrix`: Gaussian closed-form and empirical FIM computation
    with Tikhonov regularization for numerical stability
  - `NaturalGradient`: F^{-1} * g_euclidean with Cholesky decomposition
  - `FisherRaoAdaptiveThreshold`: Derives thresholds as tau = mu + k*sqrt(tr(F^{-1}))
    with drift detection and automatic recalibration
  - `StatisticalManifold`: Riemannian manifold of probability distributions
  - Reference: IGEOOD (ICLR 2022)

- **PHASE 4D — Riemannian Optimization** (`core/riemannian_optimization.py`):
  - `SimplexManifold`: Probability simplex with projection (Duchi et al. 2008),
    exponential/logarithmic maps, geodesic distance
  - `SPDManifold`: Symmetric positive definite matrices with affine-invariant metric
  - `RiemannianGradientDescent`: Manifold optimization with Armijo line search
  - `RiemannianAdam`: Adam optimizer adapted for Riemannian manifolds
  - `ConstrainedParameterOptimizer`: High-level API for OAE weights on simplex
    and covariance parameters on SPD manifold

- **PHASE 5 — Calibration Pipeline** (`core/calibration_pipeline.py`): Threshold
  auto-calibration with Youden's J, F1-optimal, cost-sensitive methods; dataset
  fingerprinting via SHA-256; distribution drift detection via KS test and KL divergence

- **PHASE 6 — System-Level Coherence** (`core/system_coherence.py`):
  - `SignalFlowGraph`: Data structure describing signal propagation through the
    detection pipeline (ingestion -> features -> detection -> fusion -> ethical
    gating -> calibration -> output) with ASCII rendering
  - `NormalizationVerifier`: Validates score range compatibility at every stage
    boundary, detecting normalization handoff mismatches
  - `LyapunovRuntimeEnforcer`: Runtime guard enforcing V_dot <= -lambda*V at
    every fusion step with violation logging and optional pipeline halt
  - `run_coherence_audit()`: Full system coherence audit function
  - Reference: Khalil (2002) "Nonlinear Systems" Chapter 4

- **Hardcoded Constant Centralization**: Replaced 38+ hardcoded 0.99 benevolence
  references across 10+ source files with `ETHICAL.BENEVOLENCE_IMMUTABLE` from
  `centralized_constants.py`. Files updated: benevolence_optimization.py,
  domain_metrics.py, enhanced_model_domains.py, gosnn_integration.py,
  gosnn_optimizer.py, engine_config.py, config.py, personality.py, engine.py,
  learnable_gosnn.py

- **PHASE 7 — Deliverables**:
  - `docs/MATH_SPEC.md`: Formal mathematical specification with LaTeX equations,
    parameter justification, convergence proofs, and sensitivity analysis
  - `docs/math_debt_backlog.md`: 15 prioritized mathematical debt items (3 high,
    6 medium, 6 low) with resolution recommendations
  - `docs/equations_inventory.md`: Complete equation catalog
  - `docs/correctness_report.md`: Correctness verification findings

### Changed

- **OAE Equation** (`core/three_r/fusion.py`): Ethical exponent now configurable
  (was hardcoded to Φ). Added `domain` and `ethical_exponent` constructor parameters.
  Added `benevolence_score` parameter to `compute()` for sigmoid gate integration.
- **Spectral Vibration** (`detectors/spectral_vibration.py`): `_compute_schumann_alignment`
  now uses domain-adaptive frequencies via `get_domain_fundamentals()`. Added
  `_detect_spectral_peaks` static method for unknown-domain frequency detection.
- **Centralized Constants** (`core/centralized_constants.py`): Added
  `BenevolenceGateConstants`, `DomainHarmonicConstants`, `RecursionConvergenceConstants`
  dataclasses and `sigmoid_benevolence_gate()`, `get_domain_fundamentals()` functions.
- **GOSNN** (`core/global_omni_scalar_network.py`): Added `compute_hierarchical_score()`
  method implementing 3-level hierarchical aggregation with configurable domain weights
  and geometric/arithmetic/harmonic mean options.

### Testing

- **124 new tests** across `test_phase3_math_audit.py` (78) and
  `test_phase4_6_math_audit.py` (46) covering all new mathematical infrastructure
- All 594 existing + new core tests passing

### Mathematical Provenance

- Sigmoid benevolence gate: Logistic function (Verhulst, 1845)
- Banach contraction recursion: Banach fixed-point theorem (Banach, 1922)
- Schumann resonance: Schumann (1952)
- HRV frequency bands: Task Force of ESC/NASPE (1996)
- Conformal prediction: Vovk et al. (2005)
- Mondrian conformal prediction: Vovk et al. (2005) Chapter 8
- Youden's J statistic: Youden (1950)
- Persistent homology: Edelsbrunner & Harer (2010)
- Fisher information metric: IGEOOD (ICLR 2022)
- Riemannian optimization: Absil et al. (2008)
- Simplex projection: Duchi et al. (2008)
- Lyapunov stability: Khalil (2002)
- Bayesian optimization: Bergstra et al. (2011) TPE

---

## [1.4.0] - 2026-02-09

### Added - Advanced Cognitive AI & Physics-Inspired Detectors

- **Physics-Inspired Anomaly Detectors**: Spectral vibration analysis, acceleration dynamics,
  dimensional analysis, spatial anomaly detection, and UI/UX behavioral anomaly detection
- **Advanced Physics Integration**: GOSNN scalar fusion with physics detector backends,
  factory functions and registry integration for all new detector types
- **Comprehensive Strict Type Checking**: MyPy strict mode enabled across entire codebase,
  resolved 274+ strict-mode errors and removed all `ignore_errors` exclusions
- **SHA3-256 Cryptographic Alignment**: Aligned cryptographic posture with AMA Cryptography,
  upgraded hash functions to SHA3-256 for tamper-evident audit trails
- **CI/CD Pipeline Compliance**: Resolved all blocking lint (Flake8), type (MyPy),
  security (Bandit), and test (pytest) failures across 408 files

### Changed

- **Codebase Consolidation**: Major refactoring for production readiness, removing
  all `cast()` workarounds in favor of proper type annotations
- **Version Alignment**: All deployment manifests, Helm charts, Kubernetes labels,
  Docker images, User-Agent strings, and documentation updated to v1.4.0

### Security

- Cryptographic hash upgrade from SHA-256 to SHA3-256 across audit trail components
- Removed type-unsafe `cast()` calls that masked potential runtime errors

---

## [1.2.0] - 2026-02-02

### Added - SaaS Infrastructure and Production Hardening

- **Streaming Infrastructure** (`infrastructure/streaming.py`): Production-grade streaming
  - `StreamProducer`/`StreamConsumer` abstract interfaces
  - `KafkaStreamProducer`/`KafkaStreamConsumer`: Apache Kafka integration with aiokafka
  - `RedisStreamProducer`/`RedisStreamConsumer`: Redis Streams for low-latency processing
  - `InMemoryStreamProducer`/`InMemoryStreamConsumer`: For testing
  - `CircuitBreaker`: Failure handling with configurable thresholds
  - `StreamingAnomalyPipeline`: End-to-end streaming anomaly detection
  - Factory classes for easy backend switching

- **Request Correlation IDs** (`api/server.py`): Distributed tracing support
  - `CorrelationIDMiddleware`: UUID-based request tracking
  - Accepts `X-Correlation-ID` or `X-Request-ID` headers
  - Context variable for access throughout request lifecycle
  - `X-Request-Duration-Ms` header for latency tracking

- **Load Testing Infrastructure** (`tests/load/`): SLO validation
  - `locustfile.py`: Locust-based load testing with multiple user classes
  - `k6_load_test.js`: k6 performance testing with scenarios
  - SLO thresholds: P50<100ms, P95<500ms, P99<1000ms
  - Smoke, load, stress, and spike test scenarios

- **Distributed K8s Deployment** (`k8s/overlays/distributed/`): Multi-node production
  - Strimzi Kafka cluster (3 brokers, 3 ZooKeeper)
  - Redis cluster with Sentinel failover (3 nodes)
  - Streaming worker deployment with HPA (2-20 replicas)
  - Network policies for secure communication

- **Live Dataset Benchmark Suite** (`benchmarks/live_dataset_benchmark.py`): 30+ datasets
  - 7 categories: Security, Industrial, Time-Series, Climate, Disaster, Environmental, ADRepository
  - Provenance tracking (live vs synthetic data)
  - Aggregate metrics and per-dataset results
  - JSON export for CI/CD integration

- **Benchmark Documentation** (`docs/BENCHMARKS.md`): Comprehensive guide
  - Dataset catalog with sources and access requirements
  - Metric definitions and expected performance
  - Reproducibility instructions

- **API Launcher Script** (`scripts/run_api.py`): Convenient server startup
  - Development mode with auto-reload
  - Production mode with multiple workers

- **New Dependencies** (`pyproject.toml`): Streaming and load testing
  - `[streaming]`: aiokafka, redis for streaming infrastructure
  - `[loadtest]`: locust for load testing

### Added - Live Oceanographic and Disaster Dataset Integration

- **Climate Dataset Loaders** (`datasets/climate.py`): Advanced marine data integration
  - `SimonsCMAPLoader`: Ocean biogeochemistry from Simons CMAP (satellite observations, in-situ measurements, model outputs)
  - `WorldOceanDatabaseLoader`: NCEI World Ocean Database temperature/salinity profiles (20M+ profiles, 1770-present)
  - `CopernicusSeaLevelLoader`: EU satellite altimetry data (0.25 degree resolution, 1993-present)
  - All loaders include synthetic fallbacks with realistic oceanographic patterns

- **Disaster Dataset Loaders** (`datasets/disaster.py`): FEMA emergency management data
  - `FEMADisasterLoader`: US disaster declarations from OpenFEMA API (hurricanes, floods, fires, earthquakes)
  - `FEMAHazardMitigationLoader`: Hazard mitigation grant program data
  - Rate limiting (500ms between requests) and SSRF protection via TrustedEndpoints
  - No API key required - free public access

- **Environmental Loader** (`datasets/environmental.py`): Contamination detection
  - `USGSGeochemistryLoader`: Heavy metal concentrations (As, Pb, Hg, Cu, Zn) from USGS MRData
  - EPA Regional Screening Levels for anomaly labeling

- **Trusted Endpoints** (`security/input_validation.py`): SSRF protection for new data sources
  - New domains: fema.gov, simonscmap.com, cds.climate.copernicus.eu, ncei.noaa.gov, mrdata.usgs.gov
  - New API endpoints: FEMA_DISASTER_DECLARATIONS, SIMONS_CMAP_API, NCEI_WOD_SELECT, COPERNICUS_CDS_API, USGS_MRDATA_GEOCHEM

- **Comprehensive Tests**: 54+ new tests for climate and disaster loaders
  - `tests/datasets/test_climate.py`: 26+ tests for oceanographic data quality
  - `tests/datasets/test_disaster.py`: 28+ tests for FEMA API integration

### Fixed - Exception Handling Improvements

- Replaced 20+ bare `except Exception` with specific exception types
- Added proper logging with exception type names
- Files improved: `empirical_benchmark.py`, `benchmarks.py`, `comm.py`, `input_validation.py`, `adaptive_domain_thresholding.py`, `gosnn_integration.py`

### Fixed - PR-AUC Calibration

- Fixed negative PR-AUC values in `datasets/benchmarks.py`
- Replaced incorrect `np.trapz()` with proper step-function integration
- Added curve sorting before area calculation
- Clamped values to [0, 1] range

### Fixed - Engineering Polish

- **JWT Exception Handling** (`api/auth.py`): Specific exception types for better debugging
  - `ExpiredSignatureError`, `InvalidTokenError` for JWT-specific errors
  - `KeyError`, `TypeError`, `ValueError` for malformed payload detection
  - Improved logging with exception type names

- **Input Validation** (`core/adaptive_fusion.py`): Production-safe assertions
  - Replaced `assert` statements with explicit `if/raise ValueError`
  - Validation remains active even with Python `-O` optimization
  - Error messages include actual values for debugging

- **Resource Management** (`infrastructure/observability.py`): FileAuditHandler lifecycle
  - Added `close()` method for explicit resource cleanup
  - Context manager protocol (`__enter__`, `__exit__`) for safe usage
  - `__del__` for GC cleanup with thread-safe `_closed` flag
  - `RuntimeError` if `emit()` called after close

- **Deprecated API** (`datasets/ocean.py`): Fixed pandas deprecation warning
  - Changed `delim_whitespace=True` to `sep=r"\s+"` (regex whitespace separator)

## [1.1.2] - 2026-01-17

### Fixed
- Statistical detector now outputs continuous scores instead of discrete boolean flags
- Adaptive contamination estimation replaces hardcoded 0.1 default
- Dynamic projection layers cached in ModuleDict to prevent memory leaks
- 3D tensor inputs properly handled in fusion model forward pass
- Soft score normalization preserves ranking information
- Semi-supervised detector fitting prevents zero-variance score cascade

### Added
- `fit_fusion()` method for training OmniFusionModel
- `calibrate_scores()` for isotonic/Platt calibration
- `tune_hyperparameters()` for Optuna-based optimization
- `visualize_embeddings()` for t-SNE/UMAP analysis
- `add_detector()` for runtime detector registration
- Numba JIT optimization for spatial distance computations
- 45+ new tests for fusion training and signal integrity

## [1.1.1] - 2026-01-16

### Fixed - Benchmark Regeneration & CodeQL Cleanup
- **Benchmark Results Regenerated**: Updated `results/latest/benchmark_results.json` with AdaptiveAnomalyDetector
  - BATADAL F1: 0.0 → 0.333 (major improvement using adaptive dataset profiling)
  - SMD F1: 0.185 → 0.164
  - covtype F1: 0.117 (newly added dataset)
  - breast_cancer F1: 0.061 (newly added dataset)
- **CodeQL Alerts Resolved**: Added explanatory comments to 26 empty except blocks
  - `tests/test_resilience.py`: Circuit breaker failure counting comments
  - `tests/resilience/test_circuit_breakers.py`: Multiple failure testing comments
- **PEP8 Formatting**: Applied blank line fixes after imports across 28 files
- **Documentation Updated**: README.md benchmark table updated with regenerated results

## [1.1.0] - 2026-01-09

### Security - v1.1.0 CodeQL Remediation
- **PBKDF2-HMAC-SHA256 API Key Hashing**: Upgraded from plain SHA256 to PBKDF2-HMAC-SHA256 with 100,000 iterations (`api/auth.py`)
  - Configurable via `API_KEY_HASH_SALT` and `API_KEY_HASH_ITERATIONS` environment variables
  - Provides strong protection against brute-force and rainbow table attacks
- **URL Sanitization**: Fixed incomplete URL substring sanitization with proper `urlparse` validation (`test_dataset_loaders.py`)
- **Sensitive Data Logging**: Changed 6 locations from `info` to `debug` level with redacted messages
- **Superclass Attribute Shadowing**: Fixed inheritance issues in `BaseVLMDetector` and `BaseVisualDetector`
- **Empty Except Blocks**: Added explanatory comments to 25+ empty except blocks
- **Explicit Exports**: Added TYPE_CHECKING imports for 33 explicit exports in `__init__.py` files
- **Redundant Assignments**: Removed 3 redundant self-assignments in `global_omni_scalar_network.py`
- **Illegal Raise**: Fixed by adding validation and using `RuntimeError` (`resilience.py`)
- **Parameter Name**: Corrected argument name from `significance_level` to `p_value_threshold` (`engine.py`)

### Changed - v1.1.0
- All documentation updated to v1.1.0 with 2026-01-09 date
- README.md comprehensively updated with latest benchmarks and features
- Test coverage increased by 20% (from 58% to 78%+)

## [0.1.0] - 2025-10-14

### Added
- Initial release of Mercury Agent (formerly Omni-Anomaly-Engine)
- 13 fused detection engines with neural network fusion
- 150+ ethical scalars from ancient wisdom and modern principles
- 17+ infrastructure modules for critical sectors (healthcare, cyber, energy, etc.)
- 3R mechanism (Recursion-Resonance-Refactoring) for self-optimization
- Multiverse exploration for scenario analysis
- CRISPR-inspired self-healing for adaptive immunity
- Humanitarian extensions: Cyber Fortress, Medical Cure Predictor, Emergent Life Detector
- 140+ experiments with statistical validation (20-48% gains vs baselines)
- 730+ tests
- Graph-based anomaly detection with NetworkX
- Multimodal fusion with attention mechanisms
- VAE for unsupervised pattern learning
- HATCN-AD for multi-scale temporal prediction
- 5 new civilization modules: Climate Resilience, AgriFood Security, Education Equity, Economic Resilience, Neuroscience
- 150+ domain-specific ethical scalars

### Changed
- Renamed from Omni-Anomaly-Engine to Mercury Agent
- Enhanced CLI with argparse for humanitarian demo
- Improved documentation with simulation disclaimers

### Security
- Added Dependabot for dependency scanning
- Implemented security.yaml workflow for automated scans

### Documentation
- Added comprehensive CONTRIBUTING.md with AI-assisted guidelines
- Created CODE_OF_CONDUCT.md using Contributor Covenant
- Enhanced README with quick-start guide and limitations section
- Added simulation disclaimers throughout documentation

### Note
**All benchmarks based on simulated data. Real-world validation recommended before production use.**

[Unreleased]: https://github.com/Steel-SecAdv-LLC/Mercury-Agent/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/Steel-SecAdv-LLC/Mercury-Agent/compare/v1.7.0...v2.0.0
[1.7.0]: https://github.com/Steel-SecAdv-LLC/Mercury-Agent/compare/v1.6.0...v1.7.0
[1.6.0]: https://github.com/Steel-SecAdv-LLC/Mercury-Agent/compare/v1.5.1...v1.6.0
[1.5.1]: https://github.com/Steel-SecAdv-LLC/Mercury-Agent/compare/v1.5.0...v1.5.1
[1.5.0]: https://github.com/Steel-SecAdv-LLC/Mercury-Agent/compare/v1.4.0...v1.5.0
[1.4.0]: https://github.com/Steel-SecAdv-LLC/Mercury-Agent/compare/v1.2.0...v1.4.0
[1.2.0]: https://github.com/Steel-SecAdv-LLC/Mercury-Agent/compare/v1.1.2...v1.2.0
[1.1.2]: https://github.com/Steel-SecAdv-LLC/Mercury-Agent/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/Steel-SecAdv-LLC/Mercury-Agent/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/Steel-SecAdv-LLC/Mercury-Agent/compare/v0.1.0...v1.1.0
[0.1.0]: https://github.com/Steel-SecAdv-LLC/Mercury-Agent/releases/tag/v0.1.0
