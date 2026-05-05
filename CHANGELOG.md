# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Reproducibility note (applies to all 1.x release entries below):**
> Headline benchmark numbers in this changelog are computed over the
> **64 reproducible datasets** (of 75 attempted). 11 datasets currently
> fail to load due to unavailable external sources (SMAP, MSL,
> CICIDS-2017, MIT-BIH, UCR, SWaT, WADI, USGS Geochemistry, NOAA
> StormEvents, NOAA ERDDAP, FEMA HazardMitigation), and 1 of the 64
> (FEMA Disaster) is a known-broken loader producing inverted scores.
> See the README "Empirical Benchmark Results" section for the full
> reproducibility footnote and `docs/ROADMAP.md` for tracked fixes.

## [Unreleased]

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
  - `docs/index.md` — landing page rewritten to surface the Wave B
    dual-gate contract, AMA Cryptography sole-backend hard-require,
    honest-benchmark framing (64/75), and pickle-removal up front;
    `COMPREHENSIVE_REPO_AUDIT` added to the toctree.
  - `docs/ROUTING_GUIDE.md` — top-of-file callout that hard ethical
    gates run *inside* the prediction call and **must not** be
    masked by fallback handlers. "Fallback only applies to
    data-source / connectivity / latency failures" clarification
    added to the Overview.
  - `docs/COMPREHENSIVE_REPO_AUDIT.md` — historical-document banner
    and resolution-status table mapping the original CRITICAL/HIGH
    findings to the PRs that remediated them (#166 / #167 / #168 /
    #144 / #162 / #165 / #179).
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
- **Tracked debt (not fixed in this sweep):** 34 unpaired
  `# type: ignore[no-redef]` suppressions across
  `safeguards/nano_safeguards.py`,
  `detectors/geological/disaster_detectors.py`,
  `detectors/acceleration_dynamics.py`,
  `detectors/dimensional.py`,
  `integrations/mercury_amacrypto.py`,
  `ml/harmonic_encoder.py`,
  `medical/abms_disciplines.py`. These guard repeated stub-class
  redefinitions across optional-dependency branches and should be
  refactored to a Protocol-or-inheritance pattern; tracked for a
  follow-up PR rather than churning the boundary code in this
  documentation/quality sweep.

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
    checks land **pending** on the bot PR and need to be unblocked
    by either re-running the CI workflow manually on the PR head
    or configuring a Personal Access Token (or GitHub App token)
    as `BENCHMARK_BOT_TOKEN` in repo secrets — the workflow reads
    `secrets.BENCHMARK_BOT_TOKEN || secrets.GITHUB_TOKEN`, so
    setting that secret automatically switches every future
    benchmark run to the trigger-aware token without further
    workflow edits.
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

- **Inline markers restored** for unresolved findings from
  `docs/COMPREHENSIVE_REPO_AUDIT.md` (Phase 2 ITEM 6). High-impact
  cited lines now carry
  `# TODO(audit-2026-03, severity=critical|high|medium|low):` markers
  at the cited locations
  (`core/ai_ethics.py:139,141`, `core/ethical_governor.py:209-210`,
  `core/three_r/fusion.py:91`). Findings closed by Phase 2 (GOSNN
  attention placeholder, GOSNN dead `_fusion`, conformal silent
  failure, ethics-decision-boundary advisory mode) are marked
  **CLOSED** in the audit doc with citations to the regression
  suites. `CONTRIBUTING.md` codifies the rule going forward: every
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
    cannot race the gate.  σ_Immutable is now trained (99.6% val_acc
    on a labelled scalar-vector corpus; weights persisted at
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
  2026-03 in-tree audit (`docs/COMPREHENSIVE_REPO_AUDIT.md` §1) flagged on
  the federated/GOSNN path are now closed:
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
- 730+ tests with 85%+ coverage
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

[0.1.0]: https://github.com/Steel-SecAdv-LLC/Mercury-Agent/releases/tag/v0.1.0
