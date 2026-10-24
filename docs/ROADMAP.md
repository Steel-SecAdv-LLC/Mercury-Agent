# Mercury Agent - Strategic Engineering Roadmap

Applies to Mercury Agent **v2.1.x**. Last updated: 2026-07-21.

The "v1.7.x Deferred Items" ledger below is the backlog deferred at the v1.7.0
cut, carried forward and tracked here (closed rows are marked inline); the
capability table and strategic-vision sections that follow are the
forward-looking roadmap.

## v1.7.x Deferred Items (consolidated)

The following items are deliberately deferred past the v1.7.0 release
cut. Each row names the surface, the precise gap, and the locking
artifact that pins behaviour today so a future PR cannot regress the
contract while the feature is built out.

| # | Surface | Gap | Locked by |
|---|---------|-----|-----------|
| 1 (closed) | σ_Immutable Wave C — narrative voice + federation | **CLOSED (2026-06-02).** `narrative/voice.py::{speak, process_detection, alert}`, `federation/aggregator.py::{submit, aggregate}`, and `federated_learning/server.py::_execute_round` now run the benevolence + σ_Immutable dual hard gate (gates constructed eagerly in each `__init__`; every domain hint routed through `sanitize_domain`; `alert` gained a `domain=` kwarg). The 256-D vector builder is promoted to the single shared `security.sigma_immutable_gate.build_sigma_immutable_vector` helper and the engine / orchestrator / hub copies all delegate to it. | `tests/ethical/test_hard_enforcement.py::{TestNarrativeVoiceBoundary, TestFederatedAggregatorBoundary, TestFederatedServerRoundBoundary}` — each pins legitimate-pass on the real gate plus `check="benevolence"` / `"sigma_immutable"` / `"gosnn_unavailable"`. |
| 2 | AMA HMAC-SHA-384 binding (HS384 path) | ✅ CLOSED. AMA Cryptography v3.3.0 adds native HMAC-SHA-256/384/512 in C with Python bindings; `native_jwt` routes HS384 through AMA's `native_hmac_sha384` fail-closed (no stdlib downgrade), alongside HS256/HS512. | `tests/security/test_native_jwt_ama_routing.py::TestSigningBackendSurface::test_hs384_routes_ama` + the RFC 4231 §4.2 SHA-384 KAT. |
| 3 (closed) | VLM detector surface | **CLOSED (2026-06-02).** `detectors/vlm/base_vlm.py`'s five contract methods (`_initialize_model`, `_create_prompt`, `_parse_response`, `detect`, `extract_features`) are genuine `@abstractmethod` declarations — no `NotImplementedError` stub on the public path; `BaseVLMDetector()` raises `TypeError`. **The surface is no longer abstract-only:** `detectors/vlm/statistical_vlm.py::StatisticalVLMDetector` is a concrete, instantiable, **fully-offline** detector implementing all five methods via deterministic salience statistics (no `transformers`, no model download). The network-backed AnyAnomaly / LAVAD / BLIP backends remain the production path; the statistical detector is the offline / CI default + documented surrogate (remediation steps in its module docstring). | `tests/test_vlm_detectors.py::{TestBaseVLMDetector, TestStatisticalVLMDetector}` — base is abstract / not instantiable + the five methods are abstract; the concrete detector is instantiated offline and its contract (shape, salience ordering, determinism, VQA round-trip) is exercised by 11 behavioural tests. |
| 4 (closed) | Visual base detector | **CLOSED (2026-06-02).** `detectors/visual/base_visual.py`'s three contract methods (`fit`, `detect`, `extract_features`) are now genuine `@abstractmethod` declarations — no `NotImplementedError` stub on the public path; `BaseVisualDetector()` raises `TypeError`. The native SOTA detectors (PatchCore, PaDiM, STFPM, ReverseDistillation, CFlow) are the concrete implementations. | `tests/test_visual_detectors.py::TestBaseVisualDetector` — base is abstract / not instantiable, the three methods are abstract, and a concrete subclass exercises the concrete `preprocess` / `postprocess` helpers. |
| 5 (closed) | GOSNN coupling wired into FL aggregator training loop | **CLOSED (2026-06-02).** `FederatedServer._execute_round` now routes every round's weights through `GOSNNCoupling{Server,Client}` (`publish → ingest → aggregate → receive`) with SHA3-256 + shape + round integrity, failing the round closed on mismatch. Unit-LR FedAvg flows through the coupling's digested weighted mean; FedAdam / SCAFFOLD / secure-aggregation keep their numerics and are installed + broadcast through the new `GOSNNCouplingServer.install_global_state`, preserving the `LocalUpdate` / `RoundResult` fields and the privacy-engine / secure-aggregation branches. | `tests/federated/test_no_silent_failure.py::test_federated_server_round_drives_gosnn_digested_fedavg_path` drives a live `FederatedServer` round end-to-end and asserts weights flow through the digested FedAvg path; the idempotent / shape-mismatch / digest-corruption / `install_global_state` companions remain. |
| 6 (closed) | Intersectional fairness metrics | **CLOSED (2026-07-21).** `ml/fairness.py` gains joint-subgroup metrics: `build_intersectional_groups` (named joint-cell construction), `FairnessAuditor.compute_intersectional_parity` + `compute_intersectional_equalized_odds` (selection-rate / TPR-FPR gaps across crossed `(race, gender)`-style cells, with a configurable `intersectional_min_group_size` floor that excludes — and reports — sparse cells so a one-sample cell cannot dominate the gap, plus an `insufficient_data` flag instead of a fabricated verdict), and `audit()` accepts multi-feature input (mapping of name → values, or 2-D array + `feature_names`), computing per-feature marginal metrics alongside the joint metrics. The engine's `_audit_fairness` dict shape is now the natively-typed `audit()` input (the `# type: ignore[arg-type]` that masked a dead runtime path is removed). Two mitigation defects fixed in the same pass: the threshold-optimization transform compared raw group arrays against stringified keys (integer-typed groups matched nothing and every prediction silently became 0.0; unseen groups were zero-filled instead of defaulting to 0.5), and `_fit_reweighting` was a placeholder — it is now a real Kamiran–Calders implementation (`P(g)·P(y)/P(g,y)`) exposing `get_sample_weights`. | `tests/fairness/` (41 tests, graduated strict-mypy lane): `test_intersectional_metrics.py` — the Simpson's-paradox regression (all marginals exactly fair, joint disparity 0.3 flagged as a violation naming the worst-off cell), small-cell exclusion, audit key-shape compatibility, hypothesis property invariants (bounded scores, permutation invariance); `test_bias_mitigation.py` — integer-group zero-fill regression, unseen-group default, reweighing closed-form values + weighted joint-independence property. |
| 7 (closed) | Concrete `AttentionProvider` implementation | **CLOSED (2026-06-02).** `core/gosnn_optimizer.py::MultiHeadAttentionProvider` is a concrete provider wired to a real `torch.nn.MultiheadAttention` surface: `observe(sequence)` runs a forward pass and caches the per-head `(num_heads, seq_len, seq_len)` weights (`average_attn_weights=False`); `get_attention()` returns them, or raises `RuntimeError` before any forward (fail-closed "model not yet run"). Defaults to 32 heads to match `AttentionOptimizer`'s triadic φ-weighting so it drives `GOSNNOptimizer` directly. The deterministic-random placeholder is gone. | `tests/core/test_attention_provider.py` (11 tests) — per-head shape, softmax-normalised rows, fail-closed pre-forward, determinism, batched input, and that a wired+observed provider drives the optimizer metric (no skip) while an unobserved one fails closed to a skip. |
| 8 (closed) | Mutation testing on σ_Immutable hot path | **CLOSED (2026-07-21).** `scripts/run_sigma_mutation_gate.py` — a deterministic AST-based mutation harness over the σ hot path (`security/sigma_immutable_gate.py` + `sigma_immutable_corpus.py`), applying mutants in place with byte-exact restoration, red-baseline abort, timeout-as-killed, stride-sampled runtime bounds, and a JSON report. `mutmut`/`cosmic-ray` were evaluated and rejected: their import/trampoline models fight the package's import-time native-PQC gate and offer no deterministic bounded-runtime mode. The first honest measurement with the pre-existing interface-level σ subset killed **9.7%** (12/124) — arithmetic mutants (projection band math, vector builders, corpus generation) all survived — so two semantic pinning suites were written specifically to kill the survivor classes (`tests/security/test_sigma_immutable_{gate,corpus}_semantics.py`, every assertion annotated with the mutation class it kills, plus fail-closed contract pins: an unavailable gate reports `passes=False, score=0.0`; an anchor exactly at the critical floor is compliant; an EthicalGate without `_trained` is refused). Final calibration: **119/120 killed — 99.2%** (gate 50/51, corpus 69/69 after two cleanups eliminated equivalent-mutant sites outright); the single survivor is an `exc_info=True` flip inside a diagnostic `logger.warning` (log verbosity, zero behavioral effect), documented in `docs/MAINT1_2_COVERAGE_REMEDIATION.md` §3. | `.github/workflows/mutation-testing.yml` — blocking, path-filtered PR lane + weekly cron, `--fail-under 90` (measured 99.2% minus a site-addition margin); harness semantics pinned by `tests/scripts/test_run_sigma_mutation_gate.py` (11 tests incl. red-baseline abort and byte-exact restoration). |
| 9 (closed) | Lyapunov-stability benchmark + λ reconciliation | **CLOSED (2026-06-02; premise was stale).** `scripts/run_ablation.py` **exists** (a real Lyapunov-pre-gated ablation runner with documented exit codes, referenced correctly by `configs/ablation_3r_lyapunov.yaml`). The `V̇ ≤ -λV` claim is certified by `tools/lyapunov_validator.py` against `configs/lyapunov_canonical.yaml`: `claimed_lambda=0.25, computed_lambda=0.5, ok=true` (2× margin). The README does **not** cite three conflicting λ values — `λ_convergence = 0.25` (Lyapunov convergence, canonical/certified) and `λ_decay = 0.18` (double-helix adaptation rate) are **distinct constants by design**, there is no `0.13`, and the `Docs λ Drift Gate` CI job (`scripts/check_readme_lyapunov.py`) enforces this — it passes ("all documented λ claims match canonical … scanned 2 files"). | `tools/lyapunov_validator.py` + `scripts/check_readme_lyapunov.py` (Docs λ Drift Gate in `iso-hardening.yml`). |
| 10 (closed) | `tests/load/` wired into CI | **CLOSED (stale row — the gate already exists and runs green).** `iso-hardening.yml`'s "Load Tests (smoke)" job invokes both `tests/load/k6_load_test.js` (k6 smoke mode) and `locust -f tests/load/locustfile.py` (headless, non-zero exit on failed requests). | `.github/workflows/iso-hardening.yml` job "Load Tests (smoke)". |
| 11 (closed) | Examples-parity CI | **CLOSED (stale row — the gate already exists and runs green).** `iso-hardening.yml`'s "Examples Parity" job runs the shippable demos (`examples/humanitarian_demo.py`, `examples/live_anomaly_demo.py`, `examples/physics_detectors_demo.py`) end-to-end and asserts on their output. | `.github/workflows/iso-hardening.yml` job "Examples Parity". |
| 12 (closed) | `tests/loaders/` + `tests/narrative/` graduate to strict mypy lane | **CLOSED (stale row — already graduated).** `ci.yml`'s "Run MyPy strict on graduated test directories" step already lists `tests/loaders/` and `tests/narrative/` next to `tests/datasets/`/`tests/ethical/`/`tests/safeguards/`/`tests/tools/`; both pass under `--disallow-subclassing-any` in CI. | `.github/workflows/ci.yml` job `type-checking` strict step. |
| 13 (closed) | Core coverage floor bump 15 → 25 | **CLOSED in v1.7.x.** Core lane expanded to include `tests/detectors/`, `tests/ml/`, `tests/datasets/`, `tests/api/`, `tests/automl/`, plus 13 root-level `test_*.py` additions. Measured combined stmt+branch coverage on the expanded lane is ≥25 % with a several-point cushion. | `.github/workflows/ci.yml` env `COVERAGE_THRESHOLD_CORE: 25` + the per-job `--cov-fail-under` flag in the `core-tests` job. |
| 14 (closed) | Serve-path explanations | **CLOSED (2026-06-02).** The validated `cognitive/explainability.py::{IntegratedGradientsExplainer, FaithfulnessEvaluator}` are wired into `OmniMercuryEngine.detect_with_fusion(explain=True)`, which attaches an IG attribution of the served `score_fusion` probability + faithfulness scores. Was deferred by PR #269 (explainer validated in-tree but not wired into the detect result). | `tests/benchmarks/test_explanation_fidelity.py::test_faithfulness_non_regression_comprehensiveness_and_recovery` (comprehensiveness > random + 0.01, recovery@k > 2× chance) + `tests/test_fusion_explainability.py`. |
| 15 (closed) | Fusion AUC/F1 + conformal-coverage CI regression gate | **CLOSED (2026-06-02).** `benchmarks/fusion_regression_guard.py` (`--check`/`--update`) deterministically trains+evaluates the fusion path on a seeded synthetic corpus and fails on AUC/F1 below `baseline − margin` or empirical conformal coverage below `target − 0.05`; committed baseline `benchmarks/fusion_capacity/fusion_gate_baseline.json` + artifact `artifacts/fusion/<ts>/metrics.json`; run by `.github/workflows/fusion-regression.yml`. Was deferred by PR #269 (pipeline + sweep artifacts existed, no CI floor gate). | `tests/benchmarks/test_fusion_regression_guard.py` (gate logic) + the workflow's `--check` step. |
| 16 | Fusion checkpoint round-trip fidelity + shipped-checkpoint quality | **CLOSED (2026-06-11).** Root causes measured and fixed: the checkpoint persisted only the network + temperature, so a loaded engine auto-fit its base detectors on the first inference batch (the leak behind both (a) and (b)) and re-randomized torch-module / population domain models; two domain-model stubs additionally emitted *fresh RNG output per call* (`affective` fed 64 columns of noise to fusion — now quarantined to deterministic neutral output) and two streaming buffers (directive RMD, neural hippocampal) leaked across calls, so even one engine drifted ±0.08 on repeated identical batches. Remediation shipped: (1) fitted base-detector state persists via `get_fitted_state()`/`set_fitted_state()` on all five detectors (incl. Oracle re-arm via the federation stats path); (2) torch-module domain-model weights (`model_state_dicts`), the multiverse population (`model_fitted_state`), and the conformal calibrator (`conformal_state`) round-trip; (3) the engine's fusion feature boundary resets transient streaming state (purity contract); (4) `default_fusion.pt` regenerated with full state. **Measured: save→load max\|Δ probability\| = 5.9e-08 (bar < 1e-3; was ≈0.76 historical / 0.427 re-measured), same-engine repeat drift exactly 0.0, conformal sets identical.** Pinned by `tests/test_fusion_checkpoint_roundtrip.py` (probability-equivalence bar, no-auto-fit-after-load, exact detector-state round-trip, conformal round-trip, legacy-checkpoint compatibility); fusion regression gate re-run green. **Follow-up (2026-10-17):** the reset in (3) ran before the first of the engine's two passes (`extract_features`, then `predict`/`detect`) only, so the second pass scored a batch's first rows against its last rows; with only the last 100 of 300 rows replaced, 99 of the first 100 neural scores moved. It now runs before each pass (`engine.py::_reset_transient_state`). | Closed by the 2026-06-11 optimization & efficiency pass; regression test keeps it closed. Follow-up pinned by `tests/core/test_engine_pass_purity.py`. |
| 17 | Headline benchmark refresh (9 failed corpora) | **OPEN (external blocker).** The committed run (`benchmarks/mercury_benchmark_results.json`, 2026-10-14) measures 76 datasets attempted / 67 successful / 9 failed (Mean AUC 0.8545, Median AUC 0.9052, n_genuine_labeled 54). A fuller refresh of the 9 failed, network-dependent corpora requires the full benchmark pipeline (many external corpora, hours); not run here, so the headline stays at its last real committed measurement rather than being fabricated. **Mitigation:** the offline-reproducible fusion floors are now CI-gated (row 15). **Offline CI-gated fallback (engineering-pass round):** `scripts/check_benchmark_integrity.py` (dependency-free, no-network; blocking step in the `code-quality`/workflow-hardening lane) fails closed unless the committed `mercury_benchmark_results.json` is internally consistent (`total == successful + failed`, `failed == #per_dataset-with-error`, `n_genuine_labeled == #successful-ground_truth`), the headline Mean/Median AUC **and** oracle-F1 **recompute** exactly from the genuine-labelled `per_dataset` rows (a restated-but-fabricated headline fails), and the README `<!-- BENCHMARK -->` block matches that JSON — so while the live refresh is deferred the headline cannot rot or be fabricated. **Dependency list — the 9 currently-failed corpora and what each needs for the live refresh:** `SMAP`, `MSL` (NASA telemetry — NetManAIOps GitHub mirror, network); `CICIDS-2017` (security — all download mirrors down, HuggingFace `datasets` or manual fetch); `MIT-BIH` (medical — PhysioNet + the `wfdb` lib via `pip install mercury-agent-global[medical]`); `SWaT`, `WADI` (industrial — iTrust SUTD, registration/access-controlled); `FEMA_HazardMitigation` (disaster — OpenFEMA v2 API, network); `USGS_Geochemistry`, `UCR` (environmental/academic — **not** network: the fetched slice had `<2` distinct label values, so these need a labelled subset with ≥2 classes, not merely connectivity). Reachability of the 7 network-blocked corpora is probed nightly by `dataset-reachability.yml`. **Remediation plan:** run `python benchmarks/mercury_benchmark.py` (or the `Benchmark Pipeline` workflow, weekly cron) with network access + `MERCURY_NETWORK_TESTS=1`, then commit the refreshed `benchmarks/*.json` + README "Latest Benchmark Results" block via the existing persistence step. | `.github/workflows/benchmark.yml` (persistence on `main`) + `scripts/check_benchmark_integrity.py` (offline integrity gate, pinned by `tests/scripts/test_check_benchmark_integrity.py`) + `dataset-reachability.yml` (nightly reachability of the 7 network-blocked corpora). |
| 18 | Multilingual natural-language interface | **OPEN (future epic).** Mercury's narrative / voice surface (`narrative/voice.py`, `narrative/interface.py`, `api/voice.py`) operates in **English only**. "Multi-language" in the README refers to the *implementation* stack (Python / Rust / C-C++), **not** localized natural-language I/O. Non-English intent parsing, response generation, and evaluation are not current capabilities; this row scopes them so the README cannot be read as claiming multilingual NL. **Concrete build-out plan (phased, so the epic is tractable rather than open-ended):** (P1) *language detection + routing* — add a `detect_language()` front-door to `narrative/interface.py` that tags each query and, until targets ship, fails closed with an explicit "English-only today" response instead of silently mis-parsing; (P2) *intent parsing per target language* — locale-specific normalizers feeding the existing intent layer; (P3) *response generation* — localized narration templates + number/date formatting; (P4) *evaluation harness* — a per-language regression set (intent-accuracy + response-adequacy) gated like the English suite. **Dependency list (external, offline-CI-gated fallback where each is unavailable):** a multilingual intent dataset (e.g. MASSIVE/xSID — offline fixture slice for CI, full set for training); a localized-response reference set (net-new, authored per language); optional MT/embedding models pinned to a vendored offline snapshot so CI never reaches the network (mirrors the VLM `StatisticalVLMDetector` offline-surrogate pattern). **CI-gated mitigation (this round):** `tests/narrative/test_language_scope.py` promotes the README honesty lock from prose to an enforced gate — it fails if the "operates in English" scope, the "multi-natural-language is not shipped" disclaimer, or the "future epic" framing is deleted/weakened, so the docs cannot silently start claiming multilingual NL before P1–P4 actually ship. | The README "Implementation Languages" block (`## Build System`) pins the implementation-vs-natural-language distinction, now **enforced** by `tests/narrative/test_language_scope.py` (English-only scope + no-shipped-multilingual-NL-claim guard). |
| 19 (closed) | Orphaned modules: `core/di.py` + `detectors/cross_domain_frequency.py` wire-or-retire | **CLOSED (2026-07-21, same round — both decisions executed).** The coverage round found both modules had **zero importers** anywhere in src/tests/scripts/tools (`core.di`'s previously-claimed importers were regex false positives; `cross_domain_frequency` was referenced only in docs). Decisions: **(a) `cross_domain_frequency` WIRED** — it is fully implemented and advertised by ARCHITECTURE.md / CAPABILITY_INVENTORY.md, so it now exports through the `omni_mercury_engine.detectors` public surface (lazy export + `__all__`: `CrossDomainFrequencyCorrelator`, `CrossDomainCorrelation`, `BandOverlap`, `DomainBandInfo`). **(b) `core/di.py` DEPRECATED** (kept functional per the DEPRECATION.md preservation policy): zero consumers since inception, its detector/model maps had bit-rotted to nonexistent class names (fixed in the same round — only possible in dead code), and its constructor injection is inert for every class defined under `from __future__ import annotations` (string annotations → silently no injection), which is all of src/. Import now emits `DIDeprecationWarning` (suppressable via `MERCURY_AGENT_SUPPRESS_DEPRECATION_WARNINGS`); DEPRECATION.md §1.3 records the evidence and preservation justification. | `tests/detectors/test_cross_domain_frequency.py::TestPublicWiring` pins the export wiring; `tests/test_compat_shims.py::TestDIDeprecation` pins the warning + suppression + still-functional contract; `tests/core/test_di.py` (56 tests, 94%) keeps the full container contract pinned. |
| 20 | Inherited suppression markers | **OPEN (separate maintenance pass).** 1,190 lines in tracked `.py` files carry a marker: `type: ignore` 921, `# noqa` 102, `pragma: no cover` 167 (by directory: src 655, tests 415, scripts 54, benchmarks 36, research 23, tools 3, examples 2, docs 2; measured 2026-10-23). `steel/version-cov5` removed the 100 on the paths it touched (`75fc079`: 59 its own, 41 from main) and adds none. Each needs a root-cause fix, not a re-justified marker. | `git grep -nE '# *noqa\|type: *ignore\|pragma: *no cover' -- '*.py'` reproduces the count. |
| 21 | Test run hides deprecation warnings | **OPEN.** `pyproject.toml` `[tool.pytest.ini_options] filterwarnings` ignores `DeprecationWarning` and `PendingDeprecationWarning` for the whole suite, so deprecations in Mercury code and in its dependencies never surface. Unhidden (`-W default::DeprecationWarning -W default::PendingDeprecationWarning`, 2026-10-23) the full run shows 166 warnings instead of 73: 76 deprecations and 17 `PytestRemovedIn10Warning` (class-scoped fixtures defined as instance methods). Those that will break on an upgrade: NumPy 2 `__array__` without a `copy` keyword (10; `models/affective.py:106`, `models/astrophysical.py:68`, `models/biometric.py:403,659` and others), NumPy's "generic" timedelta unit, which "will raise an error in the future" (7; `loaders/fema_loader.py:716`), `torch.ao.quantization`, removed in torch 2.10 (10; `ml/compression.py:413`), Starlette per-request test-client cookies (6), and `fork()` in a multi-threaded process (6). `src/` still imports Mercury's own deprecated modules: `engine.py:183` (`detectors.directive`), `detectors/advanced/__init__.py:39` (`gwo_ensemble`), `validation/__init__.py:15` (`validation.data_loaders`). | `pyproject.toml` `filterwarnings`; `pytest tests -n auto -W default::DeprecationWarning -W default::PendingDeprecationWarning`. |
| 22 | 73 warnings in the full test run, traced | **OPEN.** 2026-10-23 run (20,118 passed), every warning located: 27 scipy coherence 0/0 in `detectors/spectral_domain_frequency.py::_compute_phase_coherence` (adjacent band-filtered signals have no power outside their bands; the NaN bins are dropped afterwards), reached from `physics_integrated`, the `MercuryAnomalyDetector` ensemble (it also prints in `mercury_quickstart.py`), the drone detector and component validation; 6 divide by zero at `models/quantum.py:183`; 6 scipy "precision loss in moment calculation" (the validation A/B-tester tests and a benchmark test); 11 numpy mean/variance of empty or one-sample slices (iris single-image tests, validation pipeline); 2 `nanmean` of an empty slice (ocean buoy loader); 9 overflows (`explainability/counterfactuals.py:690-699`, the counterfactual tests' sigmoids, a `dot` and a `reduce`); 5 invalid-value results (divides at `models/neural.py:168`, its test and numpy `_function_base_impl.py:3036-3037`; a subtract in scipy `_numdiff`); 4 scipy stats invalid multiply / divide by zero; 3 torch `UserWarning`s (two tests convert a `requires_grad` tensor to a float; the quantized-tensor API). | The warnings summary of `pytest tests -n auto`. |
| 23 | CI security audit cannot fail the build | **OPEN.** `ci.yml:411` runs zizmor with `--min-severity high --no-exit-codes`: findings are printed and the job passes. zizmor 1.24.1 (offline) over `.github/workflows`: 236 findings, 94 of them suppressed by zizmor's own defaults (the repo has no inline or config ignore); 8 high, all present on main: `excessive-permissions` ×6 (`benchmark.yml:12,13,19`, `release.yml:38,39,40`), `cache-poisoning` (`release.yml:177`), `bot-conditions` (`dependabot-auto-merge.yml:36`); 125 medium, 123 of them `artipacked` (checkouts that persist credentials), 3 of which `steel/version-cov5` added in `mutation-testing.yml`. | `uvx zizmor==1.24.1 --offline .github/workflows`. |
| 24 | CI flake8 ignores unused variables | **OPEN (dormant).** `ci.yml:175` adds `F841` to `--extend-ignore`; `.flake8` and `pyproject.toml` do not. No `F841` exists in `src/ tests/ scripts/ tools/` today, so the flag hides nothing yet. | `flake8 --select F841 src tests scripts tools` (0). |
| 25 | `benchmarks/` is not type-checked | **OPEN.** CI runs mypy on `src/` and tests only. Against the installed package, `benchmarks/` has 189 errors in 29 of 111 files (`no-untyped-def` 35, `no-untyped-call` 32, `arg-type` 31, `assignment` 19, `var-annotated` 14, `name-defined` 12 — all `plt.Axes` annotations, not runtime names — and others). | `mypy --python-executable $(which python) --follow-imports=silent benchmarks/`. |
| 26 | Optional upload kept non-fatal | **OPEN (deliberate).** `dataset-reachability.yml` keeps `if-no-files-found: ignore` for its optional debug logs; every other report upload fails on a missing file since `75fc079`. | `dataset-reachability.yml`. |
| 27 | `roc_auc_score` floating-point sum | **OPEN.** `ml.mercury_ml.roc_auc_score` returns 0.9999999999999999 for a perfect ranking (trapezoid sum), not 1.0. | `tests/test_mercury_quickstart.py::test_published_protocol_fits_on_training_rows_only` compares with `pytest.approx`. |
| 28 (closed) | Deception failures are log-only | **CLOSED (2026-10-23).** The decoy surface still never blocks or breaks an answer, but every survived failure is now visible: `DeceptionSensor.failure_counts()` and `note_failure()` over the closed set `FAILURE_KINDS` (`audit_sink`, `store_record`, `store_outcomes`, `handler`, `tripwire`); `TripwireEvent.degraded` names the failures hit while recording that touch; a lost audit record is carried as `audits_failed` on the next audited event, so the tamper-evident chain accounts for it (as `audits_suppressed` does for the cap); the route layer exports `mercury_deception_failures_total{kind}` on the existing `/metrics`. | `tests/security/test_deception.py::{TestAuditVolume,TestSqliteTripwireStore}`, `tests/api/test_trust_surface_routes.py::TestFailClosed`, `tests/api/test_platform_metrics.py` (each fails on the old source). |
| 29 | `spectral_vibration`: remaining limits | **PARTLY CLOSED (2026-10-23).** Closed: the networks are seeded (`config["network_seed"]`, forked RNG) so fresh instances are bit-identical (the same data scored 0.3548, 0.3689, 0.3491, 0.3400 before); one spectral estimator at every signal length (a detector fitted on 1,024-sample signals scored every normal 1,025-sample signal 1.000: the Welch path was Hann-windowed, the single-frame path was not); the old tests that asserted `anomaly_score > 0.0` now assert flagged / not flagged. **Open:** the one-signal spread uses 8 blocks chosen on synthetic tone, noise and two-tone signals only (no recorded vibration data; the C14 row covers this); a 1,000-sample signal is analysed at 64-sample frames (15.6 Hz bins at 1 kHz), a fault close to a tone was missed in one measured case (0.38), and a single signal under 384 samples is refused. | `tests/detectors/test_spectral_vibration_estimator.py`, `tests/detectors/test_spectral_vibration_single_signal.py`, `tests/test_spectral_vibration.py`. |
| 30 | Spectral CLI and API scores | **PARTLY CLOSED (2026-10-23).** API: the `/detect/fusion` spectral score was `max|FFT| / N` of the raw data (an offset of 1000 scored 1000 and made any series an anomaly, fused 365 against 1.7); it is now Fisher's dominant-frequency test as a z-score (scale- and offset-free, white-noise false-alarm rate at or below the nominal normal tail). CLI: `mercury physics spectral` fitted on its input and scored the same input (one constant, 0.1337, for a clean tone and for a noise burst) and printed placeholder `spectral_entropy` / `harmonic_distortion` / `vibration_signature`; it now takes `--reference` (known-good recordings) for a real verdict, prints the input's own entropy, THD (IEEE 519) and signature, and without a reference says `baseline: input_itself` with a warning. **Open:** a within-recording check (each segment against the others) was tried and refused: 3 of 4 clean 4,000-sample recordings had a normal segment flagged at 1.000 with 7 reference segments; it needs a rule calibrated on labelled recordings, which the repository does not have. | `tests/api/test_fusion_spectral_score.py`, `tests/test_cli_physics_spectral.py`. |
| 31 (closed) | ML Tests lane never run on `steel/version-cov5` | **CLOSED (2026-10-23).** `ci.yml` dispatched with `run_ml_tests=true` on `b9b2690`: ML Tests, Integration Tests and Core Tests on Python 3.11-3.14 passed. That run also failed two jobs, both fixed in `0d546e9`: the README scale-block drift gate and MyPy on Python 3.11 (`MercuryAnomalyDetector` typed `object` after `detectors.statistical` became lazy). | `ci.yml` ML Tests `if:`; run 40006331744. |
| 32 | Detector coverage still open | **PARTLY CLOSED (2026-10-23).** Sorted: every `*Detector` class is a manifest entry or has a role in `core/detector_roles.py` (abstract, component, variant, deprecated or library), re-derived from the source by `tests/core/test_detector_roles.py`; the 28 library and variant classes each declare a `decision_source`, and the role table is rendered into `docs/DETECTORS.md`. Fixed on the way (each measured, then tested against the old code): untrained networks that set verdicts from random weights in endocrinology, TEMPEST, anesthesia, medical vitals/imaging, CYBINT and interstellar objects (see CHANGELOG), and BLIP's unseeded, dropout-on feature projection. **Measured (held-out ROC-AUC, five ADBench datasets, 3 seeds, `benchmarks/advanced_detector_auc.py`):** COPOD 0.869 and AdversarialAE 0.874 against 0.661 for the shipped LOF reference, so both carry signal and are candidates for manifest entries. **Wired at the registry seam (2026-10-24).** The blocker recorded earlier -- `register_detector` raising `TypeError: detector must be a BaseDetector instance` -- is the *engine* seam's requirement. The registry seam carries no such requirement, and `lstm_autoencoder` has shipped at `seams=("registry",)` without being a `BaseDetector` all along, so both are promoted there: the signal is available through `extract_all_features` with no adapter and no change to either class's released API (subclassing would have changed `predict`'s return type and `config`'s type). Three defects were found and fixed in the course of establishing they were fit to promote: the autoencoder drew its reparameterisation sample at inference, so three `predict` calls on one fitted model and one input differed by up to 0.59 per row and flagged 4, 3 and 3 of 40 rows (guarded on `self.training`, now bit-identical); its `seed` reached only the NumPy split generator, leaving the network weights unseeded so the benchmark moved between runs (0.875 then 0.880 -- `ContrastiveLearningDetector` had the identical one-line defect and got the same fix, and still varies from another source); and its threshold was cut on the *training* rows' own reconstruction error, which an autoencoder reconstructs better than anything unseen, measuring a 0.1483 false-alarm rate against the 0.05 its percentile names, now 0.0900 on a disjoint calibration block at no cost to ranking. Their false-alarm rates on normal rows are pinned in `tests/core/test_declared_modality_false_alarm_rate.py`. This clears the first half of the bar `KMeansDistanceDetector`'s docstring sets -- standalone ranking justifies making a signal *available* -- and does **not** claim the second: whether either *adds* to the live fusion ensemble still needs a fusion-marginal ablation, for which no harness exists (`scripts/run_ablation.py` is a Lyapunov-gated config runner, not a detector-marginal measurement). ContrastiveLearningDetector 0.601 does not clear even the first half and stays a library class. **Re-measured 2026-10-24** with the benchmark's own protocol: COPOD and the LOF reference reproduce exactly; AdversarialAE and Contrastive were cited at 0.866 and 0.598 and measured 0.870 and 0.621, which looked like build drift but was their own nondeterminism: both drew network weights unseeded and the autoencoder also sampled its latent at inference. With both fixed, AdversarialAE measures **0.874 reproducibly, to every per-dataset digit**; Contrastive measures 0.601 and still varies from a further source, so its figure remains indicative only. `MultiScaleTransformerDetector` is a sequence model and has no real labelled sequence measurement. **CI, measured on the branch (2026-10-23):** the dispatched Sigma-Immutable Mutation Gate passes every lane (it had failed the hazard lane at 43.5% against 75 and the distillation lane at 76.7% against 80; measured locally with the CI command after the new tests: hazard 93.5%, storm surge 92.6%, distillation 89.7%); the dispatched Security workflow passes (its gitleaks job had reported five test strings, now allowlisted exactly). **Recorded-data tests (2026-10-24):** 36 of the 49 manifest entries that had none are now pinned on recorded data, with the measured value (AUROC, and for series NAB's standard-profile score of the detector's own alarms): `tests/detectors/test_recorded_nab_series.py` (20 series detectors on four NAB files; it was 21 before `deeplog_sequence` moved to the HDFS test below), `test_recorded_hdfs_log_templates.py` (`deeplog_sequence` on 2000 real Drain-parsed HDFS log templates), `test_recorded_adbench_tabular.py` (5 tabular detectors on five ADBench tables; `topological` and `biometric` as set-level window rankers), `test_recorded_windows_and_transactions.py` (`acceleration_dynamics`, `nano_safeguard`, `frequent_pattern`), `test_recorded_nab_foundation.py` (`matrix_profile`, `foundation_ensemble`), `test_recorded_cwru_bearings.py` (`spectral_vibration`, `physics_integrated` on CWRU bearing vibration), `test_recorded_eeg_seizures.py` (`neural_cognitive` on CHB-MIT EEG). Fixtures and their SHA-256: `tests/fixtures/{adbench,cwru,eeg}/PROVENANCE.json`. Defects found by them and fixed, each with a test that fails on the old code: the LSTM autoencoder trained on raw values (mean AUROC 0.596 -> 0.829); the topological detector's threshold ignoring its sample cap (fit 167 s -> 0.1 s on thyroid) and then scoring 25-point windows against an 80-point reference profile, a cardinality mismatch that biased every feature deviation (mean window AUROC 0.821 -> 0.890 over four tables and three seeds, 0.839 -> 0.883 over six); a configured `anomaly_threshold` on the same detector silently replacing the score function (the same clean window scored 0.918 auto and 0.085 configured); Page-Hinkley never restarting after an alarm, so one sustained shift alarmed to the end of the series (NAB standard score -1785.3 -> +26.6, alarms 798/1138/1542/695 -> 4/10/19/9 on the four files, ranking AUROC 0.332 -> 0.534); `PhysicsIntegratedDetector` flagging every healthy bearing window (100% -> 12.5%, faults still 100%, fault AUROC 0.975 — the CHANGELOG's earlier "AUROC 1.0" for that change was measured without the spectral `sample_rate` and is corrected); the frequency-domain Oracle's per-band z-score comparing a dimensionless power ratio with absolute band-power statistics, so its aggregate was exactly 0.21 for every CWRU window, healthy or faulty (now healthy 0.021-0.097, every fault at AUROC 1.000; `physics_integrated` healthy flagged 12.5% -> 7.5% and fault AUROC 0.975 -> 1.000); the Oracle's inter-band phase coherence returning exactly 1.0 for every input, leaving one of the five influence-multiplier terms permanently neutral and emitting twelve `0/0` RuntimeWarnings per `detect` (now abstains explicitly — see below); a failed Oracle dropping out of `component_scores` with only a log line, so a degraded fused verdict was indistinguishable from a healthy one (now `metadata['component_failures']`); `MatrixProfileConfig.normalize` never read and a provenance claim of a z >= 2.0 screen that no code applies. **Measured calibration gaps, not fixed** (NAB standard score, 0 = no alarms, 100 = perfect): `echo_state` -368.3 (down from -1167.5: its squash scale was the 98th percentile of the readout's *in-sample* residuals, so the documented `1 - calibration_quantile` false-alarm rate was out by 12x on a stationary AR(1) series — 0.2375 against 0.02 — and the scale is now the pooled out-of-fold quantile over `calibration_folds` blocks; ranking AUROC unchanged at 0.699, mean normal false-alarm rate 0.611 -> 0.188, alarms 846/475/1134/652 -> 13/13/919/228. Two files still flood — `rogue_agent_key_hold` 919 alarms and `speed_7578` 228 — because their post-probation normal level drifts away from the probationary reference, which no quantile measured inside the training window can bound; that needs an adaptive reference), `deeplog_sequence` (was -780.3 on NAB; **that measurement is withdrawn** — it is a categorical next-key model over integer log-template keys, now declared `event`, and a continuous series gives it no grammar. Its verdict was `1 - P(key|context) > threshold`, which can only fall below 0.5 where one successor holds half the conditional mass; DeepLog's own top-g criterion was computed per position and discarded to a scalar count. The verdict is now the paper's rule conjoined with the threshold: held-out false-alarm rate on real Drain-parsed HDFS templates 0.8270 -> 0.1730, on a synthetic low-entropy grammar 0.0525 -> 0.0025, NAB -780.3 -> -530.3, with an injected unseen template still caught 10/10. It gains its first recorded-data test of its own kind, `tests/detectors/test_recorded_hdfs_log_templates.py` on `tests/fixtures/loghub` — 2000 real HDFS lines, 14 templates, unlabelled, so a false-alarm rate and a novel-template check but no AUROC), `spiking` -318.7, `matrix_profile`/`foundation_ensemble` -82.0, `diffusion_ad` -32.2, `deep_svdd` -30.0: their default thresholds flood. `lstm_autoencoder` was -88.5 here for the same reason and is now **+3.1**, net positive, after the out-of-fold conformal calibration below -- the recorded corpus corroborating the stationary-gate fix, since its AUROC (0.829) is unchanged and only the alarms moved. At chance on NAB's four files: `spot_evt` 0.503, `sprt` 0.494, `imm` 0.510, `spectral_residual` 0.489, `dimensional` 0.466; below chance: `matrix_profile` 0.356 (0.073 on road traffic, the same as raw `stumpy.stump`). `nano_safeguard` flags every NAB window and ranks at 0.33-0.58. `spatial` is at chance on cardio, breastw and Pima (0.566, 0.564, 0.587) and strong on WBC (0.982). `topological` ranks Pima windows at 0.599 (cardio 0.961, breastw 0.999, thyroid 1.000); `frequent_pattern` is at chance (0.41-0.59) on the five ADBench tables binarised at the median (no labelled transaction corpus is recorded, so that is not a measurement on transaction data). `spectral_vibration` fitted on one load flags every healthy window of another. `neural_cognitive` ranks seizure windows at 0.991 and 0.972, but a per-window standard deviation ranks them at 0.997 and 0.982, and its score is unbounded and unit-dependent (about 1e6 for background and 4e7 for seizure, in microvolts). `ChronosAdapter` (chronos-t5-small, CPU, measured once over the three NAB files that have anomaly windows after its 512-point context; the first file's window lies inside the context, whose points score 0): AUROC 0.652, 0.378, 0.598; scoring by |error| or by |error| over the context's MAD gives a mean of 0.59 and 0.63 against 0.54, too little on three files to change the shipped rule. It has no automated pin because one pass takes about 15 minutes on four CPU cores. **Still without a recorded-data test (12 entries + the Chronos pin):** `affective` (takes declared emotion distributions), `astrophysical` (event-horizon features of cosmic-event data), `chemistry` (a dict of element/compound data), `consciousness`, `neurosymbolic`, `psyop` and `uiux_anomaly` (**no corpus search has been run for the input these seven take** -- the earlier wording "no labelled recorded data exists" was never established and is withdrawn; what is established is only that none is recorded in `tests/fixtures`), `drone`, `graph_based` and `rca` (**the candidate corpora named for these three -- CMU's ALFA UAV fault log, and labelled graph-anomaly and root-cause sets -- have not themselves been reached; only unrelated URLs were tried**, so "not reachable" is unproven for them), `medical_abms` (records of board certification), and `timegpt` (needs a Nixtla API key). **Published-parity figures moved by the Oracle fix, upward:** `MercuryAnomalyDetector` fits a `SpectralDomainFrequency` Oracle on `TEMPORAL` input and multiplies its combined scores by that Oracle's influence multiplier before clipping to `[0, 1]`, so the band z-score defect was in the path of `examples/mercury_quickstart.py --published-protocols`. On the digest-pinned DTU AIS training files, same bytes both sides and reproduced locally as well as across CI runs 1170 and 1175: JMSE Fig. 12a ROC-AUC 0.8615 -> 0.8751 and AP 0.2475 -> 0.3085; Thesis Fig. 6.1 ROC-AUC unchanged at 0.8785 with AP 0.3641 -> 0.3827; the harness (0.8613) and label-free cross-fit (0.8617) rows unchanged, since they do not take that path. The mechanism is the clip: the multiplier is one scalar for the whole batch and cannot change a ranking by itself, but `np.clip(..., 0, 1)` after it ties everything the multiplier pushes past 1.0, and the saturated `aggregate_score` made the multiplier larger (1.3069 against 1.2732 on a controlled fit). The pins in `.github/workflows/iso-hardening.yml` and README's output block are updated; GeoTrackNet/VRNN at 0.87 still lies inside both intervals, so the comparison remains parity. **Fail-open closed in `RealTimeThreatDetector`:** `_validated_input` already refused a non-finite input row, naming the reason ("a security detector failing OPEN on an unmeasurable input"), but a non-finite sub-detector *output* went unguarded. A 1-D input validates to one `(1, n_features)` sample, LOF scores `inf`, so `center` was `inf`, `mad` `NaN` (the source of the module's `invalid value encountered in subtract` warnings) and `scale` `NaN` — and `scale <= 1e-12` is False for `NaN`, so the std fallback never fired. The threshold came out `NaN`, `score > NaN` is False for every sample, and the `NaN` column poisoned the ensemble mean so `_ref_threshold`/p90/p95/p99 were all `NaN`. Measured on the same input: a probe 8 sigma from the training distribution graded `is_threat: False`, `threat_level: LOW` before, `is_threat: True`, `CRITICAL` after. An uncalibratable sub-detector is now dropped with the reason logged, the fallback arm rejects a non-finite scale, and a fit where nothing calibrates raises the same refusal as the zero-fitted case. `realtime_threat` is also re-declared `matrix_2d`, the third entry this round whose declared modality its own validator contradicts. **Modality declarations, fixed:** `rca` and `kmeans_distance` declared `series_1d` while being `(n_rows, n_features)` detectors, so `DetectorRegistry.extract_all_features`'s modality filter — whose whole purpose is to keep a detector from being driven with data it cannot read — routed 1-D series to them; a series reshaped to a single row collapsed every node's or feature's training spread to its floor and 100% of a stationary held-out half flagged. Both are now `matrix_2d`. Also fixed on `kmeans_distance`: its squash was anchored at the *median* training nearest-distance, which puts score 0.632 — not 0.5 — at the median, so the 0.5 cut sat below the median training distance and flagged 93% of an in-distribution batch (now quantile-anchored like the rest of the squash tier, 0.0175 against a nominal 0.02, with the benchmarked ranking bit-identical); and `detect` compared against a hard-coded 0.5, ignoring the configured `threshold`. Its documented AUROC range is corrected from "~0.8-0.98" to the 0.76-0.98 its own `artifacts/dormant_module_revival.json` records (cardio 0.762; Pima, outside the quoted set, 0.634). **Stationary-regime false-alarm rates, now measured and pinned** (`tests/detectors/test_stationary_false_alarm_rate.py`: fit on 400 samples of a stationary AR(1), score the next 400 from the same process, so nothing there is anomalous and no drift can be blamed — the one place the squash tier's `1 - calibration_quantile` contract reads directly). Two groups, because a single number does not mean the same thing for both. **Per-sample detectors** (one verdict per held-out sample, so the number is a false-alarm rate) flooding at the default threshold: `lstm_autoencoder` 0.0670 (cured, below), `srcnn` 0.1020 (**cured**: its BCE objective weights the positive class by `(1 - inject_ratio) / inject_ratio` = 19, offsetting the trained logit by `+log(19)`, so the fixed 0.5 cut it used to carry was a 0.95 cut on the prior-calibrated probability and flagged 0.2725. `fit` now installs the 0.95 quantile of the unaugmented training series' own scores, which flags 0.1020 *and* raises the NAB standard score 12.2 -> 13.2 — better on both axes, since every candidate cut shares one ROC curve at AUROC 0.683. Subtracting the `log(19)` offset instead takes the rate to 0.0000 but costs detections, 12.2 -> 9.0, so that is not what is applied). Above the 0.02 anchor but bounded: `statistical` 0.0980, `energy_based` 0.0755, `deep_svdd` 0.0470, `extended_statistical` 0.0460. At or under contract: the other 15, down to `page_hinkley` 0.0025. **Set-level detectors** reduce the whole series to *one* verdict, so what is pinned is the share of the five seeds whose single verdict came back anomalous -- not a rate: `nano_safeguard` calls an entirely normal stationary series anomalous on every seed (**not cured, and the missing evidence is specific**: its composite rises with signal length, 0.3757 to 0.4215 at lengths 50 to 400, so a cut calibrated on windows of the training signal does not transfer — tried, and it leaves 2 of 5 seeds flagged at any quantile; closing it needs labelled micro-anomalies at a known rate, which no corpus in the tree provides. `fit` now warns and every verdict carries `threshold_calibrated: False`). **Cured**: `acceleration_dynamics` decided at the 0.5 default against its own normal regime of 0.79 to 0.83 and so flagged every normal series; its kinematic score *is* flat in length (0.7986, 0.8078, 0.8018 at lengths 100, 200, 400), so `fit` cuts at the 0.95 quantile of training-window scores — verdict share 1.0 -> 0.0 with injected level shifts and volatility bursts still flagged. Never anomalous: `frequent_pattern`, `spectral_vibration`, `spectral_domain_frequency` and `uiux_anomaly`. The gate asserts each group's verdict arity so an entry cannot drift between them unnoticed. Checked and **not** defects: `energy_based` and `deep_svdd` are high-variance rather than biased — over 20 seeds their medians are 0.0138 and 0.0188, at or under nominal, with a heavy tail (max 0.1725 and 0.1375). **Fixed — `lstm_autoencoder`'s conformal threshold** now cuts on pooled out-of-fold scores, moving its stationary false-alarm rate from 0.6265 to **0.0670** against a 0.05 nominal. Two defects stacked. The calibration block and the early-stopping block were the same rows, so the restored state had been selected to minimise error on exactly the rows setting the threshold (worth 0.6265 -> 0.4450 on its own). Dominant, though, was that a single contiguous tail cannot estimate the score distribution of an autocorrelated series: the 80-row tail's score standard deviation measured 3 to 11 times smaller than the scored block's (0.029 against 0.310 on one seed), putting the cut well below the 95th percentile. The fit series is now cut into `calibration_folds` (default 5) contiguous blocks, each scored by a model trained on the other four, and the cut is taken over the pool — every score out-of-sample, the pool spanning the series rather than one stretch of it. Reconstruction scores are bit-identical before and after, so discrimination is untouched and only the threshold moved. The residual 1.34x over nominal is the known cost of a conformal cut on serially dependent data, not a defect in how the cut is taken. **Measured, not fixed — evidence missing:** the Oracle's inter-band phase coherence is withdrawn, not replaced. Magnitude-squared coherence between disjoint band-limited copies of one signal is ill-posed; a 1:1 phase-locking-value prototype across time segments, measured on the recorded CWRU windows, separates outer-race faults from healthy at AUROC 0.892 but inner-race faults at 0.176 (the opposite direction) and ball faults at 0.521 (chance), so the "low coherence implies anomalous" direction the influence multiplier assumes does not hold. What is needed is an n:m cross-frequency phase-coupling measure (the bands' centre frequencies differ, so 1:1 locking is not well posed) with its anomalous direction established on labelled data. Separately, `SpectralDomainFrequency`'s own aggregate stays below its 0.5 `BaseDetector` default on every CWRU window even now that it ranks them perfectly: most band weight sits on bands a 12 kHz sample rate cannot resolve, so standalone it decides nothing and needs a caller-supplied threshold (`PhysicsIntegratedDetector` consumes its score and influence vector, not its verdict). Both are pinned in `tests/detectors/test_recorded_cwru_bearings.py`. **Still open:**  No shipped checkpoint for `earthquake_precursor_ca`, `tsunami_dart`, `schumann_sierra_nevada` (8 skipped tests), and no trained weights for the networks that now abstain. Several infrastructure detectors carry configuration that is never read (`CommunicationsITDetector.alert_thresholds`, `EnergyDamsDetector` displacement limit), left in place as released attributes. The TimeGPT live check needs a Nixtla API key. | `DETECTOR_MANIFEST`; `core/detector_roles.py`; the skip reasons in the test run. |
| 33 (closed) | Save/load suite compares with a tolerance | **CLOSED (2026-10-23).** The ~4e-9 drift was the graph-Laplacian eigenvalues in `spectral_vibration` (and `physics_integrated`, which holds it): scipy's ARPACK `eigsh` starts from a random vector, so the eigenvalues moved ~1e-14 between identical calls and the z-scored score ~1e-9. `eigsh` now gets a fixed start vector, and `tests/core/test_detector_save_load.py` compares bit-exactly across all detectors (72 pass, repeated). | `tests/core/test_detector_save_load.py::test_fitted_state_round_trips_exactly`, `tests/detectors/test_spectral_vibration_estimator.py::test_repeated_detection_is_bit_identical`. |

Items 1, 2, 3, 4, 5, 6, 7 also appear as status rows in the capability
table below — the rollup above is the single authoritative open-items
list. When an item closes, update both the row above and the capability
table in the same commit.

---

> **Capability status (2026-05-19 — replaces all prior status tables).**
>
> Each capability is rated against three orthogonal columns rather than a
> single ambiguous "Implemented" flag, because the previous wording
> conflated *interface exists* with *production-ready*. Definitions:
>
> - **Designed** — there is a written API or architectural design for
>   the capability (in this document, in `ARCHITECTURE.md`, or in the
>   source-tree docstrings).
> - **Stubbed** — code exists but at least one critical path raises
>   `NotImplementedError`, returns mock data, depends on an unwired
>   collaborator, or has no in-tree tests.
> - **Functional** — code exists, all critical paths are wired,
>   in-tree tests exercise them, and the behaviour matches the design.
>   "Functional" does **not** imply external audit, FIPS certification,
>   or production deployment.
>
> A capability can legitimately be Designed AND Stubbed AND Functional
> on different surfaces. The "Notes" column calls out the precise gap.
>
> | # | Capability | Designed | Stubbed | Functional | Notes |
> |---|------------|:--------:|:-------:|:----------:|-------|
> | 1 | Distributed Processing | ✓ | — | ✓ | Phase 2 audit cure (May 2026) ships a native pure-stdlib `TCPMessageTransport` (`omni_mercury_engine.distributed.tcp_transport`) — asyncio + length-prefixed binary frames + per-message Ed25519 signatures via Mercury's own AMA Cryptography surface. No third-party RPC framework. The previously-unimplemented network path is closed: the `NotImplementedError` raises that remain in `raft_consensus.py` are the abstract `MessageTransport` contract methods, and both concrete transports (`InMemoryTransport`, `TCPMessageTransport`) implement them; `RaftCluster(use_in_memory_transport=False)` constructs real network nodes. Integration test: `tests/distributed/test_tcp_transport.py::test_three_node_cluster_elects_and_re_elects` spins up 3 nodes on 3 TCP ports, elects a leader, kills it, and confirms re-election. |
> | 2 | Biometric Modalities | ✓ | — | ✓ | `iris_recognition.py` (~700 LOC), `fingerprint_recognition.py` (~1,100 LOC), `voice_recognition.py` (~870 LOC) all import-clean with no `NotImplementedError`. As of v1.7.0 `narrative/voice.py:_init_llm` no longer silently substitutes `MockLLMAdapter` — it requires `llm_provider=` to name an implemented provider (`huggingface`, `ollama`, `openai`, `anthropic`, `xai`, `gemini`, `cohere`, `deepseek`, `cursor`, or `template`). HuggingFace additionally requires an explicit `llm_model_name`; remote HuggingFace IDs also require `llm_revision=<40-char SHA>`. Missing/unavailable provider in `MERCURY_ENV=production` raises `MercuryProductionConfigError`; in development it logs a warning and the voice path falls through to deterministic template narration. Iris and fingerprint paths are functional pending dedicated test coverage. |
> | 3 | Real Quantum Computing | ✓ | — | partial | `quantum_computing` runs on a local simulator by default (Qiskit Aer when installed, a NumPy statevector otherwise). The VQE detector trains on the data (one-class compression objective; held-out AUROC 0.942 on the package's synthetic test, where the earlier fit scored 0.51-0.58), every feature is encoded, and the IBM hardware path is refused under `MERCURY_OFFLINE`. Real hardware requires user credentials and is not exercised in CI. |
> | 4 | Advanced Harmonics | ✓ | — | ✓ | `harmonics/analyzer.py`, `features.py`, `transform.py` are wired and exercised by the 21-probe ensemble and detector pipeline. |
> | 5 | AutoML | ✓ | — | ✓ | `automl/optimizer.py`, `schedulers.py`, `search_space.py` (~1,170 LOC main file). `tests/automl/test_scheduler_completion.py` exercises the scheduler. Hyperparameter search wired into training loop. |
> | 6 | Federated Learning | ✓ | — | ✓ | `federated_learning/client.py`, `server.py`, `privacy.py` implemented. **Bidirectional GOSNN coupling closed (2026-06-02):** `FederatedServer._execute_round` routes every round's weights through `GOSNNCoupling{Server,Client}` (`publish → ingest → aggregate → receive`, SHA3-256 + shape + round integrity, fail-closed), so the prior one-way (aggregator → GOSNN) gap is gone. Conformal prediction in `core/gosnn_integration.py::GOSNNIntegration.detect()` no longer returns `confidence_intervals=None` on failure — the silent path is closed by `ConformalMisconfigurationError`. Regression: `tests/federated/test_no_silent_failure.py::test_federated_server_round_drives_gosnn_digested_fedavg_path`. |
> | 7 | Explainability | ✓ | — | ✓ | `explainability/shap.py`, `counterfactuals.py`, `gdpr_compliance.py` (~2,400 LOC combined). No `NotImplementedError`; design surface present. **Serve-path wiring closed (2026-06-02):** the validated `cognitive/explainability.py::{IntegratedGradientsExplainer, FaithfulnessEvaluator}` are now integrated into `OmniMercuryEngine.detect_with_fusion(explain=True)`, which attaches an IG attribution of the served `score_fusion` probability + faithfulness scores. Faithfulness non-regression gate (`tests/benchmarks/test_explanation_fidelity.py::test_faithfulness_non_regression_comprehensiveness_and_recovery`: comprehensiveness > random + 0.01 and recovery@k > 2× chance) + serve-path lock (`tests/test_fusion_explainability.py`). |
>
> **Cross-cutting items not in the above seven, but tracked:**
>
> | Capability | Designed | Stubbed | Functional | Notes |
> |------------|:--------:|:-------:|:----------:|-------|
> | Safe training-data loader (no pickle) | ✓ | — | ✓ | `omni_mercury_engine.security.safe_load` (added in `[Unreleased]`); `tests/security/test_safe_load.py` (38 tests) covers .npz validation, HMAC signing, tamper detection. Pickle code path **deleted** from the engine. |
> | Pickle migration tool | ✓ | — | ✓ | `python -m omni_mercury_engine.tools.migrate_pkl`; `tests/security/test_migrate_pkl.py` (13 tests) covers hardened-subprocess relaunch, schema validation, refusal-by-default. |
> | VLM detectors | ✓ | experimental | ✓ | **Concrete offline detector added (2026-06-02).** `detectors/vlm/base_vlm.py`'s five contract methods are `@abstractmethod` (no `NotImplementedError` on the public path; direct instantiation raises `TypeError`), **and** `detectors/vlm/statistical_vlm.py::StatisticalVLMDetector` is a concrete, instantiable, fully-offline implementation (deterministic salience statistics; no `transformers`/download). The network-backed AnyAnomaly/LAVAD/BLIP adapters stay the production path (`transformers` + revision-pinned HF download — external dependency); the statistical detector is the offline/CI default + documented surrogate. Locked by `tests/test_vlm_detectors.py::{TestBaseVLMDetector, TestStatisticalVLMDetector}`. |
> | Visual base detector | ✓ | — | ✓ | **Transparent ABC (2026-06-02).** `detectors/visual/base_visual.py`'s three contract methods are `@abstractmethod` (no `NotImplementedError` on the public path; direct instantiation raises `TypeError`); the native SOTA detectors (PatchCore, PaDiM, STFPM, ReverseDistillation, CFlow) are the concrete implementations. Locked by `tests/test_visual_detectors.py::TestBaseVisualDetector`. |
> | Ethics enforcement | ✓ | — | ✓ | Hard-enforced at the decision boundary (Phase 2 cure, May 2026; σ_Immutable promotion completed before v1.7.0 cut, Wave B Vector 2+4 closure shipped post-cut). `CognitiveOrchestrator.analyze`, `OmniMercuryEngine.detect_with_fusion`/`detect_with_fusion_calibrated`, and `NeuroSymbolicHub.predict` all raise `EthicalConstraintViolationError` (check="benevolence") on benevolence-threshold violation via `BenevolenceScorer.enforce`; the `strict_ethics=False` flag is deprecated and ignored. The engine's boundary scorer is constructed eagerly at init so the first concurrent call cannot race the gate. σ_Immutable is trained (weights at `src/omni_mercury_engine/security/sigma_immutable_weights.pt`) and is now a **second hard gate** at every boundary surface: `EthicalConstraintViolationError(check="sigma_immutable")` is raised on sub-threshold scalar vectors and `check="gosnn_unavailable"` is raised when GOSNN itself cannot run.  Every public boundary routes the caller-supplied `domain` through `omni_mercury_engine.cognitive.ethical_bounding.sanitize_domain` (canonical helper added in `[Unreleased]`) so a hostile / typo'd hint cannot inject harm or positive keywords into the scorer or audit surface, and `NeuroSymbolicHub.predict` pre-flights the σ_Immutable gate on empty batches (closing the silent no-op bypass identified in the v1.7 audit).  Decision-boundary contract documented in `src/omni_mercury_engine/ethical/__init__.py`. Regression suite: `tests/ethical/test_hard_enforcement.py` (covers both the BenevolenceScorer first-gate and the σ_Immutable / gosnn_unavailable second-gate at all three boundary surfaces plus the Wave B Vector 2+4 closures; wired into the `Neuro-Symbolic Tests` CI job — a benevolence- or σ_Immutable-threshold regression cannot merge silently).  **σ_Immutable Wave C is CLOSED (2026-06-02):** narrative voice (`narrative/voice.py::{speak, process_detection, alert}`) and federation (`federation/aggregator.py::{submit, aggregate}`, `federated_learning/server.py::_execute_round`) now carry the same benevolence + σ_Immutable dual hard gate, built from the single shared `build_sigma_immutable_vector` helper, with eager gate construction and `sanitize_domain` on every caller hint.  Regression: `tests/ethical/test_hard_enforcement.py::{TestNarrativeVoiceBoundary, TestFederatedAggregatorBoundary, TestFederatedServerRoundBoundary}`. |
> | 21-probe Anomaly Math Arrest ensemble | ✓ | — | ✓ | Phase 2 audit complete (May 2026). All 21 probes are registered and fit-participate on representative corpora; `AnomalyMathArrest.detect` discriminates injected anomalies across `earthquake` / `tsunami` / `pandemic` / `marine` / `geomagnetic` / `default` domain affinity orderings. No live `IsolationForest` import or instantiation remains in `src/` — the only references are documentation strings explaining what the ensemble replaced. Regression suite: `tests/detectors/test_math_arrest_dominant_path.py` (11 tests). |
> | FEMA Disaster loader label polarity | ✓ | — | ✓ | v1.7.0. `FEMADisasterLoader._select_anomaly_polarity` enforces the minority-as-anomaly convention used everywhere else in Mercury; the loader exposes `labels_inverted` so benchmark reporters can surface the flip alongside their AUC numbers. Closes the README "1 of the 64 (FEMA Disaster) is a known-broken loader" footnote item. Regression suite: `tests/datasets/test_disaster.py::TestFEMAInvertedScoresCorrection`. |
> | Dataset reachability harness (unreachable-11) | ✓ | — | ✓ | v1.7.0. Two-lane harness covering all 11 historically-unreachable loaders (SMAP, MSL, CICIDS-2017, MIT-BIH, UCR, SWaT, WADI, USGS Geochemistry, NOAA StormEvents, NOAA ERDDAP, FEMA HazardMitigation): an always-on offline lane (`tests/datasets/test_unreachable_loaders_offline.py`) that asserts every loader fails loudly under simulated outage, plus a nightly network lane (`tests/datasets/test_unreachable_loaders_network.py`) wired into `.github/workflows/dataset-reachability.yml` (04:17 UTC, `MERCURY_ALLOW_SYNTHETIC=0`). Both files carry a drift-gate `test_harness_covers_*_loaders` assertion pinning the matrix to exactly 11 entries. |
> | Production-mode primitive (`MERCURY_ENV`) | ✓ | — | ✓ | v1.7.0. New `omni_mercury_engine._env` module provides the canonical `MERCURY_ENV` flag (`development` default, `production`) plus shared fail-closed helpers (`get_mercury_env`, `is_production`, `require_real_component`, `MercuryProductionConfigError`). AMA/PQC is stricter than this environment mode and is mandatory at package import regardless of `MERCURY_ENV`; `AMA_REQUIRE_REAL_PQC` is retained only for legacy workflow readability. Locked by `tests/test_env.py` and `tests/test_pqc_startup_gate.py`. |
> | PQC dependency pin (`ama-cryptography`) | ✓ | — | ✓ | v1.7.0. `pyproject.toml [project.optional-dependencies].pqc` pins `ama-cryptography` to the validated `v3.3.0` git tag rather than tracking the default branch, matching the CI `AMA_REF: v3.3.0` real-AMA gate so an upstream force-push or breaking change cannot silently bump Mercury's PQC surface mid-cycle. v3.3.0 adds native `native_hmac_sha256` / `native_hmac_sha256_2` / `native_hmac_sha384` / `native_hmac_sha512` Python bindings consumed by Mercury's `native_jwt` HS256/HS384/HS512 routing (see CHANGELOG `[Unreleased]` § "AMA-routed JWT HMAC signatures"). Bump the tag in lockstep with `tests/security/test_pqc_gate_real_ama.py` and `docs/MIGRATION-1.6-to-1.7.md` §3. |
> | AMA HMAC-SHA-384 binding (HS384 path) | ✓ | — | ✓ | Closed. AMA Cryptography v3.3.0 adds a native HMAC-SHA-384 binding (`native_hmac_sha384`); Mercury's `native_jwt` routes HS384 through AMA fail-closed (no stdlib downgrade), with `native_hmac_sha384` / `HAS_AMA_HMAC_SHA384` added to `omni_mercury_engine.security.ama_hmac` and `HS384` added to `_AMA_ROUTABLE_ALGS`. Locked by `tests/security/test_native_jwt_ama_routing.py::TestSigningBackendSurface::test_hs384_routes_ama` and the RFC 4231 §4.2 SHA-384 KAT. |
> | NIST CSF 2.0 integrator | ✓ | — | ✓ | v1.7 (PR #223). `omni_mercury_engine.compliance.nist_csf_integrator` ships all six core functions, 22 categories, 106+ subcategories with `ImplementationTier` scoring, live `NISTCSFReferenceFetcher` against `csrc.nist.gov` (XLSX, 7-day on-disk cache), `assess_function` / `create_profile` / `detect_supply_chain_anomalies` / `continuous_monitoring_detect` / `generate_compliance_report`. Locked by `tests/test_nist_csf_integrator.py` (29 unit + 2 `@pytest.mark.network` tests). See `docs/COMPLIANCE.md`. |
> | TLP 2.0 handler (FIRST.org / CISA) | ✓ | — | ✓ | v1.7 (PR #223). `omni_mercury_engine.compliance.tlp_handler` ships the full five-label, four-colour ladder (CLEAR / GREEN / AMBER / AMBER+STRICT / RED) end-to-end (classification, reasoning, sharing guidelines, ethical considerations, watermark, export metadata). **`AMBER+STRICT` is the Mercury delta** — upstream shipped only TLP 1.0 colours. Locked by `tests/test_tlp_handler.py` (37 tests). See `docs/COMPLIANCE.md`. |
> | OSHA / eCFR compliance detector | ✓ | — | ✓ | v1.7 (PR #228). `omni_mercury_engine.compliance.osha_anomaly` covers 12 hazard categories × 6 industry sectors with CFR citations. Mercury delta: NWS Rothfusz heat-index regression (replaces upstream `T + 0.5·RH` heuristic — at 95 °F / 70 % RH the heuristic over-reported by ~8 °F; at low RH it under-reported). Optional live `ECFRClient` validates citations against `ecfr.gov` (60 req/min, cached). Locked by `tests/test_osha_anomaly.py` (36 tests). |
> | Drone anomaly detector (RADD + Mercury ensemble) | ✓ | — | ✓ | v1.7 (PR #228). `omni_mercury_engine.detectors.drone.detector` ships RADD invariant rules plus Mercury's first-party `MercuryAnomalyDetector` ensemble (Resonance 40 % + Kinematic 30 % + InfoGeometry 30 %). **No sklearn runtime dependency**. Three upstream defects fixed (missing `DroneState` fields, hand-coded ensemble replaced, unvalidated "93.84 % recall" claim removed). Transport-agnostic — operator supplies PX4 ULog / MAVLink / vendor SDK ingest. Ensemble decision is a single-row kNN-density z-gate against a rolling window that excludes the judged state (the 40/30/30 blend could not reach its gate on one row); registered in `DETECTOR_MANIFEST` as `drone` (category `UAV`). Locked by `tests/test_drone_detector.py`. See `docs/drone/SETUP.md`. |
> | Endocrinology detector (CGM + FDA-aligned rules) | ✓ | — | ✓ | v1.7. `omni_mercury_engine.medical.endocrinology_detector` ships a CGM Bi-LSTM (~155 K params) with additive attention, FDA-aligned glycemic rules (preserved verbatim from upstream), GLP-1 therapy and inhaled-insulin monitors. Integration-ready via `CGMDataSource` ABC; reference `DexcomV3DataSource` adapter supplied; no vendor credentials in tree. See `docs/medical/SETUP.md`. |
> | Anesthesiology predictor (TIVA + PID + vitals) | ✓ | — | ✓ | v1.7. `omni_mercury_engine.medical.anesthesiology_predictor` ships a TIVA Bi-LSTM (~164 K params), `SmartInfusionController` (PID gains kp=0.5, ki=0.1, kd=0.2; target BIS=50; safe window 40-60), `HemodynamicMonitor` (MAP 65-110 mmHg, HR 50-100 bpm, SpO₂ ≥ 92 %, EtCO₂ 30-45 mmHg). Integration-ready via `VitalsDataSource` ABC; reference `FHIRObservationVitalsSource` adapter supplied. See `docs/medical/SETUP.md`. |
> | Performance profiling toolkit | ✓ | — | ✓ | v1.7 (PR #223). `omni_mercury_engine.utils.profiling` ships six entry points (`@profile_func`, `@profile_memory`, `@profile_time`, `@profile_time_async`, `@profile_complete`, `PerformanceBenchmark`) plus `benchmark_function`. All entry points are no-ops when `set_profiling_enabled(False)` (the default). Mercury delta: async support, opt-in global enable flag, hardened tracemalloc nesting. Locked by `tests/test_profiling.py` (32 tests). See `docs/PROFILING.md`. |
>
> The phase checklists later in this document were written **before**
> implementations were merged and use `[ ]` markers throughout. They are
> retained as design references; the table above is the authoritative
> status. When the two disagree, the table wins.

This document outlines the strategic engineering roadmap for Mercury Agent, detailing planned enhancements, architectural considerations, and implementation strategies.

---

## Executive Summary

Mercury Agent is evolving toward a distributed, privacy-preserving, and explainable AI platform. This roadmap establishes the technical foundation for seven major capability expansions:

1. **Distributed Processing** — Multi-node deployment for horizontal scalability — *Functional; native pure-stdlib `TCPMessageTransport` with per-message Ed25519 signatures (2026-06-02)*
2. **Additional Biometric Modalities** — Iris, fingerprint, and voice authentication — *Iris/fingerprint Functional; narrative-voice LLM now opt-in via explicit `llm_provider=` (no Mock fallback in any mode)*
3. **Real Quantum Computing** — Qiskit integration for production quantum workloads — *Simulator functional; real hardware untested in-tree*
4. **Advanced Harmonics** — Higher l_max spherical harmonic analysis for 3D data — *Functional*
5. **AutoML** — Automatic hyperparameter tuning and model selection — *Functional*
6. **Federated Learning** — Privacy-preserving distributed training — *Functional; bidirectional GOSNN coupling wired into `FederatedServer._execute_round` (2026-06-02); silent conformal failure closed by `ConformalMisconfigurationError`*
7. **Explainability** — SHAP values and comprehensive interpretability framework — *Functional*

---

## 1. Distributed Processing

> **Status: Functional (closed 2026-06-02).** The scheduled v1.7 fix
> shipped: `distributed/tcp_transport.py` provides a **native
> pure-stdlib TCP `MessageTransport`** (asyncio + length-prefixed
> binary frames + AMA Cryptography per-message Ed25519 signatures) and
> `RaftCluster(use_in_memory_transport=False)` constructs real network
> nodes. The only `NotImplementedError` raises remaining in
> `raft_consensus.py` are the abstract `MessageTransport` contract
> methods, implemented by both concrete transports. Integration test:
> `tests/distributed/test_tcp_transport.py::test_three_node_cluster_elects_and_re_elects`.
> No third-party RPC framework — the wire format is Mercury's own,
> owned end-to-end. The design below was written pre-implementation
> and is retained as a design reference; the capability table above is
> authoritative.

### Current State
- Single-node deployment with threading for parallelism
- Local caching via in-memory stores
- Vertical scaling limitations

### Strategic Vision
Multi-node deployment enabling horizontal scalability, fault tolerance, and geographic distribution.

### Technical Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Mercury Agent Cluster                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │  Node 1  │  │  Node 2  │  │  Node 3  │  │  Node N  │        │
│  │ (Leader) │  │(Follower)│  │(Follower)│  │(Follower)│        │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │
│       │             │             │             │                │
│       └─────────────┴──────┬──────┴─────────────┘                │
│                            │                                     │
│              ┌─────────────┴─────────────┐                       │
│              │    Consensus Layer        │                       │
│              │    (Raft Protocol)        │                       │
│              └─────────────┬─────────────┘                       │
│                            │                                     │
│              ┌─────────────┴─────────────┐                       │
│              │  Distributed State Store  │                       │
│              │  (etcd / Redis Cluster)   │                       │
│              └───────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation Phases

#### Phase 1: Cluster Foundation
- [ ] Implement node discovery and registration
- [ ] Add health check endpoints and heartbeat protocol
- [ ] Create cluster membership management
- [ ] Implement leader election using Raft consensus

#### Phase 2: Workload Distribution
- [ ] Implement task queue with distributed locking
- [ ] Add work-stealing scheduler for load balancing
- [ ] Create partitioning strategy for anomaly detection workloads
- [ ] Implement result aggregation and fusion across nodes

#### Phase 3: State Management
- [ ] Integrate distributed state store (etcd or Redis Cluster)
- [ ] Implement distributed caching layer
- [ ] Add cross-node session management
- [ ] Create checkpoint and recovery mechanisms

#### Phase 4: Observability
- [ ] Implement distributed tracing (OpenTelemetry)
- [ ] Add cluster-wide metrics aggregation
- [ ] Create unified logging with correlation IDs
- [ ] Build cluster dashboard and alerting

### Interface Design

```python
class DistributedMercuryCluster:
    """
    Multi-node Mercury Agent deployment.

    Example:
        cluster = DistributedMercuryCluster(
            nodes=["node1:8080", "node2:8080", "node3:8080"],
            replication_factor=2,
            consensus_protocol="raft",
        )

        # Distributed anomaly detection
        results = await cluster.detect_anomalies(
            data=large_dataset,
            partition_strategy="hash",
            aggregation="weighted_fusion",
        )
    """

    def __init__(
        self,
        nodes: list[str],
        replication_factor: int = 2,
        consensus_protocol: str = "raft",
        state_backend: str = "etcd",
    ) -> None: ...

    async def detect_anomalies(
        self,
        data: np.ndarray,
        partition_strategy: str = "hash",
        aggregation: str = "weighted_fusion",
    ) -> DistributedDetectionResult: ...

    async def scale_out(self, new_nodes: list[str]) -> None: ...

    async def scale_in(self, remove_nodes: list[str]) -> None: ...
```

---

## 2. Additional Biometric Modalities

> **Status: Functional (iris, fingerprint, narrative-voice template path).**
> Iris and fingerprint recognition modules (~700 + ~1,100 LOC) import
> cleanly with no `NotImplementedError`.  The narrative voice path
> (`biometric/voice_recognition.py`, ~870 LOC, plus
> `narrative/voice.py`) no longer falls back to `MockLLMAdapter` —
> as of v1.7.0, `MercuryVoice(enable_llm=True)` requires an explicit
> `llm_provider=` argument naming an implemented provider
> (`huggingface`, `ollama`, `openai`, `anthropic`, `xai`,
> `gemini`, `cohere`, `deepseek`, `cursor`, or `template`).
> HuggingFace additionally requires an explicit model name, and remote
> HuggingFace IDs require `llm_revision=<40-char SHA>`. Without a provider,
> `MERCURY_ENV=production` raises
> `MercuryProductionConfigError` and development logs a warning and
> downgrades to deterministic template narration.  The design below
> was written pre-implementation; actual API may differ.

### Current State
- Facial recognition behavioral analysis
- Multi-spectral imaging support

### Strategic Vision
Comprehensive biometric framework supporting iris, fingerprint, and voice authentication with liveness detection.

### Technical Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 Unified Biometric Framework                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐     │
│  │   Iris Module  │  │Fingerprint Mod │  │  Voice Module  │     │
│  ├────────────────┤  ├────────────────┤  ├────────────────┤     │
│  │ • IrisCode ext │  │ • Minutiae ext │  │ • MFCC extract│     │
│  │ • Daugman norm │  │ • Ridge flow   │  │ • Speaker emb │     │
│  │ • HD matching  │  │ • Singularity  │  │ • Text-indep  │     │
│  │ • Liveness det │  │ • Liveness det │  │ • Liveness det│     │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘     │
│          │                   │                   │               │
│          └───────────────────┼───────────────────┘               │
│                              │                                   │
│                 ┌────────────┴────────────┐                      │
│                 │    Fusion Engine        │                      │
│                 │  • Score-level fusion   │                      │
│                 │  • Decision-level fusion│                      │
│                 │  • Quality-weighted     │                      │
│                 └────────────┬────────────┘                      │
│                              │                                   │
│                 ┌────────────┴────────────┐                      │
│                 │   Anomaly Integration   │                      │
│                 │  • Behavioral baseline  │                      │
│                 │  • Presentation attack  │                      │
│                 │  • Identity assurance   │                      │
│                 └─────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation Phases

#### Phase 1: Iris Recognition
- [ ] Implement IrisCode extraction (Daugman algorithm)
- [ ] Add Gabor filter bank for texture analysis
- [ ] Implement Hamming distance matching
- [ ] Create iris liveness detection (pupil dynamics, specular reflection)

#### Phase 2: Fingerprint Recognition
- [ ] Implement minutiae extraction (ridge endings, bifurcations)
- [ ] Add ridge flow estimation and singularity detection
- [ ] Implement fingerprint matching with tolerance for distortion
- [ ] Create fingerprint liveness detection (sweat pore analysis)

#### Phase 3: Voice Recognition
- [ ] Implement MFCC and speaker embedding extraction
- [ ] Add text-independent speaker verification
- [ ] Implement voice activity detection and segmentation
- [ ] Create voice liveness detection (replay attack prevention)

#### Phase 4: Multi-Modal Fusion
- [ ] Implement score-level fusion algorithms
- [ ] Add quality-weighted decision fusion
- [ ] Create presentation attack detection fusion
- [ ] Implement adaptive modality selection

### Interface Design

```python
class BiometricAnomalyDetector:
    """
    Multi-modal biometric anomaly detection.

    Example:
        detector = BiometricAnomalyDetector(
            modalities=["iris", "fingerprint", "voice"],
            fusion_strategy="quality_weighted",
            liveness_required=True,
        )

        result = detector.verify(
            iris_image=iris_scan,
            fingerprint_image=fingerprint_scan,
            voice_sample=audio_recording,
            claimed_identity="user_123",
        )
    """

    SUPPORTED_MODALITIES = ["iris", "fingerprint", "voice", "face"]

    def __init__(
        self,
        modalities: list[str],
        fusion_strategy: str = "quality_weighted",
        liveness_required: bool = True,
        anomaly_threshold: float = 0.5,
    ) -> None: ...

    def enroll(
        self,
        identity: str,
        samples: dict[str, np.ndarray],
    ) -> EnrollmentResult: ...

    def verify(
        self,
        claimed_identity: str,
        **modality_samples: np.ndarray,
    ) -> VerificationResult: ...

    def detect_anomaly(
        self,
        **modality_samples: np.ndarray,
    ) -> BiometricAnomalyResult: ...
```

---

## 3. Real Quantum Computing Integration

> **Status: Functional (simulator); untested in-tree (real hardware).**
> `quantum_computing/executor.py` defaults to `BackendType.SIMULATOR`
> (Qiskit Aer when installed, a NumPy statevector otherwise). The VQE
> detector minimises the mean `1 - P(|0...0>)` of the encoded training
> samples (the quantum-autoencoder compression objective) and encodes
> every feature in layers; the kernel path projects onto the training
> data's principal axes. The IBM hardware path requires credentials, is
> refused under `MERCURY_OFFLINE`, and is not exercised in CI. The design
> below was written pre-implementation; actual API may differ.

### Current State
- Simulated quantum operations via NumPy (exact statevectors) or Qiskit Aer
- Variational one-class detector, quantum kernel and QAOA paths
- PQC (Post-Quantum Cryptography) via AMA Cryptography (ML-KEM, ML-DSA, SLH-DSA)

### Strategic Vision
Production Qiskit integration enabling execution on real quantum hardware for applicable workloads.

### Technical Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 Quantum Execution Framework                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  Quantum Circuit Builder                 │    │
│  │  • Anomaly encoding circuits                            │    │
│  │  • Variational quantum eigensolvers                     │    │
│  │  • Quantum feature maps                                 │    │
│  │  • Error mitigation circuits                            │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           │                                      │
│           ┌───────────────┴───────────────┐                      │
│           │      Execution Router         │                      │
│           └───────────────┬───────────────┘                      │
│                           │                                      │
│     ┌─────────────────────┼─────────────────────┐               │
│     │                     │                     │               │
│  ┌──┴──────────┐  ┌───────┴───────┐  ┌─────────┴─────┐         │
│  │  Simulator  │  │  IBM Quantum  │  │   IonQ/AWS   │         │
│  │  (Aer/Qsim) │  │  (via Qiskit) │  │   Braket     │         │
│  └─────────────┘  └───────────────┘  └───────────────┘         │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  Hybrid Controller                       │    │
│  │  • Workload classification (quantum-advantaged check)   │    │
│  │  • Resource allocation and queuing                      │    │
│  │  • Result post-processing and classical fallback        │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation Phases

#### Phase 1: Qiskit Integration
- [ ] Integrate Qiskit as optional dependency
- [ ] Implement quantum circuit builder for anomaly detection
- [ ] Add backend abstraction for simulator vs real hardware
- [ ] Create job management and result retrieval

#### Phase 2: Quantum Algorithms
- [ ] Implement QAOA for combinatorial anomaly problems
- [ ] Add VQE for quantum feature extraction
- [ ] Implement quantum kernel methods for SVM
- [ ] Create quantum random number generation for cryptography

#### Phase 3: Hybrid Execution
- [ ] Implement quantum-classical hybrid optimizer
- [ ] Add automatic circuit optimization and transpilation
- [ ] Create error mitigation techniques (ZNE, PEC)
- [ ] Implement resource estimation and cost optimization

#### Phase 4: Production Readiness
- [ ] Add quantum hardware provider abstraction (IBM, IonQ, AWS)
- [ ] Implement job queuing and priority management
- [ ] Create monitoring and logging for quantum workloads
- [ ] Add graceful degradation to classical when quantum unavailable

### Interface Design

```python
class QuantumAnomalyDetector:
    """
    Quantum-enhanced anomaly detection with Qiskit backend.

    Example:
        detector = QuantumAnomalyDetector(
            backend="ibmq_qasm_simulator",  # or "ibmq_manila" for real hardware
            shots=1024,
            error_mitigation="zne",
        )

        # Quantum-classical hybrid detection
        result = detector.detect(
            data=features,
            method="vqe_anomaly",
            classical_fallback=True,
        )
    """

    def __init__(
        self,
        backend: str = "aer_simulator",
        shots: int = 1024,
        error_mitigation: str | None = "zne",
        optimization_level: int = 3,
    ) -> None: ...

    def detect(
        self,
        data: np.ndarray,
        method: str = "quantum_kernel",
        classical_fallback: bool = True,
    ) -> QuantumDetectionResult: ...

    def estimate_resources(
        self,
        data_shape: tuple[int, ...],
        method: str,
    ) -> QuantumResourceEstimate: ...
```

---

## 4. Advanced Harmonics

> **Status: Functional.** `harmonics/analyzer.py`, `features.py`, and
> `transform.py` are wired into the 21-probe ensemble and exercised by
> the detector pipeline. The design below was written
> pre-implementation; actual API may differ.

### Current State
- Basic spherical harmonic decomposition
- Limited l_max for computational efficiency

### Strategic Vision
Higher-order spherical harmonic analysis (l_max > 20) for detailed 3D surface analysis in anomaly detection.

### Technical Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              Advanced Spherical Harmonics Engine                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Input Processing                            │    │
│  │  • 3D point cloud normalization                         │    │
│  │  • Spherical projection (HEALPIX/Driscoll-Healy)        │    │
│  │  • Adaptive resolution selection                         │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           │                                      │
│  ┌────────────────────────┴────────────────────────────────┐    │
│  │          GPU-Accelerated SH Transform                    │    │
│  │  • CUDA kernels for associated Legendre polynomials     │    │
│  │  • Parallel FFT for azimuthal component                 │    │
│  │  • Memory-efficient streaming for high l_max            │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           │                                      │
│  ┌────────────────────────┴────────────────────────────────┐    │
│  │           Harmonic Feature Extraction                    │    │
│  │  • Power spectrum analysis                               │    │
│  │  • Rotation-invariant descriptors                        │    │
│  │  • Multi-scale harmonic pyramids                         │    │
│  │  • Anomaly-specific harmonic patterns                    │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           │                                      │
│  ┌────────────────────────┴────────────────────────────────┐    │
│  │            Anomaly Detection Integration                 │    │
│  │  • Harmonic coefficient outlier detection               │    │
│  │  • Spectral signature matching                           │    │
│  │  • 3D shape anomaly identification                       │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation Phases

#### Phase 1: High-Order SH Foundation
- [ ] Implement stable associated Legendre polynomial computation
- [ ] Add recurrence relations for numerical stability at high l
- [ ] Create adaptive precision management (float64/float128)
- [ ] Implement fast spherical harmonic transform

#### Phase 2: GPU Acceleration
- [ ] Create CUDA kernels for SH coefficient computation
- [ ] Implement parallel FFT integration
- [ ] Add memory-efficient streaming for l_max > 100
- [ ] Create PyTorch/JAX backend support

#### Phase 3: Feature Engineering
- [ ] Implement rotation-invariant harmonic descriptors
- [ ] Add multi-scale harmonic pyramid decomposition
- [ ] Create spectral graph wavelets on sphere
- [ ] Implement harmonic correlation for pattern matching

#### Phase 4: Anomaly Integration
- [ ] Add harmonic coefficient anomaly scoring
- [ ] Implement spectral signature database
- [ ] Create real-time 3D anomaly detection
- [ ] Add visualization tools for harmonic analysis

### Interface Design

```python
class HarmonicAnalyzer:
    """
    High-order spherical harmonic analysis for 3D anomaly detection.

    Example:
        analyzer = HarmonicAnalyzer(
            l_max=64,
            backend="cuda",
            precision="float64",
        )

        # Decompose 3D surface
        coefficients = analyzer.decompose(point_cloud)

        # Extract rotation-invariant features
        features = analyzer.extract_features(
            coefficients,
            descriptors=["power_spectrum", "bispectrum"],
        )

        # Detect anomalies
        anomalies = analyzer.detect_anomalies(
            features,
            reference_database=normal_signatures,
        )
    """

    def __init__(
        self,
        l_max: int = 32,
        backend: str = "numpy",  # "cuda", "torch", "jax"
        precision: str = "float64",
    ) -> None: ...

    def decompose(
        self,
        point_cloud: np.ndarray,
        sampling: str = "healpix",
    ) -> HarmonicCoefficients: ...

    def extract_features(
        self,
        coefficients: HarmonicCoefficients,
        descriptors: list[str],
    ) -> np.ndarray: ...

    def detect_anomalies(
        self,
        features: np.ndarray,
        reference_database: HarmonicDatabase,
        threshold: float = 0.5,
    ) -> HarmonicAnomalyResult: ...
```

---

## 5. AutoML

> **Status: Functional.** `automl/optimizer.py` (~1,170 LOC) plus
> `schedulers.py` and `search_space.py`. The hyperparameter search is
> wired into the training loop, and `tests/automl/test_scheduler_completion.py`
> exercises the scheduler. The design below was written
> pre-implementation; actual API may differ.

### Current State
- Manual hyperparameter configuration
- Grid search available but not automated

### Strategic Vision
Automatic hyperparameter tuning and model selection with intelligent search strategies.

### Technical Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AutoML Framework                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │               Search Space Definition                    │    │
│  │  • Hyperparameter ranges and distributions              │    │
│  │  • Model architecture search space                       │    │
│  │  • Feature preprocessing options                         │    │
│  │  • Ensemble configuration space                          │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           │                                      │
│  ┌────────────────────────┴────────────────────────────────┐    │
│  │              Optimization Strategy                       │    │
│  │  • Bayesian Optimization (TPE, GP)                      │    │
│  │  • Hyperband / ASHA for early stopping                  │    │
│  │  • Population-based training                             │    │
│  │  • Neural Architecture Search                            │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           │                                      │
│  ┌────────────────────────┴────────────────────────────────┐    │
│  │              Trial Management                            │    │
│  │  • Parallel trial execution                              │    │
│  │  • Resource allocation and scheduling                    │    │
│  │  • Checkpointing and resumption                          │    │
│  │  • Result logging and visualization                      │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           │                                      │
│  ┌────────────────────────┴────────────────────────────────┐    │
│  │              Model Selection                             │    │
│  │  • Multi-objective optimization                          │    │
│  │  • Pareto frontier analysis                              │    │
│  │  • Performance vs. complexity tradeoffs                  │    │
│  │  • Automatic ensemble construction                       │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation Phases

#### Phase 1: Core AutoML Engine
- [ ] Implement search space definition language
- [ ] Add Bayesian optimization with TPE
- [ ] Create trial scheduler with resource management
- [ ] Implement result persistence and analysis

#### Phase 2: Advanced Search Strategies
- [ ] Implement Hyperband for efficient early stopping
- [ ] Add ASHA (Asynchronous Successive Halving)
- [ ] Create population-based training
- [ ] Implement neural architecture search for deep models

#### Phase 3: Multi-Objective Optimization
- [ ] Add Pareto optimization for multiple objectives
- [ ] Implement performance vs. latency tradeoffs
- [ ] Create accuracy vs. interpretability balancing
- [ ] Add resource-constrained optimization

#### Phase 4: Integration
- [ ] Integrate with existing detector registry
- [ ] Add automatic ensemble construction
- [ ] Create model deployment pipeline
- [ ] Implement continuous optimization (online tuning)

### Interface Design

```python
class MercuryAutoML:
    """
    Automatic hyperparameter tuning for Mercury Agent detectors.

    Example:
        automl = MercuryAutoML(
            detector_class=FusionAnomalyDetector,
            search_space={
                "learning_rate": ("log_uniform", 1e-5, 1e-1),
                "num_layers": ("int_uniform", 2, 8),
                "hidden_dim": ("categorical", [64, 128, 256]),
            },
            objective="f1_score",
            n_trials=100,
            early_stopping="hyperband",
        )

        # Run optimization
        best_config, results = automl.optimize(
            train_data=X_train,
            train_labels=y_train,
            val_data=X_val,
            val_labels=y_val,
            parallel_trials=4,
        )

        # Get optimized detector
        detector = automl.get_best_detector()
    """

    def __init__(
        self,
        detector_class: type,
        search_space: dict[str, tuple],
        objective: str | list[str] = "f1_score",
        n_trials: int = 100,
        early_stopping: str | None = "hyperband",
    ) -> None: ...

    def optimize(
        self,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        val_data: np.ndarray,
        val_labels: np.ndarray,
        parallel_trials: int = 1,
    ) -> tuple[dict, OptimizationResults]: ...

    def get_best_detector(self) -> BaseDetector: ...

    def get_pareto_frontier(self) -> list[dict]: ...
```

---

## 6. Federated Learning

> **Status: Functional (closed 2026-06-02).**
> `federated_learning/client.py`, `server.py`, and `privacy.py` are
> implemented, and both former gaps are closed: GOSNN coupling is now
> bidirectional — `FederatedServer._execute_round` routes every round's
> weights through `GOSNNCoupling{Server,Client}` with SHA3-256 + shape
> + round integrity, failing the round closed on mismatch — and
> `core/gosnn_integration.py::GOSNNIntegration.detect()` no longer
> swallows conformal failures (`ConformalMisconfigurationError`; see
> CHANGELOG). Regression:
> `tests/federated/test_no_silent_failure.py`. The design below was
> written pre-implementation and is retained as a design reference;
> the capability table above is authoritative.

### Current State
- Centralized training only
- No support for distributed data sources

### Strategic Vision
Privacy-preserving distributed training enabling learning from decentralized data without data movement.

### Technical Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│               Federated Learning Framework                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │    Client A     │  │    Client B     │  │    Client N     │  │
│  │ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │  │
│  │ │ Local Data  │ │  │ │ Local Data  │ │  │ │ Local Data  │ │  │
│  │ └──────┬──────┘ │  │ └──────┬──────┘ │  │ └──────┬──────┘ │  │
│  │        │        │  │        │        │  │        │        │  │
│  │ ┌──────┴──────┐ │  │ ┌──────┴──────┐ │  │ ┌──────┴──────┐ │  │
│  │ │Local Train  │ │  │ │Local Train  │ │  │ │Local Train  │ │  │
│  │ └──────┬──────┘ │  │ └──────┬──────┘ │  │ └──────┬──────┘ │  │
│  │        │        │  │        │        │  │        │        │  │
│  │ ┌──────┴──────┐ │  │ ┌──────┴──────┐ │  │ ┌──────┴──────┐ │  │
│  │ │Diff Privacy │ │  │ │Diff Privacy │ │  │ │Diff Privacy │ │  │
│  │ └──────┬──────┘ │  │ └──────┬──────┘ │  │ └──────┬──────┘ │  │
│  └────────┼────────┘  └────────┼────────┘  └────────┼────────┘  │
│           │                    │                    │           │
│           └────────────────────┼────────────────────┘           │
│                                │                                │
│                 ┌──────────────┴──────────────┐                 │
│                 │    Secure Aggregation       │                 │
│                 │  • Gradient encryption      │                 │
│                 │  • Secure sum protocol      │                 │
│                 │  • Byzantine fault tolerance│                 │
│                 └──────────────┬──────────────┘                 │
│                                │                                │
│                 ┌──────────────┴──────────────┐                 │
│                 │     Global Model Server     │                 │
│                 │  • FedAvg / FedProx         │                 │
│                 │  • Client selection         │                 │
│                 │  • Convergence monitoring   │                 │
│                 └─────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation Phases

#### Phase 1: Federated Core
- [ ] Implement client-server communication protocol
- [ ] Add FedAvg aggregation algorithm
- [ ] Create client selection strategies
- [ ] Implement model versioning and synchronization

#### Phase 2: Privacy Guarantees
- [ ] Implement differential privacy (DP-SGD)
- [ ] Add secure aggregation protocol
- [ ] Create gradient clipping and noise injection
- [ ] Implement privacy budget tracking

#### Phase 3: Robustness
- [ ] Add Byzantine fault tolerance
- [ ] Implement contribution validation
- [ ] Create anomaly detection for malicious clients
- [ ] Add secure computation primitives

#### Phase 4: Production Features
- [ ] Implement asynchronous federated learning
- [ ] Add heterogeneous client support
- [ ] Create adaptive aggregation strategies
- [ ] Implement model compression for bandwidth efficiency

### Interface Design

```python
class FederatedMercury:
    """
    Federated learning for privacy-preserving anomaly detection.

    Example:
        # Server setup
        server = FederatedMercury.create_server(
            aggregation="fedavg",
            min_clients=10,
            rounds=100,
        )

        # Client setup (at each data source)
        client = FederatedMercury.create_client(
            server_address="server:8080",
            local_data=local_dataset,
            differential_privacy=True,
            epsilon=1.0,
        )

        # Training
        client.participate()  # Joins federated training

        # Get final model
        global_model = server.get_model()
    """

    @staticmethod
    def create_server(
        aggregation: str = "fedavg",
        min_clients: int = 2,
        rounds: int = 100,
        client_selection: str = "random",
    ) -> FederatedServer: ...

    @staticmethod
    def create_client(
        server_address: str,
        local_data: np.ndarray,
        differential_privacy: bool = True,
        epsilon: float = 1.0,
        delta: float = 1e-5,
    ) -> FederatedClient: ...
```

---

## 7. Explainability

> **Status: Functional.** `explainability/shap.py`,
> `counterfactuals.py`, and `gdpr_compliance.py` (~2,400 LOC combined)
> contain no `NotImplementedError` and surface the designed APIs.
> Broader test coverage is pending but the core paths run. The design
> below was written pre-implementation; actual API may differ.

### Current State
- Basic feature importance via permutation
- Limited model interpretation capabilities

### Strategic Vision
Comprehensive explainability framework with SHAP values, counterfactual explanations, and audit trails.

### Technical Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 Explainability Framework                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Global Explanations                         │    │
│  │  • SHAP feature importance                               │    │
│  │  • Partial dependence plots                              │    │
│  │  • Feature interaction effects                           │    │
│  │  • Model behavior summary                                │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           │                                      │
│  ┌────────────────────────┴────────────────────────────────┐    │
│  │              Local Explanations                          │    │
│  │  • SHAP values per prediction                           │    │
│  │  • LIME approximations                                   │    │
│  │  • Counterfactual examples                               │    │
│  │  • Attention visualization                               │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           │                                      │
│  ┌────────────────────────┴────────────────────────────────┐    │
│  │              Explanation Generation                      │    │
│  │  • Natural language explanations                         │    │
│  │  • Visual explanation dashboards                         │    │
│  │  • Technical audit reports                               │    │
│  │  • Compliance documentation                              │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           │                                      │
│  ┌────────────────────────┴────────────────────────────────┐    │
│  │              Audit Trail                                 │    │
│  │  • Decision logging with explanations                   │    │
│  │  • Reproducibility guarantees                            │    │
│  │  • Bias detection and monitoring                         │    │
│  │  • Regulatory compliance reports                         │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation Phases

#### Phase 1: SHAP Integration
- [ ] Integrate SHAP library for anomaly detectors
- [ ] Implement TreeSHAP for tree-based models
- [ ] Add KernelSHAP for model-agnostic explanations
- [ ] Create DeepSHAP for neural network detectors

#### Phase 2: Counterfactual Explanations
- [ ] Implement DiCE for diverse counterfactuals
- [ ] Add contrastive explanations
- [ ] Create actionable recourse recommendations
- [ ] Implement constraint-aware counterfactuals

#### Phase 3: Explanation Generation
- [ ] Implement natural language explanation templates
- [ ] Add visualization dashboard components
- [ ] Create technical audit report generator
- [ ] Implement compliance documentation automation

#### Phase 4: Audit Infrastructure
- [ ] Implement decision logging with full context
- [ ] Add explanation storage and retrieval
- [ ] Create bias monitoring dashboards
- [ ] Implement regulatory report generation (GDPR Art. 22)

### Interface Design

```python
class ExplainableAnomalyDetector:
    """
    Explainable anomaly detection with SHAP values and audit trails.

    Example:
        detector = ExplainableAnomalyDetector(
            base_detector=FusionAnomalyDetector(),
            explanation_method="shap",
            audit_enabled=True,
        )

        # Detection with explanation
        result = detector.detect(
            data=sample,
            explain=True,
            explanation_type="local",
        )

        # Access explanations
        print(result.explanation.shap_values)
        print(result.explanation.natural_language)
        print(result.explanation.counterfactuals)

        # Generate compliance report
        report = detector.generate_gdpr_report(
            decisions=recent_decisions,
            format="pdf",
        )
    """

    def __init__(
        self,
        base_detector: BaseDetector,
        explanation_method: str = "shap",
        audit_enabled: bool = True,
        explanation_cache_size: int = 10000,
    ) -> None: ...

    def detect(
        self,
        data: np.ndarray,
        explain: bool = True,
        explanation_type: str = "local",  # "local", "global", "both"
    ) -> ExplainableDetectionResult: ...

    def get_global_explanation(
        self,
        data: np.ndarray,
        n_samples: int = 1000,
    ) -> GlobalExplanation: ...

    def generate_counterfactuals(
        self,
        sample: np.ndarray,
        n_counterfactuals: int = 5,
        constraints: dict | None = None,
    ) -> list[Counterfactual]: ...

    def generate_gdpr_report(
        self,
        decisions: list[DetectionResult],
        format: str = "pdf",
    ) -> ComplianceReport: ...
```

---

## Dependencies and Prerequisites

### Required Dependencies by Feature

| Feature | Required Libraries | Optional Libraries |
|---------|-------------------|-------------------|
| Distributed Processing | grpcio, etcd3, redis | ray, dask |
| Biometric Modalities | opencv-python, librosa | tensorflow, onnxruntime |
| Quantum Computing | qiskit | qiskit-aer, qiskit-ibmq-provider |
| Advanced Harmonics | scipy, numpy | cupy, jax |
| AutoML | optuna | ray[tune], hyperopt |
| Federated Learning | grpcio, cryptography | pysyft, tensorflow-federated |
| Explainability | shap (lime excluded: unbuildable on modern setuptools; in-repo LIME surrogate instead) | dice-ml, alibi |

---

## Timeline and Milestones

| Phase | Timeline | Milestones |
|-------|----------|------------|
| Foundation | Q1 | Core infrastructure for distributed and federated capabilities |
| Alpha | Q2 | Initial implementations of all seven capabilities |
| Beta | Q3 | Integration testing and performance optimization |
| RC | Q4 | Documentation, compliance verification, and security audit |
| GA | Q1+1 | Production release with full support |

---

## Coverage uplift

CI enforces two job-scoped coverage floors (set in `.github/workflows/ci.yml`):

| Lane                       | Current floor | Measured baseline | Headroom |
|----------------------------|:------------:|:----------------------------------------------------:|:--------:|
| `COVERAGE_THRESHOLD_FULL` (ML/full lane) | **62** | 68.95 % (2026-07-21) | ~6.9 pts |
| `COVERAGE_THRESHOLD_CORE` (core lane)    | **33** | 40.57 % (2026-07-21) | ~7.6 pts |

Previous floors, kept for provenance: v1.7.x ran FULL **50** (baseline
59.84 %, 2026-05-17, run #1182 on `main`) and CORE **25** (baseline
31.87 %, expanded lane, 2026-05-21); both graduated to 55/30 in v2.1.x
on the 2026-07-18 re-measurement.

`.coveragerc` intentionally carries no `fail_under` — the gates are
job-scoped only — and `pyproject.toml [tool.coverage.report] fail_under
= 85` remains the strict aspirational nightly bar.

The strengthening plan §5 P1 target `CORE: 25 / FULL: 50` is complete.
`FULL` graduated to 50 in v1.7.0; `CORE` graduated from 15 to 25 in
v1.7.x once the core-lane runlist was widened in
`.github/workflows/ci.yml` to include `tests/detectors/`, `tests/ml/`,
`tests/datasets/`, `tests/api/`, `tests/automl/`, plus thirteen
root-level `test_*.py` additions (full list in the `core-tests` job).
`tests/security/` is intentionally **excluded** from the core lane
because its heavy KAT suite would duplicate coverage across the
Python matrix; the core lane still builds AMA Cryptography and sets
`AMA_REQUIRE_REAL_PQC=true`, so package imports fail closed if real AMA
is not provisioned. The full security tree continues to run under the
`ml-tests` and dedicated PQC production jobs. The measured combined stmt+branch coverage on
the expanded core lane (with `tests/security/` excluded) is well above
the new 25 floor, with the per-job `--cov-fail-under` flag on the
`core-tests` matrix locking that headroom. The `[api]` extras are now
installed in the core lane so the API-surface tests collect cleanly.

When raising the floor again, the sequencing is unchanged:

1. Land core-lane tests for the highest-marginal-coverage modules
   identified in `coverage report --skip-covered --sort=cover`.
2. Re-measure the core-lane baseline on `main`.
3. Bump `COVERAGE_THRESHOLD_CORE` to within ~1 pt of the new ceiling
   in the same commit.

Do **not** lower either floor back toward 10/15 to unblock unrelated
work — the floors document a non-regression guarantee, not a
preference.

---

## Contributing

Contributions to the roadmap are welcome. Please:

1. Open an issue describing the proposed enhancement
2. Reference this roadmap section
3. Include technical design considerations
4. Address backward compatibility concerns

---

## References

- Bonawitz et al. (2019): Towards Federated Learning at Scale
- Lundberg & Lee (2017): A Unified Approach to Interpreting Model Predictions
- Li et al. (2020): Federated Optimization in Heterogeneous Networks
- Preskill (2018): Quantum Computing in the NISQ Era and Beyond
- Driscoll & Healy (1994): Computing Fourier Transforms on the 2-Sphere
