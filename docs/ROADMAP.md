# Mercury Agent - Strategic Engineering Roadmap

Applies to Mercury Agent **v1.7.x**. Last updated: 2026-05-21.

## v1.7.x Deferred Items (consolidated)

The following items are deliberately deferred past the v1.7.0 release
cut. Each row names the surface, the precise gap, and the locking
artifact that pins behaviour today so a future PR cannot regress the
contract while the feature is built out.

| # | Surface | Gap | Locked by |
|---|---------|-----|-----------|
| 1 (closed) | σ_Immutable Wave C — narrative voice + federation | **CLOSED (2026-06-02).** `narrative/voice.py::{speak, process_detection, alert}`, `federation/aggregator.py::{submit, aggregate}`, and `federated_learning/server.py::_execute_round` now run the benevolence + σ_Immutable dual hard gate (gates constructed eagerly in each `__init__`; every domain hint routed through `sanitize_domain`; `alert` gained a `domain=` kwarg). The 256-D vector builder is promoted to the single shared `security.sigma_immutable_gate.build_sigma_immutable_vector` helper and the engine / orchestrator / hub copies all delegate to it. | `tests/ethical/test_hard_enforcement.py::{TestNarrativeVoiceBoundary, TestFederatedAggregatorBoundary, TestFederatedServerRoundBoundary}` — each pins legitimate-pass on the real gate plus `check="benevolence"` / `"sigma_immutable"` / `"gosnn_unavailable"`. |
| 2 | AMA HMAC-SHA-384 binding (HS384 path) | AMA Cryptography v3.2.0 ships HMAC-SHA-256 + HMAC-SHA-512 in C with Python bindings; HMAC-SHA-384 is not yet implemented. `native_jwt` routes HS384 through stdlib `hmac` until upstream lands the binding. | `tests/security/test_native_jwt_ama_routing.py::TestSigningBackendSurface::test_hs384_is_always_stdlib`. |
| 3 (closed) | VLM detector surface | **CLOSED (2026-06-02).** `detectors/vlm/base_vlm.py`'s five contract methods (`_initialize_model`, `_create_prompt`, `_parse_response`, `detect`, `extract_features`) are genuine `@abstractmethod` declarations — no `NotImplementedError` stub on the public path; `BaseVLMDetector()` raises `TypeError`. **The surface is no longer abstract-only:** `detectors/vlm/statistical_vlm.py::StatisticalVLMDetector` is a concrete, instantiable, **fully-offline** detector implementing all five methods via deterministic salience statistics (no `transformers`, no model download). The network-backed AnyAnomaly / LAVAD / BLIP backends remain the production path; the statistical detector is the offline / CI default + documented surrogate (remediation steps in its module docstring). | `tests/test_vlm_detectors.py::{TestBaseVLMDetector, TestStatisticalVLMDetector}` — base is abstract / not instantiable + the five methods are abstract; the concrete detector is instantiated offline and its contract (shape, salience ordering, determinism, VQA round-trip) is exercised by 11 behavioural tests. |
| 4 (closed) | Visual base detector | **CLOSED (2026-06-02).** `detectors/visual/base_visual.py`'s three contract methods (`fit`, `detect`, `extract_features`) are now genuine `@abstractmethod` declarations — no `NotImplementedError` stub on the public path; `BaseVisualDetector()` raises `TypeError`. The native SOTA detectors (PatchCore, PaDiM, STFPM, ReverseDistillation, CFlow) are the concrete implementations. | `tests/test_visual_detectors.py::TestBaseVisualDetector` — base is abstract / not instantiable, the three methods are abstract, and a concrete subclass exercises the concrete `preprocess` / `postprocess` helpers. |
| 5 (closed) | GOSNN coupling wired into FL aggregator training loop | **CLOSED (2026-06-02).** `FederatedServer._execute_round` now routes every round's weights through `GOSNNCoupling{Server,Client}` (`publish → ingest → aggregate → receive`) with SHA3-256 + shape + round integrity, failing the round closed on mismatch. Unit-LR FedAvg flows through the coupling's digested weighted mean; FedAdam / SCAFFOLD / secure-aggregation keep their numerics and are installed + broadcast through the new `GOSNNCouplingServer.install_global_state`, preserving the `LocalUpdate` / `RoundResult` fields and the privacy-engine / secure-aggregation branches. | `tests/federated/test_no_silent_failure.py::test_federated_server_round_drives_gosnn_digested_fedavg_path` drives a live `FederatedServer` round end-to-end and asserts weights flow through the digested FedAvg path; the idempotent / shape-mismatch / digest-corruption / `install_global_state` companions remain. |
| 6 | Intersectional fairness metrics | Bias audits currently measure marginal demographic parity only — no `(race, gender)`-style joint subgroup metrics. | No `tests/fairness/` directory exists today; a forward PR creates it. |
| 7 (closed) | Concrete `AttentionProvider` implementation | **CLOSED (2026-06-02).** `core/gosnn_optimizer.py::MultiHeadAttentionProvider` is a concrete provider wired to a real `torch.nn.MultiheadAttention` surface: `observe(sequence)` runs a forward pass and caches the per-head `(num_heads, seq_len, seq_len)` weights (`average_attn_weights=False`); `get_attention()` returns them, or raises `RuntimeError` before any forward (fail-closed "model not yet run"). Defaults to 32 heads to match `AttentionOptimizer`'s triadic φ-weighting so it drives `GOSNNOptimizer` directly. The deterministic-random placeholder is gone. | `tests/core/test_attention_provider.py` (10 tests) — per-head shape, softmax-normalised rows, fail-closed pre-forward, determinism, batched input, and that a wired+observed provider drives the optimizer metric (no skip) while an unobserved one fails closed to a skip. |
| 8 | Mutation testing on σ_Immutable hot path | No `mutmut` / `cosmic-ray` configuration in `pyproject.toml`; no workflow runs mutation tests. The σ_Immutable hot path lives in `src/omni_mercury_engine/security/{sigma_immutable_gate.py, sigma_immutable_corpus.py}`. | None — net-new gate. |
| 9 (closed) | Lyapunov-stability benchmark + λ reconciliation | **CLOSED (2026-06-02; premise was stale).** `scripts/run_ablation.py` **exists** (a real Lyapunov-pre-gated ablation runner with documented exit codes, referenced correctly by `configs/ablation_3r_lyapunov.yaml`). The `V̇ ≤ -λV` claim is certified by `tools/lyapunov_validator.py` against `configs/lyapunov_canonical.yaml`: `claimed_lambda=0.25, computed_lambda=0.5, ok=true` (2× margin). The README does **not** cite three conflicting λ values — `λ_convergence = 0.25` (Lyapunov convergence, canonical/certified) and `λ_decay = 0.18` (double-helix adaptation rate) are **distinct constants by design**, there is no `0.13`, and the `Docs λ Drift Gate` CI job (`scripts/check_readme_lyapunov.py`) enforces this — it passes ("all documented λ claims match canonical … scanned 2 files"). | `tools/lyapunov_validator.py` + `scripts/check_readme_lyapunov.py` (Docs λ Drift Gate in `iso-hardening.yml`). |
| 10 | `tests/load/` wired into CI | `tests/load/{k6_load_test.js, locustfile.py}` exists but no workflow invokes them. | None — net-new CI workflow. |
| 11 | Examples-parity CI | No workflow asserts `examples/*.py` runs end-to-end. | None — net-new CI workflow. |
| 12 | `tests/loaders/` + `tests/narrative/` graduate to strict mypy lane | Both directories are not yet in `ci.yml`'s strict-mypy invocation (`tests/datasets/`, `tests/ethical/`, `tests/safeguards/` are). Files need full annotations first. | `.github/workflows/ci.yml` job `type-checking` step "Run MyPy strict on graduated test directories". |
| 13 (closed) | Core coverage floor bump 15 → 25 | **CLOSED in v1.7.x.** Core lane expanded to include `tests/detectors/`, `tests/ml/`, `tests/datasets/`, `tests/api/`, `tests/automl/`, plus 13 root-level `test_*.py` additions. Measured combined stmt+branch coverage on the expanded lane is ≥25 % with a several-point cushion. | `.github/workflows/ci.yml` env `COVERAGE_THRESHOLD_CORE: 25` + the per-job `--cov-fail-under` flag in the `core-tests` job. |
| 14 (closed) | Serve-path explanations | **CLOSED (2026-06-02).** The validated `cognitive/explainability.py::{IntegratedGradientsExplainer, FaithfulnessEvaluator}` are wired into `OmniMercuryEngine.detect_with_fusion(explain=True)`, which attaches an IG attribution of the served `score_fusion` probability + faithfulness scores. Was deferred by PR #269 (explainer validated in-tree but not wired into the detect result). | `tests/benchmarks/test_explanation_fidelity.py::test_faithfulness_non_regression_comprehensiveness_and_recovery` (comprehensiveness > random + 0.01, recovery@k > 2× chance) + `tests/test_fusion_explainability.py`. |
| 15 (closed) | Fusion AUC/F1 + conformal-coverage CI regression gate | **CLOSED (2026-06-02).** `benchmarks/fusion_regression_guard.py` (`--check`/`--update`) deterministically trains+evaluates the fusion path on a seeded synthetic corpus and fails on AUC/F1 below `baseline − margin` or empirical conformal coverage below `target − 0.05`; committed baseline `benchmarks/fusion_capacity/fusion_gate_baseline.json` + artifact `artifacts/fusion/<ts>/metrics.json`; run by `.github/workflows/fusion-regression.yml`. Was deferred by PR #269 (pipeline + sweep artifacts existed, no CI floor gate). | `tests/benchmarks/test_fusion_regression_guard.py` (gate logic) + the workflow's `--check` step. |
| 16 | Fusion checkpoint round-trip fidelity + shipped-checkpoint quality | **OPEN (found 2026-06-02).** Two coupled issues surfaced while building the fusion gate: (a) `save_model`→`load_model` preserves AUC (Δ≈0.002) but drifts per-sample calibrated probabilities by up to ≈0.76 (so F1@0.5 / conformal sets shift on a loaded model); (b) the shipped `src/.../checkpoints/default_fusion.pt` underperforms in-distribution (AUC≈0.70) because base-detector state is **not persisted** in the checkpoint, so detectors auto-fit (and leak) on the first inference batch. **Mitigation in place:** the regression gate trains in-process rather than loading, so it measures *achievable* performance bit-stably and is unaffected by the drift. **Remediation plan:** (1) persist fitted base-detector state (or a deterministic refit seed + reference batch) inside the checkpoint so `load_model` reproduces training-time features; (2) round-trip the temperature calibrator + conformal thresholds and add a `save→load` probability-equivalence regression test (`max|Δ_prob| < 1e-3`); (3) regenerate `default_fusion.pt` once (a) lands. | Disclosed in `benchmarks/fusion_regression_guard.py` docstring + this row; gate insulated via in-process training. |
| 17 | 65/75 headline benchmark refresh | **OPEN (external blocker).** The headline figure requires the full network-dependent benchmark pipeline (many external corpora, hours); not run here, so the figure stays at its last real measurement rather than being fabricated. **Mitigation:** the offline-reproducible fusion floors are now CI-gated (row 15). **Remediation plan:** run `python benchmarks/mercury_benchmark.py` (or the `Benchmark Pipeline` workflow, weekly cron) with network access + `MERCURY_NETWORK_TESTS=1`, then commit the refreshed `benchmarks/*.json` + README "Latest Benchmark Results" block via the existing persistence step. | `.github/workflows/benchmark.yml` (the persistence steps publish the refreshed numbers on `main`). |

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
> | 1 | Distributed Processing | ✓ | — | ✓ | Phase 2 audit cure (May 2026) ships a native pure-stdlib `TCPMessageTransport` (`omni_mercury_engine.distributed.tcp_transport`) — asyncio + length-prefixed binary frames + per-message Ed25519 signatures via Mercury's own AMA Cryptography surface. No third-party RPC framework. The five `NotImplementedError` sites in `raft_consensus.py` are gone; `RaftCluster(use_in_memory_transport=False)` constructs real network nodes. Integration test: `tests/distributed/test_tcp_transport.py::test_three_node_cluster_elects_and_re_elects` spins up 3 nodes on 3 TCP ports, elects a leader, kills it, and confirms re-election. |
> | 2 | Biometric Modalities | ✓ | — | ✓ | `iris_recognition.py` (721 LOC), `fingerprint_recognition.py` (1131 LOC), `voice_recognition.py` (884 LOC) all import-clean with no `NotImplementedError`. As of v1.7.0 `narrative/voice.py:_init_llm` no longer silently substitutes `MockLLMAdapter` — it requires `llm_provider=` to name an implemented provider (`huggingface`, `ollama`, `openai`, `anthropic`, `xai`, `gemini`, `cohere`, `deepseek`, `cursor`, or `template`). HuggingFace additionally requires an explicit `llm_model_name`; remote HuggingFace IDs also require `llm_revision=<40-char SHA>`. Missing/unavailable provider in `MERCURY_ENV=production` raises `MercuryProductionConfigError`; in development it logs a warning and the voice path falls through to deterministic template narration. Iris and fingerprint paths are functional pending dedicated test coverage. |
> | 3 | Real Quantum Computing | ✓ | — | partial | `executor.py` defaults to `BackendType.SIMULATOR` and uses `AerSimulator`. Real-hardware path (IBM Quantum, IonQ) requires user credentials and is not exercised in CI. Treat as "simulated by default; real hardware untested in-tree." |
> | 4 | Advanced Harmonics | ✓ | — | ✓ | `harmonics/analyzer.py`, `features.py`, `transform.py` are wired and exercised by the 21-probe ensemble and detector pipeline. |
> | 5 | AutoML | ✓ | — | ✓ | `automl/optimizer.py`, `schedulers.py`, `search_space.py` (~1,135 LOC main file). `tests/automl/test_scheduler_completion.py` exercises the scheduler. Hyperparameter search wired into training loop. |
> | 6 | Federated Learning | ✓ | — | ✓ | `federated_learning/client.py`, `server.py`, `privacy.py` implemented. **Bidirectional GOSNN coupling closed (2026-06-02):** `FederatedServer._execute_round` routes every round's weights through `GOSNNCoupling{Server,Client}` (`publish → ingest → aggregate → receive`, SHA3-256 + shape + round integrity, fail-closed), so the prior one-way (aggregator → GOSNN) gap is gone. Conformal prediction in `core/gosnn_integration.py::GOSNNIntegration.detect()` no longer returns `confidence_intervals=None` on failure — the silent path is closed by `ConformalMisconfigurationError`. Regression: `tests/federated/test_no_silent_failure.py::test_federated_server_round_drives_gosnn_digested_fedavg_path`. |
> | 7 | Explainability | ✓ | — | ✓ | `explainability/shap.py`, `counterfactuals.py`, `gdpr_compliance.py` (~2,400 LOC combined). No `NotImplementedError`; design surface present. **Serve-path wiring closed (2026-06-02):** the validated `cognitive/explainability.py::{IntegratedGradientsExplainer, FaithfulnessEvaluator}` are now integrated into `OmniMercuryEngine.detect_with_fusion(explain=True)`, which attaches an IG attribution of the served `score_fusion` probability + faithfulness scores. Faithfulness non-regression gate (`tests/benchmarks/test_explanation_fidelity.py::test_faithfulness_non_regression_comprehensiveness_and_recovery`: comprehensiveness > random + 0.01 and recovery@k > 2× chance) + serve-path lock (`tests/test_fusion_explainability.py`). |
>
> **Cross-cutting items not in the above seven, but tracked:**
>
> | Capability | Designed | Stubbed | Functional | Notes |
> |------------|:--------:|:-------:|:----------:|-------|
> | Safe training-data loader (no pickle) | ✓ | — | ✓ | `omni_mercury_engine.security.safe_load` (added in `[Unreleased]`); 25 tests cover .npz validation, HMAC signing, tamper detection. Pickle code path **deleted** from the engine. |
> | Pickle migration tool | ✓ | — | ✓ | `python -m omni_mercury_engine.tools.migrate_pkl`; 9 tests cover hardened-subprocess relaunch, schema validation, refusal-by-default. |
> | VLM detectors | ✓ | experimental | ✓ | **Concrete offline detector added (2026-06-02).** `detectors/vlm/base_vlm.py`'s five contract methods are `@abstractmethod` (no `NotImplementedError` on the public path; direct instantiation raises `TypeError`), **and** `detectors/vlm/statistical_vlm.py::StatisticalVLMDetector` is a concrete, instantiable, fully-offline implementation (deterministic salience statistics; no `transformers`/download). The network-backed AnyAnomaly/LAVAD/BLIP adapters stay the production path (`transformers` + revision-pinned HF download — external dependency); the statistical detector is the offline/CI default + documented surrogate. Locked by `tests/test_vlm_detectors.py::{TestBaseVLMDetector, TestStatisticalVLMDetector}`. |
> | Visual base detector | ✓ | — | ✓ | **Honest ABC (2026-06-02).** `detectors/visual/base_visual.py`'s three contract methods are `@abstractmethod` (no `NotImplementedError` on the public path; direct instantiation raises `TypeError`); the native SOTA detectors (PatchCore, PaDiM, STFPM, ReverseDistillation, CFlow) are the concrete implementations. Locked by `tests/test_visual_detectors.py::TestBaseVisualDetector`. |
> | Ethics enforcement | ✓ | — | ✓ | Hard-enforced at the decision boundary (Phase 2 cure, May 2026; σ_Immutable promotion completed before v1.7.0 cut, Wave B Vector 2+4 closure shipped post-cut). `CognitiveOrchestrator.analyze`, `OmniMercuryEngine.detect_with_fusion`/`detect_with_fusion_calibrated`, and `NeuroSymbolicHub.predict` all raise `EthicalViolation` on benevolence-threshold violation via `BenevolenceScorer.enforce`; the `strict_ethics=False` flag is deprecated and ignored. The engine's boundary scorer is constructed eagerly at init so the first concurrent call cannot race the gate. σ_Immutable is trained (99.6% val_acc; weights at `src/omni_mercury_engine/security/sigma_immutable_weights.pt`) and is now a **second hard gate** at every boundary surface: `EthicalConstraintViolationError(check="sigma_immutable")` is raised on sub-threshold scalar vectors and `check="gosnn_unavailable"` is raised when GOSNN itself cannot run.  Every public boundary routes the caller-supplied `domain` through `omni_mercury_engine.cognitive.ethical_bounding.sanitize_domain` (canonical helper added in `[Unreleased]`) so a hostile / typo'd hint cannot inject harm or positive keywords into the scorer or audit surface, and `NeuroSymbolicHub.predict` pre-flights the σ_Immutable gate on empty batches (closing the silent no-op bypass identified in the v1.7 audit).  Decision-boundary contract documented in `src/omni_mercury_engine/ethical/__init__.py`. Regression suite: `tests/ethical/test_hard_enforcement.py` (covers both the BenevolenceScorer first-gate and the σ_Immutable / gosnn_unavailable second-gate at all three boundary surfaces plus the Wave B Vector 2+4 closures; wired into the `Neuro-Symbolic Tests` CI job — a benevolence- or σ_Immutable-threshold regression cannot merge silently).  **σ_Immutable Wave C is CLOSED (2026-06-02):** narrative voice (`narrative/voice.py::{speak, process_detection, alert}`) and federation (`federation/aggregator.py::{submit, aggregate}`, `federated_learning/server.py::_execute_round`) now carry the same benevolence + σ_Immutable dual hard gate, built from the single shared `build_sigma_immutable_vector` helper, with eager gate construction and `sanitize_domain` on every caller hint.  Regression: `tests/ethical/test_hard_enforcement.py::{TestNarrativeVoiceBoundary, TestFederatedAggregatorBoundary, TestFederatedServerRoundBoundary}`. |
> | 21-probe Anomaly Math Arrest ensemble | ✓ | — | ✓ | Phase 2 audit complete (May 2026). All 21 probes are registered and fit-participate on representative corpora; `AnomalyMathArrest.detect` discriminates injected anomalies across `earthquake` / `tsunami` / `pandemic` / `marine` / `geomagnetic` / `default` domain affinity orderings. No live `IsolationForest` import or instantiation remains in `src/` — the only references are documentation strings explaining what the ensemble replaced. Regression suite: `tests/detectors/test_math_arrest_dominant_path.py` (11 tests). |
> | FEMA Disaster loader label polarity | ✓ | — | ✓ | v1.7.0. `FEMADisasterLoader._select_anomaly_polarity` enforces the minority-as-anomaly convention used everywhere else in Mercury; the loader exposes `labels_inverted` so benchmark reporters can surface the flip alongside their AUC numbers. Closes the README "1 of the 64 (FEMA Disaster) is a known-broken loader" footnote item. Regression suite: `tests/datasets/test_disaster.py::TestFEMAInvertedScoresCorrection`. |
> | Dataset reachability harness (unreachable-11) | ✓ | — | ✓ | v1.7.0. Two-lane harness covering all 11 historically-unreachable loaders (SMAP, MSL, CICIDS-2017, MIT-BIH, UCR, SWaT, WADI, USGS Geochemistry, NOAA StormEvents, NOAA ERDDAP, FEMA HazardMitigation): an always-on offline lane (`tests/datasets/test_unreachable_loaders_offline.py`) that asserts every loader fails loudly under simulated outage, plus a nightly network lane (`tests/datasets/test_unreachable_loaders_network.py`) wired into `.github/workflows/dataset-reachability.yml` (04:17 UTC, `MERCURY_ALLOW_SYNTHETIC=0`). Both files carry a drift-gate `test_harness_covers_*_loaders` assertion pinning the matrix to exactly 11 entries. |
> | Production-mode primitive (`MERCURY_ENV`) | ✓ | — | ✓ | v1.7.0. New `omni_mercury_engine._env` module provides the canonical `MERCURY_ENV` flag (`development` default, `production`) plus shared fail-closed helpers (`get_mercury_env`, `is_production`, `require_real_component`, `MercuryProductionConfigError`). AMA/PQC is stricter than this environment mode and is mandatory at package import regardless of `MERCURY_ENV`; `AMA_REQUIRE_REAL_PQC` is retained only for legacy workflow readability. Locked by `tests/test_env.py` and `tests/test_pqc_startup_gate.py`. |
> | PQC dependency pin (`ama-cryptography`) | ✓ | — | ✓ | v1.7.0. `pyproject.toml [project.optional-dependencies].pqc` pins `ama-cryptography` to the validated `v3.2.0` git tag rather than tracking the default branch, matching the CI `AMA_REF: v3.2.0` real-AMA gate so an upstream force-push or breaking change cannot silently bump Mercury's PQC surface mid-cycle. v3.2.0 extends v3.1.0 with the `native_hmac_sha256` / `native_hmac_sha256_2` Python bindings consumed by Mercury's `native_jwt` HS256 path (see CHANGELOG `[Unreleased]` § "AMA-routed JWT HMAC signatures"). Bump the tag in lockstep with `tests/security/test_pqc_gate_real_ama.py` and `docs/MIGRATION-1.6-to-1.7.md` §3. |
> | AMA HMAC-SHA-384 binding (HS384 path) | ⏳ | tracked | — | Deferred. AMA Cryptography v3.2.0 ships HMAC-SHA-256 and HMAC-SHA-512 in C with Python bindings; HMAC-SHA-384 is not yet implemented in the native C backend. Mercury's `native_jwt` therefore routes HS384 through stdlib `hmac.new(..., hashlib.sha384)` unconditionally. The deferral is encoded in `tests/security/test_native_jwt_ama_routing.py::TestSigningBackendSurface::test_hs384_is_always_stdlib` so the route flips automatically once AMA lands the binding; when that happens, add `native_hmac_sha384` to `omni_mercury_engine.security.ama_hmac` and extend `_AMA_ROUTABLE_ALGS` to include `HS384`. Locked. |
> | NIST CSF 2.0 integrator | ✓ | — | ✓ | v1.7 (PR #223). `omni_mercury_engine.compliance.nist_csf_integrator` ships all six core functions, 22 categories, 106+ subcategories with `ImplementationTier` scoring, live `NISTCSFReferenceFetcher` against `csrc.nist.gov` (XLSX, 7-day on-disk cache), `assess_function` / `create_profile` / `detect_supply_chain_anomalies` / `continuous_monitoring_detect` / `generate_compliance_report`. Locked by `tests/test_nist_csf_integrator.py` (29 unit + 2 `@pytest.mark.network` tests). See `docs/COMPLIANCE.md`. |
> | TLP 2.0 handler (FIRST.org / CISA) | ✓ | — | ✓ | v1.7 (PR #223). `omni_mercury_engine.compliance.tlp_handler` ships the full five-colour ladder (CLEAR / GREEN / AMBER / AMBER+STRICT / RED) end-to-end (classification, reasoning, sharing guidelines, ethical considerations, watermark, export metadata). **`AMBER+STRICT` is the Mercury delta** — upstream shipped only TLP 1.0 colours. Locked by `tests/test_tlp_handler.py` (45 tests). See `docs/COMPLIANCE.md`. |
> | OSHA / eCFR compliance detector | ✓ | — | ✓ | v1.7 (PR #228). `omni_mercury_engine.compliance.osha_anomaly` covers 12 hazard categories × 6 industry sectors with CFR citations. Mercury delta: NWS Rothfusz heat-index regression (replaces upstream `T + 0.5·RH` heuristic — at 95 °F / 70 % RH the heuristic over-reported by ~8 °F; at low RH it under-reported). Optional live `ECFRClient` validates citations against `ecfr.gov` (60 req/min, cached). Locked by `tests/test_osha_anomaly.py` (28 tests). |
> | Drone anomaly detector (RADD + Mercury ensemble) | ✓ | — | ✓ | v1.7 (PR #228). `omni_mercury_engine.detectors.drone.detector` ships RADD invariant rules plus Mercury's first-party `MercuryAnomalyDetector` ensemble (Resonance 40 % + Kinematic 30 % + InfoGeometry 30 %). **No sklearn runtime dependency**. Three upstream defects fixed (missing `DroneState` fields, hand-coded ensemble replaced, unvalidated "93.84 % recall" claim removed). Transport-agnostic — operator supplies PX4 ULog / MAVLink / vendor SDK ingest. Locked by `tests/test_drone_detector.py` (16+ tests). See `docs/drone/SETUP.md`. |
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

1. **Distributed Processing** — Multi-node deployment for horizontal scalability — *Designed + Stubbed; in-memory only, network transport pending v1.7*
2. **Additional Biometric Modalities** — Iris, fingerprint, and voice authentication — *Iris/fingerprint Functional; narrative-voice LLM now opt-in via explicit `llm_provider=` (no Mock fallback in any mode)*
3. **Real Quantum Computing** — Qiskit integration for production quantum workloads — *Simulator Functional; real hardware untested in-tree*
4. **Advanced Harmonics** — Higher l_max spherical harmonic analysis for 3D data — *Functional*
5. **AutoML** — Automatic hyperparameter tuning and model selection — *Functional*
6. **Federated Learning** — Privacy-preserving distributed training — *Functional; bidirectional GOSNN coupling wired into `FederatedServer._execute_round` (2026-06-02); silent conformal failure closed by `ConformalMisconfigurationError`*
7. **Explainability** — SHAP values and comprehensive interpretability framework — *Functional*

---

## 1. Distributed Processing

> **Status: Designed + Stubbed (partial Functional).** Code exists in
> `distributed/cluster.py` (688 LOC) and `distributed/raft_consensus.py`
> (894 LOC) but five `NotImplementedError` calls remain in
> `raft_consensus.py` at lines 315, 323, 331, 335, and 830 — only
> `InMemoryTransport` is implemented; no network transport exists.
> Multi-node Raft cannot communicate today. Scheduled fix in v1.7 is a
> **native pure-stdlib TCP `MessageTransport`** (asyncio + length-prefixed
> binary frames + AMA Cryptography per-message signatures) plus
> integration tests. No third-party RPC framework — the wire format is
> Mercury's own, owned end-to-end. The design below was written
> pre-implementation; actual API may differ.

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
> Iris and fingerprint recognition modules (721 + 1131 LOC) import
> cleanly with no `NotImplementedError`.  The narrative voice path
> (`biometric/voice_recognition.py`, 884 LOC, plus
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

> **Status: Functional (simulator); Untested in-tree (real hardware).**
> `quantum_computing/executor.py` defaults to `BackendType.SIMULATOR`
> and runs circuits via Qiskit's `AerSimulator`. The real-hardware code
> path (IBM Quantum, IonQ, Rigetti) requires user credentials and is
> not exercised in CI. Treat as "simulated by default; real hardware
> untested in-tree." The design below was written pre-implementation;
> actual API may differ.

### Current State
- Simulated quantum operations via NumPy
- Quantum-inspired algorithms (annealing simulation)
- PQC (Post-Quantum Cryptography) implementation

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
class AdvancedHarmonicAnalyzer:
    """
    High-order spherical harmonic analysis for 3D anomaly detection.

    Example:
        analyzer = AdvancedHarmonicAnalyzer(
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

> **Status: Functional.** `automl/optimizer.py` (~1,135 LOC) plus
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

> **Status: Designed + Stubbed (partial Functional).**
> `federated_learning/client.py`, `server.py`, and `privacy.py` are
> implemented. Two gaps keep this row at "partial": GOSNN integration
> is one-way (aggregator → GOSNN scalar update has no reverse path),
> and `core/gosnn_integration.py::GOSNNIntegration.detect()` previously
> swallowed conformal failures into `confidence_intervals=None`. The
> silent-failure path is closed via `ConformalMisconfigurationError`
> (see CHANGELOG); the bidirectional-feedback gap is tracked in the
> v1.7.x Deferred Items rollup at the top of this document. The
> design below was written pre-implementation; actual API may differ.

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
| Explainability | shap, lime | dice-ml, alibi |

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

| Lane                       | v1.7.x floor | Measured baseline | Headroom |
|----------------------------|:------------:|:----------------------------------------------------:|:--------:|
| `COVERAGE_THRESHOLD_FULL` (ML/full lane) | **50** | 59.84 % (2026-05-17, run #1182 on `main`) | ~9.8 pts |
| `COVERAGE_THRESHOLD_CORE` (core lane)    | **25** | 31.87 % (expanded lane, 2026-05-21)       | ~6.9 pts |

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
