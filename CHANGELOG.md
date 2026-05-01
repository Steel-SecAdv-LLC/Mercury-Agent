# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- All version strings bumped from 1.5.1 → 1.6.0 across 27 files:
  pyproject.toml, __init__.py, cli.py, api/health.py, api/server.py,
  crypto/__init__.py, models/sota/__init__.py, .secrets.baseline,
  README.md, SECURITY.md, data_sources/base.py, data_sources/earth_science.py,
  cognitive/anomaly_detection_enhanced.py, infrastructure/observability.py,
  integrations/cross_platform_hub.py, helm/mercury-agent/Chart.yaml,
  k8s/base/deployment.yaml, k8s/base/kustomization.yaml,
  k8s/overlays/distributed/streaming-workers.yaml, docs/MATH_SPEC.md,
  examples/physics_detectors_demo.py, benchmarks/generate_benchmark_visuals.py,
  benchmarks/generate_v1_2_visuals.py, benchmarks/live_dataset_benchmark.py,
  tests/test_cli_smoke.py, tests/test_api.py
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
