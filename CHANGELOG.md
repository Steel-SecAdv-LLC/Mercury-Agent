# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Documentation Alignment Audit
- Repository documentation, metadata, and organizational state audited for
  accuracy against actual code
- README.md: Updated module count (455), line count (268,000+), test file
  count (244); removed phantom version reference (v1.4.1); removed reference
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
  honest_benchmark_results.json

### Version Reconciliation
- All version strings bumped from 1.4.0 → 1.5.1 to match CHANGELOG entries
  (pyproject.toml, __init__.py, cli.py, api/health.py, .secrets.baseline,
  README.md, SECURITY.md)
- CHANGELOG entries [1.5.0] and [1.5.1] verified against actual code —
  all described changes are present

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
- **SHA3-256 Cryptographic Alignment**: Aligned cryptographic posture with Ava-Guardian,
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
