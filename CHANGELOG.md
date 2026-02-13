# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4] - 2026-02-12

### Added
- Real-data validation suite (ADBench, NSL-KDD, live-dataset CI gates)
- Live-dataset detector benchmarks (AUC-ROC, F1, Accuracy measured)
- Automated threshold calibration via `enable_auto_calibration()`
- CICIDS-2017 network security dataset integration
- Expanded ADBench coverage from 4 to 16 datasets
- Live-data validation CI workflow (.github/workflows/live-data-validation.yml)
- Physics-inspired anomaly detectors (spectral vibration, acceleration dynamics,
  dimensional analysis, spatial, UI/UX behavioral)
- GOSNN scalar fusion with physics detector backends
- SHA3-256 cryptographic alignment with Ava-Guardian
- Comprehensive strict type checking (MyPy strict, 274+ errors resolved)

### Changed
- CI now requires live-data validation (no synthetic-only benchmarks)
- All version references normalized to v1.4
- Documentation updated with measured real-world metrics
- Codebase consolidation: removed all `cast()` workarounds for proper type annotations
- CI/CD pipeline compliance across 408 files (Flake8, MyPy, Bandit, pytest)

### Security
- Cryptographic hash upgrade from SHA-256 to SHA3-256 across audit trail components
- Removed type-unsafe `cast()` calls that masked potential runtime errors

### Fixed
- Removed 3.5MB generated benchmark bloat
- Email standardization across all config files

---

## [1.4-pre] - 2026-02-11

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
  - **Configurable AAFE Exponent**: Made ethical scaling exponent configurable (default Φ)
    to support empirical optimization via parameter sweep
  - **NaN Guards**: Added NaN propagation prevention to AAFE fusion equation
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

- **PHASE 3B — Domain-Adaptive AAFE Weights** (`core/three_r/fusion.py`):
  `DomainAdaptiveAAFEWeights` class that learns per-domain weight profiles from empirical
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
  - `ConstrainedParameterOptimizer`: High-level API for AAFE weights on simplex
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

- **AAFE Equation** (`core/three_r/fusion.py`): Ethical exponent now configurable
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

## [Unreleased]

### Added - 2026-01-06 Security Audit
- **PII Masking Filter**: Automatic redaction of sensitive data in logs (`api/server.py`)
  - Email addresses, phone numbers, SSNs, credit cards, API keys, bearer tokens, IP addresses
  - Applied to `omni_mercury_engine.api`, `omni_mercury_engine.security`, and `uvicorn.access` loggers
- **CORS Middleware Configuration**: Environment-aware CORS security (`api/server.py`)
  - Production mode requires explicit `MERCURY_CORS_ORIGINS` configuration
  - Development mode allows localhost origins (3000, 8000, 8080)
  - Configurable via `MERCURY_CORS_ORIGINS` and `MERCURY_CORS_CREDENTIALS` environment variables
- **Cryptographic Audit Trail** (`security/pqc_backends.py`): Ava-Guardian PQC fortification
  - `CryptoAuditTrail` class for tamper-evident operation logging
  - `CryptoOperation` dataclass for audit records
  - `get_crypto_audit_trail()` for global audit instance access
  - `validate_pqc_environment()` for production readiness validation
  - Thread-safe with configurable max entries (10,000 default)
- **GOSNN Detection Cache** (`core/gosnn_integration.py`): 2x speedup for repeated queries
  - `TTLCache` class with LRU eviction and configurable TTL (300s default)
  - SHA-256 array hashing for cache keys
  - Thread-safe operations with hit/miss statistics
  - `get_detection_cache()` for global cache access
- **GOSNN Performance Monitor** (`core/gosnn_integration.py`): Latency tracking
  - `GOSNNPerformanceMonitor` class with percentile calculations (p50, p95, p99)
  - `get_bottlenecks()` for identifying slow operations
  - `get_performance_monitor()` for global monitor access
- **Gradient Cache** (`ml/advanced_optimizers.py`): 2x speedup for synthetic gradients
  - `GradientCache` class with quantized key computation
  - Integration with `SyntheticGradientPredictor.forward(use_cache=True)`
  - `get_gradient_cache()` for global cache access
- **Performance Benchmark CI Stage** (`.github/workflows/ci.yml`): Regression detection
  - TTLCache write/read performance assertions (<500ms/100ms for 1000 ops)
  - Gradient prediction performance gate (<1000ms for 100 ops)
  - Cache hit rate tracking
  - Runs on PRs to main/develop and scheduled runs
- **Comprehensive Audit Tests** (`tests/security/test_audit_improvements.py`): 15+ new tests
  - PII masking filter tests (email, phone, API keys, bearer tokens, IPs)
  - PQC audit trail tests (logging, failure summary, rotation, thread safety)
  - GOSNN cache tests (hit, miss, expiry, LRU eviction)
  - Performance monitor tests (recording, bottlenecks, percentiles)
  - Gradient cache tests (caching integration with synthetic gradients)

### Changed - 2026-01-06 Security Audit
- **GOSNNIntegration.detect()**: Added `use_cache` parameter for caching control
- **SyntheticGradientPredictor.forward()**: Added `use_cache` parameter for gradient caching
- **Performance Targets**: <2% overhead target with cache hit rate tracking

### Security - 2026-01-06 Security Audit
- **PII Protection**: All sensitive data automatically masked in logs
- **CORS Hardening**: Production requires explicit origin configuration
- **Audit Compliance**: Full cryptographic operation audit trail
- **Environment Validation**: Runtime PQC environment checks with production readiness assessment

### Added
- **Runtime Pipeline Integration**: Integrated drift.py, fairness.py, optimization.py, and llm_adapter.py into the main detection pipeline
  - `enable_drift_detection()`: Monitor for data distribution shifts with ensemble drift detection
  - `enable_fairness_auditing()`: Bias auditing and fairness assessment for ethical AI compliance
  - `enable_llm_enhancement()`: Zero-shot LLM-based anomaly explanation enhancement
  - `OptimizationConfig` and `ParallelExecutor` for performance optimization
  - Drift detection results included in `detect_with_fusion()` output
  - LLM enhancement results included for detected anomalies

### Fixed
- **ML Tests CI Failure**: Fixed initialization order in foundation model adapters
  - TimeGPTAdapter: Set `timegpt_config` before `super().__init__()` to avoid AttributeError
  - ChronosAdapter: Set `chronos_config` before `super().__init__()` to avoid AttributeError
  - MatrixProfileDetector: Set `mp_config` before `super().__init__()` to avoid AttributeError
  - FoundationEnsemble: Set `ensemble_config` before `super().__init__()` to avoid AttributeError
  - BaseModel: Use `_config_dict` for internal access to avoid property override issues
  - Added config setters to all adapter classes for base class compatibility
- **Code Quality**: All quality checks now pass
  - Black formatting: 0 issues
  - isort import sorting: 0 issues
  - flake8 linting: 0 issues
  - mypy type checking: 0 issues (down from 1,000+ previously)

### Changed
- **Deep 3R Integration in Geological Detectors**: Full RecursionEngine, ResonanceEngine, and RefactoringEngine integration
  - TornadoDetector: 3R engines with max_depth=5 for recursion, sampling_rate=1.0 for resonance
  - HurricaneDetector: 3R engines for tropical cyclone pattern analysis
  - FloodDetector: CoreRefactoringEngine alias to avoid naming conflict with local RefactoringEngine
- **NOAA Data Loaders**: New data loaders for live environmental data
  - NOAASpaceWeatherLoader: Solar activity and geomagnetic storm data
  - NOAAHurricaneLoader: Tropical cyclone track and intensity data
  - NOAAOceanLoader: Ocean temperature and marine ecosystem data
- **Comprehensive 3R Integration Tests**: 47 tests covering all 3R engine integrations
  - TestRecursionEngineIntegration: 6 tests for hierarchical feature extraction
  - TestResonanceEngineIntegration: 6 tests for FFT spectrum computation
  - TestRefactoringEngineIntegration: 5 tests for code complexity analysis
  - TestTornadoDetector3RIntegration: 6 tests for tornado detector 3R
  - TestHurricaneDetector3RIntegration: 6 tests for hurricane detector 3R
  - TestFloodDetector3RIntegration: 6 tests for flood detector 3R
  - TestCrossDetector3RConsistency: 6 tests for cross-detector consistency
  - Test3RMechanismIntegration: 3 tests for full 3R pipeline
  - TestGeologicalDetectorFeatureExtraction: 3 tests for feature extraction
- **Enhanced AI Ethics Framework**: Improved ethical scoring with keyword-based matching
  - Expanded keyword sets for compassion, evidence, justice, altruism, control, character, competence, commitment
  - Survivor-first principles for humanitarian applications
  - Bias audit integration and peer review support
- **Omni-Prefix Scalar Unification**: All 50+ ethical scalars renamed to omni- prefix
  - Core scalars: omnicompassion, omnibenevolence, omniempathy, omnimorality, omnilove, omnijustice, omniequity, omniforgiveness, omnialtruism, omnihope, omnicourage, omniwisdom, omniintegrity, omniloyalty, omnidetermination, omniaccountability, omnitransparency, omniexplainability
  - New scalars: omnigrace, omnipatience, omnihumility, omniresilience, omniperseverance, omnivigilance, omnistewardship
  - Cosmic scalars: omniuniverse_adapt, omnitelos, omnicosmicharmony, omnistellarresonance
  - Quantum scalars: omniquantum_weight, omniquantum_entanglement, omnineuroquantum, omniconsciousness_coherence, omniquantum_superposition, omniquantum_decoherence_shield
  - Humanitarian scalars: omnicrisis_response, omnidisaster_response, omnipandemic_monitoring, omnimissing_persons_priority, omnimedical_discovery, omnihumanitarian_aid, omnirefugee_protection, omnifood_security, omniclimate_resilience
  - Security scalars: omnithreat_detection, omniquantum_resistance, omniencryption_strength, omniaudit_compliance, omnicyber_fortress, omnizero_trust
- Backward-compatible legacy alias system for seamless migration (deprecated in v2.0)
- `resolve_scalar_name()` and `get_scalar()` methods for legacy alias resolution
- Nano-safeguards marked as complete in PROTECTION_OVERVIEW.md
- Real-world dataset benchmarking framework with multi-domain support
  - Medical loaders: MIMIC-III/IV, PhysioNet, Sepsis, Cardiology (with PhysioNet credential support)
  - Security loaders: NSL-KDD, CICIDS, ThreatIntel for network intrusion detection
  - Environmental loaders: USGS Earthquake, NOAA Weather, Wildfire data
  - Space loaders: SETI signals, NASA Exoplanet, Solar Dynamics
- `RealWorldBenchmarkSuite` with comprehensive metrics (precision, recall, F1, AUC-ROC, AUC-PR)
- Statistical significance testing for baseline comparisons
- Synthetic data generation fallback when real datasets unavailable
- Enhanced Neuro-Symbolic AI engine (`EnhancedNeurosymbolicEngine`)
  - Fuzzy logic with multiple semantics (Godel, Product, Lukasiewicz)
  - Temporal reasoning via `TemporalGraphReasoner` with open/closed world assumptions
  - Knowledge graph integration via `KnowledgeGraphBridge` for commonsense reasoning
  - Meta-cognition layer with uncertainty quantification and strategy selection
  - Causal reasoning module with do-calculus interventions and counterfactuals
  - Probabilistic logic layer with Frechet bounds
  - Logic Tensor Networks (PyTorch-based, optional dependency)
- `.pre-commit-config.yaml` with comprehensive hooks for security, formatting, and linting
- `.env.example` with complete configuration template and documentation

### Security
- **CI/CD Security Hardening**: All security scans and linting tools now blocking
  - Bandit security scan: Fails on medium+ severity issues (removed `|| true`)
  - Semgrep security scan: Fails on security issues (removed `|| true`)
  - Safety dependency check: Fails on known vulnerabilities (removed `|| true`)
  - Ruff linting: Fails on issues (removed `--exit-zero`)
  - Flake8 linting: Fails on violations (removed `|| true`)
  - MyPy type checking: Fails on type errors (removed `|| true`)
  - pip-audit: Fails on dependency vulnerabilities (removed `|| true`)
- **Data Loader Security**: All loaders now enforce live data requirements
  - Default `use_synthetic=False` to prevent synthetic data in production
  - Minimum 100 real samples required (`min_real_samples=100`)
  - RuntimeError raised if API is down or insufficient real samples
- **BREAKING**: JWT authentication now requires `JWT_SECRET_KEY` environment variable
  - Removed insecure default `"dev-secret-key"` (P0 security fix)
  - Clear error message with instructions for secure key generation
  
  **Migration Guide for JWT_SECRET_KEY:**
  1. Generate a secure random key: `openssl rand -hex 32`
  2. Set the environment variable before starting the server:
     - Linux/macOS: `export JWT_SECRET_KEY="your-generated-key"`
     - Windows: `set JWT_SECRET_KEY=your-generated-key`
     - Docker: Add `-e JWT_SECRET_KEY="your-generated-key"` to docker run
     - .env file: Add `JWT_SECRET_KEY=your-generated-key`
  3. For production, use a secrets manager (AWS Secrets Manager, HashiCorp Vault, etc.)
  4. **Important**: Never commit your JWT secret to version control
  
- Rate limiting middleware enforced on FastAPI server
  - Token bucket algorithm: 100 requests/min, burst of 20
  - Configurable via `OMNI_RATE_LIMIT_*` environment variables
  - Standard rate limit headers (`X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`)
- CI/CD Trivy security scans now blocking for CRITICAL/HIGH vulnerabilities
- Pre-commit hooks for secret detection (detect-secrets) and security linting (bandit)

### Changed
- **Docker**: Fixed entrypoint from `omni-anomaly` to `mercury-agent` (matches setup.py console_scripts)
- **Dependencies**: Added cryptography, fastapi, uvicorn, httpx, pydantic-settings to core requirements
- CI/CD: ML Tests now run on PRs to main/develop (previously only scheduled/manual)
- CI/CD: Documentation builds on all pushes and PRs (previously only main branch)

### Fixed
- Replaced deprecated `np.trapz` with `np.trapezoid` in benchmark metrics
- Dockerfile entrypoint mismatch with package console script

## [0.1.0] - 2025-10-14

### Added
- Initial release of Mercury Agent ♱ (formerly Omni-Anomaly-Engine)
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
- Renamed from Omni-Anomaly-Engine to Mercury Agent ♱
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

[0.1.0]: https://github.com/Steel-SecAdv-LLC/Mercury Agent ♱/releases/tag/v0.1.0
