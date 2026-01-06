# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
