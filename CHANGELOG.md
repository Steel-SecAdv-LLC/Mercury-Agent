# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
- **BREAKING**: JWT authentication now requires `JWT_SECRET_KEY` environment variable
  - Removed insecure default `"dev-secret-key"` (P0 security fix)
  - Clear error message with instructions for secure key generation
- Rate limiting middleware enforced on FastAPI server
  - Token bucket algorithm: 100 requests/min, burst of 20
  - Configurable via `OMNI_RATE_LIMIT_*` environment variables
  - Standard rate limit headers (`X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`)
- CI/CD Trivy security scans now blocking for CRITICAL/HIGH vulnerabilities
- Pre-commit hooks for secret detection (detect-secrets) and security linting (bandit)

### Changed
- **Docker**: Fixed entrypoint from `omni-anomaly` to `omni-ava` (matches setup.py console_scripts)
- **Dependencies**: Added cryptography, fastapi, uvicorn, httpx, pydantic-settings to core requirements
- CI/CD: ML Tests now run on PRs to main/develop (previously only scheduled/manual)
- CI/CD: Documentation builds on all pushes and PRs (previously only main branch)

### Fixed
- Replaced deprecated `np.trapz` with `np.trapezoid` in benchmark metrics
- Dockerfile entrypoint mismatch with package console script

## [0.1.0] - 2025-10-14

### Added
- Initial release of OMNI ♱ AVA (formerly OMNI ♱ AVA)
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
- Renamed from OMNI ♱ AVA to OMNI ♱ AVA
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

[0.1.0]: https://github.com/Steel-SecAdv-LLC/OMNI ♱ AVA/releases/tag/v0.1.0
