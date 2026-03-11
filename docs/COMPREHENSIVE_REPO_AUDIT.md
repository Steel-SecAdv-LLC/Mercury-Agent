# Mercury-Agent Comprehensive Repository Audit

**Date:** 2026-03-11
**Auditor:** Automated deep-dive analysis
**Scope:** GOSNN, 3R, Ethical Pillars, Hidden/Silenced Issues, Production Readiness

---

## Executive Summary

Mercury-Agent has strong architectural foundations but suffers from a consistent pattern:
**well-designed scaffolding with soft enforcement**. The ethical framework is advisory
rather than mandatory, GOSNN has placeholder data where real model tensors should flow,
the CI pipeline soft-fails on critical security/ethics gates, and production code contains
mock fallbacks that silently degrade functionality without operator awareness.

### Severity Overview

| Severity | Count | Category |
|----------|-------|----------|
| **CRITICAL** | 8 | Ethics not enforced, CI soft-fails security, mock objects in prod, RefactoringEngine is stub, no Respond/Recover |
| **HIGH** | 14 | Silent exception swallowing, placeholder data, bypass vectors, learnable 3R incomplete, one-way GOSNN integration |
| **MEDIUM** | 18 | Hardcoded values, disabled features, coverage gaps, unvalidated claims |
| **LOW** | 10 | Documentation drift, config inflexibility, edge cases |

---

## 1. GOSNN (Graph-Optimized Spiking Neural Network)

### Critical Issues

1. **Placeholder Attention Tensors** (`gosnn_optimizer.py:531-533`)
   - Comment: `"# (This is a placeholder - actual attention tensors would come from model)"`
   - Uses dummy random tensor `(32,16,16)` instead of real model data
   - All attention overhead metrics are computed on synthetic data

2. **Dead Code: `self._fusion`** (`gosnn_integration.py:336`)
   - `self._fusion = None` initialized but **never used anywhere**
   - Fusion strategy is hardcoded inline in `_fuse_predictions()`, not through the fusion object
   - Stacking/BMA fusion objects are created but discarded

3. **Unverified Method Assumptions** (`gosnn_optimizer.py:492, 578`)
   - Calls `_collect_all_scalars()` on GOSNN object without checking the method exists
   - No try/except wrapping - will crash if passed wrong object type

### High Issues

4. **Conformal Prediction Silently Fails** (`gosnn_integration.py:637-642`)
   - Catches `(ValueError, RuntimeError, AttributeError)`, logs, continues
   - `confidence_intervals` silently becomes `None` with no fallback strategy

5. **Silent Domain Setup Failures** (`gosnn_integration.py:416-431`)
   - Failed domain loads are logged as warnings but domain is silently disabled
   - No operator visibility into degraded state

6. **Circular Initialization Risk** (`gosnn_3r_integration.py:318`)
   - `_sync_weights_from_gosnn()` called in `__init__`, reads from `gosnn.scalar_groups` which may not be initialized

### Hardcoded Values That Should Be Configurable

| File | Line | Value | Purpose |
|------|------|-------|---------|
| gosnn_optimizer.py | 34-35 | 0.93, 0.96 | sigma_immutable thresholds |
| gosnn_optimizer.py | 101 | 100 | n_permutations |
| gosnn_optimizer.py | 145 | 0.1, 0.5 | Pruning thresholds |
| gosnn_optimizer.py | 532 | (32,16,16) | Dummy attention shape |
| learnable_gosnn.py | 107-109 | (180, 64, 8) | n_scalars, embedding_dim, n_categories |
| gosnn_integration.py | 44-45 | 1000, 300 | Cache max size and TTL |
| gosnn_3r_integration.py | 63-65 | (0.447, 0.276, 0.276) | Default 3R weights |

### Documentation vs. Reality

| Claim | Reality |
|-------|---------|
| "Stacking/BMA fusion" | Objects created but unused; inline weighted averaging instead |
| "SHAP-based importance" | Permutation importance without SHAP formalism |
| "< 2% overhead" | No actual latency tracking or verification |
| "Learnable scalar embeddings" | Only learns adjustments, base values are fixed |

---

## 2. 3R (Recursion-Resonance-Refactoring) - NOT Recognize-Respond-Recover

### Critical Clarification

**"3R" stands for Recursion-Resonance-Refactoring**, a detection-only framework.
There is NO Respond or Recover implementation. The system detects anomalies but has
no automated response or recovery pipeline.

### Completeness Assessment

| Component | Status | Completeness |
|-----------|--------|-------------|
| **Recursion (R)** - RecursionEngine | Functional | 95% |
| **Resonance (H)** - ResonanceEngine | Functional | 95% |
| **Refactoring (O)** - RefactoringEngine | **STUB** | **15%** |
| **Fusion Equation** - OmniAvaEquation | Functional | 90% |
| **Learnable Fusion** - Learnable3REngine | Incomplete | 40% |
| **Domain Adaptation** - DomainAdaptiveOAEWeights | Partial | 60% |
| **Respond** | **MISSING** | **0%** |
| **Recover** | **MISSING** | **0%** |
| **GOSNN Integration** | One-way only | 50% |
| **End-to-End Training** | Missing | 0% |

### Critical Issues

1. **RefactoringEngine Is a Stub** (`three_r_mechanism.py:2236-2270`)
   - `RefactoringTransformer._reduce_nesting()` only adds a docstring
   - No actual AST code transformations despite claiming "dynamic code optimization"
   - `should_reduce_complexity` flag is set but never used
   - Documentation claims: "AST manipulation for continuous performance improvement"
   - Reality: Demo-only stub that doesn't transform code

2. **sigma_immutable Is Not Immutable** (`three_r/fusion.py:80-91`)
   - Accepted as constructor parameter
   - Clamped to `[0.90, 0.99]` range with only a warning
   - Can be **dropped from 0.96 to 0.90** at instantiation

3. **Learnable3R Lacks Training Infrastructure** (`three_r/learnable_fusion.py:547+`)
   - `train_step()` accepts single samples, not batches
   - No `fit()` method for multi-epoch training
   - No validation loop or convergence monitoring
   - Returns 0.0 silently when PyTorch unavailable

4. **Learnable and Static 3R Are Decoupled**
   - `ThreeRMechanism` uses `OmniAvaEquation` (static weights)
   - `Learnable3REngine` is completely separate
   - Learned weights CANNOT feed back to the main mechanism

5. **GOSNN-3R Integration Is One-Way** (`gosnn_3r_integration.py`)
   - Claims "bidirectional feedback between GOSNN and 3R"
   - Reality: One-way (3R -> GOSNN weights), no reverse integration
   - GOSNN scalar update does not loop back to 3R components

6. **3R and Resilience Are Not Connected**
   - `resilience/` module exists independently
   - 3R has NO imports from resilience
   - Anomaly detection (3R) never triggers recovery (resilience)

7. **GOSNN-3R Weight Sync Assumptions** (`gosnn_3r_integration.py:454`)
   - Assumes GOSNN instance has `_collect_all_scalars()` - no validation
   - Default weights `(0.447, 0.276, 0.276)` are hardcoded, not computed from data

8. **Sliding Window Silent Fallback** (`gosnn_3r_integration.py:192-221`)
   - `normalize()` returns data unchanged if `min_samples` not met
   - No warning logged - completely silent degradation

9. **Incomplete 3R Attention** (`ml/three_r_attention.py`)
   - Contains 4 `print()` statements in production code
   - Attention mechanism works but lacks integration tests verifying end-to-end flow

10. **Ablation Script Missing** (`configs/ablation_3r_lyapunov.yaml:16-17`)
    - Config references `scripts/run_ablation.py` which does not exist
    - Comment admits: "scripts/run_ablation.py is not yet implemented"

### Test Coverage Gaps

- **Learnable3R tests are smoke-only** (`tests/cognitive/test_core_three_r_learnable.py`)
  - `assert engine is not None` - only checks instantiation
  - No functional, convergence, or accuracy tests
- **No end-to-end pipeline test** - nothing validates full R->H->O->Fusion flow
- **RefactoringTransformer untested** - no tests for actual code transformation
- **Lyapunov stability claims unvalidated** - theoretical formula, no empirical test
- **DomainAdaptiveOAEWeights.fit_domain_profiles()** - no tests at all

### Unvalidated Claims

| Claim | Reality |
|-------|---------|
| "NSL-KDD F1=0.797 -> target 0.92+" | Benchmarks show 0.796 F1 (worse than baseline) |
| "Automatic refactoring via AST" | Stub that only adds docstrings |
| "Lyapunov stability V(S_t) <= e*e^(-0.25t)" | Theoretical; no empirical validation |
| "Domain-adaptive learned weights" | Falls back to golden-ratio defaults with insufficient data |
| "Bidirectional GOSNN-3R feedback" | One-way integration only |

---

## 3. Ethical Pillars - WHERE WE SUCK THE MOST

### The Core Problem

**The ethical framework is advisory, not mandatory.** Every constraint documented as
"immutable" or "inviolable" can be configured away, disabled, or bypassed.

### Critical Finding: No Hard Blocking

The 8 ethical pillars (Compassion, Evidence, Justice, Altruism, Control, Character,
Competence, Commitment) are implemented as **scoring functions that return recommendations**.
No pillar actually blocks execution.

### Bypass Vectors (6 identified)

1. **Benevolence threshold manipulation** (`ethical_bounding.py:647`)
   - `benevolence_threshold: float = 0.99` can be set to `0.0` at construction
   - No minimum enforcement at instantiation

2. **Sigmoid gate replaces hard threshold** (`centralized_constants.py:152-178`)
   - Documentation says "benevolence >= 0.99 required"
   - Code uses logistic curve: `eta(b) = 1/(1 + exp(-k*(b - b0)))`
   - Benevolence of 0.5 still produces non-zero output - **no actual blocking**

3. **All ethical checks can be disabled** (`ethical_governor.py:209-210`)
   ```python
   enable_bias_audits: bool = True,
   enable_sigma_directives: bool = True,
   ```
   Both can be set to `False`, bypassing all ethical governance

4. **PreExecutionBlockingGate has an off switch** (`ai_ethics.py:139`)
   - `enable_blocking: bool = True` - single parameter disables ALL blocking
   - `allow_overrides: bool = False` with `override_key` allows approved bypasses

5. **Domain-specific lower bounds** (`centralized_constants.py`)
   - `SIGMA_IMMUTABLE_MEDICAL = 0.93` (lower than default 0.96)
   - Medical domain intentionally has weaker ethical threshold

6. **Rollback only recorded, not enforced** (`ethical_governor.py:246-294`)
   - `decision.rollback_triggered = True` but decision still returned to caller
   - Caller decides whether to honor rollback

### Ethics Audit Is a Placeholder

**`benchmarks/run_ethics_audit.py`** - The CI ethics audit (Stage 7) is literally:
```python
print("Ethics audit: PASS (placeholder - full audit not yet implemented)")
```
It only checks that the module is importable. No actual ethical validation runs in CI.

### What's Missing for Real Enforcement

- No `EthicalConstraintViolationError` exception thrown on violations
- No mandatory gates in the main execution path (`engine.predict()`)
- No continuous verification during long-running operations
- No escalation protocol (alerts/human review) on constraint breach
- No immutable storage for thresholds (all configurable at runtime)
- Bias audits use simple demographic parity only (no intersectional fairness)

---

## 4. Hidden / Silenced Issues

### Silent Exception Swallowing (CRITICAL)

| File | Lines | Impact |
|------|-------|--------|
| `benchmarks/live_dataset_benchmark.py` | 300-314 | ROC AUC, PR AUC, F1 failures silently `pass` |
| `infrastructure/streaming.py` | 1255 | Kafka connection errors silently `pass` |
| `infrastructure/observability.py` | 265, 287, 389 | OpenTelemetry init failures silently `pass` |
| `crypto/__init__.py` | 324 | Cryptography operations silently `pass` |
| `infrastructure/streaming.py` | 523-936 | 8+ locations return `False`/`0` masking failures |
| `security/realtime_threat_detection.py` | 243, 291 | Errors logged but never raised |

**Total: 105 bare except handlers across 20 source files**

### Mock Objects in Production Code (HIGH)

These aren't test mocks - they're fallbacks in `src/` that silently degrade:

| Mock | File | What It Replaces |
|------|------|-----------------|
| `MockLLMAdapter` | `llm_adapter.py:256-287` | Real LLM providers |
| `MockLVLMBackend` | `lvlm_backends.py:315` | Vision-language models |
| "Mock mode" | `timegpt_adapter.py:247` | TimeGPT (uses simple extrapolation) |
| "Mock mode" | `chronos_adapter.py:256` | Chronos (uses z-score) |
| "Mock Matrix Profile" | `matrix_profile.py:486` | Matrix profile computation |
| "Mock pretraining" | `ppo_trainer.py:389` | PPO pre-training |
| "Mock implementation" | `blip_vlm.py:411,440,515` | BLIP feature extraction |
| Financial stub | `stubs/financial.py:486` | Financial data source |
| Weather stub | `stubs/weather.py:341` | Weather data source |

**Risk:** Operators have no way to know if the system is running on real models
or silent mock fallbacks.

### CI Pipeline Soft-Fails (HIGH)

These `continue-on-error: true` steps mean failures DON'T block the pipeline:

| CI Step | Line | What It Allows Through |
|---------|------|----------------------|
| Pydocstyle | 99 | Docstring violations |
| Safety scan | 185 | Known CVEs in dependencies |
| pip-audit | 192 | Vulnerable packages |
| Semgrep | 197 | Security code issues |
| **Ethics audit** | **441** | **AI ethics failures** |
| Trivy scan | 587, 599 | Container vulnerabilities |
| Documentation | 632 | Documentation failures |

**The security and ethics gates are cosmetic - they log but never block.**

### Disabled Tests

- 15+ tests skip with `torch not installed` in `test_gosnn_fallback.py`
- Network tests disabled by default (`MERCURY_NETWORK_TESTS=1` required)
- ML tests only run on schedule/main PRs, NOT feature branch PRs
- VLM detector test marked `@pytest.mark.skip` (hangs without GPU)
- Coverage threshold: CI enforces **10%** while `pyproject.toml` targets **85%**

### Suppressed Linting (100+ instances)

- `# type: ignore` across multiple security modules
- `# noqa: S105` suppressing hardcoded string warnings in rate_limiting.py
- Bandit skips `B101, B311, B310` in pyproject.toml
- MyPy skipped in pre-commit hooks despite strict config
- Flake8 ignores `E402, E501, F841` in CI

### Disabled Feature Domains

- `financial` domain: `"disabled"` in ORACLE_DOMAIN_POLICY
- `humanitarian` domain: `"disabled"` in ORACLE_DOMAIN_POLICY

---

## 5. Production Readiness - 75/100

### Production Readiness Scorecard

| Category | Grade | Status |
|----------|-------|--------|
| Logging | B+ | PII masking, structured format support; 139 print() in src/ |
| Configuration | A- | Pydantic validation, env vars, K8s secrets; no secret rotation |
| Error Handling | A | Specific exceptions, proper propagation; 25+ broad handlers |
| Performance | B | Load testing infra exists (Locust + k6); no bottleneck testing |
| Security | A- | Comprehensive scanning, PQC; 12+ Trivy ignores, PQC not audited |
| Deployment | A | K8s, Helm, Docker multi-stage; no docker-compose, no GitOps |
| Monitoring | B+ | Prometheus/Grafana/AlertManager; no OpenTelemetry, limited metrics |
| Documentation | B | Good coverage; missing deployment guide, troubleshooting, runbook |
| Dependencies | A- | Security scanning; no lock file, loose version pins, git URL dep |
| Testing | A- | 6,074 tests / 256 files; 10% CI threshold vs 85% target |

### Print Statements in Production Code

139 `print()` calls in `src/` including:
- `score_calibration.py` (22 occurrences)
- `cli.py` (17 occurrences - acceptable for CLI output)
- `calibration_pipeline.py` (6 occurrences)
- `three_r_attention.py` (4 occurrences)
- `cognitive/orchestrator.py` (3 occurrences)

### No Structured Logging Standard

- Mix of `print()`, `logging.warning()`, and structured logger usage
- No consistent log format across modules
- OpenTelemetry referenced in `.env.example:125` but not implemented in code
- 289 source files import logging but no structlog adoption

### Security Concerns

- AMA Cryptography pinned to git URL without commit hash (`pyproject.toml:141`)
- PQC backend "community-tested, NOT externally audited" (`SECURITY.md:77`)
- PyTorch `>=2.2.0` with no upper bound (breaking changes possible)
- NumPy `>=1.24.0` with no upper bound
- Bandit security checks partially disabled (B101, B311, B310)
- 12+ CVEs in `.trivyignore` with quarterly review process
- No `requirements.lock` for reproducible builds

### Missing Production Infrastructure

- No `docker-compose.yml` for local development
- No OpenTelemetry distributed tracing implementation
- No chaos engineering / fault injection tests
- No rollback procedures documented
- No operational runbook for incidents
- No deployment guide (`docs/DEPLOYMENT.md` missing)
- No GitOps (Flux/ArgoCD) configuration
- No secret rotation (External Secrets Operator commented out)
- No performance regression tests in CI
- Load tests (Locust/k6) exist but not integrated into CI pipeline

### What IS Production-Ready

- Docker multi-stage builds with security hardening (no root, SUID bits removed)
- K8s manifests with liveness/readiness probes, PDB, anti-affinity, topology spread
- Helm chart with 100+ config options and environment overlays
- FastAPI with PII masking, JWT auth, rate limiting
- RestrictedUnpickler prevents arbitrary code execution
- Prometheus rules and AlertManager configuration
- 3-20 replica HPA with CPU/memory scaling targets

---

## 6. Top Priority Remediation Recommendations

### P0 - Must Fix (Integrity Risks)

1. **Make ethical gates mandatory** - Replace sigmoid soft-gate with hard threshold + exception
2. **Implement real ethics audit** - Replace placeholder `run_ethics_audit.py` with actual validation
3. **Remove `continue-on-error` from security CI steps** - Safety, pip-audit, Semgrep, ethics, Trivy
4. **Add mock-mode alerting** - Operators must know when fallback mocks are active
5. **Replace GOSNN placeholder attention** - Wire real model tensors into optimizer
6. **Implement RefactoringEngine or remove claims** - Currently a stub that only adds docstrings
7. **Complete Learnable3R training pipeline** - Add fit(), validation loop, convergence criteria

### P1 - Should Fix (Operational Risks)

8. **Replace silent exception swallowing** - 105 bare except handlers across 20 files; at minimum log at ERROR level
9. **Pin AMA Cryptography to commit hash** - Prevent supply chain drift; currently points to main branch
10. **Replace `print()` with structured logging** - 139 occurrences in source code
11. **Enforce coverage threshold** - Close gap between 10% CI and 85% target
12. **Make sigma_immutable actually immutable** - Use `__setattr__` override or frozen slots
13. **Generate `requirements.lock`** - No reproducible builds currently possible
14. **Create `docs/DEPLOYMENT.md`** - Step-by-step production deployment guide missing
15. **Implement OpenTelemetry** - Referenced in config but not implemented in code

### P2 - Should Improve (Quality)

16. **Remove dead `_fusion` code** - Dead variable in gosnn_integration.py
17. **Configure hardcoded GOSNN values** - Move 15+ magic numbers to config
18. **Enable disabled domain policies** - Financial and humanitarian are off
19. **Add intersectional fairness metrics** - Current bias audits are single-axis only
20. **Document mock fallback behavior** - Operators need to know degradation modes
21. **Create `docker-compose.yml`** - Local development requires K8s currently
22. **Integrate load tests into CI** - Locust/k6 exist but aren't automated
23. **Add operational runbook** - No incident response procedures documented
24. **Connect 3R to Resilience module** - Detection and recovery are completely decoupled
25. **Implement bidirectional GOSNN-3R feedback** - Currently one-way only

---

## 7. Validity Assessment

### Where We Are Valid
- Type system and dataclass architecture is solid
- Test structure is well-organized with proper fixtures
- 3R type definitions are comprehensive
- Security modules (PQC, audit logging, rate limiting) have real implementations
- Conformal prediction and calibration pipelines are functional

### Where We Are NOT Valid
- **Ethical claims** - "Immutable" constraints are mutable; "inviolable" principles are advisory
- **CI security** - Pipeline says "security checked" but all checks are soft-fail
- **Model integrity** - GOSNN optimizer validates against random data, not real model output
- **Production status** - Mock fallbacks mean system can silently run in degraded mode
- **Coverage claims** - 85% target with 10% enforcement is misleading
- **Ethics audit** - "PASS" output from a placeholder that checks nothing
- **3R completeness** - "Refactoring" engine is a stub; no Respond or Recover exists
- **Benchmark claims** - "F1 target 0.92+" never achieved; actual benchmarks show 0.796
- **Bidirectional GOSNN-3R** - Integration is one-way only (3R->GOSNN)
- **Learnable 3R** - Training infrastructure incomplete; no fit(), no validation loop
- **Lyapunov stability** - Theoretical claim with zero empirical validation

---

*This audit should be re-run after the `claude/apply-branding-optimize-YYHEA` branch merge
as it contains exception handling tightening and infrastructure export fixes that may
address some findings.*
