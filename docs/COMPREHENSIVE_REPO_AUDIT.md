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
| **CRITICAL** | 6 | Ethics not enforced, CI soft-fails security, mock objects in prod |
| **HIGH** | 12 | Silent exception swallowing, placeholder data, bypass vectors |
| **MEDIUM** | 15 | Hardcoded values, disabled features, coverage gaps |
| **LOW** | 8 | Documentation drift, config inflexibility, edge cases |

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

## 2. 3R (Recognize, Respond, Recover)

### Architecture Status

The 3R pipeline has strong type definitions and structural scaffolding in `three_r/types.py`,
`three_r/fusion.py`, and `three_r_mechanism.py`. The OmniAvaEquation is implemented with
sigma_immutable gating.

### Issues Found

1. **sigma_immutable Is Not Immutable** (`three_r/fusion.py:80-91`)
   - Accepted as constructor parameter
   - Clamped to `[0.90, 0.99]` range with only a warning
   - Can be **dropped from 0.96 to 0.90** at instantiation

2. **GOSNN-3R Weight Sync Assumptions** (`gosnn_3r_integration.py:454`)
   - Assumes GOSNN instance has `_collect_all_scalars()` - no validation
   - Default weights `(0.447, 0.276, 0.276)` are hardcoded, not computed from data

3. **Sliding Window Silent Fallback** (`gosnn_3r_integration.py:192-221`)
   - `normalize()` returns data unchanged if `min_samples` not met
   - No warning logged - completely silent degradation

4. **Incomplete 3R Attention** (`ml/three_r_attention.py`)
   - Contains 4 `print()` statements in production code
   - Attention mechanism works but lacks integration tests verifying end-to-end flow

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

## 5. Production Readiness Gaps

### Print Statements in Production Code

58 `print()` calls across 20 source files including:
- `score_calibration.py` (22 occurrences)
- `calibration_pipeline.py` (6 occurrences)
- `three_r_attention.py` (4 occurrences)
- `cognitive/orchestrator.py` (3 occurrences)

### No Structured Logging Standard

- Mix of `print()`, `logging.warning()`, and structured logger usage
- No consistent log format across modules
- OpenTelemetry integration silently fails if packages missing

### Security Concerns

- AMA Cryptography pinned to git URL without commit hash (`pyproject.toml:141`)
- PyTorch `>=2.2.0` with no upper bound (breaking changes possible)
- NumPy `>=1.24.0` with no upper bound
- Bandit security checks partially disabled

### Missing Production Infrastructure

- No health check endpoint validation in tests
- No load testing or performance benchmarks
- No chaos engineering / fault injection tests
- No rollback procedures documented
- No runbook for operational incidents
- Metrics endpoints exist but alerting rules are not defined

---

## 6. Top Priority Remediation Recommendations

### P0 - Must Fix (Integrity Risks)

1. **Make ethical gates mandatory** - Replace sigmoid soft-gate with hard threshold + exception
2. **Implement real ethics audit** - Replace placeholder `run_ethics_audit.py` with actual validation
3. **Remove `continue-on-error` from security CI steps** - Safety, pip-audit, Semgrep, ethics, Trivy
4. **Add mock-mode alerting** - Operators must know when fallback mocks are active
5. **Replace GOSNN placeholder attention** - Wire real model tensors into optimizer

### P1 - Should Fix (Operational Risks)

6. **Replace silent exception swallowing** - At minimum log at ERROR level; prefer raising
7. **Pin AMA Cryptography to commit hash** - Prevent supply chain drift
8. **Replace `print()` with structured logging** - 58 occurrences in source code
9. **Enforce coverage threshold** - Close gap between 10% CI and 85% target
10. **Make sigma_immutable actually immutable** - Use `__setattr__` override or frozen slots

### P2 - Should Improve (Quality)

11. **Remove dead `_fusion` code** - Dead variable in gosnn_integration.py
12. **Configure hardcoded GOSNN values** - Move 15+ magic numbers to config
13. **Enable disabled domain policies** - Financial and humanitarian are off
14. **Add intersectional fairness metrics** - Current bias audits are single-axis only
15. **Document mock fallback behavior** - Operators need to know degradation modes

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

---

*This audit should be re-run after the `claude/apply-branding-optimize-YYHEA` branch merge
as it contains exception handling tightening and infrastructure export fixes that may
address some findings.*
