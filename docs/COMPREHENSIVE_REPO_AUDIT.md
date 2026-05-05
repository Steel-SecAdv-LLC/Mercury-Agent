# Mercury-Agent Comprehensive Repository Audit

**Date:** 2026-03-11
**Auditor:** Automated deep-dive analysis
**Scope:** GOSNN, 3R, Ethical Pillars, Hidden/Silenced Issues, Production Readiness
**Status (as of 2026-05-05):** *Historical document — preserved as the
audit that motivated the Phase 1 / Phase 2 / Wave A / Wave B
remediation programme.*

> **Resolution status banner.** Many of the CRITICAL and HIGH findings
> in this document have been remediated in subsequent PRs. Do **not**
> read this audit as a description of current state without
> cross-checking against:
>
> | Finding theme                                  | Resolved by | Current state |
> |------------------------------------------------|-------------|---------------|
> | Ethics framework "advisory rather than mandatory" | PR #167 (Phase 2 Item 1), PR #179 (Wave B) | Hard-enforced dual gate at every public boundary; raises `EthicalConstraintViolationError` |
> | σ_Immutable not raised from any code path      | PR #179 (Wave B) | Mandatory hard gate; private `_enable_gosnn` plus auditable `_GOSNN_TESTING_BYPASS` flag |
> | Pickle-based training-data loader              | PR #166 (Phase 1 audit cure) | Pickle code path **deleted**; safe loaders only |
> | Honest benchmarks                              | PR #166      | 64/75 reproducibility framing canonical; aspirational ROADMAP claims removed |
> | CVE remediation, version bump, CHANGELOG       | PR #165 (v1.6.0) | v1.6.0 released |
> | AMA Cryptography fragmentation                 | PR #144, PR #162 | AMA Cryptography v2.0 is the **sole** PQC backend; Mercury refuses to start without it under `AMA_REQUIRE_REAL_PQC=true` |
> | Federated learning silent failures             | PR #168 (Wave A) | Silent-failure fixes landed; benevolence cache, threshold convergence, fibring default, seven-axis matrix |
>
> Consult `CHANGELOG.md`, `docs/ROADMAP.md`, and `ARCHITECTURE.md`
> §"Dual-Gate Hard Ethical Enforcement" for the current contract.

---

## Executive Summary

Mercury-Agent has strong architectural foundations but suffers from a consistent pattern:
**well-designed scaffolding with soft enforcement**. The ethical framework is advisory
rather than mandatory, GOSNN has placeholder data where real model tensors should flow,
the CI pipeline soft-fails on critical security/ethics gates, and production code contains
mock fallbacks that silently degrade functionality without operator awareness.

> **Status update (2026-05-05):** The "advisory rather than mandatory"
> characterization is no longer accurate as of PR #179 (Wave B). The
> Benevolence and σ_Immutable gates are mandatory hard gates at every
> public boundary surface; refer to `ARCHITECTURE.md` for the current
> contract.

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

1. ~~**Placeholder Attention Tensors**~~ **CLOSED** (`gosnn_optimizer.py:553-588`, Phase 2 ITEM 3)
   - The random-tensor fallback is gone. When no
     ``AttentionProvider`` is configured (or the configured one
     raises), ``GOSNNOptimizer.optimize`` skips the metric and
     surfaces "Attention overhead metric skipped" in
     ``recommendations`` so downstream auditors can see the metric
     was not computed. Real tensors flow only via
     ``AttentionProvider.get_attention``.
   - Regression: ``tests/core/test_gosnn_placeholder_cures.py``.

2. ~~**Dead Code: `self._fusion`**~~ **CLOSED** (Phase 2 ITEM 3)
   - The ``self._fusion = None`` attribute has been removed; the
     fusion strategy is now selected by
     ``GOSNNIntegration.fusion_method`` and applied inline in
     ``_fuse_predictions``. No object lingers unused.

3. **Unverified Method Assumptions** (`gosnn_optimizer.py:492, 578`)
   - Calls `_collect_all_scalars()` on GOSNN object without checking the method exists
   - No try/except wrapping - will crash if passed wrong object type

### High Issues

4. ~~**Conformal Prediction Silently Fails**~~ **CLOSED** (Phase 2 ITEM 3)
   - The blanket ``except (ValueError, RuntimeError, AttributeError)``
     in ``GOSNNIntegration.detect`` is gone. Conformal failures now
     propagate to the caller — silent ``confidence_intervals=None``
     is no longer the contract.
   - Regression:
     ``tests/core/test_gosnn_placeholder_cures.py::TestConformalPropagation``.

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
| **Refactoring (O)** - RefactoringEngine | Functional | 70% |
| **Fusion Equation** - OmniAvaEquation | Functional | 90% |
| **Learnable Fusion** - Learnable3REngine | Functional | 75% |
| **Domain Adaptation** - DomainAdaptiveOAEWeights | Partial | 60% |
| **Respond** | **MISSING** | **0%** |
| **Recover** | **MISSING** | **0%** |
| **GOSNN Integration** | One-way only | 50% |
| **End-to-End Training** | Missing | 0% |

### Critical Issues

1. ~~**RefactoringEngine Is a Stub**~~ **RESOLVED** (`three_r_mechanism.py:2236+`)
   - `RefactoringTransformer` now implements real AST transformations:
     - `_reduce_nesting()`: inverts the last trailing `if`-without-`else` in a function body into a guard clause with an early `return None`, leaving the happy path left-aligned (interior `if`s are left intact because injecting an early return ahead of subsequent code would change semantics)
     - `_hoist_repeated_constants()`: extracts repeated numeric/string literals (used 2+ times) from executable body statements into `_const_<n>` named locals, skipping decorators / default arguments / annotations and keeping `int` and `float` namespaces disjoint; functions containing `global`/`nonlocal` are skipped to avoid SyntaxError
   - `should_reduce_complexity` flag now activates constant hoisting
   - 8 unit tests validate correctness including compile+execute verification
   - *(Resolved by branch: `claude/improve-previous-work-k2tWf`)*

2. **sigma_immutable Is Not Immutable** (`three_r/fusion.py:80-91`)
   - Accepted as constructor parameter
   - Clamped to `[0.90, 0.99]` range with only a warning
   - Can be **dropped from 0.96 to 0.90** at instantiation

3. ~~**Learnable3R Lacks Training Infrastructure**~~ **RESOLVED** (`three_r/learnable_fusion.py:547+`)
   - `fit()` method added with: train/val split, mini-batch training, early stopping,
     per-epoch logging, configurable RNG seed, best-epoch model checkpointing
   - `train_step()` retained for single-sample use cases
   - Graceful no-op with warning when PyTorch unavailable
   - 6 unit tests validate training pipeline
   - *(Resolved by branch: `claude/improve-previous-work-k2tWf`)*

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
| "Automatic refactoring via AST" | ✅ Real guard-clause extraction + constant hoisting (8 tests) |
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

> **Phase 2 update (May 2026):** the *decision boundary* surfaces —
> ``CognitiveOrchestrator.analyze``, ``NeuroSymbolicHub.predict``, and
> ``OmniMercuryEngine.detect_with_fusion`` / ``_calibrated`` — now raise
> ``EthicalViolation`` on benevolence-threshold violation. The
> ``strict_ethics=False`` opt-out is deprecated and ignored. See
> ``src/omni_mercury_engine/ethical/__init__.py`` for the full
> contract and ``tests/ethical/test_hard_enforcement.py`` for the
> regression suite. The remaining bypass vectors below (governor,
> ai_ethics gate, ``sigma_immutable`` clamp) carry inline TODO
> markers tagged ``audit-2026-03`` and are tracked for the next
> sweep.

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

### CI Pipeline Soft-Fails (HIGH) — RESOLVED in PR #148

These `continue-on-error: true` steps previously meant failures didn't
block the pipeline. **All security/ethics gates are now blocking** as of
the commits referenced below:

| CI Step | Old line | What It Allowed Through | Status |
|---------|----------|------------------------|--------|
| Pydocstyle | 99 | Docstring violations | Still advisory (`continue-on-error: true`) — codebase-wide docstring hygiene is out of scope for this PR; tracked separately |
| Safety scan | 185 | Known CVEs in dependencies | **BLOCKING** (this PR; per-CVE ignore via `docs/PYTHON_DEP_CVE_AUDIT.md`) |
| pip-audit | 192 | Vulnerable packages | **BLOCKING** (this PR; per-CVE ignore via `docs/PYTHON_DEP_CVE_AUDIT.md`) |
| Semgrep | 197 | Security code issues | **BLOCKING** (PR #148) |
| **Ethics audit** | **441** | **AI ethics failures** | **BLOCKING** (PR #148) |
| Trivy scan | 587, 599 | Container vulnerabilities | **BLOCKING** with `exit-code: 1`, CRITICAL/HIGH severity (PR #148) |
| Documentation | 632 | Documentation failures | Still advisory (codebase-wide hygiene scope) |

**The security and ethics gates are no longer cosmetic — Safety,
pip-audit, Semgrep, Bandit, Trivy (Docker + Filesystem), and Ethics Audit
all hard-fail PRs on findings outside the documented accept-lists.**

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

### Status key: ✅ DONE | 🔲 OPEN

### P0 - Must Fix (Integrity Risks)

1. ✅ **Make ethical gates mandatory** - `EthicalConstraintViolationError` exception + `BenevolenceScorer.enforce()` added; wired into `CognitiveOrchestrator.analyze()` with `strict_ethics=True` default; `MINIMUM_BENEVOLENCE_FLOOR=0.70` clamp prevents threshold manipulation to zero (bypass vector 1 closed); benevolence score + permissibility recorded in `CognitiveAnalysisResult` *(branch: `claude/improve-previous-work-k2tWf`)*
2. ✅ **Implement real ethics audit** - `benchmarks/run_ethics_audit.py` now runs 5 test suites: module imports, 8-pillar config, PreExecutionBlockingGate hard-block verification, EthicalAutonomyGovernor end-to-end, and `ethical_compliance_threshold` immutability
3. ✅ **Remove `continue-on-error` / `|| true` from security CI steps** - Removed from Semgrep, ethics audit, Bandit, Trivy Docker scan, **Safety, and pip-audit**; Trivy narrowed to CRITICAL/HIGH with `exit-code: 1` and is one of two enforcing dep-CVE gates (mirrored by the standalone `Filesystem Security Scan` job in `security.yml`); ethics audit now returns exit code 1 on unexpected failures.  **Semgrep, Safety, and pip-audit are all BLOCKING** with hardened install (missing semgrep `exit 1`s rather than silently passing).  The Safety v3 policy-file blocker is sidestepped via per-CVE `--ignore`/`--ignore-vuln` CLI flags driven by `docs/PYTHON_DEP_CVE_AUDIT.md` (the source-of-truth audit table with per-CVE rationale and 90-day re-review dates); the ignore lists are empty as of PR #148 because every prior finding was resolved by upgrade (PR #165's 27 CVE upgrades + this PR's PyJWT pin for CVE-2026-32597 = 28 CVEs total).  `.safety-policy.yml` remains in the repo as documentation of OS-level CVE acceptances (those continue to be enforced by Trivy via `.trivyignore`); it is no longer wired into `safety check` invocations.  Findings upload as JSON artifacts (`safety-report.json`, `pip-audit-report.json`) for triage.  *(branch: `claude/improve-previous-work-k2tWf`)*
4. ✅ **Add mock-mode alerting** - `MockLLMAdapter.__init__` now emits `logger.warning` when active so operators see degraded state in logs
5. ✅ (partial) **Replace GOSNN placeholder attention** - `AttentionProvider` ABC interface added to `gosnn_optimizer.py`; `GOSNNOptimizer` accepts optional `attention_provider` parameter; when no provider is configured, a deterministic seeded placeholder is used with `logger.warning()` instead of silent `np.random.randn()`; real providers can now be plugged in without modifying the optimizer *(branch: `claude/improve-previous-work-k2tWf`)*
6. ✅ **Implement RefactoringEngine** - `RefactoringTransformer` rewritten with real AST transformations: guard-clause extraction (inverts only the **last** qualifying `if` without `else` at the end of a function body into an early-return; interior `if` statements are intentionally left intact because injecting `return None` ahead of subsequent code would change semantics), constant hoisting (extracts repeated literals ≥2 occurrences into `_const_<n>` locals; only executable body statements are scanned — decorators, default arguments, and annotations are excluded; `int` and `float` constants are tracked under disjoint `(type, value)` keys so `42` and `42.0` hoist independently; generated names are bumped past any pre-existing `_const_<digits>` identifier in the function; functions containing `global`/`nonlocal` declarations are skipped to avoid the *assigned-before-global-declaration* `SyntaxError`); 9 unit tests validate guard clauses, docstring preservation, multi-if semantic preservation, else-preservation, literal hoisting, trivial-value exclusion, string hoisting, default-argument exclusion, and compile+execute *(branch: `claude/improve-previous-work-k2tWf`)*
7. ✅ **Complete Learnable3R training pipeline** - `Learnable3REngine.fit()` implemented with: train/val split, mini-batch training, early stopping (patience + min_delta), per-epoch logging, configurable RNG seed, and **best-epoch model checkpointing** (saves `state_dict` at best val loss and restores after training); graceful no-op when PyTorch unavailable; 6 tests covering history, loss decrease, early stopping, checkpoint restore, min-samples validation, and PyTorch-absent path *(branch: `claude/improve-previous-work-k2tWf`)*

### P1 - Should Fix (Operational Risks)

8. ✅ (partial) **Replace silent exception swallowing** - Conformal prediction failure upgraded from `logger.debug` to `logger.warning` so degraded state is visible; 104 remaining bare-except handlers still open
9. 🔲 **Pin AMA Cryptography to commit hash** - Prevent supply chain drift; currently points to main branch
10. ✅ (partial) **Replace `print()` with structured logging** - `BenchmarkDiagnostics.quick_diagnose()` (10 print calls) converted to `logger.info`/`logger.warning`; `ScoreCalibrationManager.print_diagnostics()` (2 calls) and `diagnose_scores()` (7 calls) converted to `logger.info`/`logger.warning`; ~60 occurrences remain (mostly in `cli.py` which is acceptable for CLI output) *(branches: PR #146 + `claude/improve-previous-work-k2tWf`)*
11. 🔲 **Enforce coverage threshold** - Close gap between 10% CI and 85% target
12. ✅ **Make sigma_immutable actually immutable** - `OmniAvaEquation.__setattr__` now raises `AttributeError` if `ethical_compliance_threshold` is written after construction
13. 🔲 **Generate `requirements.lock`** - No reproducible builds currently possible
14. ✅ **Create `docs/DEPLOYMENT.md`** - Step-by-step production deployment guide added; covers Docker, K8s/Helm, required env vars, health checks, monitoring, secrets, upgrade/rollback, and troubleshooting
15. 🔲 **Implement OpenTelemetry** - Referenced in config but not implemented in code

### P2 - Should Improve (Quality)

16. ✅ **Remove dead `_fusion` code** - `self._fusion = None` and three unreachable `_fusion = ...` assignments removed from `gosnn_integration.py`; `_setup_fusion` simplified to always use ethical-score-weighted averaging
17. 🔲 **Configure hardcoded GOSNN values** - Move 15+ magic numbers to config
18. 🔲 **Enable disabled domain policies** - Financial and humanitarian are off
19. 🔲 **Add intersectional fairness metrics** - Current bias audits are single-axis only
20. ✅ (partial) **Document mock fallback behavior** - `MockLLMAdapter` now logs a warning; `docs/DEPLOYMENT.md` includes a troubleshooting section covering mock-adapter degradation
21. ✅ **Create `docker-compose.yml`** - Local development compose file added; starts mercury-agent API, Prometheus, and Grafana with volume-backed persistence
22. 🔲 **Integrate load tests into CI** - Locust/k6 exist but aren't automated
23. ✅ (partial) **Add operational runbook** - `docs/DEPLOYMENT.md` covers health checks, monitoring metrics, upgrade/rollback procedures, and the most common failure modes
24. 🔲 **Connect 3R to Resilience module** - Detection and recovery are completely decoupled
25. 🔲 **Implement bidirectional GOSNN-3R feedback** - Currently one-way only

---

## 7. Validity Assessment

### Where We Are Valid
- Type system and dataclass architecture is solid
- Test structure is well-organized with proper fixtures
- 3R type definitions are comprehensive
- Security modules (PQC, audit logging, rate limiting) have real implementations
- Conformal prediction and calibration pipelines are functional
- `PreExecutionBlockingGate` correctly hard-blocks destructive/exfiltration/deceptive patterns
- `ethical_compliance_threshold` is now truly immutable after `OmniAvaEquation` construction
- `BenevolenceScorer.score_action` is invoked with a fully-controlled action description in `CognitiveOrchestrator.analyze()` (the user-supplied `domain` is whitelisted before interpolation), and `strict_ethics=True` raises `EthicalConstraintViolationError` when the score is impermissible — mandatory ethical gate in execution path; orchestrator does not call `enforce()` itself because it must surface the analysis-time measurement on the exception
- `MINIMUM_BENEVOLENCE_FLOOR` is enforced via a property setter, so `scorer.benevolence_threshold = 0.0` is silently clamped to the floor (not just on `__init__`) — the floor cannot be lowered after construction
- `EthicalConstraintViolationError` propagates up call stack — cannot be silently ignored — and now carries `analysis_time_ms` for caller diagnostics
- `RefactoringTransformer` performs real AST transformations (last-`if` guard-clause + constant hoisting with `int`/`float` separation, name-collision avoidance, and `global`/`nonlocal` skip)
- `Learnable3REngine.fit()` provides proper training pipeline with best-epoch checkpointing; size-1 mini-batches are skipped to avoid the `BatchNorm1d` crash and `seed=` also seeds PyTorch
- Trivy on the built Docker image (and the standalone `Filesystem Security Scan`) hard-fail on CRITICAL/HIGH dep CVEs and honor `.trivyignore` for documented risk acceptances; Semgrep (SAST), Bandit, **Safety, and pip-audit** are all blocking; per-CVE risk acceptance for Safety/pip-audit is wired via `--ignore`/`--ignore-vuln` CLI flags driven by `docs/PYTHON_DEP_CVE_AUDIT.md` (currently empty — every prior finding was resolved by upgrade in PR #165 + PyJWT pin in PR #148).  Results upload as JSON artifacts for triage

### Where We Are NOT Valid
- **Ethical claims** - ~~"Immutable" constraints are mutable~~ `MINIMUM_BENEVOLENCE_FLOOR` + `enforce()` + `CognitiveOrchestrator` gate close bypass vector 1; **remaining**: sigmoid gate (vector 2), `enable_bias_audits`/`enable_sigma_directives` off-switches (vector 3), `enable_blocking=False` (vector 4), domain lower bounds (vector 5), rollback non-enforcement (vector 6)
- **CI security** - ~~all checks are soft-fail~~ Semgrep, Bandit, Ethics Audit, Trivy (Docker + Filesystem), **Safety, and pip-audit** are now blocking with no escape hatches; Semgrep install hardened so missing semgrep `exit 1`s rather than silently skipping.  The Safety v3 policy-file blocker is sidestepped via per-CVE `--ignore`/`--ignore-vuln` CLI flags driven by `docs/PYTHON_DEP_CVE_AUDIT.md`; the audit table is empty of accepted CVEs at the time of writing because PR #165's upgrades (cryptography 46.0.7, pillow 12.2.0, requests 2.33.1, aiohttp 3.13.4, pytest 9.0.3, black 26.3.1 — 27 CVEs) plus this PR's PyJWT pin (`>=2.12.0`, CVE-2026-32597) cleared every finding.  `.safety-policy.yml` remains in the repo as documentation of OS-level CVE acceptances enforced by Trivy via `.trivyignore`.  A new **Neuro-Symbolic Tests** job runs on every PR and is wired into the `CI Success` rollup, gating the 7-phase cognitive architecture (`tests/cognitive/`, `tests/safeguards/`, neurosymbolic/3R/ethics suites) so cognitive regressions cannot land on feature branches.  `pydocstyle` and Sphinx docs build remain advisory by design (their failures are codebase-wide hygiene work, separate from this PR's scope)
- **Model integrity** - ~~GOSNN optimizer validates against random data~~ `AttentionProvider` interface available; placeholder now uses deterministic seed with `logger.warning()`; **remaining**: no concrete `AttentionProvider` implementation wired to GOSNN model yet
- **Production status** - Mock fallbacks mean system can silently run in degraded mode (LLM adapter now warns; others remain silent)
- **Coverage claims** - 85% target with 10% enforcement is misleading
- **3R completeness** - ~~"Refactoring" engine is a stub~~ real AST transforms implemented; no Respond or Recover exists
- **Benchmark claims** - "F1 target 0.92+" never achieved; actual benchmarks show 0.796
- **Bidirectional GOSNN-3R** - Integration is one-way only (3R->GOSNN)
- ~~**Learnable 3R**~~ - ✅ `fit()` with train/val split, early stopping, and best-epoch checkpointing implemented
- **Lyapunov stability** - Theoretical claim with zero empirical validation

---

### Provenance Trail

| Date | Branch | Items Resolved |
|------|--------|----------------|
| 2026-03-11 | `claude/apply-branding-optimize-YYHEA` | Items 2, 4, 8-partial, 10-partial, 12, 14, 16, 20-partial, 21, 23-partial |
| 2026-03-11 | PRs #142, #144, #146 (cherry-picked) | Black formatting, AMA Crypto v2.0 consolidation, MyPy/monitoring fixes |
| 2026-03-11 | `claude/improve-previous-work-k2tWf` | Items 1, 3, 5-partial, 6, 7, 10-continued |

*Last updated: 2026-05-02 (PR #148 — Safety + pip-audit BLOCKING; CVE audit doc; click/typer pin)*

*Remaining high-priority open items: P1-9 (pin AMA Crypto), P1-11 (coverage threshold), P1-13 (requirements.lock), P1-15 (OpenTelemetry), P2-17 (GOSNN config), P2-18 (domain policies), P2-19 (intersectional fairness), P2-22 (load tests in CI), P2-24 (3R-Resilience), P2-25 (bidirectional GOSNN-3R).*
