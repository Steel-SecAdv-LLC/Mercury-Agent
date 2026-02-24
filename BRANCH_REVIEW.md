# Branch Review — Mercury-Agent

**Date:** 2026-02-24
**Reviewer:** Claude (automated audit)
**Scope:** All 20 remote branches evaluated against `main` (`e07658e`)
**Conflict analysis:** Dry-run merges performed for all 5 candidate branches

---

## Executive Summary

Of 20 branches audited, **2 merge cleanly into main** and contain substantial unmerged value. **18 should be deleted** — their work is either already in main, superseded, or would produce unresolvable conflicts because main has more complete implementations.

### Merge Architecture

```
main (e07658e)
  │
  ├─── PR #1: claude/audit-repo-docs-sync-if68l-lrQHe  (merge first)
  │    14 commits, 67 files, +1,633 / -1,017
  │    Focus: type safety, test infrastructure, module registration
  │    Conflict check: 0 conflicts vs main ✓
  │
  └─── PR #2: claude/merge-mercury-agent-0H8R0          (merge second)
       33 commits, 62 files, +15,024 / -536
       Focus: security hardening, +213 tests, CI fixes, Oracle features
       Conflict check: 0 conflicts vs main ✓
                       0 conflicts vs main+lrQHe ✓

Both branches tested in both merge orders — clean in either direction.
```

**Recommended order: `lrQHe` first, then `0H8R0`.** Rationale: `lrQHe` is infrastructure-focused (type fixes, test deps, module wiring) and creates a cleaner foundation for the larger feature/security PR.

---

## MERGE — 2 Branches (0 conflicts, ready for PR)

### PR #1: `claude/audit-repo-docs-sync-if68l-lrQHe`

- **Status:** 14 commits ahead, 2 behind main
- **Fork point:** `ee647e1` (between PR #126 and #127)
- **Merge test:** 0 conflicts — 8 files auto-merged cleanly
- **Scope:** 67 files, +1,633 / -1,017

**What it delivers:**

| Category | Changes |
|----------|---------|
| **Type safety** | 26 mypy errors resolved across 4 detector files (`dimensional`, `directive`, `spatial`, `temporal`) |
| **Test infrastructure** | Test dependencies declared in `pyproject.toml` optional-dependencies |
| **Torch guards** | `TORCH_AVAILABLE` guards added to bare `torch.Tensor` isinstance checks |
| **Loader fix** | CICIDS test loader short-circuited to synthetic mode |
| **Package consolidation** | `federated_robust` consolidated into `federated_learning` package |
| **Module registration** | 10+ orphan modules wired into `__init__.py` exports; 14 new smoke tests added |
| **Scheduler completion** | `HyperbandBracket.mark_complete` and `ASHAScheduler.on_trial_complete` implemented |
| **Deduplication** | `data/benchmarks` duplicate modules replaced with re-export shims (-937 lines) |
| **Safety guards** | Numerical safety for conformal prediction and GOSNN ethical gate |
| **Exception handling** | Narrowed bare `except` to specific exceptions in timeseries and optimization |
| **Documentation** | Dual Lyapunov constants documented; docs aligned with codebase across 10 files |
| **Formatting** | ruff and black applied across 23 files |

---

### PR #2: `claude/merge-mercury-agent-0H8R0`

- **Status:** 33 commits ahead, 1 behind main
- **Fork point:** PR #127 (`6584c25` — federated learning framework)
- **Merge test:** 0 conflicts — clean merge
- **Scope:** 62 files, +15,024 / -536

**What it delivers:**

| Category | Changes |
|----------|---------|
| **Security hardening** | Comprehensive hardening across 15 modules (API auth, routes, server, voice, cache HMAC, encryption) |
| **Test coverage** | +213 security-critical tests across 9 new test files |
| **CI stabilization** | All 18 remaining mypy type errors fixed; CodeQL clear-text logging alert resolved; formatting/linting cured |
| **SpectralDomainOracle** | Full production-grade neuro-symbolic detector (+1,683 lines): Selective Inference, binary segmentation, windowed DFT, spectral flux, phase coherence, cepstral analysis, phi-weighted influence, 7-domain support |
| **Cross-domain analysis** | New `cross_domain_frequency.py` module (+298 lines) |
| **F1 Precision Directive** | Inversion guard, ensemble flip, domain presets, residual frequency filter, multi-strategy threshold selection |
| **Oracle enhancements** | Noise color estimation, adaptive alpha, asymmetric bias, spectral hints, federation serialization |
| **Benchmark expansion** | 75-dataset results (64 successful, mean AUC 0.8277), domain summary reporting, per-dataset JSON output |
| **Documentation** | ARCHITECTURE.md, CROSS_DOMAIN_ANALYSIS.md, DATASETS.md, DOMAIN_PERFORMANCE.md, ORACLE_NOISE_COLOR.md |

**High-churn files (review carefully):**
- `src/omni_mercury_engine/detectors/statistical.py` — touched in 10+ commits
- `benchmarks/honest_benchmark.py` — touched in 6 commits
- `src/omni_mercury_engine/detectors/spectral_domain_oracle.py` — touched in 3 commits

---

## REMOVE — 18 Branches

### Conflict-tested and downgraded (3 branches)

These were initially retained for review but conflict analysis revealed their work is superseded:

| Branch | Conflicts | Why Remove |
|--------|:-:|---|
| `claude/setup-mercury-agent-o33Fu` | **8** | Main has a full 500+ LOC SpectralDomainOracle; this branch has a simpler 87-line version. Every conflict resolves to main's side. All features (inversion guard, domain presets, noise color) already ported into `0H8R0`. |
| `claude/audit-repo-docs-sync-5jj1y` | **20** | Strict subset of `lrQHe` (same base work, fewer commits). `lrQHe` merges cleanly and contains all of `5jj1y`'s unique value. 20 conflicts include modify/delete on 4 `data/benchmarks` files. |
| `devin/1771750539-mercury-strategic-improvements` | **14** | Architectural divergence: uses different Oracle init pattern (`_inferred_oracle_domain` vs main's `_oracle_detector`). Docs (`ARCHITECTURE.md`, `CROSS_DOMAIN_ANALYSIS.md`, `DATASETS.md`) already exist on main with different content. Code redundant with `0H8R0`. |

### Previously identified for removal (15 branches)

All branches below have had their core work merged into main via PRs #121–#130, are significantly behind main, or are strict subsets of retained branches.

#### Ensemble Integration Family (forked from #121, 6 behind main)

| Branch | Ahead | Reason |
|--------|:---:|--------|
| `claude/mercury-ensemble-integration-48eLM` | 2 | Foundational domain loaders. Absorbed into main via PR #123. |
| `claude/mercury-ensemble-integration-GaakW` | 8 | Extends 48eLM. Federated + quality work in main via #123/#127. |
| `claude/mercury-ensemble-integration-3rQdY` | 26 | Superseded by tlKK5. Both stale, work in main. |
| `claude/fix-codeql-alerts-tlKK5` | 28 | Near-identical to 3rQdY + 1 commit. CodeQL redone in later PRs. |
| `claude/mercury-ensemble-integration-OpCPx` | 4 | Independent reimplementation. Small, 6 behind. |

#### Anomaly Detection Replacement (forked from #118, 7 behind main)

| Branch | Ahead | Reason |
|--------|:---:|--------|
| `claude/replace-anomaly-detection-1S5P9` | 8 | Core work merged as PR #121. |
| `claude/validate-ensemble-fix-infrastructure-wJl7g` | 16 | Extends replace branch. Absorbed by later PRs. |

#### Early Validation & Baseline (forked from #118, 7 behind main)

| Branch | Ahead | Reason |
|--------|:---:|--------|
| `claude/mercury-agent-validation-LqBB9` | 8 | v1.4 validation. Superseded by v1.5+ work. |
| `claude/test-mercury-agent-baseline-9LGTm` | 7 | Baseline + cleanup. Superseded by #123+. |

#### Calibration & Polish (forked from #125, 4 behind main)

| Branch | Ahead | Reason |
|--------|:---:|--------|
| `claude/add-calibration-validation-ocVBf` | 30 | Core work merged as PR #126. |
| `claude/final-polish-pr-AQB4Y` | 2 | 2 README commits. Subset of ocVBf. |
| `claude/audit-mercury-agent-w45pR` | 7 | Repo-wide audit. Subset of ocVBf. |

#### Docs Alignment (forked from #123, 5 behind main)

| Branch | Ahead | Reason |
|--------|:---:|--------|
| `claude/audit-docs-alignment-NaPU8` | 28 | Adaptive weighting reimplemented in PR #130. |
| `claude/audit-repo-docs-sync-if68l` | 5 | Strict subset of if68l-lrQHe. Redundant. |

#### Superseded Setup Branch

| Branch | Ahead | Reason |
|--------|:---:|--------|
| `claude/setup-mercury-agent-T0biU` | 20 | Strict subset of 0H8R0 (all 20 commits shared). |
