# Branch Review — Mercury-Agent

**Date:** 2026-02-24
**Reviewer:** Claude (automated audit)
**Scope:** All 20 remote branches evaluated against `main` (`e07658e`)

---

## Branches to Retain

### RETAIN (High Priority) — PR Candidate

#### `claude/merge-mercury-agent-0H8R0`

- **Status:** 33 commits ahead, 1 commit behind main
- **Fork point:** PR #127 (`6584c25` — federated learning framework)
- **Recommendation:** Strongest candidate for a new PR to main

**Unique unmerged value (13 commits beyond what PR #130 delivered):**

| Commit | Description |
|--------|-------------|
| `0ec7cae` | ci: fix all 18 remaining mypy type errors for clean CI pass |
| `e998879` | ci: fix mypy unused-ignore errors and CodeQL clear-text logging alert |
| `70e1a0d` | ci: cure all failing checks — formatting, linting, types, and CodeQL |
| `8df5747` | tests: comprehensive coverage for security-critical modules (+213 tests) |
| `d69199c` | security+quality: comprehensive hardening across 15 modules |
| — | feat: port inversion guard, ensemble flip, domain presets, residual filter, dynamic severity |
| — | feat: port noise color, adaptive alpha, asymmetric bias, spectral hints into full-power Oracle |
| — | feat: port multi-strategy threshold selection and domain preset wiring into benchmark |
| — | feat: port and adapt F1 precision tests for T0biU Oracle API |
| — | feat: add domain weight presets, noise color docs, docstring rename from o33Fu |
| — | docs: merge F1 Precision Directive findings into documentation |
| — | bench: full benchmark with F1 precision improvements and system activation combined |
| — | fix: federation Oracle serialization includes noise color, fix test_from_statistics |

**Why retain:** Contains substantial security hardening (+213 tests across 15 modules), all mypy errors resolved, CodeQL alerts fixed, and comprehensive CI stabilization. This is production-readiness work not yet in main. Superset of `setup-mercury-agent-T0biU` (shares all 20 of T0biU's commits).

---

### RETAIN (Review) — Cherry-Pick Candidates

#### `claude/setup-mercury-agent-o33Fu`

- **Status:** 7 commits ahead, 1 commit behind main
- **Fork point:** PR #127 (`6584c25`)
- **Files changed:** 11 files, +2,737 / -351

**Unique commits:**

| Phase | Description |
|-------|-------------|
| Phase 1 | Rename Superintelligence Bootstrap to Cognitive Evolution Engine |
| Phase 2 | Add inversion guard and ensemble flip for below-random datasets |
| Phase 3 | Add domain-adaptive weight presets |
| Phase 4-6 | Create Spectral Domain Oracle with noise color calibration |
| Phase 7 | Add residual frequency filtering for score denoising |
| Phase 8 | Multi-strategy threshold selection in benchmark |
| Phase 10 | Tests, docs, benchmarks, and inversion fix refinement |

**Why retain:** Original implementations of Spectral Domain Oracle, noise color calibration, and residual frequency filtering. Some were ported into `0H8R0`, but the original implementations may contain cleaner or alternative approaches worth reviewing before removal.

---

#### `claude/audit-repo-docs-sync-if68l-lrQHe`

- **Status:** 14 commits ahead, 2 commits behind main
- **Fork point:** `ee647e1` (README image update, between PR #126 and #127)
- **Files changed:** 67 files, +1,633 / -1,017

**Key unique work:**

- mypy type safety fixes across 4 detector files (26 errors resolved)
- Test dependencies declared in `pyproject.toml` optional-dependencies
- `TORCH_AVAILABLE` guards added for optional PyTorch imports
- CICIDS test loader fixed for synthetic mode
- `federated_robust` consolidated into `federated_learning` package
- Orphan modules registered with smoke tests
- `HyperbandBracket.mark_complete` and `ASHAScheduler.on_trial_complete` implemented
- Dual Lyapunov constants documented

**Why retain:** Superset of `audit-repo-docs-sync-if68l` (contains all 5 of its commits + 9 more). Has test infrastructure and type safety improvements that may not overlap with `0H8R0`.

---

#### `claude/audit-repo-docs-sync-5jj1y`

- **Status:** 8 commits ahead, 2 commits behind main
- **Fork point:** `ee647e1` (README image update, between PR #126 and #127)
- **Files changed:** 48 files, +1,104 / -946

**Key unique work:**

- Numerical safety guards for conformal prediction
- GOSNN ethical gate safety guards
- Duplicate `data/benchmarks` implementations replaced with re-export shims
- Narrowed exception handling in timeseries loader and optimization modules
- ruff and black formatting fixes

**Why retain:** Conformal prediction safety guards may not overlap with PR #126's calibration work. Worth checking before discarding.

---

### RETAIN (Docs Only) — Extract Then Remove

#### `devin/1771750539-mercury-strategic-improvements`

- **Status:** 23 commits ahead, 1 commit behind main
- **Fork point:** PR #127 (`6584c25`)
- **Files changed:** 29 files, +7,248 / -377

**Unique documentation artifacts:**

- `ARCHITECTURE.md` — system architecture documentation
- `CROSS_DOMAIN_ANALYSIS.md` — cross-domain frequency correlation analysis
- `DATASETS.md` — dataset catalog and descriptions

**Why retain (temporarily):** Devin-authored parallel implementation of strategic improvements (same features as `T0biU`/`0H8R0`, zero shared commits). The code is redundant, but the documentation files may provide unique value worth extracting into main before this branch is removed.

---

## Branches to Remove — 15 Total

All branches below have had their core work merged into main via PRs #121–#130, are significantly behind main, or are strict subsets of retained branches.

### Ensemble Integration Family (forked from #121, 6 behind main)

| Branch | Ahead | Reason |
|--------|:---:|--------|
| `claude/mercury-ensemble-integration-48eLM` | 2 | Foundational domain loaders. Absorbed into main via PR #123. |
| `claude/mercury-ensemble-integration-GaakW` | 8 | Extends 48eLM. Federated + quality work in main via #123/#127. |
| `claude/mercury-ensemble-integration-3rQdY` | 26 | Superseded by tlKK5. Both stale, work in main. |
| `claude/fix-codeql-alerts-tlKK5` | 28 | Near-identical to 3rQdY + 1 commit. CodeQL redone in later PRs. |
| `claude/mercury-ensemble-integration-OpCPx` | 4 | Independent reimplementation. Small, 6 behind. |

### Anomaly Detection Replacement (forked from #118, 7 behind main)

| Branch | Ahead | Reason |
|--------|:---:|--------|
| `claude/replace-anomaly-detection-1S5P9` | 8 | Core work merged as PR #121. |
| `claude/validate-ensemble-fix-infrastructure-wJl7g` | 16 | Extends replace branch. Absorbed by later PRs. |

### Early Validation & Baseline (forked from #118, 7 behind main)

| Branch | Ahead | Reason |
|--------|:---:|--------|
| `claude/mercury-agent-validation-LqBB9` | 8 | v1.4 validation. Superseded by v1.5+ work. |
| `claude/test-mercury-agent-baseline-9LGTm` | 7 | Baseline + cleanup. Superseded by #123+. |

### Calibration & Polish (forked from #125, 4 behind main)

| Branch | Ahead | Reason |
|--------|:---:|--------|
| `claude/add-calibration-validation-ocVBf` | 30 | Core work merged as PR #126. |
| `claude/final-polish-pr-AQB4Y` | 2 | 2 README commits. Subset of ocVBf. |
| `claude/audit-mercury-agent-w45pR` | 7 | Repo-wide audit. Subset of ocVBf. |

### Docs Alignment (forked from #123, 5 behind main)

| Branch | Ahead | Reason |
|--------|:---:|--------|
| `claude/audit-docs-alignment-NaPU8` | 28 | Adaptive weighting reimplemented in PR #130. |
| `claude/audit-repo-docs-sync-if68l` | 5 | Strict subset of if68l-lrQHe. Redundant. |

### Superseded Setup Branch

| Branch | Ahead | Reason |
|--------|:---:|--------|
| `claude/setup-mercury-agent-T0biU` | 20 | Strict subset of 0H8R0 (all 20 commits shared). |
