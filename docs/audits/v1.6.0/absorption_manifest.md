# v1.6.0 Corrective Sweep — Absorption Manifest (§3.5)

**Date:** 2026-05-08
**Target:** PR #189 (`devin/1778041418-v1.6.0-corrective-sweep`)
**PR #189 HEAD:** `a8a231a7f14d6084967d2f66c84d692084a8caba`
**Base:** `origin/main` @ `de8feed5c8561815884f7073910668ab0e4354a0`

## Branch HEADs (verified 2026-05-08)

| Branch                                            | HEAD       |
|---------------------------------------------------|------------|
| `origin/main`                                     | `de8feed5` |
| PR #189 `devin/1778041418-v1.6.0-corrective-sweep`| `a8a231a7` |
| PR #188 `claude/organize-project-directory-IIqcr` | `2a3f59f1` |
| PR #190 `claude/audit-pqc-fallback-chain-Po2fD`   | `1ac1f61e` |
| PR #191 `claude/in-house-anomaly-datasets-vLn30`  | `0e076a1f` |
| `claude/review-pr-182-uNBeg`                      | `8fdc6d0c` |
| `copilot/check-mercury-agent-capabilities`        | `20e3d6af` |

## Stack topology

```
$ git merge-base --is-ancestor a8a231a7 1ac1f61e   # PR #189 ⊆ PR #190
YES
$ git merge-base --is-ancestor a8a231a7 0e076a1f   # PR #189 ⊆ PR #191
YES
$ git merge-base --is-ancestor 1ac1f61e 0e076a1f   # PR #190 ⊆ PR #191
YES
```

PRs #190 and #191 are **git-stacked descendants** of PR #189, not parallel
branches off main.

## PR #188 — Lineage attribution (7 commits)

The `_pqc_gate.py` blob (`ce64ff1bf40160ec3b07628159c010cef738aa03`), the
real-AMA test file, the job-level `MERCURY_PQC_REAL_AMA: "1"`, the
`setuptools/wheel/cmake/pip` floors, and the AMA top-level package import
path are all already on PR #189 HEAD. None of the PR #188 SHAs are ancestors
of `a8a231a7`. The seven cherry-picks are therefore **lineage-attribution
empty commits** executed with `--allow-empty --keep-redundant-commits`.

| SHA      | Subject                                                              | Files                                                                                | Classification |
|----------|----------------------------------------------------------------------|--------------------------------------------------------------------------------------|----------------|
| e198858  | fix(pqc-gate): hard-require Dilithium+Kyber, soft-require SPHINCS    | `_pqc_gate.py`, `security/pqc_guards.py`, `tests/test_pqc_startup_gate.py`           | A — empty (lineage only) |
| 12d2887  | ci(verify-real-pqc): pin explicit AMA build-system floors            | `.github/workflows/pqc-production-check.yml`                                         | A — empty (lineage only) |
| cc70fc9  | test(pqc): real AMA Cryptography end-to-end gate verification        | `.github/workflows/ci.yml`, `tests/security/test_pqc_gate_real_ama.py`               | A — empty (lineage only) |
| 7e20c5b  | fix(pqc): revert SPHINCS soft-split and the fake-AMA test mocking    | `_pqc_gate.py`, `security/pqc_guards.py`, `tests/test_pqc_startup_gate.py`           | A — empty (lineage only) |
| aa2e63e  | fix(pqc-gate): read AMA *_AVAILABLE flags from top-level package     | `_pqc_gate.py`, `tests/test_pqc_startup_gate.py`                                     | A — empty (lineage only) |
| c05aa11  | fix(review): round 7 — Copilot review on round-6 commits             | benchmark.yml, CHANGELOG.md, persist_benchmark_to_pr.py, _pqc_gate.py, pqc_guards.py | A — empty (lineage only) |
| ab9ec5a  | fix(quality): extract PQC gate to its own module                     | `__init__.py`, `_pqc_gate.py`, `tests/test_pqc_startup_gate.py`                      | A — empty (lineage only) |

### File-level evidence (PR #189 → PR #188 diff)

26 files differ between the branches. **21 of those 26 are out of scope** —
they are RNG cures, type-redef cures, security tightening, and CHANGELOG/docs
content that PR #189 has progressed beyond what the seven absorbed PR #188
SHAs touch. They will not be cherry-picked from PR #188:

```
.github/workflows/release.yml      tests/scripts/test_persist_benchmark_to_pr.py
.github/workflows/security.yml     tests/security/conftest.py            (PR #189 added)
.trivyignore                       tests/test_fallback_ethical_reraise.py (PR #189 added)
Dockerfile                         README.md                              docs/index.md
docs/COMPREHENSIVE_REPO_AUDIT.md   docs/INSTALLATION.md                   src/.../disaster.py
src/.../security.py                src/.../acceleration_dynamics.py       src/.../dimensional.py
src/.../disaster_detectors.py      src/.../arrest.py                      src/.../mercury_amacrypto.py
src/.../abms_disciplines.py        src/.../harmonic_encoder.py            src/.../nano_safeguards.py
```

The remaining 5 files **are** touched by the seven SHAs and are already at the
intended state on PR #189 HEAD:

- `src/omni_mercury_engine/_pqc_gate.py` blob `ce64ff1b…` (verified)
- `src/omni_mercury_engine/__init__.py` lines 81-83 (verified)
- `.github/workflows/pqc-production-check.yml` lines 27, 61-63 (verified)
- `tests/security/test_pqc_gate_real_ama.py` (present)
- `scripts/persist_benchmark_to_pr.py` path-traversal validation (lines 386, 392)

Therefore: each of the seven cherry-picks lands as `git cherry-pick -x
--allow-empty --keep-redundant-commits` and contributes a `(cherry picked
from commit <sha>)` trailer on an empty commit. **No content reimplementation.**

## PR #190 — Six unique commits (content-bearing)

PR #190 is stacked directly on PR #189. The 6 unique commits cherry-pick
cleanly with `git cherry-pick -x` (content-bearing).

| SHA      | Subject                                                              | Stat               | Classification |
|----------|----------------------------------------------------------------------|--------------------|----------------|
| e37ad25  | refactor(types): eliminate 17 type-redef suppressions across 8 files | 8 files, +166/-78  | C — content    |
| f7fb6d1  | fix(reproducibility): RNG cure across datasets/ in-scope (~152 sites)| 10 files, +362/-193| C — content    |
| f7a55cc  | fix(reproducibility): RNG cure across non-datasets src/              | 39 files (large)   | C — content    |
| 66e12c2  | fix(reproducibility): RNG cure for cognitive/ + models/ — 62 sites   | 21 files, +530/-74 | C — content    |
| b68875d  | fix: address PR #190 Copilot review + clear pre-existing mypy errors | 6 files, +221/-34  | C — content    |
| 1ac1f61  | fix(privacy): DifferentialPrivacy dual rng= / seed= constructor      | 1 file, +5/-1      | C — content    |

After the PR #190 picks, the residual `np.random.<global>` sites in
`src/datasets/` synthetic fallbacks not covered by the 3 RNG-cure commits
must still be cured in this PR. Target on PR #189 HEAD post-pick: **0** hits
for the §11 check #8 grep.

## PR #191 — Five unique commits (content-bearing)

PR #191 is stacked on PR #190. After the PR #190 picks, PR #191's 5 unique
commits cherry-pick cleanly with `git cherry-pick -x` (content-bearing).

| SHA      | Subject                                                              | Stat               | Classification |
|----------|----------------------------------------------------------------------|--------------------|----------------|
| a850a17  | fix(core): tighten _ConstReplacer narrowing                          | 1 file, +8/-6      | C — content    |
| 291eb2b  | fix(datasets): cure failing external-API loaders flagged in PR #189  | 8 files, +965/-243 | C — content    |
| 656170e  | fix(datasets): wave 2 — harden remaining live-data fetchers          | 7 files, +239/-122 | C — content    |
| 387fed3  | fix(datasets): wave 3 — strict-by-default helper + reviewer feedback | 9 files, +167/-130 | C — content    |
| 0e076a1  | docs(changelog): add PR #188–#191 reconciliation attribution         | 1 file, +21/-0     | C — content    |

## Rejected — σ_Immutable mock commits

| SHA      | Branch                                  | Subject                                                       | Reason                  |
|----------|-----------------------------------------|---------------------------------------------------------------|-------------------------|
| a992d0a  | copilot/check-mercury-agent-capabilities| test(truth_decipher): mock σ_Immutable gate                   | Mock = Gap (doctrine)   |
| a698012  | copilot/check-mercury-agent-capabilities| test(session3): mock σ_Immutable gate                         | Mock = Gap (doctrine)   |

These commits will be recorded in `CHANGELOG.md` under "Rejected work" with the
SHAs and the reason. They are not cherry-picked. Any test that depends on the
mocked behavior will be rewritten against the real σ_Immutable contract or
deleted.

## CONTRIBUTING.md (verify-only)

`claude/review-pr-182-uNBeg` (8fdc6d0c) has 0 changes to `CONTRIBUTING.md` vs
main. PR #189 already adds 25 lines (51 raw diff lines) to `CONTRIBUTING.md`
covering the squash-merge skip-directive guidance. **No rescue needed.**

## Workflow trigger lines (verify-only)

`workflow_dispatch:` is already on `origin/main` for `benchmark.yml`, `ci.yml`,
`docker.yml`, `format.yml`, `pqc-production-check.yml`, `release.yml`,
`security.yml`. **No rescue needed.**

## Devin's CI safeguard (5ca68b2)

Already on PR #189 HEAD. Verified by `git log origin/main..HEAD --oneline | grep 5ca68b2`.

## Total cherry-pick lineage trailer count (target)

7 (PR #188) + 6 (PR #190) + 5 (PR #191) = **≥ 18 trailers** on PR #189 HEAD
post-absorption. Plus any additional commits this PR makes for residual
RNG/type-redef/escape-hatch cures, which carry no `cherry picked from`
trailer.
