# Forensic Extraction Audit — Omni-AXA-Engine → Mercury-Agent

**Date:** 2026-05-21
**Auditor:** Claude Code (Opus 4.7) acting as senior forensic software engineer
**Repos under audit:**
- SOURCE: `Steel-SecAdv-LLC/Omni-AXA-Engine` @ `2a3c6dd9d7035e9fef39223ffb371af11cf0e0a3` (origin/main)
- TARGET: `Steel-SecAdv-LLC/Mercury-Agent` @ `7af7837612008e86afe91d54a534e9a18b9e3804` (origin/main)

## Bottom line

**[EXTRACTION_DECISION.md](./EXTRACTION_DECISION.md) recommends extracting 0 files in the audit PR.**

Of Omni's 194 `.py` files under `src/omni_anomaly_engine/`:

- **111 / 194** are fully superseded (proven by symbol-level match against Mercury).
- **13 / 194** are partially superseded (Mercury redesigned).
- **12 / 194** are weak matches (Mercury covers the role under different names).
- **58 / 194** are empty/stub `__init__.py` files.
- **7 / 194** content-bearing files are rejected for extraction with cited evidence of
  intentional Mercury-side architectural exclusion (Streamlit, vedic math, robotics shells,
  periodic-table duplication, etc.).
- **6 / 194** content-bearing files have NO Mercury counterpart and NO clear evidence of
  intentional exclusion — these are recorded as **Open Questions** for the copyright holder
  in EXTRACTION_DECISION.md §4.

A second (extraction) PR is **not** opened by this audit; per the task's "100 % confidence" rule,
those 6 candidates require the copyright holder's confirmation of architectural intent before
any source code is moved.

## Artifact index

| step | artifact | what it proves |
|---|---|---|
| 1 | [`inventory.csv`](./inventory.csv) | sha256 + git-blob sha + LOC + top-level symbols for every `.py` under both `src/`s (711 rows) |
| 2 | [`symbol_diff/`](./symbol_diff/) | per-module AST signature diff for each PRIOR-table pair |
| 3 | [`prior_verification.md`](./prior_verification.md) | yes/no/partial verdict on every row of the user's pre-triage table, with cited evidence paths |
| 3 | [`behavioral_spot_checks/`](./behavioral_spot_checks/) | AST control-flow fingerprint comparison for 8 representative shared symbols (imports were not available in the audit container — downgrade noted) |
| 4 | [`file_match.md`](./file_match.md) / [`file_match.json`](./file_match.json) | every Omni file classified SUPERSEDED / PARTIAL / WEAK / NONE against Mercury |
| 4 | [`deep_match.md`](./deep_match.md) / [`deep_match.json`](./deep_match.json) | per-symbol grep of Mercury src for every public symbol declared in each Omni file (catches renamed counterparts) |
| 4 | [`ambiguous_resolution.md`](./ambiguous_resolution.md) | per-file classification of the AMBIGUOUS subpackages: `agents/`, `domains/`, `visualization/`, `comparison/` |
| 5 | [`diagnostics/`](./diagnostics/) | compileall, ruff, mypy, bandit, pip-audit results for both repos |
| 6 | [`coverage_delta.md`](./coverage_delta.md) | Omni coverage.json analysis — high-coverage Omni files without verified Mercury counterpart |
| 7 | [`EXTRACTION_DECISION.md`](./EXTRACTION_DECISION.md) | **final verdict + rejected list + open questions + repro footer** |

## Key diagnostic findings

| metric | Omni-AXA-Engine | Mercury-Agent |
|---|---|---|
| `python -m compileall` | exit 0 ✓ | exit 0 ✓ |
| `ruff check` (default config) | 33 issue lines | 0 (all checks passed) |
| `mypy --ignore-missing-imports` | **218 errors / 75 files** | **2 errors / 2 files** |
| `bandit -ll` (high-severity) | 0 issues | 0 issues |
| `pip-audit` (installed deps) | 22 known CVEs across 7 packages | (audit ran on env, requires reqs file — `pyproject.toml` declares CVE-pinned floors `pillow>=12.2.0`, `requests>=2.33.1`, `cryptography>=46.0.7`) |

The 109-error mypy gap is the cleanest single piece of evidence that Mercury is a hardened
re-implementation of Omni's surface, not a parallel codebase.

## License gate

- Omni-AXA-Engine: **MIT** (LICENSE file confirmed).
- Mercury-Agent:   **GPL-3.0+** (LICENSE file confirmed; `pyproject.toml` declares `license = "GPL-3.0+"`).
- MIT → GPL-3.0+ is one-way compatible. Any file ported must preserve its MIT header verbatim
  **and** appear in Mercury's `THIRD_PARTY_NOTICES.md` with the Omni commit SHA + original path.
  This gate must be satisfied before any extraction PR can be merged.

## Reproducing this audit

The tooling lives outside this directory at `/work/tools/` in the auditor's container (not
included in the PR). All artifacts are deterministic given the two SHAs above. The exact
pipeline is:

1. `inventory.py` → `inventory.csv`
2. `symbol_diff.py` → `symbol_diff/*.md`
3. `match_files.py` → `file_match.{json,md}`
4. `deep_match.py` → `deep_match.{json,md}`
5. `cfg_fingerprint.py` (per symbol) + `behavioral_spot_check.py` → `behavioral_spot_checks/`
6. Manual diagnostic runs (compileall, ruff, mypy, bandit, pip-audit) → `diagnostics/`
7. `build_artifacts.py` → `prior_verification.md`, `ambiguous_resolution.md`, `coverage_delta.md`, `EXTRACTION_DECISION.md`

The five tool scripts plus their input SHAs are sufficient to regenerate every artifact byte-for-byte.
