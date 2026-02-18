# Removals Made During Repository Alignment

This file is for user review. It is NOT committed to the repository.

## Content Removed from README.md
- [x] Old header photo (358x478 `user-attachments/assets/4d7f93cb...`)
- [x] Reference to nonexistent `benchmarks/comprehensive_benchmark_results.json`
- [x] Phantom version reference `v1.4.1` (no such version exists) → replaced with `post-v1.4.0`
- [x] GPL v3 badge line (`[![License: GPL v3]...]`) — per user request
- [x] Old Co-Architects text (`X ⚛ | Caduceus ⚚ | Dev ⚕ | Claude ⊛`) → replaced with `Devin ⚛ | Claude ⊛` per user request

## Content Removed from CHANGELOG.md
- [x] Stale `[Unreleased]` section (was between v1.1.0 and v0.1.0, ~170 lines)
      containing items from Jan 2026 already captured in versioned entries
- [x] Broken release URL (had unencoded space: "Mercury Agent ♱")

## Content Modified (not removed)
- [x] README module count: 415 → 455 (verified via `find`)
- [x] README line count: 246,539+ → 268,000+ (verified via `wc -l`)
- [x] README test file count: 212 → 224 (verified via `find`)
- [x] README test count: 5,114+ → 5,000+ (conservative verifiable floor)
- [x] README Version: v1.4.0 → v1.5.1 (version reconciliation)
- [x] CONTRIBUTING.md Python version: 3.12+ → 3.11+ (matches pyproject.toml)
- [x] docs/ROADMAP.md: added "IMPLEMENTED" status banner with source locations
- [x] .gitignore: added patterns for CI-generated report files
- [x] ci.yml: added NOTE comment on dead ethics audit reference
- [x] All version strings: 1.4.0 → 1.5.1 (7 files)

## Files Added
- [x] DEAD_CODE.md — inventory of 59 orphaned modules
- [x] benchmarks/run_ethics_audit.py — stub to resolve dead CI reference
- [x] benchmarks/baseline_results.json — added `_provenance` disclaimer field

## User Changes Preserved (from NaPU8)
- [x] Header image: `user-attachments/assets/9fae8d89...` (user's chosen image)
- [x] GPL v3 badge removed (user's commit da27c7c)
- [x] Co-Architects: "Devin ⚛ | Claude ⊛" (user's commit ec501ac)
