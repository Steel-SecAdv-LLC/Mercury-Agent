# Merge Reconciliation Plan — v1.6.0 Branch & PR Convergence

**Authoring branch:** `claude/review-merge-branches-4UNr9`
**Verified against `origin/main` HEAD:** `de8feed5c8561815884f7073910668ab0e4354a0`
**Date:** 2026-05-07

This document is intended for execution by an AI coding assistant. Every claim
below was verified by direct `git rev-parse`, `git diff`, or `git show`
against the live `origin/*` refs at the time of authoring; suspected facts
that did not survive verification have been struck through and corrected.
The executor should re-run the verification commands listed in §7 before
acting, since branch HEADs may have advanced.

The goal is **lossless reconciliation**: every distinct unit of value that
exists on any side branch ends up on `main` exactly once, no work is
silently dropped, and no `# type: ignore` / disabled gate / unseeded RNG
that was supposed to be cured is allowed to escape into v1.6.0.

---

## 1. Executive summary of corrections to the original assessment

The user-supplied branch inventory contained five factual claims that did
not survive verification. The plan below replaces them with the verified
state. **The executor must operate from this document, not from the
original tasking note.**

| # | Original claim | Verified state |
|---|---|---|
| 1 | "Resurrect `claude/audit-codebase-phase-2-9xwsT`; main has Phase 2 ITEM 1 but not ITEM 2 (AMA v3.1.0 FIPS 204/205 wiring)." | **FALSE.** `pqc_backends.py`, `crypto_api.py`, `mercury_amacrypto.py` are byte-identical between `origin/main` and the Phase 2 branch (same blob SHAs). The FIPS 204/205 wiring described in commit `15e0d62` is already in `main`. |
| 2 | "Cherry-pick `8e4aa9e` (planet_mass fix) from `copilot/check-mercury-agent-capabilities`." | **FALSE.** `src/omni_mercury_engine/datasets/space.py` is byte-identical between `origin/main` and the branch (blob `499e24e3f38425be02d6278e96aec1efcb83dbfd`); the `planet_mass` derivation is on `main` at `space.py:401`. |
| 3 | "`copilot/investigate-pr-167-improvements` test files may not be in main: `test_calibration_benevolence_integration.py`, `test_no_silent_failure.py`, `test_mock_fallback_cures.py`." | **FALSE.** All three exist on `main` at `tests/ethical/`, `tests/federated/`, and `tests/`. Of the 32 files this branch touches, **15 are byte-identical to main**, and 3 differ but the branch holds the *older* version. |
| 4 | "`main` has `workflow_dispatch` on `benchmark.yml` only; PR #182's recovery is unique." | **FALSE.** `main` has `workflow_dispatch:` triggers on `benchmark.yml`, `ci.yml`, `docker.yml`, `format.yml`, `pqc-production-check.yml`, `release.yml`, **and** `security.yml` (7 of 8 workflows). The recovery branches are fully superseded. |
| 5 | "PR #189/#190/#191 stack — merge in order: #189 → rebase #190 → rebase #191." | **PARTIAL.** That order is necessary but not sufficient. **#188 collides with #189 on 30 files** (they are parallel reimplementations, not stacked); they must be reconciled first. **#190 collides with #189 on 7 files** (the description's "no overlap with PR #189's 48-file diff" is incorrect). **#191 collides with #190 on 5 files**. See §4 for the corrected order and conflict-by-conflict resolution. |

In addition: `copilot/claudein-house-anomaly-datasets-vln30` was not in
the original inventory and is at exactly `main` HEAD (`de8feed`) — it can
be deleted with full confidence.

---

## 2. Verified ground truth (per-branch, computed against `origin/main`)

| Branch | HEAD (12) | ahead | behind | Merge-base | Disposition |
|---|---|---:|---:|---|---|
| `claude/audit-codebase-iQXtX` | `5722fe621592` | 6 | 7 | `89d5a49d4cf0` | **Stale**: every differing file (9/10) is *smaller* on the branch than on main. Branch is the older fork off pre-#166; review-cure commits were absorbed into the consolidated #168 wave. **Delete after one targeted re-check (§3.B).** |
| `claude/audit-codebase-phase-2-9xwsT` | `75ad560204c2` | 22 | 6 | `455494a54017` | **Mostly stale.** PQC trio is byte-identical to main; branch is *missing* `sigma_immutable_gate.py` and `sigma_immutable_corpus.py` entirely (gate is 0 lines on branch, 501 on main); `tests/ethical/test_hard_enforcement.py` is 302 lines on branch vs 465 on main; `scripts/train_sigma_immutable.py` is 265 lines on branch vs 304 on main. Of 17 "only-in-branch" files, **12 are byte-identical to main** (rebase noise from stale base). **No commits to resurrect.** Delete after §3.A re-check. |
| `claude/audit-pqc-fallback-chain-Po2fD` | `03b88499977c` | 5 | 0 | `de8feed5c856` | **PR #190 — keep.** Type-redef cure (8 files) + RNG cure (cognitive 14 / models 7 / 21 files total). Conflicts with #189 on 7 files (§4.B). |
| `claude/in-house-anomaly-datasets-vLn30` | `b5f8e9c213c4` | 4 | 0 | `de8feed5c856` | **PR #191 — keep.** External-loader fixes + shared `http_get_with_retry`. Overlaps with #190 on 5 files (§4.C). |
| `claude/organize-project-directory-IIqcr` | `2a3f59f123b8` | 23 | 0 | `de8feed5c856` | **PR #188 — reconcile with #189 first.** 30-file overlap; on 7 of 9 sampled overlap files the branch's blob hash is identical to #189's (the same code arrived twice). The non-overlap content (PQC startup gate wired into `omni_mercury_engine/__init__.py`, `persist_benchmark_to_pr.py`, doc realign) is *unique* and must be preserved. See §4.A. |
| `claude/pr-167-post-merge-gltvV` | `51b1a29ced96` | 4 | 5 | `517314aeb4b8` | Same SHA as `claude/remove-production-stubs-efRMQ`. Wave A (#168 = `076eedc`) is on main. **Delete with §7.D verification.** |
| `claude/remove-production-stubs-efRMQ` | `51b1a29ced96` | 4 | 5 | `517314aeb4b8` | Duplicate of the above. **Delete with §7.D verification.** |
| `claude/review-pr-182-uNBeg` | `8fdc6d0cba04` | 1 | 1 | `b85fe135b61f` | `workflow_dispatch` already on all 4 target workflows on main; CONTRIBUTING addition is the only candidate-for-rescue. See §3.C. |
| `copilot/check-mercury-agent-capabilities` | `20e3d6afba7c` | 13 | 3 | `e393f955d142` | Wave B is on main; `planet_mass` is on main; σ_Immutable corpus/gate match main. **One genuinely unique file** survives: `tests/integrations/test_truth_decipher_framework.py` (a σ_Immutable test-bypass fixture for the framework integration tests) which does not exist on main. See §3.D. Branch's workflow files are *behind* main. |
| `copilot/claudein-house-anomaly-datasets-vln30` | `de8feed5c856` | 0 | 0 | `de8feed5c856` | At main HEAD. **Delete unconditionally.** |
| `copilot/claudewave-bc-sigma-automl-moe-system12-xxxx` | `95d68cd0ebcd` | 2 | 4 | `076eedc3b05d` | Diff stat: **+278 / −1623** — branch is missing huge amounts of post-Wave-A work (incl. the `dependabot-auto-merge.yml` workflow, `update_readme_benchmarks.py`, `measure_codebase_scale.py`, full Wave B σ_Immutable). **Delete unconditionally.** |
| `copilot/compare-pr-182-186` | `a02bd2dff597` | 9 | 1 | `b85fe135b61f` | **`git diff --name-only origin/main origin/copilot/compare-pr-182-186` is empty** — every file is content-identical to main. Branch holds replay-merge commits and a `Dockerfile 3.14→3.12 / Helm pin loosen` revert that was already absorbed into the #187 consolidation. **Delete unconditionally.** |
| `copilot/investigate-pr-167-improvements` | `b7757c92fded` | 1 | 6 | `455494a54017` | Of the 32 files in the diff, 15 are byte-identical to main. Of the 3 σ_Immutable files that DIFFER, the branch holds the *older* version (`sigma_immutable_gate.py` 360 vs 501 lines on main; KAT 181 vs 201 lines). **No content to recover.** Delete unconditionally. |
| `copilot/pr-167-promote-sigma-immutable` | `455494a54017` | 0 | 6 | `455494a54017` | Exactly at #166. **Delete unconditionally.** |
| `copilot/refactor-34-unpaired-type-redefinitions` | `db00b1e4b149` | 1 | 0 | `de8feed5c856` | All 7 files are also in #189's filelist. PR #190's spec keeps these files' suppressions until #189 lands; #190 must not double-touch them. After #189 merges, **verify that #190's tracked-debt sweep covers any residual no-redef on these 7 files; if so, delete this branch**. See §4.D. |
| `copilot/tighten-dependabot-settings` | `73859890214b` | 1 | 5 | `517314aeb4b8` | Branch's `dependabot.yml` change is **regressive** — it re-adds the `pypi` registry stanza that PR #181 (`b85fe13`) explicitly removed because it failed the Dependabot schema validator. The auto-merge workflow it adds already exists on main as `.github/workflows/dependabot-auto-merge.yml`. **Delete unconditionally.** |
| `devin/1777949373-fix-dependabot-yml` | `f0be50153a72` | 1 | 2 | `bab408bf64e4` | Same fix as #181 (`b85fe13`) which is on main. **Delete unconditionally.** |
| `devin/1777949743-workflow-dispatch-recovery` | `8a73416d4c23` | 3 | 1 | `b85fe135b61f` | All 4 workflow files already have `workflow_dispatch:` on main with the same recovery commentary. CONTRIBUTING.md has a 35-line addition that is a near-subset of the 45-line addition on `claude/review-pr-182-uNBeg`. See §3.C. |
| `devin/1778041418-v1.6.0-corrective-sweep` | `a8a231a7f14d` | 17 | 0 | `de8feed5c856` | **PR #189 — keep** (foundation of v1.6.0 sweep). |

---

## 3. Branch-by-branch verification & action items

### 3.A — `claude/audit-codebase-phase-2-9xwsT` (verify-and-delete)

The branch description headline (Phase 2 ITEM 2 + AMA v3.1.0 FIPS 204/205)
is already on `main`. The 484-file diff is rebase noise plus the branch
being older than main on σ_Immutable. Before deletion:

```bash
# 1. Confirm PQC trio identicality (already verified at HEAD de8feed,
#    re-verify because main may advance):
for f in src/omni_mercury_engine/security/pqc_backends.py \
         src/omni_mercury_engine/security/crypto_api.py \
         src/omni_mercury_engine/integrations/mercury_amacrypto.py; do
  test "$(git rev-parse origin/main:$f)" \
     = "$(git rev-parse origin/claude/audit-codebase-phase-2-9xwsT:$f)" \
    || echo "DIVERGED: $f — investigate before deleting"
done

# 2. Confirm σ_Immutable on main is a strict superset of branch:
test "$(git show origin/claude/audit-codebase-phase-2-9xwsT:\
src/omni_mercury_engine/security/sigma_immutable_gate.py 2>/dev/null \
| wc -l)" = "0" \
  && echo "OK: gate is missing on branch (predates #167)" \
  || echo "Gate now exists on branch — content-diff before deleting"

# 3. Confirm there is NO commit on this branch that introduces a file
#    that does not exist on main and that is not byte-identical to main.
git diff --name-only origin/main..origin/claude/audit-codebase-phase-2-9xwsT \
  | while read f; do
      a=$(git rev-parse origin/main:$f 2>/dev/null)
      b=$(git rev-parse origin/claude/audit-codebase-phase-2-9xwsT:$f)
      if [ "$a" != "$b" ] && [ -n "$b" ]; then
        # If the branch's version is *larger* than main's, that's a flag.
        amain=$(git show origin/main:$f 2>/dev/null | wc -l)
        abranch=$(git show origin/claude/audit-codebase-phase-2-9xwsT:$f | wc -l)
        if [ "$abranch" -gt "$amain" ]; then
          echo "FLAG (branch larger): $f main=$amain branch=$abranch"
        fi
      fi
    done
```

If §3.A.3 reports zero `FLAG` lines, delete the branch. If it reports any,
diff each flagged file by hand and recover content into a fresh branch off
main before deleting.

### 3.B — `claude/audit-codebase-iQXtX` (verify-and-delete)

The branch's 5 review-cure commits cured items that PR #166 (`455494a`)
landed and PRs #167/#168 evolved further. Every differing file on the
branch is *smaller* than its main counterpart. Before deletion run §3.A.3
substituting this branch's name; if it produces zero flags, delete.

### 3.C — Pick one workflow_dispatch recovery branch and delete the other

`claude/review-pr-182-uNBeg` and `devin/1777949743-workflow-dispatch-recovery`
hold near-duplicate `CONTRIBUTING.md` additions (45-line vs 35-line) and
identical workflow edits. Both workflow edits are already on main.

Action:

```bash
# Diff the two CONTRIBUTING additions:
git diff origin/devin/1777949743-workflow-dispatch-recovery:CONTRIBUTING.md \
        origin/claude/review-pr-182-uNBeg:CONTRIBUTING.md

# Diff each branch's CONTRIBUTING vs main:
git diff origin/main:CONTRIBUTING.md \
        origin/claude/review-pr-182-uNBeg:CONTRIBUTING.md
git diff origin/main:CONTRIBUTING.md \
        origin/devin/1777949743-workflow-dispatch-recovery:CONTRIBUTING.md
```

If either branch's CONTRIBUTING addition is *not* on main (the
"squash-merge skip-directive gotcha" section), cherry-pick the larger
version into a tiny branch off main, open a docs-only PR, merge, then
delete both recovery branches. If both additions are already on main,
delete both branches.

### 3.D — `copilot/check-mercury-agent-capabilities` (extract one file, then delete)

The only genuinely unique residual content is
`tests/integrations/test_truth_decipher_framework.py` (39 lines added in
commit `a992d0a`, "test(truth_decipher): mock σ_Immutable gate for
framework integration tests"). The file does not exist on main; the
underlying module being tested (`TruthDecipherFramework`) does.

Action:

```bash
# Confirm the file path on main:
git ls-tree -r origin/main --name-only | grep -i truth_decipher

# Confirm the test file is unique to the branch:
git rev-parse origin/main:tests/integrations/test_truth_decipher_framework.py 2>&1 \
  | grep -q fatal && echo "OK: only on branch"

# Inspect what it asserts (does it duplicate an existing main test?):
git show origin/copilot/check-mercury-agent-capabilities:\
tests/integrations/test_truth_decipher_framework.py | head -80
```

If the test asserts unique behavior of `TruthDecipherFramework.decipher_truth`
post-Wave-B σ_Immutable and is not duplicated on main, cherry-pick **only**
that file (plus a `tests/integrations/__init__.py` if needed) onto a fresh
branch off main and open a small PR. Then delete the source branch.

The branch's other commits (`a698012` session3 mock, `8e4aa9e` planet_mass,
`99b8b18` Wave B, `c9dc422`/`27e5039`/`eaf534f`/`c2be935` σ_Immutable
groundwork, `4864884` dependabot, `382a411` benchmark persistence,
`ba1f888` doc count alignment, `087a1fb` test refactor) are all
content-identical to or behind main. Confirm via §3.A.3 substitution and
then delete.

### 3.E — Branches to delete unconditionally (no verification beyond §7.D)

These have been verified content-empty against main or fully covered by
in-flight PRs:

- `copilot/claudein-house-anomaly-datasets-vln30` (at main HEAD)
- `copilot/claudewave-bc-sigma-automl-moe-system12-xxxx` (regressive: −1623 lines, missing post-Wave-A work)
- `copilot/compare-pr-182-186` (`git diff --name-only` vs main is empty)
- `copilot/investigate-pr-167-improvements` (15 of 32 files identical; 3 σ_Immutable files older than main)
- `copilot/pr-167-promote-sigma-immutable` (at #166 SHA, 0 commits ahead)
- `copilot/tighten-dependabot-settings` (regressive: re-adds the rejected `pypi` registry)
- `devin/1777949373-fix-dependabot-yml` (covered by #181 / `b85fe13`)
- `claude/pr-167-post-merge-gltvV` & `claude/remove-production-stubs-efRMQ` (same SHA, covered by #168)

Run the §7.D guard before any `git push origin --delete`.

---

## 4. PR-stack reconciliation (#188 / #189 / #190 / #191)

All four PRs are based on `origin/main` HEAD `de8feed`. **CI status across
all four is currently `pending` with `total_count: 0`** — no CI run has
completed for any of them at the time of this writing. The first action of
the executor must be to push a no-op commit (or use the `update_pull_request_branch`
MCP tool) to retrigger CI on each, then proceed only after the CI shape is
known.

### 4.A — Reconcile PR #188 with PR #189 *first*

These are not stacked. They are **parallel reimplementations** that
collide on 30 files. Spot-check on 9 files showed:

| File | #188 vs main | #189 vs main | #188 == #189? |
|---|---|---|---|
| `agentic/agentic_autonomy.py` | changed | changed | **identical** |
| `biometric/voice_recognition.py` | changed | changed | **identical** |
| `core/double_helix_engine.py` | changed | changed | **identical** |
| `datasets/climate.py` | changed | changed | **identical** |
| `federated_learning/client.py` | changed | changed | **identical** |
| `federated_learning/privacy.py` | changed | changed | **identical** |
| `omni_mercury_engine/__init__.py` | changed | changed | **identical** |
| `CHANGELOG.md` | changed | changed | **diverged** |
| `scripts/persist_benchmark_to_pr.py` | added | added | **diverged** |

The src/ work on the 7 identical files is one and the same change derived
twice. The CHANGELOG and `persist_benchmark_to_pr.py` are the only places
where #188 has unique value (PQC startup gate wiring, FallbackChain
ethical re-raise, doc realign for v1.6.0, the "Option A" benchmark
auto-commit script with PATCH-on-existing-PR + no-op-vs-existing-branch-head
+ regression test).

**Recommended action (single new branch, single new PR, supersedes both #188 and #189):**

```bash
git fetch origin
git checkout -b reconcile/v1.6.0-corrective-sweep origin/main

# Step 1: take #189 as the foundation (Devin's branch is the more
# recently updated PR base and matches #188 on the overlap files
# anyway). Squash-merge it in as a single commit:
git merge --squash origin/devin/1778041418-v1.6.0-corrective-sweep
git commit -m "v1.6.0 corrective sweep (foundation, was PR #189)"

# Step 2: cherry-pick #188's UNIQUE content. The 30 overlap files are
# already covered. Cherry-pick the unique-to-#188 paths:
#   - persist_benchmark_to_pr.py (use the more polished version per
#     PR #188's review history; verify by line count / test coverage)
#   - tests/scripts/test_persist_benchmark_to_pr.py
#   - The PR description's "PATCH-on-existing-PR" / "no-op-vs-existing-
#     branch-head" / GitHub-web-flow-signing logic in persist script
#   - Doc realign: SECURITY.md, README.md, ARCHITECTURE.md,
#     docs/MATH_SPEC.md, docs/ROUTING_GUIDE.md, docs/DEPLOYMENT.md,
#     etc., per #188's "Documentation (14)" list — but only the parts
#     that diverge from #189's already-applied edits. Use
#     `git diff origin/devin/1778041418-v1.6.0-corrective-sweep \
#              origin/claude/organize-project-directory-IIqcr -- <path>`
#     to find the divergence and apply only that.
#   - CHANGELOG.md: merge the two narratives into a single coherent
#     [Unreleased] section. PR #188's wording about Option A benchmark
#     persistence is unique; PR #189's narrative about the broader RNG
#     cure is the foundation.

# Step 3: push and open a new PR superseding both:
git push -u origin reconcile/v1.6.0-corrective-sweep
# Open as draft, body: "Supersedes #188 and #189; preserves all unique
# content from both. See docs/MERGE_RECONCILIATION_PLAN.md §4.A."
```

After the new PR's CI is green and review is approved, **close #188 and
#189 with a comment pointing at the supersede PR** (do not merge them).
Delete the original branches once the supersede PR merges.

**Alternative path (if the Step 2 hand-merge is too risky):** merge #189
first, then rebase #188 onto post-#189 main. The rebase will produce a
*much smaller* PR because the 30 overlap files (with identical blobs) will
fast-forward away. The result is the same set of commits on main but
loses the squash benefit. Use this fallback only if §4.A's hand-merge
introduces conflict-resolution bugs.

### 4.B — PR #190 conflict-by-conflict resolution

After #188+#189 reconciliation lands, rebase #190 onto the new main:

```bash
git checkout claude/audit-pqc-fallback-chain-Po2fD
git pull --rebase origin main
```

Expect rebase conflicts on these 7 files (`#189 ∩ #190`):

```
src/omni_mercury_engine/agentic/agentic_autonomy.py
src/omni_mercury_engine/biometric/voice_recognition.py
src/omni_mercury_engine/core/double_helix_engine.py
src/omni_mercury_engine/detectors/acceleration_dynamics.py
src/omni_mercury_engine/federated_learning/client.py
src/omni_mercury_engine/federated_learning/server.py
src/omni_mercury_engine/federation/privacy.py
```

The verified per-file analysis showed all 7 are "ALL DIFFERENT" — both
PRs change them from main, with diverging content. Resolution rule:

1. **#189 sets the RNG-cure baseline** (its commits added per-instance
   `np.random.Generator` to a curated subset of cognitive/, federated_learning/,
   federation/, security/, agentic/, biometric/, datasets/ classes).
2. **#190 adds the type-redef cure** (`# type: ignore[no-redef]` removal
   via the `TYPE_CHECKING / else` pattern documented in PR #190's body)
   *plus* an extended RNG cure on cognitive/ and models/.
3. For each of the 7 conflicting files, the post-rebase content must be
   the **union** of #189's RNG cure and #190's type-redef cure. Verification:
   - `grep -nE "type:\s*ignore\[(no-redef|assignment, unused-ignore)\]"` should match #190's spec.
   - `grep -nE "np\.random\.(seed|randn|rand|randint)\("` should be 0 inside the changed classes.
   - Class constructors should grow `seed: int | None = None` and store `self._rng: np.random.Generator = np.random.default_rng(seed)`.
   - Existing call sites should use `self._rng.standard_normal/random/integers/...`.

For each of the 7 files, the rebased version must satisfy *both* checks
simultaneously. Run the regression suite #190 added
(`tests/cognitive/test_rng_seed_reproducibility.py`) after each conflict
is resolved.

`detectors/acceleration_dynamics.py` is an additional special case: it is
also one of the 7 files in `copilot/refactor-34-unpaired-type-redefinitions`.
After #189+#190 land, verify the type-redef cure has actually been applied
(see §4.D).

### 4.C — PR #191 conflict-by-conflict resolution

After #189+#190 land, rebase #191. Expect conflicts on:

```
src/omni_mercury_engine/core/three_r_mechanism.py        (#190 ∩ #191)
src/omni_mercury_engine/datasets/base.py                 (#190 ∩ #191)
src/omni_mercury_engine/datasets/environmental.py        (#190 ∩ #191)
src/omni_mercury_engine/datasets/ocean.py                (#190 ∩ #191)
src/omni_mercury_engine/datasets/space.py                (#190 ∩ #191)
src/omni_mercury_engine/datasets/disaster.py             (#189 ∩ #191)
src/omni_mercury_engine/datasets/security.py             (#189 ∩ #191)
```

Verified line-count signatures (against `de8feed`):

| File | main | #190 | #191 |
|---|---:|---:|---:|
| `core/three_r_mechanism.py` | 2944 | 2978 (+34) | 2953 (+9) |
| `datasets/base.py` | 522 | 522 (=) | 639 (+117) |

Resolution rule:

1. **#191 owns `datasets/base.py`** — the +117 lines are the new
   `http_get_with_retry` shared helper (with strict scheme validation /
   SSRF guard / UA / exponential backoff). #190's change to this file is
   small; the post-rebase version must be #191's helper *plus* whatever
   #190 added (likely an RNG cure inside the synthetic-fallback path of
   one of the dataset classes — verify by `git show
   origin/claude/audit-pqc-fallback-chain-Po2fD:datasets/base.py |
   diff origin/main:datasets/base.py`).
2. **#190 owns `core/three_r_mechanism.py`** — its +34 lines include
   the type-redef cure for the `sph_harm` / `sph_harm_y` scipy
   compatibility shim *and* the `_ConstReplacer` type-correctness fix
   (PR #191 description mentions this fix as a side effect). The
   post-rebase version must be #190's full change (which already
   contains the type-correctness fix) plus any RNG seed plumbing #191
   added (likely none on this file).
3. For `datasets/{environmental,ocean,space,disaster,security}.py`,
   #189/#190 make per-instance Generator changes and #191 makes
   loader-fix changes. These are typically additive and conflict only on
   import lines and `__init__` signatures. Resolve manually; verify by
   running `pytest tests/datasets/ -v` post-rebase.

### 4.D — `copilot/refactor-34-unpaired-type-redefinitions` final check

This branch's 7 files are *all* in #189's filelist, but #189 does not
necessarily apply the type-redef cure on them — it changes them for
unrelated reasons (RNG cure, FallbackChain re-raise, etc.). **PR #190
explicitly carves out these 7 files** with the comment "the 7 PR #189 /
`copilot/refactor-34` files keep their suppressions until that branch
merges."

After #189+#190 land, run on each of the 7 files:

```bash
for f in src/omni_mercury_engine/detectors/acceleration_dynamics.py \
         src/omni_mercury_engine/detectors/dimensional.py \
         src/omni_mercury_engine/detectors/geological/disaster_detectors.py \
         src/omni_mercury_engine/integrations/mercury_amacrypto.py \
         src/omni_mercury_engine/medical/abms_disciplines.py \
         src/omni_mercury_engine/ml/harmonic_encoder.py \
         src/omni_mercury_engine/safeguards/nano_safeguards.py; do
  echo "=== $f ==="
  grep -nE "type:\s*ignore\[no-redef\]" "$f" || echo "  (clean)"
done
```

If any of these files still carries an `[no-redef]` suppression after
#189+#190 merge, **open a follow-up PR** that applies the same
`TYPE_CHECKING / else` pattern that #190 documents. **Do not delete
`copilot/refactor-34` until this check is clean** — it is the last
guarded copy of the cure.

---

## 5. Pre-existing weak points & production-polish tasks

The user's directive forbids leaving pre-existing errors uncorrected. The
verification work surfaced these:

1. **CI is not running on PRs #188/#189/#190/#191.** All four show
   `state: pending, total_count: 0` via the GitHub status API. The
   `format`, `Type Checking`, `Code Quality`, and security workflows
   should fire on every PR but are not. Investigate before merge — one
   plausible cause is that the `paths:` filter on `pqc-production-check.yml`
   excluded the touched files for some PRs, but that does not explain
   `format` / `ci` not running. Most likely cause: a `[skip ci]` substring
   in a commit message, or branch-protection misconfig. Re-trigger via
   `update_pull_request_branch` and confirm checks appear before any
   merge.
2. **PR #188's tracked debt** carves out 35 unpaired
   `# type: ignore[no-redef]` suppressions across 15 stub files and ~80
   files of unseeded `np.random` calls. PR #190 closes 17 + 62 of these.
   After #188/#189/#190 merge, run a full audit:

   ```bash
   grep -rnE "type:\s*ignore\[no-redef\]" src/ | wc -l    # target: 0
   grep -rnE "np\.random\.(seed|rand[a-z]*)\(" src/      # target: every match either in a docstring, in a justified deterministic constant, or inside a class that exposes seed=
   ```

   Anything that survives must be either justified inline (one-line
   comment naming the invariant) or scheduled as a follow-up issue.
3. **`benchmark.yml` `workflow_dispatch` gate** — PR #188 says it now
   requires `github.ref == 'refs/heads/main'` to prevent
   non-main-branch dispatches from opening
   `ci/benchmark-results -> main` PRs with non-main numbers. Verify the
   condition is on main after merge:
   `git show origin/main:.github/workflows/benchmark.yml | grep -A2 workflow_dispatch`.
4. **PQC startup gate** — PR #188 says
   `omni_mercury_engine/__init__.py::_enforce_pqc_production_gate` is
   now wired into package import (was previously a defined-but-uncalled
   hook). Verify:

   ```bash
   AMA_REQUIRE_REAL_PQC=1 python -c "import omni_mercury_engine"   # should RuntimeError if AMA backend is missing
   python -c "import omni_mercury_engine"                          # should succeed (soft import)
   ```
5. **`.coveragerc fail_under` removal** — PR #188 says this was removed
   to prevent partial-suite jobs from inheriting a wrong floor. Verify
   `.coveragerc` on main has no `fail_under` key, and `pyproject.toml`
   still carries `fail_under = 85` as the aspirational target.
6. **Coverage floors** — PR #188 says CORE=15%, FULL=35% (set just below
   the 36.03% measured baseline). Verify in `.github/workflows/ci.yml`
   that the floor numbers match the documented baseline; raise an
   issue if drift exists.
7. **`copilot/check-mercury-agent-capabilities`'s
   `tests/integrations/test_truth_decipher_framework.py`** (§3.D) —
   if the file is genuinely missing on main, the
   `TruthDecipherFramework` integration tests on main are already
   bypassing σ_Immutable (or not running at all). Confirm the
   framework is exercised somewhere else on main before deciding the
   file is or is not needed.

---

## 6. Final disposition matrix (after the above is executed)

| Outcome | Branches |
|---|---|
| **Open PR (merged into main)** | `reconcile/v1.6.0-corrective-sweep` (supersedes #188+#189); rebased `claude/audit-pqc-fallback-chain-Po2fD` (#190); rebased `claude/in-house-anomaly-datasets-vLn30` (#191) |
| **Possible micro-PR** | A docs-only PR for the unique CONTRIBUTING addition (§3.C) and one for `tests/integrations/test_truth_decipher_framework.py` (§3.D), both off main |
| **Deleted (closed PRs first if open)** | All other branches in §2's Disposition column |

Final branch count on `origin`: `main` + the active feature branch(es)
during the merge sequence. **Zero stale branches.**

---

## 7. Verification commands the executor must run

### 7.A — Pre-flight (before any merge or delete)

```bash
git fetch origin --prune --tags
git rev-parse origin/main                        # record this; the plan's claims are tied to a specific HEAD
for b in $(git for-each-ref --format='%(refname:short)' refs/remotes/origin/ \
            | grep -vE '^origin/(HEAD|main)$'); do
  echo "=== $b ==="
  git rev-parse "$b"
  git log --oneline origin/main.."$b" | head -5
done
```

### 7.B — Re-verify the corrections in §1 (run before trusting this plan)

```bash
# 1. Phase 2 ITEM 2 already on main:
for f in src/omni_mercury_engine/security/pqc_backends.py \
         src/omni_mercury_engine/security/crypto_api.py \
         src/omni_mercury_engine/integrations/mercury_amacrypto.py; do
  test "$(git rev-parse origin/main:$f)" \
     = "$(git rev-parse origin/claude/audit-codebase-phase-2-9xwsT:$f)" \
   && echo "OK identical $f" || echo "DIVERGED $f"
done

# 2. planet_mass already on main:
test "$(git rev-parse origin/main:src/omni_mercury_engine/datasets/space.py)" \
   = "$(git rev-parse origin/copilot/check-mercury-agent-capabilities:src/omni_mercury_engine/datasets/space.py)" \
  && echo "OK planet_mass on main"

# 3. test files already on main:
for f in tests/test_mock_fallback_cures.py \
         tests/ethical/test_calibration_benevolence_integration.py \
         tests/federated/test_no_silent_failure.py \
         tests/security/test_nist_fips_kat.py \
         tests/security/test_sigma_immutable_kat.py; do
  git cat-file -e origin/main:$f && echo "OK $f exists on main" || echo "MISSING $f"
done

# 4. workflow_dispatch on every workflow except dependabot-auto-merge:
for w in $(git ls-tree -r origin/main --name-only -- .github/workflows/); do
  c=$(git show origin/main:$w | grep -c 'workflow_dispatch:')
  echo "  count=$c $w"
done
```

### 7.C — PR-stack overlap re-verification

```bash
F189=$(git diff --name-only origin/main..origin/devin/1778041418-v1.6.0-corrective-sweep)
F190=$(git diff --name-only origin/main..origin/claude/audit-pqc-fallback-chain-Po2fD)
F191=$(git diff --name-only origin/main..origin/claude/in-house-anomaly-datasets-vLn30)
F188=$(git diff --name-only origin/main..origin/claude/organize-project-directory-IIqcr)

echo "#189∩#190:"; comm -12 <(echo "$F189"|sort) <(echo "$F190"|sort)
echo "#190∩#191:"; comm -12 <(echo "$F190"|sort) <(echo "$F191"|sort)
echo "#188∩#189:"; comm -12 <(echo "$F188"|sort) <(echo "$F189"|sort) | wc -l
```

### 7.D — Branch-deletion guard (run before every `git push origin --delete`)

For each branch slated for deletion, prove it carries no unique value:

```bash
guard_delete() {
  local b=$1
  local unique
  unique=$(git diff --name-only origin/main..origin/"$b" \
    | while read f; do
        a=$(git rev-parse origin/main:"$f" 2>/dev/null || echo MISS)
        b_=$(git rev-parse origin/"$b":"$f" 2>/dev/null || echo MISS)
        if [ "$a" = "MISS" ] || { [ "$a" != "$b_" ] && [ "$b_" != "MISS" ]; }; then
          # Branch's blob is either net-new or different; if different,
          # is the branch's blob a strict subset of main's?
          if [ "$a" != "MISS" ] && [ "$b_" != "MISS" ]; then
            am=$(git show origin/main:"$f" | wc -l)
            bm=$(git show origin/"$b":"$f" | wc -l)
            if [ "$bm" -gt "$am" ]; then echo "$f"; fi
          else
            echo "$f"
          fi
        fi
      done)
  if [ -z "$unique" ]; then
    echo "OK_TO_DELETE $b"
  else
    echo "WAIT $b — has potentially unique content:"
    echo "$unique"
  fi
}
```

Run `guard_delete` on every branch in §3.E before deleting it. Investigate
every `WAIT` reported.

### 7.E — Post-merge regression suite

After each of #188-via-supersede / #190-rebased / #191-rebased lands:

```bash
# Type-redef cure intact:
grep -rnE "type:\s*ignore\[no-redef\]" src/ | wc -l    # should trend to 0

# RNG cure intact (no new global-state seeding regressions):
grep -rnE "np\.random\.(seed|randn|rand|randint)\(" src/ \
  | grep -v "default_rng\|standard_normal" \
  | grep -vE "^.*\#.*"

# σ_Immutable hard gate intact:
pytest tests/security/test_sigma_immutable_kat.py \
       tests/ethical/test_hard_enforcement.py \
       tests/cognitive/test_rng_seed_reproducibility.py -v

# Loader resilience intact:
pytest tests/datasets/test_loader_resilience.py -v
pytest tests/datasets/test_disaster.py tests/datasets/test_cicids.py -v

# Persist-benchmark regression:
pytest tests/scripts/test_persist_benchmark_to_pr.py -v

# Federated learning seed reproducibility:
pytest tests/test_federated_learning.py -v

# Discovery-verification engine seed:
pytest tests/test_discovery_verification.py::TestDoubleHelixEngine -v

# Full type-check on the touched surface:
mypy --follow-imports=silent src/omni_mercury_engine
```

Any test failure or regression is a **gate** — do not declare reconciliation
complete until all of the above are green on main.

---

## 8. Out-of-scope items the executor should flag (not silently accept)

- Any PR description that disagrees with the verified state in §1. The
  PR bodies for #188, #189, #190 should be edited at merge time to point
  at this plan and to retract claims that did not survive verification.
- Any branch that grows new commits while this plan is being executed
  must be re-verified via §7.A and §7.D before its disposition is
  finalized.
- The `ci/benchmark-results` auto-PR machinery (PR #188's "Option A")
  should be smoke-tested end-to-end after merge; if the bot's verified
  signature contract regresses (web-flow signing on
  `github-actions[bot]` via `GITHUB_TOKEN`, *not* a PAT), open an issue
  rather than papering over.
- Any `[skip ci]` / `[ci skip]` substring in a future squash-merge
  commit message will silently re-trigger the original PR #182 defect.
  The CONTRIBUTING.md addition (§3.C) documents this; if the executor
  finds neither version of the addition on main, restore it.

---

*End of plan.* The executor should now run §7.A, then §7.B to
re-establish ground truth, and proceed with §3 deletes (gated by §7.D),
§4.A reconciliation, §4.B/C rebases, and §5 polish in that order.
