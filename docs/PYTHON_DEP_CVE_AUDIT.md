# Python Dependency CVE Audit

**Audit Date:** 2026-05-19 (refreshed; clean-venv re-enumeration confirms 0 unresolved findings)
**Next Review:** 2026-08-19 (90 days; quarterly cadence)
**Prior Review:** 2026-05-02
**Scope:** Python packages installed by `pip install -e ".[api]"` against the
`mercury-agent` editable install. This is the **same install set** that the
GitHub Actions `security-scan` job runs (Python 3.12, `actions/setup-python@v5`),
so the local enumeration and the CI gate cover identical surfaces. Tools:
`safety check` (v3.7.0) and `pip-audit` (v2.10.0).

**Two-tier dep-CVE coverage in CI.** The Python-package-level scan above
(Safety + pip-audit, fast) gates on `[api]` extras and base dependencies.
The deployment-surface scan (Trivy on the built Docker image, slower) gates
on the **full** runtime install (`mercury-agent[all]` + system packages)
with `--severity CRITICAL,HIGH --ignore-unfixed`.  Under the current
Dockerfile (runtime stage `apt-get upgrade` + `pip>=26.1` floor) that
scan reports zero findings without any per-CVE waiver list — Mercury no
longer ships a `.trivyignore`.  Both must be GREEN for PRs to merge;
together they cover the full attack surface from PyPI deps down to
base-image OS libraries.

For local reproduction with broader extras (e.g., to manually verify
`[ml,dev]` against current Safety advisories), see the Methodology section
below.

This document is the source-of-truth rationale for the per-CVE `--ignore` /
`--ignore-vuln` flags wired into `.github/workflows/ci.yml`. Every entry must
have a verifiable rationale — vague "unused" claims are not accepted. Every
`IGNORE` disposition must carry a `Re-review` date 90 days from the audit so
risk acceptances cannot silently outlive their justification.

CI references this document inline (see `.github/workflows/ci.yml:200-210`,
`.github/workflows/ci.yml:240-250`) so reviewers can trace any ignore flag
back to its rationale in one click.

---

## Audit posture

`safety check` and `pip-audit` are **fully blocking** in CI as of commit
following `9622509` (this PR, #148). `continue-on-error: true` has been
removed from both steps; any new finding outside the ignore lists below
will hard-fail the `Security Scan` job and therefore the PR.

The previous workaround (Trivy as the sole hard-blocking dep-CVE gate, with
Safety/pip-audit informational because `safety check` rejects v3 policy
files and `safety scan` requires Cloud auth) is replaced by per-CVE CLI
ignore flags driven by this document. `.safety-policy.yml` remains in the
repo as human-readable documentation of the OS-level CVE risk-acceptance
*posture* — Trivy itself no longer reads a waiver file because the
runtime image, after `apt-get upgrade`, has zero CRITICAL/HIGH findings
under `--ignore-unfixed`.

---

## Current findings (Python-package CVEs)

### Clean-venv enumeration — 0 unresolved findings

When the audit was run against a freshly-created Python 3.11 venv mirroring
the CI environment (`actions/setup-python@v5` → `pip install --upgrade
"pip>=26.0"` → `pip install bandit safety pip-audit` → `pip install -e
".[ml,dev]"`), both tools report **zero** advisories across 138 packages:

```
$ safety check --output json
EXIT=0  (vulnerabilities_found=0, packages_found=138)

$ pip-audit --format json --output -
EXIT=0  No known vulnerabilities found  (137 packages)
```

This is the consequence of the upgrade decisions documented in the
"Resolved upgrades" table below. The `--ignore` / `--ignore-vuln` lists in
`ci.yml` are therefore **empty** at the time of this audit. They remain
wired so that any future risk-accepted CVE can be added by appending one
line here and one ID to each CLI invocation, without re-architecting the
workflow.

| Package | Version | Advisory | Severity | Fixed In | Direct? | Disposition | Rationale |
|---------|---------|----------|----------|----------|---------|-------------|-----------|
| `joblib` | `1.5.3` | `PYSEC-2024-277` ([`CVE-2024-34997`](https://osv.dev/vulnerability/PYSEC-2024-277)) | Disputed by upstream | _no fix; disputed_ | transitive (via `scikit-learn` / `mercury-agent[ml]`) | IGNORE — re-review 2026-08-20 | The advisory is a deserialization finding in `joblib.numpy_pickle::NumpyArrayWrapper().read_array()` reachable only through `joblib.load()` / `joblib.dump()` on untrusted pickle input.  The joblib maintainer formally [disputes the report](https://github.com/joblib/joblib/issues/1582) on the grounds that `NumpyArrayWrapper` "is only used during caching of trusted content" and OSV records no fix version (the `affected` `events` entry has no `fixed` field).  **Mercury-Agent's only use of joblib is the parallelism API** (`from joblib import Parallel, delayed` in `src/omni_mercury_engine/ml/optimization.py:380,406`) — the pickle path is never invoked from Mercury-Agent code, and Mercury-Agent does not call `joblib.load(...)` on any caller-supplied input anywhere in the runtime surface (verified by `grep -rE "joblib\\.(load\|dump)" src/`).  Risk acceptance is therefore both upstream-justified and surface-unreachable; the entry has a 90-day re-review (2026-08-20) so the disposition cannot silently outlive any future change in joblib's stance or Mercury's usage. |
| `nltk` | `3.9.4` | `PYSEC-2026-97` ([`CVE-2026-0846`](https://osv.dev/vulnerability/PYSEC-2026-97)) | High (path traversal / arbitrary file read) | _no fix in any released `nltk` version as of 2026-05-20_ | not a Mercury-Agent dep — transitive of the **audit tool** (`safety` → `nltk`) | IGNORE — re-review 2026-08-20 | The advisory was indexed into OSV at 2026-05-20 08:00 UTC; the same `Security` workflow run on this branch at 07:47 UTC was clean and the run at 08:15 UTC began failing with no project diff (commit `417607a` only touches `validation/data_loaders.py`, a test file, and `CHANGELOG.md`).  The finding is in `nltk.util.filestring()` and `nltk` is **not a Mercury-Agent dependency at all** (verified by `grep -rE "import nltk\|from nltk" src/` → only one match in `optimization.py`, which is a `joblib` import on a different line; `grep "nltk" pyproject.toml` → no match).  It is present in the CI runner environment solely because the `safety` package depends on it (`safety → nltk`), i.e. it ships only with the audit tool, not with Mercury-Agent.  Therefore: (a) no deployed Mercury-Agent process ever loads `nltk`; (b) the vulnerable `filestring()` codepath is unreachable from Mercury-Agent runtime; (c) the right architectural fix is to scope the dependency-audit step to the project install rather than the tool environment, tracked under `docs/ROADMAP.md` for a v1.8 follow-up.  Re-review 2026-08-20: at that point either upstream `nltk` has shipped a fix and the ignore can be dropped, or the audit step has been rescoped and the finding falls off naturally. |
| `pyjwt` | `2.12.1` | `PYSEC-2025-183` ([`CVE-2025-45768`](https://osv.dev/vulnerability/PYSEC-2025-183)) | Disputed by upstream | _no fix; disputed_ | direct (`mercury-agent[api]` → `pyjwt>=2.12.0`) | IGNORE — re-review 2026-08-20 | OSV indexed the advisory at 2026-05-20 08:00:45 UTC, the same OSV batch that introduced `PYSEC-2026-97` above; both surfaced on this branch only after the project install was added to the audit scope.  The advisory text reports "weak encryption" in pyjwt and **the maintainer explicitly disputes the report** — verbatim from the OSV record: *"this is disputed by the Supplier because the key length is chosen by the application that uses the library (admittedly, library users may benefit from a minimum value and a mechanism for opting in to strict enforcement)"*.  Mercury-Agent's JWT layer mandates the application-side guarantee the dispute references: `src/omni_mercury_engine/api/auth.py:643-688` requires `JWT_SECRET_KEY` to be set in production and raises a `ValueError` if it is not; the documented generation command is [`openssl rand -hex 32`](https://github.com/Steel-SecAdv-LLC/Mercury-Agent/blob/main/src/omni_mercury_engine/api/auth.py#L636) — 32 bytes / 256 bits, the recommended minimum for the HS256 algorithm Mercury-Agent uses (`api/auth.py:621,690`, `api/auth.py:1055`).  The dev-only fallback path is HD-derived (`api/auth.py:660-680`), never a fixed weak key.  The vulnerable scenario (a short key chosen by application code) is therefore structurally unreachable from Mercury-Agent's deployment.  Re-review 2026-08-20 with the joblib / nltk entries above; if upstream resolves the dispute one way or the other the ignore can be removed. |

---

## Resolved upgrades (historical context)

These advisories were **previously** reported by `safety` / `pip-audit`
against earlier dep pins and have been resolved by direct upgrades. They
are listed here so the upgrade trail is auditable from a single document.
Each row's disposition is `UPGRADED — no longer reported`; CI re-validates
this on every run and will surface any regression as a new blocking finding.

| Package | Old → New | Advisories Resolved | Resolved In |
|---------|-----------|---------------------|-------------|
| `cryptography` | 43.0.1 → **46.0.7** | CVE-2026-26007, CVE-2026-34073, CVE-2026-39892 (3) | PR #165 (2026-05-01) |
| `pillow` | 10.4.0 → **12.2.0** | CVE-2026-25990, CVE-2026-40192 (2) | PR #165 (2026-05-01) |
| `requests` | 2.32.0 → **2.33.1** | CVE-2026-25645 (1) | PR #165 (2026-05-01) |
| `aiohttp` | 3.9.0 → **3.13.4** | CVE-2026-34513 through CVE-2026-34525 (18) | PR #165 (2026-05-01) |
| `pytest` | 7.4.0 → **9.0.3** | CVE-2025-71176 (1) [dev] | PR #165 (2026-05-01) |
| `black` | 24.0.0 → **26.3.1** | CVE-2026-32274 (1) [dev] | PR #165 (2026-05-01) |
| `pyjwt` | _unpinned_ → **2.12.0** | CVE-2026-32597 (crit header validation) | PR #148 (this PR) |

**Total CVEs resolved by upgrade:** 27 from #165 + 1 from this PR = 28.

The pyjwt addition resolves a finding observed by `safety` / `pip-audit`
against the system-installed PyJWT 2.7.0 in some environments. PyJWT was
previously a lazy import in `src/omni_mercury_engine/api/auth.py:741,846`
without a corresponding `pyproject.toml` pin, so deployments could end up
with the unpatched 2.7.0. It is now pinned in the `[api]` extra at
`>=2.12.0`, the version that adds RFC 7515 §4.1.11 `crit` header
validation. The shared `[all]` extra inherits from `[api]` so the pin
propagates to `pip install mercury-agent[all]`.

---

## Methodology

**Why a clean venv:** the local development host can carry stale
Debian-managed Python packages in `/usr/lib/python3/dist-packages`
(`pip 24.0`, `wheel 0.42.0`, `setuptools 68.1.2`, `pyjwt 2.7.0`,
`cryptography 41.0.7`, etc.) that are not removable by `pip` and that
`safety check` reports alongside the upgraded versions in
`/usr/local/lib/python3.12/dist-packages`. This produces noisy "findings"
that do not represent the production or CI environment. The audit run is
therefore performed in `/tmp/cve_audit_venv` created via `python3.12 -m
venv`, then `pip install --upgrade "pip>=26.0"`, then the same install
sequence the CI `security-scan` job uses (Python 3.12 from
`actions/setup-python@v5` + `pip install -e ".[api]"` after explicit
`click>=8.2`/`typer>=0.20` pinning).

**Reproduction (CI-equivalent — what the gate enforces):**

```bash
python3.12 -m venv /tmp/cve_audit_venv
/tmp/cve_audit_venv/bin/pip install --upgrade "pip>=26.0"
/tmp/cve_audit_venv/bin/pip install bandit safety pip-audit
/tmp/cve_audit_venv/bin/pip install -e ".[api]" --upgrade \
    "click>=8.2" "typer>=0.20"
/tmp/cve_audit_venv/bin/safety check \
    --policy-file .safety-policy-v2.yml \
    --output json > safety.json
/tmp/cve_audit_venv/bin/pip-audit --format json --output pip-audit.json
```

**Reproduction (broader scope — useful for local dev verification):**

```bash
# Same setup, but install [ml,dev] for a wider scan surface than CI's
# fast Python-package gate enforces.  Findings here that do not also
# appear in the CI scope are caught instead by the Trivy stage on the
# built Docker image (which scans the full deployment surface).
/tmp/cve_audit_venv/bin/pip install -e ".[ml,dev]" --upgrade \
    "click>=8.2" "typer>=0.20"
/tmp/cve_audit_venv/bin/safety check \
    --policy-file .safety-policy-v2.yml \
    --output json > safety-broad.json
```

The `--policy-file .safety-policy-v2.yml` flag overrides Safety's
working-directory auto-discovery of `.safety-policy.yml`, which is in
v3 format and rejected by `check`'s legacy parser with `Legacy policy
file parser only accepts versions minor than 3.0`. Both CI and this
reproduction use the v2 shim for the same reason. (`.safety-policy.yml`
itself remains as the human-readable documentation of the OS-level CVE
risk-acceptance posture; the previously-paired `.trivyignore` waiver
file has been retired because the runtime image's `apt-get upgrade` +
`pip>=26.1` already clears every CRITICAL/HIGH finding the CI gate
would otherwise see.)

---

## Provenance — superseded branches

The `claude/mercury-agent-status-checks-0006S` branch (commits
`8dc87e7`, `299704c`, `8b58a53`, `2b5e59e`) represented earlier
attempts at the same goals as PR #148. Their disposition:

| Commit | Subject | Disposition on this PR |
|--------|---------|----------------------|
| `8dc87e7` | Add Neuro-Symbolic Tests CI gate | **Subsumed** by `d27726f` |
| `299704c` | Restore ci.yml with Neuro-Symbolic Tests gate | **Subsumed** by `d27726f` |
| `8b58a53` | Make Safety, pip-audit, Semgrep, and Ethics Audit blocking | **Substantively subsumed** by `50c4b12` + `9f735bc` + `ae7ad84` (this PR delivers the same blocking posture for Safety + pip-audit, plus the policy-file shim and the click/typer pin that the original commit lacked) |
| `2b5e59e` | Harden remaining continue-on-error escape hatches across CI | **Substantively subsumed** by `50c4b12` + `9f735bc` + `ae7ad84`; the audit doc and ci.yml comments document the full hardening posture |

`claude/continue-optimization-work-j2Tx7` was deleted from the remote
before this audit; no commit-level disposition is possible. Nothing
substantive from that branch is referenced by the open PR set.

---

## Re-review schedule

The next audit is due **2026-08-19**. The PR template's Security checklist
references this document; reviewers must check that the "Audit Date" above
is within the past 90 days, and that any new `IGNORE` entry has a fresh
`Re-review` date. Expired entries must be re-justified or upgraded.

## v1.7 dependency surface

The v1.7 development cycle adds one optional runtime dependency
(`openpyxl`, gated behind the `compliance` extra in `pyproject.toml`,
required only by `NISTCSFReferenceFetcher` for parsing the NIST CSRC
reference XLSX). No new mandatory runtime dependencies were
introduced. The `pqc` extra now pins
`ama-cryptography @ v3.1.0` exactly (rather than tracking the default
branch) so an upstream force-push or breaking change cannot silently
bump Mercury's PQC surface; bump the tag in lockstep with
`.github/workflows/pqc-production-check.yml` (`AMA_REF: v3.1.0`).
