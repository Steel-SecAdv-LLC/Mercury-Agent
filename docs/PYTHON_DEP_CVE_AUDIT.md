# Python Dependency CVE Audit

**Audit Date:** 2026-05-20 (permanent supply-chain remediation; native re-implementation of pyjwt + joblib eliminates the last three open advisories with **zero risk acceptance**)
**Next Review:** 2026-08-20 (90 days; quarterly cadence)
**Prior Review:** 2026-05-19
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

This document is the source-of-truth audit ledger for Mercury-Agent's
Python supply chain.  As of 2026-05-20 the CI gates carry **no**
`--ignore` / `--ignore-vuln` flags at all: every prior advisory has
been resolved by either a direct upgrade or by a permanent
supply-chain remediation (native re-implementation in Mercury's
own source tree, eliminating the third-party dependency).  This is
the "zero risk acceptance" posture documented in
[`SECURITY.md`](../SECURITY.md): we do not carry waivers for
disputed advisories, and we do not extend re-review windows
indefinitely — we remove the exposed dependency instead.

---

## Audit posture

`safety check` and `pip-audit` are **fully blocking** in CI and run
against an **isolated Mercury install** (`/tmp/mercury-audit-env`)
so the auditor's own transitives — `nltk` via `safety`, etc. —
never appear in Mercury's audited surface.  `continue-on-error:
true` is removed from both steps; any finding hard-fails the
`Security Scan` job and blocks the PR.  No `--ignore` /
`--ignore-vuln` entries are wired into the workflows.

`.safety-policy.yml` remains in the repo as human-readable
documentation of the OS-level CVE risk-acceptance *posture* —
Trivy itself no longer reads a waiver file because the runtime
image, after `apt-get upgrade`, has zero CRITICAL/HIGH findings
under `--ignore-unfixed`.

---

## Current findings (Python-package CVEs)

### Isolated-install enumeration — 0 findings, 0 ignores

The audit is run against an **isolated Mercury install** (the same
environment CI's `Security Scan` job constructs at
`/tmp/mercury-audit-env`):

```
$ python3.12 -m venv /tmp/mercury-audit-env
$ /tmp/mercury-audit-env/bin/pip install --upgrade "pip>=26.1"
$ /tmp/mercury-audit-env/bin/pip install -e ".[api]" --upgrade \
      "click>=8.2" "typer>=0.20"
$ /tmp/mercury-audit-env/bin/pip list --format=freeze \
      > /tmp/mercury-audit-env/requirements.txt
```

Both tools report **zero** advisories with **zero** ignore flags
across the 42-package Mercury [api] surface (Python 3.12,
2026-05-20):

```
$ safety check -r /tmp/mercury-audit-env/requirements.txt \
      --policy-file .safety-policy-v2.yml
EXIT=0  (vulnerabilities_found=0, vulnerabilities_ignored=0,
          packages_found=42)

$ pip-audit --path /tmp/mercury-audit-env/lib/python3.12/site-packages \
      --skip-editable
EXIT=0  No known vulnerabilities found
```

This is the consequence of the upgrade decisions in "Resolved
upgrades" below **plus** the three permanent remediations in
"Permanent supply-chain remediations" that retired the last
upstream-disputed advisories from Mercury's audited surface.  The
`--ignore` / `--ignore-vuln` lists in `ci.yml` and `security.yml`
are therefore **empty by policy, not by accident**: Mercury does
not accept supply-chain risk via waivers.  If a new finding is
ever reported, the response is to upgrade, isolate, or re-
implement — never to ignore.

### Permanent supply-chain remediations (2026-05-20)

The three IGNORE rows that previously occupied this section
(`joblib` PYSEC-2024-277, `nltk` PYSEC-2026-97, `pyjwt`
PYSEC-2025-183) have been removed because the underlying
dependencies have been removed from Mercury's audited surface.
The replacements are listed below for full provenance.

| Advisory | Removed dependency | Replacement | Commit / PR | Notes |
|----------|-------------------|-------------|-------------|-------|
| `PYSEC-2024-277` / [`CVE-2024-34997`](https://osv.dev/vulnerability/PYSEC-2024-277) (`joblib`, disputed by upstream) | `joblib>=1.3.0` from `pyproject.toml` extras `[optimization]` + `[benchmark]` | `concurrent.futures.{ProcessPoolExecutor, ThreadPoolExecutor}` via the rewritten `ParallelExecutor` in `src/omni_mercury_engine/ml/optimization.py` | this PR (v1.7.0 cut) | The disputed deserialization advisory is moot: Mercury no longer ships `joblib` at all.  The `enable_joblib` / `joblib_backend` config field names are preserved as compatibility aliases for downstream config files; the executor honours the `loky` / `threading` / `multiprocessing` vocabulary by mapping to the equivalent stdlib executor.  Locked by `tests/ml/test_new_modules.py::test_parallel_executor_no_joblib_import`. |
| `PYSEC-2026-97` / [`CVE-2026-0846`](https://osv.dev/vulnerability/PYSEC-2026-97) (`nltk`, high — path traversal) | _never a Mercury dependency_ — it appeared in the audit scope only because `safety` itself depends on it | Audit-tool isolation: Mercury install scanned at `/tmp/mercury-audit-env`, audit tools live in the runner Python | this PR (v1.7.0 cut) | `nltk` is an auditor-internal package and was never part of Mercury's supply chain (`grep -rE "import nltk\|from nltk" src/` → no matches; `grep nltk pyproject.toml` → no matches).  Isolating the audit target permanently removes auditor-internal transitives from Mercury's reports. |
| `PYSEC-2025-183` / [`CVE-2025-45768`](https://osv.dev/vulnerability/PYSEC-2025-183) (`pyjwt`, disputed by upstream) | `pyjwt>=2.12.0` from `pyproject.toml` extra `[api]` | `src/omni_mercury_engine/security/native_jwt.py` — pure-stdlib HS256 JWT module: encode/decode/verify built on `hmac`+`hashlib`+`base64`+`json`+`time`, constant-time signature comparison via `security/constant_time.py`, `alg: none` rejected by construction (HS256-only encoder; decoder whitelists algorithms before any HMAC work).  29 unit tests in `tests/security/test_native_jwt.py`; 14 contract tests in `tests/security/test_jwt_auth.py` + `tests/api/test_auth_comprehensive.py` adapted to the new module. | this PR (v1.7.0 cut) | The upstream-disputed "weak encryption" advisory is moot: Mercury no longer ships `pyjwt` at all.  `api/auth.py` imports `omni_mercury_engine.security.native_jwt as jwt` so the encode/decode call sites are unchanged in shape but no third-party JWT library exists in the install set.  Aligns with Mercury's broader "zero-dep crypto where possible" posture (cf. `AMA-Cryptography` INVARIANT-1). |

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
| `pyjwt` | _unpinned_ → **2.12.0** → _removed_ | CVE-2026-32597 (crit header validation) | PR #148 then v1.7.0 native re-impl |

**Total CVEs resolved by upgrade:** 27 from #165 + 1 from PR #148 = 28.
PR #148 also added the `pyjwt>=2.12.0` pin to resolve a finding
against the system-installed PyJWT 2.7.0 (lazy import without a
`pyproject.toml` pin let some deployments end up on the unpatched
version).  In the v1.7.0 cut the `pyjwt` dependency is **removed
entirely** in favour of Mercury's native HS256 implementation; the
CVE-2026-32597 attack surface no longer exists in Mercury's audited
install (no third-party JWT library is present).  See the
"Permanent supply-chain remediations" section above for the full
provenance.

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
