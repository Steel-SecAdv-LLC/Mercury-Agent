# Python Dependency CVE Audit

**Audit Date:** 2026-05-02
**Next Review:** 2026-08-02 (90 days; quarterly cadence)
**Scope:** All Python packages installed by `pip install -e ".[ml,dev]"` against
the `mercury-agent` editable install. Tools: `safety check` (v3.7.0) and
`pip-audit` (v2.10.0), both run in a clean Python 3.11.15 virtual environment
to mirror the GitHub Actions `security-scan` job environment.

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
repo as human-readable documentation of OS-level CVE risk acceptances
(those are enforced by Trivy via `.trivyignore`); it is no longer wired
into `safety check` invocations.

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
| _(none)_ |  |  |  |  |  |  |  |

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
`/usr/local/lib/python3.11/dist-packages`. This produces noisy "findings"
that do not represent the production or CI environment. The audit run is
therefore performed in `/tmp/cve_audit_venv` created via
`python3 -m venv`, then `pip install --upgrade "pip>=26.0"`, then the
production install sequence — exactly matching the CI runner.

**Reproduction:**

```bash
python3 -m venv /tmp/cve_audit_venv
/tmp/cve_audit_venv/bin/pip install --upgrade "pip>=26.0"
/tmp/cve_audit_venv/bin/pip install bandit safety pip-audit
/tmp/cve_audit_venv/bin/pip install -e ".[ml,dev]"
cd /tmp  # avoid Safety auto-discovering the v3 .safety-policy.yml
/tmp/cve_audit_venv/bin/safety check --output json > safety.json
/tmp/cve_audit_venv/bin/pip-audit --format json --output pip-audit.json
```

`cd /tmp` is required because `safety check` auto-discovers
`.safety-policy.yml` in the working directory and rejects the repo's v3
file with `Legacy policy file parser only accepts versions minor than 3.0`
(known limitation of `safety` v3.x's `check` subcommand). In CI this is
a non-issue because the GHA runner workspace does not contain the policy
file at the path Safety scans from. The `cd /tmp` step is local-only.

---

## Re-review schedule

The next audit is due **2026-08-02**. The PR template's Security checklist
references this document; reviewers must check that the "Audit Date" above
is within the past 90 days, and that any new `IGNORE` entry has a fresh
`Re-review` date. Expired entries must be re-justified or upgraded.
