# Security Policy

## Mercury Agent Security Framework

Mercury Agent is a security-focused AI framework developed by Steel Security Advisors LLC. We take security seriously and are committed to maintaining the integrity, confidentiality, and availability of our systems and user data.

## Supported Versions

Security updates are provided for the latest minor release line. Older
minor lines reach end-of-life when a new minor is published; users are
expected to upgrade promptly. Critical CVEs may be back-ported to the
immediately previous line at the maintainers' discretion.

| Version | Status               | Security updates |
| ------- | -------------------- | ---------------- |
| 1.7.x   | **Current**          | :white_check_mark: |
| 1.6.x   | Previous (EOL on next minor) | Critical CVEs only |
| < 1.6   | End-of-life          | :x:              |

## Reporting a Vulnerability

We appreciate the security community's efforts in helping keep Mercury Agent secure. If you discover a security vulnerability, please follow our responsible disclosure process.

### How to Report

1. **Email**: Send a detailed report to steel.sa.llc@gmail.com
2. **Subject Line**: Use the format `[SECURITY] Mercury Agent: Brief Description`
3. **PGP Encryption**: For sensitive reports, request our PGP public key

### What to Include

Please provide as much information as possible:

- **Description**: Clear explanation of the vulnerability
- **Impact**: Potential security impact and severity assessment
- **Steps to Reproduce**: Detailed reproduction steps
- **Affected Components**: Specific modules, files, or functions involved
- **Suggested Fix**: If you have recommendations for remediation
- **Environment**: Python version, OS, and relevant dependencies

### Response Timeline

- **Initial Response**: Within 48 hours
- **Assessment**: Within 7 days
- **Resolution Target**: Within 30 days for critical issues, 90 days for lower severity

### What to Expect

1. **Acknowledgment**: We will confirm receipt of your report
2. **Investigation**: Our security team will analyze the vulnerability
3. **Updates**: We will keep you informed of our progress
4. **Credit**: With your permission, we will credit you in our security advisories

## Security Features

### Code Security

Mercury Agent implements multiple layers of security:

- **Static Analysis**: Bandit security scanning in CI/CD pipeline
- **Dependency Scanning**: Regular audits with Safety and pip-audit
- **Type Safety**: Strict MyPy type checking enabled
- **Code Review**: All changes require security review

### Cryptographic Security

- **Post-Quantum Cryptography**: PQC backends for future-proof encryption
- **Secure Hash Functions**: SHA3-256 for integrity verification
- **Key Management**: Secure key generation and storage patterns

### Post-Quantum Cryptography (PQC) Backend Audit Status

Mercury Agent uses NIST-approved post-quantum cryptographic algorithms
sourced from AMA Cryptography v3.2.0 (pinned in
`pyproject.toml [project.optional-dependencies].pqc` and the
`AMA_REF` env var of `.github/workflows/ci.yml` /
`.github/workflows/pqc-production-check.yml`):

| Algorithm | Parameter set | FIPS standard | Mercury type |
|-----------|---------------|---------------|--------------|
| ML-KEM (Kyber) | ML-KEM-1024 (NIST L5) | FIPS 203 | `KyberKeyPair`, `KyberEncapsulation` |
| ML-DSA (Dilithium) | ML-DSA-65 (NIST L3, ctx-aware §5.2) | FIPS 204 | `DilithiumKeyPair` |
| SLH-DSA (SPHINCS+) | SLH-DSA-SHAKE-128s (NIST L1) | FIPS 205 | `SlhDsaKeyPair(param_set="SHAKE-128s")` |
| SLH-DSA (SPHINCS+) | SLH-DSA-SHA2-256f (NIST L5) | FIPS 205 | `SlhDsaKeyPair(param_set="SHA2-256f")` |
| Legacy SPHINCS+ | SPHINCS+-SHA2-256f-simple | (pre-FIPS-205 NIST round-3 name) | `SphincsKeyPair` |

The legacy `SphincsKeyPair` surface (Mercury's pre-v1.6.x SPHINCS+ entry
point) and the FIPS 205 `SlhDsaKeyPair` surface coexist: callers can
continue using the legacy `sphincs_sign`/`sphincs_verify` functions, or
upgrade to `slhdsa_sign`/`slhdsa_verify` for the parameter-driven FIPS
205 contract (with FIPS 205 §10.2 binding-context support). The
authoritative source-of-truth for the algorithm names exposed to
callers is `src/omni_mercury_engine/security/pqc_backends.py` (see the
`@dataclass DilithiumKeyPair / KyberKeyPair / SphincsKeyPair /
SlhDsaKeyPair` declarations).

| Backend | Status | Recommendation |
|---------|--------|----------------|
| AMA Cryptography (Native C, v3.2.0) | Community-tested, NOT externally audited | Production (sole backend — hard-required) |

**Important Security Considerations:**

1. **Algorithm vs Implementation**: The algorithms (ML-DSA-65, ML-KEM-1024, SLH-DSA) are NIST-approved (FIPS 203 / FIPS 204 / FIPS 205) and standardized. However, implementation correctness is NOT externally verified for the AMA Cryptography backend.

2. **Production Deployments**: For production deployments requiring compliance:
   - Obtain an independent security audit of the AMA Cryptography native C library
   - Consider FIPS 140-2 Level 3+ HSM for master secrets
   - Document risk acceptance for unaudited cryptographic code

3. **Sole Backend**: Mercury Agent **hard-requires** AMA Cryptography. There is no fallback chain — if AMA Cryptography is not installed, Mercury refuses to start. The native C library must be built for PQC algorithms:
   ```bash
   pip install "ama-cryptography @ git+https://github.com/Steel-SecAdv-LLC/AMA-Cryptography.git@v3.2.0"
   cmake -B build -DAMA_USE_NATIVE_PQC=ON && cmake --build build
   ```

4. **Universal Enforcement**: Mercury Agent refuses to run without real PQC cryptography at package import. `AMA_REQUIRE_REAL_PQC=true` is retained for legacy workflow readability, but the gate is no longer optional.

5. **Constant-Time Requirement**: AMA Cryptography's native C library provides constant-time implementations. Set `AMA_REQUIRE_CONSTANT_TIME=true` to enforce this at startup.

6. **HMAC routing (v1.7.x)**: AMA Cryptography v3.2.0 also surfaces
   ACVP-validated HMAC-SHA-256 / HMAC-SHA-512 bindings
   (`native_hmac_sha256`, `native_hmac_sha256_2`). Mercury's
   `native_jwt` module routes HS256 and HS512 through these bindings
   with no stdlib fallback; HS384 remains stdlib-only until AMA ships a
   SHA-384 HMAC binding. See
   `tests/security/test_native_jwt_ama_routing.py` for the RFC 4231 KAT
   and fail-closed route locks.

**KAT and ACVP evidence:**

- `tests/security/test_ama_kat.py` pins Ed25519 RFC 8032 §7.1 vectors,
  ML-DSA-65 round-trip, ML-KEM-1024 encaps/decaps round-trip,
  SPHINCS+ round-trip, and ML-DSA deterministic-signing reproducibility.
- `tests/security/test_nist_fips_kat.py` verifies bit-for-bit
  reproducibility against curated NIST ACVP-Server test vectors
  (FIPS 203, 204, 205): ML-DSA-65 deterministic sigGen, ML-KEM-1024
  decapsulation, SLH-DSA-SHAKE-128s sigGen. Source:
  [usnistgov/ACVP-Server](https://github.com/usnistgov/ACVP-Server).

**References:**
- [NIST PQC Standardization](https://csrc.nist.gov/projects/post-quantum-cryptography)
- [FIPS 203 — ML-KEM](https://csrc.nist.gov/pubs/fips/203/final)
- [FIPS 204 — ML-DSA](https://csrc.nist.gov/pubs/fips/204/final)
- [FIPS 205 — SLH-DSA](https://csrc.nist.gov/pubs/fips/205/final)
- [AMA Cryptography](https://github.com/Steel-SecAdv-LLC/AMA-Cryptography)
- [CRYSTALS-Dilithium (pre-FIPS reference)](https://pq-crystals.org/dilithium/)
- [CRYSTALS-Kyber (pre-FIPS reference)](https://pq-crystals.org/kyber/)
- [SPHINCS+ (pre-FIPS reference)](https://sphincs.org/)

### API Security

- **Authentication**: JWT-based authentication with secure token handling
- **Rate Limiting**: Built-in rate limiting to prevent abuse (100 req/min, burst of 20)
- **Input Validation**: Comprehensive input sanitization and validation
- **CORS Configuration**: Environment-aware cross-origin resource sharing
  - Production: cross-origin access requires explicit `MERCURY_CORS_ORIGINS` configuration; unset means same-origin only (fail-closed)
  - Development: Allows localhost origins by default
- **PII Masking**: Automatic redaction of sensitive data in logs
  - Email addresses, phone numbers, SSNs, credit cards
  - API keys, bearer tokens, IP addresses

### Data Protection

- **Encryption at Rest**: AES-256 encryption for sensitive data
- **Encryption in Transit**: TLS 1.3 for all network communications
- **Data Minimization**: Collection limited to necessary data only
- **Audit Logging**: Comprehensive logging for security events
- **Cryptographic Audit Trail**: Tamper-evident logging of all PQC operations
  - `CryptoAuditTrail` class for operation tracking
  - Thread-safe with configurable max entries (10,000 default)
  - Failure summary reporting for security analysis
  - `validate_pqc_environment()` for production readiness checks

## Security Best Practices for Users

### Deployment

1. **Environment Variables**: Store secrets in environment variables, never in code
2. **Network Security**: Deploy behind a firewall with proper network segmentation
3. **Container Security**: Use minimal base images and scan for vulnerabilities
4. **Access Control**: Implement principle of least privilege

### Configuration

```yaml
# Example secure configuration
security:
  encryption:
    enabled: true
    algorithm: AES-256-GCM
  authentication:
    require_auth: true
    token_expiry: 3600
  rate_limiting:
    enabled: true
    requests_per_minute: 100
  logging:
    security_events: true
    pii_masking: true
```

### Updates

- **Regular Updates**: Keep Mercury Agent and dependencies updated
- **Security Advisories**: Watch the GitHub repository and its Security Advisories page
- **Vulnerability Monitoring**: Use tools like Dependabot or Snyk

## Ethical Security Considerations

Mercury Agent includes security intelligence capabilities. Users must:

1. **Legal Compliance**: Ensure all usage complies with applicable laws
2. **Authorization**: Obtain proper authorization before security testing
3. **Data Privacy**: Handle any personal data in accordance with privacy regulations
4. **Responsible Use**: Use security features only for legitimate purposes

## Known Security Considerations

### Third-Party Dependencies

- **PyTorch**: GPU acceleration library with its own security considerations
- **NumPy/SciPy**: Numerical libraries - keep updated for security patches
- **FastAPI**: Web framework - follow OWASP guidelines for API security

### Machine Learning Security

- **Model Integrity**: Verify model checksums before loading
- **Adversarial Inputs**: Be aware of potential adversarial attacks on ML models
- **Training Data**: Ensure training data is from trusted sources

### Autonomous Self-Modification (Governed, Fail-Closed)

**Threat.** A recursively self-improving agent that can change its own live
decision boundary without evidence or review is a security risk in its own
right: a poisoned feedback stream could drive the reflexion critic to move the
operating threshold into a blind spot, or drive the online-learning pipeline to
retrain on adversarial drift. Mercury contains both arrows — reflexion threshold
adaptation (`agentic/orchestration.py`) and drift-/performance-triggered
recalibration (`ml/online_learning.py`).

**Control (Phase 3 governed self-improvement).** Neither arrow may mutate live
behaviour autonomously. Each proposed change is routed through an engine-owned,
fail-closed governance seam (`omni_mercury_engine.governance.self_improvement`)
before it can take effect:

- The **default policy withholds every autonomous change.** The live operating
  point or model only changes through an evidence-backed, human-approved
  promotion executed out of band.
- The **gate-backed policy** routes a proposal through the Phase 2 promotion gate
  (`research/governed_fusion/promotion_gate.py`), which re-checks the σ_Immutable,
  benevolence, conformal-coverage, and Lyapunov floors and the external-label
  fitness bucket. A gate `promote` is **queued for human approval**, never
  auto-applied.
- An adversarial or malformed proposal therefore fails closed: with no held-out
  promotion evidence it is rejected; with a regressed metric it is rejected; with
  a tripped safety floor it is rejected.
- The only path that applies an autonomous change is an **explicit, named**
  `MeasurementGovernance` used solely by offline held-out measurement harnesses —
  it is never the production default, so any adapting context is auditable as a
  measurement.

Every disposition is recorded to an append-only audit trail. See
[ARCHITECTURE.md](ARCHITECTURE.md) ("Governed Self-Improvement Seam") and
[docs/PHASE3_GOVERNANCE.md](docs/PHASE3_GOVERNANCE.md) for the full contract.

## Current Vulnerability Status

*Last Review: 2026-06-10*

### Accepted Vulnerabilities (with Mitigations)

The machine-enforced accepted-risk ledger is [`.trivyignore`](.trivyignore). The blocking deployment-image Trivy gates (`.github/workflows/ci.yml` Docker job and `.github/workflows/security.yml`, both `severity: CRITICAL,HIGH` / `exit-code: 1`) run with `ignore-unfixed: false` plus that file, so:

- every acceptance is **enumerated** (one CVE per line), **expiring** (`exp:` dates at the 90-day review cadence — an expired entry fails the gate until re-reviewed), and reviewable in one place;
- a **new** unfixed CRITICAL/HIGH finding not in the ledger **blocks the merge** — the previous blanket `ignore-unfixed: true` posture, which waived the entire unfixed-OS-CVE class invisibly, is retired;
- fixable CVEs never enter the ledger: the Dockerfile's runtime-stage `apt-get upgrade -y` and the repo-wide `pip >= 26.1` floor are the fix path, and a fixable CRITICAL/HIGH finding fails the gate.

The only path-level skips are the eight `skip-files` entries in those workflows — vendored torch/tensorflow headers and the scipy/skimage dataset-registry files that Trivy's secret scanner misclassifies — documented in the CHANGELOG security entries.

Current accepted-risk posture (as measured by the blocking gates' own
built-image scan — trivy 0.70.0 via `aquasecurity/trivy-action@v0.36.0`,
2026-06-10, with a 2026-06-13 follow-up review; base-image enumeration
cross-checked with a local trivy 0.71.0 scan the same day; entries expire
2026-09-08):

- **Total accepted:** 14 CVEs — **Critical:** 4, **High:** 10
- All are Debian-bookworm OS packages with **no upstream fix available**; none sits on an untrusted-input path in the shipped API; the container runs as non-root UID 1000 with SUID/SGID bits stripped
- These are genuinely irreducible, not deferred: `libsqlite3-0`, `libexpat1`, `zlib1g` and `ncurses` (`libtinfo6`/`libncursesw6`) are linked by CPython itself (the `sqlite3`, `pyexpat`, `zlib`, and `readline`/`curses` stdlib modules), so they cannot be removed without removing the interpreter, and Debian ships no patched build to `apt-get upgrade` to; `perl-base` is pulled by apt's own `adduser` dependency. The one genuinely removable package — the Mesa GL stack — was **eliminated outright** rather than accepted (see below)

| CVE | Severity | Component | Status | Mitigation |
|-----|----------|-----------|--------|------------|
| CVE-2023-45853 | Critical | zlib/minizip | No upstream fix (will_not_fix) | No minizip usage in any Mercury code path |
| CVE-2025-7458 | Critical | SQLite (libsqlite3-0) | No upstream fix | No SQLite-backed feature ships by default; non-root execution |
| CVE-2026-11822 | High | SQLite (libsqlite3-0) | No upstream fix (affected; fixed upstream only in SQLite ≥ 3.53.2) | FTS5 full-text-search memory corruption, reachable only by opening an attacker-crafted database through FTS5; no SQLite-backed feature ships by default and Mercury never uses FTS5 |
| CVE-2026-11824 | High | SQLite (libsqlite3-0) | No upstream fix (affected; fixed upstream only in SQLite ≥ 3.53.2) | Same FTS5 attacker-crafted-database surface as CVE-2026-11822; not on any shipped-API input path |
| CVE-2026-8376 | Critical | perl-base | No upstream fix | Container executes no Perl; pulled in by Debian essential tooling |
| CVE-2026-42496 | Critical | perl-base | No upstream fix (fix_deferred) | Container executes no Perl |
| CVE-2025-69720 | High | ncurses (libncursesw6 et al.) | No upstream fix | Terminal handling only; never exposed to untrusted input |
| CVE-2026-42497 | High | perl-base | No upstream fix (fix_deferred) | Container executes no Perl |
| CVE-2026-48959 | High | perl-base | No upstream fix | Container executes no Perl |
| CVE-2026-48962 | High | perl-base | No upstream fix | Container executes no Perl |
| CVE-2026-9538 | High | perl-base | No upstream fix (fix_deferred) | Container executes no Perl |
| CVE-2025-59375 | High | libexpat1 | No upstream fix (will_not_fix) | Only XML parse path (CAP alert validation, `alerting/cap_generator.py`) is defusedxml-hardened and not exposed by any API route |
| CVE-2026-25210 | High | libexpat1 | No upstream fix (affected) | Same defusedxml-hardened, non-API XML surface |
| CVE-2026-45186 | High | libexpat1 | No upstream fix (affected) | Same defusedxml-hardened, non-API XML surface |

**Added at the 2026-06-10 review from the first enforced built-image gate run:** the three libexpat1 CVEs surface only in the built image — the gates' own scan of it is the canonical enumeration source for the ledger, which the original base-image enumeration could not cover. (The mesa CVE first enumerated at this review has since been remediated by removal — see the 2026-06-15 note below.)

**Added at the 2026-06-13 review:** the SQLite FTS5 pair CVE-2026-11822 / CVE-2026-11824 — newly published memory-corruption bugs (fixed upstream only in SQLite ≥ 3.53.2, no Debian bookworm fix) that a vuln-DB update introduced into the gate scan on the unchanged base image. They share `libsqlite3-0` and the same non-untrusted-input rationale as the existing CVE-2025-7458 acceptance, and a built-image scan confirmed they are the only new findings (no fixable CVE and no library CVE appeared).

**Resolved and removed from the ledger (2026-06-15 review):** the Mesa GL stack (CVE-2026-40393, Critical) was **eliminated rather than accepted**. It was installed only as OpenCV's `libGL` import dependency, but Mercury depends on `opencv-python-headless` — whose `cv2` extension links no `libGL` (verified: the wheel's `cv2.*.so` shows zero GL linkage) — and makes no `cv2` GUI calls, so the Dockerfile no longer installs `libgl1-mesa-glx`. The package and its CVE are gone from the image; the blocking Trivy gate (`ignore-unfixed: false`) re-verifies this on every build, failing loudly if the package ever reappears. Accepted count: 15 → 14 (Critical 5 → 4).

**Resolved and removed from the ledger (2026-06-10 review):** the pip CVE family (CVE-2025-8869, CVE-2026-1703, CVE-2026-6357) is fixed repo-wide by the `pip >= 26.1` floor and gated by `tests/security/test_cve_2026_6357_regression.py`; the formerly-listed gpgv (CVE-2025-68973), libglib2.0-0 (CVE-2025-13601), linux-pam (CVE-2025-6020), util-linux (CVE-2025-14104), and SQLite FTS5 (CVE-2025-7709) findings no longer appear at the gated severities in the current base image. The seven newly listed entries (ncurses + six perl-base) were present but invisible under the old blanket waiver — disclosed here for the first time.

### Vulnerability Assessment Process

All vulnerabilities undergo the following assessment:

1. **Severity Analysis**: CVSS score and attack vector evaluation
2. **Applicability Check**: Does the vulnerability affect Mercury Agent's use case?
3. **Mitigation Review**: What controls are in place to reduce risk?
4. **Documentation**: Justification recorded in [`.trivyignore`](.trivyignore) (machine-enforced) and the table above
5. **Quarterly Review**: Enforced mechanically — every ledger entry carries a 90-day `exp:` date and the gate fails on expiry until re-reviewed

### Container Security Hardening

Mercury Agent's Docker container implements defense-in-depth:

- **Multi-stage Build**: Separates build and runtime environments
- **Non-root User**: Application runs as `mercuryagent` (UID 1000)
- **SUID/SGID Removal**: All setuid/setgid bits removed from binaries
- **Minimal Packages**: Only essential runtime dependencies installed
- **Clean Filesystem**: No package caches or temporary files retained

### Unresolved Vulnerabilities

Accepted risks are re-reviewed at most every 90 days, enforced by the `exp:` dates in [`.trivyignore`](.trivyignore). As of the 2026-06-15 review, documented acceptances are 14 CVEs (4 Critical, 10 High), all no-upstream-fix Debian packages in the deployment image, none on an untrusted-input path in the shipped API. The ledger file and the table above are the complete record.

### Two-Tier Dependency-CVE Coverage

Mercury Agent runs **two complementary CVE gates** on every PR:

| Tier | Tool | Scope | Source of truth |
|------|------|-------|-----------------|
| Python-package | `safety check` + `pip-audit` | Editable install (`pip install -e ".[api]"`) | `.safety-policy-v2.yml` + [CHANGELOG.md](CHANGELOG.md) (per-CVE rationale tracked under the dated security entries) |
| Deployment-image | Trivy | Built Docker image (full runtime + OS) | [`.trivyignore`](.trivyignore) (enumerated, expiring acceptances) + the gate configuration in `.github/workflows/{ci,security}.yml` |

Both gates must be GREEN for any PR to merge. The Python-package gate
runs with **zero risk acceptance**: the policy files are no-op shims
and no `--ignore` / `--ignore-vuln` flags are wired into either CI
workflow. Findings are remediated by upgrade, isolation, or native
re-implementation — see the CHANGELOG entries dated 2026-05-20
("Permanent supply-chain remediations") for the current
remediation ledger. The deployment-image gate blocks every fixable
CRITICAL/HIGH finding and every unfixed finding not enumerated in
[`.trivyignore`](.trivyignore) (13 CVEs, each with a 90-day expiry that
fails the gate until re-reviewed).

## Security Assessment Posture

Mercury Agent's security analysis is automated and self-assessed:

- **Automated Scanning**: Security scans run on every pull request and push (bandit, safety, pip-audit, semgrep, Trivy) plus a weekly scheduled run (`.github/workflows/security.yml`, Sundays 00:00 UTC)
- **Dependency Audits**: Dependabot weekly update checks plus the two-tier CVE gates described below
- **Code Reviews**: All changes require human review before merge

Mercury Agent has **not** been externally audited or penetration-tested. Production deployments requiring assurance beyond self-assessment must commission an independent security review (see "Important Security Considerations" above and the README status line: Research-grade | Community-tested | Not externally audited).

## Compliance

Mercury Agent is designed with compliance in mind:

- **OWASP**: Follows OWASP Top 10 security guidelines
- **CWE**: Addresses Common Weakness Enumeration patterns
- **NIST CSF 2.0**: First-party integrator at
  `omni_mercury_engine.compliance.nist_csf_integrator` covers all six
  core functions (GOVERN, IDENTIFY, PROTECT, DETECT, RESPOND, RECOVER),
  22 categories, and 106+ subcategories. The fetcher hits the live
  NIST CSRC reference endpoint with a 7-day on-disk cache so
  assessments reflect the authoritative subcategory tree, not a
  hard-coded snapshot. See
  [`docs/COMPLIANCE.md`](docs/COMPLIANCE.md).
- **FIRST.org / CISA TLP 2.0**: First-party handler at
  `omni_mercury_engine.compliance.tlp_handler` implements the full
  five-colour ladder (CLEAR / GREEN / AMBER / AMBER+STRICT / RED) for
  every Mercury output, including watermarking and export metadata.
- **OSHA / eCFR**: First-party detector at
  `omni_mercury_engine.compliance.osha_anomaly` covers 12 hazard
  categories × 6 industry sectors with CFR citations and a NWS
  Rothfusz heat-index regression (the upstream simplified
  ``T + 0.5·RH`` heuristic over-reported by ~8 °F at high humidity and
  under-reported at low humidity — both directions removed in the
  port).

### v1.7 hard-gate boundary contract

As of the v1.7 development cycle every public boundary surface
(`OmniMercuryEngine.detect_with_fusion[_calibrated]`,
`CognitiveOrchestrator.analyze`, `NeuroSymbolicHub.predict`) runs **two
independent mandatory hard ethical gates** — Benevolence (>= 0.99) and
σ_Immutable (256-D scalar network) — and raises
`EthicalConstraintViolationError(check=…)` on failure. There is no
advisory mode. The reserved `check=` codes are `"benevolence"`,
`"sigma_immutable"`, and `"gosnn_unavailable"`. See
[`ARCHITECTURE.md`](ARCHITECTURE.md) §"Dual-Gate Hard Ethical
Enforcement" and [`docs/MIGRATION-1.6-to-1.7.md`](docs/MIGRATION-1.6-to-1.7.md) §2.

### Production-mode primitive (`MERCURY_ENV`)

`omni_mercury_engine._env` exposes the canonical environment-mode flag
`MERCURY_ENV` (`development` default, `production`) plus shared
fail-closed helpers (`get_mercury_env`, `is_production`,
`require_real_component`, `MercuryProductionConfigError`). The flag is
orthogonal to `AMA_REQUIRE_REAL_PQC`; production deployments typically
set both. An unknown value (e.g. `MERCURY_ENV=prod`) raises
`MercuryProductionConfigError` at first read, by design — typos in
deployment configuration must be loud.

## Contact

- **Security Team**: steel.sa.llc@gmail.com
- **General Support**: steel.sa.llc@gmail.com
- **GitHub Issues**: For non-sensitive security discussions

## Acknowledgments

We thank the security researchers who have helped improve Mercury Agent's security. Contributors will be acknowledged here with their permission.

---

*Last Updated: 2026-06-10*
*Version: 1.7.0*
