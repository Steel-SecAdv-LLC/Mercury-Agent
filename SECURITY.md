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

4. **Production Enforcement**: Mercury Agent refuses to run without real PQC cryptography in production. Set `AMA_REQUIRE_REAL_PQC=true` to enforce native PQC availability at startup.

5. **Constant-Time Requirement**: AMA Cryptography's native C library provides constant-time implementations. Set `AMA_REQUIRE_CONSTANT_TIME=true` to enforce this at startup.

6. **HMAC routing (v1.7.x)**: AMA Cryptography v3.2.0 also surfaces
   ACVP-validated HMAC-SHA-256 / HMAC-SHA-512 bindings
   (`native_hmac_sha256`, `native_hmac_sha256_2`). Mercury's
   `native_jwt` module routes HS256 and HS512 through these bindings
   when available, falling back transparently to the stdlib `hmac`
   path otherwise. See `tests/security/test_native_jwt_ama_routing.py`
   for the RFC 4231 KAT + stdlib byte-equivalence + interoperability
   invariants.

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
  - Production: Requires explicit `MERCURY_CORS_ORIGINS` configuration
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
- **Security Advisories**: Subscribe to our security mailing list
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

## Current Vulnerability Status

*Last Review: 2026-05-19*

### Accepted Vulnerabilities (with Mitigations)

Accepted-risk records are maintained in [`.trivyignore`](.trivyignore), which is the source of truth for CVE rationale, mitigation notes, and review/expiry metadata.

Current accepted-risk posture documented there:

- **Total accepted:** 10 CVEs
- **Critical:** 2
- **High:** 3
- **Medium:** 4
- **Low:** 1
- Includes fixed pip CVEs retained for audit continuity

| CVE | Severity | Component | Status | Mitigation |
|-----|----------|-----------|--------|------------|
| CVE-2025-7458 | Critical | SQLite | Accepted | No SQLite DB usage, non-root execution, input validation |
| CVE-2023-45853 | Critical | zlib/minizip | Accepted | No minizip usage, Debian will_not_fix context |
| CVE-2025-68973 | High | gpgv | Accepted | Not used by app paths, no runtime package installs |
| CVE-2025-13601 | High | libglib2.0-0 | Accepted | No direct glib URI handling, non-root execution |
| CVE-2025-6020 | High | linux-pam | Accepted | JWT auth path, non-root execution, SUID/SGID stripped |
| CVE-2025-14104 | Medium | util-linux | Accepted | Non-root execution, SUID/SGID bits stripped |
| CVE-2025-8869 | Medium | pip | Accepted (fixed; retained for audit continuity) | pip >=26.1, HTTPS-only, no runtime pip install |
| CVE-2025-7709 | Medium | SQLite | Accepted | No FTS5 usage, non-root execution |
| CVE-2026-6357 | Medium | pip | Accepted (fixed; retained for audit continuity) | pip >=26.1, no runtime pip install |
| CVE-2026-1703 | Low | pip | Accepted (fixed; retained for audit continuity) | pip >=26.1, trusted package sources only |

### Vulnerability Assessment Process

All vulnerabilities undergo the following assessment:

1. **Severity Analysis**: CVSS score and attack vector evaluation
2. **Applicability Check**: Does the vulnerability affect Mercury Agent's use case?
3. **Mitigation Review**: What controls are in place to reduce risk?
4. **Documentation**: Full justification recorded in `.trivyignore`
5. **Quarterly Review**: Re-evaluate accepted risks every 90 days

### Container Security Hardening

Mercury Agent's Docker container implements defense-in-depth:

- **Multi-stage Build**: Separates build and runtime environments
- **Non-root User**: Application runs as `mercuryagent` (UID 1000)
- **SUID/SGID Removal**: All setuid/setgid bits removed from binaries
- **Minimal Packages**: Only essential runtime dependencies installed
- **Clean Filesystem**: No package caches or temporary files retained

### Unresolved Vulnerabilities

Accepted risks are reviewed quarterly. As of the 2026-05-19 review, documented acceptances are 10 CVEs (2 Critical, 3 High, 4 Medium, 1 Low), including fixed pip CVEs retained for audit continuity. See [`.trivyignore`](.trivyignore) for complete details.

### Two-Tier Dependency-CVE Coverage

Mercury Agent runs **two complementary CVE gates** on every PR:

| Tier | Tool | Scope | Source of truth |
|------|------|-------|-----------------|
| Python-package | `safety check` (v3.7.0) + `pip-audit` (v2.10.0) | Editable install (`pip install -e ".[api]"`) | `.safety-policy-v2.yml` + `[CHANGELOG.md](CHANGELOG.md)` (per-CVE rationale tracked under the dated security entries) |
| Deployment-image | Trivy | Built Docker image (full runtime + OS) | [`.trivyignore`](.trivyignore) |

Both gates must be GREEN for any PR to merge. The Python-package gate
runs with **zero risk acceptance**: the policy files are no-op shims
and no `--ignore` / `--ignore-vuln` flags are wired into either CI
workflow. Findings are remediated by upgrade, isolation, or native
re-implementation — see the CHANGELOG entries dated 2026-05-20
("Permanent supply-chain remediations") for the current
remediation ledger. The deployment-image gate honours the per-CVE
acceptances enumerated in `.trivyignore` (10 CVEs, all reviewed
quarterly).

## Security Audits

Mercury Agent undergoes regular security assessments:

- **Automated Scanning**: Daily CI/CD security scans
- **Dependency Audits**: Weekly dependency vulnerability checks
- **Code Reviews**: Continuous security-focused code review
- **Penetration Testing**: Periodic external security assessments

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

*Last Updated: 2026-05-22*
*Version: 1.7.0*
