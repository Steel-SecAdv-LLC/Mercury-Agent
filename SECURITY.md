# Security Policy

## Mercury Agent Security Framework

Mercury Agent is a security-focused AI framework developed by Steel Security Advisors LLC. We take security seriously and are committed to maintaining the integrity, confidentiality, and availability of our systems and user data.

## Supported Versions

We provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.5.x   | :white_check_mark: |
| 1.4.x   | :white_check_mark: |
| 1.2.x   | :white_check_mark: |
| 1.1.x   | :white_check_mark: |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

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

Mercury Agent uses NIST-approved post-quantum cryptographic algorithms (ML-DSA-65/Dilithium, Kyber-1024, SPHINCS+-256f) for quantum-resistant security. The following table documents the audit status of available PQC backends:

| Backend | Status | Recommendation |
|---------|--------|----------------|
| AMA Cryptography (Native C) | Community-tested, NOT externally audited | Production (sole backend — hard-required) |

**Important Security Considerations:**

1. **Algorithm vs Implementation**: The algorithms (ML-DSA-65, Kyber-1024, SPHINCS+) are NIST-approved and standardized. However, implementation correctness is NOT externally verified for the AMA Cryptography backend.

2. **Production Deployments**: For production deployments requiring compliance:
   - Obtain an independent security audit of the AMA Cryptography native C library
   - Consider FIPS 140-2 Level 3+ HSM for master secrets
   - Document risk acceptance for unaudited cryptographic code

3. **Sole Backend**: Mercury Agent **hard-requires** AMA Cryptography. There is no fallback chain — if AMA Cryptography is not installed, Mercury refuses to start. The native C library must be built for PQC algorithms:
   ```bash
   pip install "ama-cryptography @ git+https://github.com/Steel-SecAdv-LLC/AMA-Cryptography.git"
   cmake -B build -DAMA_USE_NATIVE_PQC=ON && cmake --build build
   ```

4. **Production Enforcement**: Mercury Agent refuses to run without real PQC cryptography in production. Set `AMA_REQUIRE_REAL_PQC=true` to enforce native PQC availability at startup.

5. **Constant-Time Requirement**: AMA Cryptography's native C library provides constant-time implementations. Set `AMA_REQUIRE_CONSTANT_TIME=true` to enforce this at startup.

**References:**
- [NIST PQC Standardization](https://csrc.nist.gov/projects/post-quantum-cryptography)
- [AMA Cryptography](https://github.com/Steel-SecAdv-LLC/AMA-Cryptography)
- [Dilithium (CRYSTALS)](https://pq-crystals.org/dilithium/)
- [Kyber (CRYSTALS)](https://pq-crystals.org/kyber/)

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

*Last Scan: February 2026*

### Accepted Vulnerabilities (with Mitigations)

The following vulnerabilities have been assessed and accepted with documented mitigations:

| CVE | Severity | Component | Status | Mitigation |
|-----|----------|-----------|--------|------------|
| CVE-2025-14104 | Medium | util-linux | Accepted | Non-root execution, SUID bits removed, no passwd operations |
| CVE-2025-8869 | Medium | pip | Accepted | pip >=25.0, HTTPS-only, no runtime pip install |
| CVE-2025-7709 | Medium | SQLite | Accepted | No FTS5 usage, non-root execution |

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
- **Package Purge**: `login` and `passwd` packages removed from runtime
- **Clean Filesystem**: No package caches or temporary files retained

### Unresolved Vulnerabilities

As of the last scan, there are **0 high/critical** and **3 medium** severity vulnerabilities, all with documented mitigations and acceptance justifications. See `.trivyignore` for complete details.

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
- **NIST**: Aligned with NIST Cybersecurity Framework

## Contact

- **Security Team**: steel.sa.llc@gmail.com
- **General Support**: steel.sa.llc@gmail.com
- **GitHub Issues**: For non-sensitive security discussions

## Acknowledgments

We thank the security researchers who have helped improve Mercury Agent's security. Contributors will be acknowledged here with their permission.

---

**AI Co-Architects:** Eris ✠ | Eden ♱ | Devin ⚛︎ | Claude ⊛

*Last Updated: 2026-03-11*
*Version: 1.5.1*
