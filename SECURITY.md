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
| 2.1.x   | **Current**          | :white_check_mark: |
| 1.7.x   | Previous (EOL on next minor) | Critical CVEs only |
| < 1.7   | End-of-life          | :x:              |

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
sourced from AMA Cryptography v4.0.0 (pinned in
`pyproject.toml [project.optional-dependencies].pqc` and via the
`ama-ref: v4.0.0` input that `.github/workflows/ci.yml` /
`.github/workflows/pqc-production-check.yml` pass to the
`build-ama-cryptography` composite action, which exports it internally
as `AMA_REF`):

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
| AMA Cryptography (Native C, v4.0.0) | Community-tested, NOT externally audited | Production (sole backend — hard-required) |

**Important Security Considerations:**

1. **Algorithm vs Implementation**: The algorithms (ML-DSA-65, ML-KEM-1024, SLH-DSA) are NIST-approved (FIPS 203 / FIPS 204 / FIPS 205) and standardized. However, implementation correctness is NOT externally verified for the AMA Cryptography backend.

2. **Production Deployments**: For production deployments requiring compliance:
   - Obtain an independent security audit of the AMA Cryptography native C library
   - Consider FIPS 140-2 Level 3+ HSM for master secrets
   - Document risk acceptance for unaudited cryptographic code

3. **Sole Backend**: Mercury Agent **hard-requires** AMA Cryptography. There is no fallback chain — if AMA Cryptography is not installed, Mercury refuses to start. The native C library must be built for PQC algorithms:
   ```bash
   pip install "ama-cryptography @ git+https://github.com/Steel-SecAdv-LLC/AMA-Cryptography.git@v4.0.0"
   cmake -B build -DAMA_USE_NATIVE_PQC=ON && cmake --build build
   ```

4. **Universal Enforcement**: Mercury Agent refuses to run without real PQC cryptography at package import. `AMA_REQUIRE_REAL_PQC=true` is retained for legacy workflow readability, but the gate is no longer optional.

5. **Constant-Time Operation**: AMA Cryptography v4.0.0 enforces native-only operation unconditionally (INVARIANT-7 revised), so its constant-time C primitives are always in use — no configuration is needed or possible. `AMA_REQUIRE_CONSTANT_TIME` is a superseded compatibility flag: setting it changes no cryptographic behavior on a healthy install (AMA logs a deprecation warning), it redundantly fails closed on an install without the native backend (which Mercury's import-time PQC gate already refuses), and Mercury reads it only for diagnostics (`security.pqc_backends.require_constant_time()`, surfaced by `get_pqc_capabilities()` / `validate_pqc_environment()`). Leave it unset.

6. **HMAC routing (v1.7.x)**: AMA Cryptography v4.0.0 also surfaces
   ACVP-validated HMAC-SHA-256 / HMAC-SHA-384 / HMAC-SHA-512 bindings
   (`native_hmac_sha256`, `native_hmac_sha256_2`, `native_hmac_sha384`,
   `native_hmac_sha512`). Mercury's `native_jwt` module routes HS256,
   HS384, and HS512 through these bindings with no stdlib fallback
   (fail-closed). See
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

*Last Review: 2026-07-09*

### Accepted Vulnerabilities (with Mitigations)

The machine-enforced accepted-risk ledger is [`.trivyignore`](.trivyignore). The blocking deployment-image Trivy gates (`.github/workflows/ci.yml` Docker job and `.github/workflows/security.yml`, both `severity: CRITICAL,HIGH` / `exit-code: 1`) run with `ignore-unfixed: false` plus that file, so:

- every acceptance is **enumerated** (one CVE per line), **expiring** (`exp:` dates at the 90-day review cadence — an expired entry fails the gate until re-reviewed), and reviewable in one place;
- a **new** unfixed CRITICAL/HIGH finding not in the ledger **blocks the merge** — the previous blanket `ignore-unfixed: true` posture, which waived the entire unfixed-OS-CVE class invisibly, is retired;
- fixable CVEs never enter the ledger: the Dockerfile's runtime-stage `apt-get upgrade -y` and the repo-wide `pip >= 26.1` floor are the fix path, and a fixable CRITICAL/HIGH finding fails the gate.

The only path-level skips are the eight `skip-files` entries in those workflows — vendored torch/tensorflow headers and the scipy/skimage dataset-registry files that Trivy's secret scanner misclassifies — documented in the CHANGELOG security entries.

Current accepted-risk posture (as measured by the blocking gates' own
built-image scan — trivy 0.70.0 via `aquasecurity/trivy-action@v0.36.0`,
the version the enforcing workflows run), **re-enumerated 2026-06-30
against the shipped Debian trixie base** (`python:3.14-slim-trixie`,
Debian 13.5) after the Python 3.13 → 3.14 base-image bump, and
**re-checked 2026-07-02** after the vulnerability DB published four new
unfixed CRITICAL/HIGH findings (gzip, glib — both eliminated from the
image, not accepted; see below), and **re-reviewed 2026-07-09** after
the vulnerability DB published CVE-2026-53615 against `util-linux`
(added to the ledger, not eliminated — see below); every retained entry
cross-checked against the Debian Security Tracker (trixie status "open",
no fixed version):

- **Total accepted:** 4 CVEs — **Critical:** 0, **High:** 4
- All are Debian-**trixie** OS packages with **no upstream fix available**; none sits on an untrusted-input path in the shipped API; the container runs as non-root UID 1000 with SUID/SGID bits stripped
- All are genuinely irreducible: `ncurses` (`libtinfo6`/`libncursesw6`/`ncurses-base`/`ncurses-bin`) is linked by CPython itself (the `readline`/`curses` stdlib modules), `libacl1`/`libattr1` back the base image's own `coreutils`/`tar` toolchain, and the `util-linux` family (`mount`/`login` back the base image; `libuuid1`/`libblkid1` back CPython and `coreutils`) is Debian-essential — the packages `apt-get upgrade` itself depends on — so none can be removed without breaking the interpreter or the image's update path. The genuinely removable packages were **eliminated outright** rather than accepted: `perl-base` — which carried both former CRITICALs — is purged in the Dockerfile (together with its `adduser` consumer), `gzip` and `libglib2.0-0` followed on 2026-07-02, as did the Mesa GL stack before them (see below)

| CVE | Severity | Component (trixie version) | Status | Mitigation |
|-----|----------|-----------|--------|------------|
| CVE-2025-69720 | High | ncurses (libncursesw6 et al. 6.5+20250216-2) | trixie open, no fix | Terminal handling only; never exposed to untrusted input. Linked by CPython's `readline`/`_curses` stdlib modules (and `libtinfo` backs the shell). Expires 2026-09-16 |
| CVE-2026-54369 | High | acl/attr (libacl1 2.3.2-2+b1, libattr1 1:2.5.2-3) | trixie open, no fix (fixed upstream only in acl ≥ 2.4.0 / attr ≥ 2.6.0) | Local-only symlink-traversal privilege escalation in pathname-based ACL/xattr APIs, reachable only via a *privileged* caller traversing an attacker-controlled path component. Container is non-root UID 1000, SUID/SGID stripped, and never invokes the acl/attr CLIs or APIs. Retained because Debian's `coreutils`/`tar` are built against them and back `apt-get upgrade` itself. Expires 2026-09-28 |
| CVE-2026-54371 | High | acl/attr (libacl1 2.3.2-2+b1, libattr1 1:2.5.2-3) | trixie open, no fix (fixed upstream only in acl ≥ 2.4.0 / attr ≥ 2.6.0) | Same surface and mitigation as CVE-2026-54369. Expires 2026-09-28 |
| CVE-2026-53615 | High | util-linux (bsdutils/libblkid1/libmount1/libuuid1/login/mount/util-linux 2.41-5) | trixie open, no fix | Integer overflow in `libblkid`'s MS-DOS partition-table parser (`libblkid/src/partitions/dos.c`), reachable only when a *privileged* caller probes an attacker-crafted block device / disk image; the container runs non-root UID 1000, SUID/SGID stripped, and never probes block devices. `util-linux` is Debian-essential (`mount`/`login` back the base image; `libuuid1`/`libblkid1` back CPython and `coreutils`). Expires 2026-10-07 |

**Trixie re-enumeration + perl-base elimination (2026-06-18).** The deployment image migrated from `python:3.13-slim-bookworm` to `python:3.13-slim-trixie` (Debian 13.5). The ledger was rebuilt from first principles: the runtime image's OS layer was built and scanned with the gate's own trivy 0.70.0, and each finding cross-checked against the Debian Security Tracker. Six bookworm-era acceptances were **dropped, not carried inert**, because they are gone or no longer CRITICAL/HIGH on trixie — CVE-2023-45853 (zlib, resolved in trixie `1:1.3.dfsg-2`) and CVE-2025-7458 (SQLite, resolved in trixie `3.42.0-1`) are fixed by trixie's newer packages; CVE-2025-59375 / CVE-2026-25210 / CVE-2026-45186 (expat) drop because `libexpat1` is not an installed dpkg package in the trixie image, so trivy reports no OS-level expat finding (the Python `pyexpat` path remains defusedxml-hardened); and CVE-2026-48959 (perl-base) is no longer a CRITICAL/HIGH finding under the current vuln DB. That left 8. Then **`perl-base` was eliminated, not accepted**: it is a Debian-essential package carrying 5 of those 8 — both CRITICALs (CVE-2026-8376, CVE-2026-42496) plus CVE-2026-42497 / CVE-2026-48962 / CVE-2026-9538 — but nothing in the runtime needs Perl, so the Dockerfile purges `perl-base` and its `adduser` consumer right after the apt upgrade (verified on the built image: the interpreter, `sqlite3`, `pip` and `useradd` all work without it, and trivy then reports the 5 CVEs gone). Accepted count: **14 → 8 → 3 (Critical 4 → 2 → 0, High 10 → 6 → 3)**. The three retained entries each suppress a real, currently-present trixie finding (verified 1:1 against the built-image scan — no inert entries).

**Mesa GL stack — eliminated, not accepted (carried forward).** CVE-2026-40393 was installed only as OpenCV's `libGL` import dependency, but Mercury depends on `opencv-python-headless` — whose `cv2` extension links no `libGL` (verified: the wheel's `cv2.*.so` shows zero GL linkage) — and makes no `cv2` GUI calls, so the Dockerfile no longer installs `libgl1-mesa-glx`. The package and its CVE are absent from the image; the blocking Trivy gate (`ignore-unfixed: false`) re-verifies this on every build, failing loudly if the package ever reappears.

**3.14 re-enumeration (2026-06-30).** After the `python:3.13-slim-trixie` → `python:3.14-slim-trixie` base-image bump, the ledger was re-enumerated with the gate's own trivy 0.70.0. The SQLite FTS5 pair (CVE-2026-11822 / CVE-2026-11824) was re-scored by the upstream vendor from HIGH to MEDIUM — still present in `libsqlite3-0 3.46.1-7+deb13u1` with no upstream fix, but out of scope for the CRITICAL/HIGH blocking gate, so the entries were dropped (to be re-added, not silently ignored, should a future re-scoring raise them back). The unfixed acl/attr pair (CVE-2026-54369 / CVE-2026-54371) entered the ledger with the rationale in the table above. Accepted count: 3 → 3 (Critical 0, High 3); the subsequent 2026-07-09 `util-linux` addition (below) raised it to 4 (Critical 0, High 4).

**gzip + glib — eliminated, not accepted (2026-07-02).** The vulnerability DB published four new unfixed CRITICAL/HIGH findings against the image: `gzip` CVE-2026-41992 (High — LZH-decompression buffer overflow) and `libglib2.0-0t64` CVE-2026-58016 (Critical) / CVE-2026-58014 / CVE-2026-58015 (High), none with a trixie fix. Both packages turned out to be removable rather than acceptable: `libglib2.0-0` was carried only as cv2's historical `libgthread-2.0` import dependency, but the shipped `opencv-python-headless` wheel (≥ 4.13) vendors its media stack and links no glib library at all (verified via `readelf` on the wheel's `cv2.abi3.so` and a cv2 import + image-op run on a glib-less trixie base), so the Dockerfile no longer installs it; `gzip` is Debian-essential but has no runtime consumer (CPython's `gzip`/`zlib` modules use the linked `libz`, never the binary; dpkg/apt decompress internally; nothing runs `tar -z`), so the Dockerfile purges it alongside `perl-base` (verified post-purge: `apt-get update`/`install`/`upgrade`, `dpkg`, and a Python `gzip` round-trip all work). All four CVEs are gone from the image; the accepted count stays **3 (0 Critical, 3 High)** at this point.

**util-linux — accepted, not eliminated (2026-07-09).** The vulnerability DB published CVE-2026-53615 (High) against the `util-linux` family (`bsdutils`/`libblkid1`/`libmount1`/`libuuid1`/`login`/`mount`/`util-linux` 2.41-5): an integer overflow in `libblkid`'s MS-DOS partition-table parser (`libblkid/src/partitions/dos.c`), Debian trixie status "open" with no fixed version published. Unlike gzip/glib, `util-linux` is genuinely irreducible — it is Debian-essential (`mount`/`login` back the base image, and `libuuid1`/`libblkid1` back CPython and `coreutils`), so it cannot be purged without breaking the image's own toolchain. The parse path is reachable only when a *privileged* caller probes an attacker-crafted block device or disk image; the container runs non-root UID 1000 with SUID/SGID stripped and never probes block devices, so the finding was accepted as a time-boxed, enumerated entry (`exp:2026-10-07`) rather than eliminated. Accepted count: **3 → 4 (0 Critical, 4 High)**.

The bookworm-era ledger evolution (the 2026-06-10 first-enforced-gate enumeration, the 2026-06-13 SQLite FTS5 additions, and the 2026-06-15 mesa elimination) is preserved in commit history and the CHANGELOG; it is superseded as the live posture by the 2026-06-30 trixie/3.14 re-enumeration, the 2026-07-02 eliminations, and the 2026-07-09 `util-linux` addition above.

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

Accepted risks are re-reviewed at most every 90 days, enforced by the `exp:` dates in [`.trivyignore`](.trivyignore). As of the 2026-06-30 trixie/3.14 re-enumeration, the 2026-07-02 gzip/glib eliminations, and the 2026-07-09 `util-linux` addition, documented acceptances are 4 CVEs (0 Critical, 4 High), all no-upstream-fix Debian trixie packages linked by CPython itself or by the base image's own coreutils/tar and util-linux toolchain, none on an untrusted-input path in the shipped API. The ledger file and the table above are the complete record.

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
[`.trivyignore`](.trivyignore) (4 CVEs, each with a 90-day expiry that
fails the gate until re-reviewed).

## Security Assessment Posture

Mercury Agent's security analysis is automated and self-assessed:

- **Automated Scanning**: Security scans run on every pull request and push (bandit, safety, pip-audit, semgrep, Trivy, CodeQL) plus a weekly scheduled run (`.github/workflows/security.yml`, Sundays 00:00 UTC)
- **Dependency Audits**: Dependabot weekly update checks plus the two-tier CVE gates described below
- **Code Reviews**: All changes require human review before merge

Mercury Agent has **not** been externally audited or penetration-tested. Production deployments requiring assurance beyond self-assessment must commission an independent security review (see "Important Security Considerations" above and the README status line: Research-grade | Community-tested | Not externally audited).

**Scheduled external review**: an independent third-party security review (covering Mercury Agent and the AMA Cryptography native backend) is planned as a pre-1.0-production milestone. Until that review completes and its findings are published here, the posture above stands — treat every release as self-assessed only. This section will be updated with the reviewer, scope, and report reference when the engagement is scheduled.

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
  five-label, four-colour ladder (CLEAR / GREEN / AMBER / AMBER+STRICT / RED) for
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

**σ_Immutable artifact dependency (fail-closed).** The σ_Immutable gate
depends on three artifacts shipped inside the package
(`[tool.setuptools.package-data]`): the trained network weights
(`security/sigma_immutable_weights.pt`), the labelled training corpus
(`security/sigma_immutable_corpus.json`), and its detached
Ed25519 + ML-DSA-65 signatures (`security/sigma_immutable_corpus.sig.json`),
verified on first gate construction. If the weights are missing or torch is
unavailable, every gated boundary raises `check="gosnn_unavailable"`; if the
corpus is missing, tampered, or its signatures do not verify, every gated
boundary raises `check="sigma_immutable"` — there is no advisory fallback in
either case. Regenerate the artifacts with
`python scripts/train_sigma_immutable.py` (which re-signs the corpus).
Independent of these artifacts, the deterministic critical-ethical floor
(`SigmaImmutableGate.enforce_ethical_floor`) remains the authoritative
gate — the learned score is advisory (synthetic-trained; see
`docs/DORMANCY_LEDGER.md`), and shipping the artifacts does not weaken the
fail-closed contract or the floor's authority.

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

*Last Updated: 2026-07-24*
*Version: 2.1.0*
