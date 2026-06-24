<!--
Copyright (C) 2025 Steel Security Advisors LLC
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Mercury ⇄ live AMA Cryptography — end-to-end validation

This document records the validation of Mercury Agent's post-quantum
cryptography path against the **real** AMA Cryptography native C backend, built
from source and loaded at import time — no mock, no stand-in. It is the
companion evidence to PR #307 (Workstream A1), which was measured against a
throwaway, uncommitted crypto stand-in because the AMA-Cryptography repository
was egress-blocked in that session. With AMA egress cleared, the same engine
now runs against live AMA, and the numbers hold.

## What was already in place vs. what this validates

The AMA backend and the engine wiring **pre-existed and were already correct**:

- `ama_cryptography/pqc_backends.py` (AMA repo) already implements the full
  surface the engine imports, dispatching to the native `libama_cryptography.so`
  via `ctypes`.
- `omni_mercury_engine/security/pqc_backends.py` already imports that surface;
  `omni_mercury_engine/_pqc_gate.py` already enforces a fail-closed import-time
  gate; `pyproject.toml [project.optional-dependencies].pqc` already pins AMA
  `v3.2.0`; CI already builds the real backend via
  `.github/actions/build-ama-cryptography`.

The gate was failing closed for a **build/install** reason, not a code gap:
the native library simply was not built in the session. The contribution here
is therefore (1) building the real backend, (2) validating the complete real
path against live AMA, and (3) one byte-identical C cleanup in the AMA repo
(see "AMA backend change" below). **The fail-closed PQC guarantee is unchanged
and was re-verified (absent → close, partial → close, healthy → resolve).**

## Environment

| Item | Value |
|---|---|
| AMA repo HEAD | `8402317` (built `-DAMA_USE_NATIVE_PQC=ON`, Release) |
| AMA version | 3.2.0 (`libama_cryptography.so.3.2.0`) |
| Mercury base | `bdcdf6a` (origin/main) |
| Toolchain | gcc 13.3.0, CMake 3.28 (system) / 4.3.4 (pip floor), Python 3.11.15 |
| PQC backend | NATIVE — ML-DSA-65, Kyber-1024, SPHINCS+-256f, SLH-DSA (FIPS 203/204/205) |

Reproduce the build with `.github/actions/build-ama-cryptography` or
docs/INSTALLATION.md → "Post-Quantum Cryptography backend".

## Results — all against the live backend

### Import-time PQC gate (fail-closed contract)

| Scenario | Expected | Observed |
|---|---|---|
| `ama_cryptography` not importable | **close** (RuntimeError) | close ✓ |
| Partial backend (Kyber flag False) | **close** (RuntimeError) | close ✓ |
| All three algorithms loadable | resolve silently | resolve ✓ |

Both independent implementations of the contract pass:
`_pqc_gate._enforce_pqc_production_gate()` and
`security.pqc_guards.check_pqc_production_readiness()`.

### Cryptographic round-trips (`scripts/verify_live_ama_integration.py`)

All 9 checks PASS: ML-DSA-65 sign/verify + tamper-reject, Kyber-1024
encapsulate/decapsulate shared-secret equality, SPHINCS+ sign/verify +
tamper-reject, SLH-DSA SHAKE-128s and SHA2-256f (hedged + deterministic)
sign/verify + tamper-reject, and `validate_pqc_environment().production_ready`.

### Test suites

| Suite | Result | Notes |
|---|---|---|
| Mercury PQC / crypto (`tests/{security,integrations,integration}`, `test_pqc_startup_gate`, `test_crypto_backend_telemetry`) | **110 passed** | incl. `test_pqc_gate_real_ama.py` (real native lib, no skip) |
| ADBench baseline compare + selection | **17 passed** | engine-level no-regression harness |
| AMA native KAT (`build/bin/test_kat`) | **PASS** | Kyber FIPS 203 KAT, ML-DSA, SPHINCS+-SHA2-256f |

### ADBench no-regression (real data, live backend)

18/18 datasets scored, **mean AUROC 0.7634**, **bit-identical** to the committed
`research/omni_equation/adbench_results.json` (0 diffs). The PQC backend is
orthogonal to the detection ensemble, so the A1 headline is preserved against
real AMA exactly as it was against the stand-in.

### Workstream A1 deliverables re-checked against live AMA (PR #307 branch)

Validated read-only on `claude/attached-document-review-uifar2` with the live
backend installed:

- φ-numerology drift gate: **OK — 632 files scanned, 0 violations**.
- A1 56-test suite (19 integrity + `test_abms_disciplines` +
  `test_parapsychology_quarantine`): **56 passed**.

## AMA backend change (AMA-Cryptography repo, same branch)

`src/c/ama_slhdsa.c`: split the branch-on-`use_compressed_adrs` ADRS serializer
into two fixed-width functions (`slh_addr_serialize_compressed` /
`slh_addr_serialize_full`, declared `uint8_t out[static 22|32]`). This removes a
`-Wstringop-overflow` **false positive** GCC raised when the 32-byte branch was
inlined into a 22-byte (`addr_c[22]`) caller it can never reach at runtime.
Each SLH-DSA hash family already calls exactly one form with a matching-width
buffer, so the change is **byte-for-byte equivalent**: deterministic SLH-DSA
signatures for both parameter sets are SHA-256-identical before and after, and
`build/bin/test_kat` stays green.

## Pending / not claimed

- This is a session-local validation. "CI green on the real path" is asserted by
  the existing workflows (`ci.yml`, `pqc-production-check.yml`) that build AMA via
  the composite action; the GitHub-hosted runs are triggered by the PRs, not by
  this document.
- The medical-path real-label ECE measurement remains the documented A1
  follow-up (needs a labeled clinical corpus + calibration harness); unaffected
  by the crypto backend.
