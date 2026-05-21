# Mercury Agent Documentation

Version `1.7.0` — Steel-SecAdv-LLC. Last updated: 2026-05-20.

Mercury Agent is the **neuro-symbolic AI** orchestration / cognition layer
of the FIND**Ω**YOU stack — a hybrid of deep-learning detectors, an
explicit symbolic reasoning layer, a 7-phase cognitive evolution
architecture and hard ethical bounding.  It is paired with
[AMA Cryptography](https://github.com/Steel-SecAdv-LLC/AMA-Cryptography)
for the post-quantum cryptographic substrate.

## What you should know first

- **Decision-boundary contract (Wave B, PR #179).** Every public
  `detect` / `analyze` / `predict` surface runs **two independent
  mandatory hard ethical gates** — Benevolence, then σ_Immutable —
  and raises `EthicalConstraintViolationError(check=…)` on failure.
  There is no advisory mode. See the top-level
  [`ARCHITECTURE.md`](../ARCHITECTURE.md) §"Dual-Gate Hard Ethical
  Enforcement" and [`MATH_SPEC.md`](MATH_SPEC.md) §2.1.5
  "σ_Immutable Hard Gate (Wave B, PR #179)".
- **Sole PQC backend (with a soft import path for non-PQC dev).**
  AMA Cryptography (pinned to **v3.2.0** in
  `.github/workflows/pqc-production-check.yml`) is the only
  supported post-quantum backend (PR #144). v3.2.0 also exposes the
  native HMAC-SHA-256 / HMAC-SHA-512 bindings consumed by Mercury's
  `native_jwt` HS256 / HS512 signing path (see CHANGELOG
  `[Unreleased]` § "AMA-routed JWT HMAC signatures"). The package import is
  guarded — `security/pqc_backends.py` catches `ImportError` and
  keeps Mercury importable with stub functions, so a developer
  without the native library can still load the package — but
  the package import path runs a production-gate check
  (`omni_mercury_engine._pqc_gate._enforce_pqc_production_gate`,
  invoked from `__init__.py` at import time) that fails closed when
  `AMA_REQUIRE_REAL_PQC=true` and the native AMA Cryptography
  library is missing or partially built. The gate is automatic:
  `import omni_mercury_engine` raises `RuntimeError` before any
  other package state is materialised. With the env var
  unset (the dev-mode default), the gate is a no-op and the soft
  PQC stubs from `security/pqc_backends.py` carry development.
  `security/pqc_guards.check_pqc_production_readiness()` remains
  available for callers that want the same check at a finer
  boundary. See [`SECURITY.md`](../SECURITY.md) for the full
  contract.
- **Two distinct benchmark cuts.** The README headline is the
  **64/75 reproducibility set** (Mean AUC 0.8285, Mean Oracle F1
  0.6370). CI's regression-gate floor is the **51/55**
  `mercury_benchmark.py` direct path (Mean AUC 0.8030, Mean Oracle
  F1 0.5886, the legacy baseline the 64/75 run improves on). See
  [`BENCHMARKS.md`](BENCHMARKS.md) for the full reconciliation and
  the README "Empirical Benchmark Results" section for the public
  headline.
- **Pickle removed from the training-data path; not a blanket ban.**
  PR #166 deleted the `pickle` code path from the training-data
  loader; benchmark and dataset artefacts use npz / json /
  safetensors. The repo still ships
  `security/sigma_immutable_weights.pt` and loads it via
  `torch.load(..., weights_only=True)` (the safe-tensor torch
  loader path), so PyTorch's `.pt` format is still in use for
  trained-model weights — this is intentional and not a `pickle`
  fall-back.
- **Production-mode primitive (`MERCURY_ENV`).** New v1.7 module
  `omni_mercury_engine._env` exposes the canonical environment-mode
  flag (`development` default, `production`) plus shared fail-closed
  helpers (`get_mercury_env`, `is_production`,
  `require_real_component`, `MercuryProductionConfigError`). It is
  orthogonal to `AMA_REQUIRE_REAL_PQC`; production deployments
  typically set both. See
  [`MIGRATION-1.6-to-1.7.md`](MIGRATION-1.6-to-1.7.md) §3 for the full
  contract.
- **Governance modules live in `compliance/`, not `security/`.** The
  v1.7 development cycle introduced first-party NIST CSF 2.0, FIRST.org
  TLP 2.0, and OSHA / eCFR modules under
  `omni_mercury_engine.compliance`. `security/` stays reserved for
  implementation primitives (crypto, PQC, threat detection, audit
  logging). See [`COMPLIANCE.md`](COMPLIANCE.md).

## Navigation

```{toctree}
:maxdepth: 2
:caption: Contents

INSTALLATION
ARCHITECTURE
API_REFERENCE
MATH_SPEC
BENCHMARKS
DOMAIN_PERFORMANCE
ROUTING_GUIDE
DATASOURCES
LIVE_DATA_VALIDATION
ORACLE_NOISE_COLOR
DEPLOYMENT
ROADMAP
PYTHON_DEP_CVE_AUDIT
CROSS_DOMAIN_ANALYSIS
COMPREHENSIVE_REPO_AUDIT
MIGRATION-1.6-to-1.7
COMPLIANCE
PROFILING
medical/SETUP
drone/SETUP
```

## See also

- Top-level `README.md` — project overview, badges, current benchmark
  table, install quick-start.
- Top-level `ARCHITECTURE.md` — full system architecture, including
  the Dual-Gate Hard Ethical Enforcement contract and σ_Immutable
  layout.
- Top-level `CHANGELOG.md` — release history; the `[Unreleased]`
  block at the top tracks Wave A / Wave B work in flight.
- Top-level `SECURITY.md` — supported versions, PQC backend audit
  status, vulnerability reporting process.
- Top-level `CONTRIBUTING.md` — development workflow and PR
  expectations.
