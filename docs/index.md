# Mercury Agent Documentation

Version `2.0.0` — Steel-SecAdv-LLC. Last updated: 2026-06-17.

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
  [`ARCHITECTURE.md`](https://github.com/Steel-SecAdv-LLC/Mercury-Agent/blob/main/ARCHITECTURE.md) §"Dual-Gate Hard Ethical
  Enforcement" and [`MATH_SPEC.md`](MATH_SPEC.md) §2.1.5
  "σ_Immutable Hard Gate (Wave B, PR #179)".
- **Sole PQC backend, hard-gated at import.**
  AMA Cryptography (pinned to **v3.2.0** in
  `.github/workflows/pqc-production-check.yml` and the `[pqc]` extra
  of `pyproject.toml`) is the only
  supported post-quantum backend (PR #144). v3.2.0 also exposes the
  native HMAC-SHA-256 / HMAC-SHA-512 bindings consumed by Mercury's
  `native_jwt` HS256 / HS512 signing path (see CHANGELOG
  `[Unreleased]` § "AMA-routed JWT HMAC signatures"). Package import is
  hard-gated: `omni_mercury_engine._pqc_gate._enforce_pqc_production_gate`
  runs from `__init__.py`, imports `ama_cryptography.pqc_backends`,
  and refuses to start unless ML-DSA-65, Kyber-1024, and SPHINCS+
  are all backed by the native AMA library. `AMA_REQUIRE_REAL_PQC`
  is retained only for legacy workflow readability; it no longer
  creates a dev-mode escape hatch.
  `security/pqc_guards.check_pqc_production_readiness()` remains
  available for callers that want the same check at a finer
  boundary. See [`SECURITY.md`](https://github.com/Steel-SecAdv-LLC/Mercury-Agent/blob/main/SECURITY.md) for the full
  contract.
- **Committed benchmark run.** The README headline is the committed
  `mercury_benchmark_results.json` run — **66 successful / 75
  attempted**, Mean AUC **0.8251**, Median **0.8747**, Mean Oracle
  F1 **0.5998** (2026-06-21) — surfaced in the README "Latest
  Benchmark Results" block. CI's regression-gate floor is the
  historical **0.803 AUC / 0.589 F1** baseline (the gate trips 15%
  below it). Externally-comparable subset: ADBench Mean AUC 0.8251.
  See [`BENCHMARKS.md`](BENCHMARKS.md).
- **Pickle removed from the training-data path; not a blanket ban.**
  PR #166 deleted the `pickle` code path from the training-data
  loader; benchmark and dataset artefacts use npz / json /
  safetensors. The repo still ships
  `src/omni_mercury_engine/security/sigma_immutable_weights.pt` and loads it via
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
FUSION_CAPACITY_STRATEGY
DOMAIN_PERFORMANCE
ROUTING_GUIDE
DATASOURCES
LIVE_DATA_VALIDATION
ORACLE_NOISE_COLOR
DEPLOYMENT
OFFLINE_OPERATION
HARDWARE_HARNESS
ROADMAP
MIGRATION-1.6-to-1.7
COMPLIANCE
PROFILING
TOOLS
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
