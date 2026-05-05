# Mercury Agent Documentation

Version `1.6.0` — Steel-SecAdv-LLC. Last updated: 2026-05-05.

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
  There is no advisory mode. See `ARCHITECTURE.md` §"Dual-Gate Hard
  Ethical Enforcement" and `MATH_SPEC.md` §"Ethical Gating".
- **Sole PQC backend.** AMA Cryptography is hard-required as the only
  post-quantum cryptographic backend (PR #144); Mercury refuses to
  start without it when `AMA_REQUIRE_REAL_PQC=true`. See `SECURITY.md`.
- **Honest benchmarks.** All 1.x benchmark numbers in this repo are
  computed over the **64 reproducible datasets** of 75 attempted; 11
  external sources are unavailable / rate-limited and 1 loader
  (FEMA Disaster) is known-broken. See `BENCHMARKS.md` and the README
  "Empirical Benchmark Results" section.
- **No pickle code path.** The training-data loader was rewritten in
  PR #166 to remove `pickle` entirely; all artifacts use safe
  serialization formats (npz / json / safetensors).

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
