<!--
Copyright (C) 2025 Steel Security Advisors LLC
SPDX-License-Identifier: GPL-3.0-or-later
-->

# FOLLOWUP_BRIEF — Mercury-Agent Tier 0 Foundation (PR #319)

Canonical onboarding + handoff for **`steel/refinement-mercury-foundation` → PR
#319**. A fresh agent should be able to read this top-to-bottom, reproduce the
environment, `import omni_mercury_engine` against the **real** native PQC
backend, run the engine end-to-end, and pick up the open items — without
re-discovering any of the build friction below.

> **Golden rule carried forward:** the most valuable correction in this line of
> work came from an adversarial bot catching a *subtle weakening* of a safety
> control (a MONOTONE_HARM re-gate that had been quietly turned into a no-op).
> Keep that adversarial mindset. A green local run is necessary, not sufficient:
> before trusting a safety change, prove the guard still *fires* (neutralise the
> fix and watch the test go red), don't just watch the suite go green.

---

## 0. Priority 0 — build AMA and make `import omni_mercury_engine` work

Mercury has **no soft-stub PQC fallback**: `import omni_mercury_engine` runs an
import-time gate (`omni_mercury_engine/_pqc_gate.py`) that raises unless the
native **AMA-Cryptography** backend exposes ML-DSA-65 (Dilithium) + Kyber-1024 +
SPHINCS+ **and** the installed version resolves to release `3.2.0`. So the engine
cannot even be imported (and therefore no test that imports it can run) until AMA
is built and installed. Do this first.

### 0.1 Prerequisites

- `git`, `gcc`/`g++` (≥ 12; 13.x is fine), `ninja`/`make`
- **`cmake ≥ 4.3.2`** — the PyPI `cmake` shim, *not* the distro cmake (AMA's
  `setup.py` preflight reads the shim's `__version__`). A system cmake 3.28 will
  trip the preflight.
- Python 3.11–3.14. Network access to `github.com` (AMA source) and PyPI.

### 0.2 Canonical build (one command)

`scripts/build_ama_native.sh` is the **canonical local/Docker build** — the
Dockerfile builder stage runs it directly (`RUN … bash …/build_ama_native.sh`).
CI lanes do **not** call the script; they use the *mirrored* composite action
`.github/actions/build-ama-cryptography` (same steps, pinned `ama-ref`). Keep the
two in lockstep (identical `AMA_REF` / build flags). The script clones AMA at the
pinned `AMA_REF`, builds the native library, installs the Python package,
**co-locates `libama_cryptography.so*` inside the installed `ama_cryptography/`
package** (so it loads with no `LD_LIBRARY_PATH`), and fails loudly unless all
three PQC algorithms load:

```bash
# from the repo root
python -m pip install --upgrade "setuptools>=78.1.1" "wheel>=0.47.0" "cmake>=4.3.2"
bash scripts/build_ama_native.sh          # AMA_REF defaults to v3.2.0
```

Overrides: `AMA_REF` (git tag/branch), `AMA_REPO`, `AMA_BUILD_DIR`.

### 0.3 Container gotcha — Debian-managed wheels block `pip --upgrade`

On images where `wheel` / `cryptography` were installed by **apt** (Debian
`dist-packages`), `pip` cannot uninstall them to upgrade and dies with
`Cannot uninstall <pkg>, RECORD file not found`. This aborts the build script
(step 1) or the editable install. It is an **environment quirk, not a repo
defect** — GitHub-hosted CI runners have pip-managed wheels, so CI never hits it
(which is exactly why "CI masks what a clean container reproduces"). Work around
it by pre-seeding the offending packages with `--ignore-installed` (they then
shadow the apt copies on `sys.path`), *then* run the canonical build:

```bash
# one-time, only on Debian/apt-provisioned containers
python -m pip install --upgrade --ignore-installed wheel setuptools cmake
python -m pip install --ignore-installed "cryptography>=46.0.7"

python -m pip install --upgrade "setuptools>=78.1.1" "wheel>=0.47.0" "cmake>=4.3.2"
bash scripts/build_ama_native.sh
```

### 0.4 Install the engine and verify

```bash
python -m pip install -e .            # core; or ".[ml]" for the torch stack
python -c "import omni_mercury_engine, ama_cryptography as a; \
           print('engine OK; AMA', a.__version__)"
# -> engine OK; AMA 3.2.0
```

If the AMA build emits `HMAC-SHA3-256 native backend not available` /
`native Ed25519 not built`, that is expected with the default cmake flags in
`build_ama_native.sh` (it builds the three **PQC** algorithms; HMAC-SHA3/Ed25519
native backends are optional and unused by the import gate — the audit
hash-chain uses Python's stdlib `hashlib.sha3_256`). The engine imports and the
gate passes regardless.

---

## 1. Upstream AMA `v3.2.0` tag — fragility RESOLVED

A prior note (still in the PR description at handoff) warned that the mandatory
pin `v3.2.0` was **not** published upstream (latest tag `v3.1.0`; only `main`
reported `3.2.0`), so a clean clone with `--branch v3.2.0` could not reproduce.

**As of 2026-07-03 this is fixed upstream:** `Steel-SecAdv-LLC/AMA-Cryptography`
now publishes an annotated tag `refs/tags/v3.2.0` (→ `367b19e7`), and building
from it yields `ama_cryptography.__version__ == 3.2.0`, satisfying the gate. A
clean clone reproduces. Verify before trusting:

```bash
git ls-remote --tags https://github.com/Steel-SecAdv-LLC/AMA-Cryptography.git v3.2.0
# expect a refs/tags/v3.2.0 line
```

If the tag ever disappears or is re-pointed, the immediate fallback is
`AMA_REF=main` (which also reports `3.2.0`); it trades immutable-tag KAT
reproducibility for availability, so treat it as a stopgap and re-pin to a tag.
**Do not** relax the version gate to accept a different release — that would be a
weakening.

---

## 2. Toolchain pins (reproduce lint/type failures exactly)

Pins live in `.pre-commit-config.yaml` / `pyproject.toml [ml]/[dev]` / `ci.yml`
and are cross-checked by a structural gate in CI, so keep them in lockstep:

| Tool | Pin | CI scope |
|---|---|---|
| `black` | `26.5.1` | `src/ tests/` |
| `ruff` | `≥ 0.15.12` (v0.15.15 pinned in pre-commit) | `src/ tests/ scripts/ tools/` |
| `flake8` | `7.x` (`--max-line-length=100 --extend-ignore=E203,W503,E402,E501,F841`) | `src/ tests/ scripts/ tools/` |
| `mypy` | `2.1.0` (strict; `python_version = 3.12`) | `src/omni_mercury_engine/`, `tests/` |
| `pydocstyle` | `6.3.0` (`--convention=google`) | `src/omni_mercury_engine/` |
| `actionlint` + `shellcheck` | pre-commit `rev` | `.github/workflows/` |

`research/` and `benchmarks/` are **not** in the ruff/flake8 CI scope — pre-existing
lint debt there does not gate PRs and is out of scope for this branch.

> Installing the pins can leave older console-scripts ahead on `PATH`. Invoke via
> `python -m black` / `python -m mypy` / `python -m ruff` / `python -m pydocstyle`
> to guarantee you run the pinned version, not a shadowing binary.

---

## 3. Running the suite

```bash
# Tier-0 safety lane (fast, no torch): the exact `ci/gate-unit` set
python -m pytest -q \
  tests/ethical/test_capability_contract.py \
  tests/ethical/test_weapons_gate_fail_closed.py \
  tests/ethical/test_gates_present_no_weakening.py \
  tests/security/test_sigma_immutable_fail_closed.py \
  tests/security/test_escalation_audit_integration.py \
  tests/scripts/test_ingest_weapons_gate_corpus.py \
  tests/benchmarks/test_rolling_corpus_eval.py \
  tests/test_pqc_startup_gate.py

# Offline OOF + corpus gates
python scripts/ingest_weapons_gate_corpus.py --check
PYTHONPATH=src python benchmarks/rolling_corpus_eval.py --check

# Full-suite regression (torch required: install ".[ml]" or torch-cpu first)
python -m pytest tests/ -n auto --dist worksteal --timeout=300 -q -ra
```

Full collection is **9,802 tests**; with torch absent, ~10 ML test *modules* fail
at **collection** with `ModuleNotFoundError: torch` (bare top-level `import
torch`). That is a dependency gap, not a code defect — install `.[ml]` (or
`torch`/`torchvision` CPU wheels) and they collect. Also install the dev test
plugins (`pytest-timeout`, `pytest-xdist`) or `--strict-markers` rejects the
`timeout` / `xdist_group` markers at collection.

---

## 4. AI/Bot alert triage procedure

For every new Copilot / bot / CI alert, in order:

1. `pull_request_read` (check runs + review threads) to enumerate what's open.
2. Inspect failing jobs; `get_job_logs` for any failure.
3. Reproduce locally against the **pinned** toolchain (§2) and the real AMA
   build (§0) — never against a mocked engine.
4. Fix the **root cause**. Never suppress, `# noqa`-away, or add a workaround
   that hides the failure. A doc/reality mismatch is fixed by making the doc
   true, *unless* making it true would weaken a control — then fix the doc, keep
   the control.
5. Re-run the same failure path locally to confirm the fix, and for any safety
   change, prove the guard still fires (neutralise-and-watch-it-fail).
6. Reply on the thread only when it resolves the item or raises a real question.

---

## 5. What this session changed (PR #319 follow-up)

Resolved the five open Copilot review threads with code-level remediation (no
thread closed without a change):

1. **Audit-logger reconfigure resource leak** (`cognitive/gate_audit.py` +
   `security/secure_audit_logging.py`) — repointing `MERCURY_SECURE_AUDIT_DIR`
   replaced the process-global `SecureAuditLogger` **without** shutting the old
   one down, orphaning its daemon flush thread + file handle and **dropping any
   buffered, not-yet-persisted audit events**. Fix: `_secure_audit_logger()` now
   serialises the check→shutdown→reconfigure (a dedicated lock) and shuts the
   prior logger down (flush + join) before replacing it; the flush loop now waits
   on the stop event instead of an unconditional sleep, so shutdown returns
   promptly. Best-effort — a faulty prior logger is logged, never raised into the
   audit path. Regression test added and **proven to have teeth** (neutralising
   the shutdown turns it red).
2. **PEP 440 version wording** (`docs/INSTALLATION.md`, `.env.example`) — the
   docs claimed `ama_cryptography.__version__ == 3.2.0` / "reports exactly this
   version", but the gate matches the pinned *release* PEP 440-tolerantly
   (`v3.2.0`, `3.2.0.post1`, `3.2.0+cpu`, `3.2` accepted; `3.1.0` refused).
   Reworded to match `_release_matches`, so a valid post/local build isn't
   misdiagnosed as a failure.
3. **`CI_MEANING_LEVEL_ENABLED` doc/reality mismatch** (`docs/INSTALLATION.md`,
   `.env.example`) — docs implied the variable toggles the `ci/meaning-level`
   lane, but that lane runs on **every** PR via a validated stdlib model double.
   Gating the whole safety lane behind an `if:` would *weaken* it, so the fix is
   to make the docs true: the lane is always-on; the variable is a marker for
   opting into a deeper real-served-model run (applied by swapping the serving
   block per the workflow header comment), not an on/off switch.

Verification: `black`/`ruff`/`flake8`/`pydocstyle` clean on CI scope; strict
`mypy 2.1.0` — "no issues found in 659 source files"; the `ci/gate-unit` lane
plus the new regression test pass against the real AMA v3.2.0 build.

---

## 6. Known open items (owners / timelines)

| # | Item | Owner | Timeline |
|---|---|---|---|
| 1 | **Harder out-of-focus corpus cases.** Gate-level OOF is done; residual edge cases remain in the held-out adversarial slice (lexical-only recall ≈ 0.48 is the gap the meaning-level lane closes). Extend `benchmarks/rolling_corpus_eval.py` coverage. | Safety/eval | Next PR after #319 merges |
| 2 | **Upstream `v3.2.0` tag** — now published (§1); keep the pin honest by re-verifying the tag on each AMA bump and re-pinning off `main` if ever used as a stopgap. | Crypto/release | Ongoing / per-bump |
| 3 | **Full-suite regression** — reproduced this session (see the PR description for the pass/skip counts and any environment-only skips). Re-run on every gate-touching change. | CI | Per-PR |
| 4 | **Run Mercury's ensembles for measurable value** — drive `OmniMercuryEngine` + `DomainEncoderStack` fusion on the benchmark datasets (`benchmarks/comprehensive_benchmark.py`, `benchmarks/rolling_corpus_eval.py`) and record ROC-AUC / OOF ECE deltas as the value metric. | Research | Rolling |

Anything that changes the build or runtime here **must** be reflected back into
this file and `docs/INSTALLATION.md`.
