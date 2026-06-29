# Installation

Applies to Mercury Agent **v2.0.x**. Last updated: 2026-06-16.

## Requirements

- Python >= 3.11 (3.12 recommended; 3.13 and 3.14 supported)
- pip >= 26.1 (the CVE-2026-6357 floor enforced across every install
  path in CI by `tests/security/test_cve_2026_6357_regression.py`)
- GCC >= 12 and CMake >= 4.3.2 for the AMA Cryptography native PQC
  build (see "Post-Quantum Cryptography backend" below). Mercury does
  not import without real AMA/PQC.

## Production-mode primitives (`MERCURY_ENV` + AMA/PQC)

Production deployments should set these environment variables before
importing `omni_mercury_engine`:

```bash
export MERCURY_ENV=production
export AMA_REQUIRE_REAL_PQC=true
export AMA_REQUIRE_CONSTANT_TIME=true   # recommended
```

`AMA_REQUIRE_REAL_PQC` is retained for legacy workflow readability, but
AMA/PQC is mandatory regardless of its value:

- `MERCURY_ENV` (added in v1.7) is consumed by every collaborator that
  has a mock/stub fallback (`narrative.voice.MercuryVoice`, more to
  come). In `production`, the absence of a real adapter raises
  `MercuryProductionConfigError`; in `development` (the default) it
  warns and downgrades. An unknown value (e.g.
  `MERCURY_ENV=prod`) also raises — typos must be loud. See
  [`MIGRATION-1.6-to-1.7.md`](MIGRATION-1.6-to-1.7.md) §3.
- The import-time PQC check
  (`omni_mercury_engine._pqc_gate._enforce_pqc_production_gate`) always
  requires `ama_cryptography.pqc_backends` plus ML-DSA-65,
  Kyber-1024, and SPHINCS+ native availability. Missing or
  partially-built AMA Cryptography raises `RuntimeError` at
  `import omni_mercury_engine` time before any other package state is
  materialised.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Steel-SecAdv-LLC/Mercury-Agent.git
cd Mercury-Agent

# Install core (anomaly detection, no PyTorch required)
pip install -e .

# Or install with full ML stack
pip install -e ".[all]"
```

## Installation Profiles

| Profile | Command | Includes |
|---------|---------|----------|
| **Core** | `pip install -e .` | numpy, scipy, pandas, MercuryAnomalyDetector |
| **ML** | `pip install -e ".[ml]"` | Core + PyTorch, torchvision, pytorch-lightning, timm, OpenCV (headless) |
| **Visual** | `pip install -e ".[visual]"` | ML + visual anomaly detectors |
| **VLM** | `pip install -e ".[vlm]"` | transformers, accelerate |
| **API** | `pip install -e ".[api]"` | FastAPI, httpx, uvicorn, python-multipart |
| **PQC** | `pip install -e ".[pqc]"` | AMA Cryptography (pinned to `v3.2.0`) |
| **Compliance** | `pip install -e ".[compliance]"` | NIST CSF live-fetcher dependency (`openpyxl`) |
| **All** | `pip install -e ".[all]"` | Every feature extra (ml, visual, vlm, foundation, medical, face, api, sota, llm, drift, fairness, streaming, optimization, benchmark, domains, gui, explainability). `[pqc]`, `[compliance]`, and `[dev]` install separately. |
| **Dev** | `pip install -e ".[dev]"` | Tooling only: pytest (+ asyncio/cov/timeout/mock/xdist), hypothesis, black, mypy, ruff, pre-commit. Combine with `[all]` for the full stack. |

The `[compliance]` extra installs `openpyxl`, used only by
`NISTCSFReferenceFetcher` when parsing the live NIST CSRC reference
XLSX. The OSHA and TLP 2.0 modules in `omni_mercury_engine.compliance`
have no extra dependencies beyond core.

## Core Dependencies

The core anomaly detection path (`MercuryAnomalyDetector`) requires only:

- `numpy >= 2.4.0` (required for Python 3.12/3.13 wheels and the strict-mypy type contract)
- `scipy >= 1.10.0`

scikit-learn is **not** required for core detection and is not part of any
profile above. It ships in the `[benchmark-comparison]` extra and is used for
cross-domain transfer, calibration baselines, and benchmark comparisons.

## Post-Quantum Cryptography backend

Mercury Agent uses **AMA Cryptography** as the sole supported PQC
backend (see `SECURITY.md` and PRs #144, #162). Package import is
guarded unconditionally: `security/pqc_backends.py` imports the real
`ama_cryptography.pqc_backends` surface, and the import-time gate
(`omni_mercury_engine._pqc_gate._enforce_pqc_production_gate`,
invoked from `__init__.py`) requires ML-DSA-65, Kyber-1024, and
SPHINCS+ native availability. Missing or partially-built AMA
Cryptography raises `RuntimeError` before any other package state is
materialised.

For local and production installs, build and install the native library with the
canonical helper Mercury ships, **`scripts/build_ama_native.sh`** (the `cmake`
step operates on the AMA-Cryptography checkout the script clones, **not** on the
Mercury-Agent repo, which has no `CMakeLists.txt` of its own). It clones the
pinned `AMA_REF` (`v3.2.0`, matching pyproject's `ama-cryptography` git pin and
`.github/actions/build-ama-cryptography`), builds the native PQC library,
installs the Python package, **co-locates the shared object inside the installed
`ama_cryptography` package** so it loads with no `LD_LIBRARY_PATH`, and fails
loudly unless ML-DSA-65 + Kyber-1024 + SPHINCS+ all load:

```bash
# Requires git, gcc/g++ >= 12, and cmake >= 4.3.2 on PATH.
bash scripts/build_ama_native.sh
export AMA_REQUIRE_CONSTANT_TIME=true   # recommended
```

Override the ref/repo/scratch dir via `AMA_REF`, `AMA_REPO`, `AMA_BUILD_DIR`.
The script performs, in order: install the PEP 517 build floors
(`setuptools>=78.1.1`, `wheel>=0.47.0`, `cmake>=4.3.2`); `git clone --branch
v3.2.0`; `cmake -DAMA_USE_NATIVE_PQC=ON -DAMA_BUILD_SHARED=ON` + build;
`AMA_NO_CYTHON=1 pip install --no-build-isolation --force-reinstall --no-deps .`;
co-locate `libama_cryptography.so*`; verify via `get_pqc_backend_info()`.

**Docker / Kubernetes:** the production image builds this automatically — the
Dockerfile builder stage runs `scripts/build_ama_native.sh`, the `.so` travels
into the runtime image with the virtualenv, and a build-time
`import omni_mercury_engine` smoke test fails the build if the backend is
missing. No extra steps are required to deploy a PQC-complete image.

The inlined gate at
`omni_mercury_engine/__init__.py::_enforce_pqc_production_gate` runs at
package import and refuses to proceed if the native library is
unloadable — `import omni_mercury_engine` raises `RuntimeError` before
any other package state is materialised. There is no stub path and no
fallback chain to a non-AMA backend.

## Verify Installation

```bash
python -c "
from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector
import numpy as np
d = MercuryAnomalyDetector()
d.fit(np.random.randn(100, 5))
r = d.detect(np.random.randn(20, 5))
print(f'Scores shape: {r[\"scores\"].shape}')
print('Installation OK')
"
```

## Running Tests

```bash
# Core ensemble tests (fast, no extra deps)
pytest tests/validation/test_ensemble_replacement.py -v

# Full test suite (requires dev dependencies)
pip install -e ".[dev]"
pytest tests/ -x -k "not mimic and not swat and not wadi"
```

## Running Benchmarks

```bash
# Mercury benchmark on real datasets (requires ML deps for data loading)
pip install -e ".[ml]"
python benchmarks/mercury_benchmark.py
```

Results are written to `benchmarks/mercury_benchmark_results.json`.

## v1.7 cycle additions

- **Medical integrations.** The `EndocrinologyDetector` and
  `AnesthesiologyPredictor` are integration-ready (no vendor
  credentials in tree). See [`medical/SETUP.md`](medical/SETUP.md) for
  Dexcom v3 / FHIR R4 wiring and the `CGMDataSource` /
  `VitalsDataSource` adapter contract.
- **Drone telemetry.** The `DroneAnomalyDetector` is transport-agnostic;
  populate `DroneState` instances from your ingest layer (PX4 ULog via
  `pyulog`, MAVLink via `pymavlink`, or vendor SDK). See
  [`drone/SETUP.md`](drone/SETUP.md).
- **Compliance modules.** NIST CSF 2.0, FIRST.org TLP 2.0, and
  OSHA / eCFR live under `omni_mercury_engine.compliance`. See
  [`COMPLIANCE.md`](COMPLIANCE.md).
- **Performance profiling.** Six entry points (`@profile_func`,
  `@profile_memory`, `@profile_time`, `@profile_time_async`,
  `@profile_complete`, `PerformanceBenchmark`) plus `benchmark_function`
  ship in `omni_mercury_engine.utils.profiling` and are gated by
  `set_profiling_enabled(True)` at runtime. See
  [`PROFILING.md`](PROFILING.md).
