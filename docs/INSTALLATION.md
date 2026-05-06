# Installation

Applies to Mercury Agent **v1.6.x**. Last updated: 2026-05-05.

## Requirements

- Python >= 3.11
- pip >= 21.0
- A C toolchain (clang or gcc) and CMake >= 3.20 for the AMA
  Cryptography native PQC build (see "Post-Quantum Cryptography
  backend" below); only required when running with
  `AMA_REQUIRE_REAL_PQC=true`, but production deployments **must**
  enable it.

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
| **ML** | `pip install -e ".[ml]"` | Core + PyTorch, scikit-learn, torchvision |
| **Visual** | `pip install -e ".[visual]"` | ML + visual anomaly detectors |
| **VLM** | `pip install -e ".[vlm]"` | transformers, accelerate |
| **API** | `pip install -e ".[api]"` | FastAPI, uvicorn |
| **All** | `pip install -e ".[all]"` | Everything above |
| **Dev** | `pip install -e ".[dev]"` | All + pytest, black, ruff, mypy |

## Core Dependencies

The core anomaly detection path (`MercuryAnomalyDetector`) requires only:

- `numpy >= 1.24.0`
- `scipy >= 1.10.0`

scikit-learn is **not** required for core detection. It is an optional dependency
used for cross-domain transfer, calibration baselines, and benchmark comparisons.

## Post-Quantum Cryptography backend

Mercury Agent uses **AMA Cryptography** as the sole supported PQC
backend (see `SECURITY.md` and PRs #144, #162). The package import
is guarded — `security/pqc_backends.py` catches `ImportError` and
keeps Mercury importable with stub functions, so a developer
without the native library can still load the package — but
an inlined production-gate check at package import time
(`omni_mercury_engine/__init__.py::_enforce_pqc_production_gate`)
fails closed when `AMA_REQUIRE_REAL_PQC=true` and the AMA Cryptography
native C backend is not loadable. With the env var set, `import
omni_mercury_engine` raises `RuntimeError` before any other package
state is materialised, so production deployments cannot accidentally
fall through to stub PQC functions.

For production, build and install the native library from the
upstream AMA-Cryptography repository (note: the `cmake` step
operates on the AMA-Cryptography checkout, **not** on the
Mercury-Agent repo, which has no `CMakeLists.txt` of its own).
The canonical build steps are exercised by
`.github/workflows/pqc-production-check.yml` (currently pinned to
`AMA_REF: v3.1.0`):

```bash
# 1. Clone and build the AMA-Cryptography native library
git clone --depth 1 --branch v3.1.0 \
    https://github.com/Steel-SecAdv-LLC/AMA-Cryptography.git /tmp/ama-cryptography
cd /tmp/ama-cryptography
cmake -B build -DAMA_USE_NATIVE_PQC=ON
cmake --build build

# 2. Install the Python package from the same checkout
AMA_NO_CYTHON=1 pip install --no-build-isolation .

# 3. Export the runtime loader path and the production gates
export LD_LIBRARY_PATH="/tmp/ama-cryptography/build/lib:/tmp/ama-cryptography/build:${LD_LIBRARY_PATH:-}"
export AMA_REQUIRE_REAL_PQC=true
export AMA_REQUIRE_CONSTANT_TIME=true   # recommended

# 4. Return to the Mercury-Agent checkout for the rest of the install
cd /path/to/Mercury-Agent
```

With `AMA_REQUIRE_REAL_PQC=true`, the inlined production-gate
check at `omni_mercury_engine/__init__.py::_enforce_pqc_production_gate`
runs at package import and refuses to proceed if the native library
is unloadable — `import omni_mercury_engine` raises `RuntimeError`
before any other package state is materialised. Without the env
var set, Mercury imports against stub PQC functions for
development convenience. There is no fallback chain to a non-AMA
backend.

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
