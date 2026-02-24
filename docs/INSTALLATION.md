# Installation

## Requirements

- Python >= 3.11
- pip >= 21.0

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
