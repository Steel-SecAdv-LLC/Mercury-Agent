# Installation

## Requirements

- Python 3.12+
- PyTorch 2.1+

## Quick Start

```bash
# Install base package
pip install -e .

# Or install with all optional dependencies (torch, vision, lightning, etc.)
pip install -e ".[all]"
```

## Verification

```bash
python -c "import omni_mercury_engine; print(omni_mercury_engine.__version__)"
# Output: 1.4.0
```

## Running Tests

```bash
# Run core tests (no live data download)
pytest tests/ -v --ignore=tests/validation/

# Run live-data validation (downloads real datasets)
export MERCURY_RUN_LIVE_DATA=true
pytest tests/validation/test_real_data_validation.py -v
```

## Dependencies

Core dependencies are installed automatically:

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >= 1.24 | Numerical computation |
| scikit-learn | >= 1.3 | Statistical methods, IsolationForest |
| torch | >= 2.1 | Neural network detectors |
| pandas | >= 2.0 | Data loading (NSL-KDD, CICIDS) |
| click | >= 8.0 | CLI interface |

Optional dependencies (installed with `.[all]`):

| Package | Purpose |
|---------|---------|
| torchvision | Visual anomaly detectors |
| pytorch-lightning | Training infrastructure |
| fairlearn | Fairness metrics |
| matplotlib | Benchmark visualizations |

## Platform Support

- Linux (primary, CI-tested)
- macOS (community-tested)
- Windows (experimental)

## See Also

- [BENCHMARKS.md](BENCHMARKS.md) for real-world performance
- [LIVE_DATA_VALIDATION.md](LIVE_DATA_VALIDATION.md) for troubleshooting
