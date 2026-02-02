# Mercury Agent Benchmark Documentation

This document describes the comprehensive benchmark suite for Mercury Agent, covering 30+ real-world datasets across 7 domain categories.

## Overview

Mercury Agent's benchmark suite validates anomaly detection performance across diverse real-world scenarios:

| Category | Datasets | Data Sources | Focus |
|----------|----------|--------------|-------|
| Security | 2 | NSL-KDD, CICIDS | Network intrusion detection |
| Industrial | 3 | BATADAL, SWaT, WADI | Cyber-physical systems |
| Time-Series | 3 | SMD, NAB, SMAP/MSL | Server & IoT monitoring |
| Climate | 3 | Simons CMAP, WOD, Copernicus | Ocean & atmosphere |
| Disaster | 2 | FEMA | Emergency management |
| Environmental | 3 | USGS, NOAA | Seismic, weather, contamination |
| AD Repository | 21+ | Kaggle, UCI | Standard AD benchmarks |

## Running Benchmarks

### Quick Start

```bash
# Run all benchmarks
python benchmarks/live_dataset_benchmark.py

# Run specific category
python benchmarks/live_dataset_benchmark.py --category security

# Export results to JSON
python benchmarks/live_dataset_benchmark.py --output results.json

# Verbose mode with full logging
python benchmarks/live_dataset_benchmark.py -v
```

### Benchmark Scripts

| Script | Purpose | Time Estimate |
|--------|---------|---------------|
| `live_dataset_benchmark.py` | Full live dataset evaluation | 15-30 min |
| `empirical_benchmark.py` | Empirical validation suite | 10-20 min |
| `neuro_symbolic_benchmark.py` | Neuro-symbolic feature testing | 5-10 min |
| `comprehensive_benchmark.py` | All components | 30-60 min |

## Dataset Details

### Security Datasets

#### NSL-KDD (Network Intrusion Detection)
- **Source**: Canadian Institute for Cybersecurity
- **Samples**: 125,973 (train) + 22,544 (test)
- **Features**: 41 network connection attributes
- **Anomaly Types**: DoS, Probe, R2L, U2R attacks
- **Anomaly Ratio**: ~20%
- **License**: Research use

```python
from omni_mercury_engine.datasets import NSLKDDLoader
loader = NSLKDDLoader()
data = loader.load()
```

#### CICIDS-2017 (Modern Network Attacks)
- **Source**: Canadian Institute for Cybersecurity
- **Samples**: 2.8M+ flows
- **Features**: 78 flow-based features
- **Anomaly Types**: Brute Force, Web Attacks, Infiltration, Botnet, DDoS
- **Anomaly Ratio**: ~19%

### Industrial Datasets

#### BATADAL (Water Infrastructure)
- **Source**: Battle of the Attack Detection Algorithms
- **Samples**: 43,200 (train) + 17,280 (test)
- **Features**: 43 SCADA sensor readings
- **Anomaly Types**: 7 cyber-physical attack scenarios
- **Anomaly Ratio**: ~10%

#### SWaT (Secure Water Treatment)
- **Source**: Singapore University of Technology and Design
- **Samples**: 946,722
- **Features**: 51 sensors
- **Anomaly Types**: 36 attack scenarios
- **Anomaly Ratio**: ~12%
- **Access**: Requires registration

#### WADI (Water Distribution)
- **Source**: iTrust Labs
- **Samples**: 1,209,601
- **Features**: 123 sensors
- **Anomaly Types**: 15 attack scenarios
- **Anomaly Ratio**: ~6%
- **Access**: Requires registration

### Time-Series Datasets

#### SMD (Server Machine Dataset)
- **Source**: Tsinghua University
- **Samples**: 1.4M+ data points across 28 machines
- **Features**: 38 server metrics per machine
- **Anomaly Types**: Point and contextual anomalies
- **Anomaly Ratio**: ~5%

#### NAB (Numenta Anomaly Benchmark)
- **Source**: Numenta Inc.
- **Samples**: 365,558 data points across 58 files
- **Features**: Univariate time series
- **Anomaly Types**: Known anomalies with labels
- **Anomaly Ratio**: Variable

#### SMAP/MSL (NASA Spacecraft Telemetry)
- **Source**: NASA Jet Propulsion Laboratory
- **Samples**: 427,617 (SMAP) + 132,046 (MSL)
- **Features**: 25 (SMAP) + 55 (MSL) telemetry channels
- **Anomaly Types**: Spacecraft anomalies
- **Anomaly Ratio**: ~13%

### Climate & Ocean Datasets

#### Simons CMAP (Ocean Biogeochemistry)
- **Source**: Simons Collaborative Marine Atlas Project
- **Data**: Satellite observations, in-situ measurements, model outputs
- **Coverage**: Global ocean biogeochemistry
- **Features**: Chlorophyll, nutrients, temperature, salinity
- **Access**: Free API

```python
from omni_mercury_engine.datasets import SimonsCMAPLoader
loader = SimonsCMAPLoader()
data = loader.load(variable="chlorophyll", depth_range=(0, 100))
```

#### World Ocean Database
- **Source**: NOAA National Centers for Environmental Information
- **Data**: 20M+ ocean profiles (1770-present)
- **Features**: Temperature, salinity, oxygen, nutrients
- **Coverage**: Global ocean observations

#### Copernicus Sea Level
- **Source**: EU Copernicus Climate Data Store
- **Data**: Satellite altimetry (1993-present)
- **Resolution**: 0.25 degree
- **Features**: Sea surface height, trends

### Disaster Datasets

#### FEMA Disaster Declarations
- **Source**: OpenFEMA API
- **Data**: All US disaster declarations
- **Types**: Hurricanes, floods, fires, earthquakes
- **Coverage**: 1953-present
- **Access**: Free, no API key required

```python
from omni_mercury_engine.datasets import FEMADisasterLoader
loader = FEMADisasterLoader()
data = loader.load(start_date="2020-01-01", disaster_types=["Hurricane"])
```

#### FEMA Hazard Mitigation
- **Source**: OpenFEMA API
- **Data**: Hazard mitigation grant program records
- **Features**: Project costs, types, outcomes

### Environmental Datasets

#### USGS Earthquake
- **Source**: USGS Earthquake Hazards Program
- **Data**: Global earthquake catalog
- **Features**: Magnitude, location, depth, time
- **Coverage**: Real-time + historical

#### USGS Geochemistry
- **Source**: USGS Mineral Resources Data System
- **Data**: Heavy metal concentrations
- **Features**: As, Pb, Hg, Cu, Zn concentrations
- **Anomaly Labels**: EPA Regional Screening Levels

#### NOAA Weather
- **Source**: NOAA National Weather Service
- **Data**: Weather observations and forecasts
- **Features**: Temperature, precipitation, wind
- **Coverage**: US stations

### AD Repository Datasets (21+)

Mercury Agent integrates with the Anomaly Detection Repository, providing access to standard benchmark datasets:

| Dataset | Domain | Samples | Features | Anomaly % |
|---------|--------|---------|----------|-----------|
| fraud | Finance | 284,807 | 30 | 0.17% |
| thyroid | Medical | 3,772 | 6 | 2.5% |
| mammography | Medical | 11,183 | 6 | 2.3% |
| campaign | Marketing | 41,188 | 62 | 11.3% |
| backdoor | Security | 95,329 | 196 | 2.4% |
| satellite | Remote Sensing | 6,435 | 36 | 32% |
| shuttle | Space | 49,097 | 9 | 7% |
| pima | Medical | 768 | 8 | 35% |
| wine | Chemistry | 129 | 13 | 7.7% |
| glass | Materials | 214 | 9 | 4.2% |

```python
from omni_mercury_engine.datasets import load_dataset, list_available_datasets

# List all available datasets
print(list_available_datasets())

# Load a specific dataset
X, y, metadata = load_dataset('fraud')
```

## Benchmark Metrics

### Core Metrics

| Metric | Description | Range | Better |
|--------|-------------|-------|--------|
| ROC-AUC | Area under ROC curve | [0, 1] | Higher |
| PR-AUC | Average precision | [0, 1] | Higher |
| F1 Score | Harmonic mean of P & R | [0, 1] | Higher |
| Precision | True positives / Predicted positives | [0, 1] | Higher |
| Recall | True positives / Actual positives | [0, 1] | Higher |

### Time-Series Metrics

| Metric | Description | Application |
|--------|-------------|-------------|
| Event F1 | F1 on event segments | Contiguous anomaly detection |
| Time-to-Detection | Delay to first detection | Real-time systems |
| Point-Adjusted F1 | Lenient point matching | Time-series alignment |

### Ethical Metrics

| Metric | Description | Threshold |
|--------|-------------|-----------|
| Benevolence Score | Ethical compliance | >= 0.99 |
| Sigma Immutable | Stability measure | >= 0.96 |
| Fairness Ratio | Demographic parity | >= 0.80 |

## Expected Performance

Based on validated benchmarks with Mercury Agent v1.2.0:

| Dataset Category | Mean F1 | Mean ROC-AUC | Notes |
|-----------------|---------|--------------|-------|
| Security (NSL-KDD) | 0.75-0.85 | 0.90-0.95 | With 3R enabled |
| Industrial (BATADAL) | 0.30-0.45 | 0.60-0.75 | Highly imbalanced |
| Time-Series (SMD) | 0.15-0.25 | 0.55-0.70 | Challenging |
| AD Repository | 0.70-0.90 | 0.85-0.98 | Varies by dataset |

### Comparison with Baselines

| Detector | Mean F1 | Notes |
|----------|---------|-------|
| Mercury Agent (3R+Fusion) | 0.80 | Interpretable |
| IsolationForest | 0.75 | Fast, less interpretable |
| LOF | 0.65 | Distance-based |
| One-Class SVM | 0.60 | Kernel-based |

## Data Provenance

All benchmark results include provenance tracking:

```json
{
  "provenance": {
    "source": "live",
    "checksum": "sha256:abc123...",
    "used_synthetic": false,
    "n_samples": 125973,
    "n_features": 41,
    "anomaly_ratio": 0.20
  }
}
```

- `source`: "live" (real API/file), "synthetic" (generated fallback)
- `checksum`: SHA-256 hash for reproducibility
- `used_synthetic`: Whether fallback data was used

## Synthetic Fallbacks

When real data is unavailable (network issues, credentials required), the benchmark suite generates synthetic data that mimics the statistical properties of real datasets:

```python
class SyntheticDataGenerator:
    def generate_smd_like(self, n_samples=5000, n_features=38):
        """Generate data with SMD-like characteristics."""
        # Temporal correlations
        # Periodic patterns
        # ~5% anomaly rate with point/contextual anomalies
```

Synthetic data is clearly marked in results for transparency.

## Reproducing Results

### Full Reproducibility

```bash
# Set random seed
export MERCURY_SEED=42

# Run with fixed seed
python benchmarks/live_dataset_benchmark.py --seed 42 --output results_v1.2.0.json

# Verify reproducibility
python benchmarks/live_dataset_benchmark.py --seed 42 --output results_v1.2.0_verify.json
diff results_v1.2.0.json results_v1.2.0_verify.json
```

### Dataset Access

Some datasets require registration or credentials:

| Dataset | Access | Registration |
|---------|--------|--------------|
| SWaT, WADI | Registration required | iTrust Labs |
| MIMIC-III/IV | Credentials required | PhysioNet |
| NSL-KDD | Free download | UNB |
| CICIDS-2017 | Free download | UNB |

## Continuous Integration

Benchmarks run automatically in CI:

```yaml
# .github/workflows/benchmark.yml
- name: Run benchmark suite
  run: |
    python benchmarks/live_dataset_benchmark.py \
      --max-samples 1000 \
      --output benchmark_results.json
```

Results are archived and compared against baselines for regression detection.

## Contributing Datasets

To add a new dataset to the benchmark suite:

1. Create a loader in `src/omni_mercury_engine/datasets/`
2. Implement the `DatasetLoader` interface
3. Add to `DATASET_REGISTRY` in `live_dataset_benchmark.py`
4. Add documentation to this file
5. Submit a pull request

```python
class MyDatasetLoader(DatasetLoader):
    def load(self) -> DatasetSplit:
        # Load and return data
        pass
```

## References

1. NSL-KDD: Tavallaee et al., "A Detailed Analysis of the KDD CUP 99 Data Set", IEEE CISDA 2009
2. BATADAL: Taormina et al., "Battle of the Attack Detection Algorithms", J. Water Resources Planning 2018
3. SMD: Su et al., "Robust Anomaly Detection for Multivariate Time Series", KDD 2019
4. SMAP/MSL: Hundman et al., "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding", KDD 2018
