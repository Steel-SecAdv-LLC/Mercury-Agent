# Domain Performance Analysis

Per-domain precision/recall analysis from real benchmark data.

## Data Source

All metrics sourced from `benchmarks/honest_benchmark_results.json`,
specifically the `domain_summary` section. Run
`python benchmarks/honest_benchmark.py` to regenerate.

## Domain Summary Schema

Each domain entry in `domain_summary` contains:

```json
{
  "n_datasets": int,
  "n_measured": int,
  "n_below_random": int,
  "n_failed": int,
  "oracle_active_count": int,
  "stats": {
    "mean_auc": float | null,
    "median_auc": float | null,
    "std_auc": float,
    "mean_f1": float | null,
    "mean_precision": float | null,
    "mean_recall": float | null,
    "component_mean_aucs": {
      "resonance": float,
      "kinematic": float,
      "info_geometry": float
    },
    "best_component": str,
    "best_component_auc": float
  }
}
```

## Active Domains

| Domain | Datasets | Source Types |
|--------|----------|-------------|
| adbench | 47 | Local download (ADBench repository) |
| environmental | 4 | USGS, NOAA, MODIS APIs |
| ocean | 1 | NOAA NDBC buoy data |
| climate | 3 | NOAA Storm Events, GSOD, ERDDAP |
| air_quality | 1 | EPA AQS |
| disaster | 2 | FEMA OpenFEMA |
| space | 2 | NASA Exoplanet Archive, NOAA SWPC |
| academic | 3 | UCR Archive, CWRU, MSDS |
| security | 3 | NSL-KDD, CICIDS-2017, OSINT feeds |
| general | 1 | AD Repository |
| timeseries | 4 | SMD, NAB, SMAP, MSL |
| industrial | 3 | BATADAL, SWaT, WADI |
| medical | 1 | PhysioNet MIT-BIH |

## Component Performance by Domain

The benchmark tracks per-component AUC (Resonance, Kinematic,
InfoGeometry) for each domain. The `best_component` field identifies
which detection component performs best in each domain.

**Expected patterns:**

- **Temporal domains** (timeseries, environmental): Resonance
  (FFT-based) tends to outperform.
- **Tabular domains** (adbench, security): InfoGeometry
  (Mahalanobis-based) tends to outperform.
- **High-dimensional domains** (medical, industrial): Kinematic
  score provides complementary signal.

## Oracle Activation by Domain

The Oracle is auto-activated for temporal data based on the
`ORACLE_DOMAIN_POLICY` in `core/config.py`:

| Domain | Policy | Effect |
|--------|--------|--------|
| infrastructure | enabled | Full Oracle influence |
| security | enabled | Full Oracle influence |
| medical | enabled | Full Oracle influence |
| environmental | neutral | Dampened multiplier (0.5x) |
| space | neutral | Dampened multiplier (0.5x) |
| financial | disabled | Oracle skipped |
| humanitarian | disabled | Oracle skipped |

## Results

**Status: Populated after benchmark run.**

Run `python benchmarks/honest_benchmark.py` to generate results,
then `python scripts/generate_docs_images.py` to produce
visualisations from the measured data.

Every number in the generated charts is traceable to
`honest_benchmark_results.json`. No hardcoded or aspirational figures.
