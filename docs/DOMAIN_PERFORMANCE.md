# Domain Performance Analysis

Applies to Mercury Agent **v2.1.x**. Last updated: 2026-07-11.

Per-domain precision/recall analysis from real benchmark data.

> **Authoritative source.** The per-domain tables below are an older
> illustrative snapshot. The committed
> `benchmarks/mercury_benchmark_results.json` run (66 successful / 75
> attempted under the `genuine_labels_only` headline policy,
> n_genuine_labeled 53, Mean AUC 0.8251, Median 0.8747, Mean Oracle F1
> 0.5998, Median Oracle F1 0.6747, and an unsupervised subset of 13
> datasets at Mean AUC 0.9281, 2026-06-21) is surfaced in the README
> "Latest Benchmark Results" block and is the authoritative source; its
> `domain_summary` is the per-domain ground truth. The CI regression
> gate applies coarse headline backstop floors of ROC-AUC ≥ 0.75 and
> Mean Oracle F1 ≥ 0.55 (`MIN_ROC_AUC`/`MIN_F1` in
> `.github/workflows/benchmark.yml`, holding a ~9% margin under the
> post-de-leak baseline mean AUC 0.8259 / mean F1 0.6046); fine-grained,
> per-dataset regression protection is provided by the deterministic
> `benchmarks/anomaly_regression_guard.py`, not these coarse floors.
>
> Historical note (FEMA polarity fix, v1.7.0): the FEMA Disaster row's
> former pre-fix AUC ≈ 0 was corrected in v1.7.0
> (`FEMADisasterLoader._select_anomaly_polarity` enforces the
> minority-as-anomaly convention); the committed run reflects the
> corrected score (disaster AUC 0.9993).

## Data Source

All metrics sourced from `benchmarks/mercury_benchmark_results.json`,
specifically the `domain_summary` section. Run
`python benchmarks/mercury_benchmark.py` to regenerate.

## Domain Summary Schema

Each domain entry in `domain_summary` contains the following fields
(type annotations shown in JSON-schema-like shorthand — `null`
indicates the field is emitted as `null` when no measurement is
available, e.g. when every dataset in the domain `n_failed`'d):

```text
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

The `Datasets` column below reports the measured `n_datasets` from the
committed run's `domain_summary`, consistent with the "All metrics
sourced from `domain_summary`" statement above. Load availability
varies per run, so these counts reflect what actually loaded and
measured in the committed 2026-06-21 run, not configured-loader
capacity.

| Domain | Datasets | Source Types |
|--------|----------|-------------|
| adbench | 47 | Local download (ADBench repository) |
| environmental | 3 | USGS, NOAA, MODIS APIs |
| ocean | 1 | NOAA NDBC buoy data |
| climate | 3 | NOAA Storm Events, GSOD, ERDDAP |
| air_quality | 1 | EPA AQS |
| disaster | 1 | FEMA OpenFEMA |
| space | 2 | NASA Exoplanet Archive, NOAA SWPC |
| academic | 2 | UCR Archive, CWRU, MSDS |
| security | 2 | NSL-KDD, CICIDS-2017, OSINT feeds |
| general | 1 | AD Repository |
| timeseries | 2 | SMD, NAB, SMAP, MSL |
| industrial | 1 | BATADAL, SWaT, WADI |

The `medical` domain (PhysioNet MIT-BIH) is configured in
`ORACLE_DOMAIN_POLICY` but produced no measured dataset in the
committed run; it is therefore absent from that run's `domain_summary`
and is not listed as an active/measured domain here.

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

## F1 Precision Improvements

### Threshold Strategy Analysis

The multi-strategy threshold selection evaluates 4 classes of strategies:

1. **Percentile-based** (85th-99th): Works well for datasets with clear score separation
2. **MAD-based** (k=2.0-4.0): Robust to outliers, good for heavy-tailed distributions
3. **Contamination-aware**: Uses actual anomaly ratio — strongest for known contamination
4. **Linear sweep**: Baseline 101-point grid — catches edge cases

### Domain Weight Presets

Derived from measured component AUCs across benchmark datasets:

| Domain | Resonance | Kinematic | InfoGeometry | Rationale |
|--------|-----------|-----------|-------------|-----------|
| disaster | 0.30 | 0.00 | 0.70 | Tabular: kinematic adds noise |
| general | 0.30 | 0.00 | 0.70 | Tabular: kinematic adds noise |
| academic | 0.47 | 0.03 | 0.50 | Mostly tabular, minimal temporal |
| security | 0.47 | 0.00 | 0.53 | Packet data: no physics dynamics |
| industrial | 0.47 | 0.00 | 0.53 | Mostly sensor thresholds |
| ocean | 0.30 | 0.40 | 0.30 | Wave physics: kinematic valuable |
| climate | 0.35 | 0.30 | 0.35 | Temporal dynamics matter |
| air_quality | 0.35 | 0.30 | 0.35 | Temporal dynamics matter |
| environmental | 0.35 | 0.30 | 0.35 | Geophysical dynamics |
| space | 0.30 | 0.35 | 0.35 | Solar wind: physics-rich |
| timeseries | 0.35 | 0.20 | 0.45 | Mixed: moderate kinematic |
| adbench | 0.40 | 0.15 | 0.45 | Heterogeneous: conservative |
| default | 0.40 | 0.20 | 0.40 | Fallback: balanced |

### Noise Color Estimates by Domain

| Domain | Expected Beta Range | Noise Type |
|--------|-------------------|------------|
| environmental | 0.5 - 2.0 | Pink to Brown |
| ocean | 1.0 - 2.5 | Pink to Brown+ |
| space | 0.5 - 2.0 | Pink to Brown |
| security | -0.5 - 0.5 | White |
| medical | 0.5 - 1.5 | Pink |
| climate | 0.5 - 2.0 | Pink to Brown |
| adbench | -0.5 - 1.0 | White to Pink |

### Score Pipeline Architecture

```
Raw Data → fit()
  ├── Statistical baselines (mean, std, quartiles)
  ├── Kinematic baselines (jerk, acceleration)
  ├── InfoGeometry manifold (precision matrix)
  └── Oracle reference spectrum (PSD, noise color)

Test Data → detect()
  ├── Resonance scores (FFT harmonic deviation)
  ├── Kinematic scores (jerk/acceleration z-scores)
  ├── InfoGeometry scores (Mahalanobis distance)
  │
  ├── Domain weight preset blending (40% prior)
  ├── Spearman inversion guard (zero anti-correlated)
  ├── Weighted ensemble combination
  ├── Median-based ensemble flip (if median > 0.80)
  ├── Oracle frequency-domain multiplier
  ├── Residual frequency filter (30% blend)
  │
  └── Threshold → anomaly predictions
```

## Results

**Status: Populated after benchmark run.**

Run `python benchmarks/mercury_benchmark.py` to generate results,
then `python scripts/generate_docs_images.py` to produce
visualisations from the measured data.

Every number in the generated charts is traceable to
`mercury_benchmark_results.json`. No hardcoded or aspirational figures.
