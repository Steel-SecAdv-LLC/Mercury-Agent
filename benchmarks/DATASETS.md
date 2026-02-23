# Mercury Agent — Dataset Catalog

All active datasets used in `honest_benchmark.py`. Every metric in
benchmark outputs is traceable to these real data sources.

## ADBench Datasets (47)

Standard anomaly detection benchmark suite from the ADBench repository.
Downloaded automatically on first run.

| # | Dataset | Samples | Features | Anomaly % | Source |
|---|---------|---------|----------|-----------|--------|
| 1-47 | ADBench standard suite | Varies | Varies | Varies | [ADBench (NeurIPS 2022)](https://github.com/Minqi824/ADBench) |

License: MIT (ADBench framework). Individual datasets have original licenses.

## Domain Datasets (27+)

### Environmental (4)

| Dataset | Loader | Source API | Auth | Est. Samples | Features | Anomaly Labeling |
|---------|--------|-----------|------|-------------|----------|-----------------|
| USGS_Earthquake | USGSEarthquakeLoader | USGS FDSNWS | None | 1000-5000 | 8 | Magnitude threshold |
| NOAA_Weather | NOAAWeatherLoader | Open-Meteo | None | 500-2000 | 6 | Statistical extremes |
| Wildfire | WildfireDataLoader | (implemented) | None | 500-2000 | 5 | Fire detection |
| USGS_Geochemistry | USGSGeochemistryLoader | USGS MRDATA | None | 500-2000 | 10 | Concentration threshold |

### Ocean / Climate (4)

| Dataset | Loader | Source API | Auth | Est. Samples | Features | Anomaly Labeling |
|---------|--------|-----------|------|-------------|----------|-----------------|
| NOAA_Buoy | NOAABuoyLoader | NDBC Realtime | None | 500-2000 | 6 | Wave height threshold |
| NOAA_Storm_Events | NOAAStormEventsLoader | NCEI Storm Events | None | 500-5000 | 8 | Event severity |
| NOAA_GSOD | NOAAGSODLoader | NCEI GSOD Archive | None | 1000-5000 | 10 | Temperature extremes |
| NOAA_ERDDAP | NOAAERDDAPLoader | ERDDAP REST | None | 500-2000 | 6 | SST anomaly |

### Air Quality / Disaster (3)

| Dataset | Loader | Source API | Auth | Est. Samples | Features | Anomaly Labeling |
|---------|--------|-----------|------|-------------|----------|-----------------|
| EPA_Air_Quality | EPAAirQualityLoader | EPA AQS | None | 500-2000 | 6 | AQI threshold |
| FEMA_Disaster | FEMADisasterLoader | OpenFEMA | None | 500-5000 | 8 | Declaration type |
| FEMA_Hazard_Mitigation | FEMAHazardMitigationLoader | OpenFEMA | None | 500-2000 | 6 | Grant threshold |

### Space (2)

| Dataset | Loader | Source API | Auth | Est. Samples | Features | Anomaly Labeling |
|---------|--------|-----------|------|-------------|----------|-----------------|
| NASA_Exoplanet | NASAExoplanetLoader | NASA Exoplanet TAP | None | 500-5000 | 12 | Transit depth |
| Solar_Dynamics | SolarDynamicsLoader | NOAA SWPC JSON | None | 500-2000 | 8 | Solar event classification |

### Academic / Archive (3)

| Dataset | Loader | Source API | Auth | Est. Samples | Features | Anomaly Labeling |
|---------|--------|-----------|------|-------------|----------|-----------------|
| UCR_Archive | UCRLoader | UCR Archive | None | Varies | Varies | Dataset-specific |
| CWRU_Bearing | CWRUBearingLoader | CWRU Academic | None | 500-2000 | 12 | Fault type |
| MSDS | MSDSLoader | Academic | None | 500-2000 | 10 | Dataset-specific |

### Security (1)

| Dataset | Loader | Source API | Auth | Est. Samples | Features | Anomaly Labeling |
|---------|--------|-----------|------|-------------|----------|-----------------|
| Threat_Intel | ThreatIntelLoader | MITRE ATT&CK STIX | None | 500-5000 | 15 | Technique severity |

### General (1)

| Dataset | Loader | Source API | Auth | Est. Samples | Features | Anomaly Labeling |
|---------|--------|-----------|------|-------------|----------|-----------------|
| AD_Repository | ADRepositoryLoader | (implemented) | None | Varies | Varies | Dataset-specific |

### Industrial (2, conditional)

| Dataset | Loader | Source API | Auth | Est. Samples | Features | Anomaly Labeling |
|---------|--------|-----------|------|-------------|----------|-----------------|
| SWaT | SWaTLoader | iTrust Dataset | Verify | 500-5000 | 51 | Attack labels |
| WADI | WADILoader | iTrust Dataset | Verify | 500-5000 | 123 | Attack labels |

**Note**: SWaT and WADI require verification of access. When data is
unavailable, the benchmark records `{"status": "api_unavailable"}` and
continues.

## Data Quality Requirements

All loaded datasets are validated before benchmarking:

1. **Non-empty**: X must have at least 1 sample
2. **Finite values**: NaN and Inf are sanitised (replaced with 0)
3. **Label diversity**: y must contain at least 2 distinct values
4. **Shape consistency**: Feature count must be consistent across samples

Datasets failing validation are recorded with
`{"status": "invalid_data", "reason": "..."}` and skipped.

## Circuit Breaker Protection

Every API-sourced loader is wrapped with circuit breaker protection:

- **Connection timeout**: 30 seconds
- **Read timeout**: 120 seconds
- **Retry**: 3 attempts with exponential backoff (base 2 seconds)
- **Graceful degradation**: API failures logged as WARNING; dataset
  skipped with `{"status": "api_unavailable"}` in results

## CLI Flags

```bash
# Run all datasets (ADBench + domain)
python benchmarks/honest_benchmark.py

# Run only API-sourced domain datasets (skip ADBench download)
python benchmarks/honest_benchmark.py --live-only

# Filter by domain category
python benchmarks/honest_benchmark.py --domain environmental
python benchmarks/honest_benchmark.py --domain ocean
python benchmarks/honest_benchmark.py --domain space
```

## Citation

If using Mercury Agent's benchmark infrastructure in research, please cite:

```bibtex
@software{mercury_agent,
  title = {Mercury Agent: Anomaly Detection for First Responders},
  author = {Steel Security Advisors LLC},
  year = {2025},
  license = {GPL-3.0},
  url = {https://github.com/Steel-SecAdv-LLC/Mercury-Agent}
}
```
