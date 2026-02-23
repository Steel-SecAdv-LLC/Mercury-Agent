# Mercury Agent — Architecture Overview

## System Architecture

Mercury Agent is a production-grade anomaly detection engine serving first
responders and emergency managers. The system combines three original
mathematical detection frameworks with spectral-domain Oracle analysis,
cognitive reasoning modules, and cross-domain frequency correlation.

```
                    ┌─────────────────────────────┐
                    │     Data Ingestion Layer     │
                    │  41 Loaders (29 active)      │
                    │  Circuit Breaker Protection   │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │   Feature Extraction Layer    │
                    │  Data Type Auto-Detection     │
                    │  Oracle Domain Auto-Selection  │
                    └──────────┬──────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼───────┐ ┌─────▼──────┐ ┌──────▼─────────┐
    │  Resonance (40%) │ │Kinematic   │ │InfoGeometry    │
    │  FFT harmonic    │ │(30%) Jerk/ │ │(30%) Fisher    │
    │  spectral        │ │curvature   │ │info Mahalanobis│
    └─────────┬───────┘ └─────┬──────┘ └──────┬─────────┘
              │                │                │
              └────────────────┼────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │   SpectralDomainOracle        │
                    │  7 domains × 6-9 bands each   │
                    │  Selective Inference (SI)      │
                    │  Spectral Flux + Phase Coh.    │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │   Fusion Layer (OAE)          │
                    │  φ-weighted influence mult.    │
                    │  Ethical gating                │
                    └──────────┬──────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼───────┐ ┌─────▼──────┐ ┌──────▼─────────┐
    │ Cognitive Orch.  │ │ Cross-     │ │ Calibration    │
    │ 10 modules       │ │ Domain     │ │ Pipeline       │
    │ (advisory only)  │ │ Frequency  │ │ Youden/F1      │
    └─────────────────┘ │ Correlator │ └────────────────┘
                        └────────────┘
```

## Loader Registry

### Active Loaders (29 total: 9 original + 20 newly activated)

| Loader | Module | Source API | Auth | Domain |
|--------|--------|-----------|------|--------|
| ADBench (47 datasets) | datasets/adbench | UCI/OpenML | None | benchmark |
| USGSEarthquakeLoader | datasets/environmental | USGS FDSNWS | None | environmental |
| NOAAWeatherLoader | datasets/environmental | Open-Meteo | None | environmental |
| WildfireDataLoader | datasets/environmental | (implemented) | None | environmental |
| USGSGeochemistryLoader | datasets/environmental | USGS MRDATA | None | environmental |
| NOAABuoyLoader | datasets/ocean | NDBC Realtime | None | ocean |
| NOAAStormEventsLoader | datasets/noaa_storm | NCEI Storm Events | None | climate |
| NOAAGSODLoader | datasets/noaa_gsod | NCEI GSOD Archive | None | climate |
| NOAAERDDAPLoader | datasets/noaa_erddap | ERDDAP REST | None | ocean |
| EPAAirQualityLoader | datasets/epa_air | EPA AQS | None | environmental |
| FEMADisasterLoader | datasets/disaster | OpenFEMA | None | disaster |
| FEMAHazardMitigationLoader | datasets/disaster | OpenFEMA | None | disaster |
| NASAExoplanetLoader | datasets/space | NASA Exoplanet TAP | None | space |
| SolarDynamicsLoader | datasets/space | NOAA SWPC JSON | None | space |
| UCRLoader | datasets/ucr_archive | UCR Archive | None | academic |
| CWRUBearingLoader | datasets/ucr_archive | CWRU Academic | None | academic |
| MSDSLoader | datasets/ucr_archive | Academic | None | academic |
| ThreatIntelLoader | datasets/security | MITRE ATT&CK STIX | None | security |
| ADRepositoryLoader | datasets/adrepository | (implemented) | None | general |
| SWaTLoader | datasets/industrial | iTrust Dataset | Verify | industrial |
| WADILoader | datasets/industrial | iTrust Dataset | Verify | industrial |
| MIMICLoader | datasets/medical | MIMIC-III | Key | medical |
| NOAASpaceWeatherLoader | datasets/space | NOAA SWPC | None | space |
| NOAAHurricaneLoader | datasets/noaa_hurricane | NOAA IBTrACS | None | climate |
| NOAAOceanLoader | datasets/ocean | NOAA CO-OPS | None | ocean |
| NSLKDDLoader | datasets/security | NSL-KDD | None | security |
| FinancialLoader | datasets/financial | Yahoo Finance | None | financial |
| MedicalLoader | datasets/medical | (various) | Varies | medical |
| InfrastructureLoader | datasets/industrial | (various) | Varies | industrial |

### Dormant Loaders (12 remaining — require credentials)

Loaders requiring API keys or institutional access are present but
conditionally skipped when credentials are unavailable. They degrade
gracefully with `{"status": "api_unavailable"}` in benchmark output.

## Detection Ensemble

### Core Detectors (MercuryAnomalyDetector)

| Component | Weight | Method | Provenance |
|-----------|--------|--------|------------|
| Resonance Score | 40% | FFT harmonic spectral anomaly | Mercury original |
| Kinematic Score | 30% | Physics-based jerk/curvature | Mercury original |
| InfoGeometry Score | 30% | Fisher Information OOD | IGEOOD (ICLR 2022) |

Weights are adaptive — `_compute_unsupervised_adaptive_weights()` adjusts
based on data characteristics. Data type detection selects domain-optimal
defaults (temporal, tabular, image, unknown).

### SpectralDomainOracle (7 domains)

| Domain | Bands | Key Frequencies |
|--------|-------|-----------------|
| environmental | 8 | Schumann 7.83 Hz, infrasound, VLF |
| medical | 9 | HRV bands, neural theta/alpha/beta/gamma |
| infrastructure | 8 | Seismic, mains 50/60 Hz, bearing fault |
| security | 6 | Baseline to ultra-high-rate patterns |
| financial | 7 | Macro cycles to microstructure noise |
| space | 7 | Solar cycle to whistler modes |
| humanitarian | 5 | Population movement to alert propagation |

## Cognitive Module Topology

The `CognitiveOrchestrator` wires 10 modules, all **advisory only** —
they augment the analysis result but never overwrite detector scores.

| Module | Status | Integration Point | Output |
|--------|--------|-------------------|--------|
| PlasticityEngine | ACTIVE | Post-detection | Adaptation recommendations |
| CausalDiscoveryEngine | ACTIVE | Post-detection | Causal factor analysis |
| MultiHopReasoner | ACTIVE | Post-detection | Multi-step reasoning chains |
| UncertaintyQuantifier | ACTIVE | Post-detection | Uncertainty bounds |
| CaseBasedReasoner | ACTIVE | Post-detection | Similar historical cases |
| IndicatorDevelopmentSystem | ACTIVE | Post-detection | Leading indicators |
| IPBEngine | ACTIVE | Post-detection | Intent-Plan-Behaviour analysis |
| KnowledgeGraph | ACTIVE | Post-detection | Entity relationship context |
| CognitiveEvolutionEngine | ACTIVE | Post-detection | Chain-of-thought reasoning |
| MercuryPredictiveCoding | ACTIVE | Post-detection | Prediction error analysis |

## Backend Configuration

All backends default to zero-dependency local implementations. Override
via environment variables for production deployments.

| Backend | Default | Override Env Var | Production |
|---------|---------|-----------------|------------|
| Database | SQLite | `MERCURY_DB_BACKEND=postgresql` | PostgreSQL |
| Cache | In-memory LRU | `MERCURY_CACHE_BACKEND=redis` | Redis |
| Weather | Open-Meteo (free) | `MERCURY_WEATHER_BACKEND=openweathermap` | OpenWeatherMap |
| Financial | Yahoo Finance | `MERCURY_FINANCIAL_BACKEND=alphavantage` | Alpha Vantage |

### Cache TTL by Domain

| Domain | TTL (seconds) | Rationale |
|--------|--------------|-----------|
| environmental | 300 (5 min) | Sensor data updates frequently |
| ocean | 600 (10 min) | Buoy data update interval |
| climate | 3600 (1 hour) | Historical data, slow-changing |
| financial | 900 (15 min) | Market data latency tolerance |
| security | 60 (1 min) | Threat intel requires freshness |
| space | 1800 (30 min) | Solar wind data cadence |
| medical | 3600 (1 hour) | Clinical data, batch-updated |

### Database Schema

Schema at `schema/mercury.sql` with tables:
- `benchmark_runs` — Run metadata (ID, timestamp, git SHA, config)
- `dataset_results` — Per-dataset metrics (AUC, F1, precision, recall)
- `detector_state` — Serialised detector state for federation
- `api_cache` — API response cache with TTL expiry

## Cross-Domain Frequency Correlation

The `CrossDomainFrequencyCorrelator` detects spectral-band overlap
between concurrent Oracle instances. Primary validation path: Schumann
Resonance (7.83 Hz fundamental) cross-referenced between environmental
(USGS seismic) and space (NOAA solar weather) domains.

**CRITICAL**: This module provides CORRELATION only — never causation or
prediction. All outputs include mandatory human-assessment disclaimer.

## Federation

Federated learning is supported via `from_statistics()` /
`get_oracle_statistics()`. Nodes export detector statistics and Oracle
reference state; the aggregator combines them into a working detector.

## Health Checks

| Endpoint | Purpose |
|----------|---------|
| GET /health/liveness | Kubernetes liveness probe |
| GET /health/readiness | Kubernetes readiness probe |
| GET /health/startup | Kubernetes startup probe |
| GET /health/detailed | Full component health status |
| GET /health/metrics | Prometheus-format metrics |
