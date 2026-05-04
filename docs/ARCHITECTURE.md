# Mercury Agent - System Activation Architecture

Supplement to the top-level `ARCHITECTURE.md`. Covers the loader
registry, Oracle pipeline, cognitive wiring, backend configuration,
and cross-domain correlation introduced during the Mercury System
Activation.

## Dataset Loader Registry

### Module Structure

```
src/omni_mercury_engine/datasets/
├── adbench.py          # ADBenchLoader (47 tabular datasets)
├── adrepository.py     # ADRepositoryLoader
├── base.py             # DatasetConfig, BaseLoader
├── climate.py          # Climate datasets
├── disaster.py         # FEMADisasterLoader, FEMAHazardMitigationLoader
├── environmental.py    # USGSEarthquakeLoader, NOAAWeatherLoader,
│                       # WildfireDataLoader, USGSGeochemistryLoader
├── epa_air.py          # EPAAirQualityLoader
├── industrial.py       # BATADALLoader, SWaTLoader, WADILoader
├── medical.py          # MIMICLoader
├── mitbih.py           # MITBIHLoader
├── noaa_erddap.py      # NOAAERDDAPLoader
├── noaa_gsod.py        # NOAAGSODLoader
├── noaa_storm.py       # NOAAStormEventsLoader
├── ocean.py            # NOAABuoyLoader
├── security.py         # NSLKDDLoader, CICIDSLoader, ThreatIntelLoader
├── space.py            # NASAExoplanetLoader, SolarDynamicsLoader
├── timeseries.py       # SMDLoader, NABLoader, SMAPMSLLoader
└── ucr_archive.py      # UCRLoader, CWRUBearingLoader, MSDSLoader
```

### Benchmark Registration

Each domain dataset is registered in `benchmarks/mercury_benchmark.py`
via the `DOMAIN_DATASETS` list:

```python
DOMAIN_DATASETS: list[tuple[str, str, str, str, dict[str, Any]]] = [
    # (name, category, loader_class_name, module, kwargs)
    ("USGS_Earthquake", "environmental", "USGSEarthquakeLoader", "environmental", {}),
    ...
]
```

### Resilience Stack

Every API-sourced loader is protected by:

1. **Circuit breaker** (`resilience/api_circuit_breakers.py`):
   3 failure threshold, 30s recovery, exponential backoff.
2. **Retry with backoff** (3 attempts, base 2s).
3. **Graceful degradation**: API failures logged as
   `{"status": "api_unavailable"}` — benchmark continues.
4. **Data validation**: Non-empty X, no NaN/Inf, ≥2 distinct labels.

## Oracle Pipeline

### SpectralDomainOracle Integration

```
MercuryAnomalyDetector.fit(data)
    │
    ├─ _detect_data_characteristics(data) → DataCharacteristics
    ├─ _infer_oracle_domain(data, type) → domain string
    │
    ├─ if TEMPORAL and ORACLE_DOMAIN_POLICY[domain] != "disabled":
    │   ├─ SpectralDomainOracleConfig(domain=inferred_domain)
    │   ├─ SpectralDomainOracle(config)
    │   └─ oracle.fit(data)
    │
    └─ _compute_unsupervised_adaptive_weights(data)

MercuryAnomalyDetector.detect(data)
    │
    ├─ resonance = _compute_resonance_score(data)
    ├─ kinematic = _compute_kinematic_score(data)
    ├─ info_geo  = _compute_info_geometry_score(data)
    │
    ├─ combined = weights[0]*res + weights[1]*kin + weights[2]*ig
    │
    ├─ if oracle_detector is not None:
    │   ├─ oracle_result = oracle.detect(data)
    │   ├─ multiplier = influence_vector.influence_multiplier
    │   └─ combined = combined * multiplier
    │
    └─ return {..., "oracle_metadata": {...}}
```

### Oracle Domain Auto-Selection

`_infer_oracle_domain(X, detected_type)` uses:

| Heuristic | Condition | Domain |
|-----------|-----------|--------|
| Feature count | n_features >= 20 | security |
| Dominant FFT freq | < 0.05 (normalised) | environmental |
| Dominant FFT freq | 0.05-0.2 | medical |
| Dominant FFT freq | 0.2-0.4 | infrastructure |
| Dominant FFT freq | > 0.4 | security |
| Feature count | 1-3 | environmental |
| Feature count | 4-10 | space |
| Fallback | — | environmental |

User-specified domain always overrides.

### Oracle Domain Policy

From `core/config.py`:

| Domain | Policy | Effect |
|--------|--------|--------|
| infrastructure | enabled | Full influence multiplier |
| security | enabled | Full influence multiplier |
| medical | enabled | Full influence multiplier |
| environmental | neutral | Dampened (0.5x) |
| space | neutral | Dampened (0.5x) |
| financial | disabled | Oracle skipped |
| humanitarian | disabled | Oracle skipped |

## Neuro-Symbolic Fusion (NSAI Taxonomy)

Mercury's neuro-symbolic fusion is implemented at two levels:

- **Hub level** (`core/neurosymbolic_hub.py`): combines a single neural
  score with a single symbolic score per sample.
- **Ensemble level** (`core/stacking_fusion.py`): combines `N` detector
  predictions for stacking / BMA / phi-weighted ethical fusion.

### Top-level fusion mode: FIBRING (default)

The hub-level default is `FusionMode.FIBRING`. In the NSAI taxonomy
(Garcez & Lamb 2020; Sarker et al. 2021) "fibring" denotes the
architectural pattern in which one reasoning system is *fibred* over
another — a hierarchical composition rather than a sequential pipeline
or independent parallel branches. Mercury already implemented every
piece of the pattern; FIBRING simply names the composition.

The composer is `core/fibring_fusion.py::FibringComposer` and it stacks
three primitives:

1. **Phi-weighted base** — golden-ratio split between neural and
   symbolic (`φ/(1+φ) ≈ 0.618`, `1/(1+φ) ≈ 0.382`).
2. **Correlation-aware decorrelation** — a sliding window tracks recent
   `(neural, symbolic)` pairs; once the window has at least
   `MIN_SAMPLES_FOR_DECORRELATION` entries and `|Pearson r| ≥ 0.85`,
   the lower-variance (redundant) component is shrunk by `1 / (1 + |r|)`.
   Mirrors the math_arrest `CorrelationAwareDecorrelator` already used
   for the 21-probe ensemble.
3. **Per-domain affinity bias** — table in
   `core/fibring_fusion.py::DOMAIN_AFFINITY_BIAS` tilts weights toward
   the modality empirically stronger for the domain (medical, ethical,
   conflict → symbolic; geomagnetic, earthquake, tsunami, marine,
   pandemic, financial → neural). Bias is then renormalised so the
   final weights still sum to 1.

The composition is **causal**: weights for sample `t` are computed from
window contents at time `t-1`; the new pair is appended only after
composition.

The ensemble-level factory `create_fusion_ensemble(method="fibring")`
returns the existing `EthicallyConstrainedFusion` with `use_golden_ratio=True`,
which is the natural ensemble-level dual of the hub-level FIBRING mode.

Tests: `tests/core/test_fibring_default.py` pins default routing,
composer behaviour, and an ablation against BALANCED on a deterministic
channel-symmetric synthetic workload (no AUROC or Brier regression).

## Cognitive Module Wiring

```
TruthDecipherFramework
    └─ CognitiveOrchestrator.analyze(detection_result, raw_data, context)
        ├─ Step 1: UncertaintyQuantifier    (ALWAYS)
        ├─ Step 2: KnowledgeGraph           (ALWAYS)
        ├─ Step 3: MultiHopReasoner         (ALWAYS)
        ├─ Step 4: CausalDiscoveryEngine    (optional, default ON)
        ├─ Step 5: CaseBasedReasoner        (optional, default ON)
        ├─ Step 6: PlasticityEngine         (optional, default ON)
        ├─ Step 7: IndicatorDevelopmentSystem (optional, default ON)
        └─ Step 8: IPBEngine                (optional, default ON)
```

All cognitive modules are advisory — they augment results, never
overwrite detector scores. Failure in any module is caught and logged;
detection continues without cognitive enrichment.

## Integration Backend Configuration

### Database (`integrations/stubs/database.py`)

| Setting | Default | Override |
|---------|---------|----------|
| Backend | SQLite (zero deps) | `DATABASE_BACKEND=postgresql` |
| Database | `mercury.db` | `DATABASE_NAME=<path>` |
| Host | — | `DATABASE_HOST=<host>` |

Health check: `await db.health_check()` → `{"status": "healthy"|"degraded"|"unhealthy", "latency_ms": float}`

### Cache (`integrations/stubs/cache.py`)

| Setting | Default | Override |
|---------|---------|----------|
| Backend | In-memory LRU | `REDIS_HOST=<host>` for Redis |
| Domain TTL | `get_domain_ttl(domain)` | Per-domain seconds |

**Domain-specific TTL**:

| Domain | TTL (seconds) |
|--------|--------------|
| environmental | 300 |
| security | 60 |
| climate | 3600 |
| medical | 600 |
| space | 1800 |
| financial | 120 |
| industrial | 600 |
| default | 600 |

### Weather (`integrations/stubs/weather.py`)

| Setting | Default | Override |
|---------|---------|----------|
| Provider | Open-Meteo (free, no key) | `provider="openweathermap"` |
| API Key | — | Required for OpenWeatherMap |

### Financial (`integrations/stubs/financial.py`)

| Setting | Default | Override |
|---------|---------|----------|
| Provider | Yahoo Finance (no key) | Alpha Vantage with API key |

## Cross-Domain Frequency Correlation

```
CrossDomainFrequencyCorrelator.correlate({
    "environmental": env_influence_vector,
    "space":         space_influence_vector,
})
    │
    ├─ Extract bands from each domain's FrequencyInfluenceVector
    ├─ Check Hz-range overlaps for all domain pairs
    ├─ Compute geometric mean of scores for overlapping bands
    ├─ Filter significant overlaps (strength >= 0.3)
    └─ Return CrossDomainCorrelation(alert_level, description)
```

**CRITICAL**: Correlation only. Every output states
"requires human assessment."

## Federation Oracle Serialization

```python
# Export
stats = detector.get_oracle_statistics()
# → {"domain": "environmental", "ref_mean_power": [...], ...}

# Reconstruct
federated = MercuryAnomalyDetector.from_statistics(
    ...,
    oracle_ref_stats=stats,
)
# Oracle restored without re-fitting
```
