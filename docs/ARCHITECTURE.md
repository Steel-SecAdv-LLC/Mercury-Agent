# Mercury Agent - System Activation Architecture

Applies to Mercury Agent **v2.1.x**. Last updated: 2026-07-11.

Supplement to the top-level [`ARCHITECTURE.md`](https://github.com/Steel-SecAdv-LLC/Mercury-Agent/blob/main/ARCHITECTURE.md). Covers the loader
registry, Oracle pipeline, cognitive wiring, backend configuration,
and cross-domain correlation introduced during the Mercury System
Activation.

> **Baseline established in v1.7** (retained for historical reference; the
> subsystems below remain present in v2.1.x and are unchanged in scope). Three
> governance framework modules
> (`compliance.{nist_csf_integrator, osha_anomaly, tlp_handler}`),
> two medical predictors (`medical.{endocrinology_detector,
> anesthesiology_predictor}` plus the `CGMDataSource` /
> `VitalsDataSource` integration-ready ABCs), one new detector
> (`detectors.drone.detector`), one profiling toolkit
> (`utils.profiling`), the `MERCURY_ENV` production-mode primitive
> (`_env`), and the σ_Immutable second hard gate at every public
> decision boundary. See the parent
> [`ARCHITECTURE.md`](https://github.com/Steel-SecAdv-LLC/Mercury-Agent/blob/main/ARCHITECTURE.md) §"Governance Framework
> Modules (v1.7)" through §"Performance Profiling (v1.7)" for the
> per-module summary; the deep dives live in
> [`COMPLIANCE.md`](COMPLIANCE.md), [`PROFILING.md`](PROFILING.md),
> [`drone/SETUP.md`](drone/SETUP.md), and
> [`medical/SETUP.md`](medical/SETUP.md).

## Dataset Loader Registry

### Module Structure

```
src/omni_mercury_engine/datasets/
├── adbench.py          # ADBenchLoader (47 tabular datasets)
├── adrepository.py     # ADRepositoryLoader
├── base.py             # DatasetConfig, DatasetLoader, DatasetRegistry
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
    ├─ if _data_type == TEMPORAL and n >= max(50, 10*d):   # data-type gated; n-bound = rFFT min-length guard, not a temporality proxy
    │   └─ combined = _residual_frequency_filter(combined) # rFFT band-pass
    │
    ├─ if multiscale_tta and _data_type == TEMPORAL:        # opt-in, DEFAULT-OFF
    │   └─ combined = _multiscale_tta_scores(data, combined) # pool over dilations
    │
    └─ return {..., "oracle_metadata": {...}}
```

`_detect_data_characteristics` classifies `TEMPORAL` only on *strong*
per-feature lag-1 autocorrelation (> 0.55) **or** adjacent full-row coherence
(> 0.75) — two complementary signals whose thresholds sit in the empirical gap
between unordered tabular data and genuine time series. This keeps KinematicScore
and the temporal residual filter off tabular rows whose order is arbitrary, and
on real sensor/telemetry series.

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

### Fusion modes: CONJUNCTIVE (hub default) and FIBRING

The hub-level default is `FusionMode.CONJUNCTIVE` — a weighted geometric mean in
which both the neural score and the (undiluted) symbolic score must *agree* for a
high fused score, so a confident symbolic veto cannot be averaged away.
`FusionMode.FIBRING` remains a valid explicit mode; the rest of this section
describes it. In the NSAI taxonomy
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
which is the natural ensemble-level dual of the FIBRING mode.

Tests: `tests/core/test_fibring_default.py` pins the CONJUNCTIVE default at both
the class and factory level and covers FibringComposer behaviour and an ablation
against BALANCED on a deterministic channel-symmetric synthetic workload (no
AUROC or Brier regression).

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
        ├─ Step 8: IPBEngine                (optional, default ON)
        ├─ Step 9: CuriosityEngine          (optional, DEFAULT-OFF)
        └─ Step 10: EnhancedAnomalyDetector (optional, DEFAULT-OFF)
```

Steps 9 and 10 were added in PR #329 and are opt-in: `enable_curiosity`
(`cognitive/orchestrator.py:209`) and `enable_enhanced_detection`
(`cognitive/orchestrator.py:210`) both default to `False`, unlike Steps 4-8
which default ON. Step 9 scores how novel a detected anomaly is relative to the
distribution the `CuriosityEngine` has observed, setting `result.novelty_score`
and `result.is_novel` (`cognitive/orchestrator.py:566-568`). Step 10 folds each
observation into the `EnhancedAnomalyDetector` Bayesian/HMM predictive memory and
surfaces a forecast for detected anomalies; it is constructed with
`use_simulated_sources=False`, so it performs no network I/O on the runtime path.

All cognitive modules are advisory — they augment results, never
overwrite detector scores. Failure in any module is caught and logged;
detection continues without cognitive enrichment.

## Reasoning Backend (subordinate, called dependency)

Mercury Agent is the agent and the brain of record: its OODA loop,
neuro-symbolic detection, and dual ethical gates own the control flow. The
`reasoning/` subpackage gives Mercury a **pluggable, subordinate** reasoning
engine to *call* — for explanation, hypothesis proposal, and report synthesis
— and is **never** a system Mercury is wrapped around.

- **Interface:** `reasoning.ReasoningBackend` exposes `explain()`,
  `propose_hypotheses()`, and `synthesize_report()` over Mercury's own typed
  shapes (`reasoning/schemas.py`). No provider name appears in any signature;
  the model is a swappable dependency Mercury calls, not the front of the
  system.
- **Governed by Mercury, not the model:** every reasoning operation passes the
  benevolence + σ_Immutable dual hard gate (`enforce_dual_ethical_gate`) at the
  reasoning boundary before any output is surfaced. The gate fails closed — the
  backend does not get to bypass Mercury's governance.
- **Offline-first:** `LocalReasoningBackend` runs air-gap-safe over the local
  Ollama/template chain (free to run, no external call);
  `RemoteReasoningBackend` reaches an operator-declared model only when
  configured. `ReasoningRouter` defaults to local and, under hard-offline mode,
  never selects or calls a network backend.
- **Costed:** backends thread the LLM usage ledger
  (`models/foundation/llm_usage.py`) so provider-reported token spend is
  accounted wherever a call lands.

Mercury calls the backend from inside its loop; the backend never fronts
Mercury.

## Integration Backend Configuration

### Database (`integrations/stubs/database.py`)

| Setting | Default | Override |
|---------|---------|----------|
| Backend | In-memory stub (zero deps) | `DATABASE_BACKEND=sqlite\|postgresql` |
| Database | `mercury` | `DATABASE_NAME=<path>` |
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
# → {"domain": "environmental", "ref_band_means": {...}, ...}

# Reconstruct
federated = MercuryAnomalyDetector.from_statistics(
    ...,
    oracle_ref_stats=stats,
)
# Oracle restored without re-fitting
```
