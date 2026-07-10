# API Reference

Applies to Mercury Agent **v2.1.x**. Last updated: 2026-05-20.

> **Ethical-gate contract on every public surface.** Every `detect` /
> `analyze` / `predict` entry point on `OmniMercuryEngine`,
> `CognitiveOrchestrator`, and `NeuroSymbolicHub` runs two mandatory
> hard ethical gates (Benevolence ≥ 0.99, σ_Immutable) and raises
> `EthicalConstraintViolationError(check=…)` on failure. The reserved
> `check=` codes are `"benevolence"`, `"sigma_immutable"`, and
> `"gosnn_unavailable"`. There is no advisory mode. See
> [`MATH_SPEC.md`](MATH_SPEC.md) §2.1.5 and
> [`MIGRATION-1.6-to-1.7.md`](MIGRATION-1.6-to-1.7.md) §2.

## Module index

| Module | Purpose | Entry doc |
|--------|---------|-----------|
| `omni_mercury_engine.detectors.statistical` | Core 3-component anomaly ensemble | this file |
| `omni_mercury_engine.engine` | `OmniMercuryEngine` boundary surface | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| `omni_mercury_engine.cognitive` | Cognitive orchestrator, neuro-symbolic fusion | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| `omni_mercury_engine.compliance` | NIST CSF 2.0, TLP 2.0, OSHA / eCFR | [`COMPLIANCE.md`](COMPLIANCE.md) |
| `omni_mercury_engine.medical` | Cardiology, critical care, endocrinology, anesthesiology, pandemic | [`medical/SETUP.md`](medical/SETUP.md) |
| `omni_mercury_engine.detectors.drone` | RADD + Mercury-ensemble drone anomaly detection | [`drone/SETUP.md`](drone/SETUP.md) |
| `omni_mercury_engine.utils.profiling` | CPU / memory / wall-clock profiling decorators | [`PROFILING.md`](PROFILING.md) |
| `omni_mercury_engine.security.safe_http` | SSRF / DNS-rebinding defence layer | [`DATASOURCES.md`](DATASOURCES.md), [`MIGRATION-1.6-to-1.7.md`](MIGRATION-1.6-to-1.7.md) §1 |
| `omni_mercury_engine._env` | `MERCURY_ENV` production-mode primitive | [`MIGRATION-1.6-to-1.7.md`](MIGRATION-1.6-to-1.7.md) §3 |
| `omni_mercury_engine._pqc_gate` | Import-time PQC production gate | [`SECURITY.md`](https://github.com/Steel-SecAdv-LLC/Mercury-Agent/blob/main/SECURITY.md), [`INSTALLATION.md`](INSTALLATION.md) |

---

## MercuryAnomalyDetector

Mercury's original anomaly detection ensemble combining three mathematical frameworks.
No sklearn dependency in the detection path — only numpy and scipy.

```python
from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

detector = MercuryAnomalyDetector()
detector.fit(X_train)
result = detector.detect(X_test)

scores = result["scores"]                    # Continuous [0, 1]
is_anomaly = result["is_anomaly"]            # Boolean array
components = result["ensemble_components"]   # Per-component scores
```

### Ensemble Components

| Component | Default Weight | Method |
|-----------|---------------|--------|
| ResonanceScore | 40% | FFT spectral density profiling |
| KinematicScore | 30% | Derivative-based dynamics (velocity, acceleration, jerk) |
| InfoGeometryScore | 30% | Fisher information / Mahalanobis distance |

> **Note:** Weights are **adaptive** after `fit()`. The detector computes per-component
> AUC separation and assigns weights proportional to each component's discriminative
> power. Components with AUC < 0.5 (inverted signal) receive zero weight. The
> 40/30/30 split above is the **fallback default** used only when all components
> produce near-random scores. See `_compute_adaptive_weights()` in `statistical.py`.

### Config Options

```python
detector = MercuryAnomalyDetector(config={
    "z_threshold": 3.0,        # Z-score threshold for outlier detection
    "iqr_multiplier": 1.5,     # IQR fence multiplier
    "threshold": 0.5,          # Decision threshold for is_anomaly
    "auto_calibrate": False,   # Auto-calibrate threshold from score distribution
})
```

### `fit(data) -> MercuryAnomalyDetector`

Fit on training data. Computes baselines for all three components:
- Distributional statistics (mean, std, quartiles)
- Kinematic baselines (jerk/acceleration mean and std per feature)
- Information-geometric manifold (mean, regularized precision matrix)
- FFT spectral profiles per feature

**Args:** `data` — numpy array or torch tensor, shape `(n_samples,)` or `(n_samples, n_features)`.

**Returns:** Self (for method chaining).

### `detect(data) -> dict`

Run anomaly detection. Returns a dictionary with the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `scores` | `ndarray` | Combined ensemble scores in [0, 1] |
| `is_anomaly` | `ndarray[bool]` | Boolean anomaly predictions |
| `z_scores` | `ndarray` | Raw z-scores per feature |
| `z_score_continuous` | `ndarray` | Normalized z-score intensity [0, 1] |
| `iqr_scores` | `ndarray` | Continuous IQR-based scores [0, 1] |
| `resonance_scores` | `ndarray` | FFT harmonic anomaly scores [0, 1] |
| `kinematic_scores` | `ndarray` | Physics dynamics scores [0, 1] |
| `info_geometry_scores` | `ndarray` | Fisher OOD scores [0, 1] |
| `ensemble_components` | `dict` | `{"resonance": ..., "kinematic": ..., "info_geometry": ...}` |
| `threshold` | `float` | Effective threshold (may be auto-calibrated) |
| `calibration_diagnostics` | `dict\|None` | Diagnostics when auto-calibrated |
| `detector_type` | `str` | Always `"statistical"` |
| `iqr_flags` | `ndarray[bool]` | Legacy boolean IQR anomalies |
| `isolation_forest_scores` | `ndarray` | DEPRECATED — alias for `scores` |
| `isolation_forest_flags` | `ndarray[bool]` | DEPRECATED — alias for `is_anomaly` |

### Auto-Calibration

```python
detector = MercuryAnomalyDetector()
detector.fit(X_train)
detector.enable_auto_calibration(contamination=0.05)
result = detector.detect(X_test)
# result["threshold"] is now auto-calibrated
```

### Backward Compatibility

`StatisticalAnomalyDetector` is retained as an alias:

```python
from omni_mercury_engine.detectors.statistical import StatisticalAnomalyDetector
# StatisticalAnomalyDetector is MercuryAnomalyDetector
```

---

## Compliance: NIST CSF 2.0 / TLP 2.0 / OSHA

`omni_mercury_engine.compliance` ships three first-party governance
modules. Full reference (citations, public surface, live-fetcher
contracts, ethical considerations) lives in
[`COMPLIANCE.md`](COMPLIANCE.md). Quick imports:

```python
from omni_mercury_engine.compliance import (
    NISTCSFIntegrator, NISTFunction, NISTSubcategory, ImplementationTier,
    OSHAComplianceDetector, OSHASector, HazardCategory,
    compute_heat_index_fahrenheit,
    TLPHandler, TLPClassification, TLPColor,
)
```

- **NIST CSF 2.0** — all six core functions, 22 categories, 106+
  subcategories, live reference fetcher (`NISTCSFReferenceFetcher`)
  with 7-day on-disk cache, gap analysis, supply-chain anomaly
  detection, JSON compliance reports.
- **OSHA / eCFR** — 12 hazard categories × 6 industry sectors with
  CFR citations and NWS Rothfusz heat-index regression
  (`compute_heat_index_fahrenheit(temp_f, rh_pct)`).
- **TLP 2.0** — full five-label, four-colour ladder (CLEAR / GREEN / AMBER /
  AMBER+STRICT / RED; AMBER+STRICT shares AMBER's colour), single/batch classification, watermark
  generation, JSON export metadata. `AMBER+STRICT` is implemented
  end-to-end — the upstream module shipped only TLP 1.0 colours.

## Medical: Endocrinology / Anesthesiology / Cardiology / Critical Care / Pandemic

`omni_mercury_engine.medical` ships **integration-ready, not
pre-integrated** clinical modules. The platform never carries vendor
credentials and never fabricates patient data; misconfigured adapters
raise `ConfigurationError`. Full reference lives in
[`medical/SETUP.md`](medical/SETUP.md). Quick imports:

```python
from omni_mercury_engine.medical import (
    # Data-source contracts
    CGMDataSource, VitalsDataSource,
    DexcomV3DataSource, FHIRObservationVitalsSource,
    ConfigurationError, DataSourceError,
    # Endocrinology (CGM Bi-LSTM ~155K params; FDA-aligned rules)
    EndocrinologyDetector, CGMAnalyzer, GlycemicState,
    GLP1TherapyMonitor, InhaledInsulinMonitor, SmartInsulinPenMonitor,
    # Anesthesiology (TIVA Bi-LSTM ~164K params; PID infusion)
    AnesthesiologyPredictor, HemodynamicMonitor,
    SmartInfusionController, TIVAMonitoringSystem,
    AnesthesiaType, AnesthesiaRisk,
    # Cardiology
    CardiologyPredictor, ECGRhythmAnalyzer,
    FraminghamRiskCalculator, ArrhythmiaType,
    # Critical care
    SepsisDetector, SOFACalculator, QuickSOFACalculator,
    StrokeDetector, NIHSSCalculator, SeizurePredictor, ICPMonitor,
    # Pandemic
    PandemicDetector, EpidemicForecaster, PathogenDetector,
    MutationTracker, TransmissionNetworkAnalyzer,
    # Coordinator
    MedicalCoordinator,
)
```

`MedicalCoordinator` exposes a filtered registry over the modules
above by `category` (`pandemic`, `critical_care`, `cardiology`,
`general`), `priority` (`high`, `medium`), or explicit `module_names`.
See [`medical/SETUP.md`](medical/SETUP.md) for the data-source contract
(`DEXCOM_CLIENT_ID`/`DEXCOM_CLIENT_SECRET`/`DEXCOM_REFRESH_TOKEN`/`DEXCOM_REDIRECT_URI`,
`FHIR_BASE_URL`/`FHIR_PATIENT_ID`/`FHIR_BEARER_TOKEN`) and how to write
a custom adapter (e.g. Abbott LibreView, Medtronic CareLink).

## Drone: DroneAnomalyDetector

`omni_mercury_engine.detectors.drone` ships a transport-agnostic drone
anomaly detector combining rule-based RADD invariants with Mercury
Agent's first-party ensemble (no sklearn runtime dependency). Full
reference lives in [`drone/SETUP.md`](drone/SETUP.md). Quick import:

```python
from omni_mercury_engine.detectors.drone.detector import (
    DroneAnomalyDetector, DroneState,
)
```

Populate `DroneState` (always with keyword arguments; the dataclass
field order is `position`, `velocity`, `attitude`, `battery_level`,
`altitude`, `gps_satellites`, `signal_strength`, `motor_speeds`,
`temperature`, `mission_phase`, then the four derived kinematic
fields `altitude_rate` / `horizontal_velocity` / `vertical_velocity`
/ `distance_to_home`, `home_position`, and `timestamp`) from your
ingest layer of choice — PX4 ULog via `pyulog`, MAVLink via
`pymavlink`, or vendor SDK — and feed the sequence through the
detector.

## Profiling: `omni_mercury_engine.utils.profiling`

Six entry points for performance instrumentation. Full reference lives
in [`PROFILING.md`](PROFILING.md). All entry points are no-ops when
profiling is globally disabled via `set_profiling_enabled(False)`
(the default).

```python
from omni_mercury_engine.utils.profiling import (
    set_profiling_enabled, is_profiling_enabled,
    profile_func, profile_memory, profile_time, profile_time_async,
    profile_complete,
    PerformanceBenchmark, benchmark_function,
)

set_profiling_enabled(True)

@profile_time()
def expensive_op(x: int) -> int:
    return sum(range(x))

stats = benchmark_function(expensive_op, 10_000, iterations=200)
print(stats["mean_ms"], stats["std_ms"])
```

## Production-mode primitive: `omni_mercury_engine._env`

```python
from omni_mercury_engine._env import (
    get_mercury_env,           # -> Literal["development", "production"]
    is_production,             # -> bool
    require_real_component,    # raise MercuryProductionConfigError if missing in prod
    MercuryProductionConfigError,
)
```

Reads the `MERCURY_ENV` environment variable (`development` default).
Unknown values raise `MercuryProductionConfigError` at first read — typos
must be loud. Orthogonal to `AMA_REQUIRE_REAL_PQC` (the import-time PQC
gate). See [`MIGRATION-1.6-to-1.7.md`](MIGRATION-1.6-to-1.7.md) §3.
