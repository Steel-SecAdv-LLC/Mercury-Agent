# Hazard checkpoint training (T5)

Applies to Mercury Agent **v2.1.x**. Last updated: 2026-07-11.

Reproducible pipeline for training the eleven `load_neural_weights()` hooks
on **real data only**, with a hard merit gate: a checkpoint ships only when
the trained model beats the detector's deterministic physics fallback on
held-out test years, evaluated **through the public detector API**.

Entry point:

```bash
# The full pipeline for a hook (fetch → build → train → evaluate → ship):
PYTHONPATH=src python scripts/train_hazard_checkpoints.py --hook solar_storm --stage all

# The 11-hook audit (what trains where, and why not here):
python scripts/train_hazard_checkpoints.py --audit
```

Every stage is deterministic (seeded), disk-cached with sha256 provenance,
and temporal splits are by-year with enforced `train < val < test` ordering
(no random splits — Kp and seismicity autocorrelate across years). Shipped
checkpoints live in `omni_mercury_engine/models/checkpoints/` next to a
`<name>.provenance.json` sidecar recording the data sources (URL + sha256),
seed, commit, and the full learned-vs-physics evaluation. `ship` **refuses**
(exit 3) when the merit gate fails; "physics wins, not shipped" is a
first-class recorded result, not a failure of the pipeline.

## Audit table (2026-07-09)

| Hook | Detector / architecture | Cat. | Data | Result |
|---|---|---|---|---|
| `solar_storm` | `SolarStormDetector` / GeomagneticStormPredictor | **a** | NASA SPDF OMNI2 hourly solar wind + observed Kp, 2005–2024 (20 files, sha256-pinned), GFZ Kp cross-check | **SHIPPED** — see below |
| `earthquake_precursor` | `DisasterPrecursorDetector` / EarthquakePrecursorAnalyzer(128) | b | USGS FDSN catalog (reachable) reshaped into regional sequence samples; deliberately not shipped until the feature spec is literature-reviewed — a half-reviewed earthquake forecaster is worse than the abstaining physics fallback | fail-loud with spec |
| `seismic_wave` | `EarthquakeDetector` / SeismicWaveAnalyzer | b | EarthScope/IRIS FDSN dataselect waveforms (service.iris.edu unreachable here) | fail-loud with spec |
| `tsunami_waveform` | `TsunamiDetector` / WaveformFFTAnalyzer | b | NOAA DART bottom-pressure archives (unreachable here) | fail-loud with spec |
| `hurricane_wind` | `HurricaneDetector` / WindPatternAnalyzer | b | ERA5 wind fields (CDS API key) + IBTrACS labels | fail-loud with spec |
| `landslide_stability` | `LandslideDetector` / SlopeStabilityModel | b | NASA GLC/COOLR + rainfall covariate archives | fail-loud with spec |
| `tornado_radar` | `TornadoDetector` / DopplerRadarAnalyzer | b | NEXRAD Level-II volumes (AWS Open Data) + SPC reports | fail-loud with spec |
| `volcanic_eruption` | `VolcanicEruptionDetector` / EruptionForecastModel | b | Observatory multiparameter series (per-volcano, on request) | fail-loud with spec |
| `wildfire_ignition` | `WildfireDetector` / FireIgnitionDetector CNN | b | NASA FIRMS granules (requires MAP_KEY) | fail-loud with spec |
| `schumann_harmonics` | `SchumannResonanceDetector` / CNN+LSTM | b | Calibrated ELF observatory spectra; the simulated BGS client must never be training data | fail-loud with spec |
| `consciousness_field` | `ConsciousnessField` / LSTM | c | No real labeled corpus exists; training would require fabrication | physics fallback permanent |

Category **a** = real labeled data fetchable from this environment; **b** =
real data exists but needs archives/credentials absent here (the registry
entry carries the exact source, and running any training stage raises
`HazardDataUnavailableError` with that requirement); **c** = no real corpus
exists for the architecture's input contract.

## Shipped: `solar_storm_geomag`

Trained on 20 years (2005–2024) of the real NASA SPDF OMNI2 hourly archive —
multi-spacecraft L1 solar-wind/IMF measurements paired with the *observed*
planetary Kp — with the OMNI2 Kp parsing cross-checked against the GFZ
Potsdam definitive Kp service over the 2023-04 G4 storm window. Temporal
split: train 2005–2018, val 2019–2021, test **2022–2024** (25,846 held-out
hours spanning the ascent of solar cycle 25, including the 2024-05-10 G5
superstorm).

Held-out comparison, both models running through
`SolarStormDetector.predict_solar_storm` on identical hours:

| Metric | Learned | Physics (Boyle index) |
|---|---|---|
| Kp MAE | **0.574** | 1.054 |
| Kp RMSE | **0.754** | 1.344 |
| G-bucket accuracy | **97.6%** | 94.8% |
| Storm (Kp≥5) AUC | **0.972** | 0.845 |
| Storm recall @ fixed Kp5 point | 30.4% | 57.4% |
| Storm false-alarm rate @ Kp5 | **0.09%** | 3.1% |

The merit gate compares the primary metric (Kp MAE): learned wins by 45%.
Transparent caveat, recorded in the provenance sidecar: at the *fixed* Kp≥5
operating point the learned nowcast trades recall for a 36× lower
false-alarm rate; the AUC shows its ranking of storm hours is strictly
better, so operators can choose their own operating point. The detector
loads this checkpoint via `load_neural_weights()` (no arguments → shipped
default, provenance logged; corrupt or missing files raise).
