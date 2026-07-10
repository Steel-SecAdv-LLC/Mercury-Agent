# Hazard checkpoint training (T5)

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
| `earthquake_precursor` | `DisasterPrecursorDetector` / EarthquakePrecursorAnalyzer(128) | **a** | USGS ComCat FDSN catalog, California 1980–2024 M≥2.5 (45 yearly CSVs, sha256-pinned), P(M≥5.0, 30 d) per 0.5° cell per the binding review (`docs/research/EARTHQUAKE_PRECURSOR_LITERATURE_REVIEW.md`); `seismicity-catalog-v2` stacks the Reasenberg–Jones baseline's own causal forecast as inputs | **NOT SHIPPED — physics wins ranking** (2026-07-10): learned wins held-out log-loss 0.00414 vs 0.00629 (bootstrap 95% CI excludes 0) but AUC 0.8895 < RJ baseline 0.8975, so the gate's auc non-regression constraint refused; abstaining fallback stays in charge |
| `seismic_wave` | `EarthquakeDetector` / SeismicWaveAnalyzer | b | EarthScope/IRIS FDSN dataselect waveforms (service.iris.edu unreachable here) | fail-loud with spec |
| `tsunami_waveform` | `TsunamiDetector` / WaveformFFTAnalyzer | b | NOAA DART bottom-pressure archives (unreachable here) | fail-loud with spec |
| `hurricane_wind` | `HurricaneDetector` / WindPatternAnalyzer | b | ERA5 wind fields (CDS API key) + IBTrACS labels | fail-loud with spec |
| `landslide_stability` | `LandslideDetector` / SlopeStabilityModel | **a** | NASA GLC/COOLR events (AGOL mirror, sha256-pinned pages) + CHIRPS v2.0 daily 0.25° rainfall 1981–2024 (45 files, sha256-pinned); `landslide-coolr-v1` features against the fixed 1981–2006 climatology, site/geotechnical dims honestly zero | **SHIPPED** (2026-07-10): held-out 2018–2024 AUC 0.8498 vs 0.8064 for the train-fitted Caine-style rain-percentile baseline (bootstrap 95% CI on the difference [0.031, 0.057] excludes 0); deployed-rule recall 0.6555 vs 0.6189, FAR 0.1448 vs 0.1590, Brier 0.1569 vs 0.2662 all pass; validation-selected operating point τ=0.6665 carried in the checkpoint, consumed decision-only |
| `tornado_radar` | `TornadoDetector` / DopplerRadarAnalyzer | b | NEXRAD Level-II volumes (AWS Open Data) + SPC reports | fail-loud with spec |
| `volcanic_eruption` | `VolcanicEruptionDetector` / EruptionForecastModel + SeismicSwarmDetector | **a** | Smithsonian GVP Holocene catalog (day-precision onsets + VEI; USGS HANS VAN cross-check for 1–2 d-uncertain starts) + EarthScope FDSN AV-network station-day miniSEED (926 station-days, sha256-aggregated); SCOPED to 8 named AVO volcanoes (Shishaldin, Semisopochnoi, Pavlof, Great Sitkin, Veniaminof, Cleveland, Okmok, Redoubt) | **SHIPPED** (2026-07-10): held-out 2020–2024 AUC 0.9402 vs 0.7096 for the seismic physics path (train-fitted RSAM-ratio baseline 0.6978; bootstrap 95% CI on the difference [0.140, 0.319] excludes 0); deployed-rule recall 0.8393 vs 0.7857, FAR 0.0882 vs 0.5343, Brier 0.0675 vs 0.2876 all pass; all 4 covered test onsets caught at 13 d earliest lead (Cleveland's 2020-06-01 test onset had no station coverage — reported, not hidden); validation-selected operating point τ=0.4172 carried in the checkpoint, consumed decision-only |
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
Honest caveat, recorded in the provenance sidecar: at the *fixed* Kp≥5
operating point the learned nowcast trades recall for a 36× lower
false-alarm rate; the AUC shows its ranking of storm hours is strictly
better, so operators can choose their own operating point. The detector
loads this checkpoint via `load_neural_weights()` (no arguments → shipped
default, provenance logged; corrupt or missing files raise).
