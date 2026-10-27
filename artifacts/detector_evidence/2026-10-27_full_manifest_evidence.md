# Mercury-Agent Detector Evidence Table — full manifest (113 entries), rev 2

**Rev 2 corrections (same day).** (a) Two detectors were mis-driven by the rev-1 sweep harness (it called `fit().detect()` on registry-seam detectors that expose `fit()+predict()`): `info_geometry` and `adaptive_anomaly` are NOT broken — re-driven through their declared seams they score **0.891** and **0.883** mean AUROC on the clinical sets (2nd and 3rd of all graded). (b) `astrophysical`, `quantum`, `consciousness` were retrofitted with fitted-reference scoring (models/fitted_reference.py) and re-measured under the identical protocol: see their updated rows. (c) A latent width bug in `quantum.extract_features` (crash on <8-column input) was found and fixed.

Generated 2026-10-27 18:16 UTC on branch `steel/version-cov7`, head `d6e3f23f`.

**Protocol.** Real labeled data only. Series detectors: 4 NAB streams (warmup fit on first 15%, AUROC on the rest). Tabular detectors: 5 ADBench clinical/real sets (≤2000 rows). 90s/dataset budget, fast-config fallback on timeout. Exotic window-level models: 12σ planted-burst floor probe (30 clean vs 30 burst windows). Single seed; no tuning. Tiers are **computed from these numbers**, provisional, per the T0–T5 scheme (T5/T4 require production/domain evidence this sweep cannot grant).

**Known protocol limits (honest):** point-wise AUROC under-credits change-point/tail detectors (bocpd, sprt, page_hinkley, spot_evt, hawkes); ADBench subsampled to 2000 rows; one seed; the burst probe is a floor check, not a benchmark.

## A. Graded on real labeled data (44 detectors)

| Detector | Cat | thyroid | wbc | cardio | pima_real | breastw | realTraff | realKnown | realAWSCl | realAdExc | Mean | Tier | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| adversarial_autoencoder | MODEL | ⏱ | 1.00 | ⏱ | ⏱ | ⏱ | — | — | — | — | **0.997** | T2* | *single dataset only (4 timeouts) — tier capped, needs full grading |
| directive | BASE | 0.90 | 0.97 | 0.92 | 0.72 | 0.99 | — | — | — | — | **0.901** | T3 |  |
| rca | BASE | 0.93 | 0.97 | 0.93 | 0.67 | 0.96 | — | — | — | — | **0.894** | T3 |  |
| copod | MODEL | 0.77 | 0.99 | 0.91 | 0.66 | 0.99 | — | — | — | — | **0.864** | T3 |  |
| kmeans_distance | BASE | 0.93 | 0.98 | 0.74 | 0.64 | 0.97 | — | — | — | — | **0.851** | T3 |  |
| lstm_autoencoder | MODEL | — | — | — | — | — | 0.95 | 0.59 | 0.79 | 0.99 | **0.829** | T3 |  |
| echo_state | BASE | — | — | — | — | — | 0.65 | 0.71 | 0.70 | 0.71 | **0.691** | T2 |  |
| spatial | BASE | 0.84 | 0.95 | 0.55 | 0.57 | 0.44 | — | — | — | — | **0.671** | T2 |  |
| energy_based | BASE | — | — | — | — | — | 0.74 | 0.65 | 0.72 | 0.57 | **0.670** | T2 |  |
| srcnn | BASE | — | — | — | — | — | 0.74 | 0.48 | 0.63 | 0.82 | **0.668** | T2 |  |
| deep_svdd | BASE | — | — | — | — | — | 0.76 | 0.67 | 0.61 | 0.60 | **0.658** | T2 |  |
| diffusion_ad | BASE | — | — | — | — | — | 0.84 | 0.57 | 0.51 | 0.67 | **0.648** | T2 |  |
| bocpd | BASE | — | — | — | — | — | 0.64 | 0.37 | 0.88 | 0.69 | **0.645** | T2 |  |
| temporal | BASE | — | — | — | — | — | 0.75 | 0.66 | 0.61 | 0.55 | **0.642** | T2 |  |
| extended_statistical | BASE | — | — | — | — | — | 0.67 | 0.65 | 0.65 | 0.52 | **0.622** | T2 |  |
| digital_twin | BASE | — | — | — | — | — | 0.72 | 0.63 | 0.58 | 0.50 | **0.607** | T2 |  |
| survival | BASE | — | — | — | — | — | 0.69 | 0.64 | 0.49 | 0.57 | **0.597** | T1 | ~chance on this protocol |
| particle_filter | BASE | — | — | — | — | — | 0.65 | 0.44 | 0.71 | 0.56 | **0.590** | T1 | ~chance on this protocol |
| spiking | BASE | — | — | — | — | — | 0.70 | 0.66 | 0.41 | 0.58 | **0.585** | T1 | ~chance on this protocol |
| page_hinkley | BASE | — | — | — | — | — | 0.41 | 0.61 | 0.76 | 0.52 | **0.576** | T1 | point-wise AUROC is a mismatched protocol for change-point/tail detectors |
| statistical | BASE | — | — | — | — | — | 0.63 | 0.62 | 0.50 | 0.52 | **0.567** | T1 | ~chance on this protocol |
| gaussian_process | BASE | — | — | — | — | — | 0.61 | 0.40 | 0.66 | 0.58 | **0.563** | T1 | ~chance on this protocol |
| dimensional | BASE | — | — | — | — | — | 0.60 | 0.43 | 0.55 | 0.57 | **0.538** | T1 | ~chance on this protocol |
| hawkes | BASE | — | — | — | — | — | 0.57 | 0.62 | 0.45 | 0.49 | **0.533** | T1 | point-wise AUROC is a mismatched protocol for change-point/tail detectors |
| spot_evt | BASE | — | — | — | — | — | 0.50 | 0.50 | 0.50 | 0.52 | **0.503** | T1 | point-wise AUROC is a mismatched protocol for change-point/tail detectors |
| imm | BASE | — | — | — | — | — | 0.66 | 0.37 | 0.46 | 0.51 | **0.500** | T1 | ~chance on this protocol |
| spectral_residual | BASE | — | — | — | — | — | 0.49 | 0.52 | 0.48 | 0.49 | **0.494** | T1 | ~chance on this protocol |
| sprt | BASE | — | — | — | — | — | 0.31 | 0.48 | 0.60 | 0.58 | **0.492** | T1 | point-wise AUROC is a mismatched protocol for change-point/tail detectors |
| chronos | FOUNDATION | — | — | — | — | — | 0.61 | 0.48 | 0.11 | 0.28 | **0.371** | T1 | ~chance on this protocol |
| matrix_profile | FOUNDATION | — | — | — | — | — | 0.07 | 0.47 | 0.53 | 0.35 | **0.356** | T1 | ~chance on this protocol |
| foundation_ensemble | FOUNDATION | — | — | — | — | — | 0.07 | 0.47 | 0.53 | 0.35 | **0.356** | T1 | ~chance on this protocol |
| frequent_pattern | BASE | — | — | — | — | — | no-pts | no-pts | no-pts | no-pts | n/a | T1 | no per-point scores harvested |
| quantum | MODEL | 0.99 | 0.87 | 0.83 | 0.59 | 0.84 | — | — | — | — | **0.827** | T3 | FIXED (fitted reference; was burst 0.33): burst now 1.000; declared cannot_detect: amplitude collapse (scale-invariant features) |
| astrophysical | MODEL | 0.64 | 0.99 | 0.92 | 0.50 | 0.97 | — | — | — | — | **0.803** | T3 | FIXED (fitted reference; was inverted, burst 0.000): burst now 1.000 both directions |
| biometric | MODEL | — | — | — | — | — | no-pts | no-pts | no-pts | no-pts | n/a | T1 | window-level score only (no per-point); burst probe 1.00 separates |
| affective | MODEL | — | — | — | — | — | no-pts | no-pts | no-pts | no-pts | n/a | T0 | constant output on declared modality (mislabeled input contract) |
| neural_cognitive | MODEL | — | — | — | — | — | no-pts | no-pts | no-pts | no-pts | n/a | T1 | window-level score only (no per-point); burst probe 1.00 separates |
| consciousness | MODEL | 0.94 | 0.66 | 0.72 | 0.59 | 0.69 | — | — | — | — | **0.717** | T2 | FIXED (fitted reference; was non-separating 0.644): burst now 1.000; declared cannot_detect: amplitude collapse |
| chemistry | MODEL | — | — | — | — | — | no-pts | no-pts | no-pts | no-pts | n/a | T0 | constant output on declared modality (mislabeled input contract) |
| parapsychology | MODEL | — | — | — | — | — | no-pts | no-pts | no-pts | no-pts | n/a | T0 | constant output on declared modality (mislabeled input contract) |
| nano_safeguard | BASE | — | — | — | — | — | no-pts | no-pts | no-pts | no-pts | n/a | T1 | no per-point scores harvested |
| adaptive_anomaly | BASE | 0.88 | 1.00 | 0.87 | 0.69 | 0.98 | — | — | — | — | **0.883** | T3 | rev-1 harness error; correct seam drive shown |
| topological | BASE | ⏱ | refuse | ⏱ | ⏱ | refuse | — | — | — | — | n/a | T1 | exceeded 90s/dataset budget on every set |
| info_geometry | BASE | 0.94 | 0.97 | 0.90 | 0.67 | 0.97 | — | — | — | — | **0.891** | T3 | rev-1 harness error (drove detect on a fit+predict registry detector); correct seam drive shown |

## B. Exotic window-level models — burst floor probe (8 detectors)

| Detector | Burst AUROC (12σ vs clean) | Distinct scores | Verdict |
|---|---|---|---|
| astrophysical | 0.000 | 60 | fully inverted (scores bursts safest) — root-caused: score = proximity-to-horizon = 1/norm |
| quantum | 0.329 | 60 | inverted-ish / below chance — fix via fitted reference (task #24) |
| consciousness | 0.644 | 60 | no separation (0.228 vs 0.224 means) — unfitted thresholds |
| affective | 0.500 | 1 | constant 0.5 on series — wrong declared modality; needs emotion records |
| chemistry | 0.500 | 1 | constant 0 on series — parsed 0 elements; needs composition records |
| parapsychology | 0.500 | 1 | constant 0 on series — needs experiment trial records |
| biometric | 1.000 | 60 | separates at window level; per-point contract missing |
| neural_cognitive | 0.999 | 60 | separates at window level; per-point contract missing |

## C. Not gradable on generic data — needs its domain pack (69 entries)

These are *unmeasured here*, not vouched for. Each needs a domain-grounded pack (task #27) or carries `domain_supported: false`. Exception with committed domain evidence: **drone** (ALFA corpus: 35/36 real faults at 0.68% FPR → provisional T4).

| Category | Detectors (modality) |
|---|---|
| BASE (4) | geo_movement(trajectory), graph_based(graph), deeplog_sequence(event), multivariate_ts(windows_3d) |
| DISTILLATION (1) | dual_student(image_batch) |
| ECONOMIC (2) | financial_crisis(record), financial_stress(rows) |
| ENERGY (1) | emp(record) |
| FOUNDATION (1) | timegpt(series_1d) |
| GEOLOGICAL (17) | volcanic(record), landslide(rows), wildfire(record), tornado(series_1d), hurricane(series_1d), flood(series_1d), earthquake(series_1d), tsunami(series_1d), meteor(series_1d), avalanche(record), rockfall(record), subsidence(record), eq_tsunami_cascade(event), fire_debris_flow_cascade(record), tornado_conditions(rows), hurricane_pressure(rows), flood_precipitation(rows) |
| INTELLIGENCE (2) | psyop(series_1d), intelligence_fusion(record) |
| MARINE (1) | marine_biodiversity(rows) |
| MEDICAL (3) | medical_abms(record), sepsis(record), case_surge(window) |
| METEOROLOGICAL (10) | pm25_air_quality(rows), drought(series_1d), heatwave(series_1d), atmospheric_river(series_1d), lightning(series_1d), hail(series_1d), winter_storm(series_1d), derecho(series_1d), dust_storm(series_1d), surge_flood_cascade(event) |
| NEUROSYMBOLIC (1) | neurosymbolic(series_1d) |
| PHYSICS (4) | spectral_vibration(series_1d), acceleration_dynamics(series_1d), physics_integrated(matrix_2d), spectral_domain_frequency(series_1d) |
| SECURITY (2) | threat_detection(text), realtime_threat(matrix_2d) |
| SPACE (11) | schumann_resonance(series_1d), solar_storm(rows), solar_flare(series_1d), cme_arrival(rows), sep_storm(window), ionospheric_scintillation(record), gic(window), solar_gic_cascade(event), solar_wind_geomag(rows), xray_flare(rows), gic_minute_series(rows) |
| UAV (1) | drone(series_1d) **←ALFA T4** |
| UIUX (1) | uiux_anomaly(series_1d) |
| VISUAL (5) | patchcore(image_batch), padim(image_batch), stfpm(image_batch), reverse_distillation(image_batch), cflow(image_batch) |
| VLM (2) | anyanomaly(video_frames), lavad(video_frames) |

## Summary

- Manifest entries: 113. Graded here: 44 (+8 burst-probed exotics overlap). Domain-wired/unmeasured: 69.
- Tier counts (graded set, rev 2): T0: 2 (chemistry, parapsychology — modality mislabel, routes to ontology fix), T1: 19, T2: 12, T3: 10. Rev-1 counted 7 T0: two were harness errors, three are now fixed and re-measured.
- Scored on real data: 31; of those ≥0.75 mean: 6; 0.60–0.75: 10; <0.60: 15.
- Separately evidenced in committed artifacts (not this sweep): fused engine 0.8545/54 sets; ADBench harness 0.7634; ALFA drone 35/36 @0.68% FPR; UCI robot 0.968; NAB neuro-symbolic mean 0.729.

## New defects this sweep surfaced (beyond the 8 exotics)

1. **`info_geometry` — broken at the manifest seam**: declares an engine role but has no `detect` attribute; every call refused (T0). Same defect class as `matrix_profile`'s missing `fit`.
2. **FOUNDATION tier underperforms on real streams**: `matrix_profile`/`foundation_ensemble` hit **0.07** on realTraffic (an anomaly that *repeats* gets a *low* discord score — likely score-orientation/motif issue, needs root-cause), and `chronos` hits 0.11 on realAWSCloudwatch. The "foundation" label currently outranks its evidence.
3. **Contract gaps**: `frequent_pattern`, `nano_safeguard`, `adaptive_anomaly` harvest no per-point scores on their declared modality; `topological` exceeds 90s on every tabular set (needs a budget or a fast path).
