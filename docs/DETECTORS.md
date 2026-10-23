# Detector Catalog

<!-- GENERATED FILE: do not hand-edit. -->
<!-- Regenerate: python scripts/generate_detector_catalog.py -->
<!-- CI gate:    python scripts/generate_detector_catalog.py --check -->

Every entry below comes from ``DETECTOR_MANIFEST``
(``src/omni_mercury_engine/core/detector_registry.py``), the single source of
truth the seam test suite (``tests/core/test_detector_manifest_seams.py``)
drives end to end on every CI run.  **Seams** say how an entry is reached:

- ``engine`` -- a ``BaseDetector``; enable it with
  ``OmniMercuryEngine.enable_detector(name)`` and use ``fit``/``detect``.
- ``registry`` -- ``DetectorRegistry.auto_discover_detectors()`` registers it;
  ``extract_all_features(data, modality=...)`` harvests its real output.
- *library-only* -- no numeric seam; call its domain API directly (the table
  names the adapter entry covering it, where one exists).

**Modality** is the input its seams were measured with.  ``fit`` marks
detectors whose registry methods require ``fit`` first; ``ext`` marks
detectors needing an external service or model backend; ``serves X`` marks
an adapter entry and names the domain detector behind it.


## engine + registry (68)

| name | class | category | modality | notes | description |
|---|---|---|---|---|---|
| `bocpd` | `BOCPDDetector` | base | series_1d | -- | Bayesian Online Change-Point Detection via run-length posterior (Gaussian/NIG conjugate model; Adams & MacKay, 2007) |
| `deep_svdd` | `DeepSVDDDetector` | base | series_1d | -- | One-class SVDD hypersphere detector on a fixed random tanh-feature embedding (a saturating nonlinear lift, not a Fourier map); distance-to-centre is the anomaly signal (Tax & Duin, 2004) |
| `deeplog_sequence` | `DeepLogSequenceDetector` | base | series_1d | -- | DeepLog-style log/sequence detector: next-key surprisal from an n-gram transition model flags sequences leaving the learned template grammar |
| `diffusion_ad` | `DiffusionReconstructionDetector` | base | series_1d | fit | DDPM-AD: denoising-diffusion reconstruction-error detector; anomaly score is off-manifold noise-prediction error (Ho et al., 2020; requires torch) |
| `digital_twin` | `DigitalTwinResidualDetector` | base | series_1d | -- | Digital-twin simulation-residual detector: observed-vs-simulated divergence of an identified AR forward model (Grieves & Vickers, 2017) |
| `directive` | `SigmaDirectiveDetector` | base | matrix_2d | -- | Rule-based sigma detection |
| `echo_state` | `EchoStateDetector` | base | series_1d | -- | Echo-State-Network reservoir-computing predictive residual detector (fixed spectral-radius reservoir + ridge readout; Jaeger, 2001) |
| `energy_based` | `EnergyBasedDetector` | base | series_1d | -- | Explicit energy-based model detector: delay-embedding quadratic (Gaussian-family) energy fitted by score matching; anomaly score is the calibrated free energy |
| `extended_statistical` | `ExtendedStatisticalDetector` | base | series_1d | fit | Ensemble over MAD, LOF, DBSCAN, MCD, CUSUM, Grubbs and GESD statistical methods |
| `frequent_pattern` | `FrequentPatternDetector` | base | series_1d | -- | Frequent-pattern / association-rule mining detector: flags transactions violating high-confidence rules mined from normal traces (Apriori) |
| `gaussian_process` | `GaussianProcessDetector` | base | series_1d | -- | Windowed RBF Gaussian-Process one-step-ahead residual detector with calibrated predictive variance (Rasmussen & Williams, 2006) |
| `geo_movement` | `GeoMovementAnomalyDetector` | base | trajectory | fit | Trajectory movement plausibility: velocity, jump, and silence channels |
| `graph_based` | `GraphAnomalyDetector` | base | graph | fit | Graph structure anomaly detection |
| `hawkes` | `HawkesBurstDetector` | base | series_1d | -- | Hawkes self-exciting point-process event-rate / burst detector for count streams (exponential-kernel intensity residuals) |
| `imm` | `IMMDetector` | base | series_1d | -- | Interacting-Multiple-Model switching state-space residual detector (quiet + manoeuvring Kalman bank; Blom & Bar-Shalom, 1988) |
| `kmeans_distance` | `KMeansDistanceDetector` | base | series_1d | fit | Distance-to-nearest-centroid unsupervised anomaly signal (revived cognitive clusterer; ROC-AUC ~0.8-0.98 on ADBench) |
| `nano_safeguard` | `NanoSafeguardDetector` | base | series_1d | -- | Nano-safeguard micro-anomaly detection with hierarchical scanning |
| `page_hinkley` | `PageHinkleyDetector` | base | series_1d | fit | Page-Hinkley cumulative mean-shift test (both one-sided variants; PH_t = m_t - min m_s, alarm at PH_t > lambda; Page, Biometrika 1954) |
| `particle_filter` | `ParticleFilterDetector` | base | series_1d | -- | Bootstrap particle-filter state-space residual detector scoring normalised one-step-ahead predictive innovations |
| `rca` | `RootCauseGraphDetector` | base | series_1d | -- | Root-cause localisation detector: propagates per-node residuals over a causal/service graph to rank candidate root causes with attribution |
| `spatial` | `SpatialAnomalyDetector` | base | matrix_2d | -- | Geographic and spatial relationship anomalies |
| `spectral_residual` | `SpectralResidualDetector` | base | series_1d | -- | Spectral-Residual saliency detector for streaming time series (training-free FFT saliency; Ren et al., KDD 2019) |
| `spiking` | `SpikingNetworkDetector` | base | series_1d | -- | Spiking-neural-network (leaky integrate-and-fire) novelty detector scoring spike-rate divergence from the learned normal regime |
| `spot_evt` | `SPOTDetector` | base | series_1d | fit | SPOT/DSPOT Peaks-Over-Threshold EVT dynamic-threshold detector with risk-budgeted false-positive control (Siffer et al., KDD 2017) |
| `sprt` | `SPRTDetector` | base | series_1d | fit | Wald Sequential Probability Ratio Test for a Gaussian mean shift, deciding at the exact Wald boundaries A=(1-beta)/alpha, B=beta/(1-alpha) (Wald, Ann. Math. Statist. 1945) |
| `srcnn` | `SRCNNDetector` | base | series_1d | fit | SR-CNN: CNN discriminator over Spectral-Residual saliency, trained on synthetic-anomaly-augmented series (Ren et al., KDD 2019; requires torch) |
| `statistical` | `MercuryAnomalyDetector` | base | series_1d | -- | Z-score, percentile, MAD-based detection |
| `survival` | `SurvivalHazardDetector` | base | series_1d | -- | Survival / time-to-event hazard detector (Kaplan-Meier baseline + Cox proportional-hazards deviation) for inter-event-time streams |
| `temporal` | `TemporalAnomalyDetector` | base | series_1d | -- | Time-series patterns and seasonal anomalies |
| `financial_crisis` | `FinancialCrisisRecordAdapter` | economic | record | serves `FinancialCrisisDetector` | Financial crisis prediction |
| `financial_stress` | `FinancialStressAdapter` | economic | rows | -- | FinancialCrisisDetector over measured market and banking metric rows |
| `emp` | `EMPRecordAdapter` | energy | record | serves `EMPDetector` | Electromagnetic pulse detection |
| `avalanche` | `AvalancheProfileAdapter` | geological | record | serves `AvalancheDetector` | Snowpack stability / SK38 skier-stability avalanche detector |
| `eq_tsunami_cascade` | `EqTsunamiCascadeAdapter` | geological | event | serves `EqTsunamiCascadeDetector` | Earthquake -> tsunami cascade (PTWC-style screening on real events) |
| `fire_debris_flow_cascade` | `FireDebrisFlowAdapter` | geological | record | serves `FireDebrisFlowCascadeDetector` | Wildfire -> post-fire debris-flow cascade (USGS Staley 2017 M1 + Cannon I-D) |
| `flood_precipitation` | `FloodPrecipitationAdapter` | geological | rows | -- | FloodDetector over measured and forecast precipitation depth rows (inches) |
| `hurricane_pressure` | `HurricanePressureAdapter` | geological | rows | -- | HurricaneDetector over (central_pressure_mb, environmental_pressure_mb, sst_celsius) |
| `landslide` | `LandslideAdapter` | geological | rows | serves `LandslideDetector` | Landslide prediction |
| `rockfall` | `RockfallAdapter` | geological | record | serves `RockfallDetector` | Rockfall detector with Fukuzono inverse-velocity failure forecasting |
| `subsidence` | `SubsidenceAdapter` | geological | record | serves `SubsidenceDetector` | Subsidence/sinkhole detector over InSAR-style displacement series |
| `tornado_conditions` | `TornadoConditionsAdapter` | geological | rows | -- | TornadoDetector over (cape_j_kg, cin_j_kg, srh_m2_s2, bulk_shear_kt, lcl_m) rows |
| `volcanic` | `VolcanicAdapter` | geological | record | serves `VolcanicEruptionDetector` | Volcanic eruption prediction |
| `wildfire` | `WildfireSceneAdapter` | geological | record | serves `WildfireDetector` | Wildfire prediction |
| `marine_biodiversity` | `MarineEcosystemAdapter` | marine | rows | serves `MarineBiodiversityDetector` | Marine biodiversity threat detection |
| `case_surge` | `CaseSurgeWindowAdapter` | medical | window | serves `CaseSurgeDetector` | Disease-surveillance case surge: week-on-week rise in daily counts, growth-implied R_e |
| `sepsis` | `SepsisRecordAdapter` | medical | record | serves `SepsisDetector` | Possible-sepsis screen: Sepsis-3 qSOFA and SOFA (Singer et al., JAMA 2016) |
| `surge_flood_cascade` | `SurgeFloodCascadeAdapter` | meteorological | event | serves `SurgeFloodCascade` | Hurricane -> storm-surge -> coastal-flood cascade (CO-OPS residual) |
| `acceleration_dynamics` | `AccelerationDynamicsDetector` | physics | series_1d | -- | Kinematic analysis with Lyapunov stability and phase space |
| `physics_integrated` | `PhysicsIntegratedDetector` | physics | series_1d | fit | Unified physics detector with 3R and GOSNN integration |
| `spectral_domain_frequency` | `SpectralDomainFrequency` | physics | series_1d | -- | Neuro-symbolic spectral-domain anomaly detection with per-band scoring |
| `spectral_vibration` | `SpectralVibrationDetector` | physics | series_1d | -- | GNN/CNN spectral analysis with phonon interactions |
| `threat_detection` | `ThreatPayloadAdapter` | security | text | serves `ThreatDetector` | Security threat detection |
| `cme_arrival` | `CMEArrivalAdapter` | space | rows | serves `CMEArrivalDetector` | CME Earth-arrival forecaster (Gopalswamy ESA + Vrsnak DBM, DONKI-fed) |
| `gic` | `GICWindowAdapter` | space | window | serves `GICDetector` | Geomagnetically induced current detector (dB/dt + plane-wave geoelectric proxy) |
| `gic_minute_series` | `GICMinuteSeriesAdapter` | space | rows | -- | GICDetector over a (bx_nt, by_nt) minute series; score is the dB/dt tier over 3 |
| `ionospheric_scintillation` | `ScintillationAdapter` | space | record | serves `IonosphericScintillationDetector` | GNSS ionospheric scintillation detector (S4 / sigma-phi indices) |
| `sep_storm` | `SEPProtonWindowAdapter` | space | window | serves `SEPStormDetector` | Solar energetic-particle / NOAA S-scale radiation-storm detector |
| `solar_gic_cascade` | `SolarGICCascadeAdapter` | space | event | serves `SolarGICCascadeDetector` | Solar eruption -> geomagnetic storm -> GIC grid-impact cascade |
| `solar_storm` | `SolarStormSnapshotAdapter` | space | rows | serves `SolarStormDetector` | Solar storm prediction |
| `solar_wind_geomag` | `SolarWindGeomagAdapter` | space | rows | -- | SolarStormDetector over (solar_wind_speed_km_s, bz_imf_nt); score is NOAA G/5 |
| `xray_flare` | `XrayFlareAdapter` | space | rows | -- | SolarFlareDetector over GOES long-band flux rows; score is NOAA R/5 |
| `uiux_anomaly` | `UIUXAnomalyDetector` | uiux | series_1d | -- | User interaction and behavior anomaly detection |
| `cflow` | `CFlowDetector` | visual | image_batch | fit | Conditional normalizing flow anomaly detection |
| `padim` | `PaDiMDetector` | visual | image_batch | fit | Patch-wise Mahalanobis distance anomaly detection |
| `patchcore` | `PatchCoreDetector` | visual | image_batch | fit | Memory bank + coreset subsampling anomaly detection |
| `reverse_distillation` | `ReverseDistillationDetector` | visual | image_batch | fit | Reverse knowledge distillation with OCE bottleneck |
| `stfpm` | `STFPMDetector` | visual | image_batch | fit | Student-Teacher Feature Pyramid Matching |
| `lavad` | `LAVADDetector` | vlm | video_frames | fit | Training-free LLM-based VAD (CVPR 2024) |

## registry only (39)

| name | class | category | modality | notes | description |
|---|---|---|---|---|---|
| `adaptive_anomaly` | `AdaptiveAnomalyDetector` | base | matrix_2d | fit | Profile-calibrated detector routing to temporal/covariance/high-dim backends |
| `info_geometry` | `InformationGeometryDetector` | base | matrix_2d | fit | Fisher-Rao information-geometry OOD detector over a fitted reference |
| `multivariate_ts` | `MultivariateTSDetector` | base | windows_3d | fit | LTG multivariate windowed time-series detector (LSTM/temporal-conv/graph features) |
| `topological` | `TopologicalAnomalyDetector` | base | matrix_2d | fit | Persistent-homology profile scoring (Betti, entropy, landscapes, Wasserstein) |
| `dual_student` | `DualStudentDistillation` | distillation | image_batch | fit | Dual-student knowledge distillation for AD |
| `chronos` | `ChronosAdapter` | foundation | series_1d | -- | Amazon Chronos time-series foundation model |
| `foundation_ensemble` | `FoundationEnsemble` | foundation | series_1d | -- | Ensemble of foundation model adapters |
| `matrix_profile` | `MatrixProfileAdapter` | foundation | series_1d | -- | STUMPY Matrix Profile for time-series anomalies |
| `earthquake` | `EarthquakeDetector` | geological | series_1d | -- | P/S-wave spectrogram + STA/LTA earthquake detection (USGS live catalog wiring) |
| `flood` | `FloodDetector` | geological | series_1d | -- | Flood prediction with refactoring engine optimization |
| `hurricane` | `HurricaneDetector` | geological | series_1d | -- | Hurricane/cyclone/typhoon prediction with SST and resonance amplification |
| `meteor` | `MeteorDetector` | geological | series_1d | -- | Bayesian meteor/NEO threat assessment over NASA NeoWs + JPL fireball/Sentry |
| `tornado` | `TornadoDetector` | geological | series_1d | -- | Tornado prediction with Doppler radar and FFT resonance analysis |
| `tsunami` | `TsunamiDetector` | geological | series_1d | -- | Waveform-spectrum tsunami detection from observed sea-level series |
| `intelligence_fusion` | `IntelligenceFusionEngine` | intelligence | record | -- | Multi-source intelligence fusion |
| `psyop` | `PSYOPAnalyzer` | intelligence | series_1d | -- | Psychological operations analysis |
| `medical_abms` | `ABMSDisciplineDetector` | medical | record | -- | Medical discipline detection |
| `atmospheric_river` | `AtmosphericRiverDetector` | meteorological | series_1d | -- | IVT computation and Ralph et al. (2019) AR-scale classification |
| `derecho` | `DerechoDetector` | meteorological | series_1d | -- | Derecho swath detector (Johns & Hirt 1987 criteria) |
| `drought` | `DroughtDetector` | meteorological | series_1d | -- | SPI/SPEI drought index detector (McKee et al. 1993; Vicente-Serrano et al. 2010) |
| `dust_storm` | `DustStormDetector` | meteorological | series_1d | -- | Dust-storm/haboob detector (emission potential + visibility class) |
| `hail` | `HailDetector` | meteorological | series_1d | -- | SHIP significant-hail parameter detector (SPC formulation) |
| `heatwave` | `HeatwaveDetector` | meteorological | series_1d | -- | Percentile-climatology heatwave detector with Excess Heat Factor (Perkins & Alexander 2013; Nairn & Fawcett 2015) |
| `lightning` | `LightningDetector` | meteorological | series_1d | -- | 2-sigma lightning jump severe-weather precursor (Schultz et al. 2009) |
| `winter_storm` | `WinterStormDetector` | meteorological | series_1d | -- | Winter/ice-storm detector (precip-type + FRAM ice accretion) |
| `affective` | `AffectiveAnomalyModel` | model | series_1d | -- | Emotional state anomalies |
| `astrophysical` | `AstrophysicalAnomalyModel` | model | series_1d | -- | Cosmic signal anomalies |
| `biometric` | `BiometricAnomalyModel` | model | series_1d | -- | Face/biometric anomalies |
| `chemistry` | `ChemistryAnomalyDetector` | model | series_1d | -- | Chemical anomaly detection |
| `consciousness` | `ConsciousnessPreservationModel` | model | series_1d | -- | Consciousness preservation analysis |
| `lstm_autoencoder` | `LSTMAutoencoderDetector` | model | series_1d | fit | Trained LSTM-autoencoder multivariate sequence reconstruction |
| `neural_cognitive` | `NeuralCognitiveModel` | model | series_1d | -- | Brain activity anomalies |
| `parapsychology` | `ParapsychologyDetector` | model | series_1d | -- | Psi phenomena detection |
| `quantum` | `QuantumAnomalyModel` | model | series_1d | -- | Quantum state anomalies |
| `neurosymbolic` | `NeurosymbolicEngine` | neurosymbolic | series_1d | -- | Hybrid neural-symbolic reasoning |
| `realtime_threat` | `RealTimeThreatDetector` | security | series_1d | -- | Projection + kNN + robust-covariance ensemble, training-calibrated |
| `schumann_resonance` | `SchumannResonanceDetector` | space | series_1d | -- | Earth resonance detection |
| `solar_flare` | `SolarFlareDetector` | space | series_1d | covered by `xray_flare` | GOES X-ray flare classification with observed-Kp geomagnetic context |
| `drone` | `DroneAnomalyDetector` | uav | series_1d | -- | Drone fault detection: RADD phase invariants + single-row Mercury ensemble |

## engine only (2)

| name | class | category | modality | notes | description |
|---|---|---|---|---|---|
| `dimensional` | `DimensionalAnalyzer` | base | series_1d | -- | High-dimensional data anomalies |
| `anyanomaly` | `AnyAnomalyDetector` | vlm | video_frames | fit, ext | Zero-shot customizable VAD with LVLM (WACV 2026) |

## library-only (domain API) (1)

| name | class | category | modality | notes | description |
|---|---|---|---|---|---|
| `timegpt` | `TimeGPTAdapter` | foundation | series_1d | ext | Nixtla TimeGPT foundation model adapter |

## Building blocks and wrappers (importable, deliberately not manifest entries)

- ``detectors/statistical_extended.py``: ``MADDetector``, ``LOFDetector``,
  ``DBSCANDetector``, ``MCDDetector``, ``CUSUMDetector``, ``GrubbsTest``,
  ``GESDTest`` -- served through the ``extended_statistical`` ensemble entry.
- ``detectors/drift.py``: ``KolmogorovSmirnovDriftDetector``,
  ``PopulationStabilityIndexDetector``, ``ChiSquaredDriftDetector``,
  ``CovarianceDriftDetector``, ``PrincipalAxisDriftDetector``,
  ``OnlineDriftDetector`` -- served through ``EnsembleDriftDetector``, which
  the engine and ``ml/online_learning.py`` already drive.
- ``detectors/adaptive_detector.py``: ``CovarianceAwareDetector``,
  ``TemporalPatternDetector`` -- composed by the ``adaptive_anomaly`` entry.
- ``core/conformal_prediction.py`` / ``detectors/conformal_martingale.py``:
  conformal wrappers constructed around a caller's detector.
