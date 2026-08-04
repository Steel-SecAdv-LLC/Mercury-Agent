# Hazard Regression Gate — per-hazard behavior floors

Applies to Mercury Agent **v2.1.x**. Last updated: 2026-08-04.

## TL;DR

The hazard transparency wave (volcanic, space, tsunami/earthquake, and
meteorological detectors) replaced untrained-network theater with
deterministic physics paths — but nothing pinned their *behavior*.
`benchmarks/hazard_regression_guard.py` + `hazard_domain_baseline.json` close
that: every guarded detector runs over committed, hash-pinned scenario sets,
standard skill scores (`omni_mercury_engine/evaluation/hazard_metrics.py` —
each formula unit-tested against a worked literature example) are compared to
measured-minus-margin floors, and CI (`.github/workflows/hazard-regression.yml`,
required check `ci/hazard-regression`) fails on any crossing. Fully offline
and bit-deterministic.

**What these floors are — and are not.** Six of the seven domains run over
**constructed** scenario sets (seeded, physics-shaped, disclosed per-row in
the table below), and their baselines sit at or near perfect scores. Those
rows are **regression pins on the physics paths** — they prove the detector
still detects the scenarios it was designed to detect, and they catch any
change that degrades that. They are **not real-world skill claims**: no
constructed-scenario number here may be quoted as detection skill on real
events. The single exception is **solar**, whose windows are real measured
SWPC/GOES data (fetched, cached in-repo, `label_source: measured`) — that
row is a genuine real-data floor, and it is the pattern the other six
domains should migrate to (fetch-once, hash-pin real event windows) as
follow-up work.

## What is gated

| Domain | Detector path | Gated metrics | Pinned baseline | Label source |
|---|---|---|---|---|
| tornado | `TornadoDetector` velocity-couplet physics | POD, FAR, CSI, HSS, mean warning lead | 1.00 / 0.00 / 1.00 / 1.00 / 6.83 min | constructed (seed 4207) |
| flood | `FloodDetector` precip/gauge/soil physics | POD, FAR, CSI, HSS, mean warning lead | 1.00 / 0.00 / 1.00 / 1.00 / 6.83 h | constructed (explicit) |
| hurricane | `HurricaneDetector` pressure + wind kinematics | POD, FAR, CSI, HSS | 1.00 / 0.00 / 1.00 / 1.00 | constructed (seed 4210) |
| earthquake | `EarthquakeDetector` STA/LTA + S-P | POD, FAR, CSI, S-P distance MAE | 1.00 / 0.00 / 1.00 / 0.324 km | constructed (seed 4208) |
| tsunami | `TsunamiDetector` DART amplitude + resonance | POD, FAR, CSI | 1.00 / 0.00 / 1.00 | constructed (seed 4209) |
| volcano | `VolcanicEruptionDetector` multi-precursor | alert-level exact + within-one (ordinal) | 1.00 / 1.00 | constructed (seed 4211) |
| solar | `SolarStormDetector` NOAA flare chain + Boyle Kp | flare-class exact accuracy, Kp MAE, G-bucket accuracy | 1.00 / 1.089 / 0.873 | **measured (real SWPC)** |

Margins (justified in the baseline metadata): rate metrics ±0.05 absolute
(< one scenario flip on every set), lead times −15 % relative, MAE metrics
+15 % relative with a small min-abs slack. The evaluation is deterministic, so
margins define tolerated degradation, not numerical drift.

**Non-vacuous by test**: `tests/benchmarks/test_hazard_regression_guard.py`
computes the degenerate forecasters from the same committed sets and asserts
every floor beats them — always-alarm CSI/FAR, majority-class accuracy for
volcano alert levels and flare classes, and for Kp the ceiling must sit below
both the always-0 predictor (MAE 2.56) and the climatology-mean predictor
(MAE 1.28). The slow tier runs the live measurement twice (bit-identical) and
regenerates every constructed set from its seed, asserting the manifest
content hashes match.

## Scenario provenance

Everything lives in `benchmarks/hazard_scenarios/`; `manifest.json` pins a
content hash per file (JSON: file bytes; NPZ: canonical array content, since
zip containers embed timestamps) and the guard refuses to run on a mismatch.

**Real recorded data (`label_source: "measured"`)** — solar, fetched once on
2026-07-09T05:43:00Z from NOAA SWPC (raw-payload sha256 + row counts recorded
in each window file's provenance block):

* `solar_kp_windows.json` — 55 windows: SWPC propagated real-time solar wind
  (speed, IMF By/Bz; `products/geospace/propagated-solar-wind.json`) averaged
  per observed 3 h planetary-Kp interval (`products/noaa-planetary-k-index.json`);
  windows with < 60 valid wind minutes dropped, never imputed. The week spans
  Kp 0.33–7.33 including a real G3 storm (2026-07-04). Genuine forecast
  skill: Boyle-index Kp vs independently measured ground-based Kp.
* `solar_flare_windows.json` — 28 six-hour windows of GOES primary X-ray flux
  (`json/goes/primary/xrays-7-day.json`), classes B/C/M/X represented, with
  the SWPC flare-event list attached as per-window evidence. Flare-class
  labels are *definitional* (the NOAA class **is** a flux threshold), so this
  set gates the classification chain's correctness, not forecast skill —
  stated in the file's provenance block.
* `raw/` keeps the two small raw snapshots (Kp index, flare-event list)
  verbatim; the two large raw streams are pinned by sha256 in the provenance
  blocks instead of being committed.

**Constructed physics scenarios (`label_source: "constructed"`)** — tornado,
flood, hurricane, earthquake, tsunami, volcano. No allow-listed feed provides
labelled raw sensor series for these detectors (they consume Doppler velocity
fields, seismic traces, DART records, gridded winds, gauge observations —
e.g. `earthquake.usgs.gov` serves catalogs, not station waveforms). Each set
is built by the committed generator (`generate_scenarios.py`, fixed per-domain
seeds, NumPy `default_rng` stream stability) against the detector's
documented input contract, mirroring the transparency-test fixtures
(`tests/detectors/test_*_honesty.py`). Labels are the physical ground truth
of the constructed situation (a 25 m/s velocity couplet **is** a mesocyclone
signature), never the detector's output. Every `manifest.json` entry carries a
`construction` description.

## Run locally

```bash
pytest tests/evaluation/test_hazard_metrics.py -q            # metric formulas
pytest tests/benchmarks/test_hazard_regression_guard.py -q   # floors + gate logic (offline)
python benchmarks/hazard_regression_guard.py --check         # the CI gate (~12 s)
```

## Updating the baseline legitimately

1. If (and only if) a detector or scenario-set change is intentional and
   reviewed, regenerate scenarios if needed
   (`python benchmarks/hazard_scenarios/generate_scenarios.py --constructed`,
   or `--solar-from-raw DIR --fetched-at TS` with fresh SWPC snapshots), then
   `python benchmarks/hazard_regression_guard.py --update`.
2. Commit the baseline (and manifest/scenario) diff **in the same PR** as the
   change that motivated it and explain the metric movement in the PR body.
   The guard cross-pins scenario hashes into the baseline, so a scenario-set
   change without a re-pin fails `--check` loudly.
3. Never move a floor to manufacture a pass: the non-vacuity tests will
   reject floors that fall to degenerate-forecaster level (e.g. a Kp ceiling
   above the climatology predictor).

## Explicit exclusions (documented in the registry, tripwired where possible)

* **Hurricane track error** — no track model exists; the transparency wave
  deliberately *removed* the never-computed `track_forecast` /
  `landfall_probability` / `time_to_landfall_hours` fields. Gating a track
  metric would fabricate a capability. A tripwire fails the guard if the dead
  fields regrow.
* **Earthquake magnitude MAE** — the untrained detector transparently emits
  `estimated_magnitude=None`; a tripwire fails the guard if a magnitude is
  ever fabricated while untrained. `hazard_metrics.magnitude_error` and
  `location_error_km` are implemented + unit-tested for the day trained
  weights / a multi-station location path exist.
* **Volcano VEI accuracy** — the physics-path VEI is documented as a coarse
  precursor-magnitude proxy; `hazard_metrics.vei_accuracy` stands ready for a
  real forecast model.
* **Flare Brier score** — no per-class probability surface exists on the
  flare path; `hazard_metrics.brier_score` is implemented + unit-tested.
* **Flare within-one accuracy** — reported but ungated: C/M windows dominate
  the recorded week, so always-M scores 27/28 within-one; exact accuracy
  (always-M: 0.50) carries the gate.
* **Kp G-bucket accuracy** — gated as a regression tripwire, but the recorded
  week is quiet-dominated (always-G0 scores ~0.89), so skill-over-climatology
  is certified by the Kp MAE bound instead.
