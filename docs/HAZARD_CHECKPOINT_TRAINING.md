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

## Program result (2026-07-10)

Every one of the eleven hooks now has a **real, reachable data path** (all
category **a**) and has been driven through the full fetch → build → train →
evaluate → gate pipeline on real measured data. Outcome: **7 shipped through
the merit gate, 4 honest "physics-wins" refusals** — each refusal backed by a
committed evaluation record under `artifacts/hazard_training/`. Four shipped
checkpoints carry a **validation-selected operating point** (the storm/alert
decision threshold chosen on the validation years against the same
non-regression constraints the gate enforces, then consumed decision-only by
`load_neural_weights`); the Kp/probability estimate itself is never rescaled.

A refusal is the gate working, not the pipeline failing: a checkpoint that
regresses an operational metric (recall or false-alarm rate at the deployed
rule) does not ship on a primary-metric win alone. Where a refusal reflects a
recall/false-alarm trade an operator might reasonably prefer, it is flagged
for owner ratification in the PR, not silently discarded.

| Hook | Checkpoint | Cat. | Data (real, sha256-pinned) | Result |
|---|---|---|---|---|
| `solar_storm` | `solar_storm_geomag` | **a** | NASA SPDF OMNI2 hourly solar wind + observed Kp, 2005–2024 (20 files), GFZ Kp cross-check | **SHIPPED** — Kp MAE 0.574 vs 1.054; dual-rule operating point (see below): storm recall **75.9% vs 57.4%**, FAR **2.61% vs 3.14%**, CSI 0.386 vs 0.265, AUC 0.972 vs 0.845 |
| `hurricane_wind` | `hurricane_era5` | **a** | ARCO-ERA5 10 m u/v wind patches (public GCS mirror, no CDS key) + IBTrACS v04r01 labels, 1990–2024 | **SHIPPED** — intensity MAE **5.39 vs 6.12 kt** (positives 14.3 vs 17.4), detection AUC 0.998, category acc 60.8% vs 54.7%; all constraints pass |
| `wildfire_ignition` | `wildfire_firms` | **a** | NASA FIRMS VIIRS science-quality country archives (keyless), California 0.04° grid, 2012–2024 | **SHIPPED** — next-day AUC **0.875 vs 0.661** (persistence baseline; CI [0.197, 0.233]); recall 67.0% vs 59.3%, FAR **9.6% vs 29.1%**, Brier 0.134 vs 0.220; op point τ=0.579 |
| `landslide_stability` | `landslide_coolr` | **a** | NASA GLC/COOLR events (AGOL mirror) + CHIRPS v2.0 daily rainfall vs 1981–2006 climatology, 2007–2024 | **SHIPPED** — AUC **0.8498 vs 0.8064** (train-fitted Caine-style rain baseline; CI [0.031, 0.057]); recall 65.5% vs 61.9%, FAR 14.5% vs 15.9%, Brier 0.157 vs 0.266; op point τ=0.6665 |
| `volcanic_eruption` | `volcanic_avo_seismic` | **a** | Smithsonian GVP onsets + HANS VAN cross-check + EarthScope FDSN AV-network miniSEED; SCOPED to 8 named AVO volcanoes | **SHIPPED** — AUC **0.9402 vs 0.7096** (RSAM baseline 0.698; CI [0.140, 0.319]); recall 83.9% vs 78.6%, FAR **8.8% vs 53.4%**, all 4 covered onsets caught at 13 d lead; op point τ=0.417 |
| `tornado_radar` | `tornado_nexrad` | **a** | NEXRAD Level-II velocity (Unidata mirror) + SPC WCM tornado reports, 2011–2023 | **SHIPPED** — AUC **0.810 vs 0.687**; mesocyclone recall 0.35 vs 0.225, FAR **0.034 vs 0.138** (4× lower); op point τ=0.895. *Caveat: paired-bootstrap 95% CI on the AUC delta [−0.002, 0.245] marginally includes zero on a small test set (n=98) — recorded for transparency, flagged for owner review* |
| `consciousness_field` | `reg_deviation_gcp` | **a** | Global Consciousness Project per-second REG streams (Wayback archive, 2012–2024); fault-injection labels (bias / common-mode / stuck-bit), true by construction | **SHIPPED** — REG statistical-deviation detector; mixed-fault AUC **0.7875 vs 0.7246** vs the pre-registered closed-form Stouffer+χ² rule (CI [0.043, 0.080]); power@1%FAR 0.368 vs 0.337, Brier 0.183 vs 0.267. Consciousness-field reading is the *contested hypothesis under study*, neither asserted nor denied |
| `earthquake_precursor` | (`earthquake_precursor_ca`) | **a** | USGS ComCat FDSN, California 1980–2024 M≥2.5; `seismicity-catalog-v2` stacks the Reasenberg–Jones causal forecast as inputs | **REFUSED — clustering baseline wins ranking**: learned wins log-loss 0.00414 vs 0.00629 (CI excludes 0) but AUC 0.8895 < RJ 0.8975; auc constraint refused. Record: `artifacts/hazard_training/earthquake_precursor.eval.json` |
| `seismic_wave` | (`seismic_stead`) | **a** | STEAD (Mousavi et al. 2019, CC-BY-4.0), SeisBench mirror; balanced Z-component subset streamed via HTTP Range | **REFUSED — deployed-rule FAR regresses**: learned wins AUC **0.983 vs 0.923** and recall **0.995 vs 0.442** (low-SNR recall **0.986 vs 0.103**) but FAR 3.70% vs 2.84% at the deployed rule. A recall/FAR trade favouring the learned model for early warning — **flagged for owner ratification.** Record: `artifacts/hazard_training/seismic_stead.eval.json` |
| `tsunami_waveform` | (`tsunami_dart`) | **a** | NOAA NDBC DART bottom-pressure archives + NCEI HazEL arrival labels; detided 24 h windows | **REFUSED — deployed-rule FAR regresses**: learned wins AUC 0.861 vs 0.747 (CI [0.075, 0.152]) and event recall 51% vs 6.7%, but deployed FAR 0.29% vs 0.097% (validation cannot resolve FAR below the ceiling). Record: `artifacts/hazard_training/tsunami_dart.eval.json` |
| `schumann_harmonics` | (`schumann_sierra_nevada`) | **a** | Sierra Nevada ELF observatory raw int16 (Zenodo, CC-BY-4.0; Salinas et al. 2022), ranged remote-ZIP reads; GFZ Kp labels | **REFUSED — physics alarms always**: learned wins ranking AUC 0.568 vs 0.374 and type accuracy 0.60 vs 0.24, but the deterministic FFT physics alarms on every held-out hour (recall 1.0 at FAR 1.0), so the recall floor is unbeatable. The alarm-always physics is itself a detector defect the record documents. Record: `artifacts/hazard_training/schumann_sierra_nevada.eval.json` |

Category **a** = real labeled data fetchable and pipeline runs here; **b** =
real data exists but needs archives/credentials absent here (fail-loud with
the documented requirement); **c** = no real corpus exists for the
architecture's input contract. This program moved all eleven hooks to **a**.
Parenthesised checkpoint names are the basenames a future re-run would ship
under; no such file exists while the gate refuses.

## Shipped: `solar_storm_geomag`

Trained on 20 years (2005–2024) of the real NASA SPDF OMNI2 hourly archive —
multi-spacecraft L1 solar-wind/IMF measurements paired with the *observed*
planetary Kp — with the OMNI2 Kp parsing cross-checked against the GFZ
Potsdam definitive Kp service over the 2023-04 G4 storm window. Temporal
split: train 2005–2018, val 2019–2021, test **2022–2024** (25,846 held-out
hours spanning the ascent of solar cycle 25, including the 2024-05-10 G5
superstorm).

Held-out comparison, both models running through
`SolarStormDetector.predict_solar_storm` on identical hours, each scored on
its **own deployed decision rule** (physics: Kp≥5; learned: the dual rule
below):

| Metric | Learned | Physics (Boyle index) |
|---|---|---|
| Kp MAE | **0.574** | 1.054 |
| Kp RMSE | **0.754** | 1.344 |
| G-bucket accuracy | **96.0%** | 94.8% |
| Storm (Kp≥5) AUC | **0.972** | 0.845 |
| Storm recall @ deployed rule | **75.9%** | 57.4% |
| Storm false-alarm rate @ deployed rule | **2.61%** | 3.14% |
| Storm CSI @ deployed rule | **0.386** | 0.265 |

The merit gate compares the primary metric (Kp MAE): learned wins by 45%,
**and** enforces secondary non-regression constraints on storm recall,
false-alarm rate, and AUC at the deployed operating point — all pass.

**Operating point (owner-ratifiable).** The MSE-trained Kp point estimate
regresses toward the mean on a ~3%-storm dataset, so thresholding it at Kp≥5
alone halves recall versus physics (the first shipped revision did exactly
that: 30.4% recall). The BCE-trained storm-probability head carries a far
better ranking (AUC 0.972), so storm **onset** now uses a **dual rule**,
`(kp_pred ≥ 5) OR (storm_prob ≥ τ)`, with τ selected on the validation years
to maximise CSI subject to a recall floor at `max(physics recall, 0.55)` and
a false-alarm ceiling at `0.8 × physics FAR`. The threshold is carried in the
checkpoint payload, validated on load, and applied to the **alert level
only** — the emitted `kp_index` remains the honest regression estimate.
Result: the learned path now dominates physics on recall *and* false-alarm
rate *and* ranking simultaneously. The detector loads this checkpoint via
`load_neural_weights()` (no arguments → shipped default, provenance logged;
corrupt or missing files raise).

## Merit gate: secondary non-regression constraints

`EvaluationOutcome` carries optional `constraints` — metrics on which the
learned model must **match or beat** physics (not merely win the primary
metric). `ship_checkpoint` refuses if any constraint regresses, so a
checkpoint cannot ship on a primary-metric win while quietly regressing an
operational metric. This is why the four refused hooks above did not ship:
each wins its primary metric but regresses a deployed-rule constraint
(false-alarm rate for seismic/tsunami, recall against an alarm-always
baseline for schumann, ranking AUC for earthquake). The refusal, with its
full committed evaluation record, **is** the deliverable for those hooks.
