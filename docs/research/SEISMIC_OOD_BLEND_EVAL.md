# Seismic OOD posture: physics+learned blend — evaluated on held-out STEAD, not adopted

## Question

`EarthquakeDetector` serves an **either/or** alerting posture: the shipped
`seismic_stead` CNN (learned lane, ratified on real held-out STEAD) when trained
weights are present, the STA/LTA + band-resonance physics as the disclosed
fallback otherwise. Because the learned CNN scores poorly on out-of-distribution
inputs (synthetic toys: the hazard-regression guard pins `trained_pod ≈ 0.14`),
it seemed *plausible* that **blending** physics with the learned score would be
more robust in the low-SNR / OOD regime. That hypothesis was unratified. This
note records its real evaluation and the resulting decision: **do not adopt a
blend.**

## Method (real held-out STEAD)

Each held-out trace was scored through the production `EarthquakeDetector.predict_earthquake`
API on both lanes — physics = `EarthquakeDetector(load_shipped_weights=False)`
(alert at 0.96), learned = `EarthquakeDetector()` (shipped `seismic_stead.pt`,
alert at the checkpoint operating point τ=0.998360, temperature=1.269466) — and
four candidate alert rules were compared on the identical set:

- **LEARNED-only** (the ratified default): `conf > τ`.
- **PHYSICS-only**: `conf > 0.96`.
- **BLEND-max**: alert if *either* lane alerts (OR-of-alarms); ranking = `max(physics, learned)`.
- **BLEND-mean**: `0.5·(physics + learned)`, thresholded at the label-free mean of the two rules (0.97918).
- **BLEND-gated**: trust learned, but let physics fire when the learned score sits in its uncertain band `[0.90, τ]` (a label-free gate — the honest "rescue the OOD low-SNR events" strategy).

Data: **700 earthquake + 700 noise = 1,400** balanced traces, held-out test
years (2017, 2018, 2020 — the archive carries no 2019 traces), seed
**20260716** (deliberately distinct from the shipped seed 20260709 → an
independent draw). Streamed through the production harness
(`_stream_split` / `BlockCachedRangeReader` HTTP-Range over the 91 GB
`waveforms.hdf5`): **89.7 MB streamed** in 1,590 range calls; 8 traces re-read
byte-identical through a fresh h5py session. The LEARNED-only lane reproduced
the committed provenance (AUC 0.9949 / recall 0.968 / FAR 0.0158) on this fresh
draw — an independent re-validation of the shipped checkpoint.

## Results (each strategy at its deployed rule; SNR terciles 19.47 / 30.30 dB)

| strategy | AUC | recall@rule | FAR@rule | recall low-SNR | recall mid | recall high |
|---|---|---|---|---|---|---|
| **LEARNED-only** (ratified) | **0.99502** | 0.9600 | **0.0143** | **0.9021** | 0.9784 | 1.0000 |
| PHYSICS-only (@0.96) | 0.92631 | 0.4686 | 0.0257 | 0.1191 | 0.4526 | 0.8369 |
| BLEND-max (either alerts) | 0.98066 | 0.9614 | 0.0357 | 0.9021 | 0.9828 | 1.0000 |
| BLEND-mean | 0.98999 | 0.4686 | 0.0043 | 0.1191 | 0.4526 | 0.8369 |
| BLEND-gated (physics rescues in [0.90, τ]) | 0.99506 | 0.9614 | 0.0143 | 0.9021 | 0.9828 | 1.0000 |

Paired bootstrap on AUC vs LEARNED-only (B=1000, seed [20260716, 99]):
BLEND-max **−0.01429** [−0.02150, −0.00794] (worse); BLEND-mean **−0.00494**
[−0.00879, −0.00093] (worse); BLEND-gated **+0.00003** [0.00000, +0.00013]
(tie, not a strict win); PHYSICS-only −0.06859 (worse).

## Verdict: blend remains UNRATIFIED — the either/or posture stands

No blend clears the bar (strictly higher AUC **and** recall-not-lower **and**
FAR-not-higher, ideally with a low-SNR uplift):

- **BLEND-max** lowers AUC and **more than doubles the false-alarm rate**
  (0.0357 vs 0.0143) to buy +0.0014 recall (≈1 event of 700) — a FAR-for-recall
  trade, not an improvement.
- **BLEND-mean** collapses recall to the physics level (0.4686; low-SNR 0.1191).
- **BLEND-gated** matches learned on FAR and low-SNR recall and is
  statistically indistinguishable on AUC (CI lower bound 0.00000) — a tie.

The OOD hypothesis is **empirically falsified** on real STEAD: physics' own
low-SNR recall is **0.1191** versus the CNN's **0.9021**, so physics has
essentially nothing to contribute exactly where robustness would matter, and
every blend's low-SNR recall is identical to or worse than learned-only. The
learned CNN is already the far stronger ranker (AUC 0.995 vs 0.926); mixing in
the weaker physics score only injects ranking noise.

**Decision:** keep the ratified either/or posture (learned serves by default,
physics is the disclosed fallback). No code change. This evaluation is not
CI-gated (it requires streaming ~90 MB of real STEAD from the SeisBench mirror);
re-run against the `ml.hazard_training.seismic_wave` harness with a fresh seed
to reproduce.
