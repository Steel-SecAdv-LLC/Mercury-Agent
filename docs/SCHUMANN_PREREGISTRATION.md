# WS-C — Schumann sub-net: pre-registration

Registered **before** running the evaluation, to prevent post-hoc flexibility.
Commit this file in the same change as the evaluation harness; do not edit the
protocol after seeing results (edits must be dated and justified below).

## Hypothesis

A Schumann/ELF spectral encoder, trained on windows weakly labelled by
independent space-weather drivers, separates driver-coincident windows from
geomagnetically-quiet windows better than chance.

## Data

* **Labels (real, public domain):** NOAA SWPC planetary **Kp index** (storm =
  `Kp ≥ 5`) and **GOES X-ray long band** (M-class `≥ 1e-5 W/m²`, X-class
  `≥ 1e-4`). Built by `space/schumann_labeling.py`; provenance (URL + SHA-256 +
  fetch time + thresholds + lag) recorded in `artifacts/schumann_label_catalog.json`.
  Labels are **proxy / event-coincidence**, not ground truth; noise disclosed by
  `label_noise_disclosure()`.
* **Physical lag (fixed):** flares → `[onset, onset+60min]` (prompt SID);
  storms → `[start−1h, end+3h]` (storm-time response).
* **ELF signal:** an openly-licensed real ELF/Schumann corpus is **required** to
  lift quarantine. Licensing vetted (see "ELF source licensing" below); no
  source could be cleared **and** ingested in this environment. Therefore the
  encoder is exercised on a **clearly-labelled synthetic, physically-grounded
  ELF generator** keyed to the real driver windows. Synthetic results **cannot**
  lift quarantine and are never presented as real.

## Fixed evaluation protocol

* **Split:** temporal — earliest 70% train, latest 30% test (no shuffle; avoids
  leakage across the autocorrelated ELF stream). **Registered fallback:** if the
  available labelled-event distribution makes a fold single-class (space-weather
  events are often temporally clustered, so a short window can put all positives
  on one side), fall back to a fixed seeded (`12345`) stratified split and record
  `split_used` in the artifact. This is anticipated here and is not post-hoc.
* **Metric:** ROC-AUC of the encoder's anomaly score vs the weak label
  (primary); oracle-F1 secondary. Computed with `mercury_ml` (no sklearn).
* **Seeds:** 0, 1, 2; report mean ± spread.
* **Encoder:** `SchumannHarmonicAnalyzer` spectral features → linear head,
  trained supervised on the weak labels.

## A-priori bar (set before running)

* **Lift quarantine only if** mean ROC-AUC ≥ **0.70** on a **real** openly-
  licensed ELF corpus, across all 3 seeds, with the label-noise caveat
  acknowledged.
* On **synthetic** signal, the module **stays quarantined regardless of score**
  — synthetic performance validates the pipeline plumbing only.

## ELF source licensing (vetted)

| Source | Status | Decision |
|---|---|---|
| Tomsk SOS (sos70.ru) | Image-based, **explicitly copyrighted**, no API | Display/reference only — not a training corpus (per task constraint) |
| HeartMath GCMS | Research feed; redistribution **not openly licensed** | Not cleared |
| VLF Openlab / vlf.it (R. Romero) | Personal recordings; **no clear open license** | Not cleared |
| Zenodo / PANGAEA | CC-licensed Schumann sets exist but none **ingested + hash-pinned** in this environment | Deferred — candidate for the real run |

→ **Fallback taken:** synthetic-but-physically-grounded ELF, clearly labelled.

## Verdict rule

Record the measured numbers either way. `KEEP`/lift only if the real-ELF bar is
cleared; otherwise `QUARANTINE` with the result and the data blocker recorded.

---

## Results (added post-run — protocol above unchanged)

Run: `python benchmarks/schumann_eval.py --n 1000 --epochs 40`
(live NOAA fetch; content hashes recorded in `artifacts/schumann_eval.json`).

* **Real labels built successfully** — the prior session's "no labels possible"
  claim is refuted. The current NOAA window yielded 21 M-class flare windows
  (0 storms; a quiet week), positive fraction ≈ 0.10.
* **Temporal-split degeneracy observed (as anticipated):** events were
  temporally clustered, so the registered stratified fallback was used
  (`split_used = stratified_fallback_temporal_degenerate`).
* **Synthetic encoder run (original, PR #262):** per-seed ROC-AUC
  `[0.974, 1.000, 0.227]`, mean **0.734** — and **seed-unstable** (one seed
  collapsed to a sign-inverted solution).

## WS-C diagnosis (this round): the instability was an optimisation artifact

"Seed-unstable" was a *symptom*, not a verdict. `benchmarks/schumann_diagnostic.py`
root-causes it by isolating one factor at a time (offline + deterministic; no
NOAA), sweeping **optimisation regime** × **objective** over 6 seeds:

| regime | objective | per-seed AUC | collapse rate |
|---|---|---|---|
| **full-batch** | sigmoid+BCELoss (historical) | `[1,1,0.37,0.75,0.90,1]` | 0.17 |
| **full-batch** | logits+BCEWithLogitsLoss | `[1,1,0.35,0.66,0.88,1]` | 0.17 |
| full-batch | sigmoid, lr 3e-4 | `[1,1,0.03,1.0,0.83,1]` | 0.17 |
| **mini-batch** | sigmoid+BCELoss | `[1,1,1,1,1,1]` | **0.00** |
| **mini-batch** | logits+BCEWithLogitsLoss | `[1,1,1,1,1,1]` | **0.00** |

**Root cause: the optimisation regime, not the objective, the initialisation, or
the data.** The original harness trained **full-batch** — one Adam update per
epoch, ~`epochs` updates total — too few for some seeds' inits to escape a
sign-inverted basin. The collapse persists under both objectives and *worsens* at
a lower LR (AUC 0.03), so it is not a saturating-sigmoid or step-size problem.
**Mini-batch SGD removes it completely** (every seed → AUC ~1.0). The fix is now
the default in `schumann_eval.run_seed` (mini-batch + `BCEWithLogitsLoss` on the
newly-exposed `SchumannHarmonicAnalyzer.confidence_logits`; inference is
byte-identical, no parameter renames). Evidence: `artifacts/schumann_diagnostic.json`.

* **Re-run on the REAL NOAA labels with the stable recipe:** per-seed ROC-AUC
  **`[1.0, 1.0, 1.0]`** (was `[0.974, 1.000, 0.227]`); positive fraction 0.107.
  The instability is gone.

**→ VERDICT: QUARANTINE — now on ONE documented blocker, not two.** The
training-stability blocker is **resolved and root-caused** (a harness
optimisation bug, not an intrinsic ill-posedness). The quarantine stands solely
on the **data** blocker: the signal is synthetic and no openly-licensed real ELF
corpus could be cleared + ingested here, and per this pre-registration synthetic
performance **cannot** lift quarantine regardless of score. Lifting it now needs
exactly one thing — a hash-pinned real ELF corpus — to which the (now stable)
pipeline applies directly.
