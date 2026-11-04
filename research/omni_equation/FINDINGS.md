# Omni-Equation — Measured Findings (real detector, real data)

Last updated: 2026-07-11.

**What this is.** A direct measurement to decide whether the "omni-equation"
direction clears a speed/cost/accuracy bar — run on Mercury's **real**
`MercuryAnomalyDetector`, its **real** internal streams, the **real**
`score_runtime_equation_profile`, against **real** live-API data. Not a
reimplementation, not synthetic theatre.

**Data reality.** Tabular ODDS/ADBench (`load_dataset`) is blocked by the
sandbox SSRF allowlist and silently falls back to a trivially-separable
synthetic approximation (ensemble AUROC = 1.0) — **discarded as useless**. The
live-domain feeds (USGS/NOAA/etc.) *are* reachable, so all numbers below are on
real events.

**Scope.** 15 real events / 5 domains: earthquake (USGS ×5), hurricane
(NHC/IBTrACS ×3), marine (NOAA ×2), tornado (SPC ×2), tsunami (NOAA ×3).
Wildfire/flood/volcanic/landslide were unreachable in-sandbox (API key / blocked
redirect / host down) and excluded. Per-event positives are tiny (1–2 anomalies),
so single-event AUROC is noisy — but the patterns below are **consistent across
all 15 events**.

**Reproduce.** `harness_multidomain.py` with the real engine importable
(AMA built, `MERCURY_ALLOW_SYNTHETIC=0`). Raw per-event metrics: `real_results.json`.

---

## The four measured findings

| # | Question | Metric | Result |
|---|---|---|---|
| 1 | Are Mercury's streams independent? | mean \|corr\| | **0.66** — substantially redundant |
| 2 | Does fusion beat the best single stream? | ensemble − best-single AUC | **−0.074** (worse in **13/15**, worst **−0.33**) — **best-single is an ORACLE**, see note |
| 3 | Does the outer equation beat the raw ensemble? | eq − ensemble AUC | **−0.002** — inert |
| 4 | Does η^Φ ever change a verdict? | rank-flip / abs-flip / ΔAUC | **0.003 / 0.003 / −0.0002** — inert |

Means: ensemble AUC **0.836**, best-single **0.909**. And from the earthquake
benchmark (`benchmarks/earthquake_honest_benchmark.py`): mean **AUC 0.937 / F1 ≈ 0.09** — ranking is good, thresholding
is broken.

> ### The 0.909 "best single stream" is an oracle. Read every −0.074 below with this.
>
> `harness_multidomain.py:78-81` computes each stream's AUROC **against the
> evaluation labels** and then takes `max()` over them. The winning stream is
> chosen by the labels, so 0.909 is an **upper bound no label-free policy can
> reach**, not an achievable operating point. There is also no train/test
> split in that harness at all — `det.fit(feats)` then `det.detect(feats)` on
> the same rows (`:63-64`) — so the selection and the evaluation use the same
> labels. The ensemble's 0.836 consumes **zero** labels. The two numbers are
> therefore not comparable as "fusion vs a stream you could have picked".
>
> **What is actually reachable without labels.** The argmax stream is different
> on almost every event (6 distinct streams win across 15 events:
> `info_geometry` ×4, `z_score_continuous` ×3, `resonance` ×3,
> `isolation_forest` ×3, `kinematic` ×1, `iqr` ×1), so no fixed a-priori choice
> reproduces 0.909. Fixing the best stream in advance gives:
>
> | fixed stream (label-free) | mean AUROC |
> |---|--:|
> | `info_geometry` | **0.8403** |
> | `isolation_forest` | 0.8357 |
> | *(the shipped ensemble)* | *0.8355* |
> | `z_score_continuous` | 0.8293 |
> | `resonance` | 0.8172 |
> | `iqr` | 0.7720 |
> | `kinematic` | 0.6655 |
>
> The best label-free fixed choice beats the ensemble by **+0.005**, not
> +0.074 — and three of the six streams are *worse* than the ensemble. **About
> 93% of the advertised gap is the oracle selection itself.** The finding that
> survives is the weaker, true one: naive averaging of streams at |corr| 0.66
> gives up a little (~0.005) against the single best stream, and nothing like
> 0.074.

---

## What it means (the reframe)

1. **"Add more streams" is refuted on real data.** Mercury's own multi-stream
   ensemble (0.836) is *worse* than its own best single stream (0.909). Naive
   fusion of correlated streams (|corr| 0.66) **dilutes**, it doesn't combine.
   Any plan that leans on "more streams → better" is wrong as stated.

2. **CORRECTED (2026-11-04). There is no +0.074 on the table.** This item read:
   *"There is an accuracy win on the table at zero added cost: make the fusion
   reliability-weighted/selective so it reaches best-single. That is +0.074 AUC
   (0.836 → ~0.909) with no new streams and no new compute."* That target is an
   **oracle** (see the note above): reaching 0.909 requires knowing, per event,
   which stream will win — which requires the labels. The reachable, label-free
   headroom is **+0.005** (`info_geometry` fixed in advance, 0.8403 vs the
   ensemble's 0.8355), and §"Can reliability-weighting close the gap without
   labels?" below answers **no** for these streams on the sibling ADBench gap
   as well. The original wording is kept here rather than deleted because it
   was propagated into `README.md`, `PROMPT.md` and
   `research/engine_calibration/PHASE2_PROMPT.md`, where a **Kill criterion**
   was built on it; those are corrected too.

3. **The equation shape is not an accuracy lever.** Profiles move AUC by −0.002.
   Don't tune equation shape for accuracy; the levers are upstream (which
   streams, weighted how) and downstream (calibration).

4. **Calibration is the operational bottleneck.** AUC 0.94 with F1 ~0.09 means
   the value is in converting *ranking → decisions*: calibrated thresholds /
   conformal operating points. That is where F1 and precision live.

5. **The η^Φ gate is governance, not detection.** As a global scalar it is
   rank-preserving, so it provably cannot change a ranked verdict (measured: 0.003
   flip, ΔAUC −0.0002). Keep it as the locked ethics invariant, re-attached
   outermost — but it must not be sold as an accuracy feature. If it is to affect
   decisions it has to be **per-action benevolence**, not a global constant.

---

## Consequence for the plan

The omni-equation's first job is **not** breadth (more streams/modalities) — it
is **fixing fusion and calibration on the streams already present**, which is a
measured accuracy/cost win with no value surrendered. Breadth (LLM / predictive /
multimodal streams) comes **after**, each admitted only if it shows positive
conditional lift under reliability-weighting so it cannot dilute. The actionable,
gated, kill-criteria version is in `PROMPT.md`.

---

## Update — fusion frontier on real ADBench tabular (post-hardening)

The detector-hardening pass (PR #302) lifted the committed 18-set transductive
mean **0.7397 → 0.7634**, mostly by fixing the data-type gate (tabular sets were
mislabelled temporal). That re-opened FINDINGS Gate 1 — *can reliability-weighting
close the remaining gap to best-single without labels?* It was measured directly
(`harness_fusion_diagnosis.py`, `fusion_diagnosis_results.json`). **The answer is
no, for these streams.**

* **The dilution gap is real and confirmed on ADBench:** mean fused **0.7634** vs
  per-set best single stream **0.7934** (**+0.030**). It concentrates on a few
  sets where resonance is strong and gets averaged down by info-geometry —
  Cardiotocography (R 0.79 → fused 0.63), wine (0.78 → 0.68), Stamps, cardio,
  thyroid — and it is the mechanism behind both PR-302 "losses" (Waveform,
  WPBC).
* **No label-free signal ranks the streams per-set.** The self-supervised
  weighter estimates separation against a *synthetic contrast* set (3σ Gaussian
  blobs), which every stream separates ~perfectly, so its per-stream AUCs
  saturate at ~1.0 and carry no ranking information. Tested replacements all
  fail: score-distribution statistics (skew / kurtosis / tail mass) predict the
  better of {resonance, info-geometry} on only **10/18** sets (chance 9), and
  **no** contrast (Gaussian σ∈{0.5…3}, feature-shuffle, uniform) does better than
  **6–7/18**. The synthetic anomalies simply do not reflect real-anomaly
  structure, so separation on them does not transfer.
* **Every mean-improving combiner regresses real sets.** A fixed re-tilt toward
  resonance, rank-mean, and rank-max all raise the mean (to ~0.773) but regress
  **7+** sets by 0.06–0.17 (e.g. Ionosphere 0.92 → 0.74). None clears the
  pre-registered keep bar (*net-positive AND no set below −0.002*).
* **A native kNN-distance stream is genuinely complementary — but not free.** It
  beats the per-set best on **7/18** sets, exactly the weak band (glass +0.12,
  Waveform +0.11, Pima +0.04, Hepatitis, WPBC → above-random) and is weak where
  the current streams are strong (Cardiotocography, cardio, wine). It would
  convert both PR-302 losses — *but* added at any fixed weight it still regresses
  5–7 sets, because choosing when to trust it is the same unsolved per-set
  problem.

**Consequence (refines Gate 1).** Reliability-weighted fusion is the right idea,
but on these three streams it is **not realisable label-free** — the hardened
0.7634 sits on the unsupervised, zero-regression frontier. Closing the +0.030 gap
requires either (a) **ground-truth / calibration labels**, which the detector
already exploits via `fit(calibration_labels=…)` → `_compute_adaptive_weights`
(the preferred path whenever labels exist), or (b) a **per-set reliability signal
that the four tested candidate families do not provide**. The kNN stream is worth
keeping in reserve: it earns its place the moment a working per-set weighter
exists, not before. Reproduce: `python research/omni_equation/harness_fusion_diagnosis.py`.
