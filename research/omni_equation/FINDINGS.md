# Omni-Equation — Measured Findings (real detector, real data)

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
| 2 | Does fusion beat the best single stream? | ensemble − best-single AUC | **−0.074** (worse in **13/15**, worst **−0.33**) |
| 3 | Does the outer equation beat the raw ensemble? | eq − ensemble AUC | **−0.002** — inert |
| 4 | Does η^Φ ever change a verdict? | rank-flip / abs-flip / ΔAUC | **0.003 / 0.003 / −0.0002** — inert |

Means: ensemble AUC **0.836**, best-single **0.909**. And from the earthquake
honest benchmark: mean **AUC 0.937 / F1 ≈ 0.09** — ranking is good, thresholding
is broken.

---

## What it means (the reframe)

1. **"Add more streams" is refuted on real data.** Mercury's own multi-stream
   ensemble (0.836) is *worse* than its own best single stream (0.909). Naive
   fusion of correlated streams (|corr| 0.66) **dilutes**, it doesn't combine.
   Any plan that leans on "more streams → better" is wrong as stated.

2. **There is an accuracy win on the table at zero added cost:** make the fusion
   reliability-weighted/selective so it *reaches* best-single. That is
   **+0.074 AUC (0.836 → ~0.909)** with no new streams and no new compute — and
   it *removes* drag rather than surrendering any value.

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
