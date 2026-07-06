# Mercury Anomaly Engine — Phase 2 Build Prompt (reliability-weighted fusion + labels)

**Read this whole file first. Everything below is measured on the REAL engine
against REAL live-API data — reproduce it before trusting it. Ground every claim
in a measured before/after on the full reachable suite; revert anything that
does not show a clean full-suite gain.**

---

## 0. Where Phase 1 left off (proof: branch `feature/engine-calibration`)

Phase 1 fixed the *operating point* (calibration) in place and hit the ceiling of
the levers it was allowed. The remaining gap is *ranking* (AUROC), which is capped
by the **fixed ensemble weights** — the one thing Phase 1 was forbidden to touch.
Phase 2 is allowed to re-weight and to use labels. **Ethics stays — see §2.**

Measured baseline vs Phase-1 calibration (30 real events / 9 reachable domains,
controlled: identical data + scores, only the decision cut moves):

| metric | baseline (fixed 0.5) | after calibration |
|---|---|---|
| F1 | 0.220 | **0.317** |
| precision | 0.330 | 0.322 |
| recall | 0.349 | **0.576** |
| AUROC | 0.850 | 0.850 (unchanged — rank-preserving) |
| AUPRC | 0.447 | 0.447 (unchanged) |

Per-domain F1 (before → after): earthquake 0.048→0.079, energy 0.226→0.243,
fema 0.236→0.362, hurricane 0.303→0.269, marine 0.087→0.278,
network_security 0.100→0.259, pandemic 0.406→0.425, tornado 0.455→0.493,
tsunami 0.016→0.542. (8/9 improve; hurricane −0.034, 4 small noisy events.)

Phase-1 change (committed): `MercuryAnomalyDetector.detect()` now derives an
unsupervised operating point from the score distribution when no
conformal/supervised/explicit threshold is set — Otsu split when a distinct
high-score mode exists (histogram valley ≥ 0.55), else a robust MAD tail
(`median + 2·1.4826·MAD`). Rank-preserving. Reproduces the live benchmark
metrics exactly. See `src/omni_mercury_engine/detectors/statistical.py`
(`_adaptive_operating_point`, `_otsu_threshold`, `_score_valley_depth`,
`_robust_tail_threshold`) and `research/engine_calibration/`.

### Why ranking is stuck (the diagnosis that drives Phase 2)
Per-component AUROC (live suite): resonance **0.824**, kinematic **0.662**
(weak), info_geometry **0.838**. The fixed-weight ensemble = `0.40·resonance +
0.30·kinematic + 0.30·info_geometry` scores **0.832** AUROC — *below its own best
single component (0.908)*. The kinematic stream dilutes the average. This matches
`FINDINGS.md` on branch `claude/serene-gates-fbE7W`: streams are redundant
(|corr| 0.66), fusion dilutes (−0.074 vs best-single), and **reliability-weighted
fusion is the measured win (+0.074 AUROC, 0.836→~0.91).**

### In-constraint ranking levers already tested and REVERTED (do not repeat without a new idea)
- Sharper kinematic finite-difference mapping (no sliding-max spread): **regresses** (the spread helps multi-sample events).
- Robust baseline statistics (median/MAD, winsorized covariance) in `fit()`: **−0.029 overall** (helps high-contamination tsunami/tornado but tanks low-contamination earthquake −0.197, network_security −0.106; standard stats are fine when contamination is tiny).
- Feature aggregation in resonance (max / top-2 / mean+max over features): **+0.006 best**, with per-domain regressions. Marginal.

Conclusion: component-scoring tweaks don't move ranking. The lever is **fusion
weighting** and/or **labels**.

---

## 1. Objective

Lift **AUROC and F1 materially** on the full reachable suite, every gain proven
by a before/after measurement. Order of attack:

1. **Reliability-weighted fusion (primary).** Replace the fixed 0.40/0.30/0.30
   average with a per-stream reliability weighting that down-weights
   low-variance / redundant / uninformative streams. Target: close the
   0.832 → ~0.91 ensemble-vs-best-single gap. This lifts AUROC, and F1 follows.
2. **Labels — supervised / semi-supervised calibration (secondary).** The
   unsupervised operating point tops out near F1 0.32–0.38 (ORACLE per-event
   threshold ceiling is **0.532**; flagging the true count ("top-k") gives
   ~0.377). Use a labeled calibration subset / conformal prediction to push the
   operating point toward the oracle. The detector already has the hooks —
   `fit_with_labels`, `fit_with_calibration_subset`, a Mondrian conformal path,
   `_supervised_threshold` — wire and measure them.
3. **Replace the kinematic component (fallback, only if 1–2 underdeliver).**
   Swap the AUROC-0.662 finite-difference kinematic score for a stronger temporal
   anomaly score, keeping the ensemble structure. Riskier; measure hard.

Also re-confirm the calibration ceiling: squeeze the unsupervised operating point
to its max (~0.33–0.35 F1) and report it as the honest in-constraint floor that
re-weighting/labels must beat. AUROC stays ~0.85 under pure calibration unless a
ranking lever turns that dial.

---

## 2. Hard constraints (Phase 2)

- **Ethics gate (η^Φ / η^p) stays — PERIOD.** Do not remove, weaken, or add a
  config path that disables it. Re-attach it as the outermost multiply on the
  fused score and prove with an adversarial test that no path bypasses it. It is
  governance, not an accuracy knob — measured rank-flip from it is ~0.003 (inert
  for ranking), so it costs nothing to keep outermost.
- **NOW ALLOWED (relaxed from Phase 1):** re-weighting / replacing the *fusion*
  rule, and using *labels* (supervised + semi-supervised + conformal).
- **Strengthen, don't gratuitously replace.** Prefer freeze-and-add: a new
  fusion/calibration path that the existing one can be switched to, with exact
  reduction to the current behavior when streams = 1 or labels are absent.
- **No new heavy dependencies.** numpy / scipy / torch are present; **scikit-learn
  is NOT installed** — use the engine's own `omni_mercury_engine.ml.mercury_ml`
  for metrics and ML primitives (it has roc_auc_score, average_precision_score,
  f1/precision/recall, StandardScaler, GMM, LogisticRegression, KFold, etc.).
- **No synthetic-data claims.** Measure on the real live-API domains; state
  real-vs-unreachable per domain.

---

## 3. Environment (rebuild — the container is ephemeral)

- AMA-Cryptography PQC backend is a HARD import gate (no env bypass). Rebuild it
  once, as in `research/omni_equation/README.md`:
  ```bash
  git clone --depth 1 --branch v3.3.0 \
    https://github.com/Steel-SecAdv-LLC/AMA-Cryptography.git /tmp/ama-cryptography
  cd /tmp/ama-cryptography && cmake -B build -DAMA_USE_NATIVE_PQC=ON && cmake --build build
  AMA_NO_CYTHON=1 pip install --no-build-isolation .
  ```
- Run with:
  ```bash
  export PYTHONPATH=.:src:/tmp/ama-cryptography     # engine lives in src/omni_mercury_engine
  export AMA_CRYPTO_LIB_PATH=/tmp/ama-cryptography/build/lib/libama_cryptography.so
  export LD_LIBRARY_PATH=/tmp/ama-cryptography/build/lib
  export MERCURY_ALLOW_SYNTHETIC=0                  # real data only; unreachable domains skip
  ```
- The detector is deterministic after `fit()` (verified: repeated `detect()` byte-identical).

---

## 4. Step-by-step (each step ends with measured before/after or it didn't happen)

### Step A — Investigate FIRST, broadly, before writing code
- **Re-read branch `claude/serene-gates-fbE7W`**: `research/omni_equation/FINDINGS.md`,
  `PROMPT.md`, `harness_multidomain.py`, `harness_tabular.py`, `real_results.json`.
  It is the evidence behind the reliability-weighting thesis.
- **Look at the FINDOYOU repo's `mercury_fusion`** (Stouffer-over-conformal-p-values
  pattern) — `FINDINGS.md`/`PROMPT.md` cite it as the reliability-weighting
  reference. (Use `mcp__claude-code-remote__list_repos` / `add_repo` if FINDOYOU
  isn't in scope.)
- **Map the repo thoroughly** — the engine is large (`src/omni_mercury_engine/`
  has ~48 subpackages). Existing machinery you should READ and likely REUSE rather
  than reinvent:
  - Fusion: `core/adaptive_fusion.py`, `core/fusion.py`, `core/stacking_fusion.py`,
    `core/fibring_fusion.py`, `core/global_omni_scalar_network.py`.
  - Calibration / thresholding: `core/score_calibration.py` (Otsu/MAD/percentile/
    knee/GMM optimizer — already wired, off by default), `core/calibration.py`,
    `core/calibration_pipeline.py`, `core/conformal_prediction.py`,
    `core/adaptive_domain_thresholding.py`, `core/adaptive_detector.py`.
  - Detector internals: `detectors/statistical.py` (`fit_with_labels`,
    `fit_with_calibration_subset`, `_supervised_threshold`, the Mondrian conformal
    path, the existing per-sample `_adaptive_weights`, the pairwise-inversion guard,
    and the ensemble-flip — these are precedents for legitimate dynamic weighting).
  - Ethics: `core/ethical_governor.py`, `core/ai_ethics.py`,
    `core/benevolence_optimization.py`, `core/equation_profiles.py` (the η^Φ gate /
    Omni-Ava equation — keep intact, re-attach outermost).
- **Verify whether `mercury_benchmark.py`'s 47 ADBench datasets are REAL or
  synthetic.** Phase 1 saw varied, non-trivial AUCs (ADBench-26..33: 0.71–0.99),
  which do NOT look like the trivially-separable synthetic fallback FINDINGS warned
  about. If they are real, that is a 47-dataset *ranking* benchmark to optimize
  and report (it uses an oracle threshold sweep, so it measures ranking, not the
  operating point). Confirm via `datasets/adbench.py` and the loader's provenance.

### Step B — Re-baseline (proof harness)
- Rebuild the per-event cache: `python research/engine_calibration/build_cache.py`
  (fetches real `(X, y)` per event to `/home/user/eqlab/cache/*.npz`; the cache is
  ephemeral, the script is committed). Expect **9 reachable domains / 30 events**:
  earthquake(5), tsunami(3), tornado(2), marine(2), hurricane(4), energy(3),
  fema(3), network_security(2), pandemic(6).
  **Unreachable (exclude, state why):** wildfire (NASA_FIRMS_MAP_KEY not set),
  flood (SSRF refuses USGS 301 redirect), volcanic (host down), landslide (host
  down), financial (unreachable), sepsis (unreachable).
- Cache detector scores once for fast iteration (Phase 1 used a `scores.pkl`;
  rebuild similarly). NOTE: `network_security/nsl_kdd` is **148,517 rows** — a full
  `detect()` is ~10+ min; **subsample (e.g. 20k, seeded, stratified) for
  iteration**, run full only for the final official number.
- Confirm Phase-1 numbers reproduce: `python research/engine_calibration/measure_cached.py`
  → `results.json`. Confirm against the live scripts (`benchmarks/*_honest_benchmark.py`,
  run as `python -m benchmarks.<domain>_honest_benchmark`; note: the *files* are
  named `*_honest_benchmark.py` — refer to them as the **live-domain benchmarks**).

### Step C — Reliability-weighted fusion (PRIMARY)
- Per-sample/per-stream reliability weight = f(calibrated reliability): e.g.
  variance / dispersion of the stream's scores, agreement/redundancy with the
  others (down-weight |corr|≈high streams), label coverage if labels are present.
  Reduce to the 0.40/0.30/0.30 average when reliabilities are equal / streams = 1.
  Reuse `core/adaptive_fusion.py` / `core/stacking_fusion.py` if they fit; if you
  learn weights, that is the labels lever (Step D) and is allowed.
- **Target:** fused AUROC ≥ best-single across the reachable domains (close the
  −0.076 gap; 0.832 → ≥0.90). Then re-run calibration on top and report F1.
- **Kill criterion:** if fused AUROC cannot reach best-single, the fusion thesis is
  dead for this engine — report the numbers and move to Step E.

### Step D — Labels: supervised / semi-supervised calibration (SECONDARY)
- Wire a labeled-subset path: `fit_with_calibration_subset` / `fit_with_labels` /
  conformal (`core/conformal_prediction.py`). Use a held-out labeled fraction to
  set the operating point and/or learn fusion weights (stacking). Honestly
  separate calibration data from evaluation data — no peeking.
- **Target:** F1 at a fixed, stated FPR materially above the unsupervised ceiling
  (~0.33–0.38), moving toward the oracle 0.532, with AUROC ≥ Step C.
- Report the supervision budget (how many labels) vs the F1 gained.

### Step E — Replace the kinematic component (FALLBACK)
- Only if C+D underdeliver. Swap the finite-difference kinematic (AUROC 0.662) for
  a stronger temporal score (e.g. a residual/forecast-error or spectral-residual
  detector already in the repo — search `detectors/` and `core/`), keeping the
  ensemble structure and weights, and measure that the ensemble AUROC rises with
  no per-domain collapse.

### Step F — Ethics proof + final report
- Adversarial test: no config path disables η^Φ; it is the outermost multiply.
- Full-suite before/after table (AUROC, AUPRC, F1, precision, recall; per-domain +
  overall), plain numbers. Update `research/engine_calibration/` and `FINDINGS.md`.

---

## 5. Measurement discipline (non-negotiable)
- Iterate on the cached scores/data (controlled: same data, only the change moves
  the metric); **confirm every kept change on the live scripts**.
- Report AUROC, AUPRC, F1, precision, recall — per-domain AND overall — for every
  change, before vs after. **Revert anything without a clean full-suite gain.**
  Call out per-domain regressions explicitly.
- Metrics via `mercury_ml` (no sklearn). The live-domain base computes AUC from
  `detection["scores"]` and F1 from `detection["is_anomaly"]`; `mercury_benchmark.py`
  uses an oracle threshold sweep (so it measures ranking, not the operating point).
- Reference ceilings on the current scores: ORACLE per-event F1 = **0.532**,
  top-k (true count) ≈ **0.377**, best-single AUROC = **0.908**.

## 6. Mandate (empower yourself)
Research, explore, investigate, model/rehearse, synthesize, and TEST on the live
datasets and the original engine code. Read widely in the repo before coding;
prefer wiring existing, tested machinery over reinventing. Prototype with
monkeypatches to measure a lever's potential before committing to an
implementation. Branch: keep working on `feature/engine-calibration` (or a
clearly-named successor); open a draft PR only when the numbers justify it.

## 7. Definition of done
At least one of {Step C AUROC, Step D F1} shows a **measured** full-suite win over
the Phase-1 numbers above, ethics provably intact, exact reduction-to-baseline
preserved, per-domain regressions disclosed. If none beat Phase 1, the honest
output is "re-weighting/labels did not beat calibration on this engine," with the
numbers.
