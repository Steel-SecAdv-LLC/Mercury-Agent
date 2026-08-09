# Reproducing the Governed Fusion Substrate measurements (PR #278)

Last updated: 2026-07-11.

Every figure in `FINDINGS.md` (and the PR body) is reproduced **in this branch
from committed code** on the **real reachable suite**: a **live headline suite of
23 real events / 7 domains** plus a separately-reported **reconstructed-from-live
group of 7 events / 3 domains** (tsunami, energy, `ebola_2014`), always labelled
reconstruction and never folded into the headline mean. This file is the exact
recipe; the per-event results JSON in `results/` and the data manifest
`manifest.json` let a reviewer match every figure **without live API access**.

## 0. Build the AMA/PQC native backend (hard import gate)

`omni_mercury_engine` fails closed if the native PQC backend is absent and
`MERCURY_ALLOW_SYNTHETIC=0`. Build it exactly as CI does
(`.github/actions/build-ama-cryptography`, `ama-ref v3.3.0`):

```bash
git clone --branch v3.3.0 --depth 1 \
  https://github.com/Steel-SecAdv-LLC/AMA-Cryptography.git /tmp/ama-cryptography
cd /tmp/ama-cryptography
CC=gcc-12 CXX=g++-12 cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release -DAMA_USE_NATIVE_PQC=ON \
  -DAMA_BUILD_SHARED=ON -DAMA_BUILD_STATIC=ON \
  -DAMA_BUILD_TESTS=OFF -DAMA_BUILD_EXAMPLES=OFF
cmake --build build -j "$(nproc)"
AMA_NO_CYTHON=1 pip install --no-build-isolation .
```

## 1. Environment

```bash
cd <repo root>
source research/governed_fusion/gf_env.sh
```

`gf_env.sh` exports `PYTHONPATH=.:src:/tmp/ama-cryptography`,
`AMA_CRYPTO_LIB_PATH`, `LD_LIBRARY_PATH`, `MERCURY_ALLOW_SYNTHETIC=0`,
`GF_CACHE_DIR` (heavy `(X,y)`+score `.npz` cache, not committed) and
`GF_RESULTS_DIR` (`research/governed_fusion/results/`, committed JSON). Override
`AMA_HOME` if you built the backend elsewhere.

## 2. Run the measurements

The first run hits the live loaders (USGS/NOAA/FEMA/CISA/WHO) to build the
`(X,y)` cache, then fits `MercuryAnomalyDetector` once per event and caches its
scores. Subsequent runs read the cache (seconds, no network).

```bash
python research/governed_fusion/measure_baseline.py            # -> results/baseline_results.json
python research/governed_fusion/measure_conformal.py           # -> results/conformal_results.json
python research/governed_fusion/measure_reliability_fusion.py  # -> results/reliability_fusion_results.json
python research/governed_fusion/measure_survivability.py       # -> results/survivability_results.json
python research/governed_fusion/measure_calibration.py         # -> results/calibration_results.json
python research/governed_fusion/measure_calibration_levers.py  # -> results/calibration_levers_results.json
python research/governed_fusion/measure_decorrelation.py       # -> results/decorrelation_results.json
python research/governed_fusion/build_manifest.py              # -> manifest.json
```

## 2b. Verify you measured the pinned inputs (do this before quoting a number)

```bash
python research/governed_fusion/input_pin.py --check              # whole live suite
python research/governed_fusion/input_pin.py --external-only --check   # just the fitness set
```

`build_manifest.py` has always *recorded* a SHA-256 of every event's `(X, y)`;
until 2026-08-04 nothing *checked* one, which is why the suite's per-event drift
was unattributable. A fresh-environment refit moved `nsl_kdd 0.679 → 0.728` and
`batadal 0.862 → 0.889` (headline **0.770 → 0.809**) and the repository could not
say whether the inputs or the environment had moved — so neither could be ruled
out, and no improvement smaller than that drift was falsifiable.

Read the verdicts as:

| verdict | meaning |
|---|---|
| `OK` | rebuilt `(X, y)` hashes to the pin — a metric change here is **code/environment**, and is a real finding |
| `DRIFT` | the inputs changed — the metric is **not comparable**; re-pin and re-measure, do not publish across it |
| `UNPIN` | event in the suite with no manifest entry — unverifiable |
| `UNREA` | upstream unreachable — an availability fact, **not** evidence of change, so it does not fail `--check` |

`promotion_gate.py` now refuses to promote a candidate whose declared
external-label event set differs from the manifest's, and refuses a manifest
that disagrees with itself (the `provenance_summary` rollup the gate reads
versus the per-event entries the pin reads). A gate that compares numbers from
different data is not a gate.
```

## 3. What each artifact backs

| artifact | FINDINGS section |
|---|---|
| `results/baseline_results.json` | Baseline (per-event macro mean + pooled) |
| `results/conformal_results.json` | Item 4 — adaptive / **youden_f1 (displaced)** / conformal |
| `results/reliability_fusion_results.json` | Item 3 — KILL CONFIRMED |
| `results/survivability_results.json` | Item 2 — floor curve + cubic-moment escape |
| `results/calibration_results.json` | Stage 2 — Beta-MCA vs isotonic vs identity + Venn-Abers |
| `results/calibration_levers_results.json` | Stage 2 lever probe — conclusive negative (λ sweep, warm-start) |
| `results/decorrelation_results.json` | Item D — decorrelated-stream fusion (SHIP rejected; redundancy diagnosis + temporal/kNN streams + paired bootstrap) |
| `manifest.json` | per-event `n_rows`, `n_pos`, SHA-256 of `(X,y)` (full + capped) |

## 4. Notes on the suite

- `pandemic/ebola_2014` has no live WHO GHO feed (it 404s), so the loader
  reconstructs the documented 2014 epidemic curve. It is reported in the
  **reconstructed-from-live group** (tsunami / energy / `ebola_2014`, 7 events /
  3 domains) and **never folded into the 23-event live headline mean** — it is
  labelled, not excluded. `MERCURY_ALLOW_SYNTHETIC=0` does not gate the
  `loaders/` path (only `datasets/`), so the live/reconstructed split is made
  explicit in code (`suite.py: RECONSTRUCTED_DOMAINS / RECONSTRUCTED_EVENTS`).
- Unreachable domains (wildfire, flood, volcanic, landslide, financial, sepsis)
  never enter the suite; their loaders fetch live or raise
  `DataSourceUnavailableError` — none synthesise.
- `network_security/cicids_2017` is unavailable (all download sources 404 /
  refuse redirects); the domain uses `nsl_kdd` + `batadal`. The seeded stratified
  cap (`cap=6000`, `seed=42`) applies only to `nsl_kdd`, `mpox_2022`, `batadal`,
  and `fema/hurricane_2024`; every other event is used in full.
- Metrics come from `ml/mercury_ml` only — **no scikit-learn** in any reachable
  path.
