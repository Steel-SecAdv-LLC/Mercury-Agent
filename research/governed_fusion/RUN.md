# Reproducing the Governed Fusion Substrate measurements (PR #278)

Every figure in `FINDINGS.md` (and the PR body) is reproduced **in this branch
from committed code** on the **real reachable suite** (29 events / 9 domains, no
synthetic). This file is the exact recipe; the per-event results JSON in
`results/` and the data manifest `manifest.json` let a reviewer match every
figure **without live API access**.

## 0. Build the AMA/PQC native backend (hard import gate)

`omni_mercury_engine` fails closed if the native PQC backend is absent and
`MERCURY_ALLOW_SYNTHETIC=0`. Build it exactly as CI does
(`.github/actions/build-ama-cryptography`, `ama-ref v3.2.0`):

```bash
git clone --branch v3.2.0 --depth 1 \
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
python research/governed_fusion/build_manifest.py              # -> manifest.json
```

## 3. What each artifact backs

| artifact | FINDINGS section |
|---|---|
| `results/baseline_results.json` | Baseline (per-event macro mean + pooled) |
| `results/conformal_results.json` | Item 4 — adaptive / **youden_f1 (displaced)** / conformal |
| `results/reliability_fusion_results.json` | Item 3 — KILL CONFIRMED |
| `results/survivability_results.json` | Item 2 — floor curve + cubic-moment escape |
| `results/calibration_results.json` | Stage 2 — Beta-MCA vs isotonic vs identity + Venn-Abers |
| `manifest.json` | per-event `n_rows`, `n_pos`, SHA-256 of `(X,y)` (full + capped) |

## 4. Notes on the suite

- `pandemic/ebola_2014` falls back to synthetic despite the flag and is
  **excluded** (stated, never averaged in) → 29 real events.
- Unreachable domains (wildfire, flood, volcanic, landslide, financial, sepsis)
  are never synthesised and never enter the suite.
- `network_security/cicids_2017` is unavailable (all download sources 404 /
  refuse redirects); the domain uses `nsl_kdd` + `batadal`. The seeded stratified
  cap (`cap=6000`, `seed=42`) applies only to `nsl_kdd`, `mpox_2022`, `batadal`,
  and `fema/hurricane_2024`; every other event is used in full.
- Metrics come from `ml/mercury_ml` only — **no scikit-learn** in any reachable
  path.
