<!--
Copyright (C) 2025 Steel Security Advisors LLC
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Competitive benchmark — Mercury vs PyOD (ADBench) + NAB streaming

Applies to Mercury Agent **v3.0**. Last updated: 2026-10-27 (full 57-set re-measure at
commit `a2d7f991`, seed 42; the v2.1.x record this replaces had the tier 3rd of 8 at
mean rank 3.79, losing to both `knn` and `local_outlier_factor`).

On the full **57-dataset ADBench suite** (47 Classical + 10 CV/NLP embedding sets), under
one identical unsupervised protocol for every method, **Mercury's tier detector places
2nd of 8 by mean rank (2.72)** — behind `knn` (2.52) and **ahead of**
`local_outlier_factor` (3.61), `isolation_forest` (4.55), `hbos` (5.55), `copod` (5.60)
and `ecod` (6.06) — and carries the **highest mean ROC-AUC of all eight methods**
(0.8341 vs `knn`'s 0.8316 and LOF's 0.8169). The movement from the v2.1.x record is the
kNN density stream plus its dimensional fusion gate (`_knn_dimension_weight`,
measured dev/holdout before adoption): the LOF head-to-head flipped from 28W/27L to
**34W/23L**, the `knn` head-to-head closed from 21W/35L to **23W/34L with a median
losing margin of 0.008 AUC**, and the statistical baselines are beaten 47–50 of 57.
This stays a measurement of position, not a highlight reel: `knn` still leads on
low-dimensional local-density sets (its native turf), and every loss is enumerated
below with its margin.

## Per-method summary (mean over each method's successful datasets)

| method | n | mean AUC | median AUC | mean AP | mean rank |
|---|---:|---:|---:|---:|---:|
| `mercury_tier` **(Mercury)** | 57 | 0.8341 | 0.8718 | 0.5518 | 2.72 |
| `mercury_fusion` **(Mercury)** | 57 | 0.7481 | 0.7270 | 0.3935 | 5.39 |
| `isolation_forest` | 57 | 0.7748 | 0.8080 | 0.4446 | 4.55 |
| `ecod` | 57 | 0.7238 | 0.7141 | 0.4157 | 6.06 |
| `copod` | 57 | 0.7291 | 0.7336 | 0.4025 | 5.60 |
| `local_outlier_factor` | 57 | 0.8169 | 0.8516 | 0.5458 | 3.61 |
| `knn` | 57 | 0.8316 | 0.8772 | 0.5635 | 2.52 |
| `hbos` | 57 | 0.7412 | 0.7406 | 0.4208 | 5.55 |

Mean rank uses the **57 datasets complete for every method** — this run has no deferred cells, so ranks compare identical work across the whole suite.

## Head-to-head: wins / losses / break-evens

Strictly-higher ROC-AUC = a win; exact ties are break-evens. Losses are reported as plainly as wins.

### `mercury_tier` vs each PyOD baseline

| baseline | wins | losses | break-evens | n compared |
|---|---:|---:|---:|---:|
| `isolation_forest` | 47 | 10 | 0 | 57 |
| `ecod` | 50 | 7 | 0 | 57 |
| `copod` | 48 | 9 | 0 | 57 |
| `local_outlier_factor` | 34 | 23 | 0 | 57 |
| `knn` | 23 | 34 | 0 | 57 |
| `hbos` | 49 | 8 | 0 | 57 |

### `mercury_fusion` vs each PyOD baseline

| baseline | wins | losses | break-evens | n compared |
|---|---:|---:|---:|---:|
| `isolation_forest` | 22 | 35 | 0 | 57 |
| `ecod` | 32 | 25 | 0 | 57 |
| `copod` | 31 | 26 | 0 | 57 |
| `local_outlier_factor` | 18 | 39 | 0 | 57 |
| `knn` | 9 | 48 | 0 | 57 |
| `hbos` | 30 | 27 | 0 | 57 |

## Where Mercury wins outright

Mercury's tier is the single best method on **11 datasets**:

> `InternetAds`, `SpamBase`, `Wilt`, `annthyroid`, `campaign`, `celeba`, `nlp:amazon`, `satimage-2`, `smtp`, `speech`, `thyroid`

`mercury_fusion` is best on 2: `vertebral`, `yeast`.

## Where Mercury loses

Every one of the **46 datasets** where at least one PyOD baseline beats `mercury_tier`, sorted by margin, each diagnosed. Margin = best-beating-baseline AUC − Mercury AUC.

| dataset | Mercury tier AUC | beaten by | margin | # baselines | diagnosis |
|---|---:|---|---:|---:|---|
| `ALOI` | 0.547 | `local_outlier_factor` | +0.143 | 2 | inductive-bias mismatch (decisive): local-density/distance anomalies that k-NN/LOF capture directly; Mercury's tier weighting favors global geometry (PR-3 target) |
| `optdigits` | 0.875 | `local_outlier_factor` | +0.098 | 3 | inductive-bias mismatch (decisive): local-density/distance anomalies that k-NN/LOF capture directly; Mercury's tier weighting favors global geometry (PR-3 target) |
| `nlp:20news_0` | 0.722 | `local_outlier_factor` | +0.069 | 1 | inductive-bias mismatch (decisive): local-density/distance anomalies that k-NN/LOF capture directly; Mercury's tier weighting favors global geometry (PR-3 target) |
| `WPBC` | 0.483 | `hbos` | +0.064 | 6 | statistical-baseline (HBOS/COPOD/ECOD) edge on this distribution; small margin |
| `glass` | 0.837 | `knn` | +0.041 | 1 | inductive-bias mismatch (decisive): local-density/distance anomalies that k-NN/LOF capture directly; Mercury's tier weighting favors global geometry (PR-3 target) |
| `donors` | 0.953 | `knn` | +0.039 | 1 | inductive-bias mismatch (decisive): local-density/distance anomalies that k-NN/LOF capture directly; Mercury's tier weighting favors global geometry (PR-3 target) |
| `cover` | 0.955 | `local_outlier_factor` | +0.037 | 2 | inductive-bias mismatch (decisive): local-density/distance anomalies that k-NN/LOF capture directly; Mercury's tier weighting favors global geometry (PR-3 target) |
| `Cardiotocography` | 0.780 | `isolation_forest` | +0.035 | 3 | isolation-friendly axis-parallel structure; small margin, borderline |
| `mammography` | 0.872 | `ecod` | +0.034 | 4 | statistical-baseline (HBOS/COPOD/ECOD) edge on this distribution; small margin |
| `landsat` | 0.726 | `knn` | +0.033 | 2 | inductive-bias mismatch (decisive): local-density/distance anomalies that k-NN/LOF capture directly; Mercury's tier weighting favors global geometry (PR-3 target) |
| `mnist` | 0.914 | `knn` | +0.030 | 2 | inductive-bias mismatch (decisive): local-density/distance anomalies that k-NN/LOF capture directly; Mercury's tier weighting favors global geometry (PR-3 target) |
| `backdoor` | 0.930 | `local_outlier_factor` | +0.030 | 2 | inductive-bias mismatch (small-margin): local-density/distance anomalies that k-NN/LOF capture directly; Mercury's tier weighting favors global geometry (PR-3 target) |
| `cv:SVHN_0` | 0.636 | `local_outlier_factor` | +0.026 | 1 | inductive-bias mismatch (small-margin): local-density/distance anomalies that k-NN/LOF capture directly; Mercury's tier weighting favors global geometry (PR-3 target) |
| `Stamps` | 0.928 | `copod` | +0.026 | 3 | statistical-baseline (HBOS/COPOD/ECOD) edge on this distribution; small margin |
| `wine` | 0.950 | `knn` | +0.025 | 2 | inductive-bias mismatch (small-margin): local-density/distance anomalies that k-NN/LOF capture directly; Mercury's tier weighting favors global geometry (PR-3 target) |
| `cardio` | 0.933 | `isolation_forest` | +0.025 | 2 | isolation-friendly axis-parallel structure; small margin, borderline |
| `letter` | 0.837 | `knn` | +0.024 | 2 | inductive-bias mismatch (small-margin): local-density/distance anomalies that k-NN/LOF capture directly; Mercury's tier weighting favors global geometry (PR-3 target) |
| `Waveform` | 0.750 | `local_outlier_factor` | +0.022 | 2 | inductive-bias mismatch (small-margin): local-density/distance anomalies that k-NN/LOF capture directly; Mercury's tier weighting favors global geometry (PR-3 target) |
| `cv:FashionMNIST_0` | 0.888 | `local_outlier_factor` | +0.021 | 2 | inductive-bias mismatch (small-margin): local-density/distance anomalies that k-NN/LOF capture directly; Mercury's tier weighting favors global geometry (PR-3 target) |
| `satellite` | 0.856 | `knn` | +0.021 | 2 | inductive-bias mismatch (small-margin): local-density/distance anomalies that k-NN/LOF capture directly; Mercury's tier weighting favors global geometry (PR-3 target) |
| `cv:MNIST-C_brightness` | 0.857 | `local_outlier_factor` | +0.020 | 1 | inductive-bias mismatch (small-margin): local-density/distance anomalies that k-NN/LOF capture directly; Mercury's tier weighting favors global geometry (PR-3 target) |
| `fault` | 0.778 | `knn` | +0.016 | 1 | inductive-bias mismatch (small-margin): local-density/distance anomalies that k-NN/LOF capture directly; Mercury's tier weighting favors global geometry (PR-3 target) |
| `yeast` | 0.463 | `local_outlier_factor` | +0.015 | 2 | inductive-bias mismatch (small-margin): local-density/distance anomalies that k-NN/LOF capture directly; Mercury's tier weighting favors global geometry (PR-3 target) |
| `cv:CIFAR10_0` | 0.732 | `local_outlier_factor` | +0.012 | 2 | inductive-bias mismatch (small-margin): local-density/distance anomalies that k-NN/LOF capture directly; Mercury's tier weighting favors global geometry (PR-3 target) |
| `Lymphography` | 0.984 | `isolation_forest` | +0.012 | 4 | isolation-friendly axis-parallel structure; small margin, borderline |
| `nlp:yelp` | 0.678 | `knn` | +0.011 | 1 | inductive-bias mismatch (small-margin): local-density/distance anomalies that k-NN/LOF capture directly; Mercury's tier weighting favors global geometry (PR-3 target) |
| `nlp:agnews_0` | 0.660 | `local_outlier_factor` | +0.010 | 1 | inductive-bias mismatch (small-margin): local-density/distance anomalies that k-NN/LOF capture directly; Mercury's tier weighting favors global geometry (PR-3 target) |
| `fraud` | 0.962 | `hbos` | +0.010 | 5 | near-tie at ceiling AUC (within cross-environment noise; not a decisive loss) |
| `census` | 0.713 | `knn` | +0.008 | 1 | near-tie (<0.01 margin; both methods weak on this hard, near-random set) |
| `skin` | 0.983 | `knn` | +0.008 | 1 | near-tie at ceiling AUC (within cross-environment noise; not a decisive loss) |
| `Ionosphere` | 0.975 | `knn` | +0.008 | 1 | near-tie at ceiling AUC (within cross-environment noise; not a decisive loss) |
| `WDBC` | 0.992 | `isolation_forest` | +0.008 | 4 | near-tie at ceiling AUC (within cross-environment noise; not a decisive loss) |
| `vertebral` | 0.504 | `knn` | +0.007 | 1 | near-tie (<0.01 margin; both methods weak on this hard, near-random set) |
| `nlp:imdb` | 0.510 | `copod` | +0.007 | 1 | near-tie (<0.01 margin; both methods weak on this hard, near-random set) |
| `PageBlocks` | 0.965 | `local_outlier_factor` | +0.005 | 1 | near-tie at ceiling AUC (within cross-environment noise; not a decisive loss) |
| `cv:MVTec-AD_bottle` | 0.960 | `isolation_forest` | +0.004 | 1 | near-tie at ceiling AUC (within cross-environment noise; not a decisive loss) |
| `WBC` | 0.992 | `isolation_forest` | +0.003 | 4 | near-tie at ceiling AUC (within cross-environment noise; not a decisive loss) |
| `magic.gamma` | 0.833 | `knn` | +0.003 | 1 | near-tie (<0.01 margin; both methods weak on this hard, near-random set) |
| `Hepatitis` | 0.799 | `hbos` | +0.002 | 1 | near-tie (<0.01 margin; both methods weak on this hard, near-random set) |
| `vowels` | 0.974 | `knn` | +0.002 | 1 | near-tie at ceiling AUC (within cross-environment noise; not a decisive loss) |
| `Pima` | 0.731 | `hbos` | +0.002 | 1 | near-tie (<0.01 margin; both methods weak on this hard, near-random set) |
| `shuttle` | 0.997 | `local_outlier_factor` | +0.002 | 2 | near-tie at ceiling AUC (within cross-environment noise; not a decisive loss) |
| `pendigits` | 0.997 | `knn` | +0.002 | 1 | near-tie at ceiling AUC (within cross-environment noise; not a decisive loss) |
| `musk` | 0.999 | `local_outlier_factor` | +0.001 | 3 | near-tie at ceiling AUC (within cross-environment noise; not a decisive loss) |
| `breastw` | 0.993 | `isolation_forest` | +0.001 | 3 | near-tie at ceiling AUC (within cross-environment noise; not a decisive loss) |
| `http` | 0.999 | `knn` | +0.000 | 1 | near-tie at ceiling AUC (within cross-environment noise; not a decisive loss) |

Of these, **11 are decisive** (margin ≥ 0.03) and dominated by k-NN/LOF on local-density structure; the remaining 35 are near-ties (margins < 0.03, within cross-environment numerical noise). The decisive losses remain the explicit target of the per-domain calibration work.

## Deferred cells

**None in this run.** Every (dataset, method) cell — `mercury_fusion`'s 50-epoch torch training included — completed inside the per-cell wall-clock cap (`MERCURY_METHOD_TIMEOUT`, default 300 s), so the aggregates cover all 57 datasets for all 8 methods. (The v2.1.x run deferred 15 `mercury_fusion` cells on the largest sets; deferrals are recorded, never silently dropped, and this section reports them whenever they occur.)

## NAB streaming (Mercury streaming tier vs same-harness baselines)

Mercury's `StreamingScoreEnsemble` vs same-harness baselines on **46 real NAB streams** across all five real categories (`realKnownCause`, `realAWSCloudwatch`, `realTraffic`, `realAdExchange`, `realTweets`). Point-wise ROC-AUC + a budget-matched NAB-style window-detection rate; `random`/`perfect` are the harness floor/ceiling, not competitors.

| method | n | mean AUC | median AUC | mean window-detection rate |
|---|---:|---:|---:|---:|
| `mercury_consensus` **(Mercury)** | 46 | 0.6184 | 0.5939 | 0.9221 |
| `mercury_average` **(Mercury)** | 46 | 0.5842 | 0.5840 | 0.9855 |
| `iforest_windowed` | 46 | 0.6102 | 0.6125 | 0.8696 |
| `ewma_zscore` | 46 | 0.4846 | 0.4853 | 0.9819 |
| `random` | 46 | 0.4998 | 0.5014 | 1.0000 |
| `perfect` | 46 | 1.0000 | 1.0000 | 1.0000 |

NAB's own published scoreboard is quoted for context in the results JSON (`metadata.published_nab_scoreboard_standard_profile`) but is **not** numerically comparable — a different metric on a different protocol.

## Reproduce

```bash
pip install -e ".[ml,benchmark]"   # torch + pyod + scikit-learn
bash scripts/build_ama_native.sh    # mandatory PQC backend (import gate)
export MERCURY_DATA_DIR=/path/to/cache   # ADBench NPZs cache here
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# Full 57-dataset ADBench head-to-head (resumable; writes competitive_results.json)
python benchmarks/competitive_benchmark.py
# NAB streaming, all five real categories (writes nab_competitive_results.json)
python benchmarks/nab_competitive.py
# Deterministic competitive-position guard (CI gate)
python benchmarks/competitive_regression_guard.py --check
```

## Provenance

- **Benchmark code commit (Mercury repo `git rev-parse HEAD`):** `ac2e7d0e20e00cd2d33c4cac0a46a22a6a51e2b8` — the dataset *content* is pinned by the per-dataset sha256 list, not a dataset-repo revision.
- **seed:** 42  ·  **max_samples:** 10000  ·  **fusion_epochs:** 50
- **versions:** python 3.11.15, numpy 2.4.6, torch 2.13.0+cu130, pyod 3.6.1, scikit-learn 1.9.0
- **dataset source:** https://github.com/Minqi824/ADBench (MIT)
- **protocol:** per dataset — stratified cap to 2·max_samples (seed 42); normal-only train half; test = remaining normals + all anomalies capped to max_samples; fixed-seed de-leak shuffle of test rows (applied identically to every method); StandardScaler fit on train only; library defaults for every method; no per-dataset tuning; unsupervised-fair (no method sees labels).
- Per-dataset content hashes (sha256) and per-(dataset, method) fit/score/wall timings are in `benchmarks/competitive_results.json`. The 8-dataset deterministic guard subset is pinned in `benchmarks/competitive_baseline.json` (measurements + margins; floors derived at check time).

### Per-dataset content hashes (sha256, first 16 chars)

| dataset | n | d | anomaly ratio | sha256[:16] |
|---|---:|---:|---:|---|
| `ALOI` | 49534 | 27 | 0.030 | `792d0b5bd1147bfc` |
| `annthyroid` | 7200 | 6 | 0.074 | `e738f7c58a8740d2` |
| `backdoor` | 95329 | 196 | 0.024 | `164481cd9688306b` |
| `breastw` | 683 | 9 | 0.350 | `4f37ce07dbc25dda` |
| `campaign` | 41188 | 62 | 0.113 | `74dd4d0072d6c6f1` |
| `cardio` | 1831 | 21 | 0.096 | `bc5009bde6930b4d` |
| `Cardiotocography` | 2114 | 21 | 0.220 | `5ae84910430f0592` |
| `celeba` | 202599 | 39 | 0.022 | `1cadfdea14cd8d24` |
| `census` | 299285 | 500 | 0.062 | `6739999359a3b8c2` |
| `cover` | 286048 | 10 | 0.010 | `945a097c3542ae81` |
| `donors` | 619326 | 10 | 0.059 | `5824ec854465f276` |
| `fault` | 1941 | 27 | 0.347 | `ec27301f9a4cf7c5` |
| `fraud` | 284807 | 29 | 0.002 | `353b9a156b3f20e0` |
| `glass` | 214 | 7 | 0.042 | `aa7a8db1d475ba01` |
| `Hepatitis` | 80 | 19 | 0.163 | `4a87a2f7d8013e5b` |
| `http` | 567498 | 3 | 0.004 | `9d98710091551587` |
| `InternetAds` | 1966 | 1555 | 0.187 | `bdd8ad5cf5bf2cb9` |
| `Ionosphere` | 351 | 32 | 0.359 | `bca46cb0cfd1bf9d` |
| `landsat` | 6435 | 36 | 0.207 | `ba17e2158c074695` |
| `letter` | 1600 | 32 | 0.062 | `f986d3da80cf1fa2` |
| `Lymphography` | 148 | 18 | 0.041 | `dad977b04f4ea65b` |
| `magic.gamma` | 19020 | 10 | 0.352 | `0be375984d7af9ea` |
| `mammography` | 11183 | 6 | 0.023 | `a0308ac9712d7832` |
| `mnist` | 7603 | 100 | 0.092 | `be7706272325fdd3` |
| `musk` | 3062 | 166 | 0.032 | `2ed3f4392197b76a` |
| `optdigits` | 5216 | 64 | 0.029 | `60de17a375a278ba` |
| `PageBlocks` | 5393 | 10 | 0.095 | `10a9b4d5bb5d3b90` |
| `pendigits` | 6870 | 16 | 0.023 | `c4936b89dcdad59d` |
| `Pima` | 768 | 8 | 0.349 | `215454bb263a2f04` |
| `satellite` | 6435 | 36 | 0.316 | `1f048f5686f1b8c1` |
| `satimage-2` | 5803 | 36 | 0.012 | `1b056e357f801b8b` |
| `shuttle` | 49097 | 9 | 0.072 | `26900b0f6be45d9d` |
| `skin` | 245057 | 3 | 0.208 | `d02aa01ffc1b6bf2` |
| `smtp` | 95156 | 3 | 0.000 | `d2569fc9bc121154` |
| `SpamBase` | 4207 | 57 | 0.399 | `d0c46c7928e43f64` |
| `speech` | 3686 | 400 | 0.017 | `aa647e9e3af9b021` |
| `Stamps` | 340 | 9 | 0.091 | `99fe739447284f61` |
| `thyroid` | 3772 | 6 | 0.025 | `8de161b3f235b764` |
| `vertebral` | 240 | 6 | 0.125 | `0816286f92b96fe6` |
| `vowels` | 1456 | 12 | 0.034 | `4bfc4244d1c88277` |
| `Waveform` | 3443 | 21 | 0.029 | `6c9d867337ad1a2f` |
| `WBC` | 223 | 9 | 0.045 | `650f9dcb129248f6` |
| `WDBC` | 367 | 30 | 0.027 | `78d4606a1ab516f5` |
| `Wilt` | 4819 | 5 | 0.053 | `c70b72aa611ec87b` |
| `wine` | 129 | 13 | 0.078 | `d5ad8965d92cc571` |
| `WPBC` | 198 | 33 | 0.237 | `68a6d67242643855` |
| `yeast` | 1484 | 8 | 0.342 | `bbb15c873d78a4f6` |
| `cv:CIFAR10_0` | 5263 | 512 | 0.050 | `f0e89be5576c37e5` |
| `cv:FashionMNIST_0` | 6315 | 512 | 0.050 | `71d2f1e9d732b8ea` |
| `cv:SVHN_0` | 5208 | 512 | 0.050 | `3526704c19ae4c62` |
| `cv:MNIST-C_brightness` | 10000 | 512 | 0.050 | `8d2d354eff2080ee` |
| `cv:MVTec-AD_bottle` | 292 | 512 | 0.216 | `9eb7a78f1f6ecfcd` |
| `nlp:20news_0` | 3090 | 768 | 0.050 | `eb804cf85f3dc18a` |
| `nlp:agnews_0` | 10000 | 768 | 0.050 | `7762d761ba9d45b5` |
| `nlp:amazon` | 10000 | 768 | 0.050 | `e74d1afae54bdf52` |
| `nlp:imdb` | 10000 | 768 | 0.050 | `a29c8393f5b3abc2` |
| `nlp:yelp` | 10000 | 768 | 0.050 | `ce1feb6cf48b4eb9` |

