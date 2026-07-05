# WS-A — Anomaly-score "regression" (#261): root cause and guard

## TL;DR

The drop reported in **#261** — mean ROC-AUC `0.8466 → 0.8259`, Oracle-F1
`0.6428 → 0.6046` — is **not a detector regression**. It is the direct,
intended consequence of **PR #255 making the headline metric honest** by
excluding circular *manufactured-label* datasets from the supervised headline.
On an apples-to-apples basis the detector is **flat-to-slightly-up**.

Per the project's operating contract ("never move an a-priori bar to
manufacture a pass"), the corrective action is **not** to revert the de-leak to
re-inflate the number. It is to **pin the real metric floor deterministically**
so a genuine future regression cannot land silently. That guard is
`benchmarks/anomaly_regression_guard.py` + `anomaly_regression_baseline.json`.

## Reproduction / isolation

| State | Commit | mean AUC | mean F1 | basis |
|---|---|---|---|---|
| Committed baseline | `cda5a73` (JSON from CI run `79e8335`) | 0.8466 | 0.6428 | all successful datasets |
| Fresh run (#261) | `e9feeec` (#255 merge) | **0.8259** | **0.6046** | genuine-labels-only headline |

`statistical.py` and `mercury_benchmark.py` changed in exactly one commit
between those states: the **#255 merge**. So #255 is the isolating change.

## Root cause

Two changes shipped in #255 touch this number:

1. **Eval-honesty de-leak (`mercury_benchmark.py`).** The headline `mean_auc` /
   `mean_oracle_f1` are now computed over **genuine-label** datasets only
   (`ground_truth | expert_annotated`). 13 datasets whose anomaly labels were
   *manufactured* by thresholding a detector-like feature (USGS_Earthquake,
   NOAA_Weather, Wildfire, NOAA_Buoy, NOAA_StormEvents, NOAA_GSOD, NOAA_ERDDAP,
   EPA_AirQuality, FEMA_Disaster, NASA_Exoplanet, SolarDynamics, MSDS,
   ThreatIntel) are circular — scoring a detector against them inflates AUC —
   and are now reported separately under `unsupervised_eval`.

2. **`statistical.py` change = type annotations only.** The entire detector
   diff across #255 is `np.ndarray → np.ndarray[Any, Any]` signatures. **Zero
   runtime effect.** (`adbench.py` also changed, but only a dataset-selection /
   cache-existence **bug fix** that makes *more* datasets load correctly.)

The new harness records the exact split in its own `deleak_delta` field
(from #261's run on `e9feeec`):

```
mean_auc_all_datasets : 0.8495   <- apples-to-apples with the old 0.8466
mean_auc_genuine_only : 0.8259   <- new honest headline
mean_auc_delta        : -0.0237  <- removing 13 circular datasets (mean AUC 0.9479)
```

### Apples-to-apples (all-datasets basis), recomputed per-dataset

| | old baseline (all) | fresh `e9feeec` (all) | Δ |
|---|---|---|---|
| mean AUC | 0.8466 | **0.8495** | **+0.0029** |
| mean F1 | 0.6428 | **0.6422** | −0.0006 |

The detector did not regress. The 13 excluded datasets had mean AUC 0.9479 /
F1 0.7985; removing them from the headline (correctly) lowers the *headline*
without any change in detector quality.

## Why we do NOT "restore ≥ 84"

The ≥ 84 / ≥ 64 bar was measured on a **label-contaminated** dataset set.
Re-including the manufactured-label datasets to recover `0.8466` would
re-introduce circular-label inflation — exactly the theater the operating
contract forbids. The honest genuine-only headline is `0.8259 / 0.6046`, and
that is the number the repo should carry. The committed
`mercury_benchmark_results.json` is CI-managed and refreshes to the honest
format on the next `main` benchmark run.

## The guard (so a *real* regression is caught)

`benchmarks/anomaly_regression_guard.py` deterministically evaluates the
**unchanged** detector eval path (`mercury_benchmark._benchmark_single`,
seed 42) on a fixed set of 8 genuine-label ADBench datasets spanning strong and
weak signal:

| dataset | AUC | F1 |
|---|---|---|
| breastw | 0.9915 | 0.9668 |
| cardio | 0.9514 | 0.7769 |
| Ionosphere | 0.9414 | 0.8976 |
| WBC | 0.9907 | 0.9000 |
| Lymphography | 0.9836 | 0.8000 |
| Pima | 0.7047 | 0.7077 |
| glass | 0.7681 | 0.3030 |
| pendigits | 0.8741 | 0.2891 |
| **mean** | **0.9007** | **0.7051** |

Per-dataset + mean floors (measured − margin) are pinned in
`anomaly_regression_baseline.json` together with full provenance (ADBench
source + MIT license, per-dataset NPZ SHA-256, seed, metric definitions,
commit). `--check` fails non-zero below any floor.

* **Determinism:** verified byte-identical AUC/F1 across repeated runs.
* **CI:** `python benchmarks/anomaly_regression_guard.py --check` is a fast
  (~20 s) gate in the benchmark workflow; the pytest tier
  (`tests/benchmarks/test_anomaly_regression_guard.py`) runs it under
  `MERCURY_NETWORK_TESTS=1`.
* **No new deps:** uses Mercury's own `mercury_ml` metrics — **no sklearn**.

To re-pin after an *intended* detector change:
`python benchmarks/anomaly_regression_guard.py --update`.

## Generalization (this round): a repo-wide circular-label gate

The #255/#262 de-leak removed 13 circular *manufactured-label* datasets from the
supervised headline by declaring `LABEL_SOURCE = "statistical"` on those loaders.
That fix had a structural weakness: **the base class silently defaults
`LABEL_SOURCE = "ground_truth"`** (`datasets/base.py`), so any loader that
manufactures anomaly labels (a feature threshold, a z-score fence, a synthetic
generator) and simply forgets to override the attribute is counted as genuine —
re-inflating the headline. Standing vigilance (re-auditing by hand each PR) does
not scale; one canonical detector does.

`src/omni_mercury_engine/datasets/label_provenance.py` is that detector:

* **`LABEL_PROVENANCE_REGISTRY`** — the frozen result of a full audit of all
  **40** concrete loaders: each maps to `(label_source, justification)`, where
  the justification states *how the labels are derived*. Flipping a loader from
  `statistical` to `ground_truth` to inflate a number now requires a reviewable
  one-line diff here.
* **`audit_label_provenance()`** — enumerates the live loaders and returns every
  divergence: a loader missing from the registry (a *new* dataset whose
  provenance was never declared → forces an audited entry), a declared
  `LABEL_SOURCE` that disagrees with the audited value, an invalid value, or a
  stale registry entry.
* **`scan_circular_label_construction()`** — a static heuristic that reads each
  loader's own source for the exact circular pattern
  (`labels = (features[...] > threshold)` in a *real*, non-synthetic code path)
  and flags any loader that manufactures labels while declared genuine. It fires
  on the threshold pattern and is silent on genuine label-column reads
  (verified against SWaT / NSL-KDD / NAB / ADBench / UCR — no false positives).

### Residual circularity found and fixed

Auditing the remaining tree with this methodology surfaced **four** loaders that
manufactured labels by thresholding the same ocean/atmosphere features the
detector consumes, yet inherited the silent `ground_truth` default:

| loader | manufactured label rule |
|---|---|
| `climate.SimonsCMAPLoader` | `oxygen<2 \| temp>30 \| nitrate>30` |
| `climate.CopernicusSeaLevelLoader` | `\|sea-level-anomaly\| > 0.15 m` |
| `climate.CopernicusERA5Loader` | temperature / salinity-anomaly thresholds |
| `climate.WorldOceanDatabaseLoader` | no real labels; synthetic/threshold profiles |

None are in the current benchmark eval set, so the *headline today is
unaffected* — but each was a latent leak: adding any to the eval set would have
silently inflated the supervised AUC. All four are corrected to
`LABEL_SOURCE = "statistical"` at source. Two medical loaders mislabeled by the
same silent default (`PhysioNetLoader`, `CardiologyDataset`) were corrected to
`expert_annotated`. Post-fix the tree is **16 ground_truth / 3 expert_annotated /
21 statistical**, and the audit is clean.

### The gate

* **pytest:** `tests/datasets/test_label_provenance_gate.py` runs the full audit
  (offline, no downloads) in the standard `tests/datasets/` lane.
* **CI:** `python -m omni_mercury_engine.datasets.label_provenance --check` runs
  as a fast named step in `benchmark.yml`, beside the regression guard.

A future PR that adds a circular dataset now fails on one of: the loader is
unregistered, the declaration disagrees with the audited value, or the
circularity heuristic fires — so a circular dataset cannot re-enter the
supervised headline silently.
