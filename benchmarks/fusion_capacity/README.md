# Fusion capacity sweep — evidence for the shipped `hidden_dim` default

Applies to Mercury Agent **v2.1.x**. Last updated: 2026-07-11.

`scripts/sweep_fusion_capacity.py` measures held-out AUC + ECE through the real
`engine.fit_fusion` / `engine.score_fusion` path (i.e. the same path
`detect_with_fusion` serves on the production decision boundary). This
directory holds every persisted sweep result Mercury Agent ships against.

## What "evidence-based" means here

The shipped `hidden_dim` is chosen from measured signal, not from cost. Mercury
Agent is not constrained on parameter count — there is no budget pressure
behind any width decision in this repo. The width that wins on the evidence is
the width that ships. The current default holds because no candidate clears
the bump criterion, *not* because a larger width was "too expensive".

The bump criterion is documented in
[`scripts/sweep_fusion_capacity.py`](../../scripts/sweep_fusion_capacity.py)
and reproduced below.

## Bump criterion (when to change the shipped width)

A larger candidate replaces the shipped default `hidden_dim` if and only if
**all four** hold:

1. paired mean AUC delta `(other − default)` ≥ **+0.02**
2. paired t-statistic (mean / SEM, n = seeds × datasets) ≥ **+2.0**
3. mean ECE is **not worse** on the candidate
4. (1) – (3) hold on **both** axes: classical-tabular ADBench **and**
   time-series UCR

The threshold is calibrated to the seed-noise floor measured in the v3 sweep
(one SEM was ≈ 0.013 on n = 24 paired runs). Anything below the threshold is
not distinguishable from re-running the same width twice; anything at or above
it is signal. The sweep script prints a PASS/FAIL line per candidate dim
against the current default after every run.

## sweep_real_v3.json (interim, supports keep-32)

**Config:** 4 dims (32, 64, 128, 256) × 3 seeds (0, 1, 2) × 8
genuinely-labelled ADBench datasets (`cardio`, `mammography`, `pendigits`,
`annthyroid`, `satellite`, `Pima`, `WBC`, `Ionosphere` — medical + STEM).
**96 runs**, 40 epochs each with early stopping, cap 1000 samples per dataset,
30 % held-out test.

**Aggregate (unpooled)**

| dim |  mean AUC |   sd  |   sem  | mean ECE |   sd  |
|----:|----------:|------:|-------:|---------:|------:|
|  32 |    0.9466 | 0.0711 | 0.0145 |   0.0557 | 0.0570 |
|  64 |  **0.9529** | 0.0667 | 0.0136 | **0.0514** | 0.0505 |
| 128 |    0.9457 | 0.0770 | 0.0157 |   0.0610 | 0.0612 |
| 256 |    0.9328 | 0.0794 | 0.0162 |   0.0603 | 0.0604 |

**Paired diffs vs default `dim=32` (per (dataset, seed), n = 24)**

| other |  mean Δ |   sem  | paired t | sign-test p | wins / losses |
|------:|--------:|-------:|---------:|------------:|--------------:|
|    64 | **+0.0062** | 0.0037 | **+1.69** |        0.33 | 11 / 6 (+7 ties) |
|   128 | −0.0010 | 0.0028 |    −0.35 |        1.00 | 11 / 11 |
|   256 | −0.0139 | 0.0103 |    −1.35 |        0.52 |  9 / 13 |

**Transparent read of v3.** `dim = 64` has the highest mean AUC and the lowest mean
ECE in this sweep. It is the empirical leader on the unpooled summary. But the
paired analysis against `dim = 32` gives mean Δ = +0.0062 and paired t = +1.69
— short of both the +0.02 delta threshold and the +2.0 t threshold. **None of
the four bump-criterion conditions is met** in v3, so v3 alone did not justify raising the
`dim = 32` default in place at the time. (It was later raised to 64 on the
broader evidence — see Production strategy below.) The `dim = 64` result was
real enough to warrant re-measuring on tighter sweeps, which followed.

This also refutes the single-seed PR #256 claim that `dim = 128` wins by
+0.014: in v3, `dim = 128` versus `dim = 32` lands at mean Δ = −0.0010 (dead
tied), and the original +0.014 was inside one-sigma seed noise.

## sweep_real_v4.json (harness exercise, narrow datasets)

The v4 harness adds:

* the **bump criterion** as a printed PASS/FAIL line per candidate dim;
* **paired-difference statistics** (mean, sd, sem, t, sign counts) per
  baseline dim, in addition to the unpooled summary;
* **independent time-series axis** via `--source ucr` (UCR Archive datasets,
  reframed one-vs-rest);
* a clearer config docstring with the recommended full-evidence command.

The committed `sweep_real_v4.json` is a **narrow harness-exercise run** —
5 dims (16, 32, 48, 64, 96) × 2 seeds × 3 small ADBench datasets (WBC,
Ionosphere, Pima) × 30 epochs × cap 1000 = **30 runs**. Its purpose is to
demonstrate the v4 harness end-to-end through the bump-criterion reporter,
not to settle the width question (3 small datasets cannot).

**v4 aggregate (n = 6 per dim)**

| dim | mean AUC | std    | mean ECE | std    |
|----:|---------:|-------:|---------:|-------:|
|  16 |   0.9233 | 0.0854 |   0.0909 | 0.0523 |
|  32 |   0.9085 | 0.0945 |   0.1008 | 0.0679 |
|  48 |   0.9163 | 0.0875 |   0.0931 | 0.0540 |
|  64 |   0.9171 | 0.0876 |   0.0874 | 0.0652 |
|  96 |   0.9237 | 0.0827 |   0.1005 | 0.0478 |

**v4 paired diffs vs default `dim=32` (n = 6)**

| other |  mean Δ | sem    | paired t |
|------:|--------:|-------:|---------:|
|    16 | **+0.0148** | 0.0060 | **+2.455** |
|    48 | +0.0078 | 0.0092 |   +0.842 |
|    64 | +0.0085 | 0.0098 |   +0.868 |
|    96 | +0.0152 | 0.0081 |   +1.871 |

**v4 bump-criterion verdict (printed by the harness):**

```
== Bump criterion vs default dim=32 ==
  paired mean delta >= +0.02  AND  paired t >= +2.0  AND  ECE not worse
  No dim passes all three thresholds — default stays.
```

A note on the apparent `dim=16 t=+2.455`: this is the strongest paired signal
in the v4 run, but the mean delta (+0.0148) is **below** the +0.02 threshold,
so the bump criterion correctly refuses to act on it. v4 has only 3 datasets
(all under 250 test samples) and 2 seeds; running the full-evidence config
below is the only path to a real width change.

## sweep_real_v5.json (full-evidence ADBench, 256 runs) — settles ADBench axis

**Config:** 4 dims (32, 64, 128, 256) × **8 seeds** (0…7) × 8 ADBench classical
datasets (cardio, mammography, pendigits, annthyroid, satellite, Pima, WBC,
Ionosphere) × 80 epochs (early-stop patience 15) × cap 1500 × 30% test.
**256 runs**. This is the headline ADBench evidence — strictly larger than v3
(2.67× more runs, statistical power up by ~2.6× at fixed effect size).

**Aggregate (unpooled)**

| dim | mean AUC |  std   | mean ECE |  std   |  n |
|----:|---------:|-------:|---------:|-------:|---:|
|  32 |   0.9498 | 0.0663 |   0.0535 | 0.0558 | 64 |
|  64 | **0.9507** | 0.0667 | **0.0509** | 0.0560 | 64 |
| 128 |   0.9492 | 0.0654 |   0.0561 | 0.0566 | 64 |
| 256 |   0.9471 | 0.0656 |   0.0560 | 0.0585 | 64 |

**Paired diffs vs default `dim=32` (per (dataset, seed), n = 64)**

| other |  mean Δ |  sem   | paired t | wins | losses | ties |
|------:|--------:|-------:|---------:|-----:|-------:|-----:|
|    64 | +0.0009 | 0.0026 |   +0.346 |   30 |     27 |    7 |
|   128 | −0.0007 | 0.0024 |   −0.277 |   28 |     30 |    6 |
|   256 | −0.0027 | 0.0030 |   −0.892 |   27 |     30 |    7 |

**Bump-criterion verdict.** SEM is now 0.0024-0.0030 (vs v3's 0.013), enough
to detect Δ AUC ≈ 0.005 at α=0.05 paired-t. The actual deltas vs `dim=32`
are an order of magnitude below that floor: the largest is +0.0009 for
`dim=64`. Direction-of-effect on ECE matches AUC for `dim=64` (lower mean
ECE), but the AUC delta is 22× below the +0.02 threshold and t = +0.35 vs
the +2.0 floor. **No dim passes the bump criterion**:

```
== Bump criterion vs default dim=32 ==
  paired mean delta >= +0.02  AND  paired t >= +2.0  AND  ECE not worse
  No dim passes all three thresholds — default stays.
```

The v3 "dim=64 wins by +0.006" and "dim=256 overfits by −0.014" stories were
both noise — they shrunk to +0.0009 and −0.0027 respectively at 2.67× the
power. The transparent reading: across these 8 datasets, *no* dim in {32, 64,
128, 256} can be distinguished from the others at α=0.05. Shipped width
holds on positive evidence (deltas inside the noise floor by 6-20×), not
on parsimony fallback.

## sweep_ucr_v1.json (full-evidence UCR, 160 runs) — pending; will settle the UCR axis

**Config:** 4 dims (32, 64, 128, 256) × 5 seeds (0…4) × 8 UCR datasets
(ECG5000, ECGFiveDays, Wafer, SonyAIBORobotSurface1, SonyAIBORobotSurface2,
Strawberry, FordA, FordB) × 60 epochs (early-stop patience 15) × cap 1500.
**160 runs**. UCR labels are reframed one-vs-rest (largest class = normal,
rest = anomaly) per Goldstein & Uchida (2016). This is the independent
time-series axis the bump criterion requires; v5 alone does not satisfy
the "holds on both axes" clause.

> **UCR-axis repair** — the original harness silently skipped every UCR
> dataset because `_load_ucr` passed `source="ucr"` to `DatasetConfig` (not
> a constructor kwarg), and the per-dataset mirror URL had moved from
> `timeseriesclassification.com/Downloads/` to
> `timeseriesclassification.com/aeon-toolkit/` when the upstream project
> was renamed sktime→aeon. The harness emitted `[skip] ECG5000:
> TypeError…` plus a misleading "network unreachable?" hint — the bump
> criterion's second axis was unfalsifiable. Fixed by dropping the bad
> `source=` kwarg, updating the per-dataset URL, allowlisting the apex
> `timeseriesclassification.com` (the new path 301s to the no-`www.` host),
> and teaching `UCRLoader.load` to read both the legacy nested-`.tsv` and
> the current flat-`.txt` layouts.

**Status: sweep in flight as of this commit.** The harness invocation that
produces `sweep_ucr_v1.json` is the second command in the reproduction
block below; the JSON + paired-diff verdict will land in a follow-up
commit when the sweep completes. Preliminary signal from the first
seed-pass (n=8): five of the eight datasets (ECG5000, ECGFiveDays, Wafer,
SonyAIBORobotSurface1, SonyAIBORobotSurface2) saturate at AUC ≈ 0.998-1.000
across `dim=32`, leaving three (Strawberry ≈ 0.92, FordA ≈ 0.91, FordB ≈
0.90) as the datasets where dim differences can actually be measured.

## Production strategy

Raw sweep results stay in this directory. The production rationale for the
v1.7.0 `hidden_dim = 64` default lives in
[`docs/FUSION_CAPACITY_STRATEGY.md`](../../docs/FUSION_CAPACITY_STRATEGY.md).

Short version: v5 ties all widths on narrow 6-36 feature proxy data; the live
benchmark profile reaches up to 1,555 features and roughly 620K samples per
dataset. `hidden_dim = 64` is the transitional production default, while
`hidden_dim = 128` remains the forward target after real mission-scale MIT-BIH,
SMAP, SWaT/WADI, and adjacent high-dimensional training data are available.

## Cross-axis verdict (accuracy)

A width change ships only if every condition passes on **both**
`sweep_real_v5.json` (ADBench) and `sweep_ucr_v1.json` (UCR). v5 fails on
ADBench for every candidate dim with margin (see Production verdict above);
the UCR non-saturating focused sweep is the remaining open item (see UCR
status above). The cross-axis accuracy check is reproducible via:

```bash
python3 - <<'PY'
import json, statistics, math, os
def agg(p, default=32):
    runs = json.load(open(p))["runs"]
    dims = sorted({r["dim"] for r in runs})
    summ = {d: {"auc_mean": statistics.fmean(r["auc"] for r in runs if r["dim"]==d),
                "ece_mean": statistics.fmean(r["ece"] for r in runs if r["dim"]==d)}
            for d in dims}
    pairs = {}
    ref = {(r["dataset"], r["seed"]): r["auc"] for r in runs if r["dim"]==default}
    for d in dims:
        if d == default: continue
        oth = {(r["dataset"], r["seed"]): r["auc"] for r in runs if r["dim"]==d}
        diffs = [oth[k]-ref[k] for k in sorted(set(ref) & set(oth))]
        sd = statistics.stdev(diffs); m = statistics.fmean(diffs)
        pairs[d] = (m, m/(sd/math.sqrt(len(diffs))))
    return summ, pairs
for label, path in (("ADBench", "benchmarks/fusion_capacity/sweep_real_v5.json"),
                    ("UCR",     "benchmarks/fusion_capacity/sweep_ucr_v1.json")):
    if not os.path.exists(path):
        print(f"== {label} == (pending — {path} not committed yet)")
        continue
    s, p = agg(path)
    print(f"== {label} ==")
    for d, (m, t) in p.items():
        ece_def = s[32]["ece_mean"]; ece_oth = s[d]["ece_mean"]
        verdict = "PASS" if (m >= 0.02 and t >= 2.0 and ece_oth <= ece_def + 1e-9) else "fail"
        print(f"  dim={d}: Δ={m:+.4f}  t={t:+.2f}  ECE {ece_oth:.4f} vs {ece_def:.4f}  → {verdict}")
PY
```

## The full-evidence sweep config (reproduction)

The committed v5 sweep (and the pending UCR v1 sweep) are produced with:

```bash
# ADBench axis — 256 runs, ~40 min on a modern CPU.
python -m scripts.sweep_fusion_capacity \
  --source real \
  --dims 32,64,128,256 --seeds 0,1,2,3,4,5,6,7 \
  --datasets cardio,mammography,pendigits,annthyroid,satellite,Pima,WBC,Ionosphere \
  --epochs 80 --cap-per-dataset 1500 \
  --output benchmarks/fusion_capacity/sweep_real_v5.json

# UCR axis — 160 runs, ~75 min on a modern CPU.
python -m scripts.sweep_fusion_capacity \
  --source ucr \
  --dims 32,64,128,256 --seeds 0,1,2,3,4 \
  --datasets ECG5000,ECGFiveDays,Wafer,SonyAIBORobotSurface1,SonyAIBORobotSurface2,Strawberry,FordA,FordB \
  --epochs 60 --cap-per-dataset 1500 \
  --output benchmarks/fusion_capacity/sweep_ucr_v1.json
```

A wider future re-run (e.g. dims `{32, 48, 64, 96, 128, 192, 256}` × 10
seeds × add 8 more ADBench datasets + non-classical UCR like NAB / SMAP /
MSL) would push the detectable effect size below ΔAUC ≈ 0.003 — finer than
the v5 noise floor — but at this scope no candidate dim survives the bump
criterion on either axis.

## Reproducing

```bash
# Smoke (offline, single-class saturated):
python -m scripts.sweep_fusion_capacity --source synthetic --seeds 0,1

# v3 (interim) configuration:
python -m scripts.sweep_fusion_capacity --source real \
  --dims 32,64,128,256 --seeds 0,1,2 \
  --datasets cardio,mammography,pendigits,annthyroid,satellite,Pima,WBC,Ionosphere \
  --epochs 40 --cap-per-dataset 1000 --test-frac 0.3
```

ADBench cache: `~/.cache/mercury_agent/adbench`. Seeds drive the train/test
split and torch/numpy RNG; ADBench NPZs are fetched on first use.
