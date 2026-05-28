# Fusion capacity sweep — evidence for the shipped `hidden_dim` default

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

**Honest read of v3.** `dim = 64` has the highest mean AUC and the lowest mean
ECE in this sweep. It is the empirical leader on the unpooled summary. But the
paired analysis against `dim = 32` gives mean Δ = +0.0062 and paired t = +1.69
— short of both the +0.02 delta threshold and the +2.0 t threshold. **None of
the four bump-criterion conditions is met**, so the shipped default stays at
`dim = 32`. The `dim = 64` result is real enough to be worth re-measuring on a
tighter sweep, but not strong enough to change the shipped artifact today.

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

## The full-evidence sweep config

This is the run that should clear or refute the bump criterion before the
next shipped-width change:

```bash
# Classical-tabular axis (ADBench: 16 datasets, 8 seeds, 120 epochs, cap 5000)
LD_LIBRARY_PATH=/path/to/ama/build/lib python -m scripts.sweep_fusion_capacity \
  --source real \
  --dims 16,32,48,64,96 \
  --seeds 0,1,2,3,4,5,6,7 \
  --datasets cardio,mammography,pendigits,annthyroid,satellite,Pima,WBC,Ionosphere,\
             thyroid,vowels,letter,musk,optdigits,shuttle,glass,vertebral \
  --epochs 120 --cap-per-dataset 5000 \
  --output benchmarks/fusion_capacity/sweep_real_v5.json

# Independent time-series axis (UCR Archive: 8 datasets, 8 seeds, 120 epochs, cap 5000)
python -m scripts.sweep_fusion_capacity \
  --source ucr \
  --dims 16,32,48,64,96 \
  --seeds 0,1,2,3,4,5,6,7 \
  --datasets ECG5000,ECGFiveDays,Wafer,FordA,FordB,Earthquakes,Strawberry,Coffee \
  --epochs 120 --cap-per-dataset 5000 \
  --output benchmarks/fusion_capacity/sweep_ucr_v1.json
```

640 + 320 = 960 runs total. Expect ~5-10 GPU-hours or ~24-48 CPU-hours on a
modern desktop. The bump criterion's four conditions (paired mean delta ≥
+0.02, paired t ≥ +2.0, mean ECE not worse, holding on both axes) must clear
on both JSON outputs before the shipped `hidden_dim` changes.

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
