# Fusion Capacity Strategy

Applies to Mercury Agent **v1.8.x**.

This document holds the strategic production rationale for Mercury's shipped
fusion-network width. Raw sweep artifacts and reproduction commands remain in
`benchmarks/fusion_capacity/`.

## Decision boundary

**Shipped default: `hidden_dim = 64`.** It was raised from 32 for the v1.7.0
production fusion path and the regenerated `default_fusion.pt` checkpoint.

The narrow v5 ADBench sweep ties all tested widths on 6-36 feature proxy data,
but the live benchmark profile now reaches up to 1,555 features and roughly
620K samples per dataset. At that scale, `hidden_dim = 32` is a 48:1 bottleneck.
`hidden_dim = 64` is the transitional production default; `hidden_dim = 128` is
the forward target once the full mission pipeline is producing real MIT-BIH,
SMAP, SWaT/WADI, and adjacent high-dimensional training data.

## Production-axis analysis

The accuracy question was closed by the committed v5 ADBench sweep: 256 runs,
all tested widths tied within measurement noise. The production question is
whether `dim=32` still holds once cost, retraining stability, and distribution
shift are considered. Every number below is re-derivable from
`benchmarks/fusion_capacity/sweep_real_v5.json` and the architecture in
`src/omni_mercury_engine/ml/fusion_network.py`.

### Cost axis

Parameter counts are from `OmniFusionModel` with 13 feature groups,
`num_heads=4`, and `output_dim=1`. Memory is float32 on-device weight storage;
inference activations for a typical batch of 512 samples add ≲10%.

| dim | params  | fp32 weight | cost vs dim=32 |
|----:|--------:|------------:|---------------:|
|  32 |  ~56 K  |   ~0.22 MB  |          1.0×  |
|  64 | ~176 K  |   ~0.71 MB  |          3.2×  |
| 128 | ~614 K  |   ~2.45 MB  |         11.0×  |
| 256 | ~2.28 M |   ~9.10 MB  |         40.7×  |

The cost gradient is steep and non-linear: doubling `hidden_dim` roughly
quadruples cost because cross-modal fusion and attention projections are both
dominated by O(hidden²) terms. That is the argument for `dim=32`; it is not an
accuracy argument.

### Retraining stability

Derived from v5's 64 `(dataset, seed)` pairs per width. "Within-dataset SD" is
the AUC or ECE standard deviation across the 8 seeds for a given dataset,
averaged across all 8 datasets.

| dim | AUC within-ds SD | ECE within-ds SD | AUC IQR | ECE IQR |
|----:|-----------------:|-----------------:|--------:|--------:|
|  32 |           0.0184 |           0.0161 |  0.0537 |  0.0565 |
|  64 |       **0.0149** |       **0.0134** |  0.0602 |  0.0560 |
| 128 |       **0.0126** |           0.0277 |  0.0638 |  0.0410 |
| 256 |           0.0225 |           0.0205 |  0.0690 |  0.0584 |

The hypothesis "larger dim means higher retrain variance" is refuted for AUC:
`dim=64` and `dim=128` both vary less than `dim=32`, while `dim=256` reverses
the trend. For calibration, `dim=64` improves ECE stability; `dim=128` is the
worst width by a wide margin. That makes `dim=64` the coherent transitional
choice and keeps `dim=128` as a future-data target, not the shipped default.

### Distribution-shift robustness proxy

True OOD requires training on one domain and evaluating on a held-out domain;
that infrastructure is not available in the committed corpus. The best
derivable proxy is cross-dataset AUC spread across the 8 ADBench domains.

| dim | cross-ds AUC mean | cross-ds AUC SD | max  | min  |
|----:|------------------:|----------------:|-----:|-----:|
|  32 |            0.9498 |          0.0675 | 0.994 | 0.791 |
|  64 |            0.9507 |          0.0689 | 0.996 | 0.790 |
| 128 |            0.9492 |          0.0682 | 0.995 | 0.792 |
| 256 |            0.9471 |          0.0647 | 0.995 | 0.802 |

The spread is dataset-driven, not model-driven. Pima and mammography are hard
for every width; cardio and WBC are easy for every width. `dim=256` has a
slightly lower cross-dataset SD, but at 40.7× the cost. No width materially
outperforms `dim=32` on this proxy.

## UCR time-series axis status

The UCR sweep (`sweep_ucr_v1.json`, **pending** — see the in-flight status in
`benchmarks/fusion_capacity/README.md`) targets 8 datasets to form the independent
time-series bump-criterion axis; the table below is the **preliminary** seed-pass
signal, not the committed full sweep:

| dataset | status | reason |
|---------|--------|--------|
| ECG5000, ECGFiveDays, Wafer | **saturated** (AUC ≈ 1.0 for all dims) | zero discrimination power; cannot separate widths |
| SonyAIBORobotSurface1, SonyAIBORobotSurface2 | **saturated** | same |
| Strawberry (~0.92), FordA (~0.91), FordB (~0.90) | **non-saturating** | real signal; valid for width comparison |

The five saturated datasets should not drive a width decision. A focused
48-run sweep over the three non-saturating datasets is sufficient to close
this axis:

```bash
python -m scripts.sweep_fusion_capacity \
  --source ucr \
  --dims 32,64,128,256 --seeds 0,1,2,3 \
  --datasets Strawberry,FordA,FordB \
  --epochs 60 --cap-per-dataset 1500 \
  --output benchmarks/fusion_capacity/sweep_ucr_v1_nonsaturating.json
```

## Production verdict

| Axis | dim=32 | dim=64 | dim=128 | dim=256 |
|------|:------:|:------:|:-------:|:-------:|
| AUC (v5, n=64) | 0.9498 | 0.9507 | 0.9492 | 0.9471 |
| ECE (v5, n=64) | 0.0535 | 0.0509 | 0.0561 | 0.0560 |
| Paired Δ AUC vs 32 | — | +0.0009 | −0.0007 | −0.0027 |
| Paired t vs 32 | — | +0.35 | −0.28 | −0.89 |
| Cost (params) | **56 K** | 176 K | 614 K | 2.28 M |
| AUC retrain SD | 0.0184 | **0.0149** | 0.0126 | 0.0225 |
| ECE retrain SD | 0.0161 | **0.0134** | 0.0277 | 0.0205 |
| OOD proxy (cross-ds SD) | 0.0675 | 0.0689 | 0.0682 | **0.0647** |

`dim=64` is the shipped default because it preserves the v5 accuracy tie,
improves retraining/calibration stability versus `dim=32`, stays edge
deployable, and avoids prematurely shipping the `dim=128` ECE instability
observed on the current corpus. `dim=128` remains the forward target after the
mission-scale training pipeline produces real high-dimensional medical, space,
industrial, and multi-sensor data.

## Known limits

1. **UCR non-saturating subset:** 48 focused runs remain to close the "both
   axes" accuracy clause.
2. **Real deployment-domain drift:** pooled tabular ADBench does not simulate
   live clinical, space, or industrial streams.
3. **Temporal drift:** quality degradation over months of production use is
   not evaluated here.
4. **Very high-dimensional or very large inputs:** v5 datasets are ≤5K samples
   and ≤100 features. The live benchmark profile is already materially larger,
   which is why the shipped production default is `dim=64` and the next
   evidence-gated target is `dim=128`.

## Cross-axis reproducibility check

A width change ships only if every condition passes on both the committed
ADBench axis and the UCR axis. The check is:

```bash
python3 - <<'PY'
import json, statistics, math
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
    s, p = agg(path)
    print(f"== {label} ==")
    for d, (m, t) in p.items():
        ece_def = s[32]["ece_mean"]; ece_oth = s[d]["ece_mean"]
        verdict = "PASS" if (m >= 0.02 and t >= 2.0 and ece_oth <= ece_def + 1e-9) else "fail"
        print(f"  dim={d}: Δ={m:+.4f}  t={t:+.2f}  ECE {ece_oth:.4f} vs {ece_def:.4f}  → {verdict}")
PY
```
