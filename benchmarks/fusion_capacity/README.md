# Fusion capacity sweep — evidence for the shipped `hidden_dim` default

`scripts/sweep_fusion_capacity.py` measures held-out AUC + ECE through the real
`engine.fit_fusion` / `engine.score_fusion` path (i.e. the same path
`detect_with_fusion` now serves on the production decision boundary). This
directory is the persisted result that justifies the committed default.

## sweep_real_v3.json

**Config:** 4 dims (32, 64, 128, 256) × 3 seeds (0, 1, 2) × 8 genuinely-labelled
ADBench datasets (`cardio`, `mammography`, `pendigits`, `annthyroid`,
`satellite`, `Pima`, `WBC`, `Ionosphere` — medical + STEM). 96 runs, 40 epochs
each with early stopping, cap 1000 samples per dataset, 30 % held-out test.

**Aggregate**

| dim |  mean AUC | std    | mean ECE | std    |
|-----|-----------|--------|----------|--------|
|  32 |    0.9466 | 0.0696 |   0.0557 | 0.0558 |
|  64 |    0.9529 | 0.0653 |   0.0514 | 0.0494 |
| 128 |    0.9457 | 0.0754 |   0.0610 | 0.0600 |
| 256 |    0.9328 | 0.0777 |   0.0603 | 0.0591 |

**Pairwise vs dim=32 (per dataset × seed):**

| dim | mean Δ AUC | std    | wins | losses |
|-----|------------|--------|------|--------|
|  64 |    +0.0062 | 0.0176 |   11 |     13 |
| 128 |    −0.0010 | 0.0134 |   11 |     13 |
| 256 |    −0.0139 | 0.0495 |    9 |     15 |

**Recommendation:** keep `hidden_dim=32`. dim=64 has the highest mean but only by
0.006 — well inside both 1 std (0.065) and 1 SEM (0.013) of the best — and on
paired runs it wins only 11/24 times. dim=128 is statistically tied with
dim=32; dim=256 overfits (mean drop −0.014, loses 15/24). The parsimony
principle holds: don't pay for capacity that's not measurably better.

This directly refutes the single-seed claim that motivated PR #256
(`128 wins +0.014`) — that delta was inside one-sigma noise, not a real
improvement.

## Reproducing

```bash
LD_LIBRARY_PATH=/path/to/ama/build/lib python -m scripts.sweep_fusion_capacity \
  --source real --dims 32,64,128,256 --seeds 0,1,2 \
  --datasets cardio,mammography,pendigits,annthyroid,satellite,Pima,WBC,Ionosphere \
  --epochs 40 --cap-per-dataset 1000 --test-frac 0.3 \
  --output benchmarks/fusion_capacity/sweep_real_vN.json
```

Needs network on first run (ADBench NPZs are fetched into the configured cache
dir). Seed-stable for the train/test split; detector fits and the model init
are torch-seeded inside `_evaluate_once`.
