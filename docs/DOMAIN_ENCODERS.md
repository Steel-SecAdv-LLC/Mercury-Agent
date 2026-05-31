# WS-B — Differentiable domain encoders (Target 2)

## What landed (machinery, opt-in, off by default)

The production `MercuryAnomalyDetector` computes three **static** feature
families — spectral/resonance (FFT harmonic ratios), kinematic (finite-difference
velocity/accel/jerk), and Fisher/info-geometry (Mahalanobis + entropy). WS-B
reimplements each as a principled, jointly-trainable `nn.Module`
(`src/omni_mercury_engine/ml/domain_encoders.py`):

| Encoder | Differentiable basis | Static analog it generalises |
|---|---|---|
| `SpectralEncoder` | `torch.fft.rfft` magnitude → learnable filter bank | per-feature FFT harmonic ratio |
| `KinematicEncoder` | `Conv1d` kernels **initialised to** `[-1,1]`, `[1,-2,1]`, `[-1,3,-3,1]`, then learned | finite-difference jerk/accel z-scores |
| `FisherEntropyEncoder` | learned whitening → squared Mahalanobis norm + softmax entropy | precision-matrix Mahalanobis + entropy |
| `DomainEncoderStack` | concatenation + projection of the three | — |

These are **wired opt-in into the fusion path**: `engine.fit_fusion(...,
domain_encoder=True)` builds the stack, standardises the raw input (mean/std
stored for inference), pre-creates the fusion projection so its params join the
optimiser, and **jointly trains** the encoder with the fusion net (best-state
restored in sync via early stopping). Inference (`_extract_fusion_features`)
injects the encoder feature only when present.

### Off-path parity (encoder-off)

`domain_encoder=False` (the default) executes **none** of the new code:
`engine._domain_encoder is None`, no `differentiable_domain` feature group, no
extra optimiser params. Served scores match the default path within the fusion
path's own floating-point non-determinism.

**Note on "byte-identical":** the baseline `fit_fusion` is itself **not**
bit-deterministic under a fixed seed — two identical default fits differ by
~1e-15 (non-associative float reduction; pre-existing, not introduced by WS-B).
So the strongest *true* statement is: off-path is **structurally identical** and
**numerically identical within ~1e-15**. The test
(`tests/test_fusion_domain_encoder.py`) asserts structural parity and
`atol=1e-6` agreement with the default — far below any real effect, far above
the baseline noise floor.

## The ablation — and its honest verdict

`benchmarks/domain_encoder_ablation.py` runs the **faithful** comparison: the
real fusion path **without** vs **with** the encoder
(`fit_fusion(domain_encoder=False)` vs `=True`), paired on identical splits,
3 ADBench datasets × 3 seeds × 2 train-fractions, both arms fully supervised.

| dataset | frac | baseline AUC | encoder AUC | ΔAUC | seeds enc-wins |
|---|---|---|---|---|---|
| cardio | 0.25 | 0.9839 | 0.9791 | −0.0049 | 0/3 |
| cardio | 1.0 | 0.9950 | 0.9957 | +0.0007 | 1/3 |
| Pima | 0.25 | 0.5829 | 0.6026 | **+0.0197** | 2/3 |
| Pima | 1.0 | 0.6091 | 0.6192 | +0.0101 | 1/3 |
| thyroid | 0.25 | 0.9966 | 0.9963 | −0.0004 | 0/3 |
| thyroid | 1.0 | 0.9986 | 0.9983 | −0.0003 | 0/3 |

- Mean full-data ΔAUC **+0.0035**; mean low-data ΔAUC **+0.0048**; low-data
  seed agreement **0.33** (< 0.5 required).

**→ VERDICT: QUARANTINE — `domain_encoder=False` stays the default.** The
encoder helps modestly on the one hard, low-AUC dataset (Pima, +0.02/+0.01) and
washes out at the cardio/thyroid ceiling — the same low-data-helps / full-data-
washes pattern as the neurosymbolic ablation — but it does **not** clear the
conservative bar (`_AUC_MEANINGFUL=0.002` on a majority of seeds). The machinery
+ harness are kept (genuine, reusable); only the default is off. Full numbers in
`artifacts/domain_encoder_ablation.json`.

## Two confounded designs caught and rejected (anti-theater)

The first two ablation designs produced a **false positive** of mean ΔAUC
**+0.48 to +0.9** with a "KEEP" verdict. They were discarded as confounded, not
shipped:

1. **vs the production 3 component scores → tiny head.** Different information, a
   3-D bottleneck, and unsupervised-vs-supervised features. On imbalanced
   datasets the tiny head learned *inverted* rankings (AUC 0.05, 0.015) — the
   metric punishing a sign convention, not measuring the encoder.
2. **frozen-encoder vs learnable-encoder.** Same architecture/init, only encoder
   trainability differs — but a frozen *random* encoder + head still collapsed to
   inverted rankings (AUC < 0.5) on imbalanced data, so the delta was dominated
   by frozen-arm degeneracy, not a fair signal.

Both were replaced by the wired-path comparison above, where the robust
supervised fusion net removes the inversion artifact and both arms are
apples-to-apples. Reporting the +0.48 would have been theater; the honest signal
is the sub-threshold +0.005.

## Provenance & evidence

- Data: ADBench (MIT), `https://github.com/Minqi824/ADBench`; seeds 0/1/2;
  metric ROC-AUC via `mercury_ml` (**no sklearn**).
- Tests: 16 (encoders) + 5 (fusion wiring, incl. off-path parity) — all green;
  `flake8` + `mypy` clean; the 32 pre-existing fusion tests still pass.
- Artifact: `artifacts/domain_encoder_ablation.json`.
