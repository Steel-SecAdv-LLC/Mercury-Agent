# Governed Fusion Substrate — Phase 2 correction pass (PR #278)

All numbers below were reproduced **in this branch from committed code** on the
real reachable suite (cold cache; `measure_*.py` + `build_manifest.py`). Nothing
here is borrowed from FINDOYOU or any paper.

## Honest correction this pass made (read first)

The earlier framing called the suite **"29 real events / 9 domains."** Re-running
the loaders with synthetic-path instrumentation showed that is **not** accurate:
**tsunami (3) and energy (3) synthesise their series by design** — there is no live
path (`tsunami_loader.fetch_historical`: *"historical event data is synthesised
from characteristic BPR patterns"*; `energy_loader`: *"generates synthetic Kp
time-series data based on documented storm profiles"*). `pandemic/ebola_2014` also
reconstructs (the WHO GHO Ebola series 404s). `MERCURY_ALLOW_SYNTHETIC=0` does
**not** gate these: it is honoured only by the `datasets/` package, not the
`loaders/` package the suite uses.

So the suite is split, in code (`suite.py: RECONSTRUCTED_DOMAINS / RECONSTRUCTED_EVENTS`)
and in `manifest.json`, into two never-conflated classes:

- **Live headline suite — 23 real events / 7 domains.** earthquake 5, pandemic 5,
  hurricane 4, fema 3, tornado 2, marine 2, network_security 2. **Every headline
  figure below is this set.** Verified live: no synthetic-generation method is
  reached (`build_suite(kind="real")`).
- **Reconstructed-from-live — 7 events / 3 domains.** tsunami 3, energy 3,
  pandemic/ebola_2014 1. For these *documented real events* the live feed is
  unavailable, so the loader reconstructs a series mirroring the event's
  statistical properties. Per the project's data doctrine this is the **credible
  next-best source when live data is unachievable** — reported separately and
  **always labelled reconstruction, never claimed as live**.

The live-only headline AUROC (**0.823**) is *lower* than the old mixed 0.839: the
reconstructions were the easier data (reconstructed baseline AUROC 0.920). The
provable number is the smaller one.

**Provenance guard (this pass).** The static label is correct today, but `marine`
is live-labelled while its loader *silently* synthesises on an empty OBIS response
(tagging every row `dataset_id="synthetic"`; the live OBIS path never does). To
stop a *future* OBIS outage inflating the 23-event live headline with synthesised
marine data, `suite._load_event` now refuses to label any event live whose loader
returned that marker (`_looks_synthesized` / `ProvenanceError`, announced as
`PROVENANCE_SKIP`). It reads an existing column — **no loader return-contract
change** — and is cache-safe (cache hits return before the fetch), so the committed
cached/online headline is byte-unchanged; an offline fresh build now *excludes*
marine rather than mislabelling it. `energy`/`tsunami`/`ebola_2014` are
RECONSTRUCTED-labelled, so the guard never fires for them (synthesis is expected).
Bounded residual: a loader that synthesises *without* the marker is not caught —
today the only live-labelled silent-synthesizer is `marine`, and it is tagged
(`pandemic/ebola_2014` also synthesises unmarked but is RECONSTRUCTED-labelled).
Tests: `tests/research/test_governed_suite_provenance.py` (4, offline).

## Environment & suite

- AMA/PQC native backend required (hard import gate); env:
  `PYTHONPATH=.:src:/tmp/ama-cryptography`,
  `AMA_CRYPTO_LIB_PATH=…/build/lib/libama_cryptography.so`,
  `LD_LIBRARY_PATH=…/build/lib`, `MERCURY_ALLOW_SYNTHETIC=0`,
  **`PYTHONHASHSEED=0`** (the reconstruction loaders now derive their RNG seed
  from `hashlib.sha256` — process-stable — so the reconstructed group reproduces
  byte-identically *without* relying on this pin; energy/marine were previously
  salted `hash()` and were fixed this pass. The pin is retained as
  defense-in-depth and for the `datasets/` path; the 23-event live suite is
  hash-independent regardless).
- Metrics via `ml/mercury_ml` (no scikit-learn). Aggregation is the per-event
  **macro mean** (equal weight per event), matching `research/omni_equation`.
- Large events use a **seeded stratified row cap** (default 6000;
  `suite.stratified_subsample`, seed 42) for iteration. `MercuryAnomalyDetector`
  is deterministic post-`fit()`; each event's scores are cached.
- **Reproduction note:** 22 of the 23 live events reproduce their committed
  `(X,y)` fingerprint exactly; `marine/marine_heatwave_2023` shifts because OBIS
  returns live occurrence records that move between fetches — genuine live-data
  refetch on real data, disclosed in `manifest.json`.
- **Reproducibility hardening (this pass):** `energy_loader` and `marine_loader`
  seeded their reconstruction RNG from Python's per-process-salted `hash(str)`;
  both now derive the seed from `hashlib.sha256` (matching `tsunami_loader`), so
  the reconstructed group reproduces byte-identically across processes — verified
  identical across `PYTHONHASHSEED=1/2/random`. Energy's three `(X,y)`
  fingerprints moved (`quebec_1989` `e3c19e42→b01043f2`, `halloween_2003`
  `fcb55bcc→7dd9a201`, `bastille_day_2000` `915294c5→c63029f2`; labels and
  `n_pos` unchanged) and the reconstructed baseline AUROC re-reproduced
  0.916→0.920. The `tsunami`×3 + `ebola_2014` fingerprints and the **entire
  23-event live headline are byte-unchanged** (proving the change is isolated to
  energy's feature noise).

## Baseline (default fixed ensemble) — `measure_baseline.py`

| group | events / domains | AUROC | AUPRC | F1 | P | R |
|---|---|---:|---:|---:|---:|---:|
| **live headline** | 23 / 7 | **0.823** | **0.410** | **0.277** | **0.273** | **0.572** |
| reconstructed (labelled) | 7 / 3 | 0.920 | 0.552 | 0.368 | 0.476 | 0.549 |

Reproduced from committed code (`results/baseline_results.json`). Pooled
row-weighted AUROC on the live suite is 0.560 — confirming the macro mean is the
right, non-swamped statistic. Per-event rows, per-domain means and SHA-256 data
fingerprints are in `results/` and `manifest.json` (see `RUN.md`).

## Item 4 — conformal operating point: wired (opt-in, default-off); **F1-win claim is a conclusive negative against the baseline it displaces**

`measure_conformal.py` (`results/conformal_results.json`). Per event: seeded
stratified 50/50 calibration/eval split; the detector is fit unsupervised, so
calibration labels form a valid split-conformal set. The flag's threshold =
class-1 LAC quantile (`1 - q_1`) from `BinaryConformalClassifier`. All operating
points act on identical eval scores, so AUROC/AUPRC are rank-invariant — the
lever is purely the operating point. **20 of 23 live events used; 3 dropped**
(earthquake `noto_2024`, `tohoku_2011`, `nepal_2015` — no positive in a
calibration split), reported never hidden.

The flag **displaces the supervised Youden/F1 threshold**, not the adaptive
baseline (`fit_with_calibration_subset` calibrates best-of(Youden's J, F1) when
the flag is off — mirrored byte-for-byte in `measure_conformal.py`):

| metric | adaptive | **youden_f1 (displaced)** | conformal | Δ(conf − youden_f1) |
|---|---:|---:|---:|---:|
| AUROC | 0.796 | 0.796 | 0.796 | +0.000 (rank-invariant) |
| AUPRC | 0.435 | 0.435 | 0.435 | +0.000 (rank-invariant) |
| F1 | 0.307 | **0.470** | 0.398 | **−0.071** |
| precision | 0.303 | 0.406 | 0.301 | −0.104 |
| recall | 0.489 | 0.645 | 0.826 | +0.182 |

**Conclusion (honest, committed):** against the operating point it displaces, the
conformal threshold **loses F1 (−0.071)** — it is *not* an F1 win. It is a
**recall/coverage-favouring** operating point: it trades precision (−0.104) for a
large recall gain (+0.182) and a distribution-free class-1 coverage guarantee,
valuable in the missed-detection-catastrophic regime (R2) but not a strict F1
improvement. Per-domain it regresses F1 vs Youden/F1 in 5/7 domains
(network_security −0.215, tornado −0.136, fema −0.105, pandemic −0.054,
hurricane −0.035; earthquake/marine tie), disclosed not hidden. (For reference
only, conformal vs the *adaptive* baseline is +0.091 F1 — the old framing.) Wired
opt-in via `conformal_operating_point` (default off → byte-exact Youden/F1 path);
see `tests/test_conformal_operating_point.py`. The calibration thesis (Stage 2)
targets this gap: a proper-scored monotone probability with an exact-reducing
accept-gate, so the shipped path can never regress.

## Item 2 — adversarial survivability on the REAL fused score — research only

`measure_survivability.py` + `research/adversarial/governed_attacks.py`. One
representative live event per domain (most positives); `eps=0.6`, 160-row
stratified cap. The attack target is Mercury's **real fused ensemble anomaly
score**, not a toy `||x−loc||`. Fixed-budget battery (condmean / BPDA / NES /
transfer) restricted to a controlled subset of `m` channels.

### Floor curve — worst-case fused AUROC vs controlled-channel budget `m`

| domain | event | k | clean | floor curve (m: worst-case AUROC) |
|---|---|--:|--:|---|
| earthquake | turkey_syria_2023 | 8 | 0.987 | 2:0.484 4:0.327 6:0.201 |
| fema | hurricane_2024 | 7 | 0.924 | 1:0.802 3:0.829 5:0.594 |
| hurricane | milton_2024 | 8 | 0.887 | 2:0.887 4:0.850 6:0.854 |
| marine | marine_heatwave_2023 | 3 | 0.999 | 1:0.977 2:0.947 |
| network_security | nsl_kdd | 9 | 0.913 | 2:0.877 4:0.817 6:0.759 |
| pandemic | mpox_2022 | 12 | 0.795 | 3:0.793 6:0.794 9:0.791 |
| tornado | super_outbreak_2011 | 12 | 0.964 | 3:0.858 6:0.357 9:0.420 |

**Overall:** mean clean AUROC 0.924 → mean worst-case at `m=k/2` 0.707 (drop
0.217). Some domains collapse through chance under on-manifold half-channel
evasion (earthquake 0.987→0.327, tornado 0.964→0.357); others are robust
(hurricane, mpox). The masking flag fires iff NES (gradient-free) beats BPDA
(gradient-based); `win_counts` is a true per-row tally.

### Cubic-moment escape — `D_φ` over `φ(z)=[z, z²−1, z³]` vs the Gaussian floor

- **Gaussian control sanity:** escape **−0.003** — the detector vanishes to the
  floor on mean/cov-only Gaussian data.
- Real domains (escape = cubic AUC − floor AUC): hurricane **+0.119**, pandemic
  **+0.117**, network_security +0.047, tornado +0.029, fema −0.009, earthquake /
  marine +0.000 (floor already 1.000). Where the floor is not saturated, the
  cubic detector escapes it — the anomalies carry genuine 3rd-moment structure
  mean+covariance cannot see.

Zero runtime change; see `tests/research/test_governed_adversarial_smoke.py`.

## Item 3 — reliability-weighted bounded-influence fusion: **KILL CONFIRMED**

`measure_reliability_fusion.py` + `core/robust_pooling.py`. Reliability weights
(per-component AUROC-above-chance, #38 self-down-weighting) combined with the
bounded-influence **clipped** / **trimmed** log-odds pool. Per event: 50/50
cal/eval split, weights from calibration only, AUROC on eval (20 live events).

| | baseline 0.40/0.30/0.30 | rel·linear | rel·clipped | rel·trimmed | best-single |
|---|---:|---:|---:|---:|---:|
| OVERALL (20 ev) | 0.804 | 0.840 | **0.843** | 0.811 | **0.878** |

Reproduced in `results/reliability_fusion_results.json`. The bounded-influence
pool **improves over baseline** (0.804 → 0.843) but still **does not reach
best-single** (0.878); gap **−0.035**, with per-domain collapse
(hurricane −0.125, fema −0.047, network_security −0.019). On the reachable live
suite the components are too redundant for any pooling to beat the single best
stream.

**Decision:** measured from committed code with the specified machinery, the
fusion-weighting lever is **conclusively killed** (I3: no clean full-suite gain ⇒
not wired into runtime). `core/robust_pooling.py` is retained as a tested
prototype + the reproducible script; `detect()` is unchanged.

## Item 1 — info-geometry certificate: scoped honestly, boundary-correct

`core/governed_fusion.py` (`InfoGeometryCertificate`) + `detectors/statistical.py`.
`p_τ` is `g⁻¹(component_threshold)` using the info-geometry component's **own**
adaptive operating point and its **real** score map `g(p)=1−exp(−p²/2d)`. The
payload certifies the component's price level-set, **not** the fused/gated
verdict (out of scope). Optional, **default-off**, post-hoc and read-only (exact
reduction).

## Item E — certificate wiring threaded, no stale engine state

`OmniMercuryEngine._extract_detector_features` returns certificates in its tuple
(now a 3-tuple) and `detect_with_fusion` threads them straight into the result.
The mutable `self._last_detector_certificates` is gone, so interleaved `detect`
calls cannot cross-contaminate certificates
(`tests/test_governed_certificate_threading.py`). The co-training stub that
mirrors this method was updated to the 3-tuple contract
(`tests/test_fusion_symbolic_cotraining.py`).

## Stage 2 — the calibration thesis (headline): MCA lands, Venn-Abers dropped

`measure_calibration.py` (`results/calibration_results.json`). Per event: seeded
50/50 cal/eval split; fit on calibration, four metrics on held-out eval
(20 live events; the 3 earthquake events with no calibration positive drop, as in
Item 4).

### Report card — overall (20 live events)

| method | AUROC | Brier | ECE | Net-Benefit |
|---|---:|---:|---:|---:|
| identity (scaled scores) | 0.7957 | 0.1832 | 0.2738 | 0.0448 |
| isotonic | 0.7586 | 0.0920 | 0.0484 | 0.0892 |
| **Beta-MCA** | **0.7957** | **0.0893** | **0.0387** | **0.0991** |
| Beta-MCA (accept-gated) | 0.7957 | 0.0893 | 0.0387 | 0.0991 |
| Beta-MCA + Venn-Abers | 0.7952 | 0.0890 | 0.0655 | 0.0987 |

**R1 — Beta-MCA lands (opt-in, default-off, exact-reducing).** The strictly
monotone beta map (Kull 2017, `a, b ≥ 0`) fit by the composite proper objective
(Brier + λ_ECE·ECE_kernel) **ties AUROC exactly** (0.7957 = 0.7957 — I3-free) and
improves **Brier −0.0939** and **ECE −0.2351** vs identity. Head-to-head it
**beats isotonic** on both Brier (0.0893 < 0.0920) and ECE (0.0387 < 0.0484)
*and* preserves AUROC, where isotonic **drops AUROC −0.0371**. Wired behind
`calibration_map="mca"` (default off → byte-exact; additive
`calibrated_probabilities` key when on). Tests: `tests/test_beta_calibration.py`.

**R4 — exact-reducing accept-gate + four-metric report card.** `fit_accept_gated_mca`
accepts the map only if held-out-style Brier improves AND ECE ties-or-beats, else
identity; accepted in **20/20** events with **zero per-domain regressions**, so
the shipped calibration path can never regress AUROC/Brier/ECE.

**R3 — Venn-Abers validity layer: conclusive negative, NOT shipped.** Layered on
the MCA point probability, the inductive Venn-Abers predictor's marginal
contribution is **Brier −0.0002 (≈0), ECE +0.0267 (worse), AUROC −0.0005** (mean
interval width 0.0857). It adds nothing to point calibration over a good MCA, so
per I3 it is **not** wired into the runtime; `VennAbersCalibrator` is retained as
a tested prototype (`tests/test_venn_abers.py`) + a distribution-free uncertainty
band.

### Stage 2 lever probe — conclusive negative (`measure_calibration_levers.py`)

Can the composite objective, the accept-gate tolerance, or the beta warm-start
lower **held-out** Brier+ECE without breaking the AUROC tie or the no-regress
gate? Swept on the same splits (`results/calibration_levers_results.json`):

| config | Brier | ECE | d_auroc | ΔBrier | ΔECE |
|---|---:|---:|---:|---:|---:|
| **default (λ_ece=1, λ_nb=0, identity)** | 0.0893 | 0.0387 | +0.000 | — | — |
| λ_ece=0.0 | 0.0853 | 0.0469 | +0.000 | −0.0040 | +0.0081 |
| λ_ece=2.0 | 0.0909 | 0.0441 | +0.000 | +0.0016 | +0.0054 |
| λ_nb=0.5 | 0.0879 | 0.0433 | +0.000 | −0.0014 | +0.0045 |
| warm=mle | 0.1645 | 0.1297 | −0.0235 | +0.0752 | +0.0910 |

**No lever beats the shipped default on held-out Brier AND ECE** (λ_ece<1 trades
ECE for Brier; λ_nb only nudges Brier; accept-gate tol sweep accepts 20/20 at
every tol with no change). Critically, the **maximum-likelihood beta warm start**
the docstring once *claimed* both **degrades calibration** (Brier +0.075,
ECE +0.091) **and breaks the AUROC tie** (d_auroc −0.0235) — so the shipped
identity start (`a=b=1, c=0 ⇒ p=u`, d_auroc +0.000) is the correct choice and is
retained. The misleading "maximum-likelihood beta fit" comment in
`core/calibration.py` was corrected to match the code and this measurement.

## Stage 3 — decision-curve thresholding (R2) + gated η^Φ decoupling (R6)

### R2 — decision curve + ONE operating-point pathway

`core/decision_curve.py` implements net benefit `NB(t) = TP/n − FP/n·t/(1−t)`,
the treat-all/treat-none envelopes, the low-t-weighted per-domain prior
`π(t) ~ 1/t`, and the cost-driven Bayes threshold `t* = c/(c+b)`. The Net-Benefit
column of the Stage 2 report card is computed **through this module**, so the
committed NB figures reproduce from `decision_curve.py`.

**One operating-point pathway (reconciliation with Item 4).** The substrate
exposes exactly ONE principled operating point: the MCA-calibrated probability
thresholded at the cost-driven Bayes `t*` (for `b = 10c`, `t* = 0.091`), returned
by `reconciled_operating_point`. The conformal / Venn-Abers layer is reported
there as a distribution-free **coverage floor** (a recall-floor diagnostic) — not
a second, competing threshold. This module is **read-only analysis (default-off);
it changes no runtime verdict** (the detector's verdict stays `score > threshold`;
MCA only adds the additive `calibrated_probabilities` key). Item 4's
`conformal_operating_point` flag remains the *separate* opt-in lever that swaps
the detector's threshold for the conformal quantile — measured above as a
recall/coverage trade, default off. Tests: `tests/test_decision_curve.py`.

### R6 — decouple η^Φ from the probability (opt-in, default-off)

`OmniAvaEquation(decouple_ethical_scaling=True)` removes the **soft** η^Φ
multiplier from the fused-score path so a proper-scored monotone calibrator (MCA)
can own the probability. **Default-off ⇒ byte-identical fused score** (equality
test, `tests/test_eta_decoupling.py`). The two fail-closed **hard** gates
(`BenevolenceScorer` floor 0.70, `σ_Immutable`) are byte-untouched and remain the
enforcement (I1); only the soft in-score multiplier is removed when on.

**Suite measurement (honest scoping).** The reachable suite's detector score path
(`MercuryAnomalyDetector.detect`) has **no** soft η^Φ multiplier, so the
suite-measurable form of "MCA owns the probability" is exactly the Stage 2
`calibration_map="mca"` result (no-regression: AUROC exact tie 0.7957, Brier
−0.0939, ECE −0.2351). The `OmniAvaEquation` η^Φ decoupling is the
byte-identical-default-off structural analogue for the three-R fusion subsystem
(not exercised by the suite); recorded, not claimed as a separate suite gain.

## Invariants

I1 both hard gates (`BenevolenceScorer`, `σ_Immutable`) byte-untouched, ethics
tests green. I2 every addition optional + default-off + exact-reducing; no new
heavy deps (`mercury_ml`, not sklearn — guardrail `tests/test_no_sklearn_in_src.py`).
I3 AUROC-losing levers (Item 3; Venn-Abers; every Stage-2 probe lever) are not
shipped; conclusive negatives recorded with committed numbers. **I4 — live data
is the headline (23 real events / 7 domains); where the live feed is unachievable
for a documented real event, a reconstruction mirroring its statistical
properties is the credible next-best source, reported separately and labelled
reconstruction (7 events / 3 domains), never conflated with live and never
claimed as real.**

## Close-out validation sweep (this pass) — per-axis verdict

Every axis below either **landed** a reproduced improvement (freeze-and-add,
revert-on-regress) or is a **committed conclusive negative** proving the current
state is already optimal. No speculative levers, no micro-benchmark theatre. Each
verdict cites the committed artifact that proves it (numbers re-read from
`results/*.json` this pass).

| axis | verdict | committed proof |
|---|---|---|
| baseline ensemble (0.40/0.30/0.30) | shipped; live AUROC **0.8231**; no re-weighting beats best-single (see fusion pooling) | `baseline_results.json` `real.overall.auroc` |
| calibration | **Beta-MCA LANDED** — Brier −0.0939, ECE −0.2351, AUROC tie (0.7957); lever sweep + Venn-Abers are conclusive negatives | `calibration_results.json`; `calibration_levers_results.json` verdict `NEGATIVE`, `land_candidates: []` |
| operating point | conformal a disclosed **recall/coverage trade**, −0.071 F1 vs Youden/F1 (0.3982 − 0.4697) | `conformal_results.json` `overall` |
| fusion pooling | **KILL** — rel·clipped 0.8429 < best-single 0.8779 (gap −0.035); not shipped (I3) | `reliability_fusion_results.json` `gap_to_best_single`, `verdict` |
| robustness / survivability | measured floor 0.9243 → 0.7073 @ `m=k/2` + cubic-moment escape; research-only, **zero runtime change** | `survivability_results.json` `overall` |
| loaders / provenance | **read-only marine synthesis guard LANDED**; NSL-KDD schema single-sourced | `tests/research/test_governed_suite_provenance.py`; `tests/loaders/test_nsl_kdd_single_source.py` |
| reproducibility | reconstructed **7/7 byte-identical** across `PYTHONHASHSEED=1/2/random`; live 22/23 exact (marine drifts on live OBIS, disclosed); zero residual nondeterminism | `manifest.json` + `build_manifest.py` |
| latency / throughput | recorded per-dataset (`fit_ms`/`score_ms`), CPU-only numpy/scipy; **no latency lever pursued** (out of detection-quality scope — would be the micro-benchmark theatre the constraints forbid) | `benchmarks/mercury_benchmark_results.json` `per_dataset` |
| claim audit | every quantitative/capability claim reconciled to committed JSON or softened; README ratio generator fixed + regression-tested | README/docs truth-up commits |

`financial_loader`'s `pd.Timestamp.now()` is validated as **intentional real-time
fetch-window** code (90-day lookback ending "today"), not a reproducibility bug:
`financial` is unreachable (never in `suite.REACHABLE`) so it backs no committed
number; left unchanged, as instructed.

**Bottom line:** the provable improvements (Beta-MCA, the provenance guard, the
NSL-KDD single-sourcing, the claim truth-up) are landed and tested; every other
axis is a committed conclusive negative. No further reproducible improvement,
finetune, or balance remains for this PR that does not regress a landed number or
manufacture theatre. The one axis whose honest verdict is **open, not solved** —
fusion dilutes (pooling < best-single on the reachable suite) — is documented
immediately below with a kill-criteria'd experiment scoped to the **next** PR,
rather than papered over or shipped as a regression.

## Open problem — fusion dilutes (unsolved on the reachable suite)

Recorded so the next PR inherits a *measured* problem statement, not a vibe: on
the data we can actually reach, combining detectors is **worse** than picking the
single best one, and nothing in this PR changed that.

**The finding (committed numbers).** On the live suite (20 events, 50/50 cal/eval
split, weights from calibration only) **every** pooling scheme measured
underperforms the single best detector stream:

| pool | live AUROC | vs best-single |
|---|---:|---:|
| baseline `0.40/0.30/0.30` | 0.804 | −0.074 |
| reliability·linear | 0.840 | −0.038 |
| reliability·clipped | 0.843 | −0.035 |
| reliability·trimmed | 0.811 | −0.067 |
| **best single stream** | **0.878** | — |

(`results/reliability_fusion_results.json`: `overall.best_single = 0.8779`,
`gap_to_best_single = −0.0350`.) The 75-dataset third-party benchmark echoes it:
the weighted ensemble's **mean** AUC **0.8466** sits just under the best single
component, `info_geometry` **0.8504** (`benchmarks/mercury_benchmark_results.json`
→ `component_summary`).

**Why (measured, not assumed).** The components are too redundant on this suite.
Where the best stream already separates the anomaly, a log-odds pool of three
correlated streams regresses toward their average, and the bounded-influence
clip/trim caps the one informative stream instead of amplifying it. The damage is
concentrated, not diffuse: hurricane −0.125, fema −0.047, network_security −0.019.

**Not shipped (I3).** `core/robust_pooling.py` stays a tested, **default-off**
prototype; the runtime `detect()` path is byte-unchanged and keeps the baseline
ensemble. **We will not ship any fusion change that regresses best-single.**

**Measured, kill-criteria'd experiment for the NEXT PR.** The hypothesis is
*redundancy*, so the experiment measures it first and only then tries to beat
best-single on an *enlarged* pool:

1. **Diagnose redundancy.** Per event, compute the pairwise Spearman rank
   correlation of the three component score vectors; report mean `|ρ̄|`.
   *Pre-registered prediction:* `|ρ̄| ≳ 0.6` accounts for the dilution.
2. **Add one decorrelated stream** — candidate: a temporal/sequence detector over
   the same windows (a genuinely different inductive bias from the three static
   streams) — and re-measure `|ρ̄|` against the existing three.
3. **Learned stacking on the enlarged pool:** per-event 50/50 cal/eval, a logistic
   stacker fit on calibration scores only, AUROC on eval (the existing harness).
4. **Decision rule (paired, pre-registered):**
   - **SHIP** only if stacked AUROC beats best-single by a paired-bootstrap mean
     `Δ ≥ +0.01` with the 95 % CI lower bound `> 0` across the live suite, **and**
     no per-domain regression worse than `−0.01`.
   - **KILL** (abandon "fusion beats best-single" on this suite, record the
     negative with committed numbers exactly as Item 3 did) if, after adding the
     decorrelated stream, either `|ρ̄|` stays `≥ 0.5` **or** the stacked gap's CI
     upper bound is `< +0.01`.

Either outcome is a committed result, not a promise. Until that experiment runs,
the substrate ships the **best-single-preserving** path — baseline ensemble at
runtime, every re-weighting lever default-off — and this problem stays **open**.
