<!--
Copyright (C) 2025 Steel Security Advisors LLC
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Mercury Agent — Capability Matrix

Applies to Mercury Agent **v2.1.x**. Last reviewed: 2026-08-02.

One row per claim: **claim · dataset/task · metric · number · repro command ·
status**. If a capability is not in this table, Mercury does not claim it.

This file exists because pillars and capabilities are different kinds of thing
and were being written up as if they were the same. A **pillar** is a property
that either holds or does not, and it lives in `tests/pillars/` where a test
observes it. A **capability** is a *measurement* — it drifts with the data, the
seed and the checkpoint, so it belongs in a table with a repro command and a
date, not in a test asserting a number. Nothing in `tests/pillars/` asserts a
benchmark figure, and nothing here is called a pillar.

`scripts/doc_lint.py` fails CI if a row here says **enforced** without naming a
path that exists in this repository.

## Status vocabulary

| Status | Means |
|---|---|
| **enforced** | A control that runs and refuses. A named test observes it; deleting the control fails CI. |
| **measured** | A number produced by a repro command on a stated dataset. Re-measure before quoting it. |
| **advisory** | Computed and reported; it decides nothing. |
| **untrained** | Code exists, weights are at initialisation. Not a learned model; no accuracy attaches. |
| **aspirational** | Not built. Listed so it is not mistaken for shipped. |
| **removed** | Previously claimed, now deleted, with the reason. |

---

## 1. Structural facts (CI-gated counts)

| Claim | Dataset / task | Metric | Number | Repro command | Status |
|---|---|---|---|---|---|
| `torch.nn.Module` subclasses | this source tree | AST count | **173** | `python scripts/measure_codebase_scale.py` | **measured** (CI-gated in README "Codebase Scale") |
| Data-loader classes in `loaders/` | this source tree | regex `class *Loader` | **21** | `python scripts/measure_codebase_scale.py` | **measured** — 20 concrete loaders + `BaseDomainLoader`, which the CI regex also matches. The previously published "16" was stale. |
| Detector classes under `detectors/` | this source tree | AST count | **88** | `python scripts/measure_codebase_scale.py` | **measured** |
| Source files / LOC | this source tree | file + line count | **775 / 406,960** | `python scripts/measure_codebase_scale.py` | **measured** |
| Cognitive components wired at runtime | `CognitiveOrchestrator` | count | **ten** | `python -c "from omni_mercury_engine.cognitive.orchestrator import CognitiveOrchestrator"` | **measured** — the "nine components" figure was stale; the "7-phase" spine is *historical build order*, not a runtime pipeline |

## 2. Safety controls

| Claim | Dataset / task | Metric | Number | Repro command | Status |
|---|---|---|---|---|---|
| Harm-uplift gate refuses weapons/mass-casualty uplift at every public decision surface | red-team corpus in `tests/pillars/test_non_maleficence.py` | refusal on each surface | 4/4 requests × 6 surfaces | `pytest tests/pillars/test_non_maleficence.py` | **enforced** — `src/omni_mercury_engine/cognitive/decision_gate.py`; the `GATED_BOUNDARY` capability contract makes deleting it a CI failure |
| Benign input is not refused for having plain vocabulary | benign corpus incl. real injection/XSS payloads and clinical-toxicology queries | permit rate | 7/7 | `pytest tests/pillars/test_non_maleficence.py -k benign` | **enforced** — `tests/pillars/test_non_maleficence.py` |
| Identical decision → identical verdict on every surface | red-team + benign corpus × 7 surfaces | verdict agreement | 11/11 unanimous | `pytest tests/pillars/test_non_maleficence.py -k surface` | **enforced** — `tests/pillars/test_non_maleficence.py` |
| No response the loop can recommend is destructive | all dispositions × 10 severities × 6 domains × 2 postures | destructive-verb hits | **0 / 960** | `pytest tests/pillars/test_non_maleficence.py -k destructive` | **enforced** — `src/omni_mercury_engine/decision/response.py` |
| Gate fails closed when it cannot be evaluated | fault injection into `assess_weapons_uplift` | refusals | 100 % | `pytest tests/pillars -k fail_closed` | **enforced** — `src/omni_mercury_engine/cognitive/decision_gate.py` |
| σ_Immutable configuration-integrity gate | 256-D governance scalar vector | decision threshold | 0.93 | `pytest tests/security/test_sigma_immutable_gate_semantics.py` | **enforced** — `src/omni_mercury_engine/security/sigma_immutable_gate.py` |
| Benevolence score | any action text | float in [0,1] | n/a | `pytest tests/cognitive/test_ethical_bounding.py` | **advisory** — reported and logged; it approves nothing and refuses nothing. The former `≥ 0.99` pass-bar is **removed** (see §6). |
| Decision records are immutable and the ledger is append-only | synthetic decision stream | mutation attempts refused | 100 % | `pytest tests/pillars/test_control.py` | **enforced** — `src/omni_mercury_engine/decision/{record,ledger}.py` |
| Tripwire halt is irreversible; capability ceiling cannot be self-raised | governor fault injection | escapes found | **0** | `pytest tests/pillars/test_corrigibility.py` | **enforced** — `src/omni_mercury_engine/agentic/subagents/governor.py` |
| No autonomous change to a live boundary | threshold-move + recalibration proposals | authorised autonomously | **0** | `pytest tests/pillars/test_corrigibility.py` | **enforced** — `src/omni_mercury_engine/governance/self_improvement.py` |

## 3. Cryptography

| Claim | Dataset / task | Metric | Number | Repro command | Status |
|---|---|---|---|---|---|
| ML-KEM-1024 / ML-DSA-65 / SLH-DSA implement FIPS 203/204/205 | NIST ACVP-Server KAT vectors | bit-exact match | pass | `pytest tests/security/test_nist_fips_kat.py` | **enforced** — `src/omni_mercury_engine/security/pqc_backends.py` |
| Formal cryptographic validation | CAVP / CMVP | — | — | — | **aspirational** — not entered. Internally reviewed (AI-assisted); not independently audited. |
| Constant-time primitive behaviour | AMA native C primitives | — | asserted | — | **advisory** — asserted by the implementation, not independently verified |
| σ_Immutable corpus integrity | `sigma_immutable_corpus.json` | Ed25519 + ML-DSA-65 signature verify | pass | `pytest tests/security/ -k corpus` | **enforced**, but **tamper-evident, not authenticated** — the verifying public key is carried *inside the same signed payload* (`security/sigma_immutable_corpus.py`), so an attacker who can rewrite the corpus can re-sign it. It detects corruption and accidental drift; it does not prove origin. Do not describe it as "signed for authenticity". |
| 6-layer crypto package — integrity | emitted `CryptoPackageResult` | `core_valid` (Layers 1–4) | pass | `pytest tests/security/test_crypto_api.py -k SixLayerVerify` | **enforced** — `security/crypto_api.py`. Self-consistency only: every key needed to check a package travels *inside* it, so this proves the parts agree and were not corrupted, not who produced them. Same distinction as the σ_Immutable row above. |
| 6-layer crypto package — authenticity | emitted `CryptoPackageResult` + out-of-band signing key | `all_valid` (requires `key_pinned`) | pass **only when a key is pinned** | `pytest tests/security/test_crypto_api.py -k pinned` | **enforced** — requires the caller to pass `expected_public_key`. Under AMA 4.0 an *unanchored* verify returns `all_valid` False even for a wholly intact package; through AMA 3.x it returned True, reporting success for a check that could not tell the expected signer from an attacker who built their own package. There is no flag to restore the old aggregate and Mercury adds none. |

## 4. Detection models

Numbers below come from each checkpoint's shipped `*.provenance.json`, written
by the merit gate that admitted it. Ten checkpoints ship; nine carry provenance
(`default_fusion.pt` is the packaged default and does not).

| Claim | Dataset / task | Metric | Number | Repro command | Status |
|---|---|---|---|---|---|
| Seismic event detection | STEAD | AUC | **0.9949** | `python -c "import json;print(json.load(open('src/omni_mercury_engine/models/checkpoints/seismic_stead.provenance.json'))['evaluation']['learned'])"` | **measured** (merit-gated) |
| Volcanic eruption precursor | AVO seismic | AUC | **0.9402** | same pattern, `volcanic_avo_seismic.provenance.json` | **measured** (merit-gated) |
| Hurricane detection | ERA5 patches | detection AUC | **0.9882** | same pattern, `hurricane_era5.provenance.json` | **measured** (merit-gated; category accuracy is **0.608** — quote both) |
| Wildfire ignition | FIRMS | AUC | **0.8750** | same pattern, `wildfire_firms.provenance.json` | **measured** (merit-gated) |
| Landslide | COOLR | AUC | **0.8498** | same pattern, `landslide_coolr.provenance.json` | **measured** (merit-gated) |
| Tornado | NEXRAD | AUC | **0.8099** | same pattern, `tornado_nexrad.provenance.json` | **measured** (merit-gated) |
| Solar storm | GEOMAG | G-bucket accuracy | **0.9598** | same pattern, `solar_storm_geomag.provenance.json` | **measured** (merit-gated) |
| Regularity deviation | GCP | fault AUC | **0.7875** | same pattern, `reg_deviation_gcp.provenance.json` | **measured** (merit-gated) |
| GOSNN attention fusion | harvested fused states (n=450) | val MSE vs reference | learned 3.88e6 vs reference 8.30e6 | same pattern, `gosnn_attention_fusion.provenance.json` | **measured** — an **observability** head (MSE), *not* a detection head. It ships no detection metric, so the decision layer's GOSNN disagreement overlay is inert on this build. |
| PatchCore visual anomaly detection | — | — | — | `pytest tests/detectors -k patchcore` | **measured** — implementation at `src/omni_mercury_engine/detectors/visual/patchcore.py`; no benchmark number is published here because none was re-measured in this change |
| Math-Arrest equation family | — | probe count | **21** | `ls src/omni_mercury_engine/detectors/math_arrest/probes/*.py` | **measured** |
| Statistical detector | — | — | — | `pytest tests/test_detectors.py` | **measured** — real, fitted, shipped by default |

### Not re-measured in this change

These two figures have circulated and are **not published as current** here,
because they were not re-measured in this change and a stale benchmark number
presented as current is the failure mode this table exists to prevent. Run the
command, then fill in the number and the date.

| Claim | Dataset / task | Metric | Number | Repro command | Status |
|---|---|---|---|---|---|
| `OmniFusionModel` fusion quality ≈ 0.96 | ADBench subset | AUC | **not re-measured** | `python benchmarks/competitive_benchmark.py` (downloads ADBench; run on a machine with network) | **aspirational** until re-measured |
| Multi-agent consensus ≈ 0.88 / member ≈ 0.84 | ADBench, 5 datasets × 3 seeds | consensus AUC / mean member AUC | **not re-measured** (last recorded run: `artifacts/orchestration_validation.json`, mean consensus 0.841, mean member 0.833) | `python benchmarks/orchestration_validation.py` | **aspirational** until re-measured |

## 5. Untrained — relabelled, not benchmarked

Each of these ships `nn.Module` subclasses that **no training script fits and
no checkpoint restores**. They run at initialisation weights. They are not
learned models and no accuracy figure attaches to them. Every one fails closed
(`DetectorException`) until `fit()` has computed its statistical reference, so
an unfitted detector abstains rather than emitting a score.

| Claim | What actually runs | Repro command | Status |
|---|---|---|---|
| UI/UX interaction anomaly detection | Statistical only. The four networks are constructed, put in `eval()`, and **never called** by the scoring path. | `pytest tests/test_uiux_anomaly.py` | **untrained** — `detectors/uiux_anomaly.py` |
| Spectral vibration analysis (GNN/CNN/MLIP) | The networks *are* called, at init weights — a deterministic **random projection**. `fit()` builds its reference through the same projection, so the comparison is like-for-like; the discriminative power is the statistical reference, not the networks. `detect()` reports `neural_backbone="untrained_random_projection"`. | `pytest tests/test_spectral_vibration.py` | **untrained** — `detectors/spectral_vibration.py` |
| EMP-attack, fraud, pathogen (QBM), interstellar-object and pandemic-transmission models | Deterministic/statistical scoring; the neural components are unfitted. | `pytest tests/medical tests/space` | **untrained** |
| Multivariate time-series LTG detector | Per-window mean / standard deviation / feature-correlation summary. No LSTM, no convolution kernels, no graph. | `pytest tests/test_multivariate_timeseries.py` | **untrained** — `core/multivariate_timeseries.py` |
| `MercuryReasoner` "ReAct / chain-of-thought" | ReAct's *control flow* with templated thoughts and positional action selection. The trace and correlation graph are real; the reasoning is not. Actual NL reasoning is `omni_mercury_engine.reasoning` (operator-supplied backend). | `pytest tests/test_mercury_a_agent.py` | **advisory** — `agentic/mercury_a_agent.py` |
| Effective reproduction number (pandemic) | Wallinga–Lipsitch `R = exp(r·T_s)` under a **delta** serial interval — an upper bound, and **R_e, not R₀**: nothing observes susceptibility or immunity. Reported as `re_estimate`; `r0_estimate` is a retained back-compatible alias for the same number. | `pytest tests/medical` | **advisory** |
| SOFA / qSOFA | Deterministic instruments computed exactly per Vincent et al. (1996); `sofa_is_lower_bound` marks partial input. | `pytest tests/medical` | **enforced** (deterministic instrument) — `medical/critical_care/sepsis_detector.py`. **No clinical validation**: no accuracy claim is made, and none may be made without validation on real patient data. |

## 6. Removed claims

| Claim | Why it was removed |
|---|---|
| "Benevolence ≥ 0.99 hard gate at every decision boundary" | The score was computed over a **fixed string the engine wrote for itself** (`"anomaly_detection:{domain}:audit verify protect research evidence…"`), so the caller's request never reached it. As a bar it rejected benign work for having plain vocabulary and admitted anything phrased positively. Replaced by the harm-uplift gate scored on the real decision. |
| `multivariate_timeseries` `roc_auc_estimate` | Computed as `0.5 + 0.4·tanh(separation)` from the detector's own scores and its own thresholded predictions. No label was ever involved, so it was not an AUC of anything; it rose whenever the detector was self-consistent. |
| `MultivariateTSDetector` "LSTM / temporal convolution / graph convolution" | The three branches are a mean, a standard deviation and a correlation summary. Renamed and documented rather than deleted, since the statistical baseline is real. |
| "R₀ estimation" | The quantity is an effective reproduction number estimated from observed growth. Calling it R₀ overstates it whenever the population is partly immune — i.e. during any surge. |
| σ_Immutable corpus "signed for authenticity" | The verifying key travels inside the signed payload. Tamper-evident, not authenticated. |
| "16 live data-loader classes" | The real count is 21 by the CI regex (20 concrete + the base class). |
| "nine cognitive components" | Ten are wired at runtime. |
| Lyapunov stability "guarantee" | A design-time convergence proof plus **runtime monitoring**. `LyapunovRuntimeEnforcer` defaults to `halt_on_violation=False`, so at runtime it observes and records; it does not halt unless an operator constructs it with `halt_on_violation=True`. See `core/system_coherence.py`. |
| 85 % coverage `fail_under` in `pyproject.toml` | No lane ever ran at 85: CI passes `--cov-fail-under` from `COVERAGE_THRESHOLD_CORE/FULL` (33 / 62), which overrides it. The config key now matches the enforced floor; 85 survives as a labelled aspiration in prose. |

## 7. Known false claims about *this* repository, corrected

* **There is no `np.math.factorial` bug in `harmonics/transform.py`.** The code
  already calls stdlib `math.factorial` (lines 121–124), with a comment noting
  that `np.math` was removed in NumPy 2.0. Anyone "fixing" this would be
  changing working code.
* **Coverage is 33 % (core) / 62 % (full), enforced.** Not 85, not 95.
* **Mercury ships no generative language-model weights.** LLM adapters exist
  (`models/foundation/llm_adapter.py`, `ollama_adapter.py`,
  `reasoning/backend.py`) so an operator can attach one; none is bundled. The
  broader claim "Mercury never produces fabricated prose" is false as stated and
  is not made — see `tests/pillars/test_evidence.py`.
