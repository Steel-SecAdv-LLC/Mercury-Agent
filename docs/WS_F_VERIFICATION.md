# WS-F — Whole-system verification

## Suite + static analysis

* **flake8:** clean across all 13 touched files.
* **mypy:** clean across all 8 touched source modules (`engine.py`,
  `domain_encoders.py`, `schumann_labeling.py`, `gcp_ingest.py`, the four
  benchmark harnesses).
* **New tests:** 35 passed, 1 network-gated skip (WS-A: 4 + the live guard;
  WS-B: 16 encoder + 5 fusion-wiring; WS-C: 5 labeling; WS-D: 5 ingestion).
* **Regression-critical existing suite:** the 32 pre-existing fusion tests
  (`test_fusion_symbolic_cotraining`, `test_fusion_training`,
  `test_fusion_raw_path`) still pass — the WS-B wiring did not disturb the
  default path.

## Dependency reconciliation (full activation)

`pyproject.toml` is the manifest (no lockfile). The complete `[ml]` extra is
active and import-clean: torch 2.12, torchvision 0.27, pytorch-lightning 2.6,
timm 1.0, opencv 4.13; core scipy 1.17, pandas 3.0, numpy 2.4, pydantic 2.13;
`ama_cryptography` 3.2 (primary PQC backend). **No conflicts.**

**sklearn is intentionally absent.** Mercury's own `mercury_ml` supplies
`roc_auc_score` / `f1_score` / `StandardScaler`, so every artifact here is
computed without sklearn. Per project direction, scikit-learn is pulled in only
for explicit benchmark-*comparison* lanes, never as a runtime dependency; the
sklearn-gated tests remain `importorskip` + `network`/`slow` and skip cleanly.

## Off-path parity (byte-identical contract)

The baseline `fit_fusion` is **not** bit-deterministic under a fixed seed (two
identical default fits differ by ~1e-15 from non-associative float reduction —
pre-existing). The strongest true guarantee, asserted in tests:

* `symbolic_weight=0` → no symbolic state (existing).
* `domain_encoder=False` → no encoder module / feature / params, and served
  scores match the default within ~1e-15 (`atol=1e-6` test).
* `MercuryAnomalyDetector` is byte-identical across repeats (WS-A guard).

## Determinism + provenance

Every result is reproducible from a recorded (dataset, seed, metric, commit):
ADBench is MIT with per-set SHA-256 in `anomaly_regression_baseline.json`; NOAA
(public domain) and GCP provenance carry URL + content hash + fetch time; all
ablations pin seeds 0/1/2.

## Global-impact paragraphs (mission lens: life-safety, free, STEM)

**WS-A.** Catching that the "regression" was the #255 honesty de-leak — not a
detector regression — protects the integrity of every downstream life-safety
decision: a system that quietly re-inflates its headline with circular labels
would mislead operators about its real reliability. The deterministic per-dataset
guard means a *genuine* future regression now fails CI loudly and freely
(no sklearn, no paid service), so the free/open detector stays trustworthy.

**WS-B.** Differentiable domain encoders are real, reusable STEM machinery
(learnable FFT / finite-difference / Fisher operators) now available opt-in — but
the honest ablation kept them **off by default** because they don't yet beat the
static extractor on real labels. Globally this preserves whole-system
reliability: no unproven capacity is silently added to the path that flags
anomalies in a crisis.

**WS-C.** Refuting "no labels possible" and building real weak supervision from
public-domain space-weather catalogs turns a dead sub-net into a *teachable*,
reproducible pipeline — exactly the STEM-discovery mission — while the honest
quarantine (synthetic signal + unstable training) prevents a half-trained ELF
detector from ever driving a life-safety alert.

**WS-D.** Treating the GCP archive as a pre-registered signal-processing null
test models scientific integrity for learners: it shows how to engage a
controversial dataset *without* overclaiming. The faithful null and the
multiple-comparison correction (seed-0's uncorrected p=0.038 → non-significant)
are the teachable result; the module stays quarantined and never asserts psi.

**WS-E/F.** The coverage table + verification make the whole neuro-symbolic
stack auditable: every neural submodule has a dataset/seed/metric/artifact/tests
and an explicit active-or-quarantined status. For a free, life-safety system
this auditability *is* the product — operators and contributors can see exactly
what is trusted by default and what is held back, and why.
