# Mercury Omni-Equation — Build Prompt (measurement-grounded)
Applies to Mercury Agent **v2.1.x**. Last updated: 2026-07-11.

**Context.** You are working in `Mercury-Agent`. The North Star is one
ethics-gated, multi-stream, validated scoring law (the "omni-equation") that
**strengthens — never replaces** — the current Omni-Ava Equation. The earlier
direction assumed "add streams + fuse." **That assumption was measured and
refuted** (`FINDINGS.md`): on 15 real events / 5 domains, Mercury's multi-stream
ensemble (AUC **0.836**) is *worse* than its own best single stream (**0.909**);
streams are redundant (|corr| **0.66**); the outer equation is inert (**−0.002**);
the η^Φ gate is inert for ranking (**0.003** flip). Ranking is good while F1 ≈
**0.09** — calibration, not breadth, is the bottleneck.

**Order of work is fixed: fix fusion and calibration on existing streams FIRST;
add breadth only after, gated. Do not add a new stream or modality until Gates
1–2 pass.**

## Hard constraints (non-negotiable)
- **Ethics `η^p` stays the locked, outermost invariant.** Re-attach it as the
  outermost multiply on the fused score; prove with an adversarial test that no
  config path disables it. It is governance, not an accuracy lever — do not tune
  it for scores. If it must influence decisions, make it per-action benevolence,
  not a global scalar.
- **Strengthen, don't replace.** Implement as a NEW `RuntimeEquationProfile`
  (`core/equation_profiles.py`, freeze-and-add). It MUST reduce to
  `baseline_original_v1` exactly when streams = 1.
- **No value surrendered without a measured, equal-or-better trade — both
  directions.** Every added stream or unit of compute must show a measured
  accuracy or cost payback on the real domains, or it is reverted.
- **No synthetic-data claims.** Measure on the real live-API domains
  (`harness_multidomain.py`); state real-vs-unreachable per domain.

## Gate 1 — Reliability-weighted fusion (accuracy, zero added cost) — DO FIRST
Replace the fixed-weight ensemble with a per-sample reliability-weighted
combiner: weight each stream by its calibrated reliability, down-weight redundant
and low-variance streams (cf. the FINDOYOU `mercury_fusion` Stouffer-over-
conformal-p-values pattern; reliability = label coverage / variance / agreement).
- **Target:** fused AUC ≥ best-single across the 5 real domains (close the −0.074
  gap; 0.836 → ≥0.90).
- **Kill:** if it cannot reach best-single, the fusion thesis is dead — stop and
  report the numbers.

## Gate 2 — Calibration → usable decisions (operational accuracy)
Add a calibrated operating point (conformal threshold / per-domain calibration)
so ranking converts into detections.
- **Target:** materially raise F1/precision at a fixed, stated FPR vs today's
  ~0.09, without losing AUC.
- **Kill:** if F1 does not improve at controlled FPR, calibration isn't the lever
  — report and stop.

## Gate 3 — Cascade (speed / cost)
Structure evaluation as a cascade: cheap streams clear confident cases; expensive
streams (LLM / predictive) fire only on the ambiguous remainder; abstain when
underpowered.
- **Target:** per-verdict latency and $ ≤ today at equal-or-better accuracy.
- **Kill:** if always-on multi-stream cannot be made ≤ current cost, the breadth
  roadmap is cost-negative — report.

## Only after Gates 1–3 pass — add breadth (the multi-role vision)
Mercury is predictive + LLM-augmented and multi-domain. Add streams under one
**typed interface** (calibrated bits + attribution + rationale + provenance),
each admitted ONLY if it shows positive **conditional lift** under reliability-
weighting on the real domains (it must beat the diluting baseline). Priority by
measured value, not story: (a) predictive/temporal streams already in Mercury
(timeseries, drift); (b) the **LLM reasoning stream** (calibrated to bits,
cascade-gated for cost); (c) multimodal (vision/signal).

## Deliverable per gate
The change as a new profile/module + harness numbers **before/after** on the 5
real domains + a one-line trade statement (what was paid, what was gained).
Update `FINDINGS.md`.

## Definition of done (shippable)
At least one of {Gate 1 accuracy, Gate 2 F1, Gate 3 cost} shows a **measured**
win on real data, with ethics provably intact and exact reduction-to-baseline. If
none do, the transparent output is **"do not ship,"** with the numbers.
