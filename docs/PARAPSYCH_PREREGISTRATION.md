# WS-D — Parapsychology (GCP) sub-net: strict pre-registration

Applies to Mercury Agent **v2.1.x**. Last updated: 2026-07-11.

Registered **before** any analysis. The Global Consciousness Project dataset's
documented failure mode is **post-hoc analytic flexibility** (choosing the
statistic/window after seeing the data). This pre-registration fixes every
analytic degree of freedom in advance and is committed in the same change as the
code. **No claim is made that "psi" is real.** A faithful null is the expected,
scientifically valid outcome; the data decides.

## Treatment

Pure signal-processing / anomaly detection on synchronised hardware-RNG streams.
Not a test of consciousness; a test of whether the streams deviate from the
fair-coin null around a fixed, independently-defined event catalog.

## Data

* **Source:** GCP archive — per-second per-egg 200-bit sums (~65 nodes).
  Documented raw-stream endpoint: `noosphere.princeton.edu` `eggdatareq`.
* **Reachability (this environment):** the raw-stream host is **unreachable**
  (and not on the trusted allowlist); `fetch_egg_stream()` reports
  `reachable=False` transparently. The reachable mirror `global-mind.org/data/` serves
  only aggregate **daily HTML summaries**, not the raw streams the encoder needs.
* **Fallback:** a clearly-labelled **synthetic true-random** generator
  (`Binomial(200, 0.5)`) validates the statistics/encoder plumbing under a known
  null. Synthetic data **cannot** lift quarantine and is never presented as real.

## Fixed event catalog (independent source)

A small, fixed set of globally-attended events with fixed UTC windows, chosen
from an independent public record **before** analysis (not data-driven):

| Event | UTC window |
|---|---|
| New Year midnight (global) | each year, ±10 min around local midnights — canonical GCP formal event |
| Major earthquake (M≥8, USGS) | quake time → +3 h |
| Total solar eclipse totality | totality window |

(For a real run these resolve to concrete timestamps from USGS / eclipse
catalogs; fixed here so windows cannot be chosen post-hoc.)

## Fixed statistics (no alternatives evaluated)

* **Per-second network variance** = Σ z² over eggs, df = egg count (chi-square
  under null).
* **Cumulative Stouffer Z** over each event window = Σz / √N, ~N(0,1) under null.
* **Differentiable encoder:** `ConsciousnessFieldAnalyzer` (LSTM + attention)
  coherence score over the stream — reported alongside, not in place of, the
  closed-form statistic.

## Fixed parameters

* Seed: 0 (and 1, 2 for the synthetic-null distribution). Window: 300 s.
* Eggs: all available that second (NaN-skipped).

## A-priori bar (set before running)

* **Lift quarantine only if**, on **real** GCP data, the combined Stouffer Z
  across the fixed catalog exceeds a **Bonferroni-corrected** |Z| threshold for
  the number of events, AND the effect is stable across the fixed seeds.
* **Expected outcome: null.** Report it plainly either way. Synthetic-null
  results are plumbing validation only.

## Verdict rule

Record the numbers. `QUARANTINE` unless the real-data, multiple-comparison-
corrected bar is cleared. Never assert psi; a clean null is a valid contribution.

---

## Harvest (this round): the scaffolding transfers to a real mission problem

The psi null stands and is closed — re-hunting it would be motivated reasoning,
not engineering. But the *scaffolding* built here is real, reusable
scientific-integrity infrastructure: transparent ingestion with explicit
reachability, a **pre-registration** that fixes the analysis before the data is
seen, a **null test**, and a **multiple-comparison correction**. That is exactly
what a free, life-safety anomaly system needs whenever it asks *"do my detector's
flags coincide with real events more than chance?"* — a question endemic to
Mercury's space-weather / seismic / environmental-hazard mission, and the classic
home of post-hoc-flexibility self-deception.

**Generalized into a reusable tool** —
`src/omni_mercury_engine/evaluation/event_coincidence.py`:

* `PreregisteredCoincidenceTest` — fix statistic / permutation count / alpha /
  correction up front (commit it with the analysis);
* `permutation_coincidence_test` — a **circular time-shift permutation** null
  (the gold standard for autocorrelated streams: it preserves the score's own
  temporal structure and destroys only its alignment to the fixed event windows,
  so a significant result cannot be an autocorrelation artifact);
* `bonferroni` / `benjamini_hochberg` — multiple-comparison control.

It is validated as a general tool (`tests/evaluation/test_event_coincidence.py`,
8 tests): correct false-positive rate under a true null — including for
*autocorrelated* random walks, the case the circular null is designed for — and
real power against a planted signal.

**Applied to a real, non-circular, in-scope problem** —
`benchmarks/spaceweather_coincidence.py`:

> Is the geomagnetic **Kp** index elevated inside independent **GOES** solar-flare
> response windows beyond chance?

Kp (a geomagnetic instrument) and GOES flares (an X-ray instrument) are
physically coupled but **independently measured**, so the test is non-circular by
construction (no label leak — the WS-A failure mode) and a positive result would
be real. Both feeds are NOAA SWPC public domain (reused from
`space/schumann_labeling.py`); a 6 h post-flare geomagnetic-response window is
fixed a-priori. Live result (`artifacts/spaceweather_coincidence.json`):
reachable, 2 Kp samples in-window, observed Kp elevation +0.32, **p = 0.34 →
faithful NULL** (a quiet-week M-class flare cluster shows no significant
geomagnetic coincidence) — reported plainly, no overclaim, with the synthetic
positive-control + null confirming the machinery. The same tool now stands ready
for any future weak-signal coincidence question in scope.
