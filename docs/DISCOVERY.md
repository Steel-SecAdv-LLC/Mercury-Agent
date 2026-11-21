# Discovery: what a finding rests on, and how to check it

Mercury's flagship decision can carry three kinds of candidate finding: that an
observation is **novel**, that one measured quantity **leads** another, and a
**hypothesis** derived from those. This page is for the person who has to decide
whether to believe one.

The short version: every finding is delivered as an *artifact* — the claim, the
named procedure that produced it, and the numbers that procedure was given —
and you can re-run it yourself from the decision record, without Mercury's
server and without trusting it.

```bash
mercury-agent replay decision.json
```

```text
HOLDS          novelty: distance 39.3388 ranks at p=0.03448 among 28 reference distances (alpha=0.05): is_novel=True
HOLDS          causal: re-derived from the recorded sums of squares: 12 pairs, 1 edge(s) at alpha=0.05
HOLDS          hypothesis: all 3 premise(s) hold and the steps derive candidate_driven_regime
               claim: A candidate new regime in which X0 leads X1 at lag 1: unlike the reference, with a candidate driver.
```

## What "measured" means here

A number in a decision is *measured* only when the record carries the evidence
that proves it: how it was measured, on how many observations, and the SHA-256
of the numbers it was measured on. The flags `epistemic_measured`,
`aleatoric_measured`, `confidence_calibrated` and `novelty_measured` are
derived from that evidence. They cannot be set. A flag that says `true` with no
evidence beside it is read as `false` by the decision layer.

On the served path a single detection does not measure the spread of the model
(no ensemble is drawn) and no confidence calibrator has been fitted, so those
three flags are `false` and the record says so. The novelty verdict is measured
once the caller has shown enough flagged observations to rank a new one among.

## The three artifacts

Each sits in the decision record under `signals.cognitive.discovery`.

### Novelty

*Claim:* this observation is unlike the reference.

An observation is the feature-wise mean of the window you sent. Its distance
from the reference is the root-mean-square standardized deviation. It is
**novel** when that distance ranks in the top 5% of the distances the
reference observations themselves had: a rank p-value at or below 0.05. The
rule is distribution-free, so at most 5% of ordinary observations are called
novel by construction.

Measured on five recorded benchmark tables, three seeds each
(`artifacts/novelty_discovery.json`):

| Verdict rule | Ordinary rows called novel | Anomalies called novel |
|---|---|---|
| Rank test at 0.05 (current) | 3.9% (worst table 4.9%) | 73.1% |
| Score at or above 0.7 (replaced) | 14.5% | 88.2% |

The verdict is *undetermined* until there are 20 reference distances. An
undetermined verdict is never novel and carries no artifact.

The artifact carries the observation, the reference mean and standard
deviation, and the reference distances. It replays from the record alone. With
`--reference your_rows.csv` the question becomes whether the observation is
novel against a reference of your own; if it is not, the replay **fails**.

### Leads (causal)

*Claim:* within this window, X's past improves the prediction of Y beyond Y's
own past. This is predictive precedence (a Granger test). It is **not** proof
of mechanism.

Every ordered pair of the window's first five columns is tested. Three things
keep the test honest:

1. **The lags it searched.** It reports the lag with the smallest p-value, so
   that p-value is multiplied by the number of lags tried.
2. **The pairs it tested.** The pairs of one window are corrected together
   (Holm), so the chance of *any* false edge in one analysis is at most 5%.
3. **What it cannot read.** Time steps where any column is more than four
   robust deviations from its median are set aside. A window whose level is not
   stable across blocks is **not assessed**: it gets no edge, and the artifact
   says why.

Measured on generated windows whose truth is known, 200 per row
(`artifacts/causal_discovery.json`). No edge exists in any of these:

| Window | Analyses with a false edge, before | Now | Now: not assessed |
|---|---|---|---|
| Independent noise, 200 rows | 93.0% | 2.0% | 4.5% |
| Independent noise, 60 rows | 76.0% | 2.5% | 11.5% |
| Autocorrelated (AR(1) 0.7), independent columns | 94.5% | 1.0% | 76.0% |
| Common shock (9 sd, 6 rows) | 100.0% | 1.0% | 4.0% |
| Common shift (3 sd, 40 rows) | 100.0% | 0.0% | 100.0% |
| Common trend | 100.0% | 0.0% | 100.0% |
| Random walk | 100.0% | 0.0% | 100.0% |

And with one true edge (X0 drives X1 at lag 1, coefficient 0.5):

| Window | True edge found | Any other edge | Not assessed |
|---|---|---|---|
| One planted edge, 200 rows | 95.5% | 2.0% | 4.5% |
| One planted edge with a common shock, 200 rows | 96.0% | 3.0% | 4.0% |
| One planted edge, 100 rows | 70.0% | 0.5% | 23.5% |

**The limit, stated plainly.** The stability check declines most strongly
autocorrelated windows (76.0% of the AR(1) 0.7 windows above). That
costs coverage, never truth: a declined window yields no edge. Two alternatives
that decline nothing were measured and refused because they report false edges
(the artifact records both). A check that is valid under autocorrelation *and*
has power against a shift inside the window is the open problem.

The reported strength is the partial R-squared: the share of the target's
unexplained variance that the source's lags explain.

The artifact carries the residual sums of squares and degrees of freedom of
every pair tested, so each corrected p-value re-derives from the record alone.
With `--data window.csv` the sums of squares are recomputed from the window
itself; the file must be the same window (it is checked against the recorded
digest).

### Hypothesis

*Claim:* a named conclusion that follows from the verdicts above.

| Conclusion | Follows from |
|---|---|
| `candidate_new_regime` | flagged **and** novel |
| `candidate_driver` | flagged **and** a lagged dependence |
| `candidate_driven_regime` | both of the above |

A conclusion is a conjunction of measured verdicts. It is exactly as good as
its weakest premise and has **no probability of its own**: none was measured,
so none is reported. The artifact names the premise closest to failing, how
far it is from its cut, and the value at which the conclusion is withdrawn.

On replay, each premise is checked against the replay of its own artifact. If
the novelty verdict fails against your reference, the hypothesis is withdrawn
with it. If a premise cannot be replayed, the hypothesis is reported as
`CANNOT_REPLAY`, not as holding.

What is *not* claimed: that a derived conclusion is true of the world beyond
its premises holding. No measurement of that exists.

## Replaying

Three ways, same arithmetic:

| Where | How |
|---|---|
| Offline | `mercury-agent replay record.json [--data window.csv] [--reference rows.csv] [--json]` |
| API | `POST /api/v1/detect/flagship/replay` with `{"record": ..., "data": ..., "reference": ...}` |
| Console | The decision panel shows each artifact's basis, the weakest premise and the replay path |

`record.json` may be the decision record, a whole flagship response, or its
`cognitive_analysis`.

| Outcome | Meaning | CLI exit status |
|---|---|---|
| `HOLDS` | The claim reproduces. | 0 when every artifact holds |
| `FAILS` | It does not: the numbers changed, the artifact was altered, or the claim is false against what you supplied. | 1 |
| `CANNOT_REPLAY` | The artifact does not carry enough to decide, and says why. Never read as holding. | 2 |

Every artifact carries a digest over its claim, procedure and inputs. One that
does not match its own digest fails.

## Who sees what on a shared server

- A **signed-in caller** has a novelty reference of their own, built only from
  what they have sent, and gets it back in the artifact.
- **Callers without credentials** share one reference. It is built from each
  other's data, so it is not handed to any of them: their novelty artifact
  carries digests in place of the vectors and says it cannot be replayed. Their
  causal artifact is computed from their own window and still replays.
- The server's **decision ledger** keeps each artifact's claim and digest, not
  its inputs. An artifact presented later can be matched to its digest.

## Where the code is

| Piece | Location |
|---|---|
| Evidence type | `omni_mercury_engine.measurement` |
| Procedures, artifacts, replay | `omni_mercury_engine.discovery` |
| Harnesses | `research/cognitive/harness_novelty_discovery.py`, `research/cognitive/harness_causal_discovery.py` |
| Decisions and their bars | `artifacts/improvement_ledger.json` |
