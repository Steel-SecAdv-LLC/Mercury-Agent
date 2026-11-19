# Per hazard: what a score means, whether the certificate holds, how often nothing is decided

Measured 2026-11-18 by `research/hazard_calibration/measure.py`; the
numbers are `research/hazard_calibration/evidence.json`, and the package ships
the same file for the decision layer and the API to read.

## Why this exists

The hazard table in the README says how well the detector *ranks* the rows of a
real event. Three things it does not say, and a deployment needs:

1. **Is the score a probability?** When it says 0.8, is that four times in five?
2. **Does the certificate hold?** A conformal set asked for 90% coverage should
   hold the true label nine times in ten, on an event it has not seen.
3. **How often is nothing decided?** The decision layer abstains when the
   certificate cannot tell. How much of an event is that, and what becomes of
   the real anomalies?

## How it was measured

On the same real events as the hazard table, fetched from their live sources,
**leave one event out**. For each event of a hazard in turn, everything fitted
is fitted on the hazard's *other* events:

- the detector the hazard table uses (`MercuryAnomalyDetector`, fitted
  unlabelled on the event itself) scores every row;
- the other events' rows are split in two, every other row of each class: an
  isotonic map from score to probability is fitted on one half, and a
  class-conditional conformal classifier at 90% coverage on the other;
- the held-out event's rows get a probability and a label set, and the decision
  layer, as it ships, decides or abstains on each;
- then the halves change places and the event is judged again. Both judgings
  are in every figure below, and the certificate's coverage is given for each
  beside the two together: how far they differ is what the choice of half
  alone is worth. A certificate is said to have held, or to have fallen short,
  only where both agree.

An event with more than 20,000 rows is read at 20,000 evenly spaced rows.
Nothing is random: the same events give the same numbers.

## What was found

| Hazard | Labels | Events | Rows | Anomalies | Skill | Coverage, target 90% (each half) | Abstained | Decided right | Anomaly calls right |
|---|---|---|---|---|---|---|---|---|---|
| earthquake | statistical | 5 | 2,867 | 7 | -0.014 | 98.9% (97.8% / 99.9%) | 83.0% | 93.3% | 0.0% of 63 |
| energy | statistical | 5 | 6,064 | 328 | -0.011 | 88.7% (91.3% / 86.0%) | 63.3% | 69.1% | 19.8% of 1,667 |
| fema | statistical | 5 | 2,228 | 431 | -0.062 | 95.2% (95.3% / 95.1%) | 92.3% | 37.6% | 14.0% of 172 |
| hurricane | statistical | 4 | 294 | 29 | +0.030 | 93.5% (91.2% / 95.9%) | 77.0% | 71.9% | 36.8% of 57 |
| landslide | statistical | 1 | | | not measured: one event: there is no other to calibrate on | | | | |
| marine | statistical | 3 | 443 | 47 | +0.152 | 96.6% (97.7% / 95.5%) | 93.5% | 48.3% | 57.8% of 45 |
| network_security | ground_truth | 2 | 32,938 | 9,721 | -0.044 | 70.1% (70.5% / 69.7%) | 43.2% | 47.4% | 38.6% of 7,915 |
| pandemic | statistical | 5 | 22,965 | 950 | -0.535 | 85.2% (85.1% / 85.2%) | 17.2% | 82.1% | 15.4% of 7,888 |
| sepsis | ground_truth | 2 | 38,890 | 674 | +0.003 | 91.5% (91.9% / 91.0%) | 67.9% | 73.4% | 4.3% of 6,728 |
| tornado | statistical | 2 | 414 | 40 | +0.132 | 90.0% (89.4% / 90.6%) | 10.3% | 90.2% | 48.6% of 37 |
| tsunami | statistical | 3 | 1,255 | 103 | +0.183 | 89.1% (87.6% / 90.6%) | 37.0% | 82.7% | 30.8% of 386 |

- **Skill** compares the calibrated probability with a reference that knows only
  the other events' share of anomalies (`1 - Brier / reference Brier`). Above 0
  the probability says more than that base rate; below 0, less. A large value
  can also mean only that the other events' base rate was far from this one's.
- **Coverage** is how often the true label was in the set.
- **Decided right** counts every decided row, and most rows are normal: where
  anomalies are few it says little. **Anomaly calls right** is how many of the
  rows called "anomaly" were; its count is over both judgings.

Read plainly:

- **Is the score a probability?** Calibrated on the hazard's other events, it says clearly more than their base rate for marine, tornado and tsunami; about as much for earthquake, energy, hurricane and sepsis; and less for fema, network security and pandemic.
- **Does the certificate hold?** Asked for 90%, on an event it had not seen, it held whichever half it was fitted on for earthquake, fema, hurricane, marine and sepsis (5 of 10); fell short either way for network security and pandemic; and was over on one half and under on the other for energy, tornado and tsunami.
- **How often is nothing decided?** From 10.3% of rows (tornado) to 93.5% (marine).
- **Network security, whose labels are independent of what is scored:** skill -0.044, coverage 70.1%; of its true anomalies 15.7% were called, 7.9% abstained on and 76.4% called normal; 38.6% of 7,915 anomaly calls were right.
- **Sepsis, whose labels are independent of what is scored:** skill +0.003, coverage 91.5%; of its true anomalies 21.3% were called, 63.8% abstained on and 14.9% called normal; 4.3% of 6,728 anomaly calls were right.

That is the detector's ranking not carrying from one event to another as a
calibrated statement, which is what it was not built to do: it is fitted
unlabelled on each event. A deployment that wants probabilities or
certificates for a hazard has to calibrate on that hazard's own labelled past
and check them this way; for half the hazards here that check does not pass,
and where it passes most rows are abstained on.

## What this is not

- **Not the labels' independence.** `statistical` labels are a threshold on a
  column; a detector shown that column, or one that moves with it, is partly
  marking its own work (see the README's hazard table). Only network security
  and sepsis have labels the loaders declare independent.
- **Not every hazard.** Landslide has one event with both classes and cannot be
  measured this way. Financial, flood, volcanic and wildfire could not be
  fetched on 2026-11-18 (a key that was not set, or a source that did not
  answer) and are not measured.
- **Not the fusion pipeline's own certificate.** What is measured is the
  generic detector's score through calibration, certificate and decision
  layer. A certificate `detect_with_fusion` attaches from its own calibration
  data was not part of it.
- **Not much data.** Seven earthquake anomalies in five events; forty tornado
  ones in two. The small counts are in the table so that they can be read as
  small, and a hazard whose two judgings disagree is said to be near its
  target, not over it.

## Where it is served

- `GET /api/v1/catalog/hazard-evidence` returns the evidence whole, and each
  record of `GET /api/v1/catalog/domains` carries its hazard's block as
  `held_out_evidence`.
- A decision record about a measured hazard carries one caveat with that
  hazard's coverage and abstention (`decision/hazard_evidence.py`), and one
  about a hazard that could not be measured says that no held-out record
  exists. The caveat changes nothing in the verdict.

## Measuring it again

```bash
cd research/engine_calibration && MERCURY_CALIBRATION_CACHE=/some/dir python build_cache.py
python research/hazard_calibration/measure.py --cache /some/dir/cache --write
```

The first fetches the events (some need keys; those that cannot be fetched are
listed as not fetched). The second rewrites the evidence and the shipped copy;
`tests/research/test_hazard_calibration.py` fails until this document's table
matches them.
