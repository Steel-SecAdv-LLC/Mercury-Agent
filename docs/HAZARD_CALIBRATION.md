# Per hazard: what a score means, whether the certificate holds, how often nothing is decided

Measured 2026-11-20 by `research/hazard_calibration/measure.py`; the
numbers are `research/hazard_calibration/evidence.json`, and the package ships
the same file for the decision layer and the API to read. It replaces the
record of 2026-11-18, which was measured on events the loaders had built
wrongly; what changed, and how much of it is the loaders', is set out under
[What changed since the record of 2026-11-18](#what-changed-since-the-record-of-2026-11-18).

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
| earthquake | statistical | 5 | 2,214 | 7 | -0.044 | 99.1% (98.4% / 99.9%) | 97.6% | 63.2% | 2.6% of 39 |
| energy | statistical | 5 | 6,064 | 328 | -0.011 | 92.3% (90.6% / 94.0%) | 66.2% | 77.2% | 24.7% of 1,174 |
| fema | statistical | 5 | 2,228 | 431 | -0.056 | 95.5% (95.8% / 95.2%) | 93.1% | 34.0% | 15.3% of 163 |
| fema_activity | expert_annotated | 8 | 2,922 | 674 | +0.350 | 91.7% (91.8% / 91.6%) | 50.0% | 83.4% | 67.5% of 1,261 |
| flood | statistical | 1 | | | not measured: one event: there is no other to calibrate on | | | | |
| hurricane | statistical | 12 | 537 | 143 | +0.078 | 97.8% (97.8% / 97.8%) | 87.1% | 82.7% | 6.7% of 15 |
| landslide | statistical | 1 | | | not measured: one event: there is no other to calibrate on | | | | |
| marine | statistical | 3 | 443 | 47 | +0.166 | 96.0% (97.1% / 95.0%) | 90.7% | 57.3% | 63.4% of 71 |
| network_security | ground_truth | 2 | 32,938 | 9,721 | -0.023 | 72.5% (72.9% / 72.0%) | 45.6% | 49.4% | 42.4% of 12,464 |
| pandemic | statistical | 4 | 2,929 | 227 | +0.067 | 92.0% (91.9% / 92.0%) | 32.5% | 88.1% | 33.9% of 663 |
| sepsis | ground_truth | 2 | 38,890 | 674 | +0.010 | 90.9% (91.5% / 90.4%) | 72.3% | 67.2% | 4.7% of 7,242 |
| tornado | statistical | 3 | 593 | 50 | +0.251 | 95.9% (94.1% / 97.6%) | 61.5% | 89.3% | 48.6% of 74 |
| tsunami | statistical | 3 | 1,255 | 103 | +0.197 | 92.7% (91.2% / 94.2%) | 78.1% | 66.7% | 40.5% of 301 |

- **Skill** compares the calibrated probability with a reference that knows only
  the other events' share of anomalies (`1 - Brier / reference Brier`). Above 0
  the probability says more than that base rate; below 0, less. A large value
  can also mean only that the other events' base rate was far from this one's.
- **Coverage** is how often the true label was in the set.
- **Decided right** counts every decided row, and most rows are normal: where
  anomalies are few it says little. **Anomaly calls right** is how many of the
  rows called "anomaly" were; its count is over both judgings.

Read plainly:

- **Is the score a probability?** Calibrated on the hazard's other events, it says clearly more than their base rate for fema activity, hurricane, marine, pandemic, tornado and tsunami; about as much for energy and sepsis; and less for earthquake, fema and network security.
- **Does the certificate hold?** Asked for 90%, on an event it had not seen, it held whichever half it was fitted on for earthquake, energy, fema, fema activity, hurricane, marine, pandemic, sepsis, tornado and tsunami (10 of 11); fell short either way for network security; and was over on one half and under on the other for none.
- **How often is nothing decided?** From 32.5% of rows (pandemic) to 97.6% (earthquake).
- **Fema activity, whose labels are independent of what is scored:** skill +0.350, coverage 91.7%; of its true anomalies 63.1% were called, 31.3% abstained on and 5.6% called normal; 67.5% of 1,261 anomaly calls were right.
- **Network security, whose labels are independent of what is scored:** skill -0.023, coverage 72.5%; of its true anomalies 27.2% were called, 16.4% abstained on and 56.4% called normal; 42.4% of 12,464 anomaly calls were right.
- **Sepsis, whose labels are independent of what is scored:** skill +0.010, coverage 90.9%; of its true anomalies 25.4% were called, 63.2% abstained on and 11.4% called normal; 4.7% of 7,242 anomaly calls were right.

The certificate holds for ten of the eleven hazards, and abstention is what
pays for it: more than half of the rows are abstained on for eight of those
ten (pandemic, 32.5%, and FEMA declaration activity, 50.0%, are the two where
they are not). A certificate that holds while nothing is decided is a true
statement about very little. Network security, whose labels are independent of
what is scored, is the one hazard where it does not hold on either half
(72.5% against 90%), and 56.4% of its true attacks are called normal. FEMA
declaration activity is the one hazard with independent labels where the
calibrated probability clearly says more than the base rate (+0.350) and two
thirds of the anomaly calls are right. The detector is fitted unlabelled on
each event, so none of this is something it was built to carry from one event
to the next: a deployment that wants probabilities or certificates for a
hazard has to calibrate on that hazard's own labelled past and check them this
way.

## What changed since the record of 2026-11-18

The record this replaces said the certificate held for five of ten hazards,
fell short for two (network security, pandemic) and was over on one half and
under on the other for three (energy, tornado, tsunami). Three things have
changed since, and the same script was run on each state so that they can be
told apart. Every run below is of 2026-11-20, on events fetched that day.

1. **Not the loaders.** With the loaders as they were before the fixes
   (`7b422c7`), the calibration and detector code as it is now gives: held for
   seven of ten (earthquake, energy, fema, hurricane, marine, sepsis,
   tornado), short for network security and pandemic, tsunami over on one half
   and under on the other. Energy's loader was not touched by the fixes and
   its events hold the rows and anomalies they held on 2026-11-18 (6,064 and
   328), yet its coverage reads 92.3% (90.6% / 94.0%) where the record had
   88.7% (91.3% / 86.0%); network security reads 72.5% where the record had
   70.1%. That movement is not the loaders'.
2. **The loader fixes** (`7b422c7`, `5ee1f06`, `f024595`). Nine of ten held;
   network security short. Pandemic went from short to held (coverage 84.4% to
   92.0%, skill -0.653 to +0.067) because `mpox_2022`, which was the COVID
   file under another name and 20,000 of the hazard's 22,965 scored rows, is
   no longer an event. Tsunami went from undecided to held (90.7% with halves
   89.8% / 91.6%, to 92.7% with 91.2% / 94.2%) with its rolling statistics no
   longer reading later samples, and is abstained on far more (44.3% of rows
   to 78.1%). Earthquake, with each event cut to its own region, is abstained
   on almost entirely (82.5% to 97.6%). Hurricane on the 6-hourly synoptic
   track lost its skill (+0.030 to -0.050; abstained on 77.0% to 90.1%).
3. **The catalogue corrections of this record.** Seven hurricane events
   carried storm identifiers IBTrACS does not hold and had never been read;
   with them the hazard has twelve events and not five (skill -0.050 to
   +0.078, coverage 96.4% to 97.8%, abstained on 90.1% to 87.1%). The tornado
   archive file ended in 2023; with the 2024 outbreak the hazard has three
   events and not two (skill +0.188 to +0.251, coverage 91.5% to 95.9%), and
   is abstained on much more (11.0% of rows to 61.5%). FEMA declaration
   activity is measured here for the first time, because the event cache now
   holds it.

Energy, marine, network security and sepsis read the same in all three of
these runs.

## What this is not

- **Not the labels' independence.** `statistical` labels are a threshold on a
  column; a detector shown that column, or one that moves with it, is partly
  marking its own work (see the README's hazard table). Only network security
  and sepsis (`ground_truth`) and FEMA declaration activity
  (`expert_annotated`: NOAA NCEI's billion-dollar catalogue) have labels the
  loaders declare independent.
- **Not every hazard.** Landslide and flood have one event with both classes
  and cannot be measured this way. Financial, volcanic and wildfire could not
  be fetched on 2026-11-20: financial and wildfire need keys that were not
  set, and the volcanic loader's four events are not benchmarkable (the USGS
  API holds only the current alert).
- **Not the fusion pipeline's own certificate.** What is measured is the
  generic detector's score through calibration, certificate and decision
  layer. A certificate `detect_with_fusion` attaches from its own calibration
  data was not part of it.
- **Not much data.** Seven earthquake anomalies in five events; fifty tornado
  ones in three. The small counts are in the table so that they can be read as
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
export MERCURY_CALIBRATION_CACHE=/some/dir
python research/engine_calibration/build_cache.py
python research/hazard_calibration/measure.py --cache /some/dir/cache --write
```

The first fetches the events (some need keys; those that cannot be fetched are
listed as not fetched). The second rewrites the evidence and the shipped copy;
`tests/research/test_hazard_calibration.py` fails until this document's table
matches them.
