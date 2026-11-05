<!--
Copyright (C) 2025 Steel Security Advisors LLC
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# WS-E — Environmental mechanisms of reported anomalous experience: strict pre-registration

Registered **before** any data collection. Status: **design only. No data, no
code, no runtime change.** Revision 2 (2026-11-05); §1 records what revision 1
got wrong.

## 0. What this is, and what it is not

A **mechanism study with mundane hypotheses.** It asks whether reported
anomalous experience is produced by physical conditions already known to produce
it: infrasound near the human eyeball resonance, ELF/VLF magnetic fields,
air-quality excursions, and suggestibility. Every hypothesis predicts ordinary
physics, and the best-powered prior study predicts that the *fields* will not
matter and suggestibility will.

**This does not reopen WS-D.** `docs/PARAPSYCH_PREREGISTRATION.md` recorded a
faithful null on the GCP/psi question and closed it, on the stated grounds that
re-hunting a closed null is motivated reasoning rather than engineering. That
judgment stands. WS-E asks a different question of a different instrument: not
"do RNG streams deviate from chance," but "are reported experiences explained by
known environmental mechanisms."

**No claim is made that any anomalous phenomenon exists.** The residual endpoint
(§10) is specified in advance so it cannot be reached by analytic flexibility,
and §10 forbids describing it as paranormal in any output.

## 1. Corrections to revision 1

Recorded rather than quietly fixed, in keeping with AGENTS.md §6.6.

1. **The primary test did not match the mechanism.** R1 made within-site
   temporal coincidence the primary endpoint and reused
   `evaluation/event_coincidence.py` for it. But Tandy's standing wave was
   **continuous**, not episodic — the fan ran all the time. Coincidence-in-time
   cannot detect a constant exposure, because there is no onset to align to.
   The design was chosen by the available tool rather than by the mechanism.
   Corrected: the primary test is now a **controlled exposure contrast** (§3),
   and the coincidence tool is confined to the one place it fits — intermittent
   sources in the field study (§4).
2. **The residual endpoint was not computable.** R1 compared the residual
   fraction at report sites against "the same fraction at control sites."
   Control sites are defined by having no reported episodes, so that fraction
   does not exist. Corrected in §10: the comparison is against time-matched
   non-episode windows at the same site, which the circular permutation null
   already supplies.
3. **No exposure denominator.** R1 counted episodes per site-hour. Report rate
   scales with occupancy, so a busier site looks more active for reasons that
   have nothing to do with fields. This is the missing-denominator error that
   makes curated disappearance collections uninterpretable. Corrected in §9:
   the unit is **reports per person-hour**, with occupancy logged
   independently.
4. **The strongest design was missing.** R1 was purely observational. A
   controlled exposure experiment gives causal inference, needs far fewer
   participant-hours, and has better power. It is now Tier 1.
5. **No human-subjects protection.** R1 was observational and skated past it.
   Tier 1 deliberately exposes people to infrasound, so §3.4 is now a
   requirement, not a note.

## 2. The mechanism dictates the design

| Mechanism | Temporal form | Correct test |
|---|---|---|
| Standing-wave infrasound (Tandy) | **continuous** while source runs | between-condition contrast |
| Cycling HVAC / traffic infrasound | **intermittent** | within-site coincidence |
| ELF/VLF magnetic | either | both, by source |
| Suggestibility | trait, not event | covariate |

Two literatures bound the priors:

* **Tandy (1998, *JSPR*)** traced apparitional experience and cold sensation in
  a laboratory to a ~18.9 Hz standing wave from an extractor fan; removing the
  source removed the phenomenon. ~18–19 Hz is near the resonant frequency of the
  human eyeball, the proposed route to peripheral-figure percepts. A standing
  wave needs a resonant cavity, so room geometry **predicts** antinode
  locations — a prediction that can fail (H1b).
* **French, Haque, Bunton-Stasyshyn & Davis (2009, *Cortex*)** built a room with
  controllable infrasound and complex EM fields and found reported experience
  tracked participants' **suggestibility**, not the field state. This is the
  best-powered study in the area and its result is the registered expectation
  here.
* **Granqvist et al. (2005, *Neuroscience Letters*)** failed to replicate
  Persinger's temporal-lobe EM effects under double-blind, so H2 carries an
  explicitly weak prior.

**Equipment consequence, stated because it invalidates most prior field work.**
Consumer audio recorders and EMF meters roll off far above 20 Hz and physically
cannot measure the band carrying the only replicated positive result. A study
that cannot measure 15–22 Hz cannot test the leading hypothesis, whatever else
it records.

## 3. Tier 1 — controlled exposure experiment (**primary**)

Essentially a replication of French et al. with instrumentation that can
actually verify the infrasound field, which prior work often could not.

### 3.1 Design

2 × 2 between-subjects, double-blind: infrasound (18.9 Hz on / off) ×
ELF magnetic field (on / off). Participants spend a fixed period in one room and
complete a standard experience checklist afterwards. Neither participant nor
the experimenter in the room knows the condition; conditions are set by a
schedule held by a third party.

### 3.2 Hypotheses and priors

| ID | Hypothesis | Registered prior |
|---|---|---|
| **H1** | infrasound main effect on total report count | **weak-moderate** — French predicts null |
| H1b | reports cluster at cavity-predicted antinodes | moderate, conditional on H1 |
| H2 | EM main effect | weak |
| H1×H2 | interaction | weak |
| **H3** | suggestibility predicts report count independent of condition | **strong** |

A successful replication of French — H3 positive, H1 and H2 null — is a
publishable result and the expected one. Finding H1 positive with the field
verified would be the more interesting outcome, which is why the field is
measured rather than assumed.

### 3.3 Fixed measures

* **Primary endpoint:** one number — total reports per participant, infrasound
  main effect, ordinary least squares with suggestibility as a pre-specified
  covariate. Alpha 0.05, two-sided.
* **Suggestibility:** Tellegen Absorption Scale, administered **before** the
  session and before condition assignment is known. Scored by someone blind to
  condition. (The Revised Transliminality Scale is recorded as a secondary
  measure; the primary is fixed as TAS and will not be swapped.)
* **Experience checklist:** fixed item set, written before any session, with
  items for percepts the mechanisms predict (peripheral figure, cold sensation,
  pressure, unease) and filler items for percepts they do not.
* **Field verification:** the infrasound and magnetic fields are measured at the
  participant's seated head position in every session, including off-conditions.
  A session whose field does not match its assigned condition is excluded and
  reported as an exclusion.

### 3.4 Human-subjects requirements (binding)

* IRB or equivalent ethics approval before the first session. No approval, no
  sessions.
* SPL ceiling fixed in advance and recorded; exposure held below it with a
  hardware limiter, not an operator.
* Exclusion criteria screened in advance: pregnancy, seizure history, vestibular
  disorder, cochlear implant or hearing aid, cardiac device.
* Withdrawal at any moment with no reason given and no loss of compensation,
  stated in the briefing and again at the midpoint.
* Full debrief naming the mechanisms under test, because the design involves
  non-disclosure of condition.
* Consent records stored separately from research data and never committed to
  this repository.

## 4. Tier 2 — observational field study (secondary, and underpowered by design)

Run only after Tier 1. Instruments sites where experiences are reported, with
matched control sites.

**Stated honestly:** with 8 sites this tier cannot support a between-site claim.
Eight is a sample size for describing instrument behaviour, not for inference
about site properties. A between-site test needs roughly 20+ sites, which
changes the cost structure entirely. Therefore:

* The **between-site** comparison is **descriptive only** and will be labelled
  as such. No p-value from it is reported as a finding.
* The **within-site coincidence** test is inferential, and it applies to
  intermittent sources only (§2). This is the one place
  `PreregisteredCoincidenceTest` belongs.
* Preferred allocation if funding allows: more sites, fewer hours each. 20 sites
  × 80 hours beats 8 sites × 200 hours for every question this tier asks.

## 5. Instruments and calibration

| Channel | Band / range | Calibration requirement |
|---|---|---|
| Infrasound | 0.1–20 Hz, primary sub-band 15–22 Hz | published response curve; reference tone before and after every session |
| Audio | 20 Hz–20 kHz | SPL reference |
| Magnetometry (fluxgate) | DC–3 kHz | zero-offset and scale check per session |
| RF / CSI | 2.4 & 5 GHz | existing `biometric/radar_iq.py` path |
| CO / VOC / PM | per sensor spec | span gas or factory certificate, recorded |
| Confounds | temperature, humidity, barometric pressure, mains 50/60 Hz and harmonics, seismic | logged continuously on the shared clock |
| **Occupancy** | headcount per interval | independent of the experience log (§9) |

One clock across all channels, with drift measured and recorded per session.
Confound channels are not optional: mains harmonics and HVAC cycling are the two
most common sources of apparent excursions in every band here.

## 6. Negative controls — three, all mandatory

1. **Dummy transducer.** A terminated input through the identical acquisition and
   analysis pipeline. Anomalies here are pipeline artifact.
2. **Synthetic channel.** A generated noise stream, parameters fixed in advance,
   through the identical pipeline. Same device as WS-D's `Binomial(200, 0.5)`
   plumbing validation, for the same reason.
3. **Sham condition (Tier 1) / control sites (Tier 2).**

If the anomaly rate in (1) or (2) is statistically indistinguishable from the
rate in live channels, the study reports an instrumentation result and stops.

## 7. Blinding and the observer effect

* Tier 1 is double-blind, with the condition schedule held by a third party.
* Tier 2: participants log experiences through a timestamped logger writing to a
  store the analysis pipeline cannot read. Anomaly detection runs over the full
  record with labels withheld, output is **committed by hash**, and labels are
  joined only afterwards.
* The analyst running §8 does not know which sites are report sites.
* **Observer effect:** whether an investigator is present changes report rate.
  Investigator presence is logged per interval and entered as a covariate. Tier 2
  sessions with an investigator present are analysed separately from unattended
  sessions; the two are never pooled.

## 8. Fixed statistics — enumerated, with no alternatives to be evaluated

**Tier 1.** One primary endpoint (§3.3). Secondary family, k = 4: EM main
effect, interaction, suggestibility coefficient, per-symptom breakdown.
Benjamini–Hochberg over those 4.

**Tier 2.** `PreregisteredCoincidenceTest` from
`src/omni_mercury_engine/evaluation/event_coincidence.py`, unchanged:

* **Null:** circular time-shift permutation. Correct for autocorrelated streams —
  it preserves each channel's temporal structure and destroys only its alignment
  to episode windows, so a significant result cannot be an autocorrelation
  artifact.
* **Permutations:** 10,000. **Alpha:** 0.05.
* **Test count:** 5 channels (infrasound, audio, magnetometry, RF,
  air-quality) × 1 window × 1 threshold = **5 tests**. Benjamini–Hochberg over 5.
  The channel set is **closed** as of this registration.
* **Window:** asymmetric and lag-aware, exposure onset **−180 s to −5 s**
  relative to the report timestamp. Rationale: a causal exposure must precede
  the percept, and the percept must precede the person noticing and logging it.
  A symmetric window dilutes a directional effect and admits reverse-order
  coincidences. One window, no alternatives.
* **Excursion:** band power ≥ 3 robust standard deviations above the site's own
  baseline (median/MAD over the site's full record excluding episode windows),
  sustained ≥ 10 s.
* **Seeds:** 0, 1, 2. A result unstable across these three is reported as
  unstable, not as positive.

## 9. The denominator

The unit of exposure is the **person-hour**, not the site-hour. Report rates are
reported as reports per person-hour with occupancy logged independently of the
experience log. A rate without an occupancy denominator is uninterpretable, and
reporting one would repeat the error that makes curated disappearance
collections useless as evidence.

## 10. Residual endpoint R — corrected

R is the fraction of episodes with no H1–H4 excursion in the §8 window.

**Comparison (corrected from R1):** against the excursion rate in time-matched
non-episode windows at the **same site**, which the circular permutation null
already generates. If episodes and matched non-episode windows carry excursions
at the same rate, the instrumented mechanisms explain nothing at that site, and
R is the whole of the result rather than a remainder.

**R is not evidence of anything by itself.** An episode with no measured
correlate is an episode whose cause this instrument did not capture. Ordinary
candidates: channels not instrumented, sources outside the recorded band,
expectancy and suggestibility (H3), misattributed ordinary percepts, sensor
dropout (§11).

**R may not be described as paranormal, anomalous in the strong sense, or
unexplained in principle in any output of this study.** The permitted phrasing
is "residual fraction not accounted for by the instrumented mechanisms."

## 11. Missing data

Fixed in advance, because it is otherwise a degree of freedom:

* An episode window with ≥ 10% sample loss on the primary channel is **excluded**
  and counted in an exclusions table. It is not treated as no-excursion.
* A session with ≥ 5% loss on the primary channel is excluded whole.
* Exclusion counts are reported alongside every endpoint.

## 12. Power and stopping

* **Tier 1:** target n fixed in advance from a power calculation for the primary
  endpoint at the smallest effect of interest, with the calculation and its
  assumed effect size written into this file before recruitment. Recruitment
  stops at target n and is not extended for a near-threshold result.
* **Tier 2:** minimum 20 logged episodes. Below 20 the tier reports
  **underpowered**, not null — a null requires the power to have detected the
  effect.
* No interim peeking at the primary endpoint. Enrolment totals may be monitored;
  outcomes may not.

## 13. Falsification

* **H1 fails** if the infrasound main effect is non-significant at target n, or
  unstable across the fixed seeds.
* **H1b fails** if report locations are uniform with respect to predicted
  antinodes.
* **H3 fails** if suggestibility carries no coefficient at target n — which
  would contradict French and would itself be the finding.
* **The study as a whole fails** if the dummy or synthetic channel produces an
  anomaly rate indistinguishable from live channels.

## 14. Publication commitment

The result is written up and published whatever it is, including a clean null,
including a successful replication of an existing null, and including
"underpowered." Analysis code is hash-committed before labels are joined and the
hash appears in the write-up. Exclusions, field-verification failures and
occupancy denominators are reported, not summarised away.

## 15. What would void this registration

Named so the failure modes are recognised rather than discovered:

* Adding a channel, hypothesis or checklist item after seeing data.
* Changing the window, threshold, sustain duration, or suggestibility instrument
  after seeing data.
* Pooling attended and unattended Tier 2 sessions.
* Reporting a between-site Tier 2 p-value as a finding.
* Reporting a rate without its person-hour denominator.
* Re-running with new seeds and reporting the favourable set.
* Extending recruitment or collection past §12 to chase a near-threshold result.
* Reporting R as evidence of anything.

Any of these makes the output exploratory, and it must then be labelled
exploratory in every surface that cites it.
