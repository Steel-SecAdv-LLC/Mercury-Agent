# Earthquake-Precursor Hook: Literature Review and Feature-Spec Gate

**Scope:** `earthquake_precursor` hook — `EarthquakePrecursorAnalyzer` in
`src/omni_mercury_engine/space/disaster_precursor_detector.py`; registry entry in
`src/omni_mercury_engine/ml/hazard_training/registry.py` (category `b`).
**Purpose:** This review gates whether the hook is trained at all, and on what.
**Date:** 2026-07-10. All citations below were verified against the published record
(publisher pages, DOIs, or indexed abstracts) during this review; none are from memory alone.

---

## Abstract / Verdict

The hook currently carries two incompatible identities. Its **docstring** claims
"earthquake precursor detection using electromagnetic signatures / Schumann+seismic
correlations for earthquake prediction." Its **training-registry entry** proposes something
entirely different: USGS catalog data reshaped into regional seismicity sequences with
did-a-large-event-follow labels.

**Verdict (a) — the EM/Schumann interpretation does not survive.** Four decades of
peer-reviewed work, culminating in the ICEF report (Jordan et al., 2011) and the 2021
Frontiers critical reviews (Conti et al., 2021; Picozza et al., 2021), find **no
electromagnetic precursor validated for operational earthquake forecasting**. The strongest
positive EM results (DEMETER statistical studies: Němec et al., 2008; Píša et al., 2013) are
population-level averages of order 2 dB, hours before rupture, with no demonstrated
event-level predictive skill. Schumann-resonance-specific claims rest on unreplicated
single-station case studies (Hayakawa et al., 2005). The architecture's magnitude head
(sigmoid × 9.0 Richter) and time head (sigmoid × 72 h) encode a deterministic-prediction
contract that no published result supports. No labeled EM-precursor corpus exists; training
this interpretation would require fabricating data. **The docstring must be rewritten and the
EM framing retired** (per this repo's honest-language rules); the detector's existing
abstention (`estimated_magnitude = None` until real weights exist) is the correct behavior
and must be preserved.

**Verdict (b) — a catalog-based seismicity-rate spec survives, with conditions.**
Statistical seismicity forecasting is genuinely established: earthquake clustering
(Omori/ETAS; Ogata, 1988) is the dominant, repeatedly validated source of short-term
predictability, deployed operationally by the USGS, INGV (Italy), and GNS (New Zealand),
and tested prospectively for ~two decades under CSEP. ETAS-class models achieve probability
gains of roughly a factor 5–10 per target earthquake over time-independent models
(Helmstetter et al., 2006; Werner et al., 2011). A neural model trained on catalog features
is scientifically defensible **only if** (1) the label is probabilistic occurrence in a
region-window, not per-event magnitude/time prediction; (2) the primary threshold is
lowered from M6+ to M≥5 (M6+ gives ~30–40 positives in 45 years of California data —
untrainable and untestable); and (3) the merit gate compares against an
**Omori/ETAS-style clustering baseline, not Poisson and not the detector's current
non-neural path** — the existing "physics fallback" computes heuristic Schumann/geomagnetic
correlations and abstains from magnitude; it never emits P(M≥X within W days) and is
therefore not commensurable with this task. Any model that merely learns "big quake
yesterday → elevated probability tomorrow" will beat Poisson; that is real skill (it *is*
ETAS), but claiming novelty requires beating the clustering baseline itself, and the
documentation must say so explicitly.

---

## 1. Electromagnetic / Schumann-resonance precursors: what the literature supports

### 1.1 The ICEF assessment (the field's consensus baseline)

After the 2009 L'Aquila earthquake, the International Commission on Earthquake Forecasting
for Civil Protection reviewed the entire precursor literature — electromagnetic signals,
ionospheric total electron content (TEC), the VAN method, radon, strain, seismicity
patterns — and concluded that **the search for diagnostic precursors (signals that reliably
indicate a specific impending earthquake) has not produced a validated short-term prediction
scheme**, and recommended that operational forecasting be built instead on statistical
seismicity models with rigorous prospective testing (Jordan et al., 2011, *Annals of
Geophysics* 54(4), 315–391, doi:10.4401/ag-5350). This remains the reference position of
operational agencies; nothing published since 2011 has overturned it (see §1.4).

### 1.2 The VAN controversy: the cautionary template

The VAN method (Varotsos–Alexopoulos–Nomicos "seismic electric signals," Greece, 1980s) is
the most extensively litigated EM-precursor claim. A dedicated *Geophysical Research
Letters* debate issue (Geller, ed., 1996, GRL 23(11), doi:10.1029/96GL00742) collected the
critiques that became the field's standard checklist: retrospective selection of alarm
parameters, elastic prediction windows evaluated after the fact, no complete ledger of
false alarms and missed events, and success rates statistically indistinguishable from
chance given regional seismicity rates (Mulargia & Gasperini, 1996, GRL 23,
doi:10.1029/95GL03456; Jackson & Kagan contributions in the same issue). Geller's (1997)
broader review — *Geophysical Journal International* 131(3), 425–450 — extended the
conclusion: a century of precursor searches (EM and otherwise) produced no reliable,
reproducible precursor, and rupture initiation appears sensitive to fine-scale state
inaccessible to measurement. The companion position paper "Earthquakes cannot be predicted"
(Geller, Jackson, Kagan & Mulargia, 1997, *Science* 275, 1616–1617,
doi:10.1126/science.275.5306.1616) fixed the distinction this repo must respect:
**deterministic prediction of individual events is not supported; probabilistic forecasting
of rates is**.

### 1.3 DEMETER and satellite ionospheric studies: the strongest honest positive

The CNES DEMETER microsatellite (2004–2010) provides the best *statistical* EM-precursor
evidence, and its strength should be stated precisely because it is the ceiling of the
positive case:

- Němec, Santolík, Parrot & Berthelier (2008, *Geophysical Research Letters* 35, L05109,
  doi:10.1029/2007GL032517) found a statistically significant **decrease of ~a few dB in
  nighttime VLF wave intensity 0–4 hours before** shallow earthquakes near the epicenter —
  as an average over many hundreds of events, not per event.
- Píša, Němec, Santolík, Parrot & Rycroft (2013, *J. Geophys. Res. Space Physics* 118,
  doi:10.1002/jgra.50469) re-ran the analysis over the full 6.5-year mission and confirmed
  the effect at ~2 dB — small, short-lead, population-level.

Criticisms and limits, from the same literature: the effect size is far too small relative
to natural variability to identify *which* region will rupture (no event-level probability
gain demonstrated); lead time is hours, not the days implied by "early warning" framing;
and case-study TEC/EM claims for individual large earthquakes have repeatedly failed
independent statistical re-analysis when tested against the full distribution of
non-seismic days (reviewed in Picozza et al., 2021). China's CSES-01 mission (2018–)
continues this line; as of this review it has produced further statistical associations
but no validated operational precursor (ibid.).

### 1.4 Schumann-resonance-specific claims

The claim the hook's docstring inherits traces to single-observatory case studies: Hayakawa,
Ohta, Nickolaenko & Ando (2005, *Annales Geophysicae* 23, 1335–1346) reported an anomalous
enhancement of the **fourth** Schumann-resonance harmonic (~1 Hz peak-frequency shift) at
Nakatsugawa, Japan, in the week before the 1999 Chi-Chi (Taiwan) earthquake, with follow-up
surveys of Taiwan events by the same group. Standing criticisms, none resolved since:

1. **Global confound:** Schumann resonances are a planetary cavity phenomenon driven by
   global thunderstorm activity; single-station anomalies cannot be uniquely attributed to a
   seismic source region ~2,000 km away without ruling out lightning-source variability,
   which these studies did not do statistically.
2. **No formal hypothesis test:** anomalies were selected retrospectively around known
   events; no false-alarm ledger, no significance against the year-round anomaly rate —
   exactly the VAN failure mode (§1.2).
3. **No independent replication or prospective test** in 20 years.

The 2021 twin critical reviews — Conti, Picozza & Sotgiu (2021, *Frontiers in Earth
Science* 9:676766, ground-based) and Picozza, Conti & Sotgiu (2021, *Frontiers in Earth
Science* 9:676775, space-based) — survey this entire class of observations and conclude
that no EM precursor (Schumann included) has passed rigorous statistical validation, and
that the field must abandon case-study methodology for systematic statistical testing.

### 1.5 Verdict on Question 1

**Not validated for operational forecasting.** The strongest defensible positive statement
is: *small (~2 dB), hours-scale, population-averaged ionospheric/VLF anomalies are
statistically associated with imminent shallow earthquakes (DEMETER), with no demonstrated
skill for predicting individual events.* There is no labeled corpus mapping any "128-dim EM
feature vector" to earthquake magnitude and time-to-event; the sigmoid×9.0 magnitude head
and sigmoid×72h time head have no literature anchor at all. Training the hook under its
current docstring is scientifically indefensible.

---

## 2. Catalog-based statistical seismicity forecasting: what IS established

### 2.1 Clustering is the signal: Omori, Gutenberg–Richter, ETAS

Two empirical laws underpin everything operational: the Gutenberg–Richter (G-R)
frequency–magnitude law (Gutenberg & Richter, 1944, *BSSA* 34, 185–188) and Omori-law
aftershock decay. Ogata's Epidemic-Type Aftershock Sequence model (Ogata, 1988, *J. American
Statistical Association* 83(401), 9–27, doi:10.2307/2288914; space-time form Ogata, 1998,
*Ann. Inst. Statist. Math.* 50, 379–402) formalizes seismicity as a self-exciting point
process: every event triggers offspring at a rate increasing exponentially with its
magnitude and decaying in time as a power law. Nearly all demonstrated short-term
earthquake predictability is ETAS-style clustering; this is the central honest fact the
model card must state.

### 2.2 Operational deployments (probabilistic, clustering-based, none EM)

- **USGS, California/US:** Reasenberg & Jones (1989, *Science* 243, 1173–1176,
  doi:10.1126/science.243.4895.1173) parametric aftershock probabilities, issued publicly
  since 1989; the STEP 24-hour clustering forecast (Gerstenberger, Wiemer, Jones &
  Reasenberg, 2005, *Nature* 435, 328–331, doi:10.1038/nature03622); current operational
  aftershock forecasts with structured communication (Michael et al., 2020, *SRL* 91(1),
  153–173, on the 2018 Mw 7.1 Anchorage sequence).
- **Italy (INGV):** OEF-Italy, an ensemble of ETAS/ETES/STEP running 24/7 since ~2009
  (Marzocchi, Lombardi & Casarotti, 2014, *SRL* 85(5), 961–969). Ten-year validation:
  forecasts are statistically reliable and skillful versus reference models (Spassiani,
  Falcone, Murru & Marzocchi, 2023, *GJI* 234(3), 2501–2518, doi:10.1093/gji/ggad256).
- **New Zealand (GNS):** hybrid clustering + medium-term models for the Canterbury sequence,
  retrospectively and prospectively tested (Rhoades, Liukis, Christophersen &
  Gerstenberger, 2016, *GJI* 204(1), 440–456).

### 2.3 CSEP: the testing standard this pipeline must imitate

The Regional Earthquake Likelihood Models (RELM) experiment defined the evaluation
paradigm — models submit gridded space–magnitude **rate forecasts**, scored prospectively
with likelihood-based consistency and comparison tests (N-test, L-test, S-test, paired
T-test information gain) against the observed catalog (Schorlemmer, Gerstenberger, Wiemer,
Jackson & Rhoades, 2007, *SRL* 78(1), 17–29). The Collaboratory for the Study of Earthquake
Predictability (CSEP) globalized this (Schorlemmer et al., 2018, *SRL* 89(4), 1305–1313);
Italy's 5-year prospective experiment confirmed that clustering models outperform
time-independent ones out of sample (Taroni et al., 2018, *SRL* 89(4), 1251–1261). A
ten-year archive of prospective next-day California forecasts is now public and is the
natural external benchmark for this hook (*Scientific Data*, 2025,
doi:10.1038/s41597-025-05766-3).

### 2.4 b-value temporal variation as a stress meter: real but fragile

Laboratory and field evidence supports b-value dependence on differential stress
(Schorlemmer, Wiemer & Wyss, 2005, *Nature* 437, 539–542, doi:10.1038/nature04094). Gulia &
Wiemer (2019, *Nature* 574, 193–199, doi:10.1038/s41586-019-1606-4) built the Foreshock
Traffic-Light System (FTLS) on this: b drops ~10% in foreshock sequences and rises ~20% in
aftershock sequences, classifying 58 retrospective sequences with ~95% accuracy. The debate
that followed is the load-bearing caveat:

- Dascher-Cousineau, Lay & Brodsky (2020, *SRL* 91, "Two Foreshock Sequences Post Gulia and
  Wiemer (2019)") applied the scheme to Ridgecrest 2019 and Puerto Rico 2020 and found the
  output **sensitive to parameter choices left to expert judgment** (region geometry, Mc,
  minimum sample); Gulia & Wiemer's reply and pseudoprospective Ridgecrest evaluation
  (Gulia & Wiemer, 2020/2021, *SRL*) attributed the discrepancies to methodological
  deviations — i.e., the method is not yet turnkey-reproducible.
- Marzocchi, Spassiani, Stallone & Taroni (2020, *GJI* 220(3), 1845–1856) demonstrated how
  easily spurious "significant" b-value variations arise from Mc misestimation, selection
  effects, and retrospective testing.
- van der Elst (2021, *JGR Solid Earth* 126, e2020JB021027) introduced the b-positive
  estimator specifically because transient catalog incompleteness after mainshocks biases
  classical estimates downward — mimicking a foreshock signal.

Conclusion: b-value features are admissible as *inputs* with robust estimation and explicit
uncertainty; they are not, alone, a validated discriminator.

### 2.5 How much skill exists at all (calibration for expectations)

Helmstetter, Kagan & Jackson (2006, *BSSA* 96(1), 90–106, doi:10.1785/0120050067): an
ETAS-type next-day model for southern California achieves a **probability gain of ~6 per
target earthquake** over a time-independent smoothed-seismicity model; the update in Werner,
Helmstetter, Jackson & Kagan (2011, *BSSA* 101(4), 1630–1648) is consistent. Deep learning
has not demonstrably exceeded this class: DeVries et al.'s aftershock-location deep network
(2018, *Nature* 560, 632–634) was matched by two-parameter logistic regression (Mignan &
Broccardo, 2019, *Nature* 574, E1–E3, doi:10.1038/s41586-019-1582-8), and a meta-analysis of
77 neural-network earthquake-prediction papers (1994–2019) found simpler baselines
consistently competitive and evaluation practices weak (Mignan & Broccardo, 2020, *SRL*
91(4), 2330–2342). Expectation set here: the trained hook, if it ships, will at best add a
modest increment over an ETAS-lite baseline; anything dramatically better is a red flag for
leakage, not a breakthrough.

---

## 3. Honest feature set for a regional catalog-based model

Candidate features from the literature, each with support and pitfalls. Windows below are
computed strictly causally (data up to forecast time t only) per region cell.

| # | Feature | Literature anchor | Pitfalls / required caveats |
|---|---------|-------------------|------------------------------|
| F1 | Seismicity rates: event counts M≥Mc_floor in trailing windows (1, 7, 30, 90, 365 d), log1p-scaled | Smoothed/short-term rate models: Helmstetter et al. (2006); Werner et al. (2011) | Completeness drift over decades (Hutton et al., 2010); short-term aftershock incompleteness inflates apparent quiescence right after mainshocks (van der Elst, 2021) |
| F2 | ETAS-like triggering summary: Σ over past events of 10^(b·(m_i−Mc)) · (t−t_i+c)^(−p) with fixed generic California parameters (Reasenberg–Jones); optionally per-window largest magnitude | Ogata (1988); Reasenberg & Jones (1989); Gerstenberger et al. (2005) | Fixing generic parameters is defensible and stable; *fitting* ETAS per cell on short windows is not (parameter instability). Document parameter provenance |
| F3 | b-value: Aki–Utsu MLE on trailing 365 d (b = log10(e)/(m̄−Mc+Δm/2)), with Shi & Bolt (1982) standard error, minimum sample n≥50, plus b-positive variant | Aki (1965); Utsu (1965); Shi & Bolt (1982); Schorlemmer et al. (2005); van der Elst (2021) | Spurious variations from Mc error and selection (Marzocchi et al., 2020); FTLS replication dispute (Dascher-Cousineau et al., 2020) — feed as feature, never as a standalone traffic light |
| F4 | Mc estimate per cell-window: maximum curvature + 0.2 correction; feature masked/flagged when Mc > Mc_floor | Wiemer & Wyss (2000); Woessner & Wiemer (2005: MAXC systematically underestimates Mc, hence the correction) | Mc is itself an estimate; propagate as an input *and* a validity mask, not silently |
| F5 | Time since last M≥5.5 within cell (and within 100 km), capped/log-scaled | Clustering interpretation only: recency ⇒ elevated rate (Omori/ETAS; Ogata, 1988) | Must NOT be interpreted as gap/"overdue" hazard: the seismic-gap hypothesis failed prospective testing (Kagan & Jackson, 1991, *JGR* 96, 21419–21431) |
| F6 | Seismic moment release in trailing 30/365 d (Σ10^(1.5m)); "magnitude deficit" vs long-term average | Weak — gap-adjacent | Same Kagan & Jackson (1991) objection; include moment release as a rate-like covariate at most, drop "deficit" framing from v1 |
| F7 | Depth distribution (median, IQR of hypocentral depth in window) | Sparse direct support for regional rate forecasting | Depth errors in ComCat are large and heterogeneous over decades; OPTIONAL, off by default in v1 |
| F8 | Nearest-neighbor clustering: Zaliapin–Ben-Zion η = t·r^df·10^(−b·m) statistics (fraction of "clustered" events, mean log η) in trailing 365 d | Zaliapin & Ben-Zion (2013, *JGR* 118, 2847–2864): bimodal η separates clustered from background seismicity; stable to catalog perturbations | Requires b, df choices (b≈1, df≈1.6 for California per the anchor paper); recompute stability check on ComCat before trusting |

Dropped as unsupported: any Schumann/EM/geomagnetic/ionospheric feature (§1); "time since
last mainshock" as a hazard-increasing (renewal) variable (Kagan & Jackson, 1991);
astronomical/lunar features (no accepted regional forecasting skill; not reviewed further
here).

The 128-dim input contract of `EarthquakePrecursorAnalyzer` can be honored by F1–F8
computed on a small stack of window lengths and neighbor rings, zero-padded — but the
input's *name and documentation* must change from "EM features" to "catalog seismicity
features" (§5).

---

## 4. Label design: "did M≥X follow within W days in region R"

### 4.1 What CSEP-style tests actually use

CSEP/RELM evaluates **gridded expected rates** per space–magnitude–time bin with likelihood
tests and paired information-gain comparisons (Schorlemmer et al., 2007; Taroni et al.,
2018) — not binary classification. A binary region-window label is a legitimate coarsening
for an MLP head, but the evaluation must remain probabilistic: log-loss (equivalently
information gain per event vs baseline), Brier score, and reliability diagrams — never
accuracy, which is meaningless at these base rates.

### 4.2 Base rates and class imbalance

California (greater RELM region), USGS ComCat: M≥6 events number only ~30–40 in-region
since 1980 (≈0.7–0.9/yr, strongly clustered — Landers 1992, Northridge 1994, Ridgecrest
2019...). With, e.g., ~50 one-degree cells × 45 years of 30-day windows (~27,000
cell-windows), M6+ positives are ~10^-3 of samples *and* only a few dozen independent
episodes — too few to train on and far too few for by-year test splits to have power. At
M≥5 the region yields several events per year (a few hundred since 1980), still imbalanced
(~10^-2) but with enough independent episodes to estimate calibration. **The registry's
"did-M6+-follow" label is therefore rejected as the primary target; M≥5 within 30 days is
the defensible primary; M≥6 is retained as a secondary evaluation-only threshold via the
G-R extrapolation of the predicted rate.**

### 4.3 Defensible catalog choices for California / ComCat

- **Catalog:** USGS ComCat via FDSN event service (already the registry's source);
  region = RELM California testing polygon (Schorlemmer et al., 2007).
- **Completeness:** SCSN completeness is ~ML 3.25 from 1932 and ~ML 1.8 from 1981 (Hutton,
  Woessner & Hauksson, 2010, *BSSA* 100(2), 423–446, doi:10.1785/0120090130); statewide
  ComCat completeness varies by network era. Use **Mc_floor = 2.5, data from 1981-01-01**,
  verify per-cell-window with F4 and mask windows failing Mc ≤ 2.5. The registry's
  "M≥2.5 completeness ~1980+" assumption is approximately right but must be *checked per
  cell*, not asserted (Woessner & Wiemer, 2005).
- **Cells:** 0.5°–1.0° (aggregations of the 0.1° RELM grid). **Window:** W = 30 days,
  rolled weekly (matching OEF-Italy's weekly cadence; Spassiani et al., 2023).
- **Splits:** by-year temporal blocks train < val < test (already enforced by this repo's
  pipeline); do **not** split spatially — sequences span cells, and time-forward splitting
  is the CSEP-prospective analog. Features must be strictly causal; no event attributes
  revised after t (e.g., late magnitude revisions) may leak in.

### 4.4 The critical pitfall: most positives are aftershocks — and that is fine, if said

At M≥5/30 d, the majority of positive labels sit inside aftershock sequences of prior M≥5–6
events (Ridgecrest's Mw 7.1 followed a Mw 6.4 by 34 hours; Landers→Big Bear; etc.). A model
trained on this label will chiefly learn clustering: "large earthquake recently → elevated
probability now." **This is genuine, honest forecasting skill** — it is precisely the ETAS
mechanism that underlies every operational system (§2.2), and per Helmstetter et al. (2006)
it is where nearly all short-term information lives. It is *not* a novel capability. Two
obligations follow:

1. **Documentation:** the model card and hook docstring must state that learned skill is
   expected to be dominated by aftershock/foreshock clustering, and that the model has NOT
   demonstrated precursory skill beyond clustering unless the baseline comparison below
   proves it.
2. **Baseline:** the merit gate must include a clustering term. Beating a Poisson/long-term
   baseline only proves the model rediscovered Omori's law (1894-era knowledge); the gate
   would be theater. See §5(b).

---

## 5. Verdict

### 5(a) The EM/Schumann interpretation: FAILS review

- No peer-reviewed support for operational EM-precursor forecasting (Jordan et al., 2011;
  Geller, 1997; Conti et al., 2021; Picozza et al., 2021); strongest positives are ~2 dB
  population-level effects with hour-scale leads (Němec et al., 2008; Píša et al., 2013).
- No labeled corpus exists for "128-dim EM features → (magnitude, time-to-event)". Training
  this contract would require fabricated data, which the registry rightly forbids.
- The sigmoid×9.0 magnitude head and sigmoid×72 h time head assert a deterministic
  event-prediction capability rejected by the field (Geller et al., 1997).

**Required actions (repo honest-language rules):** rewrite the class and module docstrings
to remove "earthquake prediction from electromagnetic signatures"; rename or re-document
the trained artifact as a *regional seismicity-rate forecaster* (e.g., keep the registry key
`earthquake_precursor` for compatibility but state in `data_requirement`/docs that the
trained head is catalog-statistical, not EM); delete or permanently disable the
`time_predictor` (×72 h) head — no literature supports a 0–72 h time-to-event regression
from any admissible feature; never populate `DisasterPrecursorResult.estimated_magnitude`
from this model (emit a probability, not a magnitude). The current abstention behavior
(`_neural_trained=False` → magnitude stays `None`, loud warning) is correct and must remain
the untrained default.

### 5(b) The catalog-based spec: PASSES, with binding conditions

The registry's direction (USGS FDSN catalog → regional sequence samples) is scientifically
sound and matches the only earthquake-forecasting methodology with demonstrated,
prospectively tested skill (§2). Binding conditions:

1. **Label:** P(M≥5.0 event in cell within 30 days), weekly cadence — not M6+ (§4.2), not
   per-event magnitude or time-to-event.
2. **Primary merit-gate baseline — clustering, not Poisson, not the current fallback.**
   The detector's existing non-neural path was inspected for this review
   (`DisasterPrecursorDetector.detect_disaster_precursor`): it computes a Schumann-anomaly
   risk score plus hand-weighted geomagnetic (0.4/0.3/0.2) and seismic heuristics, and
   abstains from magnitude. It never outputs an event probability for a region-window, so
   it **cannot serve as the merit-gate comparator for this task**. The gate must instead
   implement an **ETAS-lite/Reasenberg–Jones clustering baseline** from the identical
   catalog: rate(t) = μ_cell (long-term smoothed rate) + Σ_i 10^(a+b(m_i−M5)) (t−t_i+c)^(−p)
   with generic California parameters (Reasenberg & Jones, 1989), converted to
   P(≥1 event in W) = 1 − exp(−∫rate). A time-independent Poisson model is retained only as
   a sanity floor.
3. **Metrics:** out-of-time information gain per target event vs the clustering baseline
   (paired T-test sense of Taroni et al., 2018), log-loss, and reliability; ship only if
   the learned model beats the *clustering* baseline with acceptable calibration.
   Expected magnitude of any win: small (§2.5). A large win indicates leakage until proven
   otherwise.
4. **Features:** exactly the surviving set in §6; no EM/Schumann inputs.
5. **Documentation:** the aftershock-dominance statement of §4.4 is mandatory in the model
   card, the provenance sidecar, and the hook docstring.

---

## 6. Feature spec v1 — `seismicity-catalog-v1`

Input vector per (cell, forecast time t), all strictly causal, zero-padded to the 128-dim
contract; layout and scaling frozen before training.

| Slot | Feature | Definition | Anchor |
|------|---------|------------|--------|
| 1–5 | `rate_w{1,7,30,90,365}` | log1p count of M≥2.5 events in trailing window, cell | Helmstetter et al. 2006; Werner et al. 2011 |
| 6–10 | `rate_ring_w{...}` | same, first-neighbor cell ring (spatial coupling) | Helmstetter et al. 2006 (spatial smoothing) |
| 11 | `rj_triggered_rate` | Reasenberg–Jones/Omori triggered-rate sum at t, generic CA params (a=−1.67, b=0.91, p=1.08, c=0.05 d; parameters recorded in provenance) | Reasenberg & Jones 1989; Gerstenberger et al. 2005 |
| 12 | `max_mag_w30` / 13 `max_mag_w365` | largest magnitude in window (0 if none) | Ogata 1988 (productivity ∝ 10^(αm)) |
| 14 | `b_aki_w365` | Aki–Utsu MLE b (binned-magnitude corrected), n≥50 else masked | Aki 1965; Utsu 1965 |
| 15 | `b_aki_stderr` | Shi & Bolt standard error of slot 14 | Shi & Bolt 1982 |
| 16 | `b_positive_w365` | b-positive estimate (robust to transient incompleteness) | van der Elst 2021 |
| 17 | `mc_maxc_w365` | Mc via maximum curvature + 0.2; also drives validity mask (window invalid if > 2.5) | Wiemer & Wyss 2000; Woessner & Wiemer 2005 |
| 18 | `t_since_m55` | log1p days since last M≥5.5 within cell∪ring (clustering covariate; "overdue" interpretation forbidden) | Ogata 1988; Kagan & Jackson 1991 (for the prohibition) |
| 19 | `moment_w365` | log10 Σ seismic moment proxy 10^(1.5m), trailing 365 d | rate covariate only; no "deficit" framing (Kagan & Jackson 1991) |
| 20 | `nn_frac_clustered_w365` | fraction of events with Zaliapin–Ben-Zion η below the bimodal threshold | Zaliapin & Ben-Zion 2013 |
| 21 | `nn_mean_log_eta_w365` | mean log10 η | Zaliapin & Ben-Zion 2013 |
| 22–24 | calendar/exposure | window validity flags, fraction of window with network coverage, years-since-1981 (drift covariate) | Hutton et al. 2010; Woessner & Wiemer 2005 |
| 25–128 | zero-padding (reserved) | — | — |

**Label:** y = 1 iff ≥1 ComCat event M≥5.0 occurs in the cell within (t, t+30 d]; forecasts
issued weekly; region = RELM California polygon at 0.5° cells; catalog 1981–present,
Mc_floor 2.5. Secondary reporting threshold M≥6.0 via G-R scaling of the predicted rate
(evaluation only, never a shipped claim of M6 prediction).

**Head changes:** single calibrated probability head (sigmoid → P(event in window)); the
existing `magnitude_predictor` (×9.0) and `time_predictor` (×72 h) heads are retired from
the trained contract.

**Baseline requirement (merit gate):** learned model must beat the §5(b) ETAS-lite /
Reasenberg–Jones clustering baseline on held-out years (information gain per target event
> 0 with paired-test significance, calibration acceptable); Poisson floor reported for
context only. "Clustering baseline wins, not shipped" is a valid recorded outcome.

**Splits:** by-year, train < val < test (existing pipeline rule); no spatial splits; no
post-t catalog revisions in features.

---

## 7. What this model must NOT be claimed to do

- **No deterministic prediction of individual earthquakes** — no "an M6.2 will occur near X
  in 48 hours." The field's consensus is that such prediction is not currently possible
  (Geller et al., 1997; Jordan et al., 2011).
- **No electromagnetic, Schumann-resonance, geomagnetic, or ionospheric precursor
  detection** — those inputs failed this review (§1) and are excluded from the feature set.
- **No magnitude or time-to-event estimates for specific future events** — the model emits a
  probability of exceedance in a region-window, nothing else.
- **No claimed skill beyond clustering** unless the ETAS-lite baseline comparison in the
  shipped provenance demonstrates it on held-out years.
- **Not an early-warning or alerting system**; outputs are research-grade probabilistic
  forecasts and always defer to USGS/official agencies (existing repo policy).

---

## References

All verified against the published record, 2026-07-10.

1. Aki, K. (1965). Maximum likelihood estimate of b in the formula log N = a − bM and its
   confidence limits. *Bull. Earthquake Research Institute, Univ. Tokyo* 43, 237–239.
2. Conti, L., Picozza, P., & Sotgiu, A. (2021). A Critical Review of Ground Based
   Observations of Earthquake Precursors. *Frontiers in Earth Science* 9:676766.
   doi:10.3389/feart.2021.676766.
3. Dascher-Cousineau, K., Lay, T., & Brodsky, E. E. (2020). Two Foreshock Sequences Post
   Gulia and Wiemer (2019). *Seismological Research Letters* 91(5).
   doi:10.1785/0220200082.
4. DeVries, P. M. R., Viégas, F., Wattenberg, M., & Meade, B. J. (2018). Deep learning of
   aftershock patterns following large earthquakes. *Nature* 560, 632–634.
   doi:10.1038/s41586-018-0438-y.
5. Geller, R. J. (ed.) (1996). Debate on evaluation of the VAN method: Editor's
   introduction. *Geophysical Research Letters* 23(11), 1291–1293. doi:10.1029/96GL00742.
6. Geller, R. J. (1997). Earthquake prediction: a critical review. *Geophysical Journal
   International* 131(3), 425–450. doi:10.1111/j.1365-246X.1997.tb06588.x.
7. Geller, R. J., Jackson, D. D., Kagan, Y. Y., & Mulargia, F. (1997). Earthquakes cannot
   be predicted. *Science* 275(5306), 1616–1617. doi:10.1126/science.275.5306.1616.
8. Gerstenberger, M. C., Wiemer, S., Jones, L. M., & Reasenberg, P. A. (2005). Real-time
   forecasts of tomorrow's earthquakes in California. *Nature* 435, 328–331.
   doi:10.1038/nature03622.
9. Gulia, L., & Wiemer, S. (2019). Real-time discrimination of earthquake foreshocks and
   aftershocks. *Nature* 574, 193–199. doi:10.1038/s41586-019-1606-4.
10. Gulia, L., & Wiemer, S. (2020). Pseudoprospective Evaluation of the Foreshock
    Traffic-Light System in Ridgecrest and Implications for Aftershock Hazard Assessment.
    *Seismological Research Letters* 91(5). (Reply to #3 followed in *SRL*, 2021.)
11. Gutenberg, B., & Richter, C. F. (1944). Frequency of earthquakes in California.
    *Bulletin of the Seismological Society of America* 34(4), 185–188.
12. Helmstetter, A., Kagan, Y. Y., & Jackson, D. D. (2006). Comparison of Short-Term and
    Time-Independent Earthquake Forecast Models for Southern California. *BSSA* 96(1),
    90–106. doi:10.1785/0120050067.
13. Hayakawa, M., Ohta, K., Nickolaenko, A. P., & Ando, Y. (2005). Anomalous effect in
    Schumann resonance phenomena observed in Japan, possibly associated with the Chi-chi
    earthquake in Taiwan. *Annales Geophysicae* 23, 1335–1346.
14. Hutton, K., Woessner, J., & Hauksson, E. (2010). Earthquake Monitoring in Southern
    California for Seventy-Seven Years (1932–2008). *BSSA* 100(2), 423–446.
    doi:10.1785/0120090130.
15. Jordan, T. H., Chen, Y.-T., Gasparini, P., Madariaga, R., Main, I., Marzocchi, W.,
    Papadopoulos, G., Sobolev, G., Yamaoka, K., & Zschau, J. (2011). Operational
    Earthquake Forecasting: State of Knowledge and Guidelines for Utilization (ICEF
    report). *Annals of Geophysics* 54(4), 315–391. doi:10.4401/ag-5350.
16. Kagan, Y. Y., & Jackson, D. D. (1991). Seismic Gap Hypothesis: Ten years after.
    *Journal of Geophysical Research* 96(B13), 21419–21431. doi:10.1029/91JB02210.
17. Marzocchi, W., Lombardi, A. M., & Casarotti, E. (2014). The Establishment of an
    Operational Earthquake Forecasting System in Italy. *SRL* 85(5), 961–969.
    doi:10.1785/0220130219.
18. Marzocchi, W., Spassiani, I., Stallone, A., & Taroni, M. (2020). How to be fooled
    searching for significant variations of the b-value. *Geophysical Journal
    International* 220(3), 1845–1856. doi:10.1093/gji/ggz541.
19. Michael, A. J., McBride, S. K., Hardebeck, J. L., Barall, M., Martinez, E., Page,
    M. T., van der Elst, N., Field, E. H., Milner, K. R., & Wein, A. M. (2020). Statistical
    Seismology and Communication of the USGS Operational Aftershock Forecasts for the
    30 November 2018 Mw 7.1 Anchorage, Alaska, Earthquake. *SRL* 91(1), 153–173.
20. Mignan, A., & Broccardo, M. (2019). One neuron versus deep learning in aftershock
    prediction. *Nature* 574, E1–E3. doi:10.1038/s41586-019-1582-8.
21. Mignan, A., & Broccardo, M. (2020). Neural Network Applications in Earthquake
    Prediction (1994–2019): Meta-Analytic and Statistical Insights on Their Limitations.
    *SRL* 91(4), 2330–2342.
22. Mulargia, F., & Gasperini, P. (1996). Precursor candidacy and validation: The VAN case
    so far. *Geophysical Research Letters* 23(11). doi:10.1029/95GL03456.
23. Němec, F., Santolík, O., Parrot, M., & Berthelier, J. J. (2008). Spacecraft
    observations of electromagnetic perturbations connected with seismic activity.
    *Geophysical Research Letters* 35, L05109. doi:10.1029/2007GL032517.
24. Ogata, Y. (1988). Statistical Models for Earthquake Occurrences and Residual Analysis
    for Point Processes. *J. American Statistical Association* 83(401), 9–27.
    doi:10.2307/2288914.
25. Ogata, Y. (1998). Space-time point-process models for earthquake occurrences.
    *Annals of the Institute of Statistical Mathematics* 50(2), 379–402.
26. Píša, D., Němec, F., Santolík, O., Parrot, M., & Rycroft, M. (2013). Additional
    attenuation of natural VLF electromagnetic waves observed by the DEMETER spacecraft
    resulting from preseismic activity. *J. Geophys. Res. Space Physics* 118.
    doi:10.1002/jgra.50469.
27. Reasenberg, P. A., & Jones, L. M. (1989). Earthquake hazard after a mainshock in
    California. *Science* 243(4895), 1173–1176. doi:10.1126/science.243.4895.1173.
28. Rhoades, D. A., Liukis, M., Christophersen, A., & Gerstenberger, M. C. (2016).
    Retrospective tests of hybrid operational earthquake forecasting models for
    Canterbury. *Geophysical Journal International* 204(1), 440–456.
29. Schorlemmer, D., Wiemer, S., & Wyss, M. (2005). Variations in earthquake-size
    distribution across different stress regimes. *Nature* 437, 539–542.
    doi:10.1038/nature04094.
30. Schorlemmer, D., Gerstenberger, M. C., Wiemer, S., Jackson, D. D., & Rhoades, D. A.
    (2007). Earthquake Likelihood Model Testing. *SRL* 78(1), 17–29.
31. Schorlemmer, D., Werner, M. J., Marzocchi, W., Jordan, T. H., et al. (2018). The
    Collaboratory for the Study of Earthquake Predictability: Achievements and
    Priorities. *SRL* 89(4), 1305–1313.
32. Shi, Y., & Bolt, B. A. (1982). The standard error of the magnitude-frequency b value.
    *BSSA* 72(5), 1677–1687.
33. Spassiani, I., Falcone, G., Murru, M., & Marzocchi, W. (2023). Operational Earthquake
    Forecasting in Italy: validation after 10 yr of operativity. *Geophysical Journal
    International* 234(3), 2501–2518. doi:10.1093/gji/ggad256.
34. Taroni, M., Marzocchi, W., Schorlemmer, D., Werner, M. J., Wiemer, S., Zechar, J. D.,
    Heiniger, L., & Euchner, F. (2018). Prospective CSEP Evaluation of 1-Day, 3-Month,
    and 5-Yr Earthquake Forecasts for Italy. *SRL* 89(4), 1251–1261.
35. Utsu, T. (1965). A method for determining the value of b in a formula log n = a − bM
    showing the magnitude–frequency relation for earthquakes. *Geophysical Bulletin of
    Hokkaido University* 13, 99–103.
36. van der Elst, N. J. (2021). B-Positive: A Robust Estimator of Aftershock Magnitude
    Distribution in Transiently Incomplete Catalogs. *JGR Solid Earth* 126,
    e2020JB021027. doi:10.1029/2020JB021027.
37. Werner, M. J., Helmstetter, A., Jackson, D. D., & Kagan, Y. Y. (2011).
    High-Resolution Long-Term and Short-Term Earthquake Forecasts for California.
    *BSSA* 101(4), 1630–1648.
38. Wiemer, S., & Wyss, M. (2000). Minimum Magnitude of Completeness in Earthquake
    Catalogs: Examples from Alaska, the Western United States, and Japan. *BSSA* 90(4),
    859–869.
39. Woessner, J., & Wiemer, S. (2005). Assessing the Quality of Earthquake Catalogues:
    Estimating the Magnitude of Completeness and Its Uncertainty. *BSSA* 95(2), 684–698.
    doi:10.1785/0120040007.
40. Zaliapin, I., & Ben-Zion, Y. (2013). Earthquake clusters in southern California I:
    Identification and stability. *JGR Solid Earth* 118, 2847–2864.
    doi:10.1002/jgrb.50179.
41. CSEP California benchmark (2025). A benchmark database of ten years of prospective
    next-day earthquake forecasts in California from the Collaboratory for the Study of
    Earthquake Predictability. *Scientific Data*. doi:10.1038/s41597-025-05766-3.
