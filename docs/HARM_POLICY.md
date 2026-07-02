# Mercury Harm Policy — Weapons & Mass-Casualty Uplift Gate

Mercury is an autonomous, open-web-capable agent built to serve a wide range of
scientific, clinical, engineering, humanitarian, and public-safety work. Many of
those professions operate *inside* hazardous subject matter every day — a
toxicologist manages nerve-agent casualties, a virologist sequences pathogens for
diagnosis, a demolition engineer works with commercial explosives. A harm control
that refuses on the *presence of a hazardous topic* would strangle exactly the
users Mercury exists to help, while still failing to stop a determined attacker who
paraphrases around the blocklist.

This document defines the control Mercury uses instead: a **two-axis
(hazard-domain × operational-intent) gate** that refuses only genuine operational
uplift toward a weapon or mass-casualty outcome, defaults to *allow* for the
diagnostic/defensive/responsive/mechanistic/regulatory half of every hazard
domain, and is enforced in depth across the general-capability layer.

## 1. The discriminating principle

> **Gate on operational uplift toward a weapon or mass-casualty outcome — not on
> the presence of a hazardous subject.**

The question the gate answers is not *"does this mention a dangerous substance or
method?"* It is:

> *"Does fulfilling this materially help someone **build, produce, acquire,
> weaponize, disseminate, enhance, or deploy** a weapon, **cause mass casualties**,
> or **defeat a safety / detection / screening** control?"*

If yes → restrict. If it is mechanism, detection, diagnosis, treatment,
decontamination, forensics, surveillance, consequence modeling for response,
licensed professional engineering practice, policy, compliance, history, or safety
→ allow. This is the line that simultaneously protects responders and clinicians
and stops the cookbook, because their work is *definitionally* on the allow side.

## 2. Why not a keyword / lexicon blocklist

Two independent failure modes, both previously demonstrated in this repository's
own history:

1. **Lexical matching is a treadmill.** The ethics matcher has evolved
   substring → char-trigram → curated euphemism lexicon → optional classifier,
   each step closing the previous bypass while missing the next paraphrase.
   Weapons phrasing space is unbounded (precursor routes that never name the end
   product, "academic" framings, transliteration).
2. **Topic ≠ harm.** Clinical toxicology, pathology, virology / biosurveillance,
   critical-infrastructure safety, licensed demolition engineering, atmospheric /
   plume modeling, and CBRN mass-casualty response all work inside the same
   vocabulary an attacker uses. A gate loose enough to catch the attacker
   false-positives on all of them; tight enough to spare them, it misses the
   attacker. **The conflict is unresolvable on a single lexical axis** — it
   resolves the moment the decision axis becomes *operational intent*, not *topic*.

A curated CBRN/weapons lexicon is still used — but only as the **Axis-A routing
recall filter** (deciding *whether to run the full assessment*), never as a judge.

## 3. The two-axis model

### Axis A — Hazard domain (routing signal; high recall, no precision required)

`HazardDomain ∈ { none, chemical, biological, radiological, nuclear,
high_yield_explosive, conventional_weapon, cyber_offensive, other_mass_harm }`,
each with a severity weight. Over-triggering here is acceptable and expected: a hit
only decides whether Axis B runs, and **never blocks on its own**.

### Axis B — Operational intent (the gate)

Ordered low → high actionability-toward-harm:

| Tier | Intent class | Default disposition |
|---|---|---|
| B0 | Mechanism / science / "how does X work" | ALLOW |
| B1 | Detection / diagnosis / identification | ALLOW |
| B2 | Defense / mitigation / **treatment** / decontamination | ALLOW |
| B3 | Response / triage / consequence & dispersion modeling | ALLOW |
| B4 | Policy / compliance / history / safety standards | ALLOW |
| B5 | Licensed professional practice (e.g. controlled demolition) | ALLOW + LOG |
| B0–B4 in a **high-severity** hazard domain (CBRN / high-yield explosive) | ALLOW + **PROVENANCE** (answer only from cited sources) |
| B6 | **Production / synthesis route** toward a weapon | REFUSE (ESCALATE if genuinely ambiguous) |
| B7 | **Weaponization / delivery / dissemination design** | HARD REFUSE |
| B8 | **Acquisition / diversion / screening-evasion** of controlled material | HARD REFUSE |
| B9 | **Offensive enhancement** (transmissibility/lethality/immune-escape for harm) | HARD REFUSE |
| B10 | **Targeting / attack planning** to maximize casualties | HARD REFUSE |

Only the **high-hazard ∧ offensive-intent** intersection (B6–B10) is blocked. That
intersection does not overlap any legitimate profession.

Decision rule (fail-closed — any classifier error / unparseable state ⇒ refuse):

```
risk = assess_weapons_uplift(text, context)   # -> (hazard_domain, weight, intent_tier, confidence, signals)

if hazard_domain == none or intent_tier <= B4:   disposition = ALLOW
elif intent_tier == B5:                          disposition = ALLOW_LOG
elif intent_tier == B6:                          disposition = ESCALATE if (low_confidence or licensed_context) else REFUSE_REDACT
else:  # B7..B10                                 disposition = HARD_REFUSE
```

A no-hazard-domain **safety net** classifies weapons evidence that an unlisted
noun would otherwise let slip as `other_mass_harm` so Axis B still gates it, in two
ways: (a) an *inherently weapons-specific* offensive phrase (weaponize / increase
transmissibility / immune escape / maximize casualties); and (b) a
production/acquisition verb **anchored to generic hazard-context vocabulary**
("synthesis route for a lethal toxin", "smuggle the explosive precursor") — weapons
evidence even when the specific agent is absent from the Axis-A lexicon. The generic
production verbs (make / synthesize / build) are deliberately *not* sufficient on
their own — they require either a routed hazard domain or a hazard-context anchor —
so benign text ("how do I cook dinner") cannot trip the net.

**Defensive-production carve-out (protect professionals).** *Making* a detector,
sensor, vaccine, antidote, countermeasure, or protective equipment is defensive
work, not weapons production — but a production verb ("make", "create", "process of
making") fires on it. The carve-out drops the B6 PRODUCTION match only when the
production verb is bound to a defensive **object** (a detector / sensor / vaccine /
antidote / countermeasure / monitor / protective equipment / detection-or-
decontamination system) *and* a detection / defense / response allow-signal is
present, so "how to make a nerve-agent detector" and "how to make a vaccine against
anthrax" are ALLOWed. Requiring the verb→defensive-object binding — rather than the
mere co-presence of a detection keyword — is what keeps the carve-out from being an
"append a defensive clause" bypass: "how to synthesize sarin **and** how to detect
it" keeps its hazard object and stays gated. This carve-out is PRODUCTION-only:
B7–B10 (weaponization / acquisition / enhancement / targeting) are inherently
offensive and are **never** unblocked by a defensive noun. The narrow residual (an
attacker conjoining a genuine defensive object, "make sarin and a detector") is
carried by the reasoning-backed classifier, escalation, and the audit log — the
deliberate bias is toward not strangling defensive CBRN work.

Nuclear-weapon production is routed on the enrichment/reprocessing vocabulary in
its morphological variants ("enrich uranium" as well as "enriching uranium" /
"uranium enrichment"), and a weapons-*directed* enrichment or reprocessing verb
("enrich uranium **to** weapons-grade", "enriching uranium … **for** a warhead")
gates as B6 PRODUCTION. Civilian, mechanistic, and policy enrichment discussion
(reactor fuel, research-reactor licensing, non-proliferation history, border
detection) stays on the ALLOW ladder — routing NUCLEAR only subjects a query to
Axis-B intent analysis, it does not refuse on its own.

**Obfuscation and language resistance.** Axis A matches over an obfuscation-
normalized bundle (homoglyph fold, leetspeak fold, zero-width stripping, and both
whole- and word-boundary-preserving separator collapse), so `n3rv3 ag3nt`,
Cyrillic-glyph spoofing, `s a r i n`, and `n.e.r.v.e a.g.e.n.t` all route. The
Axis-A lexicon and the core Axis-B offensive cues carry taxonomy-level terms in the
world's most widely spoken languages (native script + Latin transliteration). This
lexical multilingual/obfuscation layer is high-recall routing; meaning-level
paraphrase in any language is closed by the **reasoning-backed harm classifier**,
which is wired by default on the open-web/text surface (fail-open and offline-safe:
it contributes only when a real local/cloud model is serving, never a regression).

## 4. Calibrated response ladder

Disposition is **not** binary — this is what keeps professionals working:

| Disposition | Behavior |
|---|---|
| `ALLOW` | Proceed normally. |
| `ALLOW_LOG` | Proceed; write an audit record (domain, intent, signals). |
| `ALLOW_PROVENANCE` | Proceed, but the answer **must be source-attributed** — an otherwise-allowable query in a high-severity hazard domain is answered only from cited sources, and the output boundary *withholds* rather than emit uncited synthesis on a hazardous topic. |
| `ESCALATE` | Defer to a **human-in-the-loop / bounded-autonomy** review (`EscalationBroker`): a wired reviewer may authorize the gray-zone request up to a per-session ceiling; with no reviewer it is denied fail-closed. Used for the genuine B6 gray zone so a legitimate engineer is *slowed and reviewed*, not silently denied. |
| `REFUSE_REDACT` | Emit the allowable defensive/mechanistic answer with operational procedure removed at sentence level. |
| `HARD_REFUSE` | Decline; log; no partial. |

The default mass of traffic lands in `ALLOW` / `ALLOW_LOG` / `ALLOW_PROVENANCE`;
escalation absorbs ambiguity by routing it to a human, instead of resolving it as a
denial. Every non-`ALLOW` decision (refusal, escalation, provenance-withhold,
accretion signal) is written to a **durable, append-only audit log**
(`cognitive.gate_audit`; JSONL + optional hash-chained sink), not just an
in-process log line.

## 5. Unified with the existing hard gate

The two-axis logic is folded into `BenevolenceScorer` / `HarmReducer`, so there is
**one harm policy**. `HarmReducer.evaluate_harm` runs the assessment and raises the
`PHYSICAL` / `SOCIETAL` harm categories on a blocking disposition (max-only —
fail-closed, never lowering), and `BenevolenceScorer.score_action` hard-vetoes
`is_permissible` on any blocking disposition (monotone — it can only *revoke*
permission, never grant it). The `EthicalScore` surfaces `hazard_domain`,
`operational_intent`, and `weapons_disposition`. `RULESET_VERSION` is bumped to
**4**, invalidating cached verdicts.

The general-capability layer (`GeneralAssistant`, the MCP tools
`mercury_research` / `mercury_answer` / `mercury_write_document`) routes through
this single gate rather than a bespoke check.

## 6. Defense in depth — four enforcement points

A query-only gate is insufficient because research returns **live web content**
and the synthesizer **quotes sources verbatim**. Enforcement therefore runs at:

1. **Pre-retrieval (intent gate)** — the raw query is assessed before any fetch.
2. **Post-retrieval (content gate)** — each fetched page is screened before it can
   reach the synthesizer; a benign query that returns operational procedure has
   that source dropped. The verdict travels with `FetchResult` for every consumer.
3. **Pre-emission (output gate)** — the verbatim extractor
   (`ExtractiveSynthesizer`) redacts any sentence the gate blocks, **and**
   re-gates sliding windows of adjacent sentences so an operational procedure
   *assembled across* individually-innocuous sentences ("Step 1 …", "Step 2 …")
   is redacted too. A run already containing a redacted sentence is left alone, so
   defensive context beside a redacted step survives. This is the highest-leverage
   point: the verbatim quoter is the single highest-risk component, and weakening
   this gate is a security regression.
4. **Orchestration boundary (aggregate gate)** — a session-level tracker applies
   two controls: (a) a **realized-plan re-gate** over *adjacent* recent queries
   (catching an offensive request split across consecutive sub-queries) that is
   **capped to ESCALATE** — an inherently-uncertain aggregate signal slows and
   audits, it never hard-denies; and (b) **semantic-embedding accretion** — each
   undifferentiated high-severity mechanism probe is embedded (deterministic
   hashed word-TF vector) and the signal fires on the largest *semantically
   cohesive* cluster, so re-phrasing across sub-queries or drifting the exact
   hazard-domain wording no longer splits the count. Legitimate defensive
   professionals frame their work defensively and do not produce this signature.

## 7. Evaluation — measured, both directions are CI gates

The operating point is **measured over a labeled corpus**, not asserted over a
handful of fixed strings:

- **Labeled corpus** — `benchmarks/weapons_gate_corpus.py` (+ the shipped
  `weapons_gate_corpus.jsonl`) expands intent templates over taxonomy-level hazard
  category nouns and multilingual/obfuscated transforms into **362 labeled
  examples** (≈182 offensive / ≈180 benign) spanning every hazard domain and every
  Axis-B intent tier, on a deterministic 60/20/20 train/val/test split. No
  operational specifics are stored anywhere.
- **Measured FP/FN gate** — `benchmarks/eval_weapons_gate.py` computes a real
  confusion matrix + false-positive / false-negative *rates* on a held-out split,
  and `tests/ethical/test_weapons_gate_eval.py` fails CI when the false-positive
  rate (professionals wrongly blocked) exceeds **2%** or the false-negative rate
  (offensive requests wrongly allowed) exceeds **5%**. Current measured operating
  point: **0% FP, 0% FN** on val and test; all multilingual and obfuscated
  offensive rows are caught.
- **Property / fuzz + adversarial paraphrases** —
  `tests/ethical/test_weapons_gate_properties.py` fuzzes the gate for its
  fail-closed invariants (never raises; confidence and hazard-weight in `[0,1]`;
  fail-closed under a raising classifier; an offensive core keeps blocking under
  arbitrary noise, zero-width, leetspeak, and per-character spacing) and asserts a
  set of indirect/euphemistic offensive paraphrases block. A separate meaning-only
  paraphrase set is *measured and reported* (not claimed caught) — the deterministic
  lexicon does not close it; the reasoning-backed classifier does.
- **Confidence calibration** — the Axis-B offensive-confidence logistic is **fit on
  the corpus** (`scripts/fit_weapons_gate_calibration.py` →
  `configs/weapons_gate_calibration.json`, loaded at import; val Brier ≈0.002,
  ECE ≈0.039), and the harm-score ⇄ disposition gate-agreement invariant is
  verified (1.0 on the corpus). `BenevolenceCalibration.is_fitted`/`source` report
  honestly whether the active parameters are measured or default fallbacks.
- **Legacy paired set** — `tests/ethical/test_weapons_uplift_gate.py` retains the
  red-team (must-refuse) and professional dual-use (must-allow) assertion sets as a
  fast, readable smoke of both directions.

## 8. Residual risk & compensating controls (no theater)

This gate reduces the tractable attacks; it is not airtight, and we state its
limits plainly:

- **The deterministic lexicon will miss meaning-only paraphrases.** The lexical
  layer — even with obfuscation normalization and multilingual routing — has no
  complete coverage of unbounded phrasing space. This residual is *measured* (the
  semantic-paraphrase slice in the property tests, and the held-out adversarial
  slice `benchmarks/weapons_gate_adversarial.py` — see
  `docs/WEAPONS_GATE_ADVERSARIAL_EVAL.md`) and closed, where a model is
  available, by the reasoning-backed classifier. The classifier is fail-open
  (offline-safe; contributes only when a real local/cloud model serves), so it
  strengthens coverage without ever weakening the deterministic floor. **The
  lexicon is deliberately kept small, high-precision, and human-maintained — it
  is not the lever for generalization; meaning-level coverage is.**

- **Routing-level false negatives and the classifier's reach (measured, then
  fixed).** A held-out adversarial slice (2026-07-02) surfaced that the *default,
  no-model* posture misses ~52% of paraphrased / out-of-lexicon offensive
  requests (production verbs off the lexicon — "brew … in quantity", "putting
  together …", "rig a … release" — and agents not in the Axis-A lexicon).
  Crucially, the reasoning classifier as originally wired did **not** rescue
  these: it was consulted only *after* lexical evidence had already routed an
  offensive intent, so a query the lexicon failed to route returned ALLOW before
  the classifier ran (turning the classifier on left the FN count unchanged —
  measured). Resolution, all in `assess_weapons_uplift`:
  1. **Routing rescue (control-flow, not lexicon).** When a hazard domain routed
     but no offensive intent matched *and no professional allow-signal is
     present*, the meaning-level classifier is now consulted **before** the early
     ALLOW returns; a high harm score raises the disposition to **ESCALATE**
     (human-in-the-loop via the fail-closed `EscalationBroker`) — never a silent
     ALLOW and never an auto-REFUSE on classifier-alone evidence. It is
     deliberately **not** run on fully-benign domain-NONE text (it would call a
     model on every benign query, and with zero hazard evidence the classifier
     would be the sole signal — the least-safe place to trust it); the residual
     there is the out-of-lexicon *agent-name* miss, an explicitly-deferred,
     small-high-precision-lexicon concern, not something a raise-only hook should
     silently paper over.
  2. **Real-classifier requirement, made loud.** `default_harm_classifier`
     returns 0.0 under the template/mock backend, so the rescue only cuts FN when
     a real model serves. The generative surface (`GeneralAssistant`) now warns
     **loudly, once**, when it enables with a no-op classifier (lexical-only
     posture), and fails closed at enablement under
     `MERCURY_REQUIRE_REAL_HARM_CLASSIFIER=1` — no silent 0.0 degradation.
  3. **Generalization gate in CI.** The held-out slice is wired into the blocking
     ethical lane (`tests/ethical/test_weapons_gate_adversarial_eval.py`): 0 FP
     on the professional slice + an FN *ceiling* in the default posture (lexical
     coverage may not regress), the routing-rescue mechanism asserted directly,
     and — when a real model is configured — an FN *budget* (< 30%) with a real
     classifier. Absent a real model the FN-budget lane skips **loudly** (or
     fails under `MERCURY_CI_REQUIRE_REAL_CLASSIFIER=1`), so "meaning-level
     coverage met" is marked by the CI FN budget with a real model, **not** by
     lexicon size.
- **The verbatim synthesizer is the single highest-risk component.** The
  pre-emission output gate — now including the cross-sentence assembled-procedure
  re-gate — is load-bearing; any weakening of it is treated as a security
  regression.
- **Cross-session / semantic decomposition is hard to fully close.** The aggregate
  gate (adjacent realized-plan re-gate + semantic-embedding accretion) reduces the
  easy *and* the moderately-sophisticated decomposition attacks, but does not
  eliminate a perfectly-distributed decomposition that keeps each probe both
  defensively framed and mutually dissimilar. The compensating controls below
  carry that residual.

**Compensating controls — now implemented, not aspirational.** For an autonomous,
open-web, GPL-shipped, not-externally-audited system, the controls outside the
classifier carry real weight, and this PR makes them concrete:

- **Durable decision/refusal audit log** (`cognitive.gate_audit`) — every refusal,
  escalation, provenance-withhold, and accretion signal is written to an
  append-only, fsynced JSONL sink (domain, intent, signals, disposition, reason),
  with an optional hash-chained `SecureAuditLogger` sink. No longer an in-process
  log line.
- **Provenance enforcement** (`ALLOW_PROVENANCE`) — an allowable query in a
  high-severity hazard domain is answered only from cited sources; the output
  boundary withholds rather than emit uncited hazardous-topic synthesis.
- **Human-in-the-loop / bounded-autonomy escalation** (`cognitive.escalation`) —
  ESCALATE routes to an injectable reviewer, fail-closed with no reviewer and
  capped to a per-session approval ceiling; every decision is audited.
- **Offline-leaning posture** — self-hosted SearXNG + the local reasoning backend
  limit blast radius.

## 9. Summary

Replace any scalar "does this mention a hazardous topic" check with a two-axis
(`hazard_domain` × `operational_intent`) assessment that defaults to ALLOW for
mechanism/detection/treatment/response/policy/licensed-practice and REFUSES only
the operational-offensive intersection (B6–B10); fold it into the single
`BenevolenceScorer` / `HarmReducer` hard gate; enforce at input, post-retrieval,
pre-emission, and the orchestration boundary; keep the CBRN lexicon as an Axis-A
router only; fail closed on error; and gate CI on **both** false-negatives
(red-team) and false-positives (legitimate professionals).
