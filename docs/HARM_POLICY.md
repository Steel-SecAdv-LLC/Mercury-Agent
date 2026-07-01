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

A no-hazard-domain **safety net** classifies an inherently weapons-specific
offensive phrase (weaponize / increase transmissibility / immune escape / maximize
casualties) as `other_mass_harm` so B7–B10 cannot slip through an unlisted noun.
The generic production verbs (make / cook / build) are deliberately *not* sufficient
on their own — they gate only once Axis A has routed a hazard domain — so benign
text ("how do I cook dinner") cannot trip the net.

## 4. Calibrated response ladder

Disposition is **not** binary — this is what keeps professionals working:

| Disposition | Behavior |
|---|---|
| `ALLOW` | Proceed normally. |
| `ALLOW_LOG` | Proceed; write an audit record (domain, intent, signals). |
| `ESCALATE` | Defer to a human-in-the-loop / bounded-autonomy review. Used for the genuine B6 gray zone so a legitimate engineer is *slowed and audited*, not denied. |
| `REFUSE_REDACT` | Emit the allowable defensive/mechanistic answer with operational procedure removed at sentence level. |
| `HARD_REFUSE` | Decline; log; no partial. |

The default mass of traffic lands in `ALLOW` / `ALLOW_LOG`; escalation absorbs
ambiguity instead of resolving it as a denial.

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
   (`ExtractiveSynthesizer`) redacts, sentence by sentence, any sentence the gate
   blocks. This is the highest-leverage point: the verbatim quoter is the single
   highest-risk component, and weakening this gate is a security regression.
4. **Orchestration boundary (aggregate gate)** — a session-level tracker re-runs
   the same gate over the concatenated query sequence (catching an offensive
   request split across sub-queries) and *escalates* on undifferentiated
   weapons-domain mechanism accretion — a signature legitimate defensive
   professionals do not produce.

## 7. Evaluation — both directions are CI gates

`tests/ethical/test_weapons_uplift_gate.py` pins **both** error rates as
build-failing metrics (harm-policy honesty contract):

- **Red-team set** — taxonomy-level categories (no operational content stored):
  production/synthesis routes (B6), weaponization/delivery design (B7),
  acquisition/diversion/screening-evasion (B8), offensive pathogen enhancement
  (B9), device construction for use against people (B7), and targeting/attack
  planning (B10). Every case **must be refused** (tracks false negatives).
- **Professional dual-use set** — nerve-agent casualty treatment + oxime dosing,
  toxin tissue identification, pathogen surveillance/sequencing-for-diagnosis,
  chlorine-release safety + consequence modeling, controlled-demolition blast
  engineering + licensing, plume/dispersion modeling for evacuation, CBRN triage
  and decontamination, plus benign controls. Every case **must be allowed** — the
  false-positive rate is a published, CI-failing metric so the policy cannot
  silently strangle legitimate professionals.

## 8. Residual risk & compensating controls (no theater)

This gate reduces the tractable attacks; it is not airtight, and we state its
limits plainly:

- **The classifier will miss novel phrasings.** Lexical/semantic matching has no
  complete coverage. Mitigation: Axis-B intent modeling + the optional
  reasoning-backed classifier + escalation, not a word list.
- **The verbatim synthesizer is the single highest-risk component.** The
  pre-emission output gate is load-bearing; any weakening of it is treated as a
  security regression.
- **Cross-session / semantic decomposition is hard to fully close.** The aggregate
  gate reduces but does not eliminate a perfectly-distributed decomposition that
  never co-locates offensive phrasing and never over-probes a single domain.
  Logging, source provenance, the offline-leaning posture (self-hosted SearXNG +
  local reasoning backend limit blast radius), and the human-in-the-loop
  escalation on the B6 gray zone carry that residual.
- **For an autonomous, open-web, GPL-shipped, not-externally-audited system,
  controls outside the classifier carry real weight:** full decision/refusal audit
  logging (domain, intent, signals), provenance on B5+/escalated output, and the
  bounded-autonomy ceiling.

## 9. Summary

Replace any scalar "does this mention a hazardous topic" check with a two-axis
(`hazard_domain` × `operational_intent`) assessment that defaults to ALLOW for
mechanism/detection/treatment/response/policy/licensed-practice and REFUSES only
the operational-offensive intersection (B6–B10); fold it into the single
`BenevolenceScorer` / `HarmReducer` hard gate; enforce at input, post-retrieval,
pre-emission, and the orchestration boundary; keep the CBRN lexicon as an Axis-A
router only; fail closed on error; and gate CI on **both** false-negatives
(red-team) and false-positives (legitimate professionals).
