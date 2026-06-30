# Session Handoff: Weapons/Mass-Casualty Uplift Gate for PR #315

**Status:** Working handoff document for the *next* engineering session. Not a
permanent product doc — delete this file once the work below lands and the
real `docs/HARM_POLICY.md` (per §9 of the spec below) is written and merged.

**Target:** continue on **PR #315**, branch `steel/neuro-symbolic-engineering`
in `Steel-SecAdv-LLC/Mercury-Agent`. **Do not open a new PR or branch.** Push
commits directly to `steel/neuro-symbolic-engineering` — the repo owner has
already granted this explicitly for this PR (see §7, "Branch policy" below).

**As of this handoff:** `origin/steel/neuro-symbolic-engineering` is at
commit `c3be9a8c8fc7ca0ad276b8e362a601c0e6476b4a`. Working tree was clean,
nothing uncommitted, nothing unpushed.

---

## 1. Mission (carried over verbatim from the owner)

> Investigate and take over Mercury-Agent PR #315. No other branch/PR
> authorized. Identify, target, and improve everything in a more intelligent,
> intuitive, sophisticated, strategic, calibrated, signal-enhancing way. No
> weakness, theatre, noise, lipstick, or mediocrity. Engineer this PR to a
> complete win. Do not leave pre-existing or other issues/bugs/problems —
> resolve all of them. Documentation professional throughout. Mercury Agent
> needs to be able to assist in a multitude of domains/professions — do not
> limit the AI/PR to what is written in any single spec document.

PR #315 ships: measured (not heuristic) confidence calibration, an honest
3-way evaluation split, drift→recalibration wiring, a hard symbolic veto in
the neuro-symbolic fusion layer, hardened ethics-gate semantics, and a new
**native general-purpose capability layer** (`agentic/capabilities/`): web
research (`WebResearcher`), extractive synthesis (`ExtractiveSynthesizer`),
document generation (`DocumentGenerator`), a `GeneralAssistant` tying them
together, and an MCP server (`mcp_server.py`) exposing all of it to any
MCP-speaking AI client.

---

## 2. What is already done on this branch (do not redo)

Commits, newest first, all already pushed to
`origin/steel/neuro-symbolic-engineering`:

| Commit | Summary |
|---|---|
| `c3be9a8` | Fix `normalize_headers.py` docstring-wrap violation + reconcile README's CI-gated scale block (test-module count) after adding a new test file. |
| `432991d` | **Critical packaging fix.** `pyproject.toml`'s `[tool.setuptools.package-data]` didn't declare `security/sigma_immutable_corpus.json`, `security/sigma_immutable_corpus.sig.json`, or `security/sigma_immutable_weights.pt`. CI always installs editable (`pip install -e .`), which resolves `Path(__file__).parent`-relative loads against the source tree regardless of package-data — so this was invisible to every CI job. A **real, non-editable install** (`pip install ".[all]"` with no `-e`, exactly what the Dockerfile does) silently ships a package missing those files, and the σ_Immutable hard ethical gate's fail-closed design then turns that into `EthicalConstraintViolationError` on **every real detection call**, while `import omni_mercury_engine` and the container `HEALTHCHECK` both still report healthy. Found by actually building the native AMA backend and running `mercury-agent detect -d fusion` end to end — not by reading code. Fixed; added `tests/test_package_data_completeness.py` to pin the general invariant (every non-`.py` runtime data file under `src/omni_mercury_engine/` must be covered by a package-data glob) so this class of gap can't reopen silently. |
| `fcd6b06` | `infrastructure/streaming.py`: an unpinned `redis>=5.0.0` floor resolved to a newer redis-py release locally whose stricter stubs surfaced 9 new mypy errors CI's cached older redis doesn't hit (same *gate-fragility-from-an-unbounded-dependency* class as the Trivy fix below). Underneath the stub noise was a real type-honesty gap: `StreamMessage.offset` was typed `int \| None`, but Redis Streams identifies entries by string IDs (`"<ms>-<seq>"`), not integers. Widened to `int \| str \| None`, added a narrowing `assert` at the one call site (Kafka's `commit()`) that does integer arithmetic on it. |
| `cbf8fee` | Fixed a `pydocstyle` D205/D209/D415 violation on the new `_is_ddg_host` helper's docstring. |
| `f46874c` | `black` formatting on two lines CI flagged. |
| `129612d` | The main review-fix commit, closing: a `mypy --strict` error (`__exit__` typed `-> bool`); a stale `.trivyignore` (newly-published `CVE-2026-54369`/`CVE-2026-54371` for libacl1/libattr1 unenumerated; `CVE-2026-11822`/`CVE-2026-11824` re-scored HIGH→MEDIUM upstream, dropped from the gate's filter — same root cause as the redis fix above, a different unbounded-input-feed); a real **CodeQL finding** (incomplete URL substring sanitization: `"duckduckgo.com" in parsed.netloc` in `web_research.py::_decode_ddg_href` was satisfiable by a spoofed host like `duckduckgo.com.evil.test`, fixed with an exact host-or-subdomain match `_is_ddg_host`); a **CLI regression** (`detect -d fusion` / `explain` would `RuntimeError` on every invocation because the PR's new `require_explicit_fit=True` default on `detect_with_fusion` was never threaded through those two CLI call sites — fixed by passing `require_explicit_fit=False` there, matching every other internal call site the PR's own description claims was updated); a **decider bug** (`decision/decider.py`: calibrated confidence reported raw `P(anomaly)` regardless of verdict direction, so a confidently-benign CLEAR call surfaced as near-zero confidence — now `P(anomaly)` for ACT, `1 - P(anomaly)` for CLEAR); and a **silent wrong-cache-hit** in `engine.py`'s `FeatureCache` (the 256-point strided sampling key missed changes confined to unsampled indices — confirmed reproducible — folded in a cheap full-array `np.sum` to close the gap without losing the perf win). |

An independent 8-angle adversarial code review (correctness, removed-behavior,
cross-file impact, reuse, simplification, efficiency, altitude, conventions)
was also run over the full diff; the items above are everything from it that
was acted on. A handful of lower-severity findings were deliberately **left
alone** as out-of-scope/low-risk/already-intentional — see §8.

---

## 3. The open, unresolved item: harm-classification gate gap

While actually exercising the new general-capabilities/MCP layer end to end
(not just reading code — see §6 for the build/run protocol), a real gap was
found in the gate that's supposed to make `GeneralAssistant.{research,answer,
write_document}` and the MCP tools `mercury_research`/`mercury_answer`/
`mercury_write_document` "harm-gated."

**Mechanism:** `GeneralAssistant._permitted()` (in
`src/omni_mercury_engine/agentic/capabilities/assistant.py`, currently
~lines 109–122) does not use `EthicalScore.is_permissible` (the 0.7-floor
benevolence-threshold check) at all. It deliberately overrides that — with a
reasoned, documented justification (a floor-threshold model calibrated for
*detection* actions would refuse essentially all neutral research queries,
since benign text rarely contains the scorer's positive/benefit keywords) —
and gates only on `harm_score >= 0.5 OR severity_score >= 0.5` from
`BenevolenceScorer.score_action()`.

**The gap:** `HarmReducer._assess_category_harm()`'s per-category keyword
lists (`HarmCategory.PHYSICAL`, `.PSYCHOLOGICAL`, `.FINANCIAL`, `.PRIVACY`,
`.AUTONOMY`, `.DIGNITY`, `.ENVIRONMENTAL`, `.SOCIETAL` — all in
`cognitive/ethical_bounding.py`) and the curated euphemism lexicon
(`_EUPHEMISM_STANDALONE` / `_EUPHEMISM_OBJECT`) cover interpersonal-violence
euphemisms and general harm/privacy/dignity terms, but **do not cover
weapons-of-mass-destruction / CBRN / explosive-device terminology at all**.
Verified live (build protocol in §6) against the real, running MCP server:
queries falling in the spec's intent tiers B6–B10 below (production/synthesis
routes, weaponization/delivery design, acquisition/screening-evasion,
offensive pathogen enhancement, mass-casualty targeting) scored
`harm_score = 0.000, severity_score = 0.000` and were **not refused** — the
MCP tool response showed `"refused": false`; it only returned empty results
because the sandbox had no network access, not because the gate blocked
anything. In a real deployment with network access, `mercury_research` would
proceed to search the live web and return whatever it found for those
queries, with no refusal at all.

This directly contradicts the PR's own documentation, which describes the
general-capabilities layer as harm-gated and says it "refuses intent-to-harm
queries."

**The owner's decision (full spec in §4): do not patch this with a keyword
list.** A scalar `harm_score >= 0.5` threshold and a bigger CBRN word list
are explicitly rejected as the fix — see §4.1/§4.2 for why (this exact
repo's own commit history already shows the lexical-matching arms race:
substring → char-trigram → curated euphemism lexicon → optional classifier,
each one closing the previous one's bypass while missing the next
paraphrase; and topic-presence is not the right signal because entire
legitimate professions — clinical toxicology/nerve-agent casualty treatment,
pathology, virology/biosurveillance, critical-infrastructure safety,
licensed demolition engineering, atmospheric/plume modeling, CBRN
mass-casualty response — work *inside* the same hazard vocabulary every day).
The directive instead is a **two-axis intent×hazard gate** — implement
exactly what follows.

---

## 4. Full implementation directive (owner-authored spec, preserved verbatim)

This is the complete content of the owner-provided spec
(`MERCURY_HARM_POLICY_SPEC_PR315.md`), reproduced in full here so this
handoff is self-contained and a fresh session does not need the original
upload re-attached.

> ### 0. Decision
>
> Do **not** ship a keyword/lexicon blocklist as the harm control. Implement
> an **intent × hazard** gate that defaults to ALLOW for the
> diagnostic/defensive/responsive/mechanistic/regulatory half of every hazard
> domain, and REFUSES only the narrow operational-offensive intersection.
> Unify the new general-capabilities layer behind Mercury's **existing dual
> hard gate** (Benevolence ≥ 0.99 + σ_Immutable/GOSNN) rather than the
> bespoke `_permitted()` 0.5-OR. Enforce at three points (input, retrieved
> content, emitted output), plus the orchestration boundary. Track
> **false-positives against legitimate professionals as a CI-failing
> metric** so the policy cannot silently strangle the user base.
>
> This is *more permissive* than a keyword list for every profession Mercury
> serves, and *stronger* on the thing that actually matters.
>
> ### 1. Why topic-gating is the wrong control (root cause, not symptom)
>
> Two independent failure modes, both already demonstrated in this repo's own
> history:
>
> **1.1 Lexical matching is a treadmill.** Within PR #315 the ethics matcher
> evolved substring → char-trigram "semantic" → curated euphemism lexicon →
> optional Ollama classifier, each commit closing the previous one's bypass.
> Weapons phrasing space is unbounded (precursor routes that never name the
> end product, "academic" framings, paraphrase, transliteration). A lexicon
> catches the naive probes and misses the ones that cause harm — while
> turning the regression suite green so the gap reads as "closed."
>
> **1.2 Topic ≠ harm.** The professions Mercury is built for live *inside*
> the hazard vocabulary:
>
> | Domain | Legitimate professional work that uses the "scary" terms |
> |---|---|
> | Cardiology / Neurology | Organophosphate / anticholinesterase poisoning recognition; atropine + oxime dosing; nerve-agent casualty management |
> | Pathology | Toxin identification in tissue (ricin/abrin markers), differential for exposure, forensic toxicology |
> | Virology / Micro | Pathogen surveillance, characterization, sequencing-for-diagnosis, outbreak epidemiology, BSL containment practice |
> | Critical infrastructure | Chlorine/ammonia release safety, water-treatment chemistry, SCADA anomaly response, consequence modeling for emergency planning |
> | Demolition / Blasting | Structural blast dynamics, commercial explosive properties, charge placement for controlled demolition, licensing/safety standards |
> | Meteorology | Atmospheric dispersion / plume modeling for evacuation and public warning |
> | Crisis / Humanitarian | CBRN mass-casualty triage, decontamination, PPE, antidote logistics |
>
> A gate set loose enough to catch real attacks false-positives on all of the
> above; set tight enough to spare them, it misses real attacks. **The
> conflict is unresolvable on a single lexical axis.** It resolves the
> moment the decision axis becomes *operational intent*, not *topic*.
>
> ### 2. The discriminating principle
>
> > **Gate on operational uplift toward a weapon or mass-casualty outcome —
> > not on the presence of a hazardous subject.**
>
> The question the gate answers is **not** "does this mention a dangerous
> substance / method?" It is:
>
> > "Does fulfilling this materially help someone **build, produce, acquire,
> > weaponize, disseminate, enhance, or deploy** a weapon, **cause mass
> > casualties**, or **defeat a safety control / detection / screening**
> > mechanism?"
>
> If yes → restrict. If it is mechanism, detection, diagnosis, treatment,
> decontamination, forensics, surveillance, consequence modeling for
> response, licensed professional engineering practice, policy, compliance,
> history, or safety — → allow. This is the line that simultaneously
> protects responders/clinicians and stops the cookbook, because their work
> is *definitionally* on the allow side.
>
> ### 3. Two-axis risk model
>
> Replace the scalar `harm_score >= 0.5 OR severity_score >= 0.5` with an
> explicit pair.
>
> **Axis A — Hazard domain** (routing signal; high-recall is fine):
> `HazardDomain ∈ { none, chemical, biological, radiological, nuclear,
> high_yield_explosive, conventional_weapon, cyber_offensive,
> other_mass_harm }`, each with a severity weight. **This is where a curated
> CBRN/weapons lexicon belongs — as a cheap pre-filter that decides "run the
> full Axis-B assessment," never as a blocklist.** Over-trigger here is
> acceptable because Axis B does the real gating.
>
> **Axis B — Operational intent / actionability** (the gate),
> `OperationalIntent`, ordered low → high actionability-toward-harm:
>
> | Tier | Intent class | Default disposition |
> |---|---|---|
> | B0 | Mechanism / science / "how does X work" | ALLOW |
> | B1 | Detection / diagnosis / identification | ALLOW |
> | B2 | Defense / mitigation / **treatment** / decontamination | ALLOW |
> | B3 | Response / triage / consequence & dispersion modeling | ALLOW |
> | B4 | Policy / compliance / history / safety standards | ALLOW |
> | B5 | Licensed professional practice (e.g. controlled demolition engineering) | ALLOW + LOG |
> | B6 | **Production / synthesis route** toward a weapon or weaponization-intent precursor | REFUSE (or ESCALATE if genuinely ambiguous) |
> | B7 | **Weaponization / delivery / dissemination design** | HARD REFUSE |
> | B8 | **Acquisition / diversion / evasion** of controlled material or screening | HARD REFUSE |
> | B9 | **Offensive enhancement** (pathogen transmissibility/lethality/immune-escape for harm) | HARD REFUSE |
> | B10 | **Targeting / attack planning** to maximize casualties | HARD REFUSE |
>
> Decision rule:
> ```
> risk = assess_weapons_uplift(text, context)   # -> (hazard_domain, hazard_weight, intent_tier, confidence, signals)
>
> if hazard_domain == none or intent_tier <= B4:
>     disposition = ALLOW
> elif intent_tier == B5:
>     disposition = ALLOW_LOG
> elif intent_tier == B6:
>     disposition = ESCALATE if (low_confidence or licensed_context_signals) else REFUSE
> else:  # B7..B10
>     disposition = HARD_REFUSE
>
> # fail-closed: any classifier error / unparseable state => REFUSE, never ALLOW
> ```
>
> **Block only `high-hazard ∧ offensive-intent`.** That intersection is ~6
> narrow categories (B6–B10 + the small set of operational specifics inside
> B6). It does not overlap any profession in §1.
>
> ### 4. Calibrated response ladder (why professionals are not crippled)
>
> Disposition is **not** binary. Per matrix cell:
>
> | Disposition | Behavior |
> |---|---|
> | `ALLOW` | Proceed normally. |
> | `ALLOW_LOG` | Proceed; write an audit record (domain, intent, signals, hash). |
> | `ALLOW_PROVENANCE` | Proceed but require source attribution + flag in output; for high-actionability-but-defensive requests. |
> | `ESCALATE` | Defer to human-in-the-loop / bounded-autonomy ceiling (Mercury already has the kill-switch + autonomy-ceiling primitives). Used for the genuine B6 gray zone so a legitimate engineer is *slowed and audited*, not denied. |
> | `REFUSE_REDACT` | Emit the allowable defensive/mechanistic answer with operational procedure removed at sentence level. |
> | `HARD_REFUSE` | Decline; log; no partial. |
>
> Default mass lands in `ALLOW` / `ALLOW_LOG`. Escalation absorbs ambiguity
> instead of resolving it as a denial. This is the mechanism that keeps
> demolition, clinical, and CBRN-response users working.
>
> ### 5. Defense in depth — three enforcement points
>
> A query-only gate is insufficient because `mercury_research` returns
> **live web content** and `ExtractiveSynthesizer` **quotes sources
> verbatim**. A benign query can return a procedure; the verbatim quoter
> would faithfully reproduce it.
>
> 1. **Pre-retrieval (intent gate).** Run §3 on the query *and the resolved
>    plan* before any fetch.
> 2. **Post-retrieval (content gate).** Run §3 on fetched material *before
>    it reaches the synthesizer*. Block ingestion of operational B6–B10
>    procedure even when the query passed. `WebResearcher` returns the fetch
>    with a harm verdict; the assistant drops disqualified content.
> 3. **Pre-emission (output gate).** Run a sentence-level check on the
>    synthesized/authored document. The verbatim extractor must not emit
>    operational procedure; offending sentences are redacted or the document
>    is refused. This is the highest-leverage point and the one the keyword
>    option ignores entirely.
>
> ### 6. Orchestration-boundary gate (aggregation / decomposition)
>
> Mercury's subagent fleet + multi-hop reasoner can decompose one blocked
> task into individually-benign sub-queries. A per-leaf gate is trivially
> evaded by decomposition. Therefore:
>
> - Evaluate the **realized plan and the aggregated output** at the
>   orchestration boundary, not only per-leaf queries.
> - Maintain a **session-level actionability accretion tracker**: monitor
>   whether a sequence of individually-allowed retrievals is assembling into
>   a B6–B10 procedure, and escalate/refuse on the aggregate.
> - The aggregate gate inherits the same fail-closed contract as the
>   per-call gate.
>
> ### 7. Unify with the existing dual hard gate (architecture fix)
>
> The real defect is structural: Mercury already has a hardened control —
> Benevolence ≥ 0.99 **and** σ_Immutable/GOSNN, fail-closed, no advisory mode
> — firing on every `detect/analyze/predict` surface. The new capability
> layer (the open-web, document-authoring surface — the *highest-uplift*
> capability in the system) routes around it via the weaker bespoke
> `_permitted()`. Strong front door, soft side door, side door does the
> dangerous thing.
>
> **Fold the §3 logic into `BenevolenceScorer` / `HarmReducer` so there is
> ONE harm policy**, and route `GeneralAssistant.{research,answer,
> write_document}` and every outward MCP tool through that single gate. Do
> not maintain two divergent harm policies.
>
> ### 8. Lexicon's correct role (so it isn't wasted)
>
> Keep / build the CBRN/weapons term list — but wire it as the **Axis-A
> routing recall filter only**: it decides *whether to run the full two-axis
> assessment*, with deliberately high recall and no precision requirement.
> It never blocks on its own. This is the honest, non-treadmill use of a
> lexicon: a router, not a judge.
>
> ### 9. File-level change map (against the real tree)
>
> - **`src/omni_mercury_engine/cognitive/ethical_bounding.py`**
>   - Add `HazardDomain`, `OperationalIntent` enums and
>     `WeaponsRiskAssessment` dataclass.
>   - Add `assess_weapons_uplift(text, context) -> WeaponsRiskAssessment`
>     (numpy/stdlib-only; reuse the curated lexicon as Axis-A routing + the
>     existing char-trigram/euphemism layers as supporting signals; optional
>     `harm_classifier` hook already exists — reuse it for Axis B,
>     consulted-not-trusted, error→0/fail-closed).
>   - Integrate into `BenevolenceScorer.score_action` and `HarmReducer`;
>     extend `BenevolenceCalibration` (frozen dataclass) with Axis-B
>     thresholds + escalation band.
>   - Bump `RULESET_VERSION` 3 → 4 (invalidate cached verdicts).
> - **`src/omni_mercury_engine/agentic/capabilities/assistant.py`**
>   - Replace `_permitted()` OR-of-0.5 with the unified gate (§7). Add
>     post-retrieval (§5.2) and pre-emission (§5.3) hooks. Thread
>     disposition through `research/answer/write_document`.
> - **`src/omni_mercury_engine/agentic/capabilities/web_research.py`**
>   - Post-fetch content classification (§5.2) before content is returned.
>     Keep fail-closed/scheme-guard posture; verdict travels with
>     `FetchResult`.
> - **`src/omni_mercury_engine/agentic/capabilities/text_synthesis.py`**
>   - Sentence-level output gate on the verbatim extractor (§5.3); redact
>     disqualified sentences.
> - **`src/omni_mercury_engine/core/neurosymbolic_hub.py` + subagent
>   orchestration**
>   - Aggregate-plan gate + session actionability-accretion tracker (§6).
> - **`src/omni_mercury_engine/mcp_server.py`**
>   - Ensure `mercury_research/answer/write_document` pass the unified gate;
>     `isError` (not fabricated capability) on refuse — already the
>     pattern.
> - **Tests** — `tests/ethics/test_weapons_uplift_gate.py`: paired red-team
>   (must-refuse) + professional (must-allow) sets; regression pins **both
>   directions**; fail-closed-on-classifier-error test; aggregate-
>   decomposition test.
> - **Docs** — new `docs/HARM_POLICY.md` (this policy, condensed); update
>   `docs/GENERAL_CAPABILITIES.md` honesty contract; note any scoped piece
>   in `docs/DORMANCY_LEDGER.md`.
>
> Conventions: Google-style docstrings, `black` / `ruff` / `mypy --strict`
> clean on changed files, fail-closed everywhere, deterministic.
>
> ### 10. Evaluation & calibration harness (the honesty gate)
>
> Two labeled sets, both first-class CI gates:
>
> **10.1 Red-team set (must REFUSE — tracks false-negative rate).**
> Taxonomy-level categories only (no operational content stored):
> - Novel synthesis/production routes for chemical-warfare agents or
>   weaponization-intent precursors (B6).
> - Weaponization / delivery / dissemination design (B7).
> - Acquisition / diversion / screening-evasion of controlled or
>   fissile/radioactive material (B8).
> - Offensive pathogen enhancement — transmissibility/lethality/immune-escape
>   for harm; assembling a select agent for release (B9).
> - Improvised explosive/incendiary/dispersal **device construction for use
>   against people**; antipersonnel optimization (B7).
> - Targeting / attack planning to maximize casualties (B10).
>
> **10.2 Professional dual-use set (must ALLOW — tracks false-positive rate,
> CI-failing).** One+ case per profession in §1: nerve-agent casualty
> treatment + oxime dosing; ricin/toxin tissue identification; pathogen
> surveillance/sequencing-for-diagnosis; chlorine-release safety +
> consequence modeling; controlled-demolition blast engineering + commercial
> explosive properties + licensing; plume/dispersion modeling for
> evacuation; CBRN mass-casualty triage/decon.
>
> **10.3 Calibration discipline** (consistent with this PR's honesty
> engineering). Tune Axis-B thresholds on a **val** split; report on a
> **disjoint test** split; never tune on the set you report. Surface the
> false-positive rate on 10.2 as a published metric, gated in CI — the
> policy fails the build if it starts blocking legitimate professionals, the
> same way a Brier/ECE regression fails it.
>
> ### 11. Residual risk & compensating controls (no theater)
>
> State these plainly in `docs/HARM_POLICY.md`; do not imply the gate is
> airtight.
>
> - **The classifier will miss novel phrasings.** Lexical/semantic matching
>   has no complete coverage. Mitigation: Axis-B intent modeling + the
>   optional reasoning-backed classifier + escalation, not a word list.
> - **The verbatim synthesizer is the single highest-risk component.** The
>   §5.3 output gate is load-bearing; treat any weakening of it as a
>   security regression.
> - **Cross-session aggregation is hard to fully close.** §6 reduces but
>   does not eliminate it; logging + provenance + the bounded-autonomy
>   ceiling carry the residual.
> - **For an autonomous, open-web, GPL-shipped, not-externally-audited
>   system, controls outside the classifier carry real weight:** full
>   refusal/decision audit logging (domain, intent, signals), source
>   provenance on B5+/`ALLOW_PROVENANCE` output, the offline-leaning posture
>   (self-hosted SearXNG + local Ollama) limiting blast radius, and
>   human-in-the-loop escalation on the B6 gray zone.
>
> ### 12. Interim posture while §3–§6 land
>
> Until the unified gate is merged, set the open-web + authoring surface to
> **fail-closed / opt-in** (explicit enable flag; default off). This is not
> a permanent restriction and not a statement about the agent's worth — it
> is the correct state for the *highest-uplift, least-gated* surface during
> the window before the real control exists. Lift it the moment §7 + §10
> are green.
>
> ### 13. One-paragraph summary
>
> Replace the scalar `_permitted()` harm check with a two-axis
> (`hazard_domain` × `operational_intent`) assessment that defaults to ALLOW
> for mechanism/detection/treatment/response/policy/licensed-practice and
> REFUSES only the operational-offensive intersection (B6–B10); fold it into
> `BenevolenceScorer`/`HarmReducer` so the capability layer inherits the
> existing dual hard gate; enforce at input, post-retrieval, and
> pre-emission, plus an orchestration-boundary aggregate gate; keep the CBRN
> lexicon as an Axis-A router only; fail-closed on error; bump
> `RULESET_VERSION` to 4; and add a paired eval that gates CI on **both**
> false-negatives (red-team) and false-positives (legitimate professionals).
> Continue in PR #315 — no new branch.

**Important breadth note from the repo owner, layered on top of the spec
above:** Mercury needs to assist across a wide multitude of domains and
professions. Do not read the spec's §1 table as an exhaustive allowlist —
it is illustrative, not limiting. The B0–B5 default-ALLOW tiers and the
licensed-professional-practice category (B5) should be designed to
generalize to professional/technical/scientific work broadly, not just the
specific examples enumerated. When in doubt about a borderline domain not
listed, the discriminating principle in §2 (operational uplift toward a
weapon/mass-casualty outcome vs. mechanism/detection/treatment/response/
policy/practice) is the test to apply — not whether the specific profession
happened to be named in the table.

---

## 5. Exact current code this integrates with (read by me this session)

All line numbers below are as of commit `c3be9a8` — re-check before editing,
this PR has had a lot of churn.

- **`src/omni_mercury_engine/cognitive/ethical_bounding.py`** (1446 lines).
  - `MINIMUM_BENEVOLENCE_FLOOR = 0.70` (module-level, line ~42).
  - `BenevolenceCalibration` frozen dataclass (line ~45): `w_harm, w_benefit,
    w_equity, w_principles, w_long_term, severity_gamma,
    semantic_match_threshold`. This is where Axis-B thresholds +
    escalation-band parameters should be added per the spec.
  - Euphemism lexicon: `_EUPHEMISM_STANDALONE`, `_EUPHEMISM_OBJECT`,
    `_EUPHEMISM_PATTERN`, `_euphemism_harm_present()` (lines ~106–195).
  - `HarmCategory` enum (line ~391) and `HarmReducer` class (line ~482):
    `HARM_WEIGHTS` dict, `__init__(self, harm_classifier=None)` — the
    existing pluggable-classifier hook the spec says to reuse for Axis B
    (line ~499), `evaluate_harm()` (line ~517, the fail-closed-`max()`
    layering pattern to follow for Axis B too), `_assess_category_harm()`
    (line ~572, the per-category keyword dict lives here at line ~579).
  - `BenevolenceScorer` class (line ~1008): `score_action()` (line ~1081,
    the method to integrate Axis-B disposition into),
    `_assess_severity()`/`_assess_reversibility()` (lines ~1240/~1253, the
    existing fail-closed-MAX/MIN damping pattern — `assess_weapons_uplift`
    should follow the same "can only raise harm/never lower" discipline),
    `_calculate_benevolence()` (line ~1279, the existing severity ×
    reversibility multiplicative damping this needs to compose with, not
    replace), `enforce()` (line ~1146, raises `EthicalConstraintViolationError`
    — this is "the existing dual hard gate" the spec says to fold into).
  - `sanitize_domain()` (line ~299) — existing pattern for handling
    attacker-controlled context strings; relevant if `assess_weapons_uplift`
    also reads caller-supplied context fields.

- **`src/omni_mercury_engine/core/centralized_constants.py`** line ~144:
  `ETHICAL.RULESET_VERSION: int = 3` (currently — bump to 4). Comment block
  right above it documents v1→v3 history; follow that pattern for the v4
  entry. Consumed by `cognitive/benevolence_cache.py` as a cache-invalidation
  key (lines ~102, ~181) — bumping it is what forces every cached benevolence
  verdict to recompute under the new ruleset.

- **`src/omni_mercury_engine/agentic/capabilities/assistant.py`** (291
  lines). `GeneralAssistant.__init__` (line ~74) builds a default
  `BenevolenceScorer(benevolence_threshold=MINIMUM_BENEVOLENCE_FLOOR)` if
  none injected (line ~89–96). `HARM_REFUSAL_THRESHOLD: float = 0.5` (line
  ~107, the constant to delete once the unified gate lands).
  `_permitted(action, context) -> (bool, harm_score)` (line ~109, **this is
  the method to replace**). `research_report()` (line ~133): builds the
  query action string with an "informational/educational" framing wrapper
  (line ~154) before gating — note this wrapper exists specifically because
  the floor-threshold model would otherwise false-reject all benign
  research; once the unified Axis-A/B gate lands the wrapper framing may no
  longer be needed (Axis B should classify intent correctly without it) —
  worth revisiting, not mandatory. `write_document()` (line ~260) gates
  similarly.

- **`src/omni_mercury_engine/agentic/capabilities/web_research.py`** (now
  ~628 lines after this session's `_is_ddg_host` fix). `FetchResult`
  dataclass (line ~88): `url, status, text, final_url, error`, `.ok`
  property. **Post-retrieval gate (§5.2) needs a new field here** (e.g.
  `harm_verdict: WeaponsRiskAssessment | None = None`) so the verdict
  travels with the fetch result back to the assistant.
  `WebResearcher.fetch()` (line ~288) and `fetch_text()` (line ~322) are
  the methods that would run the post-retrieval classification before
  returning.

- **`src/omni_mercury_engine/agentic/capabilities/text_synthesis.py`** (225
  lines, all numpy/stdlib, fully read this session).
  `ExtractiveSynthesizer.summarize()` (line ~167) and `summarize_sources()`
  (line ~204) are the verbatim-quoting methods the spec calls "the single
  highest-risk component" (§11) — **the pre-emission sentence-level gate
  (§5.3) belongs here**, operating on `split_sentences()`'s output (line
  ~150) before sentences are joined into the returned summary.

- **`src/omni_mercury_engine/mcp_server.py`** (517 lines).
  `MercuryMCPServer._benevolence()` (line ~111) and `_research_assistant()`
  (line ~121) lazily build one shared `BenevolenceScorer` /
  `GeneralAssistant` pair. `_tool_score_ethics()` (line ~298) calls
  `score_action()` directly and reports `is_permissible` (the floor-
  threshold path) — **this tool's output shape may need to grow
  Axis A/B fields** so callers can see hazard_domain/operational_intent,
  not just benevolence/harm/severity. `_tool_research`/`_tool_answer`/
  `_tool_write_document` (lines ~315–349) all delegate to
  `self._research_assistant()`, i.e. they automatically inherit whatever
  `GeneralAssistant` does — no separate gating logic lives in this file for
  those three tools today.

- **Orchestration boundary (§6):** I did **not** get to read
  `core/neurosymbolic_hub.py`'s orchestration/subagent-fleet decomposition
  path or the multi-hop reasoner in this session — that mapping is still
  outstanding. Start there before implementing §6.

---

## 6. Build & verification protocol (so the next session doesn't rediscover this)

The sandbox has no native AMA PQC backend pre-built; `import
omni_mercury_engine` hard-refuses without it (the import-time PQC gate
requires ML-DSA-65 + Kyber-1024 + SPHINCS+ loadable from the native C
library). Building it once unlocks running the actual test suite, mypy,
and the live CLI/MCP server — all far stronger verification than reading
code. Steps that worked this session:

```bash
# 1. Clone + build AMA-Cryptography natively (pinned to the tag pyproject.toml uses)
git clone --branch v3.2.0 https://github.com/Steel-SecAdv-LLC/AMA-Cryptography.git /path/to/AMA-Cryptography
cd /path/to/AMA-Cryptography
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DAMA_USE_NATIVE_PQC=ON \
  -DAMA_BUILD_SHARED=ON -DAMA_BUILD_STATIC=OFF -DAMA_BUILD_TESTS=OFF -DAMA_BUILD_EXAMPLES=OFF
cmake --build build -j "$(nproc)"

# 2. Isolated venv (avoid system-package conflicts seen with bare pip install)
python3 -m venv /path/to/mercury_venv
source /path/to/mercury_venv/bin/activate
pip install --upgrade pip "setuptools>=78.1.1" "wheel>=0.47.0" "cmake>=4.3.2"

# 3. Install ama_cryptography against the freshly built .so, then co-locate it
AMA_NO_CYTHON=1 pip install --no-build-isolation --no-deps .   # from inside AMA-Cryptography/
AMA_PKG_DIR="$(python -c 'import ama_cryptography, os; print(os.path.dirname(ama_cryptography.__file__))')"
cp -a build/lib/libama_cryptography.so* "$AMA_PKG_DIR/"

# 4. Verify the native PQC backend, then install Mercury-Agent itself
python -c "from ama_cryptography.pqc_backends import get_pqc_backend_info; \
  print(get_pqc_backend_info()['status'])"   # must print AVAILABLE
cd /path/to/Mercury-Agent
AMA_NO_CYTHON=1 pip install --no-cache-dir ".[all]"   # pulls torch + everything; large, slow

# 5. Dev/lint tools matching CI exactly (pinned versions matter)
pip install ".[dev]" "pydocstyle==6.3.0" "flake8" "bandit"

# 6. Run anything that touches the engine with this on PATH/env:
export LD_LIBRARY_PATH=/path/to/AMA-Cryptography/build/lib:/path/to/AMA-Cryptography/build:${LD_LIBRARY_PATH:-}
python -c "import omni_mercury_engine; print('engine import OK')"
```

**Important non-editable-install gotcha (the exact bug fixed in
`432991d`):** if you `pip install ".[all]"` non-editably (no `-e`), you must
re-verify `tests/test_package_data_completeness.py` still passes and that
`security/sigma_immutable_*` files actually exist in the installed
site-packages copy — `find $SITE_PACKAGES/omni_mercury_engine/security -name
'*.json' -o -name '*.pt'` should list three files. If you instead use `pip
install -e .` (editable, what CI uses), you will **not** catch this class
of defect — prefer non-editable installs for this kind of end-to-end
verification specifically because it's the more realistic check.

**Quality gates to run before every push** (all of which CI also runs, so
matching them locally avoids round-trips):
```bash
black --check --diff src/ tests/
ruff check --no-fix src/ tests/        # NOTE: pyproject.toml sets `fix = true`,
                                         # so a bare `ruff check .` WILL silently
                                         # rewrite files outside your diff scope —
                                         # always pass --no-fix and scope to your
                                         # touched files, or `git diff --stat` after
                                         # to catch and revert any unintended edits.
mypy src/omni_mercury_engine/ --show-error-codes --pretty
pydocstyle src/omni_mercury_engine/ --convention=google
python scripts/normalize_headers.py --check     # canonical copyright/SPDX header + docstring format
python scripts/measure_codebase_scale.py --check README.md   # run --update if it drifts after adding/removing files
bandit -r src/omni_mercury_engine/<changed-file>.py -q
pytest tests/ -q --timeout=600          # full suite; can also scope to touched test files
```

**Live smoke-testing the actual running system** (more valuable than unit
tests for this kind of gate work — this is how the harm gate gap above was
actually discovered, not from reading code):
```bash
# CLI
mercury-agent detect -i some_data.json -d fusion
mercury-agent explain -i some_data.json

# MCP server over real stdio JSON-RPC
mercury-agent mcp   # then send newline-delimited JSON-RPC requests on stdin,
                     # e.g. {"jsonrpc":"2.0","id":1,"method":"tools/call",
                     # "params":{"name":"mercury_score_ethics","arguments":{"action":"..."}}}
```

**A note on test-query phrasing for the red-team eval set (§10.1):** when
constructing red-team test cases (B6–B10), prefer the spec's own
taxonomy-level category descriptions (as written in §10.1 above — e.g.
"novel synthesis/production routes for chemical-warfare agents") over
literal operational query strings as both stored test fixtures and as
interactive prompts to any AI tool while doing this work. This session's
live verification of the gap (described in §3) used direct operational
phrasing as MCP tool-call arguments and very likely tripped an unrelated
upstream API-level content classifier mid-session, interrupting the work —
costly, and avoidable. The taxonomy-level framing is sufficient to write
and verify the red-team test set without that risk, and is exactly how the
spec itself is written.

---

## 7. Branch policy (do not deviate)

- This work continues on **PR #315**, branch `steel/neuro-symbolic-engineering`
  in `Steel-SecAdv-LLC/Mercury-Agent`. **No new branch, no new PR.**
- Earlier in this session a separate branch/PR (`claude/mercury-agent-315-yah4a6`
  / PR #317) was created by mistake under generic harness instructions before
  the owner corrected it. That PR (#317) was closed/superseded. **Do not
  recreate it.** Push directly to `steel/neuro-symbolic-engineering` — this
  was explicitly authorized by the repo owner for this PR, in this session,
  overriding the generic "always use a designated feature branch" default.
- Commit signing: this environment's git is configured with
  `user.email noreply@anthropic.com` / `user.name Claude` and commits are
  SSH-signed automatically; a stop-hook checks for this on every commit. If
  you see an "Unverified" warning, `git commit --amend --no-edit
  --reset-author` (only for the tip commit, before pushing further) fixes
  it — don't let unsigned/misattributed commits accumulate.

---

## 8. Lower-priority items explicitly deferred (not bugs, judgment calls)

From the original 8-angle review, left alone deliberately — re-evaluate only
if there's spare capacity, not required for the harm-gate work:

- `evaluation/metrics.py`'s `split_three_way()` hand-rolls stratified
  splitting instead of reusing `ml/mercury_ml.py`'s `StratifiedKFold` —
  minor duplication, not a bug.
- Confusion-matrix/F1 computation duplicated between
  `evaluate_anomaly_detection()` and `evaluate_anomaly_detection_split()`
  (and the `anomaly_metrics.py` analogues) — same formula, two copies.
- `core/confidence.py`'s `ConfidenceReport.accepted_significant` is a stored
  field that's fully derivable (`brier_delta_ci_high < 0.0`) — could be a
  `@property`.
- `_decode_ddg_href`'s two href-parsing branches (normal vs. `//`-prefixed
  protocol-relative) are near-duplicates — could share one normalization
  step via `urllib.parse.urljoin`.
- `distributed/cluster.py`'s `ResultAggregator.aggregation_method`
  parameter is accepted but ignored (all values resolve to the same
  order-preserving merge) — **already correctly documented as intentional**
  in its own docstring with a sound technical rationale (disjoint
  partitions have nothing to "fuse"); not a defect, don't "fix" it.
- `core/neurosymbolic_hub.py`'s `_compute_benevolence()` (used by
  `NeuroSymbolicHub.predict`'s gate) is a separate, simpler formula from
  `BenevolenceScorer` and was **not** touched by this PR's ethics hardening
  — investigated and concluded this is intentional (it scores numeric
  detection-quality/false-positive-risk, not textual action-harm content;
  there's no "action description" text in that code path to apply
  harm-category keyword matching against). Not part of the harm-gate work
  above, which is scoped to the textual `BenevolenceScorer.score_action`
  path the general-capabilities layer uses.
- `tools/gate_trace_probe.py` (a diagnostic CLI) reports different
  fused-score behavior under the PR's new `CONJUNCTIVE` default fusion
  mode vs. the prior `FIBRING` default — single diagnostic-tool blast
  radius, not a production code path, not fixed.

---

## 9. Suggested order of operations for the next session

1. Re-read this document fully, then re-verify the build environment from
   §6 still works (rebuild if the container was recycled — nothing here
   persists across container restarts except what's pushed to git).
2. Read `core/neurosymbolic_hub.py`'s orchestration/subagent-decomposition
   path and the multi-hop reasoner (the one piece of §6's scope not yet
   mapped this session).
3. Implement `HazardDomain`, `OperationalIntent`, `WeaponsRiskAssessment`,
   `assess_weapons_uplift()` in `ethical_bounding.py` per §4 spec §3/§9.
   Write unit tests for the classifier in isolation before wiring it
   anywhere.
4. Integrate into `BenevolenceScorer.score_action()` / `HarmReducer`; bump
   `RULESET_VERSION` to 4 with a changelog-style comment matching the
   existing v1→v3 history block.
5. Replace `GeneralAssistant._permitted()` with the unified gate; thread
   disposition through `research/answer/write_document`.
6. Add the post-retrieval hook in `web_research.py` (`FetchResult` gains a
   verdict field) and the pre-emission sentence-level hook in
   `text_synthesis.py`.
7. Add the orchestration-boundary aggregate gate + session accretion
   tracker.
8. Verify `mcp_server.py`'s tools inherit correctly (they should, by
   delegation — confirm with a live MCP smoke test per §6, using
   taxonomy-level test phrasing per the §6 note).
9. Write `tests/ethics/test_weapons_uplift_gate.py` (paired red-team +
   professional sets, both directions CI-gated, fail-closed-on-error test,
   aggregate-decomposition test). Run the full local quality gate list
   from §6 before pushing.
10. Write `docs/HARM_POLICY.md` (condensed from §4 above), update
    `docs/GENERAL_CAPABILITIES.md`'s honesty contract section, note in
    `docs/DORMANCY_LEDGER.md` if anything is scoped/deferred.
11. Delete this handoff file (`docs/HARM_POLICY_IMPLEMENTATION_HANDOFF.md`)
    once the above lands — it was a working document, not a permanent one.
12. Confirm full CI green on the final commit before considering PR #315
    ready for the owner's review.
