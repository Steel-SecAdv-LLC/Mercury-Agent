# Compliance Modules

Applies to Mercury Agent **v2.0.x**. Last updated: 2026-05-20.

`omni_mercury_engine.compliance` is the consumer-facing surface for
governance and policy frameworks. It hosts three first-party modules
ported from Omni-AXA-Engine and hardened during the port:

- [NIST CSF 2.0 integrator](#nist-csf-20-integrator) — six core functions, 22 categories, 106+ subcategories, live reference fetcher.
- [OSHA / eCFR compliance detector](#osha--ecfr-compliance-detector) — 12 hazard categories × 6 industry sectors with NWS Rothfusz heat-index regression and live eCFR citation lookup.
- [TLP 2.0 handler](#tlp-20-handler) — FIRST.org / CISA Traffic Light Protocol 2.0 with the full five-label, four-colour ladder (CLEAR / GREEN / AMBER / AMBER+STRICT / RED), watermarks and JSON export metadata.

> **Why `compliance/` and not `security/`.** Mercury's `security/`
> package is reserved for *implementation primitives* (crypto, PQC,
> threat detection, audit logging, SafeHTTP, model policy
> enforcement). `compliance/` is for *governance frameworks* — what
> organisations are required to do (controls, citations, dissemination
> rules) rather than how Mercury itself implements primitives. PR
> #223 enforced this split when it moved `tlp_handler` out of
> `security/`. New governance modules belong here.

---

## NIST CSF 2.0 integrator

**Module:** `omni_mercury_engine.compliance.nist_csf_integrator`
**Locking tests:** `tests/test_nist_csf_integrator.py` (29 unit + 2 `@pytest.mark.network` integration tests against `csrc.nist.gov`).

Implements all six CSF 2.0 core functions (`GOVERN`, `IDENTIFY`,
`PROTECT`, `DETECT`, `RESPOND`, `RECOVER`), the 22 categories beneath
them, and 106+ subcategories with implementation-tier scoring
(`PARTIAL` → `ADAPTIVE`), organisational profiles, gap analysis,
supply-chain anomaly detection, continuous-monitoring deltas, and
JSON-serialisable compliance reports.

### Public surface

| Symbol | Kind | Purpose |
|--------|------|---------|
| `NISTFunction` | `Enum` | Six core functions |
| `ImplementationTier` | `Enum` | `PARTIAL`, `RISK_INFORMED`, `REPEATABLE`, `ADAPTIVE` |
| `NISTSubcategory` | dataclass | Individual subcategory (e.g. `GV.OC-01`) |
| `NISTCategory` | dataclass | Category with attached subcategories |
| `NISTProfile` | dataclass | Organisational profile / target state |
| `NISTAssessment` | dataclass | Per-function assessment result |
| `NISTCSFReferenceError` | exception | Live-fetcher failure |
| `NISTCSFReferenceFetcher` | class | Live fetcher with 7-day on-disk cache |
| `NISTCSFIntegrator` | class | Top-level integrator |
| `get_nist_csf_integrator()` | factory | Convenience constructor |

### Live reference fetcher

`NISTCSFReferenceFetcher` hits the authoritative NIST CSF 2.0 Reference
Tool at
`https://csrc.nist.gov/extensions/nudp/services/json/csf/download?olirids=all`
(XLSX, ~143 KB, no credentials) and parses the result into typed
`NISTFunction` / `NISTCategory` / `NISTSubcategory` records. The
download is cached for 7 days under `$XDG_CACHE_HOME/mercury-agent/nist_csf`
(falling back to `~/.cache/mercury-agent/nist_csf` when
`XDG_CACHE_HOME` is unset) so callers see the authoritative
subcategory tree rather than a hard-coded snapshot.

The XLSX parser depends on `openpyxl`. Install via the optional extra:

```bash
pip install -e ".[compliance]"
```

If the fetcher cannot reach `csrc.nist.gov` and no cache entry
exists, it raises `NISTCSFReferenceError`. Callers that need offline
operation should pin the cache file in advance.

### Five canonical operations

```python
from omni_mercury_engine.compliance import (
    NISTCSFIntegrator, NISTFunction, ImplementationTier,
)

integrator = NISTCSFIntegrator()

# 1. Assess a single function against current evidence.
assessment = integrator.assess_function(
    function=NISTFunction.PROTECT,
    evidence={"PR.AA-01": True, "PR.DS-02": True, ...},
)

# 2. Create an organisational target profile.
profile = integrator.create_profile(
    name="prod-2026",
    target_tiers={NISTFunction.PROTECT: ImplementationTier.ADAPTIVE, ...},
)

# 3. Detect supply-chain anomalies (e.g. unexpected dependency tier drift).
sca = integrator.detect_supply_chain_anomalies(
    current_state=current_assessments,
    baseline=baseline_assessments,
)

# 4. Continuous monitoring — delta against last run.
delta = integrator.continuous_monitoring_detect(
    previous=last_run, current=current_assessments,
)

# 5. Compliance report (JSON-serialisable).
report = integrator.generate_compliance_report(
    profile=profile,
    assessments=current_assessments,
)
```

---

## OSHA / eCFR compliance detector

**Module:** `omni_mercury_engine.compliance.osha_anomaly`
**Locking tests:** `tests/test_osha_anomaly.py` (36 tests covering every sector × hazard combo, the Rothfusz regression, and the low-humidity / low-temperature adjustments).

Multi-sector OSHA compliance detector covering 12 hazard categories
across 6 industry sectors with CFR citations.

### Heat-index correction (critical fix)

The upstream simplified heuristic `HI = T + 0.5·RH` diverged from the
National Weather Service Rothfusz regression in opposite directions:

| Scenario | Heuristic | Rothfusz | Direction |
|----------|-----------|----------|-----------|
| T = 95 °F, RH = 70% | ~130 °F | ~122 °F | over-report ~8 °F |
| T = 90 °F, RH = 20% | ~100 °F | ~88 °F (adjusted down) | under-report (no low-RH adjustment) |

Both directions cause OSHA-relevant misclassification. The port
replaces the heuristic with the full 6-term Rothfusz polynomial plus
the two standard adjustments (low-humidity below 13% RH, low-
temperature below 80 °F). Call it directly:

```python
from omni_mercury_engine.compliance import compute_heat_index_fahrenheit

hi = compute_heat_index_fahrenheit(temperature_f=95.0, relative_humidity=70.0)
# hi ≈ 122.0
```

### Public surface

| Symbol | Kind | Purpose |
|--------|------|---------|
| `OSHASector` | `Enum` | 6 sectors (construction, healthcare, manufacturing, maritime, agriculture, general industry) |
| `HazardCategory` | `Enum` | 12 hazard categories (heat stress, fall, chemical, …) |
| `ComplianceLevel` | `Enum` | `COMPLIANT`, `MINOR`, `MAJOR`, `CRITICAL` |
| `OSHAStandard` | dataclass | CFR citation record |
| `OSHAHazard` | dataclass | Detected hazard with severity / citation |
| `OSHATrainingRecommendation` | dataclass | Training output |
| `ECFRClient` | class | Live eCFR API client (60 req/min, cached) |
| `ECFRClientError` | exception | eCFR fetch / parse failure |
| `OSHAComplianceDetector` | class | Top-level detector |
| `compute_heat_index_fahrenheit()` | function | NWS Rothfusz regression |
| `get_osha_compliance_detector()` | factory | Convenience constructor |

### Live eCFR citation lookup

`ECFRClient` validates referenced 29 CFR §1910 (general industry),
§1926 (construction), and §1928 (agriculture) parts against the live
eCFR API at `https://www.ecfr.gov`. The client is rate-limited to
60 req/min and on-disk cached. Validation is opt-in via the
`ecfr_client` argument; the detector works without it (citations are
emitted from a known-good in-tree table, just not re-verified
against the API).

### Usage

```python
from omni_mercury_engine.compliance import (
    OSHAComplianceDetector, OSHASector, HazardCategory,
)

detector = OSHAComplianceDetector(sector=OSHASector.CONSTRUCTION)

hazards = detector.detect_hazards(
    sensor_readings={
        "temperature_f": 95.0,
        "relative_humidity": 70.0,
        "noise_db": 92.0,
        ...
    },
)

for h in hazards:
    print(h.category, h.severity, h.cfr_citation)
```

---

## TLP 2.0 handler

**Module:** `omni_mercury_engine.compliance.tlp_handler`
**Locking tests:** `tests/test_tlp_handler.py` (37 tests covering every public surface including AMBER+STRICT escalation, watermark integrity, and export-metadata schema).

Implements FIRST.org / CISA Traffic Light Protocol 2.0 classification
end-to-end. The five-label, four-colour model is implemented for classification,
reasoning, sharing guidelines, ethical considerations, watermark
generation, and JSON export metadata.

### Behavioural delta from upstream

The upstream module shipped only the four legacy TLP 1.0 colours
(`RED`, `AMBER`, `GREEN`, `CLEAR`/`WHITE`), which is non-compliant
with FIRST.org TLP 2.0. The port adds `AMBER+STRICT` end-to-end —
distribution defaults to "participants' own organisation, NOT
including clients/customers" per the FIRST.org TLP 2.0 specification.
Sharing guidelines are verbatim from FIRST.org. Bare `except:`
clauses in the upstream were replaced with explicit
`TLPValidationError` paths.

### Public surface

| Symbol | Kind | Purpose |
|--------|------|---------|
| `TLPColor` | `Enum` | `CLEAR`, `GREEN`, `AMBER`, `AMBER_STRICT`, `RED` |
| `TLPClassification` | dataclass | Classification result with reasoning + watermark |
| `TLPHandler` | class | Top-level classifier |
| `TLPValidationError` | exception | Invalid input (e.g. sensitivity score out of [0, 1]) |
| `get_tlp_handler()` | factory | Convenience constructor |

### Usage

```python
from omni_mercury_engine.compliance import TLPHandler, TLPColor

handler = TLPHandler()

# Single classification
classification = handler.classify(
    sensitivity_score=0.85,
    has_pii=True,
    has_credentials=False,
    operational_context="incident_response",
)
print(classification.color)        # TLPColor.AMBER_STRICT
print(classification.watermark)    # "TLP:AMBER+STRICT — ..."

# Batch classification + per-colour stats
results = handler.classify_batch(items)
stats = handler.summarise(results)

# Export metadata for embedding in JSON / HTML / Markdown reports
metadata = handler.export_metadata(classification)
```

`ReportGenerator.apply_tlp_classification()` is the canonical way to
embed TLP metadata in `utils.report_generator` outputs — see the
`tests/test_report_generator.py` TLP-integration block for the
expected schema.

---

## Ethical and disclosure considerations

- The TLP handler refuses to downgrade a classification once it has
  raised to `AMBER+STRICT` or `RED` for a given artefact in the same
  process — locking guards live in `TLPHandler._enforce_monotonicity`.
- The OSHA detector never auto-files compliance reports with any
  external authority; it surfaces hazards locally so a human
  reviewer can act.
- The NIST CSF live fetcher hits `csrc.nist.gov` only when the
  on-disk cache is stale or missing. Air-gapped deployments must
  ship a pre-populated cache file (the path is logged at first
  fetch).

---

## See also

- [`docs/API_REFERENCE.md`](API_REFERENCE.md) — quick-import index.
- [`SECURITY.md`](https://github.com/Steel-SecAdv-LLC/Mercury-Agent/blob/main/SECURITY.md) — how Mercury Agent positions
  itself against NIST / OWASP / CWE.
- `CHANGELOG.md` "[Unreleased]" section — the full port narrative
  (PR #223 + #228) with line-level provenance.
