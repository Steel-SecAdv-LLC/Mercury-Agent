# Knowledge profiles: whose knowledge every domain and detector is

Applies to Mercury Agent **v3.0.x**. Source of truth:
`src/omni_mercury_engine/core/knowledge_profiles.py`, enforced by
`tests/core/test_knowledge_profiles.py`.

## Why this layer exists

Mercury already declared three things about its detection surface:

| Layer | Module | Says |
|---|---|---|
| Detector manifest | `core/detector_registry.py` (`DETECTOR_MANIFEST`) | what a detector is: module, class, family, input modality, proven seams |
| Anomaly ontology | `core/detector_ontology.py` (`ONTOLOGY`) | which anomaly classes it is sensitive to, which it cannot see, where that was measured |
| Loader contract | `loaders/*_loader.py` class attributes | where a domain's data comes from, whether a key is needed, how labels are derived |

None of them said *whose knowledge it is*: which discipline a method comes
from, who published it, which institution is the authority for the data,
what a practitioner has to know to read the output. Without that, the
console described every loader by its file name, every detector by one
sentence, and the agent answered "what is a detector" with one template for
113 detectors. Twenty-one detectors shared an identical ontology signature
(`physical_criterion` + `threshold_exceedance`), so even the ontology could
not tell them apart.

The knowledge profiles are that fourth layer, authored by hand.

## What a profile holds

**`DomainProfile`** (one per data-loader domain and per `hazard-viz`
domain; 22 distinct profiles for 21 loaders + 9 hazards):

| Field | Meaning |
|---|---|
| `discipline` | the scientific or professional field (seismology, critical-care medicine, ...) |
| `professions` | who works with this data |
| `institutions` | who publishes, governs or defines it: name, role, `https` URL |
| `expertise` | the concepts a practitioner brings, that Mercury encodes |
| `observables` | what the raw data is |
| `method` | how Mercury turns the source into features and labels |
| `label_rule` | the label criterion and its provenance (`ground truth` or `statistical`) |
| `detectors` | the manifest detectors that serve the domain |
| `limits` | what the shipped domain does not cover |
| `references`, `standards` | the literature and the standards |

**`DetectorProfile`** (one per manifest detector, 113):

| Field | Meaning |
|---|---|
| `discipline` | the field the method comes from |
| `lineage` | who published it, when, in which tradition |
| `mechanism` | how it decides, in plain words, as the code does it |
| `practitioners` | who uses it professionally and what they need to know to read the score |
| `use_when` / `not_for` | when to reach for it, and when not to |
| `references` | the citations (DOI or URL where one exists) |

## Authoring rules (all test-enforced)

- Coverage is exact: every manifest detector and every loader/hazard domain
  has a profile; no profile is orphaned (`validate_profiles`).
- No two detector profiles share a mechanism, lineage or practitioner
  description; no two domain profiles share a method, label rule or
  observables sentence. A copied sentence is a defect.
- Every profile cites at least one reference; every domain names at least
  one institution with an `https` URL and a stated role.
- A mechanism adds to the manifest description; it never repeats it.
- An adapter's profile names the detector it serves (`wraps`).
- `use_when` never claims an anomaly class the ontology lists under
  `cannot_detect`.
- A domain's label rule starts with the loader's declared `LABEL_SOURCE`
  (`ground truth` or `statistical`), and a keyed loader's profile says which
  key.
- No placeholder text; measured performance stays in the ontology (this
  layer describes lineage and expertise, which do not change with a
  benchmark run).

## Where it flows

```
core/knowledge_profiles.py ──┬─▶ api/routes/catalog.py   GET /api/v1/catalog/domains[/{domain}]
core/loader_declarations.py ─┤                           GET /api/v1/catalog/detectors[/{name}]
core/detector_registry.py ───┤
core/detector_ontology.py ───┼─▶ scripts/build_console_catalog.py ─▶ static/console/catalog.js
                             │        (loaders[].prof, detectors[].prof, domainProfiles)
                             │        ─▶ console rows (chips, hover notes), reference cards,
                             │           :diag "Knowledge profiles" suite
                             └─▶ narrative/conversation.py  domain_answer / detector_profile_answer
                                  narrative/voice.py        "tell me about the sepsis domain", "what is bocpd",
                                                            "who publishes the earthquake data", "limits of ...",
                                                            "references for ...", "which detectors serve ..."
```

`core/loader_declarations.py` parses each loader's declared contract with
`ast` so none of these consumers imports the loaders package (which pulls in
every domain's dependencies).

## Adding or changing one

1. Edit the entry in `core/knowledge_profiles.py`. Cite real literature;
   describe the mechanism as the code does it.
2. Run `python -m pytest tests/core/test_knowledge_profiles.py
   tests/api/test_catalog_routes.py tests/narrative/test_knowledge_answers.py`.
3. Regenerate the console catalog: `python scripts/build_console_catalog.py`
   (`tests/api/test_console_catalog.py` fails until it is current).
4. A new manifest detector or a new loader fails the coverage test until its
   profile exists; that is intended.
