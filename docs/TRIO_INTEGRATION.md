# FINDOYOU ↔ Mercury ↔ AMA Cryptography: the trio contract

The three Steel Security Advisors products work as one system in service of a
civilization-first mission:

| Repository | Role in the trio |
|---|---|
| **FINDOYOU** | Runs missing-person cases; carries the biometric case workflow. |
| **Mercury Agent** (this repository) | Analyses the images: identity match, age-robust match, progression audit, sighting integrity. |
| **AMA Cryptography** | Signs Mercury's findings with ML-DSA-65 (FIPS 204), so evidence keeps its chain of custody from analysis to case file. |

Mercury hosts the integration for now. FINDOYOU is not modified by this work.

## 1. The shared honesty contract

Every finding carries one of three wire values, defined in
`omni_mercury_engine.verifiers.three_state.ThreeState` and mirrored
member-for-member by FINDOYOU's `security/three_state.py` (verified identical
on FINDOYOU `main` and `steel/systembiom-curation`):

| Value | Meaning |
|---|---|
| `grounded` | The verdict rests on a measurement this installation can make. |
| `unavailable` | The measurement cannot be made here (a missing backend or input); no verdict is invented. |
| `undecidable` | The question has no decidable answer from the evidence. |

## 2. Evidence packets (`mercury.findoyou.evidence/v1`)

`FindoyouEvidenceBridge.analyze_case(case_id, reference, candidate, source_age=None, years_elapsed=None)`
in `src/omni_mercury_engine/integrations/findoyou_bridge.py` returns a packet
with four findings:

- **`identity_match`**: `grounded` *only* when a face-identity embedding
  backend (DeepFace) produced both embeddings. Appearance similarity from the
  harmonic fallback is reported, but stays `unavailable` for identity,
  because a spectral resemblance must never become a lead in a missing-person
  case.
- **`age_robust_match`**: the reference is aged across `years_elapsed` and
  re-matched. It inherits the grounding of `identity_match`.
- **`progression_audit`**: SSIM, injected-noise ratio and lost-detail ratio
  of the progression, so an artefact-laden progression cannot pass silently.
  It is `unavailable` when `years_elapsed` is not given; the gap is never
  guessed.
- **`candidate_integrity`**: forensic screening of the sighting itself, for
  spliced noise levels, JPEG grid misalignment, copy-move duplication and
  screen-replay or print-halftone (moiré) peaks.

Each packet also carries:
- SHA3-256 digests of both images' canonical pixels;
- the producer key;
- three standing limitations, stated in every packet: the analysis supports
  investigators and is never an identification on its own; age progression
  is an investigative aid; forensic screens can miss expert manipulation.

### Signing and verification

The packet is serialised canonically (sorted keys, compact separators, ASCII)
and signed with ML-DSA-65 through AMA's `MLDSAProvider`. Sizes are 1952-byte
public keys and 3309-byte signatures.

`verify_packet(packet, expected_public_key_hex)` checks the signature against
a **pinned** producer key. A key carried inside the packet proves integrity,
not origin. It returns `(ok, reason)` and rejects:
- a changed finding, case id or schema;
- a forged or malformed signature;
- a missing signature block;
- a different producer key.

The tests in `tests/integrations/test_findoyou_bridge.py` exercise each case.

The bridge is an agent surface, so importing it requires the AMA native
backend, like every Mercury surface that signs.

## 3. What FINDOYOU already imports from Mercury

FINDOYOU's `steel/systembiom-curation` branch imports these Mercury symbols:
- `AnomalyMathArrest`
- `evaluate_anomaly_detection`
- `components_from_score_channels`
- `score_runtime_equation_profile`
- `IsotonicRegression`
- `compute_auroc` / `compute_auprc`
- `COPODDetector`
- `ConformalAnomalyDetector`
- `SplitConformalPredictor`
- the `detectors.statistical` and `loaders` modules

Every one resolves on this branch (checked by import on 2026-10-15).

## 4. Mercury's durable self-model

`src/omni_mercury_engine/self_model.py` gives Mercury an internal model of
itself. Every entry is derived from the running installation and never
written as prose:

- **Identity**: name, version, mission (`civilization-first`), code system
  (`Omni Codes`), the trio, and the three-state honesty contract.
- **Capabilities**: seven of them. Each is probed without side effects and
  given a three-state: `visual_anomaly_analysis`, `image_forensics`,
  `biometric_match` (grounded only with DeepFace installed),
  `age_progression`, `findoyou_evidence_bridge`, `post_quantum_signing` and
  `maritime_anomaly_detection`.
- **Safeguards**:
  - whether the AMA native backend is present;
  - the σ_Immutable baseline digest;
  - the retired-terminology guard.
- **Measured performance**: read from the committed benchmark artefacts,
  each pinned by SHA-256: the MVTec AD texture results and the DTU AIS
  maritime row.
- **Limitations**: derived from those measurements (for example, a
  benchmark category with image AUROC below 0.75) and from missing
  dependencies.

It is durable because `SelfModelLedger` appends snapshots to a SHA3-256
hash-chained JSON-lines file, optionally ML-DSA-65-signed:
- `verify()` detects any edit, deletion or reordering, and a broken chain is
  never extended;
- `diff_snapshots()` reports capabilities gained or lost, metric drift,
  safeguard changes and new limitations.

```bash
python -m omni_mercury_engine.self_model                 # print the current snapshot
python -m omni_mercury_engine.self_model --append        # record it (default ~/.mercury/self_model.jsonl,
                                                         #   or $MERCURY_SELF_MODEL_PATH)
python -m omni_mercury_engine.self_model --verify        # verify the chain
```

Tests: `tests/test_self_model.py`.
