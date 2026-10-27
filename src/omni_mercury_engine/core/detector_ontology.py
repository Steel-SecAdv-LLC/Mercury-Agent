# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Declared anomaly ontology for every manifest detector.

Most anomaly systems never define what they actually detect. This module
makes the declaration explicit, controlled, and enforced: every entry in
``DETECTOR_MANIFEST`` carries exactly one :class:`DetectorOntology` row
stating which anomaly classes it is sensitive to (``detects``), which it is
known or measured NOT to catch (``cannot_detect``), and in which domains it
has been validated — where a validation claim is admissible **only** with a
committed evidence path (an eval artifact, benchmark result, or competency
test), checked by ``tests/core/test_detector_ontology.py``. A detector with
an empty ``validated`` tuple is not vouched for in any domain, however smart
its description sounds.

The vocabulary is deliberately small and fixed. "What anomaly class is this
detector actually sensitive to?" has to be answerable in one shared
language, or the answer degenerates back into marketing.

Honesty rules used to author these rows (2026-10-27):

* ``detects`` comes from the detector's mechanism, not its ambitions.
* ``cannot_detect`` records *measured* blind spots (e.g. the
  scale-invariant feature maps of ``quantum``/``consciousness`` cannot see
  ``amplitude_collapse`` — pinned in
  ``tests/models/test_fitted_reference_models.py``) and structural ones
  (a change-point statistic does not localise isolated points).
* ``validated`` cites only committed, re-runnable evidence. The 2026-10-27
  full-manifest sweep (``artifacts/detector_evidence/``) is the baseline
  source for generic-data claims; domain claims cite their own artifacts.
* Domain detectors implementing published criteria (SHIP, SPI/SPEI,
  Sepsis-3, Staley 2017, ...) are *criterion-implemented*: the criterion's
  provenance lives in their docstrings, but ``validated`` stays empty until
  a domain evaluation pack (ROADMAP: evidence program) measures them on
  labeled domain data.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Controlled vocabulary of anomaly classes. Extend deliberately; every
#: term here should be meaningful to a practitioner reading an alert.
ANOMALY_CLASSES: frozenset[str] = frozenset(
    {
        "point",  # isolated out-of-pattern samples
        "contextual",  # normal value, abnormal context (season, neighbours)
        "collective",  # a group jointly anomalous, members individually fine
        "change_point",  # abrupt regime/mean/variance change in a stream
        "trend_drift",  # slow degradation / drift of level or slope
        "level_shift",  # sustained step in level
        "variance_change",  # spread change, including variance collapse
        "burst_rate",  # event-rate bursts / self-excitation
        "distributional_outlier",  # row far from the fitted data distribution
        "correlation_break",  # multivariate structure violated
        "sequence_grammar_violation",  # illegal transitions in discrete sequences
        "threshold_exceedance",  # calibrated physical/statistical threshold crossed
        "physical_criterion",  # published domain criterion satisfied (SHIP, qSOFA, ...)
        "spatial_outlier",  # geographic / spatial-relationship anomalies
        "graph_structure",  # graph topology / connectivity anomalies
        "reconstruction_outlier",  # high reconstruction error vs learned manifold
        "amplitude_collapse",  # signal energy collapse (low-side anomaly)
        "fault_signature",  # known physical fault pattern (actuator stuck, bearing)
        "novel_pattern",  # resembles nothing in the reference (novelty)
        "integrity_tamper",  # spoofed / corrupted / tampered inputs or config
    }
)

#: Controlled vocabulary of validation domains.
DOMAINS: frozenset[str] = frozenset(
    {
        "generic_series",  # real labeled generic streams (e.g. NAB)
        "generic_tabular",  # real labeled tabular sets (e.g. ADBench)
        "clinical",
        "telemetry",
        "industrial",
        "security",
        "geophysics",
        "meteorology",
        "space_weather",
        "hydrology",
        "aviation_uav",
        "robotics",
        "finance",
        "marine",
        "visual_industrial",
        "video_surveillance",
        "logs_events",
        "biometrics",
        "hci",
        "energy_grid",
        "epidemiology",
        "chemistry_lab",
    }
)


@dataclass(frozen=True)
class ValidationEvidence:
    """One admissible validation claim: a domain plus its committed proof."""

    domain: str
    evidence: str  # repo-relative path to the committed artifact or test
    note: str = ""


@dataclass(frozen=True)
class DetectorOntology:
    """The declared capability contract of one manifest detector."""

    detects: tuple[str, ...]
    cannot_detect: tuple[str, ...] = ()
    validated: tuple[ValidationEvidence, ...] = ()
    not_validated_note: str = ""


_SWEEP = "artifacts/detector_evidence/2026-10-27_ledger_seed.json"
_FITREF_PINS = "tests/models/test_fitted_reference_models.py"
_BASE_DISC = "tests/detectors/test_base_detector_discrimination.py"
_CRITERION = (
    "Implements a published domain criterion (provenance in its docstring); "
    "no labeled domain corpus evaluation is committed yet, so no domain is claimed."
)
_DOMAIN_PACK = "Awaits its domain evaluation pack; graded generically only."
_UNMEASURED = "No committed evaluation on labeled data yet; nothing is claimed."


def _v(domain: str, evidence: str, note: str = "") -> ValidationEvidence:
    return ValidationEvidence(domain=domain, evidence=evidence, note=note)


_SERIES_SWEEP_T2 = (_v("generic_series", _SWEEP, "NAB streams, 2026-10-27 sweep"),)
_TABULAR_SWEEP_T2 = (_v("generic_tabular", _SWEEP, "ADBench clinical sets, 2026-10-27 sweep"),)

#: One row per manifest entry. Keyed by the manifest ``name``.
ONTOLOGY: dict[str, DetectorOntology] = {
    # ------------------------------------------------------------- BASE
    "statistical": DetectorOntology(
        detects=("point", "distributional_outlier", "threshold_exceedance"),
        cannot_detect=("change_point", "trend_drift", "sequence_grammar_violation"),
        not_validated_note="Sweep mean 0.567 on NAB: ~chance on real streams; engine "
        "fusion carries it, standalone claims none.",
    ),
    "temporal": DetectorOntology(
        detects=("change_point", "level_shift", "trend_drift", "point", "variance_change"),
        validated=(
            _v("generic_series", _SWEEP, "NAB mean 0.642"),
            _v("generic_series", _BASE_DISC, "planted point/variance/level-shift floors"),
        ),
    ),
    "spatial": DetectorOntology(
        detects=("spatial_outlier", "distributional_outlier"),
        validated=_TABULAR_SWEEP_T2,
    ),
    "geo_movement": DetectorOntology(
        detects=("point", "contextual", "trend_drift"),
        cannot_detect=("distributional_outlier",),
        not_validated_note="Trajectory plausibility channels; " + _DOMAIN_PACK,
    ),
    "dimensional": DetectorOntology(
        detects=("correlation_break", "distributional_outlier"),
        validated=(_v("generic_series", _BASE_DISC, "Mahalanobis channel floors"),),
        not_validated_note="Sweep mean 0.538 on 1-D NAB streams: its correlation-break "
        "specialty needs multivariate input.",
    ),
    "directive": DetectorOntology(
        detects=("threshold_exceedance", "distributional_outlier", "point"),
        validated=(_v("generic_tabular", _SWEEP, "ADBench mean 0.901 (1st of all graded)"),),
    ),
    "graph_based": DetectorOntology(
        detects=("graph_structure", "collective"),
        not_validated_note="Needs graph-domain labeled data; " + _DOMAIN_PACK,
    ),
    "kmeans_distance": DetectorOntology(
        detects=("distributional_outlier", "point", "collective"),
        validated=(_v("generic_tabular", _SWEEP, "ADBench mean 0.851"),),
    ),
    "spectral_residual": DetectorOntology(
        detects=("point", "contextual"),
        cannot_detect=("trend_drift", "level_shift"),
        not_validated_note="Sweep mean 0.494 on real NAB streams - at chance; its "
        "saliency formulation needs sharp local deviations.",
    ),
    "bocpd": DetectorOntology(
        detects=("change_point", "level_shift", "variance_change"),
        cannot_detect=("point",),
        validated=(
            _v(
                "generic_series",
                _SWEEP,
                "NAB mean 0.645 under a point-wise "
                "protocol that under-credits change-point statistics",
            ),
        ),
    ),
    "sprt": DetectorOntology(
        detects=("change_point", "level_shift"),
        cannot_detect=("point", "variance_change"),
        not_validated_note="Point-wise AUROC mismatches a sequential decision test; "
        "needs a detection-delay protocol.",
    ),
    "page_hinkley": DetectorOntology(
        detects=("change_point", "level_shift", "trend_drift"),
        cannot_detect=("point",),
        not_validated_note="Same protocol mismatch as sprt.",
    ),
    "spot_evt": DetectorOntology(
        detects=("threshold_exceedance", "point"),
        cannot_detect=("change_point", "trend_drift"),
        not_validated_note="EVT tail thresholding; sweep 0.503 under a protocol that "
        "scores its calibrated threshold as a constant.",
    ),
    "hawkes": DetectorOntology(
        detects=("burst_rate", "collective"),
        cannot_detect=("point", "level_shift"),
        not_validated_note="Event-rate detector driven with value streams in the sweep; "
        "needs count/event-stream evaluation.",
    ),
    "particle_filter": DetectorOntology(
        detects=("point", "change_point", "contextual"),
        not_validated_note="Sweep mean 0.590; below the T2 bar on real streams.",
    ),
    "imm": DetectorOntology(
        detects=("change_point", "contextual"),
        not_validated_note="Sweep mean 0.500 on real streams.",
    ),
    "gaussian_process": DetectorOntology(
        detects=("point", "contextual"),
        not_validated_note="Sweep mean 0.563; below the T2 bar on real streams.",
    ),
    "echo_state": DetectorOntology(
        detects=("point", "contextual", "change_point"),
        validated=(_v("generic_series", _SWEEP, "NAB mean 0.691"),),
    ),
    "spiking": DetectorOntology(
        detects=("point", "burst_rate"),
        not_validated_note="Sweep mean 0.585; below the T2 bar on real streams.",
    ),
    "digital_twin": DetectorOntology(
        detects=("point", "contextual", "trend_drift"),
        validated=(_v("generic_series", _SWEEP, "NAB mean 0.607"),),
    ),
    "survival": DetectorOntology(
        detects=("trend_drift", "threshold_exceedance"),
        cannot_detect=("point",),
        not_validated_note="Hazard-model framing; sweep 0.597 under a point-wise protocol.",
    ),
    "energy_based": DetectorOntology(
        detects=("point", "distributional_outlier", "contextual"),
        validated=(_v("generic_series", _SWEEP, "NAB mean 0.670"),),
    ),
    "deep_svdd": DetectorOntology(
        detects=("distributional_outlier", "point", "novel_pattern"),
        validated=(_v("generic_series", _SWEEP, "NAB mean 0.658"),),
    ),
    "rca": DetectorOntology(
        detects=("correlation_break", "collective", "distributional_outlier"),
        validated=(_v("generic_tabular", _SWEEP, "ADBench mean 0.894 (2nd of all graded)"),),
    ),
    "deeplog_sequence": DetectorOntology(
        detects=("sequence_grammar_violation", "contextual"),
        cannot_detect=("point", "trend_drift"),
        not_validated_note="Needs labeled log corpora (its seam drives a key grammar); "
        + _DOMAIN_PACK,
    ),
    "frequent_pattern": DetectorOntology(
        detects=("sequence_grammar_violation", "collective"),
        cannot_detect=("point", "trend_drift"),
        not_validated_note="Transaction/itemset detector: emits per-transaction verdicts, "
        "no per-point series scores; needs a transaction corpus.",
    ),
    "srcnn": DetectorOntology(
        detects=("point", "contextual"),
        validated=(_v("generic_series", _SWEEP, "NAB mean 0.668"),),
    ),
    "diffusion_ad": DetectorOntology(
        detects=("reconstruction_outlier", "point", "novel_pattern"),
        validated=(_v("generic_series", _SWEEP, "NAB mean 0.648"),),
    ),
    "nano_safeguard": DetectorOntology(
        detects=("point", "burst_rate"),
        not_validated_note="Window-level score only (no per-point contract); ungraded "
        "by the sweep for that reason.",
    ),
    "extended_statistical": DetectorOntology(
        detects=("point", "distributional_outlier", "change_point", "collective"),
        validated=(_v("generic_series", _SWEEP, "NAB mean 0.622"),),
    ),
    "adaptive_anomaly": DetectorOntology(
        detects=("distributional_outlier", "point", "correlation_break"),
        validated=(
            _v("generic_tabular", _SWEEP, "ADBench mean 0.883 (3rd; rev-2 " "harness correction)"),
        ),
    ),
    "topological": DetectorOntology(
        detects=("collective", "novel_pattern", "correlation_break"),
        cannot_detect=("point",),
        not_validated_note="Persistent-homology profile; exceeded the sweep's 90s/dataset "
        "budget on every tabular set - needs a budgeted fast path before grading.",
    ),
    "multivariate_ts": DetectorOntology(
        detects=("correlation_break", "contextual", "collective"),
        not_validated_note="Windowed multivariate contract (windows_3d); " + _DOMAIN_PACK,
    ),
    "info_geometry": DetectorOntology(
        detects=("distributional_outlier", "novel_pattern"),
        validated=(
            _v(
                "generic_tabular",
                _SWEEP,
                "ADBench mean 0.891 (rev-2 harness " "correction: never broken)",
            ),
        ),
    ),
    # ------------------------------------------------------------ MODEL
    "quantum": DetectorOntology(
        detects=("distributional_outlier", "point", "novel_pattern"),
        cannot_detect=("amplitude_collapse",),
        validated=(
            _v("generic_tabular", _SWEEP, "ADBench mean 0.827 post fitted-reference retrofit"),
            _v("generic_tabular", _FITREF_PINS, "burst competency + held-out-normal FAR pins"),
        ),
    ),
    "astrophysical": DetectorOntology(
        detects=("distributional_outlier", "point", "amplitude_collapse", "novel_pattern"),
        validated=(
            _v("generic_tabular", _SWEEP, "ADBench mean 0.803 post fitted-reference retrofit"),
            _v("generic_tabular", _FITREF_PINS, "burst competency both directions"),
        ),
    ),
    "consciousness": DetectorOntology(
        detects=("distributional_outlier", "novel_pattern"),
        cannot_detect=("amplitude_collapse",),
        validated=(
            _v("generic_tabular", _SWEEP, "ADBench mean 0.717 post fitted-reference retrofit"),
            _v("generic_tabular", _FITREF_PINS, "burst competency + held-out-normal FAR pins"),
        ),
    ),
    "biometric": DetectorOntology(
        detects=("distributional_outlier", "point"),
        not_validated_note="Window-level tabular route separates a planted burst (1.000) "
        "but has no per-point contract and no biometric-domain corpus evaluation.",
    ),
    "affective": DetectorOntology(
        detects=("contextual", "threshold_exceedance"),
        cannot_detect=("point", "trend_drift"),
        not_validated_note="Scores declared affective records only; on undeclared series "
        "input it returns the neutral prior (measured constant). Needs an affective corpus.",
    ),
    "neural_cognitive": DetectorOntology(
        detects=("distributional_outlier", "point"),
        not_validated_note="Window-level score separates a planted burst (0.999); no "
        "per-point contract, no EEG-domain evaluation committed (CHB-MIT fixtures exist "
        "for a future pack).",
    ),
    "chemistry": DetectorOntology(
        detects=("physical_criterion", "threshold_exceedance"),
        cannot_detect=("point", "trend_drift"),
        not_validated_note="Parses chemical composition records; series input yields no "
        "elements (measured constant 0). Needs composition corpora.",
    ),
    "parapsychology": DetectorOntology(
        detects=("distributional_outlier",),
        cannot_detect=("point", "trend_drift"),
        not_validated_note="Scores experiment trial records; series input yields no "
        "trials (measured constant). Experimental by its own decision_source.",
    ),
    "copod": DetectorOntology(
        detects=("distributional_outlier", "point"),
        validated=(_v("generic_tabular", _SWEEP, "ADBench mean 0.864"),),
    ),
    "adversarial_autoencoder": DetectorOntology(
        detects=("reconstruction_outlier", "distributional_outlier", "novel_pattern"),
        not_validated_note="Sweep graded one set (wbc 1.00); four timed out at 90s - "
        "needs a budgeted re-grade before any mean is claimed.",
    ),
    "lstm_autoencoder": DetectorOntology(
        detects=("reconstruction_outlier", "contextual", "collective"),
        validated=(_v("generic_series", _SWEEP, "NAB mean 0.829"),),
    ),
    # --------------------------------------------------- NEUROSYMBOLIC
    "neurosymbolic": DetectorOntology(
        detects=("point", "contextual"),
        not_validated_note="Cognitive-stack component; engine-level NAB evidence lives in "
        "artifacts/nab_streaming.eval.json, standalone detector grading pending.",
    ),
    # -------------------------------------------------------- SECURITY
    "threat_detection": DetectorOntology(
        detects=("integrity_tamper", "sequence_grammar_violation", "contextual"),
        not_validated_note="Text/log threat screen; needs labeled security corpora "
        "(NSL-KDD engine results exist at engine level only).",
    ),
    "realtime_threat": DetectorOntology(
        detects=("distributional_outlier", "point", "integrity_tamper"),
        not_validated_note="Training-calibrated ensemble; false-alarm rate pinned in "
        "tests/core/test_declared_modality_false_alarm_rate.py, domain grading pending.",
    ),
    # ---------------------------------------------------- INTELLIGENCE
    "psyop": DetectorOntology(
        detects=("contextual", "collective"),
        not_validated_note=_UNMEASURED,
    ),
    "intelligence_fusion": DetectorOntology(
        detects=("collective", "contextual"),
        not_validated_note=_UNMEASURED,
    ),
    # ----------------------------------------------------------- SPACE
    "schumann_resonance": DetectorOntology(
        detects=("contextual", "variance_change"),
        not_validated_note="Stability tests exist (tests/space/); no labeled corpus.",
    ),
    "solar_storm": DetectorOntology(
        detects=("physical_criterion", "threshold_exceedance"),
        not_validated_note=_CRITERION,
    ),
    "solar_flare": DetectorOntology(
        detects=("physical_criterion", "threshold_exceedance", "burst_rate"),
        not_validated_note=_CRITERION,
    ),
    "cme_arrival": DetectorOntology(
        detects=("physical_criterion",),
        not_validated_note=_CRITERION,
    ),
    "sep_storm": DetectorOntology(
        detects=("physical_criterion", "threshold_exceedance"),
        not_validated_note=_CRITERION,
    ),
    "ionospheric_scintillation": DetectorOntology(
        detects=("physical_criterion", "threshold_exceedance"),
        not_validated_note=_CRITERION,
    ),
    "gic": DetectorOntology(
        detects=("physical_criterion", "threshold_exceedance"),
        not_validated_note=_CRITERION,
    ),
    "solar_gic_cascade": DetectorOntology(
        detects=("physical_criterion", "collective"),
        not_validated_note=_CRITERION,
    ),
    "solar_wind_geomag": DetectorOntology(
        detects=("physical_criterion", "threshold_exceedance"),
        not_validated_note=_CRITERION,
    ),
    "xray_flare": DetectorOntology(
        detects=("physical_criterion", "threshold_exceedance"),
        not_validated_note=_CRITERION,
    ),
    "gic_minute_series": DetectorOntology(
        detects=("physical_criterion", "threshold_exceedance"),
        not_validated_note=_CRITERION,
    ),
    # --------------------------------------------------------- MEDICAL
    "medical_abms": DetectorOntology(
        detects=("contextual", "physical_criterion"),
        not_validated_note=_UNMEASURED,
    ),
    "sepsis": DetectorOntology(
        detects=("physical_criterion", "threshold_exceedance"),
        not_validated_note="Sepsis-3 qSOFA/SOFA implementation; PhysioNet 2019 fixtures "
        "are committed for the clinical pack, evaluation not yet recorded.",
    ),
    "case_surge": DetectorOntology(
        detects=("burst_rate", "trend_drift", "physical_criterion"),
        not_validated_note=_CRITERION,
    ),
    # ------------------------------------------------------ GEOLOGICAL
    "volcanic": DetectorOntology(
        detects=("physical_criterion", "burst_rate"),
        not_validated_note=_CRITERION,
    ),
    "landslide": DetectorOntology(
        detects=("physical_criterion", "threshold_exceedance"),
        not_validated_note=_CRITERION,
    ),
    "wildfire": DetectorOntology(
        detects=("physical_criterion", "threshold_exceedance"),
        not_validated_note="FIRMS live wiring is loader-tested; detector-level labeled "
        "evaluation pending its domain pack.",
    ),
    "tornado": DetectorOntology(
        detects=("physical_criterion", "burst_rate"),
        not_validated_note=_CRITERION,
    ),
    "hurricane": DetectorOntology(
        detects=("physical_criterion", "trend_drift"),
        not_validated_note=_CRITERION,
    ),
    "flood": DetectorOntology(
        detects=("physical_criterion", "threshold_exceedance"),
        not_validated_note=_CRITERION,
    ),
    "earthquake": DetectorOntology(
        detects=("burst_rate", "change_point", "physical_criterion"),
        not_validated_note="STA/LTA + spectrogram; USGS live wiring loader-tested, "
        "labeled seismic-event evaluation pending its domain pack.",
    ),
    "tsunami": DetectorOntology(
        detects=("physical_criterion", "change_point"),
        not_validated_note=_CRITERION,
    ),
    "meteor": DetectorOntology(
        detects=("physical_criterion",),
        not_validated_note=_CRITERION,
    ),
    "avalanche": DetectorOntology(
        detects=("physical_criterion", "threshold_exceedance"),
        not_validated_note=_CRITERION,
    ),
    "rockfall": DetectorOntology(
        detects=("physical_criterion", "trend_drift"),
        not_validated_note=_CRITERION,
    ),
    "subsidence": DetectorOntology(
        detects=("trend_drift", "physical_criterion"),
        not_validated_note=_CRITERION,
    ),
    "eq_tsunami_cascade": DetectorOntology(
        detects=("physical_criterion", "collective"),
        not_validated_note=_CRITERION,
    ),
    "fire_debris_flow_cascade": DetectorOntology(
        detects=("physical_criterion", "collective"),
        not_validated_note=_CRITERION,
    ),
    "tornado_conditions": DetectorOntology(
        detects=("physical_criterion", "threshold_exceedance"),
        not_validated_note=_CRITERION,
    ),
    "hurricane_pressure": DetectorOntology(
        detects=("physical_criterion", "threshold_exceedance"),
        not_validated_note=_CRITERION,
    ),
    "flood_precipitation": DetectorOntology(
        detects=("physical_criterion", "threshold_exceedance"),
        not_validated_note=_CRITERION,
    ),
    # -------------------------------------------------- METEOROLOGICAL
    "pm25_air_quality": DetectorOntology(
        detects=("physical_criterion", "threshold_exceedance"),
        not_validated_note=_CRITERION,
    ),
    "drought": DetectorOntology(
        detects=("physical_criterion", "trend_drift"),
        not_validated_note=_CRITERION,
    ),
    "heatwave": DetectorOntology(
        detects=("physical_criterion", "threshold_exceedance", "collective"),
        not_validated_note=_CRITERION,
    ),
    "atmospheric_river": DetectorOntology(
        detects=("physical_criterion", "threshold_exceedance"),
        not_validated_note=_CRITERION,
    ),
    "lightning": DetectorOntology(
        detects=("physical_criterion", "burst_rate"),
        not_validated_note=_CRITERION,
    ),
    "hail": DetectorOntology(
        detects=("physical_criterion", "threshold_exceedance"),
        not_validated_note=_CRITERION,
    ),
    "winter_storm": DetectorOntology(
        detects=("physical_criterion", "threshold_exceedance"),
        not_validated_note=_CRITERION,
    ),
    "derecho": DetectorOntology(
        detects=("physical_criterion", "collective"),
        not_validated_note=_CRITERION,
    ),
    "dust_storm": DetectorOntology(
        detects=("physical_criterion", "threshold_exceedance"),
        not_validated_note=_CRITERION,
    ),
    "surge_flood_cascade": DetectorOntology(
        detects=("physical_criterion", "collective"),
        not_validated_note=_CRITERION,
    ),
    # -------------------------------------------- ECONOMIC/ENERGY/MARINE
    "financial_crisis": DetectorOntology(
        detects=("contextual", "trend_drift", "collective"),
        not_validated_note=_UNMEASURED,
    ),
    "financial_stress": DetectorOntology(
        detects=("physical_criterion", "threshold_exceedance"),
        not_validated_note=_CRITERION,
    ),
    "emp": DetectorOntology(
        detects=("burst_rate", "physical_criterion"),
        not_validated_note=_UNMEASURED,
    ),
    "marine_biodiversity": DetectorOntology(
        detects=("trend_drift", "collective", "physical_criterion"),
        not_validated_note=_UNMEASURED,
    ),
    # --------------------------------------------------------- UAV/HCI
    "drone": DetectorOntology(
        detects=("fault_signature", "variance_change", "point"),
        validated=(
            _v(
                "aviation_uav",
                "artifacts/alfa_uav.eval.json",
                "ALFA real-fault corpus: 35/36 at 0.68% FPR, 8.0s median delay",
            ),
        ),
    ),
    "uiux_anomaly": DetectorOntology(
        detects=("contextual", "sequence_grammar_violation"),
        not_validated_note=_UNMEASURED,
    ),
    # ---------------------------------------------------------- VISUAL
    "patchcore": DetectorOntology(
        detects=("reconstruction_outlier", "novel_pattern", "point"),
        validated=(
            _v(
                "visual_industrial",
                "tests/test_visual_detectors.py",
                "driven on real image tensors; MVTec-grade pack pending",
            ),
        ),
    ),
    "padim": DetectorOntology(
        detects=("distributional_outlier", "novel_pattern"),
        validated=(
            _v(
                "visual_industrial",
                "tests/test_visual_detectors.py",
                "driven on real image tensors; MVTec-grade pack pending",
            ),
        ),
    ),
    "stfpm": DetectorOntology(
        detects=("reconstruction_outlier", "novel_pattern"),
        validated=(
            _v(
                "visual_industrial",
                "tests/test_visual_detectors.py",
                "driven on real image tensors; MVTec-grade pack pending",
            ),
        ),
    ),
    "reverse_distillation": DetectorOntology(
        detects=("reconstruction_outlier", "novel_pattern"),
        validated=(
            _v(
                "visual_industrial",
                "tests/test_visual_detectors.py",
                "driven on real image tensors; MVTec-grade pack pending",
            ),
        ),
    ),
    "cflow": DetectorOntology(
        detects=("distributional_outlier", "novel_pattern"),
        validated=(
            _v(
                "visual_industrial",
                "tests/test_visual_detectors.py",
                "driven on real image tensors; MVTec-grade pack pending",
            ),
        ),
    ),
    # ------------------------------------------------------------- VLM
    "anyanomaly": DetectorOntology(
        detects=("novel_pattern", "contextual"),
        not_validated_note="External LVLM backend; nothing is claimed without it.",
    ),
    "lavad": DetectorOntology(
        detects=("novel_pattern", "contextual"),
        not_validated_note=_UNMEASURED,
    ),
    # ------------------------------------------------------ FOUNDATION
    "timegpt": DetectorOntology(
        detects=("point", "contextual"),
        not_validated_note="External service adapter; no seams declared, nothing claimed.",
    ),
    "chronos": DetectorOntology(
        detects=("point", "contextual"),
        cannot_detect=("change_point",),
        not_validated_note="Sweep mean 0.371 on real NAB streams (0.11 on AWS telemetry): "
        "the foundation label currently outranks its evidence.",
    ),
    "matrix_profile": DetectorOntology(
        detects=("novel_pattern", "collective", "point"),
        cannot_detect=("level_shift",),
        not_validated_note="Sweep mean 0.356; 0.07 on realTraffic - a repeating anomaly "
        "earns a LOW discord score. Orientation/motif handling needs root-cause before "
        "any claim.",
    ),
    "foundation_ensemble": DetectorOntology(
        detects=("novel_pattern", "point"),
        cannot_detect=("level_shift",),
        not_validated_note="Inherits matrix_profile's measured weaknesses (sweep 0.356).",
    ),
    # ---------------------------------------------------- DISTILLATION
    "dual_student": DetectorOntology(
        detects=("reconstruction_outlier", "novel_pattern"),
        not_validated_note=_UNMEASURED,
    ),
    # --------------------------------------------------------- PHYSICS
    "spectral_vibration": DetectorOntology(
        detects=("fault_signature", "variance_change", "contextual"),
        not_validated_note="Bearing/vibration territory; needs CWRU-style pack "
        "(fixtures exist under tests/fixtures/cwru).",
    ),
    "acceleration_dynamics": DetectorOntology(
        detects=("fault_signature", "point", "variance_change"),
        not_validated_note=_UNMEASURED,
    ),
    "physics_integrated": DetectorOntology(
        detects=("physical_criterion", "distributional_outlier"),
        not_validated_note="Set-level verdict pinned in "
        "tests/core/test_declared_modality_false_alarm_rate.py; domain grading pending.",
    ),
    "spectral_domain_frequency": DetectorOntology(
        detects=("contextual", "variance_change", "point"),
        not_validated_note=_UNMEASURED,
    ),
}


__all__ = [
    "ANOMALY_CLASSES",
    "DOMAINS",
    "ONTOLOGY",
    "DetectorOntology",
    "ValidationEvidence",
]
