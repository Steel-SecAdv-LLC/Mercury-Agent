#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
"""Mercury Agent — Neural-submodule coverage: living, self-verifying CI artifact.

`docs/NEURAL_SUBMODULE_COVERAGE.md` used to be a hand-maintained table. Hand
tables rot: an artifact gets renamed, a test is deleted, a module is removed, and
the table silently keeps claiming coverage that no longer exists. This script
makes the table a **generated, gated artifact**:

* the structured registry below (:data:`COVERAGE_ROWS`, :data:`GATE_ROWS`) is the
  single source of truth;
* ``--update`` regenerates ``docs/NEURAL_SUBMODULE_COVERAGE.md`` from it;
* ``--check`` (the CI/pytest gate) fails if the committed doc is **out of sync**
  *or* if any referenced **module / test file / committed artifact does not
  exist** — so the claimed coverage cannot drift from reality.

Usage::

    python scripts/neural_coverage.py --check    # gate (CI + pytest)
    python scripts/neural_coverage.py --update   # regenerate the doc
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs" / "NEURAL_SUBMODULE_COVERAGE.md"


@dataclass
class Row:
    """One coverage row: display fields + machine-checkable references."""

    module: str
    dataset: str
    seeds: str
    metric: str
    artifact_display: str
    tests_display: str
    contract: str
    status: str
    # machine-checkable references (verified by --check):
    source: tuple[str, str]  # (repo-relative .py, symbol that must be defined)
    test_files: list[str] = field(default_factory=list)
    artifact_path: str | None = None  # committed artifact that must exist
    artifact_ci_generated: bool = False  # if True, absence is OK (made in CI)

    def cells(self) -> list[str]:
        return [
            f"`{self.module}`",
            self.dataset,
            self.seeds,
            self.metric,
            self.artifact_display,
            self.tests_display,
            self.contract,
            f"**{self.status}**",
        ]


# --- Neural submodules in the neuro-symbolic-fusion scope ---------------------
COVERAGE_ROWS: list[Row] = [
    Row(
        module="OmniFusionModel",
        dataset="ADBench (MIT) + domain loaders",
        seeds="42",
        metric="ROC-AUC, oracle-F1",
        artifact_display="`mercury_benchmark_results.json` (CI)",
        tests_display="`test_fusion_*`",
        contract="baseline",
        status="ACTIVE (default)",
        source=("src/omni_mercury_engine/ml/fusion_network.py", "OmniFusionModel"),
        test_files=["tests/test_fusion_raw_path.py", "tests/test_fusion_training.py"],
        artifact_path="mercury_benchmark_results.json",
        artifact_ci_generated=True,
    ),
    Row(
        module="MercuryAnomalyDetector (WS-A guard)",
        dataset="ADBench (MIT), 8 fixed sets",
        seeds="42",
        metric="AUC/F1 floors",
        artifact_display="`anomaly_regression_baseline.json`",
        tests_display="`test_anomaly_regression_guard`",
        contract="deterministic; CI gate",
        status="ACTIVE",
        source=("src/omni_mercury_engine/detectors/statistical.py", "MercuryAnomalyDetector"),
        test_files=["tests/benchmarks/test_anomaly_regression_guard.py"],
        artifact_path="benchmarks/anomaly_regression_baseline.json",
    ),
    Row(
        module="SymbolicConstraintModule (LTN)",
        dataset="ADBench (MIT)",
        seeds="0,1,2",
        metric="ΔAUC, ΔFP (+confound guard)",
        artifact_display="`neurosymbolic_ablation.json`",
        tests_display="`test_symbolic_constraint`, `test_fusion_symbolic_cotraining`",
        contract="`symbolic_weight=0` byte-identical",
        status="QUARANTINE (sub-threshold)",
        source=("src/omni_mercury_engine/ml/symbolic_constraint.py", "SymbolicConstraintModule"),
        test_files=[
            "tests/ml/test_symbolic_constraint.py",
            "tests/test_fusion_symbolic_cotraining.py",
        ],
        artifact_path="artifacts/neurosymbolic_ablation.json",
    ),
    Row(
        module="DomainEncoderStack (WS-B)",
        dataset="ADBench (MIT)",
        seeds="0,1,2",
        metric="ΔAUC (+confound guard)",
        artifact_display="`domain_encoder_ablation.json`",
        tests_display="`test_domain_encoders` (16), `test_fusion_domain_encoder` (5)",
        contract="`domain_encoder=False` parity (≤1e-15)",
        status="QUARANTINE (sub-threshold)",
        source=("src/omni_mercury_engine/ml/domain_encoders.py", "DomainEncoderStack"),
        test_files=["tests/ml/test_domain_encoders.py", "tests/test_fusion_domain_encoder.py"],
        artifact_path="artifacts/domain_encoder_ablation.json",
    ),
    Row(
        module="BinaryConformalClassifier",
        dataset="ADBench + synthetic",
        seeds="fixed",
        metric="coverage @ {0.8,0.9,0.95}",
        artifact_display="in `test_*conformal`",
        tests_display="`test_binary_conformal`, `test_fusion_conformal`",
        contract="additive serve-path",
        status="ACTIVE (uncertainty)",
        source=(
            "src/omni_mercury_engine/core/conformal_prediction.py",
            "BinaryConformalClassifier",
        ),
        test_files=["tests/core/test_binary_conformal.py", "tests/test_fusion_conformal.py"],
    ),
    Row(
        module="SchumannHarmonicAnalyzer (WS-C)",
        dataset="NOAA Kp/GOES (public domain) labels; **synthetic** ELF",
        seeds="0,1,2",
        metric="ROC-AUC",
        artifact_display="`schumann_eval.json`, `schumann_diagnostic.json`",
        tests_display="`test_schumann_labeling` (5), `test_schumann_stability` (5)",
        contract="stable recipe (minibatch); quarantine on data blocker",
        status="QUARANTINE (synthetic signal; training stabilized)",
        source=("src/omni_mercury_engine/space/schumann_resonance.py", "SchumannHarmonicAnalyzer"),
        test_files=[
            "tests/space/test_schumann_labeling.py",
            "tests/space/test_schumann_stability.py",
        ],
        artifact_path="artifacts/schumann_eval.json",
    ),
    Row(
        module="ConsciousnessFieldAnalyzer (WS-D)",
        dataset="GCP (real **unreachable**); **synthetic** null",
        seeds="0,1,2",
        metric="Stouffer Z, network var",
        artifact_display="`parapsych_eval.json`",
        tests_display="`test_gcp_ingest` (5)",
        contract="abstains untrained; never asserts psi",
        status="QUARANTINE (data unreachable; null)",
        source=("src/omni_mercury_engine/models/parapsychology.py", "ConsciousnessFieldAnalyzer"),
        test_files=["tests/models/test_gcp_ingest.py"],
        artifact_path="artifacts/parapsych_eval.json",
    ),
]

# --- Verification gates that protect the above (this round's infra) -----------
GATE_ROWS: list[Row] = [
    Row(
        module="label_provenance (WS-A leak gate)",
        dataset="all 38 dataset loaders",
        seeds="—",
        metric="circular-label audit",
        artifact_display="registry in-module",
        tests_display="`test_label_provenance_gate` (11)",
        contract="repo-wide; CI `--check`",
        status="ACTIVE (gate)",
        source=("src/omni_mercury_engine/datasets/label_provenance.py", "audit_label_provenance"),
        test_files=["tests/datasets/test_label_provenance_gate.py"],
    ),
    Row(
        module="ablation_guard (WS-B confound guard)",
        dataset="paired ablation AUCs",
        seeds="—",
        metric="inverted-ranking detection",
        artifact_display="wired into both ablations",
        tests_display="`test_ablation_guard` (10)",
        contract="forces QUARANTINE on confound",
        status="ACTIVE (gate)",
        source=("src/omni_mercury_engine/evaluation/ablation_guard.py", "check_ablation_confound"),
        test_files=["tests/evaluation/test_ablation_guard.py"],
    ),
    Row(
        module="event_coincidence (WS-D null-test)",
        dataset="any score stream + event catalog",
        seeds="permutation null",
        metric="pre-registered p, FDR/Bonferroni",
        artifact_display="`spaceweather_coincidence.json`",
        tests_display="`test_event_coincidence` (offline)",
        contract="pre-registered; honest null",
        source=(
            "src/omni_mercury_engine/evaluation/event_coincidence.py",
            "permutation_coincidence_test",
        ),
        test_files=["tests/evaluation/test_event_coincidence.py"],
        artifact_path="artifacts/spaceweather_coincidence.json",
        status="ACTIVE (gate)",
    ),
]

_PREAMBLE = """# WS-E — Neural-submodule completeness sweep

> **GENERATED — do not edit by hand.** Regenerate with
> `python scripts/neural_coverage.py --update`; CI runs
> `python scripts/neural_coverage.py --check`, which fails if this file is out of
> sync *or* if any referenced module, test file, or committed artifact is
> missing. The table therefore cannot silently rot.

Coverage for every neural submodule in the neuro-symbolic fusion scope plus the
verification gates that protect it. Each is either fully-covered-and-active or
explicitly **quarantined with a recorded reason**. No half-wired module is left
in scope: every row has a dataset, seed, metric, artifact, tests, and an
off-path/quarantine contract.
"""

_HEADER = (
    "| Module | Dataset (provenance) | Seed(s) | Metric | Artifact | Tests | "
    "Off-path / quarantine contract | Status |"
)
_SEP = "|---|---|---|---|---|---|---|---|"

_POSTAMBLE = """
## Off-path / determinism invariants (asserted in tests)

* `symbolic_weight=0` and `domain_encoder=False` leave the fusion path
  **structurally identical** and numerically identical within the baseline's own
  ~1e-15 float non-determinism (the baseline is not bit-deterministic; this is
  pre-existing, not introduced here — see `docs/DOMAIN_ENCODERS.md`).
* `MercuryAnomalyDetector` is byte-identical across repeated runs (WS-A guard).
* Quarantined sub-nets are deterministic on fixed input (the #262 fix) and
  off-by-default; activation is an explicit, documented opt-in.
* `SchumannHarmonicAnalyzer.confidence_logits` is byte-identical to the sigmoid
  head at inference (WS-C); the seed-instability was a full-batch optimisation
  artifact, root-caused and fixed (mini-batch) — see
  `docs/SCHUMANN_PREREGISTRATION.md`.

## Provenance contract (met by every row)

Dataset id + license + URL + content hash, RNG seed(s), metric definition,
artifact path, and commit are recorded — in `anomaly_regression_baseline.json`,
the ablation artifacts, and the labeling/ingestion provenance dicts.

## Self-verification (the gate)

`scripts/neural_coverage.py --check` verifies, for every row above, that the
named source symbol, every referenced test file, and every committed artifact
actually exist, and that this document matches the registry byte-for-byte. It
runs in CI and as `tests/docs/test_neural_coverage_gate.py`.

## Out of scope (flagged, not silently skipped)

The wider tree has ~40 additional `nn.Module` classes (visual/VLM, SOTA
TranAD/MAAT, geological detectors, etc.). They are **not** part of the
neuro-symbolic-fusion scope and were not re-audited here. A full-tree neural
audit is a separate, larger effort; this table is exhaustive for the modules in
scope.
"""


def render_doc() -> str:
    lines = [_PREAMBLE, "## Neural submodules", "", _HEADER, _SEP]
    for r in COVERAGE_ROWS:
        lines.append("| " + " | ".join(r.cells()) + " |")
    lines += ["", "## Verification gates", "", _HEADER, _SEP]
    for r in GATE_ROWS:
        lines.append("| " + " | ".join(r.cells()) + " |")
    lines.append(_POSTAMBLE)
    return "\n".join(lines).rstrip() + "\n"


def verify_references() -> list[str]:
    """Return a list of broken references (empty = all references resolve)."""
    problems: list[str] = []
    for r in COVERAGE_ROWS + GATE_ROWS:
        src_path, symbol = r.source
        p = REPO / src_path
        if not p.exists():
            problems.append(f"{r.module}: source file missing: {src_path}")
        elif f"{symbol}" not in p.read_text():
            problems.append(f"{r.module}: symbol {symbol!r} not found in {src_path}")
        for t in r.test_files:
            if not (REPO / t).exists():
                problems.append(f"{r.module}: test file missing: {t}")
        if r.artifact_path and not r.artifact_ci_generated:
            if not (REPO / r.artifact_path).exists():
                problems.append(f"{r.module}: committed artifact missing: {r.artifact_path}")
    return problems


def check() -> list[str]:
    """Return all gate problems (doc-sync + broken references). Empty = clean."""
    problems = verify_references()
    rendered = render_doc()
    current = DOC.read_text() if DOC.exists() else ""
    if current != rendered:
        problems.append(
            "docs/NEURAL_SUBMODULE_COVERAGE.md is out of sync with the registry "
            "-- run: python scripts/neural_coverage.py --update"
        )
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="gate: exit non-zero on any problem")
    ap.add_argument("--update", action="store_true", help="regenerate the coverage doc")
    args = ap.parse_args(argv)

    if args.update:
        DOC.write_text(render_doc())
        print(f"wrote {DOC.relative_to(REPO)}")
        return 0

    problems = check()
    n = len(COVERAGE_ROWS) + len(GATE_ROWS)
    print(f"neural-coverage gate: {n} rows ({len(COVERAGE_ROWS)} modules + {len(GATE_ROWS)} gates)")
    if problems:
        print(f"PROBLEMS ({len(problems)}):")
        for p in problems:
            print(f"  - {p}")
        return 1 if args.check else 0
    print("clean: doc in sync; every module/test/artifact reference resolves.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
