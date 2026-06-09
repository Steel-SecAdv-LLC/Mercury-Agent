# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Canonical repo-wide label-provenance leak detector (WS-A).

PR #262 removed 13 *circular* manufactured-label datasets from the supervised
headline by declaring ``LABEL_SOURCE = "statistical"`` on those loaders. That
de-leak relied on each loader **honestly** declaring its provenance -- but the
base class defaults ``LABEL_SOURCE = "ground_truth"`` (see ``base.py``), so a
loader that manufactures anomaly labels (a feature threshold, a z-score fence, a
synthetic generator) and simply *forgets* to override the attribute is silently
counted as genuine, re-inflating the headline. That silent default is the leak
vector this module closes.

Instead of standing vigilance (re-auditing by hand every PR) this is **one
canonical, committed, enforced detector**:

* :data:`LABEL_PROVENANCE_REGISTRY` is the frozen result of a human audit: every
  concrete :class:`~omni_mercury_engine.datasets.base.DatasetLoader` subclass
  maps to ``(label_source, justification)`` -- the justification states *how the
  labels are derived* (genuine annotation vs threshold/synthetic). Flipping a
  loader from ``statistical`` to ``ground_truth`` to inflate a number now
  requires editing this registry, which is a reviewable one-line diff.
* :func:`audit_label_provenance` enumerates the live loaders and returns every
  divergence: a loader missing from the registry (a new dataset whose provenance
  was never declared), a declared ``LABEL_SOURCE`` that disagrees with the
  audited value, an invalid value, or a registry entry whose class vanished.
* A *static* circularity heuristic (:func:`scan_circular_label_construction`)
  reads each loader's own source for the exact pattern that defines circularity
  -- ``labels = (features[...] > threshold)`` in a real (non-synthetic) code
  path -- and flags any loader that manufactures labels while the registry calls
  it genuine. This is the audit *methodology* promoted to an automated check; it
  is what catches a circular loader even if someone (mis)declares it genuine.

The pytest gate (``tests/datasets/test_label_provenance_gate.py``) and the CI
step run :func:`audit_label_provenance` so no future PR can reintroduce a
circular dataset into the supervised headline.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
from dataclasses import dataclass

from omni_mercury_engine.datasets.base import DatasetLoader
from omni_mercury_engine.datasets.metadata import (
    GENUINE_LABEL_SOURCES,
    VALID_LABEL_SOURCES,
)

# ---------------------------------------------------------------------------
# The committed audit. Key: "<module_basename>.<ClassName>". Value:
# (label_source, justification-of-how-labels-are-derived).
#
# label_source MUST be one of datasets.metadata.VALID_LABEL_SOURCES:
#   ground_truth      -- genuine real-world labels (annotated events/attacks/
#                        outcomes, or genuine class labels remapped to anomaly).
#   expert_annotated  -- domain-expert annotation (clinician, etc.).
#   statistical       -- MANUFACTURED by thresholding a feature / z-score /
#                        percentile / domain cut, or synthetically generated.
#                        Excluded from the supervised headline as circular.
#   none              -- unlabeled.
# ---------------------------------------------------------------------------
LABEL_PROVENANCE_REGISTRY: dict[str, tuple[str, str]] = {
    # --- Genuine ground-truth anomaly benchmarks --------------------------
    "adbench.ADBenchLoader": (
        "ground_truth",
        "ADBench tabular sets ship genuine 0/1 anomaly labels (MIT).",
    ),
    "adrepository.ADRepositoryLoader": (
        "ground_truth",
        "ADBench/ADRepository .npz files carry genuine anomaly labels.",
    ),
    "industrial.SWaTLoader": (
        "ground_truth",
        "SWaT testbed records the real injected-attack flag per row.",
    ),
    "industrial.WADILoader": (
        "ground_truth",
        "WADI testbed records the real attack flag (Attack LABLE) per row.",
    ),
    "industrial.BATADALLoader": (
        "ground_truth",
        "BATADAL ships the ground-truth ATT_FLAG attack-label column.",
    ),
    "security.NSLKDDLoader": (
        "ground_truth",
        "NSL-KDD ships genuine per-connection attack/normal labels.",
    ),
    "security.CICIDSLoader": (
        "ground_truth",
        "CICIDS-2017 ships genuine labeled benign/attack flows.",
    ),
    "timeseries.NABLoader": (
        "ground_truth",
        "NAB ships hand-labeled anomaly windows (combined_windows.json).",
    ),
    "timeseries.SMAPMSLLoader": (
        "ground_truth",
        "SMAP/MSL ship NASA-curated labeled anomaly intervals.",
    ),
    "timeseries.SMDLoader": (
        "ground_truth",
        "Server Machine Dataset ships curated labeled anomaly intervals.",
    ),
    "ucr_archive.UCRLoader": (
        "ground_truth",
        "Genuine UCR class labels remapped to anomaly (minority class).",
    ),
    "ucr_archive.CWRUBearingLoader": (
        "ground_truth",
        "CWRU genuine fault classes remapped to anomaly (fault vs normal).",
    ),
    "ucr_archive.MBALoader": (
        "ground_truth",
        "MIT-BIH MBA genuine beat-class labels remapped to anomaly.",
    ),
    "medical.MIMICLoader": (
        "ground_truth",
        "Real MIMIC clinical outcome labels (mortality/etc.) from the record.",
    ),
    "medical.SepsisDataset": (
        "ground_truth",
        "Real sepsis diagnosis from MIMIC ICD codes (clinical outcome).",
    ),
    # --- Expert annotation ------------------------------------------------
    "mitbih.MITBIHLoader": (
        "expert_annotated",
        "Cardiologist-annotated MIT-BIH heartbeat classes.",
    ),
    "medical.PhysioNetLoader": (
        "expert_annotated",
        "Real PhysioNet databases carry expert ECG/event annotations.",
    ),
    "medical.CardiologyDataset": (
        "expert_annotated",
        "PhysioNet-derived expert cardiology annotations.",
    ),
    # --- Manufactured / synthetic (statistical) -- excluded from headline -
    "environmental.USGSEarthquakeLoader": (
        "statistical",
        "Labels = magnitude>=5 OR depth>300 threshold on feature columns.",
    ),
    "environmental.NOAAWeatherLoader": (
        "statistical",
        "Labels = max z-score > 3.0 threshold on the weather features.",
    ),
    "environmental.WildfireDataLoader": (
        "statistical",
        "Labels = FIRMS confidence > 70 threshold on a feature.",
    ),
    "environmental.USGSGeochemistryLoader": (
        "statistical",
        "Labels = detection-threshold cuts on concentration features.",
    ),
    "ocean.NOAABuoyLoader": (
        "statistical",
        "No ground-truth labels; manufactured from buoy signal statistics.",
    ),
    "noaa_storm.NOAAStormEventsLoader": (
        "statistical",
        "Labels = threshold on damage/casualty feature columns.",
    ),
    "noaa_gsod.NOAAGSODLoader": (
        "statistical",
        "Labels = per-station statistical anomaly mask (threshold).",
    ),
    "noaa_erddap.NOAAERDDAPLoader": (
        "statistical",
        "Labels = max z-score > 3.0 threshold on features.",
    ),
    "epa_air.EPAAirQualityLoader": (
        "statistical",
        "Labels = PM2.5 > AQI threshold (domain cut on the feature).",
    ),
    "disaster.FEMADisasterLoader": (
        "statistical",
        "No ground-truth anomaly labels; heuristic polarity selection.",
    ),
    "disaster.FEMAHazardMitigationLoader": (
        "statistical",
        "No ground-truth anomaly labels; heuristic polarity selection.",
    ),
    "space.NASAExoplanetLoader": (
        "statistical",
        "Labels derived from threshold cuts on transit features.",
    ),
    "space.SolarDynamicsLoader": (
        "statistical",
        "Labels = X-ray flux > M-class threshold / z-score fallback.",
    ),
    "space.SETILoader": (
        "statistical",
        "Synthetic signal generator (no real labels).",
    ),
    "security.ThreatIntelLoader": (
        "statistical",
        "Labels = (num_phases>=2 & num_platforms>=3) heuristic threshold.",
    ),
    "ucr_archive.MSDSLoader": (
        "statistical",
        "Synthetic multi-source generator (no real labels).",
    ),
    # Residual circularity caught by this round's audit (WS-A, PR follow-on):
    # all four climate loaders manufacture labels by thresholding the same
    # ocean/atmosphere features the detector consumes, yet inherited the
    # silent "ground_truth" default. Corrected to "statistical" at source.
    "climate.SimonsCMAPLoader": (
        "statistical",
        "Labels = oxygen<2 | temp>30 | nitrate>30 threshold on features.",
    ),
    "climate.WorldOceanDatabaseLoader": (
        "statistical",
        "No ground-truth labels; manufactured/synthetic profile anomalies.",
    ),
    "climate.CopernicusSeaLevelLoader": (
        "statistical",
        "Labels = |sea-level-anomaly| > 0.15 m threshold on a feature.",
    ),
    "climate.CopernicusERA5Loader": (
        "statistical",
        "Labels = temperature/salinity-anomaly thresholds on features.",
    ),
}


@dataclass
class ProvenanceFinding:
    """One label-provenance gate finding (a leak or inconsistency)."""

    loader: str
    kind: str  # "unregistered" | "missing_class" | "mismatch" | "invalid" | "circular"
    detail: str

    def __str__(self) -> str:
        """Return the string representation."""
        return f"[{self.kind}] {self.loader}: {self.detail}"


def _loader_key(cls: type) -> str:
    return f"{cls.__module__.split('.')[-1]}.{cls.__name__}"


def discover_loaders() -> dict[str, type[DatasetLoader]]:
    """Import every ``datasets`` submodule and return concrete loader classes.

    Concrete = a :class:`DatasetLoader` subclass with no remaining abstract
    methods (i.e. actually instantiable / usable in the benchmark).
    """
    import omni_mercury_engine.datasets as ds_pkg

    for mod in pkgutil.walk_packages(ds_pkg.__path__, ds_pkg.__name__ + "."):
        try:
            importlib.import_module(mod.name)
        except Exception:
            # A loader whose optional deps are absent still has its class
            # defined once imported elsewhere; an unimportable module simply
            # contributes no classes here. Import hygiene is covered by other
            # CI lanes, not this provenance gate.
            continue

    def _all_subclasses(cls: type) -> set[type]:
        out: set[type] = set()
        for s in cls.__subclasses__():
            out.add(s)
            out |= _all_subclasses(s)
        return out

    loaders: dict[str, type[DatasetLoader]] = {}
    for cls in _all_subclasses(DatasetLoader):
        if getattr(cls, "__abstractmethods__", None):
            continue  # still abstract -> not a usable dataset
        loaders[_loader_key(cls)] = cls
    return loaders


def scan_circular_label_construction(cls: type) -> bool:
    """Return whether a loader manufactures labels from a feature threshold.

    Applies only to *real* (non-synthetic) code paths.

    The circular pattern (the exact thing PR #255/#262 de-leaked) is an
    assignment to ``labels``/``y`` whose value compares a feature matrix
    (``features[...]`` / ``self.X`` / ``data[...]``) against a constant. Methods
    whose name contains ``synthetic`` are excluded: a clearly-labelled,
    dependency-gated synthetic *fallback* is a separate honesty mechanism
    (``ALLOW_SYNTHETIC`` / ``DataSourceUnavailableError``), not a circular
    real-label path.
    """
    try:
        src = inspect.getsource(cls)
    except (OSError, TypeError):
        return False
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False

    feature_names = {"features", "x", "data", "feats", "values"}

    class _Visitor(ast.NodeVisitor):
        found = False

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if "synthetic" in node.name.lower():
                return  # skip gated synthetic fallbacks
            self.generic_visit(node)

        def visit_Assign(self, node: ast.Assign) -> None:
            targets = {t.id.lower() for t in node.targets if isinstance(t, ast.Name)}
            if targets & {"labels", "y", "label"}:
                if self._is_feature_threshold(node.value):
                    self.found = True
            self.generic_visit(node)

        def _is_feature_threshold(self, value: ast.AST) -> bool:
            for sub in ast.walk(value):
                if isinstance(sub, ast.Compare) and self._touches_feature(sub.left):
                    return True
                # np.abs(features[...]) > c   -> Compare.left is a Call
                if isinstance(sub, ast.Compare):
                    for operand in [sub.left, *sub.comparators]:
                        if self._touches_feature(operand):
                            return True
            return False

        def _touches_feature(self, node: ast.AST) -> bool:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Subscript):
                    base = sub.value
                    name = None
                    if isinstance(base, ast.Name):
                        name = base.id.lower()
                    elif isinstance(base, ast.Attribute):
                        name = base.attr.lower()
                    if name in feature_names:
                        return True
            return False

    v = _Visitor()
    v.visit(tree)
    return v.found


def audit_label_provenance(
    loaders: dict[str, type[DatasetLoader]] | None = None,
) -> list[ProvenanceFinding]:
    """Return all label-provenance leaks/inconsistencies. Empty list = clean.

    ``loaders`` may be injected (e.g. in tests) to audit a specific mapping
    without touching the global subclass registry; it defaults to
    :func:`discover_loaders`.
    """
    findings: list[ProvenanceFinding] = []
    if loaders is None:
        loaders = discover_loaders()

    # 1. invalid registry values (guards the registry itself).
    for key, (src, _just) in LABEL_PROVENANCE_REGISTRY.items():
        if src not in VALID_LABEL_SOURCES:
            findings.append(
                ProvenanceFinding(
                    key, "invalid", f"registry value {src!r} not in VALID_LABEL_SOURCES"
                )
            )

    # 2. registry entries whose class no longer exists (stale -> remove).
    for key in LABEL_PROVENANCE_REGISTRY:
        if key not in loaders:
            findings.append(
                ProvenanceFinding(key, "missing_class", "registry entry has no live loader class")
            )

    # 3. every live loader: registered? declared value == audited value?
    for key, cls in sorted(loaders.items()):
        if key not in LABEL_PROVENANCE_REGISTRY:
            declared = getattr(cls, "LABEL_SOURCE", "ground_truth")
            findings.append(
                ProvenanceFinding(
                    key,
                    "unregistered",
                    f"loader declares LABEL_SOURCE={declared!r} but is not in "
                    "LABEL_PROVENANCE_REGISTRY -- add it with an audited justification "
                    "(a new manufactured-label dataset must be 'statistical').",
                )
            )
            continue
        expected, _just = LABEL_PROVENANCE_REGISTRY[key]
        declared = getattr(cls, "LABEL_SOURCE", "ground_truth")
        if declared != expected:
            findings.append(
                ProvenanceFinding(
                    key,
                    "mismatch",
                    f"declares LABEL_SOURCE={declared!r} but the audited provenance "
                    f"is {expected!r}",
                )
            )
        # 4. circularity heuristic: manufactures labels yet called genuine.
        if expected in GENUINE_LABEL_SOURCES and scan_circular_label_construction(cls):
            findings.append(
                ProvenanceFinding(
                    key,
                    "circular",
                    "real-path code manufactures labels from a feature threshold "
                    f"but provenance is {expected!r} (genuine). Re-audit: this is "
                    "circular and must be 'statistical', or the heuristic is a "
                    "false positive that needs an explicit waiver.",
                )
            )

    return findings


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Label-provenance leak detector (WS-A).")
    ap.add_argument("--check", action="store_true", help="exit non-zero if any leak is found")
    args = ap.parse_args(argv)

    loaders = discover_loaders()
    findings = audit_label_provenance()
    counts: dict[str, int] = {}
    for key, cls in sorted(loaders.items()):
        src = getattr(cls, "LABEL_SOURCE", "ground_truth")
        counts[src] = counts.get(src, 0) + 1

    print(f"Label-provenance audit: {len(loaders)} concrete loaders")
    for src in sorted(counts):
        print(f"  {src:<16} {counts[src]}")
    print(f"registry entries: {len(LABEL_PROVENANCE_REGISTRY)}")

    if findings:
        print(f"\nLEAKS / INCONSISTENCIES ({len(findings)}):")
        for f in findings:
            print(f"  {f}")
        if args.check:
            return 1
        return 0
    print("\nclean: every loader is registered, declared honestly, and non-circular.")
    return 0


__all__ = [
    "LABEL_PROVENANCE_REGISTRY",
    "ProvenanceFinding",
    "audit_label_provenance",
    "discover_loaders",
    "scan_circular_label_construction",
]

if __name__ == "__main__":
    raise SystemExit(main())
