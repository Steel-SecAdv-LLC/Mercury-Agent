# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Canonical label-provenance leak detector for the live-API loaders.

Phase 1 of the governed recursive self-improvement loop: the autonomous
fitness signal can only be honest if every loader feeding it has its label
provenance declared and verified.  This module is the live-API mirror of
:mod:`omni_mercury_engine.datasets.label_provenance` (which enforces the
same discipline for the ADBench / benchmark side of the codebase).

The ``loaders/`` package historically had no label-provenance discipline at
all -- :class:`~omni_mercury_engine.loaders.base.BaseDomainLoader` defaulted
``LABEL_SOURCE = "ground_truth"`` and no loader overrode it -- yet the
governed-fusion suite (``research/governed_fusion/``) was already using these
loaders to compute its headline AUROC / F1.  An audit performed for Phase 1
found that 13 of the 15 concrete loaders manufacture anomaly labels by
thresholding a column that is *also* engineered as a scored feature, or
reconstruct the entire series from documented event statistics:

* ``earthquake``  -- ``magnitude >= mainshock_mag - 1.0`` on feature[0].
* ``hurricane``   -- 24h wind delta >= 30 kt rapid-intensification window on
  the scored wind-delta feature.
* ``tornado``     -- ``mag >= 3`` (EF3+) on feature[0] ``ef_scale``.
* ``fema``        -- (declarationType == ``DR``) AND ``iaProgramDeclared``
  AND ``paProgramDeclared`` -- ``ia_program`` and ``hm_program`` flags are
  features 4 and 5.
* ``marine``      -- (baseline_richness - event_richness) / baseline > 0.70,
  while ``richness_loss`` is a scored feature.
* ``pandemic``    -- 7-day rolling new_cases > 2 * 30-day baseline mean, on
  the same 7d-smoothed feature; ``ebola_2014`` additionally reconstructs the
  series (no live WHO GHO feed).
* ``energy``      -- ``kp >= 7`` on feature[0]; series reconstructed.
* ``tsunami``     -- external arrival-time window labels, but series is
  reconstructed (NDBC files rotate out after ~45 days).
* ``financial``   -- (VIX > 30 AND yield_curve < 0) OR VIX > 45, on scored
  features.
* ``flood``       -- ``gauge_height_ft >= NWS_flood_stage`` on feature[0].
* ``landslide``   -- (fatality_count > 0) OR (size_code >= large), both
  scored features.
* ``wildfire``    -- ``FRP >= p90`` per-dataset on scored ``frp``.
* ``volcanic``    -- WARNING alert level OR RED color code, on scored
  features 1 and 2.

Genuine ground-truth labels are limited to:

* ``network_security`` -- NSL-KDD ``label`` and BATADAL ``ATT_FLAG`` columns
  (genuine per-row attack flags).
* ``sepsis``           -- PhysioNet Challenge 2019 ``SepsisLabel`` column.

Discipline applied here is identical to the ``datasets/`` side:

* :data:`LABEL_PROVENANCE_REGISTRY` is the frozen audit -- every concrete
  :class:`~omni_mercury_engine.loaders.base.BaseDomainLoader` subclass maps
  to ``(label_source, justification)``.  Flipping a loader from
  ``statistical`` to ``ground_truth`` to inflate the headline now requires
  editing this registry and getting the diff reviewed.
* :func:`audit_label_provenance` enumerates the live loaders and returns
  every divergence (unregistered, missing, mismatched, invalid, or AST-
  detected circular).
* :func:`scan_circular_label_construction` walks each loader's source for
  the canonical circular pattern -- ``labels = (df[col] > c)`` /
  ``labels = (feature_array > c)`` in a real (non-synthetic) code path --
  and flags any loader that manufactures labels while being declared
  genuine.  The scanner recognises ``df["col"].values`` /
  ``feature_array[col]`` access shapes used by the live loaders, not just
  the ``features[..]`` / ``self.X`` shapes used by ``datasets/``.

The CI gate ``tests/loaders/test_label_provenance_gate.py`` runs
:func:`audit_label_provenance` so no future PR can silently introduce a
manufactured-label loader into the governed-fusion suite without declaring
it as such.

This module is fully offline (no network, no dataset downloads).
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
from dataclasses import dataclass

from omni_mercury_engine.datasets.metadata import (
    GENUINE_LABEL_SOURCES,
    VALID_LABEL_SOURCES,
)
from omni_mercury_engine.loaders.base import BaseDomainLoader

# ---------------------------------------------------------------------------
# The committed audit. Key: "<module_basename>.<ClassName>". Value:
# (label_source, justification-of-how-labels-are-derived).
#
# label_source MUST be one of datasets.metadata.VALID_LABEL_SOURCES:
#   ground_truth      -- genuine per-row labels independent of scored signal
#                        (benchmark attack flag, clinician annotation).
#   expert_annotated  -- domain-expert annotation.
#   statistical       -- MANUFACTURED by thresholding a scored feature, or
#                        synthetically reconstructed.  Excluded from the
#                        governed-fusion headline and from the autonomous
#                        fitness signal as circular.
#   none              -- unlabeled.
# ---------------------------------------------------------------------------
LABEL_PROVENANCE_REGISTRY: dict[str, tuple[str, str]] = {
    # --- Genuine ground-truth (eligible for the honest fitness substrate) ---
    "network_security_loader.NetworkSecurityLoader": (
        "ground_truth",
        "Benchmark per-row attack flags (NSL-KDD ``label``, BATADAL ``ATT_FLAG``).",
    ),
    "sepsis_loader.SepsisLoader": (
        "ground_truth",
        "PhysioNet Challenge 2019 ``SepsisLabel`` column (clinician annotation).",
    ),
    # --- Manufactured / reconstructed (statistical) -- circular for headline -
    "earthquake_loader.EarthquakeLoader": (
        "statistical",
        "Labels = ``magnitude >= mainshock_mag - 1.0`` and ``magnitude`` is feature[0].",
    ),
    "hurricane_loader.HurricaneLoader": (
        "statistical",
        "Labels = 24h wind delta >= 30 kt; 24h wind delta is also a scored feature.",
    ),
    "tornado_loader.TornadoLoader": (
        "statistical",
        "Labels = ``mag >= 3`` (EF3+) and feature[0] is the same ``ef_scale`` column.",
    ),
    "fema_loader.FEMALoader": (
        "statistical",
        "Labels = DR + IA + PA program flags; ia/hm program flags are scored features.",
    ),
    "marine_loader.MarineLoader": (
        "statistical",
        "Labels = richness-decline threshold; ``richness_loss`` is also a scored feature.",
    ),
    "pandemic_loader.PandemicLoader": (
        "statistical",
        "Labels = 7d rolling > 2x 30d baseline on the same 7d-smoothed new_cases feature; "
        "ebola_2014 additionally reconstructs the series.",
    ),
    "energy_loader.EnergyLoader": (
        "statistical",
        "Labels = ``kp >= 7`` on feature[0]; series reconstructed from documented Kp storm profiles.",
    ),
    "tsunami_loader.TsunamiLoader": (
        "statistical",
        "Arrival-time labels are external, but the BPR series itself is reconstructed.",
    ),
    "financial_loader.FinancialLoader": (
        "statistical",
        "Labels = (VIX>30 AND yield_curve<0) OR VIX>45 on the scored VIX / yield-curve features.",
    ),
    "flood_loader.FloodLoader": (
        "statistical",
        "Labels = ``gauge_height_ft >= NWS_flood_stage`` on feature[0].",
    ),
    "landslide_loader.LandslideLoader": (
        "statistical",
        "Labels = (fatality_count > 0) OR (size_code >= large); both are scored features.",
    ),
    "wildfire_loader.WildfireLoader": (
        "statistical",
        "Labels = ``FRP >= 90th percentile`` per-dataset on the scored ``frp`` feature.",
    ),
    "volcanic_loader.VolcanicLoader": (
        "statistical",
        "Labels = WARNING alert OR RED color code; both ``alert_level_numeric`` "
        "and ``color_code_numeric`` are scored features.",
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


def discover_loaders() -> dict[str, type[BaseDomainLoader]]:
    """Import every ``loaders`` submodule and return concrete loader classes.

    Concrete = a :class:`BaseDomainLoader` subclass with no remaining abstract
    methods (i.e. actually instantiable / usable in the governed-fusion suite).
    """
    import omni_mercury_engine.loaders as loaders_pkg

    for mod in pkgutil.walk_packages(loaders_pkg.__path__, loaders_pkg.__name__ + "."):
        try:
            importlib.import_module(mod.name)
        except Exception:
            # An unimportable optional-dep loader contributes no classes here.
            # Import hygiene is covered by other CI lanes, not this gate.
            continue

    def _all_subclasses(cls: type) -> set[type]:
        out: set[type] = set()
        for s in cls.__subclasses__():
            out.add(s)
            out |= _all_subclasses(s)
        return out

    loaders: dict[str, type[BaseDomainLoader]] = {}
    for cls in _all_subclasses(BaseDomainLoader):
        if getattr(cls, "__abstractmethods__", None):
            continue  # still abstract -> not a usable domain loader
        # Test-fixture subclasses (``tests/loaders/test_*.py``) are not
        # part of the live audit surface; they exist to exercise the base
        # class in isolation, not to score data.
        if cls.__module__.split(".")[-1].startswith("test_"):
            continue
        loaders[_loader_key(cls)] = cls
    return loaders


# Source-name patterns that almost certainly refer to a row vector / feature
# matrix in the live-API loaders.  These mirror the ``datasets/`` scanner's
# vocabulary plus the live-loader-specific shapes (``df["col"].values``,
# ``raw["col"]``).  Matching is intentionally generous: a false positive must
# be re-classified or explicitly waived, never silently dropped.
_FEATURE_BASE_NAMES = {
    "features",
    "x",
    "data",
    "feats",
    "values",
    "raw",
    "df",
    "feature_array",
}


def scan_circular_label_construction(cls: type) -> bool:
    """Return whether a loader manufactures labels from a feature threshold.

    Applies only to *real* (non-synthetic) code paths -- methods whose name
    contains ``synthetic`` or ``reconstruct`` are skipped, since a clearly
    labelled, gated synthetic / reconstructed fallback is a separate honesty
    mechanism (the suite's ``RECONSTRUCTED_*`` lists), not a circular real-
    label path.

    Recognised patterns -- assignments to ``labels`` / ``y`` / ``label``
    whose right-hand side contains a comparison that touches a feature-like
    accessor: ``df["col"]``, ``df["col"].values``, ``features[...]``,
    ``self.X``, ``raw["col"]``, ``feature_array[...]``, an ``np.where(...)``
    call over those, etc.
    """
    try:
        src = inspect.getsource(cls)
    except (OSError, TypeError):
        return False
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False

    class _Visitor(ast.NodeVisitor):
        found = False

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            lname = node.name.lower()
            if "synthetic" in lname or "reconstruct" in lname:
                return
            self.generic_visit(node)

        def visit_Assign(self, node: ast.Assign) -> None:
            targets = {t.id.lower() for t in node.targets if isinstance(t, ast.Name)}
            if targets & {"labels", "y", "label"}:
                if _expr_compares_feature(node.value):
                    self.found = True
            self.generic_visit(node)

    v = _Visitor()
    v.visit(tree)
    return v.found


def _expr_compares_feature(value: ast.AST) -> bool:
    """True iff ``value`` contains a comparison whose operands touch a feature."""
    for sub in ast.walk(value):
        if isinstance(sub, ast.Compare):
            operands = [sub.left, *sub.comparators]
            if any(_touches_feature(op) for op in operands):
                return True
        # ``np.where(features[...] > c, 1, 0)`` -- the comparison sits inside
        # a call; walking the full subtree above already catches it.
    return False


def _touches_feature(node: ast.AST) -> bool:
    """True iff ``node`` reads from a feature-like accessor (df[..], X, raw[..])."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Subscript):
            base = sub.value
            name = None
            if isinstance(base, ast.Name):
                name = base.id.lower()
            elif isinstance(base, ast.Attribute):
                name = base.attr.lower()
            if name in _FEATURE_BASE_NAMES:
                return True
        if isinstance(sub, ast.Attribute):
            # ``df.column``, ``self.X``, ``raw.values``
            owner = None
            if isinstance(sub.value, ast.Name):
                owner = sub.value.id.lower()
            elif isinstance(sub.value, ast.Attribute):
                owner = sub.value.attr.lower()
            if owner in _FEATURE_BASE_NAMES:
                return True
    return False


def audit_label_provenance(
    loaders: dict[str, type[BaseDomainLoader]] | None = None,
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
                    "(a new manufactured-label loader must be 'statistical').",
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


def ground_truth_loader_keys() -> frozenset[str]:
    """Registry keys whose audited provenance is genuine ground truth.

    These are the only loaders whose events feed the honest fitness signal
    that Phase 2's promotion gate will read from. Used by the governed-fusion
    suite to partition the manifest into ``external_label`` vs leakage-flagged.
    """
    return frozenset(
        key
        for key, (src, _just) in LABEL_PROVENANCE_REGISTRY.items()
        if src in GENUINE_LABEL_SOURCES
    )


def main(argv: list[str] | None = None) -> int:
    """Print the loader provenance summary; exit non-zero if ``--check`` and dirty."""
    import argparse

    ap = argparse.ArgumentParser(description="Loader label-provenance leak detector (Phase 1).")
    ap.add_argument("--check", action="store_true", help="exit non-zero if any leak is found")
    args = ap.parse_args(argv)

    loaders = discover_loaders()
    findings = audit_label_provenance(loaders)
    counts: dict[str, int] = {}
    for _key, cls in sorted(loaders.items()):
        src = getattr(cls, "LABEL_SOURCE", "ground_truth")
        counts[src] = counts.get(src, 0) + 1

    print(f"Loader label-provenance audit: {len(loaders)} concrete loaders")
    for src in sorted(counts):
        print(f"  {src:<16} {counts[src]}")
    print(f"registry entries: {len(LABEL_PROVENANCE_REGISTRY)}")
    gt = sorted(ground_truth_loader_keys())
    print(f"ground-truth loaders ({len(gt)}): {', '.join(gt) if gt else '(none)'}")

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
    "ground_truth_loader_keys",
    "scan_circular_label_construction",
]


if __name__ == "__main__":
    raise SystemExit(main())
