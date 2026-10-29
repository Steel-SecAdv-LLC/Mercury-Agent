# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Canonical label-provenance leak detector for the live-API loaders.

Phase 1 of the governed recursive self-improvement loop: the autonomous
fitness signal can only be trusted if every loader feeding it has its label
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
import logging
import pkgutil
import textwrap
from dataclasses import dataclass

from omni_mercury_engine.datasets.metadata import (
    GENUINE_LABEL_SOURCES,
    VALID_LABEL_SOURCES,
)
from omni_mercury_engine.loaders.base import BaseDomainLoader

logger = logging.getLogger(__name__)

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
    # --- Genuine ground-truth (eligible for the transparent fitness substrate) ---
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
    "hail_loader.HailLoader": (
        "statistical",
        "Labels = ``mag >= 2.0`` in (SPC significant-hail diameter) and feature[0] "
        "is the same ``mag`` (hail size) column.",
    ),
    "landslide_loader.LandslideLoader": (
        "statistical",
        "Labels = (fatality_count > 0) OR (size_code >= large); both are scored features.",
    ),
    "airspace_loader.AirspaceLoader": (
        "statistical",
        "Kinematic-envelope outliers over live OpenSky state vectors are "
        "statistical by construction; the genuine event channel (emergency "
        "squawks 7500/7600/7700) is excluded from the scored features and "
        "carried as a separate column, never manufactured into labels.",
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
    "space_weather_loader.SpaceWeatherLoader": (
        "statistical",
        "Labels = 3-hour windows with DONKI GST observed planetary Kp >= 5 (G1); "
        "Kp is a thresholded index derived from the same ground-magnetometer "
        "disturbance physics the dB/dt features score.",
    ),
    "meteor_loader.MeteorLoader": (
        "statistical",
        "Fireball labels = CNEOS impact energy >= 1 kt on the scored log-energy "
        "feature; NEO labels = JPL PHA designation (MOID <= 0.05 au AND H <= 22), "
        "a threshold rule on the scored H/geometry features.",
    ),
    "drought_loader.DroughtLoader": (
        "statistical",
        "Labels = SPI-6 <= -1.3 (USDM D2) where the SPI is derived from the same "
        "monthly PRCP series that is feature[0] and its rolling sums.",
    ),
    "heatwave_loader.HeatwaveLoader": (
        "statistical",
        "Labels = membership in >= 3-day runs of Tmax > calendar-day 90th "
        "percentile; the same Tmax series is feature[0].",
    ),
}


@dataclass
class ProvenanceFinding:
    """One label-provenance gate finding (a leak or inconsistency)."""

    loader: str
    kind: str  # "unregistered" | "missing_class" | "mismatch" | "invalid" |
    # "circular" | "bad_exclusion" | "empty_scored_set" | "undeclared_exclusion"
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
            logger.debug(
                "Loader module %s not importable; no classes contributed", mod.name, exc_info=True
            )
            # An unimportable optional-dep loader contributes no classes here.
            # Import hygiene is covered by other CI lanes, not this gate.
            continue

    def _all_subclasses(cls: type) -> set[type]:
        out: set[type] = set()
        subclasses: list[type] = cls.__subclasses__()
        for s in subclasses:
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


def _engineered_from_columns(cls: type) -> frozenset[str]:
    """Raw column names the loader's feature engineering reads.

    Scanned off ``engineer_features`` (and any ``*feature*`` helper) so a
    label thresholding ``df["mag"]`` is recognised as circular even though the
    engineered column is called ``ef_scale``.
    """
    out: set[str] = set()
    for name, member in vars(cls).items():
        lname = name.lower()
        if "feature" not in lname and lname != "engineer_features":
            continue
        try:
            src = inspect.getsource(member)
        except (OSError, TypeError):
            continue
        try:
            out |= _columns_read(ast.parse(textwrap.dedent(src)))
        except SyntaxError:
            continue
    return frozenset(out)


def scan_circular_label_construction(cls: type) -> bool:
    """Return whether a loader manufactures labels from a feature threshold.

    Applies only to *real* (non-synthetic) code paths -- methods whose name
    contains ``synthetic`` or ``reconstruct`` are skipped, since a clearly
    labelled, gated synthetic / reconstructed fallback is a separate provenance
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

    # Two names for the same quantity: the label thresholds the *raw* column
    # (``mag``) while FEATURE_COLUMNS names the *engineered* one (``ef_scale``),
    # so membership in FEATURE_COLUMNS alone under-detects. The columns that
    # feature engineering actually reads close that gap -- if the label and the
    # feature matrix consume the same raw column, that is circularity whatever
    # the engineered column ends up called.
    scored = frozenset(getattr(cls, "FEATURE_COLUMNS", []) or ()) | _engineered_from_columns(cls)

    class _Visitor(ast.NodeVisitor):
        found = False

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            lname = node.name.lower()
            if "synthetic" in lname or "reconstruct" in lname:
                return
            # Per-function taint: the feature read is routinely a line or two
            # above the comparison, so the comparison alone shows nothing.
            tainted = _tainted_locals(node)
            for sub in ast.walk(node):
                if not isinstance(sub, ast.Assign):
                    continue
                if not _label_targets(sub):
                    continue
                if not _expr_compares_feature(sub.value, tainted, scored):
                    continue
                self.found = True
                return
            self.generic_visit(node)

    v = _Visitor()
    v.visit(tree)
    return v.found


def _label_targets(node: ast.Assign) -> bool:
    """True iff this assignment writes something named like a label vector.

    Covers ``labels = ...``, ``y = ...`` and the attribute/subscript forms
    (``self.labels = ...``, ``out["labels"] = ...``) that a plain ``ast.Name``
    check walked straight past.
    """
    names = set()
    for t in node.targets:
        if isinstance(t, ast.Name):
            names.add(t.id.lower())
        elif isinstance(t, ast.Attribute):
            names.add(t.attr.lower())
        elif isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant):
            if isinstance(t.slice.value, str):
                names.add(t.slice.value.lower())
    return bool(names & {"labels", "y", "label", "y_true", "anomaly_labels"})


def _columns_read(node: ast.AST) -> set[str]:
    """Column names this expression reads off a feature-like container.

    ``df["kp"]`` yields ``{"kp"}``; ``df.columns`` yields nothing, which is the
    point -- comparing against the *column index* (``if "label" not in
    df.columns``) is a schema check, not a threshold on a scored value, and
    reading it as one is what put a false positive on ``network_security``.
    """
    out: set[str] = set()
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Subscript):
            continue
        base = sub.value
        owner = None
        if isinstance(base, ast.Name):
            owner = base.id.lower()
        elif isinstance(base, ast.Attribute):
            owner = base.attr.lower()
        if owner not in _FEATURE_BASE_NAMES:
            continue
        key = sub.slice
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            out.add(key.value)
    return out


def _tainted_locals(func: ast.AST) -> dict[str, set[str]]:
    """Local names that carry feature data, to a fixed point.

    ``kp_values = df["kp"].values`` then ``labels = (_g_levels(kp_values) >= 7)``
    is the shape that defeated the original single-expression check: the
    comparison's operand is a call over a local, and nothing syntactically
    inside it touches a feature accessor. Propagating the taint forward --
    including through calls, so ``f(tainted)`` stays tainted -- is what makes
    the comparison legible.
    """
    tainted: dict[str, set[str]] = {}
    for _ in range(6):  # fixed point; depth beyond this does not occur in tree
        grew = False
        for sub in ast.walk(func):
            if not isinstance(sub, ast.Assign):
                continue
            cols = _columns_read(sub.value) | _tainted_columns(sub.value, tainted)
            if not cols and not _touches_feature(sub.value):
                continue
            for t in sub.targets:
                if not isinstance(t, ast.Name):
                    continue
                before = tainted.get(t.id, set())
                after = before | cols
                if after != before or t.id not in tainted:
                    tainted[t.id] = after
                    grew = True
        if not grew:
            break
    return tainted


def _tainted_columns(node: ast.AST, tainted: dict[str, set[str]]) -> set[str]:
    """Columns reachable through the already-tainted locals this node reads."""
    out: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id in tainted:
            out |= tainted[sub.id]
    return out


def _reads_tainted(node: ast.AST, tainted: dict[str, set[str]]) -> bool:
    """True iff ``node`` reads any name already known to carry feature data."""
    return any(isinstance(sub, ast.Name) and sub.id in tainted for sub in ast.walk(node))


def _expr_compares_feature(
    value: ast.AST,
    tainted: dict[str, set[str]] | None = None,
    scored: frozenset[str] | None = None,
) -> bool:
    """True iff ``value`` contains a comparison whose operands touch a feature.

    The ``Compare`` requirement is load-bearing, not incidental: it is what
    separates "thresholds a scored column" from "reads the genuine label
    column". ``labels = df["SepsisLabel"].values`` reads a dataframe and is
    perfectly honest; ``labels = df["kp"] >= 7`` reads one and is circular.
    Only the second has a comparison, which is why ``network_security`` and
    ``sepsis`` stay clean under a scan tuned for high recall.

    ``tainted`` carries the locals that already hold feature data, so a
    comparison over ``_g_levels(kp_values)`` is recognised even though nothing
    inside it is a feature accessor.
    """
    tainted = tainted or {}
    scored = frozenset(scored or ())
    for sub in ast.walk(value):
        if not isinstance(sub, ast.Compare):
            continue
        operands = [sub.left, *sub.comparators]
        columns: set[str] = set()
        for op in operands:
            columns |= _columns_read(op) | _tainted_columns(op, tainted)
        if scored:
            # Circularity is specifically "the label thresholds a column that
            # is also scored", so membership in the loader's own declared
            # FEATURE_COLUMNS is the test. A threshold on a column nobody
            # scores -- a genuine attack flag, an external index -- is honest.
            if columns & scored:
                return True
            continue
        # No declared columns to check against (a loader that engineers its
        # matrix dynamically): fall back to the shape-only heuristic.
        if any(_touches_feature(op) or _reads_tainted(op, tainted) for op in operands):
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

        # 5. the de-leak claim is verified, never taken on trust. A loader may
        #    declare LABEL_FEATURE_DISJOINT only if the exclusion it ships
        #    actually does something and actually leaves something.
        findings.extend(_audit_feature_disjointness(key, cls))

    return findings


def _audit_feature_disjointness(key: str, cls: type) -> list[ProvenanceFinding]:
    """Check a loader's ``LABEL_FEATURE_DISJOINT`` claim against its columns.

    Three ways the claim can be hollow, and all three used to be invisible:

    * a name in ``LABEL_DERIVED_FEATURES`` that is not in ``FEATURE_COLUMNS``
      excludes nothing -- one typo and a circular loader reads as de-leaked;
    * excluding *every* column leaves no scored set, which is not a fix but an
      empty measurement;
    * claiming disjointness while the AST scan still finds labels built from a
      feature, having declared no exclusions at all, is the original leak with
      a flag on top.
    """
    findings: list[ProvenanceFinding] = []
    if not getattr(cls, "LABEL_FEATURE_DISJOINT", False):
        return findings

    declared = list(getattr(cls, "FEATURE_COLUMNS", []) or [])
    excluded = tuple(getattr(cls, "LABEL_DERIVED_FEATURES", ()) or ())

    unknown = [name for name in excluded if name not in declared]
    if unknown:
        findings.append(
            ProvenanceFinding(
                key,
                "bad_exclusion",
                f"LABEL_DERIVED_FEATURES names {unknown} which are not in "
                "FEATURE_COLUMNS, so they exclude nothing -- fix the names or the "
                "disjointness claim is hollow",
            )
        )

    if declared and not [c for c in declared if c not in set(excluded)]:
        findings.append(
            ProvenanceFinding(
                key,
                "empty_scored_set",
                "every declared feature is label-derived, so excluding them leaves "
                "nothing to score; this loader needs an independent channel, not an "
                "exclusion",
            )
        )

    if not excluded and scan_circular_label_construction(cls):
        findings.append(
            ProvenanceFinding(
                key,
                "undeclared_exclusion",
                "claims LABEL_FEATURE_DISJOINT with no LABEL_DERIVED_FEATURES while "
                "its real-path code still builds labels from a feature threshold -- "
                "declare the label-bearing columns or drop the claim",
            )
        )

    return findings


def ground_truth_loader_keys() -> frozenset[str]:
    """Registry keys whose audited provenance is genuine ground truth.

    These are the only loaders whose events feed the transparent fitness signal
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
    print("\nclean: every loader is registered, provenance-declared, and non-circular.")
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
