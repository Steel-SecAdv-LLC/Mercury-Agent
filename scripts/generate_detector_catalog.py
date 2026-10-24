#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generate ``docs/DETECTORS.md`` -- the detector catalog -- from the manifest.

One page answering "what detectors do I have, and how do I reach each one",
generated from ``DETECTOR_MANIFEST`` so it can never drift from the code:
every entry's seams, input modality and category come from the same
declarations the seam test suite drives end to end.

Run::

    python scripts/generate_detector_catalog.py          # write docs/DETECTORS.md
    python scripts/generate_detector_catalog.py --check  # drift check (CI gate)

The point is anti-theatre: the catalog lists exactly what the manifest
declares and the seam suite proves, never hand-curated claims.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "DETECTORS.md"

HEADER = """\
# Detector Catalog

<!-- GENERATED FILE: do not hand-edit. -->
<!-- Regenerate: python scripts/generate_detector_catalog.py -->
<!-- CI gate:    python scripts/generate_detector_catalog.py --check -->

Every entry below comes from ``DETECTOR_MANIFEST``
(``src/omni_mercury_engine/core/detector_registry.py``), the single source of
truth the seam test suite (``tests/core/test_detector_manifest_seams.py``)
drives end to end on every CI run.  **Seams** say how an entry is reached:

- ``engine`` -- a ``BaseDetector``; enable it with
  ``OmniMercuryEngine.enable_detector(name)`` and use ``fit``/``detect``.
- ``registry`` -- ``DetectorRegistry.auto_discover_detectors()`` registers it;
  ``extract_all_features(data, modality=...)`` harvests its real output.
- *library-only* -- no numeric seam; call its domain API directly (the table
  names the adapter entry covering it, where one exists).

**Modality** is the input its seams were measured with.  ``fit`` marks
detectors whose registry methods require ``fit`` first; ``ext`` marks
detectors needing an external service or model backend; ``serves X`` marks
an adapter entry and names the domain detector behind it.
"""

FOOTER_INTRO = """\

## Other detector classes (not manifest entries)

The source holds more public ``*Detector`` classes than the manifest lists.
Each is accounted for in ``core/detector_roles.py``, and
``tests/core/test_detector_roles.py`` re-derives every claim below from the
source and fails when a new ``*Detector`` class is in neither place.  A class
named here is not a numeric seam: it is a part of something else, a base, or a
detector with its own API that is reached by importing it.
"""

_KIND_TITLES = {
    "library": (
        "Detectors with their own domain API (import them; no numeric seam)",
        "Each has at least one test and a declared ``decision_source``.",
    ),
    "variant": ("Variants of another detector", "A subclass, reached by importing it."),
    "deprecated": (
        "Deprecated",
        "The module warns on import and names its replacement; kept for compatibility.",
    ),
    "component": (
        "Components (constructed by another class)",
        "Built and driven by the class(es) in the last column; not seams of their own.",
    ),
    "abstract": (
        "Abstract bases and protocols",
        "Never instantiated on their own.",
    ),
}


def _class_summary(ref: str) -> str:
    """First docstring line of ``module:Class``, from the source (no import)."""
    import ast

    module, _, name = ref.partition(":")
    base = REPO / "src" / Path(*module.split("."))
    path = base / "__init__.py" if base.is_dir() else base.with_suffix(".py")
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            doc = ast.get_docstring(node) or ""
            return doc.splitlines()[0].rstrip(".") if doc else "--"
    return "--"


def _roles_section() -> str:
    from omni_mercury_engine.core.detector_roles import DETECTOR_ROLES

    out = [FOOTER_INTRO]
    for kind, (title, blurb) in _KIND_TITLES.items():
        rows = sorted((ref, r) for ref, r in DETECTOR_ROLES.items() if r.kind == kind)
        out.append(f"\n### {title} ({len(rows)})\n\n{blurb}\n")
        built = kind in {"component", "variant"}
        out.append(f"| class | module | what it is |{' built by |' if built else ''}")
        out.append(f"|---|---|---|{'---|' if built else ''}")
        for ref, role in rows:
            module, _, name = ref.partition(":")
            short = module.removeprefix("omni_mercury_engine.")
            extra = ""
            if built:
                parts = ", ".join(f"`{p.rsplit(':', 1)[1]}`" for p in role.parts)
                extra = f" {parts} |"
            out.append(f"| `{name}` | `{short}` | {_class_summary(ref)} |{extra}")
    return "\n".join(out) + "\n"


def _rows() -> dict[str, list[tuple[str, ...]]]:
    """Catalog rows per seam group, from the live manifest."""
    from omni_mercury_engine.core.detector_registry import DETECTOR_MANIFEST

    groups: dict[str, list[tuple[str, ...]]] = {
        "engine + registry": [],
        "registry only": [],
        "engine only": [],
        "library-only (domain API)": [],
    }
    for e in sorted(DETECTOR_MANIFEST, key=lambda e: (e.category.value, e.name)):
        seams = set(e.seams)
        if seams == {"engine", "registry"}:
            key = "engine + registry"
        elif seams == {"registry"}:
            key = "registry only"
        elif seams == {"engine"}:
            key = "engine only"
        else:
            key = "library-only (domain API)"
        notes = ", ".join(
            n
            for n in (
                "fit" if e.fit_required else "",
                "ext" if e.external else "",
                f"covered by `{e.covered_by}`" if e.covered_by else "",
                f"serves `{e.wraps.rsplit(':', 1)[1]}`" if e.wraps else "",
            )
            if n
        )
        groups[key].append(
            (e.name, e.class_name, e.category.value, e.modality, notes or "--", e.description)
        )
    return groups


def render() -> str:
    """The complete catalog page."""
    parts = [HEADER]
    for title, rows in _rows().items():
        parts.append(f"\n## {title} ({len(rows)})\n")
        parts.append("| name | class | category | modality | notes | description |")
        parts.append("|---|---|---|---|---|---|")
        for name, cls, cat, modality, notes, desc in rows:
            parts.append(f"| `{name}` | `{cls}` | {cat} | {modality} | {notes} | {desc} |")
    parts.append(_roles_section())
    return "\n".join(parts)


def main() -> int:
    """Write or check the catalog; returns a process exit code."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="fail if docs/DETECTORS.md is stale")
    args = ap.parse_args()
    content = render()
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != content:
            print(
                "DRIFT: docs/DETECTORS.md is stale. "
                "Run `python scripts/generate_detector_catalog.py`.",
                file=sys.stderr,
            )
            return 1
        print("docs/DETECTORS.md is fresh.")
        return 0
    OUT.write_text(content, encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
