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

FOOTER = """\

## Building blocks and wrappers (importable, deliberately not manifest entries)

- ``detectors/statistical_extended.py``: ``MADDetector``, ``LOFDetector``,
  ``DBSCANDetector``, ``MCDDetector``, ``CUSUMDetector``, ``GrubbsTest``,
  ``GESDTest`` -- served through the ``extended_statistical`` ensemble entry.
- ``detectors/drift.py``: ``KolmogorovSmirnovDriftDetector``,
  ``PopulationStabilityIndexDetector``, ``ChiSquaredDriftDetector``,
  ``CovarianceDriftDetector``, ``PrincipalAxisDriftDetector``,
  ``OnlineDriftDetector`` -- served through ``EnsembleDriftDetector``, which
  the engine and ``ml/online_learning.py`` already drive.
- ``detectors/adaptive_detector.py``: ``CovarianceAwareDetector``,
  ``TemporalPatternDetector`` -- composed by the ``adaptive_anomaly`` entry.
- ``core/conformal_prediction.py`` / ``detectors/conformal_martingale.py``:
  conformal wrappers constructed around a caller's detector.
"""


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
    parts.append(FOOTER)
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
