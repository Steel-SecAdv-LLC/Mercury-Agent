# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Single-writer gate for the committed charts under ``docs/images``.

Two committed charts were silently replaced in place because multiple
generations of visualization scripts wrote the same filenames into
``docs/images`` — including scripts that fabricated their data
(``generate_advanced_visuals.py`` drew its curves from seeded noise on
designed exponentials; ``generate_anomaly_panel.py`` hardcoded domain scores
and a "simulated distribution" pie). The honest generators and the fabricated
ones fought over the same paths, and whichever ran last won.

This gate makes that structural defect a test failure:

* Only the scripts named in :data:`ALLOWED_WRITERS` may reference the
  ``docs/images`` output directory in ``scripts/``, ``benchmarks/`` or
  ``src/``. A new writer must be added here deliberately, in review.
* Each allowed writer's output filenames must not overlap another allowed
  writer's, so no two scripts can ever contend for one committed chart.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: The only sources permitted to write into docs/images, with the committed
#: chart basenames each one owns.
ALLOWED_WRITERS: dict[str, frozenset[str]] = {
    "scripts/generate_docs_images.py": frozenset(
        {
            "ensemble_benchmark_report.png",
            "anomaly_detection_panel.png",
            "mercury_performance_dashboard.png",
            "benchmark_summary_live_data.png",
        }
    ),
    "scripts/generate_calibration_visuals.py": frozenset(
        {
            "calibration_improvement.png",
            "adaptive_weight_distribution.png",
        }
    ),
    "benchmarks/neuro_symbolic_benchmark.py": frozenset(
        {
            "neuro_symbolic_benchmark_report.png",
        }
    ),
}

#: Matches a docs/images path construction in either spelling:
#: ``"docs/images"`` or ``"docs" / "images"`` / ``"docs", "images"``.
_DOCS_IMAGES = re.compile(r"docs[\"'\s]*[/,]\s*[\"']?images")


def _python_sources() -> list[Path]:
    files: list[Path] = []
    for root in ("scripts", "benchmarks", "src"):
        files.extend((REPO / root).rglob("*.py"))
    return files


def test_only_allowed_writers_reference_docs_images() -> None:
    """No script outside the allowlist may target the committed chart directory."""
    offenders = []
    for path in _python_sources():
        rel = path.relative_to(REPO).as_posix()
        if rel in ALLOWED_WRITERS:
            continue
        if _DOCS_IMAGES.search(path.read_text(errors="replace")):
            offenders.append(rel)
    assert not offenders, (
        f"these sources reference docs/images but are not allowed writers: {offenders}; "
        "route their output to results/ instead, or add them here deliberately "
        "with non-overlapping filenames"
    )


def test_no_two_writers_own_the_same_chart() -> None:
    """The committed charts have exactly one generator each."""
    seen: dict[str, str] = {}
    for writer, names in ALLOWED_WRITERS.items():
        for name in names:
            assert name not in seen, f"{name} is claimed by both {seen[name]} and {writer}"
            seen[name] = writer


def test_every_allowed_writer_mentions_only_its_own_charts() -> None:
    """A writer's source must not name another writer's output file."""
    all_names = {n for names in ALLOWED_WRITERS.values() for n in names}
    for writer, names in ALLOWED_WRITERS.items():
        text = (REPO / writer).read_text(errors="replace")
        foreign = {
            n
            for n in all_names - names
            # Only output constructions count; a cross-reference in prose
            # (docstring or comment) is allowed and common.
            if re.search(rf"[/\"']{re.escape(n)}[\"']", text)
        }
        assert not foreign, f"{writer} constructs another writer's output path(s): {foreign}"


def test_every_committed_chart_has_an_owner() -> None:
    """Each committed PNG under docs/images is owned by exactly one writer."""
    committed = {p.name for p in (REPO / "docs" / "images").glob("*.png")}
    owned = {n for names in ALLOWED_WRITERS.values() for n in names}
    unowned = committed - owned
    assert not unowned, (
        f"committed charts with no declared generator: {sorted(unowned)}; "
        "declare the owner in ALLOWED_WRITERS"
    )
