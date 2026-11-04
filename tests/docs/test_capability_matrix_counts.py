# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``CAPABILITY_MATRIX.md``'s "measured" counts must match the live measurement.

Four rows of that file cite ``scripts/measure_codebase_scale.py`` and label
their value **measured**. Only ``README.md`` was gated against that script
(``--check README.md``, in ``.github/workflows/ci.yml`` and
``.pre-commit-config.yaml``), so the matrix drifted silently and ended up
publishing, as *measured*:

===========================================  ========  ==========
row                                           published   measured
===========================================  ========  ==========
``torch.nn.Module`` subclasses                     171         174
Data-loader classes in ``loaders/``                 21          22
Detector classes under ``detectors/``               88         110
Source files / LOC                      801 / 435,834  884 / ~493k
===========================================  ========  ==========

The detector row is the one that reads as a contradiction: it names the same
metric, cites the same script, and sat 22 below README's CI-gated value. (The
package-wide *public* detector count, 170, is a genuinely different metric and
not part of this inconsistency.)

A figure labelled "measured" in one file and gated in another is the drift
class this test closes: whatever the matrix publishes with that citation is now
checked against the same function the README gate uses, so the two cannot
diverge again.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from scripts.measure_codebase_scale import measure

if TYPE_CHECKING:
    from scripts.measure_codebase_scale import ScaleStats

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MATRIX = _REPO_ROOT / "CAPABILITY_MATRIX.md"

#: Row label (as it appears in the matrix's first cell) -> the ``measure()``
#: key that row publishes. Keep this in step with the matrix: a row that cites
#: the script and is not listed here is caught by
#: :func:`test_every_row_citing_the_script_is_covered`.
ROW_TO_STAT: dict[str, str] = {
    "`torch.nn.Module` subclasses": "nn_module_subclasses",
    "Data-loader classes in `loaders/`": "loader_classes",
    "Detector classes under `detectors/`": "detector_classes",
}

#: The LOC row publishes two figures in one cell, so it is checked separately.
_LOC_ROW = "Source files / LOC"

_CITATION = "scripts/measure_codebase_scale.py"


def _int_stats(stats: ScaleStats) -> dict[str, int]:
    """The integer-valued measurements, for lookup by a parametrized key."""
    return {key: value for key, value in stats.items() if isinstance(value, int)}


def _rows_citing_the_script() -> dict[str, str]:
    """Map the first cell of each citing table row to the row's full text."""
    out: dict[str, str] = {}
    for line in _MATRIX.read_text(encoding="utf-8").splitlines():
        if _CITATION not in line or not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells:
            out[cells[0]] = line
    return out


def _bold_numbers(row: str) -> list[int]:
    """Every integer inside a ``**...**`` span, in order.

    Numbers are extracted from *within* bold spans rather than requiring the
    whole span to be one number, because the LOC row publishes both of its
    figures in a single span (``**884 / ~493,000**``).
    """
    out: list[int] = []
    for span in re.finditer(r"\*\*([^*]+)\*\*", row):
        out.extend(
            int(m.group(0).replace(",", "")) for m in re.finditer(r"\d[\d,]*", span.group(1))
        )
    return out


@pytest.fixture(scope="module")
def stats() -> ScaleStats:
    return measure()


@pytest.fixture(scope="module")
def rows() -> dict[str, str]:
    return _rows_citing_the_script()


def test_the_matrix_still_cites_the_script(rows: dict[str, str]) -> None:
    """Guard the premise: if nothing cites it, this whole module is vacuous."""
    assert rows, (
        f"no row in CAPABILITY_MATRIX.md cites {_CITATION}; if the citations "
        "were removed, remove this test too rather than leaving it passing "
        "trivially"
    )


def test_every_row_citing_the_script_is_covered(rows: dict[str, str]) -> None:
    """A new citing row cannot be added without being checked."""
    uncovered = set(rows) - set(ROW_TO_STAT) - {_LOC_ROW}
    assert not uncovered, (
        f"these rows cite {_CITATION} but are not checked against it: "
        f"{sorted(uncovered)}. Add each to ROW_TO_STAT (or handle it like the "
        "LOC row) so a 'measured' figure cannot drift from the measurement."
    )


@pytest.mark.parametrize("label", sorted(ROW_TO_STAT))
def test_each_measured_count_matches_the_live_measurement(
    label: str, rows: dict[str, str], stats: ScaleStats
) -> None:
    """The published figure equals what the cited script returns now."""
    assert label in rows, f"row {label!r} no longer cites {_CITATION}"
    published = _bold_numbers(rows[label])
    assert (
        len(published) == 1
    ), f"row {label!r} publishes {len(published)} bold figures; expected one"
    expected = _int_stats(stats)[ROW_TO_STAT[label]]
    assert published[0] == expected, (
        f"CAPABILITY_MATRIX.md row {label!r} publishes {published[0]} as "
        f"**measured**, but {_CITATION} measures {expected}. Update the row."
    )


def test_the_loc_row_matches_within_the_published_rounding(
    rows: dict[str, str], stats: ScaleStats
) -> None:
    """Source-file count exact; LOC to the nearest thousand, as published."""
    assert _LOC_ROW in rows
    published = _bold_numbers(rows[_LOC_ROW])
    assert len(published) == 2, f"expected 'files / LOC', got {published}"
    files, loc = published
    assert (
        files == stats["src_files"]
    ), f"source-file count {files} vs measured {stats['src_files']}"
    measured_loc = stats["src_loc"]
    # The cell publishes a rounded figure (README uses the same convention).
    assert abs(loc - measured_loc) <= 1000, (
        f"LOC {loc:,} vs measured {measured_loc:,} -- more than one thousand "
        "apart, so the published figure is stale rather than merely rounded"
    )


def test_the_detector_row_agrees_with_the_readme_gated_block(
    rows: dict[str, str], stats: ScaleStats
) -> None:
    """The specific inconsistency this test was written for.

    The matrix read 88 while README's CI-gated block read 110 for the same
    metric from the same script. Both are checked against ``measure()`` here so
    the two documents cannot disagree again.
    """
    readme = (_REPO_ROOT / "README.md").read_text(encoding="utf-8")
    measured = stats["detector_classes"]

    matrix_value = _bold_numbers(rows["Detector classes under `detectors/`"])[0]
    assert matrix_value == measured

    match = re.search(
        r"\|\s*Detector classes \(`class \*Detector`\) in `detectors/`\s*\|\s*\*\*(\d+)\*\*",
        readme,
    )
    assert match is not None, "README's gated detector-class row was not found"
    assert int(match.group(1)) == measured == matrix_value, (
        f"README publishes {match.group(1)}, CAPABILITY_MATRIX publishes "
        f"{matrix_value}, measurement is {measured}"
    )
