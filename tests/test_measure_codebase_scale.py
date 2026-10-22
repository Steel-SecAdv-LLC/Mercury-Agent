# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for ``scripts/measure_codebase_scale.py`` — the single source of truth for the README "Codebase Scale" block and the CI drift gate.

These tests pin the two bugs the script was rewritten to cure:
  * ``__pycache__`` (and other non-package directories) must NOT be counted
    as subpackages, and
  * ``--check`` must actually detect drift (exercised against fixture
    READMEs, never the live tree).

The LIVE-TREE drift gate — ``--check README.md`` against the real
repository — deliberately does NOT live in this suite.  It runs as a
dedicated Workflow Hardening step (ci.yml) and a pre-commit hook, both
of which fail in seconds with the exact remediation command.  As a
pytest test it sat inside the ~17-minute full-sweep lanes, where every
added or removed test module turned an entire test lane red over a
documentation integer, minutes after the fast gate had already said the
same thing.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "measure_codebase_scale.py"

_spec = importlib.util.spec_from_file_location("measure_codebase_scale", SCRIPT)
assert _spec and _spec.loader
mcs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mcs)


def test_is_package_dir_excludes_pycache_and_artifacts(tmp_path: Path) -> None:
    pkg = tmp_path / "realpkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    egg = tmp_path / "mercury_agent.egg-info"
    egg.mkdir()
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    plain = tmp_path / "not_a_package"  # dir with no __init__.py
    plain.mkdir()

    assert mcs._is_package_dir(pkg) is True
    assert mcs._is_package_dir(cache) is False
    assert mcs._is_package_dir(egg) is False
    assert mcs._is_package_dir(hidden) is False
    assert mcs._is_package_dir(plain) is False


def test_measure_counts_are_sane_and_pycache_free() -> None:
    stats = mcs.measure()
    # Real package has hundreds of source files and dozens of subpackages.
    assert stats["src_files"] > 400
    assert stats["subpackage_count"] > 30
    assert stats["test_files"] > 200
    assert stats["detector_classes"] > 0
    # Detectors live well beyond detectors/ (space/, medical/, security/, ...).
    assert stats["public_detector_classes"] > stats["detector_classes"]
    assert stats["detector_subpackages"] > 1
    assert stats["loader_classes"] > 0
    assert stats["workflow_count"] > 0
    # The exact bug that motivated the rewrite: __pycache__ must never appear.
    assert "__pycache__" not in stats["subpackages"]
    assert all(not s.endswith(".egg-info") for s in stats["subpackages"])
    assert all(not s.startswith(".") for s in stats["subpackages"])


def test_subpackage_count_is_import_invariant() -> None:
    """Importing the package creates ``__pycache__`` dirs; the count must not move."""
    before = mcs.measure()["subpackage_count"]
    import omni_mercury_engine  # noqa: F401  (forces __pycache__ creation)

    after = mcs.measure()["subpackage_count"]
    assert before == after


def test_render_block_is_deterministic_and_marked() -> None:
    stats = mcs.measure()
    block_a = mcs.render_block(stats)
    block_b = mcs.render_block(stats)
    assert block_a == block_b
    assert block_a.startswith(mcs.SCALE_START)
    assert block_a.rstrip().endswith(mcs.SCALE_END)


def test_round_loc_buckets_to_nearest_thousand() -> None:
    """LOC is bucketed so the gate ignores routine churn but tracks real growth."""
    assert mcs._round_loc(317_728) == 318_000
    assert mcs._round_loc(118_467) == 118_000
    assert mcs._round_loc(0) == 0
    # Python's banker's rounding keeps the exact half-bucket boundary deterministic.
    assert mcs._round_loc(500) == 0


def test_pure_render_tracks_bucket_sized_growth() -> None:
    """A bucket of real growth must move the freshly rendered block."""
    stats = mcs.measure()
    base = mcs.render_block(stats)
    grown = dict(stats)
    grown["src_loc"] = int(stats["src_loc"]) + mcs.LOC_BUCKET
    assert mcs.render_block(grown) != base, "bucket-sized growth must move the block"


_REAL_STATS: dict[str, Any] = dict(mcs.measure())


def _stats_with_loc(src_loc: int, test_loc: int) -> Any:
    stats = dict(_REAL_STATS)
    stats["src_loc"] = src_loc
    stats["test_loc"] = test_loc
    return stats


def test_gate_is_stable_under_small_churn_at_a_rounding_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A handful of added lines must never demand a README rewrite.

    Pinned at the worst case — a measured LOC sitting just below a
    half-bucket boundary, where pure rounding alone would flip the published
    figure (the exact live-tree state that first exposed this): the
    hysteresis in ``--check`` / ``--update`` must keep the published block
    byte-identical for churn up to half a bucket in either direction.
    """
    boundary = 412_000 + mcs.LOC_BUCKET // 2 - 7  # 7 lines below the flip point
    fake = tmp_path / "README.md"
    fake.write_text(f"intro\n{mcs.SCALE_START}\nstale\n{mcs.SCALE_END}\ntail\n")

    monkeypatch.setattr(mcs, "measure", lambda: _stats_with_loc(boundary, boundary))
    assert mcs.main(["--update", str(fake)]) == 0
    published = mcs._extract_block(fake.read_text())
    assert "~412,000" in published  # rounded, bucket-aligned figures

    for churn in (7, -7, mcs.LOC_BUCKET // 2):
        monkeypatch.setattr(
            mcs, "measure", lambda c=churn: _stats_with_loc(boundary + c, boundary + c)
        )
        assert mcs.main(["--check", str(fake)]) == 0, f"churn {churn:+} flagged drift"
        assert mcs.main(["--update", str(fake)]) == 0
        assert (
            mcs._extract_block(fake.read_text()) == published
        ), f"churn {churn:+} rewrote the published block"


def test_gate_fires_once_loc_drift_exceeds_a_full_bucket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hysteresis bounds staleness: beyond one bucket the gate must fire."""
    base_loc = 412_000
    fake = tmp_path / "README.md"
    fake.write_text(f"intro\n{mcs.SCALE_START}\nstale\n{mcs.SCALE_END}\ntail\n")
    monkeypatch.setattr(mcs, "measure", lambda: _stats_with_loc(base_loc, base_loc))
    assert mcs.main(["--update", str(fake)]) == 0

    grown = base_loc + mcs.LOC_STALENESS_LIMIT + 1
    monkeypatch.setattr(mcs, "measure", lambda: _stats_with_loc(grown, grown))
    assert mcs.main(["--check", str(fake)]) == 1, "full-bucket drift must be flagged"
    assert mcs.main(["--update", str(fake)]) == 0
    assert "~413,000" in mcs._extract_block(fake.read_text())
    assert mcs.main(["--check", str(fake)]) == 0


def test_hand_typed_exact_loc_is_never_accepted_as_fresh() -> None:
    """Freshness requires bucket alignment — a hand-typed exact tally fails."""
    assert mcs._loc_figure_fresh(412_000, 412_493) is True
    assert mcs._loc_figure_fresh(412_493, 412_493) is False  # not bucket-aligned
    assert mcs._loc_figure_fresh(412_000, 413_001) is False  # beyond the limit


def test_check_detects_drift(tmp_path: Path) -> None:
    fake = tmp_path / "README.md"
    fake.write_text(
        f"intro\n{mcs.SCALE_START}\n| Measurement | Value |\n|---|---|\n"
        f"| Python source files | **1** |\n{mcs.SCALE_END}\ntail\n"
    )
    assert mcs.main(["--check", str(fake)]) == 1


def test_update_then_check_roundtrip(tmp_path: Path) -> None:
    fake = tmp_path / "README.md"
    fake.write_text(f"intro\n{mcs.SCALE_START}\nstale\n{mcs.SCALE_END}\ntail\n")
    assert mcs.main(["--update", str(fake)]) == 0
    assert mcs.main(["--check", str(fake)]) == 0
    # Surrounding prose must be preserved.
    text = fake.read_text()
    assert text.startswith("intro\n")
    assert text.rstrip().endswith("tail")


def test_missing_markers_is_error(tmp_path: Path) -> None:
    fake = tmp_path / "README.md"
    fake.write_text("no markers here\n")
    assert mcs.main(["--check", str(fake)]) == 2


def test_public_detector_pattern_skips_bases_and_private_helpers() -> None:
    text = "class SolarFlareDetector:\nclass BaseVisualDetector:\nclass _LocalDensityDetector:\nclass Detector:\n"
    assert mcs.PUBLIC_DETECTOR_CLASS_RE.findall(text) == ["class SolarFlareDetector"]
    assert len(mcs.DETECTOR_CLASS_RE.findall(text)) == 4


def test_prose_detector_counts_must_match_the_measurement() -> None:
    text = (
        "Intro: 89 detector classes.\n"
        + mcs.SCALE_START
        + "\n| Detector classes | **89** |\n"
        + mcs.SCALE_END
        + "\nLater: 166 Detector Classes and 166 detector classes.\n"
    )
    assert mcs._stale_detector_claims(text, 166) == ["89 detector classes"]
    fixed = mcs._fix_detector_claims(text, 166)
    assert mcs._stale_detector_claims(fixed, 166) == []
    assert "| Detector classes | **89** |" in fixed  # the generated block is left alone
