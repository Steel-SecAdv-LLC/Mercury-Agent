"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

Tests for ``scripts/measure_codebase_scale.py`` — the single source of truth
for the README "Codebase Scale" block and the CI drift gate.

These tests pin the two bugs the script was rewritten to cure:
  * ``__pycache__`` (and other non-package directories) must NOT be counted
    as subpackages, and
  * the README scale block must never silently drift from disk.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

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
    # Exactly on a boundary rounds deterministically.
    assert mcs._round_loc(500) == 0 or mcs._round_loc(500) == 1000


def test_render_block_is_stable_under_small_loc_churn() -> None:
    """Adding a handful of lines must not move the rendered block (no drift-gate
    churn on unrelated edits); a bucket-sized change must."""
    stats = mcs.measure()
    base = mcs.render_block(stats)

    nudged = dict(stats)
    nudged["src_loc"] = int(stats["src_loc"]) + 7
    nudged["test_loc"] = int(stats["test_loc"]) + 7
    assert mcs.render_block(nudged) == base, "small churn must not move the block"

    grown = dict(stats)
    grown["src_loc"] = int(stats["src_loc"]) + mcs.LOC_BUCKET
    assert mcs.render_block(grown) != base, "bucket-sized growth must move the block"


def test_readme_scale_block_is_in_sync() -> None:
    """This test IS the drift gate: README must match measured numbers."""
    rc = mcs.main(["--check", str(REPO_ROOT / "README.md")])
    assert (
        rc == 0
    ), "README scale block drifted — run: python scripts/measure_codebase_scale.py --update README.md"


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
