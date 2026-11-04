# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``pytest.mark.xdist_group`` does not group under pytest 9 + xdist 3.8.

Nine test modules carry ``pytestmark = pytest.mark.xdist_group(...)`` to keep
their tests on one worker. With the installed versions that marker is **inert**
and ``--dist=loadgroup`` degrades silently to plain ``load``.

Mechanism. ``xdist/remote.py`` appends the group to the nodeid on the worker::

    item._nodeid = f"{item.nodeid}@{'_'.join(sorted(gnames))}"

pytest 9 replaced the string ``_nodeid`` with a structured ``NodeId`` held in
``_id``::

    @property
    def nodeid(self) -> str:
        return str(self._id)

So xdist writes an attribute pytest no longer reads, the controller's
``LoadGroupScheduling`` never sees an ``@group``, and the tests scatter.
Measured: 8 tests sharing one group spread 2/2/2/2 over 4 workers, with
``loadgroup=True`` and the marker visible on the item at collection time.

Nothing warns: the marker is registered, so ``--strict-markers`` is satisfied.

These tests pin that state so it cannot stay invisible. They are **green while
grouping is broken** and turn red when it changes -- at which point the nine
modules below should be re-checked, since each would start being scheduled as
its author intended.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Modules relying on ``xdist_group``. Pinned so a tenth cannot be added
#: without someone reading this file and learning the marker does nothing.
XDIST_GROUP_USERS: frozenset[str] = frozenset(
    {
        "tests/test_decision_coverage_empirical.py",
        "tests/test_decision_layer_wiring.py",
        "tests/test_fusion_calibration.py",
        "tests/test_fusion_conformal.py",
        "tests/test_fusion_domain_encoder.py",
        "tests/test_fusion_pooled.py",
        "tests/test_fusion_raw_path.py",
        "tests/test_fusion_symbolic_cotraining.py",
        "tests/test_fusion_training.py",
    }
)


def test_pytest_no_longer_reads_the_attribute_xdist_writes() -> None:
    """The incompatibility itself: ``nodeid`` comes from ``_id``, not ``_nodeid``.

    Red here means pytest's ``Node`` changed. Re-check whether xdist's
    nodeid-suffix mutation works again before trusting ``xdist_group``.
    """
    spec = importlib.util.find_spec("_pytest.nodes")
    assert spec is not None and spec.origin is not None
    source = Path(spec.origin).read_text(encoding="utf-8")
    assert "return str(self._id)" in source, (
        "pytest's Node.nodeid no longer reads self._id; re-derive whether "
        "xdist's `item._nodeid = ...` suffix takes effect"
    )
    assert "self._nodeid" not in source, (
        "pytest's Node now references _nodeid again -- xdist's mutation may "
        "work, so xdist_group could be live. Re-check the modules that use it."
    )


def test_xdist_still_writes_the_attribute_pytest_ignores() -> None:
    """The other half: xdist 3.8 mutates ``_nodeid``.

    Red here means xdist changed its mechanism -- very likely the fix.
    """
    # Located rather than imported: pytest-xdist ships no type stubs, and the
    # tests/ mypy gate runs without --ignore-missing-imports.
    spec = importlib.util.find_spec("xdist.remote")
    assert spec is not None and spec.origin is not None, "pytest-xdist missing"
    source = Path(spec.origin).read_text(encoding="utf-8")
    assert "item._nodeid = f" in source, (
        "xdist no longer assigns item._nodeid; if it now sets the structured "
        "id instead, xdist_group may group correctly again"
    )


def test_grouping_does_not_actually_happen() -> None:
    """End to end: two groups, two workers, and the tests still scatter.

    The authoritative check -- it exercises the real scheduler rather than
    inferring from source. Red means grouping works now.
    """
    pytest.importorskip("xdist")
    scratch = _REPO_ROOT / ".pytest_xdist_group_probe"
    scratch.mkdir(exist_ok=True)
    try:
        # Each test records its worker id to a file: xdist does not forward
        # worker stdout to the controller, so printing would be invisible.
        (scratch / "test_probe.py").write_text(
            textwrap.dedent("""
                import os
                import pathlib
                import pytest

                pytestmark = pytest.mark.xdist_group("probe_group")

                _OUT = pathlib.Path(__file__).parent

                @pytest.mark.parametrize("i", range(8))
                def test_p(i):
                    worker = os.environ.get("PYTEST_XDIST_WORKER", "none")
                    (_OUT / f"worker_{i}.txt").write_text(worker, encoding="utf-8")
                """),
            encoding="utf-8",
        )
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                str(scratch / "test_probe.py"),
                "-q",
                "-p",
                "no:randomly",
                "-p",
                "no:cacheprovider",
                "-n",
                "4",
                "--dist=loadgroup",
                "--no-header",
            ],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
        workers = {
            path.read_text(encoding="utf-8").strip() for path in scratch.glob("worker_*.txt")
        }
        assert workers, (
            "probe recorded no worker ids; the probe run itself failed.\n"
            f"stdout:\n{proc.stdout[-1500:]}\nstderr:\n{proc.stderr[-800:]}"
        )
        assert len(workers) > 1, (
            "xdist_group now keeps a group on ONE worker "
            f"(workers seen: {sorted(workers)}). Grouping appears fixed -- drop "
            "these pins and re-check the modules in XDIST_GROUP_USERS, which "
            "have been running ungrouped and may have been relying on it."
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def test_the_set_of_modules_relying_on_grouping_is_pinned() -> None:
    """A tenth user cannot appear without someone reading this file."""
    this_file = Path(__file__).resolve()
    found = {
        str(path.relative_to(_REPO_ROOT))
        for path in (_REPO_ROOT / "tests").rglob("*.py")
        if path.resolve() != this_file
        and "pytest.mark.xdist_group" in path.read_text(encoding="utf-8")
    }
    assert found == set(XDIST_GROUP_USERS), (
        "the set of modules using pytest.mark.xdist_group changed.\n"
        f"  added:   {sorted(found - set(XDIST_GROUP_USERS))}\n"
        f"  removed: {sorted(set(XDIST_GROUP_USERS) - found)}\n"
        "That marker does NOT group with the installed pytest/xdist (see this "
        "module's docstring). If you added it for speed, it will not help; if "
        "you added it for isolation, you are not getting it -- use an explicit "
        "pytest.mark.timeout, or make the test independent of scheduling."
    )
