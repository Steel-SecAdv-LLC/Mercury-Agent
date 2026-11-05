# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``pytest.mark.xdist_group`` groups only because ``tests/conftest.py`` fixes it.

With pytest 9 and xdist 3.8 the marker is inert on its own.
``xdist/remote.py`` suffixes the nodeid on the worker::

    item._nodeid = f"{item.nodeid}@{'_'.join(sorted(gnames))}"

pytest 9 replaced the string ``_nodeid`` with a structured ``NodeId`` held in
``_id``, and ``Node.nodeid`` now returns ``str(self._id)`` -- so xdist writes
an attribute pytest no longer reads. The controller's ``LoadGroupScheduling``
never sees an ``@group``, ``--dist=loadgroup`` degrades silently to plain
``load``, and nothing warns, because the marker is registered.

``tests/conftest.py`` supplies the suffix through ``NodeId``'s own string
cache, which restores grouping. These tests measure that end to end, measure
that the shim is what does it, and pin the upstream internals it stands on, so
a pytest or xdist upgrade that moves any of them fails here with the reason
rather than silently un-grouping eleven modules.

Measured on the recorded-measurement modules (``-n 4 --dist=loadgroup``):
module-scoped fixtures went from recomputed per worker to computed once --
wall 151s to 98s, CPU 575s to 155s, and the largest setup charged to a single
test 119s to 88s, back inside the lane's ``--timeout=300``.
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

#: Scratch directory for the live grouping probe. It must sit under ``tests/``
#: for ``tests/conftest.py`` -- which holds the shim -- to apply to it, which
#: puts it in the path the module scan below walks. The scan therefore skips
#: it by name: the probe writes a file carrying the marker, and under xdist the
#: two tests can run on different workers at the same time, so without this the
#: scan intermittently sees the probe as a twelfth user.
_PROBE_DIR = "_xdist_group_probe"

#: Modules whose scheduling depends on the shim. Pinned so that losing
#: grouping is a visible change rather than a silent slowdown.
XDIST_GROUP_USERS: frozenset[str] = frozenset(
    {
        "tests/core/test_adaptive_domain_thresholding_measured.py",
        "tests/detectors/test_recorded_math_arrest.py",
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

_PROBE = textwrap.dedent("""
    import os
    import pathlib
    import pytest

    pytestmark = pytest.mark.xdist_group("probe_group")

    _OUT = pathlib.Path(__file__).parent

    @pytest.mark.parametrize("i", range(8))
    def test_p(i):
        # Recorded to a file: xdist does not forward worker stdout.
        worker = os.environ.get("PYTEST_XDIST_WORKER", "none")
        (_OUT / f"worker_{i}.txt").write_text(worker, encoding="utf-8")
    """)


def _run_probe(directory: Path, *extra: str) -> tuple[set[str], subprocess.CompletedProcess[str]]:
    """Run eight same-group tests over four workers; return the workers used."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "test_probe.py").write_text(_PROBE, encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(directory / "test_probe.py"),
            "-q",
            "-p",
            "no:randomly",
            "-p",
            "no:cacheprovider",
            "-n",
            "4",
            "--dist=loadgroup",
            "--no-header",
            *extra,
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    workers = {p.read_text(encoding="utf-8").strip() for p in directory.glob("worker_*.txt")}
    return workers, proc


def test_a_group_lands_on_one_worker() -> None:
    """The authoritative check: the real scheduler, not an inference from source."""
    pytest.importorskip("xdist")
    # Under ``tests/``, so ``tests/conftest.py`` -- which holds the shim --
    # applies to the probe exactly as it does to the eleven real modules.
    scratch = _REPO_ROOT / "tests" / _PROBE_DIR
    try:
        workers, proc = _run_probe(scratch)
        assert workers, (
            "the probe recorded no worker ids, so the probe run itself failed.\n"
            f"stdout:\n{proc.stdout[-1500:]}\nstderr:\n{proc.stderr[-800:]}"
        )
        assert workers == {next(iter(workers))} and len(workers) == 1, (
            "eight tests sharing one xdist_group spread over "
            f"{sorted(workers)} instead of landing on one worker. Grouping is "
            "broken: check _restore_xdist_group_scheduling in tests/conftest.py "
            "against the upstream internals pinned below. Until it is fixed the "
            "modules in XDIST_GROUP_USERS recompute their module-scoped fixtures "
            "on every worker."
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def test_the_shim_is_what_makes_grouping_work(tmp_path: Path) -> None:
    """Causation, not correlation: without the shim the same probe scatters.

    Run from a directory with its own ini file, so neither ``tests/conftest.py``
    nor the repo's ``pyproject.toml`` is loaded. If this ever passes a single
    worker, upstream has fixed the incompatibility and the shim can go.
    """
    pytest.importorskip("xdist")
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    workers, proc = _run_probe(tmp_path / "probe", "-c", str(tmp_path / "pytest.ini"))
    assert workers, (
        "the unshimmed probe recorded no worker ids.\n"
        f"stdout:\n{proc.stdout[-1500:]}\nstderr:\n{proc.stderr[-800:]}"
    )
    assert len(workers) > 1, (
        "xdist_group grouped without the shim "
        f"(workers seen: {sorted(workers)}), so pytest and xdist now agree on "
        "their own. Remove _restore_xdist_group_scheduling from "
        "tests/conftest.py and this test with it."
    )


def test_pytest_still_reads_the_node_id_the_shim_writes() -> None:
    """Half of the mechanism: ``nodeid`` comes from ``_id``, cached as a string."""
    spec = importlib.util.find_spec("_pytest.nodes")
    assert spec is not None and spec.origin is not None
    nodes = Path(spec.origin).read_text(encoding="utf-8")
    assert "return str(self._id)" in nodes, (
        "pytest's Node.nodeid no longer returns str(self._id); the shim writes "
        "a field nodeid may no longer be built from. Re-derive it against the "
        "installed pytest."
    )

    id_spec = importlib.util.find_spec("_pytest.nodeid")
    assert id_spec is not None and id_spec.origin is not None
    node_id = Path(id_spec.origin).read_text(encoding="utf-8")
    assert "if self._str_cache is not None:\n            return self._str_cache" in node_id, (
        "NodeId.__str__ no longer short-circuits on _str_cache; that cache is "
        "how the shim supplies the @group suffix."
    )
    assert '"_str_cache"' in node_id, (
        "pytest no longer writes _str_cache with object.__setattr__; the shim "
        "relies on that being the supported way to set this frozen dataclass."
    )


def test_xdist_still_gates_and_splits_the_way_the_shim_assumes() -> None:
    """The other half: xdist's own flag, and where it requires the ``@`` to sit."""
    # Located rather than imported: pytest-xdist ships no type stubs and the
    # tests/ mypy gate runs without --ignore-missing-imports.
    remote_spec = importlib.util.find_spec("xdist.remote")
    assert remote_spec is not None and remote_spec.origin is not None, "pytest-xdist missing"
    remote = Path(remote_spec.origin).read_text(encoding="utf-8")
    assert 'config.option.loadgroup = config.getvalue("dist") == "loadgroup"' in remote, (
        "xdist no longer derives config.option.loadgroup on the worker. The "
        "shim gates on that flag because a worker's own `dist` is reset to "
        '"no"; re-check what the worker can see.'
    )
    assert "ids=[item.nodeid for item in session.items]" in remote, (
        "xdist no longer reports collection as item.nodeid, so supplying the "
        "suffix through NodeId's string cache may no longer reach the "
        "controller's scheduler."
    )

    group_spec = importlib.util.find_spec("xdist.scheduler.loadgroup")
    assert group_spec is not None and group_spec.origin is not None
    loadgroup = Path(group_spec.origin).read_text(encoding="utf-8")
    assert 'nodeid.rfind("@") > nodeid.rfind("]")' in loadgroup, (
        "LoadGroupScheduling no longer requires the @group to follow the "
        "[params] bracket. The shim appends it to the full nodeid to satisfy "
        "exactly that condition; re-check the required format."
    )


def test_every_module_relying_on_grouping_is_pinned() -> None:
    """A twelfth user cannot appear without someone reading this file."""
    # The trailing parenthesis is what makes this an application of the marker
    # rather than a mention of it, which is how tests/conftest.py and this
    # module discuss the marker without registering as users of it.
    exempt = {this.resolve() for this in (Path(__file__), _REPO_ROOT / "tests" / "conftest.py")}
    found = {
        str(path.relative_to(_REPO_ROOT))
        for path in (_REPO_ROOT / "tests").rglob("*.py")
        if path.resolve() not in exempt
        and _PROBE_DIR not in path.parts
        and "pytest.mark.xdist_group(" in path.read_text(encoding="utf-8")
    }
    assert found == set(XDIST_GROUP_USERS), (
        "the set of modules using pytest.mark.xdist_group changed.\n"
        f"  added:   {sorted(found - set(XDIST_GROUP_USERS))}\n"
        f"  removed: {sorted(set(XDIST_GROUP_USERS) - found)}\n"
        "The marker works only through the shim in tests/conftest.py (see this "
        "module's docstring). Add the file here so the dependency stays visible."
    )
