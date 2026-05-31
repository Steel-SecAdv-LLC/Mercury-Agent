"""E/F: the neural-submodule coverage doc must stay a living, accurate artifact.

`docs/NEURAL_SUBMODULE_COVERAGE.md` is generated from the registry in
`scripts/neural_coverage.py`. This gate fails if the committed doc is out of sync
*or* if any referenced module / test file / committed artifact no longer exists,
so coverage cannot silently rot. Offline; no heavy imports.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
_SCRIPT = _REPO / "scripts" / "neural_coverage.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("neural_coverage", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass string-annotation resolution works.
    sys.modules["neural_coverage"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_coverage_doc_is_in_sync_and_references_resolve() -> None:
    mod = _load_module()
    problems = mod.check()
    assert problems == [], "neural-coverage gate problems:\n" + "\n".join(
        f"  - {p}" for p in problems
    )


def test_every_referenced_test_file_exists() -> None:
    mod = _load_module()
    for row in mod.COVERAGE_ROWS + mod.GATE_ROWS:
        for t in row.test_files:
            assert (_REPO / t).exists(), f"{row.module}: missing test file {t}"


def test_every_source_symbol_exists() -> None:
    mod = _load_module()
    for row in mod.COVERAGE_ROWS + mod.GATE_ROWS:
        src, symbol = row.source
        path = _REPO / src
        assert path.exists(), f"{row.module}: missing source {src}"
        assert symbol in path.read_text(), f"{row.module}: symbol {symbol} not in {src}"


def test_render_is_deterministic() -> None:
    mod = _load_module()
    assert mod.render_doc() == mod.render_doc()


def test_status_values_are_known() -> None:
    mod = _load_module()
    for row in mod.COVERAGE_ROWS + mod.GATE_ROWS:
        assert row.status.startswith(("ACTIVE", "QUARANTINE")), row.status


@pytest.mark.parametrize("required", ["SchumannHarmonicAnalyzer", "DomainEncoderStack"])
def test_key_modules_are_covered(required: str) -> None:
    mod = _load_module()
    modules = {r.module.split(" ")[0] for r in mod.COVERAGE_ROWS}
    assert required in modules
