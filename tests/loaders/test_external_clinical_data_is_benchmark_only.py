# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""PhysioNet is a yardstick, not a runtime dependency.

``SepsisLoader`` is one of only two ``ground_truth`` loaders in the tree, so it
carries real weight: the governed-fusion fitness signal reads genuinely
labelled events only, and half of that is this loader. That makes it tempting
to reach for from a serving path, which is exactly what must not happen.
PhysioNet Challenge 2019 is third-party clinical data Mercury is *measured
against*; nothing Mercury serves should depend on an outbound call to it.

This gate pins the boundary two ways:

* no module under ``src/`` imports ``SepsisLoader`` except the loaders package
  (which only re-exports it) and the provenance registry (which only names
  it); the callers are ``benchmarks/`` and ``research/``;
* importing the loader opens no connection -- the PhysioNet URLs are module
  constants, read at fetch time, not at import.

AST-based and offline. It does not fetch anything.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src" / "omni_mercury_engine"

#: The only ``src/`` modules allowed to name the loader, and why.
_ALLOWED_IN_SRC = {
    "loaders/__init__.py": "re-exports the loader; no call",
    "loaders/sepsis_loader.py": "is the loader",
    "loaders/label_provenance.py": "names it in the frozen provenance registry",
}


def _modules_naming(symbol: str) -> dict[str, list[int]]:
    """Map each ``src/`` module to the lines where it names ``symbol``."""
    hits: dict[str, list[int]] = {}
    for path in sorted(_SRC.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
            continue
        lines = [
            node.lineno
            for node in ast.walk(tree)
            if (isinstance(node, ast.Name) and node.id == symbol)
            or (isinstance(node, ast.Attribute) and node.attr == symbol)
            or (isinstance(node, ast.ImportFrom) and any(a.name == symbol for a in node.names))
            or (isinstance(node, ast.Constant) and node.value == symbol)
        ]
        if lines:
            hits[str(path.relative_to(_SRC))] = sorted(set(lines))
    return hits


def test_no_serving_path_in_src_reaches_the_sepsis_loader() -> None:
    """Only the re-export and the provenance registry may name it."""
    offenders = {
        module: lines
        for module, lines in _modules_naming("SepsisLoader").items()
        if module not in _ALLOWED_IN_SRC
    }

    assert not offenders, (
        "PhysioNet is benchmark-only; these src/ modules name SepsisLoader: "
        f"{offenders}. If a serving surface genuinely needs clinical data, it "
        "needs its own consented source, not the measurement corpus."
    )


def test_no_serving_surface_reaches_physionet() -> None:
    """The surfaces Mercury serves from carry no PhysioNet reference at all.

    PhysioNet hosts several corpora Mercury is measured on -- Challenge 2019
    for sepsis, MIT-BIH for ECG -- so ``datasets/`` and the row-mapping
    adapters legitimately name it. What must stay clean is everything that
    answers a request: the engine, the CLI, the API, the decision layer and
    the agentic surface. A detection Mercury serves must not depend on an
    outbound call to a third-party clinical archive.
    """
    serving = ("engine.py", "cli.py", "api", "decision", "agentic", "mcp_server.py")
    offenders: dict[str, str] = {}
    for name in serving:
        target = _SRC / name
        paths = [target] if target.is_file() else sorted(target.rglob("*.py"))
        for path in paths:
            if "__pycache__" in path.parts:
                continue
            if "physionet" in path.read_text(encoding="utf-8").lower():
                offenders[str(path.relative_to(_SRC))] = "names PhysioNet"

    assert not offenders, (
        f"serving surfaces reference PhysioNet: {offenders}. It is a corpus "
        "Mercury is measured against, not a dependency of what Mercury serves."
    )


def test_only_the_measurement_corpora_construct_a_physionet_url() -> None:
    """A PhysioNet *URL* appears only where a corpus is downloaded.

    Checks URL-bearing string literals rather than the word "PhysioNet": the
    HTTP client's allowlist comment, the provenance registries and the
    row-mapping adapter all name it legitimately and fetch nothing from it.
    What would be a finding is a new module building a request to that host.
    """
    hosts = ("physionet.org", "physionet-open.s3")
    builders = []
    for path in sorted(_SRC.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value.startswith(("http://", "https://"))
                and any(h in node.value for h in hosts)
            ):
                builders.append(str(path.relative_to(_SRC)))
                break

    allowed = {
        "loaders/sepsis_loader.py",  # Challenge 2019 (sepsis)
        "datasets/medical.py",  # clinical benchmark corpora
        "datasets/mitbih.py",  # MIT-BIH arrhythmia
        "validation/data_loaders.py",  # deprecated wrappers over the above
        "security/input_validation.py",  # the TRUSTED_DOMAINS allowlist entry
    }
    unexpected = sorted(set(builders) - allowed)
    assert not unexpected, (
        f"new module(s) building a PhysioNet request: {unexpected}. Clinical "
        "corpora are downloaded by the measurement path only."
    )


def test_importing_the_loader_opens_no_connection() -> None:
    """The URLs are module constants, resolved at fetch time, not at import.

    Guards the shape rather than the behaviour: a module-level fetch would
    make merely importing the loaders package reach out to a clinical data
    host, which is the failure this file exists to prevent.
    """
    tree = ast.parse((_SRC / "loaders" / "sepsis_loader.py").read_text(encoding="utf-8"))
    fetching = {"_fetch_url", "_fetch_json", "_fetch_csv", "get", "urlopen", "request"}

    module_level_calls = [
        node.lineno
        for stmt in tree.body
        if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in ast.walk(stmt)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Attribute) and node.func.attr in fetching)
            or (isinstance(node.func, ast.Name) and node.func.id in fetching)
        )
    ]

    assert not module_level_calls, (
        f"sepsis_loader performs a fetch at module scope (lines {module_level_calls}); "
        "importing the loaders package would then call PhysioNet"
    )
