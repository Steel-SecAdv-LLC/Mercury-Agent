"""Sphinx configuration for the Mercury Agent documentation set.

This config renders the project's markdown documentation (the ``.md``
files in this directory) into static HTML via the ``myst_parser``
extension.  It deliberately does *not* run ``autodoc`` against the
source tree: Mercury Agent's runtime depends on optional native
extensions (the AMA Cryptography shared library, the GoSNN C++
compute kernels, FAISS) that are not always importable in the
documentation build environment.  Forcing autodoc to import every
module would either gate the docs build on the entire native
toolchain or — far worse — silently render an incomplete API
reference whenever an optional dependency is missing.

The narrative docs (``ARCHITECTURE.md``, ``MATH_SPEC.md``, etc.)
are the canonical surface for human readers; the in-source
docstrings are validated separately by ``pydocstyle`` (Google
convention) on every CI run, so docstring quality is enforced
without requiring a successful autodoc import.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# Project metadata
# -----------------------------------------------------------------------------

# Make the package importable for ``versioning``-style queries that don't
# require executing the module body.  We deliberately do *not* eagerly
# import the package — see the module docstring above.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

project = "Mercury Agent"
author = "Steel-SecAdv-LLC"
# Read the version straight out of ``pyproject.toml`` so we don't have
# to import the package during the docs build.
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - tomli fallback
    import tomli as tomllib  # type: ignore[no-redef]

with (_REPO_ROOT / "pyproject.toml").open("rb") as _f:
    _pyproject = tomllib.load(_f)
release = str(_pyproject.get("project", {}).get("version", "0.0.0"))
version = ".".join(release.split(".")[:2])
copyright = "Steel-SecAdv-LLC. Released under MIT."

# -----------------------------------------------------------------------------
# Extensions
# -----------------------------------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]
# ``sphinx_autodoc_typehints`` and ``sphinx.ext.autodoc`` are
# intentionally *not* loaded.  Importing the package would require the
# AMA native shared object plus optional ML dependencies, which would
# either gate the docs build on the entire native toolchain or
# silently produce a partial API reference whenever an optional
# dependency is missing.  Docstring quality is enforced separately by
# ``pydocstyle`` in CI.

# myst_parser configuration — enable the markdown features used across
# the existing narrative docs (heading anchors for cross-references,
# colon fences for admonitions, deflists for terminology blocks).
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "smartquotes",
    "substitution",
    "tasklist",
]
# ``linkify`` is intentionally not enabled — it would require an extra
# ``linkify-it-py`` runtime dep solely to auto-link bare URLs in the
# narrative docs, which already use explicit ``[text](url)`` syntax.
myst_heading_anchors = 4

# -----------------------------------------------------------------------------
# Source files
# -----------------------------------------------------------------------------

# We render markdown sources directly.  ``index.md`` is generated below
# in ``setup`` if it does not already exist on disk so a fresh checkout
# can build the docs without an extra committed file.
source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}
master_doc = "index"
language = "en"

# Exclude build outputs and the comprehensive audit report (which is a
# point-in-time deliverable, not a stable doc surface).
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

# -----------------------------------------------------------------------------
# HTML output
# -----------------------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path: list[str] = []
html_title = f"{project} {release}"
html_short_title = project
html_show_sphinx = False

# -----------------------------------------------------------------------------
# Cross-project references
# -----------------------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# -----------------------------------------------------------------------------
# Build hooks
# -----------------------------------------------------------------------------


def _ensure_index() -> Path:
    """Generate ``index.md`` from the existing markdown set if missing.

    Sphinx requires an entry point (``master_doc``).  Mercury Agent's
    documentation lives in flat-named markdown files at the docs root;
    we synthesise an index that toctree-links them in a sensible
    reading order so the build succeeds without committing a generated
    artefact.
    """
    docs_root = Path(__file__).resolve().parent
    index = docs_root / "index.md"
    if index.exists():
        return index
    # Order chosen to mirror the README's "start here" reading path:
    # install → architecture → math/spec → operational refs → audit notes.
    ordered = [
        "INSTALLATION",
        "ARCHITECTURE",
        "API_REFERENCE",
        "MATH_SPEC",
        "BENCHMARKS",
        "DOMAIN_PERFORMANCE",
        "ROUTING_GUIDE",
        "DATASOURCES",
        "LIVE_DATA_VALIDATION",
        "ORACLE_NOISE_COLOR",
        "DEPLOYMENT",
        "CROSS_DOMAIN_ANALYSIS",
    ]
    available = {p.stem: p for p in docs_root.glob("*.md") if p.name != "index.md"}
    entries = [name for name in ordered if name in available]
    # Append any markdown file we didn't explicitly list so we never
    # silently drop a doc surface from the index.
    for name in sorted(available):
        if name not in entries:
            entries.append(name)
    body = [
        "# Mercury Agent Documentation",
        "",
        f"Version `{release}` — Steel-SecAdv-LLC.",
        "",
        "Mercury Agent is the orchestration / cognition layer of the",
        "FIND**Ω**YOU stack.  It is paired with",
        "[AMA Cryptography](https://github.com/Steel-SecAdv-LLC/AMA-Cryptography)",
        "for the post-quantum cryptographic substrate.",
        "",
        "```{toctree}",
        ":maxdepth: 2",
        ":caption: Contents",
        "",
    ]
    body.extend(entries)
    body.extend(["```", ""])
    index.write_text("\n".join(body), encoding="utf-8")
    return index


_ensure_index()

# Treat warnings as build failures *only* when explicitly opted in
# (``SPHINXOPTS="-W"``).  CI invokes ``sphinx-build`` without ``-W`` so
# the build stays decoupled from third-party warning churn (e.g. a
# myst-parser deprecation), but the local ``make linkcheck``-style
# stricter run can still surface them via ``SPHINXOPTS``.
nitpicky = bool(os.environ.get("SPHINX_NITPICKY"))
