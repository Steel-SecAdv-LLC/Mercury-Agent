#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Enumerate every class in ``src/omni_mercury_engine`` straight from source.

This is the auditable source of truth for "what can Mercury do": it walks the
package with :mod:`ast` (no imports, no runtime) and records every top-level
class, its module, a coarse capability category, and the first line of its
docstring. It writes the grouped Markdown inventory
(``docs/CAPABILITY_INVENTORY.md``) and, on request, a machine-readable JSON
blob (the build intermediate for the browsable capability artifact).

Categorization is name-suffix first, then **base-class analysis** for anything
the name leaves in the ``Other`` bucket: a two-pass ``ast`` walk builds a global
``{class: bases}`` map and resolves a class by walking its ancestor chain
(``nn.Module`` subclasses become neural components, ``Protocol``/``TypedDict``
subclasses become support types, in-tree ``Base*`` ancestors carry their own
suffix down to the subclass). This is honest about its limits: a class that
inherits only from ``object`` carries no ancestral signal, so the majority of
the residual ``Other`` bucket cannot be refined this way and is reported as
``unresolved`` rather than force-fit.

Run::

    python scripts/generate_capability_inventory.py                 # write the Markdown
    python scripts/generate_capability_inventory.py --json out.json  # + JSON blob
    python scripts/generate_capability_inventory.py --check         # drift check

The point is anti-theatre: a capability is only real if a class implements it,
so the inventory is generated from the classes themselves, never hand-curated.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src" / "omni_mercury_engine"
_MD_OUT = _REPO / "docs" / "CAPABILITY_INVENTORY.md"

# Ordered (suffix -> category). First match wins, so specific before generic.
_CATEGORY_RULES: list[tuple[tuple[str, ...], str]] = [
    (("Detector",), "Detection"),
    (("Predictor", "Forecaster", "Estimator"), "Prediction & forecasting"),
    (("Analyzer", "Analyser", "Assessor", "Calculator", "Classifier"), "Analysis & scoring"),
    (("Monitor", "Tracker"), "Monitoring"),
    (("Gate", "Enforcer", "Governor", "Guard", "Bound", "Bounder"), "Ethics & governance"),
    (("Optimizer", "Calibrator", "Trainer", "Aggregator", "Scheduler"), "Training & optimization"),
    (("Adapter", "Backend", "Client", "Server", "Connector"), "Adapters & backends"),
    (("Loader", "Source", "Retriever"), "Data sources & loaders"),
    (
        (
            "Engine",
            "System",
            "Coordinator",
            "Orchestrator",
            "Manager",
            "Controller",
            "Pipeline",
            "Hub",
            "Router",
            "Chain",
            "Agent",
        ),
        "Engines & orchestration",
    ),
    (("Recognizer", "Matcher"), "Biometric & recognition"),
    (
        (
            "Network",
            "Model",
            "Encoder",
            "Decoder",
            "Transformer",
            "Autoencoder",
            "LSTM",
            "CNN",
            "GAN",
            "VAE",
            "Layer",
            "Attention",
        ),
        "Neural models & layers",
    ),
    (("Scorer", "Solver", "Simulator", "Prover", "Verifier"), "Solvers & scorers"),
]

# Suffixes / bases that mark a class as a support type, not a capability.
_SUPPORT_SUFFIXES = (
    "Config",
    "Result",
    "Error",
    "Exception",
    "Type",
    "State",
    "Status",
    "Reading",
    "Record",
    "Info",
    "Event",
    "Response",
    "Request",
    "Report",
    "Batch",
    "Update",
    "Message",
    "Mixin",
    "Data",
    "Params",
    "Options",
    "Settings",
    "Metadata",
)


_NEURAL = "Neural models & layers"
_DATA = "Data sources & loaders"
_SUPPORT = "Support types (config / result / enum / error)"
_OTHER = "Other capability classes"

# Well-known base classes whose SIGNAL is unambiguous even when the subclass's
# own name carries no suffix. This is the base-class-analysis table that refines
# the "Other" bucket: e.g. ``class Discriminator(nn.Module)`` is a neural layer
# though "Discriminator" ends in no rule suffix. Kept conservative — only bases
# whose meaning is not in doubt (a ``Protocol`` subclass IS an interface; an
# ``nn.Module`` subclass IS a neural component). Ambiguous bases (``ABC``) are
# deliberately absent so they fall through to name/ancestor resolution.
_BASE_CATEGORY: dict[str, tuple[str, bool]] = {
    "Module": (_NEURAL, True),  # torch nn.Module (ast sees the ``.attr`` "Module")
    "BaseFusionModule": (_NEURAL, True),
    "Dataset": (_DATA, True),
    "IterableDataset": (_DATA, True),
    "Protocol": (_SUPPORT, False),
    "TypedDict": (_SUPPORT, False),
    "NamedTuple": (_SUPPORT, False),
}


def _categorize_by_name(name: str, bases: list[str]) -> tuple[str, bool] | None:
    """Return (category, is_capability) from the class's OWN name, or None.

    None means "the name carries no categorizing suffix" — the caller then falls
    back to base-class analysis before landing the class in the Other bucket.
    """
    is_enum = any(b.endswith("Enum") or b == "Enum" for b in bases)
    if is_enum or name.endswith(_SUPPORT_SUFFIXES):
        return _SUPPORT, False
    for suffixes, category in _CATEGORY_RULES:
        if name.endswith(suffixes):
            return category, True
    return None


def _resolve_via_bases(
    bases: list[str],
    class_bases: dict[str, list[str]],
    seen: set[str],
    depth: int = 0,
) -> tuple[str, bool] | None:
    """Resolve a category by walking the base-class chain (MRO-style, ast-only).

    Consults the ``_BASE_CATEGORY`` table, then each base's own name suffix, then
    recurses into that base's bases (for in-tree ancestors). A ``seen`` set and a
    depth bound guard against cycles / pathological hierarchies. Returns the
    first hit, or None if no ancestor carries a signal.
    """
    for base in bases:
        if base in _BASE_CATEGORY:
            return _BASE_CATEGORY[base]
        # Recurse through IN-TREE ancestors so a curated signal (e.g. nn.Module)
        # propagates through an intermediate project base. Name-suffix guessing on
        # base names is deliberately NOT done here: it misfires on cross-cutting
        # mixins (``LoggerMixin`` must not demote its subclass to a support type)
        # and on shadowed externals (an in-tree ``BaseModel`` that fronts pydantic
        # must not be read as a neural "Model"). Only the curated table speaks.
        if depth < 8 and base in class_bases and base not in seen:
            seen.add(base)
            resolved = _resolve_via_bases(class_bases[base], class_bases, seen, depth + 1)
            if resolved is not None:
                return resolved
    return None


def categorize(
    name: str, bases: list[str], class_bases: dict[str, list[str]]
) -> tuple[str, bool, str]:
    """Return (category, is_capability, resolution) using name then base analysis.

    ``resolution`` is ``"name"`` (own-name suffix / enum), ``"base"`` (refined via
    the ancestor chain), or ``"unresolved"`` (genuinely Other — no name suffix and
    no informative ancestor, e.g. an ``object``-only class).
    """
    own = _categorize_by_name(name, bases)
    if own is not None:
        return own[0], own[1], "name"
    resolved = _resolve_via_bases(bases, class_bases, set())
    if resolved is not None:
        return resolved[0], resolved[1], "base"
    return _OTHER, True, "unresolved"


def _base_names(node: ast.ClassDef) -> list[str]:
    out: list[str] = []
    for b in node.bases:
        if isinstance(b, ast.Name):
            out.append(b.id)
        elif isinstance(b, ast.Attribute):
            out.append(b.attr)
    return out


def collect() -> dict[str, Any]:
    """Scan the package and return the structured inventory.

    Two passes over the ``ast`` (still no imports, no runtime): pass 1 builds a
    global ``{class_name: [base_names]}`` map so pass 2 can resolve a class's
    category by walking its ancestor chain when its own name carries no suffix —
    the base-class analysis that refines the ``Other`` bucket. Name-keyed base
    resolution is a heuristic (class names can collide across modules), which is
    acceptable for this heuristic, auditable tool and is recorded per record via
    the ``resolution`` field.
    """
    scanned: list[tuple[ast.ClassDef, str, str]] = []
    class_bases: dict[str, list[str]] = {}
    for path in sorted(_SRC.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(_SRC)
        module = ".".join(rel.with_suffix("").parts)
        subsystem = rel.parts[0] if len(rel.parts) > 1 else "(top-level)"
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for node in tree.body:  # top-level classes only
            if not isinstance(node, ast.ClassDef):
                continue
            scanned.append((node, module, subsystem))
            class_bases.setdefault(node.name, _base_names(node))

    records: list[dict[str, Any]] = []
    for node, module, subsystem in scanned:
        bases = _base_names(node)
        category, is_cap, resolution = categorize(node.name, bases, class_bases)
        doc = ast.get_docstring(node) or ""
        summary = doc.strip().splitlines()[0].strip() if doc else ""
        records.append(
            {
                "name": node.name,
                "module": module,
                "subsystem": subsystem,
                "category": category,
                "is_capability": is_cap,
                "resolution": resolution,
                "summary": summary,
            }
        )

    subsystems: dict[str, list[dict[str, Any]]] = {}
    categories: dict[str, int] = {}
    for r in records:
        subsystems.setdefault(r["subsystem"], []).append(r)
        categories[r["category"]] = categories.get(r["category"], 0) + 1

    return {
        "total_classes": len(records),
        "capability_classes": sum(1 for r in records if r["is_capability"]),
        "subsystem_count": len(subsystems),
        "refined_by_base": sum(1 for r in records if r["resolution"] == "base"),
        "unresolved_other": sum(
            1 for r in records if r["category"] == _OTHER and r["is_capability"]
        ),
        "categories": dict(sorted(categories.items(), key=lambda kv: -kv[1])),
        "records": records,
    }


def render_markdown(inv: dict[str, Any]) -> str:
    """Render the grouped Markdown inventory."""
    lines: list[str] = [
        "<!--",
        "Copyright (C) 2025 Steel Security Advisors LLC",
        "SPDX-License-Identifier: GPL-3.0-or-later",
        "-->",
        "",
        "# Mercury Agent — Capability Inventory",
        "",
        "> Generated from source by `scripts/generate_capability_inventory.py` "
        "(`ast` walk of `src/omni_mercury_engine`, no runtime). Every row is a "
        "class that exists in the tree — this is the auditable answer to "
        '"what can Mercury do", not a hand-curated list. Re-run to refresh.',
        "",
        f"- **Total top-level classes:** {inv['total_classes']:,}",
        f"- **Capability-bearing classes:** {inv['capability_classes']:,} "
        "(excludes config/result/enum/error support types)",
        f"- **Subsystems (top-level packages):** {inv['subsystem_count']}",
        f"- **Refined via base-class analysis:** {inv['refined_by_base']:,} classes "
        "categorized from their ancestor chain (e.g. `nn.Module` subclasses whose "
        "own name carries no suffix)",
        f"- **Unresolved (`Other`):** {inv['unresolved_other']:,} — no name suffix and "
        "no informative ancestor (predominantly `object`-only classes, which "
        "base-class analysis cannot refine).",
        "",
        "## Capability classes by category",
        "",
        "| Category | Count |",
        "|---|---|",
    ]
    for cat, n in inv["categories"].items():
        lines.append(f"| {cat} | {n} |")
    lines.append("")
    lines.append("## Classes by subsystem")
    lines.append("")

    by_sub: dict[str, list[dict[str, Any]]] = {}
    for r in inv["records"]:
        by_sub.setdefault(r["subsystem"], []).append(r)

    for sub in sorted(by_sub):
        recs = by_sub[sub]
        caps = sum(1 for r in recs if r["is_capability"])
        lines.append(f"### `{sub}/` — {len(recs)} classes ({caps} capability)")
        lines.append("")
        # capability classes first, grouped by category, then support types
        cap_recs = [r for r in recs if r["is_capability"]]
        sup_recs = [r for r in recs if not r["is_capability"]]
        by_cat: dict[str, list[dict[str, Any]]] = {}
        for r in cap_recs:
            by_cat.setdefault(r["category"], []).append(r)
        for cat in sorted(by_cat):
            lines.append(f"**{cat}**")
            lines.append("")
            for r in sorted(by_cat[cat], key=lambda x: x["name"]):
                summary = f" — {r['summary']}" if r["summary"] else ""
                lines.append(f"- `{r['name']}` (`{r['module']}`){summary}")
            lines.append("")
        if sup_recs:
            names = ", ".join(f"`{r['name']}`" for r in sorted(sup_recs, key=lambda x: x["name"]))
            lines.append(f"<details><summary>Support types ({len(sup_recs)})</summary>")
            lines.append("")
            lines.append(names)
            lines.append("")
            lines.append("</details>")
            lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="Fail if the committed Markdown inventory is stale.",
    )
    ap.add_argument(
        "--json",
        metavar="PATH",
        help="Also write the machine-readable JSON blob here "
        "(build intermediate for the browsable artifact; not committed).",
    )
    args = ap.parse_args()

    inv = collect()
    md = render_markdown(inv)

    if args.check:
        if not _MD_OUT.exists() or _MD_OUT.read_text(encoding="utf-8") != md:
            print(
                f"STALE: {_MD_OUT.relative_to(_REPO)} — "
                "re-run scripts/generate_capability_inventory.py"
            )
            return 1
        print(f"Capability inventory fresh ({inv['total_classes']} classes).")
        return 0

    _MD_OUT.write_text(md, encoding="utf-8")
    msg = f"Wrote {_MD_OUT.relative_to(_REPO)}"
    if args.json:
        Path(args.json).write_text(
            json.dumps(inv, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        msg += f" and {args.json}"
    print(
        f"{msg}: {inv['total_classes']} classes, "
        f"{inv['capability_classes']} capability, {inv['subsystem_count']} subsystems."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
