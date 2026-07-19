#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Enumerate every class in ``src/omni_mercury_engine`` straight from source.

This is the auditable source of truth for "what can Mercury do": it walks the
package with :mod:`ast` (no imports, no runtime) and records every top-level
class, its module, a coarse capability category (by name suffix / base), and
the first line of its docstring. It writes the grouped Markdown inventory
(``docs/CAPABILITY_INVENTORY.md``) and, on request, a machine-readable JSON
blob (the build intermediate for the browsable capability artifact).

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


def _categorize(name: str, bases: list[str]) -> tuple[str, bool]:
    """Return (category, is_capability) for a class name / base list."""
    is_enum = any(b.endswith("Enum") or b == "Enum" for b in bases)
    if is_enum or name.endswith(_SUPPORT_SUFFIXES):
        return "Support types (config / result / enum / error)", False
    for suffixes, category in _CATEGORY_RULES:
        if name.endswith(suffixes):
            return category, True
    return "Other capability classes", True


def _base_names(node: ast.ClassDef) -> list[str]:
    out: list[str] = []
    for b in node.bases:
        if isinstance(b, ast.Name):
            out.append(b.id)
        elif isinstance(b, ast.Attribute):
            out.append(b.attr)
    return out


def collect() -> dict[str, Any]:
    """Scan the package and return the structured inventory."""
    records: list[dict[str, Any]] = []
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
            bases = _base_names(node)
            category, is_cap = _categorize(node.name, bases)
            doc = ast.get_docstring(node) or ""
            summary = doc.strip().splitlines()[0].strip() if doc else ""
            records.append(
                {
                    "name": node.name,
                    "module": module,
                    "subsystem": subsystem,
                    "category": category,
                    "is_capability": is_cap,
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
