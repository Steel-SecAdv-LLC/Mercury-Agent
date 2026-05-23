"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.

------------------------------------------------------------------------

Operator tool: diff the public ``omni_mercury_engine.*`` re-export
surface between two refs (or between HEAD and a saved snapshot).

Catches accidental ABI-breaking removals before release.  Operates
purely on Python introspection so it requires no special parser.

Two modes:

* ``--snapshot path.json``  — emit a snapshot of the current public
  surface (top-level public attributes of ``omni_mercury_engine``).
* ``--against path.json``  — diff the current public surface against
  the saved snapshot.  Any removal is a hard finding.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
from pathlib import Path
from typing import Any

from omni_mercury_engine.tools._base import Certificate, atomic_write_text, run_tool

_SCHEMA = "mercury.tools.api_contract_diff/v1"
_MODULE = "omni_mercury_engine"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.api_contract_diff",
        description=(
            "Snapshot or diff the public re-export surface of "
            "omni_mercury_engine to catch ABI-breaking removals."
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--snapshot",
        metavar="PATH",
        help="Emit a JSON snapshot of the current public surface to PATH.",
    )
    group.add_argument(
        "--against",
        metavar="PATH",
        help="Diff the current public surface against the snapshot at PATH.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute the snapshot/diff but do NOT write the snapshot file.",
    )
    return parser


def _classify(obj: Any) -> str:
    if inspect.isclass(obj):
        return "class"
    if inspect.isfunction(obj) or inspect.isbuiltin(obj):
        return "function"
    if inspect.ismodule(obj):
        return "module"
    return type(obj).__name__


def _signature(obj: Any) -> str | None:
    """Return the full ``inspect.signature`` string for ``obj``.

    Captures parameter names, defaults, annotations, and the return
    annotation — the same level of detail an external auditor would
    review.  Falls back to ``None`` for objects whose signature cannot
    be introspected (e.g. C builtins, classes without ``__init__``).
    """
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return None


def _signature_detail(obj: Any) -> dict[str, Any] | None:
    """Return a structured signature view: parameters, defaults, return.

    The structured form is what ``api_contract_diff`` diffs across
    snapshots so the diff highlights parameter renames and default
    changes — not just the textual signature string.
    """
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return None
    params: list[dict[str, Any]] = []
    for p in sig.parameters.values():
        params.append(
            {
                "name": p.name,
                "kind": p.kind.name,
                "default": (
                    "<no-default>" if p.default is inspect.Parameter.empty else repr(p.default)
                ),
                "annotation": (
                    "<no-annotation>"
                    if p.annotation is inspect.Parameter.empty
                    else str(p.annotation)
                ),
            }
        )
    return {
        "parameters": params,
        "return": (
            "<no-annotation>"
            if sig.return_annotation is inspect.Signature.empty
            else str(sig.return_annotation)
        ),
    }


def _snapshot_surface() -> dict[str, Any]:
    """Snapshot the public re-export surface of :mod:`omni_mercury_engine`.

    Walks ``__all__`` (or the public ``dir()`` if absent) and records the
    kind/signature/module of each exported symbol.  Two failure modes
    are distinguished so the snapshot reflects the actual cause:

    * **Optional-backend lazy import failure** (``ImportError`` /
      ``ModuleNotFoundError``).  E.g. ``OmniMercuryEngine`` pulls in
      :mod:`omni_mercury_engine.engine` → :mod:`torch`.  Recorded as
      ``"kind": "unavailable"`` with the lazy-import error preserved.
      The snapshot does NOT crash because of a single unreachable
      symbol — the diff path can still observe its presence in
      ``__all__``.
    * **Symbol listed in ``__all__`` but not actually exported**
      (``AttributeError``).  This is an ABI break the tool exists to
      detect, not a transient backend gap — recorded as
      ``"kind": "missing_export"`` so the diff path surfaces it as a
      structural problem (different from "optional backend missing").
    """
    mod = importlib.import_module(_MODULE)
    public = getattr(mod, "__all__", None)
    if public is None:
        public = sorted(name for name in dir(mod) if not name.startswith("_"))
    entries: dict[str, dict[str, Any]] = {}
    unavailable: dict[str, str] = {}
    missing_exports: dict[str, str] = {}
    for name in sorted(public):
        try:
            obj = getattr(mod, name)
        except ImportError as exc:
            # Lazy ``__getattr__`` resolution failed because an optional
            # ML backend is not installed.  Record the symbol with the
            # lazy-import error so the diff tool can still observe its
            # presence in ``__all__`` and surface a clean "unavailable"
            # classification to the operator rather than aborting the
            # entire snapshot.
            unavailable[name] = f"{type(exc).__name__}: {exc}"
            entries[name] = {
                "kind": "unavailable",
                "signature": None,
                "signature_detail": None,
                "module": _MODULE,
                "qualname": name,
                "lazy_import_error": unavailable[name],
            }
            continue
        except AttributeError as exc:
            # ``__all__`` lists a name the module does not actually
            # expose.  This is the ABI break the tool exists to detect,
            # not a transient backend gap.  Record it distinctly so the
            # diff path can surface it as a structural problem.
            missing_exports[name] = f"{type(exc).__name__}: {exc}"
            entries[name] = {
                "kind": "missing_export",
                "signature": None,
                "signature_detail": None,
                "module": _MODULE,
                "qualname": name,
                "attribute_error": missing_exports[name],
            }
            continue
        if obj is None:
            continue
        entries[name] = {
            "kind": _classify(obj),
            "signature": _signature(obj),
            "signature_detail": _signature_detail(obj),
            "module": getattr(obj, "__module__", None),
            "qualname": getattr(obj, "__qualname__", name),
        }
    snapshot: dict[str, Any] = {
        "module": _MODULE,
        "version": getattr(mod, "__version__", None),
        "public_count": len(entries),
        "entries": entries,
    }
    if unavailable:
        snapshot["unavailable_lazy_imports"] = unavailable
    if missing_exports:
        snapshot["missing_exports"] = missing_exports
    return snapshot


def _diff(saved: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    saved_e = saved.get("entries", {})
    current_e = current.get("entries", {})
    removed = sorted(set(saved_e) - set(current_e))
    added = sorted(set(current_e) - set(saved_e))
    changed: list[dict[str, Any]] = []
    for name in sorted(set(saved_e) & set(current_e)):
        s, c = saved_e[name], current_e[name]
        # We diff the textual signature, the structured detail, and the
        # kind.  The structured detail catches parameter renames and
        # default changes that a textual diff would also catch but
        # the structured form makes mechanically reviewable.
        if (
            s.get("signature") != c.get("signature")
            or s.get("signature_detail") != c.get("signature_detail")
            or s.get("kind") != c.get("kind")
        ):
            changed.append({"name": name, "before": s, "after": c})
    return {
        "removed": removed,
        "added": added,
        "changed": changed,
    }


def _collect(args: argparse.Namespace) -> Certificate:
    if args.snapshot:
        snap = _snapshot_surface()
        if not args.dry_run:
            atomic_write_text(
                Path(args.snapshot), json.dumps(snap, indent=2, sort_keys=True) + "\n"
            )
        snap_warnings: list[str] = []
        # ``missing_exports`` is a structural ABI break — a name listed in
        # ``__all__`` that does not resolve via ``getattr``.  Surface it as
        # a warning on the snapshot envelope so the operator sees it even
        # before the diff lane runs.  Optional-backend ``unavailable``
        # entries are NOT escalated: they are an expected configuration
        # gap, not an ABI defect.
        for name, err in (snap.get("missing_exports") or {}).items():
            snap_warnings.append(f"missing export in __all__: {name} ({err})")
        snap_status = "warn" if snap_warnings else "ok"
        return Certificate(
            tool="api_contract_diff",
            schema=_SCHEMA,
            status=snap_status,
            body={
                "mode": "snapshot",
                "output": args.snapshot,
                "dry_run": bool(args.dry_run),
                **snap,
            },
            warnings=snap_warnings,
        )

    saved = json.loads(Path(args.against).read_text())
    current = _snapshot_surface()
    diff = _diff(saved, current)
    warnings: list[str] = []
    for name in diff["removed"]:
        warnings.append(f"removed public symbol: {name}")
    for entry in diff["changed"]:
        warnings.append(
            f"signature changed: {entry['name']} "
            f"{entry['before'].get('signature')} → {entry['after'].get('signature')}"
        )
    status = "fail" if (diff["removed"] or diff["changed"]) else "ok"
    return Certificate(
        tool="api_contract_diff",
        schema=_SCHEMA,
        status=status,
        body={"mode": "diff", "against": args.against, "current": current, "diff": diff},
        warnings=warnings,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
