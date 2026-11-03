# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every text control in the Mercury Console has an accessible name.

A placeholder is not a label (WCAG 2.1 SC 1.3.1 and 4.1.2; WAI-ARIA 1.2
accessible-name computation uses ``aria-label`` / ``aria-labelledby`` / an
associated ``<label>`` and falls back to ``placeholder`` only as a last
resort that disappears once the field has a value). Rendered in Chromium
against ``mercury-agent serve``, the shipped template had seven controls with
only a placeholder: the command prompt, the agent composer, the menu filter,
the MCP tool-arguments field, the engine URL, the API key, its "remember"
checkbox, and the command-palette search. This test walks the compiled
template tree so the property cannot regress silently when the prototype is
rebuilt.
"""

from __future__ import annotations

import json
from typing import Any

from omni_mercury_engine.api import frontend

_TEMPLATE = frontend._STATIC_DIR / "console" / "template.js"


def _tree() -> list[Any]:
    text = _TEMPLATE.read_text(encoding="utf-8")
    body = text[text.index("=") + 1 :].strip().rstrip(";")
    tree: list[Any] = json.loads(body)
    return tree


def _rendered(attrs: dict[str, str]) -> bool:
    """A ``display:none`` control (a hidden file picker behind a button) is not in the tree."""
    return "display:none" not in attrs.get("style", "").replace(" ", "")


def _controls(
    node: Any, out: list[dict[str, str]], labelled_ids: set[str], in_label: bool = False
) -> None:
    if not (isinstance(node, list) and len(node) == 3 and isinstance(node[0], str)):
        return
    tag, attrs, kids = node
    if tag == "label" and attrs.get("for"):
        labelled_ids.add(attrs["for"])
    wrapped = in_label or tag == "label"
    if tag in ("input", "textarea", "select") and attrs.get("type") != "hidden":
        if _rendered(attrs):
            # A control nested inside ``<label>`` is named by that label.
            out.append({**attrs, "__wrapped_in_label": "1"} if wrapped else attrs)
    for kid in kids:
        _controls(kid, out, labelled_ids, wrapped)


def _has_name(attrs: dict[str, str], labelled_ids: set[str]) -> bool:
    if attrs.get("aria-label", "").strip() or attrs.get("aria-labelledby", "").strip():
        return True
    if attrs.get("__wrapped_in_label"):
        return True
    return bool(attrs.get("id")) and attrs["id"] in labelled_ids


def test_every_console_control_has_an_accessible_name() -> None:
    controls: list[dict[str, str]] = []
    labelled_ids: set[str] = set()
    for node in _tree():
        _controls(node, controls, labelled_ids)
    assert controls, "the template declares no form controls; the walk is broken"
    unnamed = [
        attrs.get("placeholder") or attrs.get("type") or "<no placeholder>"
        for attrs in controls
        if not _has_name(attrs, labelled_ids)
    ]
    assert unnamed == [], f"controls with a placeholder but no accessible name: {unnamed}"
