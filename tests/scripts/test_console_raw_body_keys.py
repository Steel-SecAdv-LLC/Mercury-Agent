# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every free-form route documents the keys its service enforces, and the console reads them.

The assessment routes take ``dict[str, Any]`` bodies, so the OpenAPI document
carries no fields for them; the service enforces the keys itself through
``_obj(payload, name, required, optional)``. The route description states the
contract on a ``Body:`` line, the console catalog reads that line
(``documented_body_keys``) into ``keys`` and ``reqKeys`` on the raw body
field, and this test keeps the three in step: for each raw route, the
documented keys and the required ones must equal the sets the service's
``_obj`` call enforces, read from the code with ``ast`` so a key added to
the service without its documentation fails here.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi", reason="the OpenAPI document needs the [api] extra")

REPO = Path(__file__).resolve().parents[2]
SERVICE = REPO / "src" / "omni_mercury_engine" / "assessment" / "service.py"
ROUTES = REPO / "src" / "omni_mercury_engine" / "api" / "routes" / "assessment.py"

_spec = importlib.util.spec_from_file_location(
    "build_console_catalog", REPO / "scripts" / "build_console_catalog.py"
)
assert _spec is not None and _spec.loader is not None
catalog = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(catalog)


def _string_set(node: ast.expr, constants: dict[str, list[str]]) -> list[str]:
    """The literal strings in a ``{"a", "b", *_NAMES}`` set expression."""
    out: list[str] = []
    for elt in getattr(node, "elts", []):
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
            out.append(elt.value)
        elif isinstance(elt, ast.Starred) and isinstance(elt.value, ast.Name):
            out.extend(constants[elt.value.id])
    return out


def _service_key_sets() -> dict[str, tuple[set[str], set[str]]]:
    """``{service function: (required, optional)}`` from each ``_obj(payload, …)`` call."""
    tree = ast.parse(SERVICE.read_text(encoding="utf-8"))
    constants: dict[str, list[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Tuple):
                constants[target.id] = _string_set(node.value, {})
    sets: dict[str, tuple[set[str], set[str]]] = {}
    for fn in tree.body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        for call in ast.walk(fn):
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "_obj"
                and call.args
                and isinstance(call.args[0], ast.Name)
                and call.args[0].id == "payload"
            ):
                continue
            required = set(_string_set(call.args[2], constants))
            optional = set(_string_set(call.args[3], constants)) if len(call.args) > 3 else set()
            sets[fn.name] = (required, optional)
    return sets


def _route_to_service() -> dict[str, str]:
    """``{"POST /api/v1/assessment/…": service function}`` from the route handlers."""
    tree = ast.parse(ROUTES.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for fn in tree.body:
        if not isinstance(fn, ast.AsyncFunctionDef):
            continue
        path = None
        for deco in fn.decorator_list:
            if (
                isinstance(deco, ast.Call)
                and isinstance(deco.func, ast.Attribute)
                and deco.func.attr == "post"
                and deco.args
                and isinstance(deco.args[0], ast.Constant)
                and isinstance(deco.args[0].value, str)
            ):
                path = deco.args[0].value
        if path is None:
            continue
        for call in ast.walk(fn):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "_run"
                and call.args
                and isinstance(call.args[0], ast.Attribute)
            ):
                out[f"POST /api/v1/assessment{path}"] = call.args[0].attr
    return out


def _raw_routes() -> dict[str, dict[str, Any]]:
    _listing, schemas = catalog.build_api()
    return {
        key: fields["body"]
        for key, fields in schemas.items()
        if "body" in fields and fields["body"].get("raw")
    }


def test_every_raw_route_documents_exactly_the_keys_its_service_enforces() -> None:
    raw = _raw_routes()
    assert raw, "no free-form routes found in the OpenAPI document"
    services = _service_key_sets()
    mapping = _route_to_service()
    for route, field in raw.items():
        assert field.get("keys"), f"{route} documents no body keys"
        assert route in mapping, f"{route} does not hand its payload to a service function"
        required, optional = services[mapping[route]]
        assert set(field["keys"]) == required | optional, route
        assert set(field["reqKeys"]) == required, route
        assert field["desc"].startswith("Free-form JSON body · documented keys: "), route


@pytest.mark.parametrize(
    ("text", "keys", "required"),
    [
        ("Body: `a` (required), `b`, `c`.", ["a", "b", "c"], ["a"]),
        ("Body: `a`, optional `b`, `c`.", ["a", "b", "c"], ["a"]),
        (
            "Body: `a`, `b`\n(Record with scope `x`), `c`; optional\n`d`.",
            ["a", "b", "c", "d"],
            ["a", "b", "c"],
        ),
        ("Body: `a`, `b`.", ["a", "b"], []),
        ("No body line here.", [], []),
        (None, [], []),
    ],
)
def test_the_body_line_parser(text: str | None, keys: list[str], required: list[str]) -> None:
    assert catalog.documented_body_keys(text) == (keys, required)
