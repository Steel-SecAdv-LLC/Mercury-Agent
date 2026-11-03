# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generate the Mercury Console catalog (``static/console/catalog.js``) by introspection.

The console's command builder, detector/loader/MCP/API catalogs, request
schemas and vocabulary chips are all read from the code they describe, so
the browser cannot drift from the engine:

* commands: the click group ``omni_mercury_engine.cli:main`` (options, types,
  defaults, choices, flags and the ``mercury-agent ...`` docstring examples);
* detectors: ``core.detector_registry.DETECTOR_MANIFEST`` joined with
  ``core.detector_ontology.ONTOLOGY``;
* loaders and hazards: ``loaders/__init__.py`` and
  ``detectors.hazard_diagnostics.KNOWN_HAZARDS`` (parsed, not imported);
* MCP tools: ``MercuryMCPServer`` tool specs, with their input schemas;
* API routes and request schemas: ``api.server.app.openapi()``;
* runs: ``benchmarks/``, ``scripts/`` and ``examples/`` with their argparse
  flags (parsed with ``ast``; never imported or executed);
* vocabularies: ``voice --domain`` from ``_build_safe_domain_labels()`` and
  ``voice --model`` from ``OllamaModel`` (parsed, so torch is not imported);
* library classes and predictors: hand-described in
  ``scripts/console_catalog_curated.json`` and verified here class by class
  and method by method with ``ast``, so a description cannot outlive its code.

Usage::

    python scripts/build_console_catalog.py           # write catalog.js
    python scripts/build_console_catalog.py --check   # exit 1 if stale
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import logging
import os
import re
import sys
import tokenize
from pathlib import Path
from typing import Any

from omni_mercury_engine.core.knowledge_profiles import DETECTOR_PROFILES, DOMAIN_PROFILES

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "omni_mercury_engine"
OUTPUT = SRC / "api" / "frontend_assets" / "static" / "console" / "catalog.js"
CURATED = REPO / "scripts" / "console_catalog_curated.json"

#: Console navigation groups for top-level CLI commands. Every top-level
#: command must appear here (the generator fails otherwise), so a new command
#: is placed deliberately rather than falling into a catch-all.
COMMAND_SECTIONS: dict[str, str] = {
    "detect": "Detection",
    "tier-detect": "Detection",
    "rca": "Detection",
    "causal": "Detection",
    "explain": "Detection",
    "stream": "Detection",
    "hazard-viz": "Hazards",
    "physics": "Physics",
    "train": "Models",
    "tune": "Models",
    "build-features": "Models",
    "symbolic-rules": "Models",
    "biometric": "Security & identity",
    "security": "Security & identity",
    "verify-corpus": "Security & identity",
    "serve": "Services",
    "mcp": "Services",
    "voice": "Services",
    "intel": "Intel",
    "assess": "Assessment",
    "adjudication": "Adjudication",
    "platform": "Platform",
    "decouple": "Platform",
    "observatory": "Observatory",
    "tool": "Tools",
}
SECTIONS: tuple[str, ...] = (
    "Detection",
    "Hazards",
    "Physics",
    "Models",
    "Security & identity",
    "Services",
    "Intel",
    "Assessment",
    "Adjudication",
    "Platform",
    "Observatory",
    "Tools",
)

#: OpenAPI tag -> console API group.
API_GROUPS: dict[str, str] = {
    "Health": "Service",
    "Service": "Service",
    "Detection": "Detection",
    "Batch Processing": "Batch",
    "Hazard Visualization": "Hazard",
    "Export": "Export",
    "Model Registry": "Models",
    "Assessment": "Assessment",
    "Voice": "Voice",
    "Authentication": "Accounts",
}

#: Run categories, first match wins (path substring rules).
_RUN_RULES: tuple[tuple[str, str], ...] = (
    ("examples/", "Examples & demos"),
    ("train", "Training & retraining"),
    ("retrain", "Training & retraining"),
    ("honest", "Domain benchmarks (honest)"),
    ("ablation", "Ablations & sweeps"),
    ("sweep", "Ablations & sweeps"),
    ("latency", "Latency & performance"),
    ("performance", "Latency & performance"),
    ("throughput", "Latency & performance"),
    ("hardware", "Latency & performance"),
    ("red_team", "Safety, ethics & red-team"),
    ("weapons", "Safety, ethics & red-team"),
    ("harm", "Safety, ethics & red-team"),
    ("ethic", "Safety, ethics & red-team"),
    ("sigma", "Safety, ethics & red-team"),
    ("calibrat", "Validation & calibration"),
    ("validat", "Validation & calibration"),
    ("verify", "Ops, verification & CI"),
    ("check_", "Ops, verification & CI"),
    ("record_", "Data, fixtures & ingestion"),
    ("prefetch", "Data, fixtures & ingestion"),
    ("ingest", "Data, fixtures & ingestion"),
    ("loader", "Data, fixtures & ingestion"),
    ("benchmarks/", "Benchmark suites"),
)


def _first_line(text: str | None) -> str:
    for line in (text or "").strip().splitlines():
        line = line.strip()
        if line:
            return line.rstrip()
    return ""


def _jsonable(value: Any) -> bool:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


# ---------------------------------------------------------------------------
# Commands (click)
# ---------------------------------------------------------------------------

_EXAMPLE = re.compile(r"^\s*(?:\$\s*)?mercury-agent\s+(.+?)\s*$")


def _examples(doc: str | None) -> list[str]:
    """``mercury-agent ...`` lines in a docstring, continuations joined, comments dropped."""
    lines: list[str] = []
    pending = ""
    for raw in (doc or "").splitlines():
        line = raw.strip()
        if pending:
            pending += " " + line.rstrip("\\").strip()
            if not line.endswith("\\"):
                lines.append(pending)
                pending = ""
            continue
        if _EXAMPLE.match(raw):
            if line.endswith("\\"):
                pending = line.rstrip("\\").strip()
            else:
                lines.append(line)
    if pending:
        lines.append(pending)
    out: list[str] = []
    for line in lines:
        match = _EXAMPLE.match(line)
        if not match:
            continue
        example = re.sub(r"\s+#.*$", "", match.group(1)).strip()
        example = re.sub(r"\s{2,}", " ", example)
        if example and example not in out:
            out.append(example)
    return out


def _click_type(param: Any) -> str | None:
    import click

    kind = param.type
    if isinstance(kind, click.Choice):
        return None  # choices carry the information
    name = getattr(kind, "name", "") or ""
    if getattr(param, "is_flag", False):
        return None
    mapping = {"float": "float", "integer": "int", "boolean": "bool", "path": "path"}
    if isinstance(kind, click.FloatRange):
        return "float"
    if isinstance(kind, click.IntRange):
        return "int"
    if isinstance(kind, click.File):
        return "path"
    return mapping.get(name)


def _option(param: Any) -> dict[str, Any]:
    import click

    flags = list(param.opts) + list(param.secondary_opts)
    longs = [f for f in param.opts if f.startswith("--")]
    entry: dict[str, Any] = {"k": longs[0] if longs else param.opts[0], "flags": flags}
    if param.required:
        entry["req"] = 1
    if getattr(param, "is_flag", False):
        entry["flag"] = 1
    if param.multiple:
        entry["multiple"] = 1
    kind = _click_type(param)
    if kind:
        entry["type"] = kind
    if isinstance(param.type, click.Choice):
        entry["choices"] = [str(c) for c in param.type.choices]
    default = param.default
    if (
        not callable(default)
        and default is not None
        and default != ()
        and not (entry.get("flag") and default is False)
        and type(default).__name__ != "Sentinel"
        and _jsonable(default)
        and repr(default) != "Sentinel.UNSET"
    ):
        entry["def"] = list(default) if isinstance(default, tuple) else default
    if param.help:
        entry["help"] = " ".join(param.help.split())
    if param.envvar:
        entry["env"] = param.envvar if isinstance(param.envvar, str) else list(param.envvar)
    return entry


def build_commands() -> list[dict[str, Any]]:
    """Walk the click group into the console's command table."""
    import click

    from omni_mercury_engine.cli import main

    missing = sorted(set(main.commands) - set(COMMAND_SECTIONS))
    if missing:
        raise SystemExit(f"top-level command(s) without a console section: {missing}")
    rows: list[dict[str, Any]] = [
        {"path": "", "kind": "group", "doc": _first_line(main.help), "section": None, "opts": []}
    ]

    def walk(group: Any, prefix: str) -> None:
        for name in sorted(group.commands):
            command = group.commands[name]
            if getattr(command, "hidden", False):
                continue
            path = f"{prefix} {name}".strip()
            is_group = isinstance(command, click.Group)
            row: dict[str, Any] = {
                "path": path,
                "kind": "group" if is_group else "command",
                "doc": _first_line(command.help or command.short_help),
                "section": COMMAND_SECTIONS[path.split(" ")[0]],
                "opts": [
                    _option(p)
                    for p in command.params
                    if isinstance(p, click.Option) and not p.hidden and p.name != "help"
                ],
            }
            args = [
                {
                    "name": p.name,
                    **({"req": 1} if p.required else {}),
                    **({"nargs": p.nargs} if p.nargs != 1 else {}),
                    **(
                        {"choices": [str(c) for c in p.type.choices]}
                        if isinstance(p.type, click.Choice)
                        else {}
                    ),
                }
                for p in command.params
                if isinstance(p, click.Argument)
            ]
            if args:
                row["args"] = args
            examples = _examples(command.help)
            if examples:
                row["ex"] = examples
            rows.append(row)
            if is_group:
                walk(command, path)

    walk(main, "")
    return rows


# ---------------------------------------------------------------------------
# Detectors, loaders, hazards
# ---------------------------------------------------------------------------


def _evidence(item: Any) -> dict[str, Any]:
    from dataclasses import asdict, is_dataclass

    data = asdict(item) if is_dataclass(item) else dict(item)
    short = {"domain": "d", "evidence": "e", "note": "n", "artifact": "e"}
    return {short.get(k, k): v for k, v in data.items() if v not in (None, "", ())}


def build_detectors() -> list[dict[str, Any]]:
    """Registry manifest joined with the detector ontology and the knowledge profile."""
    from omni_mercury_engine.core.detector_ontology import ONTOLOGY
    from omni_mercury_engine.core.detector_registry import DETECTOR_MANIFEST

    out: list[dict[str, Any]] = []
    for entry in DETECTOR_MANIFEST:
        row: dict[str, Any] = {
            "c": entry.class_name,
            "mod": entry.module_path,
            "fam": entry.category.value.replace("_", " ").title(),
            "name": entry.name,
            "desc": entry.description,
            "modality": entry.modality,
            "fit": bool(entry.fit_required),
            "seams": list(entry.seams),
        }
        if entry.external:
            row["ext"] = True
        if entry.wraps:
            row["wraps"] = entry.wraps
        if entry.covered_by:
            row["coveredBy"] = entry.covered_by
        ont = ONTOLOGY.get(entry.name)
        if ont is not None:
            row["ont"] = {
                "detects": list(ont.detects),
                "cannot": list(ont.cannot_detect),
                "validated": [_evidence(v) for v in ont.validated],
                "note": ont.not_validated_note or "",
            }
        profile = DETECTOR_PROFILES.get(entry.name)
        if profile is not None:
            row["prof"] = {
                "disc": profile.discipline,
                "lineage": profile.lineage,
                "mech": profile.mechanism,
                "who": profile.practitioners,
                "use": profile.use_when,
                "not": profile.not_for,
                "refs": [[r.citation, r.url] for r in profile.references],
            }
        out.append(row)
    return out


def build_loaders() -> list[dict[str, Any]]:
    """Every ``loaders/*_loader.py`` with its declared contract and its knowledge profile.

    The loader ids come from the ``from ...loaders.x_loader import XLoader``
    lines in ``loaders/__init__.py`` (what the package exports); the contract
    (source URL, key, label provenance, feature columns) is parsed from the
    module by ``core.loader_declarations`` without importing it; the profile
    (discipline, professions, institutions, expertise, method, limits,
    references) comes from ``core.knowledge_profiles``.
    """
    from omni_mercury_engine.core.loader_declarations import loader_declarations

    tree = ast.parse((SRC / "loaders" / "__init__.py").read_text(encoding="utf-8"))
    declared = loader_declarations()
    out: list[dict[str, Any]] = []
    for node in tree.body:
        if not (isinstance(node, ast.ImportFrom) and node.module):
            continue
        leaf = node.module.rsplit(".", 1)[-1]
        if not leaf.endswith("_loader"):
            continue
        for alias in node.names:
            if alias.name.endswith("Loader") and alias.name != "BaseDomainLoader":
                loader_id = leaf.removesuffix("_loader")
                row: dict[str, Any] = {
                    "id": loader_id,
                    "file": f"loaders/{leaf}.py",
                    "cls": alias.name,
                }
                decl = declared.get(loader_id)
                if decl is not None:
                    row["src"] = decl.source_url
                    row["key"] = decl.api_key_env_var if decl.requires_api_key else ""
                    row["label"] = decl.label_source
                    row["cols"] = list(decl.feature_columns)
                    row["scored"] = list(decl.scored_feature_columns)
                    row["ttl"] = decl.cache_ttl
                prof = _domain_profile_row(loader_id)
                if prof is not None:
                    row["prof"] = prof
                out.append(row)
    return sorted(out, key=lambda r: r["id"])


def _domain_profile_row(domain: str) -> dict[str, Any] | None:
    """The console's compact form of a domain's knowledge profile."""
    profile = DOMAIN_PROFILES.get(domain)
    if profile is None:
        return None
    return {
        "title": profile.title,
        "disc": profile.discipline,
        "who": list(profile.professions),
        "inst": [[i.name, i.role, i.url] for i in profile.institutions],
        "know": list(profile.expertise),
        "obs": profile.observables,
        "method": profile.method,
        "labels": profile.label_rule,
        "limits": list(profile.limits),
        "refs": [[r.citation, r.url] for r in profile.references],
        "dets": list(profile.detectors),
        "std": list(profile.standards),
    }


def build_domain_profiles() -> dict[str, dict[str, Any]]:
    """Every domain profile (loader and hazard domains alike), keyed by domain."""
    return {d: row for d in sorted(DOMAIN_PROFILES) if (row := _domain_profile_row(d)) is not None}


def _module_constant(path: Path, name: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            return ast.literal_eval(node.value)  # type: ignore[arg-type]
    raise SystemExit(f"{name} not found in {path}")


def build_tier() -> dict[str, Any]:
    """The streaming tier's paradigms and its torch-gated members (parsed, not imported)."""
    path = SRC / "detectors" / "detection_tier.py"
    paradigms = _module_constant(path, "TIER_PARADIGMS")
    return {
        "paradigms": {k: list(v) for k, v in paradigms.items()},
        "torch": list(_module_constant(path, "TORCH_TIER")),
    }


def build_hazards() -> list[str]:
    """``KNOWN_HAZARDS`` from the hazard diagnostics module."""
    return list(_module_constant(SRC / "detectors" / "hazard_diagnostics.py", "KNOWN_HAZARDS"))


# ---------------------------------------------------------------------------
# MCP tools and API routes
# ---------------------------------------------------------------------------


def _field(schema: dict[str, Any], required: bool, components: dict[str, Any]) -> dict[str, Any]:
    """Collapse one JSON-Schema property into the console's field descriptor."""

    def resolve(node: dict[str, Any]) -> dict[str, Any]:
        ref = node.get("$ref")
        if ref:
            # Siblings of a $ref (default, description) belong to the use
            # site and override the referenced definition.
            target = resolve(components.get(ref.rsplit("/", 1)[-1], {}))
            return {**target, **{k: v for k, v in node.items() if k != "$ref"}}
        return node

    node = resolve(schema)
    variants = [resolve(v) for v in node.get("anyOf", node.get("oneOf", []))]
    non_null = [v for v in variants if v.get("type") != "null"]
    if non_null:
        merged = dict(non_null[0])
        merged.update({k: v for k, v in node.items() if k not in ("anyOf", "oneOf")})
        node = merged
    kind = node.get("type")
    out: dict[str, Any] = {}
    if "enum" in node:
        out["type"] = "enum"
        out["enum"] = list(node["enum"])
    elif kind == "array":
        items = resolve(node.get("items", {}))
        inner = items.get("type")
        if inner == "array":
            leaf = resolve(items.get("items", {})).get("type", "number")
            out["type"] = f"{'number' if leaf in ('number', 'integer') else leaf}[][]"
        else:
            out["type"] = f"{inner or 'object'}[]"
        # The console's sample series is sized from this; without it a form's
        # default payload could be shorter than the route accepts.
        if "minItems" in node:
            out["min"] = int(node["minItems"])
    elif kind in ("number", "integer", "boolean", "string", "object"):
        out["type"] = kind
    else:
        out["type"] = "object"
    if "default" in node and _jsonable(node["default"]):
        out["def"] = node["default"]
    for src, dst in (("minimum", "min"), ("maximum", "max"), ("maxLength", "max")):
        if src in node and dst not in out:
            out[dst] = node[src]
    if required:
        out["req"] = 1
    desc = node.get("description") or schema.get("description")
    if desc:
        out["desc"] = _first_line(desc)
    return out


_BODY_LINE = re.compile(r"^Body:\s*(.+?)(?:\n\s*\n|\Z)", re.MULTILINE | re.DOTALL)
_BODY_KEY = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)`(\s*\(required\))?")


def documented_body_keys(description: str | None) -> tuple[list[str], list[str]]:
    """The keys a free-form route documents on its ``Body:`` line, and the required ones.

    A route that takes ``dict[str, Any]`` has no pydantic fields for the
    console to render, but its description states the contract: ``Body:
    `statements` (required), `evidence`, …`` or ``Body: `intel_reports`,
    optional `history`.``. The keys are read in order; a key is required when
    it carries ``(required)``, or, when no key does and the word ``optional``
    appears, when it comes before that word. A line with neither marks no key
    required rather than guessing.
    """
    if not description:
        return [], []
    match = _BODY_LINE.search(description)
    if not match:
        return [], []
    text = " ".join(match.group(1).split())
    # A parenthesis other than ``(required)`` explains a key (a scope, a
    # type); the names inside it are not keys.
    text = re.sub(r"\((?!required\))[^)]*\)", "", text)
    keys = [m.group(1) for m in _BODY_KEY.finditer(text)]
    marked = [m.group(1) for m in _BODY_KEY.finditer(text) if m.group(2)]
    if marked:
        required = marked
    elif "optional" in text:
        head = text.split("optional", 1)[0]
        required = [m.group(1) for m in _BODY_KEY.finditer(head)]
    else:
        required = []
    ordered: list[str] = []
    for key in keys:
        if key not in ordered:
            ordered.append(key)
    return ordered, [k for k in ordered if k in required]


def _object_fields(
    schema: dict[str, Any], components: dict[str, Any], *, raw_body: bool = True
) -> dict[str, Any]:
    ref = schema.get("$ref")
    if ref:
        schema = components.get(ref.rsplit("/", 1)[-1], {})
    if raw_body and schema.get("type") == "object" and not schema.get("properties"):
        # A free-form JSON body (``dict[str, Any]``): the console sends the
        # user's object as the whole body.
        return {"body": {"type": "object", "req": 1, "raw": 1, "desc": "Free-form JSON body"}}
    required = set(schema.get("required", []))
    return {
        name: _field(prop, name in required, components)
        for name, prop in (schema.get("properties") or {}).items()
    }


def build_mcp() -> tuple[list[dict[str, str]], dict[str, Any]]:
    """MCP tool list and their input schemas."""
    from omni_mercury_engine.mcp_server import MercuryMCPServer

    tools = MercuryMCPServer()._tools
    listing = []
    schemas: dict[str, Any] = {}
    for name in tools:
        spec = tools[name]
        sentence = _first_line(spec.description).split(". ")[0].rstrip(".")
        listing.append({"n": name, "d": sentence})
        schemas[name] = _object_fields(spec.input_schema, {}, raw_body=False)
    return listing, schemas


def build_vendors() -> dict[str, Any]:
    """The language-model providers the engine ships adapters for, and the MCP clients.

    Read from ``api/backends.py`` (which reads the code-grounded
    ``PROVIDER_CATALOG``), so the console's AI vendors view lists exactly the
    adapters the engine has; the live route adds whether each key is set and
    whether Ollama is reachable.
    """
    from omni_mercury_engine.api.backends import MCP_CLIENTS, PROVIDER_NAMES, PROVIDER_NOTES
    from omni_mercury_engine.models.llm_registry import PROVIDER_CATALOG

    providers = [
        {
            "id": provider,
            "name": PROVIDER_NAMES.get(provider, provider),
            "wire": facts.wire_format,
            "locality": facts.locality,
            "key": facts.api_key_env_var,
            "baseUrl": facts.requires_explicit_base_url,
            "usage": facts.reports_token_usage,
            "note": PROVIDER_NOTES.get(provider, ""),
        }
        for provider, facts in PROVIDER_CATALOG.items()
    ]
    clients = [
        {
            "id": c["id"],
            "name": c["name"],
            "file": c["config_file"],
            "snippet": json.dumps(c["snippet"], indent=2),
            "note": c["note"],
        }
        for c in MCP_CLIENTS
    ]
    return {"providers": providers, "clients": clients, "command": "mercury-agent mcp"}


def build_api() -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Every OpenAPI operation, and the JSON request schema of each."""
    logging.disable(logging.WARNING)
    try:
        from omni_mercury_engine.api.server import app

        spec = app.openapi()
    finally:
        logging.disable(logging.NOTSET)
    components = spec.get("components", {}).get("schemas", {})
    listing: list[dict[str, str]] = []
    schemas: dict[str, Any] = {}
    for path, ops in spec["paths"].items():
        for method, op in ops.items():
            if method not in ("get", "post", "put", "patch", "delete"):
                continue
            tag = (op.get("tags") or ["Service"])[0]
            row = {"g": API_GROUPS.get(tag, tag), "m": method.upper(), "path": path}
            if op.get("summary"):
                row["s"] = op["summary"]
            listing.append(row)
            key = f"{method.upper()} {path}"
            fields: dict[str, Any] = {}
            body = op.get("requestBody", {}).get("content", {}).get("application/json")
            if body:
                fields.update(_object_fields(body.get("schema", {}), components))
                raw = fields.get("body")
                if raw and raw.get("raw"):
                    keys, required = documented_body_keys(op.get("description"))
                    if keys:
                        raw["keys"] = keys
                        raw["reqKeys"] = required
                        raw["desc"] = "Free-form JSON body · documented keys: " + ", ".join(
                            k + (" (required)" if k in required else "") for k in keys
                        )
            for param in op.get("parameters", []):
                field = _field(param.get("schema", {}), bool(param.get("required")), components)
                field["p" if param.get("in") == "path" else "q"] = 1
                if param.get("description"):
                    field["desc"] = _first_line(param["description"])
                fields[param["name"]] = field
            if method != "get" or fields:
                schemas[key] = fields
    # The hazard route takes ``hazard: str`` and validates it against
    # KNOWN_HAZARDS at run time (400 otherwise); state the legal values.
    hazard = schemas.get("POST /api/v1/hazard/visualize", {}).get("hazard")
    if hazard is not None:
        hazard.update({"type": "enum", "enum": build_hazards()})
    return listing, schemas


# ---------------------------------------------------------------------------
# Runs (argparse, by ast), tools, vocabularies
# ---------------------------------------------------------------------------


def _const(node: ast.expr | None) -> Any:
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return ast.unparse(node)


def _argparse_flags(tree: ast.AST) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            continue
        flags = [
            a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)
        ]
        if not flags or not flags[0].startswith("-"):
            continue
        entry: dict[str, Any] = {"f": flags}
        for kw in node.keywords:
            if kw.arg == "default":
                value = _const(kw.value)
                if value is not None and value is not False:
                    entry["def"] = value if _jsonable(value) else str(value)
            elif kw.arg == "type" and isinstance(kw.value, ast.Name):
                entry["type"] = kw.value.id
            elif kw.arg == "help":
                value = _const(kw.value)
                if isinstance(value, str):
                    entry["h"] = " ".join(value.split())
            elif kw.arg == "choices":
                value = _const(kw.value)
                if isinstance(value, (list, tuple)):
                    entry["choices"] = [str(v) for v in value]
            elif kw.arg == "action" and _const(kw.value) in ("store_true", "store_false"):
                entry["flag"] = 1
            elif kw.arg == "required" and _const(kw.value) is True:
                entry["req"] = 1
        out.append(entry)
    return out


def _run_category(rel: str) -> str:
    name = rel.lower()
    for needle, category in _RUN_RULES:
        if needle in name:
            return category
    return "Ops, verification & CI"


def build_runs() -> list[dict[str, Any]]:
    """Benchmarks, scripts and examples, parsed (never executed)."""
    rows: list[dict[str, Any]] = []
    for group in ("benchmarks", "scripts", "examples"):
        for path in sorted((REPO / group).glob("*.py")):
            if path.name.startswith("_") or path.name == Path(__file__).name:
                continue
            rel = path.relative_to(REPO).as_posix()
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            row: dict[str, Any] = {
                "p": rel,
                "g": group,
                "cat": _run_category(rel),
                "s": _first_line(ast.get_docstring(tree)),
            }
            row["a"] = _argparse_flags(tree)  # always a list: the console iterates it
            rows.append(row)
    return rows


def build_tools() -> list[list[Any]]:
    """``tools._TOOL_NAMES`` grouped under its own comment headings."""
    source = (SRC / "tools" / "__init__.py").read_text(encoding="utf-8")
    groups: list[list[Any]] = []
    state = "seek"  # seek -> name seen -> inside the tuple -> done
    depth = 0
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if state == "seek":
            if tok.type == tokenize.NAME and tok.string == "_TOOL_NAMES":
                state = "name"
        elif state == "name":
            if tok.type == tokenize.OP and tok.string == "=":
                state = "inside"
        elif state == "inside":
            if tok.type == tokenize.OP and tok.string in "([":
                depth += 1
            elif tok.type == tokenize.OP and tok.string in ")]":
                depth -= 1
                if depth == 0:
                    break
            elif depth == 1 and tok.type == tokenize.COMMENT:
                heading = tok.string.lstrip("# ").strip()
                groups.append([heading.replace(" / ", " & "), []])
            elif depth == 1 and tok.type == tokenize.STRING:
                if not groups:
                    groups.append(["Tools", []])
                groups[-1][1].append(ast.literal_eval(tok.string))
    return [g for g in groups if g[1]]


def build_vocab(curated: dict[str, Any]) -> dict[str, Any]:
    """``voice --domain`` and ``voice --model`` value sets."""
    from omni_mercury_engine.cognitive.ethical_bounding import _build_safe_domain_labels
    from omni_mercury_engine.cognitive.ipb_engine import EnvironmentDomain

    labels = _build_safe_domain_labels()
    ordered = ["general"] + [m.value for m in EnvironmentDomain]
    assert set(ordered) == set(labels), "voice domain vocabulary drifted from EnvironmentDomain"
    domain_notes = curated["vocabNotes"]["voice --domain"]
    domain = {
        "closed": True,
        "fallback": "general",
        "unset": domain_notes["unset"],
        "note": domain_notes["note"],
        "groups": [
            {
                "g": "",
                "items": [
                    ["general", "the fallback for anything else"] if v == "general" else [v]
                    for v in ordered
                ],
            }
        ],
        "near": domain_notes["near"],
    }
    # OllamaModel parsed from source: importing the adapter would import torch.
    source = (SRC / "models" / "foundation" / "ollama_adapter.py").read_text(encoding="utf-8")
    block = source[source.index("class OllamaModel") :]
    block = block[: block.index("\n\n\n")]
    groups: list[dict[str, Any]] = []
    for line in block.splitlines()[1:]:
        stripped = line.strip()
        heading = re.match(r"#\s*([A-Za-z/]+)", stripped)
        member = re.match(r"[A-Z0-9_]+\s*=\s*\"([^\"]+)\"\s*(?:#\s*(.*))?$", stripped)
        if heading and not member:
            groups.append({"g": heading.group(1).split()[0].upper(), "items": []})
        elif member:
            if not groups:
                groups.append({"g": "", "items": []})
            item = [member.group(1)]
            if member.group(2):
                item.append(member.group(2).strip()[0].lower() + member.group(2).strip()[1:])
            groups[-1]["items"].append(item)
    model_notes = curated["vocabNotes"]["voice --model"]
    model = {
        "closed": False,
        "unset": model_notes["unset"],
        "note": model_notes["note"],
        "groups": groups,
    }
    return {"voice --domain": domain, "voice --model": model}


# ---------------------------------------------------------------------------
# Curated entries, verified
# ---------------------------------------------------------------------------


def _class_methods(module: str, cls: str) -> set[str] | None:
    path = SRC.joinpath(*module.split(".")[1:]).with_suffix(".py")
    if not path.is_file():
        return None
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            return {
                n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    return None


def verify_curated(curated: dict[str, Any], detector_names: set[str]) -> None:
    """Fail if a curated class, method or linked detector does not exist."""
    problems: list[str] = []
    for row in curated["library"] + curated["predictors"]:
        methods = _class_methods(row["mod"], row["c"])
        if methods is None:
            problems.append(f"{row['mod']}.{row['c']} not found")
            continue
        for name in re.findall(r"\b([a-z_][a-z0-9_]+)\(", row.get("call", "")):
            if name not in methods:
                problems.append(f"{row['c']}.{name} not found")
        if row.get("viaDet") and row["viaDet"] not in detector_names:
            problems.append(f"{row['c']} links to unknown detector {row['viaDet']!r}")
    if problems:
        raise SystemExit("curated catalog entries are stale:\n  " + "\n  ".join(problems))


# ---------------------------------------------------------------------------


def build() -> dict[str, Any]:
    """Assemble the whole catalog."""
    from omni_mercury_engine._version import __version__
    from omni_mercury_engine.core.detector_ontology import ANOMALY_CLASSES

    curated = json.loads(CURATED.read_text(encoding="utf-8"))
    commands = build_commands()
    detectors = build_detectors()
    verify_curated(curated, {d["name"] for d in detectors})
    mcp, mcp_schemas = build_mcp()
    api, api_schemas = build_api()
    tools = build_tools()
    return {
        "version": __version__,
        "sections": list(SECTIONS),
        "commands": commands,
        "nCommands": sum(1 for c in commands if c["kind"] == "command"),
        "tools": tools,
        "nTools": sum(len(g[1]) for g in tools),
        "detectors": detectors,
        "ontologyVocab": sorted(ANOMALY_CLASSES),
        "loaders": build_loaders(),
        "hazards": build_hazards(),
        "domainProfiles": build_domain_profiles(),
        "tier": build_tier(),
        "mcp": mcp,
        "mcpSchemas": mcp_schemas,
        "vendors": build_vendors(),
        "api": api,
        "schemas": api_schemas,
        "runs": build_runs(),
        "predictors": curated["predictors"],
        "library": curated["library"],
        "vocab": build_vocab(curated),
    }


def render(catalog: dict[str, Any]) -> str:
    """The exact text of ``catalog.js``."""
    body = json.dumps(catalog, ensure_ascii=False, separators=(",", ":"))
    return (
        "/* Copyright (C) 2025 Steel Security Advisors LLC\n"
        "   SPDX-License-Identifier: AGPL-3.0-or-later\n"
        "   GENERATED by scripts/build_console_catalog.py -- do not edit by hand. */\n"
        f"window.MERCURY_CLI = {body};\n"
    )


def main(argv: list[str] | None = None) -> int:
    """Write the catalog, or with ``--check`` report whether it is stale."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if catalog.js is stale")
    args = parser.parse_args(argv)
    os.environ.setdefault("MERCURY_LOG_LEVEL", "WARNING")
    text = render(build())
    if args.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        if current != text:
            print(f"STALE: {OUTPUT.relative_to(REPO)}; run python scripts/build_console_catalog.py")
            return 1
        print("console catalog is current")
        return 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(REPO)} ({len(text):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
