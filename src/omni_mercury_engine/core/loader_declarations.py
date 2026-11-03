# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What each domain loader declares about itself, read from source without importing it.

The loaders package pulls in every domain's dependencies on import, which the
API, the console catalog builder and the agent must not do just to say where
a domain's data comes from. Each ``loaders/<domain>_loader.py`` declares its
contract as class attributes (``DOMAIN``, ``SOURCE_URL``, ``REQUIRES_API_KEY``,
``API_KEY_ENV_VAR``, ``LABEL_SOURCE``, ``FEATURE_COLUMNS``,
``LABEL_DERIVED_FEATURES``, ``CACHE_TTL``); this module parses those
declarations with :mod:`ast` and returns them as plain data. A value that is
not a literal (a module-level alias such as ``_BASE_URL``) is resolved through
the module's own literal assignments when possible, otherwise reported as the
expression text, so nothing is guessed.
"""

from __future__ import annotations

import ast
import importlib.util
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

__all__ = ["LoaderDeclaration", "loader_declarations", "loader_domains"]

_FIELDS = (
    "DOMAIN",
    "SOURCE_URL",
    "REQUIRES_API_KEY",
    "API_KEY_ENV_VAR",
    "API_KEY_ENV_FALLBACKS",
    "LABEL_SOURCE",
    "FEATURE_COLUMNS",
    "LABEL_DERIVED_FEATURES",
    "LABEL_FEATURE_DISJOINT",
    "CACHE_TTL",
)


@dataclass(frozen=True)
class LoaderDeclaration:
    """The declared contract of one domain loader."""

    domain: str
    module: str
    class_name: str
    file: str
    docstring: str
    source_url: str = ""
    requires_api_key: bool = False
    api_key_env_var: str = ""
    api_key_env_fallbacks: tuple[str, ...] = ()
    label_source: str = "ground_truth"
    feature_columns: tuple[str, ...] = ()
    label_derived_features: tuple[str, ...] = ()
    label_feature_disjoint: bool = False
    cache_ttl: int = 3600
    scored_feature_columns: tuple[str, ...] = field(default_factory=tuple)


def _package_dir() -> Path | None:
    spec = importlib.util.find_spec("omni_mercury_engine.loaders")
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations)))


def _module_literals(tree: ast.Module) -> dict[str, Any]:
    """Module-level ``NAME = <literal>`` assignments, for resolving aliases."""
    out: dict[str, Any] = {}
    for node in tree.body:
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets, value = list(node.targets), node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        if value is None:
            continue
        try:
            literal = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            if isinstance(value, ast.JoinedStr):
                # f"{_BASE}training/": resolve the parts we can.
                parts: list[str] = []
                ok = True
                for piece in value.values:
                    if isinstance(piece, ast.Constant):
                        parts.append(str(piece.value))
                    elif (
                        isinstance(piece, ast.FormattedValue)
                        and isinstance(piece.value, ast.Name)
                        and piece.value.id in out
                    ):
                        parts.append(str(out[piece.value.id]))
                    else:
                        ok = False
                        break
                if not ok:
                    continue
                literal = "".join(parts)
            else:
                continue
        for target in targets:
            if isinstance(target, ast.Name):
                out[target.id] = literal
    return out


def _resolve(value: ast.expr, literals: dict[str, Any]) -> Any:
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        pass
    if isinstance(value, ast.Name) and value.id in literals:
        return literals[value.id]
    if (
        isinstance(value, ast.Attribute)
        and isinstance(value.value, ast.Name)
        and value.value.id == "TrustedEndpoints"
    ):
        # The egress allowlist's named endpoints (security/input_validation.py) are
        # plain constants; the module is dependency-free, so resolving one is cheap.
        from omni_mercury_engine.security.input_validation import TrustedEndpoints

        resolved = getattr(TrustedEndpoints, value.attr, None)
        if isinstance(resolved, str):
            return resolved
    return ast.unparse(value)


@lru_cache(maxsize=1)
def loader_declarations() -> dict[str, LoaderDeclaration]:
    """Every ``*_loader`` module's declared contract, keyed by domain."""
    package = _package_dir()
    if package is None:
        return {}
    out: dict[str, LoaderDeclaration] = {}
    for path in sorted(package.glob("*_loader.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        literals = _module_literals(tree)
        for node in tree.body:
            if not (isinstance(node, ast.ClassDef) and node.name.endswith("Loader")):
                continue
            declared: dict[str, Any] = {}
            for item in node.body:
                targets: list[ast.expr] = []
                value: ast.expr | None = None
                if isinstance(item, ast.Assign):
                    targets, value = list(item.targets), item.value
                elif isinstance(item, ast.AnnAssign) and item.value is not None:
                    targets, value = [item.target], item.value
                if value is None:
                    continue
                for target in targets:
                    if isinstance(target, ast.Name) and target.id in _FIELDS:
                        declared[target.id] = _resolve(value, literals)
            domain = str(declared.get("DOMAIN") or path.stem.removesuffix("_loader"))
            columns = tuple(str(c) for c in (declared.get("FEATURE_COLUMNS") or ()))
            derived = tuple(str(c) for c in (declared.get("LABEL_DERIVED_FEATURES") or ()))
            out[domain] = LoaderDeclaration(
                domain=domain,
                module=f"omni_mercury_engine.loaders.{path.stem}",
                class_name=node.name,
                file=f"loaders/{path.name}",
                docstring=(ast.get_docstring(node) or ast.get_docstring(tree) or "").strip(),
                source_url=str(declared.get("SOURCE_URL") or ""),
                requires_api_key=bool(declared.get("REQUIRES_API_KEY", False)),
                api_key_env_var=str(declared.get("API_KEY_ENV_VAR") or ""),
                api_key_env_fallbacks=tuple(
                    str(f) for f in (declared.get("API_KEY_ENV_FALLBACKS") or ())
                ),
                label_source=str(declared.get("LABEL_SOURCE") or "ground_truth"),
                feature_columns=columns,
                label_derived_features=derived,
                label_feature_disjoint=bool(declared.get("LABEL_FEATURE_DISJOINT", False)),
                cache_ttl=(
                    int(declared.get("CACHE_TTL", 3600))
                    if isinstance(declared.get("CACHE_TTL", 3600), int)
                    else 3600
                ),
                scored_feature_columns=tuple(c for c in columns if c not in derived),
            )
            break
    return out


def loader_domains() -> list[str]:
    """The declared loader domains, sorted."""
    return sorted(loader_declarations())
