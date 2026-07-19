# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the capability-inventory generator's base-class refinement.

The generator classifies every class by name suffix first, then refines the
``Other`` bucket by base-class analysis (an ``ast``-only ancestor walk). These
tests pin: the curated base signals (nn.Module -> neural, Protocol/TypedDict ->
support), the deliberate *absence* of name-suffix guessing on base names (so a
``LoggerMixin`` base or a shadowed ``BaseModel`` cannot misclassify a subclass),
that object-only classes stay unresolved, and that the committed Markdown
inventory is fresh (the previously-unguarded drift gate).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import generate_capability_inventory as gci


def test_name_suffix_wins_first() -> None:
    """A class whose own name carries a suffix is categorized by name."""
    cat, is_cap, resolution = gci.categorize("WildfireDetector", [], {})
    assert cat == "Detection"
    assert is_cap is True
    assert resolution == "name"


def test_nn_module_base_becomes_neural() -> None:
    """An nn.Module subclass with a plain name is refined to a neural component."""
    cat, is_cap, resolution = gci.categorize("Discriminator", ["Module"], {})
    assert cat == "Neural models & layers"
    assert is_cap is True
    assert resolution == "base"


def test_protocol_and_typeddict_bases_become_support() -> None:
    """Protocol / TypedDict / NamedTuple subclasses are support types."""
    for base in ("Protocol", "TypedDict", "NamedTuple"):
        cat, is_cap, resolution = gci.categorize("SomeThing", [base], {})
        assert cat.startswith("Support types"), base
        assert is_cap is False, base
        assert resolution == "base", base


def test_mixin_base_does_not_demote_subclass() -> None:
    """A cross-cutting mixin base must NOT drag a capability into Support."""
    # LoggerMixin ends in a support suffix, but it is a behavioural mixin.
    cat, is_cap, resolution = gci.categorize("CyberFortress", ["LoggerMixin"], {})
    assert cat == "Other capability classes"
    assert is_cap is True
    assert resolution == "unresolved"


def test_shadowed_basemodel_is_not_read_as_neural() -> None:
    """An in-tree BaseModel (fronting pydantic) must not be read as a 'Model'."""
    class_bases = {"BaseModel": ["object"]}
    cat, _is_cap, resolution = gci.categorize("ComponentHealth", ["BaseModel"], class_bases)
    assert cat == "Other capability classes"
    assert resolution == "unresolved"


def test_curated_signal_propagates_through_in_tree_ancestor() -> None:
    """A curated base signal reaches a subclass through an intermediate project base."""
    # NeuralBase(nn.Module); Foo(NeuralBase) -> Foo is neural via the ancestor chain.
    class_bases = {"NeuralBase": ["Module"], "Foo": ["NeuralBase"]}
    cat, is_cap, resolution = gci.categorize("Foo", ["NeuralBase"], class_bases)
    assert cat == "Neural models & layers"
    assert is_cap is True
    assert resolution == "base"


def test_object_only_class_stays_unresolved() -> None:
    """A class with no informative base and no name suffix is genuinely Other."""
    cat, is_cap, resolution = gci.categorize("Widget", [], {})
    assert cat == "Other capability classes"
    assert is_cap is True
    assert resolution == "unresolved"


def test_collect_reports_base_refinement() -> None:
    """A live collect() refines a non-trivial number of classes and is honest about limits."""
    inv = gci.collect()
    assert inv["total_classes"] > 2000
    assert inv["refined_by_base"] >= 40  # the nn.Module + Protocol/TypedDict wins
    assert inv["unresolved_other"] > 0  # object-only classes remain, reported honestly
    assert inv["categories"]["Other capability classes"] == inv["unresolved_other"]


def test_committed_inventory_is_fresh() -> None:
    """The committed CAPABILITY_INVENTORY.md matches a fresh generation (drift gate)."""
    inv = gci.collect()
    rendered = gci.render_markdown(inv)
    committed = gci._MD_OUT.read_text(encoding="utf-8")
    assert committed == rendered, "docs/CAPABILITY_INVENTORY.md is stale — re-run the generator"
