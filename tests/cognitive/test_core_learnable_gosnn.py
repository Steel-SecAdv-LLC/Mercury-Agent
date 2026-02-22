"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

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
"""

from __future__ import annotations

from omni_mercury_engine.core.learnable_gosnn import (
    LearnableGOSNN,
    ScalarCategory,
    ScalarState,
)


def test_learnable_gosnn_instantiation() -> None:
    g = LearnableGOSNN()
    assert g is not None


def test_scalar_category_importable() -> None:
    assert ScalarCategory is not None


def test_scalar_state_importable() -> None:
    assert ScalarState is not None


def test_importable_from_core() -> None:
    from omni_mercury_engine.core import (
        LearnableGOSNN as LG,
        ScalarCategory as SC,
        ScalarState as SS,
    )

    assert LG is LearnableGOSNN
    assert SC is ScalarCategory
    assert SS is ScalarState
