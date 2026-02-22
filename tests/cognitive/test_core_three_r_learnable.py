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

from omni_mercury_engine.core.three_r.learnable_fusion import (
    Learnable3RConfig,
    Learnable3REngine,
    Learnable3RResult,
)


def test_learnable_3r_config_instantiation() -> None:
    config = Learnable3RConfig()
    assert config is not None


def test_learnable_3r_engine_instantiation() -> None:
    engine = Learnable3REngine()
    assert engine is not None


def test_learnable_3r_result_importable() -> None:
    assert Learnable3RResult is not None


def test_importable_from_three_r() -> None:
    from omni_mercury_engine.core.three_r import (
        Learnable3RConfig as LC,
        Learnable3REngine as LE,
        Learnable3RResult as LR,
    )

    assert LC is Learnable3RConfig
    assert LE is Learnable3REngine
    assert LR is Learnable3RResult
