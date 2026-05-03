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

------------------------------------------------------------------------

One-shot operator tools that run outside the engine.

Modules in this package are intentionally not imported by
``omni_mercury_engine.__init__`` and never exercised by the runtime
detection / training / inference paths. They exist only so operators
can perform offline maintenance tasks (such as migrating legacy
``.pkl`` training payloads) without the engine ever loading the
dangerous code paths involved.
"""

from __future__ import annotations
