# Deferred Work — v1.6.0 Corrective Sweep (§13)

**Date:** 2026-05-09
**Branch:** `devin/1778041418-v1.6.0-corrective-sweep` (PR #189)

## mypy --strict baseline

```
$ mypy --strict src/omni_mercury_engine
Success: no issues found in 495 source files
```

**Baseline error count: 0**

The `mypy --strict` audit, which the original §13 listing assumed would
identify a substantial cleanup backlog, returns **clean** on PR #189 HEAD
post-absorption.  The three errors found in the first run (two duplicate
`self._rng` attribute definitions in `agentic_autonomy.py` and
`double_helix_engine.py`, and one `list()` narrowing issue in
`detectors/math_arrest/arrest.py`) were **fixed in this PR** rather than
deferred:

| File                                                       | Error code  | Fix                                                                                                      |
|------------------------------------------------------------|-------------|----------------------------------------------------------------------------------------------------------|
| `src/omni_mercury_engine/agentic/agentic_autonomy.py:145`  | `no-redef`  | Removed the duplicate `self._rng = np.random.default_rng(seed)` (already initialized at line 138).       |
| `src/omni_mercury_engine/core/double_helix_engine.py:177`  | `no-redef`  | Same: removed duplicate.                                                                                 |
| `src/omni_mercury_engine/detectors/math_arrest/arrest.py:213` | `arg-type` | Replaced `list(spec)` with a type-narrowing list-comprehension that filters on `isinstance(BaseEquationProbe)`.|

All three were fall-out from cherry-pick conflict resolution where the
`-X ours` strategy retained PR #189's existing `_rng` initialization but
also kept the cherry-picked addition.  The fixes restore one definition
per attribute.

## Per-file error count

`0` errors across 495 files.  No follow-up baseline table required.

## Cure approach (still binding)

Doctrine §1: "Suppression = Gap. Restructure rather than suppress."  This
section documents the cure pattern that the v1.6.0 sweep applied to the
type-redef class of suppressions; future PRs should keep using it.

### TYPE_CHECKING-guided single definition

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch                       # type-only — never imported at runtime
else:
    try:
        import torch
        TORCH_AVAILABLE = True
    except ImportError:
        TORCH_AVAILABLE = False

# Use ``torch`` in annotations everywhere; the runtime branch above is the
# single canonical definition — no second ``torch = None  # type: ignore[assignment, no-redef]``
# is needed because the annotation in the if-block is the only one mypy sees.
```

### Same-kind stub class fallback

```python
if TORCH_AVAILABLE:
    class MyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(10, 1)
else:
    # Stub: must be a ``class``, not a ``def``, so mypy sees a single
    # ``MyModel`` *type* across both branches.  Fields and methods are
    # named so call-site type checking is unchanged.
    class MyModel:  # type: ignore[no-redef] is NOT required here
        def __init__(self) -> None:
            raise ImportError("torch required for MyModel")
```

### Real type stubs over inline `# type: ignore`

When upstream packages ship `.pyi` stubs, install them
(`types-requests`, `types-PyYAML`, etc.).  When they don't, use a local
`stubs/` package (already wired in via `[tool.mypy] mypy_path =
"$MYPY_CONFIG_FILE_DIR/stubs"` if needed) instead of inline ignores.

### Narrow `[[tool.mypy.overrides]]` only with documented justification

When a third-party module is genuinely untyped (a transitive dep with no
maintainer-supplied stubs), prefer:

```toml
[[tool.mypy.overrides]]
module = "pycmap.*"
ignore_missing_imports = true
# Why: pycmap is a CMAP oceanographic data client with no PEP 561 stubs
# and no community types-pycmap package as of 2026-05.
# Re-review: when pycmap >= X.Y ships ``py.typed``.
```

Each override **MUST** include the "Why" and a "Re-review" date.

## What this is NOT

This file is a **record**, not a relaxation.  The hard rules above
(`# noqa = 0`, `# type: ignore[no-redef] = 0`, `np.random.<global> = 0`,
`continue-on-error: true = 0`, `|| true / || echo = 0`) still bind every
PR landing on `main`.

## Relationship to the §13 prose

The original §13 prose anticipated a non-trivial mypy backlog.  Empirically
the backlog does not exist on PR #189 HEAD post-absorption — every
cherry-picked SHA (PR #188 lineage + PR #190 type-redef cure +
PR #191 SSRF/loader fixes) passes `mypy --strict`, and the three residual
errors found during this audit were addressed in-tree rather than
deferred.

Future PRs that intentionally introduce `mypy --strict` regressions
(e.g. integrating an upstream library with no PEP 561 stubs) must either
ship the override + Re-review date pattern above or document the
deferral here under "Active backlog" with a tracking issue link.  As of
this PR there is no active backlog.

## Active backlog

(empty)
