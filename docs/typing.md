# Test Typing Conventions

The CI critical-tests gate enforces mypy on a growing subset of `tests/`. Use
the patterns below in any test file under a gated directory (see
`.github/workflows/ci.yml` → "Run MyPy on critical tests").

## 1. Test function annotation

Every `def test_*` function returns nothing, so annotate it as such:

```python
def test_threshold_is_positive() -> None:
    assert detector.threshold > 0
```

## 2. Generator fixture annotation

Fixtures that `yield` are generators. Use `Generator[T, None, None]`:

```python
from collections.abc import Generator

@pytest.fixture
def client() -> Generator[Client, None, None]:
    c = Client()
    yield c
    c.close()
```

## 3. Value fixture annotation

Fixtures that `return` (no yield) annotate the returned type directly:

```python
@pytest.fixture
def cfg() -> Config:
    return Config(threshold=0.5)
```

## 4. Optional narrowing — assert, not cast

When a field is `Optional[T]` and you know it is populated, narrow with
`assert`, not `cast`. `cast` lies to the type checker; `assert` is checked
at runtime so a regression in the production code fails the test instead
of producing a spurious `AttributeError`.

```python
detector.fit(data)
assert detector.mean is not None
result = detector.mean.sum()
```

## 5. Test doubles via Protocol

For duck-typed test doubles, declare a `Protocol` rather than subclassing
a heavyweight base. Protocols are structural — any object that satisfies
the shape is accepted, with no inheritance constraint.

```python
from typing import Protocol

class FakeStore(Protocol):
    def get(self, k: str) -> int: ...
```
