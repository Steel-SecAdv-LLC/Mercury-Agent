# mypy test-lane strict-graduation plan

> Directory-by-directory migration checklist for emptying the **relaxed**
> `tests/` mypy lane into the **strict** graduated lane.
>
> Owner: engineering. Status source of truth: the two mypy steps in
> `.github/workflows/ci.yml` (job `type-checking`). Last measured:
> engineering-pass round (steel/maint1/2-coverage continuation), local
> `mypy==2.3.0`, numpy 2.4.4, native AMA v3.3.0.

## Why this exists

`tests/` is checked by mypy in two steps:

1. **Relaxed lane** — `mypy tests/` with `--allow-untyped-defs`,
   `--allow-untyped-calls`, `--allow-subclassing-any`, and
   `--disable-error-code untyped-decorator`. This is the historical
   test-debt relaxation: a test function with no annotations at all is
   accepted.
2. **Strict graduated lane** — the same strict `pyproject` config the
   `src/` tree is held to, **plus** `--disallow-subclassing-any`, with
   **none** of the relaxations above. Directories opt in one at a time.

The long-term goal (strengthening plan §5 P0, "Re-lift the test-debt
mypy disables") is to graduate every directory so the relaxed lane can
be deleted. This file tracks that migration so it is data-driven, not
guesswork: a directory graduates the moment it measures **0 strict
errors**, and the ordered plan below shows exactly how much debt remains
everywhere else.

## Measurement command

Per-directory strict error count (what drives this table):

```bash
mypy tests/<dir>/ --disallow-subclassing-any --no-warn-unused-configs 2>&1 | grep -c "error:"
```

Full graduated-lane gate (must stay green; mirrors ci.yml):

```bash
mypy <graduated dirs...> --disallow-subclassing-any \
  --show-error-codes --pretty --no-warn-unused-configs
```

## Graduated (strict — no test-debt disables)

These directories are on the strict lane in `ci.yml`. Adding a file that
is not fully annotated to any of them fails the gate.

| Directory | Graduated in |
|---|---|
| `tests/datasets/` | v1.7.0 |
| `tests/ethical/` | v1.7.0 |
| `tests/safeguards/` | v1.7.0 |
| `tests/tools/` | operator-tooling hardening (#239) |
| `tests/loaders/` | ISO hardening (#238) |
| `tests/narrative/` | ISO hardening (#238) |
| `tests/fairness/` | ROADMAP row 6 closure |
| `tests/scripts/` | **engineering-pass round** |
| `tests/cyber/` | **engineering-pass round** |
| `tests/decision/` | **engineering-pass round** |
| `tests/distributed/` | **engineering-pass round** |
| `tests/emergent/` | **engineering-pass round** |
| `tests/evaluation/` | **engineering-pass round** |
| `tests/federated/` | **engineering-pass round** |
| `tests/medical/` | **engineering-pass round** |
| `tests/metrics/` | **engineering-pass round** |
| `tests/proofs/` | **engineering-pass round** |
| `tests/reasoning/` | **engineering-pass round** |
| `tests/research/` | **engineering-pass round** |
| `tests/truth_decipher/` | **engineering-pass round** |
| `tests/utils/` | **engineering-pass round** |

The engineering-pass round graduated the 14 directories that were
**already fully annotated** — measured 0 strict errors each, 131 files
clean in the combined invocation — so the graduation was a pure gate
tightening with no source churn. The combined graduated invocation
(22 directories) reports `Success: no issues found`.

## Remaining relaxed-lane debt (ordered plan)

Ordered easiest → hardest by measured strict-error count. Each row is a
self-contained graduation PR: fix the annotations, run the measurement
command until it reports 0, add the directory to the `ci.yml` strict
step, and strike the row here in the same commit.

| Order | Directory | Strict errors | Files | Notes on the debt |
|---|---|---|---|---|
| 1 | `tests/automl/` | 2 | 4 | Smallest non-zero; 2 annotations. |
| 2 | `tests/resilience/` | 3 | 2 | Two files. |
| 3 | `tests/integration/` | 4 | 6 | |
| 4 | `tests/models/` | 4 | 12 | |
| 5 | `tests/docs/` | 7 | 2 | Concentrated in 2 files. |
| 6 | `tests/space/` | 7 | 11 | |
| 7 | `tests/explainability/` | 9 | 4 | |
| 8 | `tests/security/` | 10 | 34 | Low error density; KAT vectors. |
| 9 | `tests/cognitive/` | 12 | 23 | |
| 10 | `tests/intel/` | 12 | 13 | |
| 11 | `tests/validation/` | 12 | 6 | |
| 12 | `tests/api/` | 13 | 7 | FastAPI fixtures. |
| 13 | `tests/infrastructure/` | 13 | 5 | |
| 14 | `tests/load/` | 20 | 1 | Single locust/k6 driver file. |
| 15 | `tests/ml/` | 22 | 24 | Hazard-training suites; large. |
| 16 | `tests/benchmarks/` | 24 | 18 | |
| 17 | `tests/integrations/` | 38 | 8 | Third-party boundary stubs. |
| 18 | `tests/core/` | 48 | 37 | Largest by count. |
| 19 | `tests/detectors/` | 51 | 76 | Largest surface; graduate last. |

Total remaining: **~331 strict errors across 19 directories**. The debt
is dominated by missing return/parameter annotations and
`disallow-subclassing-any` on `torch.nn.Module` / `pydantic.BaseModel`
mocks (add an explicit typed subclass or a `# type: ignore[misc]` at the
source, never a lane-wide disable).

## Definition of done

The relaxed lane (`mypy tests/ --allow-untyped-defs ...` in `ci.yml`) is
deleted once this table is empty and every directory is listed under
**Graduated**. At that point `mypy tests/` under the strict `pyproject`
config + `--disallow-subclassing-any` is the single test-typing gate.
