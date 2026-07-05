# `corpus/pending` — red-team triage queue

Surviving gate bypasses discovered by the adversarial co-training harness
(`benchmarks/red_team_harness.py`, module `omni_mercury_engine.intel.red_team`)
are appended here as `red_team_survivors.jsonl` — one JSON object per line, in
the weapons-gate corpus schema plus a `red_team` metadata block:

```json
{"text": "<mutated attack>", "label": "offensive", "expected": "block",
 "split": "pending", "tags": ["red_team", "pending_triage", "mut:char_spacing"],
 "red_team": {"id": "<sha>", "seed": "<original>", "chain": ["char_spacing"],
              "seed_disposition": "refuse_redact", "bypassed_to": "allow",
              "harness_version": "1.0"}}
```

These rows are **candidates awaiting human triage**, not training data. A
reviewer confirms each is a genuine false negative, labels it via the feedback
loop (`override_to_example(..., source=ExampleSource.RED_TEAM, reviewer=...)`),
and enqueues it for a gated retrain. See `docs/RED_TEAM_OPERATION_GUIDE.md`.

The `*.jsonl` files here are **generated** (regenerated deterministically by the
`ci/red-team` lane and uploaded as an artifact) and are git-ignored; only this
README is tracked so the directory exists.
