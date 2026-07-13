# Mercury Agent — Migration guide: v1.6.x → v1.7.0

Applies to Mercury Agent **v2.1.x**. Last updated: 2026-07-11.

This guide enumerates every breaking or behavioural change between
Mercury Agent v1.6.x and v1.7.0, with a verified replacement for each
removed surface.  Each entry links to the test that locks the new
behaviour, so you can run it locally to confirm a successful migration.

> **Scope.** This document covers user-visible API and runtime-mode
> changes only.  Internal refactors (test-debt mypy disables,
> documentation refreshes, benchmark metadata, RNG-reproducibility
> sweeps) are described in `CHANGELOG.md` and require no caller
> changes.

---

## 1. `SafeHTTPClient`: `allow_untrusted=True` removed

**What changed.** PR #210 deleted the `allow_untrusted=True` keyword
argument from every `SafeHTTPClient` method (`validate_url`,
`_request`, `get`, `get_bytes`, `get_json`, `get_text`, `post_json`)
and from the dataset/loader helpers that wrapped it
(`datasets.base.http_get_with_retry`,
`loaders.base.BaseDomainLoader._fetch_url`).  The kwarg was a
per-call bypass of the `TrustedEndpoints.TRUSTED_DOMAINS` allowlist
that had no production caller and could be misused to pivot through
an off-allowlist host.

**Migration.** Call `SafeHTTPClient` directly with
`user_configured=True` so the private-network / IMDS gate fires
explicitly:

| Before (v1.6.x)                                                          | After (v1.7.0)                                                                                            |
| ------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------- |
| `SafeHTTPClient().get(url, allow_untrusted=True)`                        | `SafeHTTPClient().get(url, user_configured=True)`                                                         |
| `SafeHTTPClient().get(rfc1918_url, allow_untrusted=True)` *(private VPC)* | `SafeHTTPClient().get(rfc1918_url, user_configured=True, allow_private=True)`                             |
| `http_get_with_retry(url, allow_untrusted=True)`                         | Configure the domain into `TrustedEndpoints.TRUSTED_DOMAINS` or migrate to direct `SafeHTTPClient` usage. |

The IMDS / loopback / multicast / reserved / CGNAT ranges remain in
the always-blocked set even with `allow_private=True`.  See
`tests/security/test_safe_http.py::TestMigrationFromAllowUntrusted`
for the documented replacement.

**DNS-fails-closed gotcha.** When `user_configured=True`,
`SafeHTTPClient.validate_url` fails closed on DNS resolution failure
(raises `UnsafeURLError`).  This is intentional SSRF / DNS-rebinding
defence.  If your operator hits this against a domain that is
reachable from the host but not from Mercury's resolver, either fix
the resolver or whitelist the IP literal directly (still subject to
the `allow_private` gate).

---

## 2. σ_Immutable is now a hard gate at every decision boundary

**What changed.** σ_Immutable (the trained 256-D scalar network at
`src/omni_mercury_engine/security/sigma_immutable_weights.pt`) was
previously *informational* at the engine boundary — its score landed
in `result["gosnn_metadata"]` but never raised.  In v1.7.0 it is the
second hard ethical gate at every boundary surface:

- `OmniMercuryEngine.detect_with_fusion` /
  `detect_with_fusion_calibrated` raises
  `EthicalConstraintViolationError(check="sigma_immutable")` on
  sub-threshold scalar vectors and
  `EthicalConstraintViolationError(check="gosnn_unavailable")` when
  GOSNN itself cannot be evaluated (missing torch, corpus signature
  failure, etc.).
- `CognitiveOrchestrator.analyze` raises the same two `check=` values.
- `NeuroSymbolicHub.predict` raises the same per-sample.

**Migration.**

1. Callers that already catch `EthicalConstraintViolationError` (or
   the `EthicalViolation` alias from
   `omni_mercury_engine.ethical`) need no code change — the exception
   type and base class are unchanged, only the `check=` value set is
   widened.  Inspect `exc.check` if you need to differentiate
   `"benevolence"` from `"sigma_immutable"` /
   `"gosnn_unavailable"`.
2. Tests that previously asserted `result["gosnn_metadata"]
   ["fallback_mode"] is True` must be deleted — the fallback path is
   gone.  A `check="gosnn_unavailable"` raise replaces it.
3. **Production callers MUST NOT toggle** the private `_enable_gosnn`
   parameter on the engine's `detect_with_fusion` variants.  Unit
   tests that genuinely need to run without GOSNN must additionally
   set the auditable module-level flag
   `omni_mercury_engine.engine._GOSNN_TESTING_BYPASS = True`.  Using
   that flag in production is a contract violation; the regression
   suite at `tests/ethical/test_hard_enforcement.py` locks the
   behaviour.

---

## 3. `MERCURY_ENV` production-mode flag

**What's new.** v1.7.0 introduces the canonical environment-mode flag
`MERCURY_ENV` (`development` default, `production`) with a shared
fail-closed helper API at `omni_mercury_engine._env`.  The first
caller to consume it is `narrative.voice.MercuryVoice` (see §4); more
modules will adopt it incrementally.

**Migration.**

- Existing deployments need no change.  An unset, empty, or
  whitespace-only `MERCURY_ENV` keeps Mercury in development mode,
  which is the pre-1.7.0 default behaviour.
- Production deployments that want stub/mock collaborators to
  hard-fail rather than silently degrade should set:

  ```bash
  export MERCURY_ENV=production
  ```

  An unknown value (e.g. `MERCURY_ENV=prod`) raises
  `MercuryProductionConfigError` at the first call to
  `omni_mercury_engine._env.get_mercury_env`, by design — typos in
  deployment configuration must be loud.

- `MERCURY_ENV` is orthogonal to `AMA_REQUIRE_REAL_PQC`.  The PQC import gate
  (`omni_mercury_engine._pqc_gate._enforce_pqc_production_gate`)
  keeps its own contract because PQC has a hard-required-build
  dependency that is independent of the development/production
  distinction.  `AMA_REQUIRE_REAL_PQC` is retained for diagnostics and
  legacy workflow readability; AMA/PQC is mandatory regardless of its value.

Locking test: `tests/test_env.py`.

---

## 4. `MercuryVoice`: explicit `llm_provider=` required for LLM mode

**What changed.** Before v1.7.0, `MercuryVoice(enable_llm=True)`
silently instantiated `MockLLMAdapter` (the heuristic-only stub).
Phase 2 of the Mercury audit (May 2026) made `MockLLMAdapter`
hard-fail at construction with `NotImplementedError`, which
*incidentally* broke `MercuryVoice(enable_llm=True)` because the
surrounding `except ImportError` did not catch it.  v1.7.0 wires the
real-provider selection that was always intended:

- `MercuryVoice(enable_llm=False)` is unchanged: pure-template
  narration, no LLM stack touched.
- `MercuryVoice(enable_llm=True, llm_provider="huggingface",
  llm_model_name="facebook/bart-large-mnli",
  llm_revision="<40-character commit SHA>")` initialises the real
  HuggingFace adapter.  HuggingFace requires an explicit
  `llm_model_name`; remote HuggingFace IDs also require the revision
  pin enforced by `SafeHFLoader`; absolute local model paths do not.
  Other implemented providers: `ollama`, `openai`, `anthropic`,
  `xai`, `gemini`, `cohere`, `deepseek`, `cursor`, `template`.
- `MercuryVoice(enable_llm=True)` with no provider:
  - `MERCURY_ENV=production` → raises `MercuryProductionConfigError`.
  - `MERCURY_ENV=development` → logs a `WARNING` and sets
    `self._llm_adapter = None`; the rest of the voice path falls
    through to deterministic template generation.
- `llm_provider="mock"` always raises
  `MercuryProductionConfigError` (in every mode), because
  `MockLLMAdapter` hard-fails at construction by design.

**Migration.**

| Before (v1.6.x)                              | After (v1.7.0)                                                                                       |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `MercuryVoice(enable_llm=True)`              | `MercuryVoice(enable_llm=True, llm_provider="huggingface", llm_model_name="facebook/bart-large-mnli", llm_revision="<40-char SHA>")` |
| `create_mercury_voice(enable_llm=True)`      | `create_mercury_voice(enable_llm=True, llm_provider=..., llm_model_name=..., llm_revision=...)`         |
| `MercuryVoice(enable_llm=True)` *(dev only)* | unchanged at the call site — now warns and disables instead of crashing                              |

Locking test: `tests/narrative/test_voice_llm.py`.

---

## 5. Verification checklist

After migrating, run the focused regression suites that lock each
contract:

```bash
# §1 SafeHTTPClient migration
pytest tests/security/test_safe_http.py::TestMigrationFromAllowUntrusted

# §2 σ_Immutable hard gate (every boundary)
pytest tests/ethical/test_hard_enforcement.py

# §3 MERCURY_ENV primitive
pytest tests/test_env.py

# §4 MercuryVoice LLM init
pytest tests/narrative/test_voice_llm.py
```

If every suite is green, the migration is complete.  Open an issue
with the failing test name and `MERCURY_ENV` value if any of them
report.
