# PR #315 — Security Report (egress, decompression, cache correctness)

*2026-07-02. Scope: the outbound-HTTP egress gate, the open-web research
transport, decompression-bomb handling, and the feature-cache correctness
surface touched or re-audited in this PR.*

## 1. SafeHTTPClient egress gate — re-audit

`src/omni_mercury_engine/security/safe_http.py` is the single outbound-HTTP
choke point. Re-audited after the web-research fix (finding A3) that made the
open-web fetch pass `user_configured=True`.

**Controls confirmed present and tested:**

| Control | Where | Test |
|---|---|---|
| Scheme allowlist (https-only; http only with `allow_http`) | `_parse_and_check_scheme` | `TestSchemeGate` |
| Trusted-domain allowlist for class-constant dataset URLs | `validate_url` | `TestTrustedDomainsGate` |
| Private/RFC1918/link-local/**IMDS 169.254.169.254**/CGNAT block for user-configured URLs | `_is_private_or_imds` | `TestPrivateNetworkGate` |
| `allow_private` opens RFC1918 but **never** IMDS/loopback/multicast/reserved | `_is_always_blocked` | `TestAllowPrivateGate` |
| Loopback-only enforcement for on-box adapters | `_is_loopback` | (safe_http suite) |
| DNS-rebinding / TOCTOU close: resolve once, re-apply policy, pin TCP to the validated IP (SNI/cert still by hostname) | `_request` + `_PinnedDNSHTTPAdapter` | (safe_http suite) |
| 3xx redirects refused (no silent redirect-following) | `_request` | (safe_http suite) |

**Open-web research transport** (`web_research._safe_http_transport`) passes
`user_configured=True` + `allow_http=True` so arbitrary research hosts are not
rejected by the *dataset* allowlist, **while the private-network / IMDS IP gate
still runs**. Egress behaviour is proven both directions:

- allowed: an arbitrary public host validates
  (`test_real_open_web_host_validates_but_imds_is_refused`);
- blocked: `http://169.254.169.254/…` (IMDS) and `http://127.0.0.1/` (loopback)
  are refused even with `user_configured=True`
  (`test_real_open_web_host_validates_but_imds_is_refused`,
  `TestPrivateNetworkGate`);
- the flag is actually forwarded (`test_default_transport_passes_user_configured`).

**Threat-model note:** the A3 fix widened the *allowlist* bypass but did **not**
widen the IP gate — the SSRF-relevant control (IMDS/private refusal) runs on
exactly the path the fix enabled. No new egress capability was granted to
untrusted input.

## 2. Decompression-bomb (gzip) mitigation

**Threat:** a small compressed response (`Content-Encoding: gzip`) that inflates
to gigabytes, exhausting memory (a classic gzip bomb). The prior transport read
`resp.raw.read(_DEFAULT_MAX_BYTES, decode_content=True)` and relied on urllib3
bounding the *decoded* read to the requested size — behaviour that varies by
urllib3 version (older urllib3 measured the amount against the *compressed*
stream, which a bomb defeats).

**Mitigation (primary):** `web_research._safe_http_transport` now streams the
already-decompressed body in bounded 64 KiB chunks and stops at
`_DEFAULT_MAX_BYTES` (2 MB) of **decoded** output. The decompressor yields
chunk-sized decoded blocks incrementally, so a 600 MB logical bomb never fully
materialises — peak is bounded regardless of urllib3's `read()` semantics.
Test: `TestDefaultTransportDecodedSizeCap::test_decoded_body_truncated_at_cap`
(a generator that would yield ~600 MB is consumed only up to the cap).

**Mitigation (defence in depth):** `urllib3>=2.5.0` floor pinned in
`pyproject.toml` so a transitive resolver cannot select an older urllib3 whose
streaming/decode behaviour reintroduces the unbounded-inflation DoS. The
explicit cap is the primary control; the pin guards against dependency drift.

## 3. FeatureCache correctness (stale-hit surface)

**Threat:** `engine.FeatureCache._make_key` keyed torch tensors purely on
identity (`data_ptr + storage_offset + stride + shape + dtype + device`). Two
distinct tensors can share a `data_ptr` (allocator reuse of a freed address) and
an in-place mutation keeps the same storage — both produce an identical key, so
`get_or_compute` could return **stale cached features for genuinely different
data** (a silent correctness bug, not merely a missed optimization). Measured:
two distinct `torch.zeros(10)` collide.

**Mitigation:** CPU tensors are now **content-keyed** (they share memory with
numpy, so `.numpy()` is a view — no host↔device copy) via the same bounded
strided-sample + finite-aware checksum the numpy path uses. This closes both the
in-place-mutation and address-reuse stale-hit surfaces for the common case. CUDA
tensors retain identity keying (content-hashing a device tensor would force a
per-lookup host↔device sync on the hot path) — a documented tradeoff bounded by
the LRU aging stale pointers out. Tests:
`TestFeatureCacheKey::test_torch_cpu_noncontiguous_distinct_tensors_do_not_collide`,
`…_inplace_mutation_changes_key`, `…_equal_content_same_key`.

## 4. Required operational mitigations (deployment)

- **Keep the container non-root, SUID/SGID-stripped** (already enforced in the
  Dockerfile) — the SSRF gate is one layer; least-privilege bounds the blast
  radius of any bypass.
- **Do not set `allow_private=True`** on the open-web research path; reserve it
  for explicitly operator-hosted internal services (SearXNG/Ollama) that need
  RFC1918, and never expose that path to untrusted input.
- **Egress network policy**: where the platform supports it, additionally block
  169.254.0.0/16 at the network layer so an application-layer regression cannot
  reach the metadata service.
- **Meaning-level harm screening**: configure a real reasoning model
  (Ollama / RemoteReasoningBackend) so the weapons-gate routing rescue is active;
  otherwise the surface runs lexical-only and warns loudly (see
  `docs/WEAPONS_GATE_ADVERSARIAL_EVAL.md`, `docs/HARM_POLICY.md` §8). Set
  `MERCURY_REQUIRE_REAL_HARM_CLASSIFIER=1` to fail closed at enablement.

## 5. Sign-off checklist

- [x] Egress allowed/blocked both proven by tests (IMDS/loopback refused on the
      user-configured open-web path).
- [x] Decompression-bomb cap implemented + tested; urllib3 floor pinned.
- [x] FeatureCache stale-hit surface closed for CPU tensors + tested; CUDA
      tradeoff documented.
- [ ] Security reviewer sign-off (requested on the PR).
- [ ] Infra reviewer sign-off on the network-layer IMDS block recommendation.
