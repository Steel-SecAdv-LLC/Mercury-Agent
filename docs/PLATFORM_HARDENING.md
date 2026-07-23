<!-- Copyright (C) 2025 Steel Security Advisors LLC -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->

# Platform Hardening — Free-Service Account, Auth, and Abuse Controls

This document is the design, security, deployment, migration, and change record
for the account/persistence/auth/metering platform that runs Mercury Agent as a
free, public, account-based service (PR #350 and its hardening pass).

**Everything here is additive and opt-in.** A solo self-hoster who clones the
repo and runs it gets byte-identical behaviour to before: in-memory stores, no
accounts, no SMTP, no quota enforcement. The machinery activates only when the
environment variables below are set. `import omni_mercury_engine` and the
default detection path do not require any of it.

- [Threat model](#threat-model)
- [Architecture](#architecture)
- [Security controls (P0/P1/P2)](#security-controls)
- [Configuration reference](#configuration-reference)
- [Deployment](#deployment)
- [Migration steps](#migration-steps)
- [Change log](#change-log)
- [Acceptance checklist](#acceptance-checklist)
- [Test coverage map](#test-coverage-map)

---

## Threat model

The service is a public, unauthenticated-to-register endpoint. The controls
here target, concretely:

| Attack | Control |
|---|---|
| Rate-limit bypass by rotating `X-Forwarded-For` | Trusted-proxy client-IP resolution (right-most trusted hop only) |
| Rate-limit multiplication across workers / reset on restart | Shared, restart-persistent SQLite bucket + counter store |
| Online password guessing | Per-IP and per-account login throttles; memory-hard KDF |
| Signup / reset-email / verification-email flooding | Per-action throttles on register, reset-request, resend |
| Account enumeration | Uniform timing (decoy KDF), silent success on unknown email, identical throttle responses |
| TOTP seed theft via DB read / backup | AES-256-GCM at-rest sealing, AAD-bound to the account |
| TOTP code replay inside its 30 s window | Last-accepted time-step tracking, non-increasing steps rejected |
| Lost authenticator → locked-out account | Single-use recovery codes |
| Session fixation / hijack survival across credential change | Session rotation on password/email change, idle + absolute timeouts |
| CSRF on state-changing POSTs | SameSite=Lax + double-submit `X-CSRF-Token` |
| Cost blowout by one heavy account | Per-account/tier quota with atomic hard request ceiling |
| Silent 2FA compromise on key loss | Sealing refuses process-lifetime keys for durable stores |

## Architecture

All platform state lives in **one SQLite file** selected by
`MERCURY_KEYSTORE_PATH` (WAL mode, multi-process safe). Tables:

- `api_keys` — durable API keys (hash only).
- `accounts` — users: email, password hash, verified/active, tier, sealed TOTP
  secret, last-used TOTP step.
- `sessions` — session token hash, CSRF token hash, created/expires/last-seen.
- `email_tokens` — single-use verify/reset/email-change tokens (hash only).
- `recovery_codes` — single-use 2FA backup codes (hash only).
- `usage_events` — append-only per-account metering fact table.
- `quota_overrides` — per-account quota overrides.
- `rate_buckets` / `rate_counters` — shared rate-limit state.

The SQLite backend migrates older database files forward in place with additive
`ALTER TABLE ADD COLUMN` steps (`SqliteIdentityStore._MIGRATIONS`), so a file
created by an earlier build keeps working.

Module map (all under `src/omni_mercury_engine/api/`):

| Module | Responsibility |
|---|---|
| `client_ip.py` | Trusted-proxy client-IP resolution |
| `rate_limit_store.py` | Shared token-bucket + fixed-window counter stores; per-action limiter |
| `passwords.py` | scrypt KDF (+ legacy PBKDF2 verify), rehash policy |
| `secret_sealer.py` | AES-256-GCM at-rest sealing over `SecureDataHandler` |
| `totp.py` | RFC 6238 TOTP with step-reporting for replay rejection |
| `email_templates.py` | Plaintext + HTML transactional emails, List-Unsubscribe |
| `mailer.py` | SMTP / console mailer seam (multipart, headers) |
| `identity_store.py` | Durable accounts/sessions/tokens/recovery-codes + migrations |
| `usage_ledger.py` | Metering with atomic check-and-reserve |
| `quota.py` | Tier/override policy + reserve/commit enforcement |
| `quota_middleware.py` | HTTP wiring of quotas on metered routes |
| `auth_audit.py` | Auth-event audit seam (SecureAuditLogger or logging) |
| `auth_service.py` | The account lifecycle orchestrator |
| `maintenance.py` | Startup + periodic pruning sweeps; TOTP sealing migration |
| `routes/accounts.py` | FastAPI routes, per-action throttles, CSRF |

## Security controls

### P0 — Security & correctness

**Rate-limiter bypass fixed.** `client_ip.resolve_client_ip` reads the client
IP from the right-most *trusted* `X-Forwarded-For` hop, governed by
`MERCURY_TRUSTED_PROXY_HOPS` (default `0` = header untrusted, use the TCP peer).
The previous left-most-entry read let any caller mint a fresh rate-limit bucket
per request. Bucket state moved to `SqliteRateLimitBackend`, so limits are
global across workers and survive restarts; consumption is a single
`BEGIN IMMEDIATE` transaction (`consume_token`) — no double-spend under
concurrency.

**Per-action auth limits.** `ActionRateLimiter` layers fixed-window counters
(atomic, shared) on login (per IP and per account), register (per IP),
password-reset request (per IP and per account), and resend-verification (per
IP and per account). Limits are configuration (`MERCURY_AUTH_RATE_<ACTION>` =
`"<max>/<window-seconds>"`). Attempts count on entry — a failed or throttled
attempt still consumes budget, and the 429 response is identical for existing
and non-existent accounts.

**Non-blocking, failure-tolerant mailer.** Account rows commit **before** any
email I/O. `AuthService._send` wraps delivery in try/except with logging and
audit; with SMTP configured, sends run on a small background executor so a slow
mail server neither blocks nor time-fingerprints the request. `POST
/api/v1/auth/resend-verification` re-issues the verification link
(enumeration-safe).

**TOTP secrets encrypted at rest.** `SecretSealer` seals seeds with
AES-256-GCM via `SecureDataHandler.encrypt_at_rest`, with the owning account id
bound in as AAD (a swapped or tampered ciphertext fails closed). The cipher is
**AMA's own native AES-256-GCM** (`libama_cryptography.so` — no OpenSSL and no
third-party `cryptography` package on this path) with a **fresh random 96-bit
nonce per call and no persistent state**. It deliberately does *not* use AMA's
counter-nonce `AESGCMProvider`, which persists a per-key counter to
`~/.ama_cryptography/aes_gcm_counters.json` — the right model for a long-lived
streaming context but wrong for at-rest sealing (it fails on a read-only
filesystem and turns a hidden JSON file into decrypt-critical state). A 256-bit
key keeps a ≥128-bit post-quantum (Grover) margin, and random nonces keep the
collision probability negligible at at-rest volumes (NIST SP 800-38D §8.2.2).
Key material is a **stable** key from `MERCURY_DATA_ENC_KEY` or derived from
`AMA_MASTER_SEED` via HKDF-SHA256 with a purpose-scoped info string. For a
durable store with no stable key, the sealer marks itself non-stable and the
write path keeps seeds unsealed rather than bricking 2FA on the next restart.

**2FA recovery.** Confirming enrollment issues ten single-use recovery codes
(shown once, stored only as hashes). They authenticate in place of a TOTP code,
work exactly once (atomic conditional `UPDATE`), tolerate case/dash
normalisation, send a security notice on use, and are voided by regeneration
and by disabling 2FA.

**TOTP replay prevented.** `verify_totp_with_step` reports the matched time
step; the auth service persists the highest accepted step per account
(`accounts.totp_last_step`) and rejects any code whose step is not strictly
greater — a code sniffed inside its 30 s window cannot be reused.

**Singleton first-build race fixed.** `get_api_key_store`,
`get_auth_service`, `get_action_limiter`, `get_rate_limiter`, and
`get_audit_logger` use lock-guarded double-checked construction, so a
first-call stampede can never hand two threads different instances.

### P1 — Missing / incomplete features

**Pruning sweeps.** `maintenance.run_maintenance_sweep` prunes expired
sessions, consumed/expired email tokens, aged usage-ledger rows
(`MERCURY_USAGE_RETENTION_DAYS`, default 30), and stale rate-limit buckets and
counters. It runs once at server startup and then every
`MERCURY_MAINTENANCE_INTERVAL_SECONDS` (default 3600; `0` disables the loop),
failure-isolated per store.

**Per-account / per-tier quotas.** `quota.QuotaEnforcer.config_for` resolves
`override > tier > default`. Tiers come from `MERCURY_QUOTA_TIER_<NAME>`; the
account's `tier` column selects one. Overrides live in `quota_overrides`
(runtime-settable). Wired into metered routes by `QuotaMiddleware`
(`/api/v1/detect`, `/api/v1/batch` by default) using **reserve → run →
commit**: an atomic check-and-insert enforces a hard request ceiling, then the
measured compute cost is back-filled onto the reserved row. Returns HTTP 429
with `Retry-After` and usage headers.

**Auth-event audit.** `AuthAuditor` records login success/failure,
registration, password change/reset, 2FA enable/disable, recovery-code use,
email change, deletion, and API-key create/revoke — to `SecureAuditLogger`
(tamper-evident, hash-chained) when `MERCURY_AUDIT_LOG_DIR` is set, else
structured logging. Events carry the account id, never the email. The sink can
never raise into the request path. Retention is operator-configurable: the
logger rotates the active file at `MERCURY_AUDIT_ROTATE_SIZE_MB` (default 100)
and keeps `MERCURY_AUDIT_MAX_FILES` rotated segments (default 10); setting
`MERCURY_AUDIT_RETENTION_DAYS` additionally makes the maintenance sweep delete
whole rotated segments (and their `.sha256` sidecars) older than that horizon —
never the active file or individual lines, so every retained segment's hash
chain stays independently verifiable.

**Account lifecycle + CSRF.** New authenticated endpoints: change-password
(re-authenticated, rotates all sessions), change-email (two-step,
re-verifies the new address), account deletion (re-authenticated, hard delete),
and data export (no secrets). Every state-changing authenticated POST requires
a matching `X-CSRF-Token` (double-submit against the session-bound token),
toggleable with `MERCURY_CSRF_PROTECTION`. The change-email *request* is
additionally throttled per IP and per acting account
(`MERCURY_AUTH_RATE_EMAIL_CHANGE_IP` / `_ACCOUNT`) because it emails a
confirmation link to a caller-supplied address — an outbound-email abuse vector
if left unbounded.

**Account-scoped API keys.** Self-service issuance links keys to accounts:
`POST /api/v1/auth/api-keys` mints a key owned by the calling account (raw key
returned exactly once; only its hash is stored), `GET /api/v1/auth/api-keys`
lists the caller's keys (metadata only), and `DELETE
/api/v1/auth/api-keys/{key_id}` revokes one (ownership-enforced — another
account's key id 404s). Issuance requires a verified account, is CSRF-protected,
enforces a per-account active-key cap (`MERCURY_MAX_API_KEYS_PER_ACCOUNT`,
default 10), and validates requested permissions against a self-service
whitelist (`read`, `detect`, `export` — never `write`/`delete`/`admin`, so a
user cannot self-grant privileges it lacks). Because a key carries its owner's
account id, `QuotaMiddleware` now charges API-key traffic at the owning
**account's tier** (not a flat `free`), and a user's session and API key share
one per-account quota bucket. Issuance and revocation are recorded on the auth
audit trail (`api_key_created` / `api_key_revoked`).

### P2 — Quality / optimization

**Memory-hard KDF.** Passwords now use `hashlib.scrypt` (OWASP setting
`n=2^15, r=8, p=3` → 32 MiB/guess), tunable via `MERCURY_SCRYPT_N/R/P` and
measurable with `scripts/calibrate_password_kdf.py`. Legacy `pbkdf2_sha256`
hashes still verify and are transparently upgraded on the next successful login
(`needs_rehash`).

**Quota atomicity.** The request ceiling is hard (atomic reserve). The
compute-ms ceiling is intentionally **soft by one**: a request's true cost is
only known after it runs, so a single in-flight request may overshoot the
compute budget once before the window closes on the account. This is the
correct trade-off (you cannot bill before you compute); the request ceiling
bounds the blast radius.

**Email deliverability & UX.** `email_templates.py` produces matched
plaintext + inline-styled HTML with a `List-Unsubscribe` header on every
message.

**Session hardening.** Idle timeout (`MERCURY_SESSION_IDLE_SECONDS`) vs
absolute lifetime (`MERCURY_SESSION_TTL_SECONDS` / `_SHORT_SECONDS`), rotation
on privilege change (password/email change drop all sessions), and a
configurable remember-me (persistent vs browser-session cookie).

### Main-branch items

- **CI false-red webhooks.** `ci_gate_watchdog.py` now excuses a `cancelled`
  run when a newer run on the same `headBranch` reached a clean pass/fail — the
  ordinary `cancel-in-progress` shape on a busy PR — while still alerting on a
  branch that *never* reaches a verdict (the #348 pathology). `gate-watchdog.yml`
  passes `headBranch` in its `gh run list` JSON.
- **`migrate_pkl` codec allowlist.** The `("_codecs", "encode")` entry is
  **deliberately retained**: it is load-bearing for protocol-0/1/2 array
  reconstruction (numpy stores the databuffer as a latin-1 `str` under the old
  protocols), pinned by `tests/tools/test_migrate_pkl.py::TestAllProtocolsRoundTrip`.
  Trimming it would silently break migration of the oldest operator pickles.
  Its security analysis (an inert byte transform with no reachable exec gadget)
  is in the `migrate_pkl` module docstring; removing it would be a regression,
  not a minimisation.
- **mypy.** All new and modified source is clean under the pinned strict
  `mypy==2.3.0` gate.

## Configuration reference

All variables are optional; unset values keep the pre-platform behaviour.

| Variable | Default | Purpose |
|---|---|---|
| `MERCURY_KEYSTORE_PATH` | *(unset → in-memory)* | One SQLite file backing all platform state |
| `MERCURY_TRUSTED_PROXY_HOPS` | `0` | Trailing `X-Forwarded-For` hops your proxy tier appends |
| `MERCURY_SMTP_HOST` etc. | *(unset → console)* | SMTP delivery (`_PORT/_USERNAME/_PASSWORD/_FROM/_STARTTLS`) |
| `MERCURY_PUBLIC_BASE_URL` | `https://mercuryagent.global` | Base URL for email links |
| `MERCURY_CONTACT_EMAIL` | `contact@mercuryagent.global` | `List-Unsubscribe` contact |
| `MERCURY_DATA_ENC_KEY` | *(unset)* | 64-hex at-rest key for TOTP sealing (`python scripts/generate_secret_key.py`) |
| `MERCURY_DATA_ENC_KEY_OLD` | *(unset)* | Retiring at-rest key, read by `scripts/reseal_totp_secrets.py` during key rotation |
| `AMA_MASTER_SEED` | *(unset)* | Fleet HD seed; TOTP sealing key derived via HKDF when set |
| `MERCURY_AUTH_RATE_<ACTION>` | *(built-in)* | Per-action limit `"<max>/<seconds>"` (LOGIN_IP, LOGIN_ACCOUNT, REGISTER_IP, RESET_IP, RESET_ACCOUNT, RESEND_IP, RESEND_ACCOUNT, EMAIL_CHANGE_IP, EMAIL_CHANGE_ACCOUNT) |
| `MERCURY_MAX_API_KEYS_PER_ACCOUNT` | `10` | Cap on active API keys per account (issuance 409s at the cap) |
| `MERCURY_SCRYPT_N/R/P` | `32768/8/3` | scrypt cost parameters |
| `MERCURY_SESSION_TTL_SECONDS` | `1209600` (14 d) | Absolute session lifetime (remember-me) |
| `MERCURY_SESSION_TTL_SHORT_SECONDS` | `86400` (1 d) | Absolute lifetime without remember-me |
| `MERCURY_SESSION_IDLE_SECONDS` | `86400` (1 d) | Idle timeout |
| `MERCURY_SESSION_COOKIE_SECURE` | `true` | `Secure` flag on cookies (set `false` only for local HTTP) |
| `MERCURY_CSRF_PROTECTION` | `true` | Require `X-CSRF-Token` on state-changing POSTs |
| `MERCURY_QUOTA_ENABLED` | `false` | Turn on quota enforcement on metered routes |
| `MERCURY_QUOTA_WINDOW_SECONDS` / `_MAX_REQUESTS` / `_MAX_COMPUTE_MS` | `3600/1000/600000` | Default quota ceilings |
| `MERCURY_QUOTA_TIER_<NAME>` | *(unset)* | Tier `"<max_requests>,<max_compute_ms>[,<window_seconds>]"` |
| `MERCURY_QUOTA_METERED_PREFIXES` | `/api/v1/detect,/api/v1/batch` | Path prefixes the quota middleware guards |
| `MERCURY_QUOTA_FAIL_CLOSED` | `false` | Deny metered requests with 503 when the quota infrastructure itself fails (see the trade-off note under Deployment) |
| `MERCURY_MAINTENANCE_INTERVAL_SECONDS` | `3600` | Sweep interval (`0` disables the periodic loop) |
| `MERCURY_USAGE_RETENTION_DAYS` | `30` | Usage-ledger retention (must exceed the largest quota window) |
| `MERCURY_AUDIT_LOG_DIR` | *(unset → logging)* | Tamper-evident audit trail directory |
| `MERCURY_AUDIT_ROTATE_SIZE_MB` | `100` | Rotate the active audit log at this size |
| `MERCURY_AUDIT_MAX_FILES` | `10` | Rotated audit segments to retain (count-based cap) |
| `MERCURY_AUDIT_RETENTION_DAYS` | *(unset → count-based only)* | Also delete rotated segments older than this many days (sweep-driven) |
| `MERCURY_FRONTEND_ENABLED` | `false` | Serve the browser account UI (`/`, `/login`, `/register`, `/dashboard`, the email-link pages, and the `/static` assets) from the API process |

## Deployment

1. **Shared store.** Set `MERCURY_KEYSTORE_PATH` to a path on durable storage
   (a mounted volume, not the container's ephemeral FS). All workers must point
   at the same file. WAL mode makes concurrent workers safe on one host; for
   multi-host, front the file with a single-writer host or migrate to a Postgres
   backend (a future sibling implementation of the same store contracts).
2. **Proxy header.** Set `MERCURY_TRUSTED_PROXY_HOPS` to the exact number of
   proxies in front of the app that append an `X-Forwarded-For` entry (e.g. `1`
   for a single nginx/Caddy reverse proxy, `2` for LB → reverse proxy). Leaving
   it at `0` behind a proxy makes every client share the proxy's IP bucket
   (over-throttling but never a bypass); setting it too high trusts
   client-supplied entries — match it to reality.
3. **Secrets.** Provide `MERCURY_DATA_ENC_KEY` (or `AMA_MASTER_SEED`) and
   `JWT_SECRET_KEY`/`AMA_MASTER_SEED` from your secret store, never the repo.
   Generate them with the bundled, dependency-free
   `python scripts/generate_secret_key.py` (stdlib CSPRNG — **no OpenSSL or any
   external tool required**; `--all` prints an export line for every key var,
   `--bytes 64` a master seed).
4. **SMTP.** Set `MERCURY_SMTP_*` for real email; unset falls back to console
   logging (fine for dev). Set `MERCURY_PUBLIC_BASE_URL` so email links resolve.
5. **Quotas.** Set `MERCURY_QUOTA_ENABLED=true` and tune the ceilings/tiers.
   By default a failure *of the quota infrastructure itself* (e.g. the SQLite
   file becomes unreadable) admits the request unmetered — availability
   outranks accounting, and the global rate limiter still bounds volume. Set
   `MERCURY_QUOTA_FAIL_CLOSED=true` to invert that: metered routes then
   return 503 until the quota store recovers. Choose fail-closed only when
   unmetered compute is a bigger risk to you than an outage window — e.g.
   expensive GPU-backed endpoints on a public deployment; keep the default
   when the service being reachable matters more than exact accounting.
6. **Audit.** Set `MERCURY_AUDIT_LOG_DIR` to a durable, append-only path.
7. **Runtime image** must ship the native AMA-Cryptography backend (the
   import-time PQC gate); the at-rest sealer and JWT paths depend on it.
8. **Frontend.** Set `MERCURY_FRONTEND_ENABLED=true` to serve the account UI
   (registration, login with 2FA, the email-link pages, and the dashboard —
   API keys, usage, password/email change, 2FA lifecycle, export, deletion).
   Pure static assets served from the installed package: vanilla HTML/CSS/JS,
   no build toolchain, no CDN; the TOTP QR code renders client-side with the
   vendored MIT `qrcode-generator`. Left unset, `/` stays a 404 and nothing
   changes. The pages use the same public API the docs describe — enabling
   the UI adds no new API surface to protect.

### Compose runbook (single-host platform deployment)

The repo ships the deployment as a compose overlay —
`docker-compose.platform.yml` layered over the base `docker-compose.yml` —
so plain `docker compose up` keeps its unchanged local-dev behaviour and the
platform profile of the deployment is one extra `-f`:

```bash
# One-time on the host: provision secrets into .env (stdlib CSPRNG, no
# OpenSSL needed). Review the file afterwards; it now holds key material.
cp .env.example .env
python scripts/generate_secret_key.py --all >> .env

# Bring up the platform: API + durable state volume + Caddy TLS edge
# (+ the base Prometheus/Grafana monitoring stack).
docker compose -f docker-compose.yml -f docker-compose.platform.yml up -d

# Verify from the host:
curl -fsS http://localhost:8000/health
docker compose -f docker-compose.yml -f docker-compose.platform.yml ps
```

What the overlay adds, and why:

* **Durable state** — a named volume (`mercury-platform-data`) mounted at
  `/var/lib/mercury`; `MERCURY_KEYSTORE_PATH` and `MERCURY_AUDIT_LOG_DIR`
  point into it, so accounts, sessions, quotas, rate-limit state, and the
  audit chain survive container replacement.
* **Configuration pass-through** — every documented `MERCURY_*` variable is
  wired host-env → container with the safe default from the reference table
  above (quotas and the frontend default *on* for this profile). The
  parameterised families (`MERCURY_AUTH_RATE_<ACTION>`,
  `MERCURY_QUOTA_TIER_<NAME>`) go in `.env`, which the app service loads.
* **TLS edge** — a Caddy service (`deploy/Caddyfile`) terminates 80/443 for
  `app.mercuryagent.global` with automatic Let's Encrypt certificates and
  actively health-checks the app on `/health`. The app runs with
  `MERCURY_TRUSTED_PROXY_HOPS=1` to match the exactly-one
  `X-Forwarded-For` hop Caddy appends.

DNS, the Wix mailbox, and Hetzner host provisioning are the human-owned
half; they are documented separately in `docs/DOMAIN_EMAIL_HOSTING_SETUP.md`
(PR #343) — the code side only ever reads the environment variables above.

## Agent runtime requirements

The agentic stack — `MultiAgentOrchestrator` episodes, `MercuryAgent`'s
sub-agent fleet (`enable_fleet()` / `delegate()`), and the fleet detection
paths — **requires the `[ml]` extra** (PyTorch). This is enforced twice, at
different depths:

1. **Import time.** `omni_mercury_engine.agentic` imports torch while
   loading (via `cognitive.orchestrator` → `utils`); on a core-lane install
   (`pip install "mercury-agent[api]"`, no `[ml]`) the import itself raises
   `ImportError`, so there is no partially-agentic mode to misconfigure.
2. **Decision time (fail-closed σ gate).** Even with the package importable,
   every orchestrated decision boundary runs the σ_Immutable hard gate. When
   the trained GOSNN network cannot load (torch or the shipped weights
   unavailable), the gate refuses with
   `EthicalConstraintViolationError(check="gosnn_unavailable", score=0.0)`
   against the 0.93 threshold — episodes are blocked, never silently
   ungated. This is deliberate design, not a bug; guarded by
   `tests/security/test_sigma_immutable_fail_closed.py` (gate level, runs in
   the no-torch lane) and
   `tests/cognitive/test_orchestrator_gosnn_unavailable.py` (orchestrator
   boundary). The passing path — a fitted episode clearing both the
   benevolence and σ gates — is pinned by
   `tests/cognitive/test_orchestration_behavioral.py::TestEthicalGating`.

Nothing extra to download at runtime: the σ_Immutable weights
(`security/sigma_immutable_weights.pt`), the signed corpus, and the domain
checkpoints ship in the repo and in the built wheel (package-data). The
Docker image installs `.[all]` (which includes `[ml]`) and builds the native
AMA backend, so the runtime image already carries the complete agent stack;
only bare-metal `[api]`-only installs are API/detection-surface only.

## Migration steps

Enabling the platform on an **existing** deployment:

1. **Schema.** No manual DDL. `SqliteIdentityStore` applies additive column
   migrations in place at first open (`totp_last_step`, `tier`, session
   `csrf_hash`/`last_seen_at`, token `payload`). Back up the SQLite file first.
2. **Encrypt existing TOTP secrets.** Set `MERCURY_DATA_ENC_KEY`, then either
   restart (the startup sweep seals them) or run one sweep:
   ```bash
   python -c "from omni_mercury_engine.api.maintenance import run_maintenance_sweep; print(run_maintenance_sweep())"
   ```
   The `sealed_totp_secrets` count reports how many were upgraded. The sweep is
   idempotent and is a no-op if no stable key is configured (it will not seal
   under a process-lifetime key).
3. **Quota schema.** `usage_events` and `quota_overrides` are created on first
   use; no backfill is needed (usage accrues forward).
4. **Audit backfill.** The audit trail is forward-only; historical events are
   not reconstructed. Point `MERCURY_AUDIT_LOG_DIR` at a fresh directory.
5. **Rehash on login.** Existing PBKDF2 password hashes upgrade to scrypt
   automatically on each user's next successful login — no bulk migration.
6. **Rotating the at-rest key.** To change `MERCURY_DATA_ENC_KEY` without
   bricking 2FA (a sealed secret only opens under the key it was sealed with),
   run the re-seal tool with the retiring key alongside the new one:
   ```bash
   export MERCURY_KEYSTORE_PATH=/var/lib/mercury/mercury.db
   export MERCURY_DATA_ENC_KEY=<new key>        # seal under this
   export MERCURY_DATA_ENC_KEY_OLD=<old key>    # unseal with this
   python scripts/reseal_totp_secrets.py --dry-run   # preview
   python scripts/reseal_totp_secrets.py             # apply
   ```
   It is idempotent (already-migrated rows are skipped), never overwrites a
   value it cannot open under either key, and exits non-zero if any such row
   remains — so a bad old key is caught, not silently lost. Deploy the new
   `MERCURY_DATA_ENC_KEY` only after the tool reports zero failures.

## Change log

| # | Change | Why | Validated by |
|---|---|---|---|
| 1 | Trusted-proxy client-IP resolution | Rotating XFF minted fresh rate-limit buckets (total bypass) | `test_client_ip.py`, `test_rate_limit_hardening.py::TestMiddlewareSpoofImmunity` |
| 2 | Shared SQLite bucket + counter stores | Per-worker limits multiplied budget; restart reset it | `test_rate_limit_hardening.py::TestSqliteBucketBackend/TestCounterStores` |
| 3 | Per-action auth throttles | Global limit still allowed 100 guesses/min per account | `test_auth_abuse_controls.py` |
| 4 | Best-effort mailer + resend endpoint | A mail outage orphaned registrations | `test_maintenance_and_mailer.py::TestMailerResilience`, `test_account_routes.py` |
| 5 | TOTP at-rest sealing + migration | DB/backup read yielded usable 2FA seeds | `test_secret_sealing_and_recovery.py::TestSecretSealer/TestSealedAtRest` |
| 6 | 2FA recovery codes | Lost authenticator = lost account | `test_secret_sealing_and_recovery.py::TestRecoveryCodes` |
| 7 | TOTP replay rejection | Sniffed code replayable within its window | `test_secret_sealing_and_recovery.py::TestTotpReplay` |
| 8 | Singleton lock-guarded init | First-call race split callers across instances | `test_singleton_race.py` |
| 9 | Maintenance sweeps | Unbounded growth of ledger/sessions/tokens/buckets | `test_maintenance_and_mailer.py::TestMaintenanceSweep` |
| 10 | Per-account/tier quotas + atomic reserve + wiring | Metering existed but never enforced; races could overrun | `test_quota_enforcement.py` |
| 11 | Auth-event audit seam | No trail of security-relevant account events | `test_maintenance_and_mailer.py::TestAuditSeam` |
| 12 | Lifecycle endpoints + CSRF | No change-pw/email/delete/export; no CSRF defense | `test_account_lifecycle.py` |
| 13 | scrypt KDF + calibration | PBKDF2 is not memory-hard (GPU/ASIC-friendly) | `test_password_kdf.py`, `calibrate_password_kdf.py` |
| 14 | Session idle/absolute/rotation/remember-me | Single fixed TTL; no rotation on privilege change | `test_account_lifecycle.py::TestSessionHardening` |
| 15 | HTML emails + List-Unsubscribe | Deliverability + UX | `test_maintenance_and_mailer.py::test_emails_carry_html_and_unsubscribe` |
| 16 | CI watchdog superseded-cancel logic | False alarms on every busy PR | `test_ci_gate_watchdog.py` |
| 17 | Account-scoped API-key issuance + owner-tier quota | Keys weren't linked to accounts; key traffic was flat-`free` tier | `test_api_key_routes.py` |
| 18 | Email-change request throttle | Emailed a caller-supplied address unthrottled (outbound-email abuse) | `test_api_key_routes.py::TestEmailChangeThrottle` |
| 19 | Operator-configurable audit retention + time-based prune | Rotation knobs were hardcoded; no time-based compliance retention | `test_audit_retention.py` |
| 20 | TOTP at-rest key rotation tool | No way to rotate `MERCURY_DATA_ENC_KEY` without bricking 2FA | `test_reseal_totp_secrets.py` |

## Acceptance checklist

Each item has a concrete verification command. Run from the repo root with the
native AMA backend on `LD_LIBRARY_PATH` (see `.github/actions/build-ama-cryptography`).

- [x] **P0 rate-limiter bypass fixed & shared.**
      `pytest tests/api/test_client_ip.py tests/api/test_rate_limit_hardening.py`
- [x] **P0 per-action auth limits.**
      `pytest tests/api/test_auth_abuse_controls.py`
- [x] **P0 non-blocking mailer + resend.**
      `pytest tests/api/test_maintenance_and_mailer.py -k Mailer`
      and `POST /api/v1/auth/resend-verification` is registered.
- [x] **P0 TOTP sealed at rest + migration.**
      `pytest tests/api/test_secret_sealing_and_recovery.py -k "Sealed or Sealer"`
- [x] **P0 2FA recovery codes.**
      `pytest tests/api/test_secret_sealing_and_recovery.py -k Recovery`
- [x] **P0 TOTP replay rejected.**
      `pytest tests/api/test_secret_sealing_and_recovery.py -k Replay`
- [x] **P0 singleton race guarded.**
      `pytest tests/api/test_singleton_race.py`
- [x] **P1 pruning sweeps.**
      `pytest tests/api/test_maintenance_and_mailer.py -k Maintenance`
- [x] **P1 quota tiers/overrides + wiring + 429.**
      `pytest tests/api/test_quota_enforcement.py`
- [x] **P1 auth-event audit.**
      `pytest tests/api/test_maintenance_and_mailer.py -k Audit`
- [x] **P1 lifecycle endpoints + CSRF.**
      `pytest tests/api/test_account_lifecycle.py`
- [x] **P2 scrypt KDF + calibration.**
      `pytest tests/api/test_password_kdf.py` and
      `python scripts/calibrate_password_kdf.py`
- [x] **P2 session hardening.**
      `pytest tests/api/test_account_lifecycle.py::TestSessionHardening`
- [x] **Account-scoped API keys + owner-tier quota + email-change throttle.**
      `pytest tests/api/test_api_key_routes.py`
- [x] **Configurable audit retention + segment prune.**
      `pytest tests/api/test_audit_retention.py`
- [x] **At-rest key rotation (TOTP re-seal).**
      `pytest tests/scripts/test_reseal_totp_secrets.py`
- [x] **Main-branch CI watchdog.**
      `pytest tests/scripts/test_ci_gate_watchdog.py`
- [x] **No regressions.**
      `pytest tests/api tests/security tests/tools tests/scripts` green;
      `black --check`, `ruff check`, `pydocstyle --convention=google` on `src/`,
      and strict `mypy` clean on new source.

## Test coverage map

| Area | Unit | Integration (HTTP) | Adversarial / mutation |
|---|---|---|---|
| Client-IP resolution | `test_client_ip.py` | `test_rate_limit_hardening.py` middleware | Hypothesis fuzz sweep; rotating-spoof regression |
| Rate-limit stores | `test_rate_limit_hardening.py` | middleware end-to-end | Concurrent overspend / undercount races |
| Per-action limits | `test_rate_limit_hardening.py` | `test_auth_abuse_controls.py` | Identical-response enumeration check |
| Password KDF | `test_password_kdf.py` | login rehash in `test_identity_and_auth.py` | Tampered-parameter memory-bomb bound |
| TOTP sealing | `test_secret_sealing_and_recovery.py` | login through sealed secret | Per-byte GCM mutation sweep; AAD swap |
| TOTP replay | `test_secret_sealing_and_recovery.py` | `test_account_routes.py` 2FA | Earlier-window code rejection |
| Recovery codes | `test_secret_sealing_and_recovery.py` | 2FA confirm route | Double-use race via atomic UPDATE |
| Quotas | `test_quota_enforcement.py` | quota middleware | 40-thread reserve overrun race |
| Sessions | `test_account_lifecycle.py` | cookie flow | Idle vs absolute vs activity |
| Lifecycle / CSRF | `test_account_lifecycle.py` | full HTTP flows | Missing/forged CSRF; email-change race |
| API keys | `test_api_key_routes.py` | issue/list/revoke HTTP flows | Cross-account revoke 404; cap 409; permission-whitelist escalation; owner-tier quota; email-change throttle |
| Audit retention | `test_audit_retention.py` | `build_auth_auditor` env config | Time-based segment prune preserves active chain integrity |
| At-rest key rotation | `test_reseal_totp_secrets.py` | SQLite round-trip + CLI | No-old-key/unopenable → reported, never overwritten; idempotent re-run |
| Singletons | `test_singleton_race.py` | — | 24-thread first-build stampede |
| Maintenance | `test_maintenance_and_mailer.py` | — | Failure isolation, idempotent migration |
| CI watchdog | `test_ci_gate_watchdog.py` | — | Superseded vs pathological cancellation |
