<!--
Copyright (C) 2025 Steel Security Advisors LLC
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Live data sources & keyed loaders

Mercury's domain loaders fetch **real** measured data from public
government/research/commercial APIs. Seven of these data sources authenticate
with a key — and USGS EROS M2M uses **two** secrets (an application token *and*
an ERS username; see its note below). Each secret is stored as a **GitHub
Actions repository secret** and is injected into the relevant CI lanes only —
never committed to the repository, echoed into logs, written into any artifact
or provenance sidecar, or exposed outside CI.

## Secret → env var → loader → endpoint → CI lane

| Repo secret | Env var the loader reads | Loader | Live endpoint (host on the SSRF allowlist) | Exercised by |
|---|---|---|---|---|
| `FIRMS_MAP_KEY` | `NASA_FIRMS_MAP_KEY` (alias — see note) | `loaders/wildfire_loader.py` (`WildfireLoader`) | `firms.modaps.eosdis.nasa.gov` | `network-tests`, `dataset-reachability`, `live-data-validation` |
| `NASA_API_KEY` | `NASA_API_KEY` | `loaders/space_weather_loader.py` (DONKI), `loaders/meteor_loader.py` (NeoWs) | `api.nasa.gov` | `network-tests`, `dataset-reachability`, `live-data-validation` |
| `FRED_API_KEY` | `FRED_API_KEY` | `loaders/financial_loader.py` (`FinancialLoader`, FRED) | `api.stlouisfed.org` | `network-tests`, `dataset-reachability`, `live-data-validation` |
| `EIA_API_KEY` | `EIA_API_KEY` | `loaders/energy_loader.py` (`EnergyLoader`) | `api.eia.gov` | `network-tests`, `dataset-reachability`, `live-data-validation` |
| `ALPHA_VANTAGE_API_KEY` | `ALPHA_VANTAGE_API_KEY` | `integrations/stubs/financial.py` (`FinancialService`, Alpha Vantage market data) | `www.alphavantage.co` | `network-tests`, `dataset-reachability`, `live-data-validation` |
| `OPENWEATHERMAP_API_KEY` | `OPENWEATHERMAP_API_KEY` | `integrations/stubs/weather.py` (`WeatherService`, OpenWeatherMap) | `api.openweathermap.org` | `network-tests`, `dataset-reachability`, `live-data-validation` |
| `USGS_KEY` **+** `EROSERS_USERNAME` | `USGS_KEY`, `EROSERS_USERNAME` | `integrations/usgs_eros.py` (`USGSErosM2MClient`, EROS M2M `login-token` → dataset search) | `m2m.cr.usgs.gov` | `network-tests`, `dataset-reachability`, `live-data-validation` |

### The two-secret source: USGS EROS M2M

The USGS EROS Machine-to-Machine API (`m2m.cr.usgs.gov`, the interface behind
EarthExplorer) authenticates with the **`login-token`** endpoint — the legacy
`login` (username + password) endpoint was **deprecated 2026-02-26**.
`login-token` requires **both** an ERS **username** and a 64-character
**application token** (generated at `ers.cr.usgs.gov/password/appgenerate`,
used in place of the password). So this one source needs two secrets:

```yaml
env:
  USGS_KEY: ${{ secrets.USGS_KEY }}                 # the application token
  EROSERS_USERNAME: ${{ secrets.EROSERS_USERNAME }} # the ERS username
```

`USGSErosM2MClient.available()` requires both to be set; with only one it fails
loudly rather than half-authenticating. The returned API key lives only in
memory, is sent as the `X-Auth-Token` header, and is invalidated by `logout()`.

### The one name mismatch: FIRMS

The repo secret is named **`FIRMS_MAP_KEY`**, but `WildfireLoader` reads
**`NASA_FIRMS_MAP_KEY`** (its `API_KEY_ENV_VAR`). Every workflow that needs it
therefore sets **both** names to the same secret value so the loader sees the
key under the exact variable it reads, e.g.:

```yaml
env:
  FIRMS_MAP_KEY: ${{ secrets.FIRMS_MAP_KEY }}
  NASA_FIRMS_MAP_KEY: ${{ secrets.FIRMS_MAP_KEY }}
```

The other five secrets are named identically to the env var their loader reads,
so they need no alias. `NASA_API_KEY` is optional for its two loaders (they fall
back to NASA's rate-limited `DEMO_KEY` when unset), but wiring the real key
raises the rate limit and is what the live smoke asserts against.

## Where the keys are consumed in CI

- **`network-tests.yml`** (weekly + dispatch) — the `@pytest.mark.network`
  suite, including the keyed live-wiring round trips.
- **`dataset-reachability.yml`** (nightly + dispatch) — the network reachability
  harness for the 11 domain loaders.
- **`live-data-validation.yml`** (weekly + dispatch) — the dedicated
  keyed-loader smoke (`scripts/live_data_smoke.py`): for every keyed loader
  whose secret is configured, it performs one real fetch and asserts real rows
  come back; a loader whose secret is absent is reported `SKIP` (never a silent
  pass), and any keyed-and-configured loader that returns empty or errors fails
  the job so live-wiring drift surfaces within a week.

None of these are PR gates — they are network- and secret-dependent and run only
on schedule/`workflow_dispatch`.

## SSRF allowlist

Every host above is on the `SafeHTTPClient` trusted-domain allowlist
(`security/input_validation.py`, `TrustedEndpoints.TRUSTED_DOMAINS`), HTTPS-only,
matched on the final host after any redirect. `www.alphavantage.co` and
`api.openweathermap.org` were added alongside this wiring.

## Handling rules (non-negotiable)

- Reference each secret **only** as `${{ secrets.NAME }}` in a workflow `env:`
  block. Never inline a value, never `echo` a key, never write one to an
  artifact, provenance sidecar, or PR body.
- The smoke script prints env-var **names** and row counts only — never key
  material.
- Keys are unavailable outside CI (no shell/agent environment carries them), so
  live wiring is verified in CI, not locally.
