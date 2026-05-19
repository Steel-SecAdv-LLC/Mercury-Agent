# Medical modules — operator setup guide

Applies to Mercury Agent **v1.6.x and the v1.7 development cycle**. Last updated: 2026-05-19.

Mercury Agent's medical modules ship **integration-ready, not pre-integrated**.
The platform never carries vendor credentials and never fabricates patient
data. This document is written for the licensed clinician (or their IT staff)
who deploys Mercury Agent in their own environment and supplies their own
data sources.

> **Decision support only.** The endocrinology detector and anesthesiology
> predictor are decision-support tools. Clinical validation by the
> appropriate licensed specialist is required before any output is used to
> influence patient care.

---

## Contents

- [Architecture](#architecture)
- [Configuration error contract](#configuration-error-contract)
- [Dexcom v3](#dexcom-v3)
- [FHIR R4 vital signs](#fhir-r4-vital-signs)
- [Custom adapter contract](#custom-adapter-contract)
- [Operational checklist](#operational-checklist)
- [Provenance](#provenance)

---

## Architecture

Both medical detectors expose the same shape:

```python
from omni_mercury_engine.medical import (
    DexcomV3DataSource,
    EndocrinologyDetector,
    FHIRObservationVitalsSource,
    AnesthesiologyPredictor,
)

cgm = DexcomV3DataSource()          # reads DEXCOM_* env vars
endo = EndocrinologyDetector(cgm)   # raises ConfigurationError otherwise

vitals = FHIRObservationVitalsSource()  # reads FHIR_* env vars
anes = AnesthesiologyPredictor(vitals)  # raises ConfigurationError otherwise

endo.fetch_and_detect(window_minutes=180)
anes.fetch_and_predict(window_minutes=5)
```

The detectors do not perform I/O themselves. They consume the abstract
contracts:

- [`CGMDataSource`](../../src/omni_mercury_engine/medical/data_sources.py) —
  produces `list[CGMReading]` for a configurable look-back window.
- [`VitalsDataSource`](../../src/omni_mercury_engine/medical/data_sources.py)
  — produces `list[VitalsReading]` for a configurable look-back window.

The reference implementations (`DexcomV3DataSource`,
`FHIRObservationVitalsSource`) speak the public Dexcom Developer API v3 and
HL7 FHIR R4 respectively. Vendor SDK adapters (Philips IntelliVue, GE
CARESCAPE, Mindray BeneVision, Abbott LibreView, Medtronic CareLink) can be
written as additional subclasses; see
[Custom adapter contract](#custom-adapter-contract).

## Configuration error contract

Mercury Agent refuses to start an integration that is misconfigured. The
following all raise
`omni_mercury_engine.medical.data_sources.ConfigurationError`:

- Instantiating `EndocrinologyDetector` with `enable_cgm=True` (the default)
  and no `data_source`.
- Instantiating `AnesthesiologyPredictor` with `enable_hemodynamics=True`
  (the default) and no `data_source`.
- Instantiating `DexcomV3DataSource()` without the required
  `DEXCOM_CLIENT_ID` / `DEXCOM_CLIENT_SECRET` / `DEXCOM_REFRESH_TOKEN` /
  `DEXCOM_REDIRECT_URI` environment variables (and no explicit
  `DexcomConfig`).
- Instantiating `FHIRObservationVitalsSource()` without `FHIR_BASE_URL` /
  `FHIR_PATIENT_ID` (and no explicit `FHIRConfig`).
- Calling `fetch_and_detect()` / `fetch_and_predict()` on a detector that
  was constructed with the relevant subsystem disabled.

There is **no demo mode, no synthetic fallback, and no silent degradation**.
If you see a `ConfigurationError` in production, an adapter is genuinely
unconfigured — surface it loudly.

## Dexcom v3

The reference CGM adapter implements the OAuth 2.0 refresh-token flow
against Dexcom's public Developer API.

### 1. Obtain credentials

1. Register an application at the [Dexcom Developer Portal](https://developer.dexcom.com/).
2. Note the **Client ID**, **Client Secret**, and **Redirect URI** for your
   registered application.
3. Complete the [Authorization Code grant](https://developer.dexcom.com/v3/docs/authentication)
   once for each patient who consents to share data. The final step exchanges
   the authorization code for an **access token** *and* a **refresh token**;
   only the refresh token needs to be stored long-term.
4. Repeat the same steps against `https://sandbox-api.dexcom.com` for any
   non-production environment. The sandbox accepts the same endpoints with
   simulated patients.

### 2. Configure environment variables

| Variable | Required | Purpose |
| --- | --- | --- |
| `DEXCOM_CLIENT_ID` | yes | OAuth2 client id from the developer portal. |
| `DEXCOM_CLIENT_SECRET` | yes | OAuth2 client secret. |
| `DEXCOM_REFRESH_TOKEN` | yes | Per-patient refresh token from the consent flow. |
| `DEXCOM_REDIRECT_URI` | yes | Redirect URI registered on the application. Must match exactly. |
| `DEXCOM_BASE_URL` | no | Defaults to `https://api.dexcom.com`. Override with `https://sandbox-api.dexcom.com` for sandbox testing. |

Store these in your secret manager of choice (HashiCorp Vault, AWS Secrets
Manager, GCP Secret Manager, Kubernetes secrets, systemd `EnvironmentFile`,
etc.). Mercury Agent never writes credentials to logs and never echoes them
to stdout.

### 3. Verify the connection

```python
from omni_mercury_engine.medical import DexcomV3DataSource

source = DexcomV3DataSource()
readings = source.fetch_recent_readings(window_minutes=60)
print(len(readings), "readings in the last hour")
```

Successful authentication will log a `Mercury-Agent/1.7 Endocrinology`
user-agent. A failure raises
`omni_mercury_engine.medical.data_sources.DataSourceError` with the
Dexcom-reported HTTP code and reason phrase — these are surfaced unchanged so
your monitoring stack can alert on them.

### 4. Refresh-token rotation

Per Dexcom policy a successful refresh **rotates the refresh token**. The
reference adapter does not persist tokens; integrators must wire the
rotation into their own secret store (re-read `DEXCOM_REFRESH_TOKEN` from
the secret manager on each process restart, or subclass
`DexcomV3DataSource._refresh_access_token` to write the new refresh token
back). For production deployments handling multiple patients, prefer a
custom subclass that loads the refresh token from a per-patient row in your
PHI-segregated database.

## FHIR R4 vital signs

The reference vitals adapter searches any spec-compliant HL7 FHIR R4 server
for vital-sign Observations.

### 1. Choose a FHIR endpoint

Any of the following work out of the box:

- Production EHR FHIR APIs: Epic, Oracle Health / Cerner, MEDITECH, Athena.
- SMART-on-FHIR sandboxes: <https://launch.smarthealthit.org/>.
- On-prem HL7 v2 gateways with a FHIR translation layer.

The adapter expects an `Observation` resource search endpoint at
`{base_url}/Observation` with `category=vital-signs` filtering.

### 2. Configure environment variables

| Variable | Required | Purpose |
| --- | --- | --- |
| `FHIR_BASE_URL` | yes | FHIR R4 base URL (no trailing slash). Must start with `http://` or `https://`. |
| `FHIR_PATIENT_ID` | yes | Logical id of the `Patient` resource being monitored. |
| `FHIR_BEARER_TOKEN` | no | Pre-issued OAuth2 bearer token. Required for most production servers; optional for open sandboxes. |

### 3. LOINC code coverage

The bundled parser recognises the standard vital-sign LOINC codes:

| LOINC | Channel |
| --- | --- |
| `8867-4` | Heart rate |
| `8480-6` | Systolic BP |
| `8462-4` | Diastolic BP |
| `8478-0` | Mean BP (used directly when reported) |
| `2708-6` / `59408-5` | SpO₂ |
| `19911-5` | End-tidal CO₂ |

When MAP is not reported directly the parser computes
`(SBP + 2 · DBP) / 3` from the systolic/diastolic component pair on the
same blood-pressure-panel Observation (LOINC 85354-9 is the typical
container code). Observations sharing the same `effectiveDateTime` are
merged into a single `VitalsReading` snapshot.

### 4. Verify the connection

```python
from omni_mercury_engine.medical import FHIRObservationVitalsSource

source = FHIRObservationVitalsSource()
readings = source.fetch_recent_vitals(window_minutes=5)
print(len(readings), "snapshots in the last five minutes")
```

## Custom adapter contract

Vendors not covered by the reference adapters can be added by subclassing
the relevant abstract base class. The contract is intentionally narrow:

```python
from datetime import datetime, timezone
from omni_mercury_engine.medical import CGMDataSource, CGMReading

class AbbottLibreSource(CGMDataSource):
    """Adapter for Abbott LibreView (https://libreview.com/)."""

    name = "abbott_libre"

    def __init__(self, account_token: str) -> None:
        if not account_token:
            raise ConfigurationError("LibreView account token is required")
        self._account_token = account_token

    def fetch_recent_readings(self, window_minutes: int = 180) -> list[CGMReading]:
        # 1. Authenticate against LibreView using ``self._account_token``.
        # 2. Fetch the latest CGM samples within ``window_minutes``.
        # 3. Map each sample to a ``CGMReading`` (timestamp must be tz-aware,
        #    value_mg_dl in milligrams per decilitre, source=self.name).
        # 4. Return the list sorted oldest-first.  Empty is acceptable when
        #    the vendor reports no samples; synthetic data is not.
        ...
```

Plug the custom adapter into the detector:

```python
endo = EndocrinologyDetector(AbbottLibreSource(account_token=...))
endo.fetch_and_detect()
```

The same contract applies to `VitalsDataSource`: implement
`fetch_recent_vitals(window_minutes)` and return tz-aware `VitalsReading`
snapshots. `VitalsReading.extra` is preserved verbatim for vendor-specific
diagnostics but is never used by the rule engine.

### Rules custom adapters must follow

1. **Real data only.** Never fabricate readings to fill gaps. Empty or
   sparse windows are permitted.
2. **Time-zone aware timestamps.** Datetimes must carry a `tzinfo`; the
   parsers assume UTC equivalence (use `astimezone(UTC)` if you receive
   local times).
3. **Raise typed errors.** Use `DataSourceError` for transient
   fetch failures and `ConfigurationError` for missing credentials. Do
   not swallow exceptions.
4. **No silent fallback.** If your adapter cannot fetch readings, raise —
   do not return synthetic placeholders or default values.
5. **Type-clean.** Type annotations and Google-style docstrings on the
   public surface; the platform's `mypy --strict` configuration must pass.

## Operational checklist

Before deploying to a real patient population:

- [ ] Credentials are provisioned via a vetted secret manager, not committed
      to source.
- [ ] Audit logging is enabled on the FHIR server / Dexcom developer
      account; you can correlate every Mercury Agent fetch to a server-side
      access log.
- [ ] The window sizes used (`fetch_and_detect`, `fetch_and_predict`) match
      your clinical protocol; the defaults (180 minutes for CGM, 5 minutes
      for vitals) are starting points, not policy.
- [ ] You have a clinical sign-off on the rule thresholds (FEV1 70 %,
      MAP 65–110 mmHg, HR 50–100 bpm, SpO₂ ≥ 92 %, EtCO₂ 30–45 mmHg,
      target BIS 50, BIS window 40–60).
- [ ] The PID infusion controller is **not** wired into any infusion pump
      without a separate clinical-trial validation and the appropriate
      regulatory approval (FDA 510(k) or equivalent).
- [ ] Output is reviewed by the responsible clinician; Mercury Agent never
      replaces a licensed practitioner.

## Provenance

Ported from the verified Omni-AXA-Engine implementation. Neural architectures
(CGM Bi-LSTM 155 K params, TIVA Bi-LSTM 164 K params), PID gains
(kp = 0.5, ki = 0.1, kd = 0.2), and all clinical thresholds match the
original module verbatim. The integration layer (data-source adapters,
ConfigurationError contract, `fetch_and_*` helpers) is new in Mercury Agent
v1.7 to support the platform's "integration-ready, not pre-integrated"
posture; the rule engine and inference paths are unchanged from the verified
source.
