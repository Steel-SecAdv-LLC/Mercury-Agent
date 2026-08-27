# GCN Kafka: MCP server + credential keep-alive

NASA's [General Coordinates Network](https://gcn.nasa.gov) (GCN) publishes
multimessenger alerts — gravitational-wave candidates, neutrino tracks, GRB
notices — on a Kafka broker at `kafka.gcn.nasa.gov:9092`. This document covers
wiring that broker to an MCP client and keeping the credential from expiring
underneath it.

The expiry is the part that bites. From GCN's FAQ:

> For security purposes, we will disable client credentials that you have not
> used to connect to a Kafka broker for the past 30 days.

An email warning arrives nine days before. A disabled credential **cannot be
re-enabled** — a new pair has to be issued and every consumer reconfigured. An
MCP server that is only attached when somebody happens to be working is exactly
the usage pattern that walks into that deadline, so this repository ships both
halves: the MCP configuration, and a scheduled keep-alive that connects whether
or not anyone is at a keyboard.

| Piece | Path |
| --- | --- |
| MCP server entry (project scope) | `.mcp.json` |
| Credentials file template | `configs/mcp/gcn-kafka.env.example` |
| Keep-alive script | `scripts/gcn_kafka_keepalive.py` |
| systemd user timer | `deploy/systemd/mercury-gcn-keepalive.{service,timer}` |
| Scheduled CI keep-alive | `.github/workflows/gcn-keepalive.yml` |

---

## 1. Issue a credential

Sign in at <https://gcn.nasa.gov/quickstart> and create a client credential.
GCN shows the **client secret once**, at creation. Copy it straight into your
secret store — not into a scratch file, a chat window, or a terminal you will
later screenshot.

## 2. Store it outside the repository

A secret pasted into a tracked file lands in git history, in every clone, and in
the transcript of any AI assistant that reads the working tree. Nothing in this
repository ever holds the value; three storage options are supported, in
descending order of preference.

**A secret manager**, exporting into the shell that launches your MCP client:

```bash
export GCN_KAFKA_CLIENT_ID="$(pass show gcn/kafka-client-id)"
export GCN_KAFKA_CLIENT_SECRET="$(pass show gcn/kafka-client-secret)"
claude
```

**An owner-only file**, for machines with no secret manager:

```bash
mkdir -p ~/.config/mercury
cp configs/mcp/gcn-kafka.env.example ~/.config/mercury/gcn-kafka.env
chmod 600 ~/.config/mercury/gcn-kafka.env
${EDITOR:-vi} ~/.config/mercury/gcn-kafka.env       # paste the real values
```

`scripts/gcn_kafka_keepalive.py` **refuses to read that file unless it is mode
600** — a credential readable by every account on the host defeats the point of
keeping it out of the checkout. To hand the same file to the MCP server, source
it into the launching shell: `set -a; . ~/.config/mercury/gcn-kafka.env; set +a`.

**Repository secrets**, for the scheduled CI keep-alive: add
`GCN_KAFKA_CLIENT_ID` and `GCN_KAFKA_CLIENT_SECRET` under *Settings → Secrets and
variables → Actions*.

`.gitignore` blocks `*.env` anywhere in the tree, so a copy dropped in the
working directory by mistake cannot be committed casually.

## 3. The MCP server entry

`.mcp.json` at the repository root configures a `gcn-kafka` server for every
Claude Code session opened in this project. Claude Code asks for approval the
first time it sees a project-scoped server.

```json
{
  "mcpServers": {
    "gcn-kafka": {
      "command": "uvx",
      "args": ["kafka-mcp-server"],
      "env": {
        "KAFKA_BOOTSTRAP_SERVERS": "${GCN_KAFKA_BOOTSTRAP_SERVERS:-kafka.gcn.nasa.gov:9092}",
        "KAFKA_SECURITY_PROTOCOL": "SASL_SSL",
        "KAFKA_SASL_MECHANISM": "OAUTHBEARER",
        "KAFKA_SASL_OAUTHBEARER_METHOD": "oidc",
        "KAFKA_SASL_OAUTHBEARER_CLIENT_ID": "${GCN_KAFKA_CLIENT_ID}",
        "KAFKA_SASL_OAUTHBEARER_CLIENT_SECRET": "${GCN_KAFKA_CLIENT_SECRET}",
        "KAFKA_SASL_OAUTHBEARER_TOKEN_ENDPOINT_URL": "${GCN_KAFKA_TOKEN_ENDPOINT:-https://auth.gcn.nasa.gov/oauth2/token}",
        "KAFKA_CLIENT_ID": "mercury-agent-mcp"
      }
    }
  }
}
```

Every credential reference is a `${VAR}` expansion resolved from the launching
shell's environment at startup. Claude Code expands `${VAR}` and
`${VAR:-default}` in `command`, `args`, `env`, `url`, and `headers`, so the file
carries variable *names* and no values. If a variable is unset, Claude Code
still loads the server and warns rather than substituting anything.

Prefer it configured for yourself rather than for the project? Use user scope,
which lives in `~/.claude.json` and applies across projects:

```bash
claude mcp add --scope user gcn-kafka \
  --env KAFKA_BOOTSTRAP_SERVERS=kafka.gcn.nasa.gov:9092 \
  --env KAFKA_SECURITY_PROTOCOL=SASL_SSL \
  --env KAFKA_SASL_MECHANISM=OAUTHBEARER \
  --env 'KAFKA_SASL_OAUTHBEARER_CLIENT_ID=${GCN_KAFKA_CLIENT_ID}' \
  --env 'KAFKA_SASL_OAUTHBEARER_CLIENT_SECRET=${GCN_KAFKA_CLIENT_SECRET}' \
  -- uvx kafka-mcp-server
```

Quote the `${...}` arguments in single quotes so the shell passes them through
literally — if the shell expands them first, the secret is written into
`~/.claude.json` in plaintext, which is the thing this setup exists to avoid.

> **Note on `~/.claude/settings.json`.** MCP servers are not configured there.
> `settings.json` holds permissions, hooks, and environment settings;
> `mcpServers` belongs in `.mcp.json` (project) or `~/.claude.json` (user/local
> scope, most easily managed with `claude mcp add`).

### What the MCP path does and does not give you today

Attaching an MCP server that authenticates against GCN counts as use and resets
the 30-day clock. That is the design intent above, and the configuration is
written for it.

Be aware of the gap: the published `kafka-mcp-server` builds on `kafka-python`
and takes its connection settings from a `kafka.properties` file.
`kafka-python` reaches SASL/OAUTHBEARER only through a token-provider object
supplied in code, so the OIDC client-credentials exchange GCN requires is not
reachable from configuration alone. Until an OAUTHBEARER-capable Kafka MCP
server is in use, treat the MCP connection as best-effort and the keep-alive
below as the mechanism that actually holds the credential open. The exact
librdkafka settings a compliant server needs are the ones
`scripts/gcn_kafka_keepalive.py::consumer_config` builds:

```python
{
    "bootstrap.servers": "kafka.gcn.nasa.gov:9092",
    "security.protocol": "SASL_SSL",
    "sasl.mechanisms": "OAUTHBEARER",
    "sasl.oauthbearer.method": "oidc",
    "sasl.oauthbearer.client.id": "<client id>",
    "sasl.oauthbearer.client.secret": "<client secret>",
    "sasl.oauthbearer.token.endpoint.url": "https://auth.gcn.nasa.gov/oauth2/token",
}
```

## 4. The keep-alive

```bash
pip install gcn-kafka                        # pulls confluent-kafka (librdkafka)
python scripts/gcn_kafka_keepalive.py
```

One authenticated metadata request against the broker — the cheapest exchange
that still makes GCN authenticate the client, which is what it counts as use.
The run is recorded in `~/.local/state/mercury/gcn-kafka-keepalive.json` so
"when did this credential last actually connect?" has an answer that is not a
guess.

| Verdict | Exit | Meaning |
| --- | --- | --- |
| `ALIVE` | 0 | Broker accepted the credential. Clock reset as of this run. |
| `FAIL` | 1 | Credential rejected — issue a new one and update the secret store. |
| `UNREACH` | 2 | Broker or network down. Says nothing about the credential; the clock was not reset. |
| `STALE` | 1 | `--check-only`: the last recorded success is older than `--max-age-days`. |
| `CONFIG` | 3 | Missing credentials, an unsafe credentials file, or `confluent-kafka` not installed. |

A failed broker probe is classified against the OIDC token endpoint before it is
reported: librdkafka's error text alone cannot reliably separate a disabled
credential from a network outage, and reporting an outage as a dead credential
would send someone to reissue a credential that was fine. A token exchange that
succeeds while the broker stays silent is reported as `UNREACH`, not `ALIVE` —
the token proves the credential, but GCN's clock is reset by a broker
connection.

Useful flags:

```bash
python scripts/gcn_kafka_keepalive.py --check-only     # read the ledger, no network
python scripts/gcn_kafka_keepalive.py --json           # machine-readable output
python scripts/gcn_kafka_keepalive.py --domain test.gcn.nasa.gov
python scripts/gcn_kafka_keepalive.py --max-age-days 7 # warn sooner
```

## 5. Schedule it

Daily, not monthly. A 30-day deadline met by a monthly job has no margin for a
laptop that was asleep or a broker that was down; daily gives thirty independent
chances.

**systemd user timer** (a laptop or workstation):

```bash
mkdir -p ~/.config/systemd/user
cp deploy/systemd/mercury-gcn-keepalive.* ~/.config/systemd/user/
${EDITOR:-vi} ~/.config/systemd/user/mercury-gcn-keepalive.service   # fix ExecStart
systemctl --user daemon-reload
systemctl --user enable --now mercury-gcn-keepalive.timer
loginctl enable-linger "$USER"          # keep the timer running while logged out
systemctl --user list-timers mercury-gcn-keepalive.timer
```

The unit sets `Persistent=true`, so a machine that was off at the scheduled time
runs the keep-alive at next boot — the 30-day clock does not pause while the
machine is off. `UNREACH` (exit 2) is accepted as success there so a transient
outage does not leave the unit in a failed state; a rejected credential and a
misconfiguration still fail loudly.

**cron**, where systemd is not available:

```cron
17 6 * * * /usr/bin/python3 "$HOME"/Mercury-Agent/scripts/gcn_kafka_keepalive.py --json >> "$HOME"/.local/state/mercury/gcn-keepalive.log 2>&1
```

**GitHub Actions** — `.github/workflows/gcn-keepalive.yml` runs daily at 06:41
UTC from a machine that is always on, which is the most reliable of the three.
It needs the two repository secrets from step 2; without them the job reports
that nothing is configured and passes, so forks are unaffected. Only a rejected
credential fails the job.

Belt and braces is fine: several schedules connecting the same credential cost
one metadata request each and remove the single point of failure.

## 6. Verify

```bash
python scripts/gcn_kafka_keepalive.py --json
python scripts/gcn_kafka_keepalive.py --check-only    # should report ~0 days
```

In an MCP session, ask the client to list Kafka topics; GCN topic names look
like `gcn.notices.icecube.lvk_nu_track_search`.

## What is never stored

* The client secret is read from the environment or an owner-only file and held
  in memory for the length of one run.
* The state ledger records a **truncated SHA-256 fingerprint** of the client id,
  never the id and never the secret, and is written mode 600.
* Both values are stripped from any error text before it is printed, so a broker
  error quoting the connection string cannot leak the credential into a CI log.
* `.mcp.json`, `configs/mcp/gcn-kafka.env.example`, and `.env.example` carry
  variable names only.

## References

* [GCN Kafka client setup](https://gcn.nasa.gov/docs/client)
* [GCN FAQ — credential expiry](https://gcn.nasa.gov/docs/faq)
* [`gcn-kafka` for Python](https://github.com/nasa-gcn/gcn-kafka-python)
* [Claude Code MCP configuration](https://code.claude.com/docs/en/mcp)
