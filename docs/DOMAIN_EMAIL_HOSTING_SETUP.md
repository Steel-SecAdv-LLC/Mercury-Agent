# Mercury Agent — Domain, Email & Hosting Setup Guide

**Audience:** the operator connecting `mercuryagent.global` (Wix domain),
Google Workspace email, and a Hetzner server so Mercury Agent has a public
website, working business email, and a place to actually run the application.

Last updated: 2026-07-23 (DKIM/DMARC verified live; deployment half handed
off to the platform runbook).

> This is an **infrastructure / operations** guide for the Wix/Google/DNS half
> (domains, DNS, email, VPS provisioning). For deploying the application
> itself, the canonical runbook is
> [`docs/PLATFORM_HARDENING.md`](PLATFORM_HARDENING.md) → **Deployment**
> (the `docker-compose.platform.yml` overlay + `deploy/Caddyfile`); general
> background lives in [`docs/DEPLOYMENT.md`](DEPLOYMENT.md) and
> [`docs/INSTALLATION.md`](INSTALLATION.md).

---

## 1. The big picture — what each vendor is for

Mercury Agent is a large **neuro-symbolic AI anomaly-detection engine**
(~406k LOC of Python, a REST API on port `8000`, Docker/Kubernetes
deployment, PyTorch-based ML in the optional `[ml]` extra, and a
post-quantum crypto substrate). It is **software that has to run on a real
computer with real CPU/RAM** — and it needs a public identity (a domain, a
website, an email address). No single vendor does all of that well, so the
stack is split across three, each doing the one thing it is good at:

```
                         mercuryagent.global  (one domain, DNS managed at Wix)
                                     |
        +----------------------------+----------------------------+
        |                            |                            |
   WIX  (identity)            GOOGLE  (communication)      HETZNER (compute)
   -------------------        --------------------------   -----------------------
   * Registers the domain     * Business email on          * A Linux cloud server
   * Hosts the public           @mercuryagent.global          (VPS) you fully control
     marketing website        * Gmail / Drive / Calendar    * Runs the actual Mercury
   * Manages the DNS zone      * Sends/receives mail           Agent app in Docker
     (all records live here)                                * Serves the API / dashboards
        |                            |                       * Reachable at a subdomain
   apex + www -> Wix site      MX/SPF/DKIM/DMARC -> Google     app.mercuryagent.global
```

**Why you need all three (and can't collapse them):**

- **Wix** is a website builder + domain registrar. It is excellent for a
  marketing/landing site and it *owns your DNS zone*, but it **cannot run
  arbitrary Python, Docker, or a long-lived API server**. Mercury Agent
  cannot live on Wix.
- **Google Workspace** gives you a professional mailbox at your own domain
  (`you@mercuryagent.global`) with the deliverability and tooling of Gmail.
  Wix can forward mail, but Workspace is the real, ownable mailbox.
- **Hetzner** gives you a cheap, powerful Linux VPS with root access — the
  only place in this stack where the Mercury Agent engine can actually
  execute. Wix/Google give you the *face*; Hetzner gives you the *body*.

Everything is stitched together with **DNS records, all managed in one place:
the Wix DNS panel.** That is the key mental model — Wix holds the zone, and you
point names within it at Google (for mail) and Hetzner (for the app).

---

## 2. Current state of `mercuryagent.global` (verified via live DNS)

Most of the plumbing is **already done**. As of this writing:

| Component | Status | What's live |
|---|---|---|
| Domain registered, DNS at Wix | ✅ Done | Nameservers `ns2.wixdns.net`, `ns3.wixdns.net` |
| Apex (`@`) + `www` point to Wix | ✅ Done | A records to Wix (`185.230.63.x`), `www` CNAME to wixdns |
| Public website **published** | ❌ **To do** | Root URL returns HTTP 404 — domain is connected but no site is published |
| Google Workspace MX (receive mail) | ✅ Done | Legacy 5-record `aspmx.l.google.com` set (valid & supported) |
| SPF (send authentication) | ✅ Done | `v=spf1 include:_spf.google.com ~all` |
| Google domain verification | ✅ Done | `google-site-verification=…` TXT present |
| **DKIM** (signs outgoing mail) | ✅ Done | `google._domainkey` live (2048-bit); Admin console shows "Authenticating email with DKIM" |
| **DMARC** (anti-spoofing policy) | ✅ Done | `_dmarc` TXT: `v=DMARC1; p=quarantine; rua=mailto:steel.sa.llc@mercuryagent.global` |
| **Hetzner** app/API endpoint | ❌ **To do** | No `app.`/`api.` subdomain; server not provisioned |

**Translation:** the domain and email — inbound *and* outbound authentication —
are fully done and verified end-to-end (Section 4.4). The two things left are
(1) publish the Wix site and (2) stand up Hetzner and point a subdomain at it.
Sections 3 and 5 cover exactly those.

> The legacy 5-record MX set works fine — do **not** rip it out. Google's newer
> setup uses a single record (`smtp.google.com`, priority `1`); it is an
> alternative, not a required upgrade. Only switch if you want to simplify.

---

## 3. Wix — domain & public website

DNS for this domain is managed **entirely in Wix**. You do not (and with a
Wix-registered domain generally cannot) move the nameservers elsewhere, so
every record — including the Google and Hetzner ones below — is added here.

### 3.1 Where the DNS panel is

1. Log in to Wix → **Account Settings → Domains** (or the **Domains** section
   of your dashboard).
2. Find `mercuryagent.global`, click the **Domain Actions (⋯)** icon.
3. Choose **Manage DNS Records**. This is the single screen where you add/edit
   **A, CNAME, MX, and TXT** records for the whole domain.

### 3.2 Publish the website (fixes the 404)

The domain currently resolves to Wix but shows a 404, which means **no
published site is attached**. To fix:

1. In Wix, open (or create) the **site** for this domain in the Wix Editor.
2. Design at minimum a landing page (what Mercury Agent is, contact email,
   links). Keep it simple to start.
3. Click **Publish**. In **Settings → Domains**, confirm the domain is set as
   the site's **primary/connected domain** (apex + `www`).
4. Re-check `https://mercuryagent.global/` — it should now serve the site.

> Keep the marketing site on Wix and the *application* on Hetzner (Section 5).
> Don't try to serve the app from the apex — use an `app.` subdomain so the two
> never collide.

### 3.3 Protect the asset

The domain was expensive — treat it like the asset it is:

- Turn on **domain auto-renew** in Wix so it can never lapse.
- Confirm the domain is **locked** (transfer lock on) at the registrar.
- Enable **two-factor authentication (2FA)** on the Wix account.

---

## 4. Google Workspace — business email

Email is **fully done**: inbound (MX + SPF + verification) and outbound
authentication (DKIM + DMARC) are live and verified end-to-end (Section 4.4).
The subsections below record what is deployed and how to repeat the setup
(e.g. for a DKIM key rotation).

### 4.1 DKIM (signs your outgoing mail) — ✅ **done**

**Deployed:** `google._domainkey` is live with a **2048-bit** key; the Google
Admin console shows the domain's status as **"Authenticating email with
DKIM."** For reference (or a future key rotation), the setup path was:

1. In the **Google Admin console** (`admin.google.com`), go to
   **Apps → Google Workspace → Gmail → Authenticate email**.
2. Select the domain `mercuryagent.global`, choose **2048-bit** key length,
   and click **Generate new record**.
3. Google shows a **host/name** (default selector `google`, i.e.
   `google._domainkey`) and a long **TXT value** (`v=DKIM1; k=rsa; p=…`).
4. In **Wix → Manage DNS Records → TXT**, click **+ Add Record**:
   - **Host Name:** `google._domainkey`
   - **Value:** the full `v=DKIM1; …` string from Google
   - Save.
5. Wait for propagation (minutes to a few hours), then return to the Google
   **Authenticate email** page and click **Start authentication**.

### 4.2 DMARC (tells the world what to do with fakes) — ✅ **done**

**Deployed** as a TXT record on `_dmarc` (via Wix → Manage DNS Records):

```
v=DMARC1; p=quarantine; rua=mailto:steel.sa.llc@mercuryagent.global
```

The policy went straight to `p=quarantine` because DKIM/SPF were verified
passing first (Section 4.4). (`p=none` is the softer monitor-only rollout
alternative if you ever need to re-stage a change.) Aggregate reports go to
the `rua` mailbox; once a few weeks of reports confirm all legitimate mail
authenticates, the policy can be tightened further to `p=reject`.

### 4.3 SPF — already correct

`v=spf1 include:_spf.google.com ~all` is present and correct. Only edit it if
you later add another service that sends mail as your domain (e.g. a
newsletter tool) — then add that provider's `include:` to the **same single**
SPF record. Never create a second SPF (`v=spf1`) TXT record; a domain may have
only one.

### 4.4 Verify email end-to-end — ✅ **done**

**Verified 2026-07-23:** a real message from the domain to a `gmail.com`
address showed, under **Show original**: **SPF: PASS**, **DKIM: PASS**
(`d=mercuryagent.global`), **DMARC: PASS**. To re-verify after any DNS or
mailbox change:

1. Sign in to `mail.google.com` as `you@mercuryagent.global`.
2. Send a message to an outside address (e.g. a personal Gmail) and reply back.
3. Open the received message → **Show original** and confirm
   **SPF: PASS**, **DKIM: PASS**, **DMARC: PASS**.

---

## 5. Hetzner — where Mercury Agent actually runs

This is the new piece. You'll provision a Linux server, deploy Mercury Agent in
Docker, and point a subdomain at it from Wix.

### 5.1 Pick a server size

Hetzner Cloud is **CPU-only** (no GPU on Cloud plans). Mercury Agent's default
detection path and REST API run fine on CPU; the full agent/`[ml]` path
(PyTorch, the agentic orchestration stack, the σ_Immutable gate) also runs on
CPU, just slower — and it needs the RAM. Note that the shipped **Docker image
installs `.[all]` — torch included** — so the container you deploy from the
platform runbook carries the complete agent stack regardless of plan size.

| Use case | Suggested plan | Why |
|---|---|---|
| API-only trial (no agentic/`[ml]` workloads) | **CX22** (2 vCPU, 4 GB, ~€4/mo) | Runs the API + reverse proxy; not enough RAM for the agent path |
| **Full agent / `[ml]` path (the real deployment)** | **CX32 / CX42 class** (4–8 vCPU, **≥ 8 GB RAM**) | PyTorch + orchestration + fusion need ≥ 8 GB |
| Serious model training | Hetzner **dedicated GPU** server (not Cloud) | Cloud has no GPU; training on CPU is slow |

For the actual launch, start at **CX32-class or larger (≥ 8 GB)**; treat CX22
as API-only trial territory. You can resize up later without rebuilding.

### 5.2 Provision the server

1. Create an account at **console.hetzner.cloud** (you said this is already
   approved) and make a new **Project**.
2. **Add Server** → Location near your users → Image **Ubuntu 24.04** →
   the plan from 5.1.
3. **SSH key:** on your own machine generate one, then paste the *public* key
   into the "SSH keys" box during creation:
   ```bash
   ssh-keygen -t ed25519 -C "mercury-agent"
   cat ~/.ssh/id_ed25519.pub      # paste this value into Hetzner
   ```
4. Create the server and copy its **public IPv4 address** (e.g. `203.0.113.45`).
5. Attach a **Hetzner Cloud Firewall**: allow inbound **22 (SSH)**,
   **80 (HTTP)**, **443 (HTTPS)**; deny everything else.

### 5.3 First login & Docker

```bash
ssh root@YOUR_SERVER_IP
apt update && apt upgrade -y
curl -fsSL https://get.docker.com | sh      # installs Docker Engine
apt install -y docker-compose-plugin git
docker --version && docker compose version   # sanity check
```

> Hardening (recommended, not optional for a public box): create a non-root
> sudo user, disable password SSH (`PasswordAuthentication no`), and enable
> `ufw`/the Hetzner firewall. See Hetzner's "Initial Server Setup" tutorial.

### 5.4 Deploy Mercury Agent — follow the platform runbook

The application-deployment half is deliberately **not duplicated here**. The
canonical, maintained runbook is
[`docs/PLATFORM_HARDENING.md`](PLATFORM_HARDENING.md) → **Deployment**
("Compose runbook"): it uses the shipped `docker-compose.platform.yml`
overlay, which adds the durable state volume, wires every documented
`MERCURY_*` variable, and includes the Caddy TLS edge (`deploy/Caddyfile`)
pre-configured for `app.mercuryagent.global`. In short, on the server:

```bash
git clone https://github.com/Steel-SecAdv-LLC/Mercury-Agent.git
cd Mercury-Agent
cp .env.example .env
python scripts/generate_secret_key.py --all >> .env   # secrets, no OpenSSL needed
docker compose -f docker-compose.yml -f docker-compose.platform.yml up -d
curl -fsS http://localhost:8000/health
```

Server-env checklist for the platform profile (details and defaults in the
runbook's configuration reference):

- Secrets from `generate_secret_key.py --all` reviewed into `.env`
- `MERCURY_FRONTEND_ENABLED=true` — serves the account UI (registration,
  login/2FA, dashboard) from the API process; the overlay defaults it on
- `MERCURY_SMTP_*` pointed at the Google mailbox from Section 4
- `MERCURY_PUBLIC_BASE_URL=https://app.mercuryagent.global` so email links
  resolve to the pages the frontend serves

For the `[ml]` / `[pqc]` extras and Python-native install, see
[`docs/INSTALLATION.md`](INSTALLATION.md); general Docker/K8s background is
in [`docs/DEPLOYMENT.md`](DEPLOYMENT.md).

### 5.5 Point a subdomain at the server (in Wix)

Back in **Wix → Manage DNS Records** (Section 3.1), add an **A record** so
`app.mercuryagent.global` resolves to your Hetzner box:

- **A (Host)** → **+ Add Record**
  - **Host Name:** `app`   (use `api` instead if you prefer `api.mercuryagent.global`)
  - **Value:** your Hetzner **IPv4** (e.g. `203.0.113.45`)
  - Save.

DNS can take up to ~48 hours to propagate (usually far less). The apex and
`www` stay pointed at Wix — only `app.` goes to Hetzner.

### 5.6 HTTPS — already handled by the platform overlay

Don't expose port `8000` raw. You don't need to install or configure a
reverse proxy by hand: the platform overlay from Section 5.4 **ships a Caddy
service** using the repo's [`deploy/Caddyfile`](../deploy/Caddyfile), which
terminates 80/443, auto-provisions and renews the Let's Encrypt certificate
for `app.mercuryagent.global`, and actively health-checks the app on
`/health`. The app runs with `MERCURY_TRUSTED_PROXY_HOPS=1` to match this
exactly-one-proxy topology.

Once the `app.` A record (5.5) has propagated,
`https://app.mercuryagent.global/health` should return the Mercury Agent
health check over HTTPS — nothing further to configure on the TLS side.

---

## 6. End-to-end checklist

- [ ] **Wix:** website designed and **Published**; `https://mercuryagent.global/`
      no longer 404s
- [ ] **Wix:** domain auto-renew ON, registrar lock ON, account 2FA ON
- [x] **Google:** DKIM generated (2048-bit) and `google._domainkey` TXT live;
      Admin console shows "Authenticating email with DKIM"
- [x] **Google:** `_dmarc` TXT live
      (`v=DMARC1; p=quarantine; rua=mailto:steel.sa.llc@mercuryagent.global`)
- [x] **Google:** test mail to gmail.com shows SPF / DKIM
      (`d=mercuryagent.global`) / DMARC = PASS (verified 2026-07-23)
- [ ] **Hetzner:** server provisioned (**≥ 8 GB / CX32-class** for the full
      agent path; CX22 only for an API-only trial), firewall 22/80/443,
      SSH key auth
- [ ] **Hetzner:** Docker installed; platform runbook followed
      (`docker compose -f docker-compose.yml -f docker-compose.platform.yml
      up -d`); `/health` OK locally
- [ ] **Hetzner:** secrets from `scripts/generate_secret_key.py --all` in
      `.env`; `MERCURY_FRONTEND_ENABLED=true`; `MERCURY_SMTP_*` +
      `MERCURY_PUBLIC_BASE_URL` set; 2FA on the Hetzner account
- [ ] **Wix:** `app` A record → Hetzner IPv4
- [ ] **Hetzner:** overlay's Caddy service serving
      `https://app.mercuryagent.global`

---

## 7. Learning resources

Official docs are the authoritative, always-current source; the videos are
useful visual walkthroughs but their content can drift — trust the docs when
they disagree.

**Wix (domain, DNS, website)**
- Managing DNS records: https://support.wix.com/en/article/adding-or-updating-mx-records-in-your-wix-account
- Connect a Wix domain to an external site/server: https://support.wix.com/en/article/connecting-a-wix-domain-to-an-external-site
- Improve mail deliverability (SPF/DKIM/DMARC): https://support.wix.com/en/article/increasing-your-mail-deliverability

**Google Workspace (email)**
- Set up MX records: https://support.google.com/a/answer/6149037
- Turn on DKIM: https://support.google.com/a/answer/174124
- Add a DMARC record: https://support.google.com/a/answer/2466580

**Hetzner (server + deployment)**
- Community tutorials index: https://community.hetzner.com/tutorials/
- Deploy an app with Docker: https://community.hetzner.com/tutorials/deploy-nodejs-with-docker/
- Caddy as a simple reverse proxy (auto-HTTPS): https://community.hetzner.com/tutorials/caddy-as-simple-reverse-proxy-and-file-server/
- DNS explained (Hetzner): https://www.hetzner.com/blog/dns-explained-how-domains-work-and-how-to-manage-them/
- Video — set up a Hetzner cloud server (short): https://www.youtube.com/watch?v=vrNitYC_qlg
- Video — Hetzner for beginners: https://www.youtube.com/watch?v=Y1Lu8NUBtrM

**Mercury Agent itself**
- [`docs/PLATFORM_HARDENING.md`](PLATFORM_HARDENING.md) — **the canonical
  deployment runbook** (compose overlay, Caddy TLS edge, every `MERCURY_*`
  variable), threat model, migration, acceptance checklist
- [`docs/DEPLOYMENT.md`](DEPLOYMENT.md) — Docker/K8s background, health checks
- [`docs/INSTALLATION.md`](INSTALLATION.md) — Python install, `[ml]`/`[pqc]` extras
- [`README.md`](../README.md) — what the system is and does
