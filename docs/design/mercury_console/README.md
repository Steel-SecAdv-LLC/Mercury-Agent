# Handoff: Mercury Console (browser front end for `mercury-agent serve`)

Checked against `Steel-SecAdv-LLC/Mercury-Agent` **main on 2026-10-30** (package `mercury-agent-global` 3.0.0, Python ≥ 3.11). Every repo path, test name and policy quoted below was read from that tree. If main has moved, re-verify the sections marked **(repo fact)** before relying on them.

---

## 1. How Claude Code gets this

Claude Code works in your local clone. It cannot see the design project, so you hand it this folder:

1. `git clone https://github.com/Steel-SecAdv-LLC/Mercury-Agent && cd Mercury-Agent && git checkout main && git pull`
2. Unzip this bundle into the repo root as `design_handoff_mercury_console/`. Add it to `.git/info/exclude` so it is never committed.
3. Make sure `gh auth status` shows you logged in (GitHub CLI). Claude Code uses it to open the PR. Without it, Claude Code pushes the branch and you open the PR in the browser.
4. Start Claude Code in the repo and paste the contents of `PROMPT.md`.

To see the design yourself, open `Mercury Console.html` in any browser. It is the whole prototype in one file and works offline, with no server.

---

## 2. Overview

The Mercury Console is a single-page browser front end for a running `mercury-agent serve`. It covers:

- **Terminal**: a command builder for all 62 `mercury-agent` CLI commands (options, examples from the cli.py docstrings, validation) with a prompt that stays in sync with the form, history, Tab completion, reverse search and a live output stream.
- **Agent**: the Mercury Agent conversation (`POST /api/v1/voice/speak`) plus agent capabilities (plan, analyse, delegate, research, verify).
- **Mass Run**: run many detection routes on one dataset and compare (consensus, agreement).
- **Results**: every server response with a summary, 30 chart types chosen per data shape, CSV/JSON/HTML report export.
- **Captures**: save sections of a result with provenance and SHA-256, tag them (ANOMALY/NORMAL/REVIEW/EVIDENCE), export labels files for `tier-detect -l` / `train -l`, record sessions (asciicast v2 + Markdown), and export an evidence pack (ZIP with `SHA256SUMS`).
- **Catalog**: commands, runs (benchmarks, scripts, examples, predictors), 113 manifest detectors and 6 energy library classes, 22 loader domains and 9 hazards, physics, operator tools, MCP tools, API routes.

---

## 3. About the design files

The files in `design/` are **design references built in HTML**: a working prototype that shows the intended look and behaviour. They are not production code to copy. `Mercury CLI.dc.html` runs on a design-tool runtime (`support.js`, React-based, uses `new Function`) that the repo's security policy would block, and it uses inline styles everywhere. Rebuild it in the repo's existing front-end style: **plain HTML + plain JavaScript + one stylesheet, no build step, no CDN** (the pattern of `api/frontend_assets/` today).

The framework-free modules (`design/mercury-cli-*.js`) hold most of the logic and can be ported nearly as-is, with the changes listed in §6.

To view the prototype, open `Mercury Console.html`: a packed single-file build of `design/` that works offline. Or serve the sources: `cd design && python -m http.server 8765`, then open `http://localhost:8765/Mercury%20CLI.dc.html`. Type `:diag` in the Terminal to run its 8,787-check self-test. The prototype includes an in-browser **simulated engine** so it works without a server; results from it are labelled SIMULATED.

## 4. Fidelity

**High fidelity.** Colours, type, spacing, copy and interactions are final. Every element's exact values are inline in `design/Mercury CLI.dc.html`; read them there. Recreate the layout and behaviour exactly, moving static styles into `console.css` classes.

---

## 5. Where it lands in the repo (repo fact)

| What | Where |
|---|---|
| Page | `src/omni_mercury_engine/api/frontend_assets/pages/console.html` |
| Scripts + stylesheet | `src/omni_mercury_engine/api/frontend_assets/static/console/` (`catalog.js`, `api.js`, `mass.js`, `viz.js`, `capture.js`, `agent.js`, `app.js`, `console.css`, `fonts/`) |
| Route | `api/frontend.py`: add `"/console": "console.html"` to `_PAGE_ROUTES` |
| Enabling | The frontend router is mounted only when `MERCURY_FRONTEND_ENABLED=true` (`frontend_enabled()`, default off). Keep that: `MERCURY_FRONTEND_ENABLED=true mercury-agent serve`, then open `http://localhost:8000/console`. Static files are served under `/static` by `register_frontend()`. |
| Packaging | `pyproject.toml` `[tool.setuptools.package-data]` lists only `api/frontend_assets/pages/*.html`, `static/*.css`, `static/*.js`, `static/vendor/*.js`. Add `static/console/*.js`, `static/console/*.css`, `static/console/fonts/*.woff2` and the font licence file, or the wheel will ship without them. |
| Catalog generator | `scripts/build_console_catalog.py` (new), writes `static/console/catalog.js` |
| Tests | update `tests/api/test_frontend.py`, add `tests/api/test_console_catalog.py`; `tests/api/test_security_headers.py` only if the CSP changes (§6.3) |

---

## 6. Hard constraints from the repo (repo fact)

### 6.1 Content-Security-Policy

`api/security_headers.py` serves HTML pages with `FRONTEND_CSP`:
`default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; font-src 'self'; connect-src 'self'; frame-ancestors 'none'; …`

What that means for the console:

- **No inline `<script>` or `<style>` blocks, no `style="…"` attributes, no `on…=` handler attributes** in the page. Attach listeners with `addEventListener`. Static styling goes in `console.css`.
- **Dynamic values** (live widths, percentages, per-chart colours) may be set from JS through the CSSOM (`el.style.width = …`, `el.style.setProperty(…)`), which CSP allows. `setAttribute("style", …)` and `innerHTML` containing `style="…"` are blocked.
- **`viz.js` has 56 `style="…"` attributes** inside the SVG strings it returns. Convert them to SVG presentation attributes (`fill`, `stroke`, `stroke-width`, `stroke-dasharray`, `font-size`, `opacity`) or classes, and move the root `display:block;max-width;font-family` to a `.mcli-chart` class.
- **Fonts:** `colors_and_type.css` `@import`s Google Fonts, which `font-src 'self'` and `style-src 'self'` block. Vendor IBM Plex Mono and IBM Plex Sans (`.woff2`, SIL OFL 1.1, include the licence) under `static/console/fonts/` with `@font-face` in `console.css`, or rely on the token's fallback stack (`ui-monospace, 'SF Mono', Menlo, Consolas, monospace`). Do not ship the `@import` lines.
- **`connect-src 'self'`:** the console may call only its own origin. In `api.js`, default the base URL to `location.origin` and drop the cross-origin base-URL field. Keep the optional `X-API-Key` header.
- **No `eval` / `new Function`.** None of the `mercury-cli-*.js` modules use them; keep it that way.
- **Images:** `mass.js` turns a hazard visualisation's `image_base64` into a `data:image/png;base64,…` URL for display. `img-src 'self'` blocks `data:` and `blob:`. Pick one and say which in the PR:
  - (a) **recommended:** add a path-scoped policy for `/console` that is `FRONTEND_CSP` with `img-src 'self' data:` (the same way `DOCS_CSP` is scoped to `DOCS_PATHS`), plus a test; or
  - (b) offer the PNG as a download only (Blob + `<a download>`), no on-page preview.

### 6.2 Tests that will fail unless you account for them

- `tests/api/test_frontend.py`
  - `test_pages_are_csp_compatible_and_self_contained`: page count must equal `len(_PAGE_ROUTES)`; no inline handlers or scripts; every `src`/`href` must start with `/` or `#` (use `/static/console/app.js`, never `./app.js`).
  - `test_page_serves_expected_markup`: add `"/console": 'id="mercury-console"'` to `_PAGE_MARKERS` and give the page root that id.
  - `test_static_assets_resolve`: add the console assets to its list.
  - `test_pages_have_accessible_status_regions`: add `console` to both loops; the page needs an `aria-live="polite"` region and `<label>` elements.
- `tests/api/test_security_headers.py::TestShippedAssetsSupportTheStrictPolicy`: no `<style>` blocks, no `style=` attributes, no `on(click|load|error|submit|change|input|focus|blur)=`, no cross-origin `src`/`href`.

### 6.3 House rules (CONTRIBUTING.md)

- New Python files start with the canonical header:
  `# Copyright (C) 2025 Steel Security Advisors LLC` / `# SPDX-License-Identifier: AGPL-3.0-or-later` / one-line docstring. Give the new HTML/CSS/JS files the same two lines in the comment syntax the existing frontend files use.
- Gates: `bash scripts/run_ci_gates.sh` (full matrix, what CI runs) or `--fast` while iterating; `black src/ tests/`, `ruff check src/ tests/`, `mypy src/omni_mercury_engine/`, `pytest tests/ -v`.
- Branch `feature/<name>`. Conventional commits (`feat`, `fix`, `security`, `docs`, `test`, `refactor`, `perf`, `chore`). Add a `CHANGELOG.md` entry (Keep a Changelog format).
- AI-assisted PRs must say so in the description and include tests and docs.
- **Never write the skip-CI directive literally** in the PR body or any commit message: squash-merge copies the body into the commit on main and GitHub Actions would skip every workflow.

---

## 7. Module plan (design file → repo file)

| Design file | Repo file | Changes |
|---|---|---|
| `mercury-cli-data.js` (catalog snapshot) | `static/console/catalog.js` | **Generate it** with `scripts/build_console_catalog.py` (§8 step 2). Never hand-edit. |
| `mercury-cli-api.js` (HTTP client + request schemas) | `static/console/api.js` | Same-origin only; derive request schemas from `app.openapi()` in the generator instead of the hand-written table; remove the WebSocket alerts URL (`api/voice.py` has no WebSocket route on main); remove simulated-engine hooks. |
| `mercury-cli-mass.js` (summaries, consensus, scripts) | `static/console/mass.js` | Image handling per §6.1. |
| `mercury-cli-viz.js` (30 chart types, SVG strings) | `static/console/viz.js` | Remove the 56 `style=` attributes (§6.1). |
| `mercury-cli-capture.js` (captures, SHA-256, CRC-32, ZIP, asciicast) | `static/console/capture.js` | Logic as-is. Its `barVals` / `panelVals` / `capsVals` return template values for the prototype; re-express them as DOM updates in `app.js`. |
| `mercury-cli-agent.js` (agent capabilities, Python snippets) | `static/console/agent.js` | As-is. |
| `Mercury CLI.dc.html` (template + `Component` logic class) | `pages/console.html` + `static/console/app.js` + `static/console/console.css` | Rebuild in plain JS. The logic class methods (`build`, `parse`, `fromInput`, `execute`, `runLive`, `items`, `refCard`, `optViews` construction in `renderVals`, …) port directly; the template becomes DOM built once plus targeted updates. |
| `mercury-cli-sim.js` (simulated engine) | — | **Do not ship.** The installed console always has the real server. |
| `mercury-cli-diag.js` (8,783-check self-test) | `static/console/diag.js` (optional) | Port the pure-function and wiring checks; replace the simulated-engine scenarios with live checks against the same-origin server. Lazy-load on `:diag`. |
| `support.js`, `_ds/…/_ds_bundle.js` | — | Design-tool runtime and design-system bundle. Not shipped (the console uses no bundle components). |

---

## 8. Implementation steps (in order)

1. `git checkout -b feature/web-console` from an up-to-date main.
2. **Catalog generator** `scripts/build_console_catalog.py` → `static/console/catalog.js` (`window.MERCURY_CLI = {…}`), built by introspection so the console cannot drift from the code:
   - **commands**: walk the click group `omni_mercury_engine.cli:main` (62 commands on main): path, help, options (flags, short flags, type, default, choices, required, is_flag, multiple, help) and the `mercury-agent …` example lines in each docstring (31 across 10 commands today).
   - **detectors**: `core/detector_registry.py` (113 entries, name, class, module, family/category, modality, seams) joined with `core/detector_ontology.py`.
   - **loaders**: `loaders/__init__.py` (22: airspace … wildfire, including `energy`). **hazards**: the 9 hazard routes in `api/routes/hazard.py`.
   - **MCP tools**: `mcp_server.py`. **API routes**: `app.routes` / `app.openapi()` (method, path, request schema).
   - **runs**: `benchmarks/*.py`, `scripts/*.py`, `examples/*.py` with their argparse flags (parse with `ast`; do not import them).
   - **voice vocabularies**: `voice --domain` = `cognitive.ethical_bounding._build_safe_domain_labels()` (EnvironmentDomain values + `general`); `voice --model` = `OllamaModel` values in `models/foundation/ollama_adapter.py`.
   - **library classes** (not in the registry): keep a short curated list in the script (the 6 energy classes, §13) and verify each with `importlib.util.find_spec` + an `ast` check that the class and method exist, without importing torch.
3. **Port the modules** per §7 into `static/console/`.
4. **Build `pages/console.html`**: header comment, `<link rel="stylesheet" href="/static/console/console.css">`, `<script src="/static/console/…" defer>` in dependency order (`catalog`, `api`, `mass`, `agent`, `viz`, `capture`, `app`), root `<div id="mercury-console">`, an `aria-live="polite"` status region, labelled inputs.
5. **Build `app.js` + `console.css`** from the template (§9–§11).
6. **Route + flag**: add `/console` to `_PAGE_ROUTES`. Optionally link it from `pages/index.html`.
7. **CSP** decision for hazard images (§6.1); implement and test it.
8. **Packaging**: extend `package-data`; build a wheel (`python -m build`) and confirm the console files are inside it.
9. **Tests** (§14). **Docs**: a short "Browser console" section in README/docs with the enable command; `CHANGELOG.md` entry.
10. **Gates**: `bash scripts/run_ci_gates.sh`; all green.
11. **Manual check**: `MERCURY_FRONTEND_ENABLED=true mercury-agent serve`, open `/console`, run the checklist in §15, check the browser console for CSP violations.
12. **PR** (§16).

---

## 9. Screens / views

### Shell
- Page: `height:100vh`, 16px padding, centred frame `max-width:1600px`, `--surface-1` fill, `1px solid --border`, `--r-md` radius, `--shadow-lg`.
- **Top bar** (`--surface-2`, 10px 16px padding, bottom border): ⌘K search button (centred, max 520px, `--void` fill: "Search commands, domains, detectors, tools, MCP, API"), run-state pill (`aria-live="polite"`, pulsing dot when live), engine connection pill, Mercury Agent chat toggle.
- **Left nav rail**, four groups:
  - WORK: Terminal `$`, Agent `@`, Mass Run `Σ`, Results `◧` (count), Captures `⊡` (count), Record session `●` (REC while recording).
  - CATALOG (shows "· N SELECTED" when items are ticked): Commands `›`, Runs `▸`, Detectors `◎`, Domains `◇`, Physics `φ`, Tools `⌗`.
  - INTERFACES: MCP tools `⇄`, API routes `{}` (account and model-registry routes hidden until enabled under Session).
  - SESSION: Session `⋯` (exports, script, settings), Diagnostics `√`, Engine `●`.
- **Right panel** (resizable): the Mercury Agent conversation, or a Reference card for the selected catalog item.
- **Command palette** (⌘K / Ctrl-K): fuzzy search across everything; Esc closes.

### Terminal
- Intro block under the banner (closable).
- Output stream: rows with a prefix, text and colour; long runs stream row by row; Ctrl-F find in output.
- **Prompt**: `$` + input ("type a command · Tab completes · ↑↓ history · Enter runs"). Ctrl-C cancel, Ctrl-L clear, Ctrl-U clear line, Ctrl-R reverse search.
- **OPTIONS panel** (collapsible, drag handle to resize; otherwise sizes to content up to `min(36vh,400px)`):
  - Bar: `▾ OPTIONS`, summary "N options · M set", RESET when anything is set.
  - EXAMPLES row (cli.py docstring examples) when the command has any; clicking one loads it.
  - One row per option: label, REQUIRED marker, badges, help text always visible on the left; the control on the right: choice chips, OFF/ON switch for flags, text input with vocabulary chips (`voice --domain`, `voice --model`) or "e.g." suggestions for free-text flags, inline warning when a value will be collapsed (e.g. `voice --domain energy`, §13).
  - CONSOLE ONLY settings (CONTEXT domain, SAMPLES) are grouped and hidden where they don't apply.
- Required options sort first; a left keyline shows state (amber = required and empty, cyan = set).

### Agent
Capability cards grouped "Plan, analyse & delegate" and others from `agent.js`, each with its inputs, the Python it runs, and copy/queue actions. Domain picker uses the agent's `DomainType` (general, medical, security, energy, infrastructure, humanitarian, scientific, financial).

### Mercury Agent conversation (right panel)
Opened by `mercury-agent voice`, the chat toggle, or Converse. Shows the mode line ("live · /api/v1/voice · domain medical" / "offline · voice --offline · local replies"), messages, the steps the agent took, and a composer. Questions typed in the Terminal are offered to the agent.

### Mass Run
Choose a dataset and a set of capabilities; run them; per-route status; consensus/agreement summary; generated bash script for the same runs.

### Results
List of responses (newest first); detail with summary key-values, flagged samples, a chart picker ("suggested" set per data shape, show all/none), HTML report (view/download), CSV and JSON downloads, raw response. **Capture panel** under the charts: a selectable strip over all samples (drag to select, flagged samples marked), from/to fields, WHOLE / FLAGGED SPAN / NEXT EVENT, tag, label, padding ±0/2/5/10, CAPTURE and CAPTURE ALL FLAGGED EVENTS.

### Captures
Tabs CAPTURES · n / RECORDINGS · n; list + detail. Capture detail: editable title, tag, note, provenance (samples, flagged, score, source, route, received, dataset, captured, engine), per-column statistics, suggested charts, the rows (first 300), SHA-256 of the CSV, CSV/JSON/Markdown export, labels file actions (download, load `tier-detect` / `train` with `--labels`), delete with confirm. Recording detail: counts, timeline (first 500 events), exports `.cast`, `.md`, `.json`, replay `.sh`. EVIDENCE PACK (.zip) button.

### Catalog views
Filterable lists grouped by family/category; selecting an item opens its Reference card (module, modality, ontology, validation status, how to run it: CLI form, API route, MCP tool, or Python).

---

## 10. Interactions & behaviour

- **Two-way sync**: editing options rebuilds the prompt (`build`); typing in the prompt re-parses into the options (`parse` / `fromInput`), including unknown-option errors in click's wording ("No such option: …", "Invalid value for '--x': 'v' is not one of …").
- **Running**: connected → the command's API route is called and the response becomes a Result; not connected → the command is prepared for you to copy and run in a shell. Long-running or writing commands (`serve`, `mcp`, `stream`, `train`, `tune`, `build-features`, anything with `-o`) are never auto-run from chat.
- **Agent chat**: sends `{message, domain, session_id}` to `POST /api/v1/voice/speak`. The domain is the one given to `voice --domain` (free text, as cli.py allows), otherwise the console CONTEXT; the mode line shows what is sent and warns when Mercury will collapse it to `general`.
- **Captures**: drag on the strip to select; a capture freezes rows, scores and flags with a SHA-256 of its CSV; labels file = 1 for samples inside ANOMALY captures, 0 elsewhere, one row per dataset sample (cli.py checks the count).
- **Recording**: `:rec [start|stop|discard]`, `:mark <note>`, `:capture <from> [to] [tag] [label]`; stops automatically (saved) when the page unloads.
- **Hover/active**: surfaces step up one rung (`--surface-2` → `--surface-3`), borders `--border` → `--border-strong`; selected rows get a 2px cyan left keyline.
- **Motion**: blinking block cursor, pulsing live dot; honour `prefers-reduced-motion`.
- **Loading/error states**: run pill shows RUNNING; errors render as amber/red rows with the server's message; empty states explain what to do next.
- **Narrow widths** (< 1180px): the right panel overlays instead of docking.

## 11. State & persistence

localStorage keys used by the prototype (keep the names so users keep their data):
- `mercury_cli_state_v1`: history, option values, selected command, CONTEXT, catalog selection, results (last 30), capture tag/padding.
- `mercury_cli_capture_v1`: captures (up to 200) and recordings (up to 10); trims on quota errors.
- `mercury_cli_conn_v1`: connection settings (API key only if "remember" is ticked; otherwise sessionStorage).
- `mercury_cli_mcp_order`: MCP list order.

## 12. Server routes the console calls

Detection: `POST /api/v1/detect/{univariate,multivariate,neurosymbolic,fusion,three-r,tier,flagship,rca}`, `POST /api/v1/batch/detect` (+ job status/results), `POST /api/v1/hazard/visualize`. Conversation: `POST /api/v1/voice/speak`, `POST /api/v1/voice/narrate`, plus the voice status/greeting GETs. Assessment: `statements/verify`, `intel/screen`, `intel/fuse`, `concealed-information-test`, `vital-signs`, `adjudication/*`. The API routes catalog lists every route on the server; account and model-registry routes are hidden by default because the console workflow does not need them.

---

## 13. Energy domain: what exists and the one gap

Energy is covered:
- Loader `energy` (`loaders/energy_loader.py`, `EnergyLoader`) in CONTEXT and the Domains catalog.
- Registry detector `emp` (family Energy, wraps `detectors/energy/emp_detector.py:EMPDetector`).
- Benchmark `benchmarks/energy_honest_benchmark.py` in Runs.
- Agent planning domain `DomainType.ENERGY` (`agentic/mercury_a_agent.py`).
- CAP alerts: `alerting/cap_generator.py` maps `energy` to category INFRA, headline "Power Grid Emergency".
- **Library classes, now in the Detectors catalog under Energy with runnable Python:** `EMPDetector.predict_emp`, `E1PulseDetector.detect_e1_pulse`, `E2PulseDetector.detect_e2_pulse`, `E3PulseDetector.detect_e3_pulse` and `IntentionalEMIDetector` (torch) are reachable through the registry detector `emp` (EMPRecordAdapter wraps EMPDetector, which runs the pulse detectors and, once weights are loaded, the IEMI network); their cards link to it. `infrastructure/energy_dams.py:EnergyDamsDetector` has no registry path and is Python-only. (`energy/energy_optimization.py` is about compute power budgets, not the energy sector, so it is not listed.)

The gap is in the repo, not the console: **`voice --domain energy` becomes `general`.** `narrative/voice.py` passes the domain through `sanitize_domain` (`cognitive/ethical_bounding.py`), whose whitelist is `EnvironmentDomain` (`cognitive/ipb_engine.py`: cyber, physical, cognitive, financial, social, infrastructure, medical, environmental, space) plus `general`. There is no `ENERGY` member. The console warns and offers `infrastructure` as the closest label.

If you want energy as a conversation domain, that is a **separate PR**: add `ENERGY = "energy"` to `EnvironmentDomain`, then check every consumer of `EnvironmentDomain` (per-domain tables in the IPB engine, anything iterating the enum) and the sanitize tests. Do not bundle it into the console PR.

## 14. Tests to add / update

- `tests/api/test_frontend.py`: `_PAGE_MARKERS["/console"]`, console assets in `test_static_assets_resolve`, `console` in the accessibility loops.
- `tests/api/test_console_catalog.py` (new):
  - the committed `catalog.js` equals a fresh generator run (staleness gate);
  - every command and option in the catalog exists in the click group with the same flags, type, default and choices, and vice versa;
  - every detector name is in the registry; every library class and method exists (`find_spec` + `ast`, no torch import);
  - every `/api/v1/…` path referenced in `static/console/*.js` exists in `app.openapi()["paths"]`;
  - the `voice --domain` vocabulary equals `_build_safe_domain_labels()`.
- `tests/api/test_security_headers.py`: only if you add the `/console` image policy; assert `/console` gets it and every other page still gets `FRONTEND_CSP`.

## 15. Manual test checklist

Run with `MERCURY_FRONTEND_ENABLED=true mercury-agent serve`, open `/console`, keep DevTools open (zero CSP violations, zero errors):
1. `detect` with a CSV: result appears, charts render, CSV/JSON/report download.
2. `voice --domain medical`, send a message: the mode line says "domain medical"; `voice --domain energy` shows the warning and USE INFRASTRUCTURE.
3. Mass Run with three routes: consensus shown, script downloads.
4. Capture a flagged event, tag ANOMALY, download the labels file, load `tier-detect … --labels …`.
5. Record a short session, stop, export `.cast` and the evidence pack; `sha256sum -c SHA256SUMS` passes.
6. Hazard visualisation: PNG shows (option a) or downloads (option b).
7. Reload: history, options, results and captures persist.

## 16. Pull request

- Branch `feature/web-console`. Suggested commits: `feat(api): add browser console page served by the frontend router`, `feat(scripts): generate the console catalog from the CLI, registry and OpenAPI`, `test(api): cover the console page, assets and catalog parity`, `docs: document the browser console`.
- `gh pr create --base main --head feature/web-console --title "feat(api): add Mercury browser console" --body-file <file>`, using `.github/PULL_REQUEST_TEMPLATE.md` sections (Description, Type of Change: New feature, Changes Made, Testing with commands and results, Documentation, Code Quality, Security Considerations: the CSP decision, Ethical Considerations, Screenshots).
- State that the PR is AI-assisted (Claude Code). Do not write the skip-CI directive anywhere.

---

## 17. Design tokens

From `design/_ds/…/colors_and_type.css` (dark default; the file also has `html.ds-light` and `html.ds-contrast` overrides the console does not use). Port the ones the console uses into `console.css` `:root`.

- Surfaces: `--void #06080B`, `--bg #0A0E14`, `--surface-1 #0F141C`, `--surface-2 #151C26`, `--surface-3 #1C2530`, `--surface-4 #243140`
- Borders: `--border-faint #161E28`, `--border #243040`, `--border-strong #36465A`
- Text: `--fg #E6EDF3`, `--fg-2 #A9B6C6`, `--fg-3 #6B7888`, `--fg-muted #495567`, `--fg-faint #313C4B`
- Mercury cyan: `--mercury-50 #E3FAFF`, `-100 #BDF0FA`, `-200 #8AE3F5`, `-300 #4FC3F7`, `-400 #18C8E0`, `-500 #00BCD4`, `-600 #00A0B6`, `-700 #007E92`, `-800 #0A5D6B`, `-900 #0C3B44`, `--mercury-glow rgba(0,188,212,.35)`
- Status: `--anomaly #FF4D52` (soft `#FF7A7E`, deep `#B3231F`, bg `rgba(255,77,82,.10)`), `--normal #2FD686` (bright `#4ADE80`, deep `#137A4E`, bg `rgba(47,214,134,.10)`), `--warning #F5A623` (bright `#F8C46B`, deep `#9A6310`, bg `rgba(245,166,35,.12)`), `--cognitive #8B7FE8`, `--benevolence #E0B341`, `--verified #3E8EF7`, `--withheld #AAB4C0`
- Brand (wordmark only): `--brand-gold #B4B124`, `--brand-cyan #11AEED`, `--brand-ink #14171B`
- Type: `--font-mono 'IBM Plex Mono', ui-monospace, 'SF Mono', 'JetBrains Mono', Menlo, Consolas, monospace`; `--font-sans 'IBM Plex Sans', ui-sans-serif, system-ui, …`. Sizes used: 9–12.5px mono for chrome and data, 11.5–13px sans for help text; uppercase labels with `.12–.16em` tracking.
- Radii: `--r-xs 2px`, `--r-sm 4px`, `--r-md 6px`, `--r-lg 10px`, `--r-pill 999px`
- Spacing: 4px grid (`--sp-1 4px` … `--sp-24 96px`)
- Shadows: `--shadow-sm 0 1px 2px rgba(0,0,0,.4)`, `--shadow-md 0 4px 14px rgba(0,0,0,.45)`, `--shadow-lg 0 18px 50px rgba(0,0,0,.55)`, `--glow-mercury 0 0 0 1px rgba(0,188,212,.5), 0 0 18px var(--mercury-glow)`
- Motion: `--dur-1 90ms`, `--dur-2 140ms`, `--dur-3 240ms`, `--ease-standard cubic-bezier(0.4,0,0.2,1)`; keyframes `mcliBlink` (cursor) and `mcliPulse` (live dot) are in the template's helmet `<style>`.

## 18. Assets

No images. Icons are Unicode glyphs (`$ @ Σ ◧ ⊡ ● › ▸ ◎ ◇ φ ⌗ ⇄ {} ⋯ √ ⌕`) and the ♱ sigil (title, description and the Mercury Agent conversation only). Fonts: IBM Plex Mono / Sans (OFL 1.1) to vendor, see §6.1.

## 19. Files in this bundle

- `README.md`: this document.
- `PROMPT.md`: the instruction to paste into Claude Code.
- `Mercury Console.html`: the whole prototype packed into one file for viewing (works offline). It is not readable source; read `design/` instead.
- `design/Mercury CLI.dc.html`: the prototype (template + logic class).
- `design/mercury-cli-data.js`, `-api.js`, `-mass.js`, `-viz.js`, `-capture.js`, `-agent.js`: logic modules to port.
- `design/mercury-cli-sim.js`, `-diag.js`: simulated engine and self-test (reference only).
- `design/support.js`, `design/_ds/…/colors_and_type.css`, `design/_ds/…/_ds_bundle.js`: needed only to open the prototype locally.

## 20. Changes since this handoff (build notes)

Added after the original handoff, all in `design/`. Nothing here changes how detection works.

- **Presence figure** (`mercury-cli-figure.js`, `mercury-cli-figure-trace.js`): canvas 2D, framework-free, no `eval`, no network. The trace file is data traced from the supplied reference image (seed points, stars, thread curves). It listens to the `mcli:call` CustomEvent that `mercury-cli-api.js` dispatches around every request (`phase: "start" | "end"`, `method`, `path`, `ok`, `status`, `data`), so its activity, signal and pulses reflect real requests. `window.MercuryPresence.state()` exposes status, activity, in-flight count, signal, pulses, frame cost and an error counter. Reduced motion draws one still frame.
- **Settings dialog**: opened from the ⚙ button; six controls (theme, accent, high contrast, reduce motion, focus outlines, scale), each backed by the design system's `window.MercurySettings` and persisted to `localStorage` key `mercury-settings`. Escape closes it.
- **Opening screen**: the progress bar streams continuously, including under reduced motion (`.mcli-progress` override). The OPTIONS drawer is collapsed on entry.
- **Sigil**: the cross is the ♱ glyph everywhere; † does not appear.
- **Assets**: `omega-cross.png` and `ama-monogram.png` are used (the "No images" statement in §18 predates them). Two `<img src="{{ … }}">` bindings inside `<sc-if>` make the browser request the literal placeholder URL once at load (a harmless 404 in the console); the images only render when bound.
- **Other additions**: world map (needs `lat`/`lon` columns), stacked histogram, fusion network, batch lifecycle strip, event dossier, agent prompts, rate-limit budget and 429 countdown, correlation ids. The self-test is now 8,739 checks (`:diag`).
- **Engine side** (`mercury-agent serve`): CORS now exposes `X-Correlation-ID` and `Retry-After`, and `POST /api/v1/voice/speak` returns the voice's full reply (intent, sources, reasoning summary, follow-ups, uncertainty note) instead of message and confidence only.
- **Porting**: the new template markup uses inline `style=` like the rest of the prototype, so it falls under the §6.1 clean-up. `mercury-cli-figure*.js` can be ported as-is.
- **Stale file**: `Mercury Console.html` (the packed single file) is the original handoff pack and does **not** include any of the above. Use `design/` as the source of truth.
