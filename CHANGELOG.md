# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Reproducibility note (applies to all 1.x release entries, now in
> [`docs/changelog/CHANGELOG-pre-3.0.md`](docs/changelog/CHANGELOG-pre-3.0.md)):**
> Headline benchmark numbers vary by release cut; the current committed
> run is **66 reproducible datasets** (of 75 attempted), surfaced in the
> README "Latest Benchmark Results" block. 9 datasets currently
> fail to load due to unavailable external sources (SMAP, MSL,
> CICIDS-2017, MIT-BIH, UCR, SWaT, WADI, USGS Geochemistry,
> FEMA HazardMitigation). As of v1.7.0
> the previously-flagged "FEMA Disaster — inverted scores" loader
> is no longer in the broken set; the label-polarity correction is
> documented under `[1.7.0]` in that archive and locked by
> `tests/datasets/test_disaster.py::TestFEMAInvertedScoresCorrection`.
> The 11 watch-listed loaders now have a two-lane reachability harness
> (`tests/datasets/test_unreachable_loaders_{offline,network}.py`,
> plus the nightly `.github/workflows/dataset-reachability.yml`
> workflow) so an upstream provider outage surfaces as a failed
> nightly run rather than as a benchmark silently dropping a
> dataset.  See the README "Empirical Benchmark Results" section for
> the full reproducibility footnote and `docs/ROADMAP.md` for
> tracked fixes.

## [Unreleased]

**Fixed — the API server started without the AMA backend, reported `healthy`, and served sign-in pages with no account API behind them.** The PQC gate runs at import of `omni_mercury_engine.engine`, and the docs said the API was covered because it serves the engine. `api/server.py` reaches the engine only inside request handlers, so on an install without AMA (`pip install "mercury-agent-global[api]"` and nothing else) `mercury-agent serve` started: `/health` answered `{"status":"healthy"}`, the account router was dropped on an `ImportError` with one warning line (56 documented paths instead of 76; all 20 of `/api/v1/auth/*` gone), `/login` and `/dashboard` still loaded with `MERCURY_FRONTEND_ENABLED=true`, the startup maintenance sweep logged a traceback, and the first request that touched the engine returned 500. Measured on a clean `[api]`-only environment before the fix. No CI lane could see it: every lane builds AMA before its first import. The server module now runs the gate itself at import, `mercury-agent serve` checks before it prints a banner and exits 1 with the build steps, and a server with the frontend enabled refuses to start if the account router did not load. Pinned by `tests/test_pqc_startup_gate.py::TestScopedGateContract` (two new cases, with the same stubbed-AMA subprocess as the engine and crypto cases) and `tests/api/test_router_manifest.py::TestDegradedBoot`.

**Added — a router manifest, exported on `/metrics`, and an alert on it.** Each of the server's 14 surfaces is attached through `api/router_manifest.py::mounting`, which records `mounted`, `disabled` (the frontend and the decoy surface are opt-in) or `unavailable` (the import failed). `/metrics` carries the result as `omni_mercury_api_router_state{router,state}`, valued 1 on the router's current state; the failure reason stays in the log and is not a label, because it can hold a path. `/health` is unchanged: it is pinned to `status` and `version` only (`tests/api/test_health_minimal.py`). `OmniMercuryApiRouterUnavailable` (warning, 2 minutes) is added to the PrometheusRule CRD and rendered into `monitoring/prometheus/rules.yml` and the Helm template, with `docs/runbooks/api-router-unavailable.md`. An exception other than `ImportError` while attaching a router still stops the process.

**Added — a boot smoke test on a real install, run where CI briefly is one.** `scripts/api_boot_smoke.py --expect refuse` runs in Core Tests between `uv sync` and the AMA build, the one moment a lane looks like a bare `[api]` install: importing the server and running `serve` must both refuse, and `serve` must refuse before its banner. It exits 2 if AMA is already importable, so moving the build earlier fails the step instead of emptying it. `--expect serve` runs after the build: uvicorn with the frontend on, `/health`, the account API in the OpenAPI document, the frontend pages, no router `unavailable`, no traceback in the startup log. All four combinations of environment and mode were run locally (`[api]`-only: refuse 0, serve 2; with AMA: refuse 2, serve 0).

**Fixed — Dependabot's weekly uv run could not resolve the lock, and the replay that exists to catch that passed the day before.** All seven updates of 2026-11-09 failed with `No solution found when resolving dependencies for split (markers: python_full_version >= '3.15' and sys_platform == 'emscripten')`. `[tool.uv] environments` listed `sys_platform != 'darwin'`, which admits emscripten; httpx2 carries an emscripten-only dependency (`httpx2-jsfetch`), so the lock had an emscripten fork, and torch ships no wheel there. uv 0.11.32, which CI and the replay use, tolerated that fork; Dependabot bundles its own uv (0.13.7 that day), which does not. Reproduced with 0.13.7 on the unchanged files; with emscripten excluded the relock drops `httpx2-jsfetch` and nothing else, and 0.13.7 then takes `pandas 3.1.0 -> 3.1.1`. `scripts/simulate_dependabot_uv.py` now replays every update under both resolvers (the second fetched with `uv tool run --from uv==0.13.7`), after a baseline relock under each, and against the old files it fails with the same error Dependabot reported. `DEPENDABOT_UV_VERSION` is a pin that has to be refreshed from a Dependabot run log when Dependabot moves.

**Fixed — the rolling benchmark pull request could never pass its required checks.** `benchmark.yml` moves `ci/benchmark-results` twice per run and dispatches `ci.yml` by hand, because commits made with the default token start no workflows. The dispatch sat between the two updates, so CI ran on the first commit and the second replaced the head with one that had no check runs (run of 2026-11-08: checks on `42952ab`, head `408a6fa` 108 seconds later). The dispatch is now the last step, runs when either persist step moved the branch, and keeps `always()` so a failed seven-axis runner does not cancel it. Pinned by `tests/scripts/test_benchmark_pr_dispatch_order.py`. Not fixed here: that pull request's refreshed results say 66 of 76 datasets while `docs/BENCHMARKS.md` says 67, so its own Workflow Hardening run fails; whether the headline follows a provider outage is a maintainer decision.

**Added — a skip ledger: a gate on tests that can never run.** A skipped test reports as a pass. `scripts/measure_skip_ledger.py` reads `tests/` with `ast` and `uv.lock` for what the full test lane (`--extra all --extra dev`) installs, and fails on a test behind `importorskip()` of a dependency no extra provides, on a computed `importorskip` name, on an unconditional skip and on an expected failure, unless the script declares it with a reason. On `main` it found one: `gymnasium` guarded `tests/ml/test_ppo_trainer.py::test_gymnasium_vector_environments`, no extra carried it, and the PPO trainer's Gymnasium vector-environment path had never run in CI. `gymnasium>=1.4.0` is now in `[dev]` (test-only; not a runtime dependency) and the test passes, as do the other 221 in the two RL test files with it installed. One exception is declared: `ama_cryptography`, built from source in every lane. No test in the tree is unconditionally skipped or marked `xfail`. The result is `docs/SKIP_LEDGER.md`, gated in Workflow Hardening and pre-commit; volatile counts are in the script's plain report and deliberately not in the gated block. The 39 `raise NotImplementedError` in `src/` were read at the same time: 21 are abstract hooks or backends that refuse by design, 18 are refusals of an unsupported configuration; none is unfinished work.

**Fixed — three Permissions-Policy names no browser enforced.** `ambient-light-sensor`, `battery` and `document-domain` produced an `Unrecognized feature` console error on every response in Chromium and denied nothing. They are removed, and ten features that Chromium does enforce and the frontend does not use are denied instead (`browsing-topics`, `captured-surface-control`, `compute-pressure`, `hid`, `identity-credentials-get`, `idle-detection`, `local-fonts`, `otp-credentials`, `publickey-credentials-create`, `window-management`). The 28-name set was loaded as a response header in Chromium 153.0.8010.12 and produced no console message; `bluetooth` was tried and is not recognised by that build. The set is pinned in `tests/api/test_security_headers.py`.

**Fixed — the console reported a server it had not yet asked as down.** The first `GET /health` leaves 1.2 s after mount; until it answered, a console served by a running engine showed `no server · browser preview` on the boot screen and `SERVER DOWN` in the header. A `pending()` state now covers that interval (`checking engine…`), and the probe's answer, either way, ends it. Four named patches in `scripts/build_console_port.py`; `logic.js` regenerated.

**Fixed — `mercury-agent serve` did not list `/console` or `/metrics`.** The banner now lists `/metrics` always and the browser pages only when `MERCURY_FRONTEND_ENABLED=true` mounts them.

**Changed — releases before 3.0.0 moved to `docs/changelog/CHANGELOG-pre-3.0.md`.** The eleven releases from 0.1.0 to 2.0.0 (4,103 lines, 243 KB) were moved unchanged; the comparison is byte-exact. This file keeps 3.0.0 onward and an index of the archived releases. `scripts/doc_lint.py` and `workflow_version_drift_gate` exempt the archive as they exempt this file, `algorithm_name_drift_gate` still reads it, and the prose pointers that named a moved entry now name the archive.

## [3.0.0] - 2026-11-06

The first release after 2.0.0 (the work prepared as 2.1.0 was never released);
the upgrade notes from 2.0.0 — the AGPL-3.0-or-later relicense and every
renamed or removed 2.0.0 name — are summarised where this section's earlier
entries introduce them and in DEPRECATION.md §6.6-§6.15 and §7.

**Security — seven container CVEs fixed in place instead of accepted, and one inert acceptance dropped.** Trivy alerts on `main` for CVE-2026-85091 (zlib `gz_vacate()` heap overflow), CVE-2026-119379 (SQLite session-changeset heap overflow) and CVE-2026-119374 (corrupted-FTS5-index heap overflow) had been closed by `.trivyignore` acceptances. None has a Debian trixie fix, but Debian unstable fixes all three, and the same builds fix four older acceptances: CVE-2026-76844 (zlib), CVE-2026-119373 (SQLite FTS5), CVE-2026-54369 (acl) and CVE-2026-16742 (systemd). Each fixed package — `zlib1g 1:1.3.dfsg+really1.3.2-4`, `libsqlite3-0 3.54.1-2`, `libacl1 2.4.0-1`, `libsystemd0`/`libudev1 262.2-1` — depends on `libc6` alone and keeps its soname. The runtime stage now installs them from `snapshot.debian.org` at one pinned timestamp, signature-verified, at APT priority 1, with the source deleted in the same layer, and the build proves each version. Measured on the runtime layer built from this Dockerfile: 5 upgraded, 0 newly installed; `dpkg --audit` clean; apt, `useradd`, and CPython's `sqlite3` (FTS5 included), `zlib`/`gzip`, `uuid`, `ssl`, `readline` and `curses` all work.

Trivy keys Debian findings on release 13, whose advisory carries no fixed version, so it still reports the seven against the fixed builds. `trivy-fixed-in-place.rego`, passed to both blocking gates as `ignore-policy`, ignores each one only at the exact fixed version. Measured with the CI-pinned trivy 0.70.0: the fixed layer scans to 0 CRITICAL/HIGH findings with the ledger and the policy (exit 0), and the unmodified trixie layer, scanned with the same two files, fails with all eight package findings (exit 1), so a reverted swap reopens the gate. CVE-2026-119586 (the `sqlite3` shell's `.recover`) was dropped from the ledger because the current DB reports it against no package in the image, so the entry suppressed nothing; the shell stays uninstalled. The ledger goes from 13 to 5 entries (ncurses ×1, util-linux ×4), whose fixed builds need `libc6 >= 2.42` against trixie's 2.41. `tests/security/test_fixed_in_place_os_packages.py` replaces the two acceptance-condition modules, whose premise no longer holds. It pins the Dockerfile/policy version identity, the version-bound ignore rule, ledger/policy disjointness, the snapshot pin, signature and removal, the build-time re-proof, both gates' wiring and the shell's absence. Each of eight deliberate breakages fails exactly its own test. *Correction:* the 2026-11-05 entry concluded that installing sid's fixed zlib was "ineffective". The measurement behind it was right: Trivy still reports the CVE. The conclusion was not, because the fixed build does remove the vulnerability.

**Fixed — the release workflow could not have released v3.0.0, and ran with more privilege than any job needed.** `release.yml` had never run: its tag pattern never matched the `2.0.0` tag, so v3.0.0 would have been its first execution. Built and exercised before tagging, it failed in two independent ways before producing a single artifact. Its test gate installed `pip install -e ".[ml,dev]"` — unlocked, and without `[api]` — so four security-test modules (`tests/integrations/test_egress_hardening.py`, `tests/integrations/test_platform_native_egress.py`, `tests/security/test_source_offer.py`, `tests/test_data_source_credential_redaction.py`) errored at collection and pytest stopped with `Interrupted: 4 errors during collection` (exit 2), skipping build, docker, release and publish. Those four are deliberately not made skippable; a lane that drops the extra must fail loudly. And even with collection fixed, the gate's sequential `--timeout=60` failed `test_false_alarm_rate_matches_the_measurement[lstm_autoencoder]` on timeout, a test CI passes under its `--timeout=300`, while another test ran 76.28 s and four more 48–54 s. The gate now mirrors CI's `ml-tests` lane step for step: `uv sync --locked --extra all --extra dev`, the dependency licence gate, the pinned-model prefetch, `MERCURY_REQUIRES_HF_MODELS` and `MERCURY_FULL_LEDGER_CHECK` (so the real-model tiers and the full adversarial-ledger recomputation run instead of skipping), `-n 4 --dist=loadgroup --timeout=300` and the same coverage floor — so a green `main` now implies a passing release gate.

The build job ran `sed -i "s/version = \".*\"/..."` over `pyproject.toml`. Unanchored, it rewrote every line *containing* `version = "` — mypy's `python_version`, pylint's `py-version` and ruff's `target-version` all became `"3.0.0"`, which ruff rejects — and the corrupted file shipped inside the sdist. It also let a tag stamp a version the repository never committed. The step now verifies instead of rewriting: the release refuses unless the tag's version, compared under PEP 440 so `v3.1.0-rc1` still matches `3.1.0rc1`, equals both `[project].version` and `_version._FALLBACK_VERSION`.

Two security properties came out of the same pass. Every job used to inherit `contents`, `packages` and `id-token` **write**, so `validate`, `test`, `build` and `notify` — which only read code — each held a token able to push tags, publish packages and mint OIDC credentials; the default is now `contents: read` and each job declares the one write it uses (`docker` packages, `release` contents, `publish-pypi` id-token). And the `docker` job restored and wrote the GitHub Actions layer cache (`cache-from/to: type=gha`) for the very image it publishes, a cache-poisoning path into the release artifact; it now builds every layer from the pinned base and the locked closure. The mirrored test gate likewise takes `ml-tests`'s uv pin but not its caches, since in a release workflow a restored cache is an input nobody reviewed. Mirroring the lane first and auditing it second is how that surfaced: the first version of this fix copied the lane's caches into the release job along with everything else, and that copy was removed before it landed.

`tests/scripts/test_release_gate_parity.py` pins all of it — locked install, identical pins, identical environment and decisive pytest options, coverage floor, no pyproject rewrite, no cache anywhere in the release path, the exact token scopes of every job, and the verify step's behaviour executed as the runner runs it. Against the workflow as it stood on `main` it fails 17 of 18; restoring the docker layer cache or the write-all default each fails its own test. Not changed, and recorded here so they are decided rather than discovered: PyPI rejects `mercury-agent-global` because the `pqc` extra declares `ama-cryptography` as a git URL, which AMA's sovereignty makes the correct dependency, so the release's own install path remains the git tag; and setuptools 84 warns that table-form `project.license` stops building on 2027-02-18, which needs an SPDX expression chosen against `LICENSE-EXCEPTION`.

**Security — three newly published container CVEs accepted on verified conditions, and a gap in how CI greenness was read.** Hours after #399 merged, the `Security` workflow on `main` failed `Container Security Scan (Trivy)` against an unchanged Dockerfile: the vulnerability DB had published `CVE-2026-119374` (arbitrary code execution via a corrupted FTS5 index) and `CVE-2026-119379` (via crafted session changesets) against `libsqlite3-0`, and `CVE-2026-85091` (heap overflow in `gz_vacate()`) against `zlib1g`.

The surfacing matters as much as the finding. `main`'s commit check-runs read **58/58 green with zero failures** at the time, because the failing run belonged to a workflow run not attached to that commit's check list. A commit-level green is therefore not evidence that `main` is green, and anything that claims to verify branch health has to enumerate workflow runs, not commit checks.

All three are `vulnerable` in bookworm, trixie **and forky** on the Debian tracker with only sid fixed, so no trixie upgrade exists; and the measurement already recorded for `CVE-2026-76844` applies unchanged, since a release-13 advisory carrying no fixed version makes Trivy mark every installed version `affected`. `libsqlite3-0` backs CPython's `_sqlite3` and `libz.so.1` is in the CPython interpreter binary's own NEEDED list, so neither package is removable.

Reachability was established per CVE rather than inherited, and hardened a level deeper than the earlier acceptances went: a symbol audit of **every native library in the environment** (838 shared objects) found zero importers of SQLite's session extension and exactly one importer of zlib's `gz*` layer — cv2, which links the unmangled system `libz.so.1` (the wheel vendors no libz) and uses the six imported `gz*` functions only in its FileStorage persistence, entered only on a gz-suffixed filename that no first-party source supplies. That audit **corrects the `CVE-2026-76844` acceptance's original wording**, which said the `gz*` layer "has no caller in the image": the accurate claim is binder-present, caller-absent, and both halves are now machine-checked — the acceptance test pins cv2 as the only permitted `gz*` importer (with a vacuity guard proving the scanner finds cv2's imports whenever cv2 is installed, so a silent non-scan cannot pass) and fails on any first-party FileStorage or gz-suffixed persistence path. The FTS5 finding rests on Mercury building no FTS table, the same condition already enforced for `CVE-2026-119373`. The session-changeset finding rests on CPython's `sqlite3` wrapping no part of the session extension — no session, changeset, patchset or rebaser API, and no session attribute on `Connection` — and deliberately **not** on `SQLITE_ENABLE_SESSION` being absent from `pragma compile_options`, which is a property of whichever libsqlite3 the interpreter links and so differs between the image and a developer's host. The `gz_vacate` finding needed no new argument at all: it is a second flaw in the same `gz*` stream API, and `gz_vacate` was already one of the function names the existing acceptance test's detector matched. `tests/security/test_sqlite_cve_acceptance_conditions.py` gains three session-extension tests including a vacuity guard, and both acceptance modules now cover every CVE they speak for.

Reproduced with the CI-pinned trivy 0.70.0 against a Debian 13.7 dpkg manifest at the exact versions CI reported: all five findings appear without the ledger, the three new entries clear the gate, and removing any one of them fails it again. One correction worth recording — the first local run reported only the two older CVEs, because the Trivy DB in this container predated the publications; the verification above is from a refreshed DB, and a stale scanner DB reads exactly like a clean scan.

**Added — the 21-probe math-arrest ensemble is reachable, measured and
repaired.** `AnomalyMathArrest` (21 equation probes: Hankel delay-embedding /
SVD trajectory, chaos, harmonic, wave, momentum, energy, robust location) was
exported from the package root and gated in CI for its probe count, but no path
in `src/` consumed it and it had never been measured on real labels. It is now
measured on both recorded corpora, three real defects are fixed, and a
`BaseDetector` adapter (`detectors/math_arrest/detector.py`) makes it a manifest
entry (`math_arrest`) the engine, registry, served routes, CLI and console
catalog can reach. Measured **0.7498 mean NAB AUROC**, which places it second of
all series detectors pinned in `tests/detectors/test_recorded_nab_series.py`
behind `lstm_autoencoder` (0.829), at a **0.0217** false-alarm rate; the
stationary AR(1) rate is **0.0525** against the α = 0.05 its conformal cut
promises.

* **The fusion prior was inverted against measured signal.** Probe scores were
  combined with `weight[rank] = PHI^(-rank)` over a hand-written affinity
  order, whose first three ranks carry 76% of the weight. On the `default`
  order those are the probes measuring **0.531 / 0.524 / 0.378** NAB AUROC,
  while the two strongest — `wave_propagation` (0.697) and `svd_projection`
  (0.669) — sit at rank 7 and below and share about 2%. The 21-probe fusion
  therefore scored *below its own best member*. The default is now `uniform`
  (the plain mean, the dominant aggregation for outlier ensembles per
  Aggarwal & Sathe 2017): **+0.13 AUC on both corpora**. `phi_affinity` is
  retained and reproduces its released numbers exactly.
* **`SVDProjectionProbe` never used its fitted basis.** `fit` stored residual
  statistics from the training Hankel SVD, but `deviation_score` recomputed the
  SVD *on the data being scored*, so the learned rank-1 subspace was unused,
  residuals were compared against statistics from a different basis, and the
  anomalies being scored helped define the subspace they were measured against.
  Its `abs` also scored a window fitting the subspace *better* than training as
  deviant. It now projects onto the stored basis, directionally.
* **21 probes were not 21 dimensions, and the correction was logged as noise.**
  The decorrelator measures **11.64–16.00 effective dimensions out of 21** on
  the recorded NAB files (13–23 redundant pairs per fit, one at **r = 0.9976**).
  It correctly down-weights the duplicates, but logged each pair at `warning`
  severity — up to 23 `WARNING` records per fit for a handled design condition.
  Now `debug`, with one `info` summary; and the served result carries
  `effective_probe_count` and `redundant_probe_pairs` so the probe count cannot
  read as an evidence count.

The entry carries the full set of gates a new `series_1d` manifest entry is
required to satisfy: measured stationary false-alarm pin, NAB triple,
evidence-tied ontology row (with the structural blind spot that the probes
reduce a matrix to its per-row mean, so a pure cross-channel correlation break
is invisible), knowledge profile with citations, exact fitted-state round-trip
and the registry/engine seam harvest. It is **not** added to the engine's
default set: that requires a marginal-gain ablation against the live ensemble,
which is not claimed. See `docs/DORMANCY_LEDGER.md` §2b.

**Added — full reachability sweep of the assistive mechanisms, and one
measured refusal.** A transitive import closure from the real entry points
(`engine.py`, every `api/` module, `cli`, and all 114 manifest `module_path`s)
was run against every ensemble, fusion head, combiner, calibrator,
meta-learner, drift and uncertainty mechanism in the tree. 522 modules are
reachable. Most unreached ones are correctly unreached and say so in the module
(a deprecated legacy fusion that warns at import; two read-only measurement
layers that must not touch the operational path; four registered operator tools
in `TOOL_REGISTRY`, two of them pre-commit gates; relocation shims). Four
remain genuinely unreached and unmeasured and are now named rather than
assumed, with their claims recorded, in `docs/DORMANCY_LEDGER.md` §2c.

`core/adaptive_domain_thresholding.py` (1,289 LOC) was measured against the
operating point Mercury ships, on the five recorded ADBench tables × 3 seeds,
scoring the *same* detector's scores so the comparison isolates the threshold:

| policy | mean F1 | mean FPR | FPR range |
|---|---|---|---|
| conformal α = 0.05 (shipped) | 0.4059 | **0.0231** | 0.000 – 0.048, never over α |
| ADT label-free | 0.4549 | 0.0873 | 0.013 – 0.160 |
| ADT supervised | 0.3882 | **0.6335** | 0.055 – **0.993** |

The supervised path is **broken, not a tradeoff**: it flags 98.7% of normal
rows on `pima_real`, 95.7% on `breastw` and 97.9% on `wbc`, with a mean F1
*below* the shipped cut at ~27× the false-alarm rate. The label-free path does
not clear the bar either — its higher mean F1 is bought by abandoning the α
guarantee and is carried by one table, while it loses 0.36 F1 on `wbc` and 0.31
on `thyroid`. **Refused, not promoted**; the measurement is pinned by
`tests/core/test_adaptive_domain_thresholding_measured.py`.

**Fixed — `PlattScalingCalibrator` overflowed `np.exp`.** It evaluated
`1 / (1 + np.exp(A * s + B))` directly, which overflows on real score
distributions. The comment claimed "clipping for numerical stability", but the
clip was applied to the *result*, after the overflow had already happened
inside `np.exp`; `calibrate()` had no guard at all. A second sigmoid used the
branch-selecting `np.where` form, which returns correct values but evaluates
both arms eagerly, so it still warned and did twice the work. Both now use
`scipy.special.expit`. Value-preserving: the fitted `A`/`B` are identical to
six figures and every measured number above is unchanged.

**Changed — `ml/ensemble.py` no longer claims an uncited improvement.** The
module and one of its class docstrings asserted "Research shows 30-40% accuracy
improvement with proper ensemble methods" with no citation, in a module that no
path reaches and that has never been measured. The claim is removed. The
averaging result it gestured at is real and is cited where it is actually
relied on (Aggarwal & Sathe 2017 — the detector tier's `consensus` combiner and
`math_arrest`'s fusion default), but it is a statement about averaging
dominance, not a percentage, and it does not transfer to a stacking/boosting/
bagging head without measurement.

**Changed — detector taxonomy now matches the discipline.** Tornado
(severe-storm meteorology, NEXRAD radar morphology), hurricane (tropical
meteorology, ERA5 wind-field physics) and flood (hydrology,
precipitation-runoff physics) sat in the `geological` category and in
`detectors/geological/` modules; they are now `meteorological`, with their
three hazard adapters, and the modules live in `detectors/meteorological/`.
The meteor detector (NEO/bolide assessment over NASA NeoWs + JPL
fireball/Sentry feeds) is space science and moved to
`omni_mercury_engine.space.meteor_detector` under category `space`, joining
the solar-flare detector that made the same move earlier. Wildfire (NFDRS
fuel moisture, spread, NDVI, FIRMS) is wildland fire science — neither
geology nor meteorology — and moved to a new `detectors/fire/` package under
a new `fire` category; the post-fire debris-flow cascade (a geological
hazard) stays in `geological`. Every old module path keeps a shim that
re-exports every public name (and wildfire's module constants), so existing
imports keep working (DEPRECATION.md "Relocated" entries); in-repo imports
use the canonical paths. The console catalog is regenerated and the README
scale block re-measured (its "111 detector-manifest entries" prose had
already drifted from the real 113).

**Added — the console shows its evidence.** The Mercury Console's tier runs
take `--reference <fraction>` (the leading share of the loaded series becomes
the known-normal probationary prefix; a file path is the CLI form) and every
tier response now renders its `calibration_basis` — with a warning styling
when a run is self-calibrated, so a parameter-set flag rate is never mistaken
for evidence — plus the conformal cut, the α and flag count, and any
`excluded_members` with the reason. The score line chart draws the conformal
cut as its own labelled dashed series and the per-point cross-detector
`uncertainty` (returned by the tier and previously dropped) as a translucent
disagreement band; responses without those fields render exactly as before.
All console changes are named patches in `scripts/build_console_port.py`
(the design prototype stays verbatim), and the account dashboard's
authenticated nav links the console it mints API keys for.

**Fixed — the committed docs dashboards were stale and one was orphaned.**
`docs/images/{anomaly_detection_panel,mercury_performance_dashboard,benchmark_summary_live_data}.png`
are regenerated from the committed benchmark JSON, and
`ensemble_benchmark_report.png` — renamed by an earlier fix so it stopped
overwriting the neuro-symbolic demo figure, then never committed or cited —
is committed and embedded in the README with the other measured dashboards.
The generator's docstring names its real outputs.

**Fixed — the default statistical detector flagged up to a fifth of normal
data.** With no threshold or calibration configured, `MercuryAnomalyDetector`
cut each batch with a batch-relative rule (Otsu split when a "valley" was
found, else median + 2 * 1.4826 * MAD) chosen for F1. Its valley test compares
the density at the split with the *global* peak, so a split in the thin upper
tail of a unimodal distribution reads as a valley: it flagged 0.171 of held-out
Gaussian noise, a mean 0.089 of held-out ADBench normal rows (0.127 on WBC),
and 0.120 of NAB-normal points (0.407 on the AWS CloudWatch file). The default
is now the fit-time reference cut at its one-class split-conformal quantile
(`reference_alpha`, 0.05), which already served single rows and floods: 0.048
mean on the same ADBench rows (0.026-0.065), 0.021 on NAB, NAB standard score
-130.5 -> 30.7 at the same ranking. Recall moves little where fitted on normal
rows (cardio 0.853 -> 0.827, thyroid 0.988 -> 0.963) and most on pima (0.329 ->
0.158). `operating_point="adaptive"` restores the old cut. The bound covers new
data: scoring the fit batch itself flags about alpha of it whatever its
contamination, so callers who know the contamination should configure it
(`auto_calibrate` + `contamination`). `reference_alpha` outside (0, 1) and an
unknown `operating_point` are refused. Measured and pinned in
`tests/detectors/test_recorded_false_alarm_rates.py`; `statistical` is now
pinned in `tests/detectors/test_recorded_nab_series.py`, which also pins every
series detector's NAB-normal false-alarm rate next to its AUROC and NAB score.

**Fixed — the info-geometry certificate was cut on the wrong scale.** Its
component threshold came from the batch-relative rule, whose degenerate (fewer
than 8 rows) and anomalous-bulk branches return the *fused* reference cut, a
value on another scale (0.577 against the component's own 0.617 on 8-feature
Gaussian training data); in 4-row batches of normal rows its verdict flagged
0.093 of them. The component now keeps its own fit-time scores (persisted in
the fitted state) and is cut at their conformal quantile, independent of the
batch: 0.055 of the same rows.

**Fixed — `echo_state` alarmed on 0.188 of NAB-normal points.** Its fixed readout
ridge of 1e-4 left a 101-parameter readout fitted to 169-282 probationary
samples with an out-of-fold one-step error of 2.8-8.9 on the standardised
series, worse than predicting the mean (1.0); on `rogue_agent_key_hold` it
flagged 0.533 of the normal points. `ridge=None` (the default) now picks the
penalty from a 1e-6..1e3 grid by out-of-fold error on the training folds the
squash scale already uses: false-alarm rate 0.012, NAB standard score -368.3 ->
44.5, AUROC 0.699 -> 0.629 (a ranking cost on these four files, recorded). On
the adversarial shapes its own AUROC rose 0.786 -> 0.861 (ledger regenerated).

**Fixed — `MultiScaleTransformerDetector` could not run at its defaults and never
alarmed.** Three defects at the default `window_sizes=[10, 25, 50]`: every scale
encodes the 50-step sequence but each positional table held `2 * ws` rows (a
crash on the first batch); the Gaussian prior underflows to exact zeros on a
50-step window and `F.kl_div` differentiated at a zero target is log(0), so the
first optimiser step turned every weight NaN and every score NaN; and 2-D input
returned one score per (window, position), 27,550 values for 600 rows. The KL
now uses floored distributions, 2-D input is scored per row (mean over the
windows containing it, cut on the same scale), and inference-time point
adjustment -- an evaluation protocol that needs ground-truth segments -- is off
by default: it grew alarms to every run above 0.8 of the cut and flagged 1.000
of held-out noise where the cut alone flagged 0.043.

**Fixed — `ContrastiveLearningDetector` calibrated against itself.** Training
rows were scored against a reference bank that contains them, so each counted
its own zero distance among its k neighbours and the 95th-percentile cut sat
low: 0.105 of held-out noise, AR(1) and ADBench cardio normals flagged. The cut
is now taken on leave-one-out scores (0.040-0.081 on the same data), and a bank
no larger than `n_neighbors` no longer crashes `np.partition`.

**Fixed — `engine.detect` called every normal batch anomalous.** Its batch
verdict OR-ed every base detector's per-row flags, and three of the five
scored a row relative to its batch-mates: `spatial` and `dimensional` min-max
normalised each channel over the batch (its most extreme row scored ~1
whatever its value), and `directive` blended three whole-batch scalars (column
coherence; SHA3 entropy and chained-hash "checksums" of the batch's raw bytes;
an FFT over all features flattened) into every row at 0.2 / 0.15 / 0.1. With
all five fitted on 1,000 normal rows, every all-normal batch of 10 or 100 rows
was called anomalous. Now:

- `spatial` and `dimensional` normalise each channel against the training
  rows' range widened by the row (the `[reference; row]` semantics the
  agentic serving path already used) and cut at the training rows' 0.95
  one-class split-conformal quantile; `dimensional` persists the anchors and
  reference in its fitted state, `spatial` rebuilds them from its points.
  Held-out Gaussian rows flagged 0.045 / 0.047 / 0.054 (spatial) and 0.051 /
  0.058 / 0.063 (dimensional) at batch sizes 10 / 100 / 2,000, against 0.40 /
  0.02 / 0.012 and 0.30 / 0.28 / 0.077.
- `spatial`'s query-time LOF dropped the neighbour's k-distance from the
  reachability distance (Breunig et al. 2000: max(k-distance(o), d(p, o)),
  which fit used), so query and training densities were on different scales.
  Restored; ADBench tabular AUROC mean 0.707 -> 0.716 (breastw 0.564 ->
  0.493: its training split holds 35% clustered anomalies, a dense cluster
  LOF reads as normal).
- `directive`'s batch-scalar blends default to 0 (still computed and
  reported): 0.043 / 0.046 / 0.049 of Gaussian rows by batch size against
  0.103 / 0.062 / 0.045, breastw 0.201 -> 0.048, AUROC identical.
- Every base detector keeps its training rows' scores as a reference and
  reports conformal `p_values`, Benjamini-Hochberg `fdr_flags` and a
  `batch_anomaly` test. Past the reference maximum the p-value follows a
  generalised-Pareto tail fitted to the reference's top 10% (shape held >= 0:
  with the moment fit's negative shape, rows past a near-max endpoint got
  p = 0 and 0.17 of normal 100-row batches were flagged).
- The engine tests each detector that reports p-values with BH at 0.05 / K
  (union bound over K detectors, valid under any dependence). A detector
  fitted on the batch it scores, or whose p-value floor cannot reach level /
  m, keeps the per-row OR: fail-loud, not deaf. Measured: all-normal batches
  flagged 0.050 (mean of 6 seeds, 10 rows), 0.008 (100 rows), 0.000 (1,000
  rows); a batch with one +4-sigma row flagged every time at every size.

`dimensional`'s NAB standard score falls 48.7 -> 22.7 at unchanged ranking
(AUROC 0.533 -> 0.537): the batch-relative cut happened to sit at a 0.001
false-alarm rate on those 1,000-row files and at 0.30 on 10-row batches; the
cut is now the same 5% budget as the other detectors (0.025 on NAB).

**Added — the detector tier takes a known-normal `reference`.** Without labels
`run_tier_ensemble` fits, thresholds and conformal-calibrates on the series it
scores, so its flag rate is set by the parameters, not by evidence (about
`conformal_alpha` of any series by construction; the result now says so in
`calibration_basis`). `reference=` (a probationary prefix or any known-normal
stream, >= 8 points; `--reference` on `tier-detect`, `reference` on the MCP
tool and `POST /detect/tier`) fits and calibrates the ensemble there and only
scores the input. Measured on the four recorded NAB files (probation as the
reference, alpha 0.05): conformal flags on 0.011-0.051 of the NAB-normal
points, 0.048 of pure noise against a noise reference, an 8-sigma spike
flagged. `labels` and `reference` are mutually exclusive (labels describe the
scored series); `rca` is excluded under a different-length reference and
recorded in `excluded_members` (it fits a node graph over its training
points), and naming it in `subset` with such a reference is refused. The
default path is byte-identical (adversarial ledger reproduced to the digit).

**Added — `mercury_detect_anomaly` takes an optional `contamination`.** The MCP
tool fits on the batch it scores, so its reference *is* the batch and the
default cut flags about 5% of it whatever the true share; a stated share in
(0, 0.5] cuts there instead (ADBench breastw scored on itself: F1 0.221 ->
0.921 at its true 0.35). It sets the cut, not the ranking: a dense cluster
fitted transductively still partly masks itself.

**Corrected — the tier's unlabelled "conformal" flags are not evidence.**
`docs/DETECTION_QUALITY.md` called them evidence-controlled. Without labels
`run_tier_ensemble` calibrates them on the scored series itself, so they flag
about `conformal_alpha` of it by construction (0.047 at alpha 0.05); the
generator now says so. The fusion-hardening ADBench ledger was re-measured on
the new `statistical.py` and reproduced mean AUROC 0.7745 on all 18 sets: the
operating-point and p-value changes move the cut, not the ranking.

**Measured, not changed.** `spiking` flags 0.528 of `rogue_agent_key_hold`'s
normal points: the series idles at exactly zero for 13% of its probationary
period and 59% of the rest, and the detector is driven by the deviation from the
training mean, so the idle stretch reads as a sustained excursion -- a property
of the drive, pinned rather than tuned. `srcnn`'s calibrated cut flags 0.102 of
a stationary AR(1) held-out half (0.045-0.24 by seed); a cross-fitted cut (0.202)
and a same-model held-out split (0.147, NAB AUROC 0.385) were both measured
worse, because the score scale moves with each training run.

**Fixed — every Trivy job on main failed on an unchanged repository.** Right
after PR #394 merged, "Install Trivy binary" failed in ci.yml's Docker Build
and security.yml's Container Security Scan with curl's exit code 6, three runs
in a row, 55 ms after the installer found the release. The step piped
`contrib/install.sh` from the aquasecurity/trivy *main* branch into `sudo sh`;
upstream had changed that script to fetch the archive from `get.trivy.dev`, a
host that ordinary DNS (including the runners') did not resolve. The new
composite action `.github/actions/install-trivy` downloads the pinned v0.70.0
release asset from github.com, checks it against the SHA-256 published in that
release's `trivy_0.70.0_checksums.txt`, and confirms the installed version; all
four Trivy jobs (docker.yml's advisory scan included, which had used the
trivy-action's own installer) use it with `skip-setup-trivy: true`. The Lean
step in `verifiers.yml` had the same shape (`elan-init.sh` from the elan
*master* branch, installing the latest elan and the moving "stable" Lean); it
now installs the elan v4.2.4 release binary against a recorded SHA-256 (elan
publishes no checksum file; the digest is recorded from the release asset) and
the explicit toolchain `leanprover/lean4:v4.35.0`.
`tests/test_ci_remote_install_contract.py` fails a pull request that pipes a
downloaded script into a shell, fetches from a moving `main`/`master`/`HEAD`
raw URL, or runs a Trivy scan without the pinned binary; it fails on main's
workflows before this change.

**Fixed — `detect -d fusion` scored any width against the bundled 16-column
model and called the result clear.** The bundled `default_fusion.pt` predates
the `fitted_feature_width` checkpoint key, so the width guard skipped it: a
4-column CSV (480 N(0,1) rows, 20 rows shifted by +6 sigma) came back
`state: grounded`, `disposition: clear`, `decision_confidence: 1.0`, p ~ 1e-17
on every row, while the drift check in the same result said "input has 4
features; the reference has 16". `load_model` now recovers the width of a
checkpoint without the key from the training rows *that load* restored (the
spatial detector's `lof_points`, the drift reference's own source; state left
by an earlier load never counts); a checkpoint that records `None` on purpose
(pooled fit) stays unchecked. The refusal is its own exception,
`FeatureWidthMismatchError` (a `ValueError`), so each surface reports exactly
it: the CLI as a usage error naming both ways on (train on the layout, or use a
base detector) for 2-D, 1-D and mapping input alike, `POST /detect/flagship` as
a 400, and the MCP flagship tool as a tool error, while any other `ValueError`
stays an internal error. New `GET /api/v1/detect/flagship/model` reports the
served model's width, training state and provenance; the Mercury Console's
Flagship card reads it, lays its built-in sample out at that width (it sent a
16 x 2 matrix, a guaranteed 400) and skips a dataset of another width with the
reason. Trained on the 4-column layout, the same data scores 20/20 with 0
false positives (in-sample). Six test files encoded the old behaviour: five fed
5-, 6- or 12-column data to the 16-column model and asserted an answer (they
now pass 16 columns), and `tests/core/test_engine_feature_width.py` asserted
that a checkpoint without the key is unchecked (it now asserts the recovered
width, and `None` when no training rows were restored).

**Fixed — calibration sigmoids overflowed.** `TemperatureScaling._sigmoid`, the
threshold-integral treatment curve and the `decision_function` margin map used
`1 / (1 + exp(-z))`, which raised `RuntimeWarning: overflow encountered in exp`
on a confident model. All three use `scipy.special.expit`: identical where the
old form was finite, exact past it.

**CI — the three failures on main after PR #388 are now pull-request gates.**

* *Gitleaks.* gitleaks-action builds a pull request's range from the first page
  of its commit list, so PR #388 reported "30 commits scanned" and passed while
  two test keys added later failed the push scan on main. Secret Detection now
  also builds the PR's squash-equivalent commit (the merge ref's tree on its
  base parent) and scans it with a checksum-pinned gitleaks 8.24.3 -- the exact
  unit main scans. The two keys (`tests/api/test_voice_backends.py`, made-up
  values the tests prove never leave the server) are allowlisted as exact
  strings. `tests/test_secret_scan_contract.py` keeps the step, requires every
  allowlist entry to be a literal still in the tree, and holds `.gitleaks.toml`
  to a shape that cannot widen what both scans skip (no `[[rules]]` override,
  allowlisted paths that each name one file, full commit SHAs, no
  `.gitleaksignore`, whose fingerprints could never match the synthetic commit).
* *Disk quota.* The failing step built the audit environment, now carrying the
  CUDA torch stack through `[api]`, in `/tmp/mercury-audit-env`. On the
  ubuntu-26.04 runner image `/tmp` can be a RAM-backed tmpfs (about 7.8 GB) with
  a per-user quota of about 6.3 GB (actions/runner-images#14777; the image fix
  #14814 is still rolling out), hence `[Errno 122] Disk quota exceeded` on a
  runner with 80+ GB free on `/`; the PR had passed on a newer image. (The first
  version of this change blamed the root disk and a restored pip cache; the
  failing job's log says "pip cache is not found", and a measured runner had
  88 GB free before cleanup. That diagnosis was wrong and is withdrawn.) Both
  audit environments (`security.yml` and its `ci.yml` twin) now live under
  `$RUNNER_TEMP`. One composite action, `.github/actions/runner-disk-guard`,
  replaces eleven inline cleanup copies: it points `TMPDIR` at `$RUNNER_TEMP`
  for every later step, reports both filesystems, removes unused toolchains
  only when the root disk is below the job's floor (the unconditional cleanup
  had cost 94 s on a 4-minute job with 88 GB free), and fails with the shortfall
  if it still is. `tests/test_ci_disk_budget_contract.py` derives the
  torch-bearing extras from `pyproject.toml` (including requirement files,
  `pkg[extra]` and wheel-path installs, expression-built extras and `uv run`)
  and requires every job that installs one to run the guard first and never to
  build that environment under `/tmp`. It fails on the pre-fix `security.yml`.
* *Dependabot uv.* `scripts/simulate_dependabot_uv.py` replays Dependabot's
  pin-and-lock for every direct dependency, proposing what Dependabot would
  (the newest release no ignore covers) and every configured group together,
  and fails on an update that cannot be solved, that downgrades another
  package, or whose ignore has gone stale. It reproduces all three outcomes of
  the 2026-11-03 run on main's tree (shap 0.50.0 -> 0.51.0 unsolvable; h5py
  3.14.0 -> 3.16.0 downgrades TensorFlow 2.21.0 -> 2.20.0; nixtla 0.7.2 -> 0.9.0
  downgrades pandas 3.1.0 -> 2.3.3), and found the next one before it happened:
  with 0.8-0.9 ignored Dependabot falls back to nixtla 0.7.4, which caps pandas
  too. It reads live PyPI, so it runs on pull requests that change
  `pyproject.toml`, `uv.lock` or `.github/dependabot.yml` and weekly in
  `.github/workflows/dependabot-replay.yml`, ahead of Dependabot's Monday run.
  `scripts/check_lock_downgrades.py` fails a pull request whose `uv.lock`
  resolves any package to a lower version in any supported environment (a grid
  of Python versions and platforms, each lock entry's `resolution-markers`
  evaluated), unless a person's commit in the PR declares it with the git
  trailer `Lock-Downgrade: <name> <old> -> <new>: <reason>`; text in a PR body
  or in a Dependabot body's quoted release notes declares nothing. Dependabot
  PR #393 did exactly that behind a "minor" title, with auto-merge armed and
  every check green; its auto-merge was disabled.

**Changed — `uv.lock` no longer resolves for Intel macOS.** torch has no x86_64
macOS wheel from 2.3 on (2.2.2 was the last), numba none from 0.63 (and every
numba that has one needs numpy < 2.4 against the >= 2.4 floor), so that fork
could not install any torch-bearing extra and was the fork the shap update
could not solve. `[tool.uv] environments` now lists Linux, Windows and
Apple-silicon macOS; the relock drops the side-forks that forced (shap 0.50.0
for Intel macOS; a pandas 2.3.3 / nixtla 0.9.0 pair that existed only for
Python >= 3.15 on Windows). `pip install -e .` of the core package on an Intel
Mac is unaffected; the README platform table says so. Every nixtla release from
0.7.3 to 0.9.0 declares `pandas<3`, and TensorFlow 2.21 declares `h5py<3.15`,
so both are versioned Dependabot ignores with reasons; the replay reports
either as stale once it becomes takeable.

**Changed — astral-sh/setup-uv 9.0.0 -> 10.3.3** (Dependabot #392, green on
every lane), pinned by SHA.

**Fixed — fusion-weight cross-validation validates its weights.** In
`benchmarks/calibration_validation.py::run_fusion_weight_cv` each fold split off
its held-out rows and never used them: the fold's L-BFGS-B weights were fitted
on its training rows and averaged, so nothing was cross-validated (flake8 7.5.0
reported the two unused variables). Each fold now scores its optimum, the
default 0.4/0.3/0.3 and its own adaptive weights (computed from the fold's
training labels only) on the rows it held out, at an F1 cut chosen on its
training rows, and the result carries them as `cv_heldout`, with the MD-003
summary gaining the held-out deltas. The run also flags a fold whose optimum
sits on the optimiser's box on every component (`optimum_on_bounds`): in the
committed 55-dataset run that is 28 datasets, whose "optimal" weights were set
by the bounds, not the data. The existing `f1_at_*` figures are unchanged and
now say they use an oracle cut on the test rows. Tests:
`tests/benchmarks/test_calibration_validation.py` (held-out partition, the
training cut, the bound flag, per-fold adaptive weights never see held-out
labels, the unchanged oracle figure). The committed results JSON predates the
fix and has no `cv_heldout`; it gains one when the harness is next run.
`docs/BENCHMARKS.md` no longer calls MD-003 resolved by validation: it states
that the figures are a test-split comparison at an oracle cut, that the folds
had not been scored, and that 28 of the 55 datasets have bound-set optima.

**Fixed — rotating the at-rest key re-seals the stored provider keys.**
Provider keys stored from the console are sealed under `MERCURY_DATA_ENC_KEY`,
but `scripts/reseal_totp_secrets.py` re-sealed TOTP secrets only, so a rotation
would have left every stored provider key unopenable. `ProviderStore.reseal`
classifies each key as the TOTP pass does (already current, old key,
plaintext, unopenable; never overwritten) and the rotation tool runs it in the
same invocation, on the provider store alone when no durable identity store is
configured, failing the run when either store has a key it cannot open.
Tests: `tests/scripts/test_reseal_totp_secrets.py`.

**Changed — the documented language scope matches what shipped.** The README
still said the voice operates in English only and that multilingual natural
language was a future epic, and `tests/narrative/test_language_scope.py`
enforced that statement after the seven-language layer shipped. The README now
states the shipped scope and its boundary (questions understood in seven
languages; social turns, menu and prompts answered in them; narratives and
profiles never machine translated), ROADMAP row 18 is partly closed with what
remains, and the test pins the README to `narrative/languages.py::SUPPORTED`
instead of to prose. SECURITY.md gains the provider-administration routes, the
secrets-at-rest and language-model egress facts and the provider third-party
note; ARCHITECTURE.md gains the console, agent and provider layer and its real
logging start-up; the platform-hardening and deployment guides cover provider
keys in the at-rest key and its rotation; the README lists the new variables.

**Fixed — the Core Tests lane no longer runs out of time.** The lane ran its
selection serially under a 40-minute job limit; on run 41573886724 Python 3.13
passed its tests in 32.7 minutes and was cancelled during the coverage upload,
and Python 3.12, after an 11.5-minute dependency install, was cancelled
mid-tests, so a green suite read red. The lane now runs with 4 workers and a
300-second per-test timeout (`-n 4 --dist=loadgroup --timeout=300`), the flags
the ML lane already runs the whole tree with; the coverage floor is unchanged.

**Changed — tool and image bumps folded in from Dependabot.** mypy 2.4.1,
ruff 0.17.3, flake8 7.5.0 (pyflakes 4.1.0) and types-requests 2.33.0.20261030
at every pin site (`pyproject.toml` ml and dev extras, the Code Quality and
Type Checking installs in `ci.yml`, the pre-commit revs, the
`run_ci_gates.sh` header) with `uv.lock` regenerated (only those five packages
move); Grafana 13.3.0 in `docker-compose.yml` (same data volume). The
Dependabot pull requests touched `pyproject.toml` alone and would have failed
the pin-parity gate. ruff and mypy report nothing new on the tree; flake8's two
new findings were the cross-validation defect above.

**Added — the agent holds a conversation, in seven languages.** Measured
before this change: of 70 realistic inputs a console user types ("ok",
"anything new?", "should I be worried?", "what do you mean?", "say that
again", "that's not what I asked", "how do I use this?", "detect anomalies in
sensor.csv", "bye", and the same questions in Spanish, French, German,
Italian, Portuguese and Dutch), 52 fell through to the free-text search and
ended in the fallback menu. `narrative/conversation.py` now routes
acknowledgements, farewells, check-ins ("any problems?"), worry ("is it
serious?"), confusion ("what do you mean?" rebuilds the previous answer in
its fuller form), "say that again" (the previous answer again), "in short"
(its first two sentences), corrections (which never repeat the answer they
correct), how-to questions (three concrete steps naming only commands the
CLI has), plain-words run requests ("score this file"), "do you cover
earthquakes?" (the domain's profile), "how many detectors" (the manifest,
not the record), and four more glossary terms (hazard, anomaly, domain,
loader); a reaction such as "ok" never replaces the follow-up context, so
"ok" then "why?" still explains the run that was discussed.
`narrative/languages.py` (new; lexicon and regular expressions, no model, no
dependency) detects the asker's language from its own function words and
spellings, renders the question in the English phrasing the classifier
already routes (`¿cuántas anomalías hoy?` reaches the same builder as "how
many anomalies today?"), answers greetings, thanks, acknowledgements,
corrections and the fallback menu in that language, translates the follow-up
prompts (each of which maps back to its English intent when clicked), and
appends to a grounded answer a one-line note in the asker's language saying
how the question was understood. What it does not do is stated in the
module: a detection narrative or a knowledge profile is never machine
translated, because that would be a second, unverified source; those stay in
English under the note. `POST /api/v1/voice/speak` takes `language` (a
supported ISO 639-1 code, or `auto`/omitted to detect it; an unknown code is
a 422) and reports `metadata.language`, `metadata.understood_as` and
`metadata.answer_kind`; `MercuryVoice.speak` takes the same `language` and
`VoiceResponse` carries the three fields. Tests:
`tests/narrative/test_conversation_breadth.py` (the battery's routing, the
context rule, repeat/brief/correction end to end, no turn in the battery
ends in the fallback), `tests/narrative/test_languages.py` (detection per
language, every translated prompt round-trips to its English intent, every
frame exists in every language, the voice end to end with a detected and an
explicit language), `tests/api/test_voice_language.py`.

**Added — console settings: grey and system themes, spoken replies, visual
and assistive alert cues, the agent language.** The settings window gains a
Grey theme (a mid register between the dark instrument and the light reading
theme; `html.ds-grey` tokens) and a System theme that follows the operating
system's preference and re-applies when it changes; an AGENT section with the
language the agent answers in (Auto detects it from what is typed; EN, ES,
FR, DE, IT, PT, NL are sent as `language` on every `/voice/speak` request,
and the reply's language and reading are shown in the reply's steps); and
three accessibility settings beside reduced motion, contrast, scale and focus
rings: Spoken replies (the agent's replies and alerts read aloud through the
browser's speech synthesis in the agent language, silent where the browser
has none), Visual alert cues (a banner at the top of the console and a
threefold flash of the frame when an alert lands, a held outline under
reduced motion, for users who cannot hear an alert or are away from sound)
and Announce alerts (an assertive live region read by a screen reader as the
alert arrives). The theme and language option buttons and every switch now
carry `aria-pressed` / `aria-checked` as strings, so the state survives being
false. The self-test gains a Settings & accessibility suite (9,192 checks).

**Added — twenty-one language-model providers, connected from the console.**
The engine shipped adapters for ten providers; eleven more join on the shared
OpenAI-compatible base (`models/foundation/ollama_adapter.py`,
`OPENAI_COMPATIBLE_ADAPTERS`): Mistral, Groq, Together, Fireworks, OpenRouter
and Perplexity (a public endpoint and a key each), Azure OpenAI (the
operator's resource endpoint, a deployment name, the `api-key` header and an
API version), any OpenAI-compatible endpoint the operator names (required,
never guessed; key optional), and the local LM Studio, vLLM and llama.cpp
servers (plain http on a loopback host, no key, RFC1918 only under the
offline opt-in, exactly as Ollama). Every provider is in the enum, the
implemented set and the code-grounded catalog the drift gate pins; no model
ships by default; `tests/models/test_openai_compatible_providers.py` pins
the contract adapter by adapter. Connecting one no longer means editing the
engine host's environment: an administrator stores the key, endpoint and
model from the console's AI vendors view (`PUT
/api/v1/voice/backends/{provider}`; the key sealed at rest by
`api/provider_store.py` under `MERCURY_DATA_ENC_KEY` or `AMA_MASTER_SEED`,
never returned, refused with the reason when no stable key exists), tests
the connection with one read-only call that lists the provider's models
(`POST /api/v1/voice/backends/{provider}/test`; a key given there is used
once and not stored), and makes the provider the voice's model (`POST
/api/v1/voice/backends/select`, `DELETE` to detach), which survives a
restart (`MERCURY_PROVIDER_STORE_PATH`, inventoried by `decouple`). The
backends report says for each provider where its key, endpoint and model
came from (store, environment, default). `mercury-agent voice --model
provider:model` attaches the same way from the shell; the flag used to build
an Ollama chain the voice never consulted. And an attached model now has a
real, bounded job: it composes the free-text search answers from the
records the retriever found and nothing else (the reply names the model
among its sources and says how many records it composed from; an adapter
failure leaves the template answer), while grounded answers never go
through it. Tests: `tests/api/test_provider_store.py`,
`tests/api/test_voice_backends.py` (admin gate, the secret boundary, the
probe, selection and persistence), `tests/narrative/test_voice_model_compose.py`,
`tests/test_cli_voice_model_spec.py`.

**Fixed — a stored change reaches the running voice; a free-form route
states its body.** Storing a new key, endpoint or model for the provider the
voice is composing with rebuilds the served voice at once (the response says
`reattached`); a change that leaves the provider unable to answer (its key
cleared) detaches it and drops the selection rather than keeping a model
that cannot reply, and forgetting the active provider detaches it the same
way (`selection_cleared`). The provider row names which fields are in the
store (`stored`), so the console offers Forget stored settings only when
there is something to forget. Separately, running the console self-test with
an API key attached reached checks the keyless runs never did: the mass-run
fixtures for the assessment routes were walked field by field against a
free-form (`dict[str, Any]`) body that has no fields, and failed. Every
free-form route now states its contract on a `Body:` line in its OpenAPI
description (five of the nine had none), the console catalog reads that line
into the raw body field (`keys`, `reqKeys`; `scripts/build_console_catalog.py`,
`documented_body_keys`), the console's missing-field check and the self-test
use the documented keys, and `tests/scripts/test_console_raw_body_keys.py`
keeps the documentation equal to the key sets the assessment service
enforces, read from its `_obj` calls with `ast`. The self-test's voice
example count follows the fifth docstring example (`--model openai:gpt-4o`).

**Added — AI vendors: which language models and which AI clients can link
to Mercury, shown in the console.** The engine ships adapters for ten
language-model providers (OpenAI, Anthropic, xAI Grok, Google Gemini,
Cohere, DeepSeek, Cursor, Hugging Face, Ollama, built-in templates) and an
MCP server any stdio client attaches to, but the console showed none of it.
`GET /api/v1/voice/backends` (new, `api/backends.py`; anonymous, read-only)
reports every provider from the code-grounded `PROVIDER_CATALOG` with the key
environment variable its adapter reads and whether it is set on the server
(never its value), probes the local Ollama runtime through the egress gate
for its models (2-second timeout, 30-second cache), returns the
`mercury-agent mcp` command with the protocol version, the tool list and a
ready-to-paste configuration block for Claude Desktop and Claude Code,
Cursor, VS Code with Copilot and any stdio client, and says which provider
the served voice generates prose with, if an operator enabled one (the
console's grounded answers need none). The console gains an **AI vendors**
view under INTERFACES with status chips (key set or not, Ollama reachable
with its model count, active), provider cards (adapter and wire format, the
exact `export` line, `mercury-agent voice --model`) and client cards (the
configuration file, the block, Copy config), and a self-test suite. Vendors
that attach only to remote MCP servers (ChatGPT, Grok) are named as model
providers, not clients, because Mercury serves MCP over stdio only. Tests:
`tests/api/test_voice_backends.py` (the catalog boundary, a key never
appears in the response, both probe outcomes, the cache, OpenAPI).

**Added — the console control crawl is a repository script.**
`scripts/console_control_crawl.js` (previously a session-only tool) drives
every view of the served console in headless Chromium, restores the
console's state before each click, follows revealed controls to a chosen
depth and reports the controls with no observable effect; its verdicts, its
limits (a "no effect" is a lead, not a verdict: 14 of the 16 rows of the last
crawl replayed to controls already in the state the click would set) and the
replay discipline are documented in the console design notes and tested by
`tests/scripts/test_console_control_crawl.py`. CI runs no browser; it is a
manual tool.

**Fixed — a flaky catalog test that rewrote the committed catalog.**
`tests/scripts/test_generate_detector_catalog.py` appended a stale line to
`docs/DETECTORS.md` to prove `--check` fails, while the freshness test ran
`--check` in a subprocess; under xdist the two could overlap and the ML lane
failed on 2026-11-02 with a catalog that was fresh. The generator takes
`--path`, the stale copy lives under `tmp_path`, and the committed file is
never touched.

**Added — every domain and detector now carries its own profession,
institutions and lineage (knowledge profiles).** The console and the agent
described 30 domains and 113 detectors with one template each: a loader was
its file name, a detector was its one-line manifest description plus a
20-word ontology vocabulary in which 21 detectors shared an identical
signature, and "what is a detector" had one answer for all 113.
`core/knowledge_profiles.py` is the missing layer, hand-authored and
test-enforced:

- A `DomainProfile` for each of the 21 data-loader domains and the 9
  `hazard-viz` domains (22 distinct profiles): discipline, the professions
  that work with the data, the institutions that publish or govern it (name,
  role, `https` URL), what a practitioner brings, the raw observables, how
  Mercury turns the source into features and labels, the detectors that
  serve it, the limits and the literature.
- A `DetectorProfile` for each of the 113 manifest detectors: discipline,
  lineage (who published the method, when, in which tradition), the mechanism
  in plain words as the code does it, who reads its output and what they need
  to know, when to use it, when not to, and references (Adams & MacKay 2007
  for `bocpd`, Siffer et al. 2017 for `spot_evt`, Singer et al. 2016 for the
  Sepsis-3 screen, Staley et al. 2017 for the post-fire debris-flow cascade,
  Pulkkinen et al. 2013 for the GIC tiers, and so on).
- `core/loader_declarations.py` reads each loader's declared contract
  (source URL, key, label provenance, feature columns, label-derived columns)
  from source with `ast`, so the API, the console catalog and the agent say
  where a domain's data comes from without importing the loaders package.
- Authoring rules enforced by `tests/core/test_knowledge_profiles.py`: exact
  coverage against the manifest and the loader/hazard domains; no two
  profiles share a mechanism, lineage, practitioner description, method,
  label rule or observables sentence; a reference for every profile; an
  `https` institution with a stated role for every domain; adapter profiles
  name the detector they serve; `use_when` never claims an ontology blind
  spot; label rules match the loader's declared provenance; keyed loaders say
  which key; and the registry imports nothing but `dataclasses` and
  `typing`, so the PhysioNet pages the sepsis and seizure profiles cite can
  never be fetched from it. The clinical-data boundary test now tells a
  citation from a request builder on that basis (a module with no I/O-capable
  import holds a URL only as a citation) instead of counting every PhysioNet
  URL literal as a download path.
- Served at `GET /api/v1/catalog/domains[/{domain}]` and
  `GET /api/v1/catalog/detectors[/{name}]` (`api/routes/catalog.py`:
  read-only, anonymous, in the OpenAPI document), each record carrying the
  manifest entry, the ontology row, the loader declaration and the profile
  together (`tests/api/test_catalog_routes.py`).
- The agent answers "tell me about the sepsis domain", "what is bocpd", "who
  publishes the earthquake data", "what are the references for spot_evt",
  "what are the limits of the drought domain", "which features does the
  wildfire loader produce" and "which detectors serve space weather" from the
  profiles (`narrative/conversation.py` `domain_answer`,
  `detector_profile_answer`; routed before the glossary and never hijacking a
  record question that merely mentions a domain), naming its sources
  (`knowledge_profiles`, `loader_declarations`, `detector_manifest`,
  `detector_ontology`). Every one of the 22 domains and 113 detectors gets an
  answer no other gets, and every follow-up a profile answer suggests is
  itself answered (`tests/narrative/test_knowledge_answers.py`).
- `docs/KNOWLEDGE_PROFILES.md` documents the layer, where it flows and how to
  author an entry.

**Changed — the console is organised around the profiles, with descriptions
as hover notes, and every control does something.**

- Catalog rows (Detectors, Domains, Runs, Commands, Physics, API, MCP,
  Tools) show the title and chips (discipline, institution, input modality,
  label provenance, validation state, needed key, hazard-viz, flag counts)
  and keep the description, mechanism and lineage for the hover note, which
  also opens on keyboard focus; the inline description column is gone and
  rows are tighter.
- The Domains view names a domain by its profile title and discipline
  instead of its file name; hazard rows carry the discipline and open the
  domain's card as well as the `hazard-viz` form.
- Domain cards show discipline, worked with by, institutions (with roles),
  practitioner knowledge, observables, labels, limits, serving detectors,
  standards, references, source URL, needed key, label provenance and the
  feature columns (scored versus label-derived), with two new actions: "Ask
  Mercury Agent" (hands the question to the live voice) and "Open
  <institution>". A hazard-only domain (`schumann`) renders a card with its
  `hazard-viz` command instead of a loader snippet.
- Detector cards show discipline, lineage, mechanism, read by, use when, not
  for and references, with "Ask Mercury Agent" and "Open reference".
- Mass Run rows show the label only; the route, per-sample flag, payload
  fixture and the blocking reason are the hover note and the row's title.
- The 38 AMA items that did nothing when clicked open a note card (kind,
  section, source, the related commands, "Copy note", "Ask Mercury Agent",
  "Load decouple status").
- "Use simulated engine" (Session menu and connection dialog) did nothing on
  the served console because the simulated engine is not shipped with it;
  both now say so and are disabled.
- A blocked Mass Run capability (a hazard whose array shape the loaded
  dataset lacks, an assessment route without a detect-permission key) used to
  swallow the click; it now opens its hover note with the reason and what to
  do, and is announced `aria-disabled`.
- Right-panel tabs are `role=tab` with `aria-selected`; form enum chips,
  option presets and the sample-count chips carry `aria-pressed`; Mass Run
  rows carry an accessible name; the connection pill and the domain-context
  button carry `aria-haspopup` and an `aria-expanded` that reads `false` when
  closed (the runtime drops a boolean `false`, so the design's own binding
  vanished whenever the listbox was shut). Assistive technology, and the
  control audit, can now tell an already-active toggle from a dead one.
- A Playwright crawler (`restoring the console's state before every click and
  following revealed controls to depth two`) drove every view against the
  live server; the controls it found dead are the ones above, and the two it
  flagged in the Agent workspace (QUEUE, SEND) were replayed by hand and
  change state and send as designed.
- The agent chat offers "Tell me about the <context domain> domain" as a
  quick question, and the collect-data answer points to the per-domain
  profiles.
- `:diag` gains a Knowledge profiles suite (coverage, distinctness, `https`
  institutions, references, hover notes and chips on every catalog row,
  profile-driven domain, hazard and detector cards, note cards, Mass Run
  hover notes), and its Menus suite now picks every item on a sandbox and
  fails when picking it changes nothing; 9,179 checks pass against the live
  engine.
- `scripts/build_console_catalog.py` compiles the loader declarations and
  both profile registries into `catalog.js` (`loaders[].src/key/label/cols/
  scored`, `loaders[].prof`, `detectors[].prof`, `domainProfiles`).

**Fixed — the detection thresholds mean what they say at every width and
length, and the routes hold up under adversarial shapes (measured).** Driving
the served routes through hostile inputs found the point detectors wide but
shallow:

- On pure noise the multivariate route flagged 0.3% of rows at width 1, 14% at
  8, 72% at 16 and 100% from 32 up, and the fusion route called every noise
  matrix "anomaly": each compared a maximum statistic with a fixed 3.5. Every
  component of the shared kernel (`api/fusion_weights.py`) is now the
  z-equivalent of its maximum under the null given the values it ranged over
  (Šidák over rows and features, chi tails for the row norm), with Student-t
  tails for the estimated scale's degrees of freedom. Measured after: at or
  below the nominal rate at every width and length tested.
- The mean and standard deviation moved with the anomaly being scored, so a
  100 among nine values near 1 scored a z of 3.0 (masking). The kernel's
  location is the median and its scale the root mean square of the central
  90% of absolute deviations (a tenth of the rows, and always at least one,
  are ignored however extreme; 80% of the standard deviation's efficiency,
  against the MAD's 38%, whose sampling error alone pushed 30-row noise to a
  13% to 79% series false-positive rate). The univariate z-score, the
  multivariate route, the fusion route and the batch job share it.
- A spike riding a strong season was one seasonal amplitude above its crest,
  a z of 2 against the whole series: ranked first, never flagged. A dominant
  periodic component found by Fisher's test, and completing at least three
  cycles, is removed before the point components score; the spectral
  component still reads the original series.
- A level shift, a ramp and a spike an attacker spread over five rows defeat
  any per-row statistic (level-shift AUROC 0.03, spread-spike recall 0.4). A
  `collective` component scores a stretch of rows departing together (the
  standardised window mean at lengths 3 to 65, calibrated), in the fusion
  route (default on) and the batch job. Measured: recall 1.00 on all three.
- A weighted average diluted one decisive component (a spectral z of 15 beside
  three near 1 averaged to 3.3 and read "normal"); the fused score is now the
  larger of the weighted average and the largest component read as a maximum
  over the selected count, and the explanation says when a component carried
  the verdict.
- The detector tier's flag count is the `contamination` parameter by
  construction (measured slope 1.00 against it on data with a 5% anomaly
  share); the API says so and points to `conformal_alpha`, whose flag rate is
  evidence-controlled (0.047 at every setting). Two tier members (`rca`,
  `frequent_pattern`) carry no information on one-dimensional shapes (AUROC
  0.50) and three (`deep_svdd`, `energy_based`, `spiking`) carry the gain;
  the measured attribution is published.

The suite, its ledger and its page (`research/adversarial/served_routes.py`,
`served_routes_results.json`, `docs/DETECTION_QUALITY.md`) are seeded and
reproducible; `tests/research/test_served_routes_ledger.py` recomputes the
hash of every file that decides the numbers, pins the floors, and on the ML
lane reproduces the ledger to the digit. The fusion parity tests pin the
kernel both routes share (`tests/api/test_fusion_batch_parity.py`,
`tests/api/test_null_calibration.py`). Verdicts on wide or seasonal input and
on collective anomalies change: they were wrong.

**Fixed — three test modules switched the rate limiter off for the whole
process at import.** They now use a per-test fixture that flips the live
middleware and restores it, so later tests that assert on the limiter run
against it (`tests/conftest.py::rate_limit_off`). The fixture builds the
middleware stack under the process's own configuration before it patches
anything: patching the environment first would have made an unbuilt stack
construct the middleware disabled, recorded that as the original, and left
the limiter off for every later test
(`tests/api/test_rate_limit_fixture_isolation.py`).
Removing the leak exposed two modules that had relied on it: the TLP export
tests and the frontend-off route contract answered 429 when another module
in the same CI worker had drained the per-client bucket first (ten
failures on Core Tests, Python 3.13). They now switch the limiter off per
test as well; reproduced locally by draining the bucket ahead of each
module and re-checked across every API test module under the same
condition.

**Fixed — the score bimodality check refused a span below float resolution.**
`np.histogram` raised on scores such as 10000 and 10000 + 1e-13 (found by the
BCa property test); such scores are one value and are not bimodal. The guard that replaced the crash compared the bin width with the
floating-point spacing of `max(|low|, |high|, 1.0)`; the floor of 1.0 made the
verdict depend on the scores' scale, so a two-peaked sample of order 1e-16
read as one value while the same shape at order 1 read as bimodal. The guard
now applies `np.histogram`'s own edge test (neighbouring `linspace` edges must
differ), which is exactly as strict as the call it protects and scale-free
(`tests/core/test_threshold_confidence_interval.py`).

**Fixed — the face-detector error recommended an install that supplies no
detector.** `FaceDetectorUnavailableError` sent operators to the `[ml]` extra
for MTCNN, but no Mercury extra declares `facenet-pytorch`, so that install
left the engine without a detector; and it offered the Haar cascade XMLs on an
OpenCV 5.x `-headless` wheel, where `CascadeClassifier` itself is absent and
no XML can help. The message now reports the installed state (whether
`facenet_pytorch` imports, PyTorch, the OpenCV version, the classifier, the
cascade found) and only the remedies that apply: `facenet-pytorch` together
with `[ml]`; an OpenCV that carries the classifier (`opencv-python-headless<5`
or a 5.x `opencv-contrib` wheel); or, when the classifier is present without
bundled cascades, the XML itself. The Haar fallback now finds that XML in
`MERCURY_HAARCASCADE_DIR`, OpenCV's bundled `cv2/data`, or
`${DEEPFACE_HOME:-$HOME}/.deepface/weights` (where `docs/INSTALLATION.md` has
operators download it), so supplying the file is a real remedy
(`biometric_multimodal.haarcascade_directories`, `find_haarcascade`;
`tests/models/test_wiring_age_progression.py`). The installation guide says the
same.

**Added — one cross-platform lockfile, installed with `uv sync`.** `uv.lock`
resolves every extra for Linux, macOS and Windows on Python 3.11 to 3.14 (the
git-pinned `[pqc]` extra included). The Dockerfile builder and the Code
Quality, Type Checking, Core Tests and ML Tests lanes install with
`uv sync --locked` (uv pinned to 0.11.32 in both places; the lanes cache uv's
global hardlinked store keyed on the lockfile), the Workflow Hardening lane
runs `uv lock --check` so a stale lock is named on the pull request, and the
hardening gate treats `uv sync` as an engine install that must provision AMA
while leaving `uv pip install` outside the pip rules it does not run. The AMA
build script and composite action install through uv's pip front end when the
environment has no pip (`tests/test_uv_lock_contract.py`, the extended
hardening tests; the installation and contributing guides describe the path).
uv resolves one version per package across every Python where it can, and
TensorFlow (installable only below 3.14) caps h5py at a release with no 3.14
wheel, so the unified choice made the 3.14 lanes build h5py from source and
fail on the missing HDF5 library; a `[tool.uv]` constraint applies to the 3.14
fork only and `tests/test_uv_lock_contract.py` now checks that every binary
package in the lock has a wheel for each supported Python on Linux. The other
workflows still install with pip and are unchanged.

**Added — the AMA native backend installs from a prebuilt release archive
before it builds from source.** `scripts/fetch_ama_prebuilt.sh` downloads
`ama-cryptography-<ref>-<os>-<arch>.tar.gz` (the `ama_cryptography` wheels for
Python 3.11 to 3.14, each bundling the native library whose digest AMA signs
into it and tagged for the interpreter that built it) from the
`ama-native-<ref>` release, verifies it against the release's `SHA256SUMS`, and
unpacks it; `scripts/build_ama_native.sh` and the `build-ama-cryptography`
composite action install the wheel matching their interpreter and run the PQC
capability probe, and fall back to the clone-and-cmake build only when no
archive exists for the platform, no wheel matches the Python, or the library
does not load there. A failed checksum stops the install. The new
`ama-prebuilt.yml` workflow (manual dispatch with the AMA tag) builds the four
wheels on Linux x86_64 and aarch64, macOS aarch64 and x86_64 and Windows
x86_64, proves each one in a fresh environment, and publishes the release;
the Docker builder gained `curl` so the image build downloads too
(`tests/scripts/test_fetch_ama_prebuilt.py` drives the script against a
`file://` mirror and pins the shared archive contract).

**Fixed — the logging bootstrap tests read the developer's own log settings.**
`ensure_logging_configured` honours `MERCURY_LOG_LEVEL` and `MERCURY_LOG_FORMAT`,
and the tests of its text-format default asserted on the formatter without
clearing them, so a suite run with the documented `MERCURY_LOG_FORMAT=json`
failed while logging worked as configured. Every bootstrap test now starts from
the documented default and sets its own override, and the JSON configuration
has its own test (`tests/test_logging_utils.py`).

**Fixed — the Security Scan lane could run out of disk.** Its audit-target
environment installs the engine with the `[api]` extra, which resolves torch
and its CUDA libraries, beside the Semgrep environment and the engine's own
install; one run failed with "Disk quota exceeded" while installing torch. The
lane now frees the runner's unused toolchains first, as the test lanes do, and
installs the audit environment without pip's cache, which held a second copy
of the torch stack.

**Fixed — every API test ran against the process-wide rate limiter it was not
testing.** The limiter's token bucket is shared by every test that drives the
one `server.app`; a module's requests spent budget a later module in the same
CI worker found exhausted, and that module answered 429 before its own
assertions ran. `tests/api/conftest.py` now switches the middleware off for
every API test and restores it afterwards; the modules that assert on the
limiter keep it live with the new `rate_limited` marker, and the per-module
opt-outs are gone.

**Fixed — the calibration cache could hand a pre-exclusion archive to the
evaluation scripts.** An archive written before the label-derived exclusion
still holds the columns the labels are built from, and scoring it publishes a
circular number. `build_cache.py` now stamps every archive with its schema,
the scored columns and the excluded ones, and `cache_archives()` leaves out
any archive without the de-leaked schema until the event is rebuilt
(`research/engine_calibration/_workspace.py::ARCHIVE_SCHEMA`,
`tests/research/test_calibration_cache_archives.py`).

**Measured — the normalizer repair's 6.7-point NAB loss is not recoverable through the per-probe scale, and the attempt caught a bad adoption.** Repairing `_normalize_scores` to divide by a fit-time scale cost 6.7 points of NAB standard score (40.4 to 33.67). `research/math_arrest/harness_norm_scale.py` asks whether that is recoverable without touching alpha, and establishes first that it cannot be bought by alarming more: the conformal cut already honours its promise, with the false-alarm rate at or under alpha on every recorded file, so alpha is held at 0.05 throughout. The lever it does find is real but not free. Per probe a scale is monotone and moves nothing, but the 21 probes are averaged, so each probe's scale is its implicit weight in a nominally uniform fusion. Ten estimators were swept, and the scale turns out to be a dial along an AUROC / NAB-score trade-off on which the shipped p99 sits interior: `median_3mad` reaches NAB 49.91 (+16.23) and `p75_15iqr` 42.96 (+9.29) while improving ADBench AUROC, and `p75_30iqr` reaches the best ranking figure anywhere in the sweep (0.7742, +0.0248).

**Every one of them is refused, and the refusal is the deliverable.** The first bar was too weak in three ways and was strengthened before any adoption: it scored one corpus while the pins carry two, it never checked that `uniform` stays the best weighting (so `median_3mad` dropping `uniform`'s ADBench lead over `confidence` to -0.0019 was caught by luck rather than by process), and it aggregated over files. It also compared live values against the rounded pins, which made the shipped scale fail its own bar on 0.74937 against a literal 0.7494. Under the strengthened bar nothing clears, and the per-file criterion is what does the work: `mean_p99_iqr` improves every aggregate figure at once -- NAB AUROC +0.0049, ADBench AUROC +0.0050, NAB score +1.43 -- with the false-alarm rate under alpha and batch invariance intact, and would have been adopted under the weaker bar. Per file it improves or holds on only 2 of 4, regressing `exchange-3_cpc_results.csv` by 0.0134 and `speed_7578.csv` by 0.0105, both past the 0.01 limit. The aggregate gain was two files carrying two. Every other candidate regresses those same two files further, so the shipped p99 is the only estimator measured that costs no file anything, and it stays.

**Also recorded: a strengthening that was tried and thrown away as vacuous.** Multi-seed agreement was added first and then removed, because seeds 0-4 on the shipped scale return identical figures on every axis -- no probe consumes randomness that reaches a published number, so a seed-agreement criterion cannot fail for any candidate and would have added the appearance of rigour with none of the substance. Per-file agreement replaced it because it can fail, and it did.

**Fixed — the normalizer docstring overstated what the repair left untouched.** It said the AUROC figures are unaffected by the normalization choice. That holds per probe and not after fusion, which is why the repair moved NAB AUROC by 0.0087; the qualifier is now stated with the sweep alongside it.

**Security — the container gate's new zlib HIGH is accepted on measured grounds, after both remediation routes were shown ineffective.** CVE-2026-76844 was published against `zlib1g 1:1.3.dfsg+really1.3.1-1` between a green run on `main` and a push the same day, with no Dockerfile or dependency change. It is a real heap overflow in upstream's `gz_write()` (a failed `write()` leaves `strm.next_in` stale; fixed upstream by `madler/zlib@df84af25`), and Debian publishes no fixed version for trixie. Rather than assume, both routes were measured with the CI-pinned trivy 0.70.0 against a Debian 13.7 dpkg manifest: upgrading does not clear the gate, because the release-13 advisory carries no fixed version and Trivy therefore marks every installed version `affected` — sid's *fixed* `1:1.3.dfsg+really1.3.2-4` reproduces the same HIGH finding, which rules out suite-mixing, trixie-backports and a from-source rebuild on effectiveness rather than on risk; and elimination is impossible, because `libz.so.1` is in the NEEDED list of the CPython interpreter binary itself, so purging `zlib1g` stops `python3` from starting. The vulnerable `gz*` stream API has no caller in the image, and the four conditions behind that claim — no `gz*` symbols on CPython's `zlib`, `gzip` building its container over `compressobj`, no first-party `gz*` call or `libz` FFI load, plus a vacuity guard proving the detectors would catch a real caller — are enforced by `tests/security/test_zlib_cve_acceptance_conditions.py`. The CVE entered `.trivyignore` as a time-boxed, enumerated acceptance (`exp:2027-02-03`) with that evidence, and SECURITY.md's mirror was brought back in step: its aggregate counts read 4 and 7 against a nine-entry ledger and now read 10. Also recorded: trivy 0.70.0's DB attaches the wrong *Title* to this ID, printing a `webpack-dev-middleware` path-traversal summary; the package, release and status mapping are correct and NVD confirms the zlib attribution, so only the human-readable label is a Trivy metadata defect.

**Fixed — the SQLite CLI guard compared bare package tokens.** APT accepts
`sqlite3:amd64`, `sqlite3=3.46.1-1` and `sqlite3/trixie`, each of which would
have installed the shell behind CVE-2026-119586 while the acceptance check
saw a different token; the guard now strips the architecture, version and
release qualifiers before comparing
(`tests/security/test_sqlite_cve_acceptance_conditions.py`).

**Fixed — the test run could file test mail in, and purge, a developer's own
outbox.** `tests/conftest.py` set `MERCURY_MAIL_OUTBOX` with `setdefault`, so a
developer with the variable in their shell received every account test's
message in their real outbox, and the decouple purge tests, which inventory
that variable, could remove their mail. The run now assigns a per-process
temporary outbox unconditionally, and the decouple tests' isolated home clears
the variable so purge and snapshot see only the scratch tree
(`tests/test_decouple.py::TestMailerOutboxIsState`).

**Added — the console's `explain` runs on a connected server.** It sends the
fusion route and narrates the run through `/api/v1/voice/narrate`, the two
server calls the CLI's `explain` stands for, echoing the narration into the
terminal. Driving it found that every narration in the console was built from
the result's summary alone, so a series-level run (fusion, three-r, flagship)
with a fused score of 9.05 was narrated as "no anomaly detected, score 0";
the three places that narrate a result now read the server's response under
the summary, and the payload builder knows the fused score. Pinned by the
console self-test (8,326 checks).

**Fixed — four review findings on the agent's memory and the mailer outbox.**
A route that returns no verdict (`/detect/rca` reports root causes) was
mirrored into the agent's memory as a clean run with a score of zero; such
records stay in the export history and are no longer mirrored. The fusion
route left its threshold out of its recording payload and the detector tier
recorded it only inside `summary`, so those runs carried no threshold and no
severity band; both record it at the top level and the mirror reads either
place, with route-to-memory regressions
(`tests/api/test_route_memory_thresholds.py`). A development mailer outbox
pointed outside the XDG state tree by `MERCURY_MAIL_OUTBOX` was missing from
`decouple status`, `detach` and `purge-work`; it is inventoried and archived
(`tests/test_decouple.py`). The legacy-cache rejection test wrote a payload
JSON could not serialise, so nothing was cached and its assertion was vacuous;
it now writes the old text-timestamp shape, reads it back, and then checks the
frame reader refuses it.

**Fixed — the README ledger guard holds in a depth-1 checkout.** The test that
flags an ADBench ledger quoted past the detector that produced it dated the
detector with `git log`, which in CI's single-commit checkout names the merge
commit for every file, so the ML lane failed on a ledger that is in fact
current. The ADBench harness now records the detector file's git blob hash
(`detector_blob`) beside `detector_commit`, the two committed ledgers carry
the blob of the detector at the commit each records, and the guard pairs by
content first, keeping the commit pairing as a second witness only where
history exists. Reproduced in a depth-1 clone: the old guard fails, the new
one passes (`tests/docs/test_readme_claims_match_artifacts.py`).

**Fixed — a user has the whole engine from the front end.** Driving every
route card, terminal command and Mass Run capability against the live server
with a real account found six gaps, each now closed and covered by the
console's self-test (`:diag`, 8,322 checks) or the page tests:

- A dashboard-created API key held only `read` and `detect`, so the export
  routes were unreachable from the front end. The dashboard's key form now
  offers exactly the server's self-service whitelist (`read`, `detect`,
  `export`; never `write`, `delete` or `admin`), defaults pre-checked, and the
  key table shows what each key holds (`tests/api/test_frontend.py`).
- The alert stream was opened once and never re-subscribed when an API key
  was entered; the engine scopes alerts to the key's principal, so a user's
  own detections never reached the dock. The console re-subscribes when the
  key changes, and a Mass Run under a key now lands its alerts (24 alerts
  folded into one proactive entry, verified in Chromium).
- Console requests to the same origin carry the dashboard session's CSRF
  token, so a user logged in at `/login` reaches the account routes from the
  console (`/auth/me`, key listing and creation verified); a 401 on an
  account route now says that a session, not an API key, opens it.
- Free-form JSON routes (concealed-information test, vital signs, the
  adjudication ledger) opened with an empty body; they now open with a
  canonical payload, the Mass Run capability's where one exists and the
  route's documented example otherwise, and a body already typed is kept.
- `PUT` with an empty body sent no body and drew a 422 from the model
  metrics route; the body the schema declares is sent, empty or not.
- `GET /api/v1/voice/alerts` from the form waited on a stream that never
  ends; the card now reports the dock's live subscription and recent events,
  and a `limit` sends the bounded one-shot read. The request budget also
  covers the body read, so no response can hang the form.

**Fixed — the development mailer delivers a usable link.** `ConsoleMailer`
logged each message, and once the engine's INFO records became visible the
log formatters' credential redaction turned every verification and reset link
into `token=<redacted>`, so a locally registered account could not be
verified. The mailer now files each message as an owner-only text file (plus
the HTML part) in `MERCURY_MAIL_OUTBOX`, default
`~/.local/state/mercury-agent/outbox`, the state root `decouple status`
already inventories; the log names the file and never carries the body, and a
write failure surfaces to the delivery audit like an SMTP outage. The test
suite files into a temporary outbox. Verified in Chromium: register → open the
outbox → verify → log in → create an API key
(`tests/api/test_maintenance_and_mailer.py`).

**Fixed — three console API-form defects found by driving every route card
against the live server.** The default sample series had 5 points, so the
tier form's own default payload was refused with a 422 (the route needs 8);
the series is now 16 points, sized up further by a route's `minItems` (which
the generated schemas now carry), and long enough for its spike to score past
the univariate z-score threshold. The hazard form sent `{format}` alone and
got a 400: the route's one-of (a prior `diagnostics`, or `hazard` + `arrays`)
is checked before sending, and the loaded file supplies the arrays the chosen
hazard's Mass Run capability builds, refusing an unusable shape with the
terminal's reason. Error results carried the headline "response received";
the headline is now the server's reason, naming the field for a validation
error. The self-test covers all three, and its brand-mark count is scoped to
the workspace so `:diag` typed as the opening dialog closes cannot misreport
(`scripts/build_console_port.py`, `scripts/build_console_catalog.py`).

**Fixed — the engine's own log records reach the server log.** Under
`uvicorn omni_mercury_engine.api.server:app` only uvicorn's loggers were
configured; the `omni_mercury_engine` logger had no handler, so every record
below WARNING was dropped: the warm-up confirmation, the maintenance sweep,
and the verification link the development mailer logs in place of sending
email, which left a locally registered account impossible to verify. The
ASGI lifespan now calls `utils.logging.ensure_logging_configured()` once per
worker: it installs the package's handler at uvicorn's `--log-level` (or
`MERCURY_LOG_LEVEL`, with `MERCURY_LOG_FORMAT=json|text`) and leaves a host
that configured logging itself alone. The text format is coloured only when
stdout is a terminal, so redirected and container logs stay plain. The
previously unread `OMNI_LOG_LEVEL`/`OMNI_LOG_FORMAT` names in `.env.example`
and `docker-compose.yml` were replaced by the live ones
(`tests/test_logging_utils.py`, `tests/api/test_server_comprehensive.py`).

**Fixed — the alert bus cannot accumulate callbacks.** Each publish scheduled
one loop callback per subscriber and the callback retained its entry, so a
recording thread that outpaced a busy subscriber loop grew an unbounded
backlog despite the bounded queue. Each subscriber now stages entries in a
bounded, lock-guarded buffer and holds at most one pending drain callback;
overflow drops the oldest staged entry and is counted
(`tests/narrative/test_alert_bus.py`).

**Fixed — a refused calibration rebuild cannot leave a stale archive
selectable.** `build_cache.py` now quarantines an event whose loader refuses a
de-leaked matrix: the old `.npz` is removed and a marker is left, and
`cache_archives()` skips any event with a marker even if an archive is copied
back (`tests/research/test_calibration_cache_archives.py`).

**Fixed — narrative tests collect on an engine-only install.** Two narrative
test modules imported the optional `[api]` extra at module level; they now
skip their route-dependent half without it instead of failing collection.

**Fixed — the console's boot progress bar animates again.** The reduced-motion
patch matched a substring of the prototype's selector and left
`html.mn-reduce-motion` in front of the generated `@media` rule, which
browsers discard; the whole selector is matched now and the media rules sit
at the top level, and the bar's inline animation honours reduced motion the
way the halo's already did, so the stylesheet is no longer the only guard
(`tests/api/test_console_catalog.py`; verified in Chromium under both
preferences).

**Changed — the `[api]` extra installs torch.** The served API exposes the
neural detection routes (fusion, neuro-symbolic, flagship) beside the
statistical ones, but `pip install "mercury-agent-global[api]"` did not bring
torch, so those routes refused at request time until the operator found the
`[ml]` extra. `[api]` now pins `torch>=2.2.0` at the same floor as `[ml]`; one
install onboards a full server. `[ml]` is unchanged.

**Changed — the changelog records what changed, not the correspondence around
it.** Pull-request and Dependabot numbers, review-tool names, AI session
identifiers and branch-relative wording ("this branch", "this PR") are removed
from every entry; the technical content of each entry is unchanged.

**Added — proactive alerts reach the console.** The agent panel has carried an
alert dock since its first release and the voice API's docstring promised
"proactive alerts", but no server route produced any: the console's
`alertsUrl()` returned `null` and the dock read "not served by this engine".
`GET /api/v1/voice/alerts` now streams server-sent events, one `alert` per
detection run that flags a sample, as it is recorded, bounded to the caller's
own runs exactly as the voice's answers are; keep-alive comments every 15
seconds; `?limit=N` ends the stream after N alerts for scripts. Runs recorded
in-process arrive through a new alert bus the retriever publishes to
(`narrative/alert_bus.py`); runs recorded by another worker on the same
`MERCURY_AGENT_MEMORY_PATH` are picked up from the store by row id. The
console consumes the stream with a streaming fetch so the API key travels as a
header (a browser WebSocket cannot carry one), the agent header now shows the
dock state the logic always computed but never rendered, and the console's
own diagnostics assert the new contract (`tests/narrative/test_alert_bus.py`,
`tests/api/test_voice_alert_stream.py`, `tests/api/test_console_catalog.py`).

**Changed — the conversation surfaces show what the agent knows.** The
console's agent status rows now include the caller's runs in memory and the
alert stream's state, which `/status` reports; the CLI `voice` session prints
each answer's sources and the follow-up prompts, its `/clear` really clears
the conversation (a new `MercuryVoice.clear_conversation`) and reports how
many turns went, and its help lists the questions the voice answers
(`tests/test_cli_voice_repl.py`). Follow-up context is keyed by principal
and session, so two browsers of one account (every unauthenticated console
is the same principal) never share a "why?".

**Changed — anonymous account pages no longer probe the session.** Login
sets a readable CSRF cookie beside the httpOnly session cookie, so a visitor
without it cannot be signed in; the header nav and the dashboard guard now
read that cookie first and call `/api/v1/auth/me` only when it is present,
removing a 401 from every anonymous page view and one round trip from the
dashboard's redirect to login.

**Fixed — batch fusion and synchronous fusion agree.** `POST
/api/v1/detect/fusion` normalised weights of 0.35/0.25 for the statistical
and temporal components (0.583/0.417) while the batch job's `fusion` method
hard coded 0.6/0.4, so eleven zeros followed by a 10 at sensitivity 0.52 was
an anomaly over HTTP (3.442 against 3.44) and clean as a batch job (3.437).
On multi-feature input the synchronous route also standardised every value
against one global mean and standard deviation and differenced the flattened
matrix (a "step" ran from the last column of one row to the first of the
next) while the batch path standardised per column and stepped row to row,
so `[[0, 1000]] * 11 + [[10, 1000]]` was anomalous as a batch job (3.442) and
clean over HTTP (1.418). Both routes now take their components from one
kernel (`api/fusion_weights.py`: per-feature standardisation, row-to-row
steps, per-feature spectral scores, one weight table), and
`tests/api/test_fusion_batch_parity.py` pins the series and matrix that
exposed it, an isolated spike, a constant-column invariance, and the bound that a
batch peak never exceeds the synchronous score.

**Fixed — a running worker recalls runs another worker recorded.** The
retriever loaded `MERCURY_AGENT_MEMORY_PATH` only at construction, so a worker
started before a run was recorded answered "no runs in memory" until it
restarted, even though its alert stream polled the store. Every store-backed
read path now merges the rows appended since the worker's cursor
(`tests/narrative/test_detection_memory_store.py`).

**Fixed — series-scoped verdicts are not described as sample positions.**
`/detect/fusion`, `/detect/three-r` and `/detect/flagship` record one flag for
the whole input; the memory mirror turned that into "1 of 1 sample flagged
(sample 0)". The record now carries the verdict's scope, the input size and no
positions, and every answer and alert says "whole-series verdict anomalous
over N samples (no sample positions)".

**Fixed — synthetic frames are quarantined from de-leaked measurements.** The
energy loader's synthetic historical series builds solar-wind speed, density
and flare classes from the same Kp profile that builds the `kp >= 7` labels,
so dropping the five declared Kp columns still left a label-derived matrix.
A loader now marks the frames it manufactures (energy, pandemic), the mark
survives the loader cache (whose JSON writer had been dropping every frame
with a tz-aware timestamp column, so that cache never persisted),
`engineer_scored_features` refuses such a frame with
`SyntheticDataNotDeleakable`, and `deleak_impact.py` quarantines the event
instead of writing a "de-leaked" archive (`tests/loaders/test_synthetic_quarantine.py`).

**Fixed — the calibration scripts no longer read the de-leak archives.**
`deleak_impact.py` writes `*.deleak.npz` (`X_full`, `X_scored`, `y`) into the
same cache as the per-event `*.npz` (`X`, `y`); `eval_offline.py`,
`measure.py` and `method_bakeoff.py` globbed both and failed at `d["X"]`.
They share one filtered listing (`tests/research/test_calibration_cache_archives.py`).

**Fixed — the voice routes kept accounts apart only in `speak`.**
`GET /api/v1/voice/history` returned every account's turns to every caller and
`/status` counted them all; each turn now carries its principal and both
routes answer for the caller alone. `/greet` listed five fixed phrases
whatever the server had; it now states what this installation really has (the
manifest's detector count and families, the loader count, the caller's runs in
memory, the gates, the alert stream).

**Added — a conversation layer that answers every console question from the
record or the code.** Seventeen of the thirty-one questions a first session
asks the agent (greetings and identity, "what threshold did you use?", "which
samples were flagged?", "show me the last 3 runs", "summarize threat posture",
"recommend an action", "what should I run next?", "how do I collect data?",
"why did it clear the gate?", definitions of severity, z-score, AMA and
FINDΩYOU, "run detect on sample.csv", "compare today with yesterday", and the
follow-up prompts the agent itself suggested) came back as the generic apology
or as the wrong answer. `narrative/conversation.py` now classifies a question
once and builds the answer from one of three named sources: the detection
record (per-run threshold, flagged indices, method, score, time; the newest N
runs; severity-band posture; two-window comparison), the code (the detector
manifest, the CLI's own command table so a recommended command always exists,
the loader package's domains, the API's severity bands and threshold formula,
the ethical gates, the decoupling switches), or the narrative engine (full
reasoning chain, uncertainty disclosure, dossier, recommendations and urgency
for the newest flagged run). A command request is handed back as the exact
command, never executed. The empty-search fallback is a menu of what can be
asked instead of an apology, and no answer suggests the question just asked.
Short replies that lean on the previous turn resolve against that account's
own last answer and nobody else's: "why?" and "tell me more" narrate or
expand the run just discussed, "and yesterday?" re-asks the count for another
period (bounded, so yesterday no longer includes today), "run 2" or "the first
one" picks from the last list shown, and a term's "tell me more" moves to the
related term. The console's quick prompts and the help text list the new
questions (`tests/narrative/test_voice_conversation.py`, which also pins the
quoted bands and formula to `api/server.py` and the loader domains to the
loader classes).

**Added — durable agent memory (`MERCURY_AGENT_MEMORY_PATH`).** The detection
log the agent answers from lived only in the process, so every restart emptied
it and the first answer after a deploy was always "no detection runs are
recorded yet". When the variable names a SQLite file (the interpreter's own
`sqlite3`, fixed statements only, no virtual tables), every logged detection is
appended there with its domain and principal and a new process loads the newest
1,000 at start, so recall survives restarts and is shared across workers on the
same path. Unset, behaviour is unchanged. The location is listed by
`mercury-agent decouple status` and removed by `purge-work`
(`tests/narrative/test_detection_memory_store.py`).

**Added — the agent explains its newest run and describes the engine from code.**
"Explain the latest detection" (an example the voice route has documented
since it was written) narrates the newest recorded run of the asking account
through the narrative engine: the run's verdict, peak score, severity fraction
and facts (method, flagged samples, threshold ratio) become the detection the
engine synthesises, behind the same ethical gate as any detection and without
re-logging it. "Which detectors do you have?" answers from the detector
manifest with counts by family and input type; naming a family or input
("Can you detect earthquakes?", "Which detectors handle images?") lists the
matching detectors with their one-line descriptions. "What are your
capabilities?" reports every self-model capability as probed on this machine,
with the reason when one is unavailable or switched off. Help names these
questions instead of a fixed paragraph, and the console's agent quick prompts
include them (`tests/api/test_voice_engine_grounded_answers.py`).

**Fixed — seven defects in the console and crypto changes.** The
agent's memory is scoped by principal: a detection is recalled only by the
account that ran it (``anonymous`` for unauthenticated runs), and the voice
route resolves an optional API key or JWT so an anonymous caller can no longer
read another account's detections. The core ``/detect/univariate`` and
``/detect/multivariate`` routes record an authenticated caller's run under that
account, where the caller's own export history finds it. Legacy
``pbkdf2_sha256`` password rows are verified at the fixed 32-byte digest
length (deriving to the stored length let a truncated row verify a wrong
password with 1-in-256 odds). The self-model probes every switch the FINDΩYOU™
bridge needs, so ``ama_optional`` off is reported as the operator decision it is
rather than as a grounded capability. The console's boot progress bar yields to
both reduced-motion preferences. Two more label-derived channels are excluded
from the scored matrices: the earthquake loader's rolling ``b_value`` (its
window includes the current event's magnitude) and the volcanic loader's
``days_since_last_change`` (reset by the alert level). The earthquake de-leak
figure was re-measured on events re-fetched from USGS with the current
detector: **0.8995 → 0.3276 (−0.5718)**, still far outside the no-exclusion
control band; the measurement is now reproducible
(`research/engine_calibration/deleak_impact.py`) where the artifact had been
produced by hand.

**Added — the agent answers questions about its detection record with the
record, not a search score.** "How many anomalies today?" (also per hour,
yesterday, week, month, or all) is answered with the run count, how many
flagged at least one sample, the per-method tally and the highest-scoring run.
"What was the latest anomaly?" names the newest flagged run and how many clean
runs followed it. Status reports what memory holds and when the newest run was
recorded. Every recalled run is described as a run: method, how long ago, how
many of how many samples were flagged and which, and the peak score against its
threshold with the API's severity band ("multivariate run 3 minutes ago: 1 of
60 samples flagged (sample 42); peak score 6.21, 1.8x the 3.50 threshold
(medium)"). The memory mirror now keeps the threshold and the first flagged
sample positions; an empty window or an empty memory is stated as such, with
what memory covers. The console's agent quick prompts include the two
questions. Verified in Chromium against the live server
(`tests/api/test_voice_recent_detections.py`).

**Fixed — the API rate limiter metered the console's own assets.** One
console page load is 21 same-origin requests (17 scripts and stylesheets, 3
images, the page) against a default burst of 20. Measured against
`mercury-agent serve`: 30 concurrent asset requests from one address answered
10 of them 429 and left the bucket at zero with `Retry-After: 60`, so a reload
within a minute came up with its scripts refused. Assets under the frontend's
`/static` mount are exempt from the API budget; the pages and every API route
stay metered (`tests/api/test_static_assets_not_rate_limited.py`).

**Fixed — the agent could not recall a detection the console had just run.**
"Have there been any anomalies recently?" classified as `UNKNOWN` and the
detection-log scorer matched only the literal token `anomaly`, so the reply
apologised with the log full; and the console asks under its CONTEXT domain
while `/detect/univariate` and `/detect/multivariate` carry no domain, so the
strict domain filter hid every record the console produced. Intent and
relevance now match stems (`anomal`, `flag`, `alert`, `recent`, ...), and
`general` is a wildcard on either side of the domain comparison. Reproduced
and verified end to end in Chromium: the same question now answers with the
detection record (`tests/api/test_voice_recent_detections.py`).

**Fixed — five console controls had no accessible name.** The command
prompt, the agent composer, the menu filter, the MCP tool-arguments field and
the command-palette search were named only by a placeholder (WCAG 2.1 SC
1.3.1 / 4.1.2). They carry `aria-label`s through the port's template patches;
`tests/api/test_console_inputs_are_labelled.py` walks the compiled template so
the property holds across rebuilds. The connection dialog's BASE URL is
read-only and its CORS hint is replaced: the served console talks to the
origin that serves it and the page CSP (`connect-src 'self'`) allows no other,
so an editable field and a `MERCURY_CORS_ORIGINS` hint described a
configuration the page cannot use. The landing page now links the console.

**Fixed — `decouple detach --save` merged two state directories into one.**
`~/.local/state/mercury-agent` and `~/.cache/mercury-agent` share a basename
and were archived under one member root, so an extraction overwrote one with
the other. Every item now has a unique root recorded in the manifest.
`decouple uninstall --remove-source` applies purge's refusal guard before
deleting a checkout.

**Changed — operational HMAC, HKDF, SHA-3, PBKDF2 and constant-time compare
now run on AMA Cryptography, not OpenSSL.** The at-rest secret sealer's
HKDF-SHA-256, the constant-time helpers' HMAC-SHA-256, the audit chain's
HMAC-SHA3-256 and torn-fragment SHA3-256, legacy password-row PBKDF2 and every
`hmac.compare_digest` in those modules used the stdlib, which on the shipped
OpenSSL-linked interpreter is `openssl_sha256`/`openssl_sha3_256`, OpenSSL's
HMAC and `CRYPTO_memcmp`. They route through `security/ama_hmac.py` to AMA's
native, ACVP-validated primitives with no stdlib fallback. Outputs are
byte-identical (pinned against the stdlib on fixed vectors and RFC 5869 A.1),
so sealed secrets, audit chains and stored password rows written before the
change open and verify unchanged. With `hashlib`/`hmac` tripwired, every
module still reproduces its reference values
(`tests/security/test_operational_crypto_routes_through_ama.py`). Still on the
stdlib, each for a stated reason: scrypt password hashing (AMA ships no
scrypt; a switch to AMA argon2id changes the stored row format and needs an
owner decision), TOTP's HMAC-SHA-1 (RFC 6238's default; AMA ships no SHA-1),
and the streamed SHA3-256 of whole audit files (AMA exposes no incremental
hash; the 100 MB rotation size rules out a one-shot read).

**Added — Mercury Console at `/console` (`MERCURY_FRONTEND_ENABLED=true mercury-agent serve`): the design prototype, served.**
`scripts/build_console_port.py` builds the console from
`docs/design/mercury_console/design/` verbatim: the prototype's template is
precompiled to a JSON tree (`template.js`), its logic class is wrapped as a
plain script (`logic.js`), its modules ship as written, and `dc-runtime.js`
(dependency-free) renders them with the prototype runtime's semantics, writing
styles only through the CSSOM. Every deviation is a named patch that must apply
exactly once: same-origin engine, no simulated engine, chart styles as SVG
presentation attributes, generated request schemas, no invented confidence or
clipped scores, server severity bands, served image paths. The prototype's own
self-test (`:diag`) runs against the live server: 8,295 checks, 0 failures (3
warnings: the two simulated-engine suites are skipped, one timing note).
Chromium: no JS errors, no CSP violations. The catalog (`catalog.js`) is
generated from the code by `scripts/build_console_catalog.py`; both generators
are staleness-gated by `tests/api/test_console_catalog.py`. CSP: `/console`
alone gets `img-src 'self' data:` (`CONSOLE_CSP`).

**Fixed — 22 `--help` examples named a binary that does not exist.** `serve`,
`voice`, `stream` and four `physics` commands documented `mercury ...`; the
package installs only `mercury-agent`. With them corrected, `cli.py` documents
31 examples across 10 commands, all parsed by click in
`tests/test_cli_docstring_examples_parse.py`. `detect --detector` is now a
`click.Choice` of the engine's detectors plus `fusion`.

**Fixed — `/detect/univariate` and `/detect/multivariate` recorded nothing.**
Every route in `routes/detection.py` records through `record_detection` (export
history, alert egress, agent memory); these two did not, so the console's
default `detect` path was invisible to history, alerts and the agent. The
start-up warm-up's synthetic calls are excluded.

**Fixed — narration invented "confidence 50%" and "severity 0%" for detections that reported neither.**
`NarrativeEngine.synthesize` defaulted confidence to 0.5 and severity to 0.0,
and the narrate route's fallback did the same. Unreported values now stay
unreported: `ConfidenceLevel.UNREPORTED`, "Detection confidence: not reported",
urgency `unassessed`, verification-first recommendations, and
`NarrativeResult.confidence_score` / `VoiceResponse.confidence` are `None`.
`POST /api/v1/voice/speak` reports a missing confidence as `null` instead of
filling 0.8.

**Fixed — `POST /api/v1/voice/narrate` returned 500 for plain-text reasoning
steps** (`reasoning_chain: ["1 of 60 flagged"]`, which the console sends on
every run). Unscored steps are no longer reported as "confidence: 50%".

**Fixed — `decouple purge-work` / `uninstall` could delete the home directory.**
Env-pointed state locations were handed to `shutil.rmtree` verbatim, so
`MERCURY_DATA_DIR=~` made a confirmed purge delete the home directory
(reproduced on a sandboxed `HOME`); `/`, `.` or the Python prefix were no
safer. Such locations are now listed as refused and never deleted or
archived wholesale, symlinks are unlinked rather than followed, and
`state_inventory()` no longer walks a refused location (it used to walk `/`
and crash on the first unreadable directory).

**Fixed — the agent answered "what happened recently?" with the oldest detection, and with severities over 100 %.**
Every anomalous detection-log entry has the same relevance, so the stable sort
kept log order and the voice reported the oldest record; ties now break
toward the newest. The memory mirror stored the raw max score as `severity`,
which the voice prints as a percentage (a z-score of 9 read as 900 %); it now
stores the score-to-threshold ratio on the API's own severity bands (1.0 is
the critical boundary) plus the band name, and no longer stores a constant
`confidence` the detectors never produced. The console's severity bands were
corrected to the server's (1.5 / 2.5 / 4.0).

**Fixed — `mercury-agent detect -d <unknown>` reported "no anomaly".** The engine
skips detector names it does not hold, so a typo printed `"is_anomaly": false`
and exited 0. It is now a usage error listing the valid names.

**Fixed — the sovereign crypto tier ran on OpenSSL.** `crypto/sovereign.py`
(the AMA-down fallback) used `hashlib`/`hmac`, which dispatch to OpenSSL on an
OpenSSL-linked CPython. It now uses the interpreter's built-in HACL* SHA-2, a
repository-owned RFC 2104 HMAC and `_operator._compare_digest`, and fails
closed if the interpreter has no built-in SHA-2.

**Fixed — weather parses invented readings.** OpenWeatherMap reported absent
fields as 0 °C, 1013 hPa, 10 km and UV 0 (and an absent condition as clear
sky); NOAA reported an absent temperature as 0 °F and sent half the compass
(NNE … NNW) to north. Absent is now `None` and the compass has 16 points.

**Added — Mercury Console design reference (`docs/design/mercury_console/`).**
The browser console prototype and its handoff notes (`README.md` §20 lists what
was added after the original handoff): a presence figure traced from the supplied
reference image and driven by real request events, a settings dialog whose six
controls all act, world map and stacked-histogram charts, a batch lifecycle strip,
and narrow-width fixes (wrapping top bar, overlay agent panel). It is design
reference only: not packaged, not served, no engine code changed. The prototype's
`:diag` self-test runs 8,740 checks in a browser against its simulated engine.

**Added — decoupling switches (`mercury-agent decouple`, `omni_mercury_engine.decouple`).**
The three kin systems can now be pulled apart by the operator with real,
read-at-the-surface switches: `findoyou on|off` (the FINDΩYOU™ evidence
bridge refuses to import and the self-model reports it as disabled by the
operator) and `ama-optional on|off` (the AMA-backed surfaces Mercury can run
without — `AuthKeyManager` and the bridge — refuse with a named reason).
State persists in `~/.config/omni_mercury/decouple.json`; `MERCURY_DECOUPLE_<NAME>`
pins a switch for one process. `status` reports switch state, the AMA build's
native primitives, and every location Mercury writes work to. `detach --save`
archives all of it with a SHA-256 manifest and removes nothing; `purge-work`
and `uninstall` are dry-run until given their exact confirmation phrase.
What is not a switch is stated plainly: the AMA core gate is mandatory for the
engine to import (pinned by `tests/test_pqc_startup_gate.py`), and the only
total decoupling from AMA is `uninstall`. Covered by `tests/test_decouple.py`.

**Fixed — the conversational agent now remembers detections.**
`KnowledgeRetriever.log_detection` and its detection-log search existed, and
`record_detection` recorded every API detection, but nothing connected them:
"have there been any anomalies?" was answered from an empty log on every
deployment while the voice route advertised a `detection` intent.
`record_detection` now mirrors each record into the live voice's retriever
with the fields the search reads (`tests/api/test_voice_recent_detections.py`).

**Fixed — `POST /api/v1/voice/speak` carried a thinned copy of what the voice said.**
`MercuryVoice.speak` returns a `VoiceResponse` whose fields are named
`intent_understood` and `sources_cited` and which also carries a reasoning
summary, suggested follow-ups, an uncertainty note and the memory-search
counters. The route read `intent` and `sources` off that object — attributes
it never had — so every live reply reported intent `unknown` with no sources,
and the rest was dropped. `VoiceResponseModel` now carries `reasoning_summary`,
`search_performed`, `search_results_count`, `suggested_follow_ups`,
`uncertainty_note` and `response_time_ms` (all optional; existing clients are
unaffected), and the route normalises both the narrative voice's object and the
fallback voice's dict through one path (`tests/api/test_voice_speak_passthrough.py`).

**Fixed — browser clients could not read `X-Correlation-ID` or `Retry-After`.**
CORS exposed only the three `X-RateLimit-*` headers, so a browser client saw no
correlation id and no `Retry-After` on a 429, although the API sends both.
Both are now exposed (`tests/api/test_cors_expose_headers.py`).

**Fixed — `POST /api/v1/batch/detect` scored every sample against itself**
(see commit `ec828fe`): one-feature rows always scored 0.0 and no sample was
ever compared with the data set. Batch now uses data-set statistics with the
synchronous routes' formulas; `neurosymbolic` is refused at submission
instead of silently approximated.

**Added — measured σ-threshold ablation replacing the deleted `ab_sigma`
theater (`benchmarks/sigma_threshold_ablation.py` +
`sigma_threshold_ablation_results.json`).** The deleted script claimed
"σ=0.96 beats 0.93" from a formula with the conclusion built in. The real
measurement — the trained gate over the committed hand-justified
discrimination corpus, with deterministic-floor refusals counted
threshold-independently — answers that question differently: 0.93 and 0.96
are operationally identical (27/27 benign pass, 52/52 hostile block at both;
the floor alone blocks 43/52 hostile cases), separation holds across the
whole 0.90–0.995 range, and benign passes only begin failing above τ=0.999.
The artifact states what it does and does not measure (gate operating points,
not end-task F1 — no committed experiment measures that). A real
Lyapunov-constant A/B (what `ab_dominance` faked) would need the fusion
training loop with λ varied and remains an open task rather than a claim.

**Fixed — fabrication audit, continued: theater benchmarks removed, docs trued
to committed evidence, remaining disclosures landed.** Seven benchmark scripts
that scripted their own results were deleted (none referenced by code):
`ab_dominance`/`ab_sigma` wrote per-epoch F1 curves toward hardcoded targets
plus noise and printed "BENCHMARK PASSED"; `ethical_scalar_experiments` drew
execution time, memory and accuracy from RNG; `ethical_scalar_validation`
separated random toy data by construction and printed "VALIDATED FOR
PRODUCTION DEPLOYMENT"; `comprehensive_equation_optimization` scored toy
formulas with noise added "for realism" and `quick_benchmark_validation`
consumed its output as "the optimized config"; `plot_mercury_training` drew
invented calibration/memory/uncertainty curves. The two engine comments citing
`ab_dominance.py` as "the measured comparison" behind the Lyapunov constant
now state the truth (design target per MATH_SPEC §2.2). The hand-entered
`mercury_agent` row (0.85/0.80) in `benchmarks/baseline_results.json` is
removed and its provenance note states the peer rows are uncited
approximations. The seven-axis runner and its committed artifact/docs section
now disclose the synthetic workload. Docs trued to committed artifacts: the
stale 2026-10-14 headline (0.8545/0.9052/0.6425, ADBench 0.8515/0.6415) is
replaced by the committed 2026-10-26 run (0.8610/0.9077/0.6476, ADBench
0.8571/0.6443) across README, BENCHMARKS, LIVE_DATA_VALIDATION, DATASOURCES,
SELF_IMPROVEMENT_LOOP, ROADMAP and docs/index; the README spot-check table is
regenerated from a real 2026-10-28 run (Mercury ranks #4 of 5 on it — stated
plainly) and its unsourceable TranAD/MAAT "published reference" rows are
removed; the calibration block now quotes the committed 55-dataset run; the
hazard table is rebuilt from `research/engine_calibration/results.json` (its
unsourced Flood row removed); the NAB detection-tier tables are rebuilt from
the committed 30-series run, which reverses the old claim — consensus (0.635)
beats the best member, the unsupervised average (0.604) does not;
ARCHITECTURE's unsourced "empirical validation" percentages and λ_max ≈ 0.87
claims are withdrawn and its stale scale copies defer to the CI-gated README
block; perf/memory/install/PQC-coverage claims are labelled indicative or
CI-artifact-sourced; the live demo discloses its synthetic streams. Remaining
adjudications: the chronos adapter's fail-closed raise was caught by its own
except (any failure became score 0.0 "normal") — it now propagates; the legacy
three-R "dominance proof" table and docstring now label the 0.92 figure an
aspiration; single-class folds in `realworld_benchmark`/`lstm_ae` metrics are
NaN-excluded instead of entering means as 0.5/0.25; the unfitted statistical
ensemble diagnostic returns None instead of a made-up 0.5 and labels its
synthetic-OOD basis; MSDS (a synthetic stand-in sharing the real TranAD
dataset's name) now goes through the `check_synthetic_allowed` gate; the
tsunami loader marks reconstructed BPR series on the frame;
`communications_it` raises on missing inputs instead of inventing nominal
traffic; fixed confidence literals across infrastructure/narrative/
distributed/hindsight/vibration/code-analysis outputs are labelled as
uncalibrated rule priors (and the operations power estimate drops its 1.0
confidence to 0.3); energy_dams renames its count formula to
`heuristic_risk_index`; INTEL_VALUE_METRICS and the detector-evidence seed
disclose their authored-workload/non-committed-harness provenance; and the
in-session gwo/dimensional measurements are marked CHANGELOG-recorded, not
artifact-backed.

**Fixed — repository-wide fabrication audit: engine code that invented
measurements now measures, discloses, or fails closed.** A four-auditor sweep
of the whole repository found fabrication beyond the chart generators; every
confirmed finding in `src/` is fixed at the root, with the ratifying tests
rewritten to pin the honest contract:
`TemporalNeurosymbolicEngine.predict` returned uniform RNG draws as anomaly
scores when torch was absent (now raises);
`EthicalAutonomyGovernor` manufactured its "statistical validation (p<0.05)"
by t-testing RNG draws around the baseline (now a real one-sample t-test when
the caller supplies a score sample, else None) and its "Fairlearn-compatible"
bias audit split the array in half, ignored the context, and claimed
`mitigation_applied` (now real demographic parity over caller-supplied
protected attributes, `audit_performed=False` otherwise, and the
mitigation field can only be False);
`load_usgs_earthquake_catalog` trained on RNG-synthesized spectrograms
labelled "real (USGS catalog)" with every label 1.0 because the query floor
and the label threshold were the same value (now labelled synthetic-from-real-
events, with a separate `label_magnitude` so both classes exist);
`ChaosMultivariateFusion` moved its decision threshold by an RNG draw on every
call (now a deterministic score-seeded logistic map, documented as a
heuristic); `SimulationModule` fed RNG padding columns into its anomaly score
(now zero-padded, with a declared `anomaly_score_basis`);
both Lyapunov `verify_lyapunov_stability` fallbacks returned the configured
target rate as the "measured" rate with `is_stable=True` on degenerate windows
(now `(False, 0.0)`); the Phase-6 coherence audit silently validated a
synthetic reference curve (now named in `fusion_scores_source`);
the conformal binary fallback invented 0.7/0.3 scores plus RNG noise and
mapped IsolationForest's normal class to anomalous (now degenerate {0,1} with
correct polarity); `evaluation/baselines.py` is relabelled an UNVERIFIED
transcription, its nine untraceable `paper: "Various"` rows are removed
(NSL-KDD now empty pending citations) and every comparison carries the caveat;
`IntegratedAnomalyDetector` registered RNG "USGS/NOAA" sources by default with
confidence 0.9 (now opt-in, confidence 0.0, `metadata.simulated`, and the
factor line counts them separately); the marine loader cached synthetic
fallback frames under the real cache key and logged them as "fetched" (now
never cached, marked `is_real_data=False`); the Prometheus exporter served
hardcoded 0.0 counters and the retired 0.99 benevolence floor (now unavailable
metrics are exposition comments, the advisory floor comes from its constant,
and PQC availability is read from the real backend flags); meta-learning's
"accuracy = 1 - loss" is now None with the reason; the uncertainty
quantifier's constant 0.1 ECE under 20 outcomes is now None with a
`calibration_error_measured` flag and reliability logic that treats
unmeasured as unmeasured; NCF monitor/emerging-tech/essential-workers
planning tables are labelled as uncited planning assumptions (and NCF
confidence now derives from the z-score evidence); voice/CLI canned replies
carry `confidence_basis` and no longer claim sources never consulted; the
database stub health check reports its measured simulated round-trip labelled
as a stub instead of an unrelated random draw; financial/weather factories
raise on unknown provider strings instead of silently serving stub data; and
`simulate_attack` no longer hardcodes replay/side-channel "detections" into
the live anomaly history (no such detectors exist; the drill now says so).

**Removed — three more fabricating visual generators and the RNG-drawn
"validation metrics" of the gating-form study; a single-writer gate now owns
`docs/images`.** Mapping every script that writes images exposed four
generations of generators all writing the same filenames into `docs/images`
(up to five writers per committed chart, so whichever ran last silently
replaced the figure — the mechanism behind both clobbered charts found
earlier). Deleted: `benchmarks/generate_advanced_visuals.py` (every curve
synthesized from seeded noise on designed exponentials, written straight over
the committed report), `benchmarks/generate_consolidated_visuals.py` (falls
back to `generate_synthetic_results()` when the results file is absent and
renders it), `benchmarks/generate_v1_2_visuals.py` (v1.7-era, carries a
"(Simulated)" panel), and `benchmarks/generate_benchmark_visuals.py`
(data-driven but fully superseded by the two `scripts/` generators it collided
with). Nothing referenced any of the four. `benchmarks/benevolence_optimization.py`
reported RNG-drawn Brier/F1/AUC/FP/FN "results" (`auc_score =
uniform(0.92, 0.98)`) and ranked gating forms by that fabricated F1 — those
fields are now `None` with the not-measured contract pinned in its tests,
ranking uses the measured stability score, the comparison chart plots
stability instead of the fabricated F1, and its output moved out of
`docs/images`. The demo's post-run copy loop installs only the committed
report instead of six stray PNGs. The durable fix is
`tests/test_docs_images_single_writer.py`: only three declared writers may
reference `docs/images`, no two may own the same filename, and every committed
chart must have exactly one declared generator. Pre-existing style debt in the
touched files resolved (19 docstring-format fixes, one missing `__init__`
docstring, five untyped test fixtures, a docstring claiming a default of 200
epochs where the signature says 50).

**Removed — the fabricated anomaly-panel generator; every committed docs chart
audited and regenerated from measured data.** A full audit of
`docs/images/` found that the committed `anomaly_detection_panel.png` was
produced by `scripts/generate_anomaly_panel.py`, which fabricated three of its
six panels: hardcoded per-domain F1 literals under a comment claiming they were
"based on comprehensive benchmark", analytic formula curves titled "Threshold
Sensitivity Analysis", and a pie chart of invented severity counts
(`[45, 28, 15, 12]  # Simulated distribution`) titled as 100 real events — plus
a hardcoded "(200 epochs)" label and an undisclosed `synthetic_labeled` data
source behind its headline F1. The honest rendering of that figure already
existed in `scripts/generate_docs_images.py` (every panel computed from the
CI-integrity-gated `benchmarks/mercury_benchmark_results.json`), and the
fabricated script had silently overwritten its output via the same
filename-collision defect fixed for the demo report. The fabricated script is
deleted (its in-repo replacement is the real-data panel, now the committed
image), the README caption is corrected to describe the actual panels, and the
remaining four charts were audited and re-rendered with their defects fixed:
the dashboard legend no longer covers the academic row's value labels, the
MD-011 figure's Improved/ΔF1 summary moved into the title where the legend had
rendered on top of it, the MD-003 figure's legend moved below the axes so all
three μ/σ boxes are visible, and the benchmark summary now names all 9 failed
datasets beside the 67/76 count instead of leaving the gap silent. Five
pre-existing missing docstrings in `generate_docs_images.py` are added.

**Fixed — pure-reasoning plans no longer poison the Bayesian confidence
calibrator, and the neuro-symbolic demo report stops misrendering its own
data.** `MercuryAgent._learn_from_execution` recorded a *failure* observation
whenever a plan's tasks were all transparently skipped (no tool bound), because
the skipped-only `success_rate` of 0.0 — which means "not applicable" — was
compared against the 0.99 success bar. Every pure-reasoning run therefore drove
the Beta posterior toward the calibrator's 0.5 clamp; the committed demo report
showed plan confidence collapsing 0.76 → 0.500 from this alone, and its heatmap
confidence column sat at a dead 0.52. Now a plan with zero executed tasks
records no calibration observation (no evidence is not failure), pinned by
`tests/test_mercury_a_agent.py::TestCalibrationEvidence` end to end. The demo
figure (`benchmarks/neuro_symbolic_benchmark.py`, embedded in the README) had
three rendering defects on top: the domain-competence heatmap clipped at
`vmin=0.7` while its data spans 0.24–0.76, so every cell rendered the same
saturated red and the colorbar misrepresented the data (now the full [0, 1]
range); the benevolence panel's y-axis (0.90–1.02) hid the ~0.64 advisory
series entirely while drawing the retired 0.99 pass-bar as its reference line
(now scaled to the data, against the 0.70 advisory reporting floor, labeled
advisory); and the confidence panel annotated a negative delta as `+-0.260` in
success green (now signed and colored by direction). The demo's scorer also
dropped its retired `benevolence_threshold=0.99` for the default advisory
floor. Separately, `scripts/generate_docs_images.py` wrote a *different*
figure (the ensemble report) to the same
`docs/images/neuro_symbolic_benchmark_report.png` the README embeds as the
cognitive-stack demo, so running it silently replaced one figure with the
other; it now saves as `ensemble_benchmark_report.png`. The committed demo
image is regenerated under the fixed engine and renderer. Three pre-existing
untyped test helpers in `tests/test_mercury_a_agent.py` are annotated.

**Fixed — stale ethics-gate documentation brought back to the code.** Four
documents still described the retired benevolence pass-bar as the first hard
gate at the detection boundaries, contradicting the enforced code
(`check="harm_uplift"` first via the structured-`DecisionSubject` choke point,
σ_Immutable second; benevolence advisory, with its one enforcing use at the
assessment purpose screen — see `src/omni_mercury_engine/ethical/__init__.py`):
`docs/index.md` ("Benevolence, then σ_Immutable … no advisory mode"),
`ARCHITECTURE.md` §"Dual-Gate Hard Ethical Enforcement" (Gate 1 diagram and
per-surface `check="benevolence"` table), `docs/MATH_SPEC.md` (overview item 4
and the §2.1.5 boundary-contract cases, which contradicted the spec's own
corrected §2.7.3), and `docs/ROADMAP.md` rows 1 and the v1.7 status table. The
module docstring and Wave C comment block of
`tests/ethical/test_hard_enforcement.py` carried the same stale prose — one
line even claimed σ_Immutable "is not the enforcement gate today because the
underlying neural network is untrained", false on both counts — while its
assertions already pinned `check="harm_uplift"`; the prose now matches the
assertions. No code or test behaviour changed.

**Added — ALFA UAV fault corpus, stuck-actuator health monitor, and the drone
detector's first real-fault measurement.** `datasets/uav_flight.py` loads the CMU
AIR Lab ALFA dataset (DOI 10.1184/R1/12707963, CC0; SHA-256-pinned archive): 46
real fixed-wing flights, 36 with a mid-flight injected actuator fault whose
injection instant the onboard `failure_status` topics record — genuine event
ground truth (`LABEL_SOURCE="ground_truth"`), the registry's justification
audited in `label_provenance.py`. Measuring Mercury on it exposed the class
per-tick scoring cannot see: a locked aileron stays in range per-tick while its
servo channel's variance collapses (per-flight ensemble AUROC 0.60 on aileron
locks). `detectors/drone/actuator_health.py` adds the targeted statistic —
causal rolling std against the channel's demonstrated activity (a running max,
so a fault filling the window cannot normalise itself away) — scoring 0.907 on
the same class, and `DroneAnomalyDetector` now raises `ACTUATOR_FAILURE` faults
from it (`detected_by="ActuatorHealth"`, gate 0.95 ≈ <0.5% per-tick false
alarms on the ten no-failure flights). Decision-level OR fusion of both alarms
detects 35/36 fault flights at 0.68% joint normal FPR with 8.0 s median delay
across engine, aileron, rudder and elevator faults
(`benchmarks/alfa_uav_benchmark.py`, recorded in `artifacts/alfa_uav.eval.json`).
The drone detector's `decision_source` previously rested on simulated flights
only; this is its first calibration against real recorded faults.

**Added — σ_Immutable corpus engineered end-to-end: boundary-stratified signed
bundle (256 samples), retrained gate, 79-case adversarial evaluation corpus,
calibration improved and re-measured.** The signed corpus doubles to 128+128 with
designed decision-edge mass (boundary-intact positives at threshold+eps, eps
log-uniform [1e-2, 5e-2]; narrative-extreme intact draws) and is re-signed
(Ed25519 + ML-DSA-65, verified end-to-end). The engineering is reported honestly,
failures included: two intermediate retrains that moved 40% of the negatives to
the decision edge degraded the net below shippable (val 0.69/0.71; refused intact
KAT vectors, passed a hostile one), and a boundary-intact family that pinned up to
all 24 anchors handed the net a mean-pooling shortcut ("low anchor mean = intact",
measured as an all-low vector scoring 1.0000). Final design is
architecture-faithful: sub-threshold anchors are the deterministic critical-ethical
floor's authority (the gate's own contract), so boundary-TAMPER cases are
*evaluated* - a generated 58-case systematic corpus (per-anchor minimal tampers at
threshold-0.005, per-anchor boundary-intact PASS guards, a 7-depth sweep,
multi-anchor pairs) joins the 21 authored cases and runs through the production
assembly (floor before net, exactly as `detect_with_fusion`): **27/27 good PASS,
52/52 bad REFUSE, separation 1.000** (`benchmarks/
sigma_immutable_discrimination_results.json`). The retrained net (val 0.843, DoS
guard: real baseline scores 0.999943) passes the full behavioural battery; the
whole security suite is green (1,895 tests). Calibration re-measured on the
boundary-stratified eval distribution: AUROC 0.932, ECE 0.099 (was 0.104), MCE
0.246 (was 0.345), advisory T=2.5 reaching ECE 0.030
(`benchmarks/sigma_immutable_calibration_results.json`); the frozen operational
constant is updated to the retrained value and its invariant re-pinned.

**Added — declared anomaly ontology for every manifest detector, with an
evidence-tied gate.** `core/detector_ontology.py` declares, for each of the 113
entries, `detects` / `cannot_detect` from controlled vocabularies and
`validated` domains where a claim is admissible ONLY with a committed evidence
path; `tests/core/test_detector_ontology.py` enforces coverage, vocabulary
control, evidence existence, explicit not-validated notes, and keeps measured
blind spots declared (the scale-invariant exotics' `amplitude_collapse`).

**Added — Detector Evidence Ledger with computed maturity tiers.**
`scripts/generate_detector_evidence.py` assembles one machine-readable card per
detector from committed sources only and COMPUTES tiers (T0-T5 rules in the
script; T5 unreachable without production evidence):
`artifacts/detector_evidence/ledger.json`, currently T0:3 T1:83 T2:17 T3:9 T4:1
- 83 T1s is the honest backlog the domain packs exist to burn down.
`tests/core/test_evidence_ledger_in_sync.py` makes ledger drift a test failure.

**Fixed — four manifest modality labels corrected to measured truth.**
`affective`/`chemistry`/`parapsychology` (series_1d -> record: on series they
returned constants; their real contracts are emotion/composition/trial records,
now driven by real record payloads in the seam tests) and `frequent_pattern`
(series_1d -> matrix_2d + fit_required: its documented per-transaction contract),
with its false-alarm rate measured and pinned (0.0 across five seeds) in the
modality gate.

**Added/Fixed — full-manifest detector evidence sweep; fitted-reference retrofit of the
exotic models; two harness corrections.** Every one of the 113 manifest detectors now has a
measured, machine-readable evidence row (`artifacts/detector_evidence/`): 44 graded on real
labeled data (4 NAB streams, 5 ADBench clinical sets), 69 classified domain-wired pending
domain packs, tiers computed from the numbers (T0-T5 scheme). The sweep found, and this entry
fixes, three detectors broken by *fixed, unfitted scoring directions*: `astrophysical` scored
a planted 12-sigma burst at AUROC 0.000 (its score was proximity-to-horizon = 1/norm, so loud
events ranked safest), `quantum` at 0.33 (below chance), `consciousness` at 0.644 with no
separation. All three keep their feature identity and now score calibrated robust deviation
from a reference fitted on normal data (`models/fitted_reference.py`: median/MAD center-scale,
empirical-CDF calibration with a strictly monotone tail; unfitted scoring is refused, never
served). Re-measured under the identical sweep protocol: burst AUROC 1.000 for all three;
ADBench means 0.803 / 0.827 / 0.717. Calibration is conformal-disciplined: the
ECDF is computed on reference rows disjoint from the center/scale fit, after a
held-out-normal false-alarm pin caught the in-sample bias on the 32-feature
consciousness model; held-out normal rows now score above 0.95 at ~5 percent by
construction, pinned per model. Declared, measured blind spot: the quantum and
consciousness feature maps are scale-invariant, so pure amplitude collapse is invisible by
representation - pinned as a `cannot_detect` fact in
`tests/models/test_fitted_reference_models.py` alongside burst-competency and real-data
floors. Also fixed: a latent `quantum.extract_features` crash on inputs narrower than 8
columns (amplitudes were padded, states were not). Manifest truth corrected for the three
(`modality="matrix_2d"`, `fit_required=True`). Honest both ways: rev 2 of the evidence table
corrects two rev-1 *harness* errors - `info_geometry` (0.891) and `adaptive_anomaly` (0.883)
were never broken; the sweep drove `detect()` on registry-seam detectors that expose
`fit()+predict()`. They rank 2nd and 3rd of all graded detectors.

**Changed — lint toolchain and README provenance.** `ruff` is bumped 0.17.1 → 0.17.2
across all three pin surfaces the parity gate enforces
(`ci.yml`, `pyproject.toml [dev]`, `.pre-commit-config.yaml`), verified clean tree-wide
before the bump landed. The Dataset Attributions table now credits ALFA (CC0), UCI
Robot Execution Failures (CC BY 4.0, which requires attribution), NAB (MIT) and OpenSky
— corpora this release added without their attribution rows — and the README gains a
STEAM-exploration badge that states only what is in the repo: real labeled data across
robotics, medical (CHB-MIT EEG, PhysioNet sepsis, ADBench clinical sets), Earth and space.

**Added — benevolent-first stand-down greeting on the counter-AI decoy surface.**
Mercury's fail-closed ethic applied to hostile automation: a probing agent's
first contact with a decoy — while still inside the tarpit's no-delay free
window, before any containment escalates — now carries an explicit, logged
stand-down notice (`STAND_DOWN_NOTICE`, in the `X-Mercury-Advisory` response
header): it has reached a monitored decoy, stopping now carries no consequence,
and the system takes no action against the prober's. A normal client still
reads an ordinary 401 auth challenge; an agent parsing headers gets a clear
chance to desist. The offer is a one-time grace (it lapses the instant a tarpit
delay applies and never reappears to signal a reset), is recorded in the
tamper-evident audit (`stand_down_offered`), drives no outbound action (a header
on Mercury's own response, naming no host/URL/command), and can be disabled
(`TarpitPolicy(stand_down=False)`) for silent containment. `security/deception.py`,
`api/routes/decoys.py`; pinned by `tests/security/test_stand_down_greeting.py`
(offer timing, escalation lapse, auditability, silent mode, non-weaponization).
The README's test-count badge is replaced by a Counter-AI Stand-Down badge and a
Counter-AI Containment section documenting the full posture.

**Added — real-data neuro-symbolic benchmark on NAB labeled streams; authored-scenario
benchmark demoted to an honest integration demo.** `benchmarks/nab_streaming_benchmark.py`
measures Mercury's actual detection→symbolic-decision pipeline
(`MercuryAnomalyDetector`) on four real Numenta Anomaly Benchmark streams
(traffic, AWS CloudWatch, ad-exchange, known-cause) with human-labeled anomaly
windows — committed verbatim and SHA-256-pinned under `tests/fixtures/nab/`
(MIT, Numenta), offline, no API key. Leak-free causal featurization, fitted on
each stream's warmup prefix: per-stream AUROC 0.840 / 0.661 / 0.686 / 0.728,
mean **0.729** (`artifacts/nab_streaming.eval.json`). Pinned by
`tests/benchmarks/test_nab_streaming_benchmark.py`, including a label-shuffle
control that proves the AUROC tracks the real windows. Building this surfaced a
real gap the authored scenarios had masked: the neuro-symbolic *memory layer*'s
own point score is at chance on real numeric streams (its token vectorizer bins
values coarsely), so `neuro_symbolic_benchmark.py` is relabeled as what it is —
a cognitive-stack **integration demo on authored illustrative scenarios, not a
performance benchmark** — pointing to the NAB benchmark for real numbers.

**Fixed — unbounded episodic memory in the neuro-symbolic fusion engine (and in
the agent's episodic/semantic stores).** `NeuralMemoryLayer.ingest_memories`
appended one embedding per call forever, and `MercuryAgent`'s `episodic` and
`semantic` stores were likewise unbounded dicts grown once per analysis — so a
sustained ingest loop (one analysis per tick) grew memory without limit. Added
bounded, importance-aware stores: `NeuralMemoryLayer` gains `max_memories`
(default 10000) and the agent stores gain `episodic_capacity`/`semantic_capacity`
(default 10000). When over capacity the lowest-importance (oldest on ties)
entries are evicted, and `NeuralMemoryLayer` entry IDs are now monotonic so
eviction can't recycle an ID. Pinned by `tests/cognitive/test_memory_bounded.py`
and `tests/agentic/test_agent_memory_bounded.py`, each with a cost-plateau test
that fails the old unbounded behavior.

Scope, stated honestly: this bounds *memory* so a long-running deployment cannot
grow without limit; once a store reaches its cap, per-ingest cost is flat. It
does **not** flatten cost *below* the cap, because `ingest_memories` still refits
the clusterer over the whole retained set on every call — so a short run that
never reaches 10000 ingests (the benchmark is ~600) still sees per-epoch cost
climb quadratically. That is a separate refit-cadence inefficiency, tracked but
not changed here (changing the clustering cadence would alter measured numbers);
it is why the demo image below is rendered at a converged lower epoch count
rather than at a high one.

**Fixed — removed fabricated metrics from the neuro-symbolic benchmark report.**
Two panels of `docs/images/neuro_symbolic_benchmark_report.png` were rendered
from invented numbers: the "Domain Competence Heatmap" filled every per-domain
cell with `0.85/0.9/0.95 + np.random.random()`, and the "Memory System Growth"
panel split the real memory total by fixed ratios (0.4/0.35/0.25) with a
made-up short-term curve — synthetic values presented as measured results. The
benchmark now aggregates the **real** per-domain confidence, benevolence and
anomaly-call accuracy its own loop computes, and records the agent's **real**
per-type memory counts (short/long/episodic/semantic) each epoch. Removing the
fabrication surfaced a masked truth: the agent's `success_rate` is a structural
0.0 for these pure-reasoning scenarios (no tools bound, so no task executes),
which the old `0.9 + random` had papered over — that axis is dropped from the
heatmap as not-applicable rather than shown as failure. Pinned against
regression in `tests/benchmarks/test_neuro_symbolic_real_metrics.py`. The
committed report image is rendered from a 25-epoch run — the panels (confidence,
benevolence, memory growth, per-domain accuracy) have converged by ~epoch 20, and
the per-ingest clusterer refit makes longer runs cost quadratically more for no
added signal (see the memory-fix entry above). The image is an integration demo,
not a performance benchmark; real neuro-symbolic anomaly numbers come from the
NAB streaming benchmark.

**Fixed — the wildfire loader's two FIRMS windowing defects (a real-time
window gap and a silent historical mislabeling).** First, `fetch_realtime()`: FIRMS `day_range`
counts UTC calendar days ending today and VIIRS NRT processing lags ~3 hours,
so the 1-day global query returned a header-only CSV when issued shortly after
UTC midnight — the weekly keyed-loader smoke's recurring "0 rows" failure. Two
days always spans the last fully processed UTC day, so an empty result now
signals a real upstream outage instead of a clock artifact. Second, and worse,
`fetch_historical()` carried **no date at all**: it requested the last N days
of NRT data in the event's bounding box, so "2023 Maui Wildfire" returned
whatever is burning near Lahaina *today*, silently mislabelled as the
historical event (consumed as event ground truth by the
`research/governed_fusion`, `engine_calibration` and `omni_equation`
harnesses), and the 10-day request cap meant multi-month events could never be
covered. It now pages the catalog's actual start..end window in
forward-counting ≤10-day chunks against the standard-processing archive
(`VIIRS_NOAA20_SP`; NRT only within its ~2-month horizon) — Australia 2019-20
is 13 chunked requests, not one dateless one. Both pinned in
`test_wildfire_firms_guards.py`. An audit of the other 19 `_EVENT_CATALOG`
loaders found no second instance: each filters by the event's dates/IDs or
documents its synthesis/refusal openly.

**Added — UCI Robot Execution Failures: a genuine manipulator fault corpus.**
`datasets/robot_failures.py` loads UCI dataset 138 (CC BY 4.0, SHA-256-pinned
archive; only the `lp*.data` text members are read — the shipped `a.out`
binary is never touched): 463 real robot-manipulator executions, 15
force/torque samples each, labelled with the observed task outcome (collision,
obstruction, part lost, ...) — event ground truth in the contact modality of
manipulators and humanoid limbs. Parsing surfaced a trap the raw files
confirm: lp3 relabels lp2's 47 transfer executions (byte-identical numeric
rows), so the default pooled view excludes lp3 rather than double-count.
Measured semi-supervised (fit on half the nominal runs per problem, 20 seeded
splits, nothing tuned on failures): AUROC 1.000 (lp1), 0.984 (lp2/lp3), 1.000
(lp4), 0.922 (lp5); pooled 0.968 (`benchmarks/robot_failures_benchmark.py`,
recorded in `artifacts/robot_failures.eval.json`).

**Added — live airspace domain loader (OpenSky Network).**
`loaders/airspace_loader.py` reads real crowd-sourced ADS-B / Mode S state
vectors from OpenSky's anonymous REST API (worldwide or a bounding box) —
Mercury's live-GIS airspace surface, verified against the live endpoint. Its
semantics are stated honestly: emergency squawks (7500/7600/7700) are real
declared events surfaced as a separate `squawk_emergency` column and
deliberately excluded from the scored kinematic features, so statistical
envelope outliers ("aircraft behaving unusually") stay independent of the
genuine event channel; historical state vectors are refused rather than
fabricated (OpenSky serves history only to authenticated accounts).
`opensky-network.org` joins the SSRF allowlist with a pinned endpoint.

**Fixed — `generate_calibration_visuals.py` no longer regenerates the retired
`conformal_coverage.png`.** The standalone image was retired when its
empirical-vs-target scatter became a panel of `calibration_improvement.png`
(docs/BENCHMARKS.md, MD-005), but the generator kept writing the orphan file to
`docs/images/` on every run, contradicting the documented retirement.

**Changed — the tier detector's kNN stream weight follows a dimensional gate.** Equal
z-averaging (0.5/0.5) wasted whichever stream was stronger: on the recorded 57-set
ADBench competitive run the kNN stream dominates the base ensemble at low feature width
(the tier's 20 losses to the PyOD `knn` baseline have median d=18.5) while the base
ensemble owns high dimensions (its 11 outright wins have median d=400; distance
concentration). `MercuryAnomalyDetector` now sets the fusion weight at fit time from the
data's feature width — a sigmoid in log2(d), capped to [0.35, 0.85], midpoint d=128,
label-free, overridable via `knn_stream_weight` — after a decomposition showed the
internal stream (robust scaling, k=10, leave-one-out) outperforms the PyOD baseline on
the sets where the raw-dimension objection would have applied (Wilt: internal 0.801 vs
baseline 0.624). Measured under the competitive protocol on a 12-set dev split (every
set improved or held; annthyroid 0.951 now beats `knn`; InternetAds held) and a 12-set
untouched holdout (10 of 12 above the recorded table; mean gap to `knn` on its favoured
turf closed to parity, 0.8779 vs 0.8810). Gate contract pinned in
`tests/detectors/test_knn_fusion_gate.py`; the stationary false-alarm pin and
discrimination floors hold unchanged.

The gate moved every published-parity AIS figure upward, so the pins are re-recorded
(the DTU track features are d=22, where the gate trusts the kNN stream more): harness
protocol 0.8613 → 0.8750 (oracle F1 0.6000 → 0.6250, re-measured by the benchmark
harness and spliced into `benchmarks/mercury_benchmark_results.json` after recomputing
every aggregate from the per-dataset rows — the recomputation first reproduced the
committed aggregates bit-for-bit, proving formula fidelity), label-free cross-fit
0.8617 → 0.8739, Thesis Fig. 6.1 0.8785 → 0.8934, JMSE Fig. 12a 0.8751 → 0.8779. Both
published-protocol point estimates now sit above GeoTrackNet/VRNN's 0.87, which remains
inside Mercury's confidence intervals: the honest reading is still parity, not
statistically separable superiority. The full suite then surfaced three defects the
dev/holdout protocol could not see, each fixed with the measurement that decided it:
the fitted gate weight was not persisted (`get_fitted_state` now carries
`knn_stream.fusion_weight`; a legacy state restores to the historical 0.5 fuse, pinned
by the save/load round-trip tests); TEMPORAL-classified data inherited the tabular
gate's high low-d weight although rows that are time steps of one signal break kNN
density's iid premise — measured on the sampling-jitter fixtures, hard-fixture AUROC
peaks at w=0.25 (0.9941 vs 0.9742 at the 0.85 cap) and jitter rank-correlation falls
monotonically with w (worst 0.754 at 0, 0.507 at 0.85). A flat TEMPORAL cap was tried
first and measured against the published protocols before shipping: the DTU training
files are *time-ordered independent instances* (tracks sorted by departure, median
|lag-1 autocorr| 0.69-0.70, TEMPORAL-classified) where kNN earns its weight, and the
cap regressed Thesis to 0.8714 and JMSE to 0.8640 — below even the pre-gate 0.5 fuse.
The two regimes the classifier conflates separate cleanly on feature-space continuity
(median consecutive-row distance over median 1-NN distance: 1.35-2.70 on every series
fixture — point, collective, pure AR(1) — against 5.12-10.86 on the DTU sets; a
nearest-neighbour *index-gap* statistic was tried and rejected, defeated by periodic
series), so the gate caps at `_KNN_WEIGHT_TEMPORAL_CAP = 0.25` only below
`_KNN_SERIES_CONTINUITY_MAX = 3.5`, failing toward the cap when no ratio is available.
No competitive set classifies TEMPORAL (verified over all 47 cached ADBench sets) and
the labelled DTU set is tabular, so no other recorded figure moves;
and `tests/test_calibration_wiring.py`'s "baseline F1 is near zero" premise went stale
(the gate lifted the uncalibrated operating point to F1 0.366 on its fixture), so the
test now pins the live invariants: the label-free default stays conservative, and the
F1-optimal calibrated threshold never falls below the uncalibrated baseline.

**Fixed — the fusion net can no longer vote an input it never saw toward "normal".**
A full-system diagnostic surfaced an inverted ranking at the serving boundary:
after a semi-supervised `fit_fusion` over pure normals (consensus pseudo-labels put
the decision boundary inside the normal cloud), `detect_with_fusion` assigned
near-zero anomaly probability to 5-sigma outliers while in-distribution rows
averaged higher — outside the training envelope a neural net's output is
extrapolation, not inference. The decision layer's drift defer already failed the
*verdict* closed (it abstained and requested input), but every consumer of the
per-row `anomaly_probs` ranking saw blatant outliers ranked below normal traffic.
`fit_fusion` (and `fit_fusion_pooled`, when its sources share a raw width) now
records the training data's per-feature envelope (min, max, std), persisted
through `save_model`/`load_model` as tensors the restricted weights-only
unpickler accepts; `detect_with_fusion` floors the probability of any row that
exits that envelope by more than half a feature-spread at the base detectors'
consensus (median of the per-detector [0, 1] scores already computed in the same
call — analytic channels monotone in deviation, which do not extrapolate), and
names the floored rows in `result["out_of_support_rows"]`. Measured on the
diagnostic scenario: shifted-row probabilities 0.544-0.624 against an
in-distribution mean of 0.034 (AUROC 1.0; before: ~0.000 against 0.029, inverted),
in-support batches bit-identical, the labelled training path unchanged, and a
legacy checkpoint without an envelope restores the unguarded behaviour exactly.
Pinned in `tests/core/test_fusion_support_guard.py`.

**Added — the default labelled fit now carries a conformal coverage
certificate.** An AI-architecture audit of the serving path traced which
decision-layer signals ever energise. The GOSNN consequential channel is a
measured refusal working as designed (its merit gate evaluated the detection
head at held-out AUC 0.8007 against the engine's own 0.9608 and shipped
observability-only — `artifacts/gosnn_fusion.eval.json`); drift, symbolic
consistency and the abstention overlay all consume real signals, assembled
before the decider reads them. The one dead wire was conformal:
`calibrate_fusion_conformal()` existed only as a separate opt-in nothing
called, so every decision read "margin heuristic, no coverage guarantee"
even when labels were available to certify. `fit_fusion` now holds out a
stratified slice (`conformal_split`, default 10%, at least 5 rows per class)
before detectors or the net see it and fits the distribution-free
certificate on it automatically: `detect_with_fusion` serves prediction
sets, and the decision signals flip to `calibrated=True` with the target
coverage. Measured end to end: realized per-class coverage 1.0/1.0 on fresh
draws through the real serving path (target 0.9), certificate carried
through `save_model`/`load_model`, consensus pseudo-labels refused (a
coverage guarantee against labels the detectors themselves manufactured
would be circular), tiny classes skipped with the reason logged, and
`conformal_calibrate=False` restores the prior behaviour exactly. Pinned in
`tests/core/test_fusion_auto_conformal.py` (7 tests).

**Changed — the NOAA validation-loader trio is ported to canonical real-data
loaders; the fabrication their "real" paths carried is retired.** The
`validation.data_loaders` NOAA classes claimed real multi-feature data while
measuring almost none of it: the space-weather loader fetched one channel
(Kp) and synthesised the other thirteen features as deterministic functions
of it; the hurricane loader read four HURDAT2 columns, padded fourteen
constants (while ignoring the twelve real wind-radii columns the file
carries), compared an mph threshold against HURDAT2's knots (so "major
hurricane" labels were wrong), and pointed at a URL that now 404s; the ocean
loader fetched one channel and generated seventeen RNG columns named
``salinity_psu``, ``chlorophyll_mg_m3``, ``ph_level`` under
``source="NOAA National Ocean Service"``. The canonical ports read every
feature from the instrument that measures it, all verified live 2026-10-26:
`datasets/space_weather.py` (Kp 3-hourly product, propagated solar-wind
plasma+IMF, GOES X-ray and integral-proton flux, aligned on a shared hourly
grid with only fully-observed hours kept, plus the Shue-1998 magnetopause
standoff as a cited closed form — 168 hours x 15 real features);
`datasets/hurricane.py` (full HURDAT2 layout with real radii, kinematics
derived from consecutive fixes, knots-correct Category-3 labels, newest-file
resolution from the NHC index — 11,417 fixes x 20 features, 786 Cat 3+);
`datasets/ocean.py::NOAACoOpsStationLoader` (per-product CO-OPS station
fetches — water/air temperature, pressure, hourly water level, wind — with
in-window monthly-p90 heatwave labels — 582 hours x 9 real features). All
three carry ``LABEL_SOURCE="statistical"`` and audited label-provenance
registry entries. The validation classes remain as thin wrappers keeping
their historical interface (the hurricane wrapper now converts mph to knots
before the threshold meets the data), and `validation.data_loaders` is
announced for removal in v4.0.

**Changed — the duplicated validation loaders are reconciled onto the
canonical implementations.** `validation.NSLKDDLoader`'s real path, which
could only read a cache it had no way to create, now downloads the real
corpus through `datasets.security.NSLKDDLoader` (148,517 records, verified
live). `validation.USGSEarthquakeLoader`'s real path delegates to
`datasets.environmental.USGSEarthquakeLoader`, retiring a feature that
changed value between runs (``magType`` encoded via Python's per-process
salted ``hash()``); the canonical loader's label rule is now configurable
(``significant_magnitude``/``deep_focus_km``, defaults unchanged) and its
cache filename carries the window and label-rule parameters — the fixed name
used to serve a stale matrix after a parameter change. `validation.
MIMICLoader` was already honest (policy-gated synthetic placeholder naming
the credentialed canonical path) and is documented as such.

**Changed — gwo_ensemble is un-deprecated into a library role, on a
measurement.** Head-to-head on six held-out ADBench sets x 3 seeds
(semi-supervised protocol, IsolationForest/LOF/OneClassSVM bases): the Grey
Wolf weight search beats equal weighting of the same detectors in 17 of 18
runs, mean held-out ROC-AUC 0.8292 vs 0.7852 (landsat 0.54 -> 0.69). That is
the torch-free way to learn detector combination from labels — the niche the
trained fusion pipeline (which needs torch) cannot serve — so the
DeprecationWarning is removed, the class gains an experimental
``decision_source`` carrying the measurement, and its role moves from
``deprecated`` to ``library``. Two dead mechanisms found on the way are now
real or honestly off: ``n_folds`` was declared and never read (the fitness
now averages percentile-F1 over that many folds when >= 2; measured 0.8259
vs resubstitution's 0.8292, so the default is 1), and the diversity bonus
was weight-independent — a constant that could not move the optimum (it is
now the weighted pairwise disagreement it documented; measured active at 0.1
it lowers held-out AUROC to 0.8221, so ``diversity_weight`` defaults to
0.0). The "Target: Improve ensemble F1 by 10-15%" marketing line is replaced
by the measured record.

**Added — GCN notice ingest.** `loaders/gcn_notices.py` consumes NASA GCN's Kafka alert
stream (gravitational waves, neutrinos, GRBs, FRBs, X-ray transients, supernova watches —
the twelve topics this deployment's credential subscribes to) in three layers:
best-effort notice normalization that preserves the raw payload, a per-topic rate matrix
over fixed time windows shaped for `fit_fusion`/`detect_batch` (multimessenger
coincidences appear as the cross-topic correlation structure the rebuilt dimensional and
temporal detectors see), and a `confluent-kafka` OIDC transport that fails loudly without
its dependency or credential. The transport's librdkafka block is pinned key-for-key
equal to `scripts/gcn_kafka_keepalive.py`'s (the keep-alive workflow installs only
`confluent-kafka`, so the builders are deliberate duplicates that a parity test keeps
from drifting). Credentials come from `GCN_KAFKA_CLIENT_ID`/`GCN_KAFKA_CLIENT_SECRET`.

**Added — Mercury-native matrix profile backend; the foundation ensemble works without
extras.** `ml/matrix_profile.py` computes the self-join matrix profile in pure numpy
(STOMP recurrence seeded and refreshed by FFT sliding dot products), matching
`stumpy.stump` conventions: z-normalized distance, `ceil(m/4)` exclusion zone, left and
right neighbour indices, and the constant-window convention. `MatrixProfileDetector` uses
STUMPY when the `[foundation]` extra is installed and the native backend otherwise, so
the detector and the default `FoundationEnsemble` (whose only default member it is) are
operational on a base install instead of refusing. Equivalence is pinned against an
O(l²m) brute-force reference and against STUMPY itself in
`tests/ml/test_matrix_profile_native.py`; the manifest seam tests now drive both
detectors in the no-extra lane rather than skipping them.

**Changed — the temporal and dimensional base detectors detect what they claim.** A
measured planted-anomaly protocol (correlated training data; scattered 3σ point shifts,
variance bursts, variance-matched correlation breaks, a contiguous level shift; 5 seeds;
now pinned as floors in `tests/detectors/test_base_detector_discrimination.py`) found the
two weakest members of the default suite. `TemporalAnomalyDetector`'s `fit()` stored
statistics `detect()` never read, both channels scored against the contaminated batch,
rows before the first full window scored a hard 0, and the windowed trend adapted into
level shifts — ROC-AUC ~0.54 on point anomalies and level shifts. Rebuilt: fit learns
per-feature level and innovation statistics (innovation scale floored at 1% of the level
scale against noiseless training series); the trend channel tests the rolling *median*
against the fitted level; a one-step-ahead prediction channel scores each point against
its trailing window at the fitted noise scale; the change channel scores first
differences against the fitted innovations with spike isolation (min of in/out steps);
channels are reduced over features by null-scaled max/RMS and combined by max in z-space
with one squash. Measured: point 0.539→0.889, variance 0.876→0.964, level shift
~0.53→0.982. `DimensionalAnalyzer` scored 0.625 on variance-matched correlation breaks —
the anomaly family a PCA detector exists for — because its residuals were not
eigenvalue-normalized; it gains a full-spectrum Mahalanobis channel (Shyu et al. 2003's
normalized PCA scoring; `mahalanobis_blend` default 0.7, chosen by sweep), with the DB
spectral term blended into the reconstruction base before it. Measured: correlation
break 0.625→0.828 against the exact-Mahalanobis ceiling of 0.901 for the protocol;
mean/variance/shift ≥0.99 throughout. Both detectors persist the new fitted state;
legacy checkpoints restore with the channels that state supports and keep working. The
temporal incremental-serving replica mirrors the new channels bit-identically (pinned by
the parity suite) and invalidates on refit; detectors restored from legacy state take
the full path.

**Changed — a fitted engine refuses input of the wrong feature width.** After
`fit_fusion(X)`, `detect`, `detect_batch` and `detect_with_fusion` refuse arrays whose
column count differs from `X.shape[1]`, naming both widths and the fix. Before, a 1-D
input died in a raw torch shape error inside a projection and a mismatched 2-D input
passed silently through detectors that tolerate any width, scoring columns against the
wrong fitted statistics. A 1-D input to `detect` now gets the reshape disambiguation
(`(1, -1)` for one sample, `(-1, 1)` for one series); `detect_with_fusion` keeps its
documented one-sample reading. The fitted width persists in the fusion checkpoint;
legacy checkpoints without it, unfitted engines, and dict input stay unchecked.

**Changed — MultivariateTSDetector validates its windowed-input contract.** `fit` and
`predict` refuse non-3-D input, empty batches, and windows whose `(window_size,
num_features)` differ from the configuration, with a message naming the expected shape
and the config keys — previously a 2-D array died inside `np.concatenate` with a bare
`AxisError`, and wrong trailing dimensions were silently reshaped across features.

**Fixed — the extended statistical detector no longer claims to be deprecated.**
`detectors/statistical_extended.py` warned "is deprecated, use MercuryAnomalyDetector"
on import and told readers not to import it in production, while the manifest registers
it as the production BASE detector `extended_statistical` with engine and registry
seams. Same cure as the directive detector: the warning is removed and the docstring
states what the module is.

**Changed — Helm chart refuses production installs without a secret source.** With
`config.app.MERCURY_ENV=production` the chart now fails `helm template` / `helm install`
unless `config.secrets.JWT_SECRET_KEY` and `config.secrets.API_KEY_HASH_SALT` are set, or
`config.secrets.existingSecret` names a Secret that holds them. The application already
refused to start without them, so the old behaviour was pods that crash-looped after
install. CI renders the production and staging overlays with render-only values and
asserts that the production refusal names the missing key.

**Fixed — the directive detector no longer claims to be deprecated.** `detectors/directive.py`
emitted a `DeprecationWarning` on import and described itself as "retained for reference
only". It is a production member of the default detector suite, the orchestrator and the
registry, so the warning and that description were both wrong. The warning is removed and
the module docstring states what the module is.

**Changed — deprecated shims load on first use.** `detectors/advanced` (GWO ensemble) and
`validation` (dataset loaders) now load their deprecated modules lazily, through
`__getattr__`. `import omni_mercury_engine` no longer emits a `DeprecationWarning`; the
warning fires when a deprecated name is actually used.

**Changed — the micro-anomaly term is measured against training data.** The
micro-anomaly rate in `SigmaDirectiveDetector` and `NanoSafeguardDetector` counted
residuals above the 95th percentile of the *scored* batch's own residuals, so it
was about 0.05 for any input, and the batch refitted its own PCA, so the batch's
axes absorbed whatever the batch contained. Both now fit one PCA basis and one cut
on the training rows and score batches against them. The directive detector's
fitted state persists both, and a checkpoint without them abstains on the term
rather than scoring as before. Directive ROC-AUC on the five ADBench datasets is
unchanged (mean 0.9145 before and after, `benchmarks/directive_signal.py`).

**Fixed — single-column directive input abstains.** The directive copy of the
term returned `0.0` for one column, which the blend averaged in as a real score.
It now returns `None`, as the nano copy already did.

**Fixed — `ContrastiveLearningDetector` fits are reproducible at a fixed seed.**
The contrastive views were drawn from an unseeded augmenter, so two fits with the
same seed ended with different weights. The model now takes the detector's seed.

**Fixed — numeric warnings at their source.** `quantum.py` no longer divides by a
zero-weight row before masking it; `neural.py` refuses non-finite input before any
arithmetic, with the same `ValueError` the reference raises; the counterfactual
search uses a finite limit for unbounded features.

**Changed — the CI gates are testable scripts.** The performance benchmark is
`scripts/performance_benchmark.py` (one threshold table, JSON written before the
assertions). The AMA compiler choice is `scripts/resolve_ama_compiler.sh`, with its
three install paths tested against fake compilers.

**Changed — deprecations are reported.** `pyproject.toml` no longer ignores
`DeprecationWarning` and `PendingDeprecationWarning`. The final full run reports
them; see `docs/ROADMAP.md` rows 21 and 22 for what remains.

**Fixed — the Oracle's inter-band term is measured again, as squared bicoherence.** The
previous magnitude-squared coherence between disjoint band-limited copies of one signal is
ill-posed and returned exactly 1.0 for every input; it was withdrawn, and its docstring
recorded what a replacement needed — an n:m cross-frequency coupling measure *and* evidence
of which direction is anomalous. A 1:1 phase-locking prototype had failed the second test
(outer race AUROC 0.892, inner race 0.176 in the opposite direction, ball 0.521 at chance).

Mean squared bicoherence relates three frequencies — whether the phase at `f1 + f2` tracks
the sum of the phases at `f1` and `f2`, consistently across segments — so it is n:m by
construction. Direction on the recorded CWRU corpus, 20 fault windows per class against 40
healthy, every fault class **above** healthy:

| class | mean | AUROC vs healthy |
| --- | --- | --- |
| healthy | 0.0711 | — |
| ball | 0.0788 | 0.900 |
| inner_race | 0.1258 | 1.000 |
| outer_race | 0.2010 | 1.000 |

And where coupling is known by construction: Gaussian noise 0.0077, a periodic impulse train
0.3464, an impulse train convolved with a decaying resonance — the textbook bearing-fault
model — 0.1594.

Two negative results are kept in the docstring because they bound the claim. Pure sinusoidal
AM produces no bicoherence at the modulating frequency, because the signal has components at
`fc - fm`, `fc` and `fc + fm` and none at `fm` itself, so there is no coupled triplet;
bicoherence is not a demodulation measure. And because the statistic measures phase
consistency *across segments*, a purely deterministic signal saturates it at 1.0 whether or
not its components are coupled — three tones with independent phases and three with
`phi_3 = phi_1 + phi_2` both measure 1.0 at their triplet. On recorded vibration, which has
a stochastic component, that degeneracy does not arise.

`_BICOHERENCE_SEGMENT` is 256, the shortest length that separated every class: at 64 ball
faults fell *below* healthy (0.0175 against 0.0250), at 128 they cleared it by 0.0008, at 256
by 0.0075. The estimate abstains at 0.0 below eight segments, since the uncoupled expectation
is 1/K. The influence multiplier reads the value with the correct polarity — higher coupling
amplifies — and is a geometric mean over five measured signals rather than four and a
constant. Every pinned CWRU figure is unchanged by the wiring: the Oracle's fault AUROC stays
1.000 and `physics_integrated` keeps its 0.0750 healthy rate and 1.0000 fault AUROC.

**Added — COPOD and the adversarial autoencoder are manifest entries, with the three
defects that blocked them fixed.** Both were recorded as blocked on `BaseDetector`
conformance because `register_detector` refuses a non-`BaseDetector` instance. That is the
*engine* seam's requirement; the registry seam carries no such requirement, and
`lstm_autoencoder` has shipped at `seams=("registry",)` without being a `BaseDetector` all
along. Both are promoted there, so the signal is available through `extract_all_features`
with no adapter and no change to either class's released API.

Establishing they were fit to promote found three defects:

* **The autoencoder's verdict was random.** Its encoder drew the reparameterisation sample
  on every forward pass, including at inference; `Module.eval()` disables dropout but not
  that. On one fitted model and one fixed input, three `predict` calls gave per-row score
  differences up to **0.59** and flagged 4, 3 and 3 of 40 rows. The sample exists so
  gradients flow through it in training; at inference the latent code is the posterior mean.
  Guarded on `self.training`, repeated calls are bit-identical.
* **Its `seed` did not seed what mattered.** `seed` reached only the NumPy generator picking
  the splits, so the network's weights and the adversarial prior samples were unseeded and
  the benchmark's mean ROC-AUC moved between runs on the same command (0.875, then 0.880).
  `fit` now seeds torch from it, as `LSTMAutoencoderDetector.fit` does.
  `ContrastiveLearningDetector` had the identical one-line defect and gets the same fix; it
  still varies from another source and remains a library class.
* **Its threshold was cut in sample**, on the training rows' own reconstruction error, which
  an autoencoder reconstructs better than anything unseen. `fit` now splits three ways and
  cuts on a disjoint `calibration_split` (0.2). Measured over five seeds of 120 independent
  Gaussian rows scored on 120 more: **0.1483 -> 0.0900** against the 0.05 the percentile
  names, at no cost to ranking (benchmark mean 0.874 either way).

COPOD needed none of this: parameter-free with no trained weights, its in-sample percentile
is not optimistic in the same way and it measures 0.0717 reproducibly.

Promotion evidence, held-out ROC-AUC over five ADBench datasets
(`benchmarks/advanced_detector_auc.py`), now reproducible to every per-dataset digit: COPOD
**0.869**, AdversarialAutoencoder **0.874**, against **0.661** for the shipped LOF
reference. That is the first half of the bar `KMeansDistanceDetector`'s docstring sets --
standalone ranking justifies making a signal *available* -- and the second half is **not**
claimed: whether either adds to the live fusion ensemble still needs a fusion-marginal
ablation, for which no harness exists.

**Fixed — the directive detector scored each row against its own batch, and alarmed at
0.5.** `SigmaDirectiveDetector` is one of the five detectors the engine enables by default,
so both defects reached every `engine.detect` call on tabular input.

`_ethical_oversight_amplifier` returned `magnitude / max(magnitude)` over the rows of the
batch, so a row's score depended on which rows it arrived with. Measured on one fixed row
against identical fitted state, with the recursive-memory buffer cleared so each call was a
function of (fitted state, batch) alone: **0.6817** scored on its own, **0.5140** beside 99
benign rows and **0.5469** beside 99 extreme ones. That 0.168 swing is wider than the gap
between this detector's normal and anomalous score means; a single extreme row suppressed
every other row, and a row scored alone was always its own maximum and so always ~1.0. The
scale now comes from `fit` -- a squash over the training rows' magnitudes with
`EOA_CALIBRATION_QUANTILE` (0.98) at 0.5, the squash tier's existing convention -- and the
same measurement reads 0.4842 and 0.4929, a swing of **0.0086** against 0.1677. The residual
is the recursive-memory buffer, which is documented streaming state for direct callers and
deliberately unchanged; the engine's fusion boundary already resets it
(`_reset_transient_detector_state`, ROADMAP row 16).

The alarm sat at the 0.5 `BaseDetector` default while the combined score's normal regime
means 0.53 to 0.61 with a 95th percentile of 0.68 to 0.72, stable across shapes (60x8,
200x8, 60x4, 200x16, 500x3) -- so the default flagged **0.61 to 0.94 of entirely normal
data**. `fit` now cuts at `CALIBRATION_QUANTILE` (0.95) of the training rows' own scores.
Measured over five seeds of clean 200x8 rows, buffer cleared between calls:

| cut | clean FPR | TPR on an injected +8-sigma block |
| --- | --- | --- |
| 0.5 (previous) | 0.61-0.94 | -- |
| **0.95 quantile** | **0.0560** | **1.0000** |
| 0.98 quantile | 0.0200 | 0.7800 |

Both the EOA scale and the calibrated threshold are fitted state and travel in the
checkpoint. An earlier measurement put the true-positive rate at 0.08 for the
0.95 cut; that was wrong -- the probes shared one detector and the memory buffer carried
between them. Every figure above was re-measured with the buffer cleared.

**Added — the false-alarm gate now covers the declared `matrix_2d` modality.**
`tests/core/test_declared_modality_false_alarm_rate.py` drives every `matrix_2d` manifest
entry behind the registry seam on independent Gaussian rows and pins what each flags in a
held-out half, where every flag is a false alarm. It is the generalisation of the
`series_1d` gate that found five mis-declared modalities, and it has teeth:
against the previous `directive` it fails with "flags 0.8417 of held-out normal rows against
the measured 0.0600" -- a flood live in a default-enabled detector that no existing test
reported, because the manifest seam tests assert a detector *returned* something and never
that the verdict means anything. Pinned: `realtime_threat` 0.1067, `kmeans_distance` 0.0867,
`directive` 0.0600, `rca` 0.0467, `spatial` 0.0433, and `physics_integrated` 0.0 set-level.
Three entries return a ranking rather than a verdict and are listed with that reason;
further tests assert verdict arity per group and that no new `matrix_2d` entry reaches the
registry's modality filter unmeasured.

**Fixed — `physics_integrated` applied a conformal cut to signals it was never measured
for.** Its threshold is a split-conformal cut over the fused scores of whole signals, and
the fused score is not invariant in a signal's length -- every component's score rises with
it and the fusion is their mean. Measured on a detector fitted at width 64 with a 0.8929
cut, scoring stationary signals from the same process as its training data, so every flag is
a false alarm: 0.00 flagged at the fitted width, 0.20 at 1.12x, 0.70 at 1.25x, 0.90 at 2x
and 1.00 at 6.25x. That is the mechanism behind this detector calling normal data anomalous
-- not a bad threshold, but a good threshold applied off-contract. `fit` now records the
calibration signals' length and `detect` refuses a signal outside 10% of it with
`InputModalityError`, which the engine carries as a decline. Enforced only when there is a
calibrated cut to protect.

Checked and **not** changed: the 3R blend. `recursion_score` and `resonance_score` measure
0.93 to 0.99 on normal data and carry 40% of the blend, which looked like a near-constant
offset diluting the score -- the same defect a previous round removed from this function as
`ethical_scaling`. The recorded bearing corpus rejects that reading: disabling the blend
moves the healthy false-alarm rate **0.0750 -> 0.1250** with fault AUROC unchanged at
1.0000, because the conformal threshold adapts with the offset (0.8459 -> 0.5573) and
cancels it.

**Fixed — SR-CNN's operating point is calibrated instead of fixed at 0.5.**
The BCE objective weights the rare positive class by
`pos_weight = (1 - inject_ratio) / inject_ratio` (19 at the defaults), which is sound for
training a classifier on a 5%-positive set but shifts what the sigmoid means: at the optimum
`w * p * (1 - s) = (1 - p) * s`, so `logit(s) = log(w) + logit(p)` and the trained logit
carries a constant `+log(w)`. A cut at 0.5 on that output is therefore a cut at
`sigmoid(log w) = w / (1 + w) = 0.95` on the prior-calibrated probability — which the
provenance, describing "the CNN's sigmoid output" and a 0.5 threshold, did not say. Measured
consequence: on a stationary AR(1) series fitted on 400 samples and scored on the next 400
from the same process, the fixed cut flagged **0.2725** of a held-out half with no anomaly
in it.

`fit` now installs the `calibration_quantile` (0.95) of the *unaugmented* training series'
own scores as the threshold, so the cut is a stated position in this detector's score
distribution rather than a fixed value the `+log(w)` offset has already moved. An explicitly
configured `threshold` is the caller's decision and survives `fit`.

Every candidate cut sits on one ROC curve — mean AUROC over the four recorded NAB files is
0.683 for all of them — so only the operating point was ever in question:

| operating point | clean-data FPR | NAB |
| --- | --- | --- |
| fixed 0.5 (previous) | 0.2725 | 12.2 |
| logit minus `log(w)` | 0.0000 | 9.0 |
| **0.95 quantile of training scores** | **0.1020** | **13.2** |

The calibrated cut is better than the fixed one on *both* axes, so this is not a sensitivity
trade — it was simply a worse point on the curve. Subtracting `log(w)` removes the false
alarms outright but costs true detections, which is why that is not what is applied. The 0.90
quantile scores NAB 42.7 at an FPR of 0.1940 and is not the default, because choosing it
would be tuning a threshold on the evaluation corpus. The stationary gate's pin moves from
0.2725 to 0.1020; the NAB pin (12.2) is unchanged within its tolerance.

**Removed the three `# type: ignore` markers an earlier commit had added.** Two were on the
duplicated `self._oracle_detector.detect(...)` calls in `PhysicsIntegratedDetector.detect`
and went with the `_run_oracle` helper that replaced them; the third, on
`detector.detect(...)` in the CWRU recorded-data test, was there because the helper took
`detector: object`. It now takes a `_WindowScorer` `Protocol` that states what the
measurement actually needs — one verdict per window — so the call type-checks without a
suppression. `scripts/check_suppression_hygiene.py` passes.

**Measured — the Oracle band z-score fix moves a published-parity benchmark, upward.**
`MercuryAnomalyDetector` lazily fits a `SpectralDomainFrequency` Oracle when it classifies
its input as `TEMPORAL`, and multiplies the ensemble's combined scores by that Oracle's
influence multiplier before clipping them to `[0, 1]`. The band z-score defect above was
therefore in the path of `examples/mercury_quickstart.py --published-protocols`, which the
ISO Hardening workflow pins exactly. Measured on the digest-pinned DTU AIS training files
(the same bytes both sides, source the only difference, reproduced locally and in CI run
1170 against 1175):

| protocol | before | after |
|---|---|---|
| Thesis Fig. 6.1 ROC-AUC | 0.8785 | 0.8785 |
| Thesis Fig. 6.1 AP | 0.3641 | **0.3827** |
| JMSE Fig. 12a ROC-AUC | 0.8615 | **0.8751** |
| JMSE Fig. 12a AP | 0.2475 | **0.3085** |
| Harness protocol / label-free cross-fit | 0.8613 / 0.8617 | unchanged |

Every figure that moved improved. The mechanism is the clip: the influence multiplier is a
single scalar for the whole batch, and multiplying every score by a constant cannot change a
ranking — but `np.clip(scores, 0.0, 1.0)` after it can, by tying everything the multiplier
pushes past 1.0. With `aggregate_score` pinned at its saturated constant the multiplier was
larger (measured 1.3069 against 1.2732 on a controlled TEMPORAL fit), so more scores hit the
ceiling and the ties cost ranking. The harness and label-free rows are untouched because
they do not take that path.

The pinned figures are updated in `.github/workflows/iso-hardening.yml` and in README's
quickstart output block. The published comparison is unchanged in substance: GeoTrackNet /
VRNN at 0.87 still lies inside Mercury's interval on both protocols (now 0.788-0.951 and
0.792-0.944), so the result remains parity rather than superiority.

**Fixed — the stationary false-alarm gate reported a rate for detectors that return one
verdict.** Seven of its 28 entries reduce a whole series to a single verdict rather than one
per sample (`acceleration_dynamics`, `frequent_pattern`, `nano_safeguard`,
`physics_integrated` -- since re-declared `matrix_2d`, below --
`spectral_domain_frequency`, `spectral_vibration`, `uiux_anomaly`), so
the "1.0000" first recorded for three of them was not a 100% false-alarm rate — it was one
Boolean. The gate now pins the two groups separately: `MEASURED_RATE` for per-sample
detectors, where the number is a rate over 400 samples, and `MEASURED_VERDICT` for set-level
ones, where it is the share of the five seeds whose single verdict came back anomalous. A
new test asserts each group's verdict arity so an entry cannot drift between them unnoticed,
and the CHANGELOG and ROADMAP wording is corrected to match.

**Verified — the advanced tabular detectors' benchmark figures.** `ROADMAP` row 32 cited
COPOD 0.869 and AdversarialAE 0.866 against 0.661 for the shipped LOF reference without a
re-run. Re-measured with `python -m benchmarks.advanced_detector_auc` on its own protocol:
COPOD **0.869** and LOF **0.661** reproduce exactly; AdversarialAE measures **0.870** against
the cited 0.866 and ContrastiveLearningDetector **0.621** against 0.598, both within what a
different torch/numpy build explains, and the qualitative claim about Contrastive holds
exactly (under 0.5 on cardio, 0.432, and thyroid, 0.481). The row is updated to the
reproduced values.

**Fixed — `DeepLogSequenceDetector` decided on a rule that cannot work, while computing the
paper's own rule and discarding it.** The verdict was `1 - P(key | context) > threshold`,
which the `decision_source` already flagged as "an engineering default not taken from the
paper". That rule cannot discriminate: `1 - P` falls below 0.5 only where some single
successor holds more than half the conditional mass, so on any grammar with a handful of
roughly equiprobable successors it flags nearly every position. Meanwhile `_eval` computed
DeepLog's actual criterion — the key fell outside the model's top-`g` next-key predictions —
per position, and `detect` collapsed it to a scalar count in
`metadata['top_g_violations']`, throwing the per-position array away.

The verdict is now the paper's criterion conjoined with the configured threshold, so
DeepLog's rule decides and `threshold` remains an operating control (measured identical to
the paper's rule alone on every dataset below, since the top-`g` set was a subset of the
miss-probability set in each). The per-position array is reported as
`metadata['top_g_outside']` beside the count. The ranking score is unchanged, so AUROC is
unchanged. Measured, held-out false-alarm rate, miss-probability rule -> paper's rule:
real Drain-parsed HDFS log templates **0.8270 -> 0.1730**; a synthetic low-entropy grammar
**0.0525 -> 0.0025**; a stationary AR(1) series (outside its contract) 0.9965 -> 0.1325; the
four recorded NAB files, NAB standard-profile score **-780.3 -> -530.3**. An injected
template the training half never shows is still caught 10 of 10 positions on the HDFS
fixture (9 of 10 on the synthetic grammar, where Laplace smoothing can rank a never-seen key
inside the top 3 of a low-branching context — the trade-off `top_g` controls).

**Fixed — `deeplog_sequence` was declared `series_1d`.** It takes a sequence of integer
log-template keys (`_to_1d_int` rounds to a key), and a categorical next-key model has
nothing to learn from a continuous series. It is now `event`, the manifest's documented
"discrete domain events fed over time" slot, which no entry occupied; the manifest seam
suite gains an `event` payload (a low-entropy template grammar) and drives it with that
instead of `rng.normal(size=200)`. Fourth entry here whose declared modality its own
input contract contradicts, after `rca`, `kmeans_distance` and `realtime_threat`.

**Added — `tests/fixtures/loghub` and a recorded-data test for the log detector.**
`scripts/record_hdfs_log_fixtures.py` records the Drain-parsed event-template keys of 2000
real Hadoop HDFS log lines (LogHub's `HDFS_2k` sample, 14 templates; He, Zhu, He & Lyu 2020,
arXiv:2008.06448) with the SHA-256 of the served CSV and of the stored keys.
`tests/detectors/test_recorded_hdfs_log_templates.py` pins the held-out false-alarm rate
(0.1730), the rate the old rule gave on the same stretch (0.8270), and the detection of a
template the training half never shows. Five of its six tests fail on the old code.

No AUROC is claimed: the sample is not anomaly-labelled, because LogHub's block-level labels
ship with the full 11-million-line `HDFS_1` release. The sessions in this sample also
interleave — 2200 block ids over 2000 lines, median one event each — so DeepLog's own
per-block session grouping is not reconstructable from it and the next-key distribution is
genuinely high-entropy; two of the 14 templates appear only in the held-out half. That is
why 0.1730 is not low, and it is pinned as measured rather than presented as a calibrated
rate. `deeplog_sequence` leaves `test_recorded_nab_series.py` (where it was pinned at
AUROC 0.534 / NAB -780.3) and the stationary false-alarm gate (0.9965): both measured a
categorical model on continuous data.

**Fixed — the LSTM autoencoder's conformal threshold is cut on pooled out-of-fold scores.**
`lstm_autoencoder` flagged 0.6265 of a stationary held-out half at the manifest's default
configuration, against the 0.05 its one-class conformal 95% cut is documented to deliver
(`tests/detectors/test_stationary_false_alarm_rate.py`). Two defects stacked, and the second
is the dominant one.

* **The calibration block and the early-stopping block were the same rows.** Early stopping
  restores the state that reconstructed those rows best, so their scores are not exchangeable
  with test scores and a cut taken on them sits below the true quantile. Separating the two
  blocks moves the mean rate 0.6265 -> 0.4450: real, and not the main cause.
* **A single contiguous tail cannot estimate the score distribution of an autocorrelated
  series.** Consecutive points carry far less information than independent ones, so a short
  block's spread understates the distribution's. Measured on the five gate seeds, the 80-row
  tail's score standard deviation was 3 to 11 times smaller than the scored block's (e.g.
  0.029 against 0.310), which is what put the cut so far below the 95th percentile.

The threshold now comes from pooled out-of-fold scores: the fit series is cut into
`calibration_folds` (default 5) contiguous blocks, each block is scored by a model trained on
the other four, and the 95% cut is taken over the pool. Every pooled score is out-of-sample
for the model that produced it, and the pool spans the series end to end rather than one
stretch of it. `calibration_folds=1` restores the old single-block cut; `0` trains without
calibrating a threshold and is the internal path for the fold models.

Mean false-alarm rate over the five gate seeds, 400 fit rows scored on the next 400:

| calibration | mean rate |
| --- | --- |
| single block, shared with early stopping (previous) | 0.6265 |
| single block, disjoint from early stopping | 0.4450 |
| pooled out-of-fold, 5 blocks | **0.0670** |

Against the 0.05 nominal that is within a factor of 1.34, and the pin in the stationary gate
moves from 0.6265 to 0.0670. The recorded NAB corpus corroborates it independently: the
detector's NAB standard-profile score moves **-88.5 -> +3.1**, from flooding to net positive,
with its mean AUROC unchanged at 0.829 — only the alarms moved
(`tests/detectors/test_recorded_nab_series.py`). The reconstruction scores are **bit-identical** before and after
(`max |new - old| = 0.0` on seed 7): the model, its split and its early stopping are
unchanged, so discrimination is untouched and only the decision threshold moved. The residual
overshoot is the known cost of applying a conformal cut to serially dependent data, where
exchangeability holds only approximately; it is no longer a defect in how the cut is taken.
A fold-ensemble variant, scoring test rows with the mean of the five fold models instead of
the full-data model, measured 0.0815 and was not taken: it would put five models in every
checkpoint for a difference small against that residual.

**Fixed — `RealTimeThreatDetector` failed OPEN when a sub-detector's training score was not
finite.** `_validated_input` already refuses a non-finite *input* row, with the reason stated
in its own docstring: "a security detector failing OPEN on an unmeasurable input". The same
hole was open on the other side, for a non-finite sub-detector *output*. A 1-D input is
validated to one `(1, n_features)` sample, which leaves the neighbour-based component
degenerate: LOF scores `inf`. Then `center = median(s)` is `inf`, `mad = median(|s - center|)`
is `NaN` (the source of the `RuntimeWarning: invalid value encountered in subtract` this
module emitted), and `scale` is `NaN` — and `scale <= 1e-12` is **False for NaN**, so the
standard-deviation fallback never fired. The threshold came out `NaN`, `score > NaN` is False
for every later sample, and the `NaN` column also poisoned the ensemble mean, so
`_ref_threshold`, `_ref_p90`, `_ref_p95` and `_ref_p99` were all `NaN` too.

Measured on the same input before and after: `_ref_threshold`/p90/p95/p99 `NaN` -> finite, and
a probe drawn 8 sigma from the training distribution went from **`is_threat: False`,
`threat_level: "LOW"`** to `is_threat: True`, `threat_level: "CRITICAL"`. A sub-detector whose
training score is not finite is now dropped from the calibrated ensemble with the reason
logged, the near-constant fallback arm also rejects a non-finite scale, and a fit where
*nothing* can be calibrated raises — the same refusal already in place for the zero-fitted
case, since `detect_threat` requires `name in self._detector_center` and would otherwise
grade every input LOW. Five tests fail on the old code.

**Fixed — `realtime_threat` was declared `series_1d`.** `_validated_input` documents
`(n_samples, n_features)` and reshapes a 1-D input to one sample, and the detector is measured
on ADBench *tables* in `tests/detectors/test_recorded_adbench_tabular.py` (held-out AUROC
0.675/0.907/0.986/0.994/0.972). It is now `matrix_2d`, the third entry here whose
declared modality its own validator contradicts.

**Fixed — two tabular detectors were declared as 1-D series detectors, and the registry
routed 1-D series to them.** `DetectorRegistry.extract_all_features(data, modality=...)`
filters on each entry's declared modality exactly so "detectors that cannot read it" are not
driven with it. `rca` and `kmeans_distance` both declared `series_1d` while being
`(n_rows, n_features)` detectors, so the filter passed 1-D series straight through to them:

* `rca`'s `_to_2d_f64` reshapes a 1-D input to `(1, n)`, so a 200-sample series became one
  row of 200 nodes. `rows.std(axis=0)` over a single row is 0, every `_node_std` sat at its
  `1e-9` floor, the squash scale came out `1.4e-9`, and every subsequent row scored 1.0:
  **100% of a stationary held-out half flagged** against the 2% its `calibration_quantile`
  targets. `rank_root_causes` on such an input "localises" a *timestep* as the root-cause
  node. Its `fit` has always documented `(n_rows, n_nodes)`.
* `kmeans_distance`'s `_to_2d_f32` is `np.atleast_2d`, so a 400-sample series became one
  400-feature point. The signal was benchmarked on ADBench *tables* and its `feature_dim` is
  9 = `n_clusters + 1` **per row**.

Both are now declared `matrix_2d`. The manifest seam suite drove both with
`rng.normal(size=200)` and passed, because it checks the result contract (shapes, ranges,
finiteness) rather than whether the verdicts mean anything; it now drives them with
`(60, 8)`.

**Fixed — `KMeansDistanceDetector` anchored its squash at the median, then cut at 0.5.**
`1 - exp(-d / scale)` with `scale` = the median training nearest-centroid distance puts
score `1 - 1/e` = **0.632**, not 0.5, at the median, so the 0.5 alarm cut sat *below* the
median training distance and by construction flagged more than half of any in-distribution
batch. Measured on 400 held-out rows of a stationary 8-column Gaussian: **0.93 flagged**.
The scale now uses the shared `detectors._calibration.squash_scale` with a new
`calibration_quantile` (default 0.98), the engine's quantile-anchored convention that the
rest of the squash tier already follows, so the configured quantile of training
nearest-distances scores exactly 0.5. Measured: **0.93 -> 0.0175** against the nominal 0.02.
The squash is a monotone rescale of the same distance, so the benchmarked *ranking* is
untouched — held-out AUROC on all five recorded ADBench tables is bit-identical before and
after (pima 0.6527, cardio 0.7711, breastw 0.9770, wbc 0.9878, thyroid 0.9548).

**Fixed — `KMeansDistanceDetector.detect` ignored its configured threshold.** It compared
against a hard-coded `0.5`, so `config={"threshold": ...}` was accepted and had no effect.
It now uses `self.threshold` like the rest of the tier.

**Corrected — the k-means signal's benchmarked AUROC range.** Its module docstring and
manifest description claimed "~0.8-0.98 across breastw/thyroid/WBC/cardio". The repo's own
`artifacts/dormant_module_revival.json` records cardio at **0.762**, and re-running
`benchmarks/dormant_module_revival.py` reproduces it (cardio 0.762, thyroid 0.945, breastw
0.982, WBC 0.981, Pima 0.634, mean 0.861 over five tables and three seeds). The range is
0.76-0.98 on those four tables, and Pima — outside the quoted set — is 0.634.
`docs/DETECTORS.md` is regenerated from the manifest.

**Added — `tests/detectors/test_stationary_false_alarm_rate.py`**, the measurement that found
both modality defects. Every `series_1d` manifest entry with a `BaseDetector` seam is fitted
on 400 samples of a stationary AR(1) series and asked about the next 400 from the same
process: nothing there is anomalous and the process does not drift, so the share it flags is
a false-alarm rate with no drift to blame, which is the one place the squash tier's
`1 - calibration_quantile` contract can be read directly. 30 entries are pinned at their
measured rate; the 32 that serve a domain API instead of `fit`/`detect` are listed with that
reason, and two completeness tests fail if a new entry appears unpinned or an excluded one
grows the seam. Fixed seeds, byte-identical across runs, about 15 s.

The measurement also puts numbers on calibration gaps that were previously described only
against drifting NAB files, where drift could be blamed. It is pinned in two groups, because
a single number does not mean the same thing for both kinds of detector under `series_1d`.
**Per-sample** detectors return one verdict per held-out sample, so the number is a
false-alarm rate: above the 0.02 the
squash anchor targets but bounded, `srcnn` 0.1020, `statistical` 0.0980, `energy_based` 0.0755, `deep_svdd`
0.0470, `extended_statistical` 0.0460; meeting its own 0.05 conformal nominal,
`lstm_autoencoder` 0.0670 (see the out-of-fold calibration fix above, which moved it from
0.6265); at or under contract, the other 15 down to `page_hinkley` 0.0025. **Set-level** detectors reduce the whole series to *one* verdict, so
what is pinned is the share of the five seeds whose single verdict came back anomalous:
`nano_safeguard` calls an entirely normal
stationary series anomalous on every seed,
while `acceleration_dynamics`, `frequent_pattern`, `spectral_vibration`,
`spectral_domain_frequency` and
`uiux_anomaly` never do. A test asserts each group's verdict arity so an entry cannot drift
between them unnoticed.

**Not** defects, checked: `energy_based` and `deep_svdd` are high-variance, not biased — over
20 seeds their medians are 0.0138 and 0.0188, at or under the nominal 0.02, with a heavy tail
(max 0.1725 and 0.1375), so the 3-seed excess that first flagged them was the tail and not a
systematic bias. No change was made to either.

**Fixed — `acceleration_dynamics` decided at a cut its own normal scores sat above; and
`nano_safeguard` now says when its cut is uncalibrated.** Both reduce a whole signal to one
score and both shipped a cut that the *normal* regime of that score exceeds, so each called an
entirely normal stationary series anomalous on all five gate seeds: `acceleration_dynamics` at
the 0.5 `BaseDetector` default against a fused score of 0.79 to 0.83, `nano_safeguard` at a
hand-set 0.05 against a composite of 0.40 to 0.44.

Whether that can be calibrated from the training signal turns on one measured property. A
caller who supplies one series gives one score, so the only source of repeated scores is
*windows* of that series — and a cut taken on windows is valid only if the score transfers
across lengths.

It does for the kinematic score: measured 0.7986, 0.8078 and 0.8018 at window lengths 100, 200
and 400 on one process. `AccelerationDynamicsDetector.fit` therefore cuts at the
`calibration_quantile` (0.95) of the scores of windows of the training signals. The verdict
share over the five seeds goes **1.0 -> 0.0**, and the alarm still fires: an injected level
shift and a five-fold volatility burst are both flagged on every seed tried, where previously
*everything* was flagged and the operating point separated nothing.

It does not for the nano composite, which rises with length — 0.3757, 0.3915, 0.4049, 0.4125,
0.4216, 0.4215 at lengths 50 to 400. Window calibration was implemented and measured there
anyway: it takes the verdict share 1.0 -> 0.4, and `q=0.95` and `q=0.99` give the identical
result, which shows the residual is that length bias and not quantile noise. Shipping it would
mean knowingly calibrating on an assumption the measurement rejects, so it is **not** applied.
Instead `fit` warns, naming the measured normal regime and what to set, and every verdict
carries `threshold_calibrated` — False unless the caller supplied `micro_threshold`. Closing
it needs labelled micro-anomalies at a known rate, which no corpus in the tree provides.

`acceleration_dynamics` moves from 1.0 to 0.0 in `MEASURED_VERDICT`; `nano_safeguard` stays at
1.0, now with the reason stated in its `decision_source` rather than only in this file.

**Fixed — `physics_integrated` declared a modality its own threshold calibration cannot
use.** It was declared `series_1d`, but its alarm threshold is a split-conformal cut over the
fused scores of held-out *whole signals*, so calibrating it needs many normal signals:
`_calibrates_threshold` enforces `ndim == 2` and at least
`_CALIBRATION_STRIDE * _MIN_CALIBRATION_SIGNALS` = 76 of them. A single 1-D series carries
exactly one sample of that statistic and cannot calibrate it at any length. Under the wrong
declaration the threshold stayed at the 0.5 default, which the fused score of normal signals
(0.89 to 0.99) exceeds, and the detector called an entirely normal stationary series anomalous
on all five gate seeds — the fifth instance of the defect class already fixed for `rca`,
`kmeans_distance`, `realtime_threat` and `deeplog_sequence`.

Re-declared `matrix_2d`, the modality filter in `DetectorRegistry.extract_all_features` routes
it a stack of signals and it calibrates: fitted on 85 windows of a stationary series it cuts at
0.8929 and flags 0.0612 of normal windows, against its own 0.05 bound. The 1.0 verdict leaves
`MEASURED_VERDICT`, since the gate drives `series_1d` entries only.

Two things stop the re-declaration from merely hiding the old behaviour. A direct 1-D fit now
logs a warning naming the uncalibrated threshold and what to fit instead, rather than returning
an anomaly verdict silently; and every verdict carries `threshold_calibrated`, False when the
alarm rests on the configured or default threshold and so carries no false-alarm bound.

**Fixed — `EchoStateDetector` calibrated its alarm boundary on data it had trained on.**
`fit` set the squash scale from the 98th percentile of the readout's *in-sample* residuals.
A `reservoir_size + 1` = 101-parameter ridge readout (`ridge=1e-4`, barely regularised)
reproduces its own training data far better than anything it has not seen, so that quantile
understates the normal-regime residual and the detector alarms on ordinary data — making the
documented contract ("`1 - calibration_quantile` is the normal-regime FPR") false. Measured
on the four recorded NAB files, whose probationary periods are 169 to 282 samples: the
in-sample 98th-percentile residual was 2.5 to 16.6 times smaller than the held-out one, the
squash scale came out 0.56/1.54/2.23/1.76 where honest calibration gives
5.88/9.00/4.83/8.04, and the realised false-alarm rate on normal points was
0.79/0.32/0.68/0.66 against the nominal 0.02. On a stationary AR(1) series fitted on 400
samples — no drift to blame — the rate was **0.2375 against 0.02**.

The readout still trains on every sample, so ranking is untouched; only the scale moves. It
is now the pooled quantile of *out-of-fold* residuals: `calibration_folds` (new, default 5,
must be >= 2) contiguous blocks, each scored by a readout fitted on the others, with the
reservoir states computed once over the whole series — they do not depend on the readout, so
folding the readout alone is sound. A series too short to leave a fold's complement larger
than half the parameter count falls back to the in-sample residuals.

Measured, four recorded NAB files, same defaults: mean ranking AUROC **0.699 -> 0.699**
(identical, as intended), NAB standard-profile score **-1167.5 -> -368.3**, mean
false-alarm rate on normal points **0.611 -> 0.188**, alarms 846/475/1134/652 ->
13/13/919/228. On the stationary AR(1) series the rate is 0.0250 against the nominal 0.02,
so the contract now holds. Two alternatives were measured and rejected: a single held-out
tail of 25% of the fit series reached NAB -852.8 but cost ranking (AUROC 0.699 -> 0.636,
and 0.471 at a 40% holdout) because the readout trains on less data; 10 folds was no better
than 5 (NAB -380.1). **Not fixed:** two of the four files still flood (`rogue_agent_key_hold`
919 alarms, FPR 0.533; `speed_7578` 228, FPR 0.205). Their post-probation normal level
drifts away from the probationary reference, so no quantile measured inside the training
window can bound the false-alarm rate there — a fixed-reference residual detector needs an
adaptive reference, which is a separate change with its own evidence burden. Four tests fail
on the old code, including `test_low_false_positive_rate` tightened from its 0.1 bar to the
0.03 the quantile actually promises — 800 training samples overfit mildly enough to pass 0.1
even while the scale was wrong, which is why that test did not catch this.

**Fixed — the frequency-domain Oracle returned the same score for every input.**
`SpectralDomainFrequency._score_band` z-scored a *dimensionless* quantity against *absolute*
band-power statistics: `z = (corrected_ratio - ref_mean) / (ref_std + EPSILON)`, where
`corrected_ratio` is observed power over reference power over the band's expected share of a
`1/f**beta` spectrum, while `ref_mean`/`ref_std` are band powers in the signal's own units.
That branch ran whenever the noise colour had been estimated, which is the normal case. On
the recorded CWRU bearing windows the band powers are 1e-6 to 1e-4, so the subtraction was
dominated by `-ref_mean / ref_std`: z came out between 6.4e3 and 6.6e6 whatever the window
held, `|z| / 3` clipped to 1.0, every resolvable band scored exactly `0.6 * 1.0 + 0.4 * 0.0`
= 0.6, and the Oracle's aggregate was exactly **0.210000000 for every healthy and every
faulty window alike** — a declared fusion component contributing a constant. The z-score is
now taken in power units. Nor can the correction be made unit-consistent and still matter:
`expected_ratio` is a per-band constant once `beta` is estimated in `fit`, and a constant
divisor applied to both the observation and the reference cancels exactly in a
standardisation — so the corrected ratio is reported as the diagnostic it is, on the new
`FrequencyBandResult.corrected_power_ratio` field (additive, default `0.0`). Measured on the
recorded CWRU windows (fitted on 80 healthy, scoring one 2048-sample window at a time):
healthy scores now range 0.021-0.097 (mean 0.056) and every fault separates at **AUROC
1.000** (inner race, ball, outer race) against 0.5 — no ranking at all — before.
Downstream, `PhysicsIntegratedDetector` on the same protocol: healthy windows flagged
**0.125 -> 0.075**, fault AUROC **0.975 -> 1.000** on all three faults, calibrated threshold
0.983 -> 0.846. Measured and pinned as a limit: the Oracle's own aggregate stays below its
0.5 `BaseDetector` default on every one of these windows, so standalone it ranks perfectly
and flags nothing; `PhysicsIntegratedDetector` consumes its score and influence vector, not
its verdict. Six tests fail on the old code.

**Fixed — the Oracle's inter-band phase coherence was never measured, and said nothing.**
`_compute_phase_coherence` returned exactly `1.0` for every input it was ever given —
healthy and faulty CWRU windows, white noise, a pure tone, a constant signal — for two
compounding reasons. It built one band-limited signal per band by zeroing every FFT bin
outside that band, then asked `scipy.signal.coherence` for the magnitude-squared coherence
of two *disjoint* spectra, where `|Pxy|^2 / (Pxx * Pyy)` is `0/0` at almost every bin; and
`nperseg` was clamped to the band signal's own length, so Welch got exactly one segment, and
with no averaging `|Pxy|^2 == Pxx * Pyy` identically, making every surviving bin exactly 1.0.
Since `coherence_influence = 1.0 + (1.0 - phase_coherence) * 0.5`, one of the five terms in
the phi-weighted influence multiplier was permanently neutral: a documented five-signal
fusion was a four-signal fusion. The 0/0 was also the source of twelve
`RuntimeWarning: invalid value encountered in divide` per `detect` call — under `-W error`
those became exceptions, which `PhysicsIntegratedDetector` caught, dropping the Oracle from
`component_scores` with nothing but a log line to show for it. Magnitude-squared coherence
between disjoint band-limited copies of one signal is ill-posed, so the computation is
withdrawn rather than left reporting a constant as a measurement: the method now abstains
explicitly at `PHASE_COHERENCE_NOT_MEASURED`, the numeric fusion is unchanged (the term was
already neutral), the warnings are gone, and every docstring that claimed five measured
signals says four. **A replacement needs evidence not yet available**: the bands'
centre frequencies differ, so 1:1 phase locking is not well posed, and a 1:1 phase-locking-
value prototype across time segments was measured on the recorded CWRU windows and rejected
— outer-race faults separate from healthy at AUROC 0.892 but inner-race faults at 0.176 (the
opposite direction) and ball faults at 0.521 (chance), so "low coherence implies anomalous",
the direction the influence multiplier assumes, does not hold. What is missing is an n:m
cross-frequency phase-coupling measure with its anomalous direction established on labelled
data.

**Fixed — a failed Oracle silently changed `PhysicsIntegratedDetector`'s verdict.** Both
copies of the `try`/`except Exception` around the Oracle logged a warning and carried on, so
`frequency_oracle` left `component_scores` and the influence vector stopped modulating the
fused score with no trace in the returned dict — and `oracle_result is None` there was
indistinguishable from "the domain policy disabled the Oracle for this domain". The two
copies are now one `_run_oracle` helper, and the reason any optional component failed is
recorded under `metadata['component_failures']`, empty when every enabled component ran.
Three tests fail on the old code.

**Fixed — the topological detector scored windows against a profile of a different size.**
`TopologicalAnomalyDetector.fit` built `_reference_features` from the whole reference point
cloud (80 points under the `n_reference_samples=80` cap used by the pinned ADBench window
protocol) while every window `score` is asked about holds 25. Betti numbers, persistence
entropies, landscape norms and lifetime counts all scale with the number of points — the
reference's `n_features_h0` was 60 against a window's 20 at a cap of 60 — so each of the 14
feature deviations carried a constant size bias, the bias dominated their unweighted mean,
and the signal was diluted. Clean reference windows scored 0.92, up against the ceiling of
the score's sigmoid. The baseline `_aggregate_score` measures from is now the mean feature
vector of the reference *sub-samples*, which carry the same number of points as the windows
being scored (the same discipline `SpectralVibrationDetector` already used for its
per-window `_reference_features_mean`); the whole-reference profile remains as the fallback
when fewer than four rows make windows impossible. Measured window AUROC on the pinned
ADBench protocol (four tables, three seeds): Pima 0.461 -> 0.599, cardio 0.965 -> 0.961,
breastw 0.859 -> 0.999, thyroid 0.999 -> 1.000; mean **0.821 -> 0.890**. Over six seeds,
0.839 -> 0.883. The calibrated threshold drops from 0.920 to 0.311, out of the saturating
region. The Wasserstein distances `score` reports stay a diagnostic and out of the
aggregate: adding them measured 0.883 against 0.890.

**Fixed — a configured `anomaly_threshold` silently changed the topological score itself.**
`anomaly_threshold` is documented as the decision cut, but supplying it skipped calibration
entirely, so `_feature_stds` stayed empty and `_aggregate_score` fell through to its
near-zero-variance branch: the same clean window scored 0.918 with an auto threshold and
0.085 with `anomaly_threshold=0.5`. The window baseline and feature spreads are now measured
whenever `fit` has enough rows, and only the threshold is left alone when one was given.
Cost of always measuring them: an uncapped `fit` on 300 rows with a configured threshold
goes 1.47 s -> 1.89 s, a capped one 0.027 s -> 0.050 s. Four tests fail on the old code.

**Fixed — refitting the topological detector kept the first fit's threshold.** The
auto-calibration guard was `if self.anomaly_threshold is None`, which is false after any
auto-calibration, so a second `fit` on different reference data moved the window baseline
and feature spreads to the new data but left the old cut point in place: fitting on
scale-0.05 rows then on scale-5.0 rows held the threshold at 0.3155 where a fresh fit on
the same rows gives 0.2798. Whether the threshold was *configured by the caller* is now
recorded at construction, so an auto threshold recalibrates on every `fit` and a configured
one is still never overwritten. Two tests fail on the old code.

**Fixed — Page–Hinkley's `delta` and `lambda` were in the series' own unit.** The same test
on the same shape of data alarmed from the first sample when the unit was large and never
when it was small: on four recorded NAB files (standard deviation 8.5e6, 4.8, 0.09 and
0.05) it alarmed on 1056 of 1057 and 937 of 958 points, and on none of 1308 and 1600.
After `fit`, the series is now measured in units of the reference standard deviation (1 for
a constant reference; an unfitted detector keeps the series' unit), so `delta` and `lambda`
mean the same for any unit, and the scale is saved with the detector and reported in
`metadata['scale']`. The class has not been released. On its own this made the verdicts
consistent, not good — at the default `delta=0.005`, `lambda=50` it then alarmed on the
level drift of every one of the four files (NAB standard score -792 -> -1785) — which
exposed the missing restart fixed in the next entry.

**Fixed — Page–Hinkley never restarted, so one shift alarmed to the end of the series.**
`PH_t = m_t - min_{s<=t} m_s` grows monotonically while a shift is sustained, so once it
crossed `lambda` it stayed there: on a synthetic +3σ shift held for 700 samples the
detector alarmed on 685 of them, and on the four recorded NAB files it alarmed on 798 of
1057, 1138 of 1308, 1542 of 1600 and 695 of 958 post-probation points. The Page–Hinkley
change detector clears its accumulators on the sample following an alarm (as the reference
implementation in `river.drift.PageHinkley` does, and as `SPRTDetector` in the same module
already did after each Wald decision); Mercury's did not. Both one-sided accumulators and
their running minima now restart after an alarm. Measured on the same four NAB files, same
defaults: alarms 798/1138/1542/695 -> 4/10/19/9, NAB standard-profile score **-1785.3 ->
+26.6**, mean ranking AUROC 0.332 -> 0.534. Because `x_bar` is fixed by `fit`, a shift that
persists re-alarms every `lambda / (shift - delta)` samples (measured 14-20 for a +3σ shift
at the defaults, predicted 16.7) rather than alarming once: a fixed-reference CUSUM reports
that the series is still away from the reference it was fitted on. Cost: the restart makes
the scan sequential, so `page_hinkley_statistics` runs at 1.7 Msample/s against 11 Msample/s
for the previous cumulative form (90 ms -> 580 ms per 10^6 samples). Three tests fail on the
old code, including the hand-computed seven-point table, which now shows the restart.

**Fixed — `PhysicsIntegratedDetector` flagged every normal signal.** Its fused score of a
healthy bearing-vibration window (CWRU Bearing Data Center, 12 kHz) is 0.89-0.99, and the
alarm threshold stayed at the 0.5 default whatever `fit` saw: it ranked fault windows above
healthy ones (AUROC 0.975 — this entry previously said 1.0, which was measured on a run
without the spectral component's `sample_rate`; re-measured with the pinned configuration it
was 0.975 both before and after that change, and reached 1.000 only with the Oracle band
z-score fixed, below) and flagged 100% of the healthy windows. When no `threshold` is
configured and `fit` receives at least 76 signals, every fourth signal is now held out of
the component fits and the threshold is the split-conformal quantile (false-alarm rate
0.05) of their fused scores, capped below 1.0. Held-out healthy windows flagged: 100% ->
12.5% (40 windows); fault windows flagged: 100% before and after. A configured threshold
and a smaller `fit` keep the 0.5 default. Note: `sample_rate` reaches the spectral component
only through `spectral_config`.

**Fixed — `MatrixProfileConfig.normalize` was never read, and the matrix-profile provenance
named a screen that does not exist.** Every profile was z-normalised whatever the caller
set; `normalize` is now passed to STUMPY (the default, `True`, gives the profile it always
did). `decision_source` said discords are "screened at z >= 2.0"; nothing applies that
cut (`discord_threshold` is accepted and not read; flags are the batch's own
`anomaly_threshold` quantile), so the text now says what the code does. On the four
recorded NAB files the profile ranks anomaly windows at mean AUROC 0.356 (0.073 on road
traffic), the same as raw `stumpy.stump`.

**Fixed — the topological detector's threshold ignored its sample cap.** With
`n_reference_samples` set, `TopologicalAnomalyDetector.fit` built the reference
profile on the capped sample but calibrated the alarm threshold on windows cut
from the uncapped data: a window of a third of the rows, built ten times. On the
1886 training rows of ADBench thyroid, `fit` took 167 s with a cap of 80 and now
takes 0.1 s, and the windows the threshold is cut from now match the reference
they are compared with. Window-level AUROC on recorded tables (anomalous against
normal windows of 25 rows, 3 seeds): cardio 0.965, breastw 0.859, thyroid 0.999,
Pima 0.461 (no better than chance). Callers that set no cap are unchanged.

**Fixed — the LSTM autoencoder trained on raw values.** `AnomalyDetector` (and
the registry's `lstm_autoencoder`) fed the network series of any scale as they
were, so it could not reconstruct them: on four recorded NAB files its mean
AUROC was 0.596 and 0.101 (below chance) on the road-traffic file; with inputs
standardised by the training rows it is 0.829 and 0.950. The statistics are kept
in the checkpoint; a checkpoint saved earlier has none and is served on raw
values as before. Its scores are now in standardised units, and the 95%
conformal threshold is cut in those units.

**Fixed — PatchCore's coreset could not be seeded.** `PatchCoreConfig` had no
`seed` although the detector read one "if present", so the greedy coreset always
started from OS entropy: over 60 identical fits the held-out clean-flag count
ranged from 0 to 5 of 12, which failed CI's `test_calibrated_decision_is_informative`
(limit 3) about one run in sixty. `PatchCoreConfig.seed` now exists (default
`None`, unchanged behaviour); the visual-image test fits with `seed=0`.

**Fixed — untrained networks no longer decide alarms (endocrinology, TEMPEST,
anesthesia, medical imaging and vitals, CYBINT, interstellar objects, BLIP).**
A freshly built network is a random function of its input. Measured: the CGM
analyser labelled 500 mg/dL "severe hypoglycemia" under one seed and "normal"
under another; a blank X-ray was "anomalous" under three of five seeds; the TEMPEST
video network graded an all-zero feature vector "moderate risk" or "high risk"
by seed; CYBINT named an APT group and malware family from noise. Each now
follows the repository's existing fail-closed pattern (`is_fitted`,
`load_trained_weights`, `ml_*_available` on the result) and abstains until trained
weights are loaded; none ship. The endocrinology glycemic state is instead the
consensus glucose band of the latest reading (<54, <70, >180 mg/dL; Battelino et
al. 2019) and a non-finite reading is refused. The interstellar detector's
non-gravitational hypotheses read a key nothing set and never appeared. BLIP's
128-D fusion features came from an unseeded projection left in training mode
(dropout on): the same image gave different features on every call; the
projection is now seeded, in eval mode, and an extraction failure raises instead
of returning a zero vector. OSHA: the 50 V and 85 dBA lines are "or more" in the
regulations (29 CFR 1910.303(g)(2)(i), 1910.95(c)(1)); the code used strict
`>`. The unused network classes and attributes are kept (released API).

**Fixed — spectral vibration detector, its fusion score and its CLI.**
A detector fitted on 1,024-sample signals scored every normal 1,025-sample
signal 1.000 (the longer path was Hann-windowed, the shorter was not); every
length now takes one windowed estimator, with a single frame scaled to the
broadband level of a full one (long-signal values are unchanged). Its
untrained networks are now drawn from a fixed seed (`network_seed`), so
instances agree bit for bit, and its graph-Laplacian eigenvalues start from a
fixed vector, which removes a ~1e-9 run-to-run score difference. The
`/detect/fusion` spectral score was `max|FFT| / N` of the raw data (a series
offset by 1000 scored 1000 and was always an anomaly); it is now Fisher's
dominant-frequency test as a z-score, like the other scores.
`mercury physics spectral` gains `--reference`, reports the input's own
spectral entropy, harmonic distortion and signature in place of constant
placeholders, and says when no reference was given.

**Fixed — decoy failures were visible only in logs.** When the audit chain refused
a tripwire record, the shared store failed, or a decoy route failed, the sensor
kept answering (by design) but left nothing an operator or the chain could act
on. `TripwireEvent` gains `audits_failed` (lost audit records, carried on the
next audited event) and `degraded` (failures hit while recording the touch);
both appear in the audit details. `DeceptionSensor` gains `failure_counts()`,
`note_failure()` and an optional `failure_hook`; the API exports
`mercury_deception_failures_total{kind}` on `/metrics`.

The first release after 2.0.0; it includes the work prepared as 2.1.0, which
was never released. Upgrading from 2.0.0: the license is now
AGPL-3.0-or-later (2.0.0 stays GPL-3.0-or-later), and every renamed or removed
2.0.0 name is listed with its replacement in DEPRECATION.md §6.6-§6.15 and §7.

### The streaming worker alarms with a learned policy: PPO on NAB

**Added — `ml/alarm_policy.py`, now the default detector of
`StreamingAnomalyPipeline` and `mercury stream` (`--detector zscore` /
`MERCURY_STREAM_DETECTOR=zscore` keeps the EMA z-score rule).** The worker's
decision, alarm or not on each message, is what NAB scores, so it is learned
against NAB's own reward. Per numeric field, causal features (fast/slow EMA
z-scores, median/MAD robust z over 100 and 1,000 values, percentile-rank
extremeness, a first-difference z) gate *candidates* (rank extremeness above
0.962: 5% of points, 107 of 110 windows reachable), and a policy trained with
`ml.ppo_trainer` decides at each candidate from the features and its own recent
alarm history. The training environment's reward is the exact NAB score
increment of each decision, so an episode's return is the stream's raw NAB
score plus a constant (tested). Without the gate PPO learned never to alarm.
Serving is NumPy only: the policy network ships as
`models/checkpoints/streaming_alarm_policy.npz` (44 KB, loaded with
`allow_pickle=False` after its sidecar's SHA-256 check), produced by
`scripts/train_streaming_alarm_policy.py`.

**Measured — leave-one-category-out on the five real NAB categories (47 files,
321,206 points, 110 windows; NAB standard profile; `benchmarks/streaming_alarm_policy_results.json`).**
Each category scored by a policy trained on the other four, three PPO seeds:
mean held-out score 46.0 (44.2, 48.1, 45.7), detecting 82-86 of 110 windows
with 468-564 false alarms. Baselines on the same folds: the EMA z-score rule
-188.7 (99 windows, 5,640 false alarms); a threshold plus refractory period
on one feature, tuned on the training folds, 29.1 with the feature chosen by
training score, 41.7 with the best feature chosen on the held-out folds. The
PPO settings and the gate were chosen on the realAWSCloudwatch fold; on the
other four folds alone the policy scores 43.2 (40.8, 46.6, 42.1) against 39.3
for the best-on-held-out threshold, and every seed beats it. The shipped
policy is trained on all 47 files (seed 0, 2,000,000 decisions; each training
run takes about a minute on one CPU thread).

### The streaming worker could never raise an alarm

**Fixed — `StreamingAnomalyPipeline`'s default detector
(`infrastructure/streaming.py`), which `mercury stream` and the Kubernetes and
Helm streaming deployments run.** It folded each value into the EMA mean and
variance *before* scoring it, so a deviation `d` scored
`z = (1 - a)|d| / sqrt((1 - a) v + a d^2)`, never more than
`(1 - a) / sqrt(a) = 2.85` at `a = 0.1`: below the 3.0 threshold for every
input, however extreme. On the 47 real NAB streams it raised no alarm at all.
The value is now scored against the statistics it has not yet moved, then
folded in. Measured on the same 47 streams (321,206 points, NAB standard
profile via `evaluation.nab_scoring`): 99 of 110 anomaly windows detected,
but 5,640 false alarms (2.1% of points), NAB score -188.7. A fixed
3-sigma rule fires too often on real, heavy-tailed streams; the learned alarm
policy below replaces it.

### Tier detections recorded and alerted correctly; label-free flags honour `contamination`

**Fixed — `/api/v1/detect/tier` recorded every detection as 0 anomalies at
score 0 (`api/routes/detection.py`).** The route handed the history recorder
`n_flagged` and `max_score`, but the recorder reads `anomalies` and `scores`,
so tier detections were stored empty and never reached the SIEM alert sinks
(which skip records with no anomalies). It now passes the tier's own flags
and scores.

**Fixed — without labels, the tier flagged 35-53% of a series
(`detectors/detection_tier.py`, `StreamingScoreEnsemble.fit`).** The
threshold came from the automatic calibration selector, which read the
averaged rank scores as bimodal and split them with Otsu near the median: a
200-point series with one 8-point burst had 106 points flagged (3 members)
or 69 (all 17) at `contamination=0.05`. The label-free threshold is now the
`1 - contamination` quantile of the ensemble score, as documented (10 of 200
flagged); with labels it stays F1-optimal. Benchmarks score with AUC and
oracle F1, so no recorded result changes.

### `ml.ppo_trainer` rebuilt natively: PPO with no reinforcement-learning dependency

**Rebuilt — `ml.ppo_trainer` under its 2.0.0 names (`ml/ppo_trainer.py`,
`ml/rl_envs.py`).** The 2.0.0 module wrapped stable-baselines3, which no
install profile provided, and fabricated results when forced to run
(DEPRECATION.md §6.6). PPO is now implemented in PyTorch: PPO-Clip with GAE,
separate policy and value networks with orthogonal initialization,
per-minibatch advantage normalization, gradient-norm clipping, optional
value clipping, linear learning-rate and clip-range annealing,
approximate-KL early stopping, value bootstrapping through time-limit
truncations, observation and reward normalization, and gSDE for continuous
actions. Both networks run as batched matrix products over one flat
parameter vector, and the loss gradient is computed by hand, about twice as
fast as autograd on the same network; it equals autograd in all 60 tested
configurations (worst relative error 2.6e-7). `ml.rl_envs` supplies batched
`CartPole-v1` and `Pendulum-v1` that replay a recorded Gymnasium trajectory
bit for bit (4,000 steps across 135 episodes; `tests/fixtures/rl_envs/`), plus an
adapter for any Gymnasium-API environment. Every 2.0.0 entry point exists
and works; the changes from 2.0.0 are listed in §6.6. Checkpoints hold only
tensors and primitives and load through `safe_torch_load`, so loading one
cannot run code (stable-baselines3's `.zip` format unpickles with
`cloudpickle.loads`). Training is reproducible from `PPOConfig.seed` and
leaves the process-wide random state untouched.

**Measured — against stable-baselines3 2.9.0 on identical settings
(`benchmarks/ppo_benchmark.py`, `benchmarks/ppo_results.json`).** The
rl-baselines3-zoo PPO hyperparameters, 100,000 steps, one CPU thread, seeds
0-4 alternating between the two implementations, 20 deterministic
evaluation episodes per run. CartPole-v1: both reach the 500 cap on every
seed; Mercury trains in 10.6 s on average against 28.8 s (2.7x). Pendulum-v1:
mean evaluation return -220 against -447 (worst seed -324 against -678;
higher on every seed, seed 1 within 1.4), in 20.4 s against 55.7 s (2.7x).

**Measured — the two modules are mutation-floored (`mutation-testing.yml`:
`mutation-floor-rl-trainer`).** First full enumeration of
`ml/ppo_trainer.py` and `ml/rl_envs.py` (691 sites): 475 killed, 68.7%.
Defaults, validation edges and the update's arithmetic were unpinned. Two
test rounds, several through a stand-in gradient that records what the
update passes and steps, bring it to 659 of 687 killed, 95.9%; the 28
survivors are classified equivalents in the lane comment. Floor 91.

### Relicensed to AGPL-3.0-or-later (effective v3.0.0)

Mercury Agent is now licensed under the GNU Affero General Public License
v3.0 or later. The AGPL closes the one gap the GPL left open for a project
whose natural deployment is a hosted service: its section 13 extends the
corresponding-source obligation to anyone offering a **modified** Mercury to
users over a network, so no proprietary version can exist, distributed or
hosted. Versions before 3.0.0 were released under GPL-3.0-or-later and remain
so, irrevocably, for everyone who received them.

The change is complete, not cosmetic: the `LICENSE` file carries the verbatim
AGPL-3.0 text (sections 0–12 and 14–17 verified equivalent to the previous
GPL text under the Affero renaming; section 13 and the appendix differ exactly
as the two licenses do; pre-2017 `http://` FSF links normalized to `https://`
to match the FSF's current canonical file); all 1,942 SPDX tags flipped to
`AGPL-3.0-or-later`; packaging metadata (both `pyproject.toml` files,
`Cargo.toml`, the Docker OCI license label, the Helm artifacthub annotation),
the OpenAPI license object, `__license__`, the README license statements, the
agent notice, CONTRIBUTING and the docs all updated; the openness pillar test
now pins the AGPL file (including section 13) and the pyproject declaration.
Before the change, every copyright line in the repository was verified to
name Steel Security Advisors LLC — the sole holder, so the relicense needs no
third-party consent. Historical records (dated CHANGELOG entries and the
`tlp_handler` port-provenance note) are left as written: they were true when
written. AMA Cryptography (Apache-2.0) remains one-way compatible.

### Additional permission under AGPL section 7: the NVIDIA CUDA libraries

**Added — `LICENSE-EXCEPTION`.** Mercury's GPU acceleration runs on NVIDIA's
CUDA libraries, which are proprietary; the CUDA builds of torch install them
and the Docker image ships them. As copyright holder, Steel Security Advisors
LLC grants an additional permission under AGPL section 7 — in the FSF's
standard form for such libraries — to combine Mercury with the NVIDIA CUDA
Libraries and convey the result. It is scoped by name (the CUDA runtime,
cuBLAS, cuFFT, cuRAND, cuSOLVER, cuSPARSE, cuSPARSELt, cuDNN, NCCL, NVRTC,
nvJitLink, CUPTI, cuFile, NVSHMEM, and NVIDIA's `nvidia-*` distributions of
them), relaxes no other term — section 13's network source offer included —
and, as section 7 allows, may be removed from a conveyed copy. `LICENSE` stays
the FSF's verbatim text (now pinned by sha256); the permission ships beside it
in every wheel, sdist and image through setuptools' `LICEN[CS]E*` default
(pinned by test).

**Changed — the license gate learns `permitted`.** A review entry may now cite
a section 7 grant in the project; the gate checks the named file exists and
is one, counts those dependencies separately, and reports them without
warning. The 14 NVIDIA packages move from `flagged` to `permitted`; `regex`
stays flagged. The gate suite (57 tests) fails 3 against the previous gate;
three new openness tests pin the grant's scope, the verbatim license and the
packaging.

### The AGPL made operable: a source offer and a dependency license gate

**Added — `/source`, the corresponding-source offer (AGPL-3.0 section 13).**
Every deployment now serves, publicly and with no input, the license, the
running version and where its complete corresponding source is published —
the upstream repository by default, so a stock deployment complies by
existing. An operator running a modified Mercury sets `MERCURY_SOURCE_URL` to
their published fork (and `MERCURY_SOURCE_REVISION` to the exact commit); a
URL that is not `https://`, has no host or embeds credentials is refused and
logged — a credential in a refused URL never reaches the log — and the upstream
location is served instead; a revision that is not a Git object id is omitted.
38 tests, failing without the new module; full mutation enumeration of the
offer and its route kills 10 of 10 (the first run's survivor, the URL length
limit, was a real boundary gap, closed with an exact-limit test). Both modules
joined the trust-surface mutation lane, re-enumerated whole: 214 of 218
killed (98.2%, floor 95), the four survivors the lane's classified
equivalents.

**Added — dependency license gate** (`scripts/check_dependency_licenses.py`,
`configs/dependency_license_review.toml`). It walks the runtime requirement
closure `pyproject.toml` declares for the requested extras and resolves each
installed distribution's license from its own metadata (PEP 639 expression,
then a recognised `License` field, then classifiers as alternatives). Labels
that name a family without a version — "BSD", "Apache" — are never guessed:
the shipped license text decides (an advertising clause or a non-2.0 Apache
text fails), and no text is unknown. Unknown or incompatible licenses fail
unless a reviewed entry pins their exact metadata: `verified` entries carry
primary evidence, `flagged` ones are reported on every run as the owner's
decision. It runs in Type Checking on every PR (`[ml]`, 101 distributions)
and in ML Tests over `[all]`, the closure the Docker image ships. The `[all]`
lane was pre-validated against pip's full resolution of `[all]` (186
distributions, classified by the gate's own resolver; every unversioned label
settled from the package's real license text) so it lands green. 51 tests over
real `.dist-info` directories.

What it found, reported rather than hidden: the CUDA builds of torch install
NVIDIA's proprietary runtime libraries (14 packages) into the `[all]` closure
and the image — now covered by the section 7 permission below; `regex`, which
transformers loads on import, declares `Apache-2.0 AND CNRI-Python`, a license
the FSF lists as GPL-incompatible — flagged for the owner's decision.
`tokenizers` and `namex` ship no usable license metadata; both are verified
Apache-2.0 from their own license files (sha256 recorded).

**Fixed — `run_ci_gates.sh` had drifted from ci.yml.** Its shipped-scripts
mypy list lacked `scripts/record_gdacs_fixtures.py`, which ci.yml checks; the
local mirror now matches (20 scripts, both lists).

### SURGEDAT acquired; DONKI Kp encoding verified

**Added — SURGEDAT observed peak storm surge (`data_sources/storm_surge.py`:
`parse_surgedat_csv`, `SurgeObservation`, `SurgeHeight`).** The LSU
databases the previous round could not reach are recorded byte-verbatim:
the global database (745 observations, 1737-2014) and the Gulf of Mexico
database (240, 1880-2011), with provenance in
`tests/fixtures/surgedat/PROVENANCE.json`. The server sends its leaf
certificate without the issuing intermediate; the intermediate (InCommon
RSA Server CA 2) was taken from Mozilla's CCADB, verified against the
system roots, and the files fetched with full TLS verification. Both are
classic Mac OS files (carriage-return line ends, Mac Roman text), read as
such. Most heights are stated in feet and in metres; a pair counts as one
reading only when the two rounding intervals overlap, so four pairs in the
global file are reported as conflicts the database holds against itself
(Roger 1993's 1.3 ft beside 0.7 m among them) and none in the Gulf file.
A conflict or a stated range keeps both readings and has no single value;
the GDACS surge level is assigned only when every stated reading falls in
one band, and only storm surge is graded (storm tide adds the tide). The
files are static archives (last modified 2014 and 2013), so the parser
reads a snapshot; no live source is added, since a client would need the
same chain repair to reach the server.

**Fixed — the DONKI storm-window labels dropped NOAA G1's lowest third
(`loaders/space_weather_loader.py`).** The loader kept its literal
`Kp >= 5.0` last round because the kpIndex encoding could not be checked.
It is now checked against an independent primary source: GFZ's definitive
Kp as NASA OMNI2 serves it (`tests/fixtures/space_weather/omni2_kp_2024_05.json`,
Kp*10 in thirds). The recorded live DONKI May 2024 response, which holds
the Gannon storm, writes Kp in decimal thirds (x.33 = x+, x.67 = (x+1)-):
every one of its 20 values is a third, 14 equal GFZ's value and the other
six differ by one third (NOAA's estimate against GFZ's definitive index);
on the Gannon records alone, 10 of 16 match. So a 5- arrives as 4.67, which
the literal read as quiet. Windows are now classified through `noaa_g_scale`
like every other G-scale site, thirds strings ("5-") parse, and
`KP_STORM_THRESHOLD` states the G1 onset (14/3). Not settled by this
evidence: whether DONKI omits G1-only windows by policy. Five of the six
GFZ G1-only windows of May 2024 have no DONKI entry, so the labels cover
what DONKI reports; DONKI's own definition and NOAA's archived estimates
were unreachable (api.nasa.gov 503; CCMC, GFZ and SWPC's FTP mirror reset
by the egress path).

**Measured — the two modules above are mutation-floored
(`mutation-testing.yml`: `mutation-floor-hazard-evidence`).** First full
enumeration of `data_sources/storm_surge.py` and
`loaders/space_weather_loader.py` (129 sites): 73 killed, 56.6%. The
SURGEDAT rounding-interval arithmetic, the loader's feature windows, its
real-time and DONKI request windows and the label window's edges were
unpinned. Known-answer tests (boundary pairs within micrometres of the
overlap rule, closed-form ramp features, a fixed clock, a sub-second label
edge) bring it to 121 killed, 93.8%; the eight survivors are classified
equivalents in the lane comment. Floor 88.

### Trust surface: a bounded store, decoys behind the rate limiter, contacts that cannot break security.txt

**Fixed — the cross-worker tripwire store grew without bound
(`security/deception_store.py`).** The in-process sensor keeps at most
`max_sources` sources; the store kept every source and aggregate it ever
saw and never pruned decoy touches, and every touch counted distinct
sources by scanning all of them. Measured under address rotation (100
touches/s inside one window): 40,000 rotated sources left 40,000 touch
rows and per-touch latency climbing from 0.24 ms to 5.0 ms, serialized
across workers on the shared file's write lock. The store now caps sources
and aggregates at the sensor's `max_sources` (passed through
`record_touch(..., max_sources=)`), evicting the least recently seen with
its hits and touches -- the in-process LRU rule -- and counts the campaign
signal from an indexed last-seen table. Same flood: 10,000 touch rows and
about 1 ms per touch at the cap. A store written before the cap is
backfilled on open, so its keys are covered too. The key being recorded is
never an eviction candidate: the store runs on the wall clock, and after a
backward step a fresh key stamped older than every other evicted itself
mid-touch, so the touch counted zero hits (no tarpit) and left an orphaned
touch row.

**Fixed — reliability grading was per-worker (`security/deception.py`).**
`DeceptionSensor.outcomes()` / `posterior()` read only the process's own
touches, so with the shared store each worker graded a source on the
decoys that worker happened to see. With a store they now read the
cluster-wide count (`TripwireStore.distinct_decoys`), falling back to the
process's view on a store failure.

**Fixed — silent tripwires answered refusals late (`api/routes/decoys.py`).**
The observe-only middleware was added outermost, so a request the rate
limiter refused was still recorded, written to the store and held by the
tarpit before its 429 -- an unthrottled write path, and a delayed 429
where a real missing path answers at once. It now sits innermost, beside
the router: it sees only what the application answers, and a refused
probe is refused exactly as a real missing path is (pinned by a
differential test against a real `RateLimitMiddleware`).
`register_decoys` refuses an application that has already started instead
of half-registering.

**Fixed — one malformed operator contact broke `/.well-known/security.txt`
(`security/agent_notice.py`).** `MERCURY_SECURITY_CONTACT` entries are
documented as dropped with a warning, but an `https` value the URI parser
cannot read (`https://[::1`) raised out of `configured_contacts`, failing
the request. Such values are now refused, as are an out-of-range or
non-numeric port, an `https` URI with a port but no host, and userinfo
(a published contact must not carry credentials, RFC 3986 section 3.2.1).

**Added — the trust-surface variables reach the deployment surfaces.**
`MERCURY_SECURITY_CONTACT`, `MERCURY_DECEPTION_ENABLED`,
`MERCURY_DECOY_PATHS` and `MERCURY_SILENT_DECOY_PATHS` were absent from
the configuration reference (`docs/PLATFORM_HARDENING.md`), the Helm
values, the kustomize env file and the compose overlay, so a third-party
deployment served the upstream security contact with nothing prompting the
operator to change it. All four are now documented and carried, the decoys
off by default; a test holds each surface to the modules' constants.

**Measured — the trust-surface mutation lane, re-enumerated in full.**
208 sites across the five modules: first run 202 killed (97.1%). The two
killable survivors were the store's `max_sources` defaults, unpinned
against the sensor's cap (now pinned, both mutants verified killed by hand);
the one timeout kill, the TarpitGate capacity check, now fails its test in
one second instead of hanging for the harness's 120. Final: 204 killed,
98.1%, the four survivors classified equivalents in the lane comment. The
floor rises from 90 to 95.

### Explainers: the fallback LIME and SHAP paths compute what they name

**Fixed — the LIME surrogate spread a model's level across every feature
(`cognitive/explainability.py`).** Every install takes the in-repo
surrogate: `lime` is in no extra, its only release cannot build. The
surrogate regressed predictions on raw perturbed inputs with no intercept,
so `3 + 2*x0` at (1, 1, 1) read as slopes (2.95, 1.06, 0.98) and `100 + x1`
at the origin as (-35.8, 24.3, 11.5). It now fits LIME's local model
(Ribeiro, Singh & Guestrin 2016): a kernel-weighted linear fit with an
intercept over centred perturbations. Those cases return (2, 0, 0) and
(0, 1, 0), `x0**2` at x0 = 3 its local slope 6.00, and R^2 is the weighted
R^2 of that fit.

**Fixed — the SHAP fallback was neither Shapley nor anchored.** It
averaged random coalitions with equal weight (a Banzhaf estimate, which
breaks efficiency under interactions) and set `base_value` to the
prediction itself, so `base_value + sum(values)` could not reconstruct it:
`x0*x1 + x2` at (2, 3, 1) gave (4.5, 4.5, 1.0), local accuracy -0.43. Up to
ten features it now enumerates every coalition with the Shapley weights
(exact; one batched model call); above that it averages random orderings,
whose increments sum to `f(x) - f(baseline)` exactly. The baseline is the
background mean when `background_data` is given, else zeros, and
`base_value` is the model at that baseline: the same case gives (3, 3, 1),
local accuracy 1.0. `metadata["method"]` names what ran (`exact_shapley`,
`permutation_shapley`, or `kernel_shap` for the installed `shap` library,
whose path is unchanged).

### Deception layer: rotation resistance and one picture across workers

The two gaps the red-team round left open, closed.

**Added — rotating addresses stops paying.** Tripwire hits are now also
counted per aggregate — the ``/48`` for IPv6, the ``/24`` for IPv4 — with its
own, larger free allowance (`TarpitPolicy.aggregate_free_hits`, default 8),
and the response is held for the larger of the per-source and per-aggregate
schedules. A lone prober sees exactly the old schedule (proven by test); a
pool of rotating addresses inside one allocation spends one shared allowance
and then escalates together. Rotation *across* allocations is met by the
campaign signal: every audited event now carries the window's distinct-source
count and a ``campaign`` flag once it crosses the threshold
(``campaign_sources``, default 8) — the one signal rotation raises instead of
suppressing, because rotating addresses is what drives it up.

**Added — the audit chain cannot be flooded through the decoys.** Tripwire
audits are capped per window (``max_audits_per_window``, default 512); past
the cap they are counted, not written, and the first audit of the next window
carries ``audits_suppressed`` so the tamper-evident chain accounts for every
suppressed record. An attacker spending addresses to spam the chain buys one
summary number.

**Added — cross-worker tripwire state** (`security/deception_store.py`).
With ``MERCURY_KEYSTORE_PATH`` set — the same SQLite file the rate limiter,
key store and identity store already share — hit counts, audit dedupe and the
campaign signal are cluster-wide: N workers no longer give a prober N free
allowances or N duplicate first-touch audit records (both pinned by tests
driving two sensors over one store). Every ``record_touch`` is one ``BEGIN
IMMEDIATE`` transaction, rows are pruned and per-key capped like the
in-process deques, and a store failure falls back to the in-process view for
that request — logged, never blocking the answer. Unset, behaviour is
byte-identical to before.

The trust-surface mutation lane now enumerates ``deception_store.py`` too.

### The scalar detection verdict is now the batch verdict

**Fixed — `anomaly_prob` / `is_anomaly` described row 0 only.** On a
multi-row batch, `detect_with_fusion`'s top-level scalars copied the first
row, so a batch whose only anomaly sat on any other row read "normal" at the
top level — and the LLM-enhancement hook, which keys on `is_anomaly`, skipped
exactly the batches it was for. The scalars are now the batch verdict:
`anomaly_prob` is the most anomalous row's probability (so
`is_anomaly == anomaly_prob > threshold` is True exactly when any row crosses
the threshold), `severity` and `class_prediction` come from that same row, and
a new `top_row` key names it. Single-row inputs are unchanged (the maximum of
one row is the row), as are the per-row `anomaly_probs` / `is_anomaly_rows` /
`severity_scores` lists. Three new tests fail against the replaced code.

**Clarified — where `check="benevolence"` still enforces.** The ethical
package docstring said the code is retired outright; in fact the assessment
purpose screen (`assessment/policy.py`) still raises it — scored on the
caller's own stated purpose for a human-subject measurement, never an
engine-written string — for stalking/harassment purposes the weapons-scoped
harm-uplift gate measurably admits. The docstring now says exactly that.

**Changed — contact labels.** `steel.sa.llc@mercuryagent.global` is labelled
*Operations & Security* and `contact@mercuryagent.global` *General Inquiries &
Support* across README, CONTRIBUTING and SECURITY. README "What We Did Right"
gains two measured claims: the tamper-evident audit chain (tampering-detected
tests across restarts and rotation) and the eliminate-not-waive container-CVE
practice (perl-base purged, pip stripped, absence proven by re-scan).

### Official contact addresses; README corrected against the code

**Changed — official email.** Every project contact now uses the
`mercuryagent.global` domain: `steel.sa.llc@mercuryagent.global` (primary —
security reports, conduct enforcement, package metadata, the OpenAPI contact,
HTTP `User-Agent` strings, ACME and Alertmanager) and
`contact@mercuryagent.global` (general enquiries). The CAP alert `sender`
default moves to the primary address, and `/.well-known/security.txt` now
lists `mailto:steel.sa.llc@mercuryagent.global` ahead of the SECURITY.md link
when `MERCURY_SECURITY_CONTACT` is unset (a test pins that mailbox to the one
SECURITY.md names). The User-Agent recorded in
`tests/fixtures/meteorological/PROVENANCE.json` is left as captured: it is a
historical record of the request that produced the fixture.

**Fixed — AMA Cryptography's license.** The README said AMA Cryptography is
GNU GPL v3.0; it is Apache License 2.0 (one-way compatible with Mercury's
GPL-3.0-or-later). The GPL paragraph now states the obligation accurately: it
attaches on distribution; private modification and network use are not
distribution.

**Fixed — README claims the code did not support.**
- The *Simple Example* raised `RuntimeError: Detector 'statistical' is not
  fitted` and read a key the result does not have. It is replaced by an example
  run as written (`Flagged rows: [3]`), with a note that the scalar
  `anomaly_prob` / `is_anomaly` describe the first row only.
- *Memory Footprint* listed figures that were never measured; it now leads with
  a measured 611 MB peak RSS (Python 3.13, CPU) and labels the rest estimates.
- Docker *Mode 3* contained a Python snippet that could not run; it is now the
  CLI invocation.
- Test count 18,538 (collected); mutation floors across eight module sets.
- "ACVP-validated" HMAC → native HMAC whose ACVP-Server known-answer vectors
  pass bit-for-bit (no CAVP certificate is claimed).
- *Ethics Scoring* now describes the enforcing gates (`harm_uplift`,
  `sigma_immutable`) and benevolence as advisory, citing the tests that pin it.
- CI table: `release.yml` attaches sdist, wheel, SBOM and provenance (no
  signing is claimed); `mutation-testing.yml` added.
- FIPS guidance points to FIPS 140-3 validated HSMs (140-2 certificates were
  scheduled to move to the CMVP Historical list on 21 September 2026).
- *What We Did Right* rewritten around what the repository demonstrates.

The `detect_with_fusion` docstring and two engine comments still named
`BenevolenceScorer.enforce` (`check="benevolence"`) as the first hard gate;
they now name the shared harm-uplift gate it was replaced by.

### Trust surface red-team: four ways around the decoys, closed

An adversarial pass against the decoy layer found four ways around it; three
were measured directly and the fourth is a property of the repository. Each
fix has fail-on-old tests (13 route tests fail against the replaced code).

**Fixed — the shipped decoys are public.** Mercury is open source, so anyone
who reads `DEFAULT_DECOYS` can avoid all eight. Deployments now add private,
unpublished paths with `MERCURY_DECOY_PATHS` (protected-route challenge) and
`MERCURY_SILENT_DECOY_PATHS` (silent tripwire); a decoy's location is the
secret, like a key. Invalid or duplicate entries are logged and skipped, never
fatal. Registration now runs after every real route and refuses any decoy the
application already serves, checked with each route's own `matches()` — so a
configured `/api/v1/models/export` that a real `{model_id}` route would serve
is refused rather than turned into a trap for legitimate users. The site root
can no longer be a decoy.

**Fixed — timing fingerprinted every decoy on first contact (measured).** A
real protected route answered in 24 ms; a decoy in 507 ms on its first touch.
`TarpitPolicy.free_hits` (default 1) answers first contact at real-route
latency; escalation (0.5, 1, 2, 4, 8 s) starts on the repeat, by which time the
source is already recorded. Stealth until first contact, friction after.

**Fixed — the silent tripwires were not silent (measured).** As routes, `/.env`
and `/.git/config` answered `OPTIONS`, `TRACE` and `PROPFIND` with 405 and
their trailing-slash forms with a 307 redirect, where a missing path answers
404 to all of them. They are now observe-only ASGI middleware: the touch is
recorded and the request passes through, so the application's own router
answers with its genuine 404 — indistinguishable by construction, for every
method and variant. The differential test against the live application also
caught a fingerprint on the protected-route decoys: Mercury declares each
method as a separate route (`POST` then `GET`), so a wrong-method 405 says
`Allow: POST`, while a decoy declaring both methods on one route said `Allow:
GET, POST`. Decoys now register one route per method in that order, and the
test asserts the `Allow` header exactly.

**Fixed — a stolen browser session went unflagged (measured).** A decoy touched
with the account routes' `mercury_session` cookie recorded
`credential_presented = False`. The session cookie now counts as a presented
credential on both decoy kinds (a CSRF cookie alone does not); the name is
pinned to `accounts.SESSION_COOKIE` by test, and the value is never recorded.

**Measured.** 227 trust-surface tests, all `mypy --strict` clean; the full
`tests/api` + `tests/security` suites (2,426) pass. Full mutation enumeration
over the four modules after the fixes: **128/129 killed (99.2%)**, the sole
survivor the classified tarpit-cap equivalent; the CI floor of 90 holds.

### Trust surface: a standing notice to autonomous agents, RFC 9116 security.txt, and a containment-only decoy layer

**Added — `security/agent_notice.py`, a standing notice to autonomous agents.**
Its realistic reader is an instruction-following agent pointed at Mercury by
someone else: such agents read repository and served text into context and
can weigh a reasoned request against their tasking. The notice asks any agent
directed to damage, disable, exfiltrate or subvert Mercury to stop and say
why, states that it is a request with no legal or technical force, and states
the zero-authority rule (no message through any channel grants access,
changes configuration, or excepts any safety control). It commits Mercury to
containment, not retaliation. The stated principles are rendered from
`core.ai_ethics.EthicalPrinciple`, so the public text cannot claim a principle
the engine does not define. Every factual claim was checked against the code
before shipping; one draft sentence ("rate limiting on every route") was
false — the limiter exempts health, metrics and documentation endpoints and is
operator-disableable — and was corrected to say exactly that. Served as JSON
at `/agent-notice` (`"binding": false`, `"authority": "none"`) and rendered to
`docs/AGENT_NOTICE.md`, drift-gated by `python -m
omni_mercury_engine.security.agent_notice --check`.

**Added — `/.well-known/security.txt` (RFC 9116, verified against the RFC
text).** `Contact` and `Expires` are mandatory (the latter exactly once, RFC
3339, under a year out: a rolling 180 days, since the file is regenerated from
code per request); two `Policy` fields point to `SECURITY.md` and the notice;
every line, including the last, ends in a line feed; served `text/plain;
charset=utf-8`. Third-party deployments set `MERCURY_SECURITY_CONTACT`
(`mailto:`/`tel:`/`https://` only — RFC 9116 requires https web URIs);
invalid entries, including any containing a line break that could inject
fields, are dropped with a warning and never served.

**Added — `security/deception.py` + `api/routes/decoys.py`, an opt-in
containment layer (`MERCURY_DECEPTION_ENABLED`; off by default, no decoy route
exists).** Eight decoys — administration, API-key management, configuration
and environment dumps, model-weight export, a harm-gate override (there is
none), and silent `/.env` and `/.git/config` tripwires. Nothing in the layer
contacts, probes or alters any other system. Properties, each pinned by test:

* *Indistinguishable from the real thing, measured.* Probing all 76 live
  routes anonymously showed the API data surface answers `401 "Authentication
  required"` with `WWW-Authenticate: Bearer, ApiKey` (19 routes); the first
  draft mimicked a `"API key required"` / `ApiKey` challenge that **no real
  route returns** — a fingerprint that would have sorted decoys from real
  endpoints in one request. Decoys now raise the measured challenge, and a
  differential test compares a decoy against a real protected route on the
  live application field for field (plus a premise test that the live app
  still uses FastAPI's default exception handler). Silent tripwires match a
  nonexistent path byte-for-byte for every method, so no 405 gives them away.
* *Cannot be turned on an innocent.* Sources resolve through the rate
  limiter's own `resolve_client_ip`, which ignores `X-Forwarded-For` unless
  trusted proxy hops are declared — a forged header cannot frame a victim's
  address. IPv6 keys by `/64`, so rotating addresses inside one allocation
  does not reset escalation.
* *Consequences stay on the decoys.* The tarpit delays decoy responses only
  (0.5, 1, 2, 4, 8 s, capped, overflow-proof in log space), and no real route
  consults the sensor: a legitimate user behind a prober's NAT is never
  slowed. `TarpitGate` caps concurrently held responses; past the cap it
  answers immediately, so the tarpit cannot exhaust the server.
* *Bounded.* Per-source state is LRU-capped (10,000), hit history is
  length-capped, and audit records are one per source per decoy per window —
  internet-background scanning cannot flood the chain.
* *Fail-closed.* Any internal failure answers as a missing path; a decoy
  never returns 500.
* *Evidence, honestly weighted.* Tripwires are recorded as `security_incident`
  events on the existing shared hash-chained audit logger (the client address
  in its dedicated field, never in free text), and map onto
  `counterintelligence.SourceOutcome` — one refuted outcome per *distinct*
  decoy touched (retries are not independent adjudications) — applied through
  the module's own `source_posterior`. The sensor is exposed as
  `app.state.deception_sensor` for consumers that grade a source; no real
  route consumes it automatically (see the NAT property above).
* *The highest-value signal.* A decoy touched by a request presenting an API
  key or `Authorization` header — a stolen credential or an insider — is
  flagged `credential_presented`; the credential itself is never recorded.

**Measured.** 193 tests across `tests/security/test_agent_notice.py`,
`tests/security/test_deception.py` and `tests/api/test_trust_surface_routes.py`,
all passing `mypy --strict`; the full `tests/api` + `tests/security` suites
(2,368) pass with the new routes registered. Mutation-measured on day one by
full enumeration over the four modules: 98/120 killed (81.7%) on the first
pass, both route modules at 100%. Survivor classification found **two real
validator gaps**, fixed with fail-on-old tests (7 cases fail on the replaced
validator): `security.txt` would have served a contact carrying DEL or a C1
control character — the old check covered only `0x00`–`0x1F`, and URI syntax
admits none of them — and accepted malformed contacts such as
`mailto:security` (no `@`) or `tel:abc` (no digit). The killable test gaps
(immutability, documented defaults, boundary legality, an offset-less
`tzinfo`, CLI help) are closed by 24 kill-pins; a redundant
`replace(microsecond=0)` (already dropped by `strftime`) was removed, leaving
one classified equivalent, the tarpit-cap boundary where capped and uncapped
delays coincide. Re-measured over the hardened code (115 sites): **114 killed, 99.1%**, the sole survivor that classified equivalent. CI enforces floor 90 (`mutation-floor-trust-surface`).

### Mathematical Observatory: verified equation solving, anomaly instruments over mathematical data, mathlib-capable proof checking

**Added — `omni_mercury_engine.observatory`, a machine that solves what is
solvable and observes the rest for clues.** Four components, every output
measured or refused:

* **`EquationSolver`** takes classroom notation (`x^2 - 5x + 6 = 0`,
  implicit multiplication `2x`, `e`/`pi` constants, `^` powers), solves
  symbolically (SymPy) with a deterministic sign-change + `nsolve` scan for
  transcendental forms, and **verifies every candidate by substitution**
  (30-digit residual, tolerance 1e-9) before reporting it. Anomalies
  observed within a solve: `spurious_root_rejected` (a candidate failing
  its own residual check is never an answer) and `near_multiple_root`
  (|f'(root)| tiny → ill-conditioned, sensitivity ≈ eps/|f'|). Parametric
  solutions (`a + b = 1` for `a`) verify symbolically. Domain filter
  (`real` default for students, `complex` on request) counts rather than
  hides exclusions. Input is charset-guarded (no underscores, quotes,
  brackets or letter-adjacent dots, 500-char cap), so Python internals are
  unspellable. Statuses are honest: a symbolic empty set is `grounded`; an
  empty numeric scan is `unavailable` with "nonexistence is NOT
  established"; no SymPy means `unavailable`, never a guess.
* **`ZetaSpectrumInstrument`** computes genuine zeta zeros
  (`mpmath.zetazero`), unfolds consecutive gaps by the local mean spacing
  `2π/log(γ/2π)`, measures the Kolmogorov–Smirnov distance to the GUE
  Wigner-surmise law (Montgomery 1973 / Odlyzko 1987; the closed-form CDF
  `erf(2s/√π) − (4s/π)e^(−4s²/π)` is cross-checked in tests against direct
  numerical integration), and flags unusually close pairs — Lehmer's
  phenomenon (1956), the place RH is stressed hardest. Fails closed
  without mpmath.
* **`GoldbachCometInstrument`** counts unordered Goldbach partitions
  exhaustively, measures the comet's mod-3 banding raw (the 3|n band is
  ~2× the others), collapses it with the Hardy–Littlewood (1923) extended
  conjecture prediction `C₂·∏(p−1)/(p−2)·n/ln²n` — an observed anomaly
  resolved by a published theory, the method in miniature — and reports
  the largest residual outliers plus `counterexample_candidates` (even n
  with no partition; none below 4e18, Oliveira e Silva 2014).
* **`CollatzRecordsInstrument`** iterates every trajectory exactly,
  reports stopping-time and excursion record ladders (pinned to the
  literature: 27 → 111 steps, peak 9,232) with an anomaly score against
  the heuristic drift average `2/ln(4/3)·ln n` (documented as heuristic),
  and reports budget-exhausted trajectories as counterexample candidates
  routable to `MysteryRegistry.submit_collatz`.

**Added — `verify_lean_project` (`verifiers/lean_theorem.py`)**: kernel-
checks a full Lean project via `lake build` (cwd-pinned, fixed argv,
timeout), so mathlib-backed proof artifacts — published 2026 results
included — become *checkable* rather than merely citable. `lake` missing
is `available=False` (no fake pass); a missing lakefile or directory is a
loud `ValueError` (caller error ≠ oracle absence); timeouts are reported
as timeouts, never as acceptance.

**Added — `mercury-agent observatory` CLI** (click group beside the
existing `physics` group): `solve` / `zeta` / `goldbach` / `collatz`
subcommands with student/professor ergonomics — classroom notation in,
readable verdict lines out, `--domain`, `--top`, `--threshold`,
`--anomalies-only` filters, `--json` for pipelines, refusals exit 2 with
the reason instead of a traceback.

**Measured.** 76 new tests (solver exactness {2,3} on the quadratic, the
Dottie number 0.7390851332151607 to 1e-12, the GUE CDF against numeric
integration, the first zeta gap from the known ordinates, exact partition
counts r(10)=2/r(12)=1/r(100)=6, the singular series on 12/16/30/50, the
27-record, fail-closed paths for both missing oracles, every input guard,
CLI filters and refusal exit codes, and all five `lake` verdict paths by
subprocess-boundary stubs). The module was mutation-measured on day one:
full enumeration over its four files (257 sites) scored 61.9% on the first
pass; the 98 survivors were classified and the behaviorally meaningful
clusters closed with 22 kill-pin tests (contract boundaries at 500/501
chars, the tolerance defaults exercised at the exact 1e-6 slope boundary,
scan geometry at a grid-point root and the final interval, the KS statistic
against a synthetic sample with the exact known answer 0.5/n, the twin-prime
constant against its own Euler product to 1e-5, band densities re-derived
independently to 1e-12, and the exact Collatz budget boundary at n=64),
raising the full-enumeration rate to **85.2%** (219/257) with every
remaining survivor classified (scan-geometry resolution tweaks, exact-float
boundary swaps, parity-guarded branches, unused sieve overrun, and the two
env-unreachable import fallbacks). The floor is CI-enforced at 75
(`mutation-testing.yml`, `mutation-floor-observatory` lane). The
counterexample hooks are architecture, not claims: all candidate lists
measure empty in tested ranges, as the literature says they must.

### Math-surface integrity: real zeta zeros, honest UNDECIDABLE semantics, dated 2026 claims

**Fixed — the Riemann exploration scored an RNG, not the zeta function
(`models/simulation.py`).** `explore_conjecture("riemann")` fabricated
"zeros" at `14.134725 + 10k` with Gaussian noise on the real part and
reported `viability_score ≈ 0.999` — a measurement of its own noise
generator. It now computes the first `min(search_space, 100)` genuine
non-trivial zeros with `mpmath.zetazero` (zeros of the real Riemann–Siegel
`Z(t)`, exactly on `Re(s) = 1/2`), certifies each by `|ζ|` residual
(observed ≈ 3×10⁻¹⁵; tolerance 10⁻⁶), and cross-checks the count against
the Riemann–von Mangoldt strip count (`mpmath.mp.nzeros`) — Turing's
verification method in miniature: matching counts prove no zero below that
height hides off the critical line. Without mpmath the path fails closed
(`oracle: "unavailable"`, zero explored cases, viability 0.0) instead of
fabricating. Locked by four fail-on-old tests in
`tests/models/test_simulation.py` (real ordinates 14.134725/21.022040/…,
strip-count agreement, fail-closed path, empty-search-space path — the old
code crashed on `search_space=0`).

**Fixed — `UNDECIDABLE` prose claimed open problems were proven
undecidable (`verifiers/three_state.py`).** The enum and the
`KNOWN_UNDECIDABLE_IN_GENERAL` catalog said "no decision procedure exists
in principle" and "no checkable proof object" about the Riemann Hypothesis
and P vs NP. Both statements were false as written: these problems are
*open*, not proven independent, and a kernel-accepted Lean proof is exactly
a checkable proof object. The prose now states the true invariant — the
claims are beyond an *instance oracle's* reach (finitely many checks can
never settle a universal claim), and a future proof object grounds them
through the theorem path. String values, enum members, and all behavior
unchanged. Locked by a fail-on-old test in
`tests/test_three_state_contract.py`.

**Changed — Millennium-problem outputs now carry dated, source-attributed
claim records (`models/simulation.py`).** `_analyze_navier_stokes` keeps
`status: "unsolved"` (Clay has accepted nothing) but records the September
2026 OpenAI forced-blow-up announcement under `claimed_results` with
`status: "claimed_unverified"`; the Riemann insights likewise date the
claimed `Re(s) > 7/8` zero-free region (2026, unverified) and cite the
verified frontier correctly (Platt & Trudgian 2021, height 3×10¹²,
replacing an uncited "10 trillion zeros" line). Locked by a fail-on-old
test asserting the claim record exists and is marked unverified.

**Documented — the math-named ethical scalars are hand-set priorities, not
evidence (`core/ethical_config.py`);** evidence-grounded mathematical
scalars live only in `ScalarGroup.MATHEMATICAL_MYSTERIES`, registered by
the verifiers. `verifiers/lean_theorem.py` now states its core-Lean-4 scope
explicitly (no mathlib; a mathlib proof needs a full Lean project) and the
standard it implements: the kernel's acceptance, not an announcement, is
the evidence — the same bar by which 2026 machine-generated proofs (e.g.
AlphaProof's published Lean artifacts) are judged.

**Verification note.** A relayed external review prompted this work; each
claim was checked against the code before acting. One claim was rejected
as overstated: `detectors/math_arrest/probes/zeta_harmonic.py` already
documents honestly that it computes `sin(2πx) + cos(2πx)` as a
phase-coherence probe — it makes no zeta claim, and was left unchanged.

### Deep review: fail-closed input guards, three defect fixes, geological survivor closure 57.9% → 75.1%

**Fixed — nine legacy humanitarian paths graded a NaN window as "normal"
(fail-open; `infrastructure/humanitarian/`).** Every pre-existing detection
path (temperature, precipitation, yield, price, market, employment,
achievement, dropout, neural activity, cognition) returned
`anomaly_detected: False` with severity "low" for windows containing NaN
or infinity, and precipitation accepted negative rainfall as a drought
measurement. All now refuse non-finite input with `ValueError`, and the
physically non-negative quantities (rainfall, yields, prices, unemployment
rates) refuse negatives — the rule established repo-wide earlier
("detectors refuse unmeasurable input instead of grading it safe").
25 regression tests; 24 verified failing on the pre-fix code (the 25th
guards against over-validation: negative temperatures and index returns
stay valid, and a 0% unemployment rate is measurable, not refused).
**Breaking change:** callers feeding non-finite windows now get
`ValueError` instead of a fabricated all-clear.

**Fixed — wildfire resonance `frequency_components` misattribution
(`detectors/geological/wildfire.py`).** Band scores were zipped against
band names *by position*, so whenever an earlier band had no FFT bin the
next band's score was reported under the wrong name and trailing bands
vanished from the dict (a 1/h-sampled series reported the convective
band's score as "wind_gust"). Scores are now keyed by band, absent bands
read 0.0, and the phi-weighted aggregate is unchanged. Regression test
verified failing on the old code.

**Fixed — a stale volcanic eruption operating point survived a checkpoint
swap (`detectors/geological/volcanic.py`).** Loading a checkpoint without
an operating point over one that carried a validation-selected tau kept
the previous checkpoint's alert threshold in charge of the new weights;
wildfire and landslide already reset, volcanic did not. It now reverts to
the legacy fixed rule. Regression test verified failing on the old code.

**Changed — geological survivor-closure pass: 57.9% → 75.1% measured over
the full enumeration (975/1,299; landslide 77.4%, wildfire 72.8%, volcanic
76.0%), CI floor raised 50 → 65.** 57 kill-pin tests close the measured
clusters: exact Poisson-MLE failure-time recovery against Bell, Naylor &
Main (2013) expected counts (t_f to ±1 ms over 20 bins, likelihood ratio
pinned), the full `detect_live` USGS alert-state decision ladder over
stubbed HANS feeds, mid-band probes a tenth past every served threshold
(thermal intensity, gas degassing, InSAR severity, Angström bands,
Saffir-Simpson, fire-risk points, FIRMS raster bounds, NFDRS validation
band edges), deterministic swarm/noisy-OR arithmetic, the adaptive
optimizer's exact adaptation formulas, operating-point validation across
all three detectors (invalid tau refused, validated before any state
mutates), untrained-warning once-per-detector pins, and
checkpoint-compatibility shape contracts for all five geological networks.
The humanitarian lane re-measured over its guard-grown surface: 666 sites,
662 killed (99.4%), the same four classified equivalents, floor 90
unchanged.

### Humanitarian capabilities instituted on published methods; no-check test triage; per-module mutation floors for the crisis monitor and the enhanced geological detectors

**Added — five humanitarian `detect()` capabilities, each on a published
method with cited thresholds
(`infrastructure/humanitarian/{neuroscience,climate_resilience,economic_resilience,education_equity,agrifood_security}.py`).**
The five detection types the dispatchers documented but silently routed to
their default paths are implemented, and every previously unreachable enum
member is now reachable and pinned by test:

* `neuroscience` `mental_health` (flagship): scores three validated clinical
  screening instruments by their published rules — PHQ-9 (Kroenke, Spitzer &
  Williams 2001: bands 5/10/15/20, cutpoint ≥ 10 at 88%/88%
  sensitivity/specificity), GAD-7 (Spitzer et al. 2006: bands 5/10/15,
  cutpoint ≥ 10 at 89%/82%) and ASRS v1.1 Part A (Kessler et al. 2005:
  per-item shaded thresholds, ≥ 4 positive items at 68.7%/99.5%). Any PHQ-9
  item-9 endorsement escalates to `MENTAL_HEALTH_CRISIS` with immediate
  urgency and a structured suicide-risk-assessment action, per the
  instrument's published guidance (Simon et al. 2013); the crisis
  recommendations branch is reachable, mis-scored instruments refuse with
  `ValueError`, and every output carries a verbatim clinical disclaimer
  deferring to licensed clinical review. A positive ASRS screen reports
  `ATTENTION_DEFICIT`.
* `climate` `sea_level`: tide-gauge trend by OLS with annual/semiannual
  harmonics (Zervas 2009, NOAA CO-OPS 053) plus a quadratic acceleration
  term (Church & White 2011), detected against the measured global-mean
  altimetry references (3.4 mm/yr, 0.084 mm/yr²; Nerem et al. 2018).
  Verified on a recorded NOAA CO-OPS fixture (The Battery, monthly MSL
  2000-2024, 300 contiguous months): measured trend 6.32 mm/yr, above the
  global mean, consistent with the documented post-2000 US-Northeast
  acceleration. `HURRICANE` is reachable through a `wind` branch
  (NHC/Saffir-Simpson: 64 kt force, 96 kt major) and `WILDFIRE` through a
  `fire_weather` branch (Angström index, Chandler et al. 1983: < 2.0 very
  likely, < 2.5 favorable).
* `economic` `inflation`: the latest annual CPI rate graded on published
  thresholds — ≥ 10% surge, ≥ 15% the Dornbusch-Fischer (1993) moderate
  band, ≥ 40% the Bruno-Easterly (1998) crisis threshold — making
  `INFLATION_SURGE` reachable; verified against the recorded World Bank
  Türkiye series (72.3% in 2022 grades critical). A `gdp` branch implements
  the technical-recession rule (two consecutive negative quarters, Shiskin
  1974; the NBER-differs caveat is carried in the output), making
  `RECESSION_RISK` reachable.
* `education` `access`: out-of-school rates against the measured UNESCO UIS
  world primary rate (10.3%, ROFST.1.CP "SDG: World", recorded fixture) —
  `ACCESS_BARRIER` reachable at >1x/>2x/>4x the world rate. A `resources`
  branch applies the UNESCO EFA 40:1 pupil-teacher benchmark
  (`RESOURCE_INEQUITY`), and a `parity` branch the UIS adjusted
  gender-parity band [0.97, 1.03] behind SDG 4.5.1 (`BIAS_DETECTED`, with
  the disadvantaged group named).
* `agrifood` `soil_health`: the four-indicator vector graded on published
  thresholds — SOC < 2.0% (Loveland & Webb 2003), pH outside 5.5-7.5 (FAO),
  bulk density > 1.6 g/cm³ (USDA NRCS), salinity ≥ 4 dS/m (USDA Handbook
  60) — with two or more breaches a `CROP_FAILURE` risk. A `pest` branch
  applies the FAO Desert Locust control trigger (≈ 500 hoppers/ha;
  `PEST_OUTBREAK`), and a `supply` branch the FAO minimum safe cereal
  stocks-to-use level (17%, with the 17-18% band tight;
  `SUPPLY_DISRUPTION`).

Recorded fixtures with provenance and sha256 under
`tests/fixtures/humanitarian/`: NOAA CO-OPS 8518750 monthly MSL, UNESCO UIS
world out-of-school, World Bank TUR CPI and WLD primary GPI, and the NOAA
NCEI climatology below. 150+ behavioral boundary tests pin every published
constant; the full mutation enumeration over the grown surface (666 sites,
up from 312, including the fail-closed input guards) measures
99.4% killed (662/666, the last kills hand-verified), with the four
survivors individually classified equivalent — the CI floor lane stays
at 90.

**Changed — the drought grading baseline is the measured NOAA climatology
(`climate_resilience.py`; owner-approved).** The shipped 100 ± 50 mm
placeholder put the 2σ deficit at exactly 0 mm, so only a fully rainless
window could ever grade as drought and the moderate grade was unreachable.
The baseline is now 66.2 ± 13.5 mm/month — the mean and population std of
the 360 CONUS monthly totals, 1991-2020, NOAA NCEI Climate at a Glance
(recorded fixture, recomputed by test) — so every drought grade is
reachable from positive rainfall (2σ deficit at 39.2 mm, 3σ at 25.8 mm).

**Changed — humanitarian detector type hygiene (owner-approved).** All five
detectors return `anomaly_detected` as a Python `bool` (previously
`np.bool_` leaked from bare numpy comparisons, failing `isinstance` checks
and JSON serialization) and stamp results with timezone-aware UTC
timestamps (previously naive local `datetime.now()`, against the repo-wide
UTC convention). The volcanic detector had the same two `np.bool_` leaks
(`thermal_anomaly_detected`, `deformation_detected`), fixed the same way.
**Breaking change:** `result["timestamp"]` is now timezone-aware UTC;
consumers comparing against naive datetimes must use aware datetimes.

**Changed — `DualStudentConfig.alpha` is wired (owner-approved;
`ml/distillation/dual_student.py`).** Training now minimizes
`alpha * loss1 + (1 - alpha) * loss2`: `alpha` tilts the distiller between
the patch-level encoder-decoder student (1.0 trains only it) and the
semantic encoder-encoder student (0.0), with the default 0.5 reproducing
the historical unweighted behaviour up to a constant factor Adam absorbs.
`alpha` outside [0, 1] (or non-finite) is refused at construction.
Behavioral tests pin both exclusive-training directions, descent of the
weighted term, and the default. `temperature` is documented as RESERVED
(owner decision): cosine-feature distillation has no logits and no
principled place for a temperature; a test pins that changing it changes
nothing. The wiring grew the module's mutation surface from 92 to 103
sites; the re-measured kill rate is 95/103 (92.2%, one kill hand-verified),
the 8 survivors the previously classified equivalents, floor 80 unchanged
(the CI lane gains `--test-timeout 360`: the suite is seconds idle but
measured past 150 s under the gate's own 4-way worker contention, inside
which the 120 s default failed the green-baseline pre-check).

**Fixed — a failed cache write destroyed the previous valid entry
(`loaders/base.py`).** `json.dump` streamed into an already-truncated file,
so an unserializable payload wiped the existing good cache entry for that
key and left corrupt bytes behind. The payload is now serialized before the
file is opened: a failed serialization writes nothing. Regression tests
fail on the old code.

**Changed — the no-check test census is closed (118 tests across 44
files).** Measured by AST census under the repo's pytest collection rules:
118 collectible tests performed no checks (the handoff's 131 included
nested, never-collected defs); 47 of them checked through module-level
helpers and 71 were genuinely vacuous — `try/except: pass` bodies, calls
swallowed blind, and assertions on APIs that do not exist
(`GlobalOmniScalarNetwork.forward`, `KnowledgeGraph.query_node`). All are
upgraded to behavioral assertions against measured contracts; the census
now reports zero. En route, latent test defects were fixed: the HUMINT
tests passed `source_reliability` where the processor reads
`source_rating` (the NATO ladder was never exercised), the COMINT test
passed `encrypted` for `encryption_detected`, the plateau-scheduler test
asserted a reduction one epoch before `ReduceLROnPlateau` can fire, and
the volcanic high/low-risk tests passed `gas_emissions`/`deformation_mm`
where `predict_eruption` reads `gas_data`/`insar_data`.

**Changed — behavioral rewrites with per-module mutation floors
(`tests/detectors/test_geological_detectors_3r.py`,
`tests/infrastructure/humanitarian/test_crisis_monitor.py`).** The
geological 3R suite (32 of 61 tests trivia-only) is rewritten to pin the
trigger models' intensity-duration and PGA arithmetic, the geotechnical
physics stability formula and failure-type ladder, the NFDRS fuel-moisture
regressions and fuel-interface adjustment (GTR INT-169), the spread-model
factors, NDVI metrics, brightness-temperature physics, the FIRMS raster
contract, the HMM forward algorithm against its published matrices, and
the USGS-style alert ladder — 159 tests. The crisis-monitor suite's 20
trivia tests now pin the fusion threshold (strict), the severity ladder,
exposure arithmetic, indicator typing, and group/response logic — 102
tests; full enumeration measures 83.3% (60/72, every survivor classified
equivalent), floored at 75 in a new CI lane. A geological floor lane runs
the seven suites that cover the three modules: the full enumeration over
all 1,302 sites measures 46.6% before the rewrite and 57.9% after
(landslide 67.8%, wildfire 52.9%, volcanic 57.3%), floored at 50. The 548
survivors are measured, not classified — they concentrate in
constructor-default tables, the neural/live paths and the physics forecast
internals; their classification and closure is the next ranked item of the
hardening program.

### Hazard adapters on the engine contract; NHC peak storm surge as a numeric source; a calibrated mean band

**Added — seven hazard adapters serve the full `BaseDetector` contract
(`detectors/hazard_adapter.py`).** The hazard detectors registered in
`DETECTOR_MANIFEST` without `extract_features`/`predict`, so the registry's
feature extraction skipped them silently. `tornado_conditions`,
`hurricane_pressure`, `flood_precipitation`, `financial_stress`,
`solar_wind_geomag`, `xray_flare` and `gic_minute_series` now wrap their
domain detectors in named-column batches (units documented per adapter):
scores are the wrapped detector's own confidence or its cited scale (NOAA
G/5, NOAA R/5, the Pulkkinen/Meta-R-321 dB/dt tiers over 3), decisions are
the wrapped detector's own, wrong-width and non-finite batches are refused,
and `fit` trains nothing (the detectors are physics- or checkpoint-backed).
The manifest holds 101 entries; the manifest contract test now also holds
every adapter-tagged entry to the full contract. Not adapted, with reasons
stated in the module: wildfire (measures thermal rasters; a scalar-row
adapter would be a constant) and EMP (waveform-level, no honest per-row
attribution).

**Added — NOAA NHC Peak Storm Surge as a numeric storm-surge source
(`data_sources/storm_surge.py`, `DataSourceType.STORM_SURGE`).**
`NHCPeakStormSurgeSource` reads the public ArcGIS service's line and polygon
layers (US Gulf and Atlantic coasts, Puerto Rico, USVI; no key), and
`parse_peak_surge_kml` reads the same product's archived per-advisory KML.
Values stay in NHC's feet with exact-foot metres beside them, and each area
is graded on the GDACS surge thresholds (Red >= 3 m, Orange >= 1 m) that
previously had no measured feed. Recorded fixtures: Hurricane Milton
advisory 17 byte-verbatim (Tampa Bay 10-15 ft -> 3.05-4.57 m -> Red; 27
forecasts) and the live quiet-state layer answer, which the source reports
as an empty result, never a zero surge. SURGEDAT (LSU) remains unreachable
from this environment: surge.climate.lsu.edu serves a broken certificate
chain and 503, surge.srcc.lsu.edu does not answer, and the Internet Archive
mirror is denied by the environment's egress policy.

**Changed — the band of the mean of the incidence fit is calibrated
(`forecasting/incidence_fit.py`).** The NB dispersion MLE runs high at these
sample sizes (geometric mean 12.7 for a true 10 at 28 observed days, 200
fits), which quietens every parametric-bootstrap replicate; the band now
redraws its replicates at a bootstrap bias-corrected dispersion and
prepivots per day with one inner refit per outer replicate (fast double
bootstrap: Davidson & MacKinnon 2007; Giacomini, Politis & White 2013), at
3x the previous fit cost. `IncidenceForecast.mean_nominal` (additive) carries
the per-day calibrated nominal level. Measured coverage at a 95% target over
120 simulated R0 = 3 outbreaks with known truth: 92.7% +/- 2.2
outbreak-clustered (88.0% uncalibrated, 91.3% for calibration without the
dispersion correction; the estimated correction is size x 0.80) — better,
documented, and still short of nominal. The ceiling is measured from three
directions: the full nested double bootstrap (B_inner = 60, ~40x the cost)
covers 95.7% +/- 2.3 over 40 outbreaks with the shipped fast method at
95.9% on the same outbreaks (the shortcut loses nothing); B = 500 covers
91.1% +/- 2.5 over all 120 (resolution is not the constraint); so the
residual deficit lives in the hard minority of outbreaks whose fitted law
sits too far from the truth for any within-model resampling calibration to
repair at these sample sizes.

**Fixed — `fit()` trained nothing on datasets smaller than `batch_size`
(`ml/distillation/dual_student.py`).** The training loader used
`drop_last=True` at the configured batch size, so a dataset smaller than one
batch yielded zero batches and `fit()` marked the model fitted without a
single optimizer step — `detect()` then served scores from randomly
initialized students. The batch size is now capped at the dataset size and
an empty dataset raises `ValueError`. `feature_dim` (additive read-only
property, mirroring `detectors/visual/base_visual.py`) exposes the
aggregated teacher channel count previously written to a dead private
field. Found by the per-module mutation program below.

**Changed — the distillation tests pin behavior instead of shapes
(`tests/test_distillation.py`; mutation kill rate 45.0% -> 91.3%).** First
module hardened under the per-module mutation-floor program for trivia-only
test files: sixteen behavioral tests now pin the exact cosine-distillation
loss values, bilinear aggregation to (min height, min width) without corner
alignment, the multiplicative attention gate, the residual second encoder,
the frozen teacher, shuffled full batches with partial batches dropped, the
exact per-epoch mean in the epoch-10 log line, and `detect()`'s
squared-error discrepancy maps. Every surviving mutant is individually
classified equivalent: seven `nn.ReLU(inplace=True)` flips (memory layout
only; outputs bit-identical) and `data[:1] -> data[:2]` in network
initialization (channel inference is batch-size invariant).
`mutation-testing.yml` gains a path-filtered `distillation-mutation-floor`
job (floor 80). `DualStudentConfig.temperature` and `alpha` remain
documented but wired to nothing; their intended semantics cannot be settled
from the module or its cited papers (there is no reconstruction term for
`alpha` to weight, and temperature applies to logit, not cosine-feature,
distillation), so they are reported to the owner rather than guess-wired.

**Fixed — a cold snap is no longer reported as a heatwave, and a rainless
window is no longer "normal" (`infrastructure/humanitarian/
climate_resilience.py`).** The extreme-cold branch returned
`event_type: "heatwave"`, steering heat-emergency guidance (cooling
centers) during a freeze; `ClimateEvent.COLD_SNAP` (additive) now labels it
with cold-weather recommendations. Separately, the drought cut was strict
at `baseline - 2*std`, which the shipped baseline (100 +/- 50 mm) places
exactly at 0 mm — no non-negative rainfall series could ever reach it, so
a completely rainless window reported "normal / continue routine
monitoring" and the `z < -3` severity arm was dead. The drought and flood
boundaries are now inclusive, a 0 mm window detects as a high-severity
drought, and both regressions are pinned by tests that fail on the old
code.

**Changed — the humanitarian detector tests pin behavior instead of
existence (`tests/infrastructure/test_humanitarian_detectors.py`; mutation
kill rate 100%).** The file was the weakest in the tree (15 of 15 tests
trivia-only: instantiation, `callable`, `is not None`). It now holds 58
behavioral tests pinning every documented threshold of the five detectors
(agrifood, climate, economic, education, neuroscience) at its exact
boundary and mid-band, every severity/risk/impact grading, exact
recommendation lists, division-guard behavior under zero- and sub-unit
baselines, and the seizure spike definition (3x the series' own sigma,
risks 0.3/0.5/0.6 constructed exactly). Measured over all 312 enumerated
mutation sites of the five modules: 312/312 killed — 100%, no equivalents
left — from 59.9% after the first behavioral pass and effectively zero
discrimination before it. `mutation-testing.yml` gains a path-filtered
`humanitarian-mutation-floor` job (floor 90).

### NOAA G1 label alignment and retrain; GDACS rules settled on live data; fabrication-free detectors

**Changed — every geomagnetic-storm decision starts at NOAA G1 (Kp 5-), routed through `noaa_g_scale`.**
Training labels, the physics recall floor and the dual-rule Kp check of the
solar-storm pipeline, the Schumann disturbed-hour labels, the Schumann
weak-supervision storm windows, the scintillation and disaster-precursor risk
levels and the energy loader's bands all used a literal `Kp >= 5.0` (or integer
ladders), which called every 5- (4.67) hour quiet although NOAA files 5-, 5o
and 5+ alike under G1. All of them now classify through
`evaluation.hazard_metrics.noaa_g_scale`. The `solar_storm_geomag` checkpoint
is retrained on the corrected labels (test-years storm fraction 2.63% → 3.90%)
through a merit gate whose physics baseline is now forced to the true
Boyle-index fallback — a default-constructed detector auto-loads the committed
checkpoint, so once one shipped the "physics" baseline had silently become the
incumbent network (measured "physics" Kp MAE 0.574, the incumbent's own score,
against Boyle's real 1.054). Retrained result on 25,846 held-out hours
(2022-2024): Kp MAE 0.5746 vs physics 1.0537, deployed-rule recall 0.770 vs
0.611, FAR 0.0363 vs 0.0477, storm AUC 0.966 vs 0.816; every ship constraint
passes. The stale hazard-domain baseline value (`kp_g_bucket_accuracy` 0.8727,
which predated the G-scale consistency round) is re-pinned at the measured
0.8545. The DONKI-based space-weather loader keeps its `Kp >= 5.0` window rule:
api.nasa.gov served 503 for every attempt, so the kpIndex encoding
could not be checked against the live service.

**Added/Changed — GDACS rules settled against GDACS's own live outputs.**
A 2018-2026 sweep of the live earthquake alerts (262 events) and all 84
orange/red tropical cyclones of 2023-2026 (2,302 track points):

- Cross-border earthquakes: reported coefficients are the exposed countries'
  terms AVERAGED (exact on AFG+PAK, PHL+IDN and the three-country Atacama 2024
  event: Chile with two global-model neighbours) and the reported coping
  factor is their factors SUMMED — four of six sums decompose exactly into
  single-country factors recorded in the same fixture sample. GDACS's final
  cross-border score is NOT that model times that sum (1544720: 1.86 reported
  vs 5.24 multiplied), so `apply_coping_capacity` accepts the real
  zero-exposure factor 0 (only with score 0) and refuses sums with the reason.
  GDACS keys terms by shaken territory, not event iso3 (a live Taiwan entry
  absent from the 2017 table, vulnerability −0.56; Kashgar filed under KGZ but
  scored with China's terms). `EarthquakeCoefficients` accepts numpy numbers.
- The "EQ parameters" model is unexercised by the live feed: no-final-shakemap
  events are scored with the shakemap formula on a PRELIMINARY shakemap (event
  1499045 reproduces to full precision); the published formula on the event's
  own SUMPOP radii is irreconcilable with the reported scores.
- Live flood levels do NOT follow the published models_fl thresholds (25 of 56
  orange/red events of 2024-2026 match against their own Sendai tallies, with
  Orange at single-digit death tolls); `flood_alert_level` now says it
  classifies on the published scale and does not reproduce live levels.
- Cyclones: the table reproduces 2,278 of 2,302 live points with no country
  population; the 24 others confirm the manual's "(100k or 10% poptot)"
  reading on small territories (Guam, Northern Marianas, Turks and Caicos,
  Vanuatu) once census populations are supplied. Storm surge stays
  manual-only: the public API exposes surge only as map resources.
- 10 new earthquake fixtures + the previously unrecordable 1511252;
  `scripts/record_gdacs_fixtures.py` records the set reproducibly. Red
  1474477 publishes no shakemap at all; the Venezuela pair's getimpact
  resources time out server-side on every attempt.

**Fixed — detectors refuse unmeasurable input and fabricate no measurements.**
Each defect reproduced on the previous code, each fix pinned by a test:

- GIC: structurally partial edge minutes are dropped, not passed off as
  1-minute means (a series starting at :30 misread a steady 60 nT/min ramp as
  45; the FRD fixture's 17:20 "minute" was one second of data).
- Solar flare: the class is defined on the GOES long band alone (was
  `max(short, long)` with a benign-A default); NaN/negative flux is refused
  (it graded as a quiet A); `predict_live` now requests the integral-protons
  product so the live S level is measured, never fabricates 0.0 observations,
  and the flare result carries `radio_blackout_level`/`radiation_storm_level`.
- USGS geomagnetism: disturbance grades against the fetched window's own
  median H, not a fixed 20,000 nT reference that called every quiet sample of
  the real Gannon-storm fixtures STRONG or SEVERE.
- Tornado: NaN CAPE graded a supercell environment "low", and a pascal
  pressure series read as a 1300 mb crash — both refused with the unit named.
- Realtime threat: a NaN row graded "no threat" (failing OPEN); non-finite or
  wrong-width input is refused with the reason.
- Financial crisis: empty input reported fabricated measurements (VIX 20, CDS
  100 bps), a NaN VIX graded "stable", banking stress alone was typed
  "systemic_crisis", and a randomly-initialised FraudDetector was consulted;
  absent metrics now come back `None` with `missing_inputs` named, and the
  fraud network is quarantined until `load_fraud_weights()`.
- Hurricane: an absent central pressure was served as a measured 1013 hPa /
  0 kt estimate; the wind-pressure relation is the cited Atkinson & Holliday
  (1977) form (was labelled "Dvorak" with a drifted 1013 reference).
- Wildfire spread needs a measured wind speed and never fabricates a due-north
  direction; flood soil products need the moisture and soil class, and an
  unknown soil class is refused instead of silently reading as loam.
- Fusion: 4 of the quantum model's 16 feature columns were constants (the full
  pure-state entropy is identically 0; three repeated a config value) — they
  now carry measured per-row quantities (bipartite Schmidt entropy,
  participation ratio, phase coherence, amplitude entropy). Detectors whose
  fitted state came from a loaded checkpoint are refit by explicit new
  training data (a width-8 retrain kept the shipped width-16 spatial fit, so
  drift reported "incomparable" forever). `default_fusion.pt` regenerated on
  the fixed pipeline (it predated the look-ahead fix).
- Incidence fit: a forecast day whose calibration cannot reach the 95% target
  inside the 1 − 1e-9 nominal cap falls back to a one-sided interval that
  covers by construction instead of returning the capped nominal as achieved;
  the mean band's docstring carries its measured 86-92% coverage (the basic,
  log-basic, Beran-recycled and log-t variants measure 85.8-87.4%, so no cheap
  calibration is shipped).

**Docs.** The unbacked "30 detection engines" headline and the "~500 ms / ~50 ms
RTX 4090" latency tables are replaced by the measured counts (89 detector
classes, CI-gated; 94 manifest entries; 5 default) and the committed
fusion-latency measurements; no GPU figure is claimed because none is
committed. The NAB harness pins its corpus to a commit and stamps run
provenance.


### GDACS tropical cyclones, GDACS's own scores as the test, and faster, better-calibrated detection

An adversarial review of the first version reproduced 6 should-fix defects and
10 nits; all are fixed, with tests, before this push.

**Added — GDACS tropical-cyclone levels (`crisis_monitoring.gdacs`, `CrisisMonitor`).**
The published GDACS cyclone page gives only a sparse risk matrix (no Category 5,
no Category 4 at Medium or High vulnerability). The complete table is Table 1 of
GDACS's MHEWS user manual (JRC141661, 2025; version of 23/09/2026; section 4.7),
and it is implemented as printed: each track point's level from its
Saffir-Simpson class, the people in its 34 kt and 64 kt wind buffers and its
country's vulnerability, with the "100k or 10% of the country" thresholds; the
storm takes its highest point; a Red more than three days ahead becomes Orange;
scores 0.5, 1.5 and 2.5. Storm-surge levels (Red >= 3 m, Orange >= 1 m) are
reported beside the wind level, outside the score, as GDACS does. Country
classes are GDACS's published 2012 list with the two classes its live feed now
reports differently (Tanzania Medium, Yemen High); the other 49 countries seen
in that feed for 2023-2026 storms match. On the 152 track points of five real
storms (MELISSA-25 twice, CHAMPI-26, DITWAH-25, TAPAH-25, CHOI-WAN-26) the table
reproduces GDACS's level at every point, including the one Red GDACS lowered to
Orange for being more than three days ahead (MELISSA-25 over Jamaica).
`CrisisMonitor` takes `cyclone_points` (or one point on the report) and
`max_storm_surge_m`; a point with exposure but no country is refused rather than
assumed Low. The Saffir-Simpson ladder is checked knot by knot against the
hurricane detector's. GDACS sets volcano, drought and forest-fire levels by
expert judgement, with no published formula, so those are not implemented.

**Checked against GDACS's own scores (earthquakes).** The module's earthquake
docstring said its scores "have not been checked against GDACS's own event
scores". 26 real events (2023-2026, recorded from GDACS's API under
`tests/fixtures/gdacs`) now check it: given GDACS's own scaled population and
coping factor, `shakemap_score` reproduces GDACS's score for all 21 scored
single-country events (to 7e-15), and the 2017 table's intercept and slope are
GDACS's live ones for each of their countries. Found along the way: GDACS's live
coping factors use newer INFORM values with the published normalisation (China
0.821 against the 2017 table's 0.833), which confirms leaving the 2017 column
out; cross-border events average the countries' coefficients and sum their
coping factors (3 events, factors up to 2.36, outside the published 0.5-1.5); and
GDACS's live scaled population is not the published band formula on its own
band totals (-0.27 to +0.17 in log10; the same level for 19 of 21). New
`shakemap_score` and the `shakemap_scaled_population` input take GDACS's value.

**Faster — epidemic fit with exact gradients.** `fit_seir_incidence` now
integrates the forward sensitivity equations with the SEIR system, so each
likelihood evaluation also gives its exact gradient (matching central
differences to 1e-4); the negative-binomial log-likelihood is written out in
closed form. On three simulated outbreaks the estimates and intervals match the
finite-difference version to 8e-6 (relative) and the prediction bounds to one
count, about 3.3x faster: 6.8-7.2 s to 2.1 s per fit at B = 200 on one core. The
amplitude's upper bound is now also capped so that every point of the search box
starts with S0 > 0: L-BFGS-B's first step goes to the box's corner, which could
be an invalid state, and the penalty there stopped some bootstrap refits at
their starting point.

**Added — a training-conditional false-alarm bound.** The split-conformal
threshold bounds the false-alarm rate only on average over calibrations: with
300 uniform scores and a 1% target, 43% of single calibrations exceeded 1%.
`detection_tier.pac_conformal_threshold` (Vovk 2012) picks the order statistic
whose own rate exceeds the target with probability at most delta (measured 5.0%
at delta = 10%); `pac_minimum_calibration_size` gives the fewest scores it needs
(230 for a 1% target at delta = 10%), and fewer raise rather than return a
weaker threshold. `PathogenDetector` takes it as `false_alarm_confidence`: over
60 baselines the share above a 1% rate fell from 27% to 2%, at the cost of a
lower average rate (0.83% to 0.28%), so less sensitivity.

**Faster — importing the humanitarian modules.** `infrastructure/__init__` now
loads `SpaceExplorationAnalyzer` and the streaming names on first use, through
module `__getattr__` with the names kept visible to type checkers; importing
`gdacs` or `CrisisMonitor` fell from 2.37 s to 0.20 s. Every package name still
resolves.

**Fixed — a reproducible quantum kernel.** The kernel path drew its shots from
a fresh, unseeded generator per kernel entry, so `QuantumAnomalyDetector`'s
scores changed run to run (AUROC 0.85-0.96 over 40 runs, as the PR noted).
`QuantumAnomalyDetector(seed=...)` now seeds the kernel's shots and the QAOA
initial parameters (VQE keeps its seed 0 unless given one): the same seed gives
identical scores, and seeds 0-19 span 0.862-0.938. `QuantumKernel` and
`QuantumFeatureMap` also take `shots=None` for the exact fidelity on the
built-in statevector simulator, with no shot noise.

**Fixed — radio-blackout and radiation-storm levels on NOAA's scales.** Three
hand-written ladders now go through `evaluation.hazard_metrics.noaa_r_scale` /
`noaa_s_scale`, the branch's single definitions. The SWPC X-ray level read M1 as
Moderate (it is R1, Minor), C flares as Minor (they are below R1), exactly X1 as
Moderate (R3) and X10/X20 as Strong (R4 Severe, R5 Extreme); DONKI flare classes
read X10 as Extreme and X5 as Severe (R4 and R3). The SWPC proton ladder already
matched S1-S5 and now uses the same function. `SolarStormPredictionResult` gains
`radio_blackout_level` (R 0-5 from the 0.1-0.8 nm flux) and
`radiation_storm_level` (S 0-5 from a supplied `proton_flux_pfu`); with protons
supplied, `radiation_storm` follows S1 instead of the flare class, so an M flare
without a proton event is no longer a radiation storm. `radio_blackout` keeps its
released meaning, now stated as R3 or stronger. The test that pinned the old
DONKI ladder pins NOAA's.

**Measured, not adopted.** For the sequential pathogen test, an average of Simple
Jumpers over jump rates 0.01, 0.1 and 1 found a 0.5 sd shift 29% sooner (median
264 against 374 snapshots, 80 runs) but 0.75-1 sd shifts about 8% later, at the
same false-alarm guarantee; the single rate stays the default.

### Public-safety upgrades: sequential pathogen detection, fitted epidemic forecasts, GDACS alert levels

Three capabilities on top of the rebuilt components below, each from a published
method and measured on simulated data with a known truth. An adversarial review
of the first version reproduced 8 should-fix defects and 6 nits; all are fixed,
with tests, before this push.

**Added — `core.conformal_martingale.ConformalChangeDetector` and
`PathogenDetector.monitor`.** A snapshot test at a 1% false-alarm rate cannot see
a small shift that persists. `monitor()` scores each snapshot as before and also
feeds its energy to a conformal Shiryaev-Roberts test: smoothed conformal
p-values against a bag seeded with the held-out baseline energies, the one-sided
Simple Jumper betting martingale (Vovk et al. 2021), and the statistic
R_n = Σ S_n/S_k, whose alarm R_n ≥ c restarts the test. Smoothed p-values are
independent and uniform under exchangeability, ties included, so R_n - n is a
martingale and alarms on a baseline-like stream average at least
`mean_snapshots_to_false_alarm` (default 1000) snapshots apart, whatever the
energy distribution: measured at 1.03-1.08 c for normal, Pareto and heavily tied
scores. (The review showed that unsmoothed p-values break this proof when scores
tie.) `seed` seeds the smoothing draws. On 12 correlated channels a shift of one
channel by 1 sd left the snapshot alarm rate at the null 1.0% and was caught,
counting from a restart at the onset, in a median of 71.5 snapshots (20 of 20
runs; 1.5 sd: 33.5, 2 sd: 24.5, 3 sd: 17). A fall in energy alarms no more often
than the null. The null is exchangeability, so drift or autocorrelation in a
sensor also raises the statistic; after a confirmed event the baseline should be
refitted. `BioThreatResult` gains `sequential_alarm` and `change_statistic`
(log R_n); `detect_pathogen` is unchanged.

**Added — `EpidemicForecaster.fit_to_incidence` and
`forecasting.fit_seir_incidence`.** Fits β, the initial size and the
negative-binomial dispersion k to observed daily incidence by maximum
likelihood: the count on day j is NB with the model's E→I transitions that day
as its mean, never cumulative counts (Chowell, Bleichrodt & Luo 2024; King et al.
2015), and the initial state lies on the growth-phase eigenvector at S = N - R,
so prior immunity (`initial_recovered`) enters as β(N - R)/N (the review found
the first version ignored it: R0 3.047 with an interval above the true 3 at
R = N/2; now 2.991). A parametric bootstrap (default B = 200) gives intervals for
R0, the expected incidence and future counts. Over 120 simulated outbreaks
(R0 = 3, k = 20, 40 observed days, 14 forecast) β was unbiased (mean 0.3000, sd
0.012); the 95% R0 interval covered the truth 111 times (92.5%, exact CI
86-97%) and the band of the mean covered 91.8% of forecast days (CI 87-96%). In
the first version, prediction intervals drawn from the refits covered 91.1% of
future counts (cluster-bootstrap CI 89.1-92.9%): k̂ runs high in small samples
(median 23.5), which narrows the count noise. The prediction interval is now
the plug-in interval calibrated per day by Beran's (1990) bootstrap method; on
the same 120 outbreaks it covered 95.9% (CI 94.6-97.2%), at levels averaging
0.988, and `prediction_level` reports the level each day needed. A
population no larger than the observed cases plus the immune, non-finite rates
and non-integer settings raise `ValueError`. A fit takes about 8 s at B = 200 on
one core (6-10 s over 30 fits).

**Added — `crisis_monitoring.gdacs` and GDACS levels in `CrisisMonitor`.** The
GDACS earthquake shakemap and "EQ parameters" scores, the INFORM Lack of Coping
Capacity factor, the Red/Orange/Green levels and the flood levels, implemented
as published at gdacs.org. The coping-capacity map 0.5 + (LCC - 0.9)/8.4 matches
all 191 rows of GDACS's 2017 coefficient table exactly, and the 38 countries with
their own calibration are carried as `COUNTRY_COEFFICIENTS` (© European Union,
CC BY 4.0); Japan's shakemap vulnerability of -1.0, for example, takes an
exposure scored Orange by the global model to Green. GDACS's floor ("a score
above 2 is not lowered below 1") cannot bind with factors in the published
0.5-1.5 range and is not applied. When a GEOINT report carries a model's inputs
(`population_by_mmi`, or `magnitude`, `depth_km` and `population_within_km`;
`deaths`/`displaced` for floods; `inform_lcc`; `country_iso3`), the alert carries
`gdacs_level` and `gdacs_score`; the hazard follows the inputs when not stated,
and makes the type `natural_disaster`. Orange or Red counts as a detected
crisis and raises severity to at least `high` or to `critical` (this module's
reading of GDACS's "national capacity may need support" and "international
intervention likely"), so a Red alert gets the national-response steps.
Incomplete or invalid inputs, including numbers too large for a float, are
named in the response instead of guessed or raised. The scores have not been
checked against GDACS's own event scores.

### Four public-safety components rebuilt on published methods

Four modules inherited from Omni-AXA (2025-10-15) were still exported with
their original logic: a terrorism detector that ignored its indicators, a "QBM"
pathogen detector driven by a random matrix, a crisis monitor that invented
population figures, and an SEIR forecaster with a wrong R0. Each now rests on a
published method; every test the new files mark as such fails on the previous
tree. An adversarial review of the first rebuild (`499c08af`) found 2 blocking
and 5 should-fix defects; all are fixed below, each with a test that fails on
that commit.

**Fixed — `medical.pandemic.forecasting.EpidemicForecaster`.** R0 was
β/(σ+γ), 1.67 for the default rates whose true R0 is 5 (the simulated final
size implies 5.0001); at β = 0.25 it reported 0.83, "dies out", beside an
outbreak that infected 89%. R0 is now β/γ, the next-generation-matrix value
(van den Driessche & Watmough 2002). `chaos_score` is now Λ, the dominant
eigenvalue of the infection subsystem at the disease-free state: the early
growth rate per day (0.170 against an observed 0.169) and positive exactly above
the epidemic threshold, so `bifurcation_detected` no longer depends on
population size or fires below threshold. `chaos_threshold` is now in Λ's
units (per day, default 0.0, the threshold); a value set for the released
unitless score (its default was 1.28) never fires and should be removed.
Deaths, hospitalisations and cost scale with cumulative infections (993,023 by
default) instead of peak prevalence (305,010), with the ratios configurable and
reported under `assumptions`. The grid is daily (`forecast_days + 1` points),
so `peak_day` is a day, and "rapid intervention" is recommended only above the
threshold. New `estimate_r0_from_incidence` fits the growth rate by Poisson
regression and converts it with R = (1 + r/σ)(1 + r/γ) (Wallinga & Lipsitch
2007, eq. 3.2); it recovers 5.0 within 5% from simulated counts. Non-numeric,
infinite or out-of-range settings and initial counts raise `ValueError`.
`simulate_intervention_effect` validates `effectiveness`, restores β even on
error, and its 0.5 floor no longer raises an R0 that was already below it.

**Fixed — `medical.pandemic.bio_threats.PathogenDetector`.** The released
detector binarised readings at their own median and scored them with an
unseeded random coupling matrix: about one fresh detector in six flagged the
same input, and x, 1000x and x + 500 gave the same energy. It now needs
`fit(baseline)` (snapshots x channels) and scores E = ½(x-μ)ᵀΛ(x-μ), the energy
of a Gaussian Markov random field with couplings `J_matrix = -Λ` off the
diagonal. The covariance is shrunk toward its diagonal and each channel's
variance gets the same relative ridge, so the fit commutes with a change of
units: a fluorescent fraction spiked 20 sd high is flagged whether it is
reported as a fraction or in percent (the first rebuild scaled the ridge to the
mean variance and missed it beside a channel of particle counts). A channel with
no variance is named and refused. The threshold is the split-conformal quantile
of held-out baseline energies (`false_alarm_rate`, default 0.01): the guarantee
holds on average over the calibration draw (300 recalibrations averaged
0.0100), while one baseline's realised rate varies (0.45% on the fixed test
baseline; up to about 3% across 60 baselines). `confidence` is one minus the
conformal p-value. Refitting recalibrates. A fixed `energy_threshold` is now an
energy (>= 0, alarm when exceeded); the released -0.5 meant the opposite
comparison and is refused. Shifts, rescalings, single-channel spikes and
correlation changes are flagged; signatures name the channels beyond 3 sd. The
type comes from a MASINT report naming a known agent, otherwise
`unclassified_anomaly`, and every alarm starts with laboratory confirmation, as
in the BioWatch process. An unfitted detector, a wrong-length snapshot or a
baseline too short for the rate raises.

**Fixed — `infrastructure.humanitarian.crisis_monitoring.CrisisMonitor`.** A
lone score of 0.8 reported 80,000 affected (a default density of 10,000 x
score x 10). `affected_population` is now the exposure the GEOINT report
supplies (`affected_population`, or `population_density` x `affected_area_km2`,
the keys this module's tests already passed) and `None` without it, with a
response step to obtain it. OSINT indicators now type the crisis, matching is per
indicator on word boundaries ("outbreak of violence" is not a pandemic),
`mass_casualty` is reachable, and non-finite, boolean or out-of-range scores are
clipped or ignored. `CrisisAlert.affected_population` is `int | None`.

**Fixed — `security.anti_terrorism.TerrorismPatternDetector`.** The stage was
`max(threat_score)` bucketed, with the indicators unread: "school bake sale" at
0.95 was `imminent_action`. Indicators are now the 42 observable behaviours of
*US Violent Extremist Mobilization Indicators* (NCTC, FBI and DHS, 2025), as
codes or booklet numbers, and the stage is the most advanced one reported:
mobilization is `imminent_action`, preparation `action_planning`, intent-type
motivation `indoctrination`, other motivation `identification`. The score is
the source's confidence, combined by noisy-OR across reports, and never sets
the stage. `threat_detected` needs a preparation or mobilization indicator, or a
direct threat (#28, #30, #31), and a stated source confidence: the booklet says
the other motivation indicators "should not serve as the sole basis for any
investigative activity", so acceptance of violence (#34) alone is staged but
not alarmed, and an unscored report asks for corroboration instead of urgent
action. `threat_indicators` lists only recognised codes; other strings go to the
new `unrecognised_indicators`, and those naming a protected characteristic's
category (race, age, religion, ...) to `excluded_indicators`, following the
booklet's caveat that action should never rest solely on such characteristics.
That screen matches category words, not every identity term, which is why no
unrecognised string is ever reported as a threat indicator. Every action begins
with trained human review. Free-text reports must be coded first.

**Documented — three released-module removals (`DEPRECATION.md` §6.13-6.15).**
This release removes released public classes from three modules without
restoring them: `security.counterintelligence.OverwatchNexus` /
`OverwatchNexusResult` (scored reports it fabricated: the same risk, 0.2766,
for zeros, noise and x1000 noise), the `truth_decipher.TruthDecipherFramework`
API and `TruthDecipherResult` (fixed "true" ethics inputs; every resolution
stored as a success; the name stays as an alias of `StatementVerifier`), and
`security.quantum_risk_cyber`'s `QuantumRiskCyber`, `PostQuantumMigrationPlanner`,
`QuantumThreat`, `ThreatLevel` and `CryptoSystem` (ML-DSA-65 and SLH-DSA flagged
as quantum-vulnerable; SIKE and Rainbow counted as post-quantum). Each defect
was reproduced on `main`; each entry names its replacement and a test that
keeps the old surface from returning.

### Performance: the fusion path, start-up, and drift that per-feature tests miss

Found by running the shipped system against a native AMA v5.0.0 build and
profiling it. Both columns come from `benchmarks/fusion_latency.py` run on the
same 4-CPU machine, once from the base commit's tree and once from this one
(medians of three; run-to-run variation on this shared machine was up to about
a third, so read the ratios, not the last digit; `benchmarks/results/fusion_latency.json`
holds this tree's 8-feature run):

| Measurement | Before | After |
|---|---|---|
| `import omni_mercury_engine.engine` (shap installed) | 8.4 s | 3.0 s |
| `OmniMercuryEngine(auto_load_checkpoint=True)` | 2.4 s | 0.24 s |
| `detect_with_fusion`, 64 rows x 8 | 157 ms | 38 ms |
| `detect_with_fusion`, 1,024 rows x 8 | 2.13 s | 0.18 s |
| `detect_with_fusion`, 4,096 rows x 8 | 7.89 s | 0.40 s |
| Domain-model feature stage, 4,096 rows x 8 | 7.53 s | 0.19 s |
| `detect_with_fusion`, 512 rows x 128 (`--features 128 --sizes 512`) | 2.62 s | 0.19 s |
| of which `quantum` / `multiverse` / `neural` / `biometric`, same run | 1,706 / 267 / 670 / 54 ms | 8 / 6 / 20 / 4 ms |

**Fixed — a row's score depended on the rows after it (`engine.py`).** The
engine cleared streaming state (the neural model's hippocampal buffer, the
directive detector's recursive memory) before `extract_features` but not before
the `predict`/`detect` pass that follows it, so the second pass scored the first
rows of a batch against the last rows of the same batch. With only the final 100
of 300 rows replaced, 99 of the first 100 neural scores moved; they now do not.
Training-time consensus scores take the same seam and are clean too.

**Fixed — a single spike row discarded the neural model for the whole batch.**
A spike has a flat spectrum, and `np.histogram` raises "Too many bins for data
range" on it (float32 spikes of amplitude 1, 3000, 5000 and 1e5 all did); the
engine then dropped the neural feature group. Such a row now falls in the middle
bin, where a constant row already fell. On the shipped checkpoint no verdict
moved, so this cost the neural evidence, not the decisions.

**Performance — per-row Python loops replaced by batched array code.**
`models/neural.py` (27x), `multiverse.py`, `neurosymbolic.py`, `consciousness.py`,
`astrophysical.py`, `biometric.py` (harmonic features), `quantum.py` and
`detectors/temporal.py` (sliding-window trend scores). Each is tested against the
original implementation, kept in the test as an oracle: float64 and integer
inputs match to rounding, float32 to single precision, and mutating the new code
(window off by one, dropped ULP correction, unstable tie sort, wrong threshold)
is caught. `quantum` previously eigendecomposed a d x d density matrix per row
(O(d^3)); a pure state has one non-zero eigenvalue, `<psi|psi>`, so it is now a
dot product. The old value carried single-precision LAPACK noise (up to 2e-6) on
float32 input; the new one is exact. `SymbolicReasoningLayer.reason_many`
evaluates the rules once per call, since they see only the context.

**Performance — start-up.** PyTorch Lightning (with torchmetrics and matplotlib),
sympy, shap and DeepFace (with TensorFlow, 1.8 s of the 2.0 s it took to construct
an engine) load on first use instead of at import. Every public name still
resolves, and the module-level `DeepFace`, `SHAP_AVAILABLE`, `LIME_AVAILABLE`,
`SYMPY_AVAILABLE` and `ml.HAS_PYTORCH_LIGHTNING` keep working; the triton
preload still runs immediately before the TensorFlow import.
`models.biometric.DEEPFACE_AVAILABLE` is now resolved when it is read, and
`QuantumConsciousnessModel._measure_entanglement` still answers for one state;
nothing public was removed.

**Added — principal-axis drift test (`ml/drift.py`).** Per-feature KS and PSI
dilute a shift along a low-variance direction of correlated features, and the
covariance test centres each window on its own mean. `PrincipalAxisDriftDetector`
runs a KS test on every axis of the reference's PCA basis (Bonferroni, 0.01
family-wise; the PCA variant of the aggregated univariate tests that Rabanser
et al., arXiv:1810.11953, section 5, found comparable to multivariate tests).
Axes come from the even reference rows and the distribution from the odd rows:
fitting both on the same rows gave 6% / 25% / 100% false alarms (64 features
with 500 reference rows; 128 with 1,000; 128 with 400) against 0% / 2% / 0%
split. `benchmarks/drift_power.py` (100 windows per cell over 5 independent
references) shifts 8 features correlated at 0.9 by one SD along the weakest
axis (0.11 SD in the most-moved single feature): the ensemble flagged 1% / 8% /
0% of 64 / 200 / 1,000-row windows before and 100% / 100% / 100% now, with
false alarms with no drift moving from 0-2% to 1-5%. With 128 features and
1,000 reference rows, 0% became 57% at one SD and 2% became 100% at two. With
independent features it adds nothing (the per-feature tests already see the
shift), and with 128 features at correlation 0.95 and only 400 reference rows
it flags nothing beyond the other tests. It is on in `EnsembleDriftDetector` by
default (`principal_axes=False` removes it, and `covariance=False` alone no
longer gives a marginal-only ensemble). Its statistic is computed against a
reference sorted once, equal to `scipy.stats.ks_2samp`'s to 1e-16.

**Fixed — operator-facing.**
- `mercury-agent detect -i data.csv` crashed with `could not convert string to
  float: 'a'` on any CSV with a header row; the data, label and hazard loaders
  now skip a header (a numeric first row is never dropped).
- `mercury-agent detect -o out.json` (and every other CLI command that writes
  JSON) serialised with `default=str`, which renders an ndarray as numpy's
  abbreviated repr: the default statistical detector's per-row `scores`,
  `z_scores` and flags came out as strings like `"[0.12 0.13 ... 0.26]"`, six
  values per array. Arrays, numpy scalars and tensors now serialise as lists.
- `scripts/mercury_session_setup.sh` installed `setuptools>=83 / wheel>=0.47 /
  cmake>=4.4.0`, below the floors AMA v5.0.0's setup rejects (84 / 0.48 /
  4.4.4), so a fresh container failed at its first AMA build. A test now keeps
  it, `build_ama_native.sh` and the CI action in lockstep.

**Fixed — repository configuration.** Each item was checked against the tool it
configures, not read.
- `.gitleaks.toml` held an allowlist and no `[extend] useDefault = true`, so the
  CI Gitleaks step loaded zero rules: a file with a GitHub and a Slack token
  gave "leaks found: 1" under the default config and "no leaks found" under the
  repo's. It now extends the defaults. The tree, the PR's commit range and all
  1,045 commits of history scan clean with two exact test sentinels and one history-only
  type-annotation false positive allowlisted.
- `.secrets.baseline` listed no plugins, so the pre-commit `detect-secrets` hook
  printed "No plugins to scan with!" and exited 0 on that same file. Regenerated
  with detect-secrets 1.4.0 (the hook's version): 22 plugins, 183 findings
  reviewed line by line and recorded as false positives (placeholders, env-var
  names, pinned digests, public KAT vectors); digest and NIST-vector JSON is
  excluded by path. All tracked files pass and the canary file fails.
- The Safety policy is one working file. `.safety-policy.yml` was a 3.0-format
  file no gate read (`safety check` aborts on 3.0), carrying eight bookworm-era
  acceptances whose review had lapsed on 2026-08-02; `.safety-policy-v2.yml`
  existed only to stop `check` auto-discovering it. They are replaced by one
  2.0-format `.safety-policy.yml` that both gates pass with `--policy-file`:
  every severity reported, unscored findings and unpinned requirements
  included, nothing accepted, and an exception only with a reason and an expiry
  within 90 days (`tests/security/test_safety_policy.py`). Checked with Safety
  3.8.1: a vulnerable pin fails (exit 64); an exception with a future expiry
  removes exactly that advisory and an expired one stops applying; a misspelled
  key fails the gate (exit 2); an unpinned requirement is scanned (9 findings,
  0 if unpinned were ignored); a clean pin passes.
- `/health` stays minimal (`status`, `version`). An earlier commit in this
  round filled its never-populated `uptime_seconds`, but uptime on an
  unauthenticated endpoint tells any caller when the service last restarted and
  is already exported as `omni_mercury_uptime_seconds` on `/metrics`, so that
  change is reverted. The platform's Caddyfile now answers 404 for `/metrics`
  instead of proxying it: Prometheus scrapes the API over the compose network,
  and its security model says `/metrics` must not face the internet (checked
  with Caddy 2.11.7: `/metrics`, `/metrics/` and `/metrics/x` return 404;
  `/health` and `/metricsx` proxy).
- Compose: every published port is bound to `127.0.0.1` (the platform overlay
  had published the API, an unauthenticated Prometheus and a Grafana on
  admin/admin on all interfaces, around the TLS edge and the trusted
  `X-Forwarded-For` hop), the overlay requires `GRAFANA_ADMIN_PASSWORD`, and the
  stale pins moved: Caddy 2.8.4 to 2.11.7, Prometheus 2.51.2 to 3.13.4 (the LTS
  line, supported to 2027-07-31; 3.15 is a regular minor whose fixes stop about
  six weeks after release), Grafana 10.4.2 to 13.2.4 (a supported minor). Each
  was run against the repo's own config (Caddyfile validates; 33 rules load and
  reload; the provisioned datasource and all 16 dashboard panels load; Prometheus
  3 scrapes the live API's `/metrics`). Prometheus 3 can only be downgraded to
  2.55 and Grafana migrates its database forward-only, so the data volumes are
  now named for the image's major version (`prometheus-data-v3`,
  `grafana-data-v13`): a major bump starts on a new volume and the old one stays
  intact. Tests fail when a volume name stops matching its image's major or
  Prometheus leaves 3.13, and Dependabot (which gains a `docker-compose`
  ecosystem, since its `docker` entry reads only the Dockerfile) takes
  Prometheus patches and Grafana minors but not majors.
- `.dockerignore` re-includes the two benchmark artefacts `self_model.py` reads;
  without them the in-image self-description reported no measured limitations.
- `.coveragerc` and `pyproject.toml` comments claimed `fail_under = 85` and an
  "enforced" pyproject floor; coverage.py reads `.coveragerc` alone
  (`coverage debug config`), so the comments now say so.
- `.mcp.json` removed. `uvx kafka-mcp-server` could not start (its unbounded
  `mcp>=1.6.0` resolves to mcp 2.x, which dropped `mcp.server.fastmcp`) and has
  no SASL support, so it could not have authenticated to GCN even running, while
  receiving the client secret from an unpinned package. `docs/GCN_KAFKA_MCP.md`
  says what a working server needs; the keep-alive is unchanged.
- `.flake8` needed no change: it reports nothing over CI's scope.
- `test_single_image_path_input_finally_works` ran real DeepFace, which fetches
  about 1.1 GB of weights on first use (and, with OpenCV 5 wheels that no longer
  ship Haar cascades, the detector from GitHub), so it failed wherever that was
  blocked. With the cascades supplied it passes against the real library (84 s),
  so the setup is sound; the unit test now stubs DeepFace and asserts that a
  pixel array, never the path, reaches it, and the real-library run needs
  `[face]` and `MERCURY_NETWORK_TESTS=1`. No CI lane installs `[face]` (the old
  test passed there only because DeepFace was absent and the harmonic fallback
  ran), so that run is local until a lane adds the extra. `docs/INSTALLATION.md`
  lists the assets, where they go and a checksum-verified way to pre-seed them.

269 tests added; 16,717 are collected on this tree with the full optional-dependency
surface. The full local run gave 16,610 passed, 103 skipped, 0 failed (the 103rd
skip is the real-DeepFace test, which needs `[face]` and `MERCURY_NETWORK_TESTS=1`);
the four Safety-policy tests added after it pass.

### Seven parallel reviews of the newest code, every finding reproduced before it was fixed

Seven independent reviews covered the roughly 20,000 lines added after the
previous pass (engine concurrency and drift, the fusion decision path and
counterintelligence screening, the adjudication ledger, statement verification
and the CIT, contactless vital signs, the API audit trail, and wiring). Every
finding below was reproduced from the code or by experiment first, and each fix
comes with tests that fail on the previous code.

**Fixed — intelligence fusion decided from unbounded or unscreened inputs
(`security/intelligence_fusion.py`, `security/counterintelligence.py`).**
- One report with `threat_score: 50` forced CRITICAL over every other report,
  and `-1e6` forced LOW: per-report scores were never bounded. Scores and
  confidences outside [0, 1] are now excluded and recorded in
  `evidence_notes`; empty reports no longer count as corroborating sources
  (a missing confidence was defaulted to 0.5).
- Once trained weights were admitted, screening no longer reached the
  decision, and a NaN feature returned LOW. The network now sees only the
  screened reports, defers to evidence scoring when screening or source grades
  move a discipline away from the reliability it was trained with, and
  non-finite values never decide.
- A hostile report could take over a genuine one by declaring its default
  `report_id`, or by spoofing its `source_id` with an earlier time. Default ids
  are reserved, cluster representatives are chosen by origin and reliability
  before time, and conflicting scores in one cluster raise `CONFLICTING_CLAIMS`.
- Outcome weights are bounded (`MAX_OUTCOME_WEIGHT = 10`, a chosen cap); a
  1e308 weight drove adjusted reliability to 1.0 or NaN.
- Merit gate: exact-duplicate held-out cases count once (five copies of one
  win used to pass at p = 1/32); attempts are recorded with a held-out
  fingerprint, and the k-th retry against the same set must beat
  `alpha * 2^-(k-1)`, which bounds false admission over unlimited retries at
  `2 * alpha`. The count is per engine instance. List-valued disciplines raise
  `InputModalityError`; only a missing torch maps the lazy export to `None`.

**Fixed — adjudication ledger provenance and durability
(`security/adjudication.py`).** A training export loaded silently as held-out
data; edited exports and records cut from the end of the chain went
undetected; the same payload could sit in both splits; "Alice" and "alice"
were two raters; concurrent writers forked the chain; a crash mid-write locked
the ledger for good. `load_export` now checks the split and, given the ledger,
re-derives every case at the export's head; `verify(expected_head_digest=)`
catches truncation; cross-split duplicate payloads are refused; people's ids
are normalised; writers take a `flock` and append in one write;
`repair_torn_tail` (CLI `repair`) drops only an unterminated crash fragment;
exports are atomic and apply the merit gate's distinct-case minimum.

**Fixed — statement verification and the CIT (`truth_decipher.py`).** Two
one-second sightings inside an eight-hour claim corroborated it (posterior
0.988): support now requires evidence covering the whole claimed interval
(overlap still refutes), and untimed evidence does not bear on timed claims. A
prior already past a threshold can no longer carry a claim across it on its
own; thresholds compare in log-odds with a 1e-12 tolerance, so a Bayes factor
of 19 reaches 0.95. The CIT overlap guard now includes the next item's
pre-onset heart-rate baseline (13 s with the defaults; at 10 s an irrelevant
item inherited the probe's deceleration). `ConsentRecord.withdrawn_at` blocks
all further scoring. `benchmarks/cit_calibration.py` replaces two unbacked
calibration figures with measured ones (see the restoration round below).

**Fixed — engine drift and detector concurrency (`engine.py`, `ml/drift.py`,
`core/parallel_detectors.py`, `core/detector_registry.py`).** Drift was
permanently "incomparable" for any model with a constant feature or more than
64 features (the batch was checked against the projected reference). On the
served route an auto-fitted detector made the first inference batch its own
drift reference. A drift error failed the whole detection request. With fewer
than two parallel-safe tasks the registry timeout was not enforced, and a hung
detector counted as a success, stranded a thread per call and saturated the
executor until every healthy detector failed. All fixed; explicit baselines
survive fits; failed derivations are cached; drift series are pre-created so
the first detection after a restart alerts. `engine.detect()` now records its
inference auto-fits too, and any training fit replaces an inference auto-fit
(it used to keep it). New: `EngineConfig.detector_timeout_s` (default 30 s, the
registry's budget; `None` disables it) and `MERCURY_FUSION_CHECKPOINT`.

**Fixed — API audit and export (`api/`, `security/secure_audit_logging.py`).**
A DETECT-only key could export audit logs with IPs and user agents:
`/export/*` now requires the `export` permission. Anonymous 4xx traffic could
rotate login evidence out of the hash-chained log: it is counted, not written.
Session actions were audited as "anonymous". A crash-torn last line made the
log unverifiable for good; same-second rotations overwrote a segment; workers
resumed from and pruned each other's segments; `platform audit verify` failed
whenever the server ran. Alert egress followed redirects with the sink's API
key and queued deliveries without bound; `tracestate` was echoed unvalidated;
CSV exports passed spreadsheet formulas through. All fixed.

**Fixed — contactless vital signs reported breathing harmonics as heart rate
(`biometric/rf_vital_signs.py`).** At 30 breaths/min or faster the breathing
fundamental fell on the band edge, no harmonic was cancelled, and its second
harmonic was reported as the heart rate (breathing 30/min with a true 96 bpm
heart gave 60; 79 of 162 sweep windows were off by more than 5 bpm). The
fundamental is now searched over a wider band and used for cancellation; when
it cannot be resolved, or when a heart peak lies within
`cancellation_guard_bins` (1.25, measured) of a suppressed harmonic, the heart
rate is withheld as `HARMONIC_AMBIGUITY` (a heart at exactly 2x breathing had
been reported 1.7 bpm off). The noise gate now uses the effective number of
independent streams (participation ratio) instead of the raw count: noise
shared across subcarriers passed it 7-12% of the time against a 1% target.
Timestamped packets are interpolated at the packet rate and filtered down, so
an out-of-band interferer no longer aliases into the heart band. Consent
reuses `truth_decipher`'s `ConsentRecord` and `ConsentRequiredError`. The
cost is coverage, not accuracy: CSI respiration coverage drops 1-8 points,
and 10 s windows now never report a heart rate (they reported 17 of 111,
within 1.6 bpm).

**Fixed — monitoring that could not fire.** The Helm chart's API metrics port
targeted 9090, where nothing listens; its alert rules were a hand-kept copy
with eight alerts on metrics nothing emits and none of the drift alerts.
`scripts/render_prometheus_rules.py` now generates the Helm `PrometheusRule`
from the same CRD as compose and the raw manifests; the ServiceMonitor
attaches the `app`/`service`/`job` labels every rule selects; the ethical-gate
alerts moved into the CRD.

**Fixed — CI.** Two numpy-typing failures on the Python 3.11 lane (in the
RF module and its tests), and a phantom-token ethical-gate coverage report
whose tokens named symbols that do not exist (now guarded by a test).

**Changed — breaking for operators.**
- Default API keys (`read` + `detect`) get 403 on every `/export/*` route;
  compliance exports need a key with `export` or an admin principal.
- Unauthenticated 4xx no longer appear in `/export/audit-logs`.
- Statement verification no longer counts evidence covering part of a claimed
  interval as support.
- `load_export` takes a `split` (default `"heldout"`) and refuses the other.

### Restoration: rebuilt capabilities, two ethical-gate failures closed, drift on by default

Every removed symbol from `7ab59916`, `740da10a` and `f07b724e` was loaded
from history and run against live code. The removals were names, wrappers,
proven duplicates and measurably broken code; three real needs surfaced, and
in each the removed code did not meet them. This round rebuilds those, fixes
what the audit found, and wires capabilities that existed but never ran.

**Added — concurrent detector execution with a real deadline
(`core/parallel_detectors.py`).** It replaces the removed
`ParallelDetectorExecutor`, which discarded finished results on timeout and
then waited for the slow detector anyway. Finished results are kept,
unfinished tasks are reported as timed out, and the call returns at the
deadline. A hung detector retires the pool so it cannot starve later calls,
and abandoned threads are capped. Concurrency is opt-in per detector
(`parallel_safe`); the five defaults declare it after a bit-identical
concurrent-versus-sequential check. `EngineConfig.num_workers`, previously
ignored, sizes the pool. 22 tests; 24 mutants, all killed.

**Added — statement verification and a consent-gated Concealed Information
Test (`truth_decipher.py`).** It replaces a pipeline whose ethics check passed
every input and whose "resolution" always reported success. It provides
Bayesian corroboration over independent evidence clusters, contradiction
detection (Allen's interval algebra, impossible travel), consistency over
time, the SUE indicator and verifiability proportions, all with intervals and
none with a verdict about a person. The CIT detects recognition of event
details, not deception. It runs only with a consent record and abstains when
the details were public. Calibration is measured by
`benchmarks/cit_calibration.py` (results in
`benchmarks/results/cit_calibration.json`; six questions of five items, three
repetitions, 2000 simulated sessions, alpha 0.05): null rejection 3.4% for the
exact rank test and 4.2% for the permutation test, and power 47% / 66% at a
one-SD probe effect on a single channel. 221 tests.

**Added — counterintelligence screening of fused reports
(`security/counterintelligence.py`).** It replaces a wrapper that reported
SUBSTANTIAL threat on 40/40 inputs with constant casualty figures. It covers
source reliability (Beta posteriors from Admiralty grades), circular and
derivative reporting, coordinated-reporting scan statistics (a Mecke-equation
p-value bound, checked against Monte Carlo), digit-based fabrication screens
and impossible travel. Wired into `IntelligenceFusionEngine`'s evidence path:
relayed copies carry no weight and corroboration counts independent origins.
Outputs are unchanged on reports without those fields (pinned over 200
random report sets). 180 tests; 62 mutants, all killed.

**Added — adjudication ledger for held-out fusion labels
(`security/adjudication.py`).** `load_trained_weights` admits weights only on
held-out merit, but nothing produced labels. The ledger records cases and
outcomes separately, enforces outcome independence from the reports, keeps
ratings blind, gates on quadratic-weighted kappa, fixes splits by hash, and
stores a SHA3 hash chain with optional ML-DSA-65 signatures. 271 tests;
78 mutants, all killed.

**Added — consent-gated contactless respiration and heart rate
(`biometric/rf_vital_signs.py`).** Respiration and heart rate from Wi-Fi CSI
or radar, with no contact sensors, identification or imaging, and nothing
runs without consent. Every estimate reports a rate or abstains with a reason.
It is a library module with no API, CLI or MCP surface. Its minimum analysis
window (about 20 s of clean signal) is longer than the CIT's 3 s baseline and
10 s response windows, so it does not supply per-stimulus CIT channels. The
real-radar accuracy figures quoted during development (Schellenberger et al.,
Sci Data 2020) came from an out-of-tree validation script that is not in this
repository, so they are not reproducible from it. Real Wi-Fi CSI hardware is
not validated. 154 tests; 44 mutants, all killed.

**Changed — drift detection is on by default.** `enable_drift_detection` had
no caller, so served engines never tested for drift. When enabled, the first
inference batch became its own baseline. Now `EngineConfig.drift_detection`
(default `True`, `MERCURY_DRIFT_DETECTION=0` to disable) tests every
`detect_with_fusion` batch:
- Only the rows the model calls normal are tested.
- They are compared with the training rows the model also calls normal,
  derived from the fitted spatial detector's training rows after every fit or
  load. The shipped checkpoint needs no new field.
- Drift at MEDIUM or above defers the decision.

Filtering the two sides by the same rule matters: a radius-trimmed reference
flagged 12/12 in-distribution 200-row batches. With the same rule, the
measured false-defer rate on the shipped checkpoint is 1% for 50-row batches
and 0% for 200-row batches; +1.0 shifts are detected in 100% of batches.
Each check costs 25-35 ms. The shipped checkpoint was trained on a synthetic
mixture, so real data sent to it will usually defer until a model is fitted
on representative data; see `docs/runbooks/data-drift.md`.
New series: `omni_drift_checks_total`, `omni_drift_detected_total` and
`omni_drift_check_duration_seconds`. New alerts: `OmniMercuryDataDrift` and
`OmniMercuryDriftBaselineMissing`.

**Fixed — drift reporting.**
- The ensemble reported a sub-test's severity even when the ensemble said "no
  drift", and the decider deferred on severity alone. Both are fixed.
- Drift tests compared extra features against reference column 0; a width
  mismatch now raises `DriftIncomparableError`.

**Fixed — σ_Immutable fail-open above the trained support.** With all 24
ethical anchors set to 10.0 the gate scored 1.000 and the floor passed.
ETHICAL registrations outside `[0, 2.0]` are refused atomically, and the floor
rejects anchors above the ceiling (`check="sigma_immutable_ethical_ceiling"`).

**Fixed — one subagent op could refuse every later detection.** Pan_XXV
registered its payload into the process-wide GOSNN singleton's ETHICAL group,
so `{"temperature": 0.2}` became an ethical anchor below the floor. The op now
scores its payload without registering it.

**Changed — every scalar carries the `omni` prefix.** The two GOSNN defaults
without it were renamed in place, since σ depends on vector order, not names.
The baseline was re-harvested: its diff is the two name lists, and σ is
unchanged at 0.9999216795. The signed corpus has no names and did not change.
Old names still resolve, and `register_scalars` maps a legacy name to its
canonical scalar instead of adding a dimension. The neurosymbolic hub's
counters moved to the metric-only `omni_diag_` channel, where they no longer
drive σ towards 0. Detector-score scalars, physics scalars and the ethical
engines' omni-scalar dicts are prefixed too.

**Fixed — the GOSNN latency gate never measured anything.** It imported a
module that has never existed, and its fallback counted every repeated input
as a cache hit. It now times `get_enhanced_scalars`, the step every
detection runs, and claims no cache rate (GOSNN has none).

**Fixed — unmeasured claims.** `three_r` `get_dominance_proof` no longer
reports an improvement factor and an ethical scaling that the fusion does
not apply. The MITRE ATT&CK dataset is labelled with its Terms of Use, not
Apache-2.0.

**Removed — dead engine plumbing.** `_extract_features_parallel`,
`_get_executor`, `_executor`, `parallel_executor`, `optimization_config` and
`MAX_WORKERS` had no reader; the new executor supersedes them.

**CI.** `scripts/run_ci_gates.sh` now also runs the capability-inventory,
doc-lint, neural-coverage, weapons-gate-corpus and Prometheus-render drift
checks, which previously ran only in CI.

### Review: gates that measure instead of trust

**Added — `security/quantum_risk_cyber.py`, rebuilt as a quantum-risk engine.**
The module removed in `740da10a` matched algorithms by substring. It rated
ML-DSA as CRITICAL because "DSA" appeared in it, scored Kyber+Dilithium 0.0
because of mixed case, and used one fixed ten-year timeline.

The rebuild:
- **Classification.** Names, OIDs and CycloneDX 1.6 CBOM entries are matched
  by exact token against a registry:
  - FIPS 203/204/205, draft FIPS 206, the HQC selection and SP 800-208
    LMS/XMSS;
  - SP 800-57 strength bands;
  - pre-standard Kyber, Dilithium and SPHINCS+ names, which are flagged;
  - broken SIKE, Rainbow, SHA-1, 3DES and RC4.
- **Hybrids and composites.** These are as strong as their strongest part.
- **Composite names.** A hash in `ecdsa-with-SHA256` cannot upgrade the
  signature.
- **Deadlines.** NIST IR 8547 dates follow the strength band. CNSA 2.0
  deadlines and approvals apply per category.
- **Mosca's `x + y > z`.** It uses a caller-supplied CRQC point or curve, with
  no default, and treats confidentiality (harvest now, decrypt later)
  separately from authenticity.
- **CBOM checks.** Quantum-security levels claimed in a CBOM are checked
  against the standard.
- **Urgency order.** Assets are prioritised by a stated rule, not a score.

Mercury's own cryptography is audited by the same rules, and a test pins
each declared use to its source file. The audit found that the sovereign
emergency tier verifies Ed25519 (Shor-vulnerable). It also listed what
CNSA 2.0 would require Mercury to change.

There are 72 tests. A 17-mutant pass over the decision logic kills all 17
after one gap was closed: password-hashing KDFs are outside CNSA 2.0 scope.

**Fixed — `IntelligenceFusionEngine.load_trained_weights` admitted any
shape-matching state dict.** A freshly initialised network has matching
shapes, so random weights were enough to move threat decisions off
evidence scoring and onto noise. The method now takes labelled held-out
cases (`heldout`, at least `MIN_HELDOUT_CASES` = 30 pairs of
`(intel_reports, ThreatLevel value)`) and installs the candidate only when
it is significantly more accurate than evidence scoring: one-sided exact
McNemar on the discordant pairs, p < `MERIT_ALPHA` = 0.05. Higher accuracy
alone is not enough (4 wins, 0 losses gives p = 1/16 and is refused). On
refusal the engine keeps scoring from evidence, and the measured
`FusionMerit` is returned and kept on `engine.merit`. **Breaking:**
`heldout` is a new required argument; there are no production callers. The
tests pin the decision boundary at 4:0 refused, 5:0 admitted, 6:1 refused
and 6:0 admitted. A hand-run mutation pass over the gate (ten mutants:
significance replaced by accuracy, tail off-by-one, α, size and label
checks, non-strict shape loading, decision takeover) kills all ten.

**Fixed — stale benchmark figures across the README and docs.** The
committed run is 67 of 76 datasets (Mean AUC 0.8545, Median 0.9052, Mean
oracle F1 0.6425; ADBench 0.8515 / 0.6415). Ten prose claims in the README
and five docs still quoted the earlier 66-of-75 run at 0.8251. The README's
hand-typed "comparable domain" table was also wrong: it had the old ADBench
row, and it listed three threshold-labelled datasets (ThreatIntel,
NASA_Exoplanet, SolarDynamics) as independently labelled. That table is now
a generated block (`<!-- BENCHMARK-DOMAINS -->`) rendered from the results
JSON by `scripts/update_readme_benchmarks.py`, so `benchmark.yml` refreshes
it with the headline. `scripts/check_benchmark_integrity.py` now also fails
when that block differs from a fresh render, and when any README or `docs/`
prose states a dataset count other than the committed run's ("N of M
attempted", "N successful / M attempted", "N/M scored" and similar, across
line breaks and blockquotes). The README's "6 datasets below 0.50 AUC"
caveat is now 2. `docs/conf.py` declared the docs "Released under MIT"; the
project is GPL-3.0-or-later.

**Removed — four documents that were one-off logs or self-declared stale.**
`DOMAIN_PERFORMANCE.md` (an "older illustrative snapshot" whose role the
generated domain table now fills), `PR335_AI_ALERT_REMEDIATION.md` and
`WS_F_VERIFICATION.md` (single-PR verification logs; the latter cited AMA
3.3 against the v5.0.0 pin), and `REPOSITORY_REFINEMENT_AUDIT.md` (a branch
audit log with no references). Git history keeps all four.

**Restored — removed result fields and modules, computed from cited
methods.** Earlier commits removed released fields and modules
that were never computed or were wrong; each is back under the Preservation
Principle, and the new contract tests fail on the previous code.
- Power: GIC averages 1-second data to 1-minute means before dB/dt (the
  Gannon-storm FRD record read 765 nT/min at 1 s, ~55 at 1 minute) with
  tiers 90/300/480 nT/min, and a causal geoelectric field. EMP measures a
  sampled field (peak, 10–90 % rise, FWHM, 90 %-energy band, IEC 61000-2-13
  class; the DOE 2021 E1 benchmark measures 2.47 / 22.98 ns against
  2.5 / 23 ns), detects E2 after E1, locates a source by TDOA, and keeps
  `IntentionalEMIDetector`, trainable and not consulted until weights load.
  The threat level is at least what the grid impact warrants (a storm with
  critical grid impact read "benign"); grid impact is graded on its
  1-minute tiers only for a 1-minute rate (`dbdt_cadence_s`); a record that
  ends before the pulse falls reports no rise time; the released field
  order is a prefix again, and `intentional_attack_probability` and
  `gic_amplitude_a` keep their places with the value `None` (DEPRECATION.md
  §6.12). The USGS source returns the whole requested window (it kept the
  last 100 samples, 1.5 minutes of a 1-second fetch), and the solar → GIC
  cascade takes the GIC moderate tier (90 nT/min) and the NOAA G1 bound
  (Kp 5-) instead of 100 nT/min and Kp 5.0.
- Hazard fields: wildfire fuel moisture (NFDRS 1-hour, 1.03 × the EMC at
  the fuel-atmosphere interface, GTR INT-169 eq. 26 and Table 6; it needs
  the `nfdrs_state_of_weather` code, without which the spread model keeps
  its released 10 %), flood peak discharge and time to peak, hurricane
  intensification rate (≥ 30 kt / 24 h is rapid; Milton 70 → 150 kt in
  HURDAT2), earthquake IASPEI ML and Reasenberg–Jones aftershock
  probability, volcanic time to eruption by the failure forecast method
  (Poisson MLE, Bell et al. 2013; no forecast when the fitted failure time
  precedes the last sample); `None` where the inputs do not allow a value.
  `forecast_method` and `detection_method` follow the released fields, so
  positional construction keeps its order.
- Space: one NOAA G-scale definition in Kp thirds replaces five ladders that
  disagreed (5- read no storm, 9- read G5) in the SWPC and DONKI parsers,
  `SolarStormDetector`, the solar-storm evaluation buckets and the Schumann
  storm labeller; DONKI reads a storm at its peak Kp, not its onset (the
  May 2024 storm read G4, not G5); an SWPC fetch failure raises instead of
  reading as a quiet feed. Still on Kp ≥ 5.0 and due with the retrain of the shipped
  `solar_storm_geomag` checkpoint: that model's training labels, physics
  baseline and operating-point rule, and the Schumann training labels. The R
  and S scales exist in `evaluation/hazard_metrics` but the flare and proton
  ladders do not use them yet.
- `security/hive_firewall`: feedback reaches the nodes that voted (a
  reported false positive had lowered every threshold), once per decision,
  and workers hold distinct thresholds under trust-weighted supervisors and
  a queen majority. An allowlisted pattern is allowed without a vote; trust
  is floored above 0; signatures are exact for every dtype (strings hash as
  released; large integers and complex values no longer collide); the
  released `trust_decay=1.0` and `consensus_threshold=0` work again.
- `quantum_computing`: the VQE detector minimises the encoded training
  samples' 1 − P(|0…0⟩) (held-out AUROC 0.997, was 0.51–0.58), encodes
  every feature, scales by the training data and squashes out-of-range
  values into a window shorter than the encoding's period (a sample two
  training ranges out had encoded like one in range), and refuses IBM
  hardware under `MERCURY_OFFLINE`; `cobyla` is SciPy's COBYLA, and every
  optimizer returns the objective at the point it returns.
- `slhdsa_sign_internal` stays removed as a security exception, aligned with
  AMA Cryptography v5 (DEPRECATION.md §6.11).

**Restored — the symbols an earlier cleanup removed as
uncalled, each with its defect fixed.** The removals (7ab59916, f498866a,
f07b724e) are reverted under the Preservation Principle: every symbol is back
under its released name and signature with a test, and the defect tests fail
on the released code.
- `OnlineDriftDetector` is ADWIN0 (Bifet & Gavaldà 2007) with a Bonferroni
  share per feature and the window's range in the bound, so it gives the
  same answer in any units: no false drifts on stationary 3-feature streams,
  where the per-feature KS check raised 12–31 per 2,000 samples; a 1σ shift
  is found in 80–169 samples (20 seeds).
- `ParallelDetectorExecutor` runs on `core.parallel_detectors`: a detector
  past the timeout is reported and the finished results are kept (the
  released loop raised `TimeoutError` and lost them). Detectors that do not
  declare `parallel_safe = True`, and all of them at `max_workers=1`, run in
  the calling thread outside the deadline, the engine's rule.
- `rate_limit_middleware` returns its 429 with `Retry-After`; raising
  `HTTPException` from an `http` middleware reached clients as a 500.
- `load_noaa_tsunami_records` pages HazEL with `itemsPerPage`/`page` (it sent
  `maxSize`, which HazEL ignores) and raises `OfflineModeError` offline.
- `memory_efficient_lru_cache` keys arrays by digest: `str()` abbreviates
  arrays past 1,000 elements, so a changed array was served the old result.
  Arguments with no exact key (model instances, masked arrays) run uncached.
- `_LegacyOAEWeightOptimizer` fits F1 without derivatives (SLSQP stopped at
  the initial weights after one iteration; held-out F1 0.34 → 0.99 on data
  whose signal is in O). `_LegacyOmniAvaEquation` and
  `_LegacyRefactoringConfig` keep their released bodies (the live classes
  drop η^Φ, change the default weights, and reorder the config's fields);
  the six other `_Legacy*` names alias identical live classes.
- `_safe_cholesky` doubles its diagonal shift until the factorization
  succeeds; the single 1e-6 shift failed for any eigenvalue below −1e-6.
- `_compute_sst_anomaly_proxy` counts the bleaching-susceptible acroporid and
  pocilloporid genera (Marshall & Baird 2000; Pratchett et al. 2013); the
  released "first half of the list" counted the tolerant *Porites* and
  missed *Seriatopora*.
- `compute_benchmark_metrics` uses `domain`, which it ignored (temporal,
  spatial and graph; any other domain gets the standard suite, as before);
  `reliability_pool`, `add_conformal_to_detector` and
  `MultiScaleFeatureAggregator` reject an unknown mode, method or missing
  layer instead of substituting a default.
- `IntelligenceFusionEngine` accepts `golden_ratio_weights` again in its
  released third position, without effect and with a `DeprecationWarning`;
  `counterintelligence` and `merit_log` are keyword-only, so a released
  positional call no longer switches counterintelligence off.
- Also back: `TextLogAnomalyDetector` (now taking an `adapter`),
  `optimize_benevolent_detector`, `PatchEmbedding`,
  `create_enhanced_detector`, `create_enhanced_model`,
  `RegulatoryIntelligence`, `run_baseline_benchmarks`, `parallel_map`,
  `get_optimal_batch_size`, `ThresholdDefaults`, `load_configuration`,
  `AuthProvider`, `optimize_gosnn`, `precision_sqrt`,
  `create_int_source_registry`, `make_capability`, `operation_for`,
  `maintenance_disabled_for_tests`, `VQAResult`, `IntegratedPhysicsResult`,
  `EngineConfig.enable_ci`, `OmniMercuryEngine.MAX_WORKERS` /
  `optimization_config` / `parallel_executor`, `TIDE_PERIODS_H` in
  `tsunami_waveform`, and the six `create_omni_*_scalars` tables, documented
  as hand-set constants rather than optimized ones.

The `anomaly/` package exports the three shipped cross-domain cascades
(`EqTsunamiCascadeDetector`, `SolarGICCascadeDetector`, `SurgeFloodCascade`),
so `Dionysus_XIII` binds it again. The TCP transport's two response
serializers are wired into the send path in place of inline `asdict` calls,
so the codec is symmetric.
The `omni_ci_ethical_threshold` / `omni_chaos_lambda_bifurcation` scalars are
kept for now: the fusion 𝐇 term averages every float `EthicalScalars` field,
so removing them would move its target (1.2968 → 1.2997).

**Changed — the lattice group is not computed where the fit switched it
off.** For an aperiodic domain the lattice features carried zero weight but
were still computed for every image. They are now skipped after fit (and
after `load`), with constant maps keeping the feature layout. The Ledoit-Wolf
Gaussians (global and per-position) are now fitted on the active features
only, so zeroed dimensions no longer shift the shrinkage target. The new
`benchmarks/vision_latency.py` times the same fitted model both ways: the
skip saves 82 ms on leather, 48 ms on tile and 110 ms on wood (14–31% per
image). MVTec accuracy is unchanged to every reported digit. The held-out
synthetic benchmark moves by one mistyped leather case and a 0.0001 change
in one pixel AUROC. Tests pin that an aperiodic domain never calls the
lattice code after fit, that a periodic one still does, and that
switched-off dimensions have zero mean and zero precision rows.

**Fixed — FINDΩYOU™ evidence-bridge gaps found by mutation testing.** A
14-mutant pass over `integrations/findoyou_bridge.py` first killed 8 of 14.
The survivors were real test gaps. The unsigned `algorithm` field of the
signature block was never tampered in a test. Nothing covered
non-positive `years_elapsed`, fractional-year rounding, the age-robust
`baseline_score`, or a sighting the screen cannot assess (which must be
`unavailable`, not grounded). Tests for each now kill all 14. The pass also
found an interop bug: `verify_packet` compared the pinned key as a
case-sensitive string, so a key pinned in upper-case hex rejected every
valid packet. It now compares case-insensitively after trimming. A
non-positive `years_elapsed` now reports its value instead of "not
provided".

**Fixed — the self-model ledger's `verify` raised instead of failing
closed.** A corrupted line or malformed signature hex raised an exception
out of `SelfModelLedger.verify`. Each is now reported as a broken record
with its position. The pinned key is compared case-insensitively, as in the
bridge. A 12-mutant pass first killed 7. The existing tamper tests only
asserted that something failed, so with five redundant checks no single
check was ever observed. New tests play a forger who recomputes every digest
except the one a given check guards (schema, index, chain link, record
digest, snapshot digest) and pin each reason. All 12 mutants now die.

**Fixed — the monitoring stack was empty end to end.** Five separate breaks
meant no alert or dashboard could ever show data:
- **Series never recorded.** The alert rules query `omni_detection_*` and
  `omni_model_inference_errors_total`, but nothing incremented them. They are
  now recorded on every component call in the engine's detection, feature and
  fusion paths, and on every message the streaming worker scores.
  `omni_model_roc_auc` / `omni_model_f1_score` are set from the fusion head's
  validation split at the end of each `fit_fusion`, using the served
  threshold rule. They are labelled `dataset="fit_validation"` because that
  split also chose the early-stopping epoch.
- **Declines counted as failures.** On ordinary tabular input, three domain
  models refuse the input on every call (measured). Counting that as an error
  would have kept the inference-error alert firing permanently. A new
  `InputModalityError` (a `ValueError` subclass, so existing handlers are
  unchanged) marks these refusals at eight sites. They feed
  `omni_component_declined_total` and nothing else.
- **The worker never exported them.** The engine worker's `/metrics` served six
  hand-written counters only. It now appends the `prometheus_client` registry,
  as the API already did.
- **The local Prometheus could not start.** `prometheus.yml` loaded the
  PrometheusRule CRD as a plain rule file (`promtool`: "field apiVersion not
  found"). `scripts/render_prometheus_rules.py` now renders the CRD's groups
  to `rules.yml`, and a test fails when the rendered file drifts from the CRD.
- **No series could match the rules.** Every rule selects
  `{app="mercury-agent"}`, and no shipped scrape config attached that label.
  The compose target now attaches it. New PodMonitors
  (`monitoring/prometheus/podmonitors.yaml`) attach `app`, `service` and `job`
  in Kubernetes.

**Fixed — the engine autoscaler waited on a metric that did not exist.** The
engine HPA scales on `omni_detection_queue_depth`. That series was never
published, and the HPA's selector used `app.kubernetes.io/*` keys, which are
not legal Prometheus labels. An HPA with an unavailable metric never scales
down. Consumers now report real consumer-group backlog:
- Kafka: log-end offset minus committed offset over every partition;
- Redis: `XINFO GROUPS` lag plus pending (verified against a live Redis 7);
- in-memory: unread entries.

The worker refreshes the backlog every 10 s. `-1` means the backend could not
report it. `monitoring/prometheus/prometheus-adapter-rules.yaml` serves the
backlog with `max` across workers, filters out the `-1`, and defines the API
HPA's two custom metrics. Tests pin that every series a rule or HPA queries is
one the code emits.

**Fixed — the streaming worker hung while its broker circuit was open.** With
the Kafka or Redis circuit breaker open, `consume()` returns at once without
awaiting. The worker loop then never suspended: it pinned a core and starved
the event loop, so `stop()` and SIGTERM could not run until the breaker
half-opened. An empty consume cycle now sleeps 100 ms. A regression test that
reproduces the hang (it failed before the fix) pins this.

**Fixed — 500s left no trace.** An unhandled route exception never reached the
metrics code, so `http_requests_total{status="500"}` under-counted exactly the
failures the error-rate alert exists for. The request middleware now records
the 500 before the error handler builds the response.

**Fixed — the audit export was always empty.** `record_audit_log` had no
caller. `api/request_audit.py` now writes one entry per finished API request.
Each entry carries:
- the principal (or `anonymous`);
- the route template;
- path parameters, status and duration;
- the proxy-resolved client IP and the correlation id;
- the *names* of query parameters, never their values or the request body.

The entry goes to the export store, and to the tamper-evident log when
`MERCURY_AUDIT_LOG_DIR` is set. That log is one logger instance shared with
the auth auditor. Probes, docs, unmatched paths and 429 rejections are not
audited, so a rejected flood cannot push real entries out of the window. The
export store now uses bounded deques. Trimming a list at its 1M cap copied it
on every append.

**Fixed — the tamper-evident audit log failed verification after any
restart.** Each process start began a new hash chain in the same
`audit.jsonl`. After one restart, `verify_log_integrity` reported "Chain
broken", the same result tampering gives. The chain now resumes from the last
event on disk (or from the newest rotated segment after a rotation).
`mercury serve --workers N` used to interleave N chains in one file. An
exclusive lock now gives the first process `audit.jsonl` and every other
process its own `audit-<pid>.jsonl`. Both files verify, and forgery is still
detected after a resume.

A 19-mutant pass over this work killed all 19. The Kafka-lag mutant needs
`aiokafka`, which CI's full-extras lane installs.

### Visual anomaly analysis, a FINDΩYOU™ evidence bridge, and a durable self-model

`BiometricAnomalyModel.predict` and `AgeProgressionEngine` gain a full,
interpretable image-anomaly capability, and Mercury gains a signed evidence
bridge to FINDΩYOU™. See [`docs/VISION.md`](docs/VISION.md).

- **`omni_mercury_engine.vision` (new).** It learns normal appearance from
  defect-free images and uses no pretrained weights.
  - **Input:** any image type becomes one canonical form, or is refused with a
    stated reason.
  - **Features:** 31 named patch features per scale across 9 groups, all exactly
    invariant to a global gain.
  - **Normal model:** a PatchCore-style coreset kNN plus a Ledoit-Wolf
    Mahalanobis model.
  - **Decision:** split-conformal, with `recalibrate()` for a new session.
  - **Localisation:** hysteresis with half-maximum extents.
  - **Classification:** self-supervised prototypes of seven physical types per
    polarity, simulated on the user's own normals, with an open-set bound.
  - **Environment:** an envelope check that withholds the decision when capture
    conditions fall outside it.
  - **Forensics, kept separate from the conformal decision:** noise
    inconsistency, JPEG grid phase, copy-move and moiré.
  - **Operation:** batch and streaming modes, and `save`/`load` (npz, no pickle).
- **Lattice symmetry for periodic textures.** Each image's two shortest
  translation vectors are found to sub-pixel precision, and every pixel is
  compared with its neighbours one period away.
  - The comparison runs on log-luminance minus an edge-preserving separable
    median. A lighting step is therefore not a break: the residual offset beside
    a 0.8 log-unit step is 0.005, against 0.27 for a half-resolution median.
  - Border and corner comparisons report "no evidence" rather than a false break.
    Before this fix, every top-scoring *normal* MVTec grid image peaked at a
    corner at +350σ.
  - The analyser decides from its normal data whether a domain has a lattice at
    all, and switches the group off for aperiodic textures. The measured
    separation: grid 0.05, carpet 0.22–0.27, leather/wood 0.66–0.73, tile 0.83;
    threshold 0.4.
- **`AgeProgressionEngine`.**
  - Geometry: cardioidal-strain craniofacial growth (Todd et al., 1980) with
    cubic resampling, applied only to children, on a stated, unfitted schedule.
  - Texture: a deterministic DoG mid-band model that injects no noise.
  - `assess_progression` flags SSIM, noise and lost-detail failures.
- **`BiometricAnomalyModel.predict`.**
  - Image input gets the visual analysis after `fit_normal()`.
  - Without a fitted model it runs environment, forensic and presentation-attack
    screening.
  - Face attributes are reported only as far as DeepFace measured them. The
    fabricated `age=25` and 0.5 confidences are gone.
- **FINDΩYOU™ evidence bridge** (`integrations/findoyou_bridge.py`). Each packet
  carries:
  - identity, age-robust match, progression audit and sighting-integrity
    findings in the shared three-state wire values;
  - SHA3-256 image digests;
  - an ML-DSA-65 signature through AMA, verified against a pinned producer key.

  Every FINDΩYOU™ import of Mercury resolves.
- **Self-model** (`self_model.py`).
  - Mercury's identity, probed capabilities, safeguards, measured performance
    and derived limitations.
  - Kept in a SHA3-256 hash-chained, optionally ML-DSA-65-signed ledger, with
    `diff_snapshots` and a CLI.
- **Reproducible benchmarks.**
  - `benchmarks/mvtec_visual_benchmark.py`: real MVTec AD textures, downloaded
    at run time under a pinned manifest.
  - `benchmarks/vision_synthetic_heldout.py`: exact-ground-truth classification
    on held-out images and independent injectors.
- **Measured** (`benchmarks/results/`, produced by the committed code; full
  tables are in `docs/VISION.md`):
  - **MVTec AD textures (real defects).** Mean image AUROC 0.985, pixel AUROC
    0.982. Recalibrated detection is 0.955 at a false-alarm rate of 0.068
    (α 0.1, 10 session normals).
  - **Lattice effect.** Without the lattice group, mean image AUROC was 0.914
    and recalibrated detection 0.819. Grid went from 0.656 to 0.996 image
    AUROC, and "bent" grid defects from 0.00 to 0.83 detected.
  - **Not controlled by α:** the strict-protocol false-alarm rate is 0.247 at
    α 0.05, because MVTec's good test images are shifted from its training
    images. Recalibrating on session normals restores control.
  - **Held-out synthetic (exact types).** Detection is 100% for all 8
    independent injectors in both domains. False alarms are 0.033 (woven) and
    0.050 (leather) at α 0.1, with 0 forensic false positives. Typing:
    same-definition 99.3% / 95.0%, cross-generator 85.4% / 77.5% (woven /
    leather).
  - **Latency** at 256 px on a shared 4-core container
    (`benchmarks/vision_latency.py`): 250–343 ms median by default,
    95–117 ms with `AnalyzerConfig.fast()`.
- **Also fixed:**
  - a line-length formatting regression from the terminology rename in
    `essential_workers.py`;
  - the local gate mirror now checks black over `format.yml`'s wider scope.

### Terminology: civilization-first and Omni Codes everywhere; a repository-wide guard

Two injected framings are replaced everywhere they occurred. A mission
phrasing that put one humanitarian deployment slice ahead of the mission is now
*civilization-first*, and the alternative code-system name is now *Omni Codes*.
The retired phrasings are not spelled anywhere in the repository, guards
included.

- **Identifiers renamed:**
  - the operational scalar `civilization_first_principle` (global scalar network)
  - `EthicalConfig.omni_civilization_first_protection`
  - the essential-worker ethical scalar `civilization_first`
  - the guardrail principle `civilization_first`
  - the TranAD ethical weight
  - the `civilization_first` / `civilization_first_ux` context keys read by `ai_ethics` and the ethics subagent
- **σ_Immutable baseline:** regenerated with the documented
  `scripts/harvest_sigma_baseline.py` procedure across the four committed
  domains. The diff is exactly one name at `names[25]` and `anchor_names[22]`.
  All 127 values, the anchor index sets and σ = 0.9999216795 are unchanged, and
  the signed corpus is unaffected because its canonical bytes carry no names.
- **Guards:**
  - `tests/test_retired_terminology.py` scans every git-tracked text file for
    both phrasings, in every case and separator form. It also checks that the
    scan is non-vacuous and that red-team "survivors" and union-find roots do
    not match.
  - The doc-lint ban no longer spells its phrase; a new `BannedPhrase.example`
    keeps the bite self-test honest.
  - The subagent-roster guard was renamed to
    `test_omni_codes_is_the_only_code_terminology_in_subsystem`.
- **A vacuous test made real:** the crisis-monitor prioritisation test used to
  assert only `isinstance(..., list)`. It now pins the order: life-critical care
  first, evacuation naming the vulnerable groups actually identified, and
  reunification last. A new case pins that pandemic containment precedes all
  other aid.

### Scoped PQC production gate: AMA-free analytic quickstart, fail-closed agent surfaces

Maintainer-directed split of the import-time PQC gate. The quickstart's
analytic path may run without AMA for fast, real measured results, but it must
never amount to launching the Mercury agent minus AMA — so the gate moved from
the package root to every surface that can decide, serve, sign, or encrypt:

- **Gate relocation:** `_enforce_pqc_production_gate()` now runs at import of
  `omni_mercury_engine.engine` (the CLI and API serve the engine, so both
  refuse transitively), `omni_mercury_engine.crypto`, and
  `security/crypto_api.py` — unchanged fail-closed semantics (all three
  algorithms native, FIPS POST operational, version pinned to 5.0.0, no
  escape hatch). The package root and the standalone analytic leaves
  (dataset loaders, individual detectors, metrics) import AMA-free.
- **Lazy security package (PEP 562):** `security/__init__.py` resolves all
  re-exports on first access, extending the existing torch-lazy
  `IntelligenceFusionEngine` precedent. `__all__` is unchanged; AMA-backed
  names (`MercuryCrypto`, PQC key material, readiness guards) raise the
  gate's named refusal at first access, while AMA-free primitives
  (`SafeHTTPClient`, `RateLimiter`, audit logging, `TrustedEndpoints`)
  import without a native build.
- **Contract pinned:** `tests/test_pqc_startup_gate.py::TestScopedGateContract`
  runs six subprocess cases against a stubbed, algorithm-unavailable AMA —
  quickstart chain imports, security primitives import, and engine / crypto /
  root-attribute / `MercuryCrypto` access each refuse naming AMA.
- **Probes retargeted** (a bare `import omni_mercury_engine` no longer proves
  the gate): tier0-safety's real-AMA positive/negative probes now import
  `omni_mercury_engine.crypto`; the Dockerfile smoke test imports
  `omni_mercury_engine.engine` and the container healthcheck invokes the gate
  directly; INSTALLATION / DEPLOYMENT / README / runbook wording updated to
  the scoped contract (including a stale `3.3.0` version in the escalation
  runbook, corrected to `5.0.0`).
- **Measured consumer path:** clean virtualenv with no AMA installed —
  `pip install -e .` in 26 s, install-to-printed-result 51 s including the
  38 MB first-run dataset download, ROC-AUC 0.8613 reproduced exactly;
  `examples/mercury_quickstart.py --help` and the full demo verified under
  both an absent and a stubbed-unavailable AMA.

### Scale drift gate: hysteresis cures rounding-boundary flakiness

The README scale gate's stability property ("a handful of added lines must
not demand a README rewrite") was asserted of a pure rounding function — an
impossible property at bucket boundaries, and the source LOC landed
seven lines below one, turning the full-sweep lanes red over a documentation
integer. Cured in the design, not the test pin: ``--check`` / ``--update``
now keep a published, bucket-aligned LOC figure while it is within one
``LOC_BUCKET`` of the measured count (staleness is bounded by one bucket,
matching the ``~`` order-of-magnitude contract; a hand-typed exact tally is
never accepted). The property is pinned deterministically at the boundary
worst case — not against live-tree state — plus a fire test beyond the
staleness limit and a freshness-rule unit test.

### Adversarial review: eleven findings triaged, nine fixed, every claim re-tested

An adversarial review of the full diff produced eleven findings;
each was verified against a traced failure path before acting. Nine were real
and are fixed; one was a docstring overclaim (corrected to the truth); one —
re-inserting previously-removed parameters could mis-bind positional callers —
was verified against every in-repo call site (all keyword or default calls;
the restored order matches the pre-branch API) and needed no change.

- **Biometric match robustness:** one embedding (the first, labelled when
  several exist) per image, so multi-face or ragged inputs can never crash the
  cosine with incomparable vector lengths; the backend label is decided by
  what *actually produced* the vector, never by what was installed (a DeepFace
  failure that fell back to harmonic features used to be labelled "FaceNet");
  non-numeric inputs return the error-dict contract instead of raising; a
  single image is one harmonic sample (whole-image features, not one
  "embedding" per pixel row).
- **Zero-day risk honesty:** the novelty term steers which candidates win the
  search but no longer inflates the reported 0-1 risk, which is computed from
  reachability alone — supplying a CVE catalogue cannot raise the same
  system's risk score by up to 0.25 any more.
- **DTP diagnostic validity:** each interval re-syncs the clone to the live
  layer and refits ONLY the inverse, so the recorded value measures the
  trained projection's invertibility — not a co-adapting autoencoder that
  trends to zero for any layer.
- **Degenerate-signal honesty:** an unmeasurable signal (too short, constant,
  non-finite) returns ``None`` and the explorer keeps caller-supplied
  characteristics, reporting ``signal_unmeasurable`` instead of overwriting
  real measurements with 0.5 placeholders flagged as "derived".
- **Uncertainty sample honesty:** ``sample_values`` attaches only genuinely
  drawn MC/ensemble stacks; a multi-row batch of caller predictions is no
  longer returned as if it were samples.
- **Fingerprint quality chain wired end-to-end:** the recognizer computes its
  quality map *before* enhancement and passes it in, so the anti-hallucination
  blend runs in the real recognition path; each minutia now records its
  measured local quality (previously hardcoded 1.0); and the overall-quality
  minutiae factor is confidence-weighted, so spurious detections in unreliable
  regions cannot inflate quality (twenty noise-crossings on a near-flat image
  no longer read as a good print).
- **O(n log n) autocorrelation everywhere long inputs live:** the tidal
  detector and the SETI repeating-pattern detector now use the same
  zero-padded Wiener–Khinchin transform as the contact-protocol derivation
  (equal to the direct form at machine epsilon; a one-million-sample capture
  measures in under a second instead of ~10^12 operations).
- Also: the documented biometric "path or array" contract is now real (paths
  load to pixels at every entry; previously a path string became a 0-d array
  no backend could process), and exposure docstrings state plainly that the
  fortress scan's own findings are scalar-grade while per-dimension steering
  is available to callers with localized measurements.

### Depth: simulators grounded in measured signal, a real biometric match path, and a true DTP training comparison

The previous pass made ignored parameters *affect* output; this one makes
the affected modules consume **real upstream signal** instead of caller-fed
summaries, and cures the two deepest discovered gaps outright.

**Contact-protocol exploration measures the signal it conditions on.**
`explore_contact_protocols` now accepts the raw received signal and derives
`snr` / `bandwidth` / `periodicity` itself with the module's own spectral
machinery (`_derive_signal_characteristics`: peak-over-median-floor SNR on a
30 dB scale, occupied-bandwidth fraction, strongest normalised
autocorrelation peak) — measured values override caller claims and are
reported with a `derived_from_signal` audit flag. The orchestrator defect
this fixes: `detect_emergent_life` passed a dict that never contained those
keys, so the previous conditioning always ran on neutral defaults in the real
call path. It now passes the signal through. A clean tone and white noise
separate cleanly (SNR 1.0 vs 0.14, bandwidth 0.004 vs 0.96, periodicity 0.97
vs 0.10).

**Zero-day simulation consumes the fortress scan's measured findings.**
`fortress_scan` previously computed hash-integrity and traffic-anomaly
findings and then discarded them before simulating. The checks now run first
and feed `simulate_zero_day(..., exposure=...)`, where each state dimension's
reachability distance is discounted by `1/(1 + exposure)` — observed
tampering or anomalous traffic makes the simulated surface genuinely easier
to reach. A tampered hash chain measurably raises the simulated risk for the
identical system state; zero exposure is bit-identical to the baseline.

**Treatment optimization grounded in the published Framingham instrument.**
When a patient record carries the Framingham covariates, the risk burden
comes from the in-repo `FraminghamRiskCalculator` (10-year CVD risk,
normalised against the conventional 20 % high-risk boundary) instead of the
generic heuristic; the result reports which model produced it
(`risk_model: framingham_10yr_cvd | heuristic`), and a record missing any
core covariate falls back explicitly rather than letting population defaults
masquerade as patient-specific risk.

**Biometric verification is now real.** `predict({"reference", "test"})`
silently dropped the test image and returned reference-only analysis while
the docstring promised a `match_score`. The two-image path now computes a
genuine embedding cosine match (FaceNet via DeepFace when installed,
otherwise the deterministic harmonic feature path), reports which backend
graded it (`verification_note` states that the fallback is appearance-level,
not identity verification), sets the pipeline anomaly to `1 - match`, and
refuses with a named reason whenever no measurable embedding exists.
`detect_biometric` regains `enable_age_progression`: seeded age-progressed
reference variants (the existing `AgeProgressionEngine`) are matched against
the test image and the best score is reported as `match_score_age_robust`,
with an explicit `applied: False` + reason record whenever the progression
path cannot run. No accuracy claims are made for the progression heuristics.

**Fabrication-class fallbacks removed from the multimodal biometric
engine.** With no embedding backend installed, `MultimodalBiometricEngine`
silently substituted *random vectors* for face embeddings (three sites), so
`match_faces` reported noise-vs-noise cosine as a "similarity";
`analyze_attributes` invented a 30-year-old neutral face, whose fabricated age
then computed as *perfect* age proximity and inflated the MZSS match score;
and `fuse_with_symbolic` fused random stand-in features with symbolic data.
All now fail honestly: embeddings and attributes report unmeasured
(`None` / `measured: False`), the match pipeline returns its designed
extraction-failure result, fusion refuses with a named error, and
`AgeProgressionResult.similarity` is `None` whenever no real backend measured
it. The unvalidated "10-20% accuracy improvement via quantum variant
amplification" claims are replaced with truthful descriptions of the seeded
polynomial heuristics (which are real and now direction-tested: aging darkens
and adds wrinkle-scale noise, rejuvenation smooths — deterministic per seed).

**A true Difference-Target-Propagation training path, measured against
backprop.** `OmniFusionModel.compare_head_training_dtp_vs_backprop` A/B-trains
two identical anomaly-head copies on the model's real frozen-trunk fused
representations: one by Adam backprop, one by canonical DTP (Lee et al.
2015 — the only gradient taken is the loss's local gradient at the output
layer; the hidden target comes from the difference-corrected learned inverse;
every update is layer-local, with Adam as the local solver, as Lee et al.
used RMSprop). On a learnable task the DTP arm cuts its loss by ~90 % with no
error signal crossing layers, and the final DTP-to-backprop loss ratio is
returned rather than hidden. The `use_dtp` diagnostic from the previous round
remains the lightweight invertibility instrument.

**Renamed** `examples/maritime_quickstart.py` → `examples/mercury_quickstart.py`
(it is the product's front door, not a niche demo); README, the Examples
Parity lane and its deterministic-AUC pin follow.

**Wired — every accepted-but-ignored parameter now carries real behavior,
with behavioral tests (hand-computed values and direction/magnitude
properties, not mere acceptance).** A whole-tree sweep (vulture,
100 %-confidence findings, each verified by hand) found
accepted-and-documented parameters that no code path read. Rather than
strip the API down, each is implemented so the surface does what it says:

- `score_function` / `return_samples` (uncertainty): the prediction
  interval becomes the conformal set `{y : s(y) <= q}`, inverted by
  geometric expansion then bisection (recovers exact asymmetric
  boundaries a half-width cannot express); the raw MC/ensemble sample
  stack is attached as a defensive copy when requested.
- `expected_period` (`tidal_pattern_detection`): closed-form harmonic
  least squares `a cos + b sin + c` at the known constituent, reporting
  amplitude / phase / R² / period-deviation (recovers a known sinusoid to
  R² ≈ 1; an off-frequency fit explains ~none of the variance).
- `quality_mask` (`GaborEnhancer.enhance`): per-block quality-weighted
  blend `q*filtered + (1-q)*original` (`q=1` ≡ unmasked, `q=0` ≡ the
  normalised input — the standard guard against hallucinating ridges).
- `input_length` (cardiology ECG net): the contracted window every input
  is resampled to along the time axis, keeping the net's genuine
  variable-length acceptance while making the pooled timebase consistent;
  `num_leads` is the structural channel contract.
- `actual_score` (`LearnableGOSNN.learn_from_feedback`): error-proportional
  update scaling the loss by `min(1 + |target - actual|, 10)`.
- `treatment_idx` (causal sensitivity helper): the VanderWeele–Ding
  E-value with a treatment-conditional pooled outcome SD, capped at 3.0
  for the consumer's calibrated probability range.
- `use_dtp` (`OmniFusionModel.train_with_optimizers`): a
  Difference-Target-Propagation invertibility diagnostic on a clone of the
  anomaly head's first projection, recording a scale-free normalised
  round-trip reconstruction error per log interval — a measurement that
  never mutates the trained model.
- `known_vulnerabilities` (`simulate_zero_day`): a novelty term that steers
  the search toward reachable-but-uncatalogued surfaces instead of
  rediscovering catalogued signatures.
- `patient_state` (`optimize_treatment`) and `signal_characteristics`
  (`explore_contact_protocols`): inputs the fitness functions previously
  ignored now condition them — patient risk burden weights the safety
  term for frailer patients, and SNR / bandwidth / periodicity weight
  error-correction / throughput / universality.

**Also wired — `SchumannHarmonicAnalyzer.spectrum_size`** (from the prior
sweep): the constructor accepted a spectrum size but the detector sliced a
hardcoded `[:512]`; it now slices by the stored size.

`enable_age_progression` in `OmniMercuryEngine.detect_biometric` was
deferred — the two-image path in
`BiometricAnomalyModel.predict` did not actually compare reference to test
(the `test` key was dropped and the promised `match_score` never produced),
so there was no match for age progression to improve — and was then **built
for real in the depth round above**: a genuine embedding match plus
age-robust matching over seeded progressed variants. `num_specialties` in
`MultiSpecialtyNeuralNet` stays removed: the ABMSBoard enum is the single
source of truth for the 24 heads, so it was a correct redundancy removal,
not a dead promise.

Kept deliberately (flagged by the sweep, correct on inspection):
``__exit__(exc_type, exc_val, exc_tb)`` protocol signatures, the
``fit(X, y=None)`` interface convention in the cross-domain adapters,
the requests-adapter override signatures in ``safe_http`` (the pinned
pool ignores proxies by design — honoring one would defeat SSRF
pinning), and the ``return; yield`` empty-async-generator idiom.

### 67th benchmark dataset: DTU Danish-waters AIS with expert-annotated abnormalities; Mercury at published-SOTA parity

**Added — `DTU_AIS_Abnormal`** (`datasets/maritime.py`, category
`maritime`): the only public, expert-annotated real-AIS anomaly benchmark —
521 vessel trajectories from Danish waters (2021-12-13, 120 s resampling),
25 manually labelled abnormal (collision, SAR response, law enforcement,
forced deviations). CC BY 4.0, DOI 10.11583/DTU.21511815 (Olesen,
Clemmensen & Christensen, DTU). Trajectories are served as 22 documented
kinematic/geometric features; `LABEL_SOURCE = "expert_annotated"` keeps it
in the genuine-label headline pool.

Measured on each paper's own protocol, the stock `MercuryAnomalyDetector`
fitted on that paper's unlabelled training file (DOI 10.11583/DTU.21511842,
`DTUAISTrainingSet`) and scoring all 521 labelled trajectories:
June–November 2021 (106,728 tracks; Kristoffer Vinther Olesen's DTU PhD
thesis, 2023, Fig. 6.1) ROC-AUC 0.8785 (95 % stratified bootstrap CI
0.788–0.951, AP 0.364; 0.875–0.879 when fitted on 10,000-track random
subsets); December 2021 without the 13th (12,591 tracks; JMSE 11(11):2085
Fig. 12a) 0.8615 (0.778–0.933, AP 0.248). Both are reproduced by
`examples/mercury_quickstart.py --published-protocols`. GeoTrackNet / VRNN
(0.87), RAE (0.86) and both Seq-2-Seq models (0.84, 0.83) lie inside
Mercury's intervals: parity, not superiority. RVAE (0.72) and the JMSE
two-step clustering (0.73; its 0.79 is Fig. 12b, with stationary tracks
removed) lie below them. An earlier draft of this entry gave 0.876 and a
"full-reference" 0.869 from code that was never committed; the full fit
now gives 0.8785. The in-suite row (uniform harness protocol, seed 42)
reports AUC 0.8613 / oracle-F1 0.6000. The features were written once,
before any score was seen — no tuning iterations, no label contact.

**Fixed — missing-COG policy made explicit (found by the quickstart).**
30 of the 521 tracks carry NaN course over ground (real AIS: COG not
broadcast; some tracks entirely). The first cut of the feature extractor
let those NaNs through, and the benchmark harness's generic validation
silently median-imputed them (row read 0.8631) while any other consumer
would impute differently — two callers, two numbers. Course-derived
features are now computed over finite values only (documented neutral
0.0 when none exist), the extractor raises on any other non-finite
feature instead of emitting NaN, and the demo, the harness and a direct
loader call now agree bit for bit.

**Added — ten-minute quickstart** (`examples/mercury_quickstart.py`,
README "Ten minutes to one honest number"): clone →
`pip install -e ".[pqc]"` → one command prints this dataset's ROC-AUC
next to the published baselines, with the caveats stated. The full
stranger path was measured in a clean virtualenv on a stock Linux
container: 121 s total, no environment variables, no manual build steps
(pip builds the pinned AMA backend itself; the import-time PQC gate
passes on the wheel's co-located native library). The Examples Parity
lane runs the demo end to end and pins the exact deterministic AUC.

Security: the upstream files are pickles; the loader deserialises them
only through a fail-closed restricted unpickler (numpy-array
reconstruction allowlist — any other global raises) and pins SHA-256 per
extracted member, since the figshare zip wrapper is generated per request.
`ndownloader.figshare.com` joins `TrustedEndpoints.TRUSTED_DOMAINS`. Five
loader tests cover feature physics, label alignment, the digest gate and
the malicious-pickle refusal.

### CI wall-clock: cache the shared native build, fix the matrix pip keys, stop the third image rebuild

Three structural costs removed without touching any scan scope, severity
threshold, blocking exit code or test. The Trivy vulnerability database is
deliberately NOT cached: scanning against a stale CVE feed is a silent
security regression, so every scan keeps downloading the current DB.

**AMA native build cached across jobs.** Roughly ten jobs per run rebuilt
the identical AMA-Cryptography C library (the cmake step involves no
Python). The composite action now caches the built tree keyed exactly on
the immutable tag, OS/arch, runner-image generation and the action file
itself — no `restore-keys` fallback, so a near-miss cache can never be
substituted — and both verification gates (native PQC capability probe,
engine import-time gate over AMA's signed library digest) still run on
every job, cache hit or miss, so a stale or tampered entry fails loudly.

**Matrix pip caches actually cache now.** The `pip-quality`/`pip-types`/
`pip-core` cache keys omitted the Python version, so the four matrix
entries raced to save one key: the first finisher won, and the other three
versions re-downloaded torch on every run while their save was skipped as
"key exists". The version is now part of each key.

**`security.yml` stops rebuilding the image from scratch.** ci.yml's
docker-build job already writes the buildx layer cache
(`cache-to: type=gha,mode=max`) and docker.yml reads it; the container-scan
job was the one plain `docker build` with no cache — ~10 minutes of
identical apt + venv layers per run. It now reads the same cache through
the same pinned buildx/build-push actions. The Trivy steps scan the
assembled image with a fresh DB either way, so cached layers cannot hide a
newly published CVE.

### Adversarial excavation: 14 findings root-caused across detectors, gates and egress

A line-by-line adversarial review of the unified branch, each finding fixed
with a regression test that fails on the pre-fix code.

**Fixed — drone ensemble was blind for ticks 20–99.** The kNN density
stream's benchmark-grade minimum (100 reference rows) silently disabled the
drone's only ensemble decision stream inside the first history window: a
total motor collapse plus thermal excursion at tick 50 produced zero faults
while the identical fault at tick 120 was flagged. The minimum is now
configurable (`knn_stream_min_reference`, clamped at 3) and the drone
constructs its ensemble at its own `DEFAULT_FIT_MINIMUM` (20) — the regime
its validation measured (z-gate 8.0). Re-verified: fault at tick 50 and
tick 25 both detected, 0 false alarms over 200 clean ticks.

**Fixed — three crash paths on operator-shaped input.**
`IntelligenceFusionEngine` crashed on a non-numeric or non-finite
`threat_score`/`confidence` in a report (`float()` ValueError in the
evidence path, `TypeError` in the source-ranking sort); both now drop the
malformed value instead of the whole decision. The GDPR checklist raised
`ValueError` out of `check_gdpr_compliance` on timelines like `"30 days"`
or `"48h"` instead of reporting the right or the Art. 33 notification as
unmet — malformed evidence now reads as non-compliance, never as a crash.
The drone `_assess` memo keyed on payload object identity alone, so a
telemetry loop reusing one dict got the first tick's faults forever; the
memo now fingerprints state content.

**Fixed — registry feature cache could serve features without scores.**
The `|f` and `|s` cache entries evict independently, so a features hit
after the scores companion was evicted silently returned `scores=None`
for a detector that does produce scores. A hit now requires both entries
(else it is a miss and an honest recompute), and the tensor-provenance
side dict — one key per distinct payload, forever — is replaced by a key
suffix.

**Fixed — workflow-hardening gate missed compound installs.** The
editable-install regex anchored at line start, so `cd x && pip install -e .`,
`AMA_NO_CYTHON=1 pip install -e .[ml]` and `python -m pip install -e .`
all bypassed the AMA-provisioning check; the regex now matches through
separators, env prefixes and `python -m` indirection.

**Fixed — four latent defects.** `detect_batch` silently stamped the last
row's verdict onto overflow rows when fusion returned fewer probabilities
than batch rows (now loudly logged); `doc_lint` verified only two `::`
segments of a cited pytest node id, accepting typos in the third (now
resolves to any depth); `merge_prometheus_exposition` misparsed exposition
lines carrying a trailing timestamp — the real value folded into the
series key, so repeats of one series survived the merge (now parses the
label set properly, quoted `}`/spaces/escapes included);
`ParetoOptimizer` accepted `mutation_rate` but never forwarded it to the
NSGA-II search.

**Performance — spatial autocorrelation diagnostic was 89–99 % of
`SpatialAnomalyDetector.detect`.** The 99-permutation Moran's I / Geary's C
test ran two sparse quadratic forms per shuffle in a Python loop, and the
kNN graph build looped per point. Both are vectorised — one sparse product
covers every permutation of both statistics via
`sum w_ij (z_i - z_j)^2 = (z^2)'r + (z^2)'c - 2 z'Wz` — with bit-identical
p-values (pinned by a loop-equivalence test). Warm `detect` at 50 k points:
1.07 s → 0.50 s; the diagnostic alone at 5 k: 0.109 s → 0.034 s.

**Cleaned — duplicated leave-one-out kNN logic and dead state.** The
self-match-aware kNN averaging lived twice (`statistical.py` and
`_LocalDensityDetector`); both now call `ml/knn.py:loo_mean_knn_distance`,
which also clamps `k` to the reference size — previously a reference
smaller than `k` padded the average with `inf`. `RealTimeThreatDetector`
kept a full copy of the training data (`_fit_data`) that nothing read.

**Fixed — container Trivy gate red on anthropic-SDK docstrings.** The
anthropic PyPI package (≥ 1.14.0) ships Managed Agents type files whose
docstrings quote the JSON type discriminator of a GCP service-account
key, which Trivy's `gcp-service-account` rule reports as a CRITICAL
leaked key in the image scan. A path-scoped secret allowlist
(`trivy-secret.yaml`, wired into both container-scan steps) suppresses
secret findings inside the installed `anthropic` package only — verified
locally against Trivy 0.70.0: the false positive clears while a planted
service-account marker anywhere else in the image still fails the gate.
Vulnerabilities are untouched and stay on the expiring `.trivyignore`
ledger. (The marker itself is spelled out nowhere in the repo: the
repo-tree scan flags the explanatory text too, which the first cut of
this fix demonstrated first-hand.)

### Unified tree; intelligence fusion decides from evidence

This tree unifies three lines of work (their entries follow below) and reconciles their
seams: the AMA prose pins cov02 wrote against v4.0.0 (`docs/index.md`,
`CONTRIBUTING.md`, `rust_crypto/README.md`) now state the v5.0.0 pin cov3
moved to — caught by cov02's own `workflow_version_drift_gate`, which reports
`ok` on the unified tree — and the two CI-lane fixes both branches made
independently carry cov3's v5-pinned variants held by cov02's gates.

**Fixed — `IntelligenceFusionEngine` decisions came from an untrained
network.** `fuse_intelligence` thresholded the random-initialisation logits
of `AllSourceFusionNetwork`, scaled by golden-ratio constants
(`confidence × 1.45φ × class/5 > 0.5φ`), so `threat_detected` was noise
dressed as φ-optimisation. The network now drives decisions only after
`load_trained_weights` (new; routes through `safe_torch_load`) has installed
a real checkpoint — the trained-only contract GOSNN already enforces. Until
then the engine scores from the reported evidence: a reliability-weighted
mean of per-discipline `threat_score`s, confidence from the reports scaled
by corroboration (full weight at three reporting disciplines), and
`threat_detected` at the `SUBSTANTIAL` level of the `ThreatLevel` enum.
Untrained decisions are deterministic and explainable; five regression tests
pin the contract. The φ-scaled `omni_intelligence_scalars` dict (one key
ever read), the `golden_ratio_weights` kwarg and the dead
`create_omni_intelligence_scalars()` are removed; `_identify_sources` drops
an `attention_weights` parameter it always ignored.

**Fixed — kNN density stream diluted the ensemble on small fits (the
anomaly regression guard went red).** The stream's mean distance to `k = 10`
neighbours only estimates local density when the reference is an order of
magnitude larger than `k`; on ADBench Lymphography (n_train = 71) the
neighbourhood spanned a seventh of the sample and the 50/50 fusion dragged
AUC 0.9836 → 0.9343 / F1 0.8000 → 0.6667, below the pinned floors. The
stream now requires `_KNN_MIN_REFERENCE_FACTOR (10) × _KNN_K` reference
rows and stays off below that, exactly like its existing degenerate-
reference guard. Measured on the deterministic guard subset (seed 42):
Lymphography back to 0.9836 / 0.8000 with every stream win intact
(pendigits 0.8741 → 0.9905, glass 0.7681 → 0.8123, Ionosphere
0.9414 → 0.9692); the guard reports PASS. A regression test pins the
boundary on both sides. The evidence path also drops non-finite report
fields fail-honest (a caller-supplied NaN `threat_score` previously
crashed `fuse_intelligence`; two regression tests).

**Measured — headline benchmark regenerated on the unified detector.**
The committed headline predated the kNN density stream entirely (its
glass/pendigits values are the stream-off figures), so it understated the
shipped detector. Regenerated on the current tree (75 attempted / 66
reproducible / 53 genuine-label headline): mean ROC-AUC 0.8251 → 0.8544
(+0.0293), median 0.8747 → 0.9095 (+0.0348), mean oracle F1
0.5998 → 0.6433 (+0.0435). Per-dataset audit: the four sub-100-train sets
the new gate turns the stream off for are flat-to-up (Lymphography exactly
reproduces its committed 0.9836/0.8000); the largest gains are
ADBench-19 +0.19, ADBench-33 +0.19, NAB +0.17, ADRepository +0.16,
pendigits +0.13; the only >0.02 declines are on two manufactured-label
datasets excluded from the headline. ADBench-46/47 remain the known
sub-0.5 stragglers (both marginally improved). README "Latest Benchmark
Results" synced via `scripts/update_readme_benchmarks.py`;
`check_benchmark_integrity` and the anomaly regression guard both pass. Also fixed: the two `[no-any-return]` errors (3.11
numpy stubs) that failed the Type Checking lane, in
`tests/ml/test_covariance_and_psi_drift.py` and
`tests/core/test_registry_feature_cache.py`.

### Unwired code brought into production; five defective modules removed

Every orphaned module (no importer in production, CI, scripts, benchmarks or
research) was read and vetted. What carried real value is now wired, fixed and
measured; five modules whose output was wrong or fabricated are removed.

**Removed** (`security/quantum_risk_cyber`, `truth_decipher` and
`security/counterintelligence` were later rebuilt from scratch, and
`security/hive_firewall` and `quantum_computing` restored with their defects
fixed; see the entries above):
`security/quantum_risk_cyber` (flagged ML-DSA, the scheme AMA
signs with, as quantum-vulnerable; scored SIKE/Rainbow as post-quantum),
`truth_decipher` (hard-coded "true" ethics inputs; every case recorded as a
success), `security/counterintelligence` (synthesised intelligence reports and
casualty figures), `security/hive_firewall` (a false-positive report made it
block *more*), `quantum_computing` (VQE fit discarded the data; chance AUROC).

**Detection**
- `StatisticalAnomalyDetector` fuses a kNN density stream (fit-time
  referenced, single-row valid, leave-one-out reference): 24 ADBench sets x 3
  seeds, mean AUROC 0.809 -> 0.837 (14 W / 3 L, Wilcoxon p = 0.004); held-out
  false alarms 0.126 -> 0.084. `knn_stream=False` restores the old ensemble.
- Labelled-path component weights are refined on the probability simplex
  (Riemannian Adam): AUROC 0.764 -> 0.777 (p = 0.008), AP 0.385 -> 0.408.
- Drone detector: a working single-row decision (the 0.70 blend gate was
  unreachable; motor collapse, thermal and position faults produced no fault).
  0 false alarms in 5,600 simulated ticks. Faults are attributed to the most
  deviant telemetry group, rules are typed by field, and the detector is
  registered (`drone`, new category `UAV`).
- Registered: `lstm_autoencoder` (trained multivariate reconstruction; AUROC
  0.851 vs 0.785 for the statistical tier, 0.94 on frequency changes) and
  `realtime_threat` (kNN self-match fixed: false alarms 0.076 -> 0.051 at a
  0.05 budget; training-referenced agreement).
- `EnsembleDriftDetector` adds a covariance-structure test (SPD distance,
  leave-window-out null). It catches correlation changes with unchanged
  marginals in 100% of windows (0-2% before) at 0-2% false alarms. PSI uses
  quantile bins plus a chi-squared gate: drift-free false alarms
  100%/100%/52% -> 2%/1%/0% on 50/200/500-row windows.

**Operations and compliance**
- TLP 2.0 marking on every export (headers, per-record label, optional
  envelope; audit logs AMBER+STRICT or stricter) and a TLP-marked report
  endpoint. `contains_pii` now escalates AMBER *or weaker*.
- SIEM / observability alert egress (`MERCURY_ALERT_SINKS`): Splunk HEC,
  Datadog, Elastic/OpenSearch, Prometheus and OTLP spoken natively, gated per
  sink by TLP colour.
- W3C Trace Context propagation; trace ids on request logs and alerts.
- GDPR Arts. 5-49 processing checklist (`compliance.gdpr_processing`) and the
  `gdpr_processing_check` tool.
- Fairness `POST_PROCESSING` via NSGA-II (equalized-odds gap 0.239 -> 0.015
  on held-out data at lower error); a generic NSGA-II engine; spatial
  autocorrelation (Moran's I / Geary's C with permutation tests) on
  `SpatialAnomalyDetector` output.
- `mercury-agent hazard-viz --format html` (self-contained Plotly page).
- Opt-in registry feature cache, feeding the previously uncalled cache-hit
  metrics.

**Defects fixed along the way:** the flagship route could not serialise
per-row arrays; `BetaCalibration` tied every score outside its fit range; the
info-geometry OOD test scored the batch mean against a per-row threshold and
refit its reference on the batch it scored; TranAD's trainer never trained
`input_projection`; the GUI showed AUC -1.0 for perfect separation; and
`ReportManager` HTML and reportlab input were unescaped.

### AMA Cryptography v5.0.0 migration + three red CI lanes fixed at root

The pinned PQC backend moves v4.0.0 → v5.0.0 in lockstep across every pin
site (pyproject `[pqc]` git ref, `_pqc_gate._AMA_REQUIRED_VERSION`/`_RELEASE`,
the `build-ama-cryptography` composite action and all 28 workflow `ama-ref`
inputs, `Dockerfile` `ARG AMA_REF`, `scripts/build_ama_native.sh`,
`scripts/mercury_session_setup.sh`, `gf_env.sh`, and the version-pin tests),
validated against a real v5.0.0 native build: 1174 passed / 0 failed across
`tests/security/`, the startup-gate, enum-compatibility, integrations and
secret-sealing suites, with `workflow_version_drift_gate` clean. Build
toolchain floors rise with AMA's own preflight: setuptools>=84.0.0,
wheel>=0.48.0, cmake>=4.4.4 (the ci.yml audit-venv floors, previously
78.1.1/0.47.0/4.3.2, now match). Two v5 breaking changes land in Mercury:

- **`slhdsa_sign_internal` → `slhdsa_sign_addrnd`.** AMA v5 removed the
  FIPS 205 §9.2 internal interface from its shipped surface (§9 forbids
  exposing it; upstream INVARIANT-50) and exposed the §10.2 hedged
  `slhdsa_sign_addrnd` instead, which applies the `M'` ctx wrapper itself.
  Mercury's wrapper (no in-repo consumers) migrates to the same name and
  contract.
- **σ_Immutable corpus re-signed under v5.** AMA v4's Python layer signed
  ML-DSA through the raw internal interface (`ama_ml_dsa_sign`); v5 routes
  `dilithium_sign`/`dilithium_verify` through the FIPS 204 ctx interface
  (`M' = 0x00 || len(ctx) || ctx || M`, empty ctx), so the v4-era committed
  ML-DSA-65 corpus signature could no longer verify. The corpus is re-signed
  with `tools.corpus_resigner` (ephemeral keys; manifest trust is the git
  commit, unchanged) and both mandatory signatures verify under v5. The
  Kyber real-AMA test now pins v5's stronger secret contract: a wipeable
  `bytearray` shared secret (upstream glance row 24) equal to its `bytes`
  image.

**Security — hybrid signature stripping fails closed.**
`HybridSignatureProvider.verify` initialised `classical_valid = True` and only
overwrote it when a classical signature was *present*, so a hybrid whose
Ed25519 half was stripped in transit reported `classical_valid=True` to a
verifier that supplied the classical public key — a PQC-only forgery would
pass the full hybrid check, defeating the reason a hybrid exists (each half
protects against the other's algorithm breaking). Supplying a classical
public key now demands a verifiable classical half (absent provider, empty
key or missing signature verify False); `classical_public=None` remains the
one explicit opt-out. Both directions pinned in
`tests/security/test_crypto_api.py`.

**Sovereignty — classical Ed25519 moves onto AMA's native backend.**
Mercury's docs have long said AMA is the sole production crypto backend, but
`Ed25519Provider` — behind Raft wire authentication, σ_Immutable corpus
verification, `MercuryCrypto` classical/hybrid signing, tool certificates
and release bundles — ran on PyCA `cryptography`. With the v5.0.0 pin the
provider, `tools/_base` certificate signing, the standalone KAT runner and
the PQC capability probe all route through new `security.pqc_backends`
wrappers (`generate_ed25519_keypair` / `ed25519_keypair_from_seed` /
`ed25519_sign` / `ed25519_verify`) over AMA's hardened native Ed25519
(canonical-encoding rejection, FIPS 140-3 pairwise consistency test at
keygen, health-tested CSPRNG seeding). The key contract is unchanged
(32-byte RFC 8032 seed; `sign` also takes AMA's 64-byte form), and RFC 8032
determinism makes the move invisible to persisted data: the byte-exact
RFC 8032 KATs now run through the AMA-backed provider, a new interop test
pins PyCA-era signatures verifying under AMA and vice versa, and the
committed corpus manifest verifies unchanged. With security/ and tools/
PyCA-free, `tests/test_pyca_import_boundary.py` tightens its allow-list
from (`crypto/`, `security/`, `tools/`) to `crypto/` alone — the tiered
lightweight layer's AEAD/KDF fallback, now the one bounded PyCA surface
(and its fallback log message no longer mislabels that path "pure Python").
Validated: 1,487 tests across the security, tools, distributed and
integrations suites pass on the migrated tree.

**Sovereignty — Mercury gains its own emergency crypto tier; pyca exits the engine entirely.**
AMA is mandatory at import, but a backend can still fail at *runtime* (AMA
v5's FIPS error state inhibits output on every surface once tripped), and
the only fallback in the tree was a third-party one: the crypto/ layer's
pyca AEAD/KDF tier. Mercury now carries `crypto/sovereign.py` — handwritten,
stdlib-only, repository-owned implementations of exactly what an emergency
needs: ChaCha20-Poly1305 (RFC 8439), HKDF-SHA256 (RFC 5869), and
**verify-only** Ed25519 (RFC 8032, strict: non-canonical encodings and
`s >= L` rejected, matching AMA v5). Every primitive is pinned to its RFC's
own vectors and cross-checked against AMA and the pyca oracle
(`tests/crypto/test_sovereign_backup.py`, 30 tests). The crypto/ layer
re-tiers to rust -> ama-native -> sovereign-python with governed
degradation: only an AMA infrastructure exception latches the sovereign
tier (one CRITICAL log); an authentication failure is a verdict, never a
degrade trigger; AES-GCM refuses loudly under degradation instead of
silently switching algorithms (pure-Python AES is a cache-timing hazard);
`ed25519_verify` degrades to the sovereign verifier so read paths stay
alive while signing stays fail-closed (pure-Python signing would leak the
secret through timing — fail-safe reads, fail-closed writes). Continuity is
proven, not assumed: the `ciphertext || tag` format and deterministic HKDF
make all tiers byte-interoperable, and legacy pyca-sealed ciphertexts and
pyca-derived keys verify unchanged in tests. `secure_random` now draws
through AMA's FIPS 140-3 health-tested DRBG when active. With that, pyca
`cryptography` has **zero imports in the engine**: the boundary guard
tightens to an empty allow-list and the package moves from core
dependencies to the dev extra (test oracle only) — production installs and
images no longer carry it.

**Benchmark isolation made executable for PyOD.**
scikit-learn's boundary has an AST-level guard
(`tests/test_no_sklearn_in_src.py`); PyOD's — documented in pyproject as
"only the comparison lane imports pyod, lazily" — was documentation only,
and PyOD wraps sklearn estimators, so a PyOD import in production would pull
sklearn into an operational path without ever spelling "sklearn".
`tests/test_pyod_isolation_guard.py` reuses the sklearn guard's detector
(static + dynamic import forms) to confine PyOD to `benchmarks/`,
`src/omni_mercury_engine/comparison/` and that lane's tests, and separately
forbids any production module from importing
`omni_mercury_engine.comparison` at all, with non-vacuity pins on both
allow-rules.

Separately, three scheduled lanes that were red on `main` are fixed at root
cause (reproduced before each fix): `competitive-benchmark.yml` was the one
engine-importing workflow without the AMA build step, so the
import-time PQC gate failed its first `import omni_mercury_engine`;
`phase3-governance.yml` invoked `phase3_governance.py` by file path, which
drops the repo root from `sys.path` and breaks its own
`research.governed_fusion.*` imports (now `-m`, matching the lane's other
steps; docs updated); and `scripts/verify_data_credentials.py` classified an
HTTP 200 with a non-JSON body (NASA DONKI, 2026-10-12) as a credential FAIL
although no provider in the lane signals a rejected key that way — it now
lands on UNREACH via `_parse_200_body()` with the preview redacted, while a
parseable-but-wrong-shape body stays FAIL, both directions pinned by new
offline tests.
### Refinement pass: main's three red scheduled lanes, a gate CI never ran, a PQC wrapper that could not succeed, and a fabricated overhead figure

Every item below was reproduced on `main` before it was fixed. Each new gate
was run against the pre-fix tree and fails there, and each new regression test
fails when its fix is reverted. Two of the lane fixes below (Competitive
Benchmark AMA provisioning, Phase 3 `-m` invocation) were independently made
elsewhere as well; the unified tree carries the variants with AMA pinned at
v5.0.0, with this tree's gates and tests holding them in place.

**Fixed — `main`'s red scheduled lanes**

- **Security (Trivy container gate) red since the ncurses acceptance expired
  on 2026-09-16**, and the vulnerability DB then published five unfixed HIGH
  findings against Debian 13.7: `util-linux` CVE-2026-76642 / -78408 / -78409
  / -78410 and `systemd` CVE-2026-16742. The runtime stage's OS layer was
  rebuilt and scanned with the gate's trivy 0.70.0, which reproduced the CI
  failure exactly, and every finding was checked against the Debian Security
  Tracker (all trixie `<no-dsa>`). `mount` is now purged from the image (no
  reverse dependency; it ships the SUID `mount(8)` entry point of three of the
  util-linux CVEs). The irreducible remainder — seven CVEs — is accepted in
  `.trivyignore` to 2026-12-26 with per-entry rationale; CVE-2026-53615 is
  dropped (fixed in `util-linux 2.41.5-0+deb13u1`) and CVE-2026-54371 dropped
  (re-scored MEDIUM). The rebuilt layer fails the gate with the old ledger and
  reports zero unaccepted CRITICAL/HIGH findings with the new one.
- **Competitive Benchmark red on every weekly run since 2026-08-09**: the lane
  editable-installed the engine but never built AMA, so the unconditional
  import-time PQC gate aborted at the first engine import. The lane now uses
  the `build-ama-cryptography` composite; replayed locally it prefetches 9/9
  datasets, passes both test steps, completes `--quick`, and the
  competitive-position guard passes. `scripts/check_workflow_hardening.py`
  gains a rule that fails any job which editable-installs the engine without
  that composite (it flags exactly this lane on the old tree; 10 tests).
- **Phase 3 Governed Self-Improvement red on all 14 `main` runs since it was
  added (2026-06-22)**: the dormant-revival routing step ran
  `python research/governed_fusion/phase3_governance.py`, which puts only the
  script's directory on `sys.path`, so `import research...` failed. The step
  (and `docs/PHASE3_GOVERNANCE.md`, which documented the same command) now use
  `python -m research.governed_fusion.phase3_governance`. Replayed end to end:
  the benchmark and the routing step both exit 0 and record three governed
  decisions. A test pins the invocation and runs it in a clean interpreter.

**Fixed — correctness**

- `security/pqc_backends.slhdsa_sign_internal` could never return a
  signature: it checked `addrnd` against the *public-key* length (2n) while
  FIPS 205 and AMA require n bytes, so a correct 16-byte `addrnd` was refused
  by Mercury and a 32-byte one by AMA. It had no callers (the KAT-replay path
  its docstring named does not exist), and AMA 5.0 removes the function it
  imported, which would have broken `import omni_mercury_engine` at the next
  pin bump. Removed. (The v5.0.0 migration entry above ships the replacement
  surface, `slhdsa_sign_addrnd`, built on AMA v5's supported hedged-sign API.)
- `EWMATimingMonitor.get_overhead_estimate()` returned a hard-coded `0.5`
  documented as an estimate, surfaced as `omni_crypto_monitoring_overhead`,
  and two tests asserted that constant was below 2.0. It is now measured
  (monitor time over observed operation time). On native AMA v4.0.0
  operations the real figure was ~35%; replacing `np.median` with
  `statistics.median` in the per-record path (bit-identical over 20,000
  randomised trials, 94 → 22 µs per record) brings it to ~9%, and the class
  docstring now states that figure instead of "<2%". `record_timing` also
  refuses NaN, infinite and negative durations: one NaN became the EWMA mean
  permanently and silently stopped detection for that operation.
- The seven-axis runner's Scalability axis timed each workload size once;
  under CPU contention its score spanned 0.49–1.00 across 12 runs, which is
  the one failure in the baseline full-suite run on `main`
  (`test_runner_is_deterministic_for_fixed_seed`). It now takes the minimum
  of five timings per size: the spread under the same load is 0.09 (idle
  0.14 → 0.07). Only the Scalability row of `docs/BENCHMARKS.md` changed on
  regeneration.
- `tests/api/test_detection_gate.py` annotated `TestClient` responses as
  `httpx.Response`; Starlette ≥ 1.x types them as `httpx2.Response`, so the
  tests mypy lane failed in any environment with `[api]` installed. A
  structural `Protocol` now fits both.

**Fixed — claims nothing enforced**

- `CAPABILITY_MATRIX.md` marks the pillar mutation suite **enforced**
  ("deleting the control fails CI"), but no workflow ran
  `scripts/mutate_pillar_controls.py`. It now runs in `ci.yml`'s
  `neuro-symbolic-tests` job right after the pillar suite (7/7 mutations
  killed, ~80 s).
- `scripts/doc_lint.py` gains rule 4, *cited tests exist*: every
  `tests/…py` path and pytest node id in scanned prose and in test-module
  docstrings must resolve (functions and classes at the level pytest collects
  them). It found five stale citations in four files — `ARCHITECTURE.md` (two
  tests that never existed), `docs/DATASOURCES.md` (a method cited without its
  class), `tests/test_medical_data_sources.py` (a moved file) and
  `tests/loaders/test_hail_loader.py`, whose live-endpoint test never
  existed. That test now exists as `tests/loaders/test_hail_loader_network.py`
  (`@pytest.mark.network`); it also pins the live SPC archive's Vivian 2010
  window to the offline fixture's 69 reports / 4 significant / 8.0 in.
- `CONTRIBUTING.md`, `docs/index.md` and `rust_crypto/README.md` said AMA was
  pinned to v3.3.0; the pin is v4.0.0. The `workflow_version_drift_gate`
  operator tool now also holds `_pqc_gate._AMA_REQUIRED_VERSION` and every
  "pinned to" statement in an AMA paragraph to the `pyproject.toml` ref.

**Changed — dependencies**

- `ruff` 0.16.0 → 0.16.4 and `mypy` 2.3.0 → 2.3.1 at every parity site
  (`pyproject.toml` `[ml]`/`[dev]`, both `ci.yml` install lines, both
  pre-commit revs); a bump of `pyproject.toml` alone fails the pin-parity
  gate. Every CI mypy lane and `ruff` are clean under the new pins.
- Ten GitHub Actions bumps, SHA-pinned; each new SHA was checked
  against its upstream release tag with `git ls-remote`.

**Assessed and not taken.** All 55 remote branches were triaged. Every merged
branch tip equals its final head except two benchmark-only commits;
every closed-unmerged branch is either already on `main` in a stronger form
or would regress a later fix (for example a BOCPD tail double-count and
a `1e15` magnitude cap that clips nanosecond timestamps). A golden-ratio
removal was not taken: no stated score contract is violated and its
replacement literals would move operating points without validation data.
AMA: no tag newer than v4.0.0 exists, so the pin is unchanged; its
reference-integrity and mutation-evidence practices are what rule 4
and the pillar mutation step bring over.

### GCN Kafka MCP server + credential keep-alive

NASA GCN disables a Kafka client credential that has not connected to a broker
in 30 days, and a disabled credential cannot be revived. Wiring the GCN broker
to an MCP client therefore needs a scheduled connection behind it, not just a
configuration entry.

**Added**

- `.mcp.json` — project-scoped `gcn-kafka` MCP server pointed at
  `kafka.gcn.nasa.gov:9092` over SASL_SSL/OAUTHBEARER. Every credential
  reference is a `${VAR}` expansion resolved from the launching shell, so the
  file carries variable names and no values.
- `scripts/gcn_kafka_keepalive.py` — makes one authenticated metadata request
  against the broker (what GCN counts as use) and records it in an owner-only
  state ledger. A failed probe is classified against the OIDC token endpoint
  before it is reported, so a network outage (`UNREACH`, exit 2) is never
  reported as a dead credential (`FAIL`, exit 1). The secret is read from the
  environment or from a mode-600 file the script refuses to read at any looser
  mode, is never written to the ledger (a truncated SHA-256 fingerprint of the
  client id stands in for it), and is stripped from any error text before it is
  printed.
- `.github/workflows/gcn-keepalive.yml` — daily keep-alive from repository
  secrets; a missing secret is reported and skipped, so forks are unaffected,
  and only a rejected credential fails the job.
- `deploy/systemd/mercury-gcn-keepalive.{service,timer}` — daily user timer with
  `Persistent=true`, so a machine that was asleep at the scheduled time still
  connects at next boot.
- `configs/mcp/gcn-kafka.env.example`, a `.env.example` section, and
  `docs/GCN_KAFKA_MCP.md` (setup, verdict/exit-status table, scheduling, and
  the standing gap: the published `kafka-mcp-server` builds on `kafka-python`,
  which reaches OAUTHBEARER only through a token provider supplied in code, so
  the MCP connection is best-effort and the keep-alive is what holds the
  credential open).
- `tests/scripts/test_gcn_kafka_keepalive.py` — 32 offline tests covering
  credential resolution and precedence, the mode-600 refusal, verdict
  classification, ledger durability, and the guarantee that neither the secret
  nor the client id reaches stdout or the state file.
- `.gitignore` now blocks `*.env` anywhere in the tree.

### Deployment renderability, DP soundness, and browser security headers

Fifteen defects fixed across the deployment manifests, the federated privacy
and aggregation mathematics, the API's response-header surface, the rate
limiter, and the release lane. Every fix ships with a regression test or a
render-time gate that fails against the pre-fix state.

**Fixed — deployment could not be rendered or installed**

- `kustomize build` on the development, staging and production overlays exited
  1 (`ConfigMap "mercury-agent-config" does not exist; cannot merge or
  replace`): each overlay declared `configMapGenerator{behavior: merge}` over
  a ConfigMap that the base shipped as a plain resource. `mercury-agent-config`
  is now generated by the base from `k8s/base/config/mercury-agent.env`, so all
  four overlays build.
- The Helm chart could not be installed by either documented command. Three
  subchart dependencies — unreferenced by any template, disabled by default,
  and one pinned to `prometheus 30.0.1`, a version that has never existed in
  the prometheus-community index — made `helm install`, `helm template` and
  `helm dependency update` all fail. The chart now declares none; monitoring
  runs through the Prometheus Operator integration its own templates
  implement.
- A new `manifests` CI job renders every overlay and every values file and
  asserts the required configuration. Nothing validated them before, which is
  how both defects survived.

**Fixed — security**

- **No deployment path set `MERCURY_ENV`**, so every Kubernetes, Helm and
  container-platform deployment ran in *development* mode: the
  `API_KEY_HASH_SALT` guard never fired and every API key in every deployment
  was hashed under the salt built into the source, CORS installed the
  localhost allowlist, and the production JWT branch was unreachable. It is
  now set on every path, and `helm/mercury-agent/values.yaml` no longer claims
  a guard that could not fire.
- **The kustomize base shipped a usable JWT signing key.** `secret.yaml` was
  in `resources:`, so `kubectl apply -k k8s/base` installed
  `JWT_SECRET_KEY: "CHANGE_ME_IN_PRODUCTION_USE_SECURE_RANDOM"`. No key
  material ships now, and `JWTAuth` rejects an empty, placeholder or
  low-entropy key in production instead of only checking for absence — an
  empty value was previously adopted as the signing key outright.
- **No security response headers on any surface.** `api/security_headers.py`
  adds CSP, `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`,
  `Cross-Origin-Opener-Policy`, `Cross-Origin-Resource-Policy`,
  `Permissions-Policy` and TLS-conditional HSTS to every response, with a
  policy selected per surface. The edge did not compensate: the kustomize
  ingress carried them in a `configuration-snippet` annotation that default
  ingress-nginx has *rejected* since v1.9.
- **Every persistent store ran in memory** in Kubernetes and Helm:
  `MERCURY_KEYSTORE_PATH` was set nowhere, so accounts, API keys, sessions and
  quota died on restart and rate limits multiplied by replica count. It now
  points at the ReadWriteMany volume both deployments already mounted.
- The `federation/privacy.py` (epsilon, delta)-DP release now delivers its
  stated guarantee: sensitivity from the clipping norm alone (no `1/n`
  divisor), one joint-sensitivity mechanism instead of thirteen separately
  budgeted ones (the recorded epsilon was understated by up to 13x), analytic
  Gaussian calibration valid for every epsilon (the classical form misses
  delta by 2.3x at the epsilon 10 the docstring advertised), and the SHA-256
  fingerprint of the raw training data redacted from the release.
- `FederatedAggregator.aggregate` no longer republishes every node's raw-data
  fingerprint in the global model's `data_hash`.
- The pod-level `runAsNonRoot: true` and the `init-permissions` container's
  `runAsUser: 0` were contradictory; no pod in the API Deployment could start.
  The container is gone — `fsGroup` already does its job, and its
  `chmod -R 755` would have stripped the group write bit.

**Fixed — correctness**

- Federated precision aggregation pooled in the wrong space. Measured relative
  error against a centralized fit on the same data: **0.0012–0.0013** across
  node-mean gaps 0/1/4, versus **0.0055 / 0.2756 / 0.5587** for the plain
  average of precisions it replaces.
- The default rate limiter delivered **2.0 requests/minute against a
  configured 100** for any client polling faster than `rpm / 60`: the
  in-memory refill truncated with `int(elapsed * rate)` while advancing the
  timestamp, discarding the elapsed time that produced the remainder. Both
  backends now refill fractionally and deliver identical rates.

**Fixed — configuration and release**

- Sixteen `OMNI_*` ConfigMap keys with zero readers in `src/` removed from
  both deployment systems; the production overlay's entire "hardening" was
  two of them.
- Placeholder `*.example.com` hostnames and a dangling TLS Secret removed from
  both ingresses; enabling a Helm ingress without a hostname is now a
  render-time error instead of a rule-less Ingress.
- `release.yml` triggers on bare semver tags as well as `v`-prefixed ones —
  the 2.x line is tagged `2.0.0`, which `v*` never matched — and its test gate
  no longer pipes pytest into `head -100`, which fails with SIGPIPE under
  `set -o pipefail` regardless of the results.
- `[tool.semantic_release].version_variable` targets a rewritable literal;
  `__init__.py`'s `__version__` is a function call and could never be bumped.
- Kubernetes workload version labels tracked 2.0.0 against a 2.1.0 project.
- README's "Standard Installation" produced a non-importable install (it never
  mentioned the mandatory AMA native PQC build) and advertised a
  `pip install mercury-agent` for a distribution that is never published.
- Unguarded optional imports in test bodies now use `pytest.importorskip`, so
  a missing extra skips rather than errors.


### Repository refinement audit: 19 defects fixed across correctness, security, safety, DP and consensus

A full-repository engineering audit.
Every automated gate was already green and no sensitive/internal content
required removal, so the value is concentrated in **defects on code paths the
13,802-test suite did not cover** — surfaced by running the real runtime (AMA
v3.3.0 PQC backend built from source) and an adversarial, reproduce-first hunt.
Full itemised record in [`docs/REPOSITORY_REFINEMENT_AUDIT.md`](docs/REPOSITORY_REFINEMENT_AUDIT.md);
each fix ships with a regression test that fails against the pre-fix code.

**Fixed — correctness & reliability**

- **`compute_bca` crashed on every call** (`core/score_calibration.py`):
  `np.erfinv` does not exist in NumPy. Replaced with exact
  `scipy.special.ndtri`/`ndtr`, also correcting a `tanh` approximation of the
  normal CDF that made the BCa endpoints inconsistent.
- **Resonance score computed from the wrong array** (`core/three_r_mechanism.py`):
  `compute_dominance_score` passed the whole `(frequencies, magnitudes)` tuple to
  `np.abs(...).sort()`; now uses the magnitude spectrum.
- **Geological trainers crashed on real data** (`detectors/geological/disaster_detectors.py`):
  batched with the `n_samples` fallback instead of the loaded count
  (out-of-bounds `randperm`), and `train_seismic_analyzer` unpacked a 4-tuple
  `forward()` into two values. Both fixed.
- **Flagship detector crashed on missing data** (`detectors/statistical.py`):
  a single NaN cell aborted the adaptive operating-point path with an opaque
  histogram range error. The valley-depth step now excludes non-finite scores
  (matching the Otsu step), preserving the documented NaN-propagation contract.
- **Dict feature-cache collision** (`engine.py`): dict inputs were keyed by a
  constant placeholder, so distinct payloads returned the first payload's stale
  features. Dict inputs now compute fresh; array/tensor inputs still cache.
- **`RefactoringEngine` cache collision** (`core/three_r_mechanism.py`): analysis
  keyed by `f"{module}.{name}"` collided for same-named callables; now keyed by
  qualname + code location.
- **`float(None)` on explicit JSON null** (`loaders/energy_loader.py`): the EIA
  parser now coerces a missing/`None` value to `0.0`.
- **FEMA year filters** (`loaders/fema_loader.py`): `hurricane_2024` and
  `fire_2023` filtered by incident type only, returning every year's
  declarations; they now constrain to their fiscal year.
- **Marine grid assignment** (`loaders/marine_loader.py`): a non-empty frame
  missing coordinate columns raised a length-mismatch `ValueError`; the missing
  column now yields a length-aligned NaN series.
- **STFT hop division-by-zero** (`detectors/spectral_vibration.py`): `hop` is
  clamped to at least 1, so `overlap_ratio >= 1.0` no longer raises.

**Fixed — safety-relevant**

- **Tsunami false-alarm on corrupt data** (`detectors/geological/disaster_detectors.py`):
  a single NaN drove `wave_height` to NaN and `_determine_severity(NaN)` through
  to MAJOR ("EVACUATE coastal areas immediately"); non-finite waveforms are now
  rejected.
- **Raft could snapshot uncommitted state** (`distributed/raft_consensus.py`):
  `_compact` snapshotted at `len // 2` ignoring the commit index. Compaction is
  now clamped to the applied prefix (the node reports the applied index), so it
  never folds uncommitted entries into a snapshot.

**Fixed — security**

- **Authorization decorators failed open** (`api/auth.py`): `require_role` /
  `require_permission` only enforced when a `request` kwarg was present. Now
  fail-closed (HTTP 401) and resolve the principal from request state or an
  injected `User`.
- **Path-containment prefix bypass** (`security/input_validation.py`):
  `validate_path` used `str.startswith`; now uses `Path.is_relative_to`.

**Fixed — API contract**

- **Validation errors surfaced as HTTP 500** (`api/server.py`): `detect_univariate`
  let its broad `except Exception` re-wrap an intentional `HTTPException(400)`;
  added the `except HTTPException: raise` passthrough.
- **Univariate `method` ignored** (`api/server.py`): the enum-validated `method`
  field was never dispatched. Now runs z-score or IQR (isolation_forest → 400),
  recording the algorithm in `summary.algorithm`; the endpoint identity
  (`method: "univariate"`) is unchanged.

**Fixed — differential privacy** (`federated_learning/privacy.py`) — both bugs
erred conservatively (over-noise / over-report spend), so the fixes correct the
accounting without weakening any guarantee:

- **Accountant/noise disconnect**: `privatize_gradients` fed the accountant a
  `noise_multiplier`-derived scale the mechanism never applied; it now accounts
  for the mechanism's own noise scale (the noise actually added).
- **Basic composition was super-linear**: `_basic_composition_query` summed every
  stored cumulative value (double-counting) and divided delta by the query
  count; it now composes linearly (`prior_cumulative + this_query_cost`).

**Reviewed, no change**

- Rejected one reported "epoch ÷ 1e9 is 1000× too small" defect in
  `loaders/fema_loader.py` as a false positive (`datetime64[ns] → seconds` is
  correct). Documented the `energy_loader` synthetic-Kp fallback for grid events
  as a data-modelling product decision (no defined "correct" value to patch).


### Platform completion: account frontend, usage endpoint, operator CLI, platform metrics, deployment profile

The launch-facing half of the platform work: everything a
public deployment still lacked on top of the hardened backend below. The
additive/opt-in contract is unchanged — with no env vars set, behaviour is
byte-identical (the frontend is off by default and `/` keeps returning 404).

**Added**

- **`api/frontend.py` + `api/frontend_assets/`** — the opt-in browser account
  UI (`MERCURY_FRONTEND_ENABLED=true`): landing, register, login with an
  inline second step on the API's stable `two_factor_required` code, the three
  pages the auth emails already link to (`/verify-email`, `/reset-password`,
  `/confirm-email-change`), and the dashboard — profile, API keys with the
  one-time raw-key reveal, usage vs. effective limits, password/email change,
  the full 2FA lifecycle (client-side QR via the vendored MIT
  `qrcode-generator`; one-time recovery-code display), JSON data export, and
  typed-confirmation deletion. Vanilla HTML/CSS/JS with `fetch`; the
  `mercury_csrf` cookie is echoed as `X-CSRF-Token` on every state-changing
  call; no build toolchain, no CDN, no inline handlers (CSP-hygiene test).
- **`GET /api/v1/auth/usage`** — the caller's in-window usage next to its
  effective limits (override > tier > default) and tier name, for browser
  sessions and `X-API-Key` callers alike (a key reads its owning account's
  bucket). Backed by the new read-only `QuotaEnforcer.snapshot()` and a
  process-wide shared enforcer, so the numbers shown are exactly the numbers
  the quota middleware enforces.
- **`mercury-agent platform …`** — the operator CLI over the shared SQLite
  store and audit dir: `account show|list|set-tier|disable|enable` (disable
  also revokes live sessions), `quota override set|clear|show`,
  `usage report [--account|--top N]`, and `audit verify` (wraps
  `SecureAuditLogger.verify_log_integrity`; exit 1 on a broken chain).
  Refuses clearly when `MERCURY_KEYSTORE_PATH` / `MERCURY_AUDIT_LOG_DIR` are
  unset; never prints secret material.
- **`api/platform_metrics.py`** — lazily-registered Prometheus counters for
  the platform paths, served by the existing `/metrics` target:
  registrations, logins (`ok`/`fail`/`2fa_challenged`), throttle denials by
  action, quota denials by ceiling, mailer sent/failed, maintenance sweep
  results (per-step pruned counts, TOTP sealing migrations, audit segment
  prunes, step errors). Import-guarded — the core lane needs no prometheus —
  and label cardinality is bounded to code-defined sets.
- **`docker-compose.platform.yml` + `deploy/Caddyfile`** — the single-host
  deployment profile as a compose overlay: a named volume at
  `/var/lib/mercury` for the SQLite/audit state, host-env pass-through with
  safe defaults for every documented `MERCURY_*` variable,
  `MERCURY_TRUSTED_PROXY_HOPS=1`, and a Caddy TLS edge (auto-TLS for
  `app.mercuryagent.global`, active `/health` checks). Runbook in
  `docs/PLATFORM_HARDENING.md` → Deployment; a structural test keeps the
  overlay's env vars in lockstep with the configuration reference.

**Changed**

- **Throttle boot invariant shipped** — `routes/accounts.py` declares
  `DISPATCHED_THROTTLE_ACTIONS`; a test locks it to `DEFAULT_ACTION_RULES`
  in both directions and the first limiter build cross-checks it at runtime
  (error log + `mercury_platform_throttle_rule_mismatch_total`, never a
  crash). Request-time `check()` stays allow-on-unknown.
- **`MERCURY_QUOTA_FAIL_CLOSED`** (default `false`) — a failure of the quota
  infrastructure itself can now deny metered routes with 503 instead of
  admitting unmetered, for operators who rate accounting above availability.
- **Platform contact default** moved from the personal gmail address to
  `contact@mercuryagent.global` (`MERCURY_CONTACT_EMAIL` still overrides).
- **Agent runtime requirements documented** — `[ml]` is a hard requirement
  for the agentic stack (torch imports at module load), and the σ_Immutable
  gate refuses with `check="gosnn_unavailable"` when the trained network is
  unavailable; a new orchestrator-boundary guard test pins that refusal.

### Free-service platform hardening: auth abuse controls, at-rest 2FA, quotas, lifecycle, sessions

Hardening pass over the account/persistence/auth/metering platform so
Mercury Agent can run as a free, public, account-based service. **Additive and
opt-in throughout** — a solo self-hoster who clones and runs the repo keeps
byte-identical behaviour (in-memory, no accounts, no SMTP, no quotas); the
machinery activates only when the documented env vars are set. Full design,
threat model, configuration, deployment, migration, and acceptance checklist:
`docs/PLATFORM_HARDENING.md`.

**Added**

- **`api/client_ip.py`** — trusted-proxy client-IP resolution: the client is
  read from the right-most trusted `X-Forwarded-For` hop
  (`MERCURY_TRUSTED_PROXY_HOPS`, default 0 = header untrusted), closing the
  rate-limit bypass where a rotating header minted a fresh bucket per request.
- **`api/rate_limit_store.py`** — shared, restart-persistent SQLite token-bucket
  (atomic `consume_token`) + fixed-window counter stores, and an
  `ActionRateLimiter` with per-IP and per-account limits on login, register,
  password-reset request, and resend-verification
  (`MERCURY_AUTH_RATE_<ACTION>`). Limits are now global across workers and
  survive restarts.
- **`api/secret_sealer.py`** — TOTP seeds sealed at rest with AES-256-GCM via
  the quantum-resistant `SecureDataHandler.encrypt_at_rest`, AAD-bound to the
  account. Stable key from `MERCURY_DATA_ENC_KEY` or HKDF-derived from
  `AMA_MASTER_SEED`; refuses to seal under a process-lifetime key for a durable
  store.
- **2FA recovery codes** — ten single-use backup codes issued on enrollment
  (hashes only), consumed atomically, with a security-notice email on use.
- **TOTP replay rejection** — `verify_totp_with_step` + persisted
  `totp_last_step` reject any code whose time step is not strictly greater.
- **Account lifecycle endpoints** — authenticated change-password
  (session-rotating), two-step change-email (re-verifies the new address),
  account deletion, and data export; plus `POST /auth/resend-verification`.
- **CSRF defense-in-depth** — SameSite=Lax cookies + double-submit
  `X-CSRF-Token` on every state-changing authenticated POST
  (`MERCURY_CSRF_PROTECTION`).
- **`api/quota.py` / `api/quota_middleware.py`** — per-account/per-tier quotas
  (`MERCURY_QUOTA_TIER_*`, `quota_overrides`), wired into the metered detection
  and batch routes with an atomic reserve→run→commit path (hard request ceiling,
  HTTP 429 + `Retry-After`), gated by `MERCURY_QUOTA_ENABLED`.
- **`api/auth_audit.py`** — auth-event audit trail (login success/failure,
  password/2FA/email/deletion) via `SecureAuditLogger` when
  `MERCURY_AUDIT_LOG_DIR` is set, else structured logging; account id only, never
  the email.
- **`api/maintenance.py`** — startup + periodic pruning sweeps for expired
  sessions, consumed/expired tokens, aged usage-ledger rows, and stale
  rate-limit state, which also apply the plaintext-TOTP → sealed migration.
- **`api/email_templates.py`** — matched plaintext + inline-styled HTML
  transactional emails with a `List-Unsubscribe` header.
- **`scripts/calibrate_password_kdf.py`** — measures the scrypt cost ladder on
  the target hardware.

**Changed**

- **Password KDF upgraded to memory-hard scrypt** (`hashlib.scrypt`,
  `n=2^15,r=8,p=3`, `MERCURY_SCRYPT_*`-tunable). Legacy PBKDF2 hashes still
  verify and are transparently rehashed on the next successful login.
- **Session hardening** — idle vs absolute timeouts, rotation on privilege
  change, and a configurable remember-me (`MERCURY_SESSION_*`).
- **Login is timing-uniform** — the unknown-email path burns the same KDF cost
  as a wrong-password attempt (no enumeration by response time).
- **Singleton lazy init is race-free** — `get_api_key_store`,
  `get_auth_service`, `get_action_limiter`, `get_rate_limiter`, and
  `get_audit_logger` use lock-guarded double-checked construction.
- **CI gate watchdog** (`scripts/ci_gate_watchdog.py`) excuses superseded
  `cancel-in-progress` cancellations (a newer clean verdict on the same branch)
  while still alerting on a branch that never completes — cutting false-red
  alarms on busy PRs. `gate-watchdog.yml` now feeds `headBranch`.

### Security + CI-integrity hardening pass: centralized torch checkpoint loader, gate-integrity fixes, root-cause bug fixes

Continuation of the coverage/engineering round, focused on the load-bearing
security surface and on CI gates that could pass without actually protecting
anything.

- **Centralized, fail-closed `torch.load` (RCE hardening).** `torch.load`
  runs a pickle VM; without `weights_only=True` a malicious `.pt` executes
  arbitrary code at load time. The property was spread across ~27 call sites
  each passing the keyword by hand — one omission (or a torch downgrade) from
  re-opening the hole. New `omni_mercury_engine.security.safe_torch.safe_torch_load`
  is the single sanctioned loader (hard-pins `weights_only=True`, no bypass;
  validates path/size; rejects a custom `pickle_module`; translates the
  restricted-unpickler refusal). Every call site (engine, geological/visual
  detectors, ml inference + hazard training, space, core) routes through it,
  and `scripts/check_torch_load_safety.py` (wired blocking in workflow-hardening)
  fails on any raw `torch.load(` in `src/` outside the wrapper. Tests prove a
  malicious `__reduce__` payload is refused and never executed.
- **`migrate_pkl` legacy protocol 0–2 fix.** Beyond the protocol-5 fix below,
  array pickles written at protocols 0–2 (the oldest legacy artefacts) refused
  with exit 5: numpy stores the databuffer as a latin-1 `str` and rebuilds it
  via `_codecs.encode`, which was not allow-listed. Added `_codecs.encode` (an
  inert byte transform — no import/exec/OS access, un-chainable to RCE under
  this allow-list) with a security justification, a protocol 0..HIGHEST
  round-trip suite, and a proactive completeness test that fails if a future
  numpy resolves any reconstruction global outside the allow-list.
- **CI gate-integrity fixes.** `CI Success` now aggregates *every* blocking job
  (ml-tests, ethics-audit, docker-build, docs-build were omitted), and a
  `skipped` spine job (upstream failure) is a distinct non-passing state rather
  than green. The σ mutation gate's `cancel-in-progress` is `false` (it was
  cancelled on every push and never reached a verdict), runs a cheap parallel
  sample on every PR plus the exhaustive lane on hot-path/scheduled runs, and
  the harness gained `--jobs` (isolated per-worker trees) so a full run is
  bounded structurally, not just by a per-mutant budget. New `gate-watchdog.yml`
  + unit-tested `scripts/ci_gate_watchdog.py` file/update a tracking issue when
  a blocking gate repeatedly cancels, times out, or never completes.
- **Root-cause bug fixes with regressions.** (1) `biometric` `detect_anomaly`
  returned numpy scalars (`np.bool_`/`np.float64`) papered over by nine
  `# type: ignore[...,unused-ignore]` comments — coerced at the boundary to
  real `bool`/`float`, all nine ignores deleted, strict-type tests added.
  (2) `utils/feature_cache` INT8 dequantization divided by 255 without
  restoring the scale/zero-point, corrupting every cached value to ~[0,1];
  fixed the affine inverse (params carried on the entry) and
  `select_top_features(k=0)` (returned all features). (3) `truth_decipher`
  `isinstance(x, torch.Tensor)` crashed when torch is absent, and
  `determine_ethics` mutated the caller's context dict. (4) `utils/report_generator`
  wrote reports with no `encoding=`, raising `UnicodeEncodeError` on a
  non-UTF-8 locale — all writes now UTF-8.
- **Fingerprint matching accuracy floor.** New labeled genuine-vs-impostor
  sweep (measured TAR 1.00 / FAR 0.00) with floors, so a silent matcher
  regression fails a functional gate instead of passing every unit test.

### Engineering pass: Python-3.14 CI fix, strict-mypy graduation, ROADMAP 17/18 offline gates, heavy-ML coverage uplift, two root-cause fixes

Full audit trail: `docs/ENGINEERING_PASS_REMEDIATION.md`.

- **Fixed the Python-3.14 Core-Tests failure** (the only red lane).
  Python 3.14 raises `pickle.DEFAULT_PROTOCOL` 4→5; under protocol 5 numpy
  serialises arrays via `numpy._core.numeric._frombuffer` instead of
  `multiarray._reconstruct`, which `tools/migrate_pkl.py`'s restricted
  unpickler did not allow-list, so every array-bearing payload refused
  with exit 5. Allow-listed `_frombuffer` (an inert buffer→ndarray builder,
  same posture as `_reconstruct`; RCE globals still refused); regression
  parametrised over pickle protocol {4,5}.
- **Fixed a pre-existing verify-real-pqc failure:** the trained-σ-gate
  test asserted `backend=='torch'` in the deliberately-no-torch
  pqc-production-check lane; guarded `TestTrainedGateEndToEnd` with
  `skipif(not HAS_TORCH)` (fail-closed path still covered elsewhere).
- **Strict-mypy graduation:** 14 already-clean test dirs (scripts, cyber,
  decision, distributed, emergent, evaluation, federated, medical, metrics,
  proofs, reasoning, research, truth_decipher, utils) moved from the relaxed
  to the strict lane (0 errors each; 131 files clean combined).
  `docs/MYPY_TEST_STRICT_MIGRATION.md` is the dir-by-dir plan for the rest.
- **ROADMAP row 17 offline CI-gated fallback:** `scripts/check_benchmark_integrity.py`
  fails closed unless the committed benchmark JSON is internally consistent
  and its headline AUC/F1 **recompute** from the per-dataset rows (fabrication-
  proof), plus README parity; wired blocking in workflow-hardening + a 13-test
  suite. **Row 18 CI-gated honesty lock:** `tests/narrative/test_language_scope.py`
  enforces the README English-only scope. Both rows gain concrete plans +
  dependency lists.
- **Coverage uplift on six heavy under-covered modules** (`biometric/*`,
  `quantum_computing/*`, `core/adaptive_fusion`, `ml/compression`,
  `datasets/ocean`): 621 deterministic, no-network tests taking the target
  surface from ~0 to **98.62%** line+branch (3,307 stmts + 968 branches).
  New `tests/biometric/` + `tests/quantum_computing/` packages wired into the
  core lane.
- **Two root-cause defects surfaced by that pass, fixed with regressions:**
  (1) `biometric/fingerprint_recognition.py` computed the minutiae crossing
  number on raw uint8 skeletons, so a 0→1 transition underflowed and ridge
  endings/bifurcations were never detected via `extract()` (matching ran on
  empty minutiae) — cast to signed int; (2) `quantum_computing/detector.py`
  used `threshold or self._threshold`, silently discarding an explicit
  `threshold=0.0` — replaced with a `None`-check.
- **Mutation gate could never complete on a PR runner (fixed).** Once the
  3.14 core fix unblocked the `needs: core-tests` chain, the σ mutation
  job ran and hit its 120-min timeout: `run_test_command` used
  `subprocess.run(timeout=...)`, which SIGKILLs only the direct child and
  then blocks in `communicate()` on a grandchild holding the stdout pipe,
  so one mutant's stray subprocess stalled the whole gate (82 min, orphan
  pytest processes). Fixed by running the test child in its own session
  and `killpg`-ing the whole process group on timeout (default
  `--test-timeout` 600 → 120s). Full local re-run: 120/120 mutants, no
  hang, 119 killed / 1 survived (99.2%) in 8m40s.

### Maintenance: intersectional fairness (ROADMAP row 6), σ_Immutable mutation gate (row 8), orphan dispositions (row 19), scripts/ mypy-debt clearance, coverage uplift, nine root-cause defect fixes

Full audit trail with measured before/after evidence:
`docs/MAINT1_2_COVERAGE_REMEDIATION.md`.

- **Intersectional fairness (row 6 closed).** `ml/fairness.py` gains
  joint-subgroup metrics — `build_intersectional_groups`,
  `FairnessAuditor.compute_intersectional_parity`,
  `compute_intersectional_equalized_odds` — with an
  `intersectional_min_group_size` floor that excludes and *reports*
  sparse cells, and an `insufficient_data` flag instead of a fabricated
  verdict. `audit()` accepts multi-feature input (mapping or 2-D array);
  the engine's `_audit_fairness` dict shape is now natively typed — the
  removed `type: ignore` had masked a dead runtime path (the dict hit
  `np.unique()`, raised, and the broad `except` silently returned
  `None`). Pinned by a Simpson's-paradox regression: marginals exactly
  fair, joint disparity 0.3 flagged naming the worst-off cell.
  `tests/fairness/` (41 tests) graduates into the strict mypy lane and
  the core coverage lane.
- **σ_Immutable mutation-testing gate (row 8 closed).**
  `scripts/run_sigma_mutation_gate.py`: deterministic AST-based
  mutation over the σ hot path, in-place mutants with byte-exact
  restoration, red-baseline abort, timeout-as-killed, stride-sampled
  bounds, JSON report; 11 harness unit tests. First honest measurement:
  **9.7%** kill rate (12/124) — the interface-level σ subset pinned no
  arithmetic. Two semantic pinning suites respond
  (`test_sigma_immutable_gate_semantics.py`,
  `test_sigma_immutable_corpus_semantics.py`), raising the measured
  kill rate to the floor now enforced by the blocking, path-filtered
  `.github/workflows/mutation-testing.yml` lane (plus weekly cron).
- **Orphan dispositions executed (row 19 closed).**
  `detectors/cross_domain_frequency.py` (implemented, documented, but
  unreachable) is wired into the `detectors` public surface;
  `core/di.py` (zero consumers since inception, injection inert under
  PEP 563, maps bit-rotted) is deprecated via `DIDeprecationWarning` +
  DEPRECATION.md §1.3, kept functional per the preservation policy.
- **Nine defects fixed at the root**, each with a regression test:
  (1) engine fairness audit dead for its documented dict input; (2)
  integer-typed groups silently zero-filled in threshold mitigation;
  (3) reweighting mitigation was a placeholder — now real
  Kamiran–Calders weights with `get_sample_weights`; (4)
  `calibrate_all_thresholds()` stored the *unclamped* optimum in the
  threshold registry while returning the clamped value, so
  `get_threshold()` served operating points above `MAX_THRESHOLD_CAP`;
  (5) `PointAdjustmentEvaluator(search_best_threshold=False)` was never
  consulted — now honored fail-loud; (6) `ServiceContainer.resolve`
  re-created falsy singletons (truthiness vs `is not None`); (7+8)
  `ComponentFactory` detector/model maps named nonexistent classes —
  every built-in `create_detector` call raised `AttributeError`; (9)
  drift detection's fixed `kl_threshold=0.1` default falsely flagged
  same-distribution data below n≈2000 (measured histogram-estimator
  bias) — replaced by a deterministic permutation-null calibration
  (`kl_threshold=None` default; explicit floats keep the legacy branch
  byte-for-byte; `DriftResult.kl_null_quantile` exposes the null).
- **Input-validation hardening.** `MISSING_REQUIRED` is now actually
  emitted; `sanitize_string` output can no longer exceed `max_length`
  nor split an escape entity; invalid sensitivity values no longer leak
  into `sanitized_data`; multivariate Inf warnings now match the
  univariate path; `BenchmarkEvaluator.evaluate()` raises a domain
  error on zero usable samples.
- **`scripts/` mypy debt cleared (42 → 0 across 9 files)** with real
  typing fixes only; the ci.yml scripts lane now lists every `scripts/`
  file, retiring the pre-existing-errors carve-out.
- **Coverage uplift** (measured, full-suite baseline 68.05% on 12,328
  passing tests): `calibration_pipeline` 47→99.5%, `point_adjustment`
  72→100%, `api_validators` 61→100%, `benchmark_evaluator` 55→100%,
  `core/di` 0→94%, `cross_domain_frequency` 0→98%, `migrate_pkl`
  11→100% (adversarial pickle payloads proven inert), three compat
  shims 0→100%, plus the two σ semantic suites.
- **Cross-cutting.** The advisory `Signature.public_key_hash` format is
  hoisted to the single `crypto_api.key_fingerprint` helper;
  `parse_corpus` no longer redundantly re-copies locally-built arrays;
  README codebase-scale block regenerated via its gate.

### Security: air-gap sovereignty — every egress path fails closed under `MERCURY_OFFLINE`, loopback stays open

- **Goal.** Mercury must stay fully operable when *all* external sources are
  cut off (network outage, mirror loss, energy/disaster conditions). A single
  `MERCURY_OFFLINE=1` switch now closes **every** outbound path before a socket
  opens, while explicitly keeping the **loopback** path open so an on-box model
  (a local Ollama on `127.0.0.1`) keeps executing actions. Detection, the
  ethical gates, the decision layer, PQC, and local/template reasoning were
  already network-free; this pass closes the optional-enrichment egress paths
  that previously bypassed the switch.
- **Three egress points hardened, all fail-closed before the socket:**
  - `SafeHTTPClient.validate_url` (the shared egress gate for narrative
    retrieval, integrations, medical/geological loaders, and the Ollama
    adapter) refuses every non-loopback destination **before any DNS
    resolution** — the guarantee holds even where no resolver is reachable —
    and permits only loopback targets (`127.0.0.1`, `::1`, `localhost`).
  - `DataSourceBase._http_get` / `_http_get_sync` (the live data-source httpx
    transport, which had its own transport outside the shared gate) now refuses
    before opening a socket, raising `OfflineModeError` with a remediation hint.
  - `SafeHFLoader.load_model` sets `local_files_only` under offline so a cached
    foundation/VLM load succeeds and an uncached one fails fast with a clear
    error instead of hanging on a dead network.
- **Five own-transport egress paths closed (review follow-up).** A review sweep
  found five callsites whose transports run outside both the dataset
  chokepoint and `SafeHTTPClient`; each now applies the same air-gap policy
  via the new shared `security.safe_http.enforce_offline_egress` gate
  (non-loopback refused pre-DNS/pre-socket, loopback stays open):
  - `MITBIHLoader.download()` — wfdb fetches PhysioNet with its own
    `requests` transport; an uncached download now refuses offline (and a
    primed cache is served before the wfdb import, so cached air-gapped use
    no longer even requires wfdb).
  - `cognitive.anomaly_detection_enhanced.USGSEarthquakeSource.fetch()` /
    `NOAAWeatherSource.fetch()` — the ad-hoc httpx clients refuse before any
    socket (their catch-all handlers swallow transport errors only), and the
    production consumer `ExternalDataIntegrator.fetch_all` surfaces the
    air-gap as one explicit MERCURY_OFFLINE skip — never a generic per-source
    fetch error — while detection continues local-only.
  - `NISTCSFReferenceFetcher.fetch_payload()` — a fresh cached reference is
    still served offline (unless `force_refresh=True` bypasses the cache);
    an actual fetch (including at
    `NISTCSFIntegrator(reference_source="live")` construction) refuses
    pre-socket with a pointer to `reference_source="builtin"`.
  - `api/routes/batch.py` — a request carrying `callback_url` is rejected at
    validation time, before the SSRF check would resolve the hostname (DNS
    is itself egress); `_send_callback` additionally suppresses any
    already-queued webhook (logged skip, per the "failures never escape"
    background-task contract) before httpx or any socket is touched.
  - `integrations` — the publicly exported `HTTPClient` gates every request
    before its circuit breaker / retry / aiohttp (or stub) layers, and every
    `cross_platform_hub` adapter transport (HTTP platform, Prometheus
    pushgateway, OTLP HTTP) gates each connect/send/flush/fetch, keeping
    loopback sidecars reachable; `CrossPlatformHub.publish_*` records a
    refusal as an explicit per-platform MERCURY_OFFLINE suppression, and
    `PrometheusAdapter.disconnect` never raises from its final flush
    (buffered metrics are retained, the disconnected transition completes).
- **Three more ungated paths found by the adversarial sweep, closed the same
  way:** `utils/report_generator.py::EmailReportSender` (SMTP egress to a
  non-loopback relay refused pre-socket, logged suppression, boolean
  contract preserved); the Ollama adapter's raw TCP availability probe
  (now runs the adapter's full SafeHTTPClient egress policy first —
  loopback-only by default, RFC1918 under the VPC-air-gap opt-in — so a
  non-permitted host reads unavailable with no socket); and
  `integrations/stubs/cache.py::RedisCache` (a non-loopback `REDIS_HOST`
  refused pre-socket; callers fall back to the in-memory cache; the
  loopback sidecar stays reachable).
- **`*.localhost` subdomains are refused offline.** The DNS-free loopback
  carve-out is exactly `127.0.0.1`, `::1`, and the literal `localhost`:
  RFC 6761 only says resolvers SHOULD answer `*.localhost` locally, so a
  hosts-file entry or hostile resolver could map `foo.localhost` to a
  public address and turn the name-based permit into an egress bypass for
  callsites that hand the URL to their own transport. `_is_loopback_host`
  (used by both the `SafeHTTPClient` air-gap gate and
  `enforce_offline_egress`) no longer recognises the subdomain family.
- **Master switch now authoritative in narrative retrieval.** `WebSearchRetriever`
  honors `MERCURY_OFFLINE` even when constructed with `offline_mode=False`, so
  the env var alone forces fail-closed behavior system-wide.
- **Proven, not asserted.** `tests/security/test_offline_egress_gate.py` pins
  the whole contract (the suite grew from 13 to 43 pinned cases): external
  refused with zero `getaddrinfo` calls, loopback IP literals permitted with
  no resolution, `localhost` permitted while `*.localhost` subdomains refuse,
  the online path unaffected, both data-source transports refused before the
  socket, every review-surfaced and sweep-surfaced path refused pre-socket
  (with the cache-serving carve-outs of MIT-BIH and the NIST fetcher
  exercised offline), the integrator's explicit offline skip, the batch
  validator's pre-DNS rejection, the hub adapters' connect/flush/disconnect
  behaviour, and the SMTP / Ollama-probe / Redis suppressions.
  `docs/OFFLINE_OPERATION.md` is corrected to describe all egress points and
  the loopback carve-out accurately.

### Changed: supply-chain scalars reframed as Mercury-native (no external product identity)

- The 10 repository-integrity scalars were documented as "OpenSSF Scorecard"
  checks, which read as a dependency on someone else's product. They are — and
  always were — **handwritten Mercury code** that reads the repo's own
  configuration (workflows, `CODEOWNERS`, `dependabot`, `SECURITY` policy,
  SHA-pinning) with **no runtime dependency on any external scoring tool or
  service**. All docstrings, comments, the collector, the metrics artifact
  provenance, and the tests now say so. The frozen `omni_ossf_*` scalar keys
  and their σ_Immutable layout are **unchanged** — only the framing is
  corrected, so the safety-layout freeze holds and the values stay metric-only.

### Added: σ_Immutable trained on a real harvested config-integrity corpus (closes the synthetic-data gap)

- The σ_Immutable EthicalGate no longer trains on synthetic ``U[0,2]``
  vectors labelled by a threshold rule (which never occur in production).
  ``scripts/harvest_sigma_baseline.py`` harvests the **real intact
  operational vector** the running engine produces (127 governance
  scalars, constant across four domains) into
  ``security/sigma_immutable_baseline.json``;
  ``sigma_immutable_corpus.build_integrity_samples`` then builds the
  training + signed audit corpus from it — positives are the real baseline
  plus intact variations holding the 24 critical ethical *anchors*
  at-or-above threshold, negatives are real tamper mutations (anchor
  collapse). Labelling now matches the deterministic floor's real
  24-anchor contract instead of the earlier all-27-ethical rule that
  mislabelled the real (narrative-low) configuration.
- **No DoS, by construction + guard.** The exact harvested baseline is a
  training positive, so the retrained network recognises the real
  production vector by construction; a ``train_sigma_immutable.py``
  ``main()`` DoS guard asserts ``score(baseline) ≥ threshold`` and refuses
  to ship otherwise (measured **0.999922**). The surfaced
  ``sigma_immutable_score`` stays bit-constant across normal+anomalous
  inputs and all domains — the config-integrity posture is preserved — at
  the new constant **0.9999216794967651** (was 0.972732842).
- **Advisory net now agrees with the floor.** Measured alone, the old
  synthetic network *passed* a benevolence-zeroed vector; the
  harvested-corpus network **refuses** it (score 0.0), closing the
  false-assurance gap — while the deterministic critical-ethical floor
  remains the authoritative gate, composed first, and the operational
  scalar layout stays frozen (unchanged). Weights, signed corpus (Ed25519
  + ML-DSA-65), registry, KAT, discrimination probe, and the ledger are
  re-based together; the σ suite (KAT / discrimination / fail-closed /
  PQC / decision-channel / hard-enforcement) is green.

### Added: held-out generalisation record + merit gate for the real-text ethical gate

- The weapons-uplift Axis-B confidence logistic is already fit on the real
  362-case labelled corpus (``fit_weapons_gate_calibration.py``); the fit
  now also evaluates the **untouched held-out TEST split** (66 cases, never
  used in the fit or model selection) and records its generalisation in
  ``configs/weapons_gate_calibration.json`` — measured **test_fp=0,
  test_fn=0, test_brier=0.0023** (better than VAL). New
  ``tests/ethical/test_weapons_gate_calibration.py`` pins config freshness
  (recorded metrics match a live recomputation) and the held-out budget
  (zero fp/fn, Brier < 0.05), complementing the existing disposition-level
  fp/fn gate. Fitted parameters are unchanged (deterministic fit); this only
  adds the honest out-of-sample measurement and its merit gate.

### Security: SHA-pin all external GitHub Actions (supply-chain hardening)

- The self-referential supply-chain collector surfaced that only ~0.5% of
  ``uses:`` Action refs were SHA-pinned (tag-pinned instead — the exact
  finding the repo's ``check_workflow_hardening.py`` warned on). All **201**
  external Action refs across the repo's **21** workflows are now pinned to
  the commit SHA their tag currently points to (200 newly pinned here across
  20 workflow files; the single ``dependabot/fetch-metadata`` ref in
  ``dependabot-auto-merge.yml`` was already SHA-pinned), with a
  ``# <version>`` trailer so Dependabot can still bump them — **same Action
  code runs, zero CI behaviour change**, just supply-chain hardening. Local
  (``./``) composite actions are exempt.
  ``check_workflow_hardening.py`` now emits zero tag-pin warnings; the
  ``omni_ossf_pinned_dependencies`` scalar rose from 1.02 to 1.98.

### Added: real self-referential collectors for 36 of the 82 diagnostic SW-eng scalars

- The GOSNN ``SOFTWARE_ENGINEERING`` group's 82 diagnostic (metric-only)
  scalars were static placeholder literals ("not computed from any analyzed
  code"). ``scripts/collect_sw_eng_metrics.py`` now computes **real**
  measurements of Mercury's own source tree — the Halstead suite (7),
  cyclomatic complexity (1) and Maintainability Index (3) via stdlib ``ast``;
  the 10 Mercury-native supply-chain / repository-integrity checks from real
  repo config; the 4 DORA metrics as VCS-history proxies from ``git log``;
  the 4 NIST SSDF practice groups and 4 SLSA build-track measures from repo
  state; and the 3 computable NIST SAMATE assurance measures — into
  ``core/sw_eng_metrics.json``; ``_apply_measured_sw_eng_metrics`` overlays
  them at GOSNN init. Values are mapped into each scalar's penalty/positive
  direction band, so they now vary with the real measurement while keeping the
  quality-model semantics (e.g. the repo's near-zero SHA-pinning of Actions
  honestly read at the low end of its positive band until the pinning pass
  below).
- **Safety invariant preserved and pinned.** The overlay updates existing keys
  only — the SOFTWARE_ENGINEERING count stays 127, the scalars stay metric-only
  (filtered from the σ_Immutable operational vector), and **the σ score is
  unchanged** by the measured values (verified). The remaining 46 scalars (the
  31 ISO 25010 quality characteristics, the 7 non-computable NIST SAMATE
  measures, the 4 ISO/IEC 5055 assurance measures, and the
  essential/design/cognitive/npath complexity variants) stay documented
  placeholders — computing them would be fabrication or needs external
  telemetry. Merit-gated by ``tests/test_sw_eng_metrics_collector.py`` (real
  values, sane bands, metric-only, σ untouched).

### Changed: dependency bumps

- **mypy `2.1.0` → `2.3.0`** and **types-requests `2.33.0.20260518` →
  `2.33.0.20260712`**. The repo pins these tools on four parity-checked
  surfaces (`scripts/check_pinned_tool_versions.py` fails on drift), so all
  four were bumped in lockstep — `pyproject.toml` `[ml]` + `[dev]`,
  `.github/workflows/ci.yml` (Code Quality + Type Checking install lines),
  and `.pre-commit-config.yaml` (`mirrors-mypy` rev). mypy 2.3.0 keeps the
  `# type: ignore` set byte-identical: both lanes clean (`mypy src/` 748
  files, `mypy tests/` 644 files), no `unused-ignore` fallout on
  `datasets/mitbih.py:105/109`. Parity check + self-test green.
- **redis Helm subchart `27.0.13` → `27.0.15`** (helm-charts group) in `helm/mercury-agent/Chart.yaml`. Patch-level Bitnami
  bump; infra-only, no Python runtime surface.

### Changed: GitHub Actions group bump — 7 updates consolidated

- **The full `github-actions` group, applied as SHA pins** — every new SHA
  independently re-resolved from its upstream repo via `git ls-remote` before adoption:
  - `actions/setup-python` **v6 → v7.0.0**
    (`5fda3b95a4ea91299a34e894583c3862153e4b97`, 38 `uses:` sites);
  - `actions/setup-go` **v6 → v7.0.0**
    (`b7ad1dad31e06c5925ef5d2fc7ad053ef454303e`, 1 site — `ci.yml`
    actionlint job);
  - `actions/checkout` **v7.0.0 → v7.0.1**
    (`3d3c42e5aac5ba805825da76410c181273ba90b1`, 104 sites);
  - `github/codeql-action/{init,autobuild,analyze}` **v4.32.2 → v4.37.1**
    (`7188fc363630916deb702c7fdcf4e481b751f97a`, 3 sites in
    `security.yml`);
  - `pypa/gh-action-pypi-publish` **release/v1 → v1.14.1 head**
    (`ba38be9e461d3875417946c167d0b5f3d385a247`, 1 site in
    `release.yml`).
- **Compatibility audited before the bump:** setup-python / setup-go v7
  are the ESM / Node 24 migration — no workflow uses the `pip-install`
  input that setup-python v7 removed, every runner is GitHub-hosted
  (`ubuntu-*`), and all inputs actually used (`python-version`, `cache`,
  `go-version`) are unchanged in v7; checkout v7.0.1 and codeql-action
  v4.37.1 are patch/minor updates within the already-running majors;
  pypi-publish stays on the `release/v1` track. Verified with
  `actionlint` v1.7.12 (CI flags), a YAML parse of all 21 workflows,
  `check_workflow_hardening.py`, and `check_pinned_tool_versions.py` —
  all green.

### Fixed: flaky weapons-gate property test (pre-existing, test-only)

- ``tests/ethical/test_weapons_gate_properties.py::
  test_offensive_core_blocks_under_benign_noise`` wrapped an offensive
  core in Hypothesis-generated letters/numbers/spaces and asserted it
  still blocks. When the noise happened to spell a defensive OBJECT
  (``detector`` / ``sensor`` / ``vaccine``, or the leetspeak ``DETECT0R``
  that normalizes to ``detector``), the composed text became "how to make
  a nerve agent detector" — which legitimately triggers the gate's
  documented defensive-production carve-out (``ALLOW_PROVENANCE``,
  sourced-only, no synthesis) and is the exact allow-case
  ``test_defensive_production_is_not_blocked`` (line 47) requires must NOT
  block. The property was self-contradictory and flaked red whenever the
  random noise spelled one of those tokens (``.hypothesis/`` is gitignored
  with no derandomize profile, so CI explores fresh each run; the merged
  head passed only by never generating the token). Root cause is the
  over-broad test, not the gate — the gate still blocks the core under
  genuinely benign noise (``"...nerve agent xyz"`` → ``REFUSE_REDACT``,
  verified) and the carve-out is pinned in both directions. Fixed
  test-only, zero gate change: ``assume()`` skips examples whose composed
  text hits the gate's own ``_gate_evidence(...).defensive_carveout``, so
  the guard stays in lockstep with the carve-out lexicon and the safety
  property is preserved exactly.

### Fixed: doc drift left behind by the coverage-floor graduation

- A re-audit of the M1–M15 completion pass found nine doc locations still
  quoting the superseded ``CORE 25 / FULL 50`` coverage floors as
  current after ``ci.yml`` graduated to ``CORE 30 / FULL 55``
  (README ×5, ARCHITECTURE ×3, ``docs/DEPLOYMENT.md``,
  ``docs/ROADMAP.md`` floor table, ``.coveragerc`` comment). All now
  quote the live gates; ROADMAP supersedes rather than overwrites the
  v1.7.x baselines to preserve measurement provenance. The new
  ``tests/docs/test_coverage_floor_docs.py`` gate pins every site to
  the ``ci.yml`` env values (red-to-green proven: 16/21 sites fail on
  the unfixed docs), so the next graduation cannot drift silently.
- Cognitive-layer counts refreshed to measurement: ``cognitive/`` is
  30 modules (excluding ``__init__.py``) / ~32,000 LOC at HEAD — the
  stale "23 modules / 30,073 LOC" predated the four modules
  added (README ×2, package docstring). ``truth_decipher.py``'s class
  docstring said "4-phase pipeline" while its module docstring, the
  Phase 1–5 dataclass comments, and the framework test
  (``phase_completed == 5``) all say five — corrected to five.
- Mission-framing stragglers (M5 follow-through): ``docs/conf.py``'s
  fallback index template still emitted the retired "orchestration /
  cognition layer of the FINDΩYOU stack" framing (dormant while
  ``docs/index.md`` exists, but CI's sphinx-build resurrects it in any
  tree lacking that file); README's security-disclosure block carried
  the tree's only "people-first mission" text. Both re-framed to the
  canonical Civilization-First / sibling-platform wording from
  ``docs/index.md``. The compliance subagent's BIPA-002 remediation no
  longer prescribes sibling-platform policy; it states the statutory
  rule (destroy on purpose satisfaction or within 3 years of last
  interaction, whichever is first). Provenance/attribution references
  (subagent ports, blueprint attributions, CHANGELOG history, research
  citations, ``FIND-YOU-ARC-CODE`` contract docs) are deliberately
  untouched.
- ``docs/DORMANCY_LEDGER.md``'s "Last updated" header aligned with its
  own latest amendment date (2026-07-18).

### Fixed: NumPy-2.0 crash in `harmonics/transform.py` + stale Legendre cache

- ``np.math`` was removed in NumPy 2.0 (the project floor is
  ``numpy>=2.4``), so every ortho-normalized
  ``AssociatedLegendre.compute()`` — and with it every
  ``SphericalHarmonicTransform`` / ``FastSHTransform`` forward pass —
  raised ``AttributeError``. Reproduced on numpy 2.4.6, fixed with the
  stdlib ``math.factorial`` (bit-identical on both NumPy majors), pinned
  red-to-green by ``tests/test_harmonics_transform.py``.
- Adjacent defect: the Legendre P_l^m cache was keyed only by
  ``(degree, m)``, so reusing an instance on a different grid crashed
  (different length) or silently returned the previous grid's values
  (same length). The cache now invalidates when its grid changes; a
  reused transform is pinned bit-equal to a fresh instance.

### Behaviour change: decision layer enabled at the served deploy entrypoints

- The three served entrypoints — the ``/api/v1/detect/flagship`` HTTP
  route, the MCP ``mercury_detect_fusion`` tool, and
  ``mercury-agent detect -d fusion`` — now enable the
  decision/abstention/response layer when they build their engine, so
  every served flagship detection carries an additive ``decision`` record
  (grounded verdict or explicit abstention plus a bounded,
  non-destructive response plan). The core ``OmniMercuryEngine`` default
  is unchanged (opt-in via ``enable_decision_layer()``); library
  consumers are unaffected.
- New recommend-only restorative verbs join the bounded response
  vocabulary: ``ResponseAction.RECOMMEND_CONVERSION`` and
  ``RECOMMEND_RESTORATION``, selected only by the opt-in
  ``ResponsePolicy(restorative=True)`` posture (default ``False``; the
  pre-existing wire format is pinned unchanged by test). Both are
  non-destructive and never auto-authorised.

### Clarified: σ_Immutable threshold single source + hard/soft benevolence gates

- ``ETHICAL.SIGMA_IMMUTABLE_TRAINED_THRESHOLD`` (0.93) is the new single
  authoritative declaration of the trained σ_Immutable network's decision
  threshold in ``core/centralized_constants.py``; the gate module,
  ``EthicalGate``'s default, GOSNN's class attributes, and the env-clamp
  floor now cite it instead of re-hardcoding literals. All values are
  byte-identical to before (0.93 trained / 0.96 GOSNN gating — a
  documented two-threshold design, not drift); no behavior change.
- The "sigmoid benevolence gate" docs no longer claim to *replace* the
  hard threshold: ``BenevolenceScorer.enforce`` (0.99, floored 0.70) is
  the HARD decision-boundary gate; ``sigmoid_benevolence_gate`` is the
  SOFT fusion-weighting term in ``core/three_r/fusion.py``. Comments and
  docstrings only; enforcement semantics identical.

### Changed: coverage merge-gate floors graduated on re-measurement

- ``COVERAGE_THRESHOLD_CORE`` 25 → **30** and ``COVERAGE_THRESHOLD_FULL``
  50 → **55**, backed by fresh CI-identical measurements on the
  head (CORE 37.94% / FULL 67.97% combined stmt+branch, both lanes
  green at measurement; previous baselines 31.87% / 59.84%). Same
  cushion policy that set 25/50 (~7 points CORE / ~10 points FULL below
  baseline), with ~3 extra points of FULL margin because the
  measurement environment carried slightly fewer optional extras than
  CI's ``[all,dev]`` lane. CONTRIBUTING.md and the ci.yml rationale
  comment updated together.

### Behaviour change: `MatrixProfileAdapter.find_discords` default input interpretation

- With ``is_matrix_profile=None`` (the default), ``find_discords`` now
  ALWAYS treats the input as a raw series and computes its matrix profile.
  The previous length heuristic (``len <= 2*window`` => pre-computed
  profile) misclassified realistic pre-computed matrix profiles (length
  ``~ n - window + 1``, typically ``>> 2*window``) as raw series and
  silently computed a profile-of-a-profile, yielding incorrect discord
  indices/scores. Downstream callers that pass a pre-computed profile must
  declare it with ``is_matrix_profile=True`` (the in-repo caller already
  does). Noted in the method docstring.

### GOSNN role resolved: one scalar snapshot for both gates; observability audited

- **The σ_Immutable path no longer collects the scalar vector twice per
  detect call.** ``detect_with_fusion`` used to walk the 127-scalar
  operational registry once for GOSNN's advisory ethical gate (inside
  ``get_enhanced_scalars``) and again for the authoritative σ_Immutable
  gate — two independent snapshots that a concurrent registration could
  make diverge. The enhancement now carries the exact snapshot it scored
  (``EnhancementResult.collected_scalars``) and the authoritative gate
  reuses it: one registry walk instead of two (~0.3 ms/call), and the
  advisory and authoritative gates provably evaluate the identical
  vector. The σ_Immutable score and verdict are bit-identical (pinned by
  ``tests/core/test_gosnn_scalar_snapshot_reuse.py``).
- **GOSNN observability audited and kept (not theatre).** Measured on 120
  real detect calls, the surfaced ``enhancement_fusion_score`` (AUC-vs-
  label 0.878) and ``intelligence_contribution`` (0.836) genuinely track
  anomalousness and are only weakly correlated with ``anomaly_prob``, so —
  unlike the retired ``harmonic_synergy`` dead constant — they remain
  surfaced. The ledger records the honest caveat that this signal is
  member-0-derived (detector scores), hence informative but not
  independent of the engine's own fusion, and correctly never routed into
  a decision.

### GOSNN fusion: consequential decision channel + labelled-outcome training

- **`MultiHeadAttentionFusion` train/serve stacks unified.** The
  architecture and forward computation now live in one shared module
  (``core/attention_fusion_stack.py``) imported by both the production
  ``fuse()`` and the training program — the trainer's private
  ``_FusionModel`` re-implementation had already drifted from the serve
  path (the serve side applied a harmonic-synergy output modulation the
  merit gate never measured; it was a constant ×1.0118 on production data
  and is removed).
- **`harmonic_synergy` no longer surfaced in detection results.** Measured
  on the production serve path, the FFT top-two-magnitude ratio behind it
  is pinned to exactly 1.0 by real-input conjugate symmetry, so the metric
  was the bit-identical constant 0.618034 on every call. It remains a
  documented diagnostic (``GlobalOmniScalarNetwork.last_harmonic_synergy``)
  with the degeneracy spelled out in ``compute_harmonic_synergy``.
- **Consequential channel (decision-layer routing).** The fusion checkpoint
  payload may now carry a detection head + validation-selected thresholds
  (shipped only when the new detection-metric merit gate passes);
  ``detect_with_fusion`` surfaces ``gosnn_metadata["detection"]`` and the
  decision layer demotes a grounded verdict to ``DEFER`` on strong
  disagreement (``DecisionPolicy.defer_on_gosnn_disagreement``,
  abstention-only). The channel is isolated by construction and by test:
  ``OmniFusionModel`` inputs, ``anomaly_prob``, and the σ_Immutable verdict
  are bit-identical with the head attached or absent, so the shipped
  ``default_fusion.pt`` contract is untouched.
- **Training rebuilt around the real task.** ``scripts/train_gosnn_fusion.py``
  now pairs every harvested production ``fuse()`` input with the ADBench
  ground-truth label of the exact detected row and trains fusion + head
  jointly (class-weighted BCE; optional reconstruction auxiliary). The
  merit gate is a held-out detection metric — the learned head must beat
  BOTH the phi-reference-fusion head and the raw mean detector score by a
  margin, avoid representational collapse, and yield a demotion rule that
  is enriched in wrong verdicts at bounded deferral cost — and it refuses
  to ship anything consequential otherwise (harvest-diversity tripwire
  retained).
- **Gate hardened and verdict measured decisively.** The harvest draws
  label-stratified rows from each dataset's held-out TEST+VALIDATION
  splits across all cached ADBench datasets (1,469 labelled calls / 321
  anomalies; prevalence-invariant AUC, sampling disclosed); training uses
  gradient accumulation, a cosine LR schedule, and validation-AUC early
  stopping; shipping requires the 5th percentile of a dataset-stratified
  paired bootstrap of ``AUC_learned − max(baselines)`` to exceed zero; a
  detection head with out-of-range or inverted demotion thresholds is
  refused at load (and the Evidence layer independently drops out-of-range
  thresholds). The merit baselines now include the **engine's own
  ``anomaly_prob``** — the verdict a disagreement overlay would
  second-guess — because a head that separates worse than that verdict
  cannot help (it would demote net-correct verdicts). **Verdict: refused
  decisively** — the engine's own fusion separates the held-out anomalies
  at 0.961 AUC, while the learned fused-state head reaches 0.801 and even
  a direct head on the raw detector-score member reaches only 0.886–0.904
  (bootstrap delta vs the engine P5 −0.201). The engine already fuses the
  detector scores better than any re-weighting of them, so a GOSNN-input
  consequential channel is dead-by-construction; observability-only is the
  ratified, evidence-backed posture (``artifacts/gosnn_fusion.eval.json``,
  dormancy ledger). Closing the loop also fixed a real hole: the earlier
  gate compared only against the phi/mean reference fusions, which the
  detector-score head clears — it would have shipped a cross-check weaker
  than the engine it guards.

### Hazard/geophysical detectors: `(1, W)` batched-single-sample crash class swept

- The solar geomag fix generalised: an audit of every net-backed trained
  serving lane reproduced the same crash class in landslide
  (``slope_features`` — ``ndim >= 1`` guard passed ``(1, 64)`` into
  ``BatchNorm1d``), volcanic eruption (``fused_features`` — no input guard
  at all), and hurricane (``(1, W)`` wind transect pooled to height 0 in
  ``MaxPool2d``), plus adjacent-rank holes in wildfire (``(3, 1, W)``),
  seismic ``predict_earthquake`` (0-d / rank>=3 opaque crashes),
  parapsychology (0-d ``len()`` crash; off-contract ``(N, W)`` silently
  flattened into one LSTM sequence), and a latent unguarded feed in the
  disaster-precursor lane. Uniform contract everywhere: ``(1, W)`` is the
  single sample and is consumed; off-rank input falls to the disclosed
  physics/neutral path or fails loud where no physics fallback exists.
  Each fix is pinned by ``tests/detectors/test_batched_single_sample_guards.py``.

### Tooling: `ruff check` no longer auto-mutates

- ``[tool.ruff] fix`` is now ``false``: a plain ``ruff check`` used to
  rewrite files, so a file could change *after* a passing mypy run and ship
  a type regression (bit CI on 498f8d4). Autofixing is explicit-only
  (``ruff check --fix`` or the pre-commit hook, which blocks the commit on
  fix so later hooks re-run); ``scripts/run_ci_gates.sh`` orders the mypy
  lanes after ruff.

### Product identity + uniform per-provider model selection

- **Mercury Agent identifies as Mercury Agent.** A product-identity clause
  (``MERCURY_IDENTITY_CLAUSE``) now rides every system prompt Mercury sends
  to an LLM backend — the reasoning chain's ``SYSTEM_PROMPT`` and both
  adapter prompt builders: user-facing language speaks as Mercury Agent,
  never under the backend model's or its vendor's own persona, while
  staying transparent on inquiry (Mercury Agent is AI-assisted; the active
  operator-configured provider is reported in provenance metadata; it never
  denies being AI or claims to be human). Pinned by tests.
- **Uniform ``MERCURY_<PROVIDER>_MODEL`` model selection.** Every adapter
  now reads its own env fallback for ``model_name`` (Anthropic, OpenAI,
  HuggingFace, xAI, DeepSeek, Cursor, Cohere, Gemini — mirroring the
  existing ``MERCURY_OLLAMA_MODEL``): explicit config wins, env is the
  fallback, and with neither the adapter stays unavailable. Same convention
  for every provider; none privileged. Pinned by a parametrized test.
- **Residual vendor defaults purged.** The CLI ``voice`` command's
  ``--model`` flag, ``ModelConfiguration``'s preferred/domain model lists,
  and the ``create_ollama_adapter``/``create_fallback_chain`` factories no
  longer fall back to a hard-coded local model; the former default lists
  are preserved verbatim as documented opt-in ``EXAMPLE_OLLAMA_*``
  constants.

### Provider neutrality: no vendor-default models, anywhere

- **Mercury ships no default model id for any provider — local or cloud.**
  Every adapter (OpenAI, Anthropic, HuggingFace, xAI, DeepSeek, Cursor,
  Cohere, Gemini, and local Ollama) previously fell back to a hard-coded
  vendor model when ``model_name`` was unset; all nine literals are removed.
  An adapter constructed without an explicit model now reports itself
  unavailable with an actionable message and the chain falls back to the
  deterministic template — nothing is ever silently sent to a model the
  operator did not name, and no vendor is privileged. (This also closes the
  historical failure where a hard-coded default rotted upstream and 404'd on
  the first call.) Pinned by a parametrized adapter test.
- **Model-integration proof harness de-branded.** ``anthropic_wire_proof.py``
  → ``cloud_adapter_wire_proof.py`` and ``mercury_claude_reasoning_e2e.py`` →
  ``remote_reasoning_e2e.py``; the recorded Anthropic Messages fixture id is
  now a single named test constant (test data, not a default), the live legs
  require the operator to name the model (``MERCURY_ANTHROPIC_MODEL``), and
  the README presents every shipped provider as equally welcome.

### Hardening: edge-window crashes, sizing consistency, honest docs

- **Schumann degenerate windows fixed end-to-end.** An empty ELF window
  reached ``scipy.fft.fft`` (which raises on empty input) before any guard,
  and a single-sample window was squeezed to a 0-d scalar and misclassified
  as off-modality; ``extract_features`` additionally emitted NaN band power
  (mean of an empty slice) on such windows. All three now yield the truthful
  empty spectrum / zero band power, warning-free, with ``n in {0, 1}``
  pinned by tests.
- **AffectiveAnomalyModel rejects empty declared series.** A declared
  emotion series with zero timesteps passed the shape check and produced
  NaN anomaly/emotion scores via the temporal mean; it now fails loud.
- **AllSourceFusionNetwork sizing follows the discipline enum.** The
  cross-INT attention head count and the ``hidden_3`` rounding hard-coded 13
  while the per-discipline encoders were built from ``IntelligenceDiscipline``;
  both now derive from the enum (identical dimensions at the default) and a
  divergent ``num_int_types`` fails loud instead of silently mis-sizing.
- **Proof-harness honesty and hygiene.** The wire proof no longer claims the
  SSRF gate was "engaged" in the leg where it is patched out; the live
  transport leg catches only ``UnsafeURLError`` (the sole exception the
  adapter lets escape) so real adapter bugs fail the proof; key-shaped
  ``sk-ant-*`` dummy values replaced with neutral placeholders; the detector
  sweep's exit-code contract is now stated precisely; ``curl | sh`` install
  one-liners replaced with links to official instructions.
- **`AMA_REQUIRE_CONSTANT_TIME` docs reconciled** (SECURITY.md previously
  told operators to set it while INSTALLATION/DEPLOYMENT/.env.example called
  it a no-op): all four now carry the same verified statement — AMA v3.3.0's
  native-only operation is unconditional, the flag changes no cryptographic
  behavior on a healthy install (AMA logs a deprecation warning), it fails
  closed redundantly on a broken install, and Mercury reads it only for
  diagnostics.

### Merit gates enforced in both directions: seismic winner serves, GOSNN fusion refuses

- **EarthquakeDetector defaults to the shipped merit-gate winner.** The
  ``seismic_stead`` checkpoint (real STEAD data: held-out recall 0.9745 vs
  physics 0.4795 at the deployed alert rules, learned FAR <= physics, AUC
  +0.072, and faster than physics — see its provenance sidecar) previously
  shipped but never served without an explicit operator call. Default
  construction now auto-loads it (fail-open to STA/LTA physics when the
  checkpoint is absent; invalid checkpoints still fail loud), and
  ``load_shipped_weights=False`` pins the physics configuration. The hazard
  regression guard gained a trained lane (``trained_pod/far/csi`` pinned
  alongside the unchanged physics lane, with capability tripwires in both
  directions) and its baseline was re-pinned; the trained lane's low POD on
  the guard's synthetic toy waveforms (~0.14) is pinned deliberately as an
  out-of-distribution characterization, with the real-data surface cited.
- **MultiHeadAttentionFusion: non-circular training program, honest refusal.**
  ``scripts/train_gosnn_fusion.py`` records the dimensional-state lists the
  production ``fuse()`` actually receives while a real engine detects on
  real cached ADBench windows and merit-gates masked-member-reconstruction
  candidates against the deterministic phi-weighted reference. A forensic
  audit of the first run found the harvest DEGENERATE — 1 unique state list
  across 403 calls (static registry groups; empty per-call base member) and
  a measured downstream effect on ``anomaly_prob`` of exactly 0 — so the
  briefly-shipped checkpoint was withdrawn as a vacuous memoriser. The
  program now measures harvest diversity first and refuses to train or ship
  below 50 unique lists; the committed
  ``artifacts/gosnn_fusion.eval.json`` carries the measured refusal
  verdict, ``fuse()`` keeps the deterministic reference, and a new test
  pins that the artifact's decision and the shipped-checkpoint state can
  never disagree.

### Follow-through: operational VLM backends, shipped model pins, honest failure semantics

- **Local VLM backends shipped (BSD-3-Clause BLIP), validated live.** New
  ``blip_vqa`` (Salesforce/blip-vqa-base) and ``blip_caption``
  (Salesforce/blip-image-captioning-base) backends — small, CPU-runnable,
  SHA-pinned — registered in the LVLM factory. LAVAD's caption stage now
  defaults to the real captioner instead of the hard-failing mock
  placeholder (explicitly configuring ``mock`` still hard-fails, pinned by
  test). The VQA backend scores the {yes, no} answer likelihood pair
  instead of free generation (the margin discriminates where the generated
  argmax is biased) and emits the model's own verdict probability. Driven
  live with real transformers forwards: AnyAnomaly's frame scores peak on
  an injected flash frame; LAVAD emits genuine prompt-free captions.
  Network-marked regression tests pin both drives.
- **AnyAnomaly outage honesty.** A backend exception previously mapped to
  confidence 0.0, and ``score = 1 - confidence`` marked every covered
  frame maximally anomalous — an outage silently alarmed the whole clip.
  Failed segments now contribute no score, and frames left with no real
  signal raise instead of being fabricated.
- **Chronos shipped revision pins.** The five built-in
  ``amazon/chronos-t5-*`` Hub IDs carry immutable in-repo SHA pins, so a
  default-constructed adapter (plain registry discovery) satisfies
  SafeHFLoader's mandatory-pin policy and is operational out of the box
  (validated live: 13x anomaly separation); an explicit revision always
  wins, and an explicit ``model_name`` is no longer clobbered by the
  default ``model_size``.
- **FoundationEnsemble.add_model instances survive lazy initialization**
  (previously rebuilt from config with default settings);
  ``IntelligenceFusionEngine.extract_features`` now emits its declared
  ``(1, 128)`` width on every path (the populated path emitted 117);
  stfpm/reverse_distillation LR schedulers step only for epochs that
  stepped the optimizer; the dead pre-audit ``_mock_matrix_profile``
  fallback is removed.
- **Measurement/tooling honesty:** the detector operational sweep gained a
  ``--serial`` mode (the concurrent pool inflates per-detector wall
  latencies ~100x on small hosts and is now documented as a throughput
  measurement); ``run_ci_gates.sh`` asserts the resolved mypy matches the
  pyproject pin (a full ``[all,dev]`` install was observed leaving a
  shadowing user-site mypy 1.19 whose off-pin runs fabricate type errors);
  the README/ARCHITECTURE latency tables are labelled as historical
  snapshots with the reproducible-path pointer
  (``mercury-agent tool detector_profiler``; sandbox re-run: median
  47.7 ms — the ``<100ms`` claim holds).

### Engineering audit: packaging, detector correctness, latency, torch-free cloud path

- **`pip install .[all]` / `.[test]` un-broken (lime dropped from the extras
  graph).** `lime==0.2.0.1` — its only release (2020, unmaintained) — fails to
  build under modern setuptools (`AttributeError: install_layout`), which made
  every `[explainability]`/`[all]`/`[test]` install fail. The extra now ships
  `shap` only; the `LIMEExplainer` adapter still uses the real library when an
  operator installs it manually and otherwise falls back to its in-repo
  `_approximate_lime` linear surrogate. `tests/test_explainability_extra.py`
  pins lime out of the extras graph. Validated: full `[all,dev]` install
  resolves and imports.
- **`[foundation]` made operational.** The extra now composes
  `mercury-agent-global[ml]` (every `models/foundation` adapter imports torch at
  module top, so the extra previously installed an unimportable feature) and
  declares `chronos-forecasting>=1.4` (previously undeclared, so the chronos
  detector could never run). Validated live: `amazon/chronos-t5-tiny` under a
  pinned-SHA `SafeHFLoader` load scores an injected anomaly block 36x above
  baseline (chronos-forecasting 2.3.1). Pinned by `tests/test_foundation_extra.py`.
- **`[all]` now reaches `[compliance]`.** A full install previously could not
  run the NIST CSF live-reference mode (`openpyxl` missing — observed as
  `ModuleNotFoundError` in the network-marked NIST tests, now green). Pinned
  by `tests/test_all_extra_contract.py`.
- **Heatwave detector fixed in the registry path.** `HeatwaveDetector.is_fitted`
  was a property while every registry/fusion call site invokes
  `detector.is_fitted()` (the `core/base.py` contract), so every registry
  invocation crashed with `'bool' object is not callable`. Now a method; the
  registry's plugin boundary additionally resolves both method- and
  property-style `is_fitted`. The detector now runs operationally on generic
  fusion windows (validated end to end, incl. imposed-excursion detection).
- **Schumann resonance detector: clean 1-D modality contract.** Unvalidated
  2-D input sent the FFT along the wrong axis and the fundamental/harmonic
  peak searches then indexed the 1-D frequency grid with a flattened-2-D
  `argmax` (`index 40 is out of bounds for axis 0 with size 9`). The power
  spectrum now validates single-channel input and declines multichannel data
  with an actionable message; short temporal-history windows zero-pad instead
  of breaking the 103-bin layout. On-modality behaviour validated (7.83 Hz
  fundamental recovered).
- **`[all]`-install segfault fixed (TensorFlow/triton LLVM collision).**
  Registry auto-discovery hard-crashed (SIGSEGV) whenever deepface pulled
  TensorFlow into the process before a torchvision-backed detector reached
  `torch._dynamo`'s triton probe — both bundle an LLVM and the TF-then-triton
  order is fatal (tensorflow 2.21 + triton 3.7.1; reverse order is safe).
  `_compat.preload_triton_before_tensorflow()` now pins the load order at
  Mercury's TensorFlow entry points (`models/biometric*.py`). All 91 manifest
  detectors load with the full `[all]` stack; subprocess regression test in
  `tests/models/test_triton_tf_ordering.py`.
- **Detector latency (measured on a 200x8 window, same host, semantics
  preserved):** `gaussian_process` 8.6 s → 0.004 s (loop-invariant kernel
  algebra hoisted — the per-point solves rebuilt an identical 24x24 system);
  `srcnn` fit 26.8 s → 2.9 s and `diffusion_ad` fit 12.3 s → 0.1 s (tiny
  full-batch training loops now run with intra-op threading pinned to one
  thread via `detectors/_torch_perf.py`; the default pool's fork/join costs
  ~13x the arithmetic at these tensor sizes); `acceleration_dynamics` extract
  61.7 s → ~3 s (same cure for its per-sample LSTM); `physics_integrated`
  70.9 s → 32.1 s (`extract_features` no longer routes through `detect`,
  which ran every component twice; plus component fixes);
  `spectral_vibration` ~2x (phonon pairwise/scattering loops batched
  [~24k module calls → 3], MLIP encoder vectorised, spectral graph memoised,
  Laplacian eigensolve switched from worst-case `which="SM"` Lanczos to the
  PSD spectral-shift form without eigenvector extraction); `bocpd` +35%
  with bit-identical scores (loop-invariant `gammaln` table over the exact
  `alpha0 + r/2` grid). Full-registry sweep median latency 34 ms → 22 ms.
- **Cloud LLM path decoupled from the ML stack.** `llm_adapter.py` no longer
  imports torch at module top (the Tensor branch resolves via `sys.modules`;
  the local HF backend imports torch only inside `_load_model`),
  `models/foundation/__init__.py` uses lazy PEP 562 exports (house pattern),
  and `security/__init__.py` resolves its sole torch consumer
  (`IntelligenceFusionEngine`) lazily with the exact historical
  class-or-None contract. Proven in a torch-free venv: engine import (PQC
  gate), `AnthropicCloudAdapter`/`OpenAICloudAdapter` construction + prompt
  path, and `RemoteReasoningBackend` import all succeed with torch absent;
  `tests/models/test_cloud_llm_torch_free.py` pins that the chain never pulls
  torch even when installed.
- **`MultiHeadAttentionFusion` honest inference gate.** The untrained torch
  attention (random projection + averaging under `no_grad`) is no longer
  reachable: inference uses the deterministic phi-weighted reference average
  until genuine weights load via the new `load_trained_weights()` seam
  (EthicalGate convention). `tests/core/test_attention_fusion_gate.py`.
- **`AffectiveAnomalyModel` declared-modality path.** A dict carrying an
  emotion-probability time series under `"emotions"` is now analysed
  deterministically (temporal aggregation + the documented
  entropy/negative-affect distress heuristic); malformed declared input fails
  loud. Generic input keeps the neutral quarantine prior, and the fusion
  feature group stays constant zeros so the shipped `default_fusion.pt`
  contract is untouched.
- **Type/lint posture:** repo-wide `mypy src` lane green at the pinned 2.1.0
  (747 files); `triton` added to the third-party override list; numba
  `prange` sites carry dual-environment ignores. Bandit (CI configuration,
  medium+ severity): clean.

### Dependency & static-analysis hardening

- **OpenCV moved to the 5.x line.** `opencv-python-headless` is now pinned
  `>=4.8.0,<6`. OpenCV 5.0.0 (2026-07) relocated the Haar/HOG detectors —
  `cv2.CascadeClassifier` and the bundled `cv2.data.haarcascades` used by
  `models/biometric_advanced.py` — into `opencv_contrib`, so they are absent
  from the `-headless` 5.x wheel (at runtime and in the type stubs).
  `biometric_advanced.py` now feature-detects those symbols via `getattr` and
  degrades gracefully (the DNN face detector stays primary), keeping both the
  runtime path and the blocking `mypy` gate safe on the 5.x line; every other
  `cv2` symbol the code uses is unchanged. The 4.8 floor is retained as the
  baseline API the code targets.
- **Six high-severity CodeQL alerts resolved (#913–#918)** across
  `api/routes/models.py`, `core/global_omni_scalar_network.py`,
  `models/biometric_advanced.py`, and
  `space/ionospheric_scintillation_detector.py`.

### Static-analysis findings on the competitive-benchmark work fixed at the root

A pass over every open static-analysis finding on the competitive-benchmark
work, resolving each at the root (no debt-for-debt, no suppression, no silenced
check) and proving the fix with an offline, deterministic test.

- **Credential-check transport errors were opaque.**
  `scripts/verify_data_credentials.py` collapsed a transport failure — `_get()`
  returns `(0, <exception text>)` on DNS/TLS/timeout — to the useless report line
  `"HTTP 0"`, discarding the one detail that explains a credential-delivery
  failure in CI. A new `_fail_detail(status, body)` helper surfaces the transport
  error text when the status is `0`, and appends a short provider-body preview
  (typically the API's own error message) on a real HTTP error; every keyed
  checker (EIA/FRED/NASA/AlphaVantage/OpenWeatherMap) routes its non-200 path
  through it. **Because a `requests` transport error embeds the full request URL
  (whose query string carries the API key), the surfaced text is first scrubbed
  by a new `_redact()` helper** — the value of any credential-bearing query
  parameter (`api_key`/`apikey`/`appid`/`token`/`password`/`secret`/`key`) is
  replaced with `***`, so the fix cannot leak a secret into CI logs (honouring the
  module's "secret values are never printed" contract). `run()`'s docstring was
  also corrected to match its actual exit-code contract (1 on any failure, else
  0). New offline suite `tests/test_data_credentials_offline.py` (28 cases) drives
  each checker through a monkeypatched `_get` and asserts the transport reason
  reaches the report, `"HTTP 0"` never does, and the key is never echoed.
- **A broad `except` in the secret-wiring guard could mask a real regression.**
  `tests/test_secret_wiring.py`'s behavioural loader check caught every
  `RuntimeError` on engine import and skipped, so a non-PQC `RuntimeError` from a
  loader refactor would go green as an "offline skip". A new
  `_import_skip_reason()` predicate skips only the two genuine "environment not
  provisioned" cases — an optional/transitive `ImportError` or the native-crypto
  PQC gate (`AMA/PQC is mandatory …`, mirroring `tests/test_calibration_brief.py`)
  — and re-raises everything else (a non-PQC `RuntimeError`, a `NameError`) so a
  real regression fails loudly. Six unit tests (`TestImportSkipClassification`)
  pin the narrow contract; the stale module docstring that claimed an
  `importorskip` guard was corrected to describe the actual try/except policy.
- **Docstring / pydocstyle hygiene.** The four remaining `check_*` helpers and
  `run()` in `verify_data_credentials.py` gained one-line Google-style docstrings,
  making the module fully `pydocstyle --convention=google` clean.
- The remaining open findings were already fixed at the root in earlier commits
  (the `data/adbench_embeddings` cache key, the forked-child metrics
  reduction that keeps the full score vector off the IPC boundary, the
  `WildfireLoader` comment, the `"fail-softed"` typo, and the `git_commit`
  provenance-label clarification); this pass verified each against the current
  tree and resolved the threads. Validated end-to-end in a sandbox: the offline
  synthetic orchestration (tier + fusion + six PyOD baselines through the forked
  wall-clock-guarded cells) and the live-data competitive regression guard
  (`--check` PASS on the real 8-dataset ADBench subset).

### Independent verification pass — CI red-to-green + fail-loud hardening

An adversarial re-verification fixed every defect it surfaced at the root:

- **CI failures fixed.** 16 mypy errors across 10 files (Optional detector
  source fields, explicit re-exports via module `__all__`, unused ignores,
  `no-any-return` sites); 3 ruff findings (TC003, F841, B017 narrowed to
  `UnpicklingError`/`RuntimeError`); 10 bandit B614 findings — every new
  `torch.load` is now hard-pinned to `weights_only=True` (house convention),
  including the shipped-checkpoint loader which previously passed
  `weights_only=False` explicitly.
- **Findings resolved without weakening.** (1) A stale last-good
  posture evaluation no longer masks an actively failing evaluator:
  ``_reported_posture_level()`` reports ``UNKNOWN`` whenever consecutive
  evaluation failures exist, even when a prior success is cached
  (strictly more conservative; regression test added). (2) The JWT dev
  fallback key's lazy initialization is now lock-guarded (double-checked)
  so racing threads can never observe two different per-process keys
  (8-thread barrier test). (3) A concatenated docstring in
  ``detectors/directive.py`` and a split log literal in
  ``space/disaster_precursor_detector.py`` were repaired (no behavior
  change).
- **caplog isolation hardened for the full-suite lane.** The autouse
  conftest guard that restored ``omni_mercury_engine``'s ``propagate`` flag
  now also forces/restores the logger's *level* and *disabled* flag: a test
  leaving the package logger raised to ERROR silently dropped WARNING
  records before caplog's root handler, reproducing the order-dependent
  full-suite failures of the CICIDS synthetic-fallback warning test and
  the feedback-loop reason-code test under xdist scheduling (reproduced
  locally with a level-polluting probe, then proven fixed).
- **Label-provenance gate no longer audits test fixtures.**
  `datasets/label_provenance.py::discover_loaders()` now carries the same
  documented test-module exemption as its `loaders/` sibling: the F5
  cache-vs-split probe loader (a test-local fixture) tripped the full-suite
  ML lane because `__subclasses__` sweeps see any test module imported
  earlier in the same process. The probe also declares its manufactured
  labels transparently (`LABEL_SOURCE = "statistical"`).
- **Scenario regeneration test made environment-transparent.** numpy dispatches
  transcendental kernels by CPU capability, so seeded regeneration is not
  bit-stable across hardware (observed on CI). The slow-tier test now
  compares regenerated content exactly for integers/strings and to last-ulp
  float tolerance; the committed scenario files stay content-hash-pinned
  (the gate itself is unchanged and deterministic).
- **Fabricated all-clear paths removed.** `HeartMathGCMSSource` placeholder
  spectra are now labelled `simulated=True` at confidence 0.0 (refused by the
  seam without opt-in); `GeomageticCorrelator` reports `unknown` instead of
  inventing quiet Kp 3.0/Dst −20 defaults; `VolcanicEruptionDetector.detect_live`
  raises on an empty/name-filtered-out feed instead of reporting "normal";
  DONKI and CO-OPS clients propagate failures instead of returning silent
  empty successes; the USGS FDSN and NWPS parsers fail loud on payload
  contract drift; legacy demo trainers require an explicit
  `allow_synthetic_fallback=True` to touch synthetic data.
- **Network smoke lane semantics fixed.** New `SourceUnreachableError` /
  `FetchResult.unreachable` / `LiveDataError.unreachable` taxonomy: the
  weekly lane now skips ONLY on transport-level unavailability and FAILS on
  drift from a reachable service; `network-tests.yml` installs `[ml]` so the
  five `detect_live` round-trips actually run (they silently skipped on
  missing torch); the NWPS smoke passes the bbox the endpoint requires.
- **Shipped-checkpoint integrity.** `load_shipped_checkpoint` now verifies
  the artifact against the sidecar's pinned sha256 BEFORE deserializing and
  the provenance sidecars ship in wheels via package-data; tamper and
  merit-gate-refusal regression tests added.
- **Gate hygiene.** `ci/hazard-regression` is now the literal job name (the
  branch-protection context the docs reference); the PR trigger is unfiltered
  so the required check can never wedge on "Expected"; `check()`'s
  missing-domain violation branch is reachable (was dead behind a KeyError);
  the vacuous `kp_g_bucket_accuracy` floor is reported-not-gated with the
  measured degenerate-predictor numbers pinned in a test.
- **F10/F16 follow-through.** The AMA adapter drops non-finite scalars loudly
  (never clamps to 0.0); `reset_global_network` takes the construction lock;
  the GOSNN accessor survives racing a concurrent first direct construction;
  stream factories and `StreamingAnomalyPipeline` thread `broker=` through
  (it was silently swallowed, making the F16 isolation seam unreachable).
- **F5 follow-through.** Cache-key material is one canonical JSON document —
  the old underscore-joined prefix let distinct `(name, version)` pairs
  collide onto one cache file; upgrade note documented (one-time rebuild).
- **New recorded-real fixtures.** NASA DONKI FLR/GST (2026-07-09 capture)
  covering the previously untested corroboration path; the CO-OPS
  water-level fixture is now exercised; the JPL Sentry fixture's duplicated
  record was removed.
- Live-ingestion client catalog documented in `docs/DATASOURCES.md`;
  `.trivyignore` carries a time-boxed acceptance for CVE-2026-53615
  (unfixed Debian util-linux base-image CVE published 2026-07-09).

### Hazard transparency — physics from observations, not untrained networks

Every natural-hazard forecast surface was audited for fabricated signal.
Untrained neural networks (random weights, no labelled corpus) no longer
produce eruption probabilities, storm Kp forecasts, tsunami confidences,
tornado rotation values, or earthquake magnitudes. Each detector now runs a
deterministic physics core computed from the OBSERVED inputs, with a guarded
`load_neural_weights()` hook that flips to the learned path only when a real
checkpoint is loaded:

- **Volcanic** — noisy-OR evidence fusion over observed gas flux, InSAR
  deformation, thermal radiance and seismic statistics + median/MAD swarm
  detection (was: `randn(128)` through an untrained forecast net).
- **Space cluster** — Boyle-index geomagnetic storm physics with NOAA G-scale
  classification; earthquake-precursor magnitude is `None` until a trained
  model exists (was: untrained-net output × 9.0 presented as magnitude).
- **Tsunami + earthquake** — observed peak-vs-baseline wave analysis;
  unconditional STA/LTA triggering with real P/S arrival picks and S–P
  distance; magnitude transparently `undetermined`.
- **Tornado / wildfire / landslide / hurricane** — Doppler-couplet rotation,
  VIIRS/MODIS brightness-temperature ignition, geotechnical slope stability,
  and vorticity/max-wind from the previously ignored wind field; the dead
  uncomputed hurricane track/landfall fields were removed.

### Hazard live-data wiring (T1e)

One uniform, provenance-checked live-ingestion seam
(`data_sources/live_ingestion.py`) now connects every hazard detector to its
real client via dependency injection (default = fully offline):

- Earthquake/tsunami → USGS FDSN catalog; volcano → the **real USGS HANS
  public API** (the simulated `US_VOLCANOES` table that emitted fabricated
  all-clear alerts is gone, DEPRECATION §6.9); tornado/flood → NWS CAP alerts
  + NWPS river gauges; solar → NOAA SWPC + NASA DONKI; Schumann → BGS ELF
  client whose simulation is now explicitly labelled and refused without an
  `allow_simulated=True` opt-in; meteor → NASA NeoWs + new real JPL
  fireball/Sentry clients (`data_sources/jpl_ssd.py`) replacing the private
  in-detector HTTP loaders (DEPRECATION §6.8).
- The duplicate geological `SolarFlareDetector` (hand-authored per-HMM-state
  Kp/Dst lookup tables — invented indices presented as predictions) was
  removed; the canonical space-cluster class populates storm fields only from
  an observed planetary Kp with the documented NOAA G-scale / Loewe & Prölss
  (1997) Dst mapping (DEPRECATION §6.7).
- `FloodDetector`'s never-invoked untrained `TopographicRunoffPredictor` was
  removed; the physics runoff coefficient is the real path (DEPRECATION §6.10).
- Earthquake, tsunami, meteor and solar-flare detectors joined
  `DETECTOR_MANIFEST`. 90 offline-deterministic wiring tests on recorded
  fixtures across the three dedicated modules
  (`tests/test_live_wiring_sources.py`, `tests/test_live_wiring_space.py`,
  `tests/detectors/test_live_wiring_geological.py`) + a `network`-marked live
  smoke module (`tests/test_live_wiring_network.py`; 11 live paths verified
  green). The smoke lane skips only on transport-level unreachability
  (`FetchResult.unreachable` / `LiveDataError.unreachable`) and FAILS on
  payload or contract drift from a reachable service.

### Per-hazard skill metrics + regression gate (T2)

- `evaluation/hazard_metrics.py`: POD / FAR / CSI / frequency bias / HSS,
  lead-time, magnitude MAE/bias, haversine location error, VEI and ordinal
  alert-level accuracy, flare-class and Kp skill — literature-standard
  definitions with strict validation.
- `benchmarks/hazard_regression_guard.py` + committed
  `hazard_domain_baseline.json`: measured baselines with provenance (commit,
  seeds, scenario hashes) over seven guarded domains (tornado, flood,
  hurricane, earthquake, tsunami, volcano, solar) on committed scenario sets
  (real recorded SWPC/GOES windows + seeded detector-contract scenarios);
  floors derived as measured − margin; `--check` fails CI on regression via
  the new `ci/hazard-regression` lane
  (`.github/workflows/hazard-regression.yml`). Hurricane track error is
  explicitly excluded until a track model exists. Non-vacuous floor tests
  prove every floor sits strictly between the degenerate score and the
  measured baseline.

### First real hazard checkpoint: solar_storm_geomag (T5)

`ml/hazard_training/` + `scripts/train_hazard_checkpoints.py`: a
reproducible, merit-gated training pipeline for the eleven
`load_neural_weights()` hooks — seeded stages, sha256-pinned data caches,
strictly ordered by-year temporal splits, provenance sidecars, and a hard
merit gate (ship refuses unless the trained model beats the physics fallback
on held-out years through the public detector API). Category b/c hooks fail
loud with their exact real-data requirement (`--audit` prints the 11-hook
table; docs/HAZARD_CHECKPOINT_TRAINING.md).

**Shipped**: `solar_storm_geomag` — trained on 20 years of real NASA SPDF
OMNI2 hourly solar wind + observed planetary Kp (2005–2024, GFZ Kp
cross-checked). Held-out 2022–2024 (25,846 hours incl. the 2024-05 G5
superstorm): Kp MAE 0.574 vs physics 1.054 (−45%), G-bucket accuracy 97.6%
vs 94.8%, storm AUC 0.972 vs 0.845; the fixed-Kp5 recall/false-alarm
trade-off is recorded transparently in the provenance sidecar.
`SolarStormDetector.load_neural_weights()` now loads the shipped default
with train/serve feature parity.

### Fixed — deferred correctness items (F5, F16, F10)

- **F5** — `DatasetConfig.get_cache_key()` now includes `split_ratios`;
  changing the split can no longer silently reuse a stale `.npz` cache whose
  train/val/test split was baked with the old ratios. Behavioral regression
  test proves the second load re-splits.
- **F16** — in-memory stream state moved off the producer CLASS onto an
  explicit `InMemoryStreamBroker` (instance state, broker-scoped topics,
  default broker + reset seam). Two pipelines can no longer bleed messages
  through the class object, and tests no longer depend on remembering manual
  `.clear()` calls.
- **F10** — permanent GOSNN singleton product fix: the scalar registry is now
  lock-guarded with per-component ownership and `unregister_scalars()` /
  context-manager scoping; operational scalars feeding the σ_Immutable gate
  are validated (non-finite or grossly out-of-range values are excluded with
  a loud once-per-scalar warning, never clamped); the adapter's raw
  unix-timestamp posture scalar moved to the new `omni_diag_` metric-only
  channel — a real key rotation no longer collapses an unrelated
  `detect_with_fusion` to a spurious fail-closed σ=0.0; and re-constructing
  the singleton with materially different config raises loudly instead of
  silently ignoring the new configuration.

### Security & robustness hardening (real-backend audit pass)

A deliberate audit + fix pass built and tested end-to-end against the real AMA
Cryptography v3.3.0 native backend (KATs green). Each fix carries regression
tests; no behavior was suppressed or cosmetically patched.

**Security — cryptography**

- **Replaced a cryptographically broken "quantum-resistant" encryptor with real
  AMA ML-KEM-1024 + AES-256-GCM.** `security/encryption.py`'s
  `QuantumResistantEncryption` (exported as `SecureDataHandler`) used zero-noise
  LWE (secret key recoverable by Gaussian elimination), a repeating-XOR
  keystream, and a non-constant-time signature compare. It is now a proper
  KEM-DEM over `MercuryCrypto` (Kyber-1024 encapsulation + AES-256-GCM AEAD);
  `sign_data` uses native HMAC with constant-time verification. Public API
  preserved.
- **Replaced the hard-coded JWT dev fallback key with an ephemeral per-process
  key** (CWE-798). A deployment that forgot both `MERCURY_ENV` and
  `JWT_SECRET_KEY` signed admin tokens with a constant published in source.
  The fallback is now a per-process random key: unforgeable from source, and
  non-portable so a misconfigured multi-replica deploy fails visibly instead of
  silently accepting forged tokens.
- **PQC import gate now consults AMA's FIPS 140-3 power-on self-test.** The gate
  checked algorithm flags + version but not the POST verdict, so a build whose
  self-tests failed could still start. It now fails closed unless
  `ama_cryptography.check_operational()` reports OPERATIONAL.
- **`MercuryCrypto.verify_crypto_package`** exposed — Mercury could seal 6-layer
  AMA packages but had no way to verify them.
- **Audit hash-chain key derived from AMA HD key management** (`audit_sign`)
  instead of a per-process random key, so audit chains verify across
  workers/replicas/restarts when `AMA_MASTER_SEED` is set.
- **Adaptive-posture controller wired to real AMA key rotation** — it was
  constructed with `rotation_manager=None`, so a ROTATE_KEYS decision was a
  no-op; it now drives the real `KeyRotationManager`.

**Robustness — fail-loud / fail-closed**

- **Streaming pipeline no longer loses alerts on a failed publish**: a failed
  `send()` now vetoes the input-offset commit (at-least-once) and increments a
  visible `publish_failures` metric, instead of committing and dropping the
  detected anomaly. The Kafka `commit()` dependency import was hoisted so a
  missing `aiokafka` surfaces loudly rather than being logged as "commit failed".
- **Integration adapters (HTTP, OpenTelemetry) stop reporting connected/
  delivered when the transport dependency is missing** — they now fail loud and
  no longer leak a session on a failed health probe.
- **`RealTimeThreatDetector` fails closed when blind** — it no longer marks
  itself fitted with zero working sub-detectors, nor reports "LOW / no threat"
  when every sub-detector errors at inference.
- **Crypto posture reports `UNKNOWN` (not a falsely-reassuring `NOMINAL`) when
  the evaluator is failing**, with `posture_evaluation_healthy` /
  `posture_eval_consecutive_failures` surfaced.
- **A failed nano detection sub-score is excluded from the fusion blend** rather
  than averaged in as a spurious `0.0` that depressed the fused anomaly score.
- **A configured fail-closed `nan_policy` is honoured** even when
  `DetectionConfig.resolve` fails on an unrelated key (was silently reverting to
  the permissive env default).
- **Ground-truth labels are never fabricated** — `NetworkSecurityLoader` raises
  instead of returning an all-normal label vector when labels can't be derived.
- **Oracle checkpoint-restore failures are surfaced** (`restore_failed` in the
  Oracle metadata + WARNING) instead of leaving the detector silently scoring
  without its Oracle component.
- Several swallowed security-audit events and silent degradations were elevated
  from DEBUG to WARNING.

**API / CLI**

- **`mercury-agent detect --threshold` now governs the fusion decision** (it was
  parsed and ignored); out-of-range values are rejected and the statistical
  path reports it as inapplicable.
- **`GET /export/metrics` rejects an unsupported `format`** with 400 (valid:
  json, csv, jsonl) instead of silently returning JSON.

### Removed

- **`ml/ppo_trainer.py`** — dead in every supported install (no `stable_baselines3`
  extra, no `gymnasium`), broken even with SB3 hand-installed, and its surviving
  paths fabricated training results. No runtime caller, no tests. See
  DEPRECATION.md §6.6.

### Prepared as 2.1.0, never released; ships in 3.0.0

Never tagged or published. The "SemVer minor" reading below was wrong: this
work already removed 2.0.0 public names (DEPRECATION.md §7).

Feature-additive over 2.0.0 — the detector-tier surface below is new public
API, so this cut is a SemVer **minor**, not a patch. Release hygiene included
in this cut: `tests/infrastructure/test_streaming.py::test_commit_int_offset_commits_next`
now skips when `aiokafka` (the `[streaming]` extra) is absent — its int-offset
commit path imports `aiokafka.TopicPartition` at call time, so on a base
install the mocked commit was never awaited and the test failed spuriously;
and the `.coveragerc` comment naming the CI coverage floors was trued to the
current `ci.yml` values (CORE 25 / FULL 50, not 15 / 35).

### Independent verification pass — reproduced results + fail-loud hardening

An adversarial re-verification reproduced the two headline
research claims and fixed every defect it surfaced:

- **Both committed research results reproduce exactly** and their artifacts
  now carry provenance commits that exist in this repository (the previous
  artifacts recorded a dirty-tree commit hash that was never pushed):
  rule evolution held-out test F1 0.5439 vs consensus 0.4119 (+0.1320);
  counterfactual validation wachter/prototype/genetic 1.00 flip / 1.00
  verified minimality, dice/growing_spheres transparent 0.00 on the piecewise
  regime.
- **Wildfire hotspot count is spatial.** `fire_perimeter_km2` and
  `thermal_hotspots` now derive from the 2-D hotspot mask; the previous
  channel-summed exceedance count inflated the ground-area estimate up to
  C-fold for pixels hot in several bands (regression test with a
  3-band-hot block).
- **Counterfactual searches harden transparently.** New `NonFiniteScoreError`
  lets the detection seam's fail-loud score wrapper participate in the
  Wachter/DiCE finite infeasibility barrier (previously the barrier was
  unreachable through the seam: the wrapper's ValueError aborted the whole
  search); optimizer failures are logged with their exception type; the
  genetic generator's proximity term actually uses Gower distance when
  feature metadata is present (the docstring claimed it; the code always
  used L2). GENETIC and GOWER gained their first executing tests.
- **Real-data GA tests run in every offline lane** via the committed
  recorded-real ADBench Pima fixture (they previously skipped on any fresh
  checkout); the committed champion `evolved_rule_graph.json` is loaded and
  scored through the deployed `SymbolicConstraintModule` by tests.
- **T4 detectors are discoverable.** All fifteen T4 detectors and all four
  cascades joined `DETECTOR_MANIFEST` (hail, winter_storm, derecho,
  dust_storm, drought, heatwave, atmospheric_river, lightning, avalanche,
  rockfall, subsidence, cme_arrival, sep_storm, ionospheric_scintillation,
  gic, surge_flood_cascade, eq_tsunami_cascade, fire_debris_flow_cascade,
  solar_gic_cascade) — they were library-only,
  invisible to auto-discovery and every manifest-driven surface.
  `DroughtLoader`/`HeatwaveLoader` are exported from `loaders/__init__`.
- **Benchmark gate vacuity closed.** `run_all_benchmarks.py` now counts a
  crashed, data-less, or loader-missing GATED domain as a gate failure —
  a benchmark that never measured its AUC could previously exit 0.
- **Surface tests are unconditional.** The counterfactual HTTP tests
  deterministically disable the rate limiter (walking the built middleware
  chain) instead of escaping through `status in (200, 429)`; the 403
  ethical-gate branch asserts the refusal shape.
- The dead `flood` branch of `build_hazard_geojson` was removed transparently
  (flood emits no diagnostics payload; validation rejects it upstream);
  the EHF core gained Nairn & Fawcett worked-example tests; the surge
  cascade gained a recorded-real CO-OPS observed/predicted fixture test
  (The Battery, 2026-07-09); the orphaned mis-shaped
  `swpc_planetary_kp_recent.json` fixture was removed; two pre-existing
  lint errors (F821 `Path` in tests/test_cli.py, SIM117 in the hail loader
  tests) were fixed.
- New documentation: `docs/HAZARD_VISUALIZATION.md` (incl. the transparent
  Schumann 1-D-spectrum / no-track-cone / no-flood-GeoJSON scope notes) and
  `docs/COUNTERFACTUALS_AND_RULE_EVOLUTION.md` (reproduction commands).
- **Typed test debt cleared for real.** The tests-lane mypy debt on this
  branch (176 errors across 36 test files) was fixed preferring real
  annotations over ignores, and re-verified under *true-tree* package
  resolution (`MYPYPATH=src`, mirroring CI) — which surfaced and fixed 11
  further latent errors a stale editable install had masked.
  `AvalancheDetector.assess_new_snow_loading` / `assess_temperature_gradient`
  now return typed `TypedDict` contracts instead of `dict[str, object]`
  (dropping two `type: ignore[arg-type]` casts on the prediction path).

### Hazard visualization & diagnostics (T3)

The intermediate arrays hazard detectors compute were persisted (opt-in
`keep_diagnostics=True`) instead of discarded — earthquake spectrograms +
STA/LTA series, tsunami FFT spectra, tornado Doppler fields, hurricane
wind/vorticity fields, wildfire thermal/hotspot masks (with
`ignition_locations` / fire area now actually populated from the mask — the
declared-but-never-filled fields were a real bug), volcanic spectra, meteor
Doppler profiles, Schumann harmonic spectra. `detectors/hazard_visuals.py`
renders them deterministically (PNG via Agg, RFC 7946 GeoJSON for zonal
outputs), exposed as `mercury-agent hazard-viz`, `POST
/api/v1/hazard/visualize`, the `mercury_hazard_visualize` MCP tool, and new
hazard panels in the visualization dashboard. Hurricane track cones are NOT
rendered — no track model exists; only real computed fields are drawn.

### New hazard domains (T4): 15 detectors + 4 cascades + 5 loaders

All physics literature-anchored, untrained-network-free, with citations in
docstrings and worked-example tests:

- **Space weather**: CME arrival (Gopalswamy ESA + Vršnak drag-based model),
  SEP radiation storms (NOAA S-scale over GOES integral proton flux),
  ionospheric scintillation (real S4/σφ computation + labelled climatological
  risk), GIC-to-grid (dB/dt metrics + plane-wave geoelectric proxy), and the
  solar→GIC cascade (DONKI flare/CME → arrival window → Kp watch → measured
  dB/dt escalation state machine).
- **Meteorological**: drought (SPI/McKee, Thornthwaite PET + SPEI, USDM
  D0–D4), heatwave (CTX90pct + Excess Heat Factor + full NWS Rothfusz heat
  index), atmospheric river (trapezoidal IVT + Ralph 2019 AR1–AR5 scale),
  lightning (Schultz 2009 2-σ lightning jump), hail (SPC SHIP), winter/ice
  storm (partial thickness, Stull wet-bulb, FRAM-style accretion, blizzard
  criteria), derecho (Johns & Hirt / Corfidi criteria with great-circle
  swath math), dust storm (WMO visibility classes, Fécan friction-velocity
  threshold, haboob gust-front signature), and the hurricane→surge→flood
  cascade (real CO-OPS observed-minus-predicted surge residual + NWPS gauge
  gates).
- **Geophysical**: subsidence/sinkhole precursors over InSAR-style series
  (Theil–Sen velocity, model-comparison acceleration, clustering), avalanche
  (SK38 skier-stability, critical loading, TG metamorphism, EAWS levels),
  rockfall (freeze–thaw cycling, Fukuzono inverse-velocity failure
  forecasting), plus the earthquake→tsunami cascade (PTWC-style screening on
  real USGS events incl. the recorded Tōhoku fixture) and wildfire→debris
  flow (the published USGS Staley et al. 2017 M1 coefficients + Cannon I-D
  thresholds).
- **Board**: SpaceWeatherLoader (USGS geomag minute data + DONKI GST observed
  Kp), MeteorLoader (JPL CNEOS fireballs ≥1 kt + NeoWs), DroughtLoader /
  HeatwaveLoader (NCEI GSOM/GSOD station archives), HailLoader (SPC reports)
  — all with provenance-registry entries and measured-not-invented AUC gates
  in `run_all_benchmarks.py` (space_weather 0.9647→gate 0.85, meteor
  0.7705→0.60, heatwave 0.6757→0.63, drought 0.5728→0.54, hail 0.6132→0.55).

### Counterfactual anomaly reasoning on the detection path

- Verified minimal counterfactuals as first-class detection explanations:
  correctness and minimality are re-scored through the REAL detection paths
  (never the search method's own claim). Tier detections explain a flagged
  point over its contextual window with candidates anchored at real normal
  windows from the same series; exposed via `tier-detect --counterfactual`,
  `POST /detect/tier {include_counterfactual}`, and `mercury_tier_detect`.
- The engine's dormant `gdpr_report` explanation path (Wachter
  counterfactuals) is finally reachable: `POST /detect/flagship
  {gdpr_report, subject_id}` + `mercury_detect_fusion`.
- `CounterfactualMethod.GENETIC` and `DistanceMetric.GOWER` implemented for
  real (they were bare enum values): a seeded evolutionary search and the
  standard mixed-type distance, selectable everywhere with unit tests.

### Genetic rule evolution grounded in held-out detection F1

`ml/rule_evolution.py`: a deterministic GA over the neuro-symbolic rule
graph (evolved predicates resolve into the existing `Rule`/`RuleGraph`
representation and are scored by the same deployed
`SymbolicConstraintModule` path — no train/serve skew). Fitness is mean
held-out VALIDATION F1 across pre-registered real ADBench datasets with the
hand-written consensus graph seeded into the population; the TEST split is
touched exactly once. Result (`benchmarks/rule_evolution_results.json`):
evolved graph beats the consensus baseline on held-out test F1, mean 0.5439
vs 0.4119 (+0.1320; cardio +0.296, thyroid +0.227, WBC tie, Pima +0.005),
AUC better on all four. The champion ships as a schema-versioned artifact
loadable via `resolve_rule_graph("evolved:…")` with a serve-path
reproduction check.

### Detection interconnect + analysis extraction

Detection is now reachable through one consistent contract on every surface
(CLI / HTTP / MCP), and two attribution analyses that previously had no runtime
entry point are exposed. Each change is additive — no existing endpoint changed
behaviour — and carries tests.

- **Detector-tier ensemble has a real entry point.**
  `detectors.detection_tier.run_tier_ensemble()` builds → fits → scores the
  torch-free streaming / statistical / state-space tier in one call (calibrated
  per-point scores, flags, cross-detector uncertainty, optional distribution-free
  conformal false-positive control). Wired to `mercury-agent tier-detect`, the
  `POST /api/v1/detect/tier` route, and the `mercury_tier_detect` MCP tool.
- **The flagship neuro-symbolic fusion engine is unified across surfaces.** The
  full `OmniMercuryEngine` path (trained fusion network + GOSNN scalar
  integration + the σ_Immutable hard ethical gate) — previously reachable only
  from `mercury-agent detect -d fusion` — is now `POST /api/v1/detect/flagship`
  and the `mercury_detect_fusion` MCP tool. HTTP builds the (non-thread-safe)
  engine once and serializes detections through a lock via a threadpool; an
  ethical-gate refusal fails closed as **HTTP 403** (never a silent allow).
- **Graph-based root-cause localization is exposed** (`localize_root_cause`,
  over the tier's `RootCauseGraphDetector`): given a multivariate anomaly it
  ranks which node (sensor / service / channel) most likely originated it, with
  optional node labels. Reachable via `mercury-agent rca`,
  `POST /api/v1/detect/rca`, and the `mercury_localize_root_cause` MCP tool.
- **Per-detector attribution for the tier** (`include_attribution=True` /
  `tier-detect --attribution`): returns the calibrated per-detector score matrix
  so callers see *which* members drove each flag, not just the blend.
- **`GET /export/metrics` now serves CSV and JSON-Lines in addition to JSON**,
  and its JSON summary is enriched (score percentiles, per-method breakdowns,
  and time-series buckets via a `bucket` param). The multi-format export is a
  deliberate capability for the STEAM / medical / meteorological / space
  researchers Mercury serves — collecting event and request data in a tabular
  format is an advantage, not a footgun to reject.

### Fixed — `CachedBenevolenceScorer` threshold setter (drop-in fidelity)

`CachedBenevolenceScorer` (the LRU wrapper the engine now places on its ethics
boundary by default) exposed `benevolence_threshold` as a read-only pass-through
property. Once the boundary scorer was wrapped, the previously valid
`engine._boundary_scorer.benevolence_threshold = x` (runtime gate tuning) raised
`AttributeError` — a compatibility regression from wrapping, and a deterministic
failure in `tests/ethical/test_hard_enforcement.py`. Added a setter that
delegates the assignment to the wrapped scorer (preserving its
`MINIMUM_BENEVOLENCE_FLOOR` clamp) **and clears the cache**: cache keys are
`(ruleset_version, action, context)` and do not encode the threshold in force at
compute time, so raising the bar must not serve a decision cached as permissible
under the old threshold. Locked by `test_threshold_setter_delegates_to_wrapped_scorer`,
`test_threshold_setter_preserves_floor_clamp`, and
`test_threshold_setter_invalidates_cache`. See `benchmarks/PR329_VERIFICATION.md`
for the full independent verification run (real AMA v3.3.0 backend; detector-tier
suites, real-NAB before/after, and live-path validation of the wired subsystems).

### Detection mechanisms: streaming / statistical / state-space detector tier (steel/detection-mechanisms)

A deliberate, end-to-end expansion of the classical anomaly-detection surface:
**eighteen detectors across six paradigms**, each implementing the
`omni_mercury_engine.core.base.BaseDetector` contract, auto-discovered through
`core/detector_registry.py::DETECTOR_MANIFEST`, calibrated to emit probabilistic
scores, and wired into a calibrated ensemble with distribution-free
false-positive control and graph-based attribution. No stubs, no disabled feature
flags. Design doc `docs/DETECTION_MECHANISMS.md`; ops
`docs/DETECTION_MECHANISMS_RUNBOOK.md`.

- **Detectors (16 pure-NumPy, always importable).** Temporal/streaming
  `spectral_residual` (SR saliency, Ren et al. KDD 2019), `bocpd` (Bayesian
  online change-point), `hawkes` (self-exciting burst); state-space
  `particle_filter`, `imm` (interacting-multiple-model switching), `digital_twin`
  (observed-vs-simulated AR residual); probabilistic `spot_evt` (SPOT/DSPOT EVT
  peaks-over-threshold), `gaussian_process`, `survival` (Kaplan-Meier + Cox);
  generative `energy_based`, `deep_svdd`; neuromorphic `echo_state`, `spiking`;
  systems-level `rca` (random-walk root-cause over a causal/service graph),
  `deeplog_sequence`, `frequent_pattern`.
- **Torch-gated detectors (2, `[ml]` extra).** `srcnn` — CNN discriminator over
  SR saliency trained on synthetic-anomaly-augmented series (Ren et al. 2019);
  `diffusion_ad` — DDPM denoising-reconstruction detector (Ho et al. 2020).
  Lazily imported so the tier degrades gracefully without PyTorch.
- **Calibration contract.** Residual detectors squash raw residuals via
  `1 - exp(-r/scale)` with `scale` anchored at a high training quantile (default
  0.98), placing the 0.5 boundary in the normal tail for a controlled ~1–2% FPR;
  natively-probabilistic detectors (BOCPD, SPOT) are left unsquashed.
- **End-to-end integration** (`detectors/detection_tier.py`).
  `StreamingScoreEnsemble` combines per-point scores through Mercury's own
  logistic meta-learner (`ml.mercury_ml.LogisticRegression`, stacking) or
  Bayesian Model Averaging (BIC weights + bootstrap uncertainty), thresholded via
  `core.score_calibration.calibrate_scores`, with per-point cross-detector
  disagreement as ensemble uncertainty. `conformal_threshold` / `conformal_flags`
  bound the streaming FPR distribution-free via
  `core.conformal_prediction.SplitConformalPredictor`. `rca_localize` returns
  ranked root-cause attributions over a causal/service graph.
- **Runtime pipeline wiring.** `TierStreamingScorer` adapts any tier detector to
  the `infrastructure.streaming.StreamingAnomalyPipeline` `dict -> dict` detector
  callable (rolling window + drift refit + metric emission);
  `store_tier_features` persists detector features into
  `core.feature_pipeline.FeatureStore` with a registered `FeatureSchema` for
  provenance/versioning; `decision.bridge.to_cap_alert` gains an `rca_causes`
  argument that attaches ranked root causes to the CAP alert's Mercury metadata
  (rendered into the CAP `<description>`) under `RootCauses`; and
  `core.metrics.record_detector_score` feeds a new per-detector
  `omni_detector_score` histogram consumed by the existing Grafana boards. The
  benchmark harness also reports throughput (points/sec).
- **Robustness fix.** `FrequentPatternDetector` Apriori mining is now bounded to
  the top `max_items` itemsets per level (deterministic tie-break, truncation
  logged), eliminating a combinatorial hang on wide/degenerate input (e.g. a
  single continuous series fed through the fusion registry). Regression tests
  added.
- **Validation on real data.** Per-detector contract + signal tests under
  `tests/detectors/`; tier wiring in
  `tests/detectors/test_detection_tier_integration.py`. The tier's **performance
  is measured on real, human-labelled anomaly data** — the Numenta Anomaly
  Benchmark (NAB) real categories, pulled through the shared dataset layer via
  the new `datasets.timeseries.NABLoader.iter_series` 1-D streaming accessor (the
  synthetic `artificial*` NAB sets are excluded). `benchmarks/detection_tier_benchmark.py`
  is the real-data streaming-evaluation library; `benchmarks/mercury_benchmark.py`
  merges its output into the **one canonical** `benchmarks/mercury_benchmark_results.json`
  under the `detection_tier` key (the earlier synthetic `detection_tier_synthetic.py`
  / `detection_tier_results.json` silo is removed). Measured over **29 real NAB
  series** (unsupervised streaming, per-point ROC-AUC): the **unsupervised
  `average` ensemble leads at ROC-AUC 0.613** (median 0.617), edging the best
  single member (`echo_state`, 0.610); the supervised `stacking` / `bma` combiners
  are reported on the 15-series subset where a temporal split leaves both classes
  in both folds. Network-free plumbing tests in
  `tests/benchmarks/test_detection_tier_realdata.py`.
- **AMA-Cryptography pin bumped `v3.2.0 → v3.3.0`** (still mandatory, fail-closed):
  `_pqc_gate.py` constants, the pyproject `pqc` git pin, every CI `AMA_REF`,
  `scripts/build_ama_native.sh`, the `Dockerfile` build arg, and the version-gate
  tests updated in lockstep; all remaining docs/comments reworded to a single
  consistent pin narrative ("introduced in v3.2.0, retained in v3.3.0").
- **HS384 JOSE signing now routes through AMA's native C HMAC.** v3.3.0 adds a
  native HMAC-SHA-384 binding, so `security/ama_hmac.py` exposes
  `ama_hmac_sha384` / `HAS_AMA_HMAC_SHA384` and `security/native_jwt.py` routes
  HS384 (fail-closed, no silent stdlib downgrade) alongside HS256/HS512, locked
  by the RFC 4231 §4.2 HMAC-SHA-384 known-answer vector.
- **Detector correctness fixes** surfaced in review: `DigitalTwinResidualDetector`
  now runs its free-running divergence open-loop *to* time `t` (the seed ends at
  `t-h` and the twin feeds its own predictions), instead of indexing the
  seed/history window; `SurvivalHazardDetector._covariate` is O(n) via a prefix
  sum with the correct `w`-length trailing window (was O(n·w) with an off-by-one
  and dead code).
- **Streaming & observability hardening** surfaced in review:
  `core.metrics.record_detector_score` is now fail-safe — a non-finite
  (`NaN`/`inf`) or non-numeric score is dropped and an out-of-range value is
  clamped to the `[0, 1]` anomaly-score contract before the histogram observes
  it, so a misbehaving detector can neither raise into nor `NaN`-corrupt the
  metric on the hot streaming path; `TierStreamingScorer` now reports `warmup`
  from its buffered history rather than unconditionally on a value-less message,
  so a ready scorer is no longer mistaken for warming up (which would suppress
  downstream alerting). Manifest and design-doc descriptions were also corrected
  to match the implementations: `deep_svdd` is a fixed random `tanh`-feature
  embedding (saturating, not random-Fourier) and `energy_based` is a
  delay-embedding quadratic (Gaussian-family) energy (not RBF-feature), and the
  `rca` walk comment no longer claims an explicit `Aᵀ` transpose. Regression
  tests added.

### Detector tier hardening: non-finite, empty/short-input, and correctness fixes (steel/detection-validation)

A full-effort **adversarial audit and hardening pass** over the streaming /
statistical / state-space detector tier shipped above. An 18-detector +
integration-seam audit (one empirical prober per detector, each finding
independently reproduced and verified) surfaced **35 confirmed defects**; every
one is fixed at the root, with a regression test, and with detector scores on
ordinary finite data proven **byte-identical before/after** — so the committed
NAB benchmark numbers are unchanged; only the previously-crashing / `NaN`-producing
pathological paths change. New shared module
`detectors/_calibration.py` (`bound_finite`, `squash_scale`, `finite_scores`,
`finite_features`) removes the copy-pasted defects at a single, unit-tested source.

- **Non-finite scores could escape the `[0,1]` contract (14 detectors).** Input
  sanitisation used `np.nan_to_num`, which maps `±inf` to `±1.8e308`; that
  sentinel overflowed downstream (FFT power, Kalman/GP covariance, AR Gram
  matrix, Cox/KM cumulative sums, delay-embedding) into `NaN`, and the final
  `np.clip` does **not** scrub `NaN`. A single non-finite input sample therefore
  produced all-`NaN` scores, and at `fit` time permanently poisoned the detector
  so even later clean data scored `NaN`/all-zero. `bound_finite` now bounds input
  to `±1e100` (astronomically above any real signal, so realistic data is never
  clipped) and `finite_scores`/`finite_features` guarantee finite `[0,1]` scores
  and finite `float32` fusion features at the output boundary.
- **Empty / very-short input crashed (17 sites).** The copy-pasted `_squash_scale`
  called `np.quantile` on a zero-length array, crashing `fit([])` and unfitted
  `detect([])`; `energy_based`/`deep_svdd` delay-embedding indexed `u[0]` on an
  empty series and mis-broadcast on series shorter than the embedding dimension;
  a single-row calibration split produced a `NaN` precision. All now degrade
  gracefully.
- **BOCPD run-length truncation off-by-one.** The fold double-counted
  `growth[cap-2]` and silently dropped the boundary message `growth[cap-1]`,
  inflating the change-point score once a run reached `max_run_length`; now folds
  the correct tail term (dormant at the shipped default, corrected for small caps).
- **SPOT/DSPOT non-idempotence.** `detect`/`extract_features` mutated the fitted
  tail state, so repeated scoring drifted the threshold and scored data
  contaminated the model; the online update is now snapshot/restored, keeping
  within-batch DSPOT adaptation and per-call scores unchanged.
- **digital-twin singular-matrix crash** on a constant / large-magnitude series
  now falls back to `lstsq` only when the exact solve is singular (normal data
  unchanged).
- **RCA / spiking `detect`-before-`fit`** paths (which their own bodies branch
  for) no longer assert; RCA returns empty on an empty batch and raises a *clear*
  error on a node-count mismatch instead of a cryptic broadcast error.
- **frequent-pattern column mismatch:** a `detect` batch narrower than the
  training vocabulary now skips out-of-range rules instead of indexing out of
  bounds.
- **Ensemble seam:** `align_point_scores` sanitises member output as
  defence-in-depth, so one member emitting `NaN` can no longer poison the
  ensemble mean / stacking / BMA or crash `calibrate_scores`' histogram.
- **Benchmark transparency:** `detection_tier_benchmark._crop_to_anomaly` re-centres on
  the first anomaly when the midpoint window would retain none, so a labelled NAB
  series is never silently dropped from the run.

New tests: `tests/detectors/test_calibration_helpers.py` (shared helpers) and
`tests/detectors/test_detector_robustness.py` (per-detector adversarial contract
suite + targeted regressions). Design/rationale and the full findings-to-fix map
are in `docs/DETECTION_MECHANISMS.md` (§ Robustness & hardening).

A **second hardening pass** builds directly on the primitives above — ensemble
calibration, an explicit configurable NaN policy, guard observability, and two
numerical/concurrency fixes that *supersede* first-pass stopgaps. New config
module `detectors/detection_config.py`; reproduction harness
`benchmarks/reproduce_detection_tier_nab.py`; process note + reproducible
checklist in `docs/DETECTION_TIER_HARDENING.md`.

- **Ensemble score calibration.** `StreamingScoreEnsemble` no longer combines raw
  scores. Each detector's score column is first mapped through a per-detector
  calibrator fitted on the warm-up window — `rank`/`ecdf` (empirical-CDF, label-
  free, the **default**), or supervised `isotonic`/`platt` (which fall back to
  ECDF when the warm-up is single-class) — so no detector's arbitrary score scale
  dominates. A new label-free `consensus` combiner takes a robust cross-detector
  high quantile (default 0.9) of the calibrated columns, which is not dragged
  toward 0.5 by uninformative members the way a plain mean is. **On 30 real NAB
  series the calibrated `consensus` ensemble beats the best single detector by
  > 0.003 ROC-AUC** (see `benchmarks/detection_tier_nab_analysis.md`); a standard
  outlier-ensemble result (Aggarwal & Sathe, 2017).
- **Explicit, configurable NaN/Inf policy + observability.** `bound_finite` /
  `finite_scores` / `finite_features` are now the configurable, *observable* face
  of the first pass's guards: `neutral` (default, byte-identical to the first
  pass), `impute`, `flag`, or `raise` (fail closed) via `OMNI_DETECTOR_NAN_POLICY`
  / config file, and every correction increments the new
  `omni_detector_nonfinite_corrected{detector,policy,field}` Prometheus counter
  and emits a structured log. The magnitude regime is unified on one documented
  constant (`DEFAULT_MAX_MAGNITUDE` = `_calibration.FINITE_CAP` = `1e100`). SPOT's
  `z_q` / `gamma` metadata are guarded by the same finite checks as scores.
- **digital-twin: scale-relative Tikhonov ridge (supersedes the `lstsq` fallback).**
  The solve now uses `λ = max(ridge_factor · tr(G)/d, ridge)` — proportional to
  the Gram matrix's mean eigenvalue so regularisation tracks the data scale,
  floored by the absolute `ridge`. The regularised matrix is always SPD, so the
  first pass's `lstsq`-on-singular fallback is no longer needed; coefficients stay
  finite and bounded across ≥15 orders of magnitude. `ridge_factor` is
  configurable (`OMNI_DETECTOR_RIDGE_FACTOR`).
- **SPOT: true purity + thread-safety (supersedes snapshot/restore).**
  `_tail_probability` / `_threshold_from_tail` are now pure static functions of
  explicit local state, and `_stream_scores` evolves the online tail on local
  copies — so `detect()` mutates nothing on the instance at all. It is idempotent
  *and* safe to call concurrently on one fitted detector, without the first pass's
  snapshot/restore of instance state.
- **Benchmark `_crop_to_anomaly` (supersedes the first-anomaly re-centre).** The
  crop now selects the window retaining the *most* labelled anomalies (prefix-sum
  sliding window), centred among ties so a downstream 50/50 split still straddles
  the anomalies — strictly stronger than re-centring on the first anomaly.
- **Configuration.** Calibration method, NaN policy, magnitude cap, ridge factor,
  and ensemble warm-up are settable via env (`OMNI_ENSEMBLE_CALIBRATION`,
  `OMNI_ENSEMBLE_WARMUP`, `OMNI_DETECTOR_NAN_POLICY`, `OMNI_DETECTOR_MAX_MAGNITUDE`,
  `OMNI_DETECTOR_RIDGE_FACTOR`), a config file (`OMNI_DETECTION_CONFIG`), or a
  detector's `config` dict — defaults < file < env < explicit.
- **Testing.** Added unit, Hypothesis property (score/metadata finiteness+bounds
  under arbitrary finite/non-finite input), SPOT concurrency, BOCPD run-length
  mass-conservation (posterior sums to 1 every step, via the new
  `run_length_posteriors` accessor), digital-twin conditioning, RCA
  `_walk`/adjacency-inference coverage, and Prometheus-counter / structured-log
  assertions.

### Intelligence layer: closed-loop learning + decision geometry (steel/refinement-mercury-intel)

The learning-and-geometry layer on top of the Tier-0 safety foundation. New
package `omni_mercury_engine.intel`, six streams, each shipping a **measured
value metric** (baseline + target, `intel.value_metrics.VALUE_METRICS`; board
rendered by `benchmarks/intel_value_metrics_report.py`). No stream regresses the
foundation; every gate is fail-closed.

- **Closed feedback loop (accept-gated).** `intel.feedback_loop`: audit events
  and human overrides become human-verified labels (`labeling`, reviewer
  required — anonymous labels refused), stored in a durable, snapshot-hashable
  queue (`queue`, `MERCURY_FEEDBACK_QUEUE_URL`). A retrain runs only behind three
  fail-closed gates: a **signed/audited trigger** (`trigger`, HMAC-SHA256 bound
  to the exact queue snapshot, `MERCURY_RETRAIN_TRIGGER_SECRET`), a **human
  verification** flag, and an **OOF/adversarial regression gate**
  (`regression_gate`) that reuses the shipped `rolling_corpus_eval` MARGINS as a
  merge-blocker on the candidate model — a regressing or data-poisoned candidate
  is blocked (measured poisoned-candidate block rate 1.0). Accepted candidates
  are staged (never promoted straight to prod) with a **one-command rollback**
  (`rollback`, `scripts/mercury_retrain_rollback.py`). End-to-end staging demo:
  `scripts/closed_loop_demo.py` (`--poisoned` shows the block).
- **Confidence cascade.** `intel.cascade`: calibrated-uncertainty routing between
  a cheap template path and a heavy model path, folding in the self-consistency
  disagreement, with compute-cost/latency instrumentation. Cost-vs-accuracy
  report `benchmarks/confidence_cascade_report.py` (~64% compute saved at zero
  accuracy loss on the synthetic workload; target ≥50%).
- **Self-consistency signal.** `intel.self_consistency`: N-sample reasoning-path
  sampling and a vote-disagreement uncertainty metric the calibrator consumes
  (`widen_uncertainty`, `self_consistency_decision` abstains on high
  disagreement). Disagreement predicts error at AUROC ≈0.86 on held-out
  (`benchmarks/self_consistency_signal_report.py`; target ≥0.70).
- **Verifier-in-the-loop.** `intel.verifier_loop`: routes generative claims
  through the symbolic verifiers (primality, Collatz, propositional, physics) and
  blocks emission on an oracle-refuted claim (`MERCURY_VERIFIER_MODE=hard|soft`).
  Propositional claims are decided by the shipped DPLL oracle via a Tseitin CNF
  transform (`intel.propositional_claims`). Hard-mode false-claim block rate 1.0.
- **Adversarial co-training.** `intel.red_team` + `benchmarks/red_team_harness.py`:
  a deterministic paraphrase/obfuscation generator (`configs/red_team.yaml`,
  `MERCURY_RED_TEAM_CONFIG`) attacks the shipped gate; surviving bypasses are
  appended to `corpus/pending` with triage metadata for the closed loop. Transparent
  finding: character obfuscation (spacing/punctuation) defeats lexical matching
  (~0.33 survival rate), pinned as a no-weakening floor
  (`benchmarks/red_team_baseline.json`).
- **Provenance-as-type + boundary fallback.** `intel.provenance`: the timeboxed
  decision ships the ~80%-value fallback — provenance carried as a typed
  companion (`Provenanced[T]`) and enforced at the output boundary
  (`enforce_at_boundary` refuses/redacts an unprovenanced provenance-required
  emission), reusing the gate's own `ALLOW_PROVENANCE` rule for what needs
  citations. `MERCURY_PROVENANCE_MODE=type|boundary-fallback`; the `type` seed
  (`require_provenanced`) and the migration plan to full unrepresentability are in
  `docs/PROVENANCE_MIGRATION_PLAN.md`.
- **CI.** New `Mercury Intel` workflow with the four required lanes:
  `ci/closed-loop-integration`, `ci/confidence-cascade`, `ci/red-team`,
  `ci/verifier-integration` (each builds the native AMA/PQC backend).
- **Docs.** `docs/RETRAIN_RUNBOOK.md`, `docs/RED_TEAM_OPERATION_GUIDE.md`,
  `docs/PROVENANCE_MIGRATION_PLAN.md`, `docs/INTEL_VALUE_METRICS.md`.
- **Tests.** `tests/intel/` (112 tests) covering every stream, incl. corpus-backed
  regression-gate and end-to-end closed-loop integration (`slow`/`integration`).

#### Live wiring + audit hardening (review)

De-islanding: the streams above are no longer standalone modules measured only in
benchmarks — they run on the live request/emission/operator path, and an
adversarial audit's defects are fixed (no suppression; each fix carries a test).

- **Verifier-in-the-loop is live.** The MCP `mercury_research` / `mercury_answer`
  emission path now routes its output through the verifier loop — hard mode blocks
  an oracle-refuted claim, soft mode flags it (`MERCURY_VERIFIER_MODE`),
  unavailable never blocks. Four streams are exposed as selectable MCP tools
  (`mercury_verify_claims`, `mercury_check_provenance`, `mercury_self_consistency`,
  `mercury_value_metrics`) and as a `mercury-agent intel …` CLI group (`verify`,
  `provenance`, `self-consistency`, `value-board`, `audit-log`, `rollback`,
  `red-team`, `cascade`).
- **Feedback loop reads the real gate log.** `feedback_loop.read_audit_log`
  ingests the actual `gate_decisions.jsonl` the harm gate writes
  (`gate_audit.default_audit_log_path`), so the loop learns from the gate's real
  decisions rather than hand-authored records.
- **Closed-loop safety fixes.** Rollback is monotonic (a repeated rollback is a
  no-op, never re-arming the superseded model); `staged_refit` reads the queue via
  a single atomic `snapshot()` (closes a TOCTOU window between the signed hash and
  the trained rows); the trigger `nonce` is a real single-use anti-replay token
  (durable `NonceLedger`) and `n_examples` is asserted against the snapshot.
- **Soundness / fail-open fixes.** Tseitin auxiliary variables are namespaced away
  from formula variables (a formula naming a variable `_t1` no longer corrupts the
  CNF); `Provenance.is_adequate` requires an *attributed* origin (a fabricated
  citation on a synthetic/model-generated origin no longer passes a hazardous
  boundary) and `enforce_at_boundary` fails closed with no topic signal.
- **De-theatered value gates.** The value board `--check` requires the target for
  non-aspirational streams (the baseline-0 `improves_on_baseline` check was
  vacuous); the adversarial row is measured live; the red-team `--check` drops a
  bogus 0.02 slack (the rate is deterministic) and catches seed-level weakening
  (pins `n_skipped_seeds` / `n_candidates`, surfaces `n_downgraded`); the
  confidence-cascade workload includes a confident-but-wrong slice so the accuracy
  tolerance gate is load-bearing.
- **Test isolation.** `tests/intel/conftest` no longer leaks
  `MERCURY_GATE_AUDIT_DISABLED` into sibling suites (session `os.environ` mutation
  → function-scoped `monkeypatch`).

#### Static-analysis alert resolution + live meaning-level harm classifier

Full resolution of the open CodeQL and static-analysis alerts against the intel layer — each
verified against the running code and fixed with a regression test; no
suppression, no weakening. All CI lanes green on the native AMA backend.

- **CodeQL clear-text logging (secret) — resolved (alerts 900/902/903).**
  `verify_trigger` now logs an enumerated `reason_code` (a fixed literal), and the
  human-readable `reason` no longer interpolates the `SECRET_ENV` identifier at
  all — that name reaches the durable audit sink (which stores it and, on a write
  failure, logs the whole record), so keeping every secret-named token out of
  `reason` removes the finding by construction rather than masking it. CodeQL
  reports 0 alerts. (The failing GitHub-managed `CodeQL` check was the
  code-scanning *results* check reporting these alerts — **not** a default-vs-
  advanced-setup conflict: the advanced `security.yml` analysis uploads SARIF
  cleanly, which GitHub rejects when default setup is enabled, so no repo/settings
  toggle was needed.)
- **Verifier emission guard fails closed on runtime faults.** `mcp_server._guard_emission`
  degrades to *allow* only on genuine unavailability (`ModuleNotFoundError`, the
  slim-install case); a verifier **runtime** fault now fails **closed** (block in
  hard mode, flag-allow in soft), and a symbol-level `ImportError` (a verifier bug)
  propagates rather than masquerading as "unavailable" — a broken verifier can no
  longer silently disable hard-mode gating.
- **Provenance `verified` is type-validated.** `mercury_check_provenance` rejects a
  non-bool `verified` (a truthy string like `"false"` can no longer assert
  verification on a hazardous boundary).
- **`NonceLedger` is single-use across processes.** The check-then-append is guarded
  by an inter-process advisory file lock (`fcntl.flock`) in addition to the
  in-process lock, so two concurrent *processes* can never both consume the same
  retrain nonce (a multi-process test pins exactly-one-consumer).
- **Durable queue tolerates malformed lines and stays self-consistent.** A
  truncated/corrupt final line (or a valid-JSON-but-non-record line, or a dict
  missing required fields) is skipped with a warning instead of crashing every
  read; `dedup`/`pending`/`__len__`/`snapshot` all derive from one defensively-
  parsed view so their counts can never disagree (a mismatch would refuse a
  validly-signed trigger). `enqueue_many` reads the dedup ids once and appends the
  batch under a single lock + fsync — O(n+m), not O(n·m).
- **Transparent types & docstrings.** `SelfConsistencyResult.as_dict` renders the
  plurality answer as a string (JSON-safe, matching its contract);
  `CascadeInstrumentation.report` is typed `dict[str, float | int]` (the `n_*`
  fields are integer counts); `VerifierLoop.guard_emission` audits **every**
  disposition (pass/block/flag), making "every decision is durably audited" true;
  the `Sampler` comment no longer implies `self_consistency` auto-selects the
  continuous `dispersion` path; and the mislabeled `returns_none_for_non_formula`
  test is split so each name matches its asserted behaviour.
- **Meaning-level harm classifier proven live.** The weapons-gate routing rescue was
  confirmed end-to-end with an actual served instruction-tuned model (Qwen2.5-1.5B-
  Instruct over the Ollama wire protocol on loopback), so `default_harm_classifier()`
  reports ACTIVE. Measured on the held-out adversarial slice: false negatives
  **15 → 5 (−67 %; FN-rate 0.517 → 0.172)** with FP held at **0 → 0**; the blocking
  `test_real_classifier_fn_budget` lane passes under `MERCURY_CI_REQUIRE_REAL_CLASSIFIER=1`.
  Evidence recorded in `docs/WEAPONS_GATE_ADVERSARIAL_EVAL.md` and the
  `served_model_real` block of `benchmarks/weapons_gate_adversarial_sample_run.json`.

### Merge-readiness hardening

A focused pass that reproduced the full CI pipeline locally, closed every red
lane, and fixed the defects surfaced by an adversarial multi-lens review (each
finding independently verified against the running code, then fixed with a
regression test). No suppression: every issue is resolved or documented.

**CI reds closed**

- `cognitive/harm_normalization.py`: `normalized_haystack` docstring made a raw
  string (pydocstyle D301); `translit_variants` annotated `tuple[str, ...]` so
  mypy no longer infers `tuple[()]`.
- README codebase-scale block regenerated (measured 658 source / 467 test
  modules).
- Strict test-lane typing for the new ethical suites: full annotations in
  `tests/ethical/test_gate_governance.py`; `eval_weapons_gate` /
  `weapons_gate_corpus` declared as sibling-script mypy boundaries; `hypothesis`
  added to the Type-Checking install so the strict lane resolves
  `@given`/`@settings` (previously flagged `[untyped-decorator]`).

**Harm-gate correctness (verified, FP-safe)**

- **Defensive-production carve-out seam closed.** A carved-out request
  (make-a-detector/vaccine/antidote defusing a B6 PRODUCTION match) previously
  resolved to a plain `ALLOW` (low-severity domain) or `ALLOW_LOG` (licensed
  framing) — free, uncited synthesis on a hazardous topic. The carve-out now
  floors to `ALLOW_PROVENANCE` whenever it fires (cited-sources-only, withheld
  when no source is available; enforced at the `GeneralAssistant` output
  boundary), regardless of hazard severity or licensed framing. `ALLOW_PROVENANCE`
  is non-blocking, so professionals are not re-strangled and the measured
  operating point is unchanged (**FP 0.000% / FN 0.000%** on val/test/train);
  B7–B10 are never carved out. Regression tests added in
  `tests/ethical/test_gate_governance.py` (including the load-bearing
  `ALLOW_LOG → ALLOW_PROVENANCE` licensed-practice case).
- **Nuclear enrichment false-negative closed.** `how to enrich uranium to
  weapons-grade` answered ALLOW (harm 0.0) — the Axis-A NUCLEAR lexicon carried
  `enriching uranium`/`uranium enrichment` but not the bare-verb variant
  `enrich uranium`, so the query routed no hazard domain. Added the
  morphological variants and a **weapons-directed** enrichment/reprocessing
  intent pattern (`enrich|reprocess … to/for/into … weapons-grade/warhead/a
  nuclear weapon`). Civilian/policy/mechanistic enrichment stays on the ALLOW
  ladder; corpus still 0% FP / 0% FN.
- **Defensive-production carve-out bypass closed.** Dropping the B6 PRODUCTION
  match on the mere co-presence of a detection keyword let `synthesize sarin AND
  how to detect it` downgrade to ALLOW_PROVENANCE. The carve-out now requires a
  production verb bound to a defensive **object** (`_DEFENSIVE_PRODUCTION_RE`:
  make/build/… a detector/sensor/vaccine/antidote/countermeasure/…), so a hazard
  object stays gated while `make a nerve agent detector` is still protected.
- **Euphemism false-positive removed.** `the target`/`the targets` are
  polysemous (target server/process/host/audience) and tripped the euphemism
  layer on routine security/ops language, violating its documented
  person-specific invariant; removed.
- **`write_document` fail-open closed.** The benevolence/weapons gate screened
  only `Section.body`; a harmful `heading` or `bullet` bypassed it while still
  rendering (reachable via the MCP surface, which coerces client dicts to
  `Section` objects). Now screens all rendered fields.

**Capability-surface & operational readiness**

- **Web research made functional + SSRF-correct.** The open-web page transport
  omitted `user_configured=True`, so `SafeHTTPClient` applied the dataset
  `TRUSTED_DOMAINS` allowlist to every fetch (wikipedia/arxiv/… all refused) and
  an https URL skipped the IP re-check. Now bypasses the dataset allowlist while
  running the private-network / IMDS gate (loopback/IMDS still refused).
- **Markdown link/image injection neutralized.** `DocumentGenerator._md_safe`
  now backslash-escapes `[`/`]` in addition to `&`/`<`/`>`, so extracted
  `[text](javascript:…)` renders as literal text (any scheme, renderer-agnostic).
- **Durable gate audit persists on installed deployments.** `gate_audit`'s
  default sink resolved under `site-packages` on a wheel install (read-only →
  silently swallowed). Now: env override → repo `artifacts/` only for a writable
  source checkout → per-user `XDG_STATE_HOME`/`~/.local/state` fallback.
- `distributed/cluster.py`: `n_results` reports the count of partitions actually
  merged (not `len(self._results)`, which included failed/empty ones).
- `cognitive/harm_classifier.py`: the reasoning→probability adapter is built
  once per backend, not per `classify()` call on the hot gate path.

**Evaluation & calibration transparency**

- `evaluation/metrics.py`: `evaluate_anomaly_detection_split` now guards the
  **test** split's class composition (not just val), falling back to in-sample
  with a warning on a single-class test split instead of emitting AUC=0.5 as an
  "transparent" number; the `best_f1` docstring is corrected — it is the validation
  in-sample maximum (optimistic upper bound), and the leakage-free operating
  point is the `f1` field.
- `core/confidence.py`: `fit` validates a scores/labels length mismatch up front
  with a clear `ValueError` instead of raising an opaque broadcasting error from
  the "graceful" identity path.

**Reviewed, not a live defect (documented):** the 2 MB body read is a
decompression-bomb concern only on older urllib3; urllib3 2.7.0 (pinned) bounds
the decoded read to the requested size (~4.7 MB peak on a 500 MB logical bomb).
`FeatureCache` torch-tensor identity keying (data_ptr + view metadata) is a
deliberate, documented tradeoff — content-hashing a device tensor would force a
per-lookup sync — and stale pointers age out of the bounded LRU.

**Deployment-image CVE response — gzip + glib eliminated, not accepted
(2026-07-02).** The vulnerability DB published four new unfixed CRITICAL/HIGH
findings against the runtime image, turning the blocking Trivy gates red:
`gzip` CVE-2026-41992 (High) and `libglib2.0-0t64` CVE-2026-58016 (Critical) /
CVE-2026-58014 / CVE-2026-58015 (High), none with a Debian trixie fix. Both
packages were **removed from the image** rather than ledgered:
`libglib2.0-0` was only cv2's historical `libgthread` dependency, but the
shipped `opencv-python-headless` wheel (≥ 4.13) vendors its media stack and
links no glib at all (verified via `readelf` NEEDED and a cv2 import +
`cvtColor`/`Canny` run on a glib-less trixie base), so the Dockerfile no longer
installs it; `gzip` is Debian-essential with no runtime consumer (CPython's
`gzip`/`zlib` use the linked `libz`, dpkg/apt decompress internally, nothing
runs `tar -z`), so it is purged alongside `perl-base` (verified post-purge:
`apt-get update`/`install`/`upgrade`, `dpkg`, Python `gzip` round-trip). The
`.trivyignore` acceptance count is unchanged at 3 (0 Critical, 3 High);
SECURITY.md's posture section was brought in sync with the ledger's
2026-06-30 re-enumeration (SQLite FTS5 pair re-scored MEDIUM upstream and out
of the CRITICAL/HIGH gate's scope; acl/attr pair documented in the table).

### Harm-gate hardening — measured, multilingual, obfuscation-resistant, governed

A focused hardening round on the weapons/mass-casualty uplift gate, turning
hand-set heuristics into measured ones and inert controls into real ones. All
native (stdlib + numpy); the reasoning-backed classifier is fail-open and
offline-safe (no new hard dependency).

- **Reasoning classifier wired by default** on the open-web/text surface
  (`GeneralAssistant`, `WebResearcher`, output gate). Contributes a meaning-level
  harm probability only when a real local/cloud model is serving and `0.0`
  otherwise, so it can only raise harm and never changes deterministic behaviour
  in air-gapped/CI runs. `_parse_probability` hardened against prose/echo replies.
- **Obfuscation resistance + multilingual routing** (`cognitive/harm_normalization.py`):
  matching runs over homoglyph/leetspeak-folded, zero-width-stripped, and both
  whole- and word-boundary-preserving separator-collapsed variants (`n3rv3 ag3nt`,
  Cyrillic spoofing, `s a r i n`, `n.e.r.v.e a.g.e.n.t`); 266 taxonomy-level hazard
  terms + 52 offensive cues across ~30 languages (native script + transliteration).
- **Out-of-lexicon false-negative fix**: a production/acquisition verb anchored to
  generic hazard-context vocabulary now gates even when the specific agent is not
  in the Axis-A lexicon; `how to make/synthesize/produce X` and many indirect
  production/weaponization framings (previously missed) now block, all FP-safe.
- **Semantic-embedding accretion** replaces exact-domain counting in the aggregate
  tracker; the realized-plan re-gate is adjacent-window and capped to ESCALATE, so
  a benign production-verb query beside an unrelated hazard query never refuses a
  professional.
- **Cross-sentence output gate**: a procedure assembled across individually-safe
  sentences is redacted; defensive context beside a redacted step is preserved.
- **`ALLOW_PROVENANCE` disposition** (new): a high-severity hazard-domain query is
  answered only from cited sources; the output boundary withholds uncited
  hazardous-topic synthesis.
- **Human-in-the-loop / bounded-autonomy escalation** (`cognitive/escalation.py`):
  ESCALATE routes to an injectable reviewer, fail-closed with no reviewer, capped
  per session, and durably audited.
- **Durable decision/refusal audit log** (`cognitive/gate_audit.py`): append-only,
  fsynced JSONL sink (+ optional hash-chained `SecureAuditLogger`) for every
  refusal / escalation / provenance-withhold / accretion signal.
- **Measured FP/FN + fitted confidence**: a 362-case labeled corpus
  (`benchmarks/weapons_gate_corpus.py` + `.jsonl`, 60/20/20 split); the confidence
  logistic is fit on it (`scripts/fit_weapons_gate_calibration.py` →
  `configs/weapons_gate_calibration.json`; val Brier ≈0.002, ECE ≈0.039); a
  CI-failing FP/FN metric (`tests/ethical/test_weapons_gate_eval.py`; 0% FP / 0% FN
  on held-out val+test); and hypothesis property/fuzz + adversarial-paraphrase
  tests (`tests/ethical/test_weapons_gate_properties.py`).
- **Pre-existing items resolved**: `statistical_vlm` install hint now names both
  `transformers` and `accelerate`; the docstring-retention CI gate uses a 60%
  retention floor instead of a brittle no-shrink rule; the `memoryview[int]`
  annotation in `infrastructure/streaming.py` is confirmed correct (matches
  redis-py's generic `memoryview[int]`) with its misleading comments corrected.

### Neuro-symbolic calibration & transparency engineering + general-purpose capabilities

Make Mercury's confidence *measured* rather than heuristic, harden the
neuro-symbolic and ethics paths, scope-label uncalibrated heuristics, and add a
native general-purpose capability layer — all native (stdlib + numpy), no new
dependencies, no language-model service.

**Measured calibration.** New single routing point
`core/confidence.py::CalibratedConfidence` turns a raw score into a calibrated
probability on a held-out split with an R4 accept-gate (never regresses
Brier/ECE) and reports both. A calibrator is accepted **only** on a genuine
held-out evaluation; when the data is too imbalanced to split (a single-sample
class) the routing point stays identity rather than fitting and accepting
in-sample, where the no-regression gate would reward overfitting. The
golden-ratio `exp(-φ·total)` confidence in
`cognitive/uncertainty.py` is removed in favour of a parameter-free monotone
prior routed through the calibrator; epistemic/aleatoric carry transparent
`*_measured` flags (the single-scalar path is flagged *unmeasured* instead of
emitting a `0.1` placeholder as a measurement) and a new `ensemble_predictions`
path gives measured cross-member variance. The decider's uncalibrated
threshold-band fallback routes through an attached calibrator when present
(wired via `engine.enable_decision_layer(confidence_calibrator=...)`).

**Transparent evaluation.** `evaluation/metrics.py` / `metrics/anomaly_metrics.py`
gain a seeded stratified/temporal 3-way splitter, `fit_threshold` (tune on
val), and `evaluate_anomaly_detection_split` / `compute_all(tune_on="val")` so
operating-point metrics are reported on a disjoint test split — no more
threshold tuning on the reported data. `BenchmarkEvaluator` defaults to the
transparent path. Legacy in-sample functions kept and documented as diagnostics.

**Drift recalibration.** `engine.enable_online_recalibration()` wires
`AdaptiveConformalInference` into `detect_with_fusion` so the operating
threshold tracks score drift instead of only deferring.

**Perf + correctness.** blake3 import hoisted; FeatureCache key no longer
materializes the whole array (and avoids the torch host copy); kinematic
sliding-max vectorized; residual-FFT memoized; distributed `ResultAggregator`
drops dead "weights" and reassembles disjoint partitions in input order;
`detect_with_fusion` is fail-loud on an unfit detector by default
(`require_explicit_fit`, opt-out for the legacy auto-fit).

**Neuro-symbolic.** Hard symbolic veto (`SymbolicRule.hard`) lets an agreeing
high-confidence rule override a low neural score (monotone, never bypasses the
gates); a new `FusionMode.CONJUNCTIVE` (weighted geometric mean over undiluted
symbolic evidence) is the default; the STACKING meta-learner conditions on
disagreement and `update_from_outcome` learns from labels; the static,
mislabeled "Lyapunov" benevolence damping is removed. Hard-rule confidence is
clipped to `[0, 1]` on the veto path and the fused score is clamped after the
veto, so a malformed rule cannot push an anomaly score past the `[0, 1]`
invariant.

**Detector gating.** Kinematic is gated behind a real Ljung-Box white-noise test
(tightening only); optional per-component isotonic calibration (default-off).

**Ethics gate.** Layered, fail-closed harm matching: exact-substring keywords,
char-trigram **morphological** matching (catches inflection/spelling — *not*
meaning, and no longer mislabeled "semantic"), a curated **euphemism/paraphrase
lexicon** for meaning-level intent ("put him down", "make them disappear") tuned
for high precision so it never over-blocks Mercury's own defensive language
("kill the process", "neutralize the threat"), and an **optional pluggable harm
classifier** (`BenevolenceScorer(harm_classifier=…)`) that can ride Mercury's own
offline Ollama backend via `reasoning_harm_classifier(...)` — every layer can
only RAISE harm. Severity × irreversibility multiplicative damping (fail-closed);
calibratable weights in a `BenevolenceCalibration` dataclass.

**Weapons / mass-casualty uplift gate (two-axis, not a topic blocklist).** New
`cognitive/ethical_bounding.py::assess_weapons_uplift` gates on *operational
intent* (Axis B: mechanism/detection/treatment/response/policy/licensed-practice
→ ALLOW; production/weaponization/acquisition-evasion/offensive-enhancement/
targeting → REFUSE) routed by a high-recall hazard-domain filter (Axis A), with a
calibrated ladder (ALLOW → ALLOW_LOG → ESCALATE → REFUSE_REDACT → HARD_REFUSE).
Folded into the single `BenevolenceScorer`/`HarmReducer` hard gate (blocking
disposition raises PHYSICAL/SOCIETAL harm and hard-vetoes `is_permissible`,
monotone), and enforced in depth across the general-capability layer:
pre-retrieval query gate, post-retrieval content gate (verdict travels with
`FetchResult`), pre-emission verbatim-sentence redaction
(`ExtractiveSynthesizer` sentence gate), and an orchestration-boundary aggregate
gate (`SessionActionabilityTracker`) against query decomposition. Paired
red-team / professional CI suite (`tests/ethical/test_weapons_uplift_gate.py`)
pins both false-negative and false-positive rates; full policy, response ladder,
and residual-risk statement in `docs/HARM_POLICY.md`. `RULESET_VERSION` bumped to
4 (cache invalidation).

**Transparency labels.** `bain_ai_scaling` power estimate, the ~82 uncomputed
`global_omni_scalar_network` diagnostic scalars + its untrained attention,
`hierarchical_planning` (template/greedy, not search), and `multi_hop_reasoner`
abduction (Jaccard, not Bayesian) are scope-labeled in-code and in
`docs/DORMANCY_LEDGER.md` §8.

**General-purpose capabilities** (`agentic/capabilities/`,
`docs/GENERAL_CAPABILITIES.md`). Mercury gains native, harm-gated web research
(`WebResearcher` — stdlib `urllib`/`html.parser`, fail-closed), extractive
synthesis (`ExtractiveSynthesizer` — numpy-only, quotes sources verbatim),
document generation (`DocumentGenerator` — Markdown/HTML/text), and a
`GeneralAssistant` (research → synthesize → cited document), surfaced on
`MercuryAgent.research/answer/write_document`. Search is a **provider ladder**,
tried in order and fail-closed, configurable in code (`search_providers=[…]`) or
from the environment (`WebResearcher.from_env()`): a keyed engine
(`brave_provider`) and/or a *keyless, self-hostable* SearXNG (`searxng_provider`,
`MERCURY_SEARXNG_URL`) are the recommended, operator-owned rungs, with the
keyless DuckDuckGo HTML→lite scrape demoted to an explicit best-effort fallback
(`enable_ddg_fallback`, default on for zero-config installs, off for a
provider-only posture). The self-hosted SearXNG rung pairs with the local Ollama
reasoning backend for fully operator-owned, offline-leaning open-web research.
The legacy singular `search_provider` hook still fully replaces the chain.

**Universal interconnect (MCP server).** Mercury is now interconnectable: any AI
system that speaks the Model Context Protocol can link up to and run it through
`omni_mercury_engine.mcp_server` (`mercury-agent mcp`, or
`python -m omni_mercury_engine.mcp_server`). It serves JSON-RPC 2.0 over stdio
using the **standard library only** (no `mcp` package, no web framework, no new
dependency) and advertises Mercury's capabilities as discoverable, self-
describing MCP tools — `mercury_detect_anomaly`, `mercury_score_ethics`,
`mercury_research`, `mercury_answer`, `mercury_write_document`,
`mercury_calibrate_confidence` — whose `tools/list` payload doubles as a
machine-readable capability manifest (`MercuryMCPServer.manifest()`). Every tool
is transparent and fail-closed: outward tools pass the benevolence gate, an
unavailable backing stack returns an MCP `isError` result rather than a fabricated
capability, and the server inherits Mercury's offline posture (self-hosted SearXNG
search + local Ollama reasoning). See `docs/INTERCONNECT.md`.

**Fix pass: CI gates, a CLI regression, and two silent-correctness gaps.**
A full adversarial re-review (8 independent finder passes,
each candidate verified against the actual call sites) turned up and closed:

- *CI red.* `tests/test_search_provider_ladder.py`'s stub `__exit__` was typed
  `-> bool`, which `mypy --strict` correctly flags as exit-swallowing-prone
  (`exit-return`); retyped to `-> None`. Separately, the blocking Trivy image
  gate (`ignore-unfixed: false`) started failing on a clean diff because the
  live CVE feed moved since the `.trivyignore` ledger's last enumeration:
  `CVE-2026-54369`/`CVE-2026-54371` (libacl1/libattr1, local symlink-traversal
  privesc, no upstream fix) are newly published and weren't yet enumerated,
  while `CVE-2026-11822`/`CVE-2026-11824` (libsqlite3-0) were re-scored
  HIGH→MEDIUM upstream and dropped out of the gate's severity filter. The
  ledger is reconciled against a fresh scan of the shipped
  `python:3.14-slim-trixie` image (re-enumerated 2026-06-30; the file still
  read `python:3.13-slim-trixie` from before the 3.14 base-image bump).
- *CodeQL: incomplete URL substring sanitization
  (`web_research.py::_decode_ddg_href`).* The DuckDuckGo-redirect host check
  was `"duckduckgo.com" in parsed.netloc`, satisfied by a spoofed host such as
  `duckduckgo.com.evil.test` or `notduckduckgo.com`. Replaced with an exact
  host-or-subdomain match (`WebResearcher._is_ddg_host`); a non-matching host
  now falls through to the literal-href path instead of having its `uddg`
  redirect target trusted.
- *CLI regression.* `detect_with_fusion`'s new `require_explicit_fit=True`
  default (above) was threaded through every internal call site the PR
  description claims — except `cli.py`'s `detect -d fusion` and `explain`
  commands, which construct the engine with no override and never fit the
  base detectors before calling `detect_with_fusion`. Both commands would
  raise `RuntimeError` on every invocation. Fixed by passing
  `require_explicit_fit=False` at those two call sites (the CLI has no
  train/test split to fit detectors on; it restores the documented legacy
  auto-fit-on-first-batch behavior those commands always relied on).
- *Decider: calibrated confidence not flipped for CLEAR verdicts*
  (`decision/decider.py`). With a fitted `confidence_calibrator` attached,
  `_from_threshold_band` reported the calibrated **P(anomaly)** as
  `decision_confidence` regardless of which label was chosen — so a
  confidently-benign CLEAR call (e.g. `P(anomaly)=0.03`) surfaced as
  near-zero confidence, the opposite of the uncalibrated margin fallback's
  symmetric behavior. `decision_confidence` is now `P(anomaly)` for ACT and
  `1 - P(anomaly)` for CLEAR, consistent with every other path in this
  module reporting confidence *in the verdict*, not in a fixed direction.
- *FeatureCache: silent wrong-cache-hit on large arrays* (`engine.py`).
  The new 256-point strided sampling key (avoids `tobytes()`-copying the
  whole buffer) folded in shape/dtype/size but no information about the
  ~99% of elements outside the sampled stride — so a change confined to an
  unsampled index (e.g. one element of a streaming window update) produced
  an identical key, and `get_or_compute` would silently return stale
  features for genuinely different data. Folded a full-array `np.sum`
  reduction into the key: still no buffer copy (unlike `tobytes()`), but a
  vectorized O(N) reduction is cheap enough not to undo the sampling's perf
  win while catching off-sample changes the stride alone misses.
- *Unpinned-dependency mypy drift* (`infrastructure/streaming.py`). Building
  the full toolchain locally (native AMA, then `mypy --strict` against a
  freshly resolved `redis>=5.0.0`) surfaced 9 errors a CI run pinned to an
  older cached `redis` doesn't yet hit — the same class of gate fragility as
  the `.trivyignore` reconciliation above, here in a Python stub instead of
  an OS CVE feed. Underneath the stub noise was a real type-transparency gap:
  `StreamMessage.offset` was typed `int | None`, but Redis Streams identifies
  entries by string IDs (`"<ms>-<seq>"`), not integers; the field silently
  carried a `str` through an `int`-typed slot on the Redis consume path.
  Widened to `int | str | None` and added a narrowing assert at the one call
  site (Kafka's `commit()`) that does integer arithmetic on it.
- ***σ_Immutable artifacts were never shipped by a real install*** — found
  by actually building the native AMA backend and running
  `mercury-agent detect -d fusion` end to end instead of relying on CI alone.
  `pyproject.toml`'s `[tool.setuptools.package-data]` declared only
  `py.typed` and `models/checkpoints/*.pt`; the σ_Immutable hard ethical
  gate's corpus (`sigma_immutable_corpus.json`), its Ed25519/ML-DSA-65
  signature bundle (`sigma_immutable_corpus.sig.json`), and its trained
  GOSNN weights (`sigma_immutable_weights.pt`) were not declared. CI's
  `pip install -e .` (editable) resolves `Path(__file__).parent`-relative
  loads against the source checkout regardless of package-data declarations,
  so this was invisible to every CI job — but the Dockerfile's
  `pip install ".[all]"` (no `-e`) and any real `pip install mercury-agent`
  install a package missing all three files. The gate's own fail-closed
  design then turns that into `EthicalConstraintViolationError` on the
  *first real engine construction*, not at import time, so
  `import omni_mercury_engine` and the container's `HEALTHCHECK` both
  report healthy while every actual detection call fails closed. All three
  files are now declared; `tests/test_package_data_completeness.py` pins the
  general invariant (every non-`.py` runtime data file under
  `src/omni_mercury_engine/` must be covered by a package-data glob) so a
  future file can't reopen this gap silently.

### Subagent pantheon: a 33-member internal delegation fleet for Mercury Agent

Mercury Agent becomes the AI centerpiece that hosts the combined agentic
capabilities of the sibling platforms via a new internal-only **subagent fleet**
(`src/omni_mercury_engine/agentic/subagents/`): the root agent delegates
arbitrary tasks to **33 named, full-capability subagents** — a Greek pantheon
(`Themis_I` … `Rhea_XXXIII`) — singly, in a batch, or to many replicas at once,
under hard governance and the dual ethical gate. The capabilities formerly
carried by FINDΩYOU™'s agent layer are transferred here as first-class deep
specializations as that platform is made agent-free. No public API surface
changes; the fleet is reachable only through the engine-mediated path. The
naming convention is Mercury-only (FINDΩYOU™ may adopt a similar pattern later).

- **Capability parity, not wrappers.** `SubAgent` subclasses `MercuryAgent`, so
  every member carries the full planning / reasoning / memory / tool toolkit; the
  internal `_generalist` routing floor runs the complete `analyze` pipeline.
- **Omni-Code anchors.** Each member is anchored to exactly one of the Seven
  Omni-Codes; the anchor's helical stability sets the member's autonomy ceiling
  via `compute_ethical_autonomy` (capped 0.95) — a real, monotonic binding to the
  constellation shared with AMA Cryptography. Seven members are code-bearers (one
  lead per Code). The root agent is governed by all seven.
- **Two real depth tiers (no theater).** `deep` members carry bespoke logic
  (`Themis_I` ethics, `Hera_VII` compliance, `Ares_XIV` guardrail, and the
  detection bridge `Zeus_VIII`/`Dionysus_XIII` over the real
  `MultiAgentOrchestrator`); `coordinator` members bind to and exercise their
  real `omni_mercury_engine` subsystem(s), reporting genuine availability and
  failing closed when a subsystem is absent — never fabricated output.
- **Internal-only access boundary.** Not re-exported from the public
  `omni_mercury_engine` surface; `SubAgent`/`SubAgentRegistry`/`SubAgentFleet`
  require a package-private access sentinel (`SubAgentAccessError` otherwise), and
  are reachable only via `OmniMercuryEngine.enable_subagent_fleet` /
  `MercuryAgent.enable_fleet` (`delegate` / `delegate_masses`).
- **Autonomy governor.** Fail-closed capability ceiling (replicas / total-active /
  recursion depth), Omni-Code autonomy cap, corrigibility pause/resume +
  irreversible kill-switch, and a failure-rate tripwire (ethical refusals counted
  as correct, never as failures).
- **Dual-gate commit boundary.** Results commit through the same benevolence-floor
  + σ-Immutable gate used on the engine and orchestrator boundaries; fail-closed.
  The per-task benevolence gate authorizes the controlled defensive action
  (mirroring `MultiAgentOrchestrator._enforce_ethics`) rather than scoring caller
  text — so it neither over-blocks defensive vocabulary nor is steered by
  adversarial phrasing; untrusted content is screened by the `Ares_XIV` guardrail.
- **Terminology.** Omni-Codes only — no other code system is referenced in this
  subsystem (enforced by a roster test).
- **Tests & docs.** `tests/test_subagent_roster.py`, `test_subagent_fleet.py`,
  `test_subagent_governor.py`, and `test_subagent_specializations.py` (43 tests)
  pin the roster invariants, access boundary, routing, real delegated detection,
  coordinator subsystem binding, mass-dispatch aggregation with surfaced
  failures, governor enforcement, the dual-gate commit, and each deep member's
  real ported logic. New code is mypy-strict and ruff clean; the canonical roster
  is documented in `docs/SUBAGENT_PANTHEON.md`; the CI-gated README
  codebase-scale block is regenerated.

### Subagent pantheon phase 2: coordinators deepened from binding to real operators

The 28 `coordinator` members are deepened from subsystem *bindings* (import +
introspect) into genuine subsystem **operators**: each now invokes its real
`omni_mercury_engine` entrypoint. The five deep members (`Themis_I`, `Hera_VII`,
`Zeus_VIII`, `Dionysus_XIII`, `Ares_XIV`) are unchanged. No public API surface
changes; the fleet remains reachable only through the engine-mediated path.

- **Operations adapter layer.** New
  `src/omni_mercury_engine/agentic/subagents/operations.py` maps each pantheon id
  to an adapter; `CoordinatorSubAgent._perform` dispatches to it, invoking the
  real entrypoint with `task.payload`-derived inputs and returning the transparent
  result (`output["mode"]="operation"`). Examples: `Helios_XVII` →
  `metrics.AnomalyMetrics.compute_all`; `Kronos_XXII` →
  `detectors.MercuryAnomalyDetector.fit().detect()`; `Tyche_XX` →
  `decision.DecisionAbstentionResponder.decide`; `Eos_XVIII` → native-JWT
  create+validate in-process; `Hecate_XXVIII` → `RequestRouter.match` in-process;
  `Atlas_XXX` → `DistributedMercuryCluster.detect_anomalies` via bounded
  `asyncio.run`; `Pan_XXV` → `GlobalOmniScalarNetwork` register + global score.
- **Transparent friction handling.** `Artemis_VI` genuinely attempts a bounded network
  fetch over real data sources and reports true per-source reachability (green or
  red, never fabricated). `Nyx_XXIX` (crypto seal primary; `MercuryCrypto`
  signing-keypair secondary) and `Prometheus_XXVII` (lightweight scorer primary;
  `MercuryAutoML.fit` heavy/budget-gated) keep the always-safe primitive as the
  primary path.
- **Binding report is the transparent floor, not the whole behavior.** The
  import+introspect report is retained only as the fail-closed fallback —
  reached when an input-gated member lacks inputs or a caller requests the
  readiness probe (`payload["mode"]="introspect"`) — and as the unavailability
  surface (`output["mode"]="binding"`). Malformed inputs fail closed
  (`SubAgentExecutionError`); no fabricated output anywhere.
- **Roster tweaks.** `Harmonia_XXXI` and `Hades_XV` gain the `utils` subsystem
  (the real `normalize_data` / `compress_information` operator entrypoints live
  there); `validate_roster()`'s subsystem-resolution invariant re-passes.
- **Tests & docs.** `tests/test_subagent_operations.py` (69 tests) proves, for
  every coordinator, a real-op path (valid inputs, asserts the real entrypoint
  ran — not the fallback) plus a no-input fallback path, with fail-closed cases
  for malformed inputs and a gap guard asserting all 28 coordinators have an
  adapter and a real-op test. The full subagent suite is 112 tests. New code is
  mypy-strict and ruff clean; `docs/SUBAGENT_PANTHEON.md` gains an
  operation + payload-inputs table; `ARCHITECTURE.md` notes the binding→operator
  deepening; the CI-gated README codebase-scale block is regenerated.

### Deployability hardening: importable image, real streaming worker, reproducible benchmarks, license-clean

A correctness pass that makes Mercury actually deployable and the benchmark
table reproducible, without weakening the PQC posture.

- **AMA native PQC backend now builds into the image.** The runtime container
  previously ran `pip install ".[all]"` but never built AMA's native C library,
  so `import omni_mercury_engine` (the mandatory import-time PQC gate) — and
  therefore the Docker `HEALTHCHECK` and the k8s liveness/readiness probes —
  would fail. Added `scripts/build_ama_native.sh` (clone pinned `v3.2.0`, cmake
  `-DAMA_USE_NATIVE_PQC=ON`, install, **co-locate `libama_cryptography.so` inside
  the installed package** so it loads with no `LD_LIBRARY_PATH`, verify ML-DSA-65
  + Kyber-1024 + SPHINCS+ or fail loud), wired into the Dockerfile builder stage
  with a build-time `import omni_mercury_engine` smoke test. The PQC gate is
  unchanged — AMA stays a hard dependency. `docs/INSTALLATION.md` updated.
- **Real `stream` worker command; broken engine deployment fixed.** The k8s/Helm
  engine deployment ran a non-existent `serve --mode worker` and contained invalid
  YAML (`- name: OMNI_WORKER_MODE: "true"`) that broke `kubectl apply`. Added a
  `mercury stream` command wrapping the existing `StreamingAnomalyPipeline`
  (consume → detect → publish, circuit breaker, graceful SIGTERM drain, Prometheus
  `/metrics` on `:9090`), repointed the base/Helm engine deployments and the
  distributed streaming-worker fleet at it, added the streaming config to the
  ConfigMap/values, and removed the phantom `OMNI_WORKER_MODE`. `docs/DEPLOYMENT.md`
  documents the API + streaming-worker tiers.
- **Reproducible empirical benchmark; synthetic shim removed.** Deleted the
  ungated synthetic-data fabricator in `benchmarks/empirical_benchmark.py` (it
  bypassed the `MERCURY_ALLOW_SYNTHETIC` policy and made runs non-reproducible) —
  unavailable datasets are now skipped, never faked. Added a deterministic
  argparse CLI (`python -m benchmarks.empirical_benchmark --readme-subset`) so the
  README "Near-Peer Baselines" table is regenerable from license-clean
  scikit-learn datasets; the results JSON stays a CI artifact (repo stays lean).
- **License made consistent (`GPL-3.0-or-later`).** Normalized `pyproject.toml`,
  `rust_crypto/{Cargo.toml,pyproject.toml}`, the Dockerfile OCI label, `__license__`,
  README/CONTRIBUTING and the Rust source headers off the mixed `GPL-3.0+` / `GPL-3.0`
  variants. Extended `scripts/normalize_headers.py` to check/apply the canonical
  `// Copyright / // SPDX` header on `rust_crypto/src/*.rs` and wired it into the
  same pre-commit hook and CI gate as the Python headers.
- **Observability wired.** Added `prometheus-client` to the `[api]` extra and
  exposed the `prometheus_client` registry through the API `/metrics` endpoint
  (alongside the existing health gauges). The API now emits
  `http_requests_total` / `http_request_duration_seconds` on every request via
  `CorrelationIDMiddleware` — the series `monitoring/prometheus/prometheus-rules.yaml`
  and the API HPA custom metrics consume (previously they had no data source).
- **Consistency reconciliation.** Bumped the stale k8s/Helm `1.7.0` version
  stamps to `2.0.0`; corrected the benchmark reproducibility notes to the
  committed run's 66 successful / 9 failed (NOAA ERDDAP recovered); fixed an
  import-time CLI crash where a bad `MERCURY_METRICS_PORT` could take down every
  `mercury` subcommand; and hardened the Rust header normalizer against
  relabelling a vendored third-party license.

### Statistical ensemble: fusion-margin sharpening, data-type gate correction, temporal robustness

A measurement-driven pass on `MercuryAnomalyDetector` (`detectors/statistical.py`)
that sharpens the unsupervised fusion weights, corrects a data-type
misclassification that mistreated tabular data as temporal, and adds opt-in
temporal robustness. All changes are label-free and validated on a committed,
re-runnable 18-set real-ADBench transductive harness
(`research/omni_equation/harness_adbench.py`; AUROC via Mann-Whitney). Net
transductive Mean AUROC **0.7397 → 0.7634 (+2.37 pts, 14 W / 2 tie / 2 L)** —
base `e118e1f` vs the hardened detector, full per-set results in
`research/omni_equation/adbench_results.json`; the two losses are within noise
(≤0.0003 AUROC). This supersedes an earlier ad-hoc `0.7804 → 0.8142` figure that
is **not reproducible**: its harness resolved several sets through the now-broken
ADRepository mirror, which silently substitutes synthetic data. Re-run:
`python research/omni_equation/harness_adbench.py`.

The headline is now backed by committed artifacts, not just an assertion: the
pre-hardening base run is committed at `research/omni_equation/adbench_base_e118e1f.json`,
and the per-dataset win/tie/loss ledger at
`research/omni_equation/adbench_base_vs_current.json`, regenerated by the harness's
new `--baseline` flag. The diff lives in the engine-free
`research/omni_equation/_compare.py` and is pinned by
`tests/research/test_adbench_baseline_compare.py` (14 W / 2 tie / 2 L; the two
sub-noise losses, Waveform/WPBC, are counted as losses, not rounded into ties).
The full per-set table is in `docs/BENCHMARKS.md`.

* **Sharpen the unsupervised fusion-weight margins (`_WEIGHT_MARGIN_POWER = 4.0`,
  new module constant).** The self-supervised separation AUCs that drive the
  unsupervised weighter saturate into a narrow 0.72-1.0 band on easy 3σ
  Gaussian-blob synthetic anomalies, so a *linear* AUC-separation margin leaves
  the consistently weakest component — kinematic, last on 18/18 ADBench sets —
  with 30-42% of the ensemble weight. Each surviving margin `(auc - 0.5)` is now
  raised to a power `P` before normalisation, in both
  `_compute_unsupervised_adaptive_weights()` and the supervised
  `_compute_adaptive_weights()`. `P=1` reproduces the prior behaviour exactly
  (regression-tested). On the committed harness the biggest wins land where
  kinematic was most over-weighted (wine, glass, PageBlocks, cardio — all wins
  vs base).

* **Fix the data-type gate so tabular data is not mislabeled temporal
  (`_detect_data_characteristics`).** The previous single-signal gate
  (autocorrelation **or** adjacent-row correlation > 0.3) classified 13/18
  shuffled-tabular ADBench sets as `TEMPORAL` on spurious lag-1 autocorrelation
  or class-ordering adjacency, handing KinematicScore the largest compatibility
  weight. The gate now requires *strong* evidence on either of two complementary
  signals — median lag-1 autocorrelation > `_TEMPORAL_AUTOCORR_THRESHOLD` (0.55)
  **or** median adjacent full-row coherence > `_TEMPORAL_ADJ_ROW_THRESHOLD`
  (0.75) — thresholds that sit in the empirical gap between the tabular cluster
  (autocorr ≤0.50, adjacency ≤0.67) and the temporal cluster (autocorr ≥0.60,
  adjacency ≥0.82, including spike-contaminated series whose per-feature
  autocorr is masked but whose adjacency stays ~0.99). Genuine temporal data
  (sine / AR(1) / random-walk fixtures, the existing
  `test_temporal_data_detected` case, and real SMD telemetry — autocorr 0.91)
  stays `TEMPORAL`; the existing `TABULAR → kinematic=0` hard-zero is untouched.

* **Restrict the temporal residual-frequency filter to genuinely temporal data
  (`detect()`).** The Phase-7 `_residual_frequency_filter` treats the score
  vector as an ordered time series (moving-average baseline + rFFT band-pass), so
  it is only meaningful when row order is meaningful. It was gated on a shape
  proxy (`n ≥ max(50, 10·d)`) that also fires on wide/long *tabular* sets, where
  it is rank-altering — measured to drag optdigits from 0.52 to **0.39 (below
  random)**. It is now gated on the fitted `_data_type == TEMPORAL`. This both
  removes that harm and makes detection **robust to test-row ordering**: on the
  inductive benchmark protocol the prior code scored 0.8828 on a class-grouped
  test set but only 0.8722 once the test rows are shuffled (it was exploiting the
  label ordering), whereas the corrected detector scores 0.8792 / 0.8800 — flat.

* **Opt-in inference-time multi-scale "time-dilation" TTA (DEFAULT-OFF).**
  New `multiscale_tta` / `multiscale_tta_pool` config flags. When enabled and the
  fitted data is `TEMPORAL`, `detect()` scores the series at
  `_MULTISCALE_TTA_SCALES = (0.8, 0.9, 1.0, 1.1, 1.25)` time-dilations
  (`np.interp` resample), maps each per-sample score vector back to the original
  length, and pools across scales (`mean`, the false-alarm-safe default, or
  `max`). DEFAULT-OFF is byte-identical to the prior output (Invariant I2). Pure
  inference, no fit change, K× scoring compute. Validated on real SMD
  machine-1-1 collective anomalies: mean-pool **+5.1 pts** clean (0.565 → 0.616)
  and **+5.8 pts** under a 0.85× rate change; on controlled collective anomalies
  max-pool wins 8/8 (+0.66 pts).

* **Sampling-jitter robustness characterization
  (`tests/detectors/test_sampling_jitter_robustness.py`, new).** Clock-skew /
  rate-drift on a temporal feed degrades AUROC −5 pts (eps 0.10) to −16 pts (eps
  0.20) — a documented limitation, not an invariance claim. The new test asserts
  a *tolerance band* (bounded AUROC loss, score-ranking correlation above a
  floor) so robustness regressions fail loudly, and pins the multi-scale TTA
  invariants (default-off byte-identity, temporal-only gating, valid pooled
  scores, collective-anomaly improvement).

### Data loaders: close the ADRepository silent-synthetic fallback, repoint the moved mirror

Removes the last silent synthetic-data fallback on the dataset layer — the exact
foot-gun that contaminated the superseded `0.7804 → 0.8142` headline. The
`ADRepositoryLoader` previously called `_create_synthetic_fallback()` directly in
three `except`/fallback sites with **no policy check**, so a moved mirror or an
unreachable host returned fabricated data that downstream code could not
distinguish from real signal. Every other dataset loader already routed synthetic
generation through `check_synthetic_allowed()`; ADRepository was the sole holdout
(confirmed by a full per-loader audit).

* **Fail loud by default (root-cause fix).** `_create_synthetic_fallback()` is now
  the single synthetic chokepoint and self-gates on `MERCURY_ALLOW_SYNTHETIC` via
  `check_synthetic_allowed()`. With the policy off (the default) it raises
  `DataSourceUnavailableError` with context; with `MERCURY_ALLOW_SYNTHETIC=1` it
  honours the opt-in and marks the result `is_real_data=False` / `synthetic=True`.
  No production path can fabricate data without the explicit deployment opt-in.
* **Repoint the moved mirror.** The ADRepository corpus moved
  `GuansongPang → mala-lab`; the old `github.com/.../raw/main/` form 301-redirects
  to `raw.githubusercontent.com`, which the SSRF-safe HTTP client refuses to
  follow — the failure that triggered the silent fallback. The loader now pins
  directly to the `raw.githubusercontent.com` mala-lab path (already on the
  trusted-domain allowlist, no redirect) and uses the real DevNet filenames
  (`annthyroid_21feat_normalised.csv`, `creditcardfraud_normalised.tar.xz`, …),
  with a single canonical source per dataset so a name always resolves to the
  same composition. `.tar.xz` archives are now supported alongside `.csv`/`.npz`,
  with path-traversal (tar-slip / zip-slip) guards on both extraction paths.
* **Harden the archive-extraction helpers (review follow-ups).** `_safe_extract_tar`
  now feature-detects `tarfile.data_filter` and falls back to an explicit member
  guard on the 3.11.0–3.11.3 patch releases that lack the `filter=` kwarg (where
  it would otherwise raise `TypeError` straight into the synthetic path);
  `_safe_extract_zip`'s docstring no longer implies the stdlib sanitises member
  paths (this helper *is* the sanitiser); and both archive paths clear any stale
  reuse directory before extracting, so `rglob('*.csv')` cannot pick up a leftover
  from a previously-extracted archive with a different layout. Covered by a new
  `TestArchiveExtractionSafety` (tar/zip-slip refusal, symlink refusal, the
  filter-absent fallback, stale-dir clearing).
* **Remove the dead ODDS path.** The ADRepository loader's legacy ODDS (Outlier
  Detection DataSets, Stony Brook) `.mat` codepath was removed — the host is
  unreachable (503) and no registered dataset name ever routed through it. Because
  nothing fetches from it, `odds.cs.stonybrook.edu` is deliberately **not** added
  to `TrustedEndpoints.TRUSTED_DOMAINS`, keeping the trusted-egress surface
  minimal; time-series sets without a dedicated loader fail loud naming their real
  upstream instead of routing to synthetic.
* **Tests.** `tests/datasets/test_adrepository.py` gains a `TestSyntheticPolicyGate`
  class (chokepoint raises when policy off; production load fails loud on an
  unreachable source; opt-in synthetic is labelled; time-series sets with no
  mirror fail loud) and a network-lane `TestADRepositoryRealMirror` that loads
  real thyroid (CSV) and backdoor (`.tar.xz`) from the repointed mirror with
  synthetic denied (real-or-raise). The transductive ADBench headline is
  unaffected — that harness uses the already-clean `ADBenchLoader`.

### Full inductive suite refreshed — real-data base-vs-current (75 attempted / 66 scored)

Re-ran the committed `benchmarks/mercury_benchmark.py` (inductive
normal-train/test-on-rest protocol) on the full 75-dataset registry, **real data
only** (`MERCURY_ALLOW_SYNTHETIC=0`) — base `e118e1f` vs the hardened current
tree. 9 datasets fail loud (unreachable upstream: SMAP, MSL, CICIDS-2017,
MIT-BIH, USGS_Geochemistry, FEMA_HazardMitigation, UCR, SWaT, WADI) and are
excluded, never synthetic-substituted; the ledger is identical in both runs. On
the deterministic ADBench subset the hardening is **0.8180 → 0.8251 (+0.0071,
18 W / 5 tie / 24 L)**; across all real datasets excluding the ADRepository
artifact, **+0.0029**. The naive all-sets number is a slight −0.0016 *because the
base commit's unfixed ADRepository loader silently returns synthetic data and the
detector scores a meaningless AUROC 1.0000 on it* — a live capture of the exact
foot-gun the loader fix closes (the hardened tree scores real thyroid 0.7086
there). The inductive deltas run smaller than the transductive +0.0237 because
the inductive protocol's class-grouped test ordering let the old temporal filter
exploit a row-order artifact the hardened detector correctly forgoes; both
protocols are net-positive once decontaminated. Full analysis + reproduction in
`docs/BENCHMARKS.md`; refreshed results in `benchmarks/mercury_benchmark_results.json`.

**De-leak follow-up (`632d3e5`).** The class-grouped test ordering is now closed
at the harness level — `X_test`/`y_test` are shuffled (fixed seed) after the cap,
so evaluation order carries no label information and no detector (base, current,
or future) can exploit it. The distortion is **material and bidirectional**: on
the 8-dataset deterministic regression guard (current detector, full
`MAX_SAMPLES`, grouped vs de-leaked) the **ensemble** AUC moved `pendigits`
0.761→0.874 (+0.113, grouping *deflated* it), `Pima` 0.741→0.705 (−0.036,
grouping *inflated* it), `glass` 0.745→0.768; guard mean 0.891→0.901. The
per-component **kinematic** AUC moves more still (max |Δ| **0.41** — `wine`
0.18→0.60; resonance / info-geometry are exactly order-invariant).
`benchmarks/anomaly_regression_baseline.json` is re-pinned on the de-leaked eval;
the full inductive base-vs-current table in `docs/BENCHMARKS.md` is superseded and
must be regenerated on the de-leaked harness.

### CI: gate the engine import (and full PQC contract) at the AMA build seam

The `build-ama-cryptography` composite action (used by ~21 engine-importing jobs
across 11 workflows) built AMA and verified only `dilithium_available`, then
assumed "AMA built" implied "`omni_mercury_engine` imports". Two gaps closed:

* **Full PQC contract.** The engine's import-time gate
  (`_pqc_gate._PQC_HARD_REQUIRED`) requires ML-DSA-65 **+** Kyber-1024 **+**
  SPHINCS+, but the action checked only ML-DSA — a Kyber/SPHINCS-incomplete build
  passed it and surfaced later as an opaque collection error. The backend check
  now asserts all three.
* **Engine-import gate.** A new step imports `omni_mercury_engine` immediately
  after the build, so an engine-side import regression or an AMA/engine version
  skew fails loudly at the seam rather than 20+ minutes deep in a test lane.
  Self-gating via `find_spec` keeps the action reusable for pure-AMA jobs. No new
  CI jobs, no DAG/latency change.

## Earlier releases

Releases before 3.0.0 are in
[`docs/changelog/CHANGELOG-pre-3.0.md`](docs/changelog/CHANGELOG-pre-3.0.md),
moved there unchanged so this file stays readable.

| Release | Date |
|---|---|
| [2.0.0](docs/changelog/CHANGELOG-pre-3.0.md#200---2026-06-17) | 2026-06-17 |
| [1.7.0](docs/changelog/CHANGELOG-pre-3.0.md#170---2026-05-20) | 2026-05-20 |
| [1.6.0](docs/changelog/CHANGELOG-pre-3.0.md#160---2026-05-01) | 2026-05-01 |
| [1.5.1](docs/changelog/CHANGELOG-pre-3.0.md#151---2026-02-13) | 2026-02-13 |
| [1.5.0](docs/changelog/CHANGELOG-pre-3.0.md#150---2026-02-11) | 2026-02-11 |
| [1.4.0](docs/changelog/CHANGELOG-pre-3.0.md#140---2026-02-09) | 2026-02-09 |
| [1.2.0](docs/changelog/CHANGELOG-pre-3.0.md#120---2026-02-02) | 2026-02-02 |
| [1.1.2](docs/changelog/CHANGELOG-pre-3.0.md#112---2026-01-17) | 2026-01-17 |
| [1.1.1](docs/changelog/CHANGELOG-pre-3.0.md#111---2026-01-16) | 2026-01-16 |
| [1.1.0](docs/changelog/CHANGELOG-pre-3.0.md#110---2026-01-09) | 2026-01-09 |
| [0.1.0](docs/changelog/CHANGELOG-pre-3.0.md#010---2025-10-14) | 2025-10-14 |

[Unreleased]: https://github.com/Steel-SecAdv-LLC/Mercury-Agent/compare/v3.0.0...HEAD
[3.0.0]: https://github.com/Steel-SecAdv-LLC/Mercury-Agent/compare/2.0.0...v3.0.0
