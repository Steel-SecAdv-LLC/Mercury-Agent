Paste everything below the line into Claude Code, started in the root of your Mercury-Agent clone with `design_handoff_mercury_console/` unzipped there.

---

Implement the Mercury Console described in `design_handoff_mercury_console/README.md` and open a pull request.

Before writing code:
1. Read `design_handoff_mercury_console/README.md` in full.
2. Re-verify every "(repo fact)" section against this checkout: `src/omni_mercury_engine/api/frontend.py`, `api/security_headers.py`, `tests/api/test_frontend.py`, `tests/api/test_security_headers.py`, `pyproject.toml` package-data, `CONTRIBUTING.md`, `.github/PULL_REQUEST_TEMPLATE.md`. If anything differs, follow the repo and note the difference in the PR.
3. Open `design_handoff_mercury_console/design/Mercury CLI.dc.html` and the `design/mercury-cli-*.js` modules. They are the source of truth for layout, copy, exact style values and behaviour.

Then:
- Create branch `feature/web-console` from up-to-date `main`.
- Follow README §8 step by step: catalog generator, ported modules, `pages/console.html`, `static/console/app.js` + `console.css`, the `/console` route, the CSP decision for hazard images (prefer option a, path-scoped `img-src 'self' data:`), package-data, tests (§14), docs and CHANGELOG.
- Respect the CSP rules in §6.1: no inline scripts, styles, `style=` attributes or handler attributes in the page; no CDN; same-origin only; remove the 56 `style=` attributes from `viz.js`.
- Do not ship `mercury-cli-sim.js`, `support.js` or `_ds_bundle.js`.
- Do not change `EnvironmentDomain` in this PR (README §13 describes it as a separate follow-up).
- Run `bash scripts/run_ci_gates.sh` and `pytest tests/api -v` until everything passes; then run the full `pytest tests/ -v` if time allows and report results.
- Build a wheel and confirm the console files are packaged.
- Never commit `design_handoff_mercury_console/`.
- Commit with conventional commits as in README §16. Never write the skip-CI directive in any commit message or the PR body.
- Push and open the PR with `gh pr create --base main` using the repository's PR template sections, marking it AI-assisted. Include the test commands you ran and their results, and list anything you could not finish.

Stop and ask me only if a repo fact contradicts the README in a way that changes the plan.
