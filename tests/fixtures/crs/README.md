# OWASP Core Rule Set regression payloads (extracted fixture)

`crs_v470_payloads.json` is extracted from the OWASP ModSecurity Core Rule
Set regression test suite, tag **v4.7.0**
(https://github.com/coreruleset/coreruleset, `tests/regression/tests/`),
licensed **Apache License 2.0** (Copyright OWASP Core Rule Set Project).

Extraction: for the rule families `REQUEST-942-*` (SQL injection, key
`sqli`), `REQUEST-941-*` (XSS, key `xss`) and `REQUEST-930-*` (local-file
inclusion / path traversal, key `traversal`), each test stage's payload
(`input.data`, the decoded query string of `input.uri`, and Cookie /
Referer / User-Agent header values) is collected under:

- `pos` — stages whose expected output asserts the family's rule fires
  (attack payloads);
- `neg` — stages whose expected output asserts it does **not**
  (`no_log_contains` / `no_expect_ids`): the CRS project's own
  false-positive regression payloads.  Note that a `neg` payload is only
  "this specific rule must not fire": some are genuine attacks scoped to
  other rules, so a detector's hit rate on `neg` is an upper bound on its
  false-positive rate, not an exact one.

Consumed by `tests/security/test_threat_crs_corpus.py`, which enforces the
measured recall floors and false-positive ceilings for
`omni_mercury_engine.security.threat_detection.ThreatDetector`.
