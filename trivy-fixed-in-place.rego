# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Trivy ignore policy for OS packages the Dockerfile has FIXED IN PLACE.
#
# This is not an acceptance ledger (that is .trivyignore). Every entry here
# names a CVE the shipped binary no longer contains: the Dockerfile's runtime
# stage installs Debian's own fixed build from a pinned snapshot, and the build
# proves the installed version. Trivy still reports these CVEs because it keys
# Debian findings on the image's release (13, trixie), whose advisory carries
# no fixed version, so every installed version reads as `affected` -- including
# the fixed one from Debian unstable.
#
# Each entry is bound to one exact installed version. If the package is ever
# at any other version -- trixie's vulnerable build, a later unstable build,
# a downgrade -- the policy matches nothing and the blocking gate reports the
# finding again. Any CVE not listed here is reported normally, so a new zlib
# or sqlite finding still fails the gate.
#
# Kept identical to the Dockerfile's *_FIXED_VERSION build arguments by
# tests/security/test_fixed_in_place_os_packages.py.
#
# Fix status (Debian Security Tracker, 2026-11-06):
#   zlib    -- bookworm, trixie vulnerable; forky, sid fixed 1:1.3.dfsg+really1.3.2-4
#              (madler/zlib@df84af25)
#   sqlite3 -- bookworm, trixie, forky vulnerable; sid fixed 3.54.1-2 (upstream 3.54.0)
#   acl     -- bookworm, trixie vulnerable; forky, sid fixed 2.4.0-1
#   systemd -- bookworm, trixie vulnerable; forky 262-1, sid 262.2-1 fixed (upstream 261.2)

package trivy

import rego.v1

default ignore := false

fixed_in_place := {
	"zlib1g": {
		"version": "1:1.3.dfsg+really1.3.2-4",
		"cves": {
			"CVE-2026-76844", # gz_write() stale next_in heap overflow
			"CVE-2026-85091", # gz_vacate() heap overflow
		},
	},
	"libsqlite3-0": {
		"version": "3.54.1-2",
		"cves": {
			"CVE-2026-119373", # fts5MergeRowidLists() heap overflow
			"CVE-2026-119374", # fts5DoSecureDelete() heap overflow (corrupted FTS5 index)
			"CVE-2026-119379", # sqlite3changeset_apply() heap overflow (session extension)
		},
	},
	"libacl1": {
		"version": "2.4.0-1",
		"cves": {"CVE-2026-54369"}, # acl_*_file() symlink-traversal privilege escalation
	},
	"libsystemd0": {
		"version": "262.2-1",
		"cves": {"CVE-2026-16742"}, # systemd-homed home-record signature verification
	},
	"libudev1": {
		"version": "262.2-1",
		"cves": {"CVE-2026-16742"},
	},
}

ignore if {
	entry := fixed_in_place[input.PkgName]
	input.InstalledVersion == entry.version
	input.VulnerabilityID in entry.cves
}
