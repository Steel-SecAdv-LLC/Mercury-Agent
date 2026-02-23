-- Mercury Agent Database Schema
-- Copyright (C) 2025 Steel Security Advisors LLC
-- Licensed under GPL-3.0
--
-- Default backend: SQLite (no external deps for local/CI)
-- Override: MERCURY_DB_BACKEND=postgresql
--
-- Tables:
--   benchmark_runs    – top-level benchmark execution metadata
--   dataset_results   – per-dataset scores from each run
--   detector_state    – serialised detector / Oracle state
--   api_cache         – HTTP response cache with domain-aware TTL

CREATE TABLE IF NOT EXISTS benchmark_runs (
    id              TEXT PRIMARY KEY,           -- UUID
    timestamp       TEXT NOT NULL,              -- ISO-8601
    git_sha         TEXT,
    git_branch      TEXT,
    python_version  TEXT,
    mercury_version TEXT,
    config_json     TEXT,                       -- serialised run config
    total_datasets  INTEGER NOT NULL DEFAULT 0,
    active_datasets INTEGER NOT NULL DEFAULT 0,
    api_unavailable INTEGER NOT NULL DEFAULT 0,
    mean_auc        REAL,
    mean_f1         REAL,
    duration_seconds REAL
);

CREATE TABLE IF NOT EXISTS dataset_results (
    id              TEXT PRIMARY KEY,           -- UUID
    run_id          TEXT NOT NULL REFERENCES benchmark_runs(id),
    name            TEXT NOT NULL,
    domain          TEXT,
    loader          TEXT,
    status          TEXT NOT NULL DEFAULT 'success',  -- success | api_unavailable | invalid_data
    n_samples       INTEGER,
    n_features      INTEGER,
    anomaly_ratio   REAL,
    detected_type   TEXT,
    inferred_domain TEXT,
    auc             REAL,
    f1              REAL,
    precision_val   REAL,
    recall          REAL,
    progressive_mean_auc        REAL,
    temporal_leakage_detected   INTEGER DEFAULT 0,
    duration_seconds            REAL,
    error_message   TEXT
);

CREATE TABLE IF NOT EXISTS detector_state (
    id              TEXT PRIMARY KEY,           -- UUID
    detector_name   TEXT NOT NULL,
    version         TEXT,
    state_blob      BLOB,                      -- pickled / JSON state
    oracle_ref_stats TEXT,                      -- JSON serialised Oracle reference stats
    created_at      TEXT NOT NULL,              -- ISO-8601
    updated_at      TEXT NOT NULL               -- ISO-8601
);

CREATE TABLE IF NOT EXISTS api_cache (
    cache_key       TEXT PRIMARY KEY,
    domain          TEXT NOT NULL,
    response_blob   BLOB,
    content_type    TEXT DEFAULT 'application/json',
    created_at      TEXT NOT NULL,              -- ISO-8601
    expires_at      TEXT NOT NULL,              -- ISO-8601
    hit_count       INTEGER NOT NULL DEFAULT 0
);

-- Indices for common queries
CREATE INDEX IF NOT EXISTS idx_dataset_results_run_id ON dataset_results(run_id);
CREATE INDEX IF NOT EXISTS idx_dataset_results_domain ON dataset_results(domain);
CREATE INDEX IF NOT EXISTS idx_detector_state_name ON detector_state(detector_name);
CREATE INDEX IF NOT EXISTS idx_api_cache_domain ON api_cache(domain);
CREATE INDEX IF NOT EXISTS idx_api_cache_expires ON api_cache(expires_at);
