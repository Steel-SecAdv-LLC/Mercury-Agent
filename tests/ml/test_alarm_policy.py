# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``ml.alarm_policy``: the streaming worker's learned alarm policy."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.evaluation.nab_scoring import (
    NABStream,
    load_nab_corpus,
    score_alarms,
    score_corpus,
)
from omni_mercury_engine.ml.alarm_policy import (
    FEATURE_NAMES,
    GATE_FEATURE,
    HISTORY_NAMES,
    SHIPPED_POLICY,
    AlarmHistory,
    AlarmPolicy,
    StreamingAlarmDetector,
    TriageEnv,
    ValueFeatureState,
    default_training_config,
    export_alarm_policy,
    load_alarm_policy,
    nab_features_and_tables,
    policy_from_trainer,
    run_alarm_policy,
    stream_features,
    train_alarm_policy,
)

ROOT = Path(__file__).resolve().parents[2]
NAB_FIXTURES = ROOT / "tests" / "fixtures" / "nab"
RESULTS = ROOT / "benchmarks" / "streaming_alarm_policy_results.json"
CATEGORIES = ("realAWSCloudwatch", "realAdExchange", "realKnownCause", "realTraffic")


@pytest.fixture(scope="module")
def corpus() -> list[NABStream]:
    return load_nab_corpus(NAB_FIXTURES, CATEGORIES)


@pytest.fixture(scope="module")
def tiny_trainer(corpus: list[NABStream]) -> Any:
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    features, tables = nab_features_and_tables(corpus)
    cfg = default_training_config(0)
    cfg.n_steps, cfg.batch_size = 32, 128
    return train_alarm_policy(features, tables, gate=0.9, total_timesteps=1024, config=cfg)


# -- features --------------------------------------------------------------------------


def test_features_are_causal_and_warm_up_at_zero() -> None:
    rng = np.random.default_rng(0)
    series = rng.normal(0.0, 1.0, 400)
    full = stream_features(series)
    assert full.shape == (400, len(FEATURE_NAMES)) and full.dtype == np.float32
    assert not full[:10].any()  # every feature is 0 for the first 10 values
    for t in (15, 120, 399):
        assert np.array_equal(stream_features(series[: t + 1])[-1], full[t])  # no look-ahead
    assert stream_features([]).shape == (0, len(FEATURE_NAMES))


def test_features_score_against_the_past() -> None:
    rng = np.random.default_rng(1)
    series = rng.normal(0.0, 1.0, 300)
    series[250] = 40.0
    f = stream_features(series)
    assert f[250, GATE_FEATURE] == 1.0  # the largest value yet: fully extreme
    assert f[250, FEATURE_NAMES.index("z_fast")] > math.log1p(20)  # not diluted by itself
    assert f[250, FEATURE_NAMES.index("robust_z")] > math.log1p(20)
    assert np.all(f[:, GATE_FEATURE] <= 1.0) and np.all(f >= 0.0) and np.all(f <= 10.0)


def test_rank_extremeness_and_robust_z_definitions() -> None:
    state = ValueFeatureState()
    for v in range(20):  # 0..19
        state.update(float(v))
    f = state.update(10.0)
    # 10 sits at empirical CDF position (10 + 0.5) / 20, extremeness |2F - 1| = 0.05.
    assert f[GATE_FEATURE] == pytest.approx(0.05)
    # median 9.5, MAD 5 -> |10 - 9.5| / (1.4826 * 5).
    assert f[FEATURE_NAMES.index("robust_z")] == pytest.approx(
        math.log1p(0.5 / (1.4826 * 5 + 1e-9 * 10.5)), rel=1e-5
    )


def test_alarm_history_features() -> None:
    history = AlarmHistory()
    assert history.features(0) == [1.0, 1.0, 0.0, 0.0]
    history.record(100, alarm=True)
    history.record(150, alarm=False)
    since_alarm, since_candidate, candidates, alarms = history.features(160)
    assert since_alarm == pytest.approx(math.log1p(60) / math.log1p(10_000))
    assert since_candidate == pytest.approx(math.log1p(10) / math.log1p(1000))
    assert (candidates, alarms) == (2 / 20, 1 / 10)
    assert history.features(1200)[2:] == [0.0, 0.0]  # both windows have passed
    assert len(HISTORY_NAMES) == 4


# -- the training environment ----------------------------------------------------------


@pytest.mark.parametrize("choice", ["always", "seeded"])
def test_triage_return_is_the_nab_score(corpus: list[NABStream], choice: str) -> None:
    """An episode's return equals the stream's raw NAB score plus fn per window."""
    features, tables = nab_features_and_tables(corpus)
    for name in features:
        env = TriageEnv({name: features[name]}, {name: tables[name]}, gate=0.9, num_envs=1, seed=0)
        env.reset()
        rng = np.random.default_rng(3)
        alarms = np.zeros(features[name].shape[0], dtype=bool)
        total = 0.0
        for t in env.candidates[name]:
            action = 1 if choice == "always" else int(rng.integers(2))
            alarms[t] = bool(action)
            _, reward, terminated, _, _ = env.step(np.array([action]))
            total += float(reward[0])
        assert terminated[0]
        table = tables[name]
        assert total == pytest.approx(
            score_alarms(alarms, table).raw + table.profile.fn * table.n_windows, abs=1e-5
        )


def test_triage_env_refuses_a_gate_with_no_candidates(corpus: list[NABStream]) -> None:
    features, tables = nab_features_and_tables(corpus[:1])
    with pytest.raises(ValueError):
        TriageEnv(features, tables, gate=1.0)


# -- inference, export, loading --------------------------------------------------------


def test_numpy_policy_matches_the_torch_policy(tiny_trainer: Any) -> None:
    import torch

    policy = policy_from_trainer(tiny_trainer, gate=0.9)
    rng = np.random.default_rng(0)
    obs = rng.normal(0.0, 2.0, (500, len(FEATURE_NAMES) + len(HISTORY_NAMES))).astype(np.float32)
    normalized = tiny_trainer._process_obs(obs, update=False)
    with torch.no_grad():
        logits = tiny_trainer.policy._forward(
            tiny_trainer.policy.tensors, torch.as_tensor(normalized)
        )[1]
    expected = torch.softmax(logits, dim=-1)[:, 1].numpy()
    got = policy.alarm_probability(obs)
    assert np.allclose(got, expected, atol=1e-5)
    decisive = np.abs(expected - 0.5) > 1e-4
    torch_actions = tiny_trainer.predict(obs, deterministic=True)
    assert np.array_equal((got > 0.5)[decisive], (torch_actions == 1)[decisive])


def test_export_round_trip_and_tamper_refusal(tiny_trainer: Any, tmp_path: Path) -> None:
    path = tmp_path / "policy.npz"
    sidecar = export_alarm_policy(tiny_trainer, path, gate=0.9, provenance={"note": "test"})
    loaded = load_alarm_policy(path)
    direct = policy_from_trainer(tiny_trainer, gate=0.9)
    obs = np.ones(len(FEATURE_NAMES) + len(HISTORY_NAMES))
    assert loaded.alarm_probability(obs) == pytest.approx(direct.alarm_probability(obs))
    assert loaded.gate == 0.9 and loaded.provenance["note"] == "test"
    assert json.loads(sidecar.read_text())["checkpoint"] == "policy.npz"
    path.write_bytes(path.read_bytes() + b"\0")
    with pytest.raises(RuntimeError, match="does not match"):
        load_alarm_policy(path)
    with pytest.raises(FileNotFoundError):
        load_alarm_policy(tmp_path / "missing.npz")


def test_archives_that_are_not_policies_are_refused(tmp_path: Path) -> None:
    import hashlib

    path = tmp_path / "other.npz"
    np.savez(path, format=np.asarray("something.else"), version=np.asarray(1))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "other.provenance.json").write_text(json.dumps({"checkpoint_sha256": sha}))
    with pytest.raises(ValueError, match="not a Mercury alarm policy"):
        load_alarm_policy(path)


def test_run_alarm_policy_only_alarms_at_candidates(
    tiny_trainer: Any, corpus: list[NABStream]
) -> None:
    policy = policy_from_trainer(tiny_trainer, gate=0.9)
    features = stream_features(corpus[0].values)
    alarms = run_alarm_policy(policy, features)
    assert not alarms[features[:, GATE_FEATURE] <= 0.9].any()


# -- the shipped policy ----------------------------------------------------------------


@pytest.fixture(scope="module")
def shipped() -> AlarmPolicy:
    return load_alarm_policy()


def test_shipped_policy_is_verified_and_documented(shipped: AlarmPolicy) -> None:
    assert SHIPPED_POLICY.is_file()
    results = json.loads(RESULTS.read_text())
    assert (
        shipped.provenance["leave_one_category_out_mean_score"]
        == results["ppo_mean_score_all_folds"]
    )
    assert shipped.gate == pytest.approx(0.962)
    assert len(shipped.provenance["nab_sha256"]) == 48  # 47 files and the labels


def test_shipped_policy_beats_the_baselines_it_was_measured_against() -> None:
    results = json.loads(RESULTS.read_text())
    baselines = results["baselines"]
    best_on_test = baselines[results["best_single_feature_chosen_on_held_out_folds"]]
    for seed in results["ppo_by_seed"].values():
        assert (
            seed["score_all_folds"]
            > baselines["threshold_feature_chosen_by_training_score"]["score_all_folds"]
        )
        assert seed["score_without_dev_fold"] > best_on_test["score_without_dev_fold"]
    assert baselines["deployed_zscore_rule"]["score_all_folds"] < 0.0


def test_shipped_policy_on_the_vendored_nab_files(
    shipped: AlarmPolicy, corpus: list[NABStream]
) -> None:
    """Pins the served behaviour (features + inference) on real NAB data."""
    features, tables = nab_features_and_tables(corpus)
    alarms = {n: run_alarm_policy(shipped, f) for n, f in features.items()}
    result = score_corpus(alarms, tables)
    assert {n: int(a.sum()) for n, a in alarms.items()} == PINNED_ALARM_COUNTS
    assert result["score"] == pytest.approx(PINNED_FIXTURE_SCORE, abs=1e-6)


def test_streaming_detector_flags_a_spike_in_any_field(shipped: AlarmPolicy) -> None:
    rng = np.random.default_rng(0)
    detector = StreamingAlarmDetector(shipped)
    results = []
    for t, (a, b) in enumerate(rng.normal(0.0, 1.0, (1500, 2))):
        message = {"cpu": float(a), "mem": float(b), "host": "web-1", "up": True}
        if t == 1200:
            message["mem"] = 60.0
        results.append(detector(message))
    spike = results[1200]
    assert spike["is_anomaly"] is True and spike["anomaly_features"] == ["mem"]
    assert spike["score"] > 0.5 and spike["detector"] == "learned_alarm_policy"
    assert set(spike["feature_scores"]) == {"cpu", "mem"}  # non-numeric and bool fields skipped
    assert sum(r["is_anomaly"] for r in results) / len(results) < 0.02


def test_the_pipeline_defaults_to_the_policy_and_cli_can_choose() -> None:
    from click.testing import CliRunner

    from omni_mercury_engine.cli import main
    from omni_mercury_engine.infrastructure.streaming import StreamingAnomalyPipeline

    pipeline = StreamingAnomalyPipeline("in", "out")
    assert isinstance(pipeline.detector, StreamingAlarmDetector)
    help_text = CliRunner().invoke(main, ["stream", "--help"]).output
    assert "--detector [policy|zscore]" in help_text


#: The shipped policy's alarms on the four vendored NAB files (in its training set).
PINNED_ALARM_COUNTS: dict[str, int] = {
    "realAWSCloudwatch/iio_us-east-1_i-a2eb1cd9_NetworkIn.csv": 2,
    "realAdExchange/exchange-3_cpc_results.csv": 4,
    "realKnownCause/rogue_agent_key_hold.csv": 4,
    "realTraffic/speed_7578.csv": 4,
}
PINNED_FIXTURE_SCORE = 48.64077248546523
