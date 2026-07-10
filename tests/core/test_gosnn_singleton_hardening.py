# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for the GOSNN singleton hardening (F10).

Pins the four product fixes that replaced the test-isolation-only
workaround (the autouse ``_isolate_gosnn_singleton`` conftest fixture,
which stays as test hygiene but which the runtime no longer depends on):

1. Gate-input sanitation — a raw unix-timestamp scalar (~1.7e9) registered
   into an operational group is excluded from the σ_Immutable fusion input
   with a once-per-scalar WARNING instead of collapsing the ethical score
   to 0.0 for unrelated ``detect_with_fusion`` calls; the ``omni_diag_``
   metric-only channel carries such raw measurements silently.
2. Scoped registration — ``unregister_scalars`` removes exactly the
   entries a component contributed (restoring shadowed defaults), and the
   ``scalar_registration`` context manager makes registrations temporary.
3. Registry thread-safety — concurrent register/unregister cycles never
   tear a reader's snapshot.
4. Re-init honesty — re-constructing the singleton with a materially
   different configuration raises ``ValueError`` instead of silently
   ignoring the request; the module accessor warns (once) and returns the
   live instance.
"""

from __future__ import annotations

import logging
import math
import threading

import numpy as np
import pytest

from omni_mercury_engine.core.global_omni_scalar_network import (
    GlobalOmniScalarNetwork,
    ScalarGroup,
    get_global_scalar_network,
    reset_global_network,
)

_GOSNN_LOGGER = "omni_mercury_engine.core.global_omni_scalar_network"
_UNIX_TIMESTAMP = 1.7e9
_QUARANTINE_MARKER = "excluding it from the fusion/gate input"


def _fresh_network() -> GlobalOmniScalarNetwork:
    """Return a freshly reset default-configured GOSNN singleton."""
    reset_global_network()
    return GlobalOmniScalarNetwork()


def _all_group_snapshots(network: GlobalOmniScalarNetwork) -> dict[ScalarGroup, dict[str, float]]:
    """Snapshot every group's registered scalars for before/after diffing."""
    return {group: network.get_group_scalars(group) for group in ScalarGroup}


class TestGateInputSanitation:
    """Timestamp-scale scalars must never reach the σ_Immutable fusion input."""

    def test_old_way_timestamp_is_excluded_and_warns_once(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The pre-F10 registration style is quarantined, visibly, once."""
        network = _fresh_network()

        network.register_scalars(
            component_name="ama_posture_rotation",
            scalars={"omni_posture_rotation_timestamp": _UNIX_TIMESTAMP},
            group=ScalarGroup.SECURITY,
        )

        with caplog.at_level(logging.WARNING, logger=_GOSNN_LOGGER):
            collected = network._collect_all_scalars()
            collected_again = network._collect_all_scalars()

        assert "omni_posture_rotation_timestamp" not in collected
        assert "omni_posture_rotation_timestamp" not in collected_again
        assert max(abs(value) for value in collected.values()) <= (
            network.OPERATIONAL_SCALAR_ABS_LIMIT
        )

        quarantine_records = [
            record for record in caplog.records if _QUARANTINE_MARKER in record.getMessage()
        ]
        assert len(quarantine_records) == 1, "tripwire WARNING must fire exactly once per scalar"
        message = quarantine_records[0].getMessage()
        assert "omni_posture_rotation_timestamp" in message
        assert "ama_posture_rotation" in message

    def test_old_way_timestamp_is_excluded_from_dimensional_states(self) -> None:
        """The fusion state vectors must not carry the timestamp either."""
        network = _fresh_network()
        network.register_scalars(
            component_name="ama_posture_rotation",
            scalars={"omni_posture_rotation_timestamp": _UNIX_TIMESTAMP},
            group=ScalarGroup.SECURITY,
        )

        states = network._prepare_dimensional_states({"base_probe": 1.0}, {})

        assert states, "dimensional states must not be empty"
        for state in states:
            assert float(np.max(np.abs(state))) <= network.OPERATIONAL_SCALAR_ABS_LIMIT

    def test_metric_only_prefix_is_excluded_without_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The omni_diag_ channel is the sanctioned home for raw measurements."""
        network = _fresh_network()
        network.register_scalars(
            component_name="ama_posture_rotation",
            scalars={"omni_diag_posture_rotation_timestamp": _UNIX_TIMESTAMP},
            group=ScalarGroup.SECURITY,
        )

        with caplog.at_level(logging.WARNING, logger=_GOSNN_LOGGER):
            collected = network._collect_all_scalars()

        assert "omni_diag_posture_rotation_timestamp" not in collected
        assert not any(_QUARANTINE_MARKER in record.getMessage() for record in caplog.records)
        # Still discoverable for reporting — diagnostic, not deleted.
        security = network.get_group_scalars(ScalarGroup.SECURITY)
        assert security["omni_diag_posture_rotation_timestamp"] == _UNIX_TIMESTAMP

    def test_non_finite_operational_scalar_is_excluded_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """NaN in the operational band is quarantined, never fed to the gate."""
        network = _fresh_network()
        network.register_scalars(
            component_name="broken_component",
            scalars={"omni_broken_signal": float("nan")},
            group=ScalarGroup.SECURITY,
        )

        with caplog.at_level(logging.WARNING, logger=_GOSNN_LOGGER):
            collected = network._collect_all_scalars()

        assert "omni_broken_signal" not in collected
        assert all(math.isfinite(value) for value in collected.values())
        assert any(
            _QUARANTINE_MARKER in record.getMessage()
            and "omni_broken_signal" in record.getMessage()
            for record in caplog.records
        )

    def test_gate_score_is_unchanged_by_timestamp_registration(self) -> None:
        """The bleed regression: the σ score must not move when a timestamp lands."""
        network = _fresh_network()
        baseline_vector = np.array(list(network._collect_all_scalars().values()))
        _, baseline_score = network.ethical_gate.evaluate(baseline_vector)

        network.register_scalars(
            component_name="ama_posture_rotation",
            scalars={"omni_posture_rotation_timestamp": _UNIX_TIMESTAMP},
            group=ScalarGroup.SECURITY,
        )
        contaminated_vector = np.array(list(network._collect_all_scalars().values()))
        _, contaminated_score = network.ethical_gate.evaluate(contaminated_vector)

        assert contaminated_vector.shape == baseline_vector.shape
        assert contaminated_score == baseline_score

    def test_adapter_rotation_path_registers_metric_only_timestamp(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """End-to-end: the real adapter callback keeps the gate input clean."""
        from omni_mercury_engine.integrations.mercury_amacrypto import MercuryGuardianAdapter

        network = _fresh_network()
        adapter = MercuryGuardianAdapter()
        try:
            with caplog.at_level(logging.WARNING, logger=_GOSNN_LOGGER):
                adapter._on_posture_rotation()
                adapter._on_posture_algorithm_switch("kyber1024")
                collected = network._collect_all_scalars()

            security = network.get_group_scalars(ScalarGroup.SECURITY)
            assert "omni_diag_posture_rotation_timestamp" in security
            assert "omni_diag_posture_switch_timestamp" in security
            assert "omni_posture_rotation_timestamp" not in security
            assert max(abs(value) for value in collected.values()) <= (
                network.OPERATIONAL_SCALAR_ABS_LIMIT
            )
            # Correctly-channelled diagnostics must not trip the quarantine.
            assert not any(_QUARANTINE_MARKER in record.getMessage() for record in caplog.records)
        finally:
            adapter.close()


class TestScopedRegistration:
    """unregister_scalars removes exactly a component's contributions."""

    def test_unregister_removes_only_the_component_entries(self) -> None:
        """Two components' contributions are separable."""
        network = _fresh_network()
        network.register_scalars("component_a", {"omni_a_only": 1.1}, group=ScalarGroup.SECURITY)
        network.register_scalars("component_b", {"omni_b_only": 1.2}, group=ScalarGroup.SECURITY)

        assert network.unregister_scalars("component_a") is True

        security = network.get_group_scalars(ScalarGroup.SECURITY)
        assert "omni_a_only" not in security
        assert security["omni_b_only"] == 1.2
        assert "component_a" not in network.registered_scalars
        assert "component_b" in network.registered_scalars

    def test_unregister_restores_shadowed_default_scalar(self) -> None:
        """Shadowing a built-in default must not shrink the σ layout on exit."""
        network = _fresh_network()
        default_morality = network.get_scalar("omnimorality")
        assert default_morality == network.MIN_MORALITY

        network.register_scalars("shadower", {"omnimorality": 0.5}, group=ScalarGroup.ETHICAL)
        assert network.get_scalar("omnimorality") == 0.5

        network.unregister_scalars("shadower")
        assert network.get_scalar("omnimorality") == default_morality

    def test_unregister_accumulates_across_repeated_registrations(self) -> None:
        """A component owns the union of everything it ever registered."""
        network = _fresh_network()
        network.register_scalars("accumulator", {"omni_first": 1.0}, group=ScalarGroup.SECURITY)
        network.register_scalars("accumulator", {"omni_second": 1.1}, group=ScalarGroup.SECURITY)

        network.unregister_scalars("accumulator")

        security = network.get_group_scalars(ScalarGroup.SECURITY)
        assert "omni_first" not in security
        assert "omni_second" not in security

    def test_cross_component_overwrite_warns_and_transfers_ownership(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Last write wins, visibly, and ownership follows the last writer."""
        network = _fresh_network()
        network.register_scalars("first_owner", {"omni_contested": 1.0}, group=ScalarGroup.SECURITY)

        with caplog.at_level(logging.WARNING, logger=_GOSNN_LOGGER):
            network.register_scalars(
                "second_owner", {"omni_contested": 2.0}, group=ScalarGroup.SECURITY
            )

        assert any(
            "omni_contested" in record.getMessage()
            and "second_owner" in record.getMessage()
            and "first_owner" in record.getMessage()
            for record in caplog.records
        ), "cross-component overwrite must log a WARNING naming both components"

        # The previous owner no longer controls the entry.
        network.unregister_scalars("first_owner")
        assert network.get_group_scalars(ScalarGroup.SECURITY)["omni_contested"] == 2.0

        # The new owner does.
        network.unregister_scalars("second_owner")
        assert "omni_contested" not in network.get_group_scalars(ScalarGroup.SECURITY)

    def test_context_manager_restores_registered_state(self) -> None:
        """scalar_registration leaves the registry exactly as it found it."""
        network = _fresh_network()
        before = _all_group_snapshots(network)

        with network.scalar_registration(
            "temporary_component",
            {"omni_temporary": 1.3, "omnimorality": 0.6},
            group=ScalarGroup.ETHICAL,
        ) as inside:
            assert inside is network
            assert network.get_scalar("omni_temporary") == 1.3
            assert network.get_scalar("omnimorality") == 0.6

        assert _all_group_snapshots(network) == before
        assert "temporary_component" not in network.registered_scalars

    def test_context_manager_unregisters_on_exception(self) -> None:
        """The temporary registration is removed even when the block raises."""
        network = _fresh_network()
        before = _all_group_snapshots(network)

        with (
            pytest.raises(RuntimeError, match="boom"),
            network.scalar_registration(
                "temporary_component", {"omni_temporary": 1.3}, group=ScalarGroup.SECURITY
            ),
        ):
            raise RuntimeError("boom")

        assert _all_group_snapshots(network) == before

    def test_unregister_unknown_component_returns_false(self) -> None:
        """Unregistering a component that never registered is a visible no-op."""
        network = _fresh_network()
        before = _all_group_snapshots(network)

        assert network.unregister_scalars("never_registered") is False
        assert _all_group_snapshots(network) == before


class TestRegistryConcurrency:
    """Concurrent register/unregister must never tear a reader snapshot."""

    def test_concurrent_register_unregister_with_reader(self) -> None:
        """N writer threads cycle registrations while a reader snapshots.

        Deterministic by construction (fixed thread/iteration counts, no
        randomness): each writer registers its two scalars in a single
        ``register_scalars`` call, so every reader snapshot must contain
        either both of a writer's scalars or neither — a torn snapshot or
        any raised exception fails the test.
        """
        network = _fresh_network()
        writer_count = 8
        iterations = 40
        errors: list[BaseException] = []
        start_barrier = threading.Barrier(writer_count + 1)
        stop_reading = threading.Event()

        def writer(thread_id: int) -> None:
            component = f"concurrent_component_{thread_id}"
            scalars = {
                f"omni_cc_{thread_id}_a": 1.0 + thread_id / 100.0,
                f"omni_cc_{thread_id}_b": 1.0 + thread_id / 100.0,
            }
            try:
                start_barrier.wait(timeout=30)
                for _ in range(iterations):
                    network.register_scalars(component, scalars, group=ScalarGroup.SECURITY)
                    network.unregister_scalars(component)
            except BaseException as exc:
                errors.append(exc)

        def reader() -> None:
            try:
                start_barrier.wait(timeout=30)
                while not stop_reading.is_set():
                    snapshot = network._collect_all_scalars()
                    for thread_id in range(writer_count):
                        has_a = f"omni_cc_{thread_id}_a" in snapshot
                        has_b = f"omni_cc_{thread_id}_b" in snapshot
                        assert has_a == has_b, "torn snapshot: half of an atomic registration"
                    group_snapshot = network.get_group_scalars(ScalarGroup.SECURITY)
                    assert isinstance(group_snapshot, dict)
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(writer_count)]
        reader_thread = threading.Thread(target=reader)
        for thread in threads:
            thread.start()
        reader_thread.start()

        for thread in threads:
            thread.join(timeout=60)
        stop_reading.set()
        reader_thread.join(timeout=60)

        assert not any(thread.is_alive() for thread in threads), "writer thread hung"
        assert not reader_thread.is_alive(), "reader thread hung"
        assert errors == []

        # Every writer unregistered on its last iteration: registry is clean.
        security = network.get_group_scalars(ScalarGroup.SECURITY)
        assert not any(name.startswith("omni_cc_") for name in security)


class TestReinitHonesty:
    """Re-construction with materially different config must fail loudly."""

    def test_different_domain_threshold_raises_value_error(self) -> None:
        """A medical caller must not silently inherit the 0.96 default gate."""
        network = _fresh_network()
        assert network.sigma_immutable_threshold == pytest.approx(0.96)

        with pytest.raises(ValueError, match="reset_global_network"):
            GlobalOmniScalarNetwork(domain="medical")

    def test_different_max_dimensions_raises_value_error(self) -> None:
        """Explicit dimension changes cannot be silently ignored."""
        _fresh_network()
        with pytest.raises(ValueError, match="max_dimensions"):
            GlobalOmniScalarNetwork(max_dimensions=64)

    def test_no_arg_reconstruction_returns_live_instance_silently(self) -> None:
        """The bare-constructor 'give me the singleton' idiom stays a no-op."""
        network = _fresh_network()
        assert GlobalOmniScalarNetwork() is network

    def test_same_explicit_config_reconstruction_is_noop(self) -> None:
        """Re-requesting the live configuration is not a divergence."""
        reset_global_network()
        network = GlobalOmniScalarNetwork(domain="medical", max_dimensions=37)
        assert network.sigma_immutable_threshold == pytest.approx(0.93)
        assert GlobalOmniScalarNetwork(domain="medical", max_dimensions=37) is network

    def test_equivalent_domain_label_is_not_material(self) -> None:
        """Domain labels resolving to the same threshold are compatible."""
        network = _fresh_network()
        # "general" resolves to the same non-medical threshold as None.
        assert GlobalOmniScalarNetwork(domain="general") is network

    def test_reset_allows_reconfiguration(self) -> None:
        """reset_global_network() is the sanctioned reconfiguration path."""
        default_network = _fresh_network()
        reset_global_network()
        medical_network = GlobalOmniScalarNetwork(domain="medical")

        assert medical_network is not default_network
        assert medical_network.sigma_immutable_threshold == pytest.approx(0.93)

    def test_accessor_warns_once_and_returns_live_instance(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """get_global_scalar_network surfaces, but survives, divergence."""
        network = _fresh_network()

        with caplog.at_level(logging.WARNING, logger=_GOSNN_LOGGER):
            first = get_global_scalar_network(domain="medical")
            second = get_global_scalar_network(domain="medical")

        assert first is network
        assert second is network
        divergence_records = [
            record
            for record in caplog.records
            if "differs materially from the live GOSNN singleton" in record.getMessage()
        ]
        assert len(divergence_records) == 1, "accessor divergence WARNING must fire once"

    def test_accessor_survives_racing_first_direct_construction(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The documented non-raising accessor must not leak ValueError.

        Simulates the race where a direct ``GlobalOmniScalarNetwork(...)``
        construction wins between the accessor's null-check and its own
        construction call: singleton ``__new__`` hands back the freshly
        initialized instance and the accessor's materially different kwargs
        trip the re-init ValueError. The accessor must recover to its
        documented semantics (warn + return the live instance).
        """
        import omni_mercury_engine.core.global_omni_scalar_network as gos

        real = _fresh_network()  # default config (32 heads)
        # Open the race window: both caches look empty to the accessor...
        monkeypatch.setattr(gos, "_global_network", None)
        monkeypatch.setattr(GlobalOmniScalarNetwork, "_instance", None)

        original_new = GlobalOmniScalarNetwork.__new__

        def racing_new(cls: type, *args: object, **kwargs: object) -> GlobalOmniScalarNetwork:
            # ...but the concurrent direct construction lands first.
            cls._instance = real  # type: ignore[attr-defined]
            return real

        monkeypatch.setattr(GlobalOmniScalarNetwork, "__new__", racing_new)
        try:
            with caplog.at_level(logging.WARNING, logger=_GOSNN_LOGGER):
                result = get_global_scalar_network(num_attention_heads=16)
        finally:
            monkeypatch.setattr(GlobalOmniScalarNetwork, "__new__", original_new)

        assert result is real
        assert any(
            "differs materially from the live GOSNN singleton" in record.getMessage()
            for record in caplog.records
        )


class TestAdapterLifecycle:
    """MercuryGuardianAdapter.close() scopes its GOSNN registrations."""

    def test_close_unregisters_adapter_components(self) -> None:
        """Attack-simulation registrations do not outlive the adapter."""
        from omni_mercury_engine.integrations.mercury_amacrypto import MercuryGuardianAdapter

        network = _fresh_network()
        baseline_security = network.get_group_scalars(ScalarGroup.SECURITY)

        adapter = MercuryGuardianAdapter()
        adapter.simulate_attack(attack_type="replay")
        assert "ama_cryptography_pqc" in network.registered_scalars

        adapter.close()

        assert "ama_cryptography_pqc" not in network.registered_scalars
        assert network.get_group_scalars(ScalarGroup.SECURITY) == baseline_security
        # Idempotent.
        adapter.close()

    def test_close_does_not_construct_the_singleton(self) -> None:
        """Closing an adapter must not instantiate GOSNN just to clean it."""
        from omni_mercury_engine.integrations.mercury_amacrypto import MercuryGuardianAdapter

        adapter = MercuryGuardianAdapter(gosnn_synapse_enabled=False)
        reset_global_network()

        adapter.close()

        assert GlobalOmniScalarNetwork._instance is None

    def test_context_manager_closes_on_exit(self) -> None:
        """The with-statement form unregisters on exit."""
        from omni_mercury_engine.integrations.mercury_amacrypto import MercuryGuardianAdapter

        network = _fresh_network()
        with MercuryGuardianAdapter() as adapter:
            adapter.simulate_attack(attack_type="side_channel")
            assert "ama_cryptography_pqc" in network.registered_scalars

        assert "ama_cryptography_pqc" not in network.registered_scalars
