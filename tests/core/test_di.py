# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Behavioral tests for the dependency-injection container (``core/di.py``).

Covers:
- Lifecycle registration: singleton / transient / scoped, fluent chaining,
  re-registration (last registration wins)
- Resolution: pre-built instances, factory functions, constructor injection
  via real type annotations, unregistered-dependency fallbacks
- Scoping: ``resolve(scope_id=...)``, ``create_scope`` / ``dispose_scope``,
  ``ServiceScope`` context-manager semantics
- Error paths: ``ServiceNotFoundError``, ``CircularDependencyError`` (self and
  two-node cycles), resolution-stack cleanup after failure
- Thread safety: concurrent singleton resolution yields one instance
- Protocols: runtime ``isinstance`` checks for Detector / Model / Encoder /
  Configurable protocols
- ComponentFactory: plugin registration and precedence, lazy-import map
  (happy path via the numpy-only quantum model), unknown-type errors,
  import-failure propagation
- Global container: ``get_container`` memoisation, ``configure_container``
- Regressions for fixed defects: falsy-singleton re-creation (truthiness
  vs ``is not None``), stale ``detector_map`` / ``model_map`` class names
  (every built-in entry is instantiated for real)
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, cast

import pytest

from omni_mercury_engine.core import di
from omni_mercury_engine.core.di import (
    CircularDependencyError,
    ComponentFactory,
    ConfigurableProtocol,
    DetectorProtocol,
    EncoderProtocol,
    Lifecycle,
    ModelProtocol,
    ServiceContainer,
    ServiceDescriptor,
    ServiceNotFoundError,
    ServiceScope,
    configure_container,
    get_container,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

# ---------------------------------------------------------------------------
# Helper service classes
# ---------------------------------------------------------------------------


class Greeter:
    """Trivial no-dependency service."""

    def __init__(self) -> None:
        self.greeting = "hello"


class Engine:
    """Leaf dependency for constructor-injection tests."""

    def __init__(self) -> None:
        self.started = True


class Car:
    """Service with a constructor dependency on :class:`Engine`."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine


class Widget:
    """Service whose annotated ctor param is never registered."""

    def __init__(self, size: int = 3) -> None:
        self.size = size


class NeedsUnregistered:
    """Service with a required dependency that is never registered."""

    def __init__(self, greeter: Greeter) -> None:
        self.greeter = greeter


class EmptySequence:
    """An object that is *falsy* (``len() == 0``) but perfectly valid."""

    def __len__(self) -> int:
        return 0


def _force_real_annotations(cls: type, annotations: dict[str, Any]) -> None:
    """Replace PEP 563 string annotations with real type objects.

    This test module uses ``from __future__ import annotations``, which turns
    ``__init__`` annotations into strings.  ``ServiceContainer._create_instance``
    matches ``param.annotation in self._services`` against *type objects*, so
    constructor injection only engages for classes whose annotations are real
    types.  Restoring concrete objects here mirrors a class defined in a
    module without postponed evaluation.
    """
    init: Any = vars(cls)["__init__"]
    init.__annotations__ = annotations


_force_real_annotations(Car, {"engine": Engine, "return": None})
_force_real_annotations(Widget, {"size": int, "return": None})
_force_real_annotations(NeedsUnregistered, {"greeter": Greeter, "return": None})


@pytest.fixture
def container() -> ServiceContainer:
    """A fresh, empty container per test."""
    return ServiceContainer()


# ---------------------------------------------------------------------------
# Registration and lifecycle
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_register_singleton_returns_self_for_chaining(
        self, container: ServiceContainer
    ) -> None:
        result = container.register_singleton(Greeter).register_transient(Engine)
        assert result is container

    def test_register_scoped_returns_self(self, container: ServiceContainer) -> None:
        assert container.register_scoped(Greeter) is container

    def test_descriptor_defaults(self) -> None:
        descriptor = ServiceDescriptor(service_type=Greeter)
        assert descriptor.implementation_type is None
        assert descriptor.factory is None
        assert descriptor.instance is None
        assert descriptor.lifecycle is Lifecycle.SINGLETON
        assert descriptor.dependencies == []

    def test_last_registration_wins(self, container: ServiceContainer) -> None:
        container.register_singleton(Greeter)
        container.register_transient(Greeter)
        first = container.resolve(Greeter)
        second = container.resolve(Greeter)
        # Re-registered as transient: a new instance per resolve.
        assert first is not second


class TestSingletonResolution:
    def test_prebuilt_instance_is_returned_verbatim(self, container: ServiceContainer) -> None:
        instance = Greeter()
        container.register_singleton(Greeter, instance=instance)
        assert container.resolve(Greeter) is instance

    def test_lazy_singleton_is_created_once_and_cached(self, container: ServiceContainer) -> None:
        container.register_singleton(Greeter)
        first = container.resolve(Greeter)
        second = container.resolve(Greeter)
        assert isinstance(first, Greeter)
        assert first is second

    def test_singleton_factory_called_exactly_once(self, container: ServiceContainer) -> None:
        calls: list[ServiceContainer] = []

        def factory(c: ServiceContainer) -> Greeter:
            calls.append(c)
            return Greeter()

        container.register_singleton(Greeter, factory=factory)
        first = container.resolve(Greeter)
        second = container.resolve(Greeter)
        assert first is second
        assert len(calls) == 1

    def test_factory_receives_the_container(self, container: ServiceContainer) -> None:
        seen: list[ServiceContainer] = []

        def factory(c: ServiceContainer) -> Greeter:
            seen.append(c)
            return Greeter()

        container.register_singleton(Greeter, factory=factory)
        container.resolve(Greeter)
        assert seen == [container]

    def test_concurrent_singleton_resolution_yields_one_instance(
        self, container: ServiceContainer
    ) -> None:
        container.register_singleton(Greeter)
        results: list[Greeter] = []
        barrier = threading.Barrier(8)

        def worker() -> None:
            barrier.wait(timeout=5)
            results.append(container.resolve(Greeter))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert len(results) == 8
        assert all(r is results[0] for r in results)


class TestTransientResolution:
    def test_transient_creates_new_instance_per_resolve(self, container: ServiceContainer) -> None:
        container.register_transient(Greeter)
        first = container.resolve(Greeter)
        second = container.resolve(Greeter)
        assert isinstance(first, Greeter)
        assert first is not second

    def test_transient_factory_called_per_resolve(self, container: ServiceContainer) -> None:
        counter: list[int] = []

        def factory(c: ServiceContainer) -> Greeter:
            counter.append(1)
            return Greeter()

        container.register_transient(Greeter, factory=factory)
        container.resolve(Greeter)
        container.resolve(Greeter)
        assert len(counter) == 2


class TestConstructorInjection:
    def test_registered_dependency_is_injected(self, container: ServiceContainer) -> None:
        container.register_singleton(Engine)
        container.register_transient(Car)
        car = container.resolve(Car)
        assert isinstance(car.engine, Engine)
        assert car.engine.started is True

    def test_injected_dependency_respects_singleton_lifecycle(
        self, container: ServiceContainer
    ) -> None:
        container.register_singleton(Engine)
        container.register_transient(Car)
        car_a = container.resolve(Car)
        car_b = container.resolve(Car)
        assert car_a is not car_b
        assert car_a.engine is car_b.engine

    def test_unregistered_annotated_param_falls_back_to_default(
        self, container: ServiceContainer
    ) -> None:
        # ``size: int`` is annotated but ``int`` is not registered, so the
        # container must leave it alone and let the default apply.
        container.register_transient(Widget)
        widget = container.resolve(Widget)
        assert widget.size == 3

    def test_required_unregistered_dependency_raises_type_error(
        self, container: ServiceContainer
    ) -> None:
        # ``Greeter`` is annotated but never registered and has no default:
        # instantiation fails with the ordinary missing-argument TypeError.
        container.register_transient(NeedsUnregistered)
        with pytest.raises(TypeError):
            container.resolve(NeedsUnregistered)

    def test_class_without_custom_init_resolves(self, container: ServiceContainer) -> None:
        class Bare:
            pass

        container.register_transient(Bare)
        assert isinstance(container.resolve(Bare), Bare)

    def test_implementation_type_is_instantiated_for_service_type(
        self, container: ServiceContainer
    ) -> None:
        class GreeterImpl(Greeter):
            pass

        container.register_singleton(Greeter, implementation_type=GreeterImpl)
        resolved = container.resolve(Greeter)
        assert type(resolved) is GreeterImpl


# ---------------------------------------------------------------------------
# Scoped lifecycle
# ---------------------------------------------------------------------------


class TestScopedResolution:
    def test_same_instance_within_scope(self, container: ServiceContainer) -> None:
        container.register_scoped(Greeter)
        first = container.resolve(Greeter, scope_id="request-1")
        second = container.resolve(Greeter, scope_id="request-1")
        assert first is second

    def test_different_instances_across_scopes(self, container: ServiceContainer) -> None:
        container.register_scoped(Greeter)
        a = container.resolve(Greeter, scope_id="request-1")
        b = container.resolve(Greeter, scope_id="request-2")
        assert a is not b

    def test_scoped_without_scope_id_behaves_transiently(self, container: ServiceContainer) -> None:
        # Without a scope_id there is nowhere to cache the instance, so each
        # resolve builds a fresh one.
        container.register_scoped(Greeter)
        assert container.resolve(Greeter) is not container.resolve(Greeter)

    def test_dispose_scope_drops_cached_instances(self, container: ServiceContainer) -> None:
        container.register_scoped(Greeter)
        first = container.resolve(Greeter, scope_id="s")
        container.dispose_scope("s")
        second = container.resolve(Greeter, scope_id="s")
        assert first is not second

    def test_dispose_unknown_scope_is_a_noop(self, container: ServiceContainer) -> None:
        container.dispose_scope("never-created")  # must not raise

    def test_create_scope_returns_service_scope(self, container: ServiceContainer) -> None:
        scope = container.create_scope("s1")
        assert isinstance(scope, ServiceScope)

    def test_service_scope_resolves_within_its_scope(self, container: ServiceContainer) -> None:
        container.register_scoped(Greeter)
        scope = container.create_scope("s1")
        assert scope.resolve(Greeter) is scope.resolve(Greeter)

    def test_service_scope_context_manager_disposes_on_exit(
        self, container: ServiceContainer
    ) -> None:
        container.register_scoped(Greeter)
        with container.create_scope("ctx") as scope:
            assert scope is not None
            inside = scope.resolve(Greeter)
            assert scope.resolve(Greeter) is inside
        # Scope disposed on __exit__: same scope_id now yields a new instance.
        after = container.resolve(Greeter, scope_id="ctx")
        assert after is not inside

    def test_distinct_scopes_are_isolated_via_context_managers(
        self, container: ServiceContainer
    ) -> None:
        container.register_scoped(Greeter)
        with container.create_scope("a") as scope_a, container.create_scope("b") as scope_b:
            assert scope_a.resolve(Greeter) is not scope_b.resolve(Greeter)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestErrorPaths:
    def test_resolve_unregistered_raises_service_not_found(
        self, container: ServiceContainer
    ) -> None:
        with pytest.raises(ServiceNotFoundError, match="Greeter is not registered"):
            container.resolve(Greeter)

    def test_self_referential_factory_raises_circular_dependency(
        self, container: ServiceContainer
    ) -> None:
        container.register_transient(Greeter, factory=lambda c: c.resolve(Greeter))
        with pytest.raises(CircularDependencyError, match="Circular dependency detected"):
            container.resolve(Greeter)

    def test_two_node_cycle_raises_circular_dependency(self, container: ServiceContainer) -> None:
        container.register_transient(Engine, factory=lambda c: c.resolve(Car))
        container.register_transient(Car, factory=lambda c: c.resolve(Engine))
        with pytest.raises(CircularDependencyError):
            container.resolve(Engine)

    def test_resolution_stack_is_cleaned_after_failure(self, container: ServiceContainer) -> None:
        # After a circular failure the resolution stack must be unwound so a
        # subsequently fixed registration resolves cleanly.
        container.register_transient(Greeter, factory=lambda c: c.resolve(Greeter))
        with pytest.raises(CircularDependencyError):
            container.resolve(Greeter)
        container.register_transient(Greeter)  # fixed registration
        assert isinstance(container.resolve(Greeter), Greeter)

    def test_factory_exception_propagates_and_unwinds_stack(
        self, container: ServiceContainer
    ) -> None:
        def broken(c: ServiceContainer) -> Greeter:
            raise RuntimeError("boom")

        container.register_singleton(Greeter, factory=broken)
        with pytest.raises(RuntimeError, match="boom"):
            container.resolve(Greeter)
        # Stack unwound: a second attempt fails with the *factory* error
        # again, not a bogus CircularDependencyError.
        with pytest.raises(RuntimeError, match="boom"):
            container.resolve(Greeter)


# ---------------------------------------------------------------------------
# Pinned defect: falsy singleton instances are not cached
# ---------------------------------------------------------------------------


class TestFalsySingletonDefect:
    """Regressions: resolve() must test instance presence with `is not None`.

    The historical truthiness check re-created any falsy singleton (an
    empty container, `__len__ == 0`) on every resolve and ignored a
    pre-registered falsy instance — both silent singleton-contract
    violations.
    """

    def test_falsy_singleton_is_cached(self, container: ServiceContainer) -> None:
        container.register_singleton(EmptySequence)
        first = container.resolve(EmptySequence)
        second = container.resolve(EmptySequence)
        assert first is second

    def test_registered_falsy_instance_is_returned(self, container: ServiceContainer) -> None:
        instance = EmptySequence()
        container.register_singleton(EmptySequence, instance=instance)
        assert container.resolve(EmptySequence) is instance


# ---------------------------------------------------------------------------
# Protocols (structural subtyping, runtime_checkable)
# ---------------------------------------------------------------------------


class FakeDetector:
    """Structurally conforms to DetectorProtocol."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self._fitted = False

    def fit(self, data: Any) -> FakeDetector:
        self._fitted = True
        return self

    def detect(self, data: Any) -> dict[str, Any]:
        return {"is_anomaly": False, "score": 0.0}

    def extract_features(self, data: Any) -> Any:
        return data

    def is_fitted(self) -> bool:
        return self._fitted


class FakeModel:
    """Structurally conforms to ModelProtocol."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def predict(self, data: Any) -> dict[str, Any]:
        return {"prediction": 0}

    def extract_features(self, data: Any) -> Any:
        return data


class FakeEncoder:
    """Structurally conforms to EncoderProtocol."""

    def __init__(self) -> None:
        self.input_dim = 4
        self.output_dim = 2

    def forward(self, x: Any) -> Any:
        return x


class FakeConfigurable:
    """Structurally conforms to ConfigurableProtocol."""

    def __init__(self) -> None:
        self._config: dict[str, Any] = {}

    def configure(self, config: dict[str, Any]) -> None:
        self._config = dict(config)

    def get_config(self) -> dict[str, Any]:
        return dict(self._config)


class TestProtocols:
    def test_conforming_detector_passes_isinstance(self) -> None:
        assert isinstance(FakeDetector(), DetectorProtocol)

    def test_non_conforming_object_fails_isinstance(self) -> None:
        assert not isinstance(Greeter(), DetectorProtocol)
        assert not isinstance(Greeter(), ModelProtocol)

    def test_conforming_model_passes_isinstance(self) -> None:
        assert isinstance(FakeModel(), ModelProtocol)

    def test_detector_is_also_a_model_structurally(self) -> None:
        # DetectorProtocol is a superset of ModelProtocol minus predict();
        # FakeDetector lacks predict so it must NOT satisfy ModelProtocol.
        assert not isinstance(FakeDetector(), ModelProtocol)

    def test_encoder_protocol_checks_data_members(self) -> None:
        assert isinstance(FakeEncoder(), EncoderProtocol)
        assert not isinstance(Greeter(), EncoderProtocol)

    def test_configurable_protocol(self) -> None:
        obj = FakeConfigurable()
        assert isinstance(obj, ConfigurableProtocol)
        obj.configure({"a": 1})
        assert obj.get_config() == {"a": 1}


# ---------------------------------------------------------------------------
# ComponentFactory
# ---------------------------------------------------------------------------


class TestComponentFactory:
    @pytest.fixture
    def factory(self, container: ServiceContainer) -> ComponentFactory:
        return ComponentFactory(container)

    def test_registered_plugin_detector_is_instantiated_with_config(
        self, factory: ComponentFactory
    ) -> None:
        factory.register_plugin("fake", FakeDetector, version="2.0.0")
        detector = factory.create_detector("fake", config={"threshold": 0.5})
        assert isinstance(detector, FakeDetector)
        assert detector.config == {"threshold": 0.5}

    def test_plugin_detector_satisfies_detector_protocol(self, factory: ComponentFactory) -> None:
        factory.register_plugin("fake", FakeDetector)
        assert isinstance(factory.create_detector("fake"), DetectorProtocol)

    def test_unknown_detector_type_raises_value_error(self, factory: ComponentFactory) -> None:
        with pytest.raises(ValueError, match="Unknown detector type: nope"):
            factory.create_detector("nope")

    def test_unknown_model_type_raises_value_error(self, factory: ComponentFactory) -> None:
        with pytest.raises(ValueError, match="Unknown model type: nope"):
            factory.create_model("nope")

    def test_create_model_lazy_imports_quantum(self, factory: ComponentFactory) -> None:
        model = factory.create_model("quantum", config={"num_qubits": 4})
        assert type(model).__name__ == "QuantumAnomalyModel"
        assert isinstance(model, ModelProtocol)
        # Config must be forwarded to the constructor.
        assert cast("Any", model).num_qubits == 4

    def test_plugin_takes_precedence_over_builtin_map(self, factory: ComponentFactory) -> None:
        factory.register_plugin("quantum", FakeModel)
        model = factory.create_model("quantum")
        assert isinstance(model, FakeModel)

    def test_create_model_reraises_import_error(
        self, factory: ComponentFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import importlib

        def broken_import(name: str, package: str | None = None) -> Any:
            raise ImportError(f"forced failure importing {name}")

        monkeypatch.setattr(importlib, "import_module", broken_import)
        with pytest.raises(ImportError, match="forced failure"):
            factory.create_model("astrophysical")

    def test_create_detector_reraises_import_error(
        self, factory: ComponentFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import importlib

        def broken_import(name: str, package: str | None = None) -> Any:
            raise ImportError(f"forced failure importing {name}")

        monkeypatch.setattr(importlib, "import_module", broken_import)
        with pytest.raises(ImportError, match="forced failure"):
            factory.create_detector("statistical")

    # -- Regressions: built-in maps must name real, constructible classes --
    #
    # Every detector_map entry had bit-rotted to a class name that never
    # existed post-rename (StatisticalDetector vs the real
    # MercuryAnomalyDetector, etc.), and model_map's 'neural' /
    # 'consciousness' entries likewise — every built-in create_detector()
    # call raised AttributeError.  These tests instantiate each entry for
    # real, so a future rename fails loudly here instead of rotting again.

    @pytest.mark.parametrize(
        "detector_type",
        ["statistical", "temporal", "spatial", "dimensional", "directive"],
    )
    def test_builtin_detector_map_entries_are_constructible(
        self, factory: ComponentFactory, detector_type: str
    ) -> None:
        detector = factory.create_detector(detector_type)
        assert isinstance(detector, DetectorProtocol)

    @pytest.mark.parametrize("model_type", ["neural", "consciousness"])
    def test_builtin_model_map_entries_are_constructible(
        self, factory: ComponentFactory, model_type: str
    ) -> None:
        model = factory.create_model(model_type)
        assert isinstance(model, ModelProtocol)


# ---------------------------------------------------------------------------
# Global container helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def _restore_global_container() -> Iterator[None]:
    """Snapshot and restore the module-level global container."""
    saved = di._global_container
    try:
        yield
    finally:
        di._global_container = saved


class TestGlobalContainer:
    @pytest.mark.usefixtures("_restore_global_container")
    def test_get_container_creates_and_memoises(self) -> None:
        di._global_container = None
        first = get_container()
        second = get_container()
        assert isinstance(first, ServiceContainer)
        assert first is second

    @pytest.mark.usefixtures("_restore_global_container")
    def test_configure_container_replaces_global(self) -> None:
        replacement = ServiceContainer()
        configure_container(replacement)
        assert get_container() is replacement
