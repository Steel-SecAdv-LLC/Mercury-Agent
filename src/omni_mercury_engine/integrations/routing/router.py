# Copyright (C) 2025 Steel Security Advisors LLC
"""(at your option) any later version."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RouteMethod(Enum):
    """HTTP methods for routing."""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"
    ANY = "*"


@dataclass
class Route:
    """Route definition.

    Attributes:
        pattern: URL pattern with optional parameters (e.g., "/users/{id}").
        handler: Async function to handle matched requests.
        methods: Allowed HTTP methods.
        name: Optional route name for reverse lookup.
        middleware: List of middleware to apply.
        metadata: Additional route metadata.
    """

    pattern: str
    handler: Callable[..., Awaitable[Any]]
    methods: list[str] = field(default_factory=lambda: ["GET"])
    name: str | None = None
    middleware: list[Callable[..., Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Compiled regex pattern (set by router)
    _regex: re.Pattern[str] | None = field(default=None, repr=False)
    _param_names: list[str] = field(default_factory=list, repr=False)

    def compile(self) -> None:
        """Compile the route pattern into regex."""
        # Convert {param} to named capture groups
        regex_pattern = self.pattern
        param_names = []

        # Find all {param} patterns
        param_pattern = re.compile(r"\{(\w+)\}")
        for match in param_pattern.finditer(self.pattern):
            param_name = match.group(1)
            param_names.append(param_name)
            # Replace with named capture group
            regex_pattern = regex_pattern.replace(match.group(0), f"(?P<{param_name}>[^/]+)")

        # Add anchors
        regex_pattern = f"^{regex_pattern}$"
        self._regex = re.compile(regex_pattern)
        self._param_names = param_names

    def match(self, path: str) -> dict[str, str] | None:
        """Match path against this route.

        Args:
            path: URL path to match.

        Returns:
            Dictionary of matched parameters, or None if no match.
        """
        if self._regex is None:
            self.compile()

        if self._regex is None:
            raise RuntimeError("Route regex compilation failed")
        match = self._regex.match(path)  # type: ignore[union-attr, unused-ignore]
        if match:
            return match.groupdict()
        return None


@dataclass
class RouteMatch:
    """Result of route matching.

    Attributes:
        route: Matched route.
        handler: Handler function.
        params: Extracted path parameters.
        middleware: Middleware chain to apply.
    """

    route: Route
    handler: Callable[..., Awaitable[Any]]
    params: dict[str, str]
    middleware: list[Callable[..., Any]]


class RouteNotFoundError(Exception):
    """Raised when no route matches."""

    def __init__(self, path: str, method: str) -> None:
        """Initialize the instance."""
        super().__init__(f"No route found for {method} {path}")
        self.path = path
        self.method = method


class MethodNotAllowedError(Exception):
    """Raised when route exists but method not allowed."""

    def __init__(self, path: str, method: str, allowed: list[str]) -> None:
        """Initialize the instance."""
        super().__init__(
            f"Method {method} not allowed for {path}. " f"Allowed: {', '.join(allowed)}"
        )
        self.path = path
        self.method = method
        self.allowed = allowed


class RequestRouter:
    """Request router with pattern matching and middleware.

    Features:
    - URL pattern matching with parameters
    - HTTP method filtering
    - Middleware chains
    - Route groups with prefixes
    - Named routes for reverse lookup

    Example:
        >>> router = RequestRouter()
        >>> @router.get("/users/{user_id}")
        ... async def get_user(request, user_id):
        ...     return {"id": user_id}
        >>>
        >>> match = router.match("/users/123", method="GET")
        >>> result = await match.handler(request, **match.params)
    """

    def __init__(
        self,
        prefix: str = "",
        middleware: list[Callable[..., Any]] | None = None,
    ):
        """Initialize router.

        Args:
            prefix: URL prefix for all routes.
            middleware: Default middleware for all routes.
        """
        self.prefix = prefix.rstrip("/")
        self.middleware = middleware or []
        self._routes: list[Route] = []
        self._named_routes: dict[str, Route] = {}
        self._request_count = 0
        self._route_hits: dict[str, int] = {}

    def add_route(
        self,
        pattern: str,
        handler: Callable[..., Awaitable[Any]],
        methods: list[str] | None = None,
        name: str | None = None,
        middleware: list[Callable[..., Any]] | None = None,
        **metadata: Any,
    ) -> Route:
        """Add a route.

        Args:
            pattern: URL pattern (supports {param} syntax).
            handler: Async handler function.
            methods: Allowed HTTP methods.
            name: Optional route name.
            middleware: Route-specific middleware.
            **metadata: Additional route metadata.

        Returns:
            Created route.
        """
        full_pattern = f"{self.prefix}{pattern}"
        route = Route(
            pattern=full_pattern,
            handler=handler,
            methods=methods or ["GET"],
            name=name,
            middleware=middleware or [],
            metadata=metadata,
        )
        route.compile()

        self._routes.append(route)
        if name:
            self._named_routes[name] = route

        logger.debug(f"Added route: {methods} {full_pattern}")
        return route

    def route(
        self,
        pattern: str,
        methods: list[str] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> Callable[..., Any]:
        """Decorator to register a route.

        Args:
            pattern: URL pattern.
            methods: Allowed HTTP methods.
            name: Optional route name.
            **kwargs: Additional route options.

        Returns:
            Decorator function.
        """

        def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Any]:
            self.add_route(
                pattern,
                func,
                methods=methods,
                name=name or func.__name__,
                **kwargs,
            )
            return func

        return decorator

    def get(self, pattern: str, **kwargs: Any) -> Callable[..., Any]:
        """Decorator for GET routes."""
        return self.route(pattern, methods=["GET"], **kwargs)

    def post(self, pattern: str, **kwargs: Any) -> Callable[..., Any]:
        """Decorator for POST routes."""
        return self.route(pattern, methods=["POST"], **kwargs)

    def put(self, pattern: str, **kwargs: Any) -> Callable[..., Any]:
        """Decorator for PUT routes."""
        return self.route(pattern, methods=["PUT"], **kwargs)

    def patch(self, pattern: str, **kwargs: Any) -> Callable[..., Any]:
        """Decorator for PATCH routes."""
        return self.route(pattern, methods=["PATCH"], **kwargs)

    def delete(self, pattern: str, **kwargs: Any) -> Callable[..., Any]:
        """Decorator for DELETE routes."""
        return self.route(pattern, methods=["DELETE"], **kwargs)

    def match(self, path: str, method: str = "GET") -> RouteMatch:
        """Match a path and method to a route.

        Args:
            path: URL path to match.
            method: HTTP method.

        Returns:
            RouteMatch with handler and parameters.

        Raises:
            RouteNotFoundError: If no route matches path.
            MethodNotAllowedError: If route exists but method not allowed.
        """
        self._request_count += 1
        method = method.upper()
        matched_route: Route | None = None

        for route in self._routes:
            params = route.match(path)
            if params is not None:
                if method in route.methods or "*" in route.methods:
                    # Update metrics
                    route_key = f"{route.methods[0]}:{route.pattern}"
                    self._route_hits[route_key] = self._route_hits.get(route_key, 0) + 1

                    # Combine middleware
                    all_middleware = self.middleware + route.middleware

                    return RouteMatch(
                        route=route,
                        handler=route.handler,
                        params=params,
                        middleware=all_middleware,
                    )
                else:
                    # Path matches but method doesn't
                    matched_route = route
                    _ = params  # Params captured for potential future use

        if matched_route:
            raise MethodNotAllowedError(path, method, matched_route.methods)

        raise RouteNotFoundError(path, method)

    def get_route(self, name: str) -> Route | None:
        """Get route by name.

        Args:
            name: Route name.

        Returns:
            Route if found, None otherwise.
        """
        return self._named_routes.get(name)

    def url_for(self, name: str, **params: Any) -> str:
        """Generate URL for named route.

        Args:
            name: Route name.
            **params: Parameters to substitute.

        Returns:
            Generated URL.

        Raises:
            KeyError: If route not found.
        """
        route = self._named_routes.get(name)
        if not route:
            raise KeyError(f"Route not found: {name}")

        url = route.pattern
        for param_name, param_value in params.items():
            url = url.replace(f"{{{param_name}}}", str(param_value))

        return url

    def include_router(
        self,
        router: RequestRouter,
        prefix: str = "",
    ) -> None:
        """Include routes from another router.

        Args:
            router: Router to include.
            prefix: Additional prefix for included routes.
        """
        for route in router._routes:
            new_pattern = f"{prefix}{route.pattern}"
            self.add_route(
                new_pattern.replace(router.prefix, ""),
                route.handler,
                methods=route.methods,
                name=route.name,
                middleware=route.middleware,
                **route.metadata,
            )

    def get_routes(self) -> list[Route]:
        """Get all registered routes."""
        return list(self._routes)

    def get_metrics(self) -> dict[str, Any]:
        """Get routing metrics.

        Returns:
            Dictionary with request counts and route hits.
        """
        return {
            "total_requests": self._request_count,
            "routes_count": len(self._routes),
            "route_hits": dict(self._route_hits),
        }


class RouterGroup:
    """Group of routes with shared prefix and middleware.

    Example:
        >>> api = RouterGroup("/api/v1")
        >>> api.add_middleware(auth_middleware)
        >>>
        >>> @api.get("/users")
        ... async def list_users(request):
        ...     return {"users": []}
    """

    def __init__(
        self,
        prefix: str,
        router: RequestRouter | None = None,
        middleware: list[Callable[..., Any]] | None = None,
    ):
        """Initialize router group.

        Args:
            prefix: URL prefix for all routes in group.
            router: Parent router (creates new if None).
            middleware: Group-specific middleware.
        """
        self.prefix = prefix
        self.router = router or RequestRouter()
        self.middleware = middleware or []

    def route(self, pattern: str, **kwargs: Any) -> Callable[..., Any]:
        """Decorator to add route to group."""
        full_pattern = f"{self.prefix}{pattern}"
        group_middleware = kwargs.pop("middleware", [])
        all_middleware = self.middleware + group_middleware

        return self.router.route(full_pattern, middleware=all_middleware, **kwargs)

    def get(self, pattern: str, **kwargs: Any) -> Callable[..., Any]:
        """Decorator for GET routes."""
        return self.route(pattern, methods=["GET"], **kwargs)

    def post(self, pattern: str, **kwargs: Any) -> Callable[..., Any]:
        """Decorator for POST routes."""
        return self.route(pattern, methods=["POST"], **kwargs)

    def add_middleware(self, middleware: Callable[..., Any]) -> None:
        """Add middleware to group."""
        self.middleware.append(middleware)
