"""The App Manager registers, invokes, and unregisters applications."""

from __future__ import annotations

import asyncio
import importlib.util
import uuid
from collections import defaultdict
from types import ModuleType
from typing import Any, get_type_hints

from opentelemetry import trace

from cornserve.app.base import AppConfig, AppRequest, AppResponse
from cornserve.logging import get_logger
from cornserve.services.gateway.app.models import AppClasses, AppDefinition, AppState
from cornserve.services.gateway.task_manager import TaskManager
from cornserve.task.base import discover_unit_tasks

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


def load_module_from_source(source_code: str, module_name: str) -> ModuleType:
    """Load a Python module from source code string.

    Creates an isolated module namespace without modifying sys.modules.
    """
    spec = importlib.util.spec_from_loader(module_name, loader=None, origin="<cornserve_app>")
    if spec is None:
        raise ImportError(f"Failed to create spec for module {module_name}")

    module = importlib.util.module_from_spec(spec)

    try:
        # Execute in isolated namespace without touching sys.modules
        exec(source_code, module.__dict__)
        return module
    except Exception as e:
        raise ImportError(f"Failed to execute module code: {e}") from e


def validate_app_module(module: ModuleType) -> AppClasses:
    """Validate that a module contains the required classes and function."""
    errors = []

    # Check Request class
    if not hasattr(module, "Request"):
        errors.append("Missing 'Request' class")
    elif not issubclass(module.Request, AppRequest):
        errors.append("'Request' class must inherit from cornserve.app.base.AppRequest")

    # Check Response class
    if not hasattr(module, "Response"):
        errors.append("Missing 'Response' class")
    elif not issubclass(module.Response, AppResponse):
        errors.append("'Response' class must inherit from cornserve.app.base.AppResponse")

    # Check Config class
    if not hasattr(module, "Config"):
        errors.append("Missing 'Config' class")
    elif not issubclass(module.Config, AppConfig):
        errors.append("'Config' class must inherit from cornserve.app.base.AppConfig")

    # Check serve function
    if not hasattr(module, "serve"):
        errors.append("Missing 'serve' function")
    elif not callable(module.serve):
        errors.append("'serve' must be a callable")
    elif not asyncio.iscoroutinefunction(module.serve):
        errors.append("'serve' must be an async function")

    # Validate serve function signature
    # Expectation is async def serve([ANYTHING]: Request) -> Response
    serve_signature = get_type_hints(module.serve)
    return_type = serve_signature.pop("return", None)
    if return_type is None:
        errors.append("'serve' function must have a return type annotation")
    elif not issubclass(return_type, module.Response):
        errors.append("'serve' function must return an instance of 'Response' class")
    if len(serve_signature) != 1:
        errors.append("'serve' function must have exactly one parameter of type 'Request'")
    request_type = next(iter(serve_signature.values()), None)
    assert request_type is not None
    if not issubclass(request_type, module.Request):
        errors.append("'serve' function must accept an instance of 'Request' class")

    if errors:
        raise ValueError("\n".join(errors))

    return AppClasses(
        request_cls=module.Request,
        response_cls=module.Response,
        config_cls=module.Config,
        serve_fn=module.serve,  # type: ignore
    )


class AppManager:
    """Manages registration and execution of user applications."""

    def __init__(self, task_manager: TaskManager) -> None:
        """Initialize the AppManager."""
        self.task_manager = task_manager

        # One lock protects all app-related state dicts below
        self.app_lock = asyncio.Lock()
        self.apps: dict[str, AppDefinition] = {}
        self.app_states: dict[str, AppState] = {}
        self.app_driver_tasks: dict[str, list[asyncio.Task]] = defaultdict(list)

    @tracer.start_as_current_span(name="AppManager.register_app")
    async def register_app(self, source_code: str) -> str:
        """Register a new application with the given ID and source code.

        This will deploy all unit tasks discovered in the app's config.

        Args:
            source_code: Python source code of the application

        Returns:
            str: The app ID

        Raises:
            ValueError: If app validation fails
            RuntimeError: If errors occur during registration
        """
        span = trace.get_current_span()
        async with self.app_lock:
            # Generate a unique app ID
            while True:
                app_id = f"app-{uuid.uuid4().hex}"
                if app_id not in self.app_states:
                    break

            self.app_states[app_id] = AppState.NOT_READY
        span.set_attribute("app_manager.register_app.app_id", app_id)

        try:
            # Load and validate the app
            module = load_module_from_source(source_code, app_id)
            app_classes = validate_app_module(module)
            tasks = discover_unit_tasks(app_classes.config_cls.tasks.values())

            # Deploy all unit tasks discovered
            await self.task_manager.declare_used(tasks)

            # Update app state and store app definition
            async with self.app_lock:
                self.app_states[app_id] = AppState.READY
                self.apps[app_id] = AppDefinition(
                    app_id=app_id,
                    module=module,
                    classes=app_classes,
                    source_code=source_code,
                )

            logger.info("Successfully registered app '%s'", app_id)

            return app_id

        except Exception as e:
            logger.exception("Failed to register app: %s", e)

            # Clean up any partially registered app
            async with self.app_lock:
                self.apps.pop(app_id, None)
                self.app_states.pop(app_id, None)
                self.app_driver_tasks.pop(app_id, None)

            raise RuntimeError(f"Failed to register app: {e}") from e

    @tracer.start_as_current_span(name="AppManager.unregister_app")
    async def unregister_app(self, app_id: str) -> None:
        """Unregister an application.

        Args:
            app_id: ID of the application to unregister

        Raises:
            KeyError: If app_id doesn't exist
            RuntimeError: If errors occur during unregistration
        """
        async with self.app_lock:
            if app_id not in self.apps:
                raise KeyError(f"App ID '{app_id}' does not exist")

            # Clean up app from internal state
            app = self.apps.pop(app_id)
            self.app_states.pop(app_id, None)

            # Cancel all running tasks
            for task in self.app_driver_tasks.pop(app_id, []):
                task.cancel()

        # Let the task manager know that this app no longer needs these tasks
        tasks = discover_unit_tasks(app.classes.config_cls.tasks.values())

        try:
            await self.task_manager.declare_not_used(tasks)
        except Exception as e:
            logger.exception("Errors while unregistering app '%s': %s", app_id, e)
            raise RuntimeError(f"Errors while unregistering app '{app_id}': {e}") from e

        logger.info("Successfully unregistered app '%s'", app_id)

    @tracer.start_as_current_span(name="AppManager.invoke_app")
    async def invoke_app(self, app_id: str, request_data: dict[str, Any]) -> Any:
        """Invoke an application with the given request data.

        Args:
            app_id: ID of the application to invoke
            request_data: Request data to pass to the application

        Returns:
            Response from the application

        Raises:
            KeyError: If app_id doesn't exist
            ValueError: On app invocation failure
            ValidationError: If request data is invalid
        """
        async with self.app_lock:
            if self.app_states[app_id] != AppState.READY:
                raise ValueError(f"App '{app_id}' is not ready")

            app_def = self.apps[app_id]

        # Parse and validate request data
        request = app_def.classes.request_cls(**request_data)

        # Invoke the app
        app_driver: asyncio.Task | None = None

        try:
            # Create a task to run the app
            app_driver = asyncio.create_task(app_def.classes.serve_fn(request))

            async with self.app_lock:
                self.app_driver_tasks[app_id].append(app_driver)

            response = await app_driver

            # Validate response
            if not isinstance(response, app_def.classes.response_cls):
                raise ValueError(
                    f"App returned invalid response type. "
                    f"Expected {app_def.classes.response_cls.__name__}, "
                    f"got {type(response).__name__}"
                )

            return response

        except asyncio.CancelledError:
            logger.info("App %s invocation cancelled", app_id)
            raise ValueError(
                f"App '{app_id}' invocation cancelled. The app may be shutting down.",
            ) from None

        except Exception as e:
            logger.exception("Error invoking app %s: %s", app_id, e)
            raise ValueError(f"Error invoking app {app_id}: {e}") from e

        finally:
            if app_driver:
                async with self.app_lock:
                    self.app_driver_tasks[app_id].remove(app_driver)

    async def list_apps(self) -> dict[str, AppState]:
        """List all registered applications and their states.

        Returns:
            dict[str, AppState]: Mapping of app IDs to their states
        """
        return self.app_states

    async def shutdown(self) -> None:
        """Shut down the server."""
        await self.task_manager.shutdown()
