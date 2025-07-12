"""The App Manager registers, invokes, and unregisters applications."""

from __future__ import annotations

import asyncio
import importlib.util
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator
from types import ModuleType
from typing import Any, get_args, get_origin, get_type_hints

from opentelemetry import trace
from pydantic import BaseModel

from cornserve.app.base import AppConfig
from cornserve.logging import get_logger
from cornserve.services.gateway.app.models import AppComponents, AppDefinition, AppState
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


def validate_app_module(module: ModuleType) -> AppComponents:
    """Validate that a module contains the required classes and function."""
    errors = []

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

    # Extract request and response classes from serve function annotations
    # Expectation is async def serve([ANYTHING]: Request) -> Response | AsyncIterator[Response]
    serve_signature = get_type_hints(module.serve)
    if len(serve_signature) != 2:
        errors.append("'serve' function must have exactly one parameter and a return type annotation")
        raise ValueError("\n".join(errors))
    request_type = next(iter(serve_signature.values()), None)
    return_type = serve_signature.pop("return", None)

    if request_type is None or return_type is None:
        errors.append("'serve' function must have both parameter and return type annotations")
        raise ValueError("\n".join(errors))
    else:
        # Validate request class
        if not hasattr(module, request_type.__name__):
            errors.append(f"Request class '{request_type.__name__}' is not defined in the module")
        elif not issubclass(request_type, BaseModel):
            errors.append(f"Request class '{request_type.__name__}' must inherit from pydantic.BaseModel")

        # Validate response class
        response_type = None
        origin = get_origin(return_type)
        if origin is AsyncIterator:
            response_type = get_args(return_type)[0]
            is_streaming = True
        else:
            response_type = return_type
            is_streaming = False

        if not hasattr(module, response_type.__name__):
            errors.append(f"Response class '{response_type.__name__}' is not defined in the module")
        elif not issubclass(response_type, BaseModel):
            errors.append(f"Response class '{response_type.__name__}' must inherit from pydantic.BaseModel")

    if errors:
        raise ValueError("\n".join(errors))

    return AppComponents(
        request_cls=getattr(module, request_type.__name__),
        response_cls=getattr(module, response_type.__name__),
        config_cls=module.Config,
        serve_fn=module.serve,  # type: ignore
        is_streaming=is_streaming,
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

    @tracer.start_as_current_span(name="AppManager.validate_and_create_app")
    async def validate_and_create_app(self, source_code: str) -> tuple[str, list[str]]:
        """Validate and create an app from source code.

        Args:
            source_code: Python source code of the application

        Returns:
            tuple[str, list[str]]: A tuple containing the App ID and a list of unit task names

        Raises:
            ValueError: If app validation fails
        """
        span = trace.get_current_span()

        async with self.app_lock:
            # Generate a unique app ID
            while True:
                app_id = f"app-{uuid.uuid4().hex}"
                if app_id not in self.app_states:
                    break

        span.set_attribute("app_manager.validate_and_create_app.app_id", app_id)

        try:
            module = load_module_from_source(source_code, app_id)
            app_components = validate_app_module(module)
            tasks = discover_unit_tasks(app_components.config_cls.tasks.values())
            task_names = [t.execution_descriptor.create_executor_name().lower() for t in tasks]
        except (ImportError, ValueError) as e:
            raise ValueError(f"App source code validation failed: {e}") from e

        async with self.app_lock:
            self.apps[app_id] = AppDefinition(
                app_id=app_id,
                module=module,
                components=app_components,
                source_code=source_code,
                tasks=tasks,
            )
            self.app_states[app_id] = AppState.NOT_READY

        return app_id, task_names

    @tracer.start_as_current_span(name="AppManager.deploy_app_tasks")
    async def deploy_app_tasks(self, app_id: str) -> None:
        """Deploy tasks for an app and return final result.

        Args:
            app_id: The App's ID

        Raises:
            RuntimeError: If task deployment gets any Exception
        """
        span = trace.get_current_span()
        span.set_attribute("app_manager.deploy_app_tasks.app_id", app_id)

        tasks_to_deploy = []
        try:
            async with self.app_lock:
                app_def = self.apps[app_id]

            tasks_to_deploy = app_def.tasks

            # Deploy unit tasks
            await self.task_manager.declare_used(tasks_to_deploy)

            # Update app state
            async with self.app_lock:
                self.app_states[app_id] = AppState.READY

            logger.info("Successfully deployed %s tasks for app '%s'.", len(tasks_to_deploy), app_id)

        except Exception as e:
            logger.exception("Failed to deploy tasks (count: %s) for app '%s': %s", len(tasks_to_deploy), app_id, e)
            async with self.app_lock:
                self.apps.pop(app_id, None)
                self.app_states.pop(app_id, None)
                self.app_driver_tasks.pop(app_id, None)

            # Re-raise as a runtime error to be caught by the router
            raise RuntimeError(f"Failed to deploy tasks: {e}") from e

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
        tasks = discover_unit_tasks(app.components.config_cls.tasks.values())

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
            Response from the application or AsyncIterator for streaming responses

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
        request = app_def.components.request_cls(**request_data)

        # Invoke the app
        app_driver: asyncio.Task[BaseModel | AsyncIterator[BaseModel]] | None = None

        try:
            # Create a task to run the app
            app_driver = asyncio.create_task(app_def.components.serve_fn(request))

            async with self.app_lock:
                self.app_driver_tasks[app_id].append(app_driver)

            response = await app_driver

            # For non-streaming apps, validate the single response.
            if not app_def.components.is_streaming and not isinstance(response, app_def.components.response_cls):
                raise ValueError(
                    f"App returned invalid response type. "
                    f"Expected {app_def.components.response_cls.__name__}, "
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

    async def is_app_streaming(self, app_id: str) -> bool:
        """Check if an app is configured for streaming responses.

        Args:
            app_id: ID of the application to check

        Returns:
            bool: True if the app is streaming, False otherwise

        Raises:
            KeyError: If app_id doesn't exist
        """
        async with self.app_lock:
            if app_id not in self.apps:
                raise KeyError(f"App ID '{app_id}' does not exist")
            return self.apps[app_id].components.is_streaming

    async def list_apps(self) -> dict[str, AppState]:
        """List all registered applications and their states.

        Returns:
            dict[str, AppState]: Mapping of app IDs to their states
        """
        async with self.app_lock:  # Ensure thread-safe access for reading states
            return self.app_states.copy()

    async def shutdown(self) -> None:
        """Shut down the server."""
        await self.task_manager.shutdown()
