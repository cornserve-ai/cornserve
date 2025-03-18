"""Gateway FastAPI app definition."""

from typing import Any

from fastapi import FastAPI, APIRouter, Request, Response, status
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ValidationError

from cornserve.services.gateway.app.manager import AppManager
from cornserve.constants import K8S_RESOURCE_MANAGER_GRPC_URL
from cornserve.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


class RegisterAppRequest(BaseModel):
    """Request for registering a new application.

    Attributes:
        app_id: The unique identifier for the application.
        source_code: The Python source code of the application.
    """

    source_code: str


class AppRegistrationResponse(BaseModel):
    """Response for registering a new application.

    Attributes:
        app_id: The unique identifier for the registered application.
    """

    app_id: str


class AppRequest(BaseModel):
    """Request for invoking a registered application.

    Attributes:
        request_data: The input data for the application. Should be a valid
            JSON object that matches the `Request` schema of the application.
    """

    request_data: dict[str, Any]


@router.post("/admin/register_app", response_model=AppRegistrationResponse)
async def register_app(request: RegisterAppRequest, raw_request: Request):
    """Register a new application with the given ID and source code."""
    app_manager: AppManager = raw_request.app.state.app_manager

    try:
        app_id = await app_manager.register_app(request.source_code)
        return AppRegistrationResponse(app_id=app_id)
    except ValueError as e:
        logger.info("Error while registering app: %s", e)
        return Response(status_code=status.HTTP_400_BAD_REQUEST, content=str(e))
    except Exception as e:
        logger.exception("Unexpected error while registering app")
        return Response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=str(e),
        )


@router.post("/admin/unregister_app/{app_id}")
async def unregister_app(app_id: str, raw_request: Request):
    """Unregister the application with the given ID."""
    app_manager: AppManager = raw_request.app.state.app_manager

    try:
        await app_manager.unregister_app(app_id)
        return Response(status_code=status.HTTP_200_OK)
    except KeyError as e:
        return Response(status_code=status.HTTP_404_NOT_FOUND, content=str(e))
    except Exception as e:
        logger.exception("Unexpected error while unregistering app")
        return Response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=str(e),
        )


@router.post("/v1/apps/{app_id}")
async def invoke_app(app_id: str, request: AppRequest, raw_request: Request):
    """Invoke a registered application."""
    app_manager: AppManager = raw_request.app.state.app_manager

    try:
        return await app_manager.invoke_app(app_id, request.request_data)
    except ValidationError as e:
        raise RequestValidationError(errors=e.errors()) from e
    except KeyError as e:
        return Response(status_code=status.HTTP_404_NOT_FOUND, content=str(e))
    except ValueError as e:
        logger.info("Error while running app {%s}: {%s}", app_id, e)
        return Response(status_code=status.HTTP_400_BAD_REQUEST, content=str(e))
    except Exception as e:
        logger.exception("Unexpected error while running app {%s}", app_id)
        return Response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=str(e),
        )


@router.get("/admin/apps")
async def list_apps(raw_request: Request):
    """List all registered applications."""
    app_manager: AppManager = raw_request.app.state.app_manager

    try:
        return await app_manager.list_apps()
    except Exception as e:
        logger.exception("Unexpected error while listing apps")
        return Response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=str(e),
        )


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return Response(status_code=status.HTTP_200_OK)


def init_app_state(app: FastAPI) -> None:
    """Initialize the app state with required components."""
    app.state.app_manager = AppManager(K8S_RESOURCE_MANAGER_GRPC_URL)


def create_app() -> FastAPI:
    """Create a FastAPI app for the Gateway service."""
    app = FastAPI(title="CornServe Gateway")
    app.include_router(router)
    init_app_state(app)
    return app
