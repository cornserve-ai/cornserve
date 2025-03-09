"""Gateway FastAPI app definition."""

from fastapi import FastAPI, APIRouter, Request, Response, status
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ValidationError
from typing import Any

from cornserve.logging import get_logger
from cornserve.services.gateway.app.manager import AppManager

router = APIRouter()
logger = get_logger(__name__)


class RegisterAppRequest(BaseModel):
    app_id: str
    source_code: str


class AppRegistrationResponse(BaseModel):
    app_id: str


class AppRequest(BaseModel):
    request_data: dict[str, Any]


@router.post("/admin/register_app", response_model=AppRegistrationResponse)
async def register_app(request: RegisterAppRequest, raw_request: Request):
    """Register a new application with the given ID and source code."""
    app_manager: AppManager = raw_request.app.state.app_manager

    try:
        app_id = await app_manager.register_app(request.app_id, request.source_code)
        return AppRegistrationResponse(app_id=app_id)
    except ValueError as e:
        logger.info("Error while registering app {%s}: {%s}", request.app_id, e)
        return Response(status_code=status.HTTP_400_BAD_REQUEST, content=str(e))
    except Exception as e:
        logger.exception("Unexpected error while registering app {%s}", request.app_id)
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
        raise RequestValidationError(errors=e.errors())
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


def init_app_state(app: FastAPI) -> None:
    """Initialize the app state with required components."""
    app.state.app_manager = AppManager("resource-manager:50051")


def create_app() -> FastAPI:
    """Create a FastAPI app for the Gateway service."""
    app = FastAPI(title="Cornserve Gateway")
    app.include_router(router)
    init_app_state(app)
    return app
