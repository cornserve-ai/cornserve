from __future__ import annotations
from pathlib import Path

from cornserve.services.gateway.models import (
    AppRegistrationRequest,
    RegistrationErrorResponse,
    RegistrationFinalResponse,
    RegistrationInitialResponse,
    RegistrationStatusEvent,
)
import requests

GATEWAY_URL = "http://localhost:30080"

def register_app(filename: str) -> str:
    """
    Register an app with the CornServe gateway, and return the app ID.
    """
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"File {filename} does not exist.")
    request = AppRegistrationRequest(source_code=path.read_text().strip())
    response = requests.post(
        f"{GATEWAY_URL}/app/register",
        json=request.model_dump(),
        timeout=(5, 1200),
        stream=True,
    )
    response.raise_for_status()
    response_iter = response.iter_lines(decode_unicode=True)

    app_id = None
    # Get immediate initial response
    for line in response_iter:
        if not line or not line.startswith("data: "):
            continue
        # For parsing the line, see gateway.router.register_app for SSE format
        event = RegistrationStatusEvent.model_validate_json(line[6:]).event
        if isinstance(event, RegistrationInitialResponse):
            if app_id is not None:
                raise RuntimeError("Received more than one initial responses.")
            app_id = event.app_id
        if isinstance(event, RegistrationErrorResponse):
            raise RuntimeError(f"Registration failed: {event.message}")
        if isinstance(event, RegistrationFinalResponse):
            break

    if not app_id:
        raise RuntimeError("No app ID received during registration.")
    return app_id

if __name__ == "__main__":
    epd_id = register_app("epd.py")
    print(f"Registered EPD app with ID: {epd_id}")
