"""Cornserve CLI entry point."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import requests
import rich
import tyro
from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cornserve.services.gateway.models import (
    AppInvocationRequest,
    AppRegistrationRequest,
    AppRegistrationResponse,
)

try:
    GATEWAY_URL = os.environ["CORNSERVE_GATEWAY_URL"]
except KeyError:
    print(
        "Environment variable CORNSERVE_GATEWAY_URL is not set. Defaulting to http://localhost:30080.\n",
    )
    GATEWAY_URL = "http://localhost:30080"

app = tyro.extras.SubcommandApp()


@app.command(name="register")
def register(
    path: Annotated[Path, tyro.conf.Positional],
) -> None:
    """Register an app with the Cornserve gateway.

    Args:
        path: Path to the app's source file.
    """
    request = AppRegistrationRequest(source_code=path.read_text().strip())
    raw_response = requests.post(
        f"{GATEWAY_URL}/app/register",
        json=request.model_dump(),
    )
    raw_response.raise_for_status()
    response = AppRegistrationResponse.model_validate(raw_response.json())

    app_id = response.app_id

    table = Table(box=box.ROUNDED)
    table.add_column("App ID")
    table.add_row(app_id)
    rich.print(table)


@app.command(name="unregister")
def unregister(
    app_id: Annotated[str, tyro.conf.Positional],
) -> None:
    """Unregister an app from Cornserve.

    Args:
        app_id: ID of the app to unregister.
    """
    raw_response = requests.post(
        f"{GATEWAY_URL}/app/unregister/{app_id}",
    )
    if raw_response.status_code == 404:
        rich.print(
            Panel(
                f"App {app_id} not found.",
                style="red",
                expand=False,
            )
        )
        return

    raw_response.raise_for_status()

    rich.print(
        Panel(
            f"App {app_id} unregistered successfully.",
            expand=False,
        )
    )


@app.command(name="list")
def list_apps() -> None:
    """List all registered apps."""
    raw_response = requests.get(f"{GATEWAY_URL}/app/list")
    raw_response.raise_for_status()
    response: dict[str, str] = raw_response.json()

    table = Table(box=box.ROUNDED)
    table.add_column("App ID")
    table.add_column("Status")
    for app_id, status in response.items():
        table.add_row(app_id, Text(status, style="green" if status == "ready" else "yellow"))
    rich.print(table)


@app.command(name="invoke")
def invoke(
    app_id: Annotated[str, tyro.conf.Positional],
    data: Annotated[str, tyro.conf.Positional],
) -> None:
    """Invoke an app with the given data.

    Args:
        app_id: ID of the app to invoke.
        data: Input data for the app, formatted as a JSON string.
    """
    print(data[0])
    print(data[-1])
    request = AppInvocationRequest(request_data=json.loads(data))
    raw_response = requests.post(
        f"{GATEWAY_URL}/app/invoke/{app_id}",
        json=request.model_dump(),
    )

    table = Table(box=box.ROUNDED, show_header=False)
    for key, value in raw_response.json().items():
        table.add_row(key, value)
    rich.print(table)


def main() -> None:
    """Main entry point for the Cornserve CLI."""
    app.cli(description="Cornserve CLI")
