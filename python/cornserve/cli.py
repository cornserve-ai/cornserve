"""Cornserve CLI entry point."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Annotated, Any

import requests
import rich
import tyro
import yaml
from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.status import Status
from tyro.constructors import PrimitiveConstructorSpec

from cornserve.services.gateway.models import (
    AppInvocationRequest,
    AppRegistrationRequest,
    AppRegistrationResponse,
)
from cornserve.services.gateway.app.models import AppState
from cornserve.misc import LogStreamer


try:
    GATEWAY_URL = os.environ["CORNSERVE_GATEWAY_URL"]
except KeyError:
    print(
        "Environment variable CORNSERVE_GATEWAY_URL is not set. Defaulting to http://localhost:30080.\n",
    )
    GATEWAY_URL = "http://localhost:30080"

STATE_DIR = Path.home() / ".local/state/cornserve"
STATE_DIR.mkdir(parents=True, exist_ok=True)

app = tyro.extras.SubcommandApp()


def _load_payload(args: list[str]) -> dict[str, Any]:
    """Load a literal JSON or a JSON/YAML file."""
    payload = args[0]

    # A hyphen indicates stdin
    if payload == "-":
        payload = str(sys.stdin.read().strip())
    # An actual file path
    elif Path(payload).exists():
        payload = Path(payload).read_text().strip()

    # Now, payload should be either a literal JSON or YAML string
    json_error = None
    yaml_error = None

    try:
        return json.loads(payload)
    except json.JSONDecodeError as e:
        json_error = e

    try:
        return yaml.safe_load(payload)
    except yaml.YAMLError as e:
        yaml_error = e

    # Nothing worked, raise an error
    raise ValueError(
        f"Invalid payload format. JSON failed with: '{json_error}'. YAML failed with: '{yaml_error}'",
    )


class Alias:
    """App ID aliases."""

    def __init__(self, file_path: Path = STATE_DIR / "alias.json") -> None:
        """Initialize the Alias class."""
        self.file_path = file_path
        # Alias -> App ID
        self.aliases = {}
        if file_path.exists():
            with open(file_path) as file:
                self.aliases = json.load(file)

    def get(self, alias: str) -> str | None:
        """Get the app ID for an alias."""
        return self.aliases.get(alias)

    def reverse_get(self, app_id: str) -> str | None:
        """Get the alias for an app ID."""
        for alias, id_ in self.aliases.items():
            if id_ == app_id:
                return alias
        return None

    def set(self, app_id: str, alias: str) -> None:
        """Set an alias for an app ID."""
        if alias.startswith("app-"):
            raise ValueError("Alias cannot start with 'app-'")
        self.aliases[alias] = app_id
        with open(self.file_path, "w") as file:
            json.dump(self.aliases, file)

    def remove(self, alias: str) -> None:
        """Remove an alias for an app ID."""
        self.aliases.pop(alias, None)
        with open(self.file_path, "w") as file:
            json.dump(self.aliases, file)


@app.command(name="register")
def register(
    path: Annotated[Path, tyro.conf.Positional],
    alias: str | None = None,
    is_async: bool = False,
) -> None:
    """Register an app with the Cornserve gateway.

    Args:
        path: Path to the app's source file.
        alias: Optional alias for the app.
        is_async: If true, exit immediately after submission without waiting.
    """
    request = AppRegistrationRequest(source_code=path.read_text().strip())
    try:
        raw_response = requests.post(
            f"{GATEWAY_URL}/app/register",
            json=request.model_dump(),
            timeout=10,
        )
        raw_response.raise_for_status()
        response = AppRegistrationResponse.model_validate(raw_response.json())
        app_id = response.app_id
        task_names = response.task_names
    except requests.exceptions.RequestException as e:
        rich.print(Panel(f"Failed to send registration request: {e}", style="red", expand=False))
        return
    except Exception as e:
        rich.print(Panel(f"Failed to initiate registration: {e}", style="red", expand=False))
        return

    if not app_id:
        rich.print(Panel("Failed to get app_id from registration response.", style="red", expand=False))
        return

    current_alias = alias or path.stem
    Alias().set(app_id, current_alias)

    # Initial display before polling starts
    initial_table = Table(box=box.ROUNDED)
    initial_table.add_column("App ID")
    initial_table.add_column("Alias")
    initial_table.add_column("Status")
    initial_table.add_row(app_id, current_alias, Text(AppState.NOT_READY.value.title(), style="yellow"))
    rich.print(initial_table)

    if task_names:
        tasks_table = Table(box=box.ROUNDED, title="Discovered Unit Tasks")
        tasks_table.add_column("Task Name")
        for name in task_names:
            tasks_table.add_row(name)
        rich.print(tasks_table)
    
    if is_async:
        rich.print(Panel(f"App '{app_id}' with alias '{current_alias}' is submitted for registration.", style="green", expand=False))
        return

    console = rich.get_console()
    # Have a spinner while we are waiting
    status_str = AppState.NOT_READY.value
    spinner_message = f" Waiting for app '{app_id}' to initialize ... Current status: {status_str.title()}"

    log_streamer: LogStreamer | None = None
    try:
        with Status(spinner_message, spinner="dots", console=console) as status:
            start_time = time.time()
            streamer_attempted = False

            time.sleep(0.2)

            while status_str == AppState.NOT_READY.value:
                # Start log streamer after a delay
                if not streamer_attempted and (time.time() - start_time) > 3 and task_names:
                    log_streamer = LogStreamer(task_names, console=console)
                    if log_streamer.k8s_available:
                        log_streamer.start()
                    else:
                        rich.print(
                            Panel(
                                Text(
                                    "Could not connect to Kubernetes cluster. Logs will not be streamed.",
                                    style="yellow",
                                ),
                                title="[bold yellow]Log Streaming[/bold yellow]",
                                border_style="dim",
                            )
                        )
                    streamer_attempted = True  # Attempt to start only once

                try:
                    status_response = requests.get(f"{GATEWAY_URL}/app/status/{app_id}", timeout=5)
                    status_response.raise_for_status()
                    status_data = status_response.json()
                    status_str = status_data.get("status", "unknown").lower()

                    current_style = "yellow"
                    if status_str == AppState.READY.value:
                        current_style = "green"
                    elif status_str == AppState.REGISTRATION_FAILED.value:
                        current_style = "red"

                    spinner_message = f" Registering app '{app_id}'... Current status: {status_str.title()}"
                    status.update(status=Text(spinner_message, style=current_style))

                    if status_str != AppState.NOT_READY.value:
                        break
                    time.sleep(1)
                except requests.exceptions.Timeout:
                    spinner_message = f" Polling timeout for app '{app_id}'. Retrying..."
                    status.update(status=Text(spinner_message, style="orange"))
                    time.sleep(1)
                except requests.exceptions.RequestException as e:
                    status_str = "polling_error"
                    rich.print(Text(f"Error polling status for '{app_id}': {e}", style="red"))
                    break
                except Exception as e:
                    status_str = "unexpected_error"
                    rich.print(
                        Text(f"An unexpected error occurred while polling for '{app_id}': {e}", style="red"),
                    )
                    break
    finally:
        if log_streamer:
            log_streamer.stop()

    if status_str == AppState.READY.value:
        rich.print(Panel(f"App '{app_id}' registered successfully with alias '{current_alias}'.", style="green", expand=False))
    elif status_str == AppState.REGISTRATION_FAILED.value:
        Alias().remove(current_alias)
        rich.print(Panel(f"App '{app_id}' failed to register. Alias '{current_alias}' removed.", style="red", expand=False))
    elif status_str == "polling_error":
        rich.print(Panel(f"Could not determine final status for app '{app_id}' due to polling errors. Please check with 'cornserve list'.", style="red", expand=False))
    elif status_str == "unexpected_error":
        rich.print(Panel(f"An unexpected error occurred while checking status for '{app_id}'. Please check with 'cornserve list'.", style="red", expand=False))
    else:
        rich.print(Panel(f"App '{app_id}' registration ended with an inconclusive status: '{status_str.title()}'. Please check with 'cornserve list'.", style="yellow", expand=False))

@app.command(name="unregister")
def unregister(
    app_id_or_alias: Annotated[str, tyro.conf.Positional],
) -> None:
    """Unregister an app from Cornserve.

    Args:
        app_id_or_alias: ID of the app to unregister or its alias.
    """
    if app_id_or_alias.startswith("app-"):
        app_id = app_id_or_alias
    else:
        alias = Alias()
        app_id = alias.get(app_id_or_alias)
        if not app_id:
            rich.print(Panel(f"Alias {app_id_or_alias} not found.", style="red", expand=False))
            return
        alias.remove(app_id_or_alias)

    raw_response = requests.post(
        f"{GATEWAY_URL}/app/unregister/{app_id}",
    )
    if raw_response.status_code == 404:
        rich.print(Panel(f"App {app_id} not found.", style="red", expand=False))
        return

    raw_response.raise_for_status()

    rich.print(Panel(f"App {app_id} unregistered successfully.", expand=False))


@app.command(name="list")
def list_apps() -> None:
    """List all registered apps."""
    raw_response = requests.get(f"{GATEWAY_URL}/app/list")
    raw_response.raise_for_status()
    response: dict[str, str] = raw_response.json()

    alias = Alias()

    table = Table(box=box.ROUNDED)
    table.add_column("App ID")
    table.add_column("Alias")
    table.add_column("Status")
    for app_id, status in response.items():
        table.add_row(
            app_id, alias.reverse_get(app_id) or "", Text(status, style="green" if status == "ready" else "yellow")
        )
    rich.print(table)


@app.command(name="invoke")
def invoke(
    app_id_or_alias: Annotated[str, tyro.conf.Positional],
    data: Annotated[
        dict[str, Any],
        PrimitiveConstructorSpec(
            nargs=1,
            metavar="JSON|YAML",
            instance_from_str=_load_payload,
            is_instance=lambda x: isinstance(x, dict),
            str_from_instance=lambda d: [json.dumps(d)],
        ),
        tyro.conf.Positional,
    ],
) -> None:
    """Invoke an app with the given data.

    Args:
        app_id_or_alias: ID of the app to invoke or its alias.
        data: Input data for the app. This can be a literal JSON string,
            a path to either a JSON or YAML file, or a hyphen to read in from stdin.
    """
    if app_id_or_alias.startswith("app-"):
        app_id = app_id_or_alias
    else:
        alias = Alias()
        app_id = alias.get(app_id_or_alias)
        if not app_id:
            rich.print(Panel(f"Alias {app_id_or_alias} not found.", style="red", expand=False))
            return

    request = AppInvocationRequest(request_data=data)
    raw_response = requests.post(
        f"{GATEWAY_URL}/app/invoke/{app_id}",
        json=request.model_dump(),
    )

    if raw_response.status_code == 404:
        rich.print(Panel(f"App {app_id} not found.", style="red", expand=False))
        return

    raw_response.raise_for_status()

    table = Table(box=box.ROUNDED, show_header=False)
    for key, value in raw_response.json().items():
        table.add_row(key, value)
    rich.print(table)


@app.command(name="check-status")
def check_status(
    app_id_or_alias: Annotated[str, tyro.conf.Positional],
) -> None:
    """Check the registration status of an application."""
    if app_id_or_alias.startswith("app-"):
        app_id = app_id_or_alias
    else:
        alias = Alias()
        app_id = alias.get(app_id_or_alias)
        if not app_id:
            rich.print(Panel(f"Alias '{app_id_or_alias}' not found.", style="red", expand=False))
            return

    try:
        status_response = requests.get(f"{GATEWAY_URL}/app/status/{app_id}", timeout=5)

        if status_response.status_code == 404:
            rich.print(f"App '{app_id}' not found.")
            return

        status_response.raise_for_status()
        status_data = status_response.json()
        status_str = status_data.get("status", "unknown").lower()

        status_style = "yellow"
        if status_str == AppState.READY.value:
            status_style = "green"
        elif status_str == AppState.REGISTRATION_FAILED.value:
            status_style = "red"
        
        rich.print(f"Status for app '{app_id}': [{status_style}]{status_str.title()}[/{status_style}]")

    except requests.exceptions.RequestException as e:
        rich.print(Panel(f"Error checking status for '{app_id}': {e}", style="red", expand=False))
    except Exception as e:
        rich.print(Panel(f"Unexpected error while checking '{app_id}': {e}", style="red", expand=False))


def main() -> None:
    """Main entry point for the Cornserve CLI."""
    app.cli(description="Cornserve CLI")
