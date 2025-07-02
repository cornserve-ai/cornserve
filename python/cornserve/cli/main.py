"""Cornserve CLI entry point."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Annotated, Any

import requests
import rich
import tyro
import yaml
from rich import box
from rich.panel import Panel
from rich.status import Status
from rich.table import Table
from rich.text import Text
from tyro.constructors import PrimitiveConstructorSpec

from cornserve.cli.log_streamer import LogStreamer
from cornserve.services.gateway.app.models import (
    RegistrationErrorResponse,
    RegistrationFinalResponse,
    RegistrationInitialResponse,
    RegistrationStatusEvent,
)
from cornserve.services.gateway.models import AppInvocationRequest, AppRegistrationRequest

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
) -> None:
    """Register an app with the Cornserve gateway.

    Args:
        path: Path to the app's source file.
        alias: Optional alias for the app.
    """
    current_alias = alias or path.stem
    aliases = Alias()

    try:
        aliases.set("pending-registration-no-id-yet", current_alias)
    except ValueError as e:
        rich.print(Panel(f"{e}", style="red", expand=False, title="Alias Error"))
        return

    request = AppRegistrationRequest(source_code=path.read_text().strip())

    try:
        response = requests.post(
            f"{GATEWAY_URL}/app/register",
            json=request.model_dump(),
            timeout=(5, 1200),  # Short connection timeout but longer timeout waiting for streaming response
            stream=True,
        )
        response.raise_for_status()
    except Exception as e:
        aliases.remove(current_alias)
        rich.print(Panel(f"Failed to process registration: {e}", style="red", expand=False))
        return

    console = rich.get_console()

    # Parse responses from stream
    response_iter = response.iter_lines(decode_unicode=True)

    app_id: str | None = None
    task_names: list[str] = []
    log_streamer: LogStreamer | None = None
    final_message: str | None = None
    success = False

    # Get immediate initial response
    for line in response_iter:
        if not line or not line.startswith("data: "):
            continue

        try:
            event = RegistrationStatusEvent.model_validate_json(line[6:]).event
        except Exception as e:
            rich.print(Panel(f"Failed to parse response from gateway: {e}", style="red", expand=False))
            aliases.remove(current_alias)
            return

        if isinstance(event, RegistrationInitialResponse):
            app_id = event.app_id
            task_names = event.task_names
            # Update alias with actual app ID
            aliases.set(app_id, current_alias)

            app_info_table = Table(box=box.ROUNDED)
            app_info_table.add_column("App ID")
            app_info_table.add_column("Alias")
            app_info_table.add_row(app_id, current_alias)
            rich.print(app_info_table)

            if task_names:
                tasks_table = Table(box=box.ROUNDED)
                tasks_table.add_column("Unit Tasks")
                for name in task_names:
                    tasks_table.add_row(name)
                rich.print(tasks_table)

                # Start log streamer
                log_streamer = LogStreamer(task_names, console=console)
                if log_streamer.k8s_available:
                    log_streamer.start()
                else:
                    rich.print(
                        Panel(
                            Text("Could not connect to Kubernetes cluster. Logs will not be streamed.", style="yellow")
                        )
                    )
            break

        if isinstance(event, RegistrationErrorResponse):
            rich.print(Panel(f"Registration failed: {event.message}", style="red", expand=False))
            aliases.remove(current_alias)
            return

    if not app_id:
        aliases.remove(current_alias)
        rich.print(Panel("Invalid initial response from gateway", style="red", expand=False))
        return

    # Wait for final response with spinner
    spinner_message = f" Registering app '{app_id}'. Waiting for tasks deployment..."
    try:
        with Status(spinner_message, spinner="dots", console=console):
            for line in response_iter:
                if not line or not line.startswith("data: "):
                    continue

                try:
                    event = RegistrationStatusEvent.model_validate_json(line[6:]).event
                except Exception as e:
                    final_message = f"Failed to parse response from gateway: {e}"
                    break

                if isinstance(event, RegistrationErrorResponse):
                    final_message = event.message
                    break
                if isinstance(event, RegistrationFinalResponse):
                    final_message = event.message
                    success = True
                    break
    finally:
        if log_streamer:
            log_streamer.stop()

    if success:
        rich.print(
            Panel(f"App '{app_id}' registered successfully with alias '{current_alias}'.", style="green", expand=False)
        )
    else:
        aliases.remove(current_alias)
        final_message = final_message or "Failed to receive or parse final response"
        rich.print(
            Panel(
                f"App '{app_id}' registration failed. {final_message}\nAlias '{current_alias}' removed.",
                style="red",
                expand=False,
            )
        )


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


def main() -> None:
    """Main entry point for the Cornserve CLI."""
    app.cli(description="Cornserve CLI")
