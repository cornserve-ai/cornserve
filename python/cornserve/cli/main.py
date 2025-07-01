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
from cornserve.services.gateway.app.models import AppState
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
        rich.print(Panel(f"Failed to process registration: {e}", style="red", expand=False))
        return

    current_alias = alias or path.stem
    console = rich.get_console()

    # Parse responses from single stream
    initial_resp = None
    final_resp = None

    response_iter = response.iter_lines(decode_unicode=True)

    # Get immediate initial response
    for line in response_iter:
        if line and line.startswith("data: "):
            data = json.loads(line[6:])
            if data.get("type") == "error_response":
                error_msg = data.get("message", "Registration failed without details")
                rich.print(Panel(f"Registration failed: {error_msg}", style="red", expand=False))
                return
            elif data.get("type") == "initial_response":
                initial_resp = data
                break

    if not initial_resp or not initial_resp.get("app_id"):
        rich.print(Panel("Invalid initial response from gateway", style="red", expand=False))
        return

    app_id = initial_resp["app_id"]
    task_names = initial_resp.get("task_names", [])

    # Set up alias and show initial registration info
    Alias().set(app_id, current_alias)

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
    log_streamer = None
    if task_names:
        log_streamer = LogStreamer(task_names, console=console)
        if log_streamer.k8s_available:
            log_streamer.start()
        else:
            rich.print(
                Panel(
                    Text("Could not connect to Kubernetes cluster. Logs will not be streamed.", style="yellow"),
                    title="[bold yellow]Log Streaming[/bold yellow]",
                    border_style="dim",
                )
            )

    # Wait for final response with spinner
    spinner_message = f" Registering app '{app_id}'... Deploying tasks"
    try:
        with Status(spinner_message, spinner="dots", console=console) as status:
            for line in response_iter:
                if line and line.startswith("data: "):
                    data = json.loads(line[6:])
                    if data.get("type") == "error_response":
                        error_msg = data.get("message", "Registration failed")
                        final_resp = {"status": "registration_failed", "message": error_msg}
                        status.update(status=Text(f" Registration error: {error_msg}", style="red"))
                        break
                    elif data.get("type") == "final_response":
                        final_resp = data
                        final_status = data.get("status", "unknown")
                        final_message = data.get("message", "Registration completed")
                        style = "green" if final_status == "ready" else "red"
                        status.update(status=Text(f" Registering app '{app_id}'... {final_message}", style=style))
                        break
    finally:
        if log_streamer:
            log_streamer.stop()

    # Show final result
    if final_resp:
        final_status = final_resp.get("status", "unknown")
        final_message = final_resp.get("message", "Registration completed")
    else:
        final_status = "unknown"
        final_message = "Failed to receive or parse final response"

    if final_status == "ready":
        rich.print(
            Panel(f"App '{app_id}' registered successfully with alias '{current_alias}'.", style="green", expand=False)
        )
    else:
        Alias().remove(current_alias)
        rich.print(
            Panel(
                f"App '{app_id}' status: {final_status}. {final_message}\nAlias '{current_alias}' removed.",
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
