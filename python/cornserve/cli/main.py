"""Cornserve CLI entry point."""

from __future__ import annotations

import json
import os
import sys
from contextlib import suppress
from pathlib import Path
from typing import Annotated, Any

import requests
import rich
import tyro
import yaml
from rich import box
from rich.live import Live
from rich.panel import Panel
from rich.status import Status
from rich.table import Table
from rich.text import Text
from tyro.constructors import PrimitiveConstructorSpec

from cornserve.cli.log_streamer import LogStreamer
from cornserve.services.gateway.models import (
    AppInvocationRequest,
    AppRegistrationRequest,
    UnitTasksDeploymentRequest,
    CompositeTasksDeploymentRequest,
    RegistrationErrorResponse,
    RegistrationFinalResponse,
    RegistrationInitialResponse,
    RegistrationStatusEvent,
)

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
    kube_config_path: Path | None = None,
) -> None:
    """Register an app with the Cornserve gateway.

    Args:
        path: Path to the app's source file.
        alias: Optional alias for the app.
        kube_config_path: Optional path to the Kubernetes config file.
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
            # For parsing the line, see gateway.router.register_app for SSE format
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
                log_streamer = LogStreamer(task_names, console=console, kube_config_path=kube_config_path)
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
                    # For parsing the line, see gateway.router.register_app for SSE format
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
    ],
    aggregate_keys: list[str] | None = None,
) -> None:
    """Invoke an app with the given data.

    Args:
        app_id_or_alias: ID of the app to invoke or its alias.
        data: Input data for the app. This can be a literal JSON string,
            a path to either a JSON or YAML file, or a hyphen to read in from stdin.
        aggregate_keys: Optional list of keys to aggregate streaming responses by. If provided,
            the CLI will fetch the value of each streamed response object by these keys, treat
            values as strings, and accumulate (concatenate) them across all streamed objects.
            Keys can use dot notation to access nested fields (e.g., "choices.0.delta.content").
            Pure numbers will be cast to integers to index into lists. If not specified, each
            response chunk (likely JSON) will be displayed as a new row in the table.
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

    try:
        raw_response = requests.post(
            f"{GATEWAY_URL}/app/invoke/{app_id}",
            json=request.model_dump(),
            stream=True,  # Always enable streaming to detect response type
            timeout=(5, 300),  # Connection timeout and read timeout
        )

        if raw_response.status_code == 404:
            rich.print(Panel(f"App {app_id} not found.", style="red", expand=False))
            return

        raw_response.raise_for_status()

        if "text/plain" in raw_response.headers.get("content-type", ""):
            _handle_streaming_response(raw_response, aggregate_keys)
        else:
            _handle_non_streaming_response(raw_response)

    except Exception as e:
        rich.print(Panel(f"Failed to invoke app: {e}", style="red", expand=False))


def _create_response_table(data: dict[str, Any], fields: list[str] | None = None) -> Table:
    """Create a table from the response data."""
    table = Table(box=box.ROUNDED, show_header=False, show_lines=True)
    for key in fields or data.keys():
        table.add_row(key, str(data[key]))
    return table


def _handle_non_streaming_response(response: requests.Response) -> None:
    """Handle non-streaming response."""
    # Collect all data since we opened stream=True but it's actually not streaming
    content = b"".join(response.iter_content())
    data = json.loads(content.decode())
    rich.print(_create_response_table(data))


def _handle_streaming_response(
    response: requests.Response,
    aggregate_keys: list[str] | None = None,
) -> None:
    """Handle streaming response with live-updating table.

    If aggregate_keys is provided, accumulates values for those keys across all streaming responses.
    Keys support dot notation (e.g., "choices.0.delta.content") and pure numbers are cast to integers.

    If aggregate_keys is None, displays each JSON response as a new table row with an incremented index.
    """
    console = rich.get_console()

    if aggregate_keys:
        # Aggregation mode: accumulate values for specified keys
        accumulated_data: dict[str, str] = {key: "" for key in aggregate_keys}

        try:
            with Live("Waiting for response...") as live:
                for line in response.iter_lines(chunk_size=None, decode_unicode=True):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        # Parse each JSON response
                        response_data = json.loads(line)

                        # Extract and accumulate values for each aggregate key
                        for key in aggregate_keys:
                            parts = key.split(".")
                            value = response_data
                            try:
                                for part in parts:
                                    # Try to convert to int for list indexing
                                    with suppress(ValueError):
                                        part = int(part)
                                    value = value[part]
                                # If the final value is None, skip
                                if value is not None:
                                    accumulated_data[key] += str(value)
                            except (KeyError, IndexError, TypeError):
                                # Key doesn't exist in this response, skip
                                pass

                        # Update the live table
                        table = _create_response_table(accumulated_data, aggregate_keys)
                        live.update(table, refresh=True)

                    except json.JSONDecodeError as e:
                        rich.print(Panel(f"Failed to parse JSON response: {e}", style="red", expand=False))
                        break

            # Final newline after live display ends
            console.print()

        except Exception as e:
            rich.print(Panel(f"Error processing streaming response: {e}", style="red", expand=False))
    else:
        # Default mode: show each line as a new row
        accumulated_data: dict[str, str] = {}

        try:
            with Live("Waiting for response...") as live:
                for line_idx, line in enumerate(response.iter_lines(chunk_size=None, decode_unicode=True)):
                    line = line.strip()
                    if not line:
                        continue

                    accumulated_data[str(line_idx)] = line
                    table = _create_response_table(accumulated_data)
                    live.update(table, refresh=True)

            # Final newline after live display ends
            console.print()

        except Exception as e:
            rich.print(Panel(f"Error processing streaming response: {e}", style="red", expand=False))


@app.command(name="deploy_tasklib")
def deploy_tasklib() -> None:
    """Scan cornserve_tasklib and deploy tasks/descriptors automatically."""
    try:
        from cornserve.cli.tasklib_explorer import discover_tasklib
    except Exception as e:
        rich.print(Panel(f"Failed to import tasklib explorer: {e}", style="red", expand=False))
        return

    try:
        unit_task_entries, composite_task_entries, descriptor_entries = discover_tasklib()
    except Exception as e:
        rich.print(Panel(f"Failed to explore cornserve_tasklib: {e}", style="red", expand=False))
        return

    # Deploy unit tasks + descriptors
    if unit_task_entries or descriptor_entries:
        try:
            rich.print("Deploying unit tasks and descriptors (auto-discovered)...")
            payload = UnitTasksDeploymentRequest(
                task_definitions=unit_task_entries,  # type: ignore[arg-type]
                descriptor_definitions=descriptor_entries,  # type: ignore[arg-type]
            )
            resp = requests.post(f"{GATEWAY_URL}/builtins/deploy-unit-tasks", json=payload.model_dump())
            resp.raise_for_status()
            rich.print(Panel("Unit tasks/descriptors deployed.", style="green", expand=False))
        except Exception as e:
            rich.print(Panel(f"Failed to deploy unit tasks/descriptors: {e}", style="red", expand=False))
            return
    else:
        rich.print(Panel("No unit tasks/descriptors discovered.", style="yellow", expand=False))

    # Deploy composite tasks
    if composite_task_entries:
        try:
            rich.print("Deploying composite tasks (auto-discovered)...")
            payload = CompositeTasksDeploymentRequest(
                task_definitions=composite_task_entries,  # type: ignore[arg-type]
            )
            resp = requests.post(f"{GATEWAY_URL}/builtins/deploy-composite-tasks", json=payload.model_dump())
            resp.raise_for_status()
            rich.print(Panel("Composite tasks deployed.", style="green", expand=False))
        except Exception as e:
            rich.print(Panel(f"Failed to deploy composite tasks: {e}", style="red", expand=False))
            return
    else:
        rich.print(Panel("No composite tasks discovered.", style="yellow", expand=False))

    rich.print(Panel("Tasklib deployment complete.", style="green", expand=False))


def main() -> None:
    """Main entry point for the Cornserve CLI."""
    app.cli(description="Cornserve CLI")
