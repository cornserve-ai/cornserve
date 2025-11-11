"""Cornserve CLI entry point."""

from __future__ import annotations

import base64
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
from kubernetes import client
from kubernetes.client.rest import ApiException
from rich import box
from rich.live import Live
from rich.panel import Panel
from rich.status import Status
from rich.table import Table
from rich.text import Text
from tyro.constructors import PrimitiveConstructorSpec

from cornserve.cli.audio_streamer import PCMStreamPlayer
from cornserve.cli.log_streamer import LogStreamer
from cornserve.cli.tasklib_explorer import discover_tasklib
from cornserve.cli.utils.k8s import load_k8s_config
from cornserve.constants import K8S_NAMESPACE, K8S_UNIT_TASK_PROFILES_CONFIG_MAP_NAME
from cornserve.services.gateway.models import (
    AppInvocationRequest,
    AppRegistrationRequest,
    RegistrationErrorResponse,
    RegistrationFinalResponse,
    RegistrationInitialResponse,
    RegistrationStatusEvent,
    TasksDeploymentRequest,
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


def _serialize_gr_command(**cmd) -> bytes:
    """Serialize a graphics command for Kitty TGP."""
    payload = cmd.pop("payload", None)
    cmd_str = ",".join(f"{k}={v}" for k, v in cmd.items())
    parts = [b"\033_G", cmd_str.encode("ascii")]
    if payload:
        parts.extend([b";", payload])
    parts.append(b"\033\\")
    return b"".join(parts)


def _write_png_chunked(data: bytes) -> None:
    """Write PNG data to terminal using Kitty TGP in chunks."""
    cmd = {"a": "T", "f": 100}
    while data:
        chunk, data = data[:4096], data[4096:]
        m = 1 if data else 0
        sys.stdout.buffer.write(_serialize_gr_command(payload=chunk, m=m, **cmd))
        sys.stdout.flush()
        cmd.clear()
    print()


def _extract_nested_value(data: Any, key: str) -> Any:
    """Extract a value from nested dict using dot notation."""
    parts = key.split(".")
    value = data
    try:
        for part in parts:
            # Try to convert to int for list indexing
            with suppress(ValueError):
                part = int(part)
            value = value[part]
        return value
    except (KeyError, IndexError, TypeError):
        return None


def _handle_png_from_response(response_data: dict[str, Any], png_key: str, save_path: str | None = None) -> None:
    """Extract PNG from response data and display/save it if the key exists."""
    png_value = _extract_nested_value(response_data, png_key)
    if png_value is None:
        rich.print(Panel(f"PNG key '{png_key}' not found in response", style="red", expand=False))
        return

    if not isinstance(png_value, str):
        rich.print(Panel(f"PNG key '{png_key}' does not contain string data", style="red", expand=False))
        return

    # Save to file if path is specified
    if save_path:
        try:
            with open(save_path, "wb") as f:
                f.write(base64.b64decode(png_value))
            rich.print(Panel(f"PNG saved to {save_path}", style="green", expand=False))
        except Exception as e:
            rich.print(Panel(f"Failed to save PNG to {save_path}: {e}", style="red", expand=False))

    # Print out the PNG image using the Kitty Terminal Graphics Protocol.
    # Nothing should happen if the terminal does not support it (e.g., tmux).
    try:
        _write_png_chunked(png_value.encode("ascii"))
    except Exception as e:
        rich.print(Panel(f"Failed to display PNG in terminal: {e}", style="red", expand=False))


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
    kube_config_path: str | None = None,
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
    png_key: str | None = None,
    save_png_path: str | None = None,
    audio_key: str | None = None,
    audio_sample_rate: int | None = None,
    audio_channels: int | None = None,
    audio_pcm_format: str | None = None,
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
        png_key: Optional key in the response containing a base64-encoded PNG image.
            Supports dot notation for nested fields (e.g., "image.data"). If specified,
            the PNG will be displayed using Kitty TGP (if supported) and/or saved to file.
        save_png_path: Optional path to save the PNG file. If specified along with png_key,
            the PNG data will be decoded and saved to this file path.
        audio_key: Optional key indicating the field in the response containing generated audio.
            Supports dot notation for nested fields. If specified, the audio will be played from
            the device that the CLI is running on.
        audio_sample_rate: If audio-key is provided, audio-sample-rate can also optionally be
            provided to specify the sample rate for the audio to be played.
        audio_channels: If audio-key is provided, audio-channels can also optionally be provided
            to specify the number of channels for the audio to be played.
        audio_pcm_format: If audio-key is provided, audio-pcm-format can also optionally be
            provided to specify as a string the PCM format for the audio to be played.
            Currently supported formats: pcm16, pcm24, and pcm32.
    """
    if not audio_key and (audio_sample_rate or audio_channels or audio_pcm_format):
        rich.print(
            Panel(
                "To specify audio-sample-rate, audio-channels, or audio-pcm-format, must also specify audio-key.",
                style="red",
                expand=False,
            )
        )
        return

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
            if png_key:
                rich.print(Panel("PNG display is not supported for streaming responses", style="red", expand=False))
                return
            if audio_key:
                _handle_streaming_audio_response(
                    raw_response, audio_key, aggregate_keys, audio_sample_rate, audio_channels, audio_pcm_format
                )
            else:
                _handle_streaming_response(raw_response, aggregate_keys)
        else:
            if audio_key:
                rich.print(
                    Panel("Audio playback is not supported for non-streaming responses", style="red", expand=False)
                )
                return
            _handle_non_streaming_response(raw_response, png_key, save_png_path)

    except requests.exceptions.HTTPError as e:
        error_details = f"HTTP {e.response.status_code}: {e.response.reason}"
        try:
            # Try to extract error details from response body
            if e.response.headers.get("content-type", "").startswith("application/json"):
                error_body = e.response.json()
                if isinstance(error_body, dict) and "detail" in error_body:
                    error_details += f"\nDetails: {error_body['detail']}"
                elif isinstance(error_body, dict) and "message" in error_body:
                    error_details += f"\nMessage: {error_body['message']}"
                else:
                    error_details += f"\nResponse: {error_body}"
            else:
                # For non-JSON responses, show the raw text
                response_text = e.response.text.strip()
                if response_text:
                    error_details += f"\nResponse: {response_text}"
        except Exception:
            # If we can't parse the response, just show what we have
            pass
        rich.print(Panel(f"Failed to invoke app: {error_details}", style="red", expand=False))
    except Exception as e:
        rich.print(Panel(f"Failed to invoke app: {e}", style="red", expand=False))


def _create_response_table(data: dict[str, Any], fields: list[str] | None = None) -> Table:
    """Create a table from the response data."""
    table = Table(box=box.ROUNDED, show_header=False, show_lines=True)
    for key in fields or data.keys():
        table.add_row(key, str(data[key]))
    return table


def _handle_non_streaming_response(
    response: requests.Response, png_key: str | None = None, save_png_path: str | None = None
) -> None:
    """Handle non-streaming response."""
    # Collect all data since we opened stream=True but it's actually not streaming
    content = b"".join(response.iter_content(chunk_size=None))
    data = json.loads(content.decode())

    # Handle PNG if requested
    if png_key:
        _handle_png_from_response(data, png_key, save_png_path)

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
            with Live("Waiting for response...", vertical_overflow="visible") as live:
                for line in response.iter_lines(chunk_size=None, decode_unicode=True):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        # Parse each JSON response
                        response_data = json.loads(line)

                        # Extract and accumulate values for each aggregate key
                        for key in aggregate_keys:
                            if (value := _extract_nested_value(response_data, key)) is not None:
                                accumulated_data[key] += str(value)

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


def _handle_streaming_audio_response(
    response: requests.Response,
    audio_key: str,
    aggregate_keys: list[str] | None = None,
    audio_sample_rate: int | None = None,
    audio_channels: int | None = None,
    audio_pcm_format: str | None = None,
) -> None:
    """Handle streaming response with live-updating table.

    Args:
        response: A response from which the result can be streamed.
        audio_key: Key of the field containing base64-encoded wav bytes in the response. Supports
            dot notation for nested fields (e.g., "choices.0.delta.wav"). Aaudio in the response
            will be played from the device that the CLI is running on.
        aggregate_keys: If provided, values for these keys across all streaming responses will
            be accumulated. Keys support dot notation (e.g., "choices.0.delta.content") and pure
            numbers are cast to integers. If aggregate_keys is None, only audio will be played.
        audio_sample_rate: Can optionally be supplied to specify the sample rate for the audio to be
            played. Otherwise, the `PCMStreamPlayer` class will choose a default.
        audio_channels: Can optionally be supplied to specify the number of channels for the audio to
            be played. Otherwise, the `PCMStreamPlayer` class will choose a default.
        audio_pcm_format: Can optionally be supplied as a str to specify the PCM format for the audio
            to be played. Otherwise, the `PCMStreamPlayer` class will choose a default.
            Currently supported formats: pcm16, pcm24, and pcm32.
    """
    console = rich.get_console()

    # If aggregation mode: accumulate values for specified keys
    accumulated_data = {key: "" for key in aggregate_keys} if aggregate_keys else {}
    audio_panel_shown = False

    try:
        with (
            PCMStreamPlayer(
                sample_rate=audio_sample_rate, channels=audio_channels, pcm_format=audio_pcm_format
            ) as player,
            Live("Waiting for response...", vertical_overflow="visible") as live,
        ):
            for line in response.iter_lines(chunk_size=None, decode_unicode=True):
                line = line.strip()
                if not line:
                    continue

                try:
                    # Parse each JSON response
                    response_data = json.loads(line)

                    value = _extract_nested_value(response_data, audio_key)
                    if value is not None:
                        pcm_bytes = base64.b64decode(str(value))
                        player.feed(pcm_bytes)

                        if not aggregate_keys and not audio_panel_shown:
                            # Since we won't be showing a table, show text instead.
                            live.update(Panel("Receiving audio...", style="green"))
                            audio_panel_shown = True

                    if aggregate_keys:
                        # Extract and accumulate values for each aggregate key
                        for key in aggregate_keys:
                            if (value := _extract_nested_value(response_data, key)) is not None:
                                accumulated_data[key] += str(value)

                        # Show live updated table
                        table = _create_response_table(accumulated_data, aggregate_keys)
                        live.update(table, refresh=True)

                except json.JSONDecodeError as e:
                    rich.print(Panel(f"Failed to parse JSON response: {e}", style="red", expand=False))
                    break

        # Final newline after live display ends
        console.print()

    except Exception as e:
        rich.print(Panel(f"Error processing audio streaming response: {e}", style="red", expand=False))


@app.command(name="deploy_tasklib")
def deploy_tasklib() -> None:
    """Scan cornserve_tasklib and deploy tasks/descriptors automatically.

    We firstly deploy unit tasks and descriptors because the composite ones depend on them.
    """
    try:
        unit_task_entries, composite_task_entries, descriptor_entries = discover_tasklib()
    except Exception as e:
        rich.print(Panel(f"Failed to explore cornserve_tasklib: {e}", style="red", expand=False))
        return

    # Deploy unit tasks + descriptors
    if unit_task_entries or descriptor_entries:
        try:
            rich.print("Deploying unit tasks and descriptors ...")
            payload = TasksDeploymentRequest(
                task_definitions=unit_task_entries,
                descriptor_definitions=descriptor_entries,
            )
            resp = requests.post(f"{GATEWAY_URL}/deploy-tasks", json=payload.model_dump())
            resp.raise_for_status()
            unit_list = ", ".join(e.task_class_name for e in unit_task_entries) or "-"
            desc_list = ", ".join(e.descriptor_class_name for e in descriptor_entries) or "-"
            rich.print(
                Panel(
                    f"Unit tasks/descriptors deployed: unit={unit_list}; descriptors={desc_list}",
                    style="green",
                    expand=False,
                )
            )
        except Exception as e:
            rich.print(Panel(f"Failed to deploy unit tasks/descriptors: {e}", style="red", expand=False))
            return
    else:
        rich.print(Panel("No unit tasks/descriptors discovered.", style="yellow", expand=False))

    # Deploy composite tasks
    if composite_task_entries:
        try:
            rich.print("Deploying composite tasks ...")
            payload = TasksDeploymentRequest(
                task_definitions=composite_task_entries,
                descriptor_definitions=[],
            )
            resp = requests.post(f"{GATEWAY_URL}/deploy-tasks", json=payload.model_dump())
            resp.raise_for_status()
            comp_list = ", ".join(e.task_class_name for e in composite_task_entries)
            rich.print(
                Panel(
                    f"Composite tasks deployed: {comp_list}",
                    style="green",
                    expand=False,
                )
            )
        except Exception as e:
            rich.print(Panel(f"Failed to deploy composite tasks: {e}", style="red", expand=False))
            return
    else:
        rich.print(Panel("No composite tasks discovered.", style="yellow", expand=False))

    rich.print(Panel("Tasklib deployment complete.", style="green", expand=False))


@app.command(name="deploy_profiles")
def profile_deploy(
    profiles_dir: Annotated[Path, tyro.conf.Positional] = Path("profiles"),
    kube_config_path: str | None = None,
    dry_run: bool = False,
) -> None:
    """Deploy UnitTask profiles to Kubernetes as a ConfigMap.

    Args:
        profiles_dir: Directory containing profile JSON files (default: ./profiles).
        kube_config_path: Optional path to the Kubernetes config file.
        dry_run: Show what would be deployed without actually deploying.
    """
    if not profiles_dir.exists():
        rich.print(Panel(f"Profiles directory {profiles_dir} does not exist.", style="red", expand=False))
        return

    # Load Kubernetes config
    try:
        load_k8s_config(kube_config_path, ["/etc/rancher/k3s/k3s.yaml"])
    except Exception as e:
        rich.print(Panel(f"Failed to load Kubernetes config: {e}", style="red", expand=False))
        return

    # Collect all profile JSON files
    profile_files = list(profiles_dir.glob("*.json"))
    if not profile_files:
        rich.print(Panel(f"No JSON files found in {profiles_dir}.", style="yellow", expand=False))
        return

    # Validate and load profile files
    config_data = {}
    validation_errors = []

    for file_path in profile_files:
        try:
            with open(file_path) as f:
                # Validate JSON format
                profile_data = json.load(f)
                # Store relative filename as key
                config_data[file_path.name] = json.dumps(profile_data, indent=2)
        except Exception as e:
            validation_errors.append(f"{file_path.name}: {e}")

    if validation_errors:
        rich.print(Panel("Profile validation errors:\n" + "\n".join(validation_errors), style="red", expand=False))
        return

    # Display what will be deployed
    table = Table(box=box.ROUNDED)
    table.add_column("Profile File")
    table.add_column("Content")
    for filename, content in config_data.items():
        table.add_row(filename, content)

    rich.print(f"Found {len(config_data)} profile(s) to deploy:")
    rich.print(table)

    if dry_run:
        rich.print(Panel("Dry run completed. No changes made.", style="yellow", expand=False))
        return

    # Create or update ConfigMap
    try:
        v1 = client.CoreV1Api()
        map_name = K8S_UNIT_TASK_PROFILES_CONFIG_MAP_NAME

        configmap = client.V1ConfigMap(
            metadata=client.V1ObjectMeta(
                name=map_name,
                namespace=K8S_NAMESPACE,
            ),
            data=config_data,
        )

        # Try to get existing ConfigMap
        try:
            v1.read_namespaced_config_map(name=map_name, namespace=K8S_NAMESPACE)
            # ConfigMap exists, update it
            v1.replace_namespaced_config_map(name=map_name, namespace=K8S_NAMESPACE, body=configmap)
            rich.print(
                Panel(
                    f"Updated ConfigMap '{map_name}' in namespace '{K8S_NAMESPACE}' with {len(config_data)} profiles.",
                    style="green",
                    expand=False,
                )
            )
        except ApiException as e:
            if e.status != 404:
                raise

            # ConfigMap doesn't exist, create it
            v1.create_namespaced_config_map(namespace=K8S_NAMESPACE, body=configmap)
            rich.print(
                Panel(
                    f"Created ConfigMap '{map_name}' in namespace '{K8S_NAMESPACE}' with {len(config_data)} profiles.",
                    style="green",
                    expand=False,
                )
            )

    except ApiException as e:
        rich.print(Panel(f"Kubernetes API error: {e.status} {e.reason}\n{e.body}", style="red", expand=False))
    except Exception as e:
        rich.print(Panel(f"Failed to deploy profiles: {e}", style="red", expand=False))


def main() -> None:
    """Main entry point for the Cornserve CLI."""
    app.cli(description="Cornserve CLI")
