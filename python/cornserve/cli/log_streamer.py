"""LogStreamer utilities for the Cornserve CLI."""

from __future__ import annotations

import threading
import time
from typing import cast

import rich
from kubernetes import client, config
from kubernetes.client import V1Pod
from kubernetes.client.rest import ApiException
from kubernetes.config.config_exception import ConfigException
from rich.console import Console
from rich.text import Text
from urllib3.response import HTTPResponse

from cornserve.constants import K8S_NAMESPACE

# A list of visually distinct colors from rich.
LOG_COLORS = [
    "yellow",
    "blue",
    "cyan",
    "green_yellow",
    "dark_orange",
    "purple",
    "spring_green",
]


class LogStreamer:
    """Streams logs from Kubernetes pods related to unit tasks."""

    def __init__(self, unit_task_names: list[str], console: Console | None = None) -> None:
        """Initialize the LogStreamer.

        Args:
            unit_task_names: A list of unit task names to monitor.
            console: The console object to output the logs.
        """
        self.unit_task_names = unit_task_names
        self.console = console or rich.get_console()
        self.k8s_available = self._check_k8s_access()
        if not self.k8s_available:
            return

        self.monitored_pods: set[str] = set()
        self.pod_colors: dict[str, str] = {}
        self.color_index = 0
        self.stop_event = threading.Event()
        self.threads: list[threading.Thread] = []
        self.streams: list[HTTPResponse] = []
        self.lock = threading.Lock()

    def _check_k8s_access(self) -> bool:
        self.console.print("[bold yellow]LogStreamer: Checking Kubernetes access...[/bold yellow]")

        # List of config loading attempts
        config_loaders = [
            ("default kube config (for standard k8s and Minikube)", lambda: config.load_kube_config()),
            ("K3s kube config", lambda: config.load_kube_config(config_file="/etc/rancher/k3s/k3s.yaml")),
        ]

        for description, loader in config_loaders:
            try:
                loader()
                # If loaded, try to access the API
                self.console.print(f"LogStreamer: {description.capitalize()} loaded successfully.")
                client.CoreV1Api().get_api_resources()
                self.console.print("LogStreamer: Kubernetes access confirmed. Going to stream executor logs ...")
                return True
            except (ConfigException, FileNotFoundError):
                self.console.print(f"LogStreamer: Could not load {description}. Trying next option...")
                continue
            except ApiException as e:
                self.console.print(f"LogStreamer: API error with {description}: {e}. Check aborted.")
                return False
            except Exception as e:
                # Catch all other unexpected errors during loading or API call.
                self.console.print(f"LogStreamer: Unexpected error with {description}: {e}. Trying next option...")
                continue

        self.console.print("LogStreamer: Kubernetes access failed. No executor log will be streamed.")
        return False

    def _assign_color(self, pod_name: str) -> None:
        with self.lock:
            if pod_name in self.pod_colors:
                return

            color = LOG_COLORS[self.color_index % len(LOG_COLORS)]
            self.pod_colors[pod_name] = color
            self.color_index += 1

    def _pod_discovery_worker(self) -> None:
        api = client.CoreV1Api()
        while not self.stop_event.is_set():
            try:
                pods = api.list_namespaced_pod(K8S_NAMESPACE, timeout_seconds=5)
                for pod in pods.items:
                    pod_name = pod.metadata.name
                    if pod_name in self.monitored_pods:
                        continue

                    for task_name in self.unit_task_names:
                        # Pod name convention: te-<unit_task_name>-...
                        if pod_name.startswith(f"te-{task_name}"):
                            with self.lock:
                                if pod_name in self.monitored_pods:
                                    continue
                                self.monitored_pods.add(pod_name)

                            self._assign_color(pod_name)

                            log_thread = threading.Thread(target=self._log_streaming_worker, args=(pod_name,))
                            self.threads.append(log_thread)
                            log_thread.start()
                            break  # Move to next pod
            except ApiException as e:
                error_message = Text(f"Error discovering pods: {e.reason}", style="bold red")
                self.console.print(error_message)
                time.sleep(5)  # Wait before retrying on API error
            except Exception:
                # Catch other potential exceptions from k8s client
                time.sleep(5)

            time.sleep(2)  # Poll every 2 seconds for new pods

    def _log_streaming_worker(self, pod_name: str) -> None:
        try:
            # Wait until pod is running
            api = client.CoreV1Api()
            while not self.stop_event.is_set():
                pod: V1Pod = cast(V1Pod, api.read_namespaced_pod(pod_name, K8S_NAMESPACE))
                if pod.status:
                    pod_status_str = pod.status.phase or "Empty"
                    if pod_status_str == "Running":
                        break
                    if pod_status_str in ["Succeeded", "Failed", "Unknown", "Empty"]:
                        self.console.print(
                            Text(f"Pod {pod_name} is in state {pod_status_str}, not streaming logs.", style="yellow")
                        )
                        return
                    time.sleep(1)

            resp: HTTPResponse = api.read_namespaced_pod_log(
                name=pod_name,
                namespace=K8S_NAMESPACE,
                follow=True,
                tail_lines=5,
                _preload_content=False,
            )

            with self.lock:
                self.streams.append(resp)

            for raw_line in resp:
                if self.stop_event.is_set():
                    break

                try:
                    decoded_line = raw_line.decode("utf-8", "replace").rstrip()
                except AttributeError:
                    decoded_line = str(raw_line).rstrip()

                with self.lock:
                    color = self.pod_colors.get(pod_name, "white")
                    log_text = f"{pod_name: <40} | {decoded_line}"
                    log_message = Text(log_text, style=color)
                self.console.print(log_message)

            # Close the response once streaming ends or is interrupted.
            resp.close()

        except Exception as e:
            # Gracefully handle expected errors during stopping the streamer.
            error_msg = str(e).lower()
            if self.stop_event.is_set() or "response ended prematurely" in error_msg:
                return

            self.console.print(Text(f"Error streaming logs for {pod_name}: {e}", style="red"))

    def start(self) -> None:
        """Start the executor discovery and log streaming."""
        if not self.k8s_available:
            return

        discovery_thread = threading.Thread(target=self._pod_discovery_worker)
        self.threads.append(discovery_thread)
        discovery_thread.start()

    def stop(self) -> None:
        """Stop the LogStreamer."""
        if not self.k8s_available:
            return

        self.stop_event.set()

        # Responses of read_namespaced_pod_log with follow=True hangs forever for resp.close(),
        # So we use the proper shutdown() to forcefully terminate resp connections.
        with self.lock:
            for resp in self.streams:
                try:
                    resp.shutdown()
                except Exception:
                    self.console.print(Text(f"Error closing stream for {resp}", style="red"))

        for thread in self.threads:
            thread.join(timeout=2)
