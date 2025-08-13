"""LogStreamer utilities for the Cornserve CLI."""

from __future__ import annotations

import threading
import time
from typing import cast

import rich
from kubernetes import client
from kubernetes.client import V1Pod
from kubernetes.client.rest import ApiException
from rich.console import Console
from rich.text import Text
from urllib3.exceptions import ProtocolError
from urllib3.response import HTTPResponse

from cornserve.cli.utils.k8s import load_k8s_config
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

    def __init__(
        self, unit_task_names: list[str], console: Console | None = None, kube_config_path: str | None = None
    ) -> None:
        """Initialize the LogStreamer.

        Args:
            unit_task_names: A list of unit task names to monitor.
            console: The console object to output the logs.
            kube_config_path: Optional path to the Kubernetes config file.
        """
        self.unit_task_names = unit_task_names
        self.console = console or rich.get_console()
        self.kube_config_path = kube_config_path
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

        try:
            load_k8s_config(self.kube_config_path, ["/etc/rancher/k3s/k3s.yaml"])
            # Test API access
            client.CoreV1Api().get_api_resources()
            self.console.print("LogStreamer: Kubernetes access confirmed. Executor logs will be streamed.")
            return True
        except Exception as e:
            self.console.print(
                "LogStreamer: Failed to configure Kubernetes access. Executor logs will not be streamed.",
            )
            self.console.print(f"Error: '{e}'")

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

            # This worker don't need to close resp, the stop() shuts it down.
            resp: HTTPResponse = api.read_namespaced_pod_log(
                name=pod_name,
                namespace=K8S_NAMESPACE,
                follow=True,
                _preload_content=False,
            )

            with self.lock:
                self.streams.append(resp)

            for raw_line in resp:
                if self.stop_event.is_set():
                    break

                decoded_line = raw_line.decode("utf-8", "replace").rstrip()

                with self.lock:
                    color = self.pod_colors.get(pod_name, "white")
                    log_text = f"{pod_name: <40} | {decoded_line}"
                    log_message = Text(log_text, style=color)
                self.console.print(log_message)

        except Exception as e:
            if isinstance(e, ProtocolError) and self.stop_event.is_set():
                # We only expect this ProtocolError when the response was shut down.
                return
            self.console.print(Text(f"Unexpected error streaming logs for {pod_name}: {e}", style="red"))

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
