"""Miscellaneous utilities for the Cornserve CLI."""

from __future__ import annotations

import random
import subprocess
import threading
import time
from collections import deque
from typing import Deque

import rich
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from kubernetes import client, config
from kubernetes.client.rest import ApiException
from kubernetes.config.config_exception import ConfigException

# A list of visually distinct colors from rich.
DISTINCT_COLORS = [
    "bright_red",
    "bright_green",
    "bright_yellow",
    "bright_blue",
    "bright_magenta",
    "bright_cyan",
    "red",
    "green",
    "yellow",
    "blue",
    "magenta",
    "cyan",
]


class LogStreamer:
    """Streams logs from Kubernetes pods related to unit tasks."""

    def __init__(self, unit_task_names: list[str], namespace: str = "cornserve", console: Console | None = None):
        self.unit_task_names = unit_task_names
        self.namespace = namespace
        self.console = console or rich.get_console()
        self.k8s_available = self._check_k8s_access()
        if not self.k8s_available:
            return

        self.monitored_pods: set[str] = set()
        self.pod_colors: dict[str, str] = {}
        self.stop_event = threading.Event()
        self.threads: list[threading.Thread] = []
        self.subprocesses: list[subprocess.Popen] = []
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
                self.console.print(f"[bold green]LogStreamer: {description.capitalize()} loaded successfully.[/bold green]")
                client.CoreV1Api().get_api_resources()
                self.console.print(f"[bold green]LogStreamer: Kubernetes access confirmed. Going to stream executor logs ... [/bold green]")
                return True
            except (ConfigException, FileNotFoundError):
                self.console.print(f"[bold yellow]LogStreamer: Could not load {description}. Trying next option...[/bold yellow]")
                continue
            except ApiException as e:
                self.console.print(f"[bold red]LogStreamer: API error with {description}: {e}. Aborting check.[/bold red]")
                return False
            except Exception as e:
                # Catch all other unexpected errors during loading or API call.
                self.console.print(
                    f"[bold red]LogStreamer: Unexpected error with {description}: {e}. Trying next option...[/bold red]"
                )
                continue

        self.console.print("[bold red]LogStreamer: Kubernetes access failed. No log will be streamed.[/bold red]")
        return False

    def _assign_color(self, pod_name: str):
        with self.lock:
            if pod_name in self.pod_colors:
                return

            used_colors = set(self.pod_colors.values())
            available_colors = [c for c in DISTINCT_COLORS if c not in used_colors]

            if available_colors:
                color = random.choice(available_colors)
            else:
                # All distinct colors are used, start reusing them
                color = random.choice(DISTINCT_COLORS)
            self.pod_colors[pod_name] = color

    def _pod_discovery_worker(self):
        api = client.CoreV1Api()
        while not self.stop_event.is_set():
            try:
                pods = api.list_namespaced_pod(self.namespace, timeout_seconds=5)
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

    def _log_streaming_worker(self, pod_name: str):
        try:
            # Wait until pod is running
            api = client.CoreV1Api()
            while not self.stop_event.is_set():
                pod_status = api.read_namespaced_pod_status(pod_name, self.namespace).status.phase
                if pod_status == "Running":
                    break
                if pod_status in ["Succeeded", "Failed", "Unknown"]:
                    self.console.print(
                        Text(f"Pod {pod_name} is in state {pod_status}, not streaming logs.", style="yellow")
                    )
                    return
                time.sleep(1)

            proc = subprocess.Popen(
                ["kubectl", "logs", "-f", "--tail=5", "-n", self.namespace, pod_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Redirect stderr to stdout
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            self.subprocesses.append(proc)

            for line in iter(proc.stdout.readline, ""):
                if self.stop_event.is_set():
                    break
                with self.lock:
                    color = self.pod_colors.get(pod_name, "white")
                    log_text = f"{pod_name: <40} | {line.strip()}"
                    log_message = Text(log_text, style=color)
                self.console.print(log_message)

            proc.stdout.close()
            return_code = proc.wait()
            if return_code != 0 and not self.stop_event.is_set():
                self.console.print(Text(f"Pod {pod_name} exited with code {return_code}.", style="yellow"))

        except Exception as e:
            self.console.print(Text(f"Error streaming logs for {pod_name}: {e}", style="red"))

    def start(self):
        if not self.k8s_available:
            return

        discovery_thread = threading.Thread(target=self._pod_discovery_worker)
        self.threads.append(discovery_thread)
        discovery_thread.start()

    def stop(self):
        if not self.k8s_available:
            return

        self.stop_event.set()
        for proc in self.subprocesses:
            proc.terminate()
        for thread in self.threads:
            thread.join(timeout=2)
        for proc in self.subprocesses:
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.kill() 