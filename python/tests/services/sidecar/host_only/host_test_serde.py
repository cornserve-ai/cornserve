import signal
import subprocess
import sys
import threading
import time
from typing import List, Optional

NUM_GPUS = 4
CONTAINER_IMAGE = "cornserve-test:latest"
SERVER_CONTAINER_NAME = "serde_server"
CLIENT_CONTAINER_PREFIX = "serde_client"

# Default Docker arguments
DEFAULT_DOCKER_ARGS = [
    "--network=host",
    # "--cap-add=IPC_LOCK",
    # "--cap-add=SYS_NICE",
    "--rm",
    "--ipc=host",
    "--pid=host",
]

DEFAULT_MOUNTS = [
    "/dev/shm:/dev/shm",
]


def stream_output(process: subprocess.Popen, prefix: str):
    """Stream output from a process in real-time."""
    try:
        for line in iter(process.stdout.readline, ""):  # type: ignore
            if line:
                print(f"[{prefix}] {line.rstrip()}")
        for line in iter(process.stderr.readline, ""):  # type: ignore
            if line:
                print(f"[{prefix}] ERROR: {line.rstrip()}")
    except Exception as e:
        print(f"[{prefix}] Stream error: {e}")


class DockerTestManager:
    """Manages Docker containers for the serialization test."""

    def __init__(self, docker_args: Optional[List[str]] = None, additional_mounts: Optional[List[str]] = None):
        self.container_processes: List[subprocess.Popen] = []
        self.docker_args = docker_args or DEFAULT_DOCKER_ARGS.copy()
        self.additional_mounts = additional_mounts or DEFAULT_MOUNTS.copy()
        self.output_threads: List[threading.Thread] = []

    def build_docker_command(self, base_cmd: List[str], gpu_spec: str = "all") -> List[str]:
        """Build a complete docker run command with all arguments."""
        cmd = ["docker", "run"] + self.docker_args.copy()

        # Add GPU specification
        cmd.extend(["--gpus", gpu_spec])

        # Add additional mounts
        for mount in self.additional_mounts:
            cmd.extend(["-v", mount])

        # Add the rest of the command
        cmd.extend(base_cmd)

        return cmd

    def start_server_container(self) -> subprocess.Popen:
        """Start the server container."""
        print("Starting server container...")

        base_cmd = [
            "--name",
            SERVER_CONTAINER_NAME,
            CONTAINER_IMAGE,
            "python",
            "-u",
            "server.py",
            "--clients",
            str(NUM_GPUS),  # -u for unbuffered output
        ]

        cmd = self.build_docker_command(base_cmd, gpu_spec="all")

        print(f"Running: {' '.join(cmd)}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        self.container_processes.append(process)

        # Start output streaming thread
        thread = threading.Thread(target=stream_output, args=(process, "SERVER"), daemon=True)
        thread.start()
        self.output_threads.append(thread)

        print(f"Server container started with PID: {process.pid}")
        return process

    def start_client_container(self, gpu_id: int) -> subprocess.Popen:
        """Start a client container for a specific GPU."""
        print(f"Starting client container for GPU {gpu_id}...")

        container_name = f"{CLIENT_CONTAINER_PREFIX}_{gpu_id}"

        base_cmd = [
            "--name",
            container_name,
            CONTAINER_IMAGE,
            "python",
            "-u",
            "client.py",
            str(gpu_id),
            "--device-id",
            str(0),  # -u for unbuffered output
        ]

        cmd = self.build_docker_command(base_cmd, gpu_spec=f"device={gpu_id}")

        # base_cmd = [
        #     "--name", container_name,
        #     CONTAINER_IMAGE,
        #     "python", "-u", "client.py", str(gpu_id), # -u for unbuffered output
        # ]
        # cmd = self.build_docker_command(base_cmd, gpu_spec="all")

        print(f"Running: {' '.join(cmd)}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        self.container_processes.append(process)

        # Start output streaming thread
        thread = threading.Thread(target=stream_output, args=(process, f"CLIENT-{gpu_id}"), daemon=True)
        thread.start()
        self.output_threads.append(thread)

        print(f"Client container for GPU {gpu_id} started with PID: {process.pid}")
        return process

    def wait_for_containers(self, timeout: int = 120):
        """Wait for all containers to complete."""
        print("Waiting for containers to complete...")

        start_time = time.time()
        completed = [False] * len(self.container_processes)

        while not all(completed) and (time.time() - start_time) < timeout:
            for i, process in enumerate(self.container_processes):
                if not completed[i] and process.poll() is not None:
                    container_type = "server" if i == 0 else f"client_gpu_{i - 1}"

                    if process.returncode != 0:
                        print(f"Container {container_type} failed with exit code {process.returncode}")
                        # Give some time for output threads to finish
                        time.sleep(1)
                        raise RuntimeError(f"Container {container_type} failed with exit code {process.returncode}")
                    else:
                        print(f"Container {container_type} completed successfully")
                        completed[i] = True

            time.sleep(0.1)  # Small delay to avoid busy waiting

        if not all(completed):
            # Check which containers are still running
            running_containers = []
            for i, (process, is_completed) in enumerate(zip(self.container_processes, completed)):
                if not is_completed:
                    container_type = "server" if i == 0 else f"client_gpu_{i - 1}"
                    running_containers.append(container_type)

            print(f"Timeout: These containers are still running: {running_containers}")

            # Try to get logs from running containers
            for i, process in enumerate(self.container_processes):
                if not completed[i]:
                    container_type = "server" if i == 0 else f"client_gpu_{i - 1}"
                    print(f"Terminating {container_type}...")
                    process.terminate()

            time.sleep(2)  # Wait for graceful shutdown

            # Force kill if still running
            for i, process in enumerate(self.container_processes):
                if process.poll() is None:
                    container_type = "server" if i == 0 else f"client_gpu_{i - 1}"
                    print(f"Force killing {container_type}...")
                    process.kill()

            raise RuntimeError(f"Containers timed out after {timeout} seconds")

    def cleanup(self):
        """Clean up containers."""
        print("Cleaning up...")

        # Terminate any running processes
        for process in self.container_processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

        # Clean up any remaining containers by name
        containers_to_cleanup = [SERVER_CONTAINER_NAME] + [f"{CLIENT_CONTAINER_PREFIX}_{i}" for i in range(NUM_GPUS)]

        for container_name in containers_to_cleanup:
            try:
                result = subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, check=False)
                if result.returncode == 0:
                    print(f"Cleaned up container: {container_name}")
            except Exception:
                pass


def test_multi_container_serde(docker_args: Optional[List[str]] = None, additional_mounts: Optional[List[str]] = None):
    """Test multiple Docker containers, each using a dedicated GPU."""
    manager = DockerTestManager(docker_args=docker_args, additional_mounts=additional_mounts)

    # Setup signal handler for cleanup
    def signal_handler(signum, frame):
        print("\nReceived interrupt signal, cleaning up...")
        manager.cleanup()
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        print(f"Testing with {NUM_GPUS} GPUs")

        # Start server container
        server_process = manager.start_server_container()

        # Wait for server to start up
        print("Waiting for server to start...")
        time.sleep(5)

        # Check if server is still running
        if server_process.poll() is not None:
            raise RuntimeError(f"Server container exited early with code {server_process.returncode}")

        # Start client containers
        for gpu_id in range(NUM_GPUS):
            client_process = manager.start_client_container(gpu_id)
            # Small delay between client starts
            time.sleep(0.5)

        # Wait for all containers to complete
        manager.wait_for_containers(timeout=120)

        print("All containers completed successfully!")

    except Exception as e:
        print(f"Test failed: {e}")
        raise

    finally:
        manager.cleanup()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Docker container serialization test")
    parser.add_argument("--docker-args", nargs="*", help="Additional Docker arguments")
    parser.add_argument("--mounts", nargs="*", help="Additional volume mounts (format: host:container:options)")
    parser.add_argument("--gpus", type=int, default=4, help="Number of GPUs to test with")

    args = parser.parse_args()

    # Update NUM_GPUS if specified
    if args.gpus:
        NUM_GPUS = args.gpus

    custom_docker_args = DEFAULT_DOCKER_ARGS.copy()
    if args.docker_args:
        custom_docker_args.extend(args.docker_args)

    custom_mounts = DEFAULT_MOUNTS.copy()
    if args.mounts:
        custom_mounts.extend(args.mounts)
    print(f"Mounts: {custom_mounts}")

    test_multi_container_serde(docker_args=custom_docker_args, additional_mounts=custom_mounts)
