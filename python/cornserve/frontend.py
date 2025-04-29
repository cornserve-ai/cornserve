"""The Cornserve client for communicating with the Cornserve gateway.

Developers can use this client to deploy and remove tasks from the Cornserve gateway.
"""

import os
from urllib.parse import urlparse
import threading

from pydantic import BaseModel
import websocket

from cornserve.constants import K8S_GATEWAY_SERVICE_HTTP_URL
from cornserve.services.gateway.app.manager import discover_unit_tasks
from cornserve.task.base import Task, UnitTask, UnitTaskList


class TaskRequest(BaseModel):
    """Request for (un)registering tasks.

    Attributes:
        method: The task_manager method to call.
        task_list: A UnitTaskList of tasks to be registered or unregistered.
    """

    method: str
    task_list: UnitTaskList | None

    def get_tasks(self) -> list[UnitTask]:
        """Get the list of tasks from the request."""
        if self.task_list is None:
            return []
        return self.task_list.tasks


class TaskResponse(BaseModel):
    """Response for a TaskRequest sent to the Cornserve gateway.

    Attributes:
        status: The HTTP status code of the response.
        content: The content of the response.
    """

    status: int
    content: str


class CornserveClient:
    """The Cornserve client for communicating with the Cornserve gateway."""

    def __init__(self, url=None):
        """Initialize the Cornserve client."""
        if url is None:
            url = os.environ.get(
                "CORNSERVE_GATEWAY_URL",
                K8S_GATEWAY_SERVICE_HTTP_URL,
            )

        url_parsed = urlparse(url)
        if url_parsed.scheme not in ["http", "https", "ws", "wss", ""]:
            # avoid port number being interpreted as path
            url_parsed = urlparse("http://" + url)

        if url_parsed.scheme == "" :
            # if no scheme is provided, assume http
            dispatcher_url = url_parsed._replace(scheme="http")
            session_url = url_parsed._replace(scheme="ws")
        elif url_parsed.scheme == "ws":
            dispatcher_url = url_parsed._replace(scheme="http")
            session_url = url_parsed
        elif url_parsed.scheme == "wss":
            dispatcher_url = url_parsed._replace(scheme="https")
            session_url = url_parsed
        elif url_parsed.scheme == "http":
            dispatcher_url = url_parsed
            session_url = url_parsed._replace(scheme="ws")
        elif url_parsed.scheme == "https":
            dispatcher_url = url_parsed
            session_url = url_parsed._replace(scheme="wss")
        else:
            raise ValueError(f"Invalid URL scheme: {url_parsed.scheme}")
        
        # reset env
        os.environ["CORNSERVE_GATEWAY_URL"] = dispatcher_url.geturl()

        self.url = session_url.geturl() + "/session"
        self.socket = websocket.create_connection(self.url)
        print(f"Connected to Cornserve gateway at {session_url.netloc}")

        self.message_lock = threading.Lock()
        self.keep_alive_thread = threading.Thread(
            target=self._keep_alive,
            args=(self.socket,),
            daemon=True,
        )
        self.keep_alive_thread.start()

    def _keep_alive(self, socket: websocket.WebSocket):
        """Keep the WebSocket connection alive by sending a ping message.

        Args:
            socket: The WebSocket connection to keep alive.
        """
        import time
        while True:
            try:
                request = TaskRequest(method="heartbeat", task_list=None)
                with self.message_lock:
                    socket.send(request.model_dump_json())
                    data = socket.recv()
                response = TaskResponse.model_validate_json(data)
                if response.status != 200:
                    print(f"Failed to send heartbeat: {response.content}")
                time.sleep(5)
            except websocket.WebSocketConnectionClosedException:
                break
            except Exception as e:
                print(f"Error in keep_alive: {e}")
                break

    def is_connected(self):
        """Check if the client is connected to the Cornserve gateway."""
        return self.socket.connected

    def deploy_unit_tasks(self, tasks: list[UnitTask]):
        """Deploy unit tasks to the Cornserve gateway.

        Args:
            tasks: A list of unit tasks to deploy.
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to the Cornserve gateway.")
        task_list = UnitTaskList(tasks=tasks)
        request = TaskRequest(
            method="declare_used",
            task_list=task_list,
        )
        with self.message_lock:
            self.socket.send(request.model_dump_json())
            data = self.socket.recv()
        response = TaskResponse.model_validate_json(data)
        if response.status != 200:
            raise Exception(f"Failed to deploy tasks: {response.content}")
        return response

    def deploy(self, task: Task):
        """Deploy a task to the Cornserve gateway.

        Args:
            task: The task to deploy.
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to the Cornserve gateway.")
        tasks = discover_unit_tasks([task])
        return self.deploy_unit_tasks(tasks)

    def remove_unit_tasks(self, tasks: list[UnitTask]):
        """Remove unit tasks from the Cornserve gateway.

        Args:
            tasks: A list of unit tasks to remove.
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to the Cornserve gateway.")
        task_list = UnitTaskList(tasks=tasks)
        request = TaskRequest(
            method="declare_not_used",
            task_list=task_list,
        )
        with self.message_lock:
            self.socket.send(request.model_dump_json())
            data = self.socket.recv()
        response = TaskResponse.model_validate_json(data)
        if response.status != 200:
            raise Exception(f"Failed to deploy tasks: {response.content}")
        return response

    def remove(self, task: Task):
        """Remove a task from the Cornserve gateway.

        Args:
            task: The task to remove.
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to the Cornserve gateway.")
        tasks = discover_unit_tasks([task])
        return self.remove_unit_tasks(tasks)

    def close(self):
        """Close the connection to the Cornserve gateway."""
        if self.is_connected():
            self.socket.close()
            print("Closed connection to Cornserve gateway.")
        if self.keep_alive_thread.is_alive():
            self.keep_alive_thread.join()
            print("Closed keep-alive thread.")
