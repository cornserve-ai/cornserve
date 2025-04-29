"""The Cornserve client for communicating with the Cornserve gateway.

Developers can use this client to deploy and remove tasks from the Cornserve gateway.
"""

import os

import websocket
from pydantic import BaseModel

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
    task_list: UnitTaskList

    def get_tasks(self) -> list[UnitTask]:
        """Get the list of tasks from the request."""
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
        if url.startswith("http://"):
            url = url.replace("http://", "ws://")
        if not url.endswith("/session"):
            url += "/session"
        self.url = url
        self.socket = websocket.create_connection(self.url)
        print(f"Connected to Cornserve gateway at {self.url.replace('ws://', '')}")

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
        self.socket.send(request.model_dump_json())
        data = self.socket.recv()
        response = TaskResponse.model_validate_json(data)
        if response.status != 200:
            raise Exception(f"Failed to deploy tasks: {response.content}")

    def deploy(self, task: Task):
        """Deploy a task to the Cornserve gateway.

        Args:
            task: The task to deploy.
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to the Cornserve gateway.")
        tasks = discover_unit_tasks([task])
        self.deploy_unit_tasks(tasks)

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
        self.socket.send(request.model_dump_json())
        data = self.socket.recv()
        response = TaskResponse.model_validate_json(data)
        if response.status != 200:
            raise Exception(f"Failed to deploy tasks: {response.content}")

    def remove(self, task: Task):
        """Remove a task from the Cornserve gateway.

        Args:
            task: The task to remove.
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to the Cornserve gateway.")
        tasks = discover_unit_tasks([task])
        self.remove_unit_tasks(tasks)

    def close(self):
        """Close the connection to the Cornserve gateway."""
        if self.is_connected():
            self.socket.close()
            print("Closed connection to Cornserve gateway.")
