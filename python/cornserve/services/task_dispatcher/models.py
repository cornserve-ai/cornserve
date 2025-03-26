"""Data models for the Task Dispatcher."""

from __future__ import annotations

import grpc
from pydantic import BaseModel, ConfigDict

from cornserve.frontend.tasks import Task
from cornserve.services.pb import common_pb2, task_dispatcher_pb2, task_manager_pb2_grpc
from cornserve.services.task_manager.models import TaskManagerType


class TaskDispatchRequest(BaseModel):
    """Request for invoking a task.

    Attributes:
        app_id: The unique identifier for the application.
        task_id: The unique identifier for the task.
        request_id: The unique identifier for the request.
        request_data: Serialized input data for task invocation.
    """

    app_id: str
    task_id: str
    request_id: str
    request_data: str


class TaskManagerInfo(BaseModel):
    """Task Manager info and how to reach it.

    Attributes:
        id: The ID of the task manager.
        type: The type of task manager.
        url: The URL of the task manager.
        channel: The gRPC channel to the task manager.
        stub: The gRPC stub to the task manager.
    """

    id: str
    type: TaskManagerType
    url: str
    channel: grpc.aio.Channel
    stub: task_manager_pb2_grpc.TaskManagerStub

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_pb(cls, pb: task_dispatcher_pb2.TaskManagerInfo) -> TaskManagerInfo:
        """Create a TaskManager from a protobuf struct."""
        channel = grpc.aio.insecure_channel(pb.url)
        stub = task_manager_pb2_grpc.TaskManagerStub(channel)
        return cls(
            id=pb.task_manager_id,
            type=TaskManagerType.from_pb(pb.type),
            url=pb.url,
            channel=channel,
            stub=stub,
        )


class TaskInfo(BaseModel):
    """Task info and what task managers it invokes.

    Attributes:
        id: The ID of the task.
        task_managers: The task managers that this task invokes.
    """

    id: str
    task: Task
    # XXX(J1): This assumes that the necessary task manager can be located
    # only with the task manager type. However, for instance, there can be
    # multiple encoders (e.g., one for image and one for audio), which will
    # require a more sophisticated way of passing around and locating the task manager.
    task_managers: list[TaskManagerInfo]

    @classmethod
    def from_pb(cls, pb: task_dispatcher_pb2.TaskInfo) -> TaskInfo:
        """Create a Task from a protobuf struct."""
        return cls(
            id=pb.task_id,
            task=Task.from_json(common_pb2.TaskType.Name(pb.type), pb.task_config),
            task_managers=[TaskManagerInfo.from_pb(info) for info in pb.task_manager_info],
        )
