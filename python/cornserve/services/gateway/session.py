"""A session manager for the Cornserve gateway."""

import asyncio
import uuid
from dataclasses import dataclass, field

from cornserve.frontend import TaskRequest, TaskRequestVerb, TaskResponse
from cornserve.logging import get_logger
from cornserve.services.gateway.task_manager import TaskManager
from cornserve.task.base import UnitTask

logger = get_logger(__name__)


@dataclass
class Session:
    """The session state.

    Attributes:
        tasks: A dictionary of tasks that are currently in use by this session.
    """

    tasks: dict[str, UnitTask] = field(default_factory=dict)


class SessionManager:
    """Manages debug sessions for the Cornserve gateway."""

    def __init__(self, task_manager: TaskManager) -> None:
        """Initialize the session manager."""
        self.task_manager = task_manager
        self.lock = asyncio.Lock()
        self.sessions: dict[str, Session] = {}

    async def create_session(self) -> str:
        """Create a new session."""
        session_id = str(uuid.uuid4())
        async with self.lock:
            while session_id in self.sessions:
                session_id = str(uuid.uuid4())
            self.sessions[session_id] = Session()
        logger.info("Created session with ID: %s", session_id)
        return session_id

    async def handle_request(self, session_id: str, request: dict) -> TaskResponse:
        """Handle a request for a session.

        Args:
            session_id: The ID of the session.
            request: The request data.
        """
        async with self.lock:
            if session_id not in self.sessions:
                logger.warning("Session ID %s not found", session_id)
                return TaskResponse(status=404, content="Session invalid")
            try:
                task_request = TaskRequest.model_validate(request)
            except Exception:
                logger.exception("Invalid request")
                return TaskResponse(status=400, content="Invalid request")
            if task_request.verb == TaskRequestVerb.DECLARE_USED:
                logger.info("Declaring tasks as used: %s", task_request.task_list)
                await self.task_manager.declare_used(task_request.get_tasks())
                self.sessions[session_id].tasks.update({task.id: task for task in task_request.get_tasks()})
                return TaskResponse(status=200, content="Tasks declared used")
            elif task_request.verb == TaskRequestVerb.DECLARE_NOT_USED:
                logger.info("Declaring tasks as not used: %s", task_request.task_list)
                await self.task_manager.declare_not_used(task_request.get_tasks())
                for task in task_request.get_tasks():
                    if task.id in self.sessions[session_id].tasks:
                        del self.sessions[session_id].tasks[task.id]
                return TaskResponse(status=200, content="Tasks declared not used")
            elif task_request.verb == TaskRequestVerb.HEARTBEAT:
                return TaskResponse(status=200, content="Session is alive")
            else:
                logger.warning("Unknown method %s", task_request.verb)
                return TaskResponse(status=400, content="Unknown method")

    async def destroy_session(self, session_id: str) -> bool:
        """Destroy a session. Clean up all tasks in use by this session.

        Args:
            session_id: The ID of the session to destroy.
        """
        async with self.lock:
            if session_id in self.sessions:
                logger.info("Destroying session with ID: %s", session_id)
                tasks = list(self.sessions[session_id].tasks.values())
                await self.task_manager.declare_not_used(tasks)
                del self.sessions[session_id]
                return True
        return False
