from __future__ import annotations

import inspect
import unittest.mock

import pytest
from pydantic import ValidationError

from cornserve.services.gateway.app.manager import AppManager

from . import example_vlm


@pytest.fixture(scope="module")
def app_manager():
    """Fixture to create a new AppManager instance."""
    manager = AppManager("resource-manager:50051")
    manager.resource_manager = unittest.mock.MagicMock()
    manager.resource_manager.ReconcileNewApp = unittest.mock.AsyncMock()
    manager.resource_manager.ReconcileRemovedApp = unittest.mock.AsyncMock()
    return manager


@pytest.mark.asyncio
async def test_example_vlm_app(app_manager: AppManager) -> None:
    """Test example VLM app registration, invocation, and unregistration."""

    async def llm_task_invoke(*args, **kwargs):
        return "Hi Mom!"

    unittest.mock.patch(
        "cornserve.services.gateway.app.task_impl.llm_task_invoke",
        new=llm_task_invoke,
    ).start()

    source_code = inspect.getsource(example_vlm)
    app_id = await app_manager.register_app(source_code)

    with pytest.raises(KeyError):
        await app_manager.invoke_app("non-existing-app", {})

    with pytest.raises(KeyError):
        await app_manager.unregister_app("non-existing-app")

    with pytest.raises(ValidationError):
        await app_manager.invoke_app(app_id, {})

    with pytest.raises(ValidationError):
        # Wrong type
        await app_manager.invoke_app(app_id, {"prompt": 123})

    with pytest.raises(ValidationError):
        # Missing prompt
        await app_manager.invoke_app(app_id, {"image_url": "url"})

    response = await app_manager.invoke_app(app_id, {"prompt": "test", "image_url": "url"})

    assert response == app_manager.apps[app_id].classes.response_cls(text="Hi Mom!")  # type: ignore

    await app_manager.unregister_app(app_id)
