from cornserve.frontend.app import AppConfig, AppRequest, AppResponse
from cornserve.frontend.tasks import LLMTask


class Request(AppRequest):
    """Request class for the VLM app."""
    
    image_url: str
    prompt: str


class Response(AppResponse):
    """Response class for the VLM app."""
    
    text: str


vlm = LLMTask(
    modalities={"text", "image"},
    model_id="Qwen/Qwen2-VL-7B-Instruct",
)


class Config(AppConfig):
    """Config class for the VLM app."""
    
    tasks = {"vlm": vlm}


async def serve(request: Request) -> Response:
    """Serve a single VLM request."""
    response = await vlm.invoke(
        images=[request.image_url],
        prompt=request.prompt,
    )
    return Response(text=response)
