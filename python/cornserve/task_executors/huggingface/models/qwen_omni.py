"""Qwen 2.5 Omni model wrapper using HuggingFace transformers."""

from __future__ import annotations
import base64
import os
import tempfile

from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
from qwen_omni_utils import process_mm_info
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

from cornserve.logging import get_logger
from cornserve.task.builtins.llm import OpenAIChatCompletionChunk
from cornserve.task_executors.huggingface.api import (
    HuggingFaceRequest,
    HuggingFaceResponse,
    Status,
)
from cornserve.task_executors.huggingface.models.base import HFModel

logger = get_logger(__name__)


class QwenOmniModel(HFModel):
    """Wrapper for Qwen 2.5 Omni model using HuggingFace transformers.

    Uses Qwen2_5OmniForConditionalGeneration and Qwen2_5OmniProcessor for
    multimodal generation with text and audio output.
    """

    def __init__(self, model_id: str) -> None:
        """Initialize the Qwen 2.5 Omni model.

        Args:
            model_id: Model ID to load (e.g., "Qwen/Qwen2.5-Omni-7B").
        """
        self.model_id = model_id
        logger.info("Loading Qwen 2.5 Omni model: %s", model_id)

        # Load the model and processor
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )

        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_id)

        logger.info("Successfully loaded Qwen 2.5 Omni model")

    def generate(self, request: HuggingFaceRequest) -> HuggingFaceResponse:
        """Generate audio from the Qwem Omni model."""
        assert request.messages is not None, "Messages must be provided in the request"

        # Convert messages to the format expected by the processor
        conversations = self._convert_messages(request.messages)

        # Process inputs
        text = self.processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
        tmpfiles = _materialize_videos_inplace(conversations)
        audios, images, videos = process_mm_info(conversations, use_audio_in_video=False)  # type: ignore
        _cleanup_tmpfiles(tmpfiles)
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,  # type: ignore
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        max_new_tokens = request.max_completion_tokens
        min_new_tokens = request.max_completion_tokens if request.ignore_eos else None

        # Generate response
        if not request.return_audio:
            text_ids = self.model.generate(
                **inputs,
                use_audio_in_video=False,
                return_audio=False,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
            )  # type: ignore
            text = self.processor.batch_decode(text_ids)[0]
            logger.info("Generated text: %s / max tokens %s", text, max_new_tokens)
            # create a dummy text chunk
            text_chunk = OpenAIChatCompletionChunk(
                id="ID",
                choices=[
                    Choice(
                        index=0,
                        finish_reason="stop",
                        delta=ChoiceDelta(
                            role="assistant",
                            content=text[text.rfind("<|im_start|>") :],
                        ),
                    )
                ],
                created=0,
                object="chat.completion.chunk",
                model=self.model_id,
            )
            # logger.info("Generated text chunk: %s", text_chunk)
            return HuggingFaceResponse(
                status=Status.SUCCESS,
                text_chunk=text_chunk.model_dump(),
            )

        text_ids, audio = self.model.generate(
            **inputs,
            use_audio_in_video=False,
            return_audio=request.return_audio,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
        )  # type: ignore
        text = self.processor.batch_decode(text_ids)[0]


        audio_data = audio.reshape(-1).detach().cpu().numpy()  # np.float32
        audio_b64 = base64.b64encode(audio_data.tobytes()).decode("utf-8")

        logger.info("Generated text: %s / max tokens %s", text, max_new_tokens)
        logger.info(
            "Generated audio length is %f seconds and size after base64 encoding is %.2f MiBs",
            audio.numel() / 24000,
            len(audio_b64) / (1024 * 1024),
        )

        return HuggingFaceResponse(status=Status.SUCCESS, audio_chunk=audio_b64)

    def _convert_messages(self, messages: list[dict]) -> list[dict]:
        """Convert OpenAI-style messages to Qwen format.

        Args:
            messages: List of message dictionaries.

        Returns:
            Converted messages for Qwen processor.
        """
        conversations = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if isinstance(content, str):
                # Simple text message
                conversations.append({"role": role, "content": [{"type": "text", "text": content}]})
            elif isinstance(content, list):
                # Could contain multimodal content
                converted_content = []
                for part in content:
                    if isinstance(part, dict):
                        part_type = part.get("type", "text")
                        if part_type == "text":
                            converted_content.append({"type": "text", "text": part.get("text", "")})
                        elif part_type in ["image_url", "audio_url", "video_url"]:
                            # Handle multimodal URLs
                            url_key = part_type
                            url_obj = part.get(url_key, {})
                            url = url_obj.get("url", "") if isinstance(url_obj, dict) else str(url_obj)
                            converted_content.append(
                                {"type": part_type.replace("_url", ""), part_type.replace("_url", ""): url}
                            )

                conversations.append({"role": role, "content": converted_content})

        return conversations


def _cleanup_tmpfiles(paths: list[str]) -> None:
    """Helper to remove temporary files."""
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass

def _materialize_videos_inplace(conversations: list[dict]) -> list[str]:
    """Helper to convert base64 videos to temporary files. """
    created: list[str] = []

    for msg in conversations:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not (isinstance(item, dict) and item.get("type") == "video"):
                continue
            v = item.get("video")
            if not (isinstance(v, str) and v.startswith("data:video/mp4")):
                continue  # leave non-data URLs or paths alone

            # grab the base64 payload after the first comma
            i = v.find(",")
            if i == -1:
                continue
            b64 = v[i + 1 :]

            fd, path = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            with open(path, "wb") as f:
                f.write(base64.b64decode(b64))

            item["video"] = path
            created.append(path)

    return created
