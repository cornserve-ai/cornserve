"""Qwen 2.5 Omni model wrapper using HuggingFace transformers."""

from __future__ import annotations

import base64
import uuid
from collections.abc import AsyncGenerator

import numpy as np
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

from cornserve.logging import get_logger
from cornserve.task.builtins.llm import OpenAIChatCompletionChunk
from cornserve.task_executors.huggingface.api import HuggingFaceRequest, HuggingFaceResponse, Status

logger = get_logger(__name__)


class QwenOmniModel:
    """Wrapper for Qwen 2.5 Omni model using HuggingFace transformers.

    Uses Qwen2_5OmniForConditionalGeneration and Qwen2_5OmniProcessor for
    multimodal generation with text and audio output.
    """

    def __init__(self, model_id: str):
        """Initialize the Qwen 2.5 Omni model.

        Args:
            model_id: Model ID to load (e.g., "Qwen/Qwen2.5-Omni-7B").
        """
        self.model_id = model_id
        logger.info("Loading Qwen 2.5 Omni model: %s", model_id)

        # Load the model and processor
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
        )

        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_id)

        logger.info("Successfully loaded Qwen 2.5 Omni model")

    async def generate_stream(self, request: HuggingFaceRequest) -> AsyncGenerator[HuggingFaceResponse, None]:
        """Generate streaming response for multimodal input.

        Args:
            request: The request containing messages and options.

        Yields:
            HuggingFaceResponse chunks with either text or audio content.
        """
        try:
            # Convert messages to the format expected by the processor
            conversations = self._convert_messages(request.messages)

            # Process inputs
            inputs = self.processor.apply_chat_template(
                conversations, add_generation_prompt=True, tokenize=True, return_dict=True
            )

            # Generate response
            if request.return_audio:
                # Generate both text and audio
                async for chunk in self._generate_with_audio(inputs, request):
                    yield chunk
            else:
                # Generate text only
                async for chunk in self._generate_text_only(inputs, request):
                    yield chunk

        except Exception as e:
            logger.exception("Error in Qwen 2.5 Omni generation: %s", e)
            yield HuggingFaceResponse(status=Status.ERROR, error_message=f"Error in generation: {e}")

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
                # Multimodal content
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

    async def _generate_with_audio(
        self, inputs: dict, request: HuggingFaceRequest
    ) -> AsyncGenerator[HuggingFaceResponse, None]:
        """Generate response with audio output.

        Args:
            inputs: Processed inputs from processor.
            request: Original request.

        Yields:
            Response chunks with text and audio.
        """
        try:
            # Generate with audio output (using default speaker "Chelsie")
            text_ids, audio = self.model.generate(**inputs, speaker="Chelsie")

            # Convert text tokens to text
            generated_text = self.processor.decode(text_ids[0], skip_special_tokens=True)

            # Create text chunks (simulate streaming)
            chunk_id = uuid.uuid4().hex

            # Yield text chunk
            text_chunk = OpenAIChatCompletionChunk(
                id=chunk_id,
                object="chat.completion.chunk",
                created=0,
                model=request.model_id,
                choices=[{"index": 0, "delta": {"content": generated_text}, "finish_reason": None}],
            )

            yield HuggingFaceResponse(status=Status.SUCCESS, text_chunk=text_chunk.model_dump())

            # Convert and yield audio chunks
            if audio is not None:
                # Convert audio to base64
                audio_np = audio.cpu().numpy() if hasattr(audio, "cpu") else audio
                if isinstance(audio_np, (list, tuple)):
                    audio_np = np.array(audio_np)

                # Ensure audio is float32
                if audio_np.dtype != np.float32:
                    audio_np = audio_np.astype(np.float32)

                # Split audio into chunks for streaming
                chunk_size = 4096  # samples per chunk
                for i in range(0, len(audio_np), chunk_size):
                    audio_chunk = audio_np[i : i + chunk_size]

                    # Convert to base64
                    audio_b64 = base64.b64encode(audio_chunk.tobytes()).decode("utf-8")

                    yield HuggingFaceResponse(status=Status.SUCCESS, audio_chunk=audio_b64)

            # Final chunk
            final_chunk = OpenAIChatCompletionChunk(
                id=chunk_id,
                object="chat.completion.chunk",
                created=0,
                model=request.model_id,
                choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}],
            )

            yield HuggingFaceResponse(status=Status.SUCCESS, text_chunk=final_chunk.model_dump())

        except Exception as e:
            logger.exception("Error in audio generation: %s", e)
            yield HuggingFaceResponse(status=Status.ERROR, error_message=f"Error in audio generation: {e}")

    async def _generate_text_only(
        self, inputs: dict, request: HuggingFaceRequest
    ) -> AsyncGenerator[HuggingFaceResponse, None]:
        """Generate text-only response.

        Args:
            inputs: Processed inputs from processor.
            request: Original request.

        Yields:
            Response chunks with text only.
        """
        try:
            # Generate text without audio
            # Note: For text-only generation, we might need to use a different method
            # This is a simplified implementation
            text_ids, _ = self.model.generate(**inputs)

            # Convert text tokens to text
            generated_text = self.processor.decode(text_ids[0], skip_special_tokens=True)

            # Create streaming chunks
            chunk_id = uuid.uuid4().hex

            # Simulate streaming by splitting text
            words = generated_text.split()
            current_text = ""

            for word in words:
                current_text += word + " "

                text_chunk = OpenAIChatCompletionChunk(
                    id=chunk_id,
                    object="chat.completion.chunk",
                    created=0,
                    model=request.model_id,
                    choices=[{"index": 0, "delta": {"content": word + " "}, "finish_reason": None}],
                )

                yield HuggingFaceResponse(status=Status.SUCCESS, text_chunk=text_chunk.model_dump())

            # Final chunk
            final_chunk = OpenAIChatCompletionChunk(
                id=chunk_id,
                object="chat.completion.chunk",
                created=0,
                model=request.model_id,
                choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}],
            )

            yield HuggingFaceResponse(status=Status.SUCCESS, text_chunk=final_chunk.model_dump())

        except Exception as e:
            logger.exception("Error in text generation: %s", e)
            yield HuggingFaceResponse(status=Status.ERROR, error_message=f"Error in text generation: {e}")
