"""Qwen 2.5 Omni model wrapper using HuggingFace transformers."""

from __future__ import annotations

import base64

import torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

from cornserve.logging import get_logger
from cornserve.task_executors.huggingface.api import HuggingFaceRequest, HuggingFaceResponse, Status
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

        if not request.return_audio:
            raise ValueError("Only audio generation is supported for the Qwen 2.5 Omni model")

        # Convert messages to the format expected by the processor
        conversations = self._convert_messages(request.messages)

        # Process inputs
        text = self.processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversations, use_audio_in_video=False)  # type: ignore
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

        # Generate response
        text_ids, audio = self.model.generate(**inputs, use_audio_in_video=False, return_audio=True)

        text = self.processor.batch_decode(text_ids)[0]

        audio_data = audio.reshape(-1).detach().cpu().numpy()  # np.float32
        audio_b64 = base64.b64encode(audio_data.tobytes()).decode("utf-8")

        logger.info("Generated text: %s", text[text.rfind("<|im_start|>") :])
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
