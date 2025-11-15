"""Qwen Omni model wrapper using HuggingFace transformers.

Supports both Qwen 2.5 Omni and Qwen 3 Omni models.
"""

from __future__ import annotations

import base64

import torch
from qwen_omni_utils import process_mm_info
from transformers import AutoConfig

from cornserve.logging import get_logger
from cornserve.task_executors.huggingface.api import HuggingFaceRequest, HuggingFaceResponse, Status
from cornserve.task_executors.huggingface.models.base import HFModel

logger = get_logger(__name__)

# Sampling rate for audio output (24kHz for both Qwen 2.5 and 3)
AUDIO_SAMPLE_RATE = 24000


class QwenOmniModel(HFModel):
    """Wrapper for Qwen Omni models using HuggingFace transformers.

    Supports both Qwen 2.5 Omni and Qwen 3 Omni models.
    """

    def __init__(self, model_id: str) -> None:
        """Initialize the Qwen Omni model.

        Args:
            model_id: Model ID to load (e.g., "Qwen/Qwen2.5-Omni-7B" or
                     "Qwen/Qwen3-Omni-30B-A3B-Instruct").
        """
        self.model_id = model_id
        logger.info("Loading Qwen Omni model: %s", model_id)

        # Auto-detect model version from config
        config = AutoConfig.from_pretrained(model_id)
        model_type = config.model_type

        if model_type == "qwen2_5_omni":
            logger.info("Detected Qwen 2.5 Omni model")
            from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor  # noqa: PLC0415

            self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                model_id,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                device_map="cuda",
            )
            self.processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
            self.version = "2.5"

        elif model_type == "qwen3_omni_moe":
            logger.info("Detected Qwen 3 Omni MoE model")
            from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor  # noqa: PLC0415

            self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                model_id,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                device_map="cuda",
            )
            self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_id)
            self.version = "3"

        else:
            raise ValueError(f"Unsupported model type: {model_type}. Expected 'qwen2_5_omni' or 'qwen3_omni_moe'.")

        logger.info("Successfully loaded Qwen %s Omni model", self.version)

    def generate(self, request: HuggingFaceRequest) -> HuggingFaceResponse:
        """Generate audio from the Qwen Omni model."""
        assert request.messages is not None, "Messages must be provided in the request"

        if not request.return_audio:
            raise ValueError("Only audio generation is supported for Qwen Omni models")

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
            return_tensors="pt",  # type: ignore
            padding=True,  # type: ignore
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
            audio.numel() / AUDIO_SAMPLE_RATE,
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
