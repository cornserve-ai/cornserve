"""Tests for the Qwen3-Omni model's vocoder."""

import pytest
import torch
from torch import nn
from transformers import (
    Qwen3OmniMoeForConditionalGeneration,
)
from transformers.models.auto.processing_auto import AutoProcessor

from cornserve.task_executors.geri.executor.loader import load_model
from cornserve.task_executors.geri.models.base import GeriModel, StreamGeriModel
from cornserve.task_executors.geri.models.qwen3_omni_moe import Qwen3OmniMoeCode2Wav

from ..utils import assert_similar

model_id = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
model_shorthand = "qwen3_omni_moe"


def test_model_loading() -> None:
    """Test model is correctly configured for model loader."""
    model = load_model(model_id, torch.device("cuda"))
    assert isinstance(model, GeriModel)
    assert isinstance(model, StreamGeriModel)


class CodesExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.codes = None

    def get_codes(self):
        return self.codes

    def chunked_decode(self, codes, **_):
        self.codes = codes
        return torch.tensor([])

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")


def get_talker_codes(model, processor):
    # Save the real original vocoder
    real_code2wav = model.code2wav

    # Replace it with a fake that will intercept talker codes
    codes_extractor = CodesExtractor()
    model.code2wav = codes_extractor

    # Prepare for generation
    conversation = [{"role": "user", "content": "Say 'Hello world, I am Qwen.'"}]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, return_tensors="pt").to(model.device).to(model.dtype)

    # model.generate will call chunked_decode
    with torch.no_grad():
        model.generate(
            **inputs,
            return_audio=True,
        )

    # Put back the original vocoder
    model.code2wav = real_code2wav

    return codes_extractor.codes


@pytest.fixture(scope="module")
def hf_model():
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
    ).eval()
    return model


@pytest.fixture(scope="module")
def processor():
    return AutoProcessor.from_pretrained(model_id)


@pytest.fixture(scope="module")
def talker_codes(hf_model, processor):
    # Prepare talker codes for testing
    return get_talker_codes(hf_model, processor)


def test_hf_reference(hf_model, talker_codes) -> None:
    """Generate reference outputs from the Hugging Face model."""
    torch.set_grad_enabled(False)

    talker_codes = talker_codes.to(torch.device("cuda"))

    # Generate audio using HF vocoder
    hf_code2wav = hf_model.code2wav.cuda().eval()
    hf_output = hf_code2wav.chunked_decode(
        codes=talker_codes,
    )

    # Instantiate Geri vocoder
    geri_code2wav = Qwen3OmniMoeCode2Wav(
        model_id=model_id,
        torch_dtype=torch.bfloat16,
        torch_device=torch.device("cuda"),
        config=hf_model.config.code2wav_config,
    )

    # (1, num_quantizers, seqlen) -> (seqlen, num_quantizers)
    talker_codes = talker_codes.permute(0, 2, 1).squeeze(0)
    wavs = []
    for wav_chunk in geri_code2wav.generate(prompt_embeds=[talker_codes]):
        assert len(wav_chunk) == 1
        wavs.append(wav_chunk[0])
    geri_output = torch.cat(wavs, dim=-1)

    assert_similar([hf_output], [geri_output])
