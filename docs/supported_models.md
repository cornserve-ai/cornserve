# Supported Models

Cornserve supports a wide range of multimodal models from Hugging Face Hub, with flexible deployment options that enable efficient resource sharing and component disaggregation.

## Overview

Cornserve's model support philosophy:

- **Hugging Face Native**: Models are loaded directly from Hugging Face Hub using standard model IDs.
- **Flexible Deployment**: Monolithic, encoder-fission, disaggregated, etc., you choose.
- **Component Sharing**: Multiple models ("apps") can share encoder, decoder, or other component deployments if they are the same.
- **Multimodal First**: Native support for text, image, video, and audio inputs and outputs.

## Models by Category

### Omni-modality (Any-to-Any) Models

Models that map multiple modalities to multiple modalities.

| Model Family | Example Models | Input Modalities | Output Modalities |
|--------------|----------------|------------------|-------------------|
| **Qwen3-Omni** | [`Qwen/Qwen3-Omni-30B-A3B-Instruct`](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) | Text + Image + Video + Audio | Text + Audio |

### Multimodal Input Models

Multimodal LLMs that accept multiple input modalities (e.g., text + image) and produce text outputs.

| Model Family | Example Models | Input Modalities |
|--------------|----------------|------------------|
| **Gemma 3** | [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it)<br/>[`google/gemma-3-12b-it`](https://huggingface.co/google/gemma-3-12b-it)<br/>[`google/gemma-3-27b-it`](https://huggingface.co/google/gemma-3-27b-it) | Text + Image |
| **Qwen2-VL** | [`Qwen/Qwen2-VL-7B-Instruct`](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) | Text + Image + Video |
| **Qwen2.5-VL** | [`Qwen/Qwen2.5-VL-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | Text + Image + Video |
| **Qwen3-VL** | [`Qwen/Qwen3-VL-4B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | Text + Image + Video |
| **Qwen3-VL-MoE** | [`Qwen/Qwen3-VL-30B-A3B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct) | Text + Image + Video |
| **InternVL3** | [`OpenGVLab/InternVL3-1B`](https://huggingface.co/OpenGVLab/InternVL3-1B)<br/>[`OpenGVLab/InternVL3-38B`](https://huggingface.co/OpenGVLab/InternVL3-38B) | Text + Image + Video |
| **LLaVA-OneVision** | [`llava-hf/llava-onevision-qwen2-7b-ov-hf`](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf) | Text + Image + Video |

### Multimodal Output Models

Models that generate non-text outputs (e.g., images).

| Model Family | Example Models | Input Modalities | Output Modalities |
|--------------|----------------|------------------|-------------------|
| **Qwen-Image** | [`Qwen/Qwen-Image`](https://huggingface.co/Qwen/Qwen-Image) | Text | Image |

### Text-Only Language Models

Cornserve automatically supports any text-only LLMs supported by vLLM.
Refer to [vLLM's supported models documentation](https://docs.vllm.ai/en/latest/models/supported_models) for the complete list.

!!! Note
    Cornserve uses a fork of vLLM, which can lag slightly behind the latest vLLM releases. If you need support for a newly released model, please file an issue on our repository, or we'd be happy to accept contributions.
