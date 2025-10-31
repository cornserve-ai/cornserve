# Supported Models

Cornserve supports a variety of multimodal AI models through its specialized task executors: **Eric** (encoder) for embedding multimodal data, and **Geri** (generator) for generating multimodal content.

## Eric: Multimodal Encoders

Eric is Cornserve's multimodal data embedding server that computes embeddings from images, videos, and audio. The following models are currently supported:

### Vision-Language Models

#### Qwen2-VL

- **Model**: [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
- **Modalities**: Image, Video
- **Description**: Qwen2-VL is a vision-language model with dynamic resolution support

#### Qwen2.5-VL

- **Model**: [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- **Modalities**: Image, Video
- **Description**: Enhanced version of Qwen2-VL with improved performance

### Omni-Modal Models

#### Qwen2.5-Omni

- **Model**: [Qwen/Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)
- **Modalities**: Image, Video, Audio
- **Description**: Multimodal model supporting vision and audio inputs

#### Qwen3-Omni-MoE

- **Model**: [Qwen/Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct)
- **Modalities**: Image, Video, Audio
- **Description**: Mixture-of-Experts architecture for efficient multimodal processing

### Additional Vision-Language Models

#### LLaVA-OneVision

- **Model**: [llava-hf/llava-onevision-qwen2-7b-ov-chat-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-chat-hf)
- **Modalities**: Image, Video
- **Description**: LLaVA-OneVision with Qwen2 backbone

#### Gemma 3

- **Model**: [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it)
- **Modalities**: Image
- **Description**: Google's Gemma 3 vision-language model

#### InternVL3

- **Models**: 
    - [OpenGVLab/InternVL3-1B](https://huggingface.co/OpenGVLab/InternVL3-1B)
    - [OpenGVLab/InternVL3-38B](https://huggingface.co/OpenGVLab/InternVL3-38B)
- **Modalities**: Image, Video
- **Description**: InternVL3 series with dynamic resolution support

## Geri: Multimodal Generators

Geri is Cornserve's multimodal content generation server that generates images, videos, and audio from embeddings.

### Image Generation

#### Qwen-Image

- **Model**: [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image)
- **Modalities**: Image generation
- **Description**: Text-to-image generation model

## Text-Only Language Models

For text-only language model inference (without multimodal inputs or outputs), Cornserve integrates with vLLM. vLLM supports a wide range of language models including popular families like:

- Llama 3 and Llama 2
- Mistral and Mixtral
- Qwen2 and Qwen2.5
- Gemma and Gemma 2
- And many more

For a complete list of supported text-only language models and their requirements, please refer to the [vLLM documentation on supported models](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Adding New Models

To add support for new models, please refer to the [Contributor Guide](../contributor_guide/eric.md) for information on how to implement new model encoders and register them in Cornserve.
