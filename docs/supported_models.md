# Supported Models

Cornserve supports a variety of multimodal AI models through its specialized task executors: **Eric** (encoder) for embedding multimodal data, and **Geri** (generator) for generating multimodal content.

## Multimodal models

| Model | Input Modalities | Output Modalities |
|-------|------------------|-------------------|
| [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) | Image, Video | Embeddings |
| [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | Image, Video | Embeddings |
| [Qwen/Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) | Image, Video, Audio | Embeddings |
| [Qwen/Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) | Image, Video, Audio | Embeddings |
| [llava-hf/llava-onevision-qwen2-7b-ov-chat-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-chat-hf) | Image, Video | Embeddings |
| [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) | Image | Embeddings |
| [OpenGVLab/InternVL3-1B](https://huggingface.co/OpenGVLab/InternVL3-1B) | Image, Video | Embeddings |
| [OpenGVLab/InternVL3-38B](https://huggingface.co/OpenGVLab/InternVL3-38B) | Image, Video | Embeddings |
| [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) | Text | Image |

## Text-only language models

For text-only language model inference (without multimodal inputs or outputs), Cornserve integrates with vLLM. vLLM supports a wide range of language models including popular families like Llama 3, Llama 2, Mistral, Mixtral, Qwen2, Qwen2.5, Gemma, Gemma 2, and many more.

For a complete list of supported text-only language models and their requirements, please refer to the [vLLM documentation on supported models](https://docs.vllm.ai/en/latest/models/supported_models.html).
