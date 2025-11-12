# Performance Tuning with Model Fission

One of Cornserve's main feature is **model fission**: running a model with complex structure as several independent micro-servers (e.g., Encoder / LLM / Generator) instead of a single monolithic server.
However, model fission is not always the best choice for performance.
This guide helps you decide when to use model fission versus a monolithic setup for best performance.

## Core Principle

Model fission works best when components are heterogeneous or components can be shared across different models.

## Prefer Model Fission when

- **A model has many hard-to-combine components.**
  Some models may have multiple encoders, LLMs, or even generators that are difficult to fit into a monolithic server. Also, some optimization applied to different components may conflict within a single server.
  Example: Qwen3-Omni → The thinker and the talker are in itself LLMs, so fission makes sense. Also, the audio generator can be separated out as a separate server (Geri) to prvent interfering the talker LLM generation steps.

- **A model has large encoders with heavy multimodal embeddings.**
  When encoders are big and do substantial work, and you can match encoder scale to the LLM's throughput, fission helps.  
  Example: InternVL-38B → encoder fission if you can balance encoder vs. LLM.

- **Serving multiple models that share components.**
  Fissioning the shared components lets different models reuse them.  
  Example: Gemma3 4B, 12B, and 27B are served together → fission the image encoder so that all models can use the same encoder server (Eric).

- **App requests have high heterogeneity.**
  If some requests need expensive components (e.g., audio or image generation) while others are lightweight (text-only), fission lets the heavy parts run separately so the light requests aren't slowed down or blocked.  
  Example: Any MLLM app with some requests having text-only and others potentially having heavy image/video inputs → fission will unblock text-only requests.

## Prefer Monolithic when

- **You can't balance the components given limited resources (not enough GPUs).**  
  If one component is about **1:100** slower than another and you don't have enough GPUs to bring them close, fission strands capacity.  
  Example: Qwen-Image served alone under tight GPU budgets → monolithic is often better.

- **LLM decode memory is the limiter (KV cache bound).**
  If throughput is capped by KV cache size, monolithic deployment generally wins because more memory will remain available for the LLM to increase its KV cache budget.
  Example: Qwen2.5-VL under high concurrency → monolithic is usually better since high throughput demands a large batch size, which requires a large KV cache budget.

## An Automated Planner

The Cornserve team is working on an automated planner that will profile your model components and resources, and recommend the best deployment strategy (fission vs. monolithic) for optimal performance.
Stay tuned for updates in future releases!
