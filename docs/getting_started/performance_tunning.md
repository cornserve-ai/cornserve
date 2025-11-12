# Performance Tuning with Model Fission

Cornserve supports **model fission**: running a model as several independent components (e.g., Encoder / LLM / Generator) instead of a single monolithic server.
This guide helps you decide when to use model fission versus a monolithic setup for best performance.

---

## Core Principle

Model fission works best when components are heterogeneous or components can be shared across different models.

---

## Prefer Model Fission when

- **A model has many hard-to-combine components.**
  Some models may have multiple encoders, LLMs, or even generators that are difficult to fit into a monolithic design.  
  Example: Qwen3-Omni → encoder fission to separate its thinker, talker, and vocoder components.

- **Models have large encoders with heavy multimodal embeddings.**
  When encoders are big and do substantial work, and you can match encoder scale to the LLM's throughput, fission helps.  
  Example: InternVL-38B → encoder fission if you can balance encoder vs. LLM.

- **Serving multiple models that share components.**
  Fissioning the shared components lets different models reuse them. <br>
  Example: Gemma3 4B, 12B, and 27B are served together → fission the image encoder so that all models can use the same encoder capacity.

- **App requests have heterogeneity.**
  If some requests need expensive components (e.g., audio or image generation) while others are lightweight (text-only), fission lets the heavy parts run separately so the light requests aren't slowed down.  
  Example: Qwen-Image served together with Qwen2.5-VL-8B for both image chat and image generation → encoder fission to separate the image generator from the multimodal LLM.

---

## Prefer Monolithic when

- **You can't balance the components given limited resources (not enough GPUs).**  
  If one component is about **1:100** slower than another and you don't have enough GPUs to bring them close, fission strands capacity.  
  Example: Qwen-Image served alone under tight GPU budgets → monolithic is often better.

- **LLM decode memory is the limiter (KV cache bound).**
  If throughput is capped by KV cache size, monolithic deployment generally wins because more memory will remain available for the LLM.  
  Example: Qwen2.5-VL → monolithic is usually better since performance is typically limited by LLM decode memory size.

---

## Quick Checklist

- Are components heterogeneous **and** can you scale each one close to its demand? → **Fission**  
- Are you serving **multiple models** with a **shared** encoder/backbone? → **Fission**  
- Is the workload **KV-cache bound** at the LLM decode? → **Monolithic**  
- Are GPUs **too limited** to balance a large cost gap (e.g., ~1:100)? → **Monolithic**
---

