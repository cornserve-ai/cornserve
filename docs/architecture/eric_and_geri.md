# Eric and Geri: Multimodal Task Executors

> **Mosharaf**: Hey what is this "Eric" thing in the architecture diagram?  
  **Jae-Won**: Oh. Uh, no. It says "Enc." For Encoder.  
  **Mosharaf**: Oh.  
  **Jae-Won**: Now it's Eric.

## Eric: Multimodal Data Embedding Server

Package: `cornserve.task_executors.eric`

Eric is a multimodal data embedding server that takes in a list of multimodal data (e.g., images, videos) and computes the multimodal embedding of the input data.

## Architecture

Below, components are divided at the process boundary.

### Router

The gateway router is an async FastAPI server that (1) receives modality encoding requests and (2) preprocesses modality data before running the encoder.
Preprocessing is done asynchronously in a thread pool by the `eric.router.processor.Processor` class.

Each model processes different modality data differently, so the router must instantiate the correct model-specific preprocessor.
Instantiating and invoking these model- and modality-specific preprocessors are implemented in the class `eric.models.[model_module].ModalityProcessor`, which is a subclass of `eric.models.base.BaseModalityProcessor`.

When modality preprocessing is complete, the router submits the embedding request to the engine.
The router and the engine communicate through ZMQ sockets. Especially, the router holds an instance of the engine client (`eric.engine.client.EngineClient`), which is used to send requests to the engine and receive responses.

### Engine

From the engine and below, everything is synchronous Python (i.e., not `asyncio`).

The Engine constantly receives embedding requests from the router, runs the request scheduler to create a `eric.schema.Batch`, and invokes the model executor (`eric.executor.executor.ModelExecutor`) to compute the multimodal embedding.
The model executor provides the `execute_model` method, which broadcasts input batch data to all Workers via shared memory.

The engine currently only batches data of the same modality together. This is because there are models that have different code paths for different modalities. Furthermore, due to the compute-intensive nature of multimodal encoders, it is unlikely we will scale to large batch sizes.

### Workers

There is one worker (`eric.executor.worker.Worker`) process per GPU. The number of workers is the tensor parallelism degree.
When spawned, the workers initialize PyTorch distributed and instantiate the model from weights downloaded from the Hugging Face Hub.
It then waits for the model executor to dispatch a batch to it, runs tensor parallel inference, and dispatches tensor communication to the destination Task Executor via the [sidecar](sidecar.md).

## Geri: Multimodal Content Generation Server

Package: `cornserve.task_executors.geri`

Geri (pronounced "Jerry") is a multimodal content generation server that takes in embeddings and generates multimodal content such as images, videos, or audio. It is the counterpart to Eric—where Eric encodes multimodal data into embeddings, Geri decodes embeddings into generated content.

### Architecture

Geri shares the same architectural design as Eric, with components divided at the process boundary:

**Router**: An async FastAPI server (`geri.router`) that receives generation requests via the `/generate` endpoint. The router communicates with the engine through ZMQ sockets using an `EngineClient` (`geri.engine.client.EngineClient`), similar to Eric's design.

**Engine**: A synchronous Python engine (`geri.engine.core`) that receives generation requests from the router, runs the request scheduler to create batches, and invokes the model executor (`geri.executor.executor.ModelExecutor`) to generate the requested content. Like Eric, the engine batches requests and broadcasts input data to workers via shared memory.

**Workers**: One worker process per GPU (`geri.executor.worker.Worker`), matching the tensor parallelism degree. Workers initialize PyTorch distributed, load the generative model from the Hugging Face Hub, and perform tensor parallel inference to generate content. Generated outputs are communicated via the [sidecar](sidecar.md) when needed.

The key difference from Eric is in the task being performed: Geri performs generative inference (embeddings to content) rather than encoding (content to embeddings), but the overall system architecture and communication patterns remain the same.
