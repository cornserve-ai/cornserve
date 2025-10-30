# Sidecar developer guide

## Docker container

It is recommended to run everything inside docker. Sidecar uses `UCX` as backend,
so you might find the `docker/dev.Dockerfile` helpful. Additionally, Sidecar has 
dependency over `ucxx-cu12`, meaning you need to development on an Nvidia
GPU-enabled machine at the moment.

Specifying `--shm-size` with at least 4 GB and `--ipc host` is required.

## Editable installation

```bash
pip install -e 'python[dev]'
```

## Testing

We use pytest.

```bash
pytest python/tests/services/sidecar/test_sidecar.py
```

When testing locally with task executors, you can `export SIDECAR_IS_LOCAL=true` to
route all communications through `localhost` instead of k8s network.


## Testing with Mock Sidecar

You can test components (e.g., Geri, Task Executors) that use sidecar clients without spawning the sidecar server, running k3s, or doing any communication by using the mock sidecar mode.
This is particularly useful for local development and testing.

### Setup

1. **Enable mock sidecar mode** inside your Docker container:
   ```bash
   export CORNSERVE_MOCK_SIDECAR=1
   ```

2. **Generate and export a mock mapping**. The mapping is a JSON object that maps `{data_id}-{chunk_id}` to file paths where the sidecar will read/write data:
   ```python
   import json

   tensor_data_id = "audio_code"
   json_data_id = "metadata"

   # Generate mapping
   # Streaming audio code (two chunks), single-chunk JSON metadata
   mock_mapping = {
       f"{tensor_data_id}-0": "/path/to/code0.pt",
       f"{tensor_data_id}-1": "/path/to/code1.pt",
       f"{json_data_id}-0": "/path/to/data.json",
       # Add more mappings as needed
   }

   # Export as environment variable
   os.environ["CORNSERVE_MOCK_SIDECAR_MAPPING"] = json.dumps(mock_mapping)
   ```

3. **Fire sends and receives** with the sidecar client using the same data and chunk IDs as in the mapping. The sidecar client will read from/write to the specified file paths instead of actual network communication.

### Example

See `python/tests/services/sidecar/test_mock_sidecar.py` for a complete example.
The test demonstrates creating a mock mapping with various data types (tensors, chunked tensors, `dict`, primitives like `int`).


## Debugging

To debug UCX related error, you can set `UCX_LOG_LEVEL=trace` and `UCXPY_LOG_LEVEL=DEBUG`
