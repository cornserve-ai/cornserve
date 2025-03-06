# Eric developer guide

## Docker container

All code is to be run inside a Docker container, including tests.

```bash
docker build -t cornserve/eric:latest -f docker/eric.Dockerfile .
docker run -it --gpus all --entrypoint bash --ipc host --rm --name eric-dev -v $PWD:/workspace/cornserve -v $HF_CACHE:/root/.cache/huggingface cornserve/eric:latest
```

## Editable installation

```bash
pip install -e 'python[dev]'
```

## Testing

We use pytest. Tests use GPUs.

```bash
pytest
```

Set the `CORNSERVE_TEST_DUMP_TENSOR_DIR` to an existing directory when running pytest.
This will dump output embedding tensors to the specified directory.
Refer to `build_batch` in `tests/task_executors/eric/utils.py`.

```bash
export CORNERSERVE_TEST_DUMP_TENSOR_DIR=/path/to/dump
pytest python/tests/task_executors/eric/models/test_llava_onevision.py::test_image_inference
```
