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


## Debugging

To debug UCX related error, you can set `UCX_LOG_LEVEL=trace` and `UCXPY_LOG_LEVEL=DEBUG`
