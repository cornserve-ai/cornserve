---
description: Cornserve contributor guide
---

# Contributor Guide

Here, we provide more info for contributors.
General principles are here, and child pages discuss specific topics in more detail.

We have a few principles for developing Cornserve:

1. **Strict type annotations**: We enforce strict type annotation everywhere in the Python codebase, which leads to numerous benefits including better reliability, readability, and editor support. We use `pyright` for type checking.
1. **Automated testing**: We don't aim for 100% test coverage, but non-trivial and/or critical features should be tested with `pytest`.

## Contributing process

!!! Important
    By contributing to Cornserve, you agree that your code will be licensed with Apache 2.0.

If the feature is not small or requires broad changes over the codebase, please **open an issue** at our GitHub repository to discuss with us.

1. Fork our GitHub repository. Make sure you clone with `--recurse-submodules` to get the submodules.
1. Create a new virtual environment:
    ```bash
    uv venv --python=3.12
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
1. Install Cornserve in editable mode:
    ```bash
    # For environments with GPU and nvcc (to compile FlashAttention)
    uv pip install -e 'python[dev]' --config-settings editable_mode=strict

    # For environments without GPU (e.g., editor/IDE environment)
    uv pip install -e 'python[dev-no-gpu]' --config-settings editable_mode=strict

    uv pip install -e 'tasklib/'
    ```
1. Generate Python bindings for Protobuf files with `uv run bash scripts/generate_pb.sh`.
1. Implement changes in your branch and add tests as needed.
1. Ensure `uv run bash python/scripts/lint.sh` and `pytest` passes. Note that many of our tests require GPU.
1. Submit a PR to the main repository. Please ensure that CI (GitHub Actions) passes.

## Developing on Kubernetes

Cornserve runs on top of Kubernetes, which introduces some complexity in development.
Please refer to the guide on [Local and Distributed Development on Kubernetes](kubernetes.md) for more details.

## Documentation

The documentation is written in Markdown and is located in the `docs` folder.
We use MkDocs to build the documentation and use the `mkdocs-material` theme.

To install documentation build dependencies:

```bash
uv pip install -r docs/requirements.txt
```

To build and preview the documentation:

```bash
uv run bash scripts/preview_docs.sh
```
