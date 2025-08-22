# Sidecar Host Only Test
This test validates serialization and deserialization between sidecar clients and the server across container boundaries. It is designed to run on a **host** environment.

If this test fails, it indicates that Sidecars will likely not function properly within a Kubernetes cluster.

# How-to
1. First build the `dev` docker image:
   ```bash
   cd <your-cornserve-root>
   REGISTRY="none" bash scripts/build_export_images.sh dev
   ```
2. Then run the test:
   ```bash
   cd <your-cornserve-root>/python/tests/services/sidecar/host_only
   pytest -s host_test_serde.py
