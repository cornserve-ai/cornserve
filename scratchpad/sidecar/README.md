# Sidecar example

`sender.py` and `receiver.py` are examples of how task executors can instantiate sidecar clients to transfer data.

- To run this pair, you need to first deploy the [k3s cluster](/kubernetes/README.md).
- Then initialize the cluster with `kubectl apply -k kubernetes/kustomize/base/`
- For now receivers must be launched before senders start to send, so:
   ```bash
   kubectl apply -k kubernetes/kustomize/examples/double-sender/llm
   # make sure the receiver LLM container is running
   kubectl apply -k kubernetes/kustomize/examples/double-sender/erics
   ```
- You can verify the transfer status by inspecting the container logs.
