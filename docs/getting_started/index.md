---
description: Getting started with Cornserve
---

# Getting Started

## Try it out in Minikube!

You can try out Cornserve on your local machine (with Docker and at least two NVIDIA GPUs) using [Minikube](https://minikube.sigs.k8s.io).

First, install Minikube following their [guide](https://minikube.sigs.k8s.io/docs/start/).

Then, start a Minikube cluster with GPU support:

```bash
minikube start \
    --driver docker \
    --container-runtime docker \
    --gpus all \
    --disk-size 50g  # (1)!
```

1. Give it enough disk space to download model weights and stuff. You can also give more CPU (e.g., `--cpus 8`) and memory (`--memory 16g`).

Next, and this is important, we want to increase the shared memory (`/dev/shm`) size of the Minikube container.

```console
$ minikube ssh -- sudo mount -o remount,size=16G /dev/shm
```

Next, clone the Cornserve GitHub repository and deploy Cornserve on your Minikube cluster:

```bash
git clone git@github.com:cornserve-ai/cornserve.git
cd cornserve

minikube kubectl -- apply -k kubernetes/kustomize/cornserve-system/overlays/minikube
minikube kubectl -- apply -k kubernetes/kustomize/cornserve/overlays/minikube
```

After a few moments (which largely depends on how long it takes to pull Docker images from Docker Hub), check whether Cornserve is running:

```console
$ minikube kubectl -- get -n cornserve pods   # (1)!
NAME                               READY   STATUS    RESTARTS   AGE
gateway-6c65745c5d-x8gkh           1/1     Running   0          4s
resource-manager-9b4df4687-9djc4   1/1     Running   0          4s
task-dispatcher-9954cffcd-g4rk2    1/1     Running   0          4s
sidecar-0                          1/1     Running   0          3s
sidecar-1                          1/1     Running   0          3s
sidecar-2                          1/1     Running   0          3s
sidecar-3                          1/1     Running   0          3s
```

1. The number of Sidecar pods should match the number of GPUs you gave to Minikube. They are spawned by the Resource Manager, so you will initially see only three (Gateway, Resource Manager, and Task Dispatcher) pods running.

Next, install the Cornserve CLI that helps you interact with Cornserve:

```bash
# Configure a virtual environment with Python 3.11+ as needed.
pip install cornserve
```

Try registering a simple example app that defines a Vision-Language Model:

```bash
export CORNSERVE_GATEWAY_URL=$(minikube service -n cornserve gateway-node-port --url)
cornserve register examples/mllm/app.py --alias mllm
```

You can check out what the app looks like on [GitHub](https://github.com/cornserve-ai/cornserve/blob/3fbf3c62dc7bd8019af29d1ae260b2cafc071ad8/examples/mllm/app.py).

This will take a few minutes. The two large bits are (1) pulling in the Docker images and (2) waiting for vLLM to warm up and start. But eventually, you should see something like this:

```console
$ cornserve register examples/mllm/app.py --alias mllm
╭──────────────────────────────────────┬───────╮
│ App ID                               │ Alias │
├──────────────────────────────────────┼───────┤
│ app-564b79ff446342c69821464b22585a72 │ mllm  │
╰──────────────────────────────────────┴───────╯
```

Now, you can invoke the app using the CLI:

```console
$ cornserve invoke mllm - <<EOF
prompt: "Describe what you see in the two images, in detail."
multimodal_data:
- ["image", "https://picsum.photos/id/12/480/560"]
- ["image", "https://picsum.photos/id/234/960/960"]
EOF
╭──────────┬───────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ response │ The first image depicts a serene beach scene with a rocky foreground and a sandy beach extending into the │
│          │ distance. The rocks are dark and jagged, contrasting with the smooth, golden sand. The ocean is calm,     │
│          │ with gentle waves lapping against the shore. In the background, there is a line of trees or a forested    │
│          │ area, adding depth to the scene. The sky is clear, suggesting a bright and sunny day.                     │
│          │                                                                                                           │
│          │ The second image shows a bustling city street with the Eiffel Tower prominently visible in the            │
│          │ background. The tower is tall and slender, with a metal lattice structure. The street is lined with       │
│          │ trees, some of which have bare branches, indicating a winter or early spring setting. There are several   │
│          │ buildings along the street, including a large, ornate building with multiple stories and balconies. The   │
│          │ street is filled with cars and pedestrians, giving the scene a lively and dynamic atmosphere. The overall │
│          │ color tone of the image is muted, with a sepia-like effect, adding a vintage or nostalgic feel to the     │
│          │ photograph.                                                                                               │
╰──────────┴───────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

The invocation payload and response schema are defined by [the app itself](https://github.com/cornserve-ai/cornserve/blob/3fbf3c62dc7bd8019af29d1ae260b2cafc071ad8/examples/mllm/app.py) as a `AppRequest` and `AppResponse` subclass.
You can learn more about defining apps (and tasks) [in our guide](building_apps.md).

Here's how to clean up:

```bash
minikube kubectl -- delete -k kubernetes/kustomize/cornserve/overlays/minikube
minikube kubectl -- delete -k kubernetes/kustomize/cornserve-system/overlays/minikube
minikube stop  # or minikube delete
```

## Getting started (seriously)

At a high level, there are two steps to using Cornserve:

1. [**Cornserve deployment**](cornserve.md): Deploying Cornserve on a GPU cluster managed by Kubernetes.
1. [**Building your app**](building_apps.md): Building a Cornserve app and deploying it on a Cornserve cluster for invocation.
1. [**Interactively debugging your app with Jupyter notebook**](jupyter.ipynb): Building a Cornserve app and deploying it on a Cornserve cluster for invocation.
1. [**Registering and invoking your app**](registering_apps.md): Building a Cornserve app and deploying it on a Cornserve cluster for invocation.
