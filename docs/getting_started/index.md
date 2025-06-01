---
description: Getting started with Cornserve
---

# Getting Started

## Try it out in Minikube!

You can try out Cornserve on your local machine (with Docker and at least two NVIDIA GPUs(1)) using [Minikube](https://minikube.sigs.k8s.io).
{ .annotate }

1. [Compute Capability](https://developer.nvidia.com/cuda-gpus) >= 8.0

First, install Minikube following their [guide](https://minikube.sigs.k8s.io/docs/start/).

Then, start a Minikube cluster with GPU support (1):
{ .annotate }

1. We recommend enabling rootless docker to avoid permission or `$PATH` related issues.

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
git clone https://github.com/cornserve-ai/cornserve.git
cd cornserve

minikube kubectl -- apply -k kubernetes/kustomize/cornserve-system/overlays/minikube
minikube kubectl -- apply -k kubernetes/kustomize/cornserve/overlays/minikube
```

We'll be using [Gemma 3 4B](https://huggingface.co/google/gemma-3-4b-it) for this demo, so you need to have access (requests are processed immediately with an account).
While we wait for the containers to spin up, add your HuggingFace access token to Cornserve, which can be created [here](https://huggingface.co/settings/tokens) if you don't have one already.
```
minikube kubectl -- create -n cornserve secret generic cornserve-env --from-literal=hf-token='YOUR_HUGGINGFACE_TOKEN'
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
prompt: "Write a haiku about each image."
multimodal_data:
- ["image", "https://picsum.photos/id/12/480/560"]
- ["image", "https://picsum.photos/id/234/960/960"]
EOF
╭──────────┬───────────────────────────────────────────────────────────────────────────╮
│ response │ Okay, here are haikus for each image:                                     │
│          │                                                                           │
│          │ **Image 1: Coastal Landscape**                                            │
│          │                                                                           │
│          │ Gray sea meets the shore,                                                 │
│          │ Rocks stand still, a weathered grace,                                     │
│          │ Island dreams unfold.                                                     │
│          │                                                                           │
│          │ **Image 2: Paris Scene**                                                  │
│          │                                                                           │
│          │ Fog veils city’s height,                                                  │
│          │ Eiffel stands, a ghostly trace,                                           │
│          │ Winter’s quiet grace.                                                     │
│          │                                                                           │
│          │ ---                                                                       │
│          │                                                                           │
│          │ Would you like me to create haikus for any other images you have in mind? │
╰──────────┴───────────────────────────────────────────────────────────────────────────╯
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
