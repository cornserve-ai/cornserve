## Local and Distributed Development on Kubernetes

### Local development

You are developing on a single node.
In this case, we don't need a registry.
Instead, we build containers directly within the containerd runtime of K3s.

First, follow [this guide](https://blog.otvl.org/blog/k3s-loc-sp) (Section "Switching from Docker to Containerd") to set up Nerdctl and BuildKit on your local development machine.

After that, you can use Nerdctl to build images directly within K3s containerd, and no pull is necessary whatsoever.
Use the `build_export_images.sh` script with the `REGISTRY` environment variable set to `local` (a special case):

```bash
REGISTRY=local bash scripts/build_export_images.sh 
```

Use the `local` overlay to deploy Cornserve:

```bash
kubectl apply -k kustomize/cornserve/overlays/local kustomize/cornserve-system/overlays/local
```

The `local` overlay specifies `imagePullPolicy: Never`, meaning that if the image was not found locally, it means that it was not built yet, correctly raising an error.

!!! NOTE  
    You can use the `local` overlay for the quick Minikube demo as well.

### Distributed development

You are developing on a multi-node cluster.

(1) Now, you do need a registry to push images to, so that remote nodes can pull them:

```bash
bash kubernetes/registry.sh
REGISTRY=myregisty.com:5000 bash kubernetes/set_registry.sh  # (1)!
```

1. Modifies `kustomization.yaml` and `k3s/registries.yaml`
   If you're on this dev workflow with a *single* node cluster, you can skip `kubernetes/set_registry.sh` because things default to `localhost:5000`.

(2) For K3s to work with insecure (i.e., HTTP) registries, you need to set up the `registries.yaml` file in `/etc/rancher/k3s/` on **all** nodes (master and worker) before starting K3s:

```bash
sudo cp kubernetes/k3s/registries.yaml /etc/rancher/k3s/registries.yaml
sudo systemctl start k3s  # or k3s-agent
```

(3) Build and push images to the registry using the `build_export_images.sh` script with the `REGISTRY` environment variable set to the registry address:

```bash
REGISTRY=myregistry.com:5000 bash scripts/build_export_images.sh
```

(4) Use the `dev` overlay (which specifies `imagePullPolicy: Always`) to deploy Cornserve:

```bash
kubectl apply -k kustomize/cornserve/overlays/dev kustomize/cornserve-system/overlays/dev
```
