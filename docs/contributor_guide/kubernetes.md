## Local and Distributed Development on Kubernetes

### Build and Export Modes

The first step to running Cornserve on Kubernetes is to build container images.

We have `scripts/build_export_images.sh` to build and export container images for Cornserve components.
Now, *where to push these images* changes based on your development environment, the `REGISTRY` environment variable needs to be set accordingly.

| Dev Environment | `REGISTRY` Value | Description | Use Case |
|------|-----------------|-------------|----------|
| **Local K3s** | `local` | Builds images directly in K3s containerd (no pull needed) | Single-node K3s development |
| **Minikube** | `minikube` | Builds with Docker and loads into Minikube | Minikube development |
| **Build Only** | `none` | Builds images locally without pushing anywhere | Testing builds |
| **Registry Push** | Registry URL (e.g., `myregistry.com:5000`) | Builds and pushes to specified registry | Multi-node clusters or distributed development |

### Local development

You are developing on a single node.
In this case, we don't need a registry.
Instead, we build containers directly within the containerd runtime of K3s.

To set up your local K3s development environment, first, follow [this guide](https://blog.otvl.org/blog/k3s-loc-sp) (Section "Switching from Docker to Containerd") to set up Nerdctl and BuildKit on your local development machine.
It should be something like:

```bash
wget https://github.com/containerd/nerdctl/releases/download/v2.1.3/nerdctl-2.1.3-linux-amd64.tar.gz
tar -xvf nerdctl-*.gz
sudo mv nerdctl /usr/bin
sudo mkdir -p /etc/nerdctl/
sudo tee /etc/nerdctl/nerdctl.toml > /dev/null <<'EOF'
address        = "/run/k3s/containerd/containerd.sock"
namespace      = "k8s.io"
EOF

wget https://github.com/moby/buildkit/releases/download/v0.24.0/buildkit-v0.24.0.linux-amd64.tar.gz
tar -xvf buildkit-v0.24.0.linux-amd64.tar.gz
sudo mv bin/* /usr/bin
sudo mkdir -p /etc/buildkit/
sudo tee /etc/buildkit/buildkitd.toml > /dev/null <<'EOF'
[worker.oci]
  enabled = false

[worker.containerd]
  enabled = true
  address = "/run/k3s/containerd/containerd.sock"
  namespace = "k8s.io"
EOF

sudo tee /etc/systemd/system/buildkit.service > /dev/null <<'EOF'
[Unit]
Description=BuildKit
Documentation=https://github.com/moby/buildkit

[Service]
Type=simple
TimeoutStartSec=10
Restart=always
RestartSec=10
ExecStart=/usr/bin/buildkitd

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable buildkit.service
sudo systemctl start buildkit.service
```

After that, you can use Nerdctl to build images directly within K3s containerd, and no pull is necessary whatsoever.
Use the `build_export_images.sh` script with the `REGISTRY` environment variable set to `local` (a special case):

```bash
REGISTRY=local bash scripts/build_export_images.sh 
```

Use the `local` overlay to deploy Cornserve:

```bash
kubectl apply -k kustomize/cornserve-system/overlays/local
kubectl apply -k kustomize/cornserve/overlays/local
```

The `local` overlay specifies `imagePullPolicy: Never`, meaning that if the image was not found locally, it means that it was not built yet, correctly raising an error.

!!! NOTE  
    You can use the `local` overlay for the quick Minikube demo as well.

### Distributed development

You are developing on a multi-node cluster.

(1) Now, you do need a registry to push images to, so that remote nodes can pull them:

```bash
bash kubernetes/registry.sh
REGISTRY=myregistry.com:5000 bash kubernetes/set_registry.sh  # (1)!
```

1. Modifies `kustomization.yaml` and `k3s/registries.yaml`
   If you're on this dev workflow with a *single* node cluster, you can skip `kubernetes/set_registry.sh` because things default to `localhost:5000`.

(2) For K3s to work with insecure (i.e., HTTP) registries, you need to set up the `registries.yaml` file in `/etc/rancher/k3s/` on **all** nodes (master and worker) before starting K3s:

```bash
sudo cp kubernetes/k3s/registries.yaml /etc/rancher/k3s/registries.yaml
sudo systemctl start k3s  # or k3s-agent
```

(3) On the node that you want to build the images, you also need to specify insecure registries for docker so that it can push images to it. So, in `/etc/docker/daemon.json`, you should add something like,

```
{
    "existing configs...",
    "insecure-registries": ["myregistry.com:5000"]
}
```

Then restart docker by `sudo systemctl restart docker`.

(4) Build and push images to the registry using the `build_export_images.sh` script with the `REGISTRY` environment variable set to the registry address:

```bash
REGISTRY=myregistry.com:5000 bash scripts/build_export_images.sh
```

!!! NOTE
    Building Eric can consume a lot of memory and may trigger OOMs that freeze the instance. Please set a proper `max_jobs` in `eric.Dockerfile`.

(5) Use the `dev` overlay (which specifies `imagePullPolicy: Always`) to deploy Cornserve:

```bash
kubectl apply -k kustomize/cornserve-system/overlays/dev
kubectl apply -k kustomize/cornserve/overlays/dev
```
