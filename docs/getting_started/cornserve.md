# Deploying Cornserve

Cornserve can be deployed on a GPU cluster managed by Kubernetes.

## Deploying K3s

If you don't already have Kubernetes deployed on your cluster, you can use the lightweight [k3s](https://k3s.io) distribution.
Their [Documentation](https://docs.k3s.io/quick-start/) provides a quick start guide.

!!! Tip
    If you're deploying on-premise with k3s, make sure you have plenty of disk space under `/var/lib/rancher` because `containerd` stores images there.
    If not, you can create a directory in a secondary storage (e.g., `/mnt/data/rancher`) and symlink it to `/var/lib/rancher` prior to starting k3s.

### NVIDIA Device Plugin

The [NVIDIA GPU Device Plugin](https://github.com/NVIDIA/k8s-device-plugin) is required to expose GPUs to the Kubernetes cluster as resources.
You can deploy a specific version like this:

```bash
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.2/deployments/static/nvidia-device-plugin.yml
```

### Clone the Repository

```bash
git clone git@github.com:cornstarch-org/cornserve.git
cd cornserve/kubernetes
```

### Master Node

Start a private image registry:

```bash
bash registry.sh
```

Replace the endpoint in `registries.yaml` with the registry address.

Install and start K3s:

```bash
curl -sfL https://get.k3s.io | INSTALL_K3S_SKIP_ENABLE=true sh -
sudo mkdir -p /etc/rancher/k3s
sudo cp k3s/server-config.yaml /etc/rancher/k3s/config.yaml
sudo cp k3s/registries.yaml /etc/rancher/k3s/registries.yaml
sudo systemctl start k3s
```

Note the master node address (`$MASTER_ADDRESS`) and the node token (`$NODE_TOKEN`):

```bash
NODE_TOKEN="$(sudo cat /var/lib/rancher/k3s/server/node-token)"
```

### Worker Nodes

Install and start K3s:

```bash
curl -sfL https://get.k3s.io | K3S_URL=https://$MASTER_ADDRESS:6443 K3S_TOKEN=$NODE_TOKEN INSTALL_K3S_SKIP_ENABLE=true sh -
sudo mkdir -p /etc/rancher/k3s
sudo cp k3s/agent-config.yaml /etc/rancher/k3s/config.yaml
sudo cp k3s/registries.yaml /etc/rancher/k3s/registries.yaml
sudo systemctl start k3s-agent
```

## Deploying Cornserve

On top of a Kubernetes cluster, you can deploy Cornserve with a single command:

```bash
kubectl apply -k kustomize/cornserve/base kustomize/cornserve-system/base
```

!!! Note
    The `cornserve` namespace is used for most of our control plane and data plane objects.
    On the other hand, the `cornserve-system` namespace is used for components that look over and manage the Cornserve system itself (under `cornserve`), like Jaeger and Prometheus.

We also have an overlay for development (`dev`), which, for instance, adds a `NodePort` service for the Cornserve Gateway and Jaeger for local testing.

```bash
kubectl apply -k kustomize/cornserve/overlays/dev kustomize/cornserve-system/overlays/dev
```

We suggest that you tweak the manifest files to suit your needs.
