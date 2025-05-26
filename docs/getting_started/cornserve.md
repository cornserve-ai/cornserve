# Deploying Cornserve

Cornserve can be deployed on a GPU cluster managed by Kubernetes.

!!! Note
    The `cornserve` namespace is used for most of our control plane and data plane objects.
    On the other hand, the `cornserve-system` namespace is used for components that look over and manage the Cornserve system itself (under `cornserve`), like Jaeger and Prometheus.
If you already have a Kubernetes cluster running, you can deploy Cornserve on it with the `prod` overlay:

## Deploying K3s

!!! Tip
    If you have a Kubernetes cluster running, you can skip this section.

If you don't have a Kubernetes cluster running, you can deploy Cornserve on a K3s cluster.
We also use the [K3s](https://k3s.io/) distribution of Kubernetes for our development.
Refer to their [Documentation](https://docs.k3s.io/quick-start/) for more details.

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
git clone git@github.com:cornserve-ai/cornserve.git
cd cornserve/kubernetes
```

### Master Node

Install and start K3s:

```bash
curl -sfL https://get.k3s.io | INSTALL_K3S_SKIP_ENABLE=true sh -
sudo mkdir -p /etc/rancher/k3s
sudo cp k3s/server-config.yaml /etc/rancher/k3s/config.yaml
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
sudo systemctl start k3s-agent
```

## Deploying Cornserve

If you haven't already, clone the Cornserve repository:

```bash
git clone git@github.com:cornserve-ai/cornserve.git
cd cornserve
```

On top of a Kubernetes cluster, you can deploy Cornserve with a single command:

```bash
kubectl apply -k kubernetes/kustomize/cornserve-system/base
kubectl apply -k kubernetes/kustomize/cornserve/overlays/prod
```

!!! Note
    The `cornserve` namespace is used for most of our control plane and data plane objects.
    On the other hand, the `cornserve-system` namespace is used for components that look over and manage the Cornserve system itself (under `cornserve`), like Jaeger and Prometheus.
