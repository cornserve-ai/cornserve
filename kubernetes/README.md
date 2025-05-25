# Deploying Cornserve

Please refer to our [documentation](https://docs.cornserve.ai/getting-started/cornserve/) for instructions on deploying Cornserve on Kubernetes.

!!! Note
    The `cornserve` namespace is used for most of our control plane and data plane objects.
    On the other hand, the `cornserve-system` namespace is used for components that look over and manage the Cornserve system itself (under `cornserve`), like Jaeger and Prometheus.
If you already have a Kubernetes cluster running, you can deploy Cornserve on it with the `prod` overlay:

## Deploying K3s

If you don't have a Kubernetes cluster running, you can deploy Cornserve on a K3s cluster.
If you do have a cluster running, you can skip this section.
We also use the [K3s](https://k3s.io/) distribution of Kubernetes for our development.
Refer to their [Documentation](https://docs.k3s.io/quick-start/) for more details.

### Disk space

Make sure the path under `/var/lib/rancher` has enough disk space, particularly for containerd to store container filesystems in.
If not, create a directory in secondary storage (e.g., `/mnt/data/rancher`) and symlink it to `/var/lib/rancher`.

### Master node

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

On all other (worker) nodes:

```bash
curl -sfL https://get.k3s.io | K3S_URL=https://$MASTER_ADDRESS:6443 K3S_TOKEN=$NODE_TOKEN INSTALL_K3S_SKIP_ENABLE=true sh -
sudo mkdir -p /etc/rancher/k3s
sudo cp k3s/agent-config.yaml /etc/rancher/k3s/config.yaml
sudo cp k3s/registries.yaml /etc/rancher/k3s/registries.yaml
sudo systemctl start k3s-agent
```

## Deploying on Kubernetes

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

### Deploy the `prod` overlay

```bash
kubectl apply -k kustomize/cornserve-system/base kustomize/cornserve/overlays/prod
```

Update `kustomize/overlays/prod/kustomization.yaml` to change the version of the image to deploy.
The default is the latest release version of Cornserve.
