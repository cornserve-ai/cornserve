# Deploying on Kubernetes

## Deploying Kubernetes

We use the lightweight [K3s](https://k3s.io/) distribution of Kubernetes for our deployments.
Refer to their [Documentation](https://docs.k3s.io/quick-start/) for details.

### Master node

Start a private registry:

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

On all other (worker) nodes:

```bash
# NOTE: This snippet has not been validated yet. I just wrote it as I would first try.
# Install and start K3s
curl -sfL https://get.k3s.io | K3S_URL=https://$MASTER_ADDRESS:6443 K3S_TOKEN=$NODE_TOKEN INSTALL_K3S_SKIP_ENABLE=true sh -
sudo mkdir -p /etc/rancher/k3s
sudo cp k3s/agent-config.yaml /etc/rancher/k3s/config.yaml
sudo cp k3s/registries.yaml /etc/rancher/k3s/registries.yaml
sudo systemctl start k3s-agent
```

## Cluster Bootstrapping

After deploying Kubernetes, deploy the control plane and workers:

```bash
# Deploying the `base` overlay
kubectl apply -k kustomize/base

# Deploying the `dev` overlay
kubectl apply -k kustomize/overlays/dev
```

## Install NVIDIA device plugin 
After installing the Nvidia container toolkit, install the NVIDIA device plugin
so that containers can request GPUs as resources. More in the officail [doc](https://github.com/NVIDIA/k8s-device-plugin/blob/main/README.md)
`kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.0/deployments/static/nvidia-device-plugin.yml`

TODO: The number of workers depends on the number of GPUs on the node. Later, this can be configured more flexibly with Kustomize (i.e., override the default number of replicas for the `base` environment `Deployment` object with a patch in the `internal` environment).
