# Gateway Horizontal Scaling

The Cornserve Gateway is designed to scale horizontally to handle increased application registration and invocation traffic. Scaling out the gateway means running multiple replicas of the gateway service behind a load balancer, with shared state managed through Kubernetes Custom Resource Definitions (CRDs).

## Configuring Replicas

Horizontal scaling is controlled by the `replicas` field in the gateway deployment configuration. By default, Cornserve runs with a single gateway replica.

To scale out, update the `replicas` count in `kubernetes/kustomize/cornserve/base/gateway/deployment.yaml`.

After updating the configuration, apply it to the cluster using `kubectl apply -k kubernetes/kustomize/cornserve/base/gateway/`.

## Load Balancing

The gateway is exposed via a standard Kubernetes Service (`ClusterIP`). This service does not use `sessionAffinity` (it defaults to `None`), meaning that `kube-proxy` distributes incoming HTTP and WebSocket requests across all available gateway pods using a round-robin strategy.

Because all critical control-plane state is distributed via CRDs, the system does not require sticky sessions for application registration or invocation. A client can register an application through one gateway replica and immediately invoke it through another. The development overlay's `NodePort` service behaves in the same manner.

## Distributed State via CRDs

The gateway uses the Kubernetes API server as its shared source of truth. The CRDs involved fall into two categories:

### Pre-existing CRDs (unchanged)

These CRDs existed before the scale-out work and are used as-is:

- **AppInstance CRD** (`appinstances.cornserve.ai`): Stores the authoritative definition for each application, including its unique ID, Python source code, required task keys, streaming flags, and lifecycle state (`NOT_READY` or `READY`).
- **LatestAppRV CRD** (`latestapprvs.cornserve.ai`): A singleton resource that tracks the latest resource version (RV) of app-related changes. It acts as a synchronization barrier to ensure all replicas have observed a specific update.

### Extended CRD

- **UnitTaskInstance CRD** (`unittaskinstances.cornserve.ai`): This CRD was pre-existing and manages the lifecycle of unit tasks (storing `definitionRef`, `config`, `executionDescriptorName`). For scale-out, we **extended** its schema with a `usageRefcount` integer field. This field enables cross-gateway task deduplication and shared reference counting — multiple gateways can share the same physical unit task deployment, and the task is only torn down when the last reference is released.

## AppRegistry Watcher

Each gateway replica runs an `AppRegistry` background watcher that uses the Kubernetes "list then watch" pattern. The watcher maintains a local in-memory cache of `AppInstance` resources.

On `ADDED` or `MODIFIED` events from the Kubernetes API server, the watcher updates its local cache. When an application is requested on a gateway that did not handle the original registration, the gateway uses this cache to discover the app's definition and materialize its execution logic.

## Cross-Gateway Task Refcounting

The `usageRefcount` field on `UnitTaskInstance` CRs is the core mechanism for safe task lifecycle management across gateway replicas.

### Registration (`declare_used`)

When a gateway registers an app and needs to deploy unit tasks:

1. The gateway computes a canonical task key and acquires a Kubernetes `Lease` (`coordination.k8s.io/v1`) for that key.
2. While holding the lease, it searches existing `UnitTaskInstance` CRs for an equivalent configuration (matching `definitionRef`, `config`, and `executionDescriptorName`).
3. **If a matching CR is found**: The task is already deployed in the cluster by another gateway. The gateway increments the CR's `usageRefcount` and skips the gRPC deployment call to the Resource Manager.
4. **If no matching CR is found**: The gateway creates a new CR with `usageRefcount=1` and issues a gRPC `DeployUnitTask` call to the Resource Manager.
5. The lease is released after the critical section completes.

### Unregistration (`declare_not_used`)

When a gateway unregisters an app and releases unit tasks:

1. The gateway decrements its local usage counter. If it reaches 0, the gateway proceeds to check the CR.
2. The gateway acquires the same per-task Kubernetes `Lease` used by registration.
3. While holding the lease, it decrements the CR's `usageRefcount`.
4. **If the CR refcount reaches 0**: The gateway performs gRPC `TeardownUnitTask` and CR deletion while still holding the lease.
5. **If the CR refcount is still > 0**: The gateway releases only local state; the shared deployment remains active.

### Concurrency Notes

Lease-based serialization removes the lost-update race for concurrent register/unregister operations on the same logical task by enforcing a single writer per task key.

The lock is time-bounded (`leaseDurationSeconds`), so very long critical sections can still require careful tuning of lease duration and retry behavior.

### Invocation on Non-Registering Gateways

When an invocation request lands on a gateway that did not handle the original registration, its local `TaskManager` state may be empty. Before matching tasks, the gateway calls `_ensure_tasks_from_crs()` which lists all `UnitTaskInstance` CRs and reconstructs `UnitTask` objects via the `TaskRegistry`. This populates local state on-demand so the invocation can proceed normally.

## What Stays Local

While control-plane metadata is shared, certain state remains local to each gateway process:

- **WebSocket sessions** (`SessionManager`): Active sessions are bound to specific TCP connections and cannot be migrated between replicas.
- **In-process module caches**: Each gateway replica independently executes (`exec()`) the application source code from the CRD to materialize Python `ModuleType` objects.
- **Active streams**: HTTP streaming responses and SSE connections are tied to the replica handling the request.
- **Transport clients**: Local `aiohttp` sessions, gRPC channels, and OpenTelemetry exporters are per-process.

## Registration Flow

When a client registers an application, the following distributed flow occurs:

1. A client sends a `POST /app/register` request, which is routed to any gateway replica.
2. Validation of the source code happens first, followed by the creation of an `AppInstance` CR with `state=NOT_READY`.
3. Next, the gateway requests task deployment through the `TaskManager`. For each task, the CR-based refcount mechanism is used (see above) to either reuse an existing deployment or create a new one.
4. Upon successful deployment, the `AppInstance` CR state is updated to `READY` and the `LatestAppRV` singleton is bumped.
5. A call to `sync_watchers()` then blocks until the local watcher has caught up to the new resource version.
6. Finally, the gateway returns success to the client. The application is now ready to be invoked from any replica in the cluster.

## Invocation Flow

When an invocation request hits a gateway replica:

1. The gateway checks its local cache for the application's materialized module.
2. If the module is not found, a lookup occurs in the `AppRegistry` cache (populated by the CRD watcher).
3. When an app is found and its state is `READY`, the gateway materializes the `ModuleType` from the stored source code and caches it locally.
4. If the local `TaskManager` has no tasks, it synchronizes from `UnitTaskInstance` CRs to populate task state.
5. Normal invocation logic then proceeds as usual.
