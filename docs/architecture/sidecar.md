# Sidecar: P2P Communication Library

Sidecar is the P2P communication library in Cornserve that allows task executors
to send/receive intermediate data to/from each other. It's mainly designed for
tensor, but it also supports any other types.

Code lives under `python/services/sidecar`

## Architecture
Sidecars have servers and clients. Servers are long running services in the
cluster, created upon cluster deployment. Clients live within Task Executors,
and task executors invoke clients for send and receive operations which are then
fulfilled by the servers.


### Servers
Each GPU in the cluster is usually paired with one dedicated Sidecar server, but
Sidecar servers can also act as a group when, for example, a task executor runs
a model with tensor parallel.

All Sidecar servers and clients send control signals through gRPC, while Sidecar
servers use `UCX` as the backend for tensor transfer, which uses RDMA if possible.
Small objects are directly sent over through gRPC to reduce contention.

#### Forwarding Tensors
The tensor transfer among Sidecar servers do not use NVLink, as they are preserved
for maximizing throughput when running models with tensor parallel. Throughout a
tensor forward from a producer with a Sidecar sender server to a consumer with a
Sidecar receiver server, the Sidecar sender will copy the tensor from the producer's
GPU to CPU, and the tensor will arrive at the receiver server's CPU. Therefore,
the consumer has the responsibility for copying the received tensor to its devices.
If the producer and the consumer locates within the same node, there will be no
addition transfer over the network.

When multiple Sidecars are grouped, Sidecars assume each producer in the group
holds a full replica of the tensor to forward, and the Sidecar Servers could
choose to either use one single GPU or use every GPU in the group when copying
-- adjusted through a configuration knob.

#### Chunking
Producers are free to chunk the forwarding tensors in any way. However, it's not
recommended to have chunks with non-contiguous memory view due overhead.
Sidecars view each chunk as independent, so there is no guarantee that all the
chunks will be in order or are placed a slab of contiguous memory. Consumers can
decide to process chunks in order, or decide to process all chunks together if
the consumer cannot utilize chunks independently.


#### Memory Management
Sidecar servers manage CPU memory for placing the tensors to send and receive. To
reduce internal fragmentation, sidecar clients, thus task executors, are currently
required to provide memory hint for the servers. The memory hint is conceptually
the memory allocation unit size for the servers, and typically this could be the 
hidden size of a model the executor is running.


### Clients
Clients are the front ends for task executors to interact with servers.

Task executors can define a `SidecarConfig` from `python/sidecar/schema.py` and 
then instantiate a `Sidecar` client from `python/sidecar/api.py`. The client will
setup the Sidecar server for the task executor's use upon creation. The client
mainly provides three sets of APIs, namely, `send`, `recv`, and `mark_done`.

#### `send`
`send` can be used to broadcast some data to a list of Sidecar groups. When chunking
is involved, the producer need to fill in the `chunk_id` and `num_chunks` parameters.

#### `recv`
`recv` can be used to receive data at chunk-granularity, where `chunk_id` can be
specified. The returning data is either a tensor with CPU storage or a small python
object. Receive operations are idempotent for Sidecars, so multiple consumer processes
can consume the data concurrently. There is also a synchronous version called `recv_sync`.

#### `mark_done`
`mark_done` is used to free the backing memory of a received tensor in the Sidecar
server. As the Sidecar server allows for idempotent receive operations, the data
is always held within the server until a corresponding `mark_done` called.

See `python/sidecar/api.py` for more details.
