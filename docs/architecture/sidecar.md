# Sidecar

_NOTE: Naming is temporary_

Sidecar is a P2P tensor communication library that launches long living processes
to send/receive tensors to/from each other.

Code lives under `python/services/sidecar`

## Motivation
The distributed processing group in PyTorch is rigid and it's costly to
dynamically scale the processing group or adjust the role of each distributed
process. The Cornserve project requires auto-scaling of distributed
task executors. To support intermediate data (tensor) transfer adaptively, we
decouple communication from task executors to dedicated `sidecars` that can be
long-living within the cluster.

## Architecture
Sidecars are implemented with Servers and Clients. Conceptually, servers are long
running, and clients register to servers and request servers to perform send or
receive operations. All control signals among servers and clients use gRPC, and
the tensor transfer is implemented using `gloo`. Servers and clients use shared 
memory buffer to reduce memory copies. A sender client puts some data in the
shared memory buffer and provide a handle to the sender server, the sender server
then transmits the data to the receiver server. Upon the receiver client calling
receive on that data, a shared memory handle used by the receiver server will be
returned.

### Servers
Servers are the long-running processes within the cluster, and each GPU is
expected to be paired with at least one server (duplicate servers for
fault-tolerance, work in the future). The role of a server is unique but also
dynamic. When a client registers as a sender to a server, the requested server
becomes a Sender Server, and vice versa.

Classes within `python/services/sidecar/server.py` are servers.

#### `CommSidecarSender`
An intermediate data can be chunked, and each chunk could be further sharded
when there are multiple senders holding the same chunk. This reduces the
communication volume when a producer task executor has TP size > 1.

#### `CommSidecarReceiver`
The receiver server manages the shared memory buffer. A `recv` call will only
return when the all chunks of a data has been received (in near future, this 
will become an async generator that returns chunks in order). When the consumer
no longer needs the data, a corresponding `mark_done` must be called to free the
buffer used by that data.

### Clients
Clients are the front ends for task executors to interact with servers.

Classes within `python/services/sidecar/api.py` are clients.

#### `TensorSidecarSender`
All methods are blocking. It has a shared memory manager. The underlying send
uses a thread pool to be non-blocking.

#### `TensorSidecarAsyncReceiver`
All methods are asynchronous. It allows for grouping, where a consumer task
executor may use multiple sidecar receiver servers. In that case, all send
communication will go through the sidecar server with the lowest rank.

#### `TensorSidecarReceiverExecutor`
All methods are synchronous. This class acts as a reader to the received data.
The `recv` will return the requested data, but it should be called after the
`TensorSidecarAsyncReceiver` has confirmed the completion of data transfer.


