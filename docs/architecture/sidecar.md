# Sidecar

_NOTE: Naming is temporary_

Sidecar is the P2P communication library that allows task executors
to send/receive intermediate data to/from each other. The primary data type is
tensors, but it also supports any other types.

Code lives under `python/services/sidecar`

## Architecture
Sidecars arewimplemented with Servers and Clients. Conceptually, sidecar servers
are long running services inside the cluster, and each task executor should
create sidecar clients that register to servers and request servers to perform
send or receive operations. All control signals among servers and clients use
gRPC, and tensor transfer is implemented using `ucx-py`, which uses RDMA if available.
Sidecars expect producers to provide GPU tensors, and return CPU tensors to 
consumers. Servers and clients use shared memory buffer to reduce memory copies.
When the client receives a GPU tensor from its parent task executor, it uses CUDA
IPC to share the memory handle with the server, the server will then launches CUDA
copy from GPU memory to a dedicated shared memory buffer, then forward the tensor
to the destination sidecar. If the destination is in a different node, the tensor
will be transferred the dedicated buffer on that node, then later transferred to
the consumer's GPU if the consumer wishes. If the destination is within, the
same node, the consumer is directly passed with a tensor backed by the same
buffer without additional copy. If the intermediate data is not a tensor, the data
will be serialized to bytes and transferred through gRPC, such data is expected
to be very small.

### Servers
Servers are the long-running services within the cluster, and each GPU is
expected to be paired with at least one server (duplicate servers for
fault-tolerance, work in the future). A server and perform send and receive
operations at the same time. Currently, the server has centralized memory
management for the shared memory file. To reduce fragmentation, servers
require clients (task executors) to provide memory hint on the unit of
transferring tensor.

See `python/services/sidecar` for implementation details.

#### Chunking
One important need of sidecar is chunking. Task executors are free to chunk the
tensor and assemble the chunks in any way, but sidecars require parameters of 
`chunk_id` and `num_chunks` if the task executors wish to do so. Sidecars view 
each chunk as independent, so there is no guarantee that all the chunks will be
contiguous in CPU memory.


### Clients
Clients are the front ends for task executors to interact with servers.

See `python/sidecar/api.py` for more details.
