import hashlib
import torch


def tensor_hash(t: torch.Tensor) -> str:
    """Compute a hash of a tensor, used to check if sent and received tensors match."""
    t_cpu = t.cpu().contiguous()
    if t_cpu.dtype == torch.bfloat16:
        t_cpu = t_cpu.to(torch.float32)
    tensor_bytes = t_cpu.numpy().tobytes()
    return hashlib.sha256(tensor_bytes).hexdigest()
