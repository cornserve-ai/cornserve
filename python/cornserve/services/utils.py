import hashlib
import torch

def tensor_hash(t: torch.Tensor) -> str:
    """Compute a hash of a tensor, used to check if sent and received tensors match."""
    t_cpu = t.cpu().contiguous()
    if t_cpu.dtype == torch.bfloat16:
        t_cpu = t_cpu.to(torch.float32)
    tensor_bytes = t_cpu.numpy().tobytes()
    return hashlib.sha256(tensor_bytes).hexdigest()


def get_tensor_size(t: torch.Tensor) -> int:
    """Compute the size of a tensor from its shape."""
    return t.numel() * t.element_size()


def get_dtype_size(dtype: torch.dtype) -> int:
    """Get the element size of a tensor dtype."""
    return torch.empty(0, dtype=dtype).element_size()
