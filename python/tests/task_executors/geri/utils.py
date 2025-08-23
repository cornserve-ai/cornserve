"""Testing utilities for Geri."""

import torch


def create_dummy_embeddings(
    batch_size: int = 1, seq_len: int = 77, hidden_size: int = 3584, dtype: torch.dtype = torch.bfloat16
) -> list[torch.Tensor]:
    """Create dummy prompt embeddings for testing.

    Args:
        batch_size: Number of prompts in the batch.
        seq_len: Sequence length of embeddings.
        hidden_size: Hidden dimension size.
        dtype: Data type for embeddings.

    Returns:
        List of dummy embedding tensors, one per batch item.
    """
    return [torch.randn(seq_len, hidden_size, dtype=dtype, device=torch.device("cuda")) for _ in range(batch_size)]


def assert_valid_png_bytes_list(png_bytes_list: list[bytes], expected_batch_size: int = 1) -> None:
    """Assert that the generated PNG bytes list is valid.

    Args:
        png_bytes_list: List of PNG-encoded image bytes.
        expected_batch_size: Expected batch size.
    """
    assert isinstance(png_bytes_list, list), f"Expected list, got {type(png_bytes_list)}"
    assert len(png_bytes_list) == expected_batch_size, (
        f"Expected batch size {expected_batch_size}, got {len(png_bytes_list)}"
    )

    for i, png_bytes in enumerate(png_bytes_list):
        assert isinstance(png_bytes, bytes), f"Item {i} should be bytes, got {type(png_bytes)}"
        assert len(png_bytes) > 0, f"Item {i} should not be empty"
        # Check PNG header
        assert png_bytes.startswith(b"\x89PNG"), f"Item {i} should start with PNG header"
