"""Testing utilities for Geri."""

import base64

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


def assert_valid_png_results_list(results: list[str], expected_batch_size: int = 1) -> None:
    """Assert that the generated PNG results list is valid."""
    assert isinstance(results, list), f"Expected list, got {type(results)}"
    assert len(results) == expected_batch_size, f"Expected batch size {expected_batch_size}, got {len(results)}"

    for i, png_str in enumerate(results):
        assert isinstance(png_str, str), f"Item {i} should be str, got {type(png_str)}"
        png_bytes = base64.b64decode(png_str.encode("ascii"))
        assert len(png_bytes) > 0, f"Item {i} should not be empty"
        # Check PNG header
        assert png_bytes.startswith(b"\x89PNG"), f"Item {i} should start with PNG header"
