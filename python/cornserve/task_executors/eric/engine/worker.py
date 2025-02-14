import torch
from transformers import AutoModel, AutoConfig

from cornserve.task_executors.eric.models import Modality


class Worker:
    """Runs model inference."""

    def __init__(
        self,
        model_id: str,
        modality: Modality,
        tp_rank: int,
        tp_degree: int,
    ) -> None:
        """Initialize the worker.

        1. Initialize distributed process group
        """
        self.device = torch.device("cuda:0")
        config = AutoConfig.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, config=config).to(self.device)
        self.model.eval()

    def execute_model(self, items):
        # "tensors" is a Python list or nested list from JSON, so convert to torch.Tensor
        # each item: { "request_id", "modality", "tensors": nested list }
        # We'll accumulate them, shape them, move to GPU, run forward pass
        # Return one output embedding per item

        # Convert each item’s "tensors" back to a torch.Tensor
        # They were originally [batch_size, 3, H, W] or similar
        # Here we do something simple: just cat them
        batch_list = []
        batch_sizes = []
        for it in items:
            arr = torch.tensor(it["tensors"], dtype=torch.float32)
            batch_list.append(arr)
            batch_sizes.append(arr.shape[0])

        batched = torch.cat(batch_list, dim=0).to(self.device)
        with torch.no_grad():
            out = self.model(batched)
            # assume out.last_hidden_state is [total_batch, seq_len, hidden_dim]
            # do a mean pooling
            emb = out.last_hidden_state.mean(dim=1)  # [total_batch, hidden_dim]

        results = []
        idx = 0
        for size in batch_sizes:
            # slice out the portion for this request
            sub_emb = emb[idx: idx + size]
            idx += size
            # maybe store final embedding as CPU tensor
            results.append(sub_emb.cpu())
        return results
