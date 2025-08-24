import json
import os
import uuid
from contextlib import contextmanager

import torch

from cornserve.sidecar.api import Sidecar
from cornserve.sidecar.schema import SidecarConfig

os.environ["CORNSERVE_MOCK_SIDECAR"] = "1"


@contextmanager
def mock_sidecar_files(mock_mapping):
    """Context manager that sets up mock mapping and cleans up files afterwards."""
    os.environ["CORNSERVE_MOCK_SIDECAR_MAPPING"] = json.dumps(mock_mapping)
    try:
        yield
    finally:
        # Clean up files
        for file_path in mock_mapping.values():
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except OSError:
                pass


def test_mock_send_recv():
    """Tests sending and receiving data using mock sidecar."""
    tensor_id = uuid.uuid4().hex
    chunked_tensor_id = uuid.uuid4().hex
    dict_data_id = uuid.uuid4().hex
    int_data_id = uuid.uuid4().hex
    mock_mapping = {
        f"{tensor_id}-0": "tmp_tensor.pt",
        f"{chunked_tensor_id}-0": "tmp_tensor-chunk0.pt",
        f"{chunked_tensor_id}-1": "tmp_tensor-chunk1.pt",
        f"{dict_data_id}-0": "tmp_dict.json",
        f"{int_data_id}-0": "tmp_int.json",
    }
    with mock_sidecar_files(mock_mapping):
        config = SidecarConfig(
            sidecar_rank=0,
            send_tensor_shape=(-1, 1),
            send_tensor_dtype=torch.bfloat16,
        )
        sender = Sidecar(config=config)
        receiver = Sidecar(config=config)

        torch.manual_seed(0)
        tensor_data = torch.randn(3, 5, 8, device="cuda")
        tensor_chunk0 = torch.randn(3, 5, 4, device="cuda")
        tensor_chunk1 = torch.randn(3, 5, 4, device="cuda")
        dict_data = {"key1": "value1", "key2": 123, "key3": [1, 2, 3]}
        int_data = 42
        sender.send(data=tensor_data, id=tensor_id, dst_sidecar_ranks=[[]])
        sender.send(data=tensor_chunk0, id=chunked_tensor_id, dst_sidecar_ranks=[[]])
        sender.send(data=tensor_chunk1, id=chunked_tensor_id, chunk_id=1, dst_sidecar_ranks=[[]])
        sender.send(data=dict_data, id=dict_data_id, dst_sidecar_ranks=[[]])
        sender.send(data=int_data, id=int_data_id, dst_sidecar_ranks=[[]])

        recv_tensor = receiver.recv_sync(id=tensor_id)
        recv_tensor_chunk0 = receiver.recv_sync(id=chunked_tensor_id, chunk_id=0)
        recv_tensor_chunk1 = receiver.recv_sync(id=chunked_tensor_id, chunk_id=1)
        recv_dict = receiver.recv_sync(id=dict_data_id)
        recv_str = receiver.recv_sync(id=int_data_id)

        assert torch.allclose(tensor_data, recv_tensor.to(tensor_data.device))
        assert torch.allclose(tensor_chunk0, recv_tensor_chunk0.to(tensor_chunk0.device))
        assert torch.allclose(tensor_chunk1, recv_tensor_chunk1.to(tensor_chunk1.device))
        assert dict_data == recv_dict
        assert int_data == recv_str
