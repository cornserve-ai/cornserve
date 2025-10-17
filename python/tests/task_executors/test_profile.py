from __future__ import annotations

from pathlib import Path

from cornserve_tasklib.task.unit.llm import LLMUnitTask
from cornserve.task_executors.profile import ProfileInfo, UnitTaskProfile, UnitTaskProfileManager

GEMMA_3_27B_PROFILE_JSON = """
{
  "task": {
    "__class__": "LLMUnitTask",
    "model_id": "google/gemma-3-27b-it",
    "receive_embeddings": true,
    "execution_descriptor_name": null
  },
  "num_gpus_to_profile": {
    "2": {},
    "4": {"launch_args": ["--max-num-seqs", "768"]}
  }
}
"""


def test_example_profile(tmp_path: Path) -> None:
    """Test whether the example Gemma 3 27B profile works as expected."""
    tempfile = tmp_path / "gemma_3_27b_profile.json"
    tempfile.write_text(GEMMA_3_27B_PROFILE_JSON)

    unit_task = LLMUnitTask(model_id="google/gemma-3-27b-it")

    profile = UnitTaskProfile.from_json_file(tempfile)

    assert profile.task.is_equivalent_to(unit_task)
    assert profile.num_gpus_to_profile == {2: ProfileInfo(), 4: ProfileInfo(launch_args=["--max-num-seqs", "768"])}

    manager = UnitTaskProfileManager(profile_dir=tmp_path)
    assert manager.get_profile(task=unit_task).num_gpus_to_profile == {
        2: ProfileInfo(),
        4: ProfileInfo(launch_args=["--max-num-seqs", "768"]),
    }

    assert not profile.num_gpus_to_profile[2].launch_args
    assert profile.num_gpus_to_profile[4].launch_args == ["--max-num-seqs", "768"]
