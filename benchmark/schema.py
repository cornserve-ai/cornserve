"""Schema definitions for benchmark configurations and data handling."""

from __future__ import annotations

import datetime
import json
from abc import abstractmethod
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, SerializeAsAny

DATA_ROOT = "data"


class BackendConfig(BaseModel):
    """Base class for backend configurations."""

    @abstractmethod
    def to_subdir_name(self) -> str:
        """Return the file name for the backend configuration."""
        pass


class EricConfig(BackendConfig):
    """Eric only backedend configuration."""

    num_replicas: int
    tp_size: int = 1
    max_batch_size: int = 1
    modality: Literal["image", "video", "audio"] = "image"

    def to_subdir_name(self) -> str:
        """Return the file name for the backend configuration."""
        subdir_name = f"eric+replicas{self.num_replicas}+tp{self.tp_size}+maxbs{self.max_batch_size}"
        if self.modality != "image":
            subdir_name += f"+{self.modality}"
        return subdir_name

class CornserveOmniConfig(BackendConfig):
    """Configuration for the  backend with a specific number of erics and vLLMs."""
    num_image_erics: int = 0
    num_video_erics: int = 0
    num_audio_erics: int = 0
    num_thinkers: int
    num_talker_vocoders: int
    # tp not supported bc only 7B

    modalities: list[Literal["image", "video", "audio"]] = ["image", "video", "audio"]

    def to_subdir_name(self) -> str:
        """Return the subdirectory name for the Cornserve configuration."""
        suffix = f"cornserve_omni+image_erics{self.num_image_erics}+video_erics{self.num_video_erics}+audio_erics{self.num_audio_erics}" + \
                f"+thinkers{self.num_thinkers}+talker_vocoders{self.num_talker_vocoders}"
        return suffix

class VLLMOmniConfig(BackendConfig):
    """Configuration for the  backend with a specific number of erics and vLLMs."""
    num_thinkers: int
    num_talker_vocoders: int
    # tp not supported bc only 7B

    def to_subdir_name(self) -> str:
        """Return the subdirectory name for the Cornserve configuration."""
        suffix = f"vllm_omni+thinkers{self.num_thinkers}+talker_vocoders{self.num_talker_vocoders}"
        return suffix

class CornserveConfig(BackendConfig):
    """Configuration for the Cornserve backend with a specific number of erics and vLLMs."""

    num_erics: int
    eric_tp_size: int = 1
    num_vllms: int
    vllm_tp_size: int = 1

    modalities: list[Literal["image", "video", "audio"]] = ["image"]
    # only used when using omni dataset
    num_image_erics: int = 0
    num_video_erics: int = 0
    num_audio_erics: int = 0

    def model_post_init(self, __context: Any) -> None:
        if set(self.modalities) == {"image"} or self.num_image_erics + self.num_video_erics + self.num_audio_erics == 0:
            self.num_image_erics = self.num_erics

    def to_subdir_name(self) -> str:
        """Return the subdirectory name for the Cornserve configuration."""
        suffix = f"cornserve+erics{self.num_erics}+tp{self.eric_tp_size}" + f"+vllms{self.num_vllms}+tp{self.vllm_tp_size}"
        if set(self.modalities) != {"image"}:
            suffix += f"+modalities{'-'.join(sorted(self.modalities))}"
            suffix += f"+image_erics{self.num_image_erics}+video_erics{self.num_video_erics}+audio_erics{self.num_audio_erics}"
        return suffix

    @classmethod
    def create_backend_configs(
        cls,
        num_gpus: int,
    ) -> list[CornserveConfig]:
        """Create a list of CornserveConfig instances based on the number of GPUs."""
        configs = []
        for num_eric in range(1, num_gpus):
            num_vllm = num_gpus - num_eric
            configs.append(cls(num_erics=num_eric, num_vllms=num_vllm))
        return configs


class VLLMConfig(BackendConfig):
    """Configuration for the vLLM backend with a specific number of replicas."""

    num_replicas: int = Field(
        default=8,
        description="Number of vLLM replicas to use for the benchmark.",
    )
    tp_size: int = 1

    def to_subdir_name(self) -> str:
        """Return the subdirectory name for the vLLM configuration."""
        return f"vllm+replicas{self.num_replicas}+tp{self.tp_size}"


class PDConfig(BackendConfig):
    """Configuration for the Cornserve backend with a specific number of erics and vLLMs."""

    num_prefills: int
    prefill_tp_size: int = 1
    num_decodes: int
    decode_tp_size: int = 1

    def to_subdir_name(self) -> str:
        """Return the subdirectory name for the Cornserve configuration."""
        return (
            f"pd+prefills{self.num_prefills}+tp{self.prefill_tp_size}"
            + f"+decodes{self.num_decodes}+tp{self.decode_tp_size}"
        )

    @classmethod
    def create_backend_configs(
        cls,
        num_gpus: int,
    ) -> list[CornserveConfig]:
        """Create a list of CornserveConfig instances based on the number of GPUs."""
        configs = []
        for num_prefills in range(1, num_gpus):
            num_decodes = num_gpus - num_prefills
            configs.append(cls(num_prefills=num_prefills, num_decodes=num_decodes))
        print(f"Created {len(configs)} PD configurations for {num_gpus} GPUs.")
        return configs

class NcclPDConfig(BackendConfig):
    """Configuration for the Cornserve backend with a specific number of erics and vLLMs."""

    num_prefills: int
    prefill_tp_size: int = 1
    num_decodes: int
    decode_tp_size: int = 1

    def to_subdir_name(self) -> str:
        """Return the subdirectory name for the Cornserve configuration."""
        return (
            f"ncclpd+prefills{self.num_prefills}+tp{self.prefill_tp_size}"
            + f"+decodes{self.num_decodes}+tp{self.decode_tp_size}"
        )

    @classmethod
    def create_backend_configs(
        cls,
        num_gpus: int,
    ) -> list[CornserveConfig]:
        """Create a list of CornserveConfig instances based on the number of GPUs."""
        configs = []
        for num_prefills in range(1, num_gpus):
            num_decodes = num_gpus - num_prefills
            configs.append(cls(num_prefills=num_prefills, num_decodes=num_decodes))
        print(f"Created {len(configs)} PD configurations for {num_gpus} GPUs.")
        return configs


class EPDConfig(BackendConfig):
    """Configuration for the Cornserve backend with a specific number of erics and vLLMs."""

    num_erics: int
    eric_tp_size: int = 1
    num_prefills: int
    prefill_tp_size: int = 1
    num_decodes: int
    decode_tp_size: int = 1

    def to_subdir_name(self) -> str:
        """Return the subdirectory name for the Cornserve configuration."""
        return (
            f"epd+erics{self.num_erics}+tp{self.eric_tp_size}"
            + f"prefills{self.num_prefills}+tp{self.prefill_tp_size}"
            + f"+decodes{self.num_decodes}+tp{self.decode_tp_size}"
        )

    @classmethod
    def create_backend_configs(
        cls,
        num_gpus: int,
    ) -> list[EPDConfig]:
        """Create a list of CornserveConfig instances based on the number of GPUs."""
        configs = []
        for num_erics in range(1, num_gpus - 1):
            for num_prefills in range(1, num_gpus - num_erics):
                num_decodes = num_gpus - num_erics - num_prefills
                configs.append(
                    cls(
                        num_erics=num_erics,
                        num_prefills=num_prefills,
                        num_decodes=num_decodes,
                    )
                )
        print(f"Created {len(configs)} EPD configurations for {num_gpus} GPUs.")
        return configs

class WorkloadConfig(BaseModel):
    """Base class for workload configurations."""

    @abstractmethod
    def to_suffix(self) -> str:
        """Return the suffix for the workload configuration."""
        pass

class DutyCycleConfig(WorkloadConfig):
    """Configuration for the duty cycle workload."""

    request_rate: float
    on_request_factor: float = 2.0
    off_request_factor: float = 0.5
    cycles: int = 20

    def to_suffix(self) -> str:
        """Return the suffix for the duty cycle workload configuration."""
        return (
            f"duty_cycle+rate{self.request_rate}+on_factor{self.on_request_factor}"
            f"+off_factor{self.off_request_factor}+cycles{self.cycles}"
        )

class ServeGenConfig(WorkloadConfig):
    """Configuration for the ServeGen workload."""

    request_rate: float
    duration: int

    no_image_prob: float = 0.0
    audio_prob: float = 0.0
    video_prob: float = 0.0

    def to_suffix(self) -> str:
        basename = f"servegen+rate{self.request_rate}+duration{self.duration}"
        if self.no_image_prob + self.audio_prob + self.video_prob > 0:
            # non default
            basename += f"+no_image_prob{self.no_image_prob}+audio_prob{self.audio_prob}+video_prob{self.video_prob}"
        return basename

class OmniConfig(WorkloadConfig):
    """Configuration for the Omni workload."""

    include_image: bool = True
    image_width: int = 1920
    image_height: int = 1080
    include_video: bool = True
    video_num_frames: int = 20
    video_width: int = 1024
    video_height: int = 768
    # 9990 image tokens for Qwen
    include_audio: bool = True
    audio_duration_sec: int = 8
    return_audio_prob: float = 0.0
    return_audio_tokens: int = 0

    def to_suffix(self) -> str:
        suffix = "omni"
        if self.include_image:
            suffix += f"+image_w{self.image_width}_h{self.image_height}"
        if self.include_video:
            suffix += f"+video_f{self.video_num_frames}_w{self.video_width}_h{self.video_height}"
        if self.include_audio:
            suffix += f"+audio_dur{self.audio_duration_sec}"
        if self.return_audio_prob > 0.0:
            # assert self.return_audio_tokens > 0, "return_audio_tokens must be > 0 if return_audio_prob > 0"
            # suffix += f"+return_audio_prob{self.return_audio_prob}+return_audio_tokens{self.return_audio_tokens}"
            suffix += f"+return_audio_prob{self.return_audio_prob}"
        return suffix



class ExperimentConfig(BaseModel):
    """Configuration for the a benchmark experiment."""

    # Backend config
    backend_config: SerializeAsAny[BackendConfig]
    app_id: str = Field(exclude=True)

    # general config
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    seed: int = Field(
        default=48105,
        description=(
            "Random seed for reproducibility. The same seed is used for sampling requests from"
            "the given dataset; synthesizing multimodal data; synthesizing requests; and generating"
            "request arrival distribution."
        ),
    )
    gpu_type: Literal["A40", "A100", "H100"] = "A40"

    # workload config
    dataset: Literal["lmarena-ai/VisionArena-Chat", "servegen", "omni"] = Field(
        default="lmarena-ai/VisionArena-Chat",
        description=("Dataset to use for the benchmark. Currently only supports lmarena-ai/VisionArena-Chat."),
    )
    num_prompts: int = 2000
    num_warmups: int = 50
    input_len: int = Field(
        default=500,
        description=(
            "Number of input text tokens in each request."
            # TODO: support variable input lengths in the future and original input len from dataset
        ),
    )
    output_len: int = Field(
        default=500,
        description=(
            "Number of output tokens to generate for each request."
            # TODO: support variable output lengths in the future and original output len from dataset
        ),
    )
    request_rate: float = Field(
        default=10.0,
        description="Request rate in requests per second for the benchmark.",
    )
    burstiness: float = Field(
        default=1,
        description="Burstiness factor for request arrival distribution.",
    )
    max_concurrency: int | None = Field(
        default=None,
        description="Maximum number of concurrent requests to send in the benchmark.",
    )
    use_synthesized_data: bool = True
    # modality data config
    image_probability: float = 1.0
    image_width: int = 1920
    image_height: int = 1080
    image_count: int = Field(
        default=1,
        description="Number of images to include in each request.",
    )
    image_choices: int = Field(
        default=10,
        description="Number of image choices to synthesize and choose from for each request.",
    )
    encoder_fission_probability: float = Field(
        default=1.0,
        description="Probability of using independent encoder during benchmark.",
    )
    # this probability is used after encoder_fission_probability
    # so to independently control each modality, set encoder_fission_probability to 1.0
    image_fission_probability: float = Field(
        default=1.0,
        description="Probability of using independent image encoder during benchmark.",
    )
    video_fission_probability: float = Field(
        default=1.0,
        description="Probability of using independent video encoder during benchmark.",
    )
    audio_fission_probability: float = Field(
        default=1.0,
        description="Probability of using independent audio encoder during benchmark.",
    )
    encoder_fission_block_size: int = Field(
        default=16,
        description="Interval based fission block size"
    )

    workload_config: SerializeAsAny[WorkloadConfig] | None = None

    def model_post_init(self, __context: Any) -> None:
        if self.workload_config is not None and isinstance(self.workload_config, ServeGenConfig):
            assert self.dataset == "servegen", "ServeGenConfig can only be used with the servegen dataset."
            assert self.use_synthesized_data == False, "ServeGen dataset already contains mm data."
            self.request_rate = self.workload_config.request_rate
        if self.workload_config is not None and isinstance(self.workload_config, OmniConfig):
            assert self.dataset == "omni", "OmniConfig can only be used with the omni dataset."
            assert self.use_synthesized_data == False, "Omni dataset already contains mm data."

    def _get_image_config_str(self) -> str:
        """Get the image configuration as a string."""
        return (
            f"image+prob{self.image_probability}+w{self.image_width}+h{self.image_height}+"
            f"count{self.image_count}+choices{self.image_choices}"
        )

    def _to_filename(self) -> str:
        """Convert the config to a filename."""
        filename = (
            f"{self.gpu_type}+dataset{self.dataset.replace('/', '_')}+warmups{self.num_warmups}+"
            f"prompts{self.num_prompts}+input{self.input_len}+output{self.output_len}+"
            f"rate{self.request_rate}+burst{self.burstiness}+seed{self.seed}+"
            f"synthesized{self.use_synthesized_data}"
        )
        if self.use_synthesized_data:
            filename += f"+{self._get_image_config_str()}"
        filename += f"+fission{self.encoder_fission_probability}"
        if self.encoder_fission_probability > 0.0:
            filename += f"+fission_block{self.encoder_fission_block_size}"
        if self.image_fission_probability < 1.0:
            filename += f"+image_fission{self.image_fission_probability}"
        if self.video_fission_probability < 1.0:
            filename += f"+video_fission{self.video_fission_probability}"
        if self.audio_fission_probability < 1.0:
            filename += f"+audio_fission{self.audio_fission_probability}"
        if self.workload_config is not None:
            filename += f"+{self.workload_config.to_suffix()}"

        return filename + ".json"

    def to_path(self) -> Path:
        """Return the path to the config file."""
        return Path(DATA_ROOT) / self.backend_config.to_subdir_name() / self.model_id / self._to_filename()

    def exists(self) -> bool:
        """Check if the config file exists."""
        return self.to_path().exists()

    def save(self, data: dict[str, Any]) -> None:
        """Save the benchmark data to a file."""
        config = self.model_dump()
        config["timestamp"] = datetime.datetime.now().isoformat()
        data["config"] = config
        path = self.to_path()
        if path.exists():
            print(f"Warning: Data file {path} already exists. Overwriting it.")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(data, f, indent=2)

    def load(self) -> dict[str, Any]:
        """Load the benchmark data."""
        path = self.to_path()
        if not path.exists():
            raise FileNotFoundError(f"Config file {path} does not exist.")
        with path.open("r") as f:
            return json.load(f)

    def batchable_with(self, other: ExperimentConfig) -> bool:
        """Check if this config is batchable with another config using the same list of sample requests."""
        return (
            self.model_id == other.model_id
            and self.dataset == other.dataset
            and self.seed == other.seed
            and self.num_prompts == other.num_prompts
            and self.input_len == other.input_len
            and self.output_len == other.output_len
            # this reduces the overhead of scaling
            and self.backend_config == other.backend_config
        )

    def __hash__(self) -> int:
        """Return a hash of the config."""
        return hash(
            (
                self.model_id,
                self.gpu_type,
                self.dataset,
                self.num_prompts,
                self.input_len,
                self.output_len,
                self.request_rate,
                self.burstiness,
                self.seed,
                self.use_synthesized_data,
                self.image_probability,
                self.image_width,
                self.image_height,
                self.image_count,
                self.image_choices,
            )
        )


if __name__ == "__main__":
    backend = CornserveConfig(num_erics=4, num_vllms=8)
    config = ExperimentConfig(backend_config=backend, app_id="example_app_id")
    dummy_data = {
        "example_key": "example_value",
        "another_key": 123,
    }
    print("path:", config.to_path())
    config.save(dummy_data)
    loaded_data = config.load()
    print("Saved data:", dummy_data)
    print("Loaded data:", loaded_data)
