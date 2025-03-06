import torch.nn as nn
from transformers import CLIPVisionConfig, SiglipVisionConfig, PixtralVisionConfig

from cornserve.task_executors.eric.models.layers.siglip import SiglipVisionModel


def _get_num_hidden_layers(hf_config) -> int:
    """Determine the number of hidden layers to initialize up to in the
    visual encoder.

    Args:
        hf_config: Model config with vision feature layer(s).
    """
    feature_layers = hf_config.vision_feature_layer
    num_hidden_layers = hf_config.vision_config.num_hidden_layers
    # If we have one feature layer, initialize up to that layer
    if isinstance(feature_layers, int):
        return _get_layer_index(feature_layers, num_hidden_layers)
    # If we have multiple feature layers, initialize up to the deepest one
    elif isinstance(feature_layers, (list, tuple)):
        return max(_get_layer_index(idx, num_hidden_layers) for idx in feature_layers)
    raise TypeError(f"vision_layer_feature type: {type(feature_layers)} is not supported")


def _get_layer_index(feature_layer_index: int, num_hidden_layers: int) -> int:
    """Given a signed vision feature layer, get the number of hidden layers
    needed to leverage it.

    Args:
        feature_layer_index: Index of a required layer in the visual encoder.
        num_hidden_layers: The total number of hidden layers in the visual
            encoder.
    """
    if feature_layer_index < 0:
        return num_hidden_layers + feature_layer_index + 1
    return feature_layer_index


def init_vision_tower_for_llava(
    hf_config,
    *,
    require_post_norm: bool | None = None,
) -> nn.Module:
    """Initialize the vision tower for Llava-family models."""
    vision_config = hf_config.vision_config

    # Initialize the vision tower only up to the deepest required feature layer
    num_hidden_layers = _get_num_hidden_layers(hf_config)

    if isinstance(vision_config, SiglipVisionConfig):
        return SiglipVisionModel(
            vision_config,
            num_hidden_layers_override=num_hidden_layers,
            require_post_norm=require_post_norm,
        )

    msg = f"Unsupported vision config: {type(vision_config)}"
    raise NotImplementedError(msg)
