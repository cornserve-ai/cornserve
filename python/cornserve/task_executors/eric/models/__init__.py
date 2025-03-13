"""The Eric model zoo.

Steps for adding a new model:

1. Create a module inside `models` named after the model type (`hf_config.model_type`).
2. Implement the model class inheriting from `models.base.EricModel`.
3. Implement a class exactly called `ModalityProcessor` in the module, inheriting from
    `models.base.BaseModalityProcessor`. For each supported modality, implement
    the corresponding method (`get_image_processor`, `get_video_processor`, etc.).
4. Add an entry in `models.registry.MODEL_REGISTRY` for the model type.
"""
