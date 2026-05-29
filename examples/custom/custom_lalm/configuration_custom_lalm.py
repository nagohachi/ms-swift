"""HF config for CustomLALM (audio-only large audio LM).

Layout mirrors `PetitMLLM-5B`'s `petit_mllm` config (Qwen3 LM + Parakeet audio
encoder + image tower) with the image side removed: only `lm_config` and
`audio_config` remain.

Sub-configs are stored as dicts on disk and rehydrated into proper
`PretrainedConfig` instances on load — same convention as `PetitMLLMConfig`,
so configs saved by this class are forward-compatible with `AutoConfig` /
`AutoModelForCausalLM` over `trust_remote_code=True`.
"""

from transformers import AutoConfig, PretrainedConfig


class CustomLALMConfig(PretrainedConfig):
    model_type = "custom_lalm"

    def __init__(
        self,
        lm_config: PretrainedConfig | dict | None = None,
        audio_config: PretrainedConfig | dict | None = None,
        audio_stack_factor: int = 4,
        audio_token_id: int | None = None,
        **kwargs,
    ) -> None:
        self.lm_config = self._to_pretrained_config(lm_config)
        self.audio_config = self._to_pretrained_config(audio_config)
        self.audio_stack_factor = audio_stack_factor
        self.audio_token_id = audio_token_id
        super().__init__(**kwargs)

    @staticmethod
    def _to_pretrained_config(value):
        if value is None or isinstance(value, PretrainedConfig):
            return value
        if isinstance(value, dict):
            # Prefer remote code (auto_map) over CONFIG_MAPPING so that
            # save_pretrained-serialized fields parse against the same class
            # that produced them.
            auto_map = value.get("auto_map") or {}
            class_ref = auto_map.get("AutoConfig")
            if class_ref:
                if "--" in class_ref:
                    upstream_repo, ref = class_ref.split("--", 1)
                else:
                    upstream_repo = value.get("_name_or_path") or value.get(
                        "name_or_path"
                    )
                    ref = class_ref
                if upstream_repo:
                    from transformers.dynamic_module_utils import (
                        get_class_from_dynamic_module,
                    )

                    config_class = get_class_from_dynamic_module(ref, upstream_repo)
                    config_class.register_for_auto_class()
                    return config_class(**value)

            from transformers.models.auto.configuration_auto import CONFIG_MAPPING

            model_type = value.get("model_type", "")
            if model_type and model_type in CONFIG_MAPPING:
                return AutoConfig.for_model(**value)
            return PretrainedConfig(**value)
        raise TypeError(
            f"sub-config must be dict or PretrainedConfig, got {type(value)}"
        )
