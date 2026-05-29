"""ms-swift custom registration for CustomLALM (audio-only large audio LM).

Pass this file to `swift sft` / `swift infer` via `--custom_register_path`:

    swift sft --custom_register_path \
        examples/custom/custom_lalm/register.py \
      --model <custom_lalm dir from assemble.py> \
      --model_type custom_lalm \
      --template custom_lalm \
      --dataset ...

The model architecture is `CustomLALMPretrainedModel`; the chat template
is the same `<|im_start|>...<|im_end|>` shell as Qwen3, with each audio block
expanding to `<|audio_start|><|audio_pad|><|audio_end|>` and the processor
then expanding `<|audio_pad|>` to `K_audio` consecutive copies per clip.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal

import torch
from transformers import PreTrainedModel

from swift.model import (
    ModelGroup,
    ModelLoader,
    ModelMeta,
    MultiModelKeys,
    register_model,
    register_model_arch,
)
from swift.template import (
    StdTemplateInputs,
    Template,
    TemplateMeta,
    register_template,
)
from swift.template.utils import Context, findall
from swift.template.vision_utils import load_audio
from swift.utils import get_env_args, get_logger

logger = get_logger()

MODEL_TYPE = "custom_lalm"
TEMPLATE_TYPE = "custom_lalm"
MODEL_ARCH = "custom_lalm"
AUDIO_TOKEN = "<|audio_pad|>"


def _k_audio_per_clip(mel_lengths: torch.Tensor, stack_factor: int) -> torch.Tensor:
    """Mirror Whisper conv stride-2 subsampling then ceil-divide by stack_factor.

    Must stay in sync with `processing_custom_lalm.compute_audio_token_count_per_clip`.
    """
    encoder_lengths = mel_lengths // 2
    return (encoder_lengths + stack_factor - 1) // stack_factor


# `MultiModelKeys` tells ms-swift which submodule paths correspond to the
# LLM / projector / audio encoder, which drives `--freeze_llm`, `--freeze_vit`,
# `--freeze_aligner` and the matching LoRA target selection.
register_model_arch(
    MultiModelKeys(
        MODEL_ARCH,
        language_model="language_model",
        aligner="audio_projector",
        vision_tower="audio_tower",
    )
)


class CustomLALMLoader(ModelLoader):
    """Loader for `custom_lalm`.

    The HF dir is `trust_remote_code` and registers `CustomLALMPretrainedModel`
    against `AutoModelForCausalLM`, so the default `ModelLoader` machinery is
    enough — we just have to set `auto_model_cls` explicitly because the
    architecture name isn't in transformers' built-in mapping.
    """

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        from transformers import AutoModelForCausalLM

        self.auto_model_cls = self.auto_model_cls or AutoModelForCausalLM
        return super().get_model(model_dir, *args, **kwargs)


register_model(
    ModelMeta(
        MODEL_TYPE,
        # No public Hub id — `custom_lalm` checkpoints are produced
        # locally by `assemble.py`. Users pass `--model <local dir>`.
        [ModelGroup([])],
        CustomLALMLoader,
        template=TEMPLATE_TYPE,
        is_multimodal=True,
        model_arch=MODEL_ARCH,
        architectures=["CustomLALMPretrainedModel"],
        requires=["transformers>=4.50", "librosa", "einops", "soundfile"],
        tags=["audio"],
    )
)


class CustomLALMTemplate(Template):
    # `<|audio_pad|>` is the post-expansion audio token. We mark it as a
    # placeholder so truncation doesn't slice through an audio run and so
    # `safe_decode` abbreviates the run on logging.
    placeholder_tokens = [AUDIO_TOKEN]
    use_model = False
    support_padding_free = False

    def init_processor(self, processor) -> None:
        if processor is None:
            return
        super().init_processor(processor)
        # `processor.audio_processor` is a `WhisperFeatureExtractor`.
        self.sampling_rate = get_env_args(
            "sampling_rate", int, processor.audio_processor.sampling_rate
        )
        self.audio_stack_factor = processor.audio_stack_factor
        # `_tokenize` returns a list of token ids; the audio token is a single
        # special token, so the list has length 1.
        self._audio_token_ids = self._tokenize(AUDIO_TOKEN)
        assert len(self._audio_token_ids) == 1, (
            f"expected `{AUDIO_TOKEN}` to be a single token, "
            f"got {self._audio_token_ids}"
        )

    def replace_tag(
        self,
        media_type: Literal["image", "video", "audio"],
        index: int,
        inputs: StdTemplateInputs,
    ) -> List[Context]:
        assert media_type == "audio", (
            f"custom_lalm only supports audio inputs, got {media_type}"
        )
        return ["<|audio_start|><|audio_pad|><|audio_end|>"]

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        if not inputs.audios:
            return encoded

        # Load audio waveforms then extract Whisper mel features in batch.
        audios = [load_audio(a, self.sampling_rate) for a in inputs.audios]
        # WhisperFeatureExtractor needs padding="max_length" to pad to the
        # encoder's fixed 30 s window — passing `padding=True` only pads to
        # batch-longest, and the encoder hard-asserts 3000 mel frames.
        audio_inputs = self.processor.audio_processor(
            audios,
            sampling_rate=self.sampling_rate,
            return_attention_mask=True,
            return_tensors="pt",
            padding="max_length",
        )
        input_features = audio_inputs["input_features"]
        audio_attention_mask = audio_inputs["attention_mask"]
        encoded["input_features"] = input_features
        encoded["audio_attention_mask"] = audio_attention_mask

        # Compute per-clip audio token counts and expand each `<|audio_pad|>`
        # placeholder into K_audio copies.
        mel_lengths = audio_attention_mask.sum(dim=-1)
        k_per_clip = _k_audio_per_clip(
            mel_lengths, self.audio_stack_factor
        ).tolist()

        input_ids = encoded["input_ids"]
        labels = encoded["labels"]
        loss_scale = encoded.get("loss_scale", None)
        idx_list = findall(input_ids, self._audio_token_ids)
        if len(idx_list) != len(k_per_clip):
            raise ValueError(
                f"audio placeholder count {len(idx_list)} != "
                f"audio clip count {len(k_per_clip)}"
            )

        def _get_new_tokens(i: int) -> List[int]:
            return self._audio_token_ids * k_per_clip[i]

        input_ids, labels, loss_scale = self._extend_tokens(
            input_ids, labels, loss_scale, idx_list, _get_new_tokens
        )
        encoded["input_ids"] = input_ids
        encoded["labels"] = labels
        encoded["loss_scale"] = loss_scale
        return encoded

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch)
        input_features = [
            b["input_features"] for b in batch if b.get("input_features") is not None
        ]
        audio_attention_mask = [
            b["audio_attention_mask"]
            for b in batch
            if b.get("audio_attention_mask") is not None
        ]
        if input_features:
            res["input_features"] = torch.concat(input_features)
            res["audio_attention_mask"] = torch.concat(audio_attention_mask)
        return res


register_template(
    TemplateMeta(
        TEMPLATE_TYPE,
        prefix=[],
        prompt=["<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n"],
        chat_sep=["<|im_end|>\n"],
        suffix=["<|im_end|>"],
        system_prefix=["<|im_start|>system\n{{SYSTEM}}<|im_end|>\n"],
        # Default to the Qwen3 system prompt so that the trained model carries
        # through the LM's chat affordances; overridden by `--system` at run time.
        default_system="You are a helpful assistant.",
        stop_words=["<|endoftext|>", "<|im_end|>"],
        agent_template="hermes",
        template_cls=CustomLALMTemplate,
    )
)


if __name__ == "__main__":
    # Smoke test: load a local dir + dispatch a tiny conversation.
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: python register.py <custom_lalm_dir> "
            "[<wav_path>]"
        )
        sys.exit(1)
    from swift.model import get_model_processor
    from swift.template import get_template

    model_dir = sys.argv[1]
    audio_path = sys.argv[2] if len(sys.argv) > 2 else None
    print(f"Loading {model_dir}")
    model, processor = get_model_processor(model_dir, model_type=MODEL_TYPE)
    template = get_template(processor, template_type=TEMPLATE_TYPE)
    template.set_mode("train")

    user_content: Any = "transcribe the audio"
    if audio_path:
        user_content = [
            {"type": "audio", "audio": audio_path},
            {"type": "text", "text": "transcribe the audio"},
        ]
    data = {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": "hello world"},
        ],
        "audios": [audio_path] if audio_path else [],
    }
    encoded = template.encode(data)
    print("input_ids:", template.safe_decode(encoded["input_ids"]))
    print("labels:   ", template.safe_decode(encoded["labels"]))
    print("keys:     ", list(encoded.keys()))
