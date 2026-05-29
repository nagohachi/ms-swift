"""Build an initial audio-only PetitMLLM checkpoint.

Combines a base LLM (default `Qwen/Qwen3-1.7B`) with a Whisper audio encoder
(default `openai/whisper-medium`) and an MLP audio projector, then writes the
result as a `trust_remote_code`-loadable HF directory.

Usage:
    python assemble.py --output ./petit_mllm_audio_init
    python assemble.py --llm Qwen/Qwen3-4B --audio openai/whisper-large-v3 \
        --output ./petit_mllm_audio_large_init

After this the directory can be passed to `swift sft --model <dir>` together
with `--custom_register_path .../register.py`.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    WhisperFeatureExtractor,
    WhisperModel,
)

# Local imports — run as a script from this directory so relative names work.
import sys

sys.path.insert(0, str(Path(__file__).parent))
from configuration_petit_mllm_audio import PetitMLLMAudioConfig  # noqa: E402
from modeling_petit_mllm_audio import PetitMLLMAudioPretrainedModel  # noqa: E402
from processing_petit_mllm_audio import (  # noqa: E402
    AUDIO_TOKEN,
    PetitMLLMAudioProcessor,
)

AUDIO_START = "<|audio_start|>"
AUDIO_END = "<|audio_end|>"

# Minimal Qwen-style chat template with text + audio content-block support.
# Mirrors the audio bracketing used by PetitMLLM-5B: `<|audio_start|>` +
# `<|audio_pad|>` + `<|audio_end|>`. The processor expands the inner
# `<|audio_pad|>` into K per-clip copies before tokenization.
CHAT_TEMPLATE = r"""{%- if messages[0].role == 'system' %}
    {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
{%- endif %}
{%- for message in messages %}
    {%- if message.role == 'system' and loop.first %}
        {# already emitted above #}
    {%- else %}
        {%- if message.content is string %}
            {%- set content = message.content %}
        {%- else %}
            {%- set ns = namespace(s='') %}
            {%- for c in message.content %}
                {%- if c.type == 'text' %}
                    {%- set ns.s = ns.s + c.text %}
                {%- elif c.type == 'audio' %}
                    {%- set ns.s = ns.s + '<|audio_start|><|audio_pad|><|audio_end|>' %}
                {%- endif %}
            {%- endfor %}
            {%- set content = ns.s %}
        {%- endif %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--llm", default="Qwen/Qwen3-1.7B", help="HF id or local path")
    ap.add_argument(
        "--audio",
        default="openai/whisper-medium",
        help="HF id or local path for the Whisper checkpoint to take the encoder from",
    )
    ap.add_argument("--audio-stack-factor", type=int, default=4)
    ap.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    ap.add_argument("--output", required=True, help="Output directory")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Loading LLM tokenizer + model: {args.llm}")
    tokenizer = AutoTokenizer.from_pretrained(args.llm, trust_remote_code=True)
    lm = AutoModelForCausalLM.from_pretrained(
        args.llm, dtype=dtype, trust_remote_code=True
    )

    # Add the audio bracketing tokens if the base tokenizer doesn't already
    # have them. PetitMLLM-5B's tokenizer ships with `<|audio_pad|>` baked in;
    # the base Qwen3 tokenizer does not. We grow the LM embedding to match.
    new_tokens = [
        t for t in (AUDIO_START, AUDIO_TOKEN, AUDIO_END) if t not in tokenizer.get_vocab()
    ]
    if new_tokens:
        print(f"      adding special tokens: {new_tokens}")
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        lm.resize_token_embeddings(len(tokenizer))
    audio_token_id = tokenizer.convert_tokens_to_ids(AUDIO_TOKEN)
    print(f"      audio_token_id = {audio_token_id}")

    print(f"[2/6] Loading audio encoder (Whisper, encoder half only): {args.audio}")
    whisper = WhisperModel.from_pretrained(args.audio, dtype=dtype)
    audio_feature_extractor = WhisperFeatureExtractor.from_pretrained(args.audio)
    # The feature extractor's default behaviour produces fixed-length 30s
    # input; the model's mask-pooling handles the per-clip valid length.

    print(f"[3/6] Assembling petit_mllm_audio config")
    lm_config = lm.config
    audio_config = whisper.config
    # Persist dtype names as strings so `from_pretrained` finds them on load.
    # `dtype` is the transformers 5.x attribute name; older 4.x versions used
    # `torch_dtype`. Setattr both for portability.
    for cfg in (lm_config, audio_config):
        setattr(cfg, "dtype", args.dtype)

    config = PetitMLLMAudioConfig(
        lm_config=lm_config,
        audio_config=audio_config,
        audio_stack_factor=args.audio_stack_factor,
        audio_token_id=audio_token_id,
        dtype=args.dtype,
    )
    config.auto_map = {
        "AutoConfig": "configuration_petit_mllm_audio.PetitMLLMAudioConfig",
        "AutoModelForCausalLM": (
            "modeling_petit_mllm_audio.PetitMLLMAudioPretrainedModel"
        ),
    }
    config.architectures = ["PetitMLLMAudioPretrainedModel"]

    print(f"[4/6] Building combined model and copying weights")
    model = PetitMLLMAudioPretrainedModel(config)
    # `from_config` reinstantiated the LM submodule; load the pretrained weights.
    model.language_model.load_state_dict(lm.state_dict())
    # The Whisper encoder lives under `whisper.encoder`. WhisperEncoder's
    # state_dict matches whisper.encoder's exactly.
    model.audio_tower.load_state_dict(whisper.encoder.state_dict())
    # audio_projector keeps its random init — it'll be trained from scratch.
    model = model.to(dtype)

    print(f"[5/6] Saving model + tokenizer + processor → {out_dir}")
    model.save_pretrained(out_dir)
    tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.save_pretrained(out_dir)

    processor = PetitMLLMAudioProcessor(
        audio_processor=audio_feature_extractor,
        tokenizer=tokenizer,
        audio_token=AUDIO_TOKEN,
        audio_stack_factor=args.audio_stack_factor,
        chat_template=CHAT_TEMPLATE,
    )
    # Register so `custom_object_save` emits the auto_map entry; the modeling
    # files themselves are copied below regardless.
    processor.register_for_auto_class("AutoProcessor")
    processor.save_pretrained(out_dir)

    print(f"[6/6] Copying remote-code files into {out_dir}")
    src_dir = Path(__file__).parent
    for fname in (
        "configuration_petit_mllm_audio.py",
        "modeling_petit_mllm_audio.py",
        "processing_petit_mllm_audio.py",
    ):
        shutil.copy(src_dir / fname, out_dir / fname)

    print("Done.")
    print(f"  audio_token_id={audio_token_id} "
          f"audio_stack_factor={args.audio_stack_factor}")
    print(f"Load with `AutoModelForCausalLM.from_pretrained({out_dir!s}, trust_remote_code=True)`")


if __name__ == "__main__":
    main()
