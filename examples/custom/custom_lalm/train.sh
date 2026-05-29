#!/usr/bin/env bash
# Example: SFT a CustomLALM (audio-only LALM) model on an ASR dataset.
#
# Prereq: run `python assemble.py --output ./custom_lalm_init` first.
#
# The audio encoder (`audio_tower`) maps to ms-swift's `vision_tower` slot, so
# `--freeze_vit true` freezes the Whisper encoder. The MLP projector is the
# `aligner` and is always trainable here so the projector can warm up from
# its random init.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${MODEL_DIR:-${SCRIPT_DIR}/custom_lalm_init}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output/custom_lalm_sft}"
DATASET="${DATASET:-AI-ModelScope/LibriSpeech#1000}"

swift sft \
    --custom_register_path "${SCRIPT_DIR}/register.py" \
    --model "${MODEL_DIR}" \
    --model_type custom_lalm \
    --template custom_lalm \
    --tuner_type full \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner false \
    --dataset "${DATASET}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.03 \
    --max_length 2048 \
    --output_dir "${OUTPUT_DIR}" \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 2 \
    --deepspeed zero2 \
    --bf16 true
