"""Processor for CustomLALM (audio-only large audio LM).

Wraps the Whisper feature extractor + Qwen3 tokenizer. The contract with the
model side is identical to the upstream `PetitMLLM-5B` processor for the audio
portion:

- The chat-template-rendered text contains exactly one `<|audio_pad|>`
  per audio clip.
- `__call__` expands each `<|audio_pad|>` into `K_audio` consecutive copies,
  where `K_audio = ceil(T_out / stack_factor)` and `T_out` is the per-clip
  post-subsampling length on the Whisper encoder side.
- The model's `forward` then asserts that the number of `<|audio_pad|>`
  tokens in `input_ids` equals the number of valid feature vectors.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoTokenizer,
    WhisperFeatureExtractor,
)
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin

AUDIO_TOKEN = "<|audio_pad|>"


def compute_audio_token_count_per_clip(
    mel_lengths: torch.Tensor,
    stack_factor: int,
) -> torch.Tensor:
    """Mirror WhisperEncoder's conv stride-2 subsampling, then ceil-divide by
    `stack_factor`.

    Args:
        mel_lengths: (N_clips,) of real (non-padded) mel-frame counts per clip.
        stack_factor: from `CustomLALMConfig.audio_stack_factor`.

    Returns:
        (N_clips,) of `K_audio` per clip.
    """
    # Whisper has two conv layers; only conv2 has stride 2. T_out = T_in // 2
    # with floor-division (consistent with how the encoder's conv reduces
    # length under PyTorch's default `floor` rounding).
    encoder_lengths = mel_lengths // 2
    return (encoder_lengths + stack_factor - 1) // stack_factor


class CustomLALMProcessor(ProcessorMixin):
    """Combines the Whisper feature extractor and Qwen3 tokenizer."""

    attributes = ["audio_processor", "tokenizer"]
    audio_processor_class = "WhisperFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    valid_kwargs = [
        "audio_token",
        "audio_stack_factor",
    ]

    def __init__(
        self,
        audio_processor=None,
        tokenizer=None,
        chat_template: str | None = None,
        audio_token: str = AUDIO_TOKEN,
        audio_stack_factor: int = 4,
    ):
        self.audio_token = audio_token
        self.audio_stack_factor = audio_stack_factor
        if chat_template is None and tokenizer is not None:
            chat_template = getattr(tokenizer, "chat_template", None)
        if tokenizer is not None and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        super().__init__(
            audio_processor, tokenizer, chat_template=chat_template
        )

    def _k_audio_per_clip(self, mel_lengths: torch.Tensor) -> torch.Tensor:
        return compute_audio_token_count_per_clip(
            mel_lengths, self.audio_stack_factor
        )

    def __call__(
        self,
        text: str | list[str] | None = None,
        audio: list[Any] | None = None,
        sampling_rate: int = 16000,
        return_tensors: str = "pt",
        padding: bool | str = True,
        **kwargs,
    ) -> BatchFeature:
        if text is None:
            raise ValueError("`text` is required (call apply_chat_template first)")
        if isinstance(text, str):
            text = [text]

        data: dict[str, Any] = {}

        if audio:
            # WhisperFeatureExtractor must pad to its `chunk_length` (30 s);
            # the encoder asserts exactly 3000 mel frames, so `padding=True`
            # (batch-longest) is not enough for short clips.
            audio_out = self.audio_processor(
                audio,
                sampling_rate=sampling_rate,
                return_tensors=return_tensors,
                return_attention_mask=True,
                padding="max_length",
            )
            data["input_features"] = audio_out["input_features"]
            # Rename to match the model's `audio_attention_mask` kwarg.
            data["audio_attention_mask"] = audio_out["attention_mask"]
            mel_lengths = audio_out["attention_mask"].sum(dim=-1)
            k_per_clip = self._k_audio_per_clip(mel_lengths).tolist()
            total_placeholders = sum(t.count(self.audio_token) for t in text)
            if total_placeholders != len(audio):
                raise ValueError(
                    f"audio count mismatch: {len(audio)} clips, "
                    f"{total_placeholders} `{self.audio_token}` placeholders "
                    f"across {len(text)} text(s)"
                )
            expanded_texts: list[str] = []
            offset = 0
            for t in text:
                n = t.count(self.audio_token)
                expanded_texts.append(
                    self._expand_audio(t, k_per_clip[offset : offset + n])
                )
                offset += n
            text = expanded_texts

        text_inputs = self.tokenizer(
            text, return_tensors=return_tensors, padding=padding
        )
        data["input_ids"] = text_inputs["input_ids"]
        data["attention_mask"] = text_inputs["attention_mask"]
        return BatchFeature(data=data)

    def _expand_audio(self, text: str, k_per_clip: list[int]) -> str:
        """Replace each occurrence of `audio_token` with `audio_token * K_i`
        in document order — the i-th placeholder consumes the i-th K.
        """
        sentinel = "\x00"
        expansions = []
        for k in k_per_clip:
            if self.audio_token not in text:
                raise ValueError(
                    "more audio clips than <|audio_pad|> placeholders"
                )
            text = text.replace(self.audio_token, sentinel, 1)
            expansions.append(self.audio_token * k)
        if self.audio_token in text:
            raise ValueError(
                "more <|audio_pad|> placeholders than audio clips"
            )
        for expansion in expansions:
            text = text.replace(sentinel, expansion, 1)
        return text

    def apply_chat_template(self, conversation, **kwargs):
        return self.tokenizer.apply_chat_template(conversation, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self) -> list[str]:
        return [
            "input_ids",
            "attention_mask",
            "input_features",
            "audio_attention_mask",
        ]

    def save_pretrained(self, save_directory: str | os.PathLike, **kwargs):
        """No-collision custom save: tokenizer + audio feature extractor both
        emit `preprocessor_config.json`, so the audio one goes in a subdir.

        We mirror the same `audio_processor/` subdir convention as the upstream
        PetitMLLM-5B processor — that way callers who already have tooling for
        the upstream model don't need to special-case this one.
        """
        import json

        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        self.tokenizer.save_pretrained(save_directory)
        audio_dir = save_directory / "audio_processor"
        audio_dir.mkdir(parents=True, exist_ok=True)
        self.audio_processor.save_pretrained(audio_dir)

        proc_cfg = {k: getattr(self, k) for k in self.valid_kwargs}
        proc_cfg["processor_class"] = self.__class__.__name__
        if self._auto_class is not None:
            proc_cfg["auto_map"] = {
                "AutoProcessor": (
                    f"processing_custom_lalm.{self.__class__.__name__}"
                )
            }
            from transformers.dynamic_module_utils import custom_object_save

            custom_object_save(self, save_directory, config=self)
        with (save_directory / "processor_config.json").open("w") as f:
            json.dump(proc_cfg, f, indent=2)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        import json

        src = pretrained_model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
        audio_processor = WhisperFeatureExtractor.from_pretrained(
            src, subfolder="audio_processor"
        )

        proc_kwargs: dict[str, Any] = {}
        proc_cfg_path: Path | None = None
        local_path = Path(str(src))
        if local_path.exists():
            candidate = local_path / "processor_config.json"
            if candidate.exists():
                proc_cfg_path = candidate
        else:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.errors import EntryNotFoundError

            try:
                proc_cfg_path = Path(
                    hf_hub_download(
                        repo_id=str(src),
                        filename="processor_config.json",
                    )
                )
            except EntryNotFoundError:
                proc_cfg_path = None
        if proc_cfg_path is not None and proc_cfg_path.exists():
            with proc_cfg_path.open() as f:
                proc_kwargs = {
                    k: v for k, v in json.load(f).items() if k in cls.valid_kwargs
                }
        proc_kwargs.update(
            {k: v for k, v in kwargs.items() if k in cls.valid_kwargs}
        )
        return cls(
            audio_processor=audio_processor,
            tokenizer=tokenizer,
            **proc_kwargs,
        )
