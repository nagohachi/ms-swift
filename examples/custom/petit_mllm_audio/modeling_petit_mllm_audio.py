"""Audio-only PetitMLLM (Whisper encoder + MLP projector + Qwen3 LM).

This is a stripped-down variant of `PetitMLLM-5B`'s `PetitMLLMPretrainedModel`:

- image_tower / image_projector removed.
- ParakeetEncoder replaced with the encoder half of a Whisper checkpoint
  (`transformers.WhisperEncoder`).

The audio token-count math (`K_audio` per clip) lives on the processor side;
this module just executes the scatter from `audio_token_id` placeholder
positions into projected audio features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import (
    AutoModelForCausalLM,
    GenerationMixin,
    PreTrainedModel,
)
from transformers.models.whisper.modeling_whisper import WhisperEncoder
from transformers.utils.generic import ModelOutput

# Loaded two ways: (1) as part of a `trust_remote_code` package out of the
# saved checkpoint dir, where the relative form is required; (2) as a flat
# module from `sys.path` (e.g. by `assemble.py`). The fallback handles (2).
try:
    from .configuration_petit_mllm_audio import PetitMLLMAudioConfig
except ImportError:
    from configuration_petit_mllm_audio import PetitMLLMAudioConfig


class PetitMLLMAudioProjector(nn.Module):
    """Stack `stack_factor` consecutive audio frames, then MLP-project to LM dim.

    Identical shape contract to `PetitMLLM-5B`'s `PetitMLLMAudioProjector` so the
    processor's K-count math carries over.
    """

    def __init__(
        self,
        audio_hidden_size: int,
        stack_factor: int,
        hidden_size: int,
        output_size: int,
    ):
        super().__init__()
        assert stack_factor >= 1, f"stack_factor must be >= 1, got {stack_factor}"
        self.stack_factor = stack_factor
        stacked_dim = audio_hidden_size * stack_factor
        self.mlp = nn.Sequential(
            nn.Linear(stacked_dim, hidden_size),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_size, output_size),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (N_clips, T_out, D_audio) audio encoder output, where T_out is
                the post-subsampling length (for Whisper conv stride 2:
                T_out = T_in // 2).
            attention_mask: (N_clips, T_out) — 1=real / 0=padded encoder frame.

        Returns:
            embeds: (N_clips, T', D_lm) projected post-stack features,
                T' = ceil(T_out / stack_factor).
            mask: (N_clips, T') — 1 if any constituent frame was real.
        """
        k = self.stack_factor
        if k > 1:
            t = x.shape[1]
            pad = (-t) % k
            if pad:
                x = F.pad(x, (0, 0, 0, pad))
                attention_mask = F.pad(attention_mask, (0, pad))
            x = rearrange(x, "n (t k) d -> n t (k d)", k=k)
            attention_mask = (
                rearrange(attention_mask, "n (t k) -> n t k", k=k).any(dim=-1).long()
            )
        return self.mlp(x), attention_mask


class PetitMLLMAudioPretrainedModel(PreTrainedModel, GenerationMixin):
    config_class = PetitMLLMAudioConfig
    # Transformers' attention-backend negotiation walks every submodule and
    # refuses dispatch if any wrapper class lacks the corresponding flag.
    # We don't implement attention ourselves — both submodules (Qwen3 LM and
    # WhisperEncoder) support sdpa/flash; advertise the same so dispatch
    # propagates through us.
    _supports_sdpa = True
    _supports_flash_attn_2 = True
    _supports_attention_backend = True

    def __init__(self, config: PetitMLLMAudioConfig):
        super().__init__(config)
        self.language_model = AutoModelForCausalLM.from_config(
            config.lm_config, trust_remote_code=True
        )
        self.audio_tower = WhisperEncoder(config.audio_config)
        self.audio_projector = PetitMLLMAudioProjector(
            audio_hidden_size=config.audio_config.d_model,
            stack_factor=config.audio_stack_factor,
            hidden_size=config.lm_config.hidden_size,
            output_size=config.lm_config.hidden_size,
        )
        # FSDP2 / mixed-precision: keep the projector's dtype aligned with the
        # LM. The LM and audio tower get their dtype from `from_pretrained`;
        # the projector is fresh `nn.Linear` and defaults to fp32.
        # `dtype` is the transformers 5.x attribute; `torch_dtype` is the 4.x
        # fallback (kept here for older checkpoints, even though we emit the
        # new name on save).
        lm_dtype = getattr(config.lm_config, "dtype", None) or getattr(
            config.lm_config, "torch_dtype", None
        )
        if isinstance(lm_dtype, str):
            lm_dtype = getattr(torch, lm_dtype, None)
        if lm_dtype is not None:
            self.audio_projector.to(dtype=lm_dtype)
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def calculate_audio_embed(
        self,
        input_features: torch.FloatTensor,
        attention_mask: torch.LongTensor,
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        """
        Args:
            input_features: (N_clips, n_mels, T_in) mel features. The Whisper
                feature extractor pads to a fixed `chunk_length` so T_in is
                constant across the batch.
            attention_mask: (N_clips, T_in) — mel-frame mask (1=real).

        Returns:
            audio_embeds: (N_clips, T', D_lm) post-stack features.
            output_attention_mask: (N_clips, T') real-frame mask, post-stack.
        """
        tower_dtype = next(self.audio_tower.parameters()).dtype
        if input_features.dtype != tower_dtype:
            input_features = input_features.to(tower_dtype)
        # WhisperEncoder runs over the full padded mel input regardless of any
        # attention mask. We track validity ourselves so the projector can
        # drop encoder outputs that came from padded frames.
        encoder_out = self.audio_tower(input_features=input_features, return_dict=True)
        hidden = encoder_out.last_hidden_state  # (N, T_out, d_model)
        # Whisper conv subsampling halves the time axis (conv2 stride=2). Pool
        # the mel mask the same way: an output frame is real iff either of
        # its two source mel frames was real.
        t_out = hidden.shape[1]
        mel_mask = attention_mask
        if mel_mask.shape[1] < t_out * 2:
            mel_mask = F.pad(mel_mask, (0, t_out * 2 - mel_mask.shape[1]))
        elif mel_mask.shape[1] > t_out * 2:
            mel_mask = mel_mask[:, : t_out * 2]
        out_mask = rearrange(mel_mask, "n (t k) -> n t k", k=2).any(dim=-1).long()

        proj_dtype = next(self.audio_projector.parameters()).dtype
        if hidden.dtype != proj_dtype:
            hidden = hidden.to(proj_dtype)
        return self.audio_projector(hidden, out_mask)

    def calculate_language_embed(
        self, input_ids: torch.LongTensor
    ) -> torch.FloatTensor:
        return self.get_input_embeddings()(input_ids)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        audio_attention_mask: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        past_key_values=None,
        position_ids: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs,
    ) -> ModelOutput:
        # Prefill builds inputs_embeds with the modality scatter; decode (post
        # `generate`'s prepare_inputs_for_generation) passes inputs_embeds=None
        # and a single-token input_ids, so we re-embed and skip the audio path.
        if inputs_embeds is None:
            inputs_embeds = self.calculate_language_embed(input_ids)

            if input_features is not None and input_features.numel() > 0:
                assert audio_attention_mask is not None, (
                    "audio_attention_mask is required when input_features is "
                    "given (it determines real-vs-padded mel frames)"
                )
                audio_embeds, downsampled_mask = self.calculate_audio_embed(
                    input_features, attention_mask=audio_attention_mask
                )
                audio_embeds = audio_embeds[downsampled_mask.bool()]

                audio_mask = input_ids == self.config.audio_token_id
                assert audio_mask.sum().item() == audio_embeds.size(0), (
                    f"audio placeholders != features, "
                    f"{audio_mask.sum().item()} != {audio_embeds.size(0)}"
                )
                # Out-of-place masked_scatter so inputs_embeds keeps an autograd
                # path back to audio_embeds when the LM embedding is frozen.
                inputs_embeds = inputs_embeds.masked_scatter(
                    audio_mask.unsqueeze(-1).expand_as(inputs_embeds),
                    audio_embeds.to(inputs_embeds.dtype),
                )

        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=use_cache,
            labels=labels,
            return_dict=True,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values=None,
        attention_mask: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        audio_attention_mask: torch.LongTensor | None = None,
        **kwargs,
    ) -> dict:
        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        # Only attach audio inputs on the prefill step. On subsequent decode
        # steps `past_key_values` is non-None and we must not re-run the audio
        # tower (the LM cache already holds the projected features).
        if past_key_values is None and input_features is not None:
            model_inputs["input_features"] = input_features
            if audio_attention_mask is not None:
                model_inputs["audio_attention_mask"] = audio_attention_mask
        return model_inputs
