# -*- coding:utf-8 -*-

from typing import  Optional, Tuple
from torch import nn
import torch
from transformers import LlamaConfig, PretrainedConfig
import transformers
import math
import pdb

def _compute_default_rope_parameters(
        config: Optional[PretrainedConfig] = None,
        device: Optional["torch.device"] = None,
        seq_len: Optional[int] = None,
        **rope_kwargs,
) -> Tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies according to the original RoPE implementation
    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
        rope_kwargs (`Dict`, *optional*):
            BC compatibility with the previous RoPE class instantiation, will be removed in v4.45.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    if config is not None and len(rope_kwargs) > 0:
        raise ValueError(
            "Unexpected arguments: `**rope_kwargs` and `config` are mutually exclusive in "
            f"`_compute_default_rope_parameters`, got `rope_kwargs`={rope_kwargs} and `config`={config}"
        )
    if len(rope_kwargs) > 0:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
    elif config is not None:
        base = config.rope_theta
        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        dim = int((config.hidden_size // config.num_attention_heads) * partial_rotary_factor)

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
    return inv_freq

def _compute_yarn_parameters(
        config: PretrainedConfig, device: "torch.device", seq_len: Optional[int] = None, **rope_kwargs
) -> Tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies with NTK scaling. Please refer to the
    [original paper](https://arxiv.org/abs/2309.00071)
    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
        rope_kwargs (`Dict`, *optional*):
            BC compatibility with the previous RoPE class instantiation, will be removed in v4.45.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin.
    """
    if len(rope_kwargs) > 0:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
    else:
        base = config.rope_theta
        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        dim = int((config.hidden_size // config.num_attention_heads) * partial_rotary_factor)

    # qwen2 yarn config
    max_position_embeddings = 32768
    # Optional config options
    # beta_fast/beta_slow: as suggested in the paper, default to 32/1 (correspondingly)
    beta_fast = 32
    beta_slow = 1

    # Compute the inverse frequencies
    def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
        """Inverse dimension formula to find the dimension based on the number of rotations"""
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings):
        """Find dimension range bounds based on rotations"""
        low = math.floor(find_correction_dim(low_rot, dim, base, max_position_embeddings))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_position_embeddings))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_mask(min, max, dim):
        if min == max:
            max += 0.001  # Prevent singularity

        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    pos_freqs = base ** (torch.arange(0, dim, 2).float().to(device) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (yarn_qwen2_factor * pos_freqs)

    low, high = find_correction_range(beta_fast, beta_slow, dim, base, max_position_embeddings)

    # Get n-dimensional rotational scaling corrected for extrapolation
    inv_freq_mask = 1 - linear_ramp_mask(low, high, dim // 2).float().to(device)
    inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
    return inv_freq


class YarnRotaryEmbedding(nn.Module):
    def __init__(
            self,
            dim=None,
            max_position_embeddings=2048,
            base=10000,
            device=None,
            scaling_factor=1.0,
            rope_type="default",
            config: Optional[LlamaConfig] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = _compute_yarn_parameters
        inv_freq = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", attention_factor * emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", attention_factor * emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if not isinstance(seq_len, int):
            seq_len = seq_len.size(-1)
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# diag_size = None
# local_window = None

MAX_NEW_TOKENS = 1024
attention_factor = 1.0
yarn_qwen2_factor = 1.0
from string_for_llama import flashdecoding_forward, forward, causal_forward


def replace_with_string(max_test_length, shifted_offset, small_local_value=128, yarn_factor=None):
    # this is used to pre-allocate KV cache, saving GPU memory
    global yarn_qwen2_factor
    global attention_factor
    # STRING will make the attention map smooth, we use the attention_factor to recover it, similar with Yarn
    attention_factor = 0.1*math.log(2) + 1
    yarn_qwen2_factor = yarn_factor

    # String parameters
    import string_for_llama 
    string_for_llama.MAX_CACHE_LEN = max_test_length + MAX_NEW_TOKENS
    string_for_llama.diag_size = shifted_offset
    string_for_llama.local_window = small_local_value
    print("============== [STRING Config for Qwen2] ===============")
    print(f"Position ids for sliding window attention: {0}-{string_for_llama.diag_size}")
    print(f"Position ids for Shifted self attention: {small_local_value}-{max_test_length-small_local_value}")
    print(f"Extrapolation Yarn factor: {yarn_qwen2_factor}")
    print(f"Attention factor {attention_factor}")

    transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM.forward = causal_forward
    transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = forward
    transformers.models.qwen2.modeling_qwen2.Qwen2Model.forward = flashdecoding_forward
    transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2.forward = forward
    transformers.models.qwen2.modeling_qwen2.Qwen2RotaryEmbedding = YarnRotaryEmbedding
