# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.33.2/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Rotary Positional Embeddings."""
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import pdb
import torch.nn as nn
import torch
from vllm.model_executor.custom_op import CustomOp
from vllm.utils import is_tpu


def _rotate_neox(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def _apply_rotary_emb(
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> torch.Tensor:
    x_ = torch.view_as_complex(
        torch.stack(torch.chunk(x.transpose(1, 2).float(), 2, dim=-1), dim=-1))
    x_out = torch.view_as_real(x_ * freqs_cis).type_as(x)
    x_out = torch.cat(torch.chunk(x_out, 2, dim=-1), dim=-2)
    x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2],
                          -1).transpose(1, 2)
    return x_out


class RotaryEmbedding(CustomOp):
    """Original rotary positional embedding."""

    def __init__(
            self,
            head_size: int,
            rotary_dim: int,
            max_position_embeddings: int,
            base: int,
            is_neox_style: bool,
            dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype

        cache = self._compute_cos_sin_cache()
        self.use_native2 = is_tpu() and is_neox_style
        if not self.use_native2:
            cache = cache.to(dtype)
            self.register_buffer("cos_sin_cache", cache, persistent=False)
        else:
            cos, sin = cache.chunk(2, dim=-1)
            freqs_cis = cos + 1j * sin
            self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
        """Compute the inverse frequency."""
        # NOTE(woosuk): The HF implementation uses `torch.arange(...).float()`.
        # However, we use `torch.arange(..., dtype=torch.float)` instead to
        # avoid numerical issues with large base values (e.g., 10000000).
        # This may cause a slight numerical difference between the HF
        # implementation and ours.
        # NOTE(woosuk): To exactly match the HF implementation, we need to
        # use CPU to compute the cache and then move it to GPU. However, we
        # create the cache on GPU for faster initialization. This may cause
        # a slight numerical difference between the HF implementation and ours.
        inv_freq = 1.0 / (base ** (torch.arange(
            0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim))
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward_native(
            self,
            positions: torch.Tensor,
            query: torch.Tensor,
            key: torch.Tensor,
            offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A PyTorch-native implementation equivalent to forward().

        This method mimics the implementation of the custom CUDA kernel
        used in `forward_cuda()`.
        """
        query = query.view(*query.shape[:-1], -1, self.head_size)
        key = key.view(*key.shape[:-1], -1, self.head_size)

        query_rot = query[..., :self.rotary_dim]
        key_rot = key[..., :self.rotary_dim]
        if self.rotary_dim < self.head_size:
            query_pass = query[..., self.rotary_dim:]
            key_pass = key[..., self.rotary_dim:]

        self.cos_sin_cache: torch.Tensor = self.cos_sin_cache.to(
            positions.device, dtype=query.dtype)
        cos_sin = self.cos_sin_cache[torch.add(positions, offsets)
        if offsets is not None else positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if self.is_neox_style:
            # NOTE(woosuk): Here we assume that the positions tensor has the
            # shape [batch_size, seq_len].
            cos = cos.repeat(1, 1, 2).unsqueeze(-2)
            sin = sin.repeat(1, 1, 2).unsqueeze(-2)
        else:
            cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
            sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)

        rotate_fn = _rotate_neox if self.is_neox_style else _rotate_gptj
        query_rot = query_rot * cos + rotate_fn(query_rot) * sin
        key_rot = key_rot * cos + rotate_fn(key_rot) * sin

        if self.rotary_dim < self.head_size:
            query = torch.cat((query_rot, query_pass), dim=-1)
            key = torch.cat((key_rot, key_pass), dim=-1)
        else:
            query = query_rot
            key = key_rot
        query = query.flatten(-2)
        key = key.flatten(-2)
        return query, key

    def forward_native2(
            self,
            positions: torch.Tensor,
            query: torch.Tensor,
            key: torch.Tensor,
            offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Another PyTorch-native implementation of forward().

        This method might perform better than `forward_native()` when compiled.
        """
        if positions.dim() == 1:
            batch_size = 1
            seq_len = positions.shape[0]
        else:
            batch_size, seq_len = positions.shape
        if offsets is not None:
            positions = positions + offsets
        freqs_cis = self.freqs_cis.index_select(0, positions.flatten())
        freqs_cis = freqs_cis.view(batch_size, 1, seq_len, -1)

        query_shape = query.shape
        query = query.view(batch_size, seq_len, -1, self.head_size)
        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        query_rot = _apply_rotary_emb(query_rot, freqs_cis)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.view(batch_size, seq_len, -1, self.head_size)
        key_rot = key[..., :self.rotary_dim]
        key_pass = key[..., self.rotary_dim:]
        key_rot = _apply_rotary_emb(key_rot, freqs_cis)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

    def forward_cuda(
            self,
            positions: torch.Tensor,
            query: torch.Tensor,
            key: torch.Tensor,
            offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from vllm import _custom_ops as ops

        self.cos_sin_cache = self.cos_sin_cache.to(positions.device,
                                                   dtype=query.dtype)
        # ops.rotary_embedding()/batched_rotary_embedding()
        # are in-place operations that update the query and key tensors.
        if offsets is not None:
            ops.batched_rotary_embedding(positions, query, key, self.head_size,
                                         self.cos_sin_cache,
                                         self.is_neox_style, self.rotary_dim,
                                         offsets)
        else:
            ops.rotary_embedding(positions, query, key, self.head_size,
                                 self.cos_sin_cache, self.is_neox_style)
        return query, key

    def forward_xpu(
            self,
            positions: torch.Tensor,
            query: torch.Tensor,
            key: torch.Tensor,
            offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from vllm._ipex_ops import ipex_ops as ops

        self.cos_sin_cache = self.cos_sin_cache.to(positions.device,
                                                   dtype=query.dtype)
        # ops.rotary_embedding()/batched_rotary_embedding()
        # are in-place operations that update the query and key tensors.
        if offsets is not None:
            ops.batched_rotary_embedding(positions, query, key, self.head_size,
                                         self.cos_sin_cache,
                                         self.is_neox_style, self.rotary_dim,
                                         offsets)
        else:
            ops.rotary_embedding(positions, query, key, self.head_size,
                                 self.cos_sin_cache, self.is_neox_style)
        return query, key

    def forward_tpu(
            self,
            positions: torch.Tensor,
            query: torch.Tensor,
            key: torch.Tensor,
            offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        forward_fn = (self.forward_native2
                      if self.use_native2 else self.forward_native)
        return forward_fn(positions, query, key, offsets)

    def extra_repr(self) -> str:
        s = f"head_size={self.head_size}, rotary_dim={self.rotary_dim}"
        s += f", max_position_embeddings={self.max_position_embeddings}"
        s += f", base={self.base}, is_neox_style={self.is_neox_style}"
        return s


class LinearScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with linear scaling.

    It supports multiple scaling factors. Since multiple LoRA adapters may have
    different scaling factors, we need multiple cos/sin caches. In this way,
    instead of running rotary embedding kernel per lora, we can run multiple
    lora in a batched way.

    In addition to that, we also keep the cos/sin cache for the scaling factor
    of 1 (default) at all times.

    Exemplary for two scaling factors x=1, y and z with embeddings
    [[x11, x12, ... x1m], ..., [xn1, xn2, ..., xnm]] and
    [[y11, y12, ... y1o], ..., [yn1, yn2, ..., yno]], and
    [[z11, z12, ... z1p], ..., [zn1, zn2, ..., znp]],

    we construct the cos/sin cache as follows:
    [[x11, x12, ... x1m, y11, y12, ... y1o, z11, z12, ... z1p],
        ...
     [xn1, xn2, ... xnm, yn1, yn2, ... yno, zn1, zn2, ... znp]]

    We then use offsets to index into the cos/sin cache for
    the respective scaling factors.

    The offset to cache can be accessed via `scaling_factor_to_offset` API.

    Credits to the Reddit user /u/kaiokendev
    """

    def __init__(
            self,
            head_size: int,
            rotary_dim: int,
            max_position_embeddings: int,
            base: int,
            is_neox_style: bool,
            scaling_factors: Union[List[float], float],
            dtype: torch.dtype,
    ) -> None:
        if isinstance(scaling_factors, float):
            scaling_factors = [scaling_factors]
        self.scaling_factors: List[float] = scaling_factors  # noqa
        super().__init__(head_size, rotary_dim, max_position_embeddings, base,
                         is_neox_style, dtype)
        # Lazy initialized.
        self._scaling_factor_to_offset: Dict[float, int]

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.base)
        cache_list: List[torch.Tensor] = []
        # offsets to the next cache in a tensor.
        # Each offset corresponds to the same index in scaling_factors.
        offsets: List[int] = []
        for scaling_factor in self.scaling_factors:
            # NOTE(woosuk): self.max_position_embeddings is the original
            # maximum length before applying the rope scaling.
            # Thus, the maximum length after applying the rope scaling is
            # self.max_position_embeddings * self.scaling_factor.
            max_len = self.max_position_embeddings * scaling_factor
            t = torch.arange(max_len, dtype=torch.float)
            t = t / scaling_factor

            freqs = torch.einsum("i,j -> ij", t, inv_freq)
            cos = freqs.cos()
            sin = freqs.sin()
            cache = torch.cat((cos, sin), dim=-1)
            if not cache_list:
                offset = 0
            else:
                last_offset = offsets[-1]
                next_max_len = cache_list[-1].shape[0]
                offset = last_offset + next_max_len
            offsets.append(offset)
            cache_list.append(cache)
        self._scaling_factor_to_offset = {
            float(scaling_factor): offsets[i]
            for i, scaling_factor in enumerate(self.scaling_factors)
        }
        assert len(self.scaling_factors) == len(offsets)
        return torch.cat(cache_list, dim=0)

    @property
    def scaling_factor_to_offset(self) -> Dict[float, int]:
        return self._scaling_factor_to_offset


class DynamicNTKScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with Dynamic NTK scaling.

    Credits to the Reddit users /u/bloc97 and /u/emozilla
    """

    def __init__(
            self,
            head_size: int,
            rotary_dim: int,
            max_position_embeddings: int,
            base: int,
            is_neox_style: bool,
            scaling_factor: float,
            dtype: torch.dtype,
    ) -> None:
        self.scaling_factor = scaling_factor
        super().__init__(head_size, rotary_dim, max_position_embeddings, base,
                         is_neox_style, dtype)

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        # NOTE(woosuk): self.max_position_embeddings is the original
        # maximum length before applying the rope scaling.
        # Thus, the maximum length after applying the rope scaling is
        # self.max_position_embeddings * self.scaling_factor.
        max_len = self.max_position_embeddings * self.scaling_factor
        base = self.base * (
                (self.scaling_factor * max_len / self.max_position_embeddings) -
                (self.scaling_factor - 1)) ** (self.rotary_dim /
                                               (self.rotary_dim - 2))
        inv_freq = self._compute_inv_freq(base)
        t = torch.arange(max_len, dtype=torch.float)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache


# Inverse dim formula to find dim based on number of rotations
def _yarn_find_correction_dim(num_rotations: int,
                              dim: int,
                              base: float = 10000,
                              max_position_embeddings: int = 2048) -> float:
    return (dim * math.log(max_position_embeddings /
                           (num_rotations * 2 * math.pi))) / (2 *
                                                              math.log(base))


# Find dim range bounds based on rotations
def _yarn_find_correction_range(
        low_rot: int,
        high_rot: int,
        dim: int,
        base: float = 10000,
        max_position_embeddings: int = 2048) -> Tuple[int, int]:
    low = math.floor(
        _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(
        _yarn_find_correction_dim(high_rot, dim, base,
                                  max_position_embeddings))
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def _yarn_linear_ramp_mask(low: float, high: float, dim: int,
                           dtype: torch.dtype) -> torch.Tensor:
    if low == high:
        high += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=dtype) - low) / (high - low)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def _yarn_get_mscale(scale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0


class YaRNScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with YaRN method.

    Credits to Peng et al. github.com/jquesnelle/yarn
    """

    def __init__(
            self,
            head_size: int,
            rotary_dim: int,
            max_position_embeddings: int,
            base: int,
            is_neox_style: bool,
            scaling_factor: float,
            dtype: torch.dtype,
            *,
            extrapolation_factor: float = 1,
            attn_factor: float = 1,
            beta_fast: int = 32,
            beta_slow: int = 1,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        # Get n-d magnitude scaling corrected for interpolation
        self.mscale = float(
            _yarn_get_mscale(self.scaling_factor) * attn_factor)
        super().__init__(head_size, rotary_dim, max_position_embeddings, base,
                         is_neox_style, dtype)

    def _compute_inv_freq(self, scaling_factor: float) -> torch.Tensor:
        pos_freqs = self.base ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float) /
                self.rotary_dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

        low, high = _yarn_find_correction_range(self.beta_fast, self.beta_slow,
                                                self.rotary_dim, self.base,
                                                self.max_position_embeddings)
        # Get n-d rotational scaling corrected for extrapolation
        inv_freq_mask = (1 - _yarn_linear_ramp_mask(
            low, high, self.rotary_dim // 2,
            dtype=torch.float)) * self.extrapolation_factor
        inv_freq = inv_freq_interpolation * (
                1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.scaling_factor)
        t = torch.arange(self.max_position_embeddings * self.scaling_factor,
                         dtype=torch.float32)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = (freqs.cos() * self.mscale)
        sin = (freqs.sin() * self.mscale)
        cache = torch.cat((cos, sin), dim=-1)
        return cache


class Phi3LongRoPEScaledRotaryEmbedding(nn.Module):
    """Phi3 family of models scaled rotary embedding.

    Based on the original RotaryEmbedding implementation.
    """

    def __init__(
            self,
            head_size: int,
            rotary_dim: int,
            max_position_embeddings: int,
            original_max_position_embeddings: int,
            base: int,
            is_neox_style: bool,
            dtype: torch.dtype,
            short_factor: List[float],
            long_factor: List[float],
            short_mscale: float = 1.0,
            long_mscale: float = 1.0,
    ):
        super().__init__()

        if rotary_dim != head_size:
            raise ValueError(
                f"`Phi3LongRoPEScaledRotaryEmbedding` does not support \
                    rotary_dim != head_size ({rotary_dim}!={head_size}).")
        if is_neox_style is False:
            raise ValueError(
                "`Phi3LongRoPEScaledRotaryEmbedding` only supports neox_style."
            )

        self.head_size = head_size
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.base = base
        self.short_factor = short_factor
        self.long_factor = long_factor
        self.short_mscale = short_mscale
        self.long_mscale = long_mscale

        scale = (self.max_position_embeddings /
                 self.original_max_position_embeddings)

        if scale <= 1.0:
            self.scaling_factor = 1.0
        else:
            self.scaling_factor = math.sqrt(
                1 + math.log(scale) /
                math.log(self.original_max_position_embeddings))

        short_cache = self._compute_cos_sin_cache(
            original_max_position_embeddings, short_factor, short_mscale)
        short_cache = short_cache.to(dtype)
        self.register_buffer("short_cos_sin_cache",
                             short_cache,
                             persistent=False)

        long_cache = self._compute_cos_sin_cache(max_position_embeddings,
                                                 long_factor, long_mscale)
        long_cache = long_cache.to(dtype)
        self.register_buffer("long_cos_sin_cache",
                             long_cache,
                             persistent=False)

        long_short_cache = torch.cat(
            [self.short_cos_sin_cache, self.long_cos_sin_cache], dim=0)
        self.register_buffer("long_short_cos_sin_cache",
                             long_short_cache,
                             persistent=False)

    def _compute_inv_freq(self, rescale_factors: List[float]) -> torch.Tensor:
        rescale_factors = torch.tensor(rescale_factors, dtype=torch.float32)
        inv_freq = 1.0 / (rescale_factors * (self.base ** (torch.arange(
            0, self.head_size, 2, dtype=torch.float) / self.head_size)))
        return inv_freq

    def _compute_cos_sin_cache(
            self,
            max_position_embeddings: int,
            rescale_factors: List[float],
            mscale: float,
    ) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(rescale_factors)
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * mscale * self.scaling_factor
        sin = freqs.sin() * mscale * self.scaling_factor
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward(
            self,
            positions: torch.Tensor,
            query: torch.Tensor,
            key: torch.Tensor,
            offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query = query.view(*query.shape[:-1], -1, self.head_size)
        key = key.view(*key.shape[:-1], -1, self.head_size)

        k = self.original_max_position_embeddings
        long_prompt_offset = (torch.any(positions > k).float() *
                              torch.full_like(positions, k)).long()
        idx = (torch.add(positions, long_prompt_offset)
               if long_prompt_offset is not None else positions)
        self.long_short_cos_sin_cache: torch.Tensor = (
            self.long_short_cos_sin_cache.to(idx.device))
        idx = torch.add(idx, offsets) if offsets is not None else idx
        cos_sin = torch.index_select(self.long_short_cos_sin_cache, 0, idx)

        cos, sin = cos_sin.chunk(2, dim=-1)
        cos = cos.repeat(1, 2).unsqueeze(-2)
        sin = sin.repeat(1, 2).unsqueeze(-2)

        query = query * cos + _rotate_neox(query) * sin
        key = key * cos + _rotate_neox(key) * sin

        return query.flatten(-2), key.flatten(-2)


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class DeepseekScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with YaRN method.

    Credits to Peng et al. github.com/jquesnelle/yarn
    """

    def __init__(
            self,
            head_size: int,
            rotary_dim: int,
            max_position_embeddings: int,
            base: int,
            is_neox_style: bool,
            scaling_factor: float,
            dtype: torch.dtype,
            *,
            extrapolation_factor: float = 1,
            attn_factor: float = 1,
            beta_fast: int = 32,
            beta_slow: int = 1,
            mscale: float = 1,
            mscale_all_dim: float = 0,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        # Get n-d magnitude scaling corrected for interpolation.
        self.mscale = float(
            yarn_get_mscale(self.scaling_factor, float(mscale)) /
            yarn_get_mscale(self.scaling_factor, float(mscale_all_dim)) *
            attn_factor)
        super().__init__(head_size, rotary_dim, max_position_embeddings, base,
                         is_neox_style, dtype)

    def _compute_inv_freq(self, scaling_factor: float) -> torch.Tensor:
        pos_freqs = self.base ** (torch.arange(
            0, self.rotary_dim, 2, dtype=torch.float, device="cuda") /
                                  self.rotary_dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

        low, high = _yarn_find_correction_range(self.beta_fast, self.beta_slow,
                                                self.rotary_dim, self.base,
                                                self.max_position_embeddings)
        # Get n-d rotational scaling corrected for extrapolation
        inv_freq_mask = (1 - _yarn_linear_ramp_mask(
            low, high, self.rotary_dim // 2,
            dtype=torch.float)) * self.extrapolation_factor
        inv_freq = inv_freq_interpolation * (
                1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.scaling_factor)
        t = torch.arange(self.max_position_embeddings * self.scaling_factor,
                         device="cuda",
                         dtype=torch.float32)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = (freqs.cos() * self.mscale)
        sin = (freqs.sin() * self.mscale)
        cache = torch.cat((cos, sin), dim=-1)
        print("Cache shape", cache.shape)
        return cache

    def forward(
            self,
            positions: torch.Tensor,
            query: torch.Tensor,
            key: torch.Tensor,
            offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        query_rot = query[..., :self.rotary_dim]
        key_rot = key[..., :self.rotary_dim]
        if self.rotary_dim < self.head_size:
            query_pass = query[..., self.rotary_dim:]
            key_pass = key[..., self.rotary_dim:]

        self.cos_sin_cache: torch.Tensor = self.cos_sin_cache.to(
            positions.device)
        cos_sin = self.cos_sin_cache[torch.add(positions, offsets)
        if offsets is not None else positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if self.is_neox_style:
            # NOTE(woosuk): Here we assume that the positions tensor has the
            # shape [batch_size, seq_len].
            cos = cos.repeat(1, 1, 2).unsqueeze(-2)
            sin = sin.repeat(1, 1, 2).unsqueeze(-2)
        else:
            cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
            sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)

        rotate_fn = _rotate_neox if self.is_neox_style else _rotate_gptj
        query_rot = query_rot * cos + rotate_fn(query_rot) * sin
        key_rot = key_rot * cos + rotate_fn(key_rot) * sin

        if self.rotary_dim < self.head_size:
            query = torch.cat((query_rot, query_pass), dim=-1)
            key = torch.cat((key_rot, key_pass), dim=-1)
        else:
            query = query_rot
            key = key_rot
        return query, key


class GemmaRotaryEmbedding(RotaryEmbedding):

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
        # https://github.com/huggingface/transformers/blob/v4.41.2/src/transformers/models/gemma/modeling_gemma.py#L107
        inv_freq = 1.0 / (base ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.int64).float() /
                self.rotary_dim))
        return inv_freq


class DualChunkRotaryEmbedding(CustomOp):
    """Rotary positional embedding for Dual Chunk Attention."""

    def __init__(
            self,
            head_size: int,
            rotary_dim: int,
            max_position_embeddings: int,
            base: int,
            is_neox_style: bool,
            dtype: torch.dtype,
            chunk_size: int,
            local_size: int,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.chunk_size = chunk_size
        self.local_size = local_size
        self.dtype = dtype

        q_cache, qc_cache, k_cache = self._compute_cos_sin_cache()
        q_cache = q_cache.to(dtype)
        qc_cache = qc_cache.to(dtype)
        k_cache = k_cache.to(dtype)
        self.register_buffer("cos_sin_q_cache", q_cache, persistent=False)
        self.register_buffer("cos_sin_qc_cache", qc_cache, persistent=False)
        self.register_buffer("cos_sin_k_cache", k_cache, persistent=False)

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
        """Compute the inverse frequency."""
        # NOTE(woosuk): The HF implementation uses `torch.arange(...).float()`.
        # However, we use `torch.arange(..., dtype=torch.float)` instead to
        # avoid numerical issues with large base values (e.g., 10000000).
        # This may cause a slight numerical difference between the HF
        # implementation and ours.
        # NOTE(woosuk): To exactly match the HF implementation, we need to
        # use CPU to compute the cache and then move it to GPU. However, we
        # create the cache on GPU for faster initialization. This may cause
        # a slight numerical difference between the HF implementation and ours.
        inv_freq = 1.0 / (base ** (torch.arange(
            0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim))
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)

        chunk_len = self.chunk_size - self.local_size
        q_t = torch.arange(chunk_len, dtype=torch.float)
        qc_t = (torch.arange(chunk_len, dtype=torch.float) +
                chunk_len).clamp(max=self.chunk_size)
        k_t = torch.arange(self.max_position_embeddings,
                           dtype=torch.float) % chunk_len

        q_freqs = torch.einsum("i,j -> ij", q_t, inv_freq)
        qc_freqs = torch.einsum("i,j -> ij", qc_t, inv_freq)
        k_freqs = torch.einsum("i,j -> ij", k_t, inv_freq)
        q_cos = q_freqs.cos()
        q_sin = q_freqs.sin()
        qc_cos = qc_freqs.cos()
        qc_sin = qc_freqs.sin()
        k_cos = k_freqs.cos()
        k_sin = k_freqs.sin()
        q_cache = torch.cat((q_cos, q_sin), dim=-1)
        qc_cache = torch.cat((qc_cos, qc_sin), dim=-1)
        k_cache = torch.cat((k_cos, k_sin), dim=-1)
        return q_cache, qc_cache, k_cache

    def forward(
            self,
            positions: torch.Tensor,
            query: torch.Tensor,
            key: torch.Tensor,
            offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        query = query.view(*query.shape[:-1], -1, self.head_size)
        key = key.view(*key.shape[:-1], -1, self.head_size)

        query_rot = query[..., :self.rotary_dim]
        key_rot = key[..., :self.rotary_dim]
        if self.rotary_dim < self.head_size:
            query_pass = query[..., self.rotary_dim:]
            key_pass = key[..., self.rotary_dim:]
        else:
            query_pass = None
            key_pass = None

        self.cos_sin_q_cache: torch.Tensor = self.cos_sin_q_cache.to(
            positions.device, dtype=query.dtype)
        self.cos_sin_qc_cache: torch.Tensor = self.cos_sin_qc_cache.to(
            positions.device, dtype=query.dtype)
        self.cos_sin_k_cache: torch.Tensor = self.cos_sin_k_cache.to(
            positions.device, dtype=query.dtype)

        def apply_rotary_embedding(cos_sin, hidden_rot, hidden_pass):
            cos, sin = cos_sin.chunk(2, dim=-1)
            if self.is_neox_style:
                # NOTE(woosuk): Here we assume that the positions tensor has the
                # shape [batch_size, seq_len].
                cos = cos.repeat(1, 1, 2).unsqueeze(-2)
                sin = sin.repeat(1, 1, 2).unsqueeze(-2)
            else:
                cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
                sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)

            rotate_fn = _rotate_neox if self.is_neox_style else _rotate_gptj
            hidden_rot = hidden_rot * cos + rotate_fn(hidden_rot) * sin
            if self.rotary_dim < self.head_size:
                hidden = torch.cat((hidden_rot, hidden_pass), dim=-1)
            else:
                hidden = hidden_rot
            hidden = hidden.flatten(-2)
            return hidden.squeeze(0)

        key = apply_rotary_embedding(
            self.cos_sin_k_cache[torch.
            add(positions, offsets
                ) if offsets is not None else positions],
            key_rot, key_pass)

        chunk_len = self.chunk_size - self.local_size
        query = apply_rotary_embedding(
            self.cos_sin_q_cache[(torch.add(positions, offsets) if offsets
                                                                   is not None else positions) % chunk_len],
            query_rot, query_pass)
        query_succ = apply_rotary_embedding(
            self.cos_sin_qc_cache[(torch.add(positions, offsets) if offsets
                                                                    is not None else positions) % chunk_len],
            query_rot, query_pass)
        query_inter = apply_rotary_embedding(
            self.cos_sin_qc_cache[chunk_len - 1].repeat(positions.shape[0], 1),
            query_rot, query_pass)

        return query, query_succ, query_inter, key

    def extra_repr(self) -> str:
        s = f"head_size={self.head_size}, rotary_dim={self.rotary_dim}"
        s += f", max_position_embeddings={self.max_position_embeddings}"
        s += f", base={self.base}, is_neox_style={self.is_neox_style}"
        s += f", chunk_size={self.chunk_size}, local_size={self.local_size}"
        return s


class Llama3RotaryEmbedding(RotaryEmbedding):
    """Rotary positional embedding for Dual Chunk Attention."""

    def __init__(
            self,
            head_size: int,
            rotary_dim: int,
            max_position_embeddings: int,
            base: int,
            is_neox_style: bool,
            dtype: torch.dtype,
            shifted_offset: int,
            local_value: int,
            rope_type=None,
            factor=None,
            low_freq_factor=None,
            high_freq_factor=None,
            original_max_position_embeddings=None
    ) -> None:

        # self.mscale = 0.1*math.log(2) + 1
        self.mscale = 1.0
        self.rope_type = rope_type
        self.max_position_embeddings = max_position_embeddings
        self.shifted_offset = shifted_offset
        self.local_value = local_value

        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype

        # yarn config
        self.factor = factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        # cache = self._compute_cos_sin_cache().to(dtype)
        # self.register_buffer("rope_cache", cache, persistent=False)
        super().__init__(head_size, rotary_dim, max_position_embeddings, base,
                         is_neox_style, dtype)

    def _compute_llama3_parameters(self):

        """
        Computes the inverse frequencies for llama 3.1.

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
        # Gets the default RoPE parameters
        inv_freq = 1.0 / (self.base ** (torch.arange(
            0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim))

        factor = self.factor  # `8` in the original implementation
        low_freq_factor = self.low_freq_factor  # `1` in the original implementation
        high_freq_factor = self.high_freq_factor  # `4` in the original implementation
        old_context_len = self.original_max_position_embeddings  # `8192` in the original implementation
        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
        # pdb.set_trace()
        new_freqs = []
        for freq in inv_freq:
            wavelen = 2 * math.pi / freq
            if wavelen < high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                new_freqs.append(freq / factor)
            else:
                assert low_freq_wavelen != high_freq_wavelen
                smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
                new_freqs.append((1 - smooth) * freq / factor + smooth * freq)
        inv_freq = torch.tensor(new_freqs, dtype=inv_freq.dtype, device=inv_freq.device)
        return inv_freq

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
        """Compute the inverse frequency."""
        # NOTE(woosuk): The HF implementation uses `torch.arange(...).float()`.
        # However, we use `torch.arange(..., dtype=torch.float)` instead to
        # avoid numerical issues with large base values (e.g., 10000000).
        # This may cause a slight numerical difference between the HF
        # implementation and ours.
        # NOTE(woosuk): To exactly match the HF implementation, we need to
        # use CPU to compute the cache and then move it to GPU. However, we
        # create the cache on GPU for faster initialization. This may cause
        # a slight numerical difference between the HF implementation and ours.
        if self.rope_type == "llama3":
            inv_freq = self._compute_llama3_parameters()
        else:
            inv_freq = 1.0 / (base ** (torch.arange(
                0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim))
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float32)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = (freqs.cos() * self.mscale)
        sin = (freqs.sin() * self.mscale)
        cache = torch.cat((cos, sin), dim=-1)
        return cache


class ShiftedRotaryEmbedding(CustomOp):
    """Rotary positional embedding for Dual Chunk Attention."""

    def __init__(
            self,
            head_size: int,
            rotary_dim: int,
            max_position_embeddings: int,
            base: int,
            is_neox_style: bool,
            dtype: torch.dtype,
            shifted_offset: int,
            local_value: int,
            rope_type=None,
            factor=None,
            low_freq_factor=None,
            high_freq_factor=None,
            original_max_position_embeddings=None
    ) -> None:
        super().__init__()
        self.mscale = 0.1*math.log(2) + 1
        self.rope_type = rope_type
        self.max_position_embeddings = max_position_embeddings
        self.shifted_offset = shifted_offset
        self.local_value = local_value

        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype

        # yarn config
        self.factor = factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        cache = self._compute_cos_sin_cache().to(dtype)
        self.register_buffer("rope_cache", cache, persistent=False)
        # super().__init__(head_size, rotary_dim, max_position_embeddings, base,
        #                  is_neox_style, dtype)

    def _compute_llama3_parameters(self):

        """
        Computes the inverse frequencies for llama 3.1.

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
        # Gets the default RoPE parameters
        inv_freq = 1.0 / (self.base ** (torch.arange(
            0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim))

        factor = self.factor  # `8` in the original implementation
        low_freq_factor = self.low_freq_factor  # `1` in the original implementation
        high_freq_factor = self.high_freq_factor  # `4` in the original implementation
        old_context_len = self.original_max_position_embeddings  # `8192` in the original implementation
        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
        # pdb.set_trace()
        new_freqs = []
        for freq in inv_freq:
            wavelen = 2 * math.pi / freq
            if wavelen < high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                new_freqs.append(freq / factor)
            else:
                assert low_freq_wavelen != high_freq_wavelen
                smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
                new_freqs.append((1 - smooth) * freq / factor + smooth * freq)
        inv_freq = torch.tensor(new_freqs, dtype=inv_freq.dtype, device=inv_freq.device)
        return inv_freq

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
        """Compute the inverse frequency."""
        # NOTE(woosuk): The HF implementation uses `torch.arange(...).float()`.
        # However, we use `torch.arange(..., dtype=torch.float)` instead to
        # avoid numerical issues with large base values (e.g., 10000000).
        # This may cause a slight numerical difference between the HF
        # implementation and ours.
        # NOTE(woosuk): To exactly match the HF implementation, we need to
        # use CPU to compute the cache and then move it to GPU. However, we
        # create the cache on GPU for faster initialization. This may cause
        # a slight numerical difference between the HF implementation and ours.
        if self.rope_type == "llama3":
            inv_freq = self._compute_llama3_parameters()
        else:
            inv_freq = 1.0 / (base ** (torch.arange(
                0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim))
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float32)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = (freqs.cos() * 1)
        sin = (freqs.sin() * 1)
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward(
            self,
            positions: torch.Tensor,
            query: torch.Tensor,
            key: torch.Tensor,
            offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = query.view(*query.shape[:-1], -1, self.head_size)
        key = key.view(*key.shape[:-1], -1, self.head_size)

        query_rot = query[..., :self.rotary_dim]
        key_rot = key[..., :self.rotary_dim]
        if self.rotary_dim < self.head_size:
            query_pass = query[..., self.rotary_dim:]
            key_pass = key[..., self.rotary_dim:]
        else:
            query_pass = None
            key_pass = None

        self.cos_sin_cache: torch.Tensor = self.rope_cache.to(
            positions.device, dtype=query.dtype)

        def apply_rotary_embedding(cos_sin, hidden_rot, hidden_pass):
            cos, sin = cos_sin.chunk(2, dim=-1)
            if self.is_neox_style:
                # NOTE(woosuk): Here we assume that the positions tensor has the
                # shape [batch_size, seq_len].
                cos = cos.repeat(1, 1, 2).unsqueeze(-2)
                sin = sin.repeat(1, 1, 2).unsqueeze(-2)
            else:
                cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
                sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)

            rotate_fn = _rotate_neox if self.is_neox_style else _rotate_gptj
            hidden_rot = hidden_rot * cos + rotate_fn(hidden_rot) * sin
            if self.rotary_dim < self.head_size:
                hidden = torch.cat((hidden_rot, hidden_pass), dim=-1)
            else:
                hidden = hidden_rot
            hidden = hidden.flatten(-2)
            return hidden.squeeze(0)
        key = apply_rotary_embedding(self.rope_cache[positions], key_rot, key_pass)
        query_diag = apply_rotary_embedding(self.rope_cache[positions], query_rot, query_pass)
        query_shifted = apply_rotary_embedding(self.rope_cache[positions-self.shifted_offset + self.local_value], query_rot, query_pass)
        return query_diag*self.mscale, query_shifted*self.mscale, key*self.mscale

    def extra_repr(self) -> str:
        s = f"head_size={self.head_size}, rotary_dim={self.rotary_dim}"
        s += f", max_position_embeddings={self.max_position_embeddings}"
        s += f", STRING offset S={self.shifted_offset}, Small local value W={self.local_value}"
        return s


_ROPE_DICT: Dict[Tuple, RotaryEmbedding] = {}


def get_rope(
        head_size: int,
        rotary_dim: int,
        max_position: int,
        base: int,
        is_neox_style: bool = True,
        rope_scaling: Optional[Dict[str, Any]] = None,
        dtype: Optional[torch.dtype] = None,
        string_config: Optional[Dict[str, Any]] = None,
) -> RotaryEmbedding:
    if dtype is None:
        dtype = torch.get_default_dtype()
    if rope_scaling is not None:
        # Transforms every value that is a list into a tuple for caching calls
        rope_scaling_tuple = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in rope_scaling.items()
        }
        rope_scaling_args = tuple(rope_scaling_tuple.items())
    else:
        rope_scaling_args = None

    if string_config is not None:
        string_tuple = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in string_config.items()
        }
        string_args = tuple(string_tuple.items())
    else:
        string_args = None

    key = (head_size, rotary_dim, max_position, base, is_neox_style,
           rope_scaling_args, dtype, string_args)
    if key in _ROPE_DICT:
        return _ROPE_DICT[key]
    if rope_scaling is None:
        if string_args is None:
            rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position,
                                         base, is_neox_style, dtype)
        else:
            assert string_config is not None
            extra_kwargs = {
                k: v
                for k, v in string_config.items()
                if k in ("shifted_offset", "local_value")
            }
            rotary_emb = ShiftedRotaryEmbedding(head_size, rotary_dim,
                                                max_position, base,
                                                is_neox_style, dtype,
                                                **extra_kwargs)
    else:
        if "type" in rope_scaling:
            scaling_type = rope_scaling["type"]
        else:
            scaling_type = rope_scaling["rope_type"]
        # The correct one should be "longrope" but keep "su" here
        # for backward compatible
        if scaling_type != "su" and scaling_type != "longrope":
            scaling_factor = rope_scaling["factor"]

        if scaling_type == "llama3":
            extra_kwargs = {
                k: v
                for k, v in rope_scaling.items()
                if k in ("rope_type", "factor", "low_freq_factor",
                         "high_freq_factor", "original_max_position_embeddings")
            }
            if string_args is not None:
                for k, v in string_config.items():
                    extra_kwargs[k] = v
            rotary_emb = ShiftedRotaryEmbedding(head_size, rotary_dim,
                                                max_position, base,
                                                is_neox_style, dtype, **extra_kwargs)
        elif scaling_type == "linear":
            rotary_emb = LinearScalingRotaryEmbedding(head_size, rotary_dim,
                                                      max_position, base,
                                                      is_neox_style,
                                                      scaling_factor, dtype)
        elif scaling_type == "dynamic":
            rotary_emb = DynamicNTKScalingRotaryEmbedding(
                head_size, rotary_dim, max_position, base, is_neox_style,
                scaling_factor, dtype)
        elif scaling_type == "yarn":
            original_max_position = rope_scaling[
                "original_max_position_embeddings"]
            extra_kwargs = {
                k: v
                for k, v in rope_scaling.items()
                if k in ("extrapolation_factor", "attn_factor", "beta_fast",
                         "beta_slow")
            }
            rotary_emb = YaRNScalingRotaryEmbedding(head_size, rotary_dim,
                                                    original_max_position,
                                                    base, is_neox_style,
                                                    scaling_factor, dtype,
                                                    **extra_kwargs)
        elif scaling_type == "deepseek_yarn":
            original_max_position = rope_scaling[
                "original_max_position_embeddings"]
            # assert max_position == original_max_position * scaling_factor
            extra_kwargs = {
                k: v
                for k, v in rope_scaling.items()
                if k in ("extrapolation_factor", "attn_factor", "beta_fast",
                         "beta_slow", "mscale", "mscale_all_dim")
            }
            rotary_emb = DeepseekScalingRotaryEmbedding(
                head_size, rotary_dim, original_max_position, base,
                is_neox_style, scaling_factor, dtype, **extra_kwargs)
        # The correct one should be "longrope" but keep "su" here
        # for backward compatible
        elif scaling_type == "su" or scaling_type == "longrope":
            short_factor = rope_scaling["short_factor"]
            long_factor = rope_scaling["long_factor"]
            original_max_position = rope_scaling[
                "original_max_position_embeddings"]
            extra_kwargs = {
                k: v
                for k, v in rope_scaling.items()
                if k in ("short_mscale", "long_mscale")
            }
            rotary_emb = Phi3LongRoPEScaledRotaryEmbedding(
                head_size, rotary_dim, max_position, original_max_position,
                base, is_neox_style, dtype, short_factor, long_factor,
                **extra_kwargs)
        else:
            raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
    _ROPE_DICT[key] = rotary_emb
    return rotary_emb
