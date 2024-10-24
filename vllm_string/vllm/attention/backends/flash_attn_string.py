"""Attention layer with Flash and PagedAttention.

NOTE(woosuk): At the moment, this file includes a lot of duplicated code from
XFormers backend. The duplicated code will be removed once we use flash-attn or
flashinfer for all the attention operations.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, Union
import pdb
import torch
from vllm_flash_attn import flash_attn_varlen_func, flash_attn_func
import vllm_flash_attn_2_cuda as flash_attn_cuda

from vllm import _custom_ops as ops
from vllm.attention.backends.abstract import (StringAttentionBackend,
                                              StringAttentionImpl,
                                              StringAttentionMetadata)



def lower_multiple_tensor(tensor, block_size):
    return (tensor // block_size) * block_size



def flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        k=None,
        v=None,
        rotary_cos=None,
        rotary_sin=None,
        cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
        cache_batch_idx: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite context window
        rotary_interleaved=True,
        alibi_slopes=None,
        num_splits=0,
        *,
        out=None,
):
    """
    If k and v are not None, k_cache and v_cache will be updated *inplace* with the new values from
    k and v. This is useful for incremental decoding: you can pass in the cached keys/values from
    the previous step, and update them with the new keys/values from the current step, and do
    attention with the updated cache, all in 1 kernel.

    If you pass in k / v, you must make sure that the cache is large enough to hold the new values.
    For example, the KV cache could be pre-allocated with the max sequence length, and you can use
    cache_seqlens to keep track of the current sequence lengths of each sequence in the batch.

    Also apply rotary embedding if rotary_cos and rotary_sin are passed in. The key @k will be
    rotated by rotary_cos and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If causal or local (i.e., window_size != (-1, -1)), the query @q will be rotated by rotary_cos
    and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If not causal and not local, the query @q will be rotated by rotary_cos and rotary_sin at
    indices cache_seqlens only (i.e. we consider all tokens in @q to be at position cache_seqlens).

    See tests/test_flash_attn.py::test_flash_attn_kvcache for examples of how to use this function.

    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Note: Does not support backward pass.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
            or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)
            page_block_size must be a multiple of 256.
        v_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
            or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)
        k [optional]: (batch_size, seqlen_new, nheads_k, headdim). If not None, we concatenate
            k with k_cache, starting at the indices specified by cache_seqlens.
        v [optional]: (batch_size, seqlen_new, nheads_k, headdim). Similar to k.
        rotary_cos [optional]: (seqlen_ro, rotary_dim / 2). If not None, we apply rotary embedding
            to k and q. Only applicable if k and v are passed in. rotary_dim must be divisible by 16.
        rotary_sin [optional]: (seqlen_ro, rotary_dim / 2). Similar to rotary_cos.
        cache_seqlens: int, or (batch_size,), dtype torch.int32. The sequence lengths of the
            KV cache.
        block_table [optional]: (batch_size, max_num_blocks_per_seq), dtype torch.int32.
        cache_batch_idx: (batch_size,), dtype torch.int32. The indices used to index into the KV cache.
            If None, we assume that the batch indices are [0, 1, 2, ..., batch_size - 1].
            If the indices are not distinct, and k and v are provided, the values updated in the cache
                 might come from any of the duplicate indices.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        rotary_interleaved: bool. Only applicable if rotary_cos and rotary_sin are passed in.
            If True, rotary embedding will combine dimensions 0 & 1, 2 & 3, etc. If False,
            rotary embedding will combine dimensions 0 & rotary_dim / 2, 1 & rotary_dim / 2 + 1
            (i.e. GPT-NeoX style).
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        num_splits: int. If > 1, split the key/value into this many chunks along the sequence.
           If num_splits == 1, we don't split the key/value. If num_splits == 0, we use a heuristic
           to automatically determine the number of splits.
           Don't change this unless you know what you are doing.

    Return:
        out: (batch_size, seqlen, nheads, headdim).
    """
    assert k_cache.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_cache.stride(-1) == 1, "v_cache must have contiguous last dimension"
    maybe_contiguous = lambda x: x.contiguous() if x is not None and x.stride(-1) != 1 else x
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if cache_seqlens is not None and isinstance(cache_seqlens, int):
        cache_seqlens = torch.full(
            (k_cache.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device
        )
        cache_seqlens = maybe_contiguous(cache_seqlens)
    cache_batch_idx = maybe_contiguous(cache_batch_idx)
    block_table = maybe_contiguous(block_table)
    out, softmax_lse = flash_attn_cuda.fwd_kvcache(
        q,
        k_cache,
        v_cache,
        k,
        v,
        cache_seqlens,
        rotary_cos,
        rotary_sin,
        cache_batch_idx,
        block_table,
        alibi_slopes,
        out,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        rotary_interleaved,
        num_splits,
    )
    return out, softmax_lse


class StringFlashAttentionBackend(StringAttentionBackend):

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "dual-chunk-flash-attn"

    @staticmethod
    def get_impl_cls() -> Type["StringFlashAttentionImpl"]:
        return StringFlashAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type["StringAttentionMetadata"]:
        return StringFlashAttentionMetadata

    @staticmethod
    def swap_blocks(
            src_kv_cache: torch.Tensor,
            dst_kv_cache: torch.Tensor,
            src_to_dst: torch.Tensor,
    ) -> None:
        src_key_cache = src_kv_cache[0]
        dst_key_cache = dst_kv_cache[0]
        ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dst)

        src_value_cache = src_kv_cache[1]
        dst_value_cache = dst_kv_cache[1]
        ops.swap_blocks(src_value_cache, dst_value_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
            kv_caches: List[torch.Tensor],
            src_to_dists: torch.Tensor,
    ) -> None:
        key_caches = [kv_cache[0] for kv_cache in kv_caches]
        value_caches = [kv_cache[1] for kv_cache in kv_caches]
        ops.copy_blocks(key_caches, value_caches, src_to_dists)


@dataclass
class StringFlashAttentionMetadata(StringAttentionMetadata):
    """Metadata for StringFlashAttentionBackend.

    NOTE: Any python object stored here is not updated when it is
    cuda-graph replayed. If you have values that need to be changed
    dynamically, it should be stored in tensor. The tensor has to be
    updated from `CUDAGraphRunner.forward` API.
    """

    # (batch_size,). The sequence length per sequence. Sequence length means
    # the computed tokens + new tokens None if it is a decoding.
    seq_lens: Optional[List[int]]
    # seq_lens stored as a tensor.
    seq_lens_tensor: Optional[torch.Tensor]

    # (batch_size,). The original prefill length per sequence.
    # None if it is decoding.
    prefill_original_seq_lens_tensor: Optional[torch.Tensor]

    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ----------------------|
    #                                   |-- query_len ---|

    # Maximum query length in the batch. None for decoding.
    max_query_len: Optional[int]
    # Maximum sequence length among prefill batch. 0 if there are decoding
    # requests only.
    max_prefill_seq_len: int
    # Maximum sequence length among decode batch. 0 if there are prefill
    # requests only.
    max_decode_seq_len: int
    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    query_start_loc: Optional[torch.Tensor]
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor]
    # (batch_size,) A tensor of context lengths (tokens that are computed
    # so far).
    context_lens_tensor: Optional[torch.Tensor]

    # (batch_size, max_blocks_per_seq).
    # Block addresses per sequence. (Seq id -> list of physical block)
    # E.g., [0, 1, 2] means tokens are stored in 0th, 1st, and 2nd blocks
    # in the kv cache. Each block can contain up to block_size tokens.
    # 2nd dimensions are padded up to max_blocks_per_seq if it is cuda-graph
    # captured.
    block_tables: Optional[torch.Tensor]

    # Whether or not if cuda graph is enabled.
    # Cuda-graph is currently enabled for decoding only.
    # TODO(woosuk): Move `use_cuda_graph` out since it's unrelated to attention.
    use_cuda_graph: bool

    _cached_prefill_metadata: Optional[
        "StringFlashAttentionMetadata"] = None
    _cached_decode_metadata: Optional["StringFlashAttentionMetadata"] = None

    @property
    def prefill_metadata(self) -> Optional["StringFlashAttentionMetadata"]:
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            return self._cached_prefill_metadata

        assert self.seq_lens is not None
        assert self.seq_lens_tensor is not None
        assert self.prefill_original_seq_lens_tensor is not None
        assert self.query_start_loc is not None
        assert self.context_lens_tensor is not None
        assert self.block_tables is not None
        assert self.seq_start_loc is not None

        self._cached_prefill_metadata = StringFlashAttentionMetadata(
            num_prefills=self.num_prefills,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=0,
            slot_mapping=self.slot_mapping[:self.num_prefill_tokens],
            seq_lens=self.seq_lens[:self.num_prefills],
            seq_lens_tensor=self.seq_lens_tensor[:self.num_prefills],
            prefill_original_seq_lens_tensor=self.
                                             prefill_original_seq_lens_tensor[:self.num_prefill_tokens],
            max_query_len=self.max_query_len,
            max_prefill_seq_len=self.max_prefill_seq_len,
            max_decode_seq_len=0,
            query_start_loc=self.query_start_loc[:self.num_prefills + 1],
            seq_start_loc=self.seq_start_loc[:self.num_prefills + 1],
            context_lens_tensor=self.context_lens_tensor[:self.num_prefills],
            block_tables=self.block_tables[:self.num_prefills],
            use_cuda_graph=False,
        )
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> Optional["StringFlashAttentionMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            return self._cached_decode_metadata
        assert self.block_tables is not None
        assert self.seq_lens_tensor is not None

        self._cached_decode_metadata = StringFlashAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=self.num_decode_tokens,
            slot_mapping=self.slot_mapping[self.num_prefill_tokens:],
            seq_lens=None,
            seq_lens_tensor=self.seq_lens_tensor[self.num_prefills:],
            prefill_original_seq_lens_tensor=None,
            max_query_len=None,
            max_prefill_seq_len=0,
            max_decode_seq_len=self.max_decode_seq_len,
            query_start_loc=None,
            seq_start_loc=None,
            context_lens_tensor=None,
            block_tables=self.block_tables[self.num_prefills:],
            use_cuda_graph=self.use_cuda_graph,
        )
        return self._cached_decode_metadata


class StringFlashAttentionImpl(StringAttentionImpl):
    """
    If the input tensors contain prompt tokens, the layout is as follows:
    |<--------------- num_prefill_tokens ----------------->|
    |<--prefill_0-->|<--prefill_1-->|...|<--prefill_N-1--->|

    Otherwise, the layout is as follows:
    |<----------------- num_decode_tokens ------------------>|
    |<--decode_0-->|..........|<--decode_M-1-->|<--padding-->|

    Generation tokens can contain padding when cuda-graph is used.
    Currently, prompt tokens don't contain any padding.

    The prompts might have different lengths, while the generation tokens
    always have length 1.

    If chunked prefill is enabled, prefill tokens and decode tokens can be
    batched together in a flattened 1D query.

    |<----- num_prefill_tokens ---->|<------- num_decode_tokens --------->|
    |<-prefill_0->|...|<-prefill_N-1->|<--decode_0-->|...|<--decode_M-1-->|

    Currently, cuda graph is disabled for chunked prefill, meaning there's no
    padding between prefill and decode tokens.
    """

    def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float,
            num_kv_heads: int,
            alibi_slopes: Optional[List[float]],
            sliding_window: Optional[int],
            kv_cache_dtype: str,
            blocksparse_params: Optional[Dict[str, Any]] = None,
            string_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if blocksparse_params is not None:
            raise ValueError(
                "FlashAttention does not support block-sparse attention.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = ((sliding_window, sliding_window)
                               if sliding_window is not None else (-1, -1))
        self.kv_cache_dtype = kv_cache_dtype

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        if sliding_window is not None:
            # NOTE(woosuk): flash-attn's sliding window does not work with
            # paged KV cache.
            raise ValueError(
                "Sliding window is not supported in FlashAttention.")

        support_head_sizes = (
            StringFlashAttentionBackend.get_supported_head_sizes())
        if head_size not in support_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by FlashAttention. "
                f"Supported head sizes are: {support_head_sizes}.")

        assert string_config is not None
        self.shifted_offset = string_config.get("shifted_offset", 42000)
        self.local_value = string_config.get("local_value", 128)
        self.original_max_position_embeddings = string_config.get(
            "original_max_position_embeddings", 0)

    def forward(
            self,
            query_diag: torch.Tensor,
            query_shifted: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: StringFlashAttentionMetadata,
            kv_scale: float = 1.0,
    ) -> torch.Tensor:
        """Forward pass with StringFlashAttention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            query_succ: shape = [num_tokens, num_heads * head_size]
            query_inter: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size, num_kv_heads * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        # NOTE(woosuk): FlashAttention does not support FP8 KV cache.
        assert kv_scale == 1.0, "kv_scale is not supported in FlashAttention."

        num_tokens, hidden_size = query_diag.shape
        # Reshape the query, key, and value tensors.
        query_diag = query_diag.view(-1, self.num_heads, self.head_size)
        query_shifted = query_shifted.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        if self.original_max_position_embeddings > 0:
            if prefill_meta := attn_metadata.prefill_metadata:
                assert prefill_meta.query_start_loc is not None
                assert prefill_meta.prefill_original_seq_lens_tensor is not None

                current_start = 0
                query_start_loc_cpu = prefill_meta.query_start_loc.cpu()
                for i in range(
                        0, prefill_meta.prefill_original_seq_lens_tensor.
                                shape[0]):
                    current_end = (current_start +
                                   (query_start_loc_cpu[i + 1] -
                                    query_start_loc_cpu[i]).item())
                    current_start = current_end

        if kv_cache is not None:
            key_cache = kv_cache[0]
            value_cache = kv_cache[1]

            # Reshape the input keys and values and store them in the cache.
            # If kv_cache is not provided, the new key and value tensors are
            # not cached. This happens during the initial memory profiling run.
            ops.reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping.flatten(),
                self.kv_cache_dtype,
            )

        num_prefill_tokens = attn_metadata.num_prefill_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens
        assert key.shape[0] == num_prefill_tokens + num_decode_tokens
        assert value.shape[0] == num_prefill_tokens + num_decode_tokens

        output = torch.empty_like(query_diag)
        # Query for decode. KV is not needed because it is already cached.
        decode_query_diag = query_diag[num_prefill_tokens:]
        decode_query_shifted = query_shifted[num_prefill_tokens:]
        # QKV for prefill.
        query_diag = query_diag[:num_prefill_tokens]
        query_shifted = query_shifted[:num_prefill_tokens]
        key = key[:num_prefill_tokens]
        value = value[:num_prefill_tokens]

        assert query_diag.shape[0] == num_prefill_tokens
        assert decode_query_diag.shape[0] == num_decode_tokens

        if prefill_meta := attn_metadata.prefill_metadata:
            # Prompt run.
            if (kv_cache is None or prefill_meta.block_tables is None
                    or prefill_meta.block_tables.numel() == 0):
                # normal attention
                # When block_tables are not filled, it means q and k are the
                # prompt, and they have the same length.
                out = self._bruteforce_dynamic_chunk_flash_attn_varlen_func(
                    q_diag=query_diag,
                    q_shifted=query_shifted,
                    k=key,
                    v=value,
                    cu_seqlens_q=prefill_meta.seq_start_loc,
                    cu_seqlens_k=prefill_meta.seq_start_loc,
                    max_seqlen_q=prefill_meta.max_prefill_seq_len,
                    max_seqlen_k=prefill_meta.max_prefill_seq_len,
                    softmax_scale=self.scale,
                    causal=True,
                    window_size=self.sliding_window,
                    alibi_slopes=self.alibi_slopes,
                    block_table=None,
                    shifted_offset=self.shifted_offset,
                    local_value=self.local_value,
                    original_max_position_embeddings=self.
                    original_max_position_embeddings,
                    prefill_original_seq_lens_tensor=prefill_meta.
                    prefill_original_seq_lens_tensor,
                )
                assert output[:num_prefill_tokens].shape == out.shape
                output[:num_prefill_tokens] = out
            else:
                # prefix-enabled attention
                assert prefill_meta.seq_lens is not None
                max_seq_len = max(prefill_meta.seq_lens)
                output[:num_prefill_tokens] = (
                    self._bruteforce_dynamic_chunk_flash_attn_varlen_func(
                        q_diag=query_diag,
                        q_shifted=query_shifted,
                        k=key_cache,
                        v=value_cache,
                        cu_seqlens_q=prefill_meta.query_start_loc,
                        max_seqlen_q=prefill_meta.max_query_len,
                        cu_seqlens_k=prefill_meta.seq_start_loc,
                        max_seqlen_k=max_seq_len,
                        softmax_scale=self.scale,
                        causal=True,
                        window_size=(-1, -1),
                        alibi_slopes=self.alibi_slopes,
                        block_table=prefill_meta.block_tables,
                        shifted_offset=self.shifted_offset,
                        local_value=self.local_value,
                        original_max_position_embeddings=self.
                        original_max_position_embeddings,
                        prefill_original_seq_lens_tensor=prefill_meta.
                        prefill_original_seq_lens_tensor,
                    ))
        if decode_meta := attn_metadata.decode_metadata:
            # Decoding run.
            output[num_prefill_tokens:] = (
                self._bruteforce_dynamic_chunk_pageattention_forward_decode(
                    decode_query_diag.unsqueeze(1),
                    decode_query_shifted.unsqueeze(1),
                    key_cache,
                    value_cache,
                    block_table=decode_meta.block_tables,
                    cache_seqlens=decode_meta.seq_lens_tensor,
                    softmax_scale=self.scale,
                    causal=True,
                    alibi_slopes=self.alibi_slopes,
                    shifted_offset=self.shifted_offset,
                    local_value=self.local_value,
                    original_max_position_embeddings=self.
                    original_max_position_embeddings,
                ).squeeze(1))

        # Reshape the output tensor.
        return output.view(num_tokens, hidden_size)

    def _bruteforce_dynamic_chunk_flash_attn_func(
            self,
            q_diag,
            q_shifted,
            k,
            v,
            block_table,
            softmax_scale,
            shifted_offset,
            local_value,
            original_max_position_embeddings,
            current_prefill_original_seq_lens_tensor,
            k_length,
    ):

        def do_flash_attn(
                query_states,
                key_states,
                value_states,
                causal=True,
                block_table=None,
                max_seqlen_k=None,
                sliding_window=None
        ):
            if sliding_window is None:
                sliding_window = [-1, -1]
            if max_seqlen_k is None:
                max_seqlen_k = key_states.shape[0]

            output, softmax_lse, _ = flash_attn_varlen_func(
                q=query_states,
                k=key_states,
                v=value_states,
                window_size = sliding_window,
                softmax_scale=softmax_scale,
                cu_seqlens_q=torch.tensor(
                    [0, query_states.shape[0]],
                    dtype=torch.int32,
                    device=query_states.device,
                ),
                max_seqlen_q=query_states.shape[0],
                cu_seqlens_k=torch.tensor(
                    [0, max_seqlen_k],
                    dtype=torch.int32,
                    device=query_states.device,
                ),
                max_seqlen_k=max_seqlen_k,
                causal=causal,
                block_table=block_table,
                return_attn_probs=True,
            )
            return output, softmax_lse

        def get_block(begin, end):
            return block_table[:, begin // block_size:(end - 1) // block_size + 1]

        if block_table is not None:
            block_size = v.shape[1]
        else:
            block_size = 1
        # sliding window attention
        diag_len = self.shifted_offset
        triangle_len = k_length - diag_len

        diag_out, diag_lse = do_flash_attn(q_diag, k,v, causal=True, block_table=block_table, sliding_window=[diag_len, 0])
        if triangle_len > 0:
            q_shifted = q_shifted[-triangle_len:]
            k_shifted = k[:triangle_len]
            v_shifted = v[:triangle_len]

            shifted_out, shifted_lse = do_flash_attn(q_shifted, k_shifted, v_shifted, causal=True, block_table=block_table)
            diag_lse = diag_lse.to(torch.float32).squeeze(0) # head x tokens
            shifted_lse = shifted_lse.to(torch.float32).squeeze(0) # head x tokens
            diag_out_head = diag_out[:self.shifted_offset]
            diag_lse_tail = diag_lse[:, self.shifted_offset:]
            diag_out_tail = diag_out[self.shifted_offset:]

            lse_gap = 1 / (1 + torch.exp(diag_lse_tail - shifted_lse))
            lse_gap_re = 1 / (1 + torch.exp(shifted_lse - diag_lse_tail))
            lse_gap = lse_gap.transpose(0, 1).unsqueeze(-1)
            lse_gap_re = lse_gap_re.transpose(0, 1).unsqueeze(-1)
            merge_out_tail = diag_out_tail * lse_gap_re.to(diag_out_tail) + shifted_out * lse_gap.to(shifted_out)
            output = torch.cat([diag_out_head, merge_out_tail], dim=0)
            return output
        else:
            return diag_out

    def _bruteforce_dynamic_chunk_flash_attn_varlen_func(
            self,
            q_diag,
            q_shifted,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale,
            causal,
            window_size,
            alibi_slopes,
            block_table,
            shifted_offset,
            local_value,
            original_max_position_embeddings,
            prefill_original_seq_lens_tensor,
    ):
        cu_seqlens_q_cpu = cu_seqlens_q.cpu().tolist()
        cu_seqlens_k_cpu = cu_seqlens_k.cpu().tolist()

        all_outputs = []
        for i in range(0, len(cu_seqlens_q_cpu) - 1):
            qs = cu_seqlens_q_cpu[i]
            qe = cu_seqlens_q_cpu[i:i + 2][-1]
            ks = cu_seqlens_k_cpu[i]
            ke = cu_seqlens_k_cpu[i:i + 2][-1]

            current_q_diag = q_diag[qs:qe]
            current_q_shifted = q_shifted[qs:qe]
            if block_table is None:
                current_k = k[ks:ke]
                current_v = v[ks:ke]
                current_block_table = None
                current_prefill_original_seq_lens_tensor = (
                    prefill_original_seq_lens_tensor[i:i + 1])
            else:
                current_block_table = block_table[i:i + 1]
                current_prefill_original_seq_lens_tensor = (
                    prefill_original_seq_lens_tensor[i:i + 1])
                current_k = k
                current_v = v

            if current_q_diag.shape[0] == 0:
                continue
            if current_k.shape[0] == 0:
                all_outputs.append(
                    torch.zeros(
                        (current_q_diag.shape[0], current_q_diag.shape[1], v.shape[2]),
                        device=q_diag.device,
                        dtype=q_diag.dtype,
                    ))
                continue

            current_output = self._bruteforce_dynamic_chunk_flash_attn_func(
                current_q_diag,
                current_q_shifted,
                current_k,
                current_v,
                current_block_table,
                softmax_scale,
                shifted_offset,
                local_value,
                original_max_position_embeddings,
                current_prefill_original_seq_lens_tensor,
                ke - ks,
            )
            all_outputs.append(current_output)

        return torch.cat(all_outputs, dim=0)

    def _bruteforce_dynamic_chunk_pageattention_forward_decode(
            self,
            query_diag: torch.Tensor,
            query_shifted: torch.Tensor,
            key_cache: torch.Tensor,
            value_cache: torch.Tensor,
            block_table: torch.Tensor,
            cache_seqlens: torch.Tensor,
            softmax_scale: float,
            causal: bool,
            alibi_slopes: Optional[torch.Tensor],
            shifted_offset: int,
            local_value: int,
            original_max_position_embeddings: int,
    ):
        assert causal
        batch_size = block_table.shape[0]
        block_size = value_cache.shape[1]

        outputs_list = []
        softmax_lses_list = []

        # diag-attention
        temp = torch.min(torch.full_like(cache_seqlens, self.shifted_offset), cache_seqlens)
        seq_lens_triangle = lower_multiple_tensor(cache_seqlens - temp, block_size)
        seq_lens_diag = cache_seqlens - seq_lens_triangle
        max_seq_len_diag = seq_lens_diag.max().item()
        block_table_diag = torch.zeros(
            batch_size,
            (max_seq_len_diag - 1) // block_size + 1,
            dtype=block_table.dtype,
            device=block_table.device,
        )
        for i in range(batch_size):
            st = (cache_seqlens[i]-seq_lens_diag[i]) // block_size
            ed = min(
                st + (max_seq_len_diag - 1) // block_size + 1,
                (cache_seqlens[i] - 1) // block_size + 1,
            )
            # pdb.set_trace()
            block_table_diag[i, :ed - st] = block_table[i, st:ed]
        diag_output, diag_softmax_lse = (
            self._pagedattention_forward_decode_with_exp_sums(
                query_diag,
                key_cache,
                value_cache,
                block_table_diag,
                seq_lens_diag,
                softmax_scale,
                alibi_slopes,
                causal=False,
            ))
        outputs_list.append(diag_output)
        softmax_lses_list.append(diag_softmax_lse)

        seq_lens_shifted = cache_seqlens - seq_lens_diag
        max_seq_len_shifted = seq_lens_shifted.max().item()
        if max_seq_len_shifted:
            shifted_output, succ_softmax_lse = (
                self._pagedattention_forward_decode_with_exp_sums(
                    query_shifted,
                    key_cache,
                    value_cache,
                    block_table[:, :max_seq_len_shifted],
                    seq_lens_shifted,
                    softmax_scale,
                    alibi_slopes,
                    causal=False,
                ))
            outputs_list.append(shifted_output)
            softmax_lses_list.append(succ_softmax_lse)

        outputs = torch.stack(outputs_list, dim=0)

        del outputs_list
        softmax_lses = torch.stack(softmax_lses_list, dim=0).to(torch.float32)
        del softmax_lses_list

        max_logits = torch.max(softmax_lses, dim=0).values
        stable_logits = softmax_lses - max_logits.unsqueeze(0)
        lse_s = torch.exp(stable_logits).detach()
        lse_sum = torch.sum(lse_s, dim=0)
        lse_s /= lse_sum
        outputs *= lse_s.unsqueeze(-1).transpose(2, 3)

        return outputs.sum(0)

    def _pagedattention_forward_decode_with_exp_sums(
            self,
            query: torch.Tensor,
            key_cache: torch.Tensor,
            value_cache: torch.Tensor,
            block_table: torch.Tensor,
            cache_seqlens: torch.Tensor,
            softmax_scale: float,
            alibi_slopes: Optional[torch.Tensor],
            causal: bool,
    ):

        out, softmax_lse = flash_attn_with_kvcache(
            query,
            key_cache,
            value_cache,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            softmax_scale=softmax_scale,
            alibi_slopes=alibi_slopes,
            causal=causal,
        )
        cache_seqlens_cpu = cache_seqlens.cpu()
        for i in range(cache_seqlens.shape[0]):
            if cache_seqlens_cpu[i] == 0:
                softmax_lse[i].fill_(-float("inf"))
                out[i].fill_(0)

        return out, softmax_lse
