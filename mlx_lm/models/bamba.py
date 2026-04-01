# Copyright © 2025 Apple Inc.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import ArraysCache, KVCache
from .rope_utils import initialize_rope
from .ssm import ssm_update


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "bamba"
    vocab_size: int = 128256
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-5
    mamba_n_heads: int = 128
    mamba_d_head: int = 64
    mamba_n_groups: int = 1
    mamba_d_state: int = 128
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_conv_bias: bool = True
    mamba_proj_bias: bool = False
    attn_layer_indices: Optional[List[int]] = None
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict] = None
    partial_rotary_factor: float = 0.5
    rope_parameters: Optional[Dict] = None
    tie_word_embeddings: bool = False
    time_step_limit: Tuple[float, float] = (0.0, float("inf"))
    time_step_min: Optional[float] = None
    time_step_max: Optional[float] = None

    def __post_init__(self):
        if self.attn_layer_indices is None:
            self.attn_layer_indices = []
        if self.rope_parameters is not None:
            self.rope_theta = self.rope_parameters.get("rope_theta", self.rope_theta)
            self.rope_scaling = self.rope_parameters.get("rope_scaling", self.rope_scaling)
        if (
            self.time_step_limit == (0.0, float("inf"))
            and self.time_step_min is not None
            and self.time_step_max is not None
        ):
            self.time_step_limit = (self.time_step_min, self.time_step_max)
        self.layers_block_type = [
            "attention" if i in self.attn_layer_indices else "mamba"
            for i in range(self.num_hidden_layers)
        ]


class BambaRMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones(hidden_size)
        self.eps = eps

    def __call__(self, x: mx.array, gate: mx.array = None) -> mx.array:
        if gate is not None:
            x = swiglu(gate, x)
        return mx.fast.rms_norm(x, self.weight, eps=self.eps)


class BambaMamba2Mixer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_heads = args.mamba_n_heads
        self.hidden_size = args.hidden_size
        self.ssm_state_size = args.mamba_d_state
        self.conv_kernel_size = args.mamba_d_conv
        self.intermediate_size = args.mamba_expand * args.hidden_size
        self.n_groups = args.mamba_n_groups
        self.head_dim = args.mamba_d_head
        self.time_step_limit = args.time_step_limit

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size

        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=args.mamba_d_conv,
            padding=0,
            groups=self.conv_dim,
            bias=args.mamba_conv_bias,
        )

        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(
            self.hidden_size, projection_size, bias=args.mamba_proj_bias
        )

        self.dt_bias = mx.ones(self.num_heads)
        self.A_log = mx.log(mx.arange(1, self.num_heads + 1, dtype=mx.float32))
        self.D = mx.ones(self.num_heads)

        self.norm = BambaRMSNormGated(self.intermediate_size, eps=args.rms_norm_eps)
        self.out_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=args.mamba_proj_bias
        )

    def _conv(
        self,
        conv_input: mx.array,
        cache: Optional[ArraysCache],
        mask: Optional[mx.array],
    ) -> mx.array:
        if mask is not None:
            conv_input = mx.where(mask[..., None], conv_input, 0)

        if cache is not None:
            if cache[0] is None:
                conv_state = mx.zeros(
                    (conv_input.shape[0], self.conv_kernel_size - 1, self.conv_dim),
                    dtype=conv_input.dtype,
                )
            else:
                conv_state = cache[0]
            padded_input = mx.concatenate([conv_state, conv_input], axis=1)
            n_keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                t = padded_input.shape[1]
                ends = mx.clip(cache.lengths, 0, t - n_keep)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(padded_input, positions, axis=1)
            else:
                cache[0] = padded_input[:, -n_keep:, :]
        else:
            padded_input = mx.pad(
                conv_input, [(0, 0), (self.conv_kernel_size - 1, 0), (0, 0)]
            )

        conv_output = self.conv1d(padded_input)
        return nn.silu(conv_output)

    def _ssm(
        self,
        hidden_states: mx.array,
        B: mx.array,
        C: mx.array,
        dt: mx.array,
        cache: Optional[ArraysCache],
        mask: Optional[mx.array],
    ) -> mx.array:
        batch_size, seq_len, _ = hidden_states.shape

        hidden_states = hidden_states.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        B = B.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size)
        C = C.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size)
        if cache:
            state = cache[1]
            lengths = cache.lengths
        else:
            state, lengths = None, None

        y, state = ssm_update(
            hidden_states,
            self.A_log,
            B,
            C,
            self.D.astype(hidden_states.dtype),
            dt,
            self.dt_bias,
            state,
            self.time_step_limit,
            mask=mask,
            lengths=lengths,
        )
        if cache:
            cache[1] = state

        return y.reshape(batch_size, seq_len, self.intermediate_size)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array],
        cache: Optional[ArraysCache] = None,
    ) -> mx.array:
        projected = self.in_proj(hidden_states)

        gate, conv_input, dt = mx.split(
            projected,
            [self.intermediate_size, self.intermediate_size + self.conv_dim],
            axis=-1,
        )
        conv_output = self._conv(conv_input, cache, mask)
        hidden_states_ssm, B, C = mx.split(
            conv_output,
            [
                self.intermediate_size,
                self.intermediate_size + self.n_groups * self.ssm_state_size,
            ],
            axis=-1,
        )
        y = self._ssm(hidden_states_ssm, B, C, dt, cache, mask)
        if cache:
            cache.advance(y.shape[1])
        y = self.norm(y, gate)
        return self.out_proj(y)


class BambaAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        rope_dims = int(self.head_dim * args.partial_rotary_factor)
        self.rope = initialize_rope(
            dims=rope_dims,
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries = self.q_proj(x).reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys = (
            self.k_proj(x)
            .reshape(B, L, self.num_key_value_heads, -1)
            .transpose(0, 2, 1, 3)
        )
        values = (
            self.v_proj(x)
            .reshape(B, L, self.num_key_value_heads, -1)
            .transpose(0, 2, 1, 3)
        )

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class BambaMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class BambaDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_type: str):
        super().__init__()
        self.layer_type = layer_type
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.pre_ff_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.feed_forward = BambaMLP(args)

        if layer_type == "mamba":
            self.mamba = BambaMamba2Mixer(args)
        else:
            self.self_attn = BambaAttention(args)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        residual = x
        hidden_states = self.input_layernorm(x)

        if self.layer_type == "mamba":
            hidden_states = self.mamba(hidden_states, mask=mask, cache=cache)
        else:
            hidden_states = self.self_attn(hidden_states, mask=mask, cache=cache)

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class BambaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            BambaDecoderLayer(args, layer_type=lt)
            for lt in args.layers_block_type
        ]
        self.final_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        hidden_states = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        # Find first attention and first mamba layer for mask creation
        attn_cache = None
        ssm_cache = None
        for layer, c in zip(self.layers, cache):
            if layer.layer_type == "attention" and attn_cache is None:
                attn_cache = c
            elif layer.layer_type == "mamba" and ssm_cache is None:
                ssm_cache = c

        attn_mask = create_attention_mask(hidden_states, attn_cache)
        ssm_mask = create_ssm_mask(hidden_states, ssm_cache)

        for layer, c in zip(self.layers, cache):
            mask = attn_mask if layer.layer_type == "attention" else ssm_mask
            hidden_states = layer(hidden_states, mask=mask, cache=c)

        return self.final_layernorm(hidden_states)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model = BambaModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.model_type = args.model_type

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        out = self.model(inputs, cache=cache)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for lt in self.args.layers_block_type:
            if lt == "mamba":
                caches.append(ArraysCache(size=2))
            else:
                caches.append(KVCache())
        return caches

    def sanitize(self, weights):
        for k, v in weights.items():
            if "conv1d.weight" in k and v.ndim == 3 and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)
        return {
            k: v
            for k, v in weights.items()
            if "rotary_emb.inv_freq" not in k
        }

    @property
    def cast_predicate(self):
        def predicate(k):
            return "A_log" not in k
        return predicate
