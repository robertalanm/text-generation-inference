# coding=utf-8
# Copyright 2023 Opentensor and Cerebras team. All rights reserved.
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
""" PyTorch BTLM model."""
import torch
import torch.distributed

from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.models.gpt_neox import GPTNeoXConfig
from typing import Optional, List, Tuple

# vllm imports
import vllm_cache_ops
import vllm_attention_ops

import math

from text_generation_server.utils.flash_attn import attention
from text_generation_server.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelHead,
    FastLayerNorm,
    PositionRotaryEmbedding,
    get_linear,
)

from text_generation_server.models.custom_modeling.btlm_config import BTLMConfig

def load_row(config, prefix: str, weights, bias: bool):
    weight = weights.get_multi_weights_row(prefix, quantize=config.quantize)

    if bias and weights.process_group.rank() == 0:
        # Rank is only on the first rank process
        bias = weights.get_tensor(f"{prefix}.bias")
    else:
        bias = None

    linear = get_linear(weight, bias, config.quantize)
    if config.use_parallel_residual:
        return linear
    else:
        return TensorParallelRowLinear(linear, process_group=weights.process_group)


def load_qkv(config, prefix: str, weights, num_heads, head_size, hidden_size):
    weight = weights.get_multi_weights_col([prefix], quantize=config.quantize, dim=0)
    if isinstance(weight, torch.Tensor):
        # Only on non quantized versions
        weight = (
            weight.view(
                num_heads,
                3,
                head_size,
                hidden_size,
            )
            .permute(1, 0, 2, 3)
            .reshape(-1, hidden_size)
        )

    bias = weights.get_sharded(f"{prefix}.bias", dim=0)
    bias = bias.view(num_heads, 3, head_size).permute(1, 0, 2).reshape(-1)

    linear = get_linear(weight, bias, config.quantize)
    if config.use_parallel_residual:
        return linear
    else:
        return TensorParallelColumnLinear(linear)


class SwiGLUActivation(nn.Module):
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return x1 * nn.functional.silu(x2)


class AlibiPositionEmbeddingLayer(nn.Module):
    def __init__(self, num_heads):
        super(AlibiPositionEmbeddingLayer, self).__init__()

        self.num_heads = num_heads
        slopes = torch.tensor(AlibiPositionEmbeddingLayer._get_alibi_slopes(num_heads)).unsqueeze(-1)
        self.slopes = nn.parameter.Parameter(slopes, requires_grad=False)

    def forward(
        self,
        seq_length,
        key_length,
        cached_qk_len,
    ):
        context_position = torch.arange(
            cached_qk_len, cached_qk_len + seq_length, device=self.slopes.device
        )[:, None]
        memory_position = torch.arange(
            key_length + cached_qk_len, device=self.slopes.device
        )[None, :]
        relative_position = memory_position - context_position
        relative_position = torch.abs(relative_position).unsqueeze(0).expand(self.num_heads, -1, -1)
        alibi = (self.slopes * -1.0).unsqueeze(1) * relative_position
        return alibi

    @staticmethod
    def _get_alibi_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(
                n
            )  # In the paper, we only train models that have 2^a heads for some a. This function has
        else:  # some good properties that only occur when the input is a power of 2. To maintain that even
            closest_power_of_2 = 2 ** math.floor(
                math.log2(n)
            )  # when the number of heads is not a power of 2, we use this workaround.
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + AlibiPositionEmbeddingLayer._get_alibi_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )



class FlashBTLMAttention(torch.nn.Module):
    def __init__(self, config, prefix, weights, is_cross_attention=False, layer_idx=None):
        super().__init__()
        num_heads = config.num_attention_heads
        hidden_size = config.hidden_size

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_size = hidden_size // num_heads
        self.is_cross_attention = is_cross_attention
        self.layer_idx = layer_idx
        self.split_size = hidden_size
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn
        self.attn_scale_power = 1.0 if config.mup_scale_qk_dot_by_d else 0.5
        self.scale_attn_weights = config.scale_attn_weights

        if self.num_heads % weights.process_group.size() != 0:
            raise ValueError(
                f"`num_heads` must be divisible by `num_shards` (got `num_heads`: {self.num_heads} "
                f"and `num_shards`: {weights.process_group.size()}"
            )
        self.num_heads = self.num_heads // weights.process_group.size()

        self.softmax_scale = self.head_size ** (-0.5)

        self.query_key_value = load_qkv(
            config,
            prefix=f"{prefix}.query_key_value",
            weights=weights,
            num_heads=self.num_heads,
            head_size=self.head_size,
            hidden_size=self.hidden_size,
        )
        self.dense = load_row(
            config, prefix=f"{prefix}.dense", weights=weights, bias=True
        )
        self.kv_head_mapping = torch.arange(
            0, self.num_heads, dtype=torch.int32, device=weights.device
        )

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def forward(
        self,
        hidden_states,
        cu_seqlen_prefill,
        kv_cache,
        block_tables,
        slots,
        input_lengths,
        max_s,
        attention_mask=None,
        head_mask=None,
        position_bias=None,
    ):
        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(-1, 3, self.num_heads, self.head_size)

        vllm_cache_ops.reshape_and_cache(
            qkv[:, 1], qkv[:, 2], kv_cache[0], kv_cache[1], slots
        )

        # output tensor
        attn_output = torch.empty_like(qkv[:, 0])

        # Prefill
        if cu_seqlen_prefill is not None:
            # flash attention
            attention(
                qkv[:, 0],
                qkv[:, 1],
                qkv[:, 2],
                attn_output,
                cu_seqlen_prefill,
                max_s,
                self.softmax_scale,
            )
        # Decode
        else:
            # kv_cache[1] => [num_blocks, num_heads, head_size, block_size]
            block_size = kv_cache[1].shape[3]
            vllm_attention_ops.single_query_cached_kv_attention(
                attn_output,
                qkv[:, 0],
                kv_cache[0],
                kv_cache[1],
                self.kv_head_mapping,
                self.softmax_scale,
                block_tables,
                input_lengths,
                block_size,
                max_s,
            )

        attn_output = self.dense(attn_output.view(-1, self.num_heads * self.head_size))
        attn_output = self.resid_dropout(attn_output)

        return attn_output  # a, present, (attentions)


class FlashBTLMMLP(torch.nn.Module):
    def __init__(self, config, prefix, weights):
        super().__init__()
        act = config.activation_function
        self.act = ACT2FN[act] if "gelu" not in act else lambda x: torch.nn.functional.gelu(x, approximate="tanh" if act in ["gelu_fast", "gelu_pytorch_tanh"] else "none",)

        self.swiglu = config.activation_function == "swiglu"
        self.dense_h_to_4h = TensorParallelColumnLinear.load(config, prefix=f"{prefix}.dense_h_to_4h", weights=weights, bias=True)
        self.dense_4h_to_h = load_row(config, prefix=f"{prefix}.dense_4h_to_h", weights=weights, bias=True)
        if self.swiglu:
            self.dense_4h_to_h2 = load_row(config, prefix=f"{prefix}.dense_4h_to_h2", weights=weights, bias=True)
        
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        if self.swiglu:
            hidden_states2 = self.dense_4h_to_h2(hidden_states)
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states, hidden_states2) if self.swiglu else self.act(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states
    
class FlashBTLMBlock(torch.nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()

        layer_norm_eps = config.layer_norm_epsilon

        prefix = f"gpt_neox.layers.{layer_id}"

        self.use_parallel_residual = config.use_parallel_residual
        self.ln_1 = FastLayerNorm.load(prefix=f"{prefix}.ln_1", weights=weights, eps=layer_norm_eps)
        self.attn = FlashBTLMAttention(config, prefix=f"{prefix}.attention", weights=weights)
        self.ln_2 = FastLayerNorm.load(prefix=f"{prefix}.ln_2", weights=weights, eps=layer_norm_eps)

        if config.add_cross_attention:
            self.crossattention = FlashBTLMAttention(config, prefix=f"{prefix}.crossattention", weights=weights, is_cross_attention=True)
            self.ln_cross_attn = FastLayerNorm.load(prefix=f"{prefix}.ln_cross_attn", weights=weights, eps=layer_norm_eps)

        self.mlp = FlashBTLMMLP(config, prefix=f"{prefix}.mlp", weights=weights)
        self.process_group = weights.process_group

    def forward(
        self,
        hidden_states,
        residual,
        cu_seqlen_prefill,
        kv_cache,
        block_tables,
        slots,
        input_lengths,
        max_s,
        attention_mask=None,
        head_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        if self.use_parallel_residual:
            hidden_states, _ = self.ln_1(hidden_states)

            attn_output = self.attn(
                hidden_states,
                cu_seqlen_prefill,
                kv_cache,
                block_tables,
                slots,
                input_lengths,
                max_s,
                attention_mask,
                head_mask,
                position_bias,
            )

            if encoder_hidden_states is not None:
                if not hasattr(self, "crossattention"):
                    raise ValueError(
                        f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                        "cross-attention layers by setting `config.add_cross_attention=True`"
                    )
                cross_attn_output = self.crossattention(
                    hidden_states,
                    cu_seqlen_prefill,
                    kv_cache,
                    block_tables,
                    slots,
                    input_lengths,
                    max_s,
                    attention_mask,
                    head_mask,
                    position_bias,
                )
                attn_output += cross_attn_output

            hidden_states, _ = self.ln_2(hidden_states)
            mlp_output = self.mlp(hidden_states)
            intermediate = mlp_output + attn_output

            if self.process_group.size() > 1:
                torch.distributed.all_reduce(intermediate, group=self.process_group)

            return intermediate + hidden_states, None
        else:
            hidden_states, residual = self.ln_1(hidden_states, residual)

            hidden_states = self.attn(
                hidden_states,
                cu_seqlen_prefill,
                kv_cache,
                block_tables,
                slots,
                input_lengths,
                max_s,
                attention_mask,
                head_mask,
                position_bias,
            )

            if encoder_hidden_states is not None:
                if not hasattr(self, "crossattention"):
                    raise ValueError(
                        f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                        "cross-attention layers by setting `config.add_cross_attention=True`"
                    )
                cross_attn_output = self.crossattention(
                    hidden_states,
                    cu_seqlen_prefill,
                    kv_cache,
                    block_tables,
                    slots,
                    input_lengths,
                    max_s,
                    attention_mask,
                    head_mask,
                    position_bias,
                )
                hidden_states += cross_attn_output

            hidden_states, residual = self.ln_2(hidden_states, residual)

            mlp_output = self.mlp(hidden_states)

            return mlp_output, residual


class FlashBTLMPreTrainedModel(PreTrainedModel):
    config_class = BTLMConfig
    base_model_prefix = "btlm"
    supports_gradient_checkpointing = False
    _no_split_modules = None

class FlashBTLMModel(FlashBTLMPreTrainedModel):
    def __init__(self, config, weights):
        super().__init__(config)
        self.config = config

        self.wte = TensorParallelEmbedding(
            prefix="btlm.wte", weights=weights
        )
        self.wpe = TensorParallelEmbedding(
            prefix="btlm.wpe", weights=weights
        ) if config.position_embedding_type != "alibi" else None
        self.embeddings_scale = config.mup_embeddings_scale

        self.drop = torch.nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [
                FlashBTLMBlock(block_id, config, weights)
                for block_id in range(config.num_hidden_layers)
            ]
        )
        self.ln_f = FastLayerNorm.load(
            prefix="btlm.ln_f",
            weights=weights,
            eps=config.layer_norm_epsilon,
        )

        self.relative_pe = (
            AlibiPositionEmbeddingLayer.load(prefix="btlm.relative_pe", weights=weights)
            if config.position_embedding_type == "alibi"
            else None
        )

        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    # Rest of your forward and other methods go here...
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        max_s: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor] = None,
        kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        block_tables: Optional[torch.Tensor] = None,
        slots: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None,
        
    ) -> torch.Tensor:
        hidden_states = self.wte(input_ids)
        hidden_states *= torch.tensor(
            float(self.embeddings_scale), dtype=hidden_states.dtype, device=hidden_states.device
        )
        
        # we ignore cu_seqlen_prefill, kv_cache, block_tables, slots, input_lengths parameters

        if self.relative_pe is not None:
            length = input_ids.shape[1]
            position_bias = self.relative_pe(length, length)
        else:
            position_bias = None

        past_key_values = (None,) * len(self.h) # we assume there are no past_key_values

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            hidden_states, residual = block(
                hidden_states,
                residual,
                cu_seqlen_prefill,
                kv_cache[i],
                block_tables,
                slots,
                input_lengths,
                max_s,
            )

        hidden_states = self.ln_f(hidden_states)

        return hidden_states
    

class FlashBTLMForCausalLM(FlashBTLMPreTrainedModel):
    def __init__(self, config, weights):
        super().__init__(config)
        self.config = config

        self.transformer = FlashBTLMModel(config, weights)
        self.lm_head = TensorParallelHead(
            prefix="lm_head", weights=weights
        )

        self.output_logits_scale = config.mup_output_alpha * config.mup_width_scale


    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        input_lengths: torch.Tensor,
        max_s: int,
        lm_head_indices: Optional[torch.Tensor] = None,       
    ):
        hidden_states = self.transformer(
            input_ids,
            position_ids,
            max_s,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
        )

        lm_logits = self.lm_head(hidden_states)
        lm_logits *= torch.tensor(
            float(self.output_logits_scale), dtype=lm_logits.dtype, device=lm_logits.device
        )

        return lm_logits
