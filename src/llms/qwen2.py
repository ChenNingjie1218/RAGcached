from typing import Any
from transformers import AutoTokenizer, Qwen2ForCausalLM

import torch
import math
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    repeat_kv,
    rotate_half,
    Qwen2DecoderLayer,
    Qwen2Model,
    Qwen2ForCausalLM,
    Qwen2Config,
    Cache,
    FlashAttentionKwargs,
    Unpack,
    apply_rotary_pos_emb,
    rotate_half,
    eager_attention_forward,
    Callable,
    logger,
    ALL_ATTENTION_FUNCTIONS
)

from typing import Optional, Tuple, Union

def apply_rotary_pos_emb_k(k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return k_embed

class Qwen2ModifiedAttention(Qwen2Attention):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        self.apply_rope = True  # 新增属性，默认开启 RoPE

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        # ✅ 控制是否应用 RoPE
        if self.apply_rope:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        sliding_window = None
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=sliding_window,  # main diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

# class Qwen2ModifiedDecoderLayer(Qwen2DecoderLayer):
#     def __init__(self, config: Qwen2Config, layer_idx: int):
#         super().__init__(config, layer_idx)
#         self.self_attn = Qwen2ModifiedAttention(config=config, layer_idx=layer_idx)

# class Qwen2ModifiedModel(Qwen2Model):
#     def __init__(self, config: Qwen2Config):
#         super().__init__(config)
#         self.layers = nn.ModuleList(
#             [Qwen2ModifiedDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
#         )
#         # Initialize weights and apply final processing
#         self.post_init()

class Qwen2ModifiedForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        # self.model = Qwen2ModifiedModel(config)
         # 替换 attention 层
        for idx, layer in enumerate(self.model.layers):
            old_attn = layer.self_attn
            modified_attn = Qwen2ModifiedAttention(config=config, layer_idx=idx)
            modified_attn.load_state_dict(old_attn.state_dict(), strict=False)
            layer.self_attn = modified_attn
    
    def set_apply_rope(self, enable: bool):
        """
        设置模型中所有 ModifiedAttention 层的 apply_rope 开关
        """
        for module in self.model.modules():
            if isinstance(module, Qwen2ModifiedAttention):
                module.apply_rope = enable
                # print(f"启动RoPE：{enable}")

    def apply_rotary_pos_emb_for_past_key_values(self, full_input_ids, past_key_values):
        # Step 1: 获取已缓存的 token 数量
        num_cached_tokens = past_key_values[0][0].shape[2]  # 假设使用 tuple 类型的 past_key_values

        # Step 2: 只取对应长度的 input_ids
        # selected_input_ids = full_input_ids[:, :num_cached_tokens]  # shape: [batch_size, num_cached_tokens]

        # Step 3: 构造 inputs_embeds 和 position_ids  ||| rotary_emb里面只用得到device和dtype,无需inputs_embeds!!!!!
        # inputs_embeds = self.model.embed_tokens(selected_input_ids)    
        shape = (1, 128)
        inputs_embeds = torch.randn(shape, device='cuda', dtype=torch.float16)
        batch_size = full_input_ids.shape[0]
        position_ids = torch.arange(num_cached_tokens, dtype=torch.long, device="cuda").unsqueeze(0).expand(batch_size, -1)
        
        # Step 4: 获取 position_embeddings (cos/sin)
        position_embeddings = self.model.rotary_emb(inputs_embeds, position_ids)
        cos, sin = position_embeddings

        # Step 5: 对每个 layer 的 key 应用 RoPE
        new_key_values = []
        for idx, layer in enumerate(self.model.layers):
            layer_device = self.get_layer_device(idx)
            key = past_key_values[idx][0]
            value = past_key_values[idx][1]
            # print(f"k.device: {key.device}, cos.device: {cos.device}, sin.device: {sin.device}")
            key_rot = apply_rotary_pos_emb_k(key.to(layer_device), cos.to(layer_device), sin.to(layer_device))
            new_key_values.append((key_rot, value))
        return tuple(new_key_values)
    
    def get_layer_device(self, layer_idx):
        try:
            # 假设模型是 HuggingFace Transformers 风格的
            layer_name = f"model.layers.{layer_idx}"
            module = self.get_submodule(layer_name)
            return next(module.parameters()).device
        except Exception as e:
            raise RuntimeError(f"无法获取第 {layer_idx} 层设备: {e}")