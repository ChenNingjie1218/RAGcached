from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, StoppingCriteria
from llama_index.core.llms import CustomLLM, CompletionResponseGen, CompletionResponse
import time
import torch
from src.retrievers.kvcachetrie import KVCacheTrie
from src.configs.config import model_path, log_path
from typing import Any, Optional, List
from pydantic import Field
from transformers.cache_utils import (
    DynamicCache,
)
import logging
logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from src.logger.performance_logger import PerformanceLogger

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids):
        self.stop_ids = stop_ids

    def __call__(self, input_ids, scores, **kwargs):
        for stop_id in self.stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

class LocalLLM(CustomLLM):    
    model: Any = Field(default=None, description="HuggingFace模型")
    tokenizer: Any = Field(default=None, description="HuggingFace分词器")
    streamer: Any = Field(default=None, description="文本流生成器")
    kvsystem: Any = Field(default=None, description="kvsystem")
    prefix_token_count: int = Field(default=0, description="prefix token count")
    PREFIX: str = Field(default='', description="PREFIX")
    use_kv_cache: bool = Field(default=False, description="是否使用kv cache进行推理")
    total_time: Any = Field(default=None, description="调度总耗时")
    total_count: Any = Field(default=None, description="调度总次数")
    stack_total_time: Any = Field(default=None, description="加载KV Cache(to GPU)及拼接的总耗时")
    stack_total_count: Any = Field(default=None, description="拼接KV Cache总次数")
    
    def __init__(self):
        """
        初始化本地语言模型
        """
        try:
            super().__init__()
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            ).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            SPECIAL_SEP_TOKEN = '<|document_sep|>'
            self.tokenizer.add_special_tokens({"additional_special_tokens": [SPECIAL_SEP_TOKEN]})
            self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)

            # 调用
            self.total_time = 0
            self.total_count = 0

            # kv cache拷贝到GPU及拼接
            self.stack_total_time = 0
            self.stack_total_count = 0
            
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {str(e)}")
        
    def __del__(self):
        if self.total_count > 0:
            avg_time = self.total_time / self.total_count
            avg_stack_time = self.stack_total_time / self.stack_total_count if self.stack_total_count > 0 else 0
            log_msg = f"LLM调用次数: {self.total_count}, Average time: {avg_time*1000:.1f}ms, Average stack time: {avg_stack_time*1000:.1f}ms"
            logger.info(log_msg)

    def set_prompt(self, prefix: str):
        self.PREFIX = prefix
        prefix_inputs = self.tokenizer(self.PREFIX, return_tensors="pt").to(self.model.device)
        token_count = prefix_inputs.input_ids.shape[1]
        print(f"prefix的 token 数量: {token_count}")
        with torch.no_grad():
            outputs = self.model(**prefix_inputs, use_cache=True)
            self.kvsystem = KVCacheTrie(outputs.past_key_values)
            self.prefix_token_count = token_count
    
    @property
    def metadata(self):
        """返回模型元数据"""
        return {
            "model_name": "Qwen2.5-7B",
            "context_window": 32768,  # Qwen2.5-7B 的上下文窗口大小
            "num_output": 1280,  # 与 max_new_tokens 保持一致
        }
    def get_layer_device(self, layer_idx):
        try:
            # 假设模型是 HuggingFace Transformers 风格的
            layer_name = f"model.layers.{layer_idx}"
            module = self.model.get_submodule(layer_name)
            return next(module.parameters()).device
        except Exception as e:
            raise RuntimeError(f"无法获取第 {layer_idx} 层设备: {e}")
        
    def stack_past_key_values(self, past_key_values_list):
        num_layers = len(past_key_values_list[0])
        batch_past_key_values = []
        for layer in range(num_layers):
            layer_device = self.get_layer_device(layer)
            keys = torch.cat([past_key_values[layer][0].to(layer_device) for past_key_values in past_key_values_list], dim=2)
            values = torch.cat([past_key_values[layer][1].to(layer_device) for past_key_values in past_key_values_list], dim=2)
            batch_past_key_values.append((keys, values))
        return tuple(batch_past_key_values)
    
    def stream_complete(self, prompt: str, key_values: Optional[List[DynamicCache]] = None) -> CompletionResponseGen:
        """
        流式生成文本
        
        Args:
            prompt: 输入提示词
            
        Yields:
            CompletionResponse: 生成的文本响应
        """
        try:
            # 在生成开始前就开始计时
            start_time = time.perf_counter()
            new_prompt = self.PREFIX+prompt+'<|document_sep|><|im_start|>assistant\n'
            first_token_received = False
            inputs = self.tokenizer(new_prompt, return_tensors="pt").to(self.model.device)
            if self.use_kv_cache:
                past_key_values = self.prefix_cache
                if key_values is not None:
                    past_key_values = self.stack_past_key_values([past_key_values] + key_values)

                generation_kwargs = {
                    **inputs,
                    "max_new_tokens": 512,
                    "streamer": self.streamer,
                    "past_key_values": past_key_values,
                }
            else:
                generation_kwargs = {
                    **inputs,
                    "max_new_tokens": 512,
                    "streamer": self.streamer
                }
            
            # 使用多线程进行生成
            import threading
            thread = threading.Thread(target=self._safe_generate, kwargs=generation_kwargs)
            thread.start()
            
            # 逐Token捕获输出
            for token in self.streamer:
                if not first_token_received:
                    ttft = time.perf_counter() - start_time  # 计算TTFT
                    first_token_received = True
                    print(f"TTFT: {ttft:.2f}s")
                yield CompletionResponse(text=token, delta=token)
                
        except Exception as e:
            raise RuntimeError(f"文本生成失败: {str(e)}")
    
    def complete(self, prompt: str, key_values: Optional[List[DynamicCache]] = None, chunk_info : Optional[List[dict]] = None) -> str:
        """
        非流式生成文本
        
        Args:
            prompt: 输入提示词
            
        Returns:
            str: 生成的完整文本
        """
        try:
            new_prompt = self.PREFIX + prompt + '<|document_sep|><|im_start|>assistant\n'
            inputs = self.tokenizer(new_prompt, return_tensors="pt").to(self.model.device)
            token_count = inputs.input_ids.shape[1]
            PerformanceLogger.record_event("Local_LLM", "input_token_count", {"token_count":token_count})
            print(f"输入的token总数：{token_count}")
            if self.use_kv_cache:
                if key_values is not None:
                    stack_time = time.perf_counter()
                    past_key_values = self.stack_past_key_values(key_values)
                    stack_duration = time.perf_counter() - stack_time
                    self.stack_total_time += stack_duration
                    self.stack_total_count += 1
                    PerformanceLogger.record_event("Local_LLM", "stack_kv_cache", {"stack_time":stack_duration*1000})
                re_token_count = past_key_values[0][0].shape[2]
                # print(f"复用kv cache token count: {re_token_count}")
                PerformanceLogger.record_event("Local_LLM", "reuse_kv_cache", {"reuse_kv_cache":re_token_count})
                generation_kwargs = {
                    **inputs,
                    # "max_new_tokens": 512,
                    "max_new_tokens": 1,
                    # "max_new_tokens": 1 if self.total_count < 100 else 512,
                    "past_key_values": past_key_values,
                    "return_dict_in_generate":True
                }
            else:
                generation_kwargs = {
                    **inputs,
                    # "max_new_tokens": 512,
                    "return_dict_in_generate":True,
                    "max_new_tokens": 1
                }
            start_time = time.perf_counter()
            outputs = self._safe_generate(**generation_kwargs)
            # response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            duration = time.perf_counter() - start_time 
            # 插新的
            if self.use_kv_cache:
                new_past_key_value = outputs.past_key_values
                # len = new_past_key_value[0][0].shape[2]
                # print(f"KV Cache:{len}")
                node_list = self._get_node_list(new_past_key_value, chunk_info)
                self.kvsystem.insert(node_list)


            self.total_time += duration
            self.total_count += 1
            PerformanceLogger.record_event("Local_LLM", "complete", {"complete_time":duration*1000})
            print(f"complete_time: {duration*1000} ms")
            # 移除输入提示词，只返回生成的文本
            return response.split('assistant')[-1].strip()
            
        except Exception as e:
            raise RuntimeError(f"文本生成失败: {str(e)}")
        
    def _safe_generate(self, **kwargs):
        with torch.no_grad():
            return self.model.generate(**kwargs)
        
    def set_use_kv_cache(self, use_kv_cache : bool = False):
        self.use_kv_cache = use_kv_cache

    def _get_node_list(self, past_key_values, chunk_info):
        node_list = []
        start_idx = self.prefix_token_count
        for chunk in chunk_info:
            node_list.append({
                'id': chunk['id'],
                'kv': self.slice_dynamic_cache(past_key_values, start_idx, start_idx + chunk['token_count']),
                'token_count':chunk['token_count']
            })
            start_idx += chunk['token_count']
        return node_list
    
    def slice_dynamic_cache(self, dynamic_cache, start, end):
        sliced_cache = DynamicCache()

        for layer_idx in range(len(dynamic_cache.key_cache)):
            key = dynamic_cache.key_cache[layer_idx]  # [bsz, n_heads, seq_len, head_dim]
            value = dynamic_cache.value_cache[layer_idx]

            # 截取中间部分
            sliced_key = key[:, :, start:end, :]
            sliced_value = value[:, :, start:end, :]

            # 存入新的 cache
            sliced_cache.update(
                key_states=sliced_key,
                value_states=sliced_value,
                layer_idx=layer_idx,
            )

        return sliced_cache