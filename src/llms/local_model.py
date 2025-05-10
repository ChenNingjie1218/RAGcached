from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
from llama_index.core.llms import CustomLLM, CompletionResponseGen, CompletionResponse
import time
import torch
from src.configs.config import model_path, log_path
from typing import Any, Optional, List
from pydantic import Field
from transformers.cache_utils import (
    DynamicCache,
)
import copy
import logging
logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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
    prefix_cache: Any = Field(default=None, description="PREFIX的KV cache")
    PREFIX: str = Field(default='', description="PREFIX")
    use_kv_cache: bool = Field(default=False, description="是否使用kv cache进行推理")
    total_time: Any = Field(default=None, description="调度总耗时")
    total_count: Any = Field(default=None, description="调度总次数")
    
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
            self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
            self.total_time = 0
            self.total_count = 0
            
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {str(e)}")
        
    def __del__(self):
        if self.total_count > 0:
            avg_time = self.total_time / self.total_count
            log_msg = f"LLM调用次数: {self.total_count}, Total time: {self.total_time:.4f}s, Average time: {avg_time:.4f}s\n"
            logger.info(log_msg)

    def set_prompt(self, prefix: str):
        self.PREFIX = prefix
        prefix_inputs = self.tokenizer(self.PREFIX, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            self.prefix_cache = DynamicCache()
            self.prefix_cache = self.model(**prefix_inputs, past_key_values = self.prefix_cache).past_key_values
    
    @property
    def metadata(self):
        """返回模型元数据"""
        return {
            "model_name": "Qwen2.5-7B",
            "context_window": 32768,  # Qwen2.5-7B 的上下文窗口大小
            "num_output": 1280,  # 与 max_new_tokens 保持一致
        }
    
    def stack_past_key_values(self, past_key_values_list):
        num_layers = len(past_key_values_list[0])
        batch_past_key_values = []
        for layer in range(num_layers):
            keys = torch.cat([past_key_values[layer][0] for past_key_values in past_key_values_list], dim=2)
            values = torch.cat([past_key_values[layer][1] for past_key_values in past_key_values_list], dim=2)
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
            # docs = '''[ 2023-07-28 16：40 ] ，正文：人民网柏林7月28日电 （记者刘仲华、张慧中）7月27日晚，中国驻德国大使馆举行庆祝中国人民解放军建军96周年招待会。德军政官员、各界人士、多国驻德使节和武官、华侨华人、留学生及中资企业代表等约350人出席招待会。吴恳大使和使馆主要外交官出席活动。国防武官吴俊辉少将在招待会上致辞。吴俊辉强调，中国人民解放军建军96周年来，在中国共产党领导下，为民族独立和解放，为国家建设和发展，为维护国家主权、安全和发展利益，建立了不朽功勋。中国军队永远是捍卫国家主权和领土完整的坚定力量。吴俊辉针对当前涉中国问题的负面杂音，简要阐述了中国和平理念和国防政策，指出历史是最好的见证者，中国是全球唯一将“坚持和平发展道路”载入宪法的国家，在涉及和平与安全问题上，中国更是纪录最好的大国。中国从不追逐霸权，也绝不走“国强必霸”的老路，中国军队始终是维护世界和平的坚定力量。吴俊辉表示，李强总理上月对德国进行了正式访问，传承了两国友谊，深化了双边合作。今年初，中德两国国防部举行了工作对话，前不久，两国防长在香格里拉对话会期间进行了会晤，为两军关系下一步发展指明了方向。中国军队愿同德方一道，加强沟通交流，增进理解互信，深化务实合作，共同维护世界和地区安全与稳定。招待会气氛热烈友好，现场还通过电子屏幕滚动播放《我在》等视频。出席来宾纷纷祝贺近年来中国国防和军队建设取得的巨大成就，赞赏中国军队为维护世界和平所做出的积极贡献。分享让更多人看到'''
            # docs = '''人民网柏林7月28日电 中国驻德国大使馆27日晚举办建军96周年招待会，德军政界、多国使节及各界代表约350人出席。国防武官吴俊辉少将强调，中国军队在中国共产党领导下为国家发展和安全作出重大贡献，是捍卫国家主权的坚定力量。他重申中国将"和平发展"写入宪法，坚持永不称霸，致力于维护世界和平。吴俊辉提及李强总理访德成果及中德国防部近期对话，表示愿深化两军交流合作。活动期间，来宾高度评价中国国防建设成就及维和贡献，现场播放相关主题视频。'''
            # docs_inputs = self.tokenizer(docs, return_tensors="pt").to(self.model.device)
            # with torch.no_grad():
            #     docs_cache = DynamicCache()
            #     docs_cache = self.model(**docs_inputs, past_key_values = docs_cache).past_key_values
            # past_key_values = self.stack_past_key_values([self.prefix_cache, docs_cache])
            # past_key_values = self.stack_past_key_values([self.prefix_cache])
            # 在生成开始前就开始计时
            start_time = time.perf_counter()
            new_prompt = self.PREFIX+prompt+'<|im_end|><|im_start|>assistant\n'
            first_token_received = False
            inputs = self.tokenizer(new_prompt, return_tensors="pt").to(self.model.device)
            if self.use_kv_cache:
                past_key_values = copy.deepcopy(self.prefix_cache)
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
    
    def complete(self, prompt: str) -> str:
        """
        非流式生成文本
        
        Args:
            prompt: 输入提示词
            
        Returns:
            str: 生成的完整文本
        """
        try:
            start_time = time.perf_counter()
            new_prompt = self.PREFIX+prompt+'<|im_end|><|im_start|>assistant\n'
            inputs = self.tokenizer(new_prompt, return_tensors="pt").to(self.model.device)
            past_key_values = copy.deepcopy(self.prefix_cache)
            generation_kwargs = {
                **inputs,
                "max_new_tokens": 512,
                "past_key_values": past_key_values,
            }
            
            outputs = self._safe_generate(**generation_kwargs)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            duration = time.perf_counter() - start_time 
            self.total_time += duration
            self.total_count += 1
            # 移除输入提示词，只返回生成的文本
            return response.split('assistant')[-1].strip()
            
        except Exception as e:
            raise RuntimeError(f"文本生成失败: {str(e)}")
        
    def _safe_generate(self, **kwargs):
        with torch.no_grad():
            return self.model.generate(**kwargs)
        
    def set_use_kv_cache(self, use_kv_cache : bool = False):
        self.use_kv_cache = use_kv_cache
    
