
from concurrent.futures import ThreadPoolExecutor
import os
import time
import torch
from src.prompt.prompt_query import QUERY_PREFIX
from src.embeddings.base import HuggingfaceEmbeddings
from src.llms.local_model import LocalLLM
from src.retrievers.milvus import MilvusDB
from cachetools import LRUCache
import logging
from src.configs.config import embedding_model_path, log_path, kv_cache_output_dir
logging.basicConfig(file_name=log_path, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
from src.logger.performance_logger import PerformanceLogger
class RAG:
    def __init__(self) -> None:
        self.llm = LocalLLM()
        self.llm.set_prompt(QUERY_PREFIX)
        self.embedding_model = HuggingfaceEmbeddings(model_name=embedding_model_path)
        self.milvus = MilvusDB()
        self.use_kv_cache = False

        # RAG流程耗时
        self.total_time = 0
        self.total_count = 0

        # LRU Cache的kv cache命中率
        self.cache_requests = 0
        self.cache_hits = 0

        # KV Cache加载
        self.load_total_time = 0
        self.load_total_count = 0

    def __del__(self):
        if self.total_count > 0:
            avg_time = self.total_time / self.total_count
            avg_load_time = self.load_total_time / self.load_total_count if self.load_total_count > 0 else 0
            cache_hit_ratio = self.cache_hits / self.cache_requests if self.cache_requests > 0 else 0
            log_msg = f"查询次数: {self.total_count}, Average time: {avg_time*1000:.1f}ms, Average load time: {avg_load_time*1000:.1f}ms, cache命中率: {cache_hit_ratio:.2%}"
            logger.info(log_msg)

    def set_use_kv_cache(self, use : bool):
        self.llm.set_use_kv_cache(use)
        self.use_kv_cache = use
        # self.cache = LRUCache(maxsize=20)

    def set_milvus_db(self, filename:str):
        self.milvus.close()
        self.milvus = MilvusDB(filename)

    def _load_key_value_cache(self, path):
        # self.cache_requests += 1
        # if path in self.cache:
        #     self.cache_hits += 1
        #     # logger.info(f'缓存命中! {path}')
        #     return self.cache[path]
        # start = time.perf_counter()
        data = torch.load(path, map_location="cpu", weights_only=False)
        # end = time.perf_counter()
        # print(f"加载 {path} 耗时: {(end - start)*1000:.2f}ms")
        # self.cache[path] = data
        return data

    def query(self, query : str):
        start_time = time.perf_counter()
        vector = self.embedding_model.embed_query(query)
        responses = self.milvus.query_vectors([vector])[0]
        docs = ''
        key_values = [] if self.use_kv_cache else None
        paths = []
        load_time = time.perf_counter()
        for resp in responses:
            docs = docs + resp['entity']['text']
            if key_values is not None:
                load_time = time.perf_counter()
                path = os.path.join(kv_cache_output_dir, resp['entity']['kv_file'])
                paths.append(path)
                # key_values.append(self._load_key_value_cache(path))
                # self.load_total_time += time.perf_counter() - load_time
                # self.load_total_count += 1

        query = '<|document_sep|>\n问题:\n' + query
        if key_values is not None:
            # 并行加载所有 key_value 文件
            with ThreadPoolExecutor() as executor:
                key_values.extend(executor.map(self._load_key_value_cache, paths))
        # print(f'加载kv cache耗时：{time.perf_counter()-load_time:.4f} s')
            load_duration = time.perf_counter() - load_time
            self.load_total_time += load_duration
            self.load_total_count += 1
            PerformanceLogger.record_event("RAG", "load_kv_cache", {"load_time":load_duration*1000})
        prompt = docs + query 
        
        # for response in self.llm.stream_complete(prompt):
            # print(response.text, end="", flush=True)
        result = self.llm.complete(prompt, key_values)
        end_time = time.perf_counter()  # 结束计时
        elapsed = end_time - start_time
        self.total_time += elapsed
        self.total_count += 1
        PerformanceLogger.record_event("RAG", "query", {"query_time":elapsed*1000})
        return result

    def stream_query(self, query: str):
        vector = self.embedding_model.embed_query(query)
        responses = self.milvus.query_vectors([vector])[0]

        docs = ''
        key_values = [] if self.use_kv_cache else None
        # paths = []
        for resp in responses:
            docs = docs + resp['entity']['text']
            if key_values is not None:
                path = os.path.join(kv_cache_output_dir, resp['entity']['kv_file'])
                key_values.append(self._load_key_value_cache(path))
                # paths.append(path)

        # if key_values is not None:
        #     # 并行加载所有 key_value 文件
        #     with ThreadPoolExecutor() as executor:
        #         key_values.extend(executor.map(self._load_key_value_cache, paths))

        prompt = docs + '<|document_sep|>\n问题：' + query
        # 流式生成
        for response in self.llm.stream_complete(prompt, key_values):
            yield response.text