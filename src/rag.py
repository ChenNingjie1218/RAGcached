
import time
import torch
from src.prompt.prompt_query import QUERY_PREFIX
from src.embeddings.base import HuggingfaceEmbeddings
from src.llms.local_model import LocalLLM
from src.retrievers.milvus import MilvusDB
import logging
from src.configs.config import embedding_model_path, log_path
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class RAG:
    def __init__(self) -> None:
        self.llm = LocalLLM()
        self.llm.set_prompt(QUERY_PREFIX)
        self.embedding_model = HuggingfaceEmbeddings(model_name=embedding_model_path)
        self.milvus = MilvusDB()
        self.use_kv_cache = False
        self.total_time = 0
        self.total_count = 0

    def __del__(self):
        if self.total_count > 0:
            avg_time = self.total_time / self.total_count
            log_msg = f"查询次数: {self.total_count}, Total time: {self.total_time:.4f}s, Average time: {avg_time:.4f}s\n"
            with open(log_path, 'a') as f:
                f.write(log_msg)

    def set_use_kv_cache(self, use : bool):
        self.llm.set_use_kv_cache(use)
        self.use_kv_cache = use

    def set_milvus_db(self, filename:str):
        self.milvus.close()
        self.milvus = MilvusDB(filename)

    def query(self, query : str):
        start_time = time.perf_counter()
        vector = self.embedding_model.embed_query(query)
        responses = self.milvus.query_vectors([vector])[0]
        docs = ''
        key_values = [] if self.use_kv_cache else None
        for resp in responses:
            docs = docs + resp['entity']['text']
            if key_values is not None:
                key_value = torch.load(resp['entity']['kv_path'], weights_only=False)
                key_values.append(key_value)
        prompt = docs + '\n问题:\n' + query
        
        # for response in self.llm.stream_complete(prompt):
            # print(response.text, end="", flush=True)
        result = self.llm.complete(prompt)
        end_time = time.perf_counter()  # 结束计时
        elapsed = end_time - start_time
        self.total_time += elapsed
        self.total_count += 1
        return result

    def stream_query(self, query: str):
        vector = self.embedding_model.embed_query(query)
        responses = self.milvus.query_vectors([vector])[0]

        docs = ''
        key_values = [] if self.use_kv_cache else None
        for resp in responses:
            docs += resp['entity']['text']
            if key_values is not None:
                key_value = torch.load(resp['entity']['kv_path'], weights_only=False)
                key_values.append(key_value)

        prompt = docs + '\n问题:\n' + query
        # 流式生成
        for response in self.llm.stream_complete(prompt):
            yield response.text