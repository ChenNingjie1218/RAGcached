
import torch
from src.prompt.prompt_query import QUERY_PREFIX
from src.embeddings.base import HuggingfaceEmbeddings
from src.llms.local_model import LocalLLM
from src.retrievers.milvus import MilvusDB

class RAG:
    def __init__(self) -> None:
        self.llm = LocalLLM()
        self.llm.set_prompt(QUERY_PREFIX)
        self.embedding_model = HuggingfaceEmbeddings(model_name='BAAI/bge-base-zh-v1.5')
        self.milvus = MilvusDB()
        self.use_kv_cache = False

    def set_use_kv_cache(self, use : bool):
        self.llm.set_use_kv_cache(use)
        self.use_kv_cache = use

    def query(self, query : str):
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
        for response in self.llm.stream_complete(prompt):
            print(response.text, end="", flush=True)