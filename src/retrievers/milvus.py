import time
from pymilvus import MilvusClient
from src.configs.config import milvus_data_path, log_path
import os

import logging
logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class MilvusDB:
    def __init__(self, file_name:str='test.db') -> None:
        path = os.path.join(milvus_data_path, file_name)
        self.client = MilvusClient(path)
        self.collection_name = "test"
        self.total_time = 0
        self.total_count = 0

    def __del__(self):
        if self.total_count > 0:
            avg_time = self.total_time / self.total_count
            log_msg = f"向量检索次数: {self.total_count}, Total time: {self.total_time:.4f}s, Average time: {avg_time:.4f}s\n"
            logger.info(log_msg)

    def close(self):
        self.client.close()

    def set_collection_name(self, collection_name:str):
        self.collection_name = collection_name

    def construct_index(self):
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.drop_collection(collection_name=self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            auto_id=True,
            dimension=768, 
        )
    
    def insert(self, data):
        self.client.insert(collection_name=self.collection_name, data=data)

    def query_vectors(self, query_vectors):
        start_time = time.perf_counter()
        output_fields = ["text", "kv_file"]
        res = self.client.search(
            collection_name=self.collection_name,  # target collection
            data=query_vectors,  # query vectors
            limit=8,  # number of returned entities
            output_fields=output_fields,  # specifies fields to be returned
        )
        end_time = time.perf_counter()  # 结束计时
        elapsed = end_time - start_time
        self.total_time += elapsed
        self.total_count += 1
        return res

