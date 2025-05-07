from pymilvus import MilvusClient
from src.configs.config import milvus_data_path
import os
class MilvusDB:
    def __init__(self) -> None:
        path = os.path.join(milvus_data_path, "milvus_demo.db")
        self.client = MilvusClient(path)

    def construct_index(self):
        if self.client.has_collection(collection_name="demo_collection"):
            self.client.drop_collection(collection_name="demo_collection")
        self.client.create_collection(
            collection_name="demo_collection",
            dimension=768,  
        )
    
    def insert(self, data):
        self.client.insert(collection_name="demo_collection", data=data)

    def query_vectors(self, query_vectors):
        res = self.client.search(
            collection_name="demo_collection",  # target collection
            data=query_vectors,  # query vectors
            limit=8,  # number of returned entities
            output_fields=["text", "kv_path"],  # specifies fields to be returned
        )
        return res

