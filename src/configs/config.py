import os

# 根目录路径
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# KV Cache暂存路径
kv_cache_output_dir = os.path.join(base_dir, "data", "tmp_kv_cache")

# 源数据路径
# src_docs_dir = os.path.join(base_dir, "data", "80000_docs")
src_docs_dir = os.path.join(base_dir, "data", "test_docs")

# 压缩处理后数据路径
docs_dir = os.path.join(base_dir, "data", "compress_docs")

# 模型路径
# model_path = "/home/data/Model/llama-7b"
model_path = "/home/data/Model/Qwen2.5-7B-Instruct"

# Milvus配置
milvus_host = "localhost"
milvus_port = "19530"
milvus_collection_name = "document_vectors"
milvus_dim = 4096  # 向量维度，根据使用的embedding模型调整


API_KEY = ""
