import os

# 根目录路径
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# KV Cache暂存路径
kv_cache_output_dir = os.path.join(base_dir, "data", "tmp_kv_cache")

# 源数据路径
src_docs_dir = os.path.join(base_dir, "data", "80000_docs")
# src_docs_dir = os.path.join(base_dir, "data", "test_docs")

# 压缩处理后数据路径
docs_dir = os.path.join(base_dir, "data", "compress_docs")
# docs_dir = os.path.join(base_dir, "data", "test_compress_docs")

# 模型路径
# model_path = "/home/data/Model/llama-7b"
model_path = "/home/data/Model/Qwen2.5-7B-Instruct"
embedding_model_path = "/home/jokerjay/rag/RAGcached/sentence-transformers/bge-base-zh-v1.5"

# Milvus配置
# milvus_host = "localhost"
# milvus_port = "19530"
# milvus_collection_name = "document_vectors"
# milvus_dim = 4096  # 向量维度，根据使用的embedding模型调整
milvus_data_path = os.path.join(base_dir, "data")

API_KEY = ""

# 日志
log_path = os.path.join(base_dir, "log", "rag.log")
performance_log_path = os.path.join(base_dir, "log")