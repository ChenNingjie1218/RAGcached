import logging 
import time
import sys
import os
# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessor import KVCachedNodeParser
from src.configs.config import docs_dir
from llama_index.core import (
    SimpleDirectoryReader,
)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# documents = SimpleDirectoryReader(docs_dir).load_data()
file_path = f"{docs_dir}/documents_dup_part_1_part_1"
documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
parser = KVCachedNodeParser()
start_time = time.perf_counter()
nodes = parser.get_nodes_from_documents(documents[0:1])
end_time = time.perf_counter()
logger.info(f"KVCache解析耗时：{end_time-start_time:.2f}s")
logger.info(f"KVCache数量：{len(nodes)}")







