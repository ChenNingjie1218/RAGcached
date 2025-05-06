import logging 
from time import time
import sys
import os
# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrievers.preprocessor import KVCachedNodeParser
from src.configs.config import src_docs_dir
from llama_index.core import (
    SimpleDirectoryReader,
)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

documents = SimpleDirectoryReader(src_docs_dir).load_data()
parser = KVCachedNodeParser()
start_time = time()
nodes = parser.get_nodes_from_documents(documents[0:1])
end_time = time()
logger.info(f"KVCache解析耗时：{end_time-start_time:.2f}s")
logger.info(f"KVCache数量：{len(nodes)}")







