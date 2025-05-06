import logging 
from time import time
import argparse

from src.retrievers.preprocessor import Preprocessor



logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行RAGcached...")
    parser.add_argument('--mode',  choices=['compress', 'chunk_kvcache','rag', 'kvcache'], required=False, help='模式:1.compress 上下文压缩 2.chunk_kvcache 制作kv cache 3.rag 常规RAG流程 4. kvcache 使用KV Cache的RAG流程')
    # parser.add_argument('--method', choices=['rag', 'kvcache'], required=True, help='Method to use (rag or kvcache)')
    
    args = parser.parse_args()

    if args.mode == 'compress':
        logger.info('正在运行 compress 模式，进行上下文压缩')
        propressor = Preprocessor()
        propressor.compress_context()
    elif args.mode == 'chunk_kvcache':
        logger.info('正在运行 chunk_kvcache 模式，制作kv cache')
    elif args.mode == 'rag':
        logger.info('正在运行 rag 模式，不用kv cache进行RAG流程')
    elif args.mode == 'kvcache':
        logger.info('正在运行 kvcache 模式，使用kv cache进行RAG流程')
    else:
        logger.info('模式错误：')
    
    


    