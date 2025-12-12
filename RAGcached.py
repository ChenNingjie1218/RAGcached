import logging 
from multiprocessing import Process
from time import time
import argparse

from src.rag import RAG
from src.preprocessor import Preprocessor
from src.configs.config import log_path
from src.llms.local_model import LocalLLM

logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from src.logger.performance_logger import PerformanceLogger

def start_rag_server(rag:RAG):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    import json
    app = FastAPI()

    @app.post("/stream_query")
    async def stream_query(request: dict):
        try:
            query_text = request.get("query", "")
            if not query_text:
                return {"error": "缺少 query 参数"}
            logger.info(f"用户问题：{query_text}")
            async def generate():
                for chunk in rag.stream_query(query_text):
                    yield f"data: {json.dumps({'answer': chunk})}\n\n"

            return StreamingResponse(generate(), media_type="text/event-stream")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")
    
    @app.post("/query")
    async def query(request: dict):
        try:
            query_text = request.get("query", "")
            if not query_text:
                return {"error": "缺少 query 参数"}
            logger.debug(f"用户问题：{query_text}")
            return {
                "answer" : rag.query(query_text)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")
    import uvicorn
    # 添加启动服务器的代码
    uvicorn.run(app, host="0.0.0.0", port=10080)

def start_llm_server(llm:LocalLLM):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    import json
    app = FastAPI()

    @app.post("/stream_query")
    async def stream_query(request: dict):
        try:
            query_text = request.get("query", "")
            if not query_text:
                return {"error": "缺少 query 参数"}
            logger.info(f"用户问题：{query_text}")
            async def generate():
                for chunk in llm.stream_complete(query_text):
                    yield f"data: {json.dumps({'answer': chunk})}\n\n"

            return StreamingResponse(generate(), media_type="text/event-stream")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")
    
    @app.post("/query")
    async def query(request: dict):
        try:
            query_text = request.get("query", "")
            if not query_text:
                return {"error": "缺少 query 参数"}
            logger.debug(f"用户问题：{query_text}")
            return {
                "answer" : llm.complete(query_text)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")
    import uvicorn
    # 添加启动服务器的代码
    uvicorn.run(app, host="0.0.0.0", port=10080)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False, description="运行RAGcached...")
    parser.add_argument('--compress', action='store_true', help='上下文压缩')
    parser.add_argument('--chunk_kvcache', action='store_true', help='正在构建预处理后的数据库')
    parser.add_argument('--origin_prepare', action='store_true', help='构建未预处理的数据库')
    parser.add_argument('--mode',  default='no', choices=['rag', 'kvcache', 'origin', 'no', 'llm'], required=False, help='模式:1.rag 常规RAG流程 2.kvcache 使用KV Cache的RAG流程 3.origin 使用未预处理的数据进行常规RAG')
    args = parser.parse_args()
    if args.compress:
        logger.info('正在进行上下文压缩...')
        propressor = Preprocessor()
        # for idx in [1, 2, 3]:
        # idx = 3
        processes = []
        for group_num in range(1,4):
            # part = f'_{group_num}_part_{idx}'
            part = f'docs_{group_num}.txt'
        # for group_num in range(1,4):
        #     part = f'_11_part_{group_num}'
            p = Process(target=propressor.compress_context, args=(part,))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        print("所有进程已完成！")

    if args.chunk_kvcache:
        logger.info('正在构建预处理后的数据库')
        Preprocessor = Preprocessor()
        Preprocessor.prepare(kv_cache=True)
        logger.info('构建完成')
    
    if args.origin_prepare:
        logger.info('正在构建未预处理的数据库...')
        Preprocessor = Preprocessor()
        Preprocessor.prepare(kv_cache=False)
        logger.info('构建完成')
    
    if args.mode == 'rag':
        logger.info('正在运行 rag 模式，不用kv cache进行RAG流程')
        PerformanceLogger.reset_logger(log_file="rag.json")
        rag = RAG()
        rag.set_milvus_db("cached.db")
        start_rag_server(rag)
    elif args.mode == 'kvcache':
        logger.info('正在运行 kvcache 模式，使用kv cache进行RAG流程')
        PerformanceLogger.reset_logger(log_file="kvcache.json")
        rag = RAG()
        rag.set_use_kv_cache(use=True)
        rag.set_milvus_db("cached.db")
        start_rag_server(rag)
    elif args.mode == 'origin':
        logger.info('正在运行 origin 模式，使用未预处理的数据进行常规RAG')
        PerformanceLogger.reset_logger(log_file="origin.json")
        rag = RAG()
        rag.set_milvus_db("origin.db")
        start_rag_server(rag)
    elif args.mode == 'llm':
        logger.info('不进行RAG流程')
        PerformanceLogger.reset_logger(log_file="no.json")
        llm = LocalLLM()
        start_llm_server(llm)
    elif args.mode == 'no':
        logger.info('仅构建数据库')
    elif args.mode == 'vllm':
        logger.info('采用vllm')
        PerformanceLogger.reset_logger(log_file='vllm.json')
        
    else:
        logger.info('模式错误')
    
    


    