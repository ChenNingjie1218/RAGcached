from llama_cloud import TokenTextSplitter
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.text_splitter import TokenTextSplitter
import torch
from llms.local_model import LocalLLM
from llms.api_model import API_LLM
from configs.config import kv_cache_output_dir, src_docs_dir, docs_dir, milvus_host, milvus_port, milvus_collection_name, milvus_dim
from typing import List
from tqdm import tqdm
import os
import logging
from embeddings.base import HuggingfaceEmbeddings
from retrievers.milvus import MilvusDB

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# LlamaIndex related
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    PromptHelper,
    Document,
    StorageContext,
    load_index_from_storage,
)

class KVCachedNodeParser(SimpleNodeParser):
    # 针对每个分块 生成KV Cache
    def process_chunk(self, chunk_text: str, chunk_id: str, **kwargs):
        model = kwargs.get('model')
        tokenizer = kwargs.get('tokenizer')
        
        if not model or not tokenizer:
            raise ValueError("Model and tokenizer must be provided in kwargs")
            
        chunk_text = "<|doc_start|>" + chunk_text + "<|doc_end|>"
        inputs = tokenizer(chunk_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True)
        
        past_key_values = outputs.past_key_values
        
        kvcache_file_path = f'{kv_cache_output_dir}/kvcache_chunk_{chunk_id}.pt'
        torch.save(past_key_values, kvcache_file_path)
        
        node = TextNode(
            text=chunk_text,
            id_=f"chunk_{chunk_id}",
            metadata={
                "kvcache_file_path": kvcache_file_path
            }
        )
        return node
    
    # 重写get_nodes_from_documents方法
    # 这里分块策略可以定制化，有待深入
    def get_nodes_from_documents(
        self,
        documents: List[Document]
    ) -> List[BaseNode]:
        llm = LocalLLM()
        # llm.set_prompt('''<|im_start|>system
        #     你是一个准确可靠的AI助手，能够借助外部文档回答问题。如果文档中包含正确答案，你会给出准确的回答。如果文档中没有包含答案，你会生成"由于文档中信息不足，我无法回答这个问题。"<|im_end|>
        #     <|im_start|>user
        #     文档:''')
        llm.set_prompt('')
        nodes = []
        for doc_id, document in tqdm(enumerate(documents)):
            for chunk_id, chunk_text in enumerate(document.text.splitlines()):
                node = self.process_chunk(chunk_text, f"{doc_id}_{chunk_id}", model=llm.model, tokenizer=llm.tokenizer)
                nodes.append(node)
        return nodes

class Preprocessor:
    def __init__(self):
        pass
    
    def _store_in_milvus(self, nodes: List[BaseNode]):
        milvus = MilvusDB()
        milvus.construct_index()

        """将节点存储到Milvus"""
        embedding_model = HuggingfaceEmbeddings(model_name="BAAI/bge-base-zh-v1.5")
        data = []
        for node in nodes:
            data.append({
                "id":node.id_,
                "vector":embedding_model.get_text_embedding(node.text),
                "text":node.text,
                "kv_path":node.metadata["kvcache_file_path"],
            })
        milvus.insert(data)
        logger.info(f"数据插入完成，总量：{len(nodes)}")
    
    def get_nodes(self) -> List[BaseNode]:
        """
        Processes documents and generates nodes with KV cache.
        """
        documents = SimpleDirectoryReader(docs_dir).load_data()
        node_parser = KVCachedNodeParser()
        nodes = node_parser.get_nodes_from_documents(documents)
        return nodes
    
    def prepare_kv_cache(self):
        nodes = self.get_nodes()
        self._store_in_milvus(nodes)

    def compress_context(self):
        # llm = API_LLM()
        llm = LocalLLM()
        llm.set_prompt('''<|im_start|>system
            请对以下文本进行去冗余处理，保留关键信息。只需要返回一个版本，且是完整的自然段落形式，不要重复输出、不要分段、不要编号、不要加标题。<|im_end|>
            <|im_start|>user
            以下是原文档内容：
            ''')
        documents = SimpleDirectoryReader(src_docs_dir).load_data()
        for doc_id, document in enumerate(tqdm(documents, desc="Processing files")):
            # 获取原始文件名（假设 metadata 中有 'file_name' 字段）
            file_name = os.path.basename(document.metadata.get('file_name', f'doc_{doc_id}.txt'))
            logger.info(f'开始处理 {file_name} ...')
            
            # 输出路径
            output_file = os.path.join(docs_dir, file_name)

            # 以追加模式打开文件，准备流式写入
            with open(output_file, 'w', encoding='utf-8') as f:
                # 处理文档内容：假设 document.text 是字符串，我们一行一行地处理
                for line in document.text.splitlines():
                    # LLM提取关键信息
                    # logger.info(f'压缩：{line[:10]}...')
                    processed_line = llm.complete(line)
                    if processed_line:  # 去除空行
                        f.write(processed_line + '\n')

        logger.info(f"All files have been processed and saved to: {docs_dir}")
        
        
