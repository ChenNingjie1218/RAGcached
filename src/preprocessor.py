from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.text_splitter import TokenTextSplitter
from pydantic import Field
import torch
from src.llms.local_model import LocalLLM
from src.llms.local_model import AutoTokenizer
from src.llms.api_model import API_LLM
from src.configs.config import kv_cache_output_dir, src_docs_dir, docs_dir, embedding_model_path, model_path
from typing import Any, List
from tqdm import tqdm
import os
import logging
from src.embeddings.base import HuggingfaceEmbeddings
from src.prompt.prompt_compress import COMPRESS_PREFIX
from src.retrievers.milvus import MilvusDB

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
            
        chunk_text = '<|document_sep|>' + chunk_text
        inputs = tokenizer(chunk_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True)
        
        past_key_values = outputs.past_key_values
        
        # kvcache_file_path = f'{kv_cache_output_dir}/kvcache_chunk_{chunk_id}.pt'
        kvcache_file = f'kvcache_chunk_{chunk_id}.pt'
        path = os.path.join(kv_cache_output_dir, kvcache_file)
        torch.save(past_key_values, path)
        
        node = TextNode(
            text=chunk_text,
            id_=f'chunk_{chunk_id}',
            metadata={
                "kvcache_file": kvcache_file
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
        nodes = []
        for doc_id, document in tqdm(enumerate(documents)):
            for chunk_id, chunk_text in enumerate(document.text.splitlines()):
                node = self.process_chunk(chunk_text, f"{doc_id}_{chunk_id}", model=llm.model, tokenizer=llm.tokenizer)
                nodes.append(node)
        return nodes

class LineNodeParser(SimpleNodeParser):
    # 重写get_nodes_from_documents方法
    # 这里分块策略可以定制化，有待深入
    def get_nodes_from_documents(
        self,
        documents: List[Document]
    ) -> List[BaseNode]:
        nodes = []
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        SPECIAL_SEP_TOKEN = '<|document_sep|>'
        tokenizer.add_special_tokens({"additional_special_tokens": [SPECIAL_SEP_TOKEN]})
        for document in tqdm(documents):
            for chunk_id, chunk_text in enumerate(document.text.splitlines()):
                chunk_text = '<|document_sep|>' + chunk_text
                chunk_tokens = tokenizer.encode(chunk_text, add_special_tokens=False)
                token_count = len(chunk_tokens)
                node = TextNode(
                    text=chunk_text,
                    id_=f'chunk_{chunk_id}',
                    metadata={
                        "token_count": token_count
                    }
                )
                nodes.append(node)
        return nodes
    
class HybridTokenNodeParser(SimpleNodeParser):
    pad_token_id: Any = Field(default=None, description="pad token")
    chunk_size: int = Field(default=128, description="固定大小分块")
    tokenizer: Any = Field(default=None, description="HuggingFace分词器")
    sentence_splitter: Any = Field(default=None, description="SentenceSplitter（用于按句子拆分）")
    needpad: bool = Field(default=False, description="是否填充")
    def __init__(
        self,
        needpad = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.chunk_size = 128

        # 加载 Qwen Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.needpad = needpad
        if needpad:
            pad_token = "."
            # 验证 pad_token 是否存在
            if pad_token not in self.tokenizer.get_vocab():
                raise ValueError(f"Padding token '{pad_token}' not found in tokenizer vocabulary.")
            self.pad_token_id = self.tokenizer.encode(pad_token, add_special_tokens=False)[0]

        # 初始化 SentenceSplitter（用于按句子拆分）
        self.sentence_splitter = SentenceSplitter(
            tokenizer=self.tokenizer.encode,
            chunk_size=self.chunk_size,
            chunk_overlap=0
        )

    def get_nodes_from_documents(
        self,
        documents: List[Document],
        **kwargs
    ) -> List[TextNode]:
        nodes = []
        node_counter = 0

        for doc_idx, document in enumerate(tqdm(documents, desc="Processing Documents")):
            # 1. 按行分割
            lines = document.text.splitlines()

            for line in lines:
                line = line.strip()
                if not line:
                    continue  # 跳过空行

                # 2. 按句子拆分
                sentences = self.sentence_splitter.split_text(line)

                # 3. 合并句子到 512 token
                current_chunk = []
                current_length = 0

                for sentence in sentences:
                    sentence_tokens = self.tokenizer.encode(sentence, add_special_tokens=False)
                    sentence_length = len(sentence_tokens)

                    if current_length + sentence_length <= self.chunk_size:
                        current_chunk.append(sentence)
                        current_length += sentence_length
                    else:
                        # 当前 chunk 已满，生成节点
                        chunk_text = " ".join(current_chunk)
                        if self.needpad:
                            padded_text = self._pad_chunk(self.tokenizer.encode(chunk_text, add_special_tokens=False))
                        else:
                            padded_text = chunk_text
                        nodes.append(
                            TextNode(text=padded_text, id_=f"doc_{doc_idx}_chunk_{node_counter}")
                        )
                        node_counter += 1

                        # 开始新 chunk
                        current_chunk = [sentence]
                        current_length = sentence_length

                # 处理剩余内容
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    if self.needpad:
                        padded_text = self._pad_chunk(self.tokenizer.encode(chunk_text, add_special_tokens=False))
                    else:
                        padded_text = chunk_text
                    nodes.append(
                        TextNode(text=padded_text, id_=f"doc_{doc_idx}_chunk_{node_counter}")
                    )
                    node_counter += 1

        return nodes

    def _pad_chunk(self, token_ids: List[int]) -> str:
        if len(token_ids) >= self.chunk_size:
            return self.tokenizer.decode(token_ids[:self.chunk_size], skip_special_tokens=False)

        pad_length = self.chunk_size - len(token_ids)
        padded_ids = token_ids + [self.pad_token_id] * pad_length
        return self.tokenizer.decode(padded_ids, skip_special_tokens=False)

class Preprocessor:
    def __init__(self):
        pass
    
    def _store_in_milvus(self, nodes: List[BaseNode], kv_cache:bool=False):
        if kv_cache:
            milvus = MilvusDB("cached.db")
        else:
            milvus = MilvusDB("origin.db")
        milvus.construct_index()
        """将节点存储到Milvus"""
        embedding_model = HuggingfaceEmbeddings(model_name=embedding_model_path)
        # 收集所有文本
        texts = [node.text for node in nodes]

        # 批量获取 embeddings
        vectors = embedding_model.embed_documents(texts)
        data = []
        for i, node in enumerate(nodes):
            item = {
                "vector": vectors[i],
                "text": node.text
            }
            if "kvcache_file" in node.metadata:
                item["kv_file"] = node.metadata["kvcache_file"]
            if "token_count" in node.metadata:
                item["token_count"] = node.metadata["token_count"]
            data.append(item)
        total = len(data)
        batch_size = 8000
        for i in range(0, total, batch_size):
            batch = data[i:i+batch_size]
            try:
                milvus.insert(batch)
                print(f"Inserted batch {i//batch_size+1} successfully.")
            except Exception as e:
                print(f"Error inserting batch {i//batch_size+1}: {e}")
        logger.info(f"数据插入完成，总量：{len(nodes)}")
    
    def get_nodes(self, use_kv_cache:bool=False) -> List[BaseNode]:
        """
        Processes documents and generates nodes with KV cache.
        """
        if use_kv_cache:
            documents = SimpleDirectoryReader(docs_dir).load_data()
            # node_parser = KVCachedNodeParser()
            node_parser = LineNodeParser()
        else:
            documents = SimpleDirectoryReader(src_docs_dir).load_data()
            node_parser = LineNodeParser()
        # if use_kv_cache:
        #     documents = SimpleDirectoryReader(docs_dir).load_data()
        #     node_parser = HybridTokenNodeParser(needpad=True)
        # else:
        #     documents = SimpleDirectoryReader(src_docs_dir).load_data()
        #     node_parser = HybridTokenNodeParser()

        nodes = node_parser.get_nodes_from_documents(documents)
        return nodes
    
    def prepare(self, kv_cache:bool):
        nodes = self.get_nodes(kv_cache)
        self._store_in_milvus(nodes, kv_cache)

    def compress_context(self, part_suffix : str):
        # llm = API_LLM()
        llm = LocalLLM()
        llm.set_prompt(COMPRESS_PREFIX)
        documents = SimpleDirectoryReader(src_docs_dir).load_data()
        # for doc_id, document in enumerate(tqdm(documents, desc="Processing files")):
        for doc_id, document in enumerate(documents):
            # 获取原始文件名（假设 metadata 中有 'file_name' 字段）
            file_name = os.path.basename(document.metadata.get('file_name', f'doc_{doc_id}.txt'))

            if part_suffix not in file_name:
                continue
            
            logger.info(f'开始处理 {file_name} ...')

            # 输出路径
            output_file = os.path.join(docs_dir, file_name)

            # 以追加模式打开文件，准备流式写入
            with open(output_file, 'w', encoding='utf-8') as f:
                # 处理文档内容：假设 document.text 是字符串，我们一行一行地处理
                for line in tqdm(document.text.splitlines()):
                    # LLM提取关键信息
                    # logger.info(f'压缩：{line[:10]}...')
                    processed_line = llm.complete(line)
                    if processed_line:  # 去除空行
                        f.write(processed_line + '\n')

        logger.info(f"All files have been processed and saved to: {docs_dir}")
        
        
