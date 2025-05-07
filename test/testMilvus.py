import sys
import os
# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrievers.milvus import MilvusDB
from src.embeddings.base import HuggingfaceEmbeddings
milvus = MilvusDB()
milvus.construct_index()
embedding_model = HuggingfaceEmbeddings(model_name='BAAI/bge-base-zh-v1.5')
text = '''
2023-08-16 18:24:57，英特尔为游戏开发者提供了最新XeSS 1.2.0版本，附带最新的SDK，并更新了7月27日编译的libxess.dll文件。英特尔XeSS 1.2向后兼容XeSS 1.0和1.1，支持动态分辨率缩放，修复了多种Bug，提高了性能和稳定性。尽管目前没有游戏支持XeSS 1.2.0.13版本，《瑞奇与叮当：时空跳转》已加入“1.2.0.10”版本。XeSS 1.1对英特尔是一次重大升级，带来了更快的XMX和DP4a内核及新的升级模型。XeSS不仅支持Intel GPU，还支持AMD和NVIDIA平台，已成为FSR或DLSS等其他竞争技术的可行替代方案。最新SDK提供了一个Demo供大家下载试用，包含文档和示例，但XeSS至今仍是闭源技术。
'''
data = {
    'id':1,
    'text':text,
    'vector':embedding_model.embed_query(text),
    'kv_path':''
}
milvus.insert(data)

query = '有游戏支持XeSS 1.2.0.13版本吗？'
res = milvus.query_vectors([embedding_model.embed_query(query)])
print(res)