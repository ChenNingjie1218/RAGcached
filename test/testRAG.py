import sys
import os
# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag import RAG

rag = RAG()
rag.set_milvus_db('cached.db')
# rag.set_use_kv_cache(True)
# query = '''庆祝中国人民解放军建军96周年招待会是什么时候举行的？'''
# query = '有游戏支持XeSS 1.2.0.13版本吗？'
# query = '国家卫生健康委在2023年7月28日开展的“启明行动”是为了防控哪个群体的哪种健康问题，并请列出活动发布的指导性文件名称。'
query = '陈宁杰是谁？'
res = rag.query(query)
rag.set_use_kv_cache(True)
res = rag.query(query)
res = rag.query(query)
# print(res)

# query = '''庆祝中国人民解放军建军96周年招待会是什么时候举行的？'''
# res = rag.query(query)

# query = '在大暑期间，为什么不能立即用冷水洗头冲凉或吹空调强风？'
# for response in rag.stream_query(query):
#     print(response, end="", flush=True)
