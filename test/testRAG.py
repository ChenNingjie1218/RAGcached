import sys
import os
# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag import RAG

rag = RAG()
# query = '''庆祝中国人民解放军建军96周年招待会是什么时候举行的？'''
query = '有游戏支持XeSS 1.2.0.13版本吗？'
rag.query(query)