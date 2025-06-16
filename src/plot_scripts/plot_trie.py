import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.configs.config import base_dir

# 数据准备
queries = [0, 20, 40, 60, 80, 100]
memory_used_gib = [16.0, 21.7, 23.6, 27.0, 29.7, 32.8]  # 已用显存 (GiB)
memory_increase = [0, 5.7, 1.9, 3.4, 2.7, 3.1]          # 显存使用增量 (GiB)
tokens = [0, 46356, 93405, 142070, 188055, 234029]  # 每次查询对应的 token 总数

rag_mean = [75.58, 77.18, 74.87, 76.33, 75.74]
rag_p50 = [74.96, 74.00, 74.63, 75.24, 74.62]
rag_p99 = [85.96, 147.92, 87.32, 100.51, 81.81]

# ========================
# 图一：显存使用趋势图（X 轴改为 token 数）
# ========================
plt.figure(figsize=(10, 5))
plt.plot(tokens, memory_used_gib, marker='o', color='tab:blue', label='Memory Used')
for i, txt in enumerate(memory_used_gib):
    plt.annotate(f'{txt:.1f}', (tokens[i], memory_used_gib[i]), textcoords="offset points", xytext=(0,10), ha='center')

# 添加增量箭头标注
for i in range(1, len(tokens)):
    plt.annotate('', 
                 xy=(tokens[i], memory_used_gib[i]), 
                 xytext=(tokens[i], memory_used_gib[i - 1]),
                 arrowprops=dict(arrowstyle='->', color='gray'))

# 设置 X 轴格式化为千分位数字
def thousands_formatter(x, pos):
    return f'{int(x):,}'

from matplotlib.ticker import FuncFormatter
plt.gca().xaxis.set_major_formatter(FuncFormatter(thousands_formatter))

# 更新标签与标题
plt.xlabel('Token Count')
plt.ylabel('GPU Memory Usage (GiB)')
plt.title('GPU Memory Usage vs Token Count')
plt.grid(True)
plt.legend()
plt.tight_layout()

# 保存图像
plt.savefig(os.path.join(base_dir, "chart", 'memory.png'), dpi=300, bbox_inches='tight')
plt.close()

# ========================
# 图二：RAG 查询延迟趋势图
# ========================
plt.figure(figsize=(10, 5))
plt.plot(queries[1:], rag_mean, marker='s', linestyle='--', color='green', label='RAG Mean')
plt.plot(queries[1:], rag_p50, marker='^', linestyle='--', color='orange', label='RAG P50')
plt.plot(queries[1:], rag_p99, marker='D', linestyle='--', color='purple', label='RAG P99')

# 添加数据标签
for i, txt in enumerate(rag_mean):
    plt.annotate(f'{txt:.2f}', (queries[i+1], rag_mean[i]), textcoords="offset points", xytext=(0,10), ha='center')
for i, txt in enumerate(rag_p50):
    plt.annotate(f'{txt:.2f}', (queries[i+1], rag_p50[i]), textcoords="offset points", xytext=(0,10), ha='center')
for i, txt in enumerate(rag_p99):
    plt.annotate(f'{txt:.2f}', (queries[i+1], rag_p99[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.xlabel('Cache Query Count')
plt.ylabel('Latency (ms)')
plt.title('RAG Query Latency Trends')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "chart", 'latency.png'), dpi=300, bbox_inches='tight')
plt.close()