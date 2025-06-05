import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.configs.config import base_dir
# === 你只需要修改下面这个字典中的数值即可 ===
ttft_data = {
    1004: {'with_kv_cache': 43, 'without_kv_cache': 281},
    1980: {'with_kv_cache': 49, 'without_kv_cache': 540},
    3157: {'with_kv_cache': 40, 'without_kv_cache': 879},
    4232: {'with_kv_cache': 41, 'without_kv_cache': 1247},
    5428: {'with_kv_cache': 44, 'without_kv_cache': 1745},
}
# === 到这里为止 ===

token_counts = list(ttft_data.keys())
with_kv = [ttft_data[k]['with_kv_cache'] for k in token_counts]
without_kv = [ttft_data[k]['without_kv_cache'] for k in token_counts]

bar_width = 0.35
index = np.arange(len(token_counts))

fig, ax = plt.subplots(figsize=(10, 6))

# 绘制柱状图
bar1 = ax.bar(index - bar_width/2, with_kv, bar_width, label='With KV Cache', color='skyblue')
bar2 = ax.bar(index + bar_width/2, without_kv, bar_width, label='Without KV Cache', color='salmon')

# 绘制折线图
ax.plot(index - bar_width/2, with_kv, color='blue', marker='o', linestyle='-', linewidth=2, label='With KV Cache (Trend)')
ax.plot(index + bar_width/2, without_kv, color='red', marker='o', linestyle='-', linewidth=2, label='Without KV Cache (Trend)')


ax.set_xlabel('Token Count')
ax.set_ylabel('TTFT (Time to First Token)')
ax.set_title('TTFT Comparison with and without KV Cache')
ax.set_xticks(index)
ax.set_xticklabels(token_counts)
ax.legend()

plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "chart", 'ttft.png'), dpi=300, bbox_inches='tight')
