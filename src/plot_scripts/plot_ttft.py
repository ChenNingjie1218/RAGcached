import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.configs.config import base_dir

# === 你只需要修改下面这个字典中的数值即可 ===
ttft_data = {
    742: {'with_kv_cache': 42, 'without_kv_cache': 224, 'system': 70},
    1803: {'with_kv_cache': 39, 'without_kv_cache': 495, 'system': 69},
    2982: {'with_kv_cache': 39, 'without_kv_cache': 819, 'system': 76},
    4232: {'with_kv_cache': 41, 'without_kv_cache': 1247, 'system': 86},
    5428: {'with_kv_cache': 45, 'without_kv_cache': 1742, 'system': 93},
    6464: {'with_kv_cache': 48, 'without_kv_cache': 2188, 'system': 103},
}
# === 到这里为止 ===

token_counts = list(ttft_data.keys())
with_kv = [ttft_data[k]['with_kv_cache'] for k in token_counts]
without_kv = [ttft_data[k]['without_kv_cache'] for k in token_counts]
system = [ttft_data[k]['system'] for k in token_counts]

bar_width = 0.25  # 更窄的宽度以容纳三组柱子
index = np.arange(len(token_counts))

fig, ax = plt.subplots(figsize=(10, 6))

# 绘制三组柱状图
bar1 = ax.bar(index - bar_width, with_kv, bar_width, label='With KV Cache', color='skyblue')
bar2 = ax.bar(index, without_kv, bar_width, label='Without KV Cache', color='salmon')
bar3 = ax.bar(index + bar_width, system, bar_width, label='My System', color='lightgreen')

# 绘制折线图
ax.plot(index - bar_width, with_kv, color='blue', marker='o', linestyle='-', linewidth=2)
ax.plot(index, without_kv, color='red', marker='o', linestyle='-', linewidth=2)
ax.plot(index + bar_width, system, color='darkgreen', marker='o', linestyle='-', linewidth=2)

ax.set_xlabel('Token Count')
ax.set_ylabel('TTFT (Time to First Token)')
ax.set_title('TTFT Comparison: With KV Cache, Without KV Cache, and My System')
ax.set_xticks(index)
ax.set_xticklabels(token_counts)
ax.legend()

plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "chart", 'ttft_with_system.png'), dpi=300, bbox_inches='tight')
# plt.show()