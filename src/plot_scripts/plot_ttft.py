import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.configs.config import base_dir

# === 你只需要修改下面这个字典中的数值即可 ===
ttft_data = {
    742: {'with_kv_cache': 42, 'without_kv_cache': 224, 'system': 70, 'from_disk': 185},
    1803: {'with_kv_cache': 39, 'without_kv_cache': 495, 'system': 69, 'from_disk': 323},
    2982: {'with_kv_cache': 39, 'without_kv_cache': 819, 'system': 76, 'from_disk': 482},
    4232: {'with_kv_cache': 41, 'without_kv_cache': 1247, 'system': 86, 'from_disk': 663},
    5428: {'with_kv_cache': 45, 'without_kv_cache': 1742, 'system': 93, 'from_disk': 801},
    6464: {'with_kv_cache': 48, 'without_kv_cache': 2188, 'system': 103, 'from_disk': 940},
}
# === 到这里为止 ===

token_counts = list(ttft_data.keys())
with_kv = [ttft_data[k]['with_kv_cache'] for k in token_counts]
without_kv = [ttft_data[k]['without_kv_cache'] for k in token_counts]
system = [ttft_data[k]['system'] for k in token_counts]
disk = [ttft_data[k]['from_disk'] for k in token_counts]

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

# ==== 新图表：仅比较 without_kv_cache 和 from_disk ====
fig2, ax2 = plt.subplots(figsize=(10, 6))

bar_width_new = 0.35  # 更宽一点，容纳两个柱子
index_new = np.arange(len(token_counts))

# 绘制两组柱状图
bar_disk = ax2.bar(index_new - bar_width_new/2, disk, bar_width_new, label='From Disk', color='lightblue')
bar_without = ax2.bar(index_new + bar_width_new/2, without_kv, bar_width_new, label='Without KV Cache', color='salmon')

# 绘制折线图
ax2.plot(index_new - bar_width_new/2, disk, color='blue', marker='o', linestyle='-', linewidth=2)
ax2.plot(index_new + bar_width_new/2, without_kv, color='red', marker='o', linestyle='-', linewidth=2)

# 设置标签、标题等
ax2.set_xlabel('Token Count')
ax2.set_ylabel('TTFT (Time to First Token)')
ax2.set_title('TTFT Comparison: From Disk vs Without KV Cache')
ax2.set_xticks(index_new)
ax2.set_xticklabels(token_counts)
ax2.legend()

plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "chart", 'ttft_disk_vs_without_kv.png'), dpi=300, bbox_inches='tight')
# plt.show()