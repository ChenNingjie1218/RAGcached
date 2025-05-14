import json
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.configs.config import performance_log_path, base_dir
# 文件路径设置
origin_file = os.path.join(performance_log_path, "origin.json")
rag_file = os.path.join(performance_log_path, "rag.json")
output_path = os.path.join(base_dir, "chart")
def load_data(file_path):
    """加载 JSON 日志文件，提取 query_time 和 complete_time"""
    query_times = []
    complete_times = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                event = entry.get("event")
                data = entry.get("data", {})

                if event == "query" and "query_time" in data:
                    query_times.append(data["query_time"])
                elif event == "complete" and "complete_time" in data:
                    complete_times.append(data["complete_time"])

            except json.JSONDecodeError:
                print(f"[警告] 解析失败: {line[:50]}...")

    return query_times, complete_times

# 加载数据
origin_query, origin_complete = load_data(origin_file)
rag_query, rag_complete = load_data(rag_file)

# 绘制箱型图函数
def plot_boxplot(data_list, labels, title, ylabel, output_file):
    plt.figure(figsize=(8, 6))
    plt.boxplot(data_list, tick_labels=labels, showfliers=False)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

# 1. Query Time 箱型图对比
# plot_boxplot(
#     [origin_query, rag_query],
#     ['Original', 'Compressed'],
#     'Query Time Distribution Comparison',
#     'Query Time (ms)',
#     os.path.join(output_path, 'query_time_comparison_boxplot.png')
# )

# 2. Complete Time 箱型图对比
plot_boxplot(
    [origin_complete, rag_complete],
    ['Original', 'Compressed'],
    'LLM Complete Time Distribution Comparison',
    'Complete Time (ms)',
    os.path.join(output_path, 'complete_time_comparison_boxplot.png')
)

print("✅ 图表已生成：query_time_comparison_boxplot.png 和 complete_time_comparison_boxplot.png")