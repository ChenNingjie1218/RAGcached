# analysis.py
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sys

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.configs.config import performance_log_path, base_dir


def load_logs(log_file):
    """加载 JSON 格式的性能日志"""
    logs = []
    path = os.path.join(performance_log_path, log_file)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"[警告] 解析失败: {line[:100]}...")
    return logs


def analyze_performance(logs):
    """按模块和事件类型提取耗时数据"""
    stats = defaultdict(list)

    for entry in logs:
        module = entry["module"]
        event = entry["event"]
        data = entry["data"]

        # 提取耗时字段
        time_key = {
            "load_kv_cache": "load_time",
            "stack_kv_cache": "stack_time",
            "complete": "complete_time",
            "query": "query_time",
            "input_token_count":"token_count",
            "reuse_kv_cache":"reuse_kv_cache"
        }.get(event, None)

        if time_key and time_key in data:
            stats[f"{module}_{event}"].append(data[time_key])

    return stats


def calculate_stats(times):
    """计算 P50、P99 和平均值"""
    return {
        "mean": np.mean(times),
        "P50": np.percentile(times, 50),
        "P99": np.percentile(times, 99)
    }


def plot_bar_chart(stats):
    """绘制柱状图对比各模块的 mean / P50 / P99"""
    labels = []
    mean_values = []
    p50_values = []
    p99_values = []

    for key, times in stats.items():
        if len(times) == 0:
            continue
        s = calculate_stats(times)
        labels.append(key)
        mean_values.append(s["mean"])
        p50_values.append(s["P50"])
        p99_values.append(s["P99"])

    x = range(len(labels))

    plt.figure(figsize=(14, 6))
    bar_width = 0.25

    plt.bar(x, p50_values, width=bar_width, label='P50', color='skyblue')
    plt.bar([i + bar_width for i in x], p99_values, width=bar_width, label='P99', color='salmon')
    plt.bar([i + 2 * bar_width for i in x], mean_values, width=bar_width, label='Mean', color='lightgreen')

    # 添加折线连接均值
    plt.plot([i + 2 * bar_width for i in x], mean_values, color='darkgreen', marker='o', linestyle='--', label='Mean Trend')

    plt.xticks([i + bar_width for i in x], labels, rotation=45)
    plt.ylabel("Latency (ms)")
    plt.title("Performance Metrics by Module (P50 / P99 / Mean)")
    plt.legend()
    plt.tight_layout()

    file_name = os.path.splitext(log_file)[0]
    plt.savefig(os.path.join(base_dir, "chart", file_name + "_with_mean.png"))
    plt.show()


def main(log_file="performance_log.json"):
    print(f"正在分析日志文件: {log_file}")
    logs = load_logs(log_file)
    stats = analyze_performance(logs)

    print("\n📊 各模块性能指标:")
    for key, times in stats.items():
        s = calculate_stats(times)
        print(f"{key}: Mean={s['mean']:.2f}ms, P50={s['P50']:.2f}ms, P99={s['P99']:.2f}ms")

    # plot_bar_chart(stats)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        log_file = "performance_log.json"

    main(log_file)