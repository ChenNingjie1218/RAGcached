import os
import matplotlib.pyplot as plt
import sys
import numpy as np

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.configs.config import src_docs_dir, docs_dir, base_dir


def read_all_lines_from_dir(directory):
    """遍历目录下所有文件，读取每行内容"""
    all_lines = []
    for filename in sorted(os.listdir(directory)):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                all_lines.extend(lines)
    return all_lines


# 设置路径
original_dir = src_docs_dir
compressed_dir = docs_dir

# 读取并统计字符数（不要求一一对应）
original_lines = read_all_lines_from_dir(original_dir)
compressed_lines = read_all_lines_from_dir(compressed_dir)

original_lengths = [len(line.strip()) for line in original_lines if line.strip()]
compressed_lengths = [len(line.strip()) for line in compressed_lines if line.strip()]

# 计算总字符数
total_original_chars = sum(original_lengths)
total_compressed_chars = sum(compressed_lengths)

# 打印统计信息
def print_length_stats(lengths, label, total_chars):
    avg = np.mean(lengths)
    p50 = np.percentile(lengths, 50)
    p99 = np.percentile(lengths, 99)
    print(f"[{label}] ")
    print(f"  总文本长度: {total_chars:,} 字符")
    print(f"  平均文本长度: {avg:.2f}, P50: {int(p50)}, P99: {int(p99)}")

print("📊 文本长度统计：")
print_length_stats(original_lengths, "Original", total_original_chars)
print_length_stats(compressed_lengths, "Compressed", total_compressed_chars)


# # 绘制双直方图
# plt.figure(figsize=(10, 6))
# plt.hist(original_lengths, bins=50, alpha=0.6, color='blue', label='Original Text')
# plt.hist(compressed_lengths, bins=50, alpha=0.6, color='green', label='Compressed Text')

# # 添加标签和样式
# plt.xlabel('Text Length (chars)')
# plt.ylabel('Frequency')
# plt.title('Distribution of Text Length Before and After Compression')
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.5)

# # 保存图像
# plt.tight_layout()
# plt.savefig(os.path.join(base_dir, "chart", 'compression.png'), dpi=300, bbox_inches='tight')

# 显示图像
# plt.show()


plt.figure(figsize=(10, 6))

# 原始文本 — 蓝色 + 斜向纹理
plt.hist(
    original_lengths,
    bins=50,
    label='Original Text',
    color='#4A90E2',            # 蓝色
    edgecolor='black',
    alpha=0.6,                  # 半透明保证重叠可见
    hatch='///'
)

# 压缩文本 — 绿色 + 横向纹理
plt.hist(
    compressed_lengths,
    bins=50,
    label='Compressed Text',
    color='#50E3C2',            # 绿色
    edgecolor='black',
    alpha=0.6,
    hatch='\\\\\\'
)

plt.xlabel('Text Length (chars)')
plt.ylabel('Frequency')
plt.title('Distribution of Text Length Before and After Compression')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.savefig(os.path.join(base_dir, "chart", 'compression.png'),
            dpi=300, bbox_inches='tight')
