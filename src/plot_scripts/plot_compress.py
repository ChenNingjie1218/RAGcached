import os
import matplotlib.pyplot as plt
import sys

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.configs.config import src_docs_dir, docs_dir

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

# 绘制双直方图
plt.figure(figsize=(10, 6))
plt.hist(original_lengths, bins=50, alpha=0.6, color='blue', label='Original Text')
plt.hist(compressed_lengths, bins=50, alpha=0.6, color='green', label='Compressed Text')

# 添加标签和样式
plt.xlabel('Text Length (chars)')
plt.ylabel('Frequency')
plt.title('Distribution of Text Length Before and After Compression')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# 保存图像
plt.tight_layout()
plt.savefig('compression.png', dpi=300, bbox_inches='tight')

# 显示图像
# plt.show()