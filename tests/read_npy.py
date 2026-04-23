import numpy as np
import sys

if len(sys.argv) != 2:
    print("用法: python read_npy.py <文件路径>")
    sys.exit(1)

file_path = sys.argv[1]

try:
    data = np.load(file_path, allow_pickle=True)
    print("数据内容:")
    print(data)
    print("\n数据类型:", type(data))
    if isinstance(data, np.ndarray):
        print("数组形状:", data.shape)
        print("数据类型 (dtype):", data.dtype)
except FileNotFoundError:
    print(f"错误: 找不到文件 '{file_path}'")
except Exception as e:
    print(f"读取文件时出错: {e}")