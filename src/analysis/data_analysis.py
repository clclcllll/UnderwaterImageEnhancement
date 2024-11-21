# data_analysis.py：用于数据加载、预处理和分析的主脚本

"""
数据加载和解析： 脚本遍历 data/processed/ 文件夹中的所有图像文件，解析文件名以提取图像特征值，并存储到列表中。
创建DataFrame： 使用 pandas 将数据列表转换为 DataFrame，方便后续的数据分析。
合并退化类型标签： 从 degradation_labels.csv 文件中读取退化类型标签，并与主 DataFrame 合并。
数据保存： 将包含所有特征和退化类型的数据保存到 analysis_results.csv，统计结果保存到 degradation_type_stats.csv。
数据检查： 打印前5行数据和统计结果，确保数据正确加载和处理。
"""

# 导入必要的库
import os
import pandas as pd

# 定义数据文件夹路径
image_folder = '../../data/processed/'  # 根据实际路径调整

# 初始化数据列表
data = []

# 遍历文件夹中的所有图像文件
for filename in os.listdir(image_folder):
    if filename.endswith('.png'):
        # 去掉文件扩展名
        name = filename[:-4]
        # 以下划线分割文件名，提取特征值
        parts = name.split('_')
        index = int(parts[0])  # 序号
        blue = float(parts[1])  # 蓝色值
        green = float(parts[2])  # 绿色值
        red = float(parts[3])  # 红色值
        brightness = float(parts[4])  # 平均亮度
        laplacian_var = float(parts[5])  # 拉普拉斯方差（清晰度）
        # 将数据添加到列表
        data.append([filename, index, red, green, blue, brightness, laplacian_var])

# 创建包含特征值的DataFrame
df = pd.DataFrame(data, columns=['filename', 'index', 'red', 'green', 'blue', 'brightness', 'laplacian_var'])

# 加载退化类型标签（假设存在一个CSV文件包含文件名和退化类型）
labels_df = pd.read_csv('../../data/processed/degradation_labels.csv')  # 根据实际路径调整

# 合并退化类型标签到主DataFrame
df = df.merge(labels_df, on='filename')

# 保存合并后的数据到CSV文件
df.to_csv('../../results/metrics/analysis_results.csv', index=False)

# 打印前5行数据进行检查
print(df.head())

# 统计各退化类型的均值和标准差
stats = df.groupby('degradation_type').agg({
    'red': ['mean', 'std'],
    'green': ['mean', 'std'],
    'blue': ['mean', 'std'],
    'brightness': ['mean', 'std'],
    'laplacian_var': ['mean', 'std']
})

# 将统计结果保存到CSV文件
stats.to_csv('../../results/metrics/degradation_type_stats.csv')

# 打印统计结果
print(stats)
