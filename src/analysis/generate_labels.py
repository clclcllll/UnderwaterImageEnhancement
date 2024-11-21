# 导入必要的库
import os
import pandas as pd

# 定义数据文件夹路径
# 假设当前工作目录为项目根目录 UnderwaterImageEnhancement/
processed_folder = './data/raw/processed/'  # 根据实际路径调整

# 初始化数据列表
data = []

# 定义分类阈值
# 根据经验或数据分布设定阈值，您可以根据实际情况调整这些值
BRIGHTNESS_THRESHOLD = 100  # 亮度阈值，低于此值认为是弱光
LAPLACIAN_THRESHOLD = 100  # 拉普拉斯方差阈值，低于此值认为是模糊
COLOR_RATIO_THRESHOLD = 1.5  # 颜色通道比例阈值，大于此值认为存在偏色

# 遍历处理后的图像文件夹中的所有图像文件
for filename in os.listdir(processed_folder):
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

        # 初始退化类型为空
        degradation_type = ''

        # 判断是否为弱光（低亮度）
        if brightness < BRIGHTNESS_THRESHOLD:
            degradation_type = '弱光'

        # 判断是否为模糊（低清晰度）
        elif laplacian_var < LAPLACIAN_THRESHOLD:
            degradation_type = '模糊'

        # 判断是否为偏色（颜色通道不平衡）
        else:
            # 计算最大和最小颜色通道值
            max_color = max(red, green, blue)
            min_color = min(red, green, blue)
            # 如果最大值与最小值的比例大于阈值，则认为存在偏色
            if max_color / min_color > COLOR_RATIO_THRESHOLD:
                degradation_type = '偏色'
            else:
                # 如果不属于以上任何一种退化类型，则标记为'正常'或'未知'
                degradation_type = '未知'

        # 将结果添加到数据列表
        data.append([filename, degradation_type])

# 创建包含文件名和退化类型的DataFrame
labels_df = pd.DataFrame(data, columns=['filename', 'degradation_type'])

# 将结果保存到 degradation_labels.csv
labels_df.to_csv('./data/processed/degradation_labels.csv', index=False, encoding='utf-8-sig')

# 打印前5行数据进行检查
print(labels_df.head())

print("degradation_labels.csv 已生成，保存在 ./data/processed/ 文件夹下。")
