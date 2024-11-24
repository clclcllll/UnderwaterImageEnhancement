import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 文件路径配置
output_excel = "../../results/metrics/T4_basedCNN.xlsx"  # 可以更改为其他文件路径

# 指定输出图片的目录
output_dir = "../../results/plots/"  # 确保该目录存在


# 根据文件名设置增强描述
def get_enhancement_description(filename):
    if filename == "T3.xlsx":
        return "Enhancement for Different Degradation Scenarios"
    elif filename == "T4_ordinary_basedT3.xlsx":
        return "Combined Enhancement"
    elif filename == "T4_optimizing_basedT3.xlsx":
        return "Combined Enhancement Using Optimization"
    elif filename == "T4_basedCNN.xlsx":
        return "Enhancement Using Machine Learning and Deep Learning (CNN+GAN)"

    else:
        return "Image Enhancement"


# 从路径中提取文件名
excel_filename = os.path.basename(output_excel)
enhancement_description = get_enhancement_description(excel_filename)

# 读取数据
data = pd.read_excel(output_excel)

# 设置图像文件名为索引
data.set_index('image file name', inplace=True)
n_images = len(data)
x = np.arange(n_images)

# 确定X轴标签的步长，最多显示10个标签
step = max(1, n_images // 10)
xticks_positions = x[::step]
xticks_labels = data.index[::step]

# 创建子图
fig, axs = plt.subplots(3, 2, figsize=(18, 18))

# 定义指标列表和颜色
metrics = [("PSNR", "skyblue"),
           ("UCIQE", "lightcoral"),
           ("UIQM", "lightgreen")]
colors = [("blue", "orange"),
          ("red", "purple"),
          ("green", "brown")]

for i, (metric, diff_color) in enumerate(metrics):
    original = data[metric]
    enhanced = data[f"{metric}-IM"]
    difference = enhanced - original

    # 对比折线图
    axs[i, 0].plot(x, original, marker='o', label=f"{metric} (Original)", color=colors[i][0])
    axs[i, 0].plot(x, enhanced, marker='o', label=f"{metric} (Enhanced)", color=colors[i][1])
    axs[i, 0].set_title(f"{metric} Comparison - {enhancement_description}")
    axs[i, 0].set_xlabel("Image File Name")
    axs[i, 0].set_ylabel(metric)
    axs[i, 0].set_xticks(xticks_positions)
    axs[i, 0].set_xticklabels(xticks_labels, rotation=45)
    axs[i, 0].legend()
    axs[i, 0].grid()

    # 差异柱状图
    axs[i, 1].bar(x, difference, color=diff_color)
    axs[i, 1].set_title(f"{metric} Improvement - {enhancement_description}")
    axs[i, 1].set_xlabel("Image File Name")
    axs[i, 1].set_ylabel("Difference")
    axs[i, 1].set_xticks(xticks_positions)
    axs[i, 1].set_xticklabels(xticks_labels, rotation=45)
    axs[i, 1].grid()

plt.tight_layout()

# 构建输出图片的文件名，包含区分特征
output_image_filename = f"{os.path.splitext(excel_filename)[0]}_comparison.png"
output_image_path = os.path.join(output_dir, output_image_filename)

# 保存图片到指定路径
plt.savefig(output_image_path)
plt.show()
